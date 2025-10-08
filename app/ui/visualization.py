from __future__ import annotations

from typing import Any, Dict, List, Tuple

import streamlit as st

from sqlmodel import select

from app.data import models
from app.data.db import session_scope

try:
    import networkx as nx
    from pyvis.network import Network
    import streamlit.components.v1 as components
except ImportError:  # pragma: no cover
    nx = None
    Network = None
    components = None


def render() -> None:
    """Render the interactive knowledge graph visualisation page."""
    st.header("Knowledge Graph Visualization")
    st.write("Interact with the approved knowledge graph edges and trace their provenance.")

    if not nx or not Network or not components:
        st.error("Graph visualization dependencies missing. Install networkx and pyvis.")
        return

    documents = _load_documents()
    doc_lookup = {doc["name"]: doc["id"] for doc in documents if doc["id"] is not None}
    selected_docs = st.multiselect("Filter by document", list(doc_lookup.keys()))
    doc_ids = [doc_lookup[name] for name in selected_docs]

    nodes, edges = _load_graph(document_ids=doc_ids)
    if not nodes or not edges:
        message = "No approved edges yet. Approve triples to populate the graph."
        if doc_ids:
            message = "No edges were found for the selected document filter."
        st.info(message)
        return

    color_lookup = _build_color_lookup(nodes)

    graph = nx.DiGraph()
    for node in nodes:
        entity_type = node.get("entity_type") or "Unspecified"
        color = color_lookup.get(entity_type, color_lookup.get("Unspecified", "#6c757d"))
        graph.add_node(
            node["id"],
            label=node["label"],
            title=_format_node_tooltip(node),
            color=color,
        )

    for edge in edges:
        graph.add_edge(
            edge["subject_id"],
            edge["object_id"],
            label=edge["predicate"],
            title=_format_edge_tooltip(edge),
        )

    net = Network(height="600px", width="100%", directed=True)
    net.force_atlas_2based()
    net.from_nx(graph)

    html = net.generate_html(notebook=False)
    components.html(html, height=650, scrolling=True)

    legend_items = _build_legend(color_lookup)
    if legend_items:
        st.markdown("**Entity type legend**<br>" + "<br>".join(legend_items), unsafe_allow_html=True)


def _load_documents() -> List[Dict[str, Any]]:
    """Return available documents for graph filtering."""
    with session_scope() as session:
        stmt = select(models.Document).order_by(models.Document.display_name)
        rows = session.exec(stmt).all()
        return [{"id": row.id, "name": row.display_name} for row in rows]


def _load_graph(*, document_ids: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return serialized nodes and edges for visualization."""

    with session_scope() as session:
        edge_stmt = select(models.Edge)

        if document_ids:
            edge_id_column: Any = models.Edge.id
            document_id_column: Any = models.DocumentChunk.document_id
            edge_stmt = edge_stmt.where(
                edge_id_column.in_(
                    select(models.EdgeSource.edge_id)
                    .join(models.DocumentChunk)
                    .where(document_id_column.in_(document_ids))
                )
            )

        edge_rows = session.exec(edge_stmt).all()

        node_lookup: Dict[Any, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        for edge in edge_rows:
            subject = edge.subject
            obj = edge.object
            if subject is None or obj is None:
                continue

            subject_payload = node_lookup.setdefault(subject.id, _serialize_node(subject))
            obj_payload = node_lookup.setdefault(obj.id, _serialize_node(obj))

            sources: List[Dict[str, Any]] = []
            for source in edge.sources:
                chunk = source.document_chunk
                doc = chunk.document if chunk and chunk.document else None
                sources.append(
                    {
                        "type": source.source_type.value,
                        "document": doc.display_name if doc else None,
                        "page_label": chunk.page_label if chunk else None,
                    }
                )

            tags = sorted({tag.label for tag in (edge.tags or []) if tag.label})

            edges.append(
                {
                    "id": edge.id,
                    "predicate": edge.predicate,
                    "subject_id": subject_payload["id"],
                    "object_id": obj_payload["id"],
                    "subject_label": subject_payload["label"],
                    "object_label": obj_payload["label"],
                    "sources": sources,
                    "tags": tags,
                }
            )

        return list(node_lookup.values()), edges


def _serialize_node(node: models.Node) -> Dict[str, Any]:
    canonical_label = node.canonical_term.label if node.canonical_term else None
    node_id = node.id if node.id is not None else f"node:{node.label}"
    return {
        "id": node_id,
        "label": node.label,
        "entity_type": node.entity_type,
        "canonical_label": canonical_label,
    }


def _build_color_lookup(nodes: List[Dict[str, Any]]) -> Dict[str, str]:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_map: Dict[str, str] = {"Unspecified": "#6c757d"}
    next_index = 0

    for node in nodes:
        raw_type = node.get("entity_type")
        entity_type = raw_type.strip() if isinstance(raw_type, str) else None
        key = entity_type or "Unspecified"
        if key in color_map:
            continue
        color_map[key] = palette[next_index % len(palette)]
        next_index += 1

    return color_map


def _build_legend(color_lookup: Dict[str, str]) -> List[str]:
    legend_items: List[str] = []
    for entity_type, color in color_lookup.items():
        label = entity_type or "Unspecified"
        swatch = (
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
            f"background:{color};margin-right:6px;vertical-align:middle;'></span>"
        )
        legend_items.append(f"{swatch}{label}")
    return legend_items


def _format_node_tooltip(node: Dict[str, Any]) -> str:
    lines = [f"<strong>{node['label']}</strong>"]
    entity_type = node.get("entity_type")
    if entity_type:
        lines.append(f"Type: {entity_type}")
    else:
        lines.append("Type: Unspecified")
    canonical = node.get("canonical_label")
    if canonical:
        lines.append(f"Canonical: {canonical}")
    return "<br/>".join(lines)


def _format_edge_tooltip(edge: Dict[str, Any]) -> str:
    lines = [f"<strong>{edge['predicate']}</strong>"]
    lines.append(f"{edge['subject_label']} → {edge['object_label']}")
    tags = edge.get("tags") or []
    if tags:
        lines.append("Tags: " + ", ".join(tags))
    for source in edge.get("sources", []):
        document = source.get("document")
        page = source.get("page_label")
        if document:
            page_suffix = f" ({page})" if page else ""
            lines.append(f"Doc: {document}{page_suffix}")
        else:
            source_type = source.get("type") or "unknown"
            lines.append(f"Source: {source_type}")
    return "<br/>".join(filter(None, lines))
