from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.data import models
from app.data.db import session_scope
from sqlmodel import select

try:
    import networkx as nx
    from pyvis.network import Network
    import streamlit.components.v1 as components
except ImportError:  # pragma: no cover
    nx = None
    Network = None
    components = None


def render() -> None:
    st.header("Knowledge Graph Visualization")
    st.write("Interact with the approved knowledge graph edges and trace their provenance.")

    if not nx or not Network or not components:
        st.error("Graph visualization dependencies missing. Install networkx and pyvis.")
        return

    documents = _load_documents()
    doc_lookup = {doc["name"]: doc["id"] for doc in documents}
    selected_docs = st.multiselect("Filter by document", list(doc_lookup.keys()))
    doc_ids = [doc_lookup[name] for name in selected_docs]

    edges = _load_edges(document_ids=doc_ids)
    if not edges:
        st.info("No approved edges yet. Approve triples to populate the graph.")
        return

    graph = nx.DiGraph()
    for edge in edges:
        graph.add_node(edge["subject"], label=edge["subject"])
        graph.add_node(edge["object"], label=edge["object"])
        graph.add_edge(
            edge["subject"],
            edge["object"],
            label=edge["predicate"],
            title=_format_edge_tooltip(edge),
        )

    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(graph)

    html = net.generate_html(notebook=False)
    components.html(html, height=650, scrolling=True)


def _load_documents() -> List[Dict[str, Any]]:
    with session_scope() as session:
        rows = session.exec(select(models.Document)).all()
        return [
            {"id": row.id, "name": row.display_name}
            for row in rows
        ]


def _load_edges(*, document_ids: List[int]) -> List[Dict[str, Any]]:
    from sqlalchemy.orm import selectinload

    with session_scope() as session:
        stmt = (
            select(models.Edge)
            .options(
                selectinload(models.Edge.subject),
                selectinload(models.Edge.object),
                selectinload(models.Edge.sources)
                .selectinload(models.EdgeSource.document_chunk)
                .selectinload(models.DocumentChunk.document),
            )
        )
        if document_ids:
            stmt = stmt.where(
                models.Edge.id.in_(
                    select(models.EdgeSource.edge_id)
                    .join(models.DocumentChunk)
                    .where(models.DocumentChunk.document_id.in_(document_ids))
                )
            )
        rows = session.exec(stmt).all()

        edges: List[Dict[str, Any]] = []
        for edge in rows:
            subject = edge.subject.label if edge.subject else str(edge.subject_node_id)
            obj = edge.object.label if edge.object else str(edge.object_node_id)
            sources = []
            for source in edge.sources:
                doc = None
                if source.document_chunk and source.document_chunk.document:
                    doc = source.document_chunk.document.display_name
                sources.append({
                    "source_type": source.source_type.value,
                    "document": doc,
                    "page_label": source.document_chunk.page_label if source.document_chunk else None,
                })
            edges.append(
                {
                    "subject": subject,
                    "predicate": edge.predicate,
                    "object": obj,
                    "sources": sources,
                }
            )
        return edges


def _format_edge_tooltip(edge: Dict[str, Any]) -> str:
    parts = [edge["predicate"]]
    for source in edge.get("sources", []):
        if source["document"]:
            parts.append(f"Doc: {source['document']} ({source.get('page_label')})")
        else:
            parts.append(source["source_type"])
    return "<br/>".join(filter(None, parts))
