from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.data.db import session_scope
from app.services import AdminService, GraphInsightService


def render() -> None:
    """Render the admin dashboard with maintenance, document, and vocabulary tools."""
    st.header("Administration")
    st.caption("Manage canonical vocabulary and uploaded documents.")

    _render_flash()
    _render_pending_confirmations()

    st.subheader("Graph Maintenance")
    _render_graph_scan_section()

    st.subheader("Documents")
    _render_documents_section()

    st.divider()
    st.subheader("Canonical Vocabulary")
    _render_canonical_terms_section()


# ---------------------------------------------------------------------------
# Flash + confirmations


def _render_flash() -> None:
    """Display any queued admin flash message."""
    flash = st.session_state.pop("admin_flash", None)
    if not flash:
        return
    level = flash.get("type", "info")
    message = flash.get("message", "")
    if level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def _render_pending_confirmations() -> None:
    """Render confirmation prompts for delete operations when required."""
    if (doc_id := st.session_state.get("admin_confirm_delete_doc")) is not None:
        doc_name = st.session_state.get("admin_confirm_delete_doc_name", "this document")
        with st.warning(
            f"Are you sure you want to delete '{doc_name}' and all associated knowledge?"
        ):
            col1, col2 = st.columns(2)
            if col1.button("Confirm delete", key="confirm_delete_doc"):
                _delete_document(doc_id)
            if col2.button("Cancel", key="cancel_delete_doc"):
                _clear_doc_confirmation()

    if (term_id := st.session_state.get("admin_confirm_delete_term")) is not None:
        term_label = st.session_state.get("admin_confirm_delete_term_label", "this term")
        with st.warning(
            f"Delete canonical term '{term_label}'? Nodes referencing it will lose canonical links."
        ):
            col1, col2 = st.columns(2)
            if col1.button("Delete term", key="confirm_delete_term"):
                _delete_canonical_term(term_id)
            if col2.button("Cancel delete term", key="cancel_delete_term"):
                _clear_term_confirmation()


def _render_graph_scan_section() -> None:
    """Render the graph maintenance controls and display scan results."""
    results_key = "admin_graph_scan_results"
    message_key = "admin_graph_scan_message"

    default_threshold = st.session_state.get("admin_graph_scan_threshold", 0.82)
    default_top_k = st.session_state.get("admin_graph_scan_top_k", 6)
    default_max_clusters = st.session_state.get("admin_graph_scan_max_clusters", 10)

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.5,
        max_value=0.95,
        value=float(default_threshold),
        step=0.01,
        help="Minimum cosine similarity required for nodes to be grouped. Lower this if you have sparse data.",
    )
    cols = st.columns(2)
    top_k = int(
        cols[0].number_input(
            "Neighbors per node",
            min_value=2,
            max_value=25,
            value=int(default_top_k),
            step=1,
            help="How many nearest labels to compare for each node when forming clusters.",
        )
    )
    max_clusters = int(
        cols[1].number_input(
            "Max clusters to review",
            min_value=1,
            max_value=30,
            value=int(default_max_clusters),
            step=1,
            help="Cap on the number of candidate clusters returned in a single scan.",
        )
    )

    scan_button = st.button("Scan and recommend graph connections")

    if scan_button:
        with st.spinner("Analyzing graph clusters..."):
            with session_scope() as session:
                insight_service = GraphInsightService(session)
                recommendations = insight_service.scan_for_recommendations(
                    similarity_threshold=float(threshold),
                    top_k=top_k,
                    max_clusters=max_clusters,
                )
                st.session_state[results_key] = [
                    {
                        "node_summaries": item.node_summaries,
                        "existing_connections": item.existing_connections,
                        "missing_pairs": item.missing_pairs,
                        "proposed_nodes": item.proposed_nodes,
                        "proposed_edges": item.proposed_edges,
                        "notes": item.notes,
                    }
                    for item in recommendations
                ]
                st.session_state["admin_graph_scan_threshold"] = float(threshold)
                st.session_state["admin_graph_scan_top_k"] = top_k
                st.session_state["admin_graph_scan_max_clusters"] = max_clusters
                st.session_state[message_key] = (
                    f"Scan complete: evaluated {len(recommendations)} cluster(s) "
                    f"(threshold {threshold:.2f}, neighbors {top_k}, max clusters {max_clusters})."
                )
        st.rerun()

    if message_key in st.session_state:
        st.success(st.session_state.pop(message_key))

    results = st.session_state.get(results_key, [])
    if not results:
        st.caption("No scan results yet. Click the button above to analyze the graph.")
        return

    for idx, item in enumerate(results, start=1):
        with st.expander(f"Cluster recommendation #{idx}", expanded=False):
            st.write("**Cluster nodes**")
            for summary in item.get("node_summaries", []):
                attrs = summary.get("attributes") or []
                attr_text = ", ".join(
                    f"{attr['name']}: {attr.get('value')}" for attr in attrs
                ) if attrs else "(no attributes)"
                st.write(f"- {summary['label']} — {attr_text}")

            existing = item.get("existing_connections") or []
            if existing:
                st.write("**Existing connections**")
                for edge in existing:
                    st.write(
                        f"- {edge['subject']} — {edge['predicate']} — {edge['object']}"
                    )
            missing_pairs = item.get("missing_pairs") or []
            if missing_pairs:
                st.write("**Unconnected pairs**")
                for source, target in missing_pairs:
                    st.write(f"- {source} ↔ {target}")

            proposed_nodes = item.get("proposed_nodes") or []
            proposed_edges = item.get("proposed_edges") or []
            if proposed_nodes:
                st.write("**LLM proposed nodes**")
                for node in proposed_nodes:
                    description = node.get("description") or ""
                    st.write(f"- {node.get('label')} — {description}")
            if proposed_edges:
                st.write("**LLM proposed edges**")
                for edge in proposed_edges:
                    rationale = edge.get("rationale") or ""
                    st.write(
                        f"- {edge.get('subject')} — {edge.get('predicate')} — {edge.get('object')}"
                        + (f" ({rationale})" if rationale else "")
                    )
            if item.get("notes"):
                st.info(item["notes"])


# ---------------------------------------------------------------------------
# Document management


def _render_documents_section() -> None:
    """List ingested documents with stats and management controls."""
    documents = _load_documents()
    if not documents:
        st.info("No documents ingested yet.")
        return

    for doc in documents:
        with st.expander(f"{doc['name']} ({doc['media_type']})", expanded=False):
            st.write(f"Uploaded: {doc['created_at']}")
            st.write(
                f"Chunks: {doc['chunk_count']} | Candidates: {doc['candidate_count']} | Approved edges: {doc['edge_count']}"
            )
            delete_key = f"delete_doc_{doc['id']}"
            if st.button("Delete document", key=delete_key):
                st.session_state["admin_confirm_delete_doc"] = doc["id"]
                st.session_state["admin_confirm_delete_doc_name"] = doc["name"]
                st.rerun()


def _load_documents() -> List[Dict[str, Any]]:
    """Return document metadata dictionaries for the admin view."""
    with session_scope() as session:
        svc = AdminService(session)
        documents = svc.list_documents()
    return [doc.__dict__ for doc in documents]


def _delete_document(document_id: int) -> None:
    """Delete the document identified by ``document_id`` and refresh the UI."""
    with session_scope() as session:
        svc = AdminService(session)
        try:
            svc.delete_document(document_id)
        except Exception as exc:  # noqa: BLE001
            st.session_state["admin_flash"] = {"type": "error", "message": str(exc)}
        else:
            st.session_state["admin_flash"] = {"type": "success", "message": "Document deleted."}
    _clear_doc_confirmation()
    st.rerun()


def _clear_doc_confirmation() -> None:
    """Remove stored confirmation state after cancelling or completing deletion."""
    st.session_state.pop("admin_confirm_delete_doc", None)
    st.session_state.pop("admin_confirm_delete_doc_name", None)


# ---------------------------------------------------------------------------
# Canonical term management


def _render_canonical_terms_section() -> None:
    """Render controls that manage canonical vocabulary entries."""
    with st.expander("Add new canonical term", expanded=False):
        with st.form("add_canonical_term_form"):
            label = st.text_input("Preferred label", key="admin_new_term_label")
            entity_type = st.text_input("Entity type", key="admin_new_term_entity")
            aliases_raw = st.text_input(
                "Aliases (comma separated)",
                key="admin_new_term_aliases",
                help="Optional list of synonyms to seed for this term.",
            )
            submitted = st.form_submit_button("Create term")
            if submitted:
                if not label.strip():
                    st.error("Label is required.")
                else:
                    aliases = [alias.strip() for alias in aliases_raw.split(",") if alias.strip()]
                    with session_scope() as session:
                        svc = AdminService(session)
                        svc.create_canonical_term(
                            label=label.strip(),
                            entity_type=entity_type.strip() or None,
                            aliases=aliases or None,
                        )
                        st.session_state["admin_flash"] = {
                            "type": "success",
                            "message": f"Canonical term '{label.strip()}' created.",
                        }
                    st.rerun()

    terms = _load_terms()
    if not terms:
        st.info("No canonical terms configured yet.")
        return

    for term in terms:
        aliases = term.get("aliases", []) or []
        with st.expander(term["label"], expanded=False):
            st.write(f"Entity type: {term.get('entity_type') or '—'}")
            if aliases:
                st.write("Aliases: " + ", ".join(aliases))
            else:
                st.write("Aliases: none")

            alias_input_key = f"alias_input_{term['id']}"
            new_alias = st.text_input(
                "Add alias",
                key=alias_input_key,
                label_visibility="collapsed",
                placeholder="Add new alias",
            )
            if st.button("Add alias", key=f"add_alias_{term['id']}"):
                if not new_alias.strip():
                    st.session_state["admin_flash"] = {
                        "type": "error",
                        "message": "Alias cannot be empty.",
                    }
                else:
                    _add_alias(term["id"], new_alias.strip())
                # Clear the input for this alias field
                st.session_state[alias_input_key] = ""
                st.rerun()

            if aliases:
                st.caption("Existing aliases")
                for alias in aliases:
                    cols = st.columns([0.8, 0.2])
                    cols[0].write(alias)
                    if cols[1].button(
                        "Remove",
                        key=f"remove_alias_{term['id']}_{alias}",
                    ):
                        _remove_alias(term["id"], alias)
                        st.rerun()

            col1, col2 = st.columns(2)
            if col1.button("Delete term", key=f"delete_term_{term['id']}"):
                st.session_state["admin_confirm_delete_term"] = term["id"]
                st.session_state["admin_confirm_delete_term_label"] = term["label"]
                st.rerun()
            if col2.button("Rename term", key=f"rename_term_{term['id']}"):
                st.session_state["admin_edit_term_id"] = term["id"]
                st.session_state["admin_edit_term_label"] = term["label"]
                st.session_state["admin_edit_term_entity"] = term.get("entity_type") or ""
                st.rerun()

    _render_term_edit_dialog()


def _load_terms() -> List[Dict[str, Any]]:
    """Fetch canonical terms and return them as plain dictionaries."""
    with session_scope() as session:
        svc = AdminService(session)
        terms = svc.list_canonical_terms()
    return [term.__dict__ for term in terms]


def _add_alias(term_id: int, alias: str) -> None:
    """Add ``alias`` to ``term_id`` and surface success feedback."""
    with session_scope() as session:
        svc = AdminService(session)
        svc.add_alias(term_id, alias)
    st.session_state["admin_flash"] = {
        "type": "success",
        "message": f"Alias '{alias}' added.",
    }


def _remove_alias(term_id: int, alias: str) -> None:
    """Remove ``alias`` from ``term_id`` and surface success feedback."""
    with session_scope() as session:
        svc = AdminService(session)
        svc.remove_alias(term_id, alias)
    st.session_state["admin_flash"] = {
        "type": "success",
        "message": f"Alias '{alias}' removed.",
    }


def _render_term_edit_dialog() -> None:
    """Show a modal for renaming a canonical term when requested."""
    term_id = st.session_state.get("admin_edit_term_id")
    if term_id is None:
        return
    default_label = st.session_state.get("admin_edit_term_label", "")
    default_entity = st.session_state.get("admin_edit_term_entity", "")

    with st.modal("Edit canonical term"):
        new_label = st.text_input("Label", value=default_label, key="admin_edit_label_input")
        new_entity = st.text_input("Entity type", value=default_entity, key="admin_edit_entity_input")
        col1, col2 = st.columns(2)
        if col1.button("Save changes", key="admin_save_term"):
            with session_scope() as session:
                svc = AdminService(session)
                try:
                    svc.update_canonical_term(
                        term_id,
                        label=new_label.strip() or default_label,
                        entity_type=new_entity.strip() or None,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.session_state["admin_flash"] = {
                        "type": "error",
                        "message": str(exc),
                    }
                else:
                    st.session_state["admin_flash"] = {
                        "type": "success",
                        "message": "Canonical term updated.",
                    }
            _clear_term_edit_state()
            st.rerun()
        if col2.button("Cancel", key="admin_cancel_term"):
            _clear_term_edit_state()
            st.rerun()


def _delete_canonical_term(term_id: int) -> None:
    """Delete a canonical term and flash the outcome."""
    with session_scope() as session:
        svc = AdminService(session)
        try:
            svc.delete_canonical_term(term_id)
        except Exception as exc:  # noqa: BLE001
            st.session_state["admin_flash"] = {"type": "error", "message": str(exc)}
        else:
            st.session_state["admin_flash"] = {
                "type": "success",
                "message": "Canonical term deleted.",
            }
    _clear_term_confirmation()
    st.rerun()


def _clear_term_confirmation() -> None:
    """Clear stored confirmation state for canonical term deletion."""
    st.session_state.pop("admin_confirm_delete_term", None)
    st.session_state.pop("admin_confirm_delete_term_label", None)


def _clear_term_edit_state() -> None:
    """Reset modal state after renaming a canonical term."""
    st.session_state.pop("admin_edit_term_id", None)
    st.session_state.pop("admin_edit_term_label", None)
    st.session_state.pop("admin_edit_term_entity", None)
    st.session_state.pop("admin_edit_label_input", None)
    st.session_state.pop("admin_edit_entity_input", None)
