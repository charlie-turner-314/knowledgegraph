from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.data.db import session_scope
from app.services import AdminService


def render() -> None:
    st.header("Administration")
    st.caption("Manage canonical vocabulary and uploaded documents.")

    _render_flash()
    _render_pending_confirmations()

    st.subheader("Documents")
    _render_documents_section()

    st.divider()
    st.subheader("Canonical Vocabulary")
    _render_canonical_terms_section()


# ---------------------------------------------------------------------------
# Flash + confirmations


def _render_flash() -> None:
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


# ---------------------------------------------------------------------------
# Document management


def _render_documents_section() -> None:
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
    with session_scope() as session:
        svc = AdminService(session)
        documents = svc.list_documents()
    return [doc.__dict__ for doc in documents]


def _delete_document(document_id: int) -> None:
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
    st.session_state.pop("admin_confirm_delete_doc", None)
    st.session_state.pop("admin_confirm_delete_doc_name", None)


# ---------------------------------------------------------------------------
# Canonical term management


def _render_canonical_terms_section() -> None:
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
    with session_scope() as session:
        svc = AdminService(session)
        terms = svc.list_canonical_terms()
    return [term.__dict__ for term in terms]


def _add_alias(term_id: int, alias: str) -> None:
    with session_scope() as session:
        svc = AdminService(session)
        svc.add_alias(term_id, alias)
    st.session_state["admin_flash"] = {
        "type": "success",
        "message": f"Alias '{alias}' added.",
    }


def _remove_alias(term_id: int, alias: str) -> None:
    with session_scope() as session:
        svc = AdminService(session)
        svc.remove_alias(term_id, alias)
    st.session_state["admin_flash"] = {
        "type": "success",
        "message": f"Alias '{alias}' removed.",
    }


def _render_term_edit_dialog() -> None:
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
    st.session_state.pop("admin_confirm_delete_term", None)
    st.session_state.pop("admin_confirm_delete_term_label", None)


def _clear_term_edit_state() -> None:
    st.session_state.pop("admin_edit_term_id", None)
    st.session_state.pop("admin_edit_term_label", None)
    st.session_state.pop("admin_edit_term_entity", None)
    st.session_state.pop("admin_edit_label_input", None)
    st.session_state.pop("admin_edit_entity_input", None)

