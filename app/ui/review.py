from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.data.db import session_scope
from app.services.review_service import ReviewService, serialize_candidate


def render() -> None:
    st.header("SME Review Queue")
    st.write("Approve, reject, or refine candidate triples before they enter the graph.")

    if flash := st.session_state.pop("review_flash", None):
        severity = flash.get("type", "info")
        message = flash.get("message", "")
        if severity == "success":
            st.success(message)
        elif severity == "error":
            st.error(message)
        else:
            st.info(message)

    pending = _load_pending_candidates()
    if not pending:
        st.info("No pending triples. Ingest a document to generate candidates.")
        return

    for candidate in pending:
        header = f"#{candidate['id']} — {candidate['subject']} / {candidate['predicate']} / {candidate['object']}"
        with st.expander(header, expanded=False):
            st.caption(
                f"Document: {candidate['document']}"
                + (f" — Page: {candidate['page_label']}" if candidate['page_label'] else "")
            )
            if candidate.get("is_duplicate"):
                duplicate_hint = (
                    f"Potential duplicate of candidate #{candidate['duplicate_of']}"
                    if candidate.get("duplicate_of")
                    else "Potential duplicate based on matching triple values"
                )
                st.warning(duplicate_hint)
            if candidate.get("chunk_text"):
                st.write(candidate["chunk_text"][:1000])

            subject_key = f"subject_{candidate['id']}"
            predicate_key = f"predicate_{candidate['id']}"
            object_key = f"object_{candidate['id']}"
            notes_key = f"notes_{candidate['id']}"

            subject_value = st.text_input("Subject", candidate["subject"], key=subject_key)
            predicate_value = st.text_input("Predicate", candidate["predicate"], key=predicate_key)
            object_value = st.text_input("Object", candidate["object"], key=object_key)
            notes_value = st.text_area("Notes", key=notes_key)

            col1, col2 = st.columns(2)
            if col1.button("Approve", key=f"approve_{candidate['id']}"):
                _resolve_candidate(
                    action="approve",
                    candidate_id=candidate["id"],
                    payload={
                        "subject": subject_value,
                        "predicate": predicate_value,
                        "object": object_value,
                        "notes": notes_value,
                    },
                )
            if col2.button("Reject", key=f"reject_{candidate['id']}"):
                _resolve_candidate(
                    action="reject",
                    candidate_id=candidate["id"],
                    payload={"reason": notes_value or "Rejected via UI"},
                )
            if candidate.get("is_duplicate"):
                if st.button(
                    "Dismiss duplicate",
                    key=f"dismiss_{candidate['id']}",
                    help="Mark this candidate as a duplicate and remove it from the queue.",
                ):
                    _resolve_candidate(
                        action="dismiss_duplicate",
                        candidate_id=candidate["id"],
                        payload={"reason": "Dismissed as duplicate"},
                    )


def _load_pending_candidates() -> List[Dict[str, Any]]:
    with session_scope() as session:
        service = ReviewService(session)
        pending = service.list_pending(limit=25)
        serialized = [serialize_candidate(item) for item in pending]
    return serialized


def _resolve_candidate(*, action: str, candidate_id: int, payload: Dict[str, Any]) -> None:
    try:
        with session_scope() as session:
            service = ReviewService(session)
            if action == "approve":
                service.approve_candidate(
                    candidate_id,
                    actor="sme-local",
                    subject_override=payload.get("subject"),
                    predicate_override=payload.get("predicate"),
                    object_override=payload.get("object"),
                    notes=payload.get("notes"),
                )
            elif action == "reject":
                service.reject_candidate(candidate_id, actor="sme-local", reason=payload.get("reason"))
            elif action == "dismiss_duplicate":
                service.reject_candidate(candidate_id, actor="sme-local", reason=payload.get("reason"))
            else:
                raise ValueError(f"Unknown action {action}")
    except Exception as exc:  # noqa: BLE001
        st.session_state["review_flash"] = {"type": "error", "message": str(exc)}
    else:
        messages = {
            "approve": "Candidate approved",
            "reject": "Candidate rejected",
            "dismiss_duplicate": "Candidate dismissed as duplicate",
        }
        st.session_state["review_flash"] = {
            "type": "success",
            "message": messages.get(action, "Candidate updated"),
        }
    st.rerun()
