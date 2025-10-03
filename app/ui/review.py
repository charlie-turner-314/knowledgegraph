from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from app.data.db import session_scope
from app.services.review_service import ReviewService
from app.llm.client import get_client


def render() -> None:
    """Render the SME review queue UI."""
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

    pending = _load_pending_serialized()
    if not pending:
        st.info("No pending triples. Ingest a document to generate candidates.")
        return

    _render_collaborative_conversation(pending)


def _load_pending_serialized() -> List[Dict[str, Any]]:
    """Load serialised pending candidates for the summary workflow."""
    with session_scope() as session:
        service = ReviewService(session)
        return service.serialize_pending(limit=50)


def _resolve_candidate(*, action: str, candidate_id: int, payload: Dict[str, Any]) -> None:
    """Execute reviewer actions and update the review flash state."""
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
                    subject_attributes=payload.get("subject_attributes"),
                    object_attributes=payload.get("object_attributes"),
                    tags=payload.get("tags"),
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


def _render_collaborative_conversation(candidates: List[Dict[str, Any]]) -> None:
    st.markdown(
        f"{len(candidates)} candidate statements awaiting validation. Engage with the assistant below to review and correct the extracted knowledge before committing it to the knowledge graph."
    )

    history: List[Dict[str, str]] = st.session_state.setdefault("collab_history", [])
    summary_cache: Optional[str] = st.session_state.get("collab_summary")

    if history:
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.info("Start by requesting a summary of what the assistant learned from the uploaded materials.")

    cols = st.columns(2)
    if cols[0].button("Generate summary & questions", key="collab_generate"):
        message = _generate_review_summary(candidates, history)
        history.append({"role": "assistant", "content": message})
        st.session_state["collab_history"] = history
        st.session_state["collab_summary"] = message
        st.rerun()

    if cols[1].button("Clear conversation", key="collab_clear"):
        st.session_state.pop("collab_history", None)
        st.session_state.pop("collab_input", None)
        st.session_state.pop("collab_summary", None)
        st.rerun()

    user_input = st.text_area("Your response", key="collab_input")
    if st.button("Send", key="collab_send"):
        if user_input.strip():
            history.append({"role": "user", "content": user_input.strip()})
            st.session_state["collab_history"] = history
            st.session_state["collab_input"] = ""
            st.rerun()
        else:
            st.warning("Please enter a response before sending.")

    summary = summary_cache or _conversation_summary(history)
    if summary:
        st.caption("Working summary (will be stored when committed):")
        st.write(summary)

    action_cols = st.columns([0.4, 0.4, 0.2])
    if action_cols[0].button("Commit to knowledge graph", key="collab_commit"):
        _finalize_candidates(candidates, summary)
    if action_cols[1].button("Mark as needs evidence", key="collab_needevidence"):
        _mark_conversation_needs_evidence(candidates, summary)
    with action_cols[2]:
        st.caption("Conversation state stored automatically.")


def _generate_review_summary(
    candidates: List[Dict[str, Any]], history: List[Dict[str, str]]
) -> str:
    client = get_client()
    payload = [
        {
            "subject": item["subject"],
            "predicate": item["predicate"],
            "object": item["object"],
            "chunk_text": item.get("chunk_text"),
            "subject_attributes": item.get("subject_attributes") or [],
            "object_attributes": item.get("object_attributes") or [],
            "tags": item.get("tags") or [],
        }
        for item in candidates
    ]
    return client.generate_review_summary(candidates=payload, history=history)


def _conversation_summary(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = [f"{message['role']}: {message['content']}" for message in history]
    return "\n".join(lines)


def _finalize_candidates(candidates: List[Dict[str, Any]], summary: str) -> None:
    with session_scope() as session:
        service = ReviewService(session)
        service.approve_candidates_bulk(candidates, actor="sme-local", summary=summary or None)
    st.session_state.pop("collab_history", None)
    st.session_state.pop("collab_summary", None)
    st.session_state["review_flash"] = {"type": "success", "message": "Knowledge graph updated."}
    st.rerun()


def _mark_conversation_needs_evidence(candidates: List[Dict[str, Any]], summary: str) -> None:
    with session_scope() as session:
        service = ReviewService(session)
        for candidate in candidates:
            service.flag_candidate_needs_evidence(
                candidate_id=candidate["id"],
                actor="sme-local",
                rationale=summary or "Additional evidence required",
                confidence=candidate.get("confidence") or None,
            )
    st.session_state.pop("collab_history", None)
    st.session_state.pop("collab_summary", None)
    st.session_state["review_flash"] = {
        "type": "info",
        "message": "Statements flagged for follow-up evidence."
    }
    st.rerun()
