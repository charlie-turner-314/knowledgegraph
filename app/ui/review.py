from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from app.data.db import session_scope
from app.services.review_service import ReviewChatOutcome, ReviewService


def render() -> None:
    """Review assistant chat mirroring the query interface."""
    st.header("Graph Review Assistant")
    st.write(
        "Discuss proposed edits with the review agent. It can inspect recent statements,"
        " add new connections, and capture provenance directly in the knowledge graph."
    )

    history: List[Dict[str, Any]] = st.session_state.setdefault("review_history", [])

    snapshot = _load_graph_snapshot()
    _render_snapshot_sidebar(snapshot)
    _render_history(history)

    prompt = st.chat_input("Ask the review agent to inspect or adjust the graph…")
    if not prompt:
        return

    history.append({"role": "user", "content": prompt})
    st.session_state["review_history"] = history

    try:
        outcome = _run_review_turn(history)
    except Exception as exc:  # noqa: BLE001
        error_message = f"Review agent failed: {exc}"
        st.session_state["review_history"].append({"role": "assistant", "content": error_message})
        st.error(error_message)
        st.rerun()
        return

    _record_outcome(history, outcome)
    st.session_state["review_history"] = history
    st.rerun()


def _render_snapshot_sidebar(snapshot: Dict[str, Any]) -> None:
    with st.sidebar:
        st.subheader("Recent activity")
        edges = snapshot.get("recent_edges", [])
        if edges:
            for edge in edges[:10]:
                doc_part = f" — {edge['document']}" if edge.get("document") else ""
                page_part = f" (page {edge['page_label']})" if edge.get("page_label") else ""
                tag_part = (
                    f" _(tags: {', '.join(edge['tags'])})_"
                    if edge.get("tags")
                    else ""
                )
                st.markdown(
                    f"• **{edge.get('subject', '⟪unknown⟫')} — {edge.get('predicate', '?')} — {edge.get('object', '⟪unknown⟫')}**"
                    f"{doc_part}{page_part}{tag_part}"
                )
        else:
            st.caption("No recent edges to display yet.")

        nodes = snapshot.get("recent_nodes", [])
        if nodes:
            st.divider()
            st.caption("Latest nodes")
            for node in nodes[:8]:
                attrs = node.get("attributes") or {}
                attr_text = ", ".join(f"{key}={value}" for key, value in attrs.items())
                detail = f" — {attr_text}" if attr_text else ""
                st.write(f"• {node.get('label')} ({node.get('entity_type') or 'unspecified'}){detail}")

        canonical_terms = snapshot.get("canonical_terms", [])
        if canonical_terms:
            st.divider()
            st.caption("Recent canonical terms")
            for term in canonical_terms[:8]:
                aliases = term.get("aliases") or []
                alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
                st.write(f"• {term.get('label')} — {term.get('entity_type') or 'generic'}{alias_text}")


def _render_history(history: List[Dict[str, Any]]) -> None:
    for entry in history:
        role = entry.get("role", "assistant")
        content = entry.get("raw_reply") or entry.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if entry.get("applied_changes"):
                st.caption("Applied changes")
                st.markdown("\n".join(f"- {item}" for item in entry["applied_changes"]))
            if entry.get("commands"):
                with st.expander("LLM command payload", expanded=False):
                    st.json(entry["commands"])


def _conversation_payload(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    payload: List[Dict[str, str]] = []
    for entry in history:
        role = entry.get("role")
        content = entry.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            payload.append({"role": role, "content": content})
    return payload[-12:]


def _run_review_turn(history: List[Dict[str, Any]]) -> ReviewChatOutcome:
    messages = _conversation_payload(history)
    with session_scope() as session:
        service = ReviewService(session)
        with st.spinner("Collaborating with the review agent…"):
            return service.chat(messages=messages, actor="review-ui")


def _record_outcome(history: List[Dict[str, Any]], outcome: ReviewChatOutcome) -> None:
    applied_changes = outcome.applied_changes or []
    commands = [command.model_dump() for command in outcome.commands]
    assistant_content = _assistant_conversation_text(outcome.reply, applied_changes)
    history.append(
        {
            "role": "assistant",
            "content": assistant_content,
            "raw_reply": outcome.reply,
            "applied_changes": applied_changes,
            "commands": commands,
        }
    )


def _assistant_conversation_text(reply: str, applied_changes: List[str]) -> str:
    if not applied_changes:
        return reply
    change_lines = "\n".join(f"- {item}" for item in applied_changes)
    return f"{reply}\n\nApplied changes:\n{change_lines}"


def _load_graph_snapshot() -> Dict[str, Any]:
    with session_scope() as session:
        service = ReviewService(session)
        return service.graph_snapshot()
