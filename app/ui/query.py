from __future__ import annotations

import streamlit as st

from app.data.db import session_scope
from app.services import QueryResult, QueryService


def render() -> None:
    st.header("Natural Language Query")
    st.write("Ask questions about the curated knowledge graph. An LLM now plans traversals and falls back to keyword search only when needed.")

    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []

    for entry in st.session_state["query_history"]:
        with st.chat_message(entry["role"]):
            if isinstance(entry["content"], str):
                st.markdown(entry["content"])
            else:
                st.write(entry["content"])

    prompt = st.chat_input("What would you like to know?")
    if not prompt:
        return

    st.session_state["query_history"].append({"role": "user", "content": prompt})

    status_placeholder = st.empty()

    def _progress(message: str) -> None:
        status_placeholder.info(f"{message}")

    with session_scope() as session:
        service = QueryService(session)
        with st.spinner("Thinking..."):
            result = service.ask(prompt, progress_callback=_progress)

    status_placeholder.success("Answer ready")

    response_content = _format_query_result(result)
    st.session_state["query_history"].append({"role": "assistant", "content": response_content})
    st.rerun()


def _format_query_result(result: QueryResult) -> str:
    lines = []
    if result.answers:
        lines.extend(result.answers)
    if result.matches:
        lines.append("\n**Supporting triples:**")
        for match in result.matches:
            doc_part = f" — {match.source_document}" if match.source_document else ""
            page_part = f" (page {match.page_label})" if match.page_label else ""
            edge_part = f" [edge #{match.edge_id}]" if match.edge_id is not None else ""
            meta_parts = []
            if match.tags:
                meta_parts.append(f"tags: {', '.join(match.tags)}")
            if match.subject_attributes:
                meta_parts.append(
                    "subject attrs: "
                    + ", ".join(
                        f"{key}={value}" for key, value in match.subject_attributes.items()
                    )
                )
            if match.object_attributes:
                meta_parts.append(
                    "object attrs: "
                    + ", ".join(
                        f"{key}={value}" for key, value in match.object_attributes.items()
                    )
                )
            meta_part = f" ({'; '.join(meta_parts)})" if meta_parts else ""
            lines.append(
                f"• {match.subject} — {match.predicate} — {match.object}{doc_part}{page_part}{edge_part}{meta_part}"
            )
    if result.note:
        lines.append(f"\n_{result.note}_")
    return "\n".join(lines)
