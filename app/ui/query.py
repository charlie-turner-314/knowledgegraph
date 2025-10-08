from __future__ import annotations

import streamlit as st

from app.data.db import session_scope
from app.services import QueryResult, QueryService


def render() -> None:
    """Render the natural language query chat interface."""
    st.header("Natural Language Query")
    st.write("Ask questions about the curated knowledge graph. Try: 'What equipment is used in mine ventilation?' or 'What are the main risks associated with the Boltec'")

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
    """Render a QueryResult using Streamlit widgets for proper formatting, and return a summary for chat history."""
    summary_lines = []
    if result.answers:
        for answer in result.answers:
            st.markdown(answer)
        summary_lines.extend(result.answers)
    else:
        st.info("No direct answers found, raw response:")
        st.json(result)
        summary_lines.append("No direct answers found. See sources below.")
    if result.matches:
        st.markdown("**Sources:**")
        seen = {}
        for match in result.matches:
            doc_title = match.source_document or "(unknown document)"
            page = match.page_label or ""
            raw_text = match.raw_text or "(no raw text available)"
            key = (doc_title, page, raw_text)
            if key not in seen:
                seen[key] = []
            seen[key].append(match)
        for idx, ((doc_title, page, raw_text), matches) in enumerate(seen.items(), 1):
            page_str = f" (page {page})" if page else ""
            badge_label = f"{doc_title}{page_str}"
            fact_summaries = [f"{m.subject} — {m.predicate} — {m.object}" for m in matches]
            facts_md = "\n".join(f"- {fact}" for fact in fact_summaries)
            with st.expander(f"{badge_label}"):
                st.markdown("**Facts:**")
                st.markdown(facts_md)
                st.markdown("**Source text:**")
                st.write(raw_text)
        summary_lines.append(f"See above for {len(seen)} supporting source(s).")
    if result.note:
        st.markdown(f"_{result.note}_")
        summary_lines.append(result.note)
    return "\n".join(summary_lines)
