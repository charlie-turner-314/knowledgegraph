from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from app.data.db import session_scope
from app.services import ExtractionOrchestrator
from app.utils.files import store_uploaded_file


def render() -> None:
    """Render the document ingestion interface for files and raw text."""
    st.header("Document Ingestion")
    st.write(
        "Provide knowledge via document upload or pasted text. "
        "Extracted statements are written to the knowledge graph immediately with provenance back to the source document."
    )

    mode = st.radio(
        "Ingestion method",
        ["File upload", "Manual text"],
        horizontal=True,
        key="ingestion_mode",
    )

    if mode == "File upload":
        _render_file_ingest()
    else:
        _render_text_ingest()


def _ingest_from_path(
    path: Path,
    *,
    plan_placeholder=None,
    chunk_progress=None,
) -> Dict[str, Any]:
    """Ingest a file from disk and return a summary dictionary."""
    plan_state: Dict[str, Any] = {}

    def _on_plan(plan: Dict[str, Any]) -> None:
        nonlocal plan_state
        plan_state = plan
        message = (
            f"Planned extraction: {plan['total_chunks']} chunk(s), "
            f"{plan['expected_llm_calls']} LLM call(s), "
            f"{plan['expected_embedding_batches']} embedding batch(es)."
        )
        if plan_placeholder is not None:
            plan_placeholder.info(message)
        else:
            st.info(message)
        if chunk_progress is not None:
            if plan["total_chunks"]:
                chunk_progress.progress(0.0, text="Starting chunk extraction…")
            else:
                chunk_progress.progress(1.0, text="No chunks to process")

    def _on_progress(done: int, total: int) -> None:
        if chunk_progress is None:
            return
        if not total:
            chunk_progress.progress(1.0, text="No chunks to process")
            return
        fraction = min(max(done / total, 0.0), 1.0)
        chunk_progress.progress(fraction, text=f"LLM extraction {done}/{total} chunk(s)")

    with session_scope() as session:
        orchestrator = ExtractionOrchestrator(session)
        result = orchestrator.ingest_file(
            path,
            plan_callback=_on_plan,
            progress_callback=_on_progress,
        )
        session.flush()

        if chunk_progress is not None:
            chunk_progress.progress(1.0, text="Extraction complete")

        duplicate_count = sum(1 for c in result.candidate_triples if c.is_potential_duplicate)
        summary = {
            "document": {
                "id": result.document.id,
                "name": result.document.display_name,
                "media_type": result.document.media_type,
                "chunks": len(result.chunks),
            },
            "candidates": len(result.candidate_triples),
            "edges": len(result.edges),
            "duplicates": duplicate_count,
            "existing_document": result.was_existing_document,
            "stats": result.stats,
        }
    if plan_placeholder is not None:
        plan_placeholder.empty()
    if chunk_progress is not None:
        chunk_progress.empty()
    return summary


def _ingest_from_text(
    text: str,
    title: str,
    *,
    plan_placeholder=None,
    chunk_progress=None,
) -> Dict[str, Any]:
    """Ingest manually supplied ``text`` and return the resulting summary."""

    def _on_plan(plan: Dict[str, Any]) -> None:
        message = (
            f"Planned extraction: {plan['total_chunks']} chunk(s), "
            f"{plan['expected_llm_calls']} LLM call(s), "
            f"{plan['expected_embedding_batches']} embedding batch(es)."
        )
        if plan_placeholder is not None:
            plan_placeholder.info(message)
        else:
            st.info(message)
        if chunk_progress is not None:
            if plan["total_chunks"]:
                chunk_progress.progress(0.0, text="Starting chunk extraction…")
            else:
                chunk_progress.progress(1.0, text="No chunks to process")

    def _on_progress(done: int, total: int) -> None:
        if chunk_progress is None:
            return
        if not total:
            chunk_progress.progress(1.0, text="No chunks to process")
            return
        fraction = min(max(done / total, 0.0), 1.0)
        chunk_progress.progress(fraction, text=f"LLM extraction {done}/{total} chunk(s)")

    with session_scope() as session:
        orchestrator = ExtractionOrchestrator(session)
        result = orchestrator.ingest_text(
            text=text,
            title=title,
            plan_callback=_on_plan,
            progress_callback=_on_progress,
        )
        session.flush()

        if chunk_progress is not None:
            chunk_progress.progress(1.0, text="Extraction complete")

        duplicate_count = sum(1 for c in result.candidate_triples if c.is_potential_duplicate)
        summary = {
            "document": {
                "id": result.document.id,
                "name": result.document.display_name,
                "media_type": result.document.media_type,
                "chunks": len(result.chunks),
            },
            "candidates": len(result.candidate_triples),
            "edges": len(result.edges),
            "duplicates": duplicate_count,
            "existing_document": result.was_existing_document,
            "stats": result.stats,
        }
    if plan_placeholder is not None:
        plan_placeholder.empty()
    if chunk_progress is not None:
        chunk_progress.empty()
    return summary


def _render_file_ingest() -> None:
    """Render the file upload form and handle submissions."""
    with st.form("file_ingest_form"):
        uploaded_files = st.file_uploader(
            "Select one or more documents",
            type=["pdf", "docx", "pptx", "txt", "md"],
            accept_multiple_files=True,
            help="You can drag-and-drop multiple files here.",
        )
        submitted = st.form_submit_button("Extract candidate triples")

    if not submitted:
        return

    if not uploaded_files:
        st.error("Please select at least one file before submitting.")
        return

    progress = st.progress(0, text="Starting ingestion…")
    summaries = []
    total = len(uploaded_files)
    for idx, file in enumerate(uploaded_files, 1):
        progress.progress((idx - 0.5) / total, text=f"Processing {file.name} ({idx}/{total})…")
        try:
            stored_path = store_uploaded_file(file.name, file)
            plan_placeholder = st.empty()
            chunk_progress = st.progress(0.0, text="Preparing extraction…")
            summary = _ingest_from_path(
                stored_path,
                plan_placeholder=plan_placeholder,
                chunk_progress=chunk_progress,
            )
            summaries.append((file.name, summary))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to ingest {file.name}: {exc}")
        progress.progress(idx / total, text=f"Completed {file.name} ({idx}/{total})")
    progress.empty()

    if not summaries:
        st.error("No documents were successfully ingested.")
        return

    # Aggregate metrics
    total_candidates = sum(s[1].get("candidates", 0) for s in summaries)
    total_edges = sum(s[1].get("edges", 0) for s in summaries)
    total_duplicates = sum(s[1].get("duplicates", 0) for s in summaries)
    existing_count = sum(1 for s in summaries if s[1].get("existing_document"))

    st.success(f"Ingestion complete — {len(summaries)} document(s) processed.")
    if existing_count:
        st.info(f"{existing_count} document(s) were previously ingested; new statements reconciled.")

    agg_cols = st.columns(3)
    agg_cols[0].metric("Candidate triples analysed", total_candidates)
    agg_cols[1].metric("Graph statements created", total_edges)
    agg_cols[2].metric("Potential duplicates skipped", total_duplicates)

    st.markdown("### Per-document results")
    for name, summary in summaries:
        with st.expander(f"{name} (ID: {summary['document']['id']})"):
            _render_ingest_summary(summary)


def _render_text_ingest() -> None:
    """Render the manual text ingestion form and handle submissions."""
    with st.form("text_ingest_form"):
        title = st.text_input("Document title", value="Manual Entry")
        text_input = st.text_area(
            "Paste text",
            height=220,
            help="Provide raw text to extract candidate triples from.",
        )
        submitted = st.form_submit_button("Extract candidate triples")

    if not submitted:
        return

    if not text_input.strip():
        st.error("Please enter some text before submitting.")
        return

    with st.spinner("Processing text and querying the LLM…"):
        ingestion_summary = _ingest_from_text(text_input, title)

    _render_ingest_summary(ingestion_summary)


def _render_ingest_summary(ingestion_summary: Dict[str, Any]) -> None:
    """Display feedback and stats produced by the ingestion pipeline."""
    # This helper is now used both for single and multi ingest; avoid repeating high-level banners.
    if ingestion_summary.get("existing_document"):
        st.caption("Previously ingested — reconciled with existing graph.")

    metrics = st.columns(3)
    metrics[0].metric("Candidates", ingestion_summary.get("candidates", 0))
    metrics[1].metric("Edges", ingestion_summary.get("edges", 0))
    metrics[2].metric("Chunks", ingestion_summary.get("document", {}).get("chunks", 0))
    duplicate_count = ingestion_summary.get("duplicates", 0)
    if duplicate_count:
        st.warning(f"Skipped {duplicate_count} potential duplicate triple(s).")
