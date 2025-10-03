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
        "All information remains local until approved by the SME."
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


def _ingest_from_path(path: Path) -> Dict[str, Any]:
    """Ingest a file from disk and return a summary dictionary."""
    with session_scope() as session:
        orchestrator = ExtractionOrchestrator(session)
        result = orchestrator.ingest_file(path)
        session.flush()

        duplicate_count = sum(1 for c in result.candidate_triples if c.is_potential_duplicate)
        summary = {
            "document": {
                "id": result.document.id,
                "name": result.document.display_name,
                "media_type": result.document.media_type,
                "chunks": len(result.chunks),
            },
            "candidates": len(result.candidate_triples),
            "duplicates": duplicate_count,
            "existing_document": result.was_existing_document,
        }
    return summary


def _ingest_from_text(text: str, title: str) -> Dict[str, Any]:
    """Ingest manually supplied ``text`` and return the resulting summary."""
    with session_scope() as session:
        orchestrator = ExtractionOrchestrator(session)
        result = orchestrator.ingest_text(text=text, title=title)
        session.flush()

        duplicate_count = sum(1 for c in result.candidate_triples if c.is_potential_duplicate)
        summary = {
            "document": {
                "id": result.document.id,
                "name": result.document.display_name,
                "media_type": result.document.media_type,
                "chunks": len(result.chunks),
            },
            "candidates": len(result.candidate_triples),
            "duplicates": duplicate_count,
            "existing_document": result.was_existing_document,
        }
    return summary


def _render_file_ingest() -> None:
    """Render the file upload form and handle submissions."""
    with st.form("file_ingest_form"):
        uploaded_file = st.file_uploader(
            "Select a document",
            type=["pdf", "docx", "pptx", "txt", "md"],
            accept_multiple_files=False,
        )
        submitted = st.form_submit_button("Extract candidate triples")

    if not submitted:
        return

    if uploaded_file is None:
        st.error("Please select a file before submitting.")
        return

    with st.spinner("Parsing document and querying the LLM…"):
        stored_path = store_uploaded_file(uploaded_file.name, uploaded_file)
        ingestion_summary = _ingest_from_path(stored_path)

    _render_ingest_summary(ingestion_summary)


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
    st.success("Ingestion complete")
    if ingestion_summary.get("existing_document"):
        st.info("Content was previously ingested; generated a fresh set of candidates.")
    duplicate_count = ingestion_summary.get("duplicates", 0)
    if duplicate_count:
        st.warning(
            f"{duplicate_count} potential duplicate triple(s) detected. Review tab has them highlighted."
        )
    st.json(ingestion_summary)
