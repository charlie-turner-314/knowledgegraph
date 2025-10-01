from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st

from app.data.db import session_scope
from app.services import ExtractionOrchestrator
from app.utils.files import store_uploaded_file


def render() -> None:
    st.header("Document Ingestion")
    st.write(
        "Upload a source file to extract candidate triples. "
        "All information remains local until approved by the SME."
    )

    uploaded_file = st.file_uploader(
        "Select a document",
        type=["pdf", "docx", "pptx", "txt", "md"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return

    st.caption(f"Selected file: {uploaded_file.name}")
    process = st.button("Extract candidate triples", type="primary")

    if not process:
        return

    with st.spinner("Parsing document and querying Gemma…"):
        stored_path = store_uploaded_file(uploaded_file.name, uploaded_file)
        ingestion_summary = _ingest_from_path(stored_path)

    st.success("Ingestion complete")
    if ingestion_summary.get("existing_document"):
        st.info("Document was previously ingested; generated a fresh set of candidates.")
    duplicate_count = ingestion_summary.get("duplicates", 0)
    if duplicate_count:
        st.warning(
            f"{duplicate_count} potential duplicate triple(s) detected. Review tab has them highlighted."
        )
    st.json(ingestion_summary)


def _ingest_from_path(path: Path) -> Dict[str, Any]:
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
