from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sqlmodel import Session

from app.data import models
from app.data.repositories import (
    CandidateRepository,
    CanonicalTermRepository,
    DocumentRepository,
    NodeEmbeddingStore
)
from app.ingestion.parsers import parse_document
from app.ingestion.types import ChunkPayload, ParsedDocument
from app.llm.client import GemmaClient, get_client
from app.llm.schemas import ExtractionResponse, ExtractionTriple

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    document: models.Document
    chunks: List[models.DocumentChunk]
    candidate_triples: List[models.CandidateTriple]
    was_existing_document: bool


def _chunk_to_model(document: models.Document, payload: ChunkPayload) -> models.DocumentChunk:
    return models.DocumentChunk(
        document_id=document.id,
        ordering=payload.ordering,
        text=payload.text,
        page_label=payload.page_label,
        token_count=payload.token_count or len(payload.text.split()),
    )


class ExtractionOrchestrator:
    def __init__(self, session: Session, llm_client: Optional[GemmaClient] = None):
        self.session = session
        self.documents = DocumentRepository(session)
        self.candidates = CandidateRepository(session)
        self.canonicals = CanonicalTermRepository(session)
        self.llm_client = llm_client or get_client()
        self._canonical_context_cache: Optional[List[dict]] = None
        self.embedding_store = NodeEmbeddingStore()

    def ingest_file(self, path: Path) -> IngestionResult:
        parsed = parse_document(path)
        return self._ingest_parsed_document(parsed)

    def ingest_text(
        self,
        *,
        text: str,
        title: Optional[str] = None,
        media_type: str = "text/plain",
    ) -> IngestionResult:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text input cannot be empty.")

        checksum = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        parsed = ParsedDocument(
            display_name=title.strip() if title and title.strip() else "Manual Entry",
            media_type=media_type,
            checksum=checksum,
            chunks=[
                ChunkPayload(
                    ordering=0,
                    text=cleaned,
                    page_label="manual",
                    token_count=len(cleaned.split()),
                )
            ],
            original_path=None,
            extra={"source": "manual_input"},
        )
        return self._ingest_parsed_document(parsed)

    def _ingest_parsed_document(self, parsed: ParsedDocument) -> IngestionResult:
        existing_doc = self.documents.get_by_checksum(parsed.checksum)
        was_existing = existing_doc is not None

        if existing_doc:
            document = existing_doc
            if parsed.extra and not document.extra:
                document.extra = parsed.extra
            chunks_models = sorted(document.chunks or [], key=lambda chunk: chunk.ordering)
            if not chunks_models:
                chunks_models = self._persist_chunks(document, parsed)
        else:
            document = self.documents.create_document(
                display_name=parsed.display_name,
                media_type=parsed.media_type,
                checksum=parsed.checksum,
                original_path=str(parsed.original_path) if parsed.original_path else None,
                extra=parsed.extra,
            )
            chunks_models = self._persist_chunks(document, parsed)

        candidates = self._extract_candidates(chunks_models)

        return IngestionResult(
            document=document,
            chunks=chunks_models,
            candidate_triples=candidates,
            was_existing_document=was_existing,
        )

    def _persist_chunks(
        self, document: models.Document, parsed: ParsedDocument
    ) -> List[models.DocumentChunk]:
        chunk_models = []
        for payload in parsed.iter_chunks():
            chunk_models.append(_chunk_to_model(document, payload))
        return self.documents.add_chunks(document, chunk_models)

    def _extract_candidates(
        self, chunks: List[models.DocumentChunk]
    ) -> List[models.CandidateTriple]:
        candidates: List[models.CandidateTriple] = []
        seen_triples = set()
        canonical_context = self._canonical_prompt_context()
        for chunk in chunks:
            response = self._call_extractor(chunk, canonical_context)
            for triple in response.triples:
                key = (
                    triple.subject.strip(),
                    triple.predicate.strip(),
                    triple.object.strip(),
                )
                duplicate_of = self.candidates.find_similar(
                    subject=key[0],
                    predicate=key[1],
                    obj=key[2],
                )
                is_duplicate = duplicate_of is not None or key in seen_triples
                if is_duplicate:
                    logger.info(f"Skipping duplicate triple: {key}")
                    continue

                # Check similarity to exising
                similar_subjects = self.embedding_store.suggest_similar(key[0])
                similar_objects = self.embedding_store.suggest_similar(key[2])


                candidate = self._build_candidate(
                    chunk,
                    triple,
                    response,
                    duplicate_of=duplicate_of,
                    is_duplicate=is_duplicate,
                    similar_objects=similar_objects,
                    similar_subjects=similar_subjects
                )
                self.session.add(candidate)
                candidates.append(candidate)
                seen_triples.add(key)
        self.session.flush()
        return candidates

    def _call_extractor(
        self,
        chunk: models.DocumentChunk,
        canonical_context: List[dict],
    ) -> ExtractionResponse:
        metadata = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "page_label": chunk.page_label,
        }
        try:
            return self.llm_client.extract_triples(
                chunk_text=chunk.text,
                metadata=metadata,
                canonical_context=canonical_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM extraction failed for chunk %s", chunk.id)
            # Return empty response preserving raw error for debugging
            return ExtractionResponse(triples=[], raw_response={"error": str(exc)})

    def _build_candidate(
        self,
        chunk: models.DocumentChunk,
        triple: ExtractionTriple,
        response: ExtractionResponse,
        *,
        duplicate_of: Optional[models.CandidateTriple],
        is_duplicate: bool,
        similar_subjects: list[tuple[str, float]],
        similar_objects: list[tuple[str, float]]
    ) -> models.CandidateTriple:
        suggested_subject = None
        suggested_object = None
        if triple.suggested_subject_label:
            suggested_subject = self.canonicals.find_best_match(triple.suggested_subject_label)
        if triple.suggested_object_label:
            suggested_object = self.canonicals.find_best_match(triple.suggested_object_label)

        return models.CandidateTriple(
            chunk_id=chunk.id,
            subject_text=triple.subject,
            predicate_text=triple.predicate,
            object_text=triple.object,
            llm_confidence=triple.confidence,
            llm_response_fragment=str(response.raw_response),
            suggested_subject_term_id=suggested_subject.id if suggested_subject else None,
            suggested_object_term_id=suggested_object.id if suggested_object else None,
            duplicate_of_candidate_id=duplicate_of.id if duplicate_of else None,
            is_potential_duplicate=is_duplicate,
        )

    def _canonical_prompt_context(self) -> List[dict]:
        if self._canonical_context_cache is not None:
            return self._canonical_context_cache
        terms = self.canonicals.list_terms()
        context: List[dict] = []
        for term in terms:
            payload = {
                "label": term.label,
                "aliases": term.aliases or [],
            }
            if term.entity_type:
                payload["entity_type"] = term.entity_type
            context.append(payload)
        self._canonical_context_cache = context
        return context
