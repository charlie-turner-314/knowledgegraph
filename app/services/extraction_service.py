from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

from sqlmodel import Session

from app.data import models
from app.data.repositories import (
    CandidateRepository,
    CanonicalTermRepository,
    DocumentRepository,
    GraphRepository,
    NodeEmbeddingStore,
)
from app.ingestion.parsers import parse_document
from app.ingestion.types import ChunkPayload, ParsedDocument
from app.llm.client import LLMClient, get_client
from app.llm.schemas import (
    ExtractionAttribute,
    ExtractionContextUpdate,
    ExtractionResponse,
    ExtractionTriple,
)
from app.utils.ontology import OntologyEvaluation, evaluate_triple, summarize_mutations

logger = logging.getLogger(__name__)

def _normalize_extracted_attributes(entries: Optional[Sequence[ExtractionAttribute]]) -> List[Dict[str, object]]:
    """Convert LLM attribute payloads into structured dictionaries."""
    normalized: List[Dict[str, object]] = []
    if not entries:
        return normalized
    for attr in entries:
        name = (attr.name or "").strip()
        if not name:
            continue
        value = attr.value
        data_type = models.NodeAttributeType.string
        value_text: Optional[str] = None
        value_number: Optional[float] = None
        value_boolean: Optional[bool] = None

        if isinstance(value, bool):
            data_type = models.NodeAttributeType.boolean
            value_boolean = value
        elif isinstance(value, (int, float)):
            data_type = models.NodeAttributeType.number
            value_number = float(value)
        else:
            text_value = str(value).strip()
            # Try to coerce booleans
            lowered = text_value.lower()
            if lowered in {"true", "false", "yes", "no"}:
                data_type = models.NodeAttributeType.boolean
                value_boolean = lowered in {"true", "yes"}
            else:
                try:
                    numeric = float(text_value)
                except ValueError:
                    if len(text_value.split()) <= 3 and lowered.replace("_", "").isalnum():
                        data_type = models.NodeAttributeType.enum
                    value_text = text_value
                else:
                    data_type = models.NodeAttributeType.number
                    value_number = numeric

        if data_type == models.NodeAttributeType.string and value_text is None:
            value_text = str(value)

        normalized.append(
            {
                "name": name,
                "data_type": data_type.value,
                "value_text": value_text,
                "value_number": value_number,
                "value_boolean": value_boolean,
            }
        )
    return normalized


@dataclass
class IngestionResult:
    """Aggregate result returned after running the ingestion pipeline."""
    document: models.Document
    chunks: List[models.DocumentChunk]
    candidate_triples: List[models.CandidateTriple]
    edges: List[models.Edge]
    was_existing_document: bool


def _chunk_to_model(document: models.Document, payload: ChunkPayload) -> models.DocumentChunk:
    """Create a ``DocumentChunk`` ORM instance from the parsed payload."""
    if document.id is None:
        raise ValueError("Document must be persisted before creating chunks")
    return models.DocumentChunk(
        document_id=cast(int, document.id),
        ordering=payload.ordering,
        text=payload.text,
        page_label=payload.page_label,
        token_count=payload.token_count or len(payload.text.split()),
    )


class ExtractionOrchestrator:
    """Coordinate chunk persistence and candidate triple extraction."""

    def __init__(self, session: Session, llm_client: Optional[LLMClient] = None):
        """Store dependencies and warm reusable caches."""
        self.session = session
        self.documents = DocumentRepository(session)
        self.candidates = CandidateRepository(session)
        self.canonicals = CanonicalTermRepository(session)
        self.graph = GraphRepository(session)
        self.llm_client = llm_client or get_client()
        self._canonical_context_cache: Optional[List[dict]] = None
        self.embedding_store = NodeEmbeddingStore()
        if not self.embedding_store.has_entries():
            self.embedding_store.bootstrap_from_session(session)

    def ingest_file(self, path: Path) -> IngestionResult:
        """Parse and ingest the file located at ``path``."""
        parsed = parse_document(path)
        return self._ingest_parsed_document(parsed)

    def ingest_text(
        self,
        *,
        text: str,
        title: Optional[str] = None,
        media_type: str = "text/plain",
    ) -> IngestionResult:
        """Ingest raw ``text`` while reusing the structured extraction pipeline."""
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
        """Persist ``parsed`` content and extract candidate triples."""
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

        candidates, edges = self._extract_candidates(document, chunks_models)

        return IngestionResult(
            document=document,
            chunks=chunks_models,
            candidate_triples=candidates,
            edges=edges,
            was_existing_document=was_existing,
        )

    def _persist_chunks(
        self, document: models.Document, parsed: ParsedDocument
    ) -> List[models.DocumentChunk]:
        """Store chunks that belong to ``document`` and return them."""
        chunk_models = []
        for payload in parsed.iter_chunks():
            chunk_models.append(_chunk_to_model(document, payload))
        return self.documents.add_chunks(document, chunk_models)

    def _extract_candidates(
        self,
        document: models.Document,
        chunks: List[models.DocumentChunk],
    ) -> tuple[List[models.CandidateTriple], List[models.Edge]]:
        """Extract candidates for each chunk while deduplicating duplicates.

        Returns a tuple of (candidates, edges) where edges represent the
        persisted graph relationships linked back to their source chunks.
        """
        candidates: List[models.CandidateTriple] = []
        edges: List[models.Edge] = []
        seen_triples = set()
        canonical_context = self._canonical_prompt_context()

        use_context = len(chunks) > 1
        document_title = document.display_name or "Untitled Document"
        running_summary = (
            f"Document title: {document_title}."
            if use_context
            else None
        )
        context_entities: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            response = self._call_extractor(
                chunk,
                canonical_context,
                running_summary=running_summary if use_context else None,
                entity_memory=context_entities if use_context else None,
            )
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


                candidate, suggested_subject, suggested_object = self._build_candidate(
                    chunk,
                    triple,
                    response,
                    duplicate_of=duplicate_of,
                    is_duplicate=is_duplicate,
                    similar_objects=similar_objects,
                    similar_subjects=similar_subjects
                )
                self.session.add(candidate)
                self.session.flush()

                evaluation = evaluate_triple(
                    subject_label=candidate.subject_text,
                    predicate=candidate.predicate_text,
                    object_label=candidate.object_text,
                    subject_attributes=candidate.subject_attributes,
                    object_attributes=candidate.object_attributes,
                )

                if evaluation.is_match:
                    edge = self.graph.create_edge_with_provenance(
                        subject_label=candidate.subject_text.strip(),
                        predicate=candidate.predicate_text.strip(),
                        object_label=candidate.object_text.strip(),
                        # Prefer explicit types passed by the extractor (subject_type/object_type) if present
                        entity_type_subject=getattr(triple, "subject_type", None),
                        entity_type_object=getattr(triple, "object_type", None),
                        canonical_subject=suggested_subject,
                        canonical_object=suggested_object,
                        candidate=candidate,
                        document_chunk=chunk,
                        sme_action=None,
                        subject_attributes=candidate.subject_attributes,
                        object_attributes=candidate.object_attributes,
                        tags=candidate.suggested_tags,
                        created_by="llm_extraction",
                        statement_rationale=None,
                        statement_confidence=triple.confidence,
                        needs_evidence=False,
                    )

                    self.candidates.update_status(
                        candidate,
                        status=models.CandidateStatus.approved,
                    )

                    edges.append(edge)
                else:
                    self._annotate_candidate_pending(candidate, evaluation)

                candidates.append(candidate)
                seen_triples.add(key)

            if use_context:
                running_summary, context_entities = self._update_running_context(
                    document_title=document_title,
                    running_summary=running_summary or f"Document title: {document_title}.",
                    context_entities=context_entities,
                    chunk=chunk,
                    chunk_index=idx,
                    triples=response.triples,
                )
        return candidates, edges

    def _call_extractor(
        self,
        chunk: models.DocumentChunk,
        canonical_context: List[dict],
        *,
        running_summary: Optional[str] = None,
        entity_memory: Optional[List[Dict[str, Any]]] = None,
    ) -> ExtractionResponse:
        """Call the extraction LLM and return a validated response."""
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
                running_summary=running_summary,
                entity_memory=entity_memory,
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
    ) -> tuple[models.CandidateTriple, Optional[models.CanonicalTerm], Optional[models.CanonicalTerm]]:
        """Translate an ``ExtractionTriple`` into a candidate row with metadata."""
        suggested_subject = None
        suggested_object = None
        if triple.suggested_subject_label:
            suggested_subject = self.canonicals.find_best_match(triple.suggested_subject_label)
        if triple.suggested_object_label:
            suggested_object = self.canonicals.find_best_match(triple.suggested_object_label)

        subject_attributes = _normalize_extracted_attributes(triple.subject_attributes)
        object_attributes = _normalize_extracted_attributes(triple.object_attributes)

        # If the LLM provided explicit subject/object types, ensure they are present as attributes
        # so existing ontology evaluation (which currently inspects attributes) can still function
        # even if a later schema change stores them directly.
        if getattr(triple, "subject_type", None):
            lowered_names = {str(a.get("name", "")).lower() for a in subject_attributes}
            if "entity_type" not in lowered_names:
                subject_attributes.append(
                    {
                        "name": "entity_type",
                        "data_type": models.NodeAttributeType.enum.value,
                        "value_text": getattr(triple, "subject_type"),
                        "value_number": None,
                        "value_boolean": None,
                    }
                )
        if getattr(triple, "object_type", None):
            lowered_names = {str(a.get("name", "")).lower() for a in object_attributes}
            if "entity_type" not in lowered_names:
                object_attributes.append(
                    {
                        "name": "entity_type",
                        "data_type": models.NodeAttributeType.enum.value,
                        "value_text": getattr(triple, "object_type"),
                        "value_number": None,
                        "value_boolean": None,
                    }
                )
        tags = sorted(
            {
                tag.strip().lower()
                for tag in (triple.tags or [])
                if isinstance(tag, str) and tag.strip()
            }
        )

        if chunk.id is None:
            raise ValueError("Document chunk must be persisted before creating candidates")

        candidate = models.CandidateTriple(
            chunk_id=cast(int, chunk.id),
            subject_text=triple.subject,
            predicate_text=triple.predicate,
            object_text=triple.object,
            llm_confidence=triple.confidence,
            llm_response_fragment=str(response.raw_response),
            suggested_subject_term_id=suggested_subject.id if suggested_subject else None,
            suggested_object_term_id=suggested_object.id if suggested_object else None,
            duplicate_of_candidate_id=duplicate_of.id if duplicate_of else None,
            is_potential_duplicate=is_duplicate,
            subject_attributes=subject_attributes,
            object_attributes=object_attributes,
            suggested_tags=tags,
        )
        return candidate, suggested_subject, suggested_object

    def _annotate_candidate_pending(
        self,
        candidate: models.CandidateTriple,
        evaluation: OntologyEvaluation,
    ) -> None:
        """Attach ontology guidance to candidates that need review."""

        tags = set(candidate.suggested_tags or [])
        tags.add("needs-ontology-review")
        candidate.suggested_tags = sorted(tags)

        if not evaluation.suggested_mutations:
            return

        summaries = summarize_mutations(evaluation.suggested_mutations)
        if not summaries:
            return

        attributes = list(candidate.subject_attributes or [])
        existing = None
        for attr in attributes:
            if str(attr.get("name", "")).lower() == "ontology_suggestions":
                existing = attr
                break

        payload = "\n".join(summaries)
        suggestion_record = {
            "name": "ontology_suggestions",
            "data_type": "string",
            "value_text": payload,
            "value_number": None,
            "value_boolean": None,
        }

        if existing is None:
            attributes.append(suggestion_record)
        else:
            existing.update(suggestion_record)

        candidate.subject_attributes = attributes

    def _canonical_prompt_context(self) -> List[dict]:
        """Return cached canonical vocabulary for LLM prompts."""
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

    def _update_running_context(
        self,
        *,
        document_title: str,
        running_summary: str,
        context_entities: List[Dict[str, Any]],
        chunk: models.DocumentChunk,
        chunk_index: int,
        triples: Sequence[ExtractionTriple],
    ) -> tuple[str, List[Dict[str, Any]]]:
        try:
            update = self.llm_client.update_extraction_context(
                document_title=document_title,
                previous_summary=running_summary,
                existing_entities=context_entities,
                chunk_order=chunk.ordering if chunk.ordering is not None else chunk_index,
                chunk_page_label=chunk.page_label,
                chunk_text=chunk.text,
                extracted_triples=[self._triple_context_payload(triple) for triple in triples],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update extraction context for chunk %s: %s", chunk.id, exc)
            return running_summary, context_entities

        new_summary = (update.summary or running_summary).strip()
        if len(new_summary) > 4000:
            new_summary = new_summary[:4000]

        merged_entities = self._merge_entity_memory(
            context_entities,
            [entity.model_dump() for entity in update.entities],
        )
        return new_summary or running_summary, merged_entities

    @staticmethod
    def _triple_context_payload(triple: ExtractionTriple) -> Dict[str, Any]:
        return {
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "subject_type": getattr(triple, "subject_type", None),
            "object_type": getattr(triple, "object_type", None),
            "tags": list(triple.tags or []),
        }

    @staticmethod
    def _merge_entity_memory(
        existing: List[Dict[str, Any]],
        updates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}

        for entity in existing:
            label = str(entity.get("label") or "").strip()
            if not label:
                continue
            key = label.lower()
            index[key] = {
                "label": label,
                "aliases": sorted({alias.strip() for alias in entity.get("aliases", []) if alias}),
                "entity_type": entity.get("entity_type"),
                "description": entity.get("description"),
                "attributes": dict(entity.get("attributes") or {}),
            }

        for entity in updates:
            label = str(entity.get("label") or "").strip()
            if not label:
                continue
            key = label.lower()
            aliases = [str(alias).strip() for alias in entity.get("aliases", []) if str(alias).strip()]
            entity_type = entity.get("entity_type")
            description = entity.get("description")
            attributes = entity.get("attributes") or {}

            if key not in index:
                index[key] = {
                    "label": label,
                    "aliases": sorted(set(aliases)),
                    "entity_type": entity_type,
                    "description": description,
                    "attributes": dict(attributes),
                }
                continue

            entry = index[key]
            alias_set = set(entry.get("aliases", []))
            alias_set.update(aliases)
            entry["aliases"] = sorted(alias_set)

            if entity_type and not entry.get("entity_type"):
                entry["entity_type"] = entity_type

            if description:
                existing_desc = entry.get("description")
                if not existing_desc or len(description) > len(existing_desc):
                    entry["description"] = description

            if attributes:
                merged_attrs = dict(entry.get("attributes") or {})
                for attr_key, attr_value in attributes.items():
                    merged_attrs[attr_key] = attr_value
                entry["attributes"] = merged_attrs

        merged = list(index.values())
        merged.sort(key=lambda item: item.get("label", "").lower())
        if len(merged) > 40:
            merged = merged[:40]
        return merged
