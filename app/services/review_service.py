from __future__ import annotations

from typing import Dict, List, Optional

from sqlmodel import Session, select
from sqlalchemy.orm import selectinload

from app.data import models
from app.data.repositories import (
    CandidateRepository,
    CanonicalTermRepository,
    GraphRepository,
    SMEActionRepository,
)


class ReviewService:
    """Business logic for SME review actions on candidate triples."""

    def __init__(self, session: Session):
        """Initialise the service with repository helpers."""
        self.session = session
        self.candidates = CandidateRepository(session)
        self.canonicals = CanonicalTermRepository(session)
        self.graph = GraphRepository(session)
        self.actions = SMEActionRepository(session)

    def list_pending(self, limit: int = 25) -> List[models.CandidateTriple]:
        """Return the oldest pending candidates up to ``limit``."""
        return self.candidates.list_pending(limit=limit)

    def approve_candidate(
        self,
        candidate_id: int,
        *,
        actor: Optional[str],
        subject_override: Optional[str] = None,
        predicate_override: Optional[str] = None,
        object_override: Optional[str] = None,
        notes: Optional[str] = None,
        subject_attributes: Optional[List[Dict[str, object]]] = None,
        object_attributes: Optional[List[Dict[str, object]]] = None,
        tags: Optional[List[str]] = None,
    ) -> models.Edge:
        """Approve a candidate triple, creating a graph edge with provenance."""
        candidate = self._load_candidate(candidate_id)
        if candidate.status != models.CandidateStatus.pending:
            raise ValueError("Candidate already resolved")

        subject_label = subject_override or candidate.subject_text
        predicate = predicate_override or candidate.predicate_text
        object_label = object_override or candidate.object_text

        def _json_ready(attrs: Optional[List[Dict[str, object]]]) -> Optional[List[Dict[str, object]]]:
            if attrs is None:
                return None
            safe: List[Dict[str, object]] = []
            for item in attrs:
                if not isinstance(item, dict):
                    continue
                converted = dict(item)
                dtype = converted.get("data_type")
                if isinstance(dtype, models.NodeAttributeType):
                    converted["data_type"] = dtype.value
                safe.append(converted)
            return safe

        action = self.actions.record_action(
            action_type=models.SMEActionType.accept_triple,
            actor=actor,
            candidate=candidate,
            payload={
                "subject_override": subject_override,
                "predicate_override": predicate_override,
                "object_override": object_override,
                "notes": notes,
                "subject_attributes": _json_ready(subject_attributes),
                "object_attributes": _json_ready(object_attributes),
                "tags": tags,
            },
        )

        subject_attrs_payload = (
            subject_attributes
            if subject_attributes is not None
            else list(candidate.subject_attributes or [])
        )
        object_attrs_payload = (
            object_attributes
            if object_attributes is not None
            else list(candidate.object_attributes or [])
        )
        tags_payload = tags if tags is not None else list(candidate.suggested_tags or [])

        edge = self.graph.create_edge_with_provenance(
            subject_label=subject_label,
            predicate=predicate,
            object_label=object_label,
            entity_type_subject=None,
            entity_type_object=None,
            canonical_subject=candidate.subject_suggestion,
            canonical_object=candidate.object_suggestion,
            candidate=candidate,
            document_chunk=candidate.chunk,
            sme_action=action if notes else None,
            subject_attributes=subject_attrs_payload,
            object_attributes=object_attrs_payload,
            tags=tags_payload,
            created_by=actor,
        )

        self.candidates.update_status(candidate, status=models.CandidateStatus.approved)
        return edge

    def reject_candidate(
        self,
        candidate_id: int,
        *,
        actor: Optional[str],
        reason: Optional[str] = None,
    ) -> models.CandidateTriple:
        """Mark a candidate as rejected and log the reviewer action."""
        candidate = self._load_candidate(candidate_id)
        if candidate.status != models.CandidateStatus.pending:
            raise ValueError("Candidate already resolved")

        self.actions.record_action(
            action_type=models.SMEActionType.reject_triple,
            actor=actor,
            candidate=candidate,
            payload={"reason": reason},
        )
        return self.candidates.update_status(candidate, status=models.CandidateStatus.rejected)

    def _load_candidate(self, candidate_id: int) -> models.CandidateTriple:
        """Return a candidate with eager-loaded relations or raise if missing."""
        statement = (
            select(models.CandidateTriple)
            .where(models.CandidateTriple.id == candidate_id)
            .options(
                selectinload(models.CandidateTriple.chunk).selectinload(models.DocumentChunk.document),
                selectinload(models.CandidateTriple.subject_suggestion),
                selectinload(models.CandidateTriple.object_suggestion),
            )
        )
        candidate = self.session.exec(statement).first()
        if candidate is None:
            raise ValueError("Candidate not found")
        return candidate


def serialize_candidate(candidate: models.CandidateTriple) -> Dict[str, object]:
    """Convert a candidate ORM object into a Streamlit-friendly dictionary."""
    chunk = candidate.chunk
    document = chunk.document if chunk else None
    return {
        "id": candidate.id,
        "subject": candidate.subject_text,
        "predicate": candidate.predicate_text,
        "object": candidate.object_text,
        "confidence": candidate.llm_confidence,
        "status": candidate.status.value,
        "document": document.display_name if document else None,
        "document_id": document.id if document else None,
        "page_label": chunk.page_label if chunk else None,
        "chunk_text": chunk.text if chunk else None,
        "is_duplicate": candidate.is_potential_duplicate,
        "duplicate_of": candidate.duplicate_of_candidate_id,
        "subject_attributes": candidate.subject_attributes or [],
        "object_attributes": candidate.object_attributes or [],
        "tags": candidate.suggested_tags or [],
    }
