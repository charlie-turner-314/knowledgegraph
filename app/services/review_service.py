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
    def __init__(self, session: Session):
        self.session = session
        self.candidates = CandidateRepository(session)
        self.canonicals = CanonicalTermRepository(session)
        self.graph = GraphRepository(session)
        self.actions = SMEActionRepository(session)

    def list_pending(self, limit: int = 25) -> List[models.CandidateTriple]:
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
        candidate = self._load_candidate(candidate_id)
        if candidate.status != models.CandidateStatus.pending:
            raise ValueError("Candidate already resolved")

        subject_label = subject_override or candidate.subject_text
        predicate = predicate_override or candidate.predicate_text
        object_label = object_override or candidate.object_text

        action = self.actions.record_action(
            action_type=models.SMEActionType.accept_triple,
            actor=actor,
            candidate=candidate,
            payload={
                "subject_override": subject_override,
                "predicate_override": predicate_override,
                "object_override": object_override,
                "notes": notes,
            },
        )

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
            subject_attributes=subject_attributes,
            object_attributes=object_attributes,
            tags=tags,
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
    }
