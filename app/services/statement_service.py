from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.data import models
from app.data.repositories import StatementRepository
from app.data.models import StatementStatus


@dataclass
class StatementView:
    id: int
    subject: Optional[str]
    predicate: str
    object: Optional[str]
    status: StatementStatus
    needs_evidence: bool
    confidence: Optional[float]
    rationale: Optional[str]
    resolution_notes: Optional[str]
    created_by: Optional[str]


class StatementService:
    """Application layer for managing graph statements and evidence gaps."""

    def __init__(self, session: Session):
        self.session = session
        self.statements = StatementRepository(session)

    def list_needs_evidence(self, limit: int = 100) -> List[StatementView]:
        rows = self.statements.list_by_status(StatementStatus.needs_evidence, limit=limit)
        return [self._to_view(row) for row in rows]

    def resolve_statement(
        self,
        statement_id: int,
        *,
        new_status: StatementStatus,
        resolution_notes: Optional[str],
        confidence: Optional[float],
        actor: Optional[str],
    ) -> models.GraphStatement:
        statement = self.statements.get(statement_id)
        if statement is None:
            raise ValueError("Statement not found")
        needs_evidence = new_status == StatementStatus.needs_evidence
        return self.statements.update_status(
            statement,
            status=new_status,
            needs_evidence=needs_evidence,
            confidence=confidence,
            resolution_notes=resolution_notes,
            actor=actor,
        )

    def _to_view(self, statement: models.GraphStatement) -> StatementView:
        subject_label = statement.subject.label if statement.subject else None
        object_label = statement.object.label if statement.object else None
        return StatementView(
            id=statement.id,
            subject=subject_label,
            predicate=statement.predicate,
            object=object_label,
            status=statement.status,
            needs_evidence=statement.needs_evidence,
            confidence=statement.confidence,
            rationale=statement.rationale,
            resolution_notes=statement.resolution_notes,
            created_by=statement.created_by,
        )
