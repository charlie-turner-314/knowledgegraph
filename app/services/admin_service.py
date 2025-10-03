from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.data import models
from app.data.repositories import (
    CanonicalTermRepository,
    CandidateRepository,
    DocumentRepository,
    GraphRepository,
)


@dataclass
class DocumentOverview:
    """Summary information for documents shown in the admin UI."""
    id: int
    name: str
    media_type: str
    created_at: str
    chunk_count: int
    candidate_count: int
    edge_count: int


@dataclass
class CanonicalTermView:
    """Lightweight view of canonical vocabulary entries."""
    id: int
    label: str
    entity_type: Optional[str]
    aliases: List[str] = field(default_factory=list)


class AdminService:
    """Administrative operations for documents and canonical vocabulary."""

    def __init__(self, session: Session):
        """Initialise repository helpers with a shared session."""
        self.session = session
        self.documents = DocumentRepository(session)
        self.candidates = CandidateRepository(session)
        self.graph = GraphRepository(session)
        self.canonicals = CanonicalTermRepository(session)

    # --- Document management -----------------------------------------------------

    def list_documents(self) -> List[DocumentOverview]:
        """Return high-level statistics for each ingested document."""
        docs = self.documents.list_documents()
        overview: List[DocumentOverview] = []
        for doc in docs:
            candidate_count = self.candidates.count_for_document(doc.id)
            edge_count = self.graph.count_edges_for_document(doc.id)
            overview.append(
                DocumentOverview(
                    id=doc.id,
                    name=doc.display_name,
                    media_type=doc.media_type,
                    created_at=doc.created_at.isoformat(),
                    chunk_count=len(doc.chunks),
                    candidate_count=candidate_count,
                    edge_count=edge_count,
                )
            )
        return overview

    def delete_document(self, document_id: int) -> None:
        """Delete the specified document and cascade related graph data."""
        statement = (
            select(models.Document)
            .where(models.Document.id == document_id)
            .options(selectinload(models.Document.chunks))
        )
        document = self.session.exec(statement).first()
        if not document:
            raise ValueError("Document not found")
        self.documents.delete_document(document)

    # --- Canonical term management ----------------------------------------------

    def list_canonical_terms(self) -> List[CanonicalTermView]:
        """Return all canonical terms mapped into simple view models."""
        terms = self.canonicals.list_terms()
        return [
            CanonicalTermView(
                id=term.id,
                label=term.label,
                entity_type=term.entity_type,
                aliases=list(term.aliases or []),
            )
            for term in terms
        ]

    def create_canonical_term(
        self,
        *,
        label: str,
        entity_type: Optional[str],
        aliases: Optional[List[str]] = None,
    ) -> models.CanonicalTerm:
        """Create a new canonical term and optional aliases."""
        term = self.canonicals.upsert(label=label, entity_type=entity_type)
        if aliases:
            for alias in aliases:
                self.canonicals.add_alias(term, alias)
        return term

    def update_canonical_term(
        self,
        term_id: int,
        *,
        label: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> models.CanonicalTerm:
        """Rename or update metadata for an existing canonical term."""
        term = self.session.get(models.CanonicalTerm, term_id)
        if not term:
            raise ValueError("Canonical term not found")
        if label:
            term.label = label
        if entity_type is not None:
            term.entity_type = entity_type or None
        self.session.add(term)
        return term

    def add_alias(self, term_id: int, alias: str) -> models.CanonicalTerm:
        """Append a new alias to ``term_id``."""
        term = self.session.get(models.CanonicalTerm, term_id)
        if not term:
            raise ValueError("Canonical term not found")
        self.canonicals.add_alias(term, alias)
        return term

    def remove_alias(self, term_id: int, alias: str) -> models.CanonicalTerm:
        """Remove ``alias`` from ``term_id``."""
        term = self.session.get(models.CanonicalTerm, term_id)
        if not term:
            raise ValueError("Canonical term not found")
        self.canonicals.remove_alias(term, alias)
        return term

    def delete_canonical_term(self, term_id: int) -> None:
        """Delete the canonical term identified by ``term_id``."""
        term = self.session.get(models.CanonicalTerm, term_id)
        if not term:
            raise ValueError("Canonical term not found")
        self.canonicals.delete(term)


__all__ = [
    "AdminService",
    "DocumentOverview",
    "CanonicalTermView",
]
