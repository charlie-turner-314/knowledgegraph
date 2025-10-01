from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Sequence

from sqlalchemy.orm import selectinload
from sqlmodel import Session, func, select

from . import models


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_checksum(self, checksum: str) -> Optional[models.Document]:
        statement = (
            select(models.Document)
            .where(models.Document.checksum == checksum)
            .options(selectinload(models.Document.chunks))
        )
        return self.session.exec(statement).first()

    def list_documents(self) -> List[models.Document]:
        statement = (
            select(models.Document)
            .options(selectinload(models.Document.chunks))
            .order_by(models.Document.created_at.desc())
        )
        return list(self.session.exec(statement))

    def create_document(
        self,
        *,
        display_name: str,
        media_type: str,
        checksum: str,
        original_path: Optional[str],
        extra: Optional[dict] = None,
    ) -> models.Document:
        document = models.Document(
            display_name=display_name,
            media_type=media_type,
            checksum=checksum,
            original_path=original_path,
            extra=extra or {},
        )
        self.session.add(document)
        self.session.flush()
        return document

    def add_chunks(
        self,
        document: models.Document,
        chunks: Sequence[models.DocumentChunk],
    ) -> List[models.DocumentChunk]:
        stored = []
        for chunk in chunks:
            chunk.document_id = document.id  # ensure FK set
            self.session.add(chunk)
            stored.append(chunk)
        self.session.flush()
        return stored

    def delete_document(self, document: models.Document) -> None:
        # Delete edge sources tied to this document's chunks first to maintain referential integrity
        chunk_ids = [chunk.id for chunk in document.chunks]
        if chunk_ids:
            # Load edge sources via select to avoid hitting ORM cascades unexpectedly
            sources_stmt = select(models.EdgeSource).where(
                models.EdgeSource.document_chunk_id.in_(chunk_ids)
            )
            sources = list(self.session.exec(sources_stmt))
            edge_ids_to_check: set[int] = set()
            for source in sources:
                edge_ids_to_check.add(source.edge_id)
                self.session.delete(source)

            if edge_ids_to_check:
                edges_stmt = select(models.Edge).where(models.Edge.id.in_(edge_ids_to_check))
                edges = list(self.session.exec(edges_stmt))
                for edge in edges:
                    if not edge.sources:
                        self.session.delete(edge)

                # Remove nodes that no longer participate in any edges
                orphan_nodes_stmt = select(models.Node).where(
                    ~models.Node.outgoing_edges.any(),
                    ~models.Node.incoming_edges.any(),
                )
                for node in self.session.exec(orphan_nodes_stmt):
                    self.session.delete(node)

        # Delete candidate triples tied to these chunks
        if chunk_ids:
            candidates_stmt = select(models.CandidateTriple).where(
                models.CandidateTriple.chunk_id.in_(chunk_ids)
            )
            for candidate in self.session.exec(candidates_stmt):
                self.session.delete(candidate)

        # Delete chunks themselves
        for chunk in document.chunks:
            self.session.delete(chunk)

        self.session.delete(document)


class CanonicalTermRepository:
    def __init__(self, session: Session):
        self.session = session

    def find_best_match(self, label: str) -> Optional[models.CanonicalTerm]:
        statement = select(models.CanonicalTerm).where(models.CanonicalTerm.label == label)
        return self.session.exec(statement).first()

    def add_alias(self, term: models.CanonicalTerm, alias: str) -> None:
        if alias not in term.aliases:
            term.aliases.append(alias)
            term.last_reviewed_at = datetime.utcnow()
            self.session.add(term)

    def remove_alias(self, term: models.CanonicalTerm, alias: str) -> None:
        if alias in term.aliases:
            term.aliases.remove(alias)
            term.last_reviewed_at = datetime.utcnow()
            self.session.add(term)

    def upsert(
        self,
        *,
        label: str,
        entity_type: Optional[str] = None,
        authority_source: Optional[str] = None,
    ) -> models.CanonicalTerm:
        statement = select(models.CanonicalTerm).where(models.CanonicalTerm.label == label)
        term = self.session.exec(statement).first()
        if term:
            if entity_type and not term.entity_type:
                term.entity_type = entity_type
            if authority_source and not term.authority_source:
                term.authority_source = authority_source
            return term
        term = models.CanonicalTerm(
            label=label,
            entity_type=entity_type,
            authority_source=authority_source,
        )
        self.session.add(term)
        self.session.flush()
        return term

    def list_terms(self) -> List[models.CanonicalTerm]:
        statement = select(models.CanonicalTerm).order_by(models.CanonicalTerm.label)
        return list(self.session.exec(statement))

    def delete(self, term: models.CanonicalTerm) -> None:
        # Nullify canonical references in nodes
        nodes_stmt = select(models.Node).where(models.Node.canonical_term_id == term.id)
        for node in self.session.exec(nodes_stmt):
            node.canonical_term_id = None
            node.sme_override = True
            self.session.add(node)

        self.session.delete(term)


class NodeRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_label(self, label: str) -> Optional[models.Node]:
        statement = select(models.Node).where(models.Node.label == label)
        return self.session.exec(statement).first()

    def ensure_node(
        self,
        *,
        label: str,
        entity_type: Optional[str],
        canonical_term: Optional[models.CanonicalTerm],
        sme_override: bool = False,
    ) -> models.Node:
        node = self.get_by_label(label)
        if node:
            if canonical_term and node.canonical_term_id is None:
                node.canonical_term_id = canonical_term.id
            return node
        node = models.Node(
            label=label,
            entity_type=entity_type,
            canonical_term_id=canonical_term.id if canonical_term else None,
            sme_override=sme_override,
        )
        self.session.add(node)
        self.session.flush()
        return node


class CandidateRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_candidates(
        self,
        *,
        chunk: models.DocumentChunk,
        triples: Iterable[models.CandidateTriple],
    ) -> List[models.CandidateTriple]:
        stored = []
        for triple in triples:
            triple.chunk_id = chunk.id
            self.session.add(triple)
            stored.append(triple)
        self.session.flush()
        return stored

    def get(self, candidate_id: int) -> Optional[models.CandidateTriple]:
        return self.session.get(models.CandidateTriple, candidate_id)

    def list_pending(self, limit: int = 50) -> List[models.CandidateTriple]:
        statement = (
            select(models.CandidateTriple)
            .where(models.CandidateTriple.status == models.CandidateStatus.pending)
            .order_by(models.CandidateTriple.created_at)
            .limit(limit)
        )
        return list(self.session.exec(statement))

    def count_for_document(self, document_id: int) -> int:
        statement = (
            select(func.count())
            .select_from(models.CandidateTriple)
            .join(models.DocumentChunk, models.CandidateTriple.chunk_id == models.DocumentChunk.id)
            .where(models.DocumentChunk.document_id == document_id)
        )
        return self.session.exec(statement).one()[0]

    def find_similar(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
    ) -> Optional[models.CandidateTriple]:
        statement = (
            select(models.CandidateTriple)
            .where(models.CandidateTriple.subject_text == subject)
            .where(models.CandidateTriple.predicate_text == predicate)
            .where(models.CandidateTriple.object_text == obj)
            .where(models.CandidateTriple.status != models.CandidateStatus.rejected)
            .order_by(models.CandidateTriple.created_at)
        )
        return self.session.exec(statement).first()

    def update_status(
        self,
        candidate: models.CandidateTriple,
        *,
        status: models.CandidateStatus,
    ) -> models.CandidateTriple:
        candidate.status = status
        candidate.updated_at = datetime.utcnow()
        self.session.add(candidate)
        return candidate


class GraphRepository:
    def __init__(self, session: Session):
        self.session = session
        self.nodes = NodeRepository(session)

    def create_edge_with_provenance(
        self,
        *,
        subject_label: str,
        predicate: str,
        object_label: str,
        entity_type_subject: Optional[str],
        entity_type_object: Optional[str],
        canonical_subject: Optional[models.CanonicalTerm],
        canonical_object: Optional[models.CanonicalTerm],
        candidate: Optional[models.CandidateTriple],
        document_chunk: Optional[models.DocumentChunk],
        sme_action: Optional[models.SMEAction],
        created_by: Optional[str] = None,
    ) -> models.Edge:
        subject_node = self.nodes.ensure_node(
            label=subject_label,
            entity_type=entity_type_subject,
            canonical_term=canonical_subject,
            sme_override=canonical_subject is None,
        )
        object_node = self.nodes.ensure_node(
            label=object_label,
            entity_type=entity_type_object,
            canonical_term=canonical_object,
            sme_override=canonical_object is None,
        )

        edge = models.Edge(
            subject_node_id=subject_node.id,
            predicate=predicate,
            object_node_id=object_node.id,
            candidate_id=candidate.id if candidate else None,
            created_by=created_by,
        )
        self.session.add(edge)
        self.session.flush()

        source_records: List[models.EdgeSource] = []
        if document_chunk:
            source_records.append(
                models.EdgeSource(
                    edge_id=edge.id,
                    source_type=models.EdgeSourceType.document_chunk,
                    document_chunk_id=document_chunk.id,
                )
            )
        if sme_action:
            source_records.append(
                models.EdgeSource(
                    edge_id=edge.id,
                    source_type=models.EdgeSourceType.sme_note,
                    sme_action_id=sme_action.id,
                )
            )

        if not source_records:
            raise ValueError("Edge must have at least one provenance source")

        for source in source_records:
            self.session.add(source)

        return edge

    def count_edges_for_document(self, document_id: int) -> int:
        statement = (
            select(func.count(func.distinct(models.Edge.id)))
            .select_from(models.Edge)
            .join(models.EdgeSource)
            .join(models.DocumentChunk)
            .where(models.DocumentChunk.document_id == document_id)
        )
        return self.session.exec(statement).one()[0]


class SMEActionRepository:
    def __init__(self, session: Session):
        self.session = session

    def record_action(
        self,
        *,
        action_type: models.SMEActionType,
        actor: Optional[str],
        candidate: Optional[models.CandidateTriple],
        payload: Optional[dict] = None,
    ) -> models.SMEAction:
        action = models.SMEAction(
            action_type=action_type,
            actor=actor,
            candidate_id=candidate.id if candidate else None,
            payload=payload or {},
        )
        self.session.add(action)
        self.session.flush()
        return action
