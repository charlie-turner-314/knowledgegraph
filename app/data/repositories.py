from __future__ import annotations
# mypy: ignore-errors

import json
import logging
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.data.embedding_persistence import load_embeddings, save_embeddings

import numpy as np
import requests
from rapidfuzz import fuzz
from sqlalchemy.orm import selectinload
from sqlmodel import Session, func, select

from app.core.config import settings

# Configurable batch size for embedding API calls (OpenAI supports up to 2048)
DEFAULT_EMBEDDING_BATCH_SIZE = 1024
from app.llm.client import LLMClient, get_client

from . import models
from app.utils.ontology import infer_entity_type


logger = logging.getLogger(__name__)


class DocumentRepository:
    """Persistence helpers for documents and their chunks."""

    def __init__(self, session: Session):
        """Store the active SQLModel session for subsequent operations."""
        self.session = session

    def get_by_checksum(self, checksum: str) -> Optional[models.Document]:
        """Return the document that matches ``checksum`` if it exists."""
        statement = (
            select(models.Document)
            .where(models.Document.checksum == checksum)
            .options(selectinload(models.Document.chunks))
        )
        return self.session.exec(statement).first()

    def list_documents(self) -> List[models.Document]:
        """Return all documents ordered by newest first with eager-loaded chunks."""
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
        """Insert a new document row and return the persisted instance."""
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
        """Persist the supplied chunk models for ``document`` in a single batch."""
        stored = []
        for chunk in chunks:
            chunk.document_id = document.id  # ensure FK set
            self.session.add(chunk)
            stored.append(chunk)
        self.session.flush()
        return stored

    def delete_document(self, document: models.Document) -> None:
        """Remove a document and all dependent graph/candidate records."""
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
    """CRUD helpers for canonical vocabulary management."""

    def __init__(self, session: Session):
        """Keep a reference to the active session."""
        self.session = session

    def find_best_match(self, label: str) -> Optional[models.CanonicalTerm]:
        """Return a canonical term whose label exactly matches ``label``."""
        statement = select(models.CanonicalTerm).where(models.CanonicalTerm.label == label)
        return self.session.exec(statement).first()

    def add_alias(self, term: models.CanonicalTerm, alias: str) -> None:
        """Append ``alias`` to ``term`` if it is not already present."""
        if alias not in term.aliases:
            term.aliases.append(alias)
            term.last_reviewed_at = datetime.utcnow()
            self.session.add(term)

    def remove_alias(self, term: models.CanonicalTerm, alias: str) -> None:
        """Remove ``alias`` from ``term`` when it exists."""
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
        """Insert or fetch a canonical term by label, updating optional metadata."""
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
        """Return all canonical terms sorted alphabetically."""
        statement = select(models.CanonicalTerm).order_by(models.CanonicalTerm.label)
        return list(self.session.exec(statement))

    def delete(self, term: models.CanonicalTerm) -> None:
        """Remove ``term`` and detach any linked nodes."""
        # Nullify canonical references in nodes
        nodes_stmt = select(models.Node).where(models.Node.canonical_term_id == term.id)
        for node in self.session.exec(nodes_stmt):
            node.canonical_term_id = None
            node.sme_override = True
            self.session.add(node)

        self.session.delete(term)


class NodeRepository:
    """Utility methods for reading or creating graph nodes."""

    def __init__(self, session: Session):
        """Bind the repository to a SQLModel session."""
        self.session = session

    def get_by_label(self, label: str) -> Optional[models.Node]:
        """Return a node matching ``label`` including its attributes."""
        statement = (
            select(models.Node)
            .where(models.Node.label == label)
            .options(selectinload(models.Node.attributes))
        )
        return self.session.exec(statement).first()

    def ensure_node(
        self,
        *,
        label: str,
        entity_type: Optional[str],
        canonical_term: Optional[models.CanonicalTerm],
        sme_override: bool = False,
    ) -> models.Node:
        """Fetch or create a node, lazily attaching canonical metadata if needed."""
        node = self.get_by_label(label)
        if node:
            changed = False
            if canonical_term and node.canonical_term_id is None:
                node.canonical_term_id = canonical_term.id
                changed = True
            if entity_type and (node.entity_type or "").strip().lower() != entity_type.strip().lower():
                node.entity_type = entity_type
                node.sme_override = True
                changed = True
            if changed:
                self.session.add(node)
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

    def find_by_label_case_insensitive(self, label: str) -> List[models.Node]:
        """Return nodes whose label matches ``label`` ignoring case."""
        cleaned = (label or "").strip()
        if not cleaned:
            return []
        stmt = (
            select(models.Node)
            .where(func.lower(models.Node.label) == cleaned.lower())
            .options(selectinload(models.Node.attributes))
        )
        return list(self.session.exec(stmt))


class NodeAttributeRepository:
    """Manage scalar attribute state for nodes."""

    def __init__(self, session: Session):
        """Store the active session for use in mutations."""
        self.session = session

    def upsert_many(
        self,
        node: models.Node,
        attributes: Sequence[Dict[str, object]],
    ) -> None:
        """Create or update the provided ``attributes`` for ``node``."""
        if not attributes:
            return
        existing = {
            attr.name.lower(): attr for attr in (node.attributes or [])
        }

        for payload in attributes:
            name = str(payload.get("name", "")).strip()
            if not name:
                continue
            key = name.lower()
            data_type = payload.get("data_type", models.NodeAttributeType.string)
            if isinstance(data_type, str):
                try:
                    data_type_enum = models.NodeAttributeType(data_type)
                except ValueError:
                    data_type_enum = models.NodeAttributeType.string
            else:
                data_type_enum = data_type

            value_text = payload.get("value_text")
            value_number = payload.get("value_number")
            value_boolean = payload.get("value_boolean")

            record = existing.get(key)
            if record is None:
                record = models.NodeAttribute(
                    node_id=node.id,
                    name=name,
                    data_type=data_type_enum,
                )
                self.session.add(record)
                if node.attributes is not None:
                    node.attributes.append(record)
            elif node.attributes is not None and record not in node.attributes:
                node.attributes.append(record)
            record.data_type = data_type_enum
            record.value_text = value_text
            record.value_number = value_number
            record.value_boolean = value_boolean
            record.updated_at = datetime.utcnow()

class CandidateRepository:
    """Operations for managing candidate triples awaiting SME review."""

    def __init__(self, session: Session):
        """Store the SQLModel session."""
        self.session = session

    def add_candidates(
        self,
        *,
        chunk: models.DocumentChunk,
        triples: Iterable[models.CandidateTriple],
    ) -> List[models.CandidateTriple]:
        """Persist ``triples`` for ``chunk`` and return the stored objects."""
        stored = []
        for triple in triples:
            triple.chunk_id = chunk.id
            self.session.add(triple)
            stored.append(triple)
        self.session.flush()
        return stored

    def get(self, candidate_id: int) -> Optional[models.CandidateTriple]:
        """Return a candidate by primary key or ``None`` when missing."""
        return self.session.get(models.CandidateTriple, candidate_id)

    def list_pending(self, limit: int = 50) -> List[models.CandidateTriple]:
        """Fetch the next ``limit`` candidates awaiting review."""
        statement = (
            select(models.CandidateTriple)
            .where(models.CandidateTriple.status == models.CandidateStatus.pending)
            .order_by(models.CandidateTriple.created_at)
            .limit(limit)
        )
        return list(self.session.exec(statement))

    def count_for_document(self, document_id: int) -> int:
        """Return how many candidates originate from ``document_id``."""
        statement = (
            select(func.count())
            .select_from(models.CandidateTriple)
            .join(models.DocumentChunk, models.CandidateTriple.chunk_id == models.DocumentChunk.id)
            .where(models.DocumentChunk.document_id == document_id)
        )
        return self.session.exec(statement).one()

    def find_similar(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
    ) -> Optional[models.CandidateTriple]:
        """Return an existing candidate matching the supplied triple values."""
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
        """Persist a status transition for ``candidate`` and return it."""
        candidate.status = status
        candidate.updated_at = datetime.utcnow()
        self.session.add(candidate)
        return candidate


class GraphRepository:
    """Coordinate node creation, edge persistence, and lightweight ontology audits."""

    def __init__(self, session: Session):
        """Initialise helper repositories and ensure embeddings are ready."""
        self.session = session
        self.nodes = NodeRepository(session)
        self.node_attributes = NodeAttributeRepository(session)
        self._embedding_store = NodeEmbeddingStore()
        if not self._embedding_store.has_entries():
            self._embedding_store.bootstrap_from_session(session)

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
        subject_attributes: Optional[Sequence[Dict[str, object]]] = None,
        object_attributes: Optional[Sequence[Dict[str, object]]] = None,
        tags: Optional[Sequence[str]] = None,
        created_by: Optional[str] = None,
        statement_rationale: Optional[str] = None,
        statement_confidence: Optional[float] = None,
        needs_evidence: bool = False,
    ) -> models.Edge:
        """Persist an edge with provenance, normalising optional attributes and tags."""
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

        normalized_subject_attributes = self._normalize_attributes(subject_attributes)
        normalized_object_attributes = self._normalize_attributes(object_attributes)

        if normalized_subject_attributes:
            self.node_attributes.upsert_many(subject_node, normalized_subject_attributes)
        if normalized_object_attributes:
            self.node_attributes.upsert_many(object_node, normalized_object_attributes)

        self._embedding_store.bulk_add([subject_node.label, object_node.label])

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

        if tags:
            normalized_tags = {
                tag.strip().lower()
                for tag in tags
                if isinstance(tag, str) and tag.strip()
            }
            for tag in sorted(normalized_tags):
                self.session.add(
                    models.EdgeTag(
                        edge_id=edge.id,
                        label=tag,
                    )
                )

        self._create_statement(
            subject_node=subject_node,
            predicate=predicate,
            object_node=object_node,
            status=models.StatementStatus.needs_evidence if needs_evidence else models.StatementStatus.validated,
            needs_evidence=needs_evidence,
            confidence=statement_confidence if statement_confidence is not None else (candidate.llm_confidence if candidate else None),
            rationale=statement_rationale,
            edge=edge,
            candidate=candidate,
            created_by=created_by,
        )

        return edge

    # ------------------------------------------------------------------
    # Candidate integration helpers

    def find_edge_by_labels(
        self,
        *,
        subject_label: str,
        predicate: str,
        object_label: str,
    ) -> Optional[models.Edge]:
        """Return an existing edge connecting subject → object via predicate if present.

        Performs a lightweight lookup by resolving subject/object node IDs first to avoid
        complex self-joins of the node table.
        """
        subject_node = self.session.exec(
            select(models.Node).where(models.Node.label == subject_label)
        ).first()
        if subject_node is None:
            return None
        object_node = self.session.exec(
            select(models.Node).where(models.Node.label == object_label)
        ).first()
        if object_node is None:
            return None
        edge_stmt = (
            select(models.Edge)
            .where(models.Edge.subject_node_id == subject_node.id)
            .where(models.Edge.object_node_id == object_node.id)
            .where(models.Edge.predicate == predicate)
            .limit(1)
        )
        return self.session.exec(edge_stmt).first()

    def ensure_edge_for_candidate(
        self,
        candidate: models.CandidateTriple,
        *,
        sme_action: Optional[models.SMEAction],
        actor: Optional[str] = None,
    ) -> models.Edge:
        """Create (or link to) an edge for an approved candidate triple.

        If an edge already exists with the same subject/predicate/object, ensure the
        candidate ID is attached and return it; otherwise create a new edge with the
        candidate's provenance.
        """
        subject_label = candidate.subject_text
        predicate = candidate.predicate_text
        object_label = candidate.object_text

        existing = self.find_edge_by_labels(
            subject_label=subject_label, predicate=predicate, object_label=object_label
        )
        if existing:
            if existing.candidate_id is None:
                existing.candidate_id = candidate.id
                self.session.add(existing)
            return existing

        document_chunk = candidate.chunk
        subj_type = infer_entity_type(candidate.subject_attributes) or None
        obj_type = infer_entity_type(candidate.object_attributes) or None

        edge = self.create_edge_with_provenance(
            subject_label=subject_label,
            predicate=predicate,
            object_label=object_label,
            entity_type_subject=subj_type,
            entity_type_object=obj_type,
            canonical_subject=None,
            canonical_object=None,
            candidate=candidate,
            document_chunk=document_chunk,
            sme_action=sme_action,
            subject_attributes=candidate.subject_attributes,
            object_attributes=candidate.object_attributes,
            tags=candidate.suggested_tags,
            created_by=actor or "candidate_approval",
            statement_rationale="Approved candidate triple",
            statement_confidence=candidate.llm_confidence,
            needs_evidence=False,
        )
        return edge

    def create_statement_placeholder(
        self,
        *,
        subject_label: Optional[str],
        predicate: str,
        object_label: Optional[str],
        created_by: Optional[str] = None,
        rationale: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> models.GraphStatement:
        """Create a statement flagged as needing evidence without creating an edge."""

        subject_node = (
            self.nodes.ensure_node(
                label=subject_label,
                entity_type=None,
                canonical_term=None,
                sme_override=True,
            )
            if subject_label
            else None
        )
        object_node = (
            self.nodes.ensure_node(
                label=object_label,
                entity_type=None,
                canonical_term=None,
                sme_override=True,
            )
            if object_label
            else None
        )

        self._embedding_store.bulk_add(
            [node.label for node in [subject_node, object_node] if node and node.label]
        )

        return self._create_statement(
            subject_node=subject_node,
            predicate=predicate,
            object_node=object_node,
            status=models.StatementStatus.needs_evidence,
            needs_evidence=True,
            confidence=confidence,
            rationale=rationale,
            edge=None,
            candidate=None,
            created_by=created_by,
        )

    def _normalize_attributes(
        self, attributes: Optional[Sequence[Dict[str, object]]]
    ) -> List[Dict[str, object]]:
        """Normalise external attribute payloads into a standard schema."""
        if not attributes:
            return []

        normalized: Dict[str, Dict[str, object]] = {}
        for raw in attributes:
            if not isinstance(raw, dict):
                continue
            name_raw = raw.get("name") or raw.get("label")
            if not name_raw:
                continue
            name = str(name_raw).strip()
            if not name:
                continue

            data_type = raw.get("data_type")
            if isinstance(data_type, models.NodeAttributeType):
                data_type_enum = data_type
            elif isinstance(data_type, str):
                try:
                    data_type_enum = models.NodeAttributeType(data_type)
                except ValueError:
                    data_type_enum = None
            else:
                data_type_enum = None

            value_text = raw.get("value_text")
            value_number = raw.get("value_number")
            value_boolean = raw.get("value_boolean")

            if value_text is None and value_number is None and value_boolean is None:
                value = raw.get("value")
                if isinstance(value, bool):
                    data_type_enum = data_type_enum or models.NodeAttributeType.boolean
                    value_boolean = value
                elif isinstance(value, (int, float)):
                    data_type_enum = data_type_enum or models.NodeAttributeType.number
                    value_number = float(value)
                elif value is not None:
                    text_value = str(value).strip()
                    if data_type_enum is None:
                        lowered = text_value.lower()
                        if lowered in {"true", "false", "yes", "no"}:
                            data_type_enum = models.NodeAttributeType.boolean
                            value_boolean = lowered in {"true", "yes"}
                        else:
                            try:
                                numeric = float(text_value)
                            except ValueError:
                                if len(text_value.split()) <= 3 and lowered.replace("_", "").isalnum():
                                    data_type_enum = models.NodeAttributeType.enum
                                else:
                                    data_type_enum = models.NodeAttributeType.string
                                value_text = text_value
                            else:
                                data_type_enum = models.NodeAttributeType.number
                                value_number = numeric
                    else:
                        value_text = text_value

            if data_type_enum is None:
                if value_number is not None:
                    data_type_enum = models.NodeAttributeType.number
                elif value_boolean is not None:
                    data_type_enum = models.NodeAttributeType.boolean
                else:
                    data_type_enum = models.NodeAttributeType.string

            if data_type_enum == models.NodeAttributeType.string and value_text is None:
                fallback_value = raw.get("value")
                if fallback_value is not None:
                    value_text = str(fallback_value)

            normalized[name.lower()] = {
                "name": name,
                "data_type": data_type_enum,
                "value_text": value_text,
                "value_number": value_number,
                "value_boolean": value_boolean,
            }

        return list(normalized.values())

    def _create_statement(
        self,
        *,
        subject_node: Optional[models.Node],
        predicate: str,
        object_node: Optional[models.Node],
        status: models.StatementStatus,
        needs_evidence: bool,
        confidence: Optional[float],
        rationale: Optional[str],
        edge: Optional[models.Edge],
        candidate: Optional[models.CandidateTriple],
        created_by: Optional[str],
        resolution_notes: Optional[str] = None,
    ) -> models.GraphStatement:
        statement = models.GraphStatement(
            subject_node_id=subject_node.id if subject_node else None,
            predicate=predicate,
            object_node_id=object_node.id if object_node else None,
            edge_id=edge.id if edge else None,
            candidate_id=candidate.id if candidate else None,
            status=status,
            confidence=confidence,
            needs_evidence=needs_evidence,
            rationale=rationale,
            created_by=created_by,
            resolution_notes=resolution_notes,
        )
        self.session.add(statement)
        self.session.flush()
        return statement

    def count_edges_for_document(self, document_id: int) -> int:
        """Return the number of distinct edges attributed to ``document_id``."""
        statement = (
            select(func.count(func.distinct(models.Edge.id)))
            .select_from(models.Edge)
            .join(models.EdgeSource)
            .join(models.DocumentChunk)
            .where(models.DocumentChunk.document_id == document_id)
        )
        return self.session.exec(statement).one()

    def audit_ontology(self) -> dict:
        """Produce heuristics highlighting potential class gaps or ambiguous predicates."""
        all_nodes = list(self.session.exec(select(models.Node)))
        isa_edges = list(self.session.exec(select(models.Edge).where(models.Edge.predicate == 'is_a')))
        isa_subject_ids = {edge.subject_node_id for edge in isa_edges}

        # Heuristic: nodes that are likely instances but lack class membership
        candidate_untyped = [
            node for node in all_nodes
            if node.entity_type in {"equipment", "material", "concept"} and node.id not in isa_subject_ids
        ]

        # Predicate normalization suggestions
        all_predicates = {edge.predicate for edge in self.session.exec(select(models.Edge))}
        ambiguous_predicates = [p for p in all_predicates if p in {"is one of", "belongs to", "type of"}]

        return {
            "candidate_untyped_nodes": candidate_untyped,
            "ambiguous_predicates": ambiguous_predicates,
        }



class SMEActionRepository:
    """Record SME actions for provenance and audit trails."""

    def __init__(self, session: Session):
        """Bind to the active session."""
        self.session = session

    def record_action(
        self,
        *,
        action_type: models.SMEActionType,
        actor: Optional[str],
        candidate: Optional[models.CandidateTriple],
        payload: Optional[dict] = None,
    ) -> models.SMEAction:
        """Create a new SME action row and return it."""
        action = models.SMEAction(
            action_type=action_type,
            actor=actor,
            candidate_id=candidate.id if candidate else None,
            payload=payload or {},
        )
        self.session.add(action)
        self.session.flush()
        return action


class OntologySuggestionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, suggestion_id: int) -> Optional[models.OntologySuggestion]:
        return self.session.get(models.OntologySuggestion, suggestion_id)

    def list_pending(self, limit: int = 20) -> List[models.OntologySuggestion]:
        statement = (
            select(models.OntologySuggestion)
            .where(models.OntologySuggestion.status == models.OntologySuggestionStatus.pending)
            .order_by(models.OntologySuggestion.created_at)
            .limit(limit)
        )
        return list(self.session.exec(statement))

    def find_for_nodes(
        self,
        node_ids: Sequence[int],
        *,
        predicate: str = "is_a",
        statuses: Optional[Sequence[models.OntologySuggestionStatus]] = None,
    ) -> Optional[models.OntologySuggestion]:
        if not node_ids:
            return None
        status_filter = tuple(statuses) if statuses else (models.OntologySuggestionStatus.pending,)
        sorted_ids = sorted(int(node_id) for node_id in node_ids)
        statement = (
            select(models.OntologySuggestion)
            .where(models.OntologySuggestion.predicate == predicate)
            .where(models.OntologySuggestion.status.in_(status_filter))
        )
        for candidate in self.session.exec(statement):
            if sorted((candidate.supporting_node_ids or [])) == sorted_ids:
                return candidate
        return None


class StatementRepository:
    """Persistence helpers for graph statements (knowledge assertions)."""

    def __init__(self, session: Session):
        self.session = session

    def get(self, statement_id: int) -> Optional[models.GraphStatement]:
        return self.session.get(models.GraphStatement, statement_id)

    def list_by_status(
        self, status: models.StatementStatus, *, limit: int = 100
    ) -> List[models.GraphStatement]:
        statement = (
            select(models.GraphStatement)
            .where(models.GraphStatement.status == status)
            .order_by(models.GraphStatement.updated_at.desc())
            .limit(limit)
            .options(
                selectinload(models.GraphStatement.subject),
                selectinload(models.GraphStatement.object),
                selectinload(models.GraphStatement.edge)
                .selectinload(models.Edge.sources)
                .selectinload(models.EdgeSource.document_chunk)
                .selectinload(models.DocumentChunk.document),
            )
        )
        return list(self.session.exec(statement))

    def update_status(
        self,
        statement: models.GraphStatement,
        *,
        status: models.StatementStatus,
        needs_evidence: Optional[bool] = None,
        confidence: Optional[float] = None,
        resolution_notes: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> models.GraphStatement:
        statement.status = status
        if needs_evidence is not None:
            statement.needs_evidence = needs_evidence
        if confidence is not None:
            statement.confidence = confidence
        if resolution_notes is not None:
            statement.resolution_notes = resolution_notes
        if actor:
            statement.created_by = actor
        statement.updated_at = datetime.utcnow()
        self.session.add(statement)
        return statement

    def create_suggestion(
        self,
        *,
        parent_label: str,
        predicate: str,
        supporting_node_ids: Sequence[int],
        supporting_node_labels: Sequence[str],
        evidence: Dict[str, object],
        guardrail_flags: Sequence[str],
        confidence: float,
        llm_confidence: Optional[float],
        llm_rationale: Optional[str],
        raw_llm_response: Optional[Dict[str, object]],
        parent_description: Optional[str],
        created_by: Optional[str] = "ontology_inference",
    ) -> models.OntologySuggestion:
        sorted_ids = sorted(dict.fromkeys(int(node_id) for node_id in supporting_node_ids))
        suggestion = models.OntologySuggestion(
            parent_label=parent_label,
            parent_description=parent_description,
            predicate=predicate,
            supporting_node_ids=sorted_ids,
            supporting_node_labels=list(supporting_node_labels),
            evidence=dict(evidence),
            guardrail_flags=list(guardrail_flags),
            confidence=float(confidence),
            llm_confidence=float(llm_confidence) if llm_confidence is not None else None,
            llm_rationale=llm_rationale,
            raw_llm_response=json.dumps(raw_llm_response, ensure_ascii=False)
            if raw_llm_response is not None
            else None,
            created_by=created_by,
        )
        now = datetime.utcnow()
        suggestion.created_at = now
        suggestion.updated_at = now
        self.session.add(suggestion)
        self.session.flush()
        return suggestion

    def update_status(
        self,
        suggestion: models.OntologySuggestion,
        *,
        status: models.OntologySuggestionStatus,
        applied_parent_node_id: Optional[int] = None,
    ) -> models.OntologySuggestion:
        suggestion.status = status
        suggestion.updated_at = datetime.utcnow()
        if applied_parent_node_id is not None:
            suggestion.applied_parent_node_id = applied_parent_node_id
        self.session.add(suggestion)
        return suggestion


class NodeEmbeddingStore:
    """Shared in-memory embedding index for node labels."""

    _BACKEND: str = "fuzzy"
    _LABELS: List[str] = []
    _LABEL_METADATA: List[Dict[str, object]] = []
    _LABEL_TO_INDEX: Dict[str, int] = {}
    _NODE_ID_TO_INDEX: Dict[int, int] = {}
    _VECTORS: Optional[np.ndarray] = None
    _LOCK = threading.Lock()
    _EMBEDDING_DIM: Optional[int] = None
    _EXTERNAL_ENDPOINT: Optional[str] = None
    _EXTERNAL_API_KEY: Optional[str] = None
    _EXTERNAL_DEPLOYMENT: Optional[str] = None

    def __init__(self, embedding_model: str = "external"):
        """Initialise the similarity backend, preferring configured external embeddings. Loads persisted embeddings if available."""
        self.embedding_model = embedding_model
        if settings.embedding_endpoint:
            NodeEmbeddingStore._EXTERNAL_ENDPOINT = settings.embedding_endpoint
            NodeEmbeddingStore._EXTERNAL_API_KEY = settings.embedding_api_key
            NodeEmbeddingStore._EXTERNAL_DEPLOYMENT = settings.embedding_deployment
            NodeEmbeddingStore._BACKEND = "external"
        else:
            NodeEmbeddingStore._BACKEND = "fuzzy"

        # Try to load persisted embeddings
        labels, label_to_index, vectors, embedding_dim, metadata = load_embeddings()
        if labels:
            NodeEmbeddingStore._LABELS = list(labels)
            if metadata and len(metadata) == len(labels):
                NodeEmbeddingStore._LABEL_METADATA = [dict(item) for item in metadata]
            else:
                NodeEmbeddingStore._LABEL_METADATA = [
                    {"label": label, "entity_type": None, "node_id": None, "kind": "node"}
                    for label in labels
                ]

            if label_to_index:
                NodeEmbeddingStore._LABEL_TO_INDEX = dict(label_to_index)
            else:
                NodeEmbeddingStore._LABEL_TO_INDEX = {
                    label.lower(): idx for idx, label in enumerate(NodeEmbeddingStore._LABELS)
                }

            NodeEmbeddingStore._VECTORS = vectors
            NodeEmbeddingStore._EMBEDDING_DIM = embedding_dim

            NodeEmbeddingStore._NODE_ID_TO_INDEX = {}
            for idx, meta in enumerate(NodeEmbeddingStore._LABEL_METADATA):
                if not isinstance(meta, dict):
                    continue
                meta.setdefault("kind", "node")
                node_id = meta.get("node_id")
                if isinstance(node_id, int):
                    NodeEmbeddingStore._NODE_ID_TO_INDEX[node_id] = idx

    @property
    def backend(self) -> str:
        """Return the current similarity backend (``external`` or ``fuzzy``)."""
        return NodeEmbeddingStore._BACKEND

    @staticmethod
    def _format_label(label: str, entity_type: Optional[str]) -> str:
        cleaned_label = (label or "").strip()
        if not cleaned_label:
            return ""
        cleaned_type = (entity_type or "").strip()
        return f"{cleaned_label} [{cleaned_type}]" if cleaned_type else cleaned_label

    @staticmethod
    def _normalise_entry(entry: object) -> tuple[str, Optional[str], Optional[int], str]:
        label: Optional[str] = None
        entity_type: Optional[str] = None
        node_id: Optional[int] = None
        kind = "node"

        if entry is None:
            return "", None, None, kind

        if isinstance(entry, models.Node):
            label = entry.label
            entity_type = entry.entity_type
            node_id = entry.id
            kind = "node"
        elif isinstance(entry, dict):
            label = entry.get("label")  # type: ignore[arg-type]
            entity_type = entry.get("entity_type")  # type: ignore[arg-type]
            node_id_raw = entry.get("node_id")
            if isinstance(node_id_raw, int):
                node_id = node_id_raw
            kind = entry.get("kind") or ("predicate" if entity_type == "predicate" else "node")  # type: ignore[arg-type]
        elif isinstance(entry, (list, tuple)):
            if entry:
                label = entry[0] if isinstance(entry[0], str) else str(entry[0])
            if len(entry) > 1 and entry[1] is not None:
                entity_type = str(entry[1])
            if len(entry) > 2 and isinstance(entry[2], int):
                node_id = entry[2]
        elif isinstance(entry, str):
            label = entry

        label = (label or "").strip()
        if not label:
            return "", None, None, kind

        if entity_type is not None:
            entity_type = str(entity_type).strip() or None
        if kind not in {"node", "predicate"}:
            kind = "node"

        return label, entity_type, node_id, kind

    @staticmethod
    def _metadata_for_index(index: int) -> Dict[str, object]:
        try:
            meta = NodeEmbeddingStore._LABEL_METADATA[index]
            if isinstance(meta, dict):
                return meta
        except IndexError:
            pass
        label = NodeEmbeddingStore._LABELS[index] if index < len(NodeEmbeddingStore._LABELS) else ""
        return {"label": label, "entity_type": None, "node_id": None, "kind": "node"}

    @staticmethod
    def _persist_state() -> None:
        save_embeddings(
            NodeEmbeddingStore._LABELS,
            NodeEmbeddingStore._LABEL_TO_INDEX,
            NodeEmbeddingStore._VECTORS,
            NodeEmbeddingStore._EMBEDDING_DIM,
            NodeEmbeddingStore._LABEL_METADATA,
        )

    @staticmethod
    def _to_float(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    def has_entries(self) -> bool:
        """Return ``True`` when at least one label has been cached."""
        return bool(NodeEmbeddingStore._LABELS)

    def encode_labels(self, labels: Sequence[str]) -> Optional[np.ndarray]:
        """Encode ``labels`` using the configured external embedding endpoint."""
        if NodeEmbeddingStore._BACKEND != "external":
            return None
        if not labels:
            if NodeEmbeddingStore._EMBEDDING_DIM is None:
                return np.empty((0, 0))
            return np.empty((0, NodeEmbeddingStore._EMBEDDING_DIM))
        return self._embed_external(labels)

    def bulk_add(self, entries: Sequence[object]) -> None:
        """Add or refresh embedding entries based on the supplied metadata."""
        normalized: List[tuple[str, Optional[str], Optional[int], str, str]] = []
        for entry in entries:
            label, entity_type, node_id, kind = self._normalise_entry(entry)
            formatted = self._format_label(label, entity_type)
            if not formatted:
                continue
            normalized.append((label, entity_type, node_id, kind, formatted))

        if not normalized:
            return

        vectors_map: Dict[int, np.ndarray] = {}
        if NodeEmbeddingStore._BACKEND == "external":
            # Batch all the formatted labels for embedding at once
            formatted_labels = [formatted for _, _, _, _, formatted in normalized]
            if formatted_labels:
                batch_vectors = self._embed_external(formatted_labels)
                if batch_vectors is not None and batch_vectors.size:
                    for idx, vector in enumerate(batch_vectors):
                        vectors_map[idx] = vector
                    NodeEmbeddingStore._EMBEDDING_DIM = batch_vectors.shape[1]

        with NodeEmbeddingStore._LOCK:
            for idx, (label, entity_type, node_id, kind, formatted) in enumerate(normalized):
                key = formatted.lower()
                existing_idx = None
                if node_id is not None:
                    existing_idx = NodeEmbeddingStore._NODE_ID_TO_INDEX.get(node_id)
                if existing_idx is None:
                    existing_idx = NodeEmbeddingStore._LABEL_TO_INDEX.get(key)

                vector = vectors_map.get(idx)
                if vector is not None:
                    NodeEmbeddingStore._EMBEDDING_DIM = vector.shape[0]

                metadata = {
                    "label": label,
                    "entity_type": entity_type,
                    "node_id": node_id,
                    "kind": kind,
                }

                if existing_idx is not None:
                    old_key = NodeEmbeddingStore._LABELS[existing_idx].lower()
                    NodeEmbeddingStore._LABEL_TO_INDEX.pop(old_key, None)
                    NodeEmbeddingStore._LABELS[existing_idx] = formatted
                    NodeEmbeddingStore._LABEL_METADATA[existing_idx] = metadata
                    NodeEmbeddingStore._LABEL_TO_INDEX[key] = existing_idx
                    if node_id is not None:
                        NodeEmbeddingStore._NODE_ID_TO_INDEX[node_id] = existing_idx
                    if vector is not None:
                        if NodeEmbeddingStore._VECTORS is None:
                            NodeEmbeddingStore._VECTORS = vector.reshape(1, -1)
                        else:
                            NodeEmbeddingStore._VECTORS[existing_idx] = vector
                else:
                    NodeEmbeddingStore._LABELS.append(formatted)
                    NodeEmbeddingStore._LABEL_METADATA.append(metadata)
                    new_index = len(NodeEmbeddingStore._LABELS) - 1
                    NodeEmbeddingStore._LABEL_TO_INDEX[key] = new_index
                    if node_id is not None:
                        NodeEmbeddingStore._NODE_ID_TO_INDEX[node_id] = new_index
                    if vector is not None:
                        if NodeEmbeddingStore._VECTORS is None:
                            NodeEmbeddingStore._VECTORS = vector.reshape(1, -1)
                        else:
                            NodeEmbeddingStore._VECTORS = np.vstack([NodeEmbeddingStore._VECTORS, vector])
                    elif NodeEmbeddingStore._VECTORS is not None and NodeEmbeddingStore._EMBEDDING_DIM:
                        zeros = np.zeros((1, NodeEmbeddingStore._EMBEDDING_DIM))
                        NodeEmbeddingStore._VECTORS = np.vstack([NodeEmbeddingStore._VECTORS, zeros])

            NodeEmbeddingStore._persist_state()

    def suggest_similar_batch(
        self,
        labels: List[str],
        entity_types: Optional[List[Optional[str]]] = None,
        *,
        top_k: int = 5,
        include_predicates: bool = False,
    ) -> List[List[Dict[str, object]]]:
        """Return up to ``top_k`` records similar to each label using batched embedding calls."""
        if not labels or not NodeEmbeddingStore._LABELS:
            return [[] for _ in labels]

        # Prepare formatted queries
        if entity_types is None:
            entity_types = [None] * len(labels)  # type: ignore[assignment]
        
        formatted_queries = [
            self._format_label(label, entity_type) 
            for label, entity_type in zip(labels, entity_types)
        ]
        formatted_keys = [query.lower() for query in formatted_queries]
        
        all_results: List[List[Dict[str, object]]] = []

        if (
            NodeEmbeddingStore._BACKEND == "external"
            and NodeEmbeddingStore._VECTORS is not None
            and NodeEmbeddingStore._LABELS
        ):
            # Batch embed all queries at once
            embeddings = self._embed_external(formatted_queries)
            if embeddings is not None and embeddings.size > 0:
                for i, (formatted_key, query_vector) in enumerate(zip(formatted_keys, embeddings)):
                    results: List[Dict[str, object]] = []
                    similarities = NodeEmbeddingStore._VECTORS @ query_vector
                    order = np.argsort(-similarities)
                    for idx in order:
                        meta = self._metadata_for_index(idx)
                        candidate_label = NodeEmbeddingStore._LABELS[idx]
                        candidate_key = candidate_label.lower()
                        if candidate_key == formatted_key:
                            continue
                        kind_value = str(meta.get("kind") or "node").lower()
                        if not include_predicates and kind_value == "predicate":
                            continue
                        results.append(
                            {
                                "label": meta.get("label"),
                                "entity_type": meta.get("entity_type"),
                                "kind": meta.get("kind"),
                                "node_id": meta.get("node_id"),
                                "similarity": float(similarities[idx]),
                            }
                        )
                        if len(results) >= top_k:
                            break
                    all_results.append(results)
                return all_results
            else:
                logger.warning("Batch embedding call failed during suggest_similar_batch; falling back to fuzzy matching")

        # Fallback to fuzzy matching
        for formatted_query, formatted_key in zip(formatted_queries, formatted_keys):
            scores: List[Dict[str, object]] = []
            for idx, candidate_label in enumerate(NodeEmbeddingStore._LABELS):
                candidate_key = candidate_label.lower()
                if candidate_key == formatted_key:
                    continue
                meta = self._metadata_for_index(idx)
                kind_value = str(meta.get("kind") or "node").lower()
                if not include_predicates and kind_value == "predicate":
                    continue
                score = fuzz.token_set_ratio(formatted_query, candidate_label) / 100.0
                if score > 0:
                    scores.append(
                        {
                            "label": meta.get("label"),
                            "entity_type": meta.get("entity_type"),
                            "kind": meta.get("kind"),
                            "node_id": meta.get("node_id"),
                            "similarity": float(score),
                        }
                    )
            scores.sort(
                key=lambda item: NodeEmbeddingStore._to_float(item.get("similarity")),
                reverse=True,
            )
            all_results.append(scores[:top_k])
        
        return all_results

    def suggest_similar(
        self,
        label: str,
        entity_type: Optional[str] = None,
        *,
        top_k: int = 5,
        include_predicates: bool = False,
    ) -> List[Dict[str, object]]:
        """Return up to ``top_k`` records similar to ``label`` using the current backend."""
        if not NodeEmbeddingStore._LABELS:
            return []

        formatted_query = self._format_label(label, entity_type)
        formatted_key = formatted_query.lower()
        results: List[Dict[str, object]] = []

        if (
            NodeEmbeddingStore._BACKEND == "external"
            and NodeEmbeddingStore._VECTORS is not None
            and NodeEmbeddingStore._LABELS
        ):
            embedding = self._embed_external([formatted_query])
            if embedding is None or embedding.size == 0:
                logger.warning(
                    "External embedding call failed during query; falling back to fuzzy matching"
                )
            else:
                query_vector = embedding[0]
                similarities = NodeEmbeddingStore._VECTORS @ query_vector
                order = np.argsort(-similarities)
                for idx in order:
                    meta = self._metadata_for_index(idx)
                    candidate_label = NodeEmbeddingStore._LABELS[idx]
                    candidate_key = candidate_label.lower()
                    if candidate_key == formatted_key:
                        continue
                    kind_value = str(meta.get("kind") or "node").lower()
                    if not include_predicates and kind_value == "predicate":
                        continue
                    results.append(
                        {
                            "label": meta.get("label"),
                            "entity_type": meta.get("entity_type"),
                            "kind": meta.get("kind"),
                            "node_id": meta.get("node_id"),
                            "similarity": float(similarities[idx]),
                        }
                    )
                    if len(results) >= top_k:
                        break
                return results

        scores: List[Dict[str, object]] = []
        for idx, candidate_label in enumerate(NodeEmbeddingStore._LABELS):
            candidate_key = candidate_label.lower()
            if candidate_key == formatted_key:
                continue
            meta = self._metadata_for_index(idx)
            kind_value = str(meta.get("kind") or "node").lower()
            if not include_predicates and kind_value == "predicate":
                continue
            score = fuzz.token_set_ratio(formatted_query, candidate_label) / 100.0
            if score > 0:
                scores.append(
                    {
                        "label": meta.get("label"),
                        "entity_type": meta.get("entity_type"),
                        "kind": meta.get("kind"),
                        "node_id": meta.get("node_id"),
                        "similarity": float(score),
                    }
                )
        scores.sort(
            key=lambda item: NodeEmbeddingStore._to_float(item.get("similarity")),
            reverse=True,
        )
        return scores[:top_k]

    def bootstrap_from_session(self, session: Session) -> None:
        """Populate the store with existing node label+class from the database."""
        nodes = [node for node in session.exec(select(models.Node)) if node.label]
        self.bulk_add(nodes)

    def query(
        self,
        text: str,
        top_k: int = 8,
        *,
        include_kinds: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return (label, score) tuples similar to ``text`` using embeddings or fuzzy fallback."""
        if not NodeEmbeddingStore._LABELS:
            return []

        kinds_filter = {kind.lower() for kind in include_kinds} if include_kinds else None

        if (
            NodeEmbeddingStore._BACKEND == "external"
            and NodeEmbeddingStore._VECTORS is not None
            and NodeEmbeddingStore._LABELS
        ):
            embedding = self._embed_external([text])
            if embedding is not None and embedding.size:
                query_vector = embedding[0]
                similarities = NodeEmbeddingStore._VECTORS @ query_vector
                order = np.argsort(-similarities)
                results: List[Tuple[str, float]] = []
                for idx in order[: top_k * 2]:
                    meta = self._metadata_for_index(idx)
                    kind = str(meta.get("kind") or "node").lower()
                    if kinds_filter and kind not in kinds_filter:
                        continue
                    label = str(meta.get("label") or "").strip()
                    if not label:
                        continue
                    results.append((label, float(similarities[idx])))
                results.sort(key=lambda item: item[1], reverse=True)
                return results[:top_k]
            logger.warning("External embedding call failed; using fuzzy similarity for '%s'", text)

        scores: List[Tuple[str, float]] = []
        for idx, candidate_label in enumerate(NodeEmbeddingStore._LABELS):
            meta = self._metadata_for_index(idx)
            kind = str(meta.get("kind") or "node").lower()
            if kinds_filter and kind not in kinds_filter:
                continue
            score = fuzz.token_set_ratio(text, candidate_label) / 100.0
            if score > 0:
                label = str(meta.get("label") or "").strip()
                if not label:
                    continue
                scores.append((label, float(score)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _embed_external(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        """Call the configured embedding endpoint and return normalised vectors."""
        if not texts:
            return np.empty((0, NodeEmbeddingStore._EMBEDDING_DIM or 0))
        if not NodeEmbeddingStore._EXTERNAL_ENDPOINT:
            return None

        payload: Dict[str, object] = {"input": list(texts)}
        if NodeEmbeddingStore._EXTERNAL_DEPLOYMENT:
            payload["model"] = NodeEmbeddingStore._EXTERNAL_DEPLOYMENT

        headers = {"Content-Type": "application/json"}
        if NodeEmbeddingStore._EXTERNAL_API_KEY:
            headers["Authorization"] = f"Bearer {NodeEmbeddingStore._EXTERNAL_API_KEY}"
            headers["api-key"] = NodeEmbeddingStore._EXTERNAL_API_KEY

        try:
            logger.info(
                "Calling embedding endpoint %s with %d inputs",  # simple one-line log
                NodeEmbeddingStore._EXTERNAL_ENDPOINT,
                len(texts),
            )
            response = requests.post(
                NodeEmbeddingStore._EXTERNAL_ENDPOINT,
                headers=headers,
                data=json.dumps(payload),
                timeout=30,
            )
            response.raise_for_status()
            body = response.json()
        except Exception:  # noqa: BLE001
            logger.exception("Embedding request failed; reverting to fuzzy similarity")
            NodeEmbeddingStore._BACKEND = "fuzzy"
            return None

        vectors: List[Sequence[float]] = []
        if isinstance(body, dict):
            data_field = body.get("data")
            if isinstance(data_field, list):
                for item in data_field:
                    if isinstance(item, dict):
                        embedding = item.get("embedding") or item.get("vector")
                        if embedding is not None:
                            vectors.append(embedding)
            elif isinstance(body.get("embeddings"), list):
                vectors = body["embeddings"]
        elif isinstance(body, list):
            vectors = body

        if not vectors:
            logger.warning("Embedding endpoint returned no vectors; using fuzzy similarity")
            return None
        if len(vectors) != len(texts):
            logger.warning(
                "Embedding endpoint returned %s vectors for %s inputs",
                len(vectors),
                len(texts),
            )

        arr = np.array(vectors, dtype=float)
        if arr.ndim != 2:
            logger.warning("Embedding response has unexpected shape %s", arr.shape)
            return None

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        NodeEmbeddingStore._EMBEDDING_DIM = arr.shape[1]
        return arr

    def embed_texts(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        """Public wrapper for obtaining embeddings when an external backend is configured."""
        if not texts:
            return np.empty((0, NodeEmbeddingStore._EMBEDDING_DIM or 0))
        if NodeEmbeddingStore._BACKEND != "external":
            return None
        return self._embed_external(texts)

    def embed_texts_batched(
        self, 
        texts: Sequence[str], 
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    ) -> Optional[np.ndarray]:
        """Embed texts in batches for better API efficiency with large inputs."""
        if not texts:
            return np.empty((0, NodeEmbeddingStore._EMBEDDING_DIM or 0))
        if NodeEmbeddingStore._BACKEND != "external":
            return None
        
        # For small inputs, use regular method
        if len(texts) <= batch_size:
            return self._embed_external(texts)
        
        # Process in batches for large inputs
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_external(batch)
            if batch_embeddings is not None:
                all_embeddings.append(batch_embeddings)
            else:
                # If any batch fails, fall back to fuzzy
                return None
        
        if not all_embeddings:
            return None
        return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
