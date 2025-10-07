import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import CheckConstraint, Column, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlmodel import Field, Relationship, SQLModel


# --- Enumerations -----------------------------------------------------------------


class CandidateStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    needs_revision = "needs_revision"


class EdgeSourceType(str, enum.Enum):
    document_chunk = "document_chunk"
    sme_note = "sme_note"


class SMEActionType(str, enum.Enum):
    accept_triple = "accept_triple"
    reject_triple = "reject_triple"
    edit_triple = "edit_triple"
    add_node_alias = "add_node_alias"
    system = "system"


class NodeAttributeType(str, enum.Enum):
    string = "string"
    number = "number"
    boolean = "boolean"
    enum = "enum"


class StatementStatus(str, enum.Enum):
    draft = "draft"
    needs_evidence = "needs_evidence"
    validated = "validated"
    rejected = "rejected"


class OntologySuggestionStatus(str, enum.Enum):
    pending = "pending"
    applied = "applied"
    rejected = "rejected"


# --- Core domain tables ------------------------------------------------------------


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    display_name: str = Field(index=True)
    original_path: Optional[str] = Field(default=None)
    media_type: str = Field(index=True)
    checksum: str = Field(index=True, unique=True)
    extra: dict = Field(default_factory=dict, sa_column=Column(SQLiteJSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    chunks: List["DocumentChunk"] = Relationship(back_populates="document")


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="documents.id", nullable=False, index=True)
    ordering: int = Field(nullable=False)
    text: str = Field(nullable=False)
    page_label: Optional[str] = Field(default=None, index=True)
    token_count: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    document: Optional[Document] = Relationship(back_populates="chunks")
    candidates: List["CandidateTriple"] = Relationship(back_populates="chunk")


class CanonicalTerm(SQLModel, table=True):
    __tablename__ = "canonical_terms"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    label: str = Field(index=True)
    entity_type: Optional[str] = Field(default=None, index=True)
    description: Optional[str] = Field(default=None)
    aliases: List[str] = Field(default_factory=list, sa_column=Column(SQLiteJSON))
    authority_source: Optional[str] = Field(default=None)
    last_reviewed_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    nodes: List["Node"] = Relationship(back_populates="canonical_term")


class Node(SQLModel, table=True):
    __tablename__ = "nodes"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    label: str = Field(index=True)
    entity_type: Optional[str] = Field(default=None, index=True)
    canonical_term_id: Optional[int] = Field(foreign_key="canonical_terms.id")
    sme_override: bool = Field(default=False, nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    canonical_term: Optional[CanonicalTerm] = Relationship(back_populates="nodes")
    outgoing_edges: List["Edge"] = Relationship(
        back_populates="subject",
        sa_relationship_kwargs={"foreign_keys": "Edge.subject_node_id"},
    )
    incoming_edges: List["Edge"] = Relationship(
        back_populates="object",
        sa_relationship_kwargs={"foreign_keys": "Edge.object_node_id"},
    )
    attributes: List["NodeAttribute"] = Relationship(
        back_populates="node",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class CandidateTriple(SQLModel, table=True):
    __tablename__ = "candidate_triples"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(foreign_key="document_chunks.id", nullable=False, index=True)
    subject_text: str = Field(nullable=False)
    predicate_text: str = Field(nullable=False)
    object_text: str = Field(nullable=False)
    llm_confidence: Optional[float] = Field(default=None)
    llm_response_fragment: Optional[str] = Field(default=None)
    status: CandidateStatus = Field(default=CandidateStatus.pending, nullable=False)
    suggested_subject_term_id: Optional[int] = Field(foreign_key="canonical_terms.id")
    suggested_object_term_id: Optional[int] = Field(foreign_key="canonical_terms.id")
    duplicate_of_candidate_id: Optional[int] = Field(
        default=None,
        foreign_key="candidate_triples.id",
    )
    is_potential_duplicate: bool = Field(default=False, nullable=False)
    subject_attributes: List[dict] = Field(default_factory=list, sa_column=Column(SQLiteJSON))
    object_attributes: List[dict] = Field(default_factory=list, sa_column=Column(SQLiteJSON))
    suggested_tags: List[str] = Field(default_factory=list, sa_column=Column(SQLiteJSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    chunk: Optional[DocumentChunk] = Relationship(back_populates="candidates")
    subject_suggestion: Optional[CanonicalTerm] = Relationship(sa_relationship_kwargs={"primaryjoin": "CandidateTriple.suggested_subject_term_id==CanonicalTerm.id"})
    object_suggestion: Optional[CanonicalTerm] = Relationship(sa_relationship_kwargs={"primaryjoin": "CandidateTriple.suggested_object_term_id==CanonicalTerm.id"})
    actions: List["SMEAction"] = Relationship(back_populates="candidate")


class Edge(SQLModel, table=True):
    __tablename__ = "edges"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    subject_node_id: int = Field(foreign_key="nodes.id", nullable=False, index=True)
    predicate: str = Field(nullable=False, index=True)
    object_node_id: int = Field(foreign_key="nodes.id", nullable=False, index=True)
    candidate_id: Optional[int] = Field(foreign_key="candidate_triples.id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    created_by: Optional[str] = Field(default=None)

    subject: Optional[Node] = Relationship(back_populates="outgoing_edges", sa_relationship_kwargs={"foreign_keys": "Edge.subject_node_id"})
    object: Optional[Node] = Relationship(back_populates="incoming_edges", sa_relationship_kwargs={"foreign_keys": "Edge.object_node_id"})
    candidate: Optional[CandidateTriple] = Relationship()
    sources: List["EdgeSource"] = Relationship(back_populates="edge", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
    tags: List["EdgeTag"] = Relationship(
        back_populates="edge",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    statements: List["GraphStatement"] = Relationship(back_populates="edge")


class EdgeSource(SQLModel, table=True):
    __tablename__ = "edge_sources"
    __table_args__ = (
        CheckConstraint(
            "(source_type='document_chunk' AND document_chunk_id IS NOT NULL) OR (source_type='sme_note' AND sme_action_id IS NOT NULL)",
            name="edge_source_valid_reference",
        ),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    edge_id: int = Field(foreign_key="edges.id", nullable=False, index=True)
    source_type: EdgeSourceType = Field(nullable=False)
    document_chunk_id: Optional[int] = Field(foreign_key="document_chunks.id", default=None)
    sme_action_id: Optional[int] = Field(foreign_key="sme_actions.id", default=None)
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    edge: Optional[Edge] = Relationship(back_populates="sources")
    document_chunk: Optional[DocumentChunk] = Relationship()
    sme_action: Optional["SMEAction"] = Relationship()


class SMEAction(SQLModel, table=True):
    __tablename__ = "sme_actions"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    action_type: SMEActionType = Field(nullable=False, index=True)
    actor: Optional[str] = Field(default=None)
    candidate_id: Optional[int] = Field(foreign_key="candidate_triples.id", default=None, index=True)
    payload: dict = Field(default_factory=dict, sa_column=Column(SQLiteJSON))
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    candidate: Optional[CandidateTriple] = Relationship(back_populates="actions")
    edge_sources: List["EdgeSource"] = Relationship(back_populates="sme_action")


class GraphStatement(SQLModel, table=True):
    __tablename__ = "graph_statements"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    subject_node_id: Optional[int] = Field(foreign_key="nodes.id", index=True)
    predicate: str = Field(nullable=False, index=True)
    object_node_id: Optional[int] = Field(foreign_key="nodes.id", index=True)
    edge_id: Optional[int] = Field(foreign_key="edges.id", index=True, default=None)
    candidate_id: Optional[int] = Field(foreign_key="candidate_triples.id", index=True, default=None)
    status: StatementStatus = Field(default=StatementStatus.draft, nullable=False, index=True)
    confidence: Optional[float] = Field(default=None)
    needs_evidence: bool = Field(default=False, nullable=False)
    rationale: Optional[str] = Field(default=None)
    resolution_notes: Optional[str] = Field(default=None)
    created_by: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    subject: Optional[Node] = Relationship(sa_relationship_kwargs={"foreign_keys": "GraphStatement.subject_node_id"})
    object: Optional[Node] = Relationship(sa_relationship_kwargs={"foreign_keys": "GraphStatement.object_node_id"})
    edge: Optional[Edge] = Relationship(back_populates="statements")
    candidate: Optional[CandidateTriple] = Relationship()


class NodeAttribute(SQLModel, table=True):
    __tablename__ = "node_attributes"
    __table_args__ = (
        CheckConstraint(
            "value_number IS NOT NULL OR value_text IS NOT NULL OR value_boolean IS NOT NULL",
            name="node_attribute_value_not_null",
        ),
        CheckConstraint(
            "data_type IN ('string','number','boolean','enum')",
            name="node_attribute_valid_type",
        ),
        UniqueConstraint("node_id", "name", name="uq_node_attribute_node_name"),
        {"extend_existing": True},
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: int = Field(foreign_key="nodes.id", nullable=False, index=True)
    name: str = Field(nullable=False, index=True)
    data_type: NodeAttributeType = Field(default=NodeAttributeType.string, nullable=False)
    value_text: Optional[str] = Field(default=None)
    value_number: Optional[float] = Field(default=None)
    value_boolean: Optional[bool] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    node: Optional[Node] = Relationship(back_populates="attributes")


class EdgeTag(SQLModel, table=True):
    __tablename__ = "edge_tags"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    edge_id: int = Field(foreign_key="edges.id", nullable=False, index=True)
    label: str = Field(nullable=False, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    edge: Optional[Edge] = Relationship(back_populates="tags")


class OntologySuggestion(SQLModel, table=True):
    __tablename__ = "ontology_suggestions"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    parent_label: str = Field(nullable=False, index=True)
    parent_description: Optional[str] = Field(default=None)
    predicate: str = Field(default="is_a", nullable=False, index=True)
    supporting_node_ids: List[int] = Field(default_factory=list, sa_column=Column(SQLiteJSON))
    supporting_node_labels: List[str] = Field(default_factory=list, sa_column=Column(SQLiteJSON))
    evidence: dict = Field(default_factory=dict, sa_column=Column(SQLiteJSON))
    llm_rationale: Optional[str] = Field(default=None)
    guardrail_flags: List[str] = Field(default_factory=list, sa_column=Column(SQLiteJSON))
    confidence: float = Field(default=0.0)
    llm_confidence: Optional[float] = Field(default=None)
    status: OntologySuggestionStatus = Field(default=OntologySuggestionStatus.pending, nullable=False)
    applied_parent_node_id: Optional[int] = Field(foreign_key="nodes.id", default=None)
    raw_llm_response: Optional[str] = Field(default=None)
    created_by: Optional[str] = Field(default="ontology_inference")
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


# --- Helper data classes -----------------------------------------------------------


class Triple(SQLModel):
    subject: str
    predicate: str
    object: str
