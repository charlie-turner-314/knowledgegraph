from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ExtractionAttribute(BaseModel):
    name: str = Field(..., min_length=1)
    value: str | float | int | bool


class ExtractionTriple(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    # Optional ontology / class labels provided directly by the extractor to avoid
    # post-hoc inference. When present they should map to an ontology entity label
    # (or a proposed new one marked for review upstream in the pipeline).
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    confidence: Optional[float] = None
    source_reference: Optional[str] = None
    suggested_subject_label: Optional[str] = None
    suggested_object_label: Optional[str] = None
    subject_attributes: List[ExtractionAttribute] = Field(default_factory=list)
    object_attributes: List[ExtractionAttribute] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    triples: list[ExtractionTriple]
    raw_response: dict


class ExtractionContextEntity(BaseModel):
    label: str = Field(..., min_length=1)
    aliases: List[str] = Field(default_factory=list)
    entity_type: Optional[str] = None
    description: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class ExtractionContextUpdate(BaseModel):
    summary: str = Field(..., min_length=1)
    entities: List[ExtractionContextEntity] = Field(default_factory=list)
    raw_response: dict = Field(default_factory=dict)


class OntologyParentSuggestion(BaseModel):
    parent_label: str = Field(..., min_length=1)
    relation: str = Field(default="is_a")
    description: Optional[str] = None
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    guardrail_flags: List[str] = Field(default_factory=list)


class OntologySuggestionResponse(BaseModel):
    suggestion: Optional[OntologyParentSuggestion] = None
    raw_response: dict = Field(default_factory=dict)


class RetrievalNode(BaseModel):
    label: str
    node_id: Optional[int] = None
    score: Optional[float] = None
    rationale: Optional[str] = None


class ConnectionEdgeProposal(BaseModel):
    subject: str
    predicate: str
    object: str
    rationale: Optional[str] = None


class ConnectionNodeProposal(BaseModel):
    label: str
    description: Optional[str] = None


class QueryRetrieval(BaseModel):
    focus_nodes: List[RetrievalNode] = Field(default_factory=list)
    focus_predicates: List[str] = Field(default_factory=list)
    intent: Optional[str] = None
    notes: Optional[str] = None


class QueryRetrievalResponse(BaseModel):
    focus_nodes: List[RetrievalNode] = Field(default_factory=list)
    focus_predicates: List[str] = Field(default_factory=list)
    intent: Optional[str] = None
    notes: Optional[str] = None
    context_nodes: List[dict] = Field(default_factory=list)
    raw_response: dict = Field(default_factory=dict)


class QueryPlanStep(BaseModel):
    start_node: str = Field(..., min_length=1)
    direction: str = Field(default="outbound")
    predicates: List[str] = Field(default_factory=list)
    depth: Optional[int] = Field(default=1)


class QueryPlan(BaseModel):
    steps: List[QueryPlanStep] = Field(default_factory=list)
    max_hops: Optional[int] = None
    rationale: Optional[str] = None


class QueryPlanResponse(BaseModel):
    plan: Optional[QueryPlan] = None
    warnings: List[str] = Field(default_factory=list)
    raw_response: dict = Field(default_factory=dict)


class QueryAnswerResponse(BaseModel):
    answers: List[str] = Field(default_factory=list)
    cited_edges: List[int] = Field(default_factory=list)
    confidence: Optional[float] = None
    notes: Optional[str] = None
    raw_response: dict = Field(default_factory=dict)


class ConnectionRecommendationResponse(BaseModel):
    new_nodes: List[ConnectionNodeProposal] = Field(default_factory=list)
    new_edges: List[ConnectionEdgeProposal] = Field(default_factory=list)
    notes: Optional[str] = None
    raw_response: dict = Field(default_factory=dict)


class NodeMergeDecision(BaseModel):
    use_existing: bool = Field(default=False)
    preferred_label: Optional[str] = None
    reason: Optional[str] = None
    raw_response: dict = Field(default_factory=dict)


class ReviewAgentCommand(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    action: Literal[
        "create_edge",
        "update_node",
        "delete_edge",
        "add_attribute",
        "noop",
        "upsert_node",
        "upsert_edge",
        "ensure_entity_type",
        "ensure_relationship_type",
        "ensure_ontology_type",
        "ensure_ontology_relationship",
        "upsert_entity",
        "update_candidate_status",
        "resolve_ontology_backlog_items",
        "accept_pending_candidate",
        "reject_pending_candidate",
        "request_clarification",
        "remove_attribute",
    ] = Field(validation_alias=AliasChoices("action", "op"))
    edge_id: Optional[int] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    node_label: Optional[str] = None
    entity_type: Optional[str] = None
    definition: Optional[str] = None
    relationship: Optional[str] = None
    candidate_id: Optional[int] = None
    id: Optional[int] = None
    name: Optional[str] = None
    domain: Optional[str] = None
    range: Optional[str] = None
    id_alias: Optional[str] = None
    subject_attributes: List[Dict[str, Any]] = Field(default_factory=list)
    object_attributes: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    confidence: Optional[float] = None
    source_document_id: Optional[int] = None
    source_chunk_id: Optional[int] = None
    source_text: Optional[str] = None
    rationale: Optional[str] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    types: Optional[List[str]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    subject_ref: Optional[Dict[str, Any]] = None
    object_ref: Optional[Dict[str, Any]] = None
    # Provenance may come through as a free-form string (e.g. "Human Input") or a structured object.
    provenance: Optional[Dict[str, Any] | str] = None
    applied_edge_ref: Optional[Dict[str, Any]] = None
    items: Optional[List[str]] = None


class ReviewAgentResponse(BaseModel):
    reply: str = Field(default="")
    commands: List[ReviewAgentCommand] = Field(default_factory=list)
    raw_response: dict = Field(default_factory=dict)
