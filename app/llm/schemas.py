from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ExtractionTriple(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    confidence: Optional[float] = None
    source_reference: Optional[str] = None
    suggested_subject_label: Optional[str] = None
    suggested_object_label: Optional[str] = None


class ExtractionResponse(BaseModel):
    triples: list[ExtractionTriple]
    raw_response: dict


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
