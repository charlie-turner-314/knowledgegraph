from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.data import models
from app.data.repositories import NodeEmbeddingStore
from app.llm.client import LLMClient, get_client
from app.llm.schemas import (
    QueryAnswerResponse,
    QueryPlanResponse,
    QueryRetrievalResponse,
)


logger = logging.getLogger(__name__)


@dataclass
class QueryMatch:
    """Supporting triple returned by the query execution pipeline."""
    edge_id: Optional[int]
    subject: str
    predicate: str
    object: str
    source_document: str | None
    page_label: str | None
    subject_attributes: Dict[str, object]
    object_attributes: Dict[str, object]
    tags: List[str]


@dataclass
class QueryResult:
    """High-level response for a natural language query."""
    answers: List[str]
    matches: List[QueryMatch]
    note: str | None = None


class QueryService:
    """Natural language query pipeline orchestrating retrieval, planning, and answering."""

    def __init__(self, session: Session, *, llm_client: Optional[LLMClient] = None):
        """Store repository helpers and configure the embedding cache."""
        self.session = session
        self.llm_client = llm_client or get_client()
        self.embedding_store = NodeEmbeddingStore()
        if not self.embedding_store.has_entries():
            self.embedding_store.bootstrap_from_session(session)

    def ask(
        self,
        question: str,
        *,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> QueryResult:
        """Answer ``question`` using the multi-stage LLM workflow with fallbacks."""
        cleaned = (question or "").strip()
        if not cleaned:
            return QueryResult(
                answers=[],
                matches=[],
                note="Enter a more descriptive question to search the knowledge graph.",
            )

        logger.info("Query received: %s", cleaned)
        self._notify(progress_callback, "Selecting relevant nodes and predicates...")

        try:
            retrieval_context = self._build_retrieval_context(cleaned)
            logger.debug("Retrieval context prepared with %d nodes and %d edges", len(retrieval_context.get("nodes", [])), len(retrieval_context.get("edges", [])))
            self._notify(progress_callback, "Analyzing question for key entities...")
            retrieval = self._run_retrieval_stage(cleaned, retrieval_context)
            logger.info(
                "Retrieval stage identified %d focus nodes and %d predicates",
                len(retrieval.focus_nodes or []),
                len(retrieval.focus_predicates or []),
            )
            self._notify(progress_callback, "Planning graph traversal...")
            plan = self._run_planning_stage(cleaned, retrieval, retrieval_context)
            step_count = len(plan.plan.steps) if plan.plan else 0
            logger.info("Planning stage produced %d steps", step_count)
            self._notify(progress_callback, "Executing traversal and gathering evidence...")
            matches = self._execute_plan(plan, retrieval)
            logger.info("Execution stage captured %d matching triples", len(matches))

            if not matches:
                logger.warning("No matches found from plan; falling back to keyword search")
                fallback = self._keyword_fallback(cleaned)
                fallback.note = (
                    "LLM produced no matching triples; falling back to keyword search."
                )
                self._notify(progress_callback, "Using keyword fallback...")
                return fallback

            self._notify(progress_callback, "Summarizing answer...")
            answer = self._run_answer_stage(cleaned, plan, matches, retrieval)
            answers = answer.answers if answer and answer.answers else []
            note = None
            if answer and answer.notes:
                note = answer.notes
            if answer and answer.confidence is not None:
                answers.append(f"Confidence: {answer.confidence:.2f}")

            logger.info("Answer stage completed with %d sentences", len(answers))
            self._notify(progress_callback, "Done")
            return QueryResult(
                answers=answers,
                matches=matches,
                note=note,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM query pipeline failed; using keyword fallback")
            fallback = self._keyword_fallback(cleaned)
            suffix = f"LLM query pipeline failed: {exc}"
            fallback.note = (
                (fallback.note + "\n") if fallback.note else ""
            ) + suffix
            self._notify(progress_callback, "Pipeline error; using keyword fallback")
            return fallback

    # ------------------------------------------------------------------
    # Pipeline stages

    def _build_retrieval_context(self, question: str) -> Dict[str, object]:
        """Assemble candidate nodes/edges and metadata for LLM retrieval planning."""
        top_nodes = self._rank_nodes(question, top_k=8)
        node_labels = [item[0] for item in top_nodes]
        nodes = self._load_nodes_by_labels(node_labels)

        score_lookup = {label.lower(): score for label, score in top_nodes}

        node_id_to_label = {node.id: node.label for node in nodes}
        edges = self._load_edges_for_nodes(node_id_to_label.keys(), sample_limit=20)

        serialized_nodes = [
            {
                "id": node.id,
                "label": node.label,
                "entity_type": node.entity_type,
                "score": score_lookup.get((node.label or "").lower()),
                "attributes": self._serialize_node_attributes(node),
            }
            for node in nodes
        ]
        serialized_edges = [
            {
                "id": edge.id,
                "subject": edge.subject.label if edge.subject else None,
                "predicate": edge.predicate,
                "object": edge.object.label if edge.object else None,
                "tags": sorted({tag.label for tag in (edge.tags or [])}),
            }
            for edge in edges
        ]

        return {
            "nodes": serialized_nodes,
            "edges": serialized_edges,
            "question": question,
        }

    def _run_retrieval_stage(
        self,
        question: str,
        context: Dict[str, object],
    ) -> QueryRetrievalResponse:
        """Call the LLM to identify focus nodes and predicates."""
        response = self.llm_client.analyze_query(question=question, context=context)
        return response

    def _run_planning_stage(
        self,
        question: str,
        retrieval: QueryRetrievalResponse,
        context: Dict[str, object],
    ) -> QueryPlanResponse:
        """Ask the LLM to convert retrieval hints into a traversal plan."""
        response = self.llm_client.plan_query(
            question=question,
            retrieval=retrieval,
            context=context,
        )
        return response

    def _execute_plan(
        self,
        plan: QueryPlanResponse,
        retrieval: QueryRetrievalResponse,
    ) -> List[QueryMatch]:
        """Traverse the graph according to ``plan`` and collect supporting triples."""
        if not plan.plan or not plan.plan.steps:
            return []

        label_map = self._candidate_label_map(retrieval)
        all_matches: Dict[int, QueryMatch] = {}

        for step in plan.plan.steps:
            start_label = step.start_node
            node_ids = label_map.get(start_label.lower())
            if not node_ids:
                continue

            predicates = [p.lower() for p in step.predicates] if step.predicates else []
            directions = self._resolve_directions(step.direction)
            depth_limit = max(1, min(step.depth or 1, plan.plan.max_hops or 3))

            current_ids = set(node_ids)
            visited_nodes: Set[int] = set(node_ids)
            for hop in range(depth_limit):
                edges = self._fetch_edges_for_step(current_ids, directions, predicates)
                next_ids: Set[int] = set()
                for edge in edges:
                    match = self._to_query_match(edge)
                    if match.edge_id is not None:
                        all_matches.setdefault(match.edge_id, match)
                    if "outbound" in directions and edge.subject_node_id in current_ids:
                        if edge.object_node_id not in visited_nodes:
                            next_ids.add(edge.object_node_id)
                    if "inbound" in directions and edge.object_node_id in current_ids:
                        if edge.subject_node_id not in visited_nodes:
                            next_ids.add(edge.subject_node_id)
                visited_nodes.update(next_ids)
                current_ids = next_ids
                if not current_ids:
                    break

        return list(all_matches.values())

    def _run_answer_stage(
        self,
        question: str,
        plan: QueryPlanResponse,
        matches: List[QueryMatch],
        retrieval: QueryRetrievalResponse,
    ) -> QueryAnswerResponse | None:
        """Request an answer summary from the LLM using traversal results."""
        payload = {
            "question": question,
            "plan": plan.model_dump(),
            "matches": [
                {
                    "edge_id": match.edge_id,
                    "subject": match.subject,
                    "predicate": match.predicate,
                    "object": match.object,
                    "source_document": match.source_document,
                    "page_label": match.page_label,
                    "subject_attributes": match.subject_attributes,
                    "object_attributes": match.object_attributes,
                    "tags": match.tags,
                }
                for match in matches[:20]
            ],
            "retrieval": retrieval.model_dump(),
        }
        return self.llm_client.answer_query(payload)

    # ------------------------------------------------------------------
    # Helpers

    def _rank_nodes(self, text: str, top_k: int) -> List[Tuple[str, float]]:
        """Use the embedding store to find labels closest to ``text``."""
        suggestions = self.embedding_store.query(text, top_k=top_k)
        return suggestions

    def _load_nodes_by_labels(self, labels: Sequence[str]) -> List[models.Node]:
        """Return nodes whose labels match the supplied list."""
        if not labels:
            return []
        stmt = (
            select(models.Node)
            .where(models.Node.label.in_(labels))
            .options(selectinload(models.Node.attributes))
        )
        return list(self.session.exec(stmt))

    def _load_edges_for_nodes(self, node_ids: Iterable[int], sample_limit: int) -> List[models.Edge]:
        """Return edges touching ``node_ids`` up to ``sample_limit`` results."""
        node_ids = list(node_ids)
        if not node_ids:
            return []
        stmt = (
            select(models.Edge)
            .where(
                (models.Edge.subject_node_id.in_(node_ids))
                | (models.Edge.object_node_id.in_(node_ids))
            )
            .options(
                selectinload(models.Edge.subject).selectinload(models.Node.attributes),
                selectinload(models.Edge.object).selectinload(models.Node.attributes),
                selectinload(models.Edge.tags),
            )
            .limit(sample_limit)
        )
        return list(self.session.exec(stmt))

    def _candidate_label_map(
        self, retrieval: QueryRetrievalResponse
    ) -> Dict[str, Set[int]]:
        """Map lowercase node labels to sets of candidate node IDs from retrieval."""
        mapping: Dict[str, Set[int]] = defaultdict(set)
        if retrieval.focus_nodes:
            for node in retrieval.focus_nodes:
                if node.node_id is not None:
                    mapping[node.label.lower()].add(node.node_id)
        if retrieval.context_nodes:
            for node in retrieval.context_nodes:
                if node.get("id"):
                    label = (node.get("label") or "").lower()
                    mapping[label].add(node["id"])
        return mapping

    def _fetch_edges_for_step(
        self,
        node_ids: Iterable[int],
        directions: Set[str],
        predicates: Sequence[str],
    ) -> List[models.Edge]:
        """Fetch edges aligned with traversal ``directions`` and ``predicates``."""
        node_ids = list(node_ids)
        if not node_ids:
            return []

        predicate_filter = [pred.lower() for pred in predicates] if predicates else []

        clauses = []
        if "outbound" in directions:
            clauses.append(models.Edge.subject_node_id.in_(node_ids))
        if "inbound" in directions:
            clauses.append(models.Edge.object_node_id.in_(node_ids))
        if not clauses:
            return []

        stmt = select(models.Edge).where(or_(*clauses))

        stmt = stmt.options(
            selectinload(models.Edge.subject).selectinload(models.Node.attributes),
            selectinload(models.Edge.object).selectinload(models.Node.attributes),
            selectinload(models.Edge.sources)
            .selectinload(models.EdgeSource.document_chunk)
            .selectinload(models.DocumentChunk.document),
            selectinload(models.Edge.tags),
        )

        edges = list(self.session.exec(stmt))
        if predicate_filter:
            edges = [
                edge
                for edge in edges
                if edge.predicate and edge.predicate.lower() in predicate_filter
            ]
        return edges

    def _resolve_directions(self, directive: str | None) -> Set[str]:
        """Normalise a direction string into a set of traversal instructions."""
        directive = (directive or "outbound").lower()
        if directive == "both":
            return {"outbound", "inbound"}
        if directive == "inbound":
            return {"inbound"}
        return {"outbound"}

    def _to_query_match(self, edge: models.Edge) -> QueryMatch:
        """Convert an edge ORM object into a ``QueryMatch`` dataclass."""
        source = edge.sources[0] if edge.sources else None
        document = None
        page = None
        if source and source.document_chunk:
            if source.document_chunk.document:
                document = source.document_chunk.document.display_name
            page = source.document_chunk.page_label
        subject_attrs = self._node_attribute_map(edge.subject)
        object_attrs = self._node_attribute_map(edge.object)
        tags = sorted({tag.label for tag in (edge.tags or [])})
        return QueryMatch(
            edge_id=edge.id,
            subject=edge.subject.label if edge.subject else "",
            predicate=edge.predicate,
            object=edge.object.label if edge.object else "",
            source_document=document,
            page_label=page,
            subject_attributes=subject_attrs,
            object_attributes=object_attrs,
            tags=tags,
        )

    def _keyword_fallback(self, question: str) -> QueryResult:
        """Perform a simple keyword-based search when the LLM pipeline fails."""
        keywords = {
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", question)
            if len(word) > 3
        }
        if not keywords:
            return QueryResult(
                answers=[],
                matches=[],
                note="No keywords with length > 3 found in query.",
            )
        logger.info("Running keyword fallback for query")

        stmt = (
            select(models.Edge)
            .options(
                selectinload(models.Edge.subject).selectinload(models.Node.attributes),
                selectinload(models.Edge.object).selectinload(models.Node.attributes),
                selectinload(models.Edge.sources)
                .selectinload(models.EdgeSource.document_chunk)
                .selectinload(models.DocumentChunk.document),
                selectinload(models.Edge.tags),
            )
        )
        edges = self.session.exec(stmt).all()
        matches: List[QueryMatch] = []
        for edge in edges:
            haystack = " ".join([
                edge.subject.label if edge.subject else "",
                edge.predicate,
                edge.object.label if edge.object else "",
            ]).lower()
            if keywords.intersection(haystack.split()):
                matches.append(self._to_query_match(edge))

        answers = [
            f"Found {len(matches)} triples matching keywords {sorted(keywords)}"
        ] if matches else []
        return QueryResult(
            answers=answers,
            matches=matches,
            note="Keyword fallback result",
        )

    @staticmethod
    def _notify(callback: Optional[Callable[[str], None]], message: str) -> None:
        """Send progress updates to the optional callback, ignoring failures."""
        if callback:
            try:
                callback(message)
            except Exception:  # noqa: BLE001
                logger.debug("Progress callback failed for message '%s'", message, exc_info=True)

    @staticmethod
    def _attribute_value(attr: Optional[models.NodeAttribute]) -> Optional[object]:
        """Return the scalar value for ``attr`` regardless of storage columns."""
        if attr is None:
            return None
        if attr.data_type == models.NodeAttributeType.number:
            return attr.value_number
        if attr.data_type == models.NodeAttributeType.boolean:
            return attr.value_boolean
        return attr.value_text

    def _serialize_node_attributes(self, node: Optional[models.Node]) -> List[Dict[str, object]]:
        """Serialise node attributes into simple dictionaries for LLM payloads."""
        if node is None or not node.attributes:
            return []
        serialized: List[Dict[str, object]] = []
        for attr in node.attributes:
            serialized.append(
                {
                    "name": attr.name,
                    "type": attr.data_type.value if hasattr(attr.data_type, "value") else str(attr.data_type),
                    "value": self._attribute_value(attr),
                }
            )
        return serialized

    def _node_attribute_map(self, node: Optional[models.Node]) -> Dict[str, object]:
        """Return a name→value mapping for a node's attributes."""
        if node is None or not node.attributes:
            return {}
        mapping: Dict[str, object] = {}
        for attr in node.attributes:
            value = self._attribute_value(attr)
            if value is not None:
                mapping[attr.name] = value
        return mapping
