from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from rapidfuzz import fuzz
from sqlmodel import Session, select

from app.data import models
from app.data.repositories import (
    GraphRepository,
    NodeEmbeddingStore,
    OntologySuggestionRepository,
    SMEActionRepository,
)
from app.llm.client import GemmaClient, get_client


logger = logging.getLogger(__name__)

_BLOCKING_GUARDRAILS: Set[str] = {"needs_review"}


@dataclass
class ClusterAnalysis:
    indices: List[int]
    node_ids: List[int]
    labels: List[str]
    backend: str
    pair_scores: List[float]
    average_similarity: float
    min_similarity: float


class OntologyInferenceService:
    def __init__(self, session: Session, *, llm_client: Optional[GemmaClient] = None):
        self.session = session
        self.graph = GraphRepository(session)
        self.actions = SMEActionRepository(session)
        self.suggestions = OntologySuggestionRepository(session)
        self.llm_client = llm_client or get_client()
        self.embedding_store = NodeEmbeddingStore()
        if not self.embedding_store.has_entries():
            self.embedding_store.bootstrap_from_session(session)

    # ---------------------------------------------------------------------
    # Suggestion generation

    def generate_suggestions(
        self,
        *,
        similarity_threshold: float = 0.82,
        min_cluster_size: int = 2,
        limit: Optional[int] = None,
        created_by: str = "ontology_inference",
    ) -> List[models.OntologySuggestion]:
        nodes: List[models.Node] = [
            node for node in self.session.exec(select(models.Node)) if node.label
        ]
        if len(nodes) < min_cluster_size:
            return []

        node_lookup: Dict[int, models.Node] = {node.id: node for node in nodes if node.id is not None}
        node_records: List[tuple[int, str]] = [
            (node.id, node.label) for node in nodes if node.id is not None
        ]

        labels = [label for _, label in node_records]
        similarity_matrix, backend = self._build_similarity_matrix(labels)
        cluster_indices = self._cluster_indices(similarity_matrix, similarity_threshold, min_cluster_size)
        if not cluster_indices:
            return []

        analyses = [
            self._describe_cluster(indices, node_records, similarity_matrix, backend)
            for indices in cluster_indices
        ]
        analyses.sort(key=lambda analysis: analysis.average_similarity, reverse=True)

        isa_edges = list(
            self.session.exec(
                select(models.Edge).where(models.Edge.predicate == "is_a")
            )
        )
        parent_map: Dict[int, Set[int]] = {}
        for edge in isa_edges:
            parent_map.setdefault(edge.subject_node_id, set()).add(edge.object_node_id)

        existing_label_map = {node.label.lower(): node.id for node in nodes if node.label}

        created: List[models.OntologySuggestion] = []
        for analysis in analyses:
            cluster_node_ids = analysis.node_ids
            cluster_labels = analysis.labels

            existing = self.suggestions.find_for_nodes(
                cluster_node_ids,
                statuses=(
                    models.OntologySuggestionStatus.pending,
                    models.OntologySuggestionStatus.applied,
                ),
            )
            if existing:
                continue

            parent_sets = [parent_map.get(node_id, set()) for node_id in cluster_node_ids]
            shared_parents = self._shared_parents(parent_sets)
            if shared_parents:
                continue

            parent_label_candidates = {
                node_lookup[parent_id].label
                for parents in parent_sets
                for parent_id in parents
                if parent_id in node_lookup and node_lookup[parent_id].label
            }

            llm_response = self.llm_client.suggest_parent_node(
                child_labels=cluster_labels,
                existing_parent_labels=sorted(parent_label_candidates),
            )
            llm_suggestion = llm_response.suggestion
            if llm_suggestion is None:
                logger.debug(
                    "LLM declined to suggest parent for cluster %s", cluster_labels
                )
                continue

            parent_label = llm_suggestion.parent_label.strip()
            if not parent_label:
                continue

            predicate = (llm_suggestion.relation or "is_a").strip() or "is_a"
            guardrail_flags = set(llm_suggestion.guardrail_flags or [])

            if parent_label.lower() in {label.lower() for label in cluster_labels}:
                guardrail_flags.add("parent_matches_child")

            existing_conflict = existing_label_map.get(parent_label.lower())
            if existing_conflict and existing_conflict not in cluster_node_ids:
                guardrail_flags.add("parent_label_conflicts_existing")

            if analysis.average_similarity < similarity_threshold + 0.05:
                guardrail_flags.add("weak_similarity_signal")

            llm_confidence = llm_suggestion.confidence
            if llm_confidence is not None and llm_confidence < 0.5:
                guardrail_flags.add("low_llm_confidence")

            final_confidence = self._blend_confidence(
                embedding_component=analysis.average_similarity,
                llm_component=llm_confidence,
            )

            evidence = {
                "backend": analysis.backend,
                "pair_scores": [round(score, 4) for score in analysis.pair_scores],
                "average_similarity": round(analysis.average_similarity, 4),
                "min_similarity": round(analysis.min_similarity, 4),
                "threshold": similarity_threshold,
            }

            suggestion = self.suggestions.create_suggestion(
                parent_label=parent_label,
                predicate=predicate,
                supporting_node_ids=cluster_node_ids,
                supporting_node_labels=cluster_labels,
                evidence=evidence,
                guardrail_flags=sorted(guardrail_flags),
                confidence=final_confidence,
                llm_confidence=llm_confidence,
                llm_rationale=llm_suggestion.rationale,
                raw_llm_response=llm_response.raw_response,
                parent_description=llm_suggestion.description,
                created_by=created_by,
            )

            created.append(suggestion)
            if limit is not None and len(created) >= limit:
                break

        return created

    # ------------------------------------------------------------------
    # Suggestion application & moderation

    def apply_suggestion(
        self,
        suggestion_id: int,
        *,
        actor: Optional[str] = None,
        enforce_guardrails: bool = True,
    ) -> models.OntologySuggestion:
        suggestion = self.suggestions.get(suggestion_id)
        if suggestion is None:
            raise ValueError("Suggestion not found")
        if suggestion.status != models.OntologySuggestionStatus.pending:
            raise ValueError("Suggestion already resolved")

        if enforce_guardrails and any(
            flag in _BLOCKING_GUARDRAILS for flag in (suggestion.guardrail_flags or [])
        ):
            raise ValueError("Suggestion flagged for manual review; cannot auto-apply")

        parent_node = self.graph.nodes.ensure_node(
            label=suggestion.parent_label,
            entity_type=None,
            canonical_term=None,
            sme_override=True,
        )
        self.embedding_store.bulk_add([parent_node.label])

        action_payload = {
            "suggestion_id": suggestion.id,
            "confidence": suggestion.confidence,
            "guardrail_flags": suggestion.guardrail_flags,
        }
        sme_action = self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload=action_payload,
        )

        for node_id, node_label in zip(
            suggestion.supporting_node_ids or [],
            suggestion.supporting_node_labels or [],
        ):
            node = self.session.get(models.Node, node_id)
            if node is None:
                logger.warning(
                    "Skipping missing node %s while applying suggestion %s",
                    node_id,
                    suggestion.id,
                )
                continue
            if self._edge_exists(node.id, parent_node.id, suggestion.predicate):
                continue

            self.graph.create_edge_with_provenance(
                subject_label=node.label,
                predicate=suggestion.predicate,
                object_label=parent_node.label,
                entity_type_subject=node.entity_type,
                entity_type_object=parent_node.entity_type,
                canonical_subject=node.canonical_term,
                canonical_object=parent_node.canonical_term,
                candidate=None,
                document_chunk=None,
                sme_action=sme_action,
                created_by=actor or "ontology_inference",
            )

        self.suggestions.update_status(
            suggestion,
            status=models.OntologySuggestionStatus.applied,
            applied_parent_node_id=parent_node.id,
        )
        return suggestion

    def reject_suggestion(
        self,
        suggestion_id: int,
        *,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> models.OntologySuggestion:
        suggestion = self.suggestions.get(suggestion_id)
        if suggestion is None:
            raise ValueError("Suggestion not found")
        if suggestion.status != models.OntologySuggestionStatus.pending:
            raise ValueError("Suggestion already resolved")

        payload = {
            "suggestion_id": suggestion.id,
            "reason": reason,
        }
        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload=payload,
        )
        self.suggestions.update_status(
            suggestion,
            status=models.OntologySuggestionStatus.rejected,
        )
        return suggestion

    # ------------------------------------------------------------------
    # Helpers

    def _build_similarity_matrix(
        self, labels: Sequence[str]
    ) -> tuple[np.ndarray, str]:
        embeddings = self.embedding_store.encode_labels(labels)
        if embeddings is not None and embeddings.size:
            similarity = embeddings @ embeddings.T
            similarity = np.clip(similarity, -1.0, 1.0)
            return similarity, "embedding"

        n = len(labels)
        similarity = np.eye(n)
        for i, j in combinations(range(n), 2):
            score = fuzz.token_set_ratio(labels[i], labels[j]) / 100.0
            similarity[i, j] = similarity[j, i] = score
        return similarity, "fuzzy"

    def _cluster_indices(
        self,
        similarity: np.ndarray,
        threshold: float,
        min_cluster_size: int,
    ) -> List[List[int]]:
        n = similarity.shape[0]
        visited: Set[int] = set()
        clusters: List[List[int]] = []
        for idx in range(n):
            if idx in visited:
                continue
            cluster = {idx}
            stack = [idx]
            while stack:
                current = stack.pop()
                for candidate in range(n):
                    if candidate in cluster:
                        continue
                    if similarity[current, candidate] >= threshold:
                        cluster.add(candidate)
                        stack.append(candidate)
            visited.update(cluster)
            if len(cluster) >= min_cluster_size:
                clusters.append(sorted(cluster))
        return clusters

    def _describe_cluster(
        self,
        indices: Sequence[int],
        node_records: Sequence[tuple[int, str]],
        similarity: np.ndarray,
        backend: str,
    ) -> ClusterAnalysis:
        pair_scores = [
            float(similarity[i, j])
            for i, j in combinations(indices, 2)
        ]
        if pair_scores:
            average_similarity = sum(pair_scores) / len(pair_scores)
            min_similarity = min(pair_scores)
        else:
            average_similarity = 0.0
            min_similarity = 0.0

        node_ids = [node_records[i][0] for i in indices]
        labels = [node_records[i][1] for i in indices]

        return ClusterAnalysis(
            indices=list(indices),
            node_ids=node_ids,
            labels=labels,
            backend=backend,
            pair_scores=pair_scores,
            average_similarity=average_similarity,
            min_similarity=min_similarity,
        )

    @staticmethod
    def _shared_parents(parent_sets: Sequence[Set[int]]) -> Set[int]:
        filtered = [parents for parents in parent_sets if parents]
        if not filtered:
            return set()
        shared = set(filtered[0])
        for parents in filtered[1:]:
            shared &= parents
        return shared

    @staticmethod
    def _blend_confidence(
        *,
        embedding_component: float,
        llm_component: Optional[float],
    ) -> float:
        embedding_component = max(0.0, min(1.0, embedding_component))
        if llm_component is None:
            return round(embedding_component, 3)
        llm_component = max(0.0, min(1.0, llm_component))
        blended = (embedding_component * 0.6) + (llm_component * 0.4)
        return round(min(1.0, blended), 3)

    def _edge_exists(self, subject_id: int, object_id: int, predicate: str) -> bool:
        statement = (
            select(models.Edge)
            .where(models.Edge.subject_node_id == subject_id)
            .where(models.Edge.object_node_id == object_id)
            .where(models.Edge.predicate == predicate)
            .limit(1)
        )
        return self.session.exec(statement).first() is not None
