from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from app.data import models
from app.data.repositories import GraphRepository, NodeEmbeddingStore
from app.llm.client import LLMClient, get_client


@dataclass
class ClusterRecommendation:
    """Result payload summarising cluster analysis and proposals."""
    node_summaries: List[Dict[str, object]]
    existing_connections: List[Dict[str, object]]
    missing_pairs: List[Tuple[str, str]]
    proposed_nodes: List[Dict[str, object]]
    proposed_edges: List[Dict[str, object]]
    notes: Optional[str]


class DisjointSet:
    """Union-find structure for clustering node IDs."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, item: int) -> int:
        """Return the canonical representative for ``item``."""
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        """Merge the sets that contain ``a`` and ``b``."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        self.parent[root_b] = root_a


class GraphInsightService:
    """Analyzes the current knowledge graph to suggest missing connections."""

    def __init__(self, session: Session, *, llm_client: Optional[LLMClient] = None):
        self.session = session
        self.graph = GraphRepository(session)
        self.embedding_store = NodeEmbeddingStore()
        if not self.embedding_store.has_entries():
            self.embedding_store.bootstrap_from_session(session)
        self.llm_client = llm_client or get_client()

    def scan_for_recommendations(
        self,
        *,
        similarity_threshold: float = 0.82,
        top_k: int = 6,
        max_clusters: int = 10,
    ) -> List[ClusterRecommendation]:
        """Cluster similar nodes and request LLM suggestions for disconnected groups."""
        nodes = self._load_nodes()
        if len(nodes) < 2:
            return []

        label_lookup: Dict[str, List[models.Node]] = {}
        for node in nodes:
            if node.label:
                label_lookup.setdefault(node.label.lower(), []).append(node)

        dsu = DisjointSet()
        for node in nodes:
            suggestions = self.embedding_store.suggest_similar(node.label, top_k=top_k)
            for label, score in suggestions:
                if score < similarity_threshold:
                    continue
                matches = label_lookup.get(label.lower())
                if not matches:
                    continue
                for match in matches:
                    dsu.union(node.id, match.id)

        clusters: Dict[int, List[models.Node]] = {}
        for node in nodes:
            root = dsu.find(node.id)
            clusters.setdefault(root, []).append(node)

        # Filter clusters by size and limit total to scan
        significant_clusters = [
            items for items in clusters.values() if len(items) > 1
        ]
        significant_clusters.sort(key=len, reverse=True)
        significant_clusters = significant_clusters[:max_clusters]

        if not significant_clusters:
            return []

        edges = list(self.session.exec(select(models.Edge)))
        adjacency = self._build_adjacency(edges)
        id_to_node = {node.id: node for node in nodes}

        recommendations: List[ClusterRecommendation] = []
        for cluster_nodes in significant_clusters:
            cluster_ids = [node.id for node in cluster_nodes]
            missing_pairs = self._find_missing_pairs(cluster_ids, adjacency)
            if not missing_pairs:
                continue

            node_summaries = [
                {
                    "node_id": node.id,
                    "label": node.label,
                    "attributes": self._serialize_node_attributes(node),
                }
                for node in cluster_nodes
            ]

            existing_connections = self._collect_existing_connections(cluster_ids, edges, id_to_node)

            payload = {
                "cluster_nodes": node_summaries,
                "existing_connections": existing_connections,
                "missing_pairs": [
                    {
                        "source": id_to_node[pair[0]].label,
                        "target": id_to_node[pair[1]].label,
                    }
                    for pair in missing_pairs
                ],
            }

            llm_response = self.llm_client.recommend_connections(payload)

            recommendations.append(
                ClusterRecommendation(
                    node_summaries=node_summaries,
                    existing_connections=existing_connections,
                    missing_pairs=[
                        (
                            id_to_node[pair[0]].label,
                            id_to_node[pair[1]].label,
                        )
                        for pair in missing_pairs
                    ],
                    proposed_nodes=[
                        proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
                        for proposal in llm_response.new_nodes
                    ],
                    proposed_edges=[
                        proposal.model_dump() if hasattr(proposal, "model_dump") else proposal
                        for proposal in llm_response.new_edges
                    ],
                    notes=llm_response.notes,
                )
            )

        return recommendations

    def create_placeholder_statement(
        self,
        *,
        subject_label: Optional[str],
        predicate: str,
        object_label: Optional[str],
        rationale: Optional[str],
        created_by: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> models.GraphStatement:
        """Record a needs-evidence statement anchoring a prospective connection."""

        return self.graph.create_statement_placeholder(
            subject_label=subject_label,
            predicate=predicate,
            object_label=object_label,
            created_by=created_by,
            rationale=rationale,
            confidence=confidence,
        )

    def _load_nodes(self) -> List[models.Node]:
        """Return all nodes with eager-loaded attributes."""
        stmt = select(models.Node).options(selectinload(models.Node.attributes))
        return list(self.session.exec(stmt))

    def _build_adjacency(self, edges: Iterable[models.Edge]) -> Dict[int, Set[int]]:
        """Create an undirected adjacency map for quick path checks."""
        adjacency: Dict[int, Set[int]] = {}
        for edge in edges:
            adjacency.setdefault(edge.subject_node_id, set()).add(edge.object_node_id)
            adjacency.setdefault(edge.object_node_id, set()).add(edge.subject_node_id)
        return adjacency

    def _find_missing_pairs(
        self,
        cluster_ids: Sequence[int],
        adjacency: Dict[int, Set[int]],
    ) -> List[Tuple[int, int]]:
        """Return node ID pairs that lack a path within the cluster."""
        missing: List[Tuple[int, int]] = []
        for a, b in itertools.combinations(cluster_ids, 2):
            if not self._has_path(a, b, adjacency):
                missing.append((a, b))
        return missing

    def _has_path(
        self,
        start: int,
        goal: int,
        adjacency: Dict[int, Set[int]],
        *,
        max_depth: int = 4,
    ) -> bool:
        """Breadth-first search to determine if two nodes connect within ``max_depth`` hops."""
        if start == goal:
            return True
        visited: Set[int] = {start}
        frontier: List[Tuple[int, int]] = [(start, 0)]
        while frontier:
            node_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for neighbor in adjacency.get(node_id, set()):
                if neighbor == goal:
                    return True
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
        return False

    def _collect_existing_connections(
        self,
        cluster_ids: Sequence[int],
        edges: Iterable[models.Edge],
        id_to_node: Dict[int, models.Node],
    ) -> List[Dict[str, object]]:
        """Return existing edges that already connect nodes within the cluster."""
        id_set = set(cluster_ids)
        connections: List[Dict[str, object]] = []
        for edge in edges:
            if (
                edge.subject_node_id in id_set
                and edge.object_node_id in id_set
            ):
                connections.append(
                    {
                        "subject": id_to_node[edge.subject_node_id].label,
                        "predicate": edge.predicate,
                        "object": id_to_node[edge.object_node_id].label,
                    }
                )
        return connections

    def _serialize_node_attributes(self, node: models.Node) -> List[Dict[str, object]]:
        """Convert a node's attribute ORM rows into serialisable dictionaries."""
        if not node.attributes:
            return []
        payload: List[Dict[str, object]] = []
        for attr in node.attributes:
            value = self._attribute_value(attr)
            payload.append(
                {
                    "name": attr.name,
                    "type": attr.data_type.value if hasattr(attr.data_type, "value") else str(attr.data_type),
                    "value": value,
                }
            )
        return payload

    @staticmethod
    def _attribute_value(attr: models.NodeAttribute) -> object:
        """Return the scalar value for ``attr`` respecting its declared type."""
        if attr.data_type == models.NodeAttributeType.number:
            return attr.value_number
        if attr.data_type == models.NodeAttributeType.boolean:
            return attr.value_boolean
        return attr.value_text


__all__ = ["GraphInsightService", "ClusterRecommendation"]
