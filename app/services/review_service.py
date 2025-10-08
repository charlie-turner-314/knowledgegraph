from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from app.data import models
from app.data.repositories import CandidateRepository, GraphRepository, SMEActionRepository
from app.llm.client import LLMClient, get_client
from app.llm.schemas import ReviewAgentCommand, ReviewAgentResponse
from app.utils.ontology import (
    ensure_entity_label,
    ensure_relationship_label,
    evaluate_triple,
    summarize_mutations,
)


logger = logging.getLogger(__name__)


@dataclass
class ReviewChatOutcome:
    """Result of a review chat turn."""

    reply: str
    applied_changes: List[str]
    commands: List[ReviewAgentCommand]


@dataclass
class CommandExecutionContext:
    """Mutable state shared across command handlers in a single turn."""

    node_aliases: Dict[str, models.Node] = field(default_factory=dict)
    edge_aliases: Dict[str, models.Edge] = field(default_factory=dict)


class ReviewService:
    """Conversational agent for graph curation and CRUD support."""

    def __init__(self, session: Session, *, llm_client: Optional[LLMClient] = None):
        self.session = session
        self.graph = GraphRepository(session)
        self.actions = SMEActionRepository(session)
        self.candidates = CandidateRepository(session)
        self.llm = llm_client or get_client()

    def chat(
        self,
        *,
        messages: List[Dict[str, str]],
        actor: Optional[str] = None,
    ) -> ReviewChatOutcome:
        """Run a chat turn with the review agent and apply any graph mutations."""

        context = self._graph_context()
        response = self.llm.review_chat(messages=messages, graph_context=context)
        applied = self._apply_commands(response, actor=actor or "graph-reviewer")
        return ReviewChatOutcome(
            reply=response.reply,
            applied_changes=applied,
            commands=response.commands,
        )

    def graph_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight snapshot of the graph for UI display."""
        return self._graph_context()

    # ------------------------------------------------------------------
    # Command execution helpers

    def _apply_commands(self, response: ReviewAgentResponse, *, actor: str) -> List[str]:
        summaries: List[str] = []
        context = CommandExecutionContext()
        for command in response.commands:
            try:
                change = self._dispatch_command(command, actor=actor, context=context)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Error executing review command",
                    extra={
                        "action": command.action,
                        "payload": command.model_dump(exclude_none=True),
                    },
                )
                change = f"Failed to execute action '{command.action}': {exc}"
            if not change:
                continue
            if isinstance(change, list):
                summaries.extend(str(item) for item in change if item)
            else:
                summaries.append(str(change))
        return summaries

    def _dispatch_command(
        self,
        command: ReviewAgentCommand,
        *,
        actor: str,
        context: CommandExecutionContext,
    ) -> Optional[object]:
        action = self._normalize_action(command.action)

        if action == "create_edge":
            return self._handle_create_edge(command, actor, context=context)
        if action == "add_attribute":
            return self._handle_add_attribute(command, actor, context=context)
        if action == "delete_edge":
            return self._handle_delete_edge(command, actor, context=context)
        if action == "update_node":
            return self._handle_update_node(command, actor, context=context)
        if action == "noop":
            return None
        if action == "upsert_node":
            return self._handle_upsert_node(command, actor, context=context)
        if action == "upsert_edge":
            return self._handle_upsert_edge(command, actor, context=context)
        if action == "ensure_entity_type":
            return self._handle_ensure_entity_type(command, actor)
        if action == "ensure_relationship_type":
            return self._handle_ensure_relationship_type(command, actor)
        if action == "accept_pending_candidate":
            return self._handle_update_candidate_status(
                command,
                actor,
                target_status=models.CandidateStatus.approved,
                context=context,
            )
        if action == "reject_pending_candidate":
            return self._handle_update_candidate_status(
                command,
                actor,
                target_status=models.CandidateStatus.rejected,
                context=context,
            )
        if action == "update_candidate_status":
            return self._handle_update_candidate_status(
                command,
                actor,
                target_status=None,
                context=context,
            )
        if action == "request_clarification":
            return self._handle_request_clarification(command, actor)
        if action == "remove_attribute":
            return self._handle_remove_attribute(command, actor)
        if action == "resolve_ontology_backlog_items":
            return self._handle_resolve_backlog_items(command, actor)

        return f"Ignored unsupported action '{command.action}'."

    @staticmethod
    def _normalize_action(action: str) -> str:
        mapping = {
            "ensure_ontology_type": "ensure_entity_type",
            "ensure_ontology_relationship": "ensure_relationship_type",
            "upsert_entity": "upsert_node",
        }
        return mapping.get(action, action)

    def _handle_create_edge(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        context: CommandExecutionContext,
    ) -> Optional[str]:
        return self._execute_edge_command(command, actor, context=context)

    def _handle_upsert_edge(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        context: CommandExecutionContext,
    ) -> Optional[str]:
        return self._execute_edge_command(command, actor, context=context)

    def _execute_edge_command(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        context: CommandExecutionContext,
    ) -> Optional[str]:
        alias = self._command_str(command, "id_alias", "alias")
        subject_label, subject_node = self._resolve_node_label(command, role="subject", context=context)
        object_label, object_node = self._resolve_node_label(command, role="object", context=context)
        predicate = self._command_str(command, "predicate", "relationship", "name")
        if not subject_label or not predicate or not object_label:
            return "Edge command missing subject, predicate, or object."

        if subject_node is None:
            subject_node = self.graph.nodes.get_by_label(subject_label)
        if object_node is None:
            object_node = self.graph.nodes.get_by_label(object_label)

        subject_attr_payload = command.subject_attributes or self._command_list(command, "subject_attributes")
        object_attr_payload = command.object_attributes or self._command_list(command, "object_attributes")
        subject_attributes = self._prepare_attributes(subject_attr_payload)
        object_attributes = self._prepare_attributes(object_attr_payload)
        tags_payload = command.tags or self._command_list(command, "tags")
        tags = self._prepare_tags(tags_payload)

        candidate = self._resolve_candidate(command)
        document_chunk = self._resolve_chunk(command.source_chunk_id)
        if document_chunk is None and candidate is not None:
            document_chunk = candidate.chunk

        subject_type = self._command_str(command, "subject_entity_type", "subject_type")
        if not subject_type and subject_node is not None:
            subject_type = subject_node.entity_type
        object_type = self._command_str(command, "object_entity_type", "object_type")
        if not object_type and object_node is not None:
            object_type = object_node.entity_type

        needs_evidence = bool(self._command_bool(command, "needs_evidence") or ("for-review" in tags))
        rationale = command.notes or command.rationale
        confidence = command.confidence

        payload = {
            "command": command.model_dump(exclude_none=True),
            "alias": alias,
            "needs_evidence": needs_evidence,
            "tags": tags,
        }
        if candidate:
            payload["candidate_id"] = candidate.id
        if command.provenance:
            payload["provenance"] = command.provenance

        sme_action = self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=candidate,
            payload=payload,
        )

        edge = self.graph.create_edge_with_provenance(
            subject_label=subject_label,
            predicate=predicate,
            object_label=object_label,
            entity_type_subject=subject_type,
            entity_type_object=object_type,
            canonical_subject=subject_node.canonical_term if subject_node else None,
            canonical_object=object_node.canonical_term if object_node else None,
            candidate=candidate,
            document_chunk=document_chunk,
            sme_action=sme_action,
            subject_attributes=subject_attributes,
            object_attributes=object_attributes,
            tags=tags,
            created_by=actor,
            statement_rationale=rationale,
            statement_confidence=confidence,
            needs_evidence=needs_evidence,
        )

        if candidate and edge.candidate_id != candidate.id:
            edge.candidate_id = candidate.id
            self.session.add(edge)

        if alias:
            context.edge_aliases[alias] = edge

        return f"Created edge #{edge.id}: {subject_label} — {predicate} — {object_label}."

    def _handle_add_attribute(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        context: CommandExecutionContext,
    ) -> Optional[str]:
        node_label = self._command_str(command, "subject", "node_label", "label")
        attributes_payload = command.subject_attributes or self._command_list(command, "subject_attributes", "attributes")
        attributes = self._prepare_attributes(attributes_payload)
        if not node_label or not attributes:
            return None

        node = self.graph.nodes.ensure_node(
            label=node_label,
            entity_type=self._command_str(command, "entity_type"),
            canonical_term=None,
            sme_override=True,
        )
        self.graph.node_attributes.upsert_many(node, attributes)

        alias = self._command_str(command, "id_alias", "alias")
        if alias:
            context.node_aliases[alias] = node

        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload={
                "node_label": node_label,
                "attributes": self._serialize_attributes(attributes),
                "command": command.model_dump(exclude_none=True),
            },
        )
        return f"Updated attributes for '{node_label}'."

    def _handle_delete_edge(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        context: CommandExecutionContext,
    ) -> Optional[str]:
        edge = self._resolve_edge_reference(command, context=context)
        if edge is None:
            edge_id = self._command_int(command, "edge_id", "id")
            if edge_id is None:
                return "Delete edge command missing a valid edge reference."
            return f"Edge #{edge_id} not found."

        edge_id = edge.id
        self.session.delete(edge)
        self.actions.record_action(
            action_type=models.SMEActionType.edit_triple,
            actor=actor,
            candidate=None,
            payload={
                "edge_id": edge_id,
                "command": command.model_dump(exclude_none=True),
            },
        )
        return f"Deleted edge #{edge_id}."

    def _handle_update_node(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        context: CommandExecutionContext,
    ) -> Optional[str]:
        # Support two formats:
        # 1. Flat: {"action":"update_node", "subject":"Node A", "entity_type":"Equipment"}
        # 2. Cypher-like: {"action":"update_node", "match":{"label":"Old"}, "set":{"label":"New", "attributes":{...}}}
        extra = self._command_extra(command)
        match_spec = extra.get("match") if isinstance(extra.get("match"), dict) else None
        set_spec = extra.get("set") if isinstance(extra.get("set"), dict) else None

        original_label = None
        if match_spec and isinstance(match_spec.get("label"), str):
            original_label = self._coerce_str(match_spec.get("label"))
        if not original_label:
            original_label = self._command_str(command, "subject", "node_label", "label")
        if not original_label:
            return "Update node command missing node label."

        node = self.graph.nodes.get_by_label(original_label)
        if node is None:
            return f"Node '{original_label}' not found."

        # Determine new label if rename requested
        new_label = None
        if set_spec and isinstance(set_spec.get("label"), str):
            new_label = self._coerce_str(set_spec.get("label"))
        # Attributes can come from set.attributes (dict), or standard attributes arrays
        attributes_payload = command.subject_attributes or self._command_list(command, "subject_attributes", "attributes")
        if not attributes_payload and set_spec and isinstance(set_spec.get("attributes"), dict):
            # Convert dict to list form
            attributes_payload = [
                {"name": key, "value": value} for key, value in set_spec.get("attributes", {}).items()
            ]
        attributes = self._prepare_attributes(attributes_payload)
        new_entity_type = self._command_str(command, "entity_type") or (
            self._coerce_str(set_spec.get("entity_type")) if set_spec else None
        )
        alias = self._command_str(command, "id_alias", "alias")

        audit_payload: Dict[str, Any] = {
            "node_label": original_label,
            "command": command.model_dump(exclude_none=True),
        }
        changed = False
        if new_label and new_label != node.label:
            audit_payload["old_label"] = node.label
            node.label = new_label
            changed = True
        if attributes:
            self.graph.node_attributes.upsert_many(node, attributes)
            audit_payload["attributes"] = self._serialize_attributes(attributes)
            changed = True
        if new_entity_type and new_entity_type != (node.entity_type or ""):
            node.entity_type = new_entity_type
            node.sme_override = True
            self.session.add(node)
            audit_payload["entity_type"] = new_entity_type
            changed = True
        if command.notes:
            audit_payload["notes"] = command.notes

        if alias:
            context.node_aliases[alias] = node

        if changed:
            self.actions.record_action(
                action_type=models.SMEActionType.system,
                actor=actor,
                candidate=None,
                payload=audit_payload,
            )
        return f"Updated node '{new_label or original_label}'."

    def _handle_upsert_node(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        context: CommandExecutionContext,
    ) -> Optional[str]:
        label = self._command_str(command, "node_label", "label", "subject")
        if not label:
            return "Upsert node command missing label."

        entity_type = self._command_str(command, "entity_type")
        if not entity_type:
            types = command.types or self._command_list(command, "types")
            if types:
                entity_type = str(types[0]).strip()

        node = self.graph.nodes.ensure_node(
            label=label,
            entity_type=entity_type,
            canonical_term=None,
            sme_override=entity_type is None,
        )
        if entity_type and node.entity_type != entity_type:
            node.entity_type = entity_type
            node.sme_override = True
            self.session.add(node)

        attributes_payload = command.subject_attributes or self._command_list(command, "subject_attributes", "attributes")
        attributes = self._prepare_attributes(attributes_payload)
        if attributes:
            self.graph.node_attributes.upsert_many(node, attributes)

        self.graph._embedding_store.bulk_add([node.label])

        alias = self._command_str(command, "id_alias", "alias")
        if alias:
            context.node_aliases[alias] = node

        payload: Dict[str, Any] = {
            "node_id": node.id,
            "label": label,
            "entity_type": entity_type,
            "command": command.model_dump(exclude_none=True),
        }
        if attributes:
            payload["attributes"] = self._serialize_attributes(attributes)
        if command.sources:
            payload["sources"] = command.sources

        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload=payload,
        )

        details: List[str] = []
        if entity_type:
            details.append(entity_type)
        if attributes:
            details.append(f"{len(attributes)} attribute(s)")
        detail_str = ", ".join(details) if details else "no metadata"
        return f"Ensured node '{label}' ({detail_str})."

    def _handle_ensure_entity_type(self, command: ReviewAgentCommand, actor: str) -> Optional[str]:
        label = self._command_str(command, "name", "label", "entity_type")
        if not label:
            return "Ontology entity type command missing name."

        added = ensure_entity_label(label)
        payload = {
            "entity_type": label,
            "description": self._command_str(command, "description", "definition"),
            "command": command.model_dump(exclude_none=True),
            "added": added,
        }
        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload=payload,
        )
        if added:
            return f"Added '{label}' to ontology entity types."
        return f"Entity type '{label}' already present."

    def _handle_ensure_relationship_type(self, command: ReviewAgentCommand, actor: str) -> Optional[str]:
        label = self._command_str(command, "predicate", "name", "relationship")
        if not label:
            return "Ontology relationship command missing name."

        added = ensure_relationship_label(label)
        payload = {
            "relationship": label,
            "domain": self._command_str(command, "domain"),
            "range": self._command_str(command, "range"),
            "description": self._command_str(command, "description", "definition"),
            "command": command.model_dump(exclude_none=True),
            "added": added,
        }
        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload=payload,
        )
        if added:
            return f"Added '{label}' to ontology relationships."
        return f"Relationship '{label}' already present."

    def _handle_update_candidate_status(
        self,
        command: ReviewAgentCommand,
        actor: str,
        *,
        target_status: Optional[models.CandidateStatus],
        context: CommandExecutionContext,
    ) -> Optional[str]:
        candidate = self._resolve_candidate(command)
        if candidate is None:
            return "Candidate update skipped; candidate not found."

        status = target_status
        if status is None:
            raw_status = self._command_str(command, "status")
            if not raw_status:
                return "Candidate update skipped; status missing."
            normalized = raw_status.strip().lower()
            alias_map = {
                "accept": models.CandidateStatus.approved,
                "accepted": models.CandidateStatus.approved,
                "approve": models.CandidateStatus.approved,
                "approved": models.CandidateStatus.approved,
                "reject": models.CandidateStatus.rejected,
                "rejected": models.CandidateStatus.rejected,
                "deny": models.CandidateStatus.rejected,
                "needs_revision": models.CandidateStatus.needs_revision,
                "needs_review": models.CandidateStatus.needs_revision,
                "pending": models.CandidateStatus.pending,
            }
            status = alias_map.get(normalized)
            if status is None:
                try:
                    status = models.CandidateStatus(normalized)
                except ValueError:
                    return f"Candidate update skipped; unknown status '{raw_status}'."

        candidate.status = status
        candidate.updated_at = datetime.utcnow()
        tags = set(candidate.suggested_tags or [])
        if status == models.CandidateStatus.approved:
            tags.discard("needs-ontology-review")
            tags.discard("for-review")
        elif status == models.CandidateStatus.rejected:
            tags.discard("for-review")
        candidate.suggested_tags = sorted(tags)
        self.session.add(candidate)

        edge = self._resolve_edge_reference(command, context=context)
        if edge and status == models.CandidateStatus.approved:
            edge.candidate_id = candidate.id
            self.session.add(edge)
        elif status == models.CandidateStatus.approved and edge is None:
            # No existing edge reference supplied; materialize from candidate
            sme_action = self.actions.record_action(
                action_type=models.SMEActionType.accept_triple,
                actor=actor,
                candidate=candidate,
                payload={
                    "candidate_id": candidate.id,
                    "auto_materialize": True,
                },
            )
            edge = self.graph.ensure_edge_for_candidate(
                candidate,
                sme_action=sme_action,
                actor=actor,
            )
            context.edge_aliases.setdefault(f"candidate:{candidate.id}", edge)

        action_type = models.SMEActionType.system
        if status == models.CandidateStatus.approved:
            action_type = models.SMEActionType.accept_triple
        elif status == models.CandidateStatus.rejected:
            action_type = models.SMEActionType.reject_triple

        payload = {
            "candidate_id": candidate.id,
            "status": status.value,
            "notes": command.notes,
            "command": command.model_dump(exclude_none=True),
        }
        if edge:
            payload["edge_id"] = edge.id

        self.actions.record_action(
            action_type=action_type,
            actor=actor,
            candidate=candidate,
            payload=payload,
        )

        return f"Candidate #{candidate.id} marked as {status.value}."

    def _handle_request_clarification(self, command: ReviewAgentCommand, actor: str) -> Optional[str]:
        candidate = self._resolve_candidate(command)
        if candidate is None:
            return "Clarification request skipped; candidate not found."

        clarification = self._command_str(command, "notes", "summary", "rationale")
        candidate.status = models.CandidateStatus.needs_revision
        candidate.updated_at = datetime.utcnow()
        self.session.add(candidate)

        payload = {
            "candidate_id": candidate.id,
            "clarification": clarification,
            "command": command.model_dump(exclude_none=True),
        }
        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=candidate,
            payload=payload,
        )
        return f"Requested clarification for candidate #{candidate.id}."

    def _handle_remove_attribute(self, command: ReviewAgentCommand, actor: str) -> Optional[str]:
        node_label = self._command_str(command, "node_label", "subject", "label")
        if not node_label:
            return "Remove attribute skipped; node label missing."

        node = self.graph.nodes.get_by_label(node_label)
        if node is None:
            return f"Node '{node_label}' not found."

        target_names = {name.lower() for name in self._collect_attribute_names(command)}
        if not target_names:
            return "Remove attribute skipped; no attribute names provided."

        removed = False
        for attr in list(node.attributes or []):
            if attr.name and attr.name.lower() in target_names:
                self.session.delete(attr)
                removed = True

        if not removed:
            return f"No matching attributes removed from '{node_label}'."

        payload = {
            "node_label": node_label,
            "removed_attributes": sorted(target_names),
            "command": command.model_dump(exclude_none=True),
        }
        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload=payload,
        )
        return f"Removed attributes {sorted(target_names)} from '{node_label}'."

    def _handle_resolve_backlog_items(self, command: ReviewAgentCommand, actor: str) -> Optional[str]:
        items = command.items or self._command_list(command, "items")
        cleaned = [str(item).strip() for item in items if isinstance(item, str) and str(item).strip()]
        if not cleaned:
            return "Ontology backlog resolution skipped; no items provided."

        payload = {
            "items": cleaned,
            "command": command.model_dump(exclude_none=True),
            "notes": command.notes,
        }
        self.actions.record_action(
            action_type=models.SMEActionType.system,
            actor=actor,
            candidate=None,
            payload=payload,
        )
        return f"Marked {len(cleaned)} ontology backlog item(s) as resolved."

    # ------------------------------------------------------------------
    # Context helpers

    def _graph_context(self) -> Dict[str, Any]:
        edges = self._recent_edges(limit=12)
        nodes = self._recent_nodes(limit=12)
        canonical_terms = self._canonical_terms(limit=12)
        pending_candidates = self._pending_candidates(limit=8)
        return {
            "recent_edges": edges,
            "recent_nodes": nodes,
            "canonical_terms": canonical_terms,
            "pending_candidates": pending_candidates,
            "ontology_backlog": self._ontology_backlog(pending_candidates),
        }

    def _recent_edges(self, limit: int) -> List[Dict[str, Any]]:
        stmt = select(models.Edge).order_by(models.Edge.id.desc()).limit(limit)  # type: ignore[attr-defined]
        serialized: List[Dict[str, Any]] = []
        for edge in self.session.exec(stmt):
            source = edge.sources[0] if edge.sources else None
            document = None
            page_label = None
            if source and source.document_chunk:
                page_label = source.document_chunk.page_label
                if source.document_chunk.document:
                    document = source.document_chunk.document.display_name
            serialized.append(
                {
                    "id": edge.id,
                    "subject": edge.subject.label if edge.subject else None,
                    "predicate": edge.predicate,
                    "object": edge.object.label if edge.object else None,
                    "tags": sorted({tag.label for tag in (edge.tags or [])}),
                    "document": document,
                    "page_label": page_label,
                }
            )
        return serialized

    def _recent_nodes(self, limit: int) -> List[Dict[str, Any]]:
        stmt = select(models.Node).order_by(models.Node.id.desc()).limit(limit)  # type: ignore[attr-defined]
        nodes: List[Dict[str, Any]] = []
        for node in self.session.exec(stmt):
            nodes.append(
                {
                    "label": node.label,
                    "entity_type": node.entity_type,
                    "attributes": {
                        attr.name: self._attr_value(attr)
                        for attr in (node.attributes or [])
                        if attr.name
                    },
                }
            )
        return nodes

    def _canonical_terms(self, limit: int) -> List[Dict[str, Any]]:
        stmt = select(models.CanonicalTerm).order_by(  # type: ignore[attr-defined]
            models.CanonicalTerm.id.desc()  # type: ignore[attr-defined]
        ).limit(limit)
        return [
            {
                "label": term.label,
                "aliases": term.aliases or [],
                "entity_type": term.entity_type,
            }
            for term in self.session.exec(stmt)
        ]

    def _pending_candidates(self, limit: int) -> List[Dict[str, Any]]:
        stmt = (
            select(models.CandidateTriple)
            .where(models.CandidateTriple.status == models.CandidateStatus.pending)
            .order_by(models.CandidateTriple.created_at)  # type: ignore[arg-type]
            .limit(limit)
        )
        pending: List[Dict[str, Any]] = []
        for candidate in self.session.exec(stmt):
            chunk = candidate.chunk
            document = chunk.document if chunk and chunk.document else None
            evaluation = evaluate_triple(
                subject_label=candidate.subject_text,
                predicate=candidate.predicate_text,
                object_label=candidate.object_text,
                subject_attributes=candidate.subject_attributes,
                object_attributes=candidate.object_attributes,
            )
            mutation_summaries = summarize_mutations(evaluation.suggested_mutations)
            pending.append(
                {
                    "id": candidate.id,
                    "subject": candidate.subject_text,
                    "predicate": candidate.predicate_text,
                    "object": candidate.object_text,
                    "confidence": candidate.llm_confidence,
                    "document": document.display_name if document else None,
                    "document_id": document.id if document else None,
                    "page_label": chunk.page_label if chunk else None,
                    "chunk_text": chunk.text if chunk else None,
                    "suggested_mutations": evaluation.suggested_mutations,
                    "mutation_summaries": mutation_summaries,
                    "tags": candidate.suggested_tags or [],
                    "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
                }
            )
        return pending

    @staticmethod
    def _ontology_backlog(pending_candidates: List[Dict[str, Any]]) -> List[str]:
        seen: set[str] = set()
        backlog: List[str] = []
        for candidate in pending_candidates:
            for summary in candidate.get("mutation_summaries") or []:
                key = summary.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                backlog.append(summary)
                if len(backlog) >= 12:
                    return backlog
        return backlog


    # ------------------------------------------------------------------
    # Utility helpers

    def _prepare_attributes(self, attributes: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for attr in attributes or []:
            if not isinstance(attr, dict):
                continue
            name = str(attr.get("name") or attr.get("label") or "").strip()
            if not name:
                continue
            data_type_raw = attr.get("data_type") or attr.get("type") or "string"
            if isinstance(data_type_raw, models.NodeAttributeType):
                data_type = data_type_raw
            else:
                try:
                    data_type = models.NodeAttributeType(str(data_type_raw))
                except ValueError:
                    data_type = models.NodeAttributeType.string
            normalized.append(
                {
                    "name": name,
                    "data_type": data_type,
                    "value_text": attr.get("value_text") or attr.get("value"),
                    "value_number": attr.get("value_number"),
                    "value_boolean": attr.get("value_boolean"),
                }
            )
        return normalized

    @staticmethod
    def _prepare_tags(tags: List[str]) -> List[str]:
        cleaned: List[str] = []
        for tag in tags or []:
            if isinstance(tag, str) and tag.strip():
                cleaned.append(tag.strip().lower())
        return cleaned

    def _serialize_attributes(self, attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        serialized: List[Dict[str, Any]] = []
        for attr in attributes:
            data_type = attr.get("data_type")
            if isinstance(data_type, models.NodeAttributeType):
                data_type_value = data_type.value
            else:
                data_type_value = data_type
            serialized.append(
                {
                    "name": attr.get("name"),
                    "data_type": data_type_value,
                    "value_text": attr.get("value_text"),
                    "value_number": attr.get("value_number"),
                    "value_boolean": attr.get("value_boolean"),
                }
            )
        return serialized

    def _resolve_chunk(self, chunk_id: Optional[int]) -> Optional[models.DocumentChunk]:
        if chunk_id is None:
            return None
        return self.session.get(models.DocumentChunk, chunk_id)

    @staticmethod
    def _attr_value(attr: models.NodeAttribute) -> Any:
        if attr.data_type == models.NodeAttributeType.number:
            return attr.value_number
        if attr.data_type == models.NodeAttributeType.boolean:
            return attr.value_boolean
        return attr.value_text

    # ------------------------------------------------------------------
    # Command parsing helpers

    @staticmethod
    def _command_extra(command: ReviewAgentCommand) -> Dict[str, Any]:
        extra = getattr(command, "model_extra", None)
        if isinstance(extra, dict):
            return extra
        return {}

    def _command_str(self, command: ReviewAgentCommand, *keys: str) -> Optional[str]:
        extra = self._command_extra(command)
        for key in keys:
            value = getattr(command, key, None)
            cleaned = self._coerce_str(value)
            if cleaned:
                return cleaned
            if key in extra:
                cleaned = self._coerce_str(extra[key])
                if cleaned:
                    return cleaned
        return None

    def _command_int(self, command: ReviewAgentCommand, *keys: str) -> Optional[int]:
        extra = self._command_extra(command)
        for key in keys:
            value = getattr(command, key, None)
            converted = self._to_int(value)
            if converted is not None:
                return converted
            if key in extra:
                converted = self._to_int(extra[key])
                if converted is not None:
                    return converted
        return None

    def _command_bool(self, command: ReviewAgentCommand, *keys: str) -> Optional[bool]:
        extra = self._command_extra(command)
        for key in keys:
            value = getattr(command, key, None)
            converted = self._to_bool(value)
            if converted is not None:
                return converted
            if key in extra:
                converted = self._to_bool(extra[key])
                if converted is not None:
                    return converted
        return None

    def _command_list(self, command: ReviewAgentCommand, *keys: str) -> List[Any]:
        extra = self._command_extra(command)
        for key in keys:
            value = getattr(command, key, None)
            converted = self._coerce_list(value)
            if converted is not None:
                return converted
            if key in extra:
                converted = self._coerce_list(extra[key])
                if converted is not None:
                    return converted
        return []

    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        if value is None or isinstance(value, (dict, list, tuple, set)):
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @staticmethod
    def _coerce_list(value: Any) -> Optional[List[Any]]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return None
            try:
                return int(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "y"}:
                return True
            if lowered in {"false", "no", "0", "n"}:
                return False
        return None

    def _resolve_node_label(
        self,
        command: ReviewAgentCommand,
        *,
        role: str,
        context: CommandExecutionContext,
    ) -> Tuple[Optional[str], Optional[models.Node]]:
        preferred_keys = (role, f"{role}_label", "node_label", "label")
        label = self._command_str(command, *preferred_keys)
        if label:
            return label, self.graph.nodes.get_by_label(label)

        extra = self._command_extra(command)
        ref = getattr(command, f"{role}_ref", None)
        if ref is None and f"{role}_ref" in extra:
            ref = extra[f"{role}_ref"]
        return self._resolve_node_ref(ref, context)

    def _resolve_node_ref(
        self,
        ref: Any,
        context: CommandExecutionContext,
    ) -> Tuple[Optional[str], Optional[models.Node]]:
        if ref is None:
            return None, None
        if isinstance(ref, dict):
            by = self._coerce_str(ref.get("by"))
            value = ref.get("value")
            if by == "alias" and isinstance(value, str):
                alias = value.strip()
                node = context.node_aliases.get(alias)
                if node:
                    return node.label, node
                existing = self.graph.nodes.get_by_label(alias)
                if existing:
                    return existing.label, existing
                return alias or None, None
            if by in {"label", "name"} and isinstance(value, str):
                label = value.strip()
                return label or None, self.graph.nodes.get_by_label(label) if label else None
            if by in {"id", "node_id"}:
                node_id = self._to_int(value)
                if node_id is not None:
                    node = self.session.get(models.Node, node_id)
                    if node:
                        return node.label, node
                return None, None
            if isinstance(value, str):
                label = value.strip()
                return label or None, self.graph.nodes.get_by_label(label) if label else None
        elif isinstance(ref, str):
            alias = ref.strip()
            if not alias:
                return None, None
            node = context.node_aliases.get(alias) or self.graph.nodes.get_by_label(alias)
            return alias, node
        elif isinstance(ref, int):
            node = self.session.get(models.Node, ref)
            if node:
                return node.label, node
        return None, None

    def _resolve_edge_reference(
        self,
        command: ReviewAgentCommand,
        *,
        context: CommandExecutionContext,
    ) -> Optional[models.Edge]:
        edge_id = self._command_int(command, "edge_id", "id")
        if edge_id is not None:
            edge = self.session.get(models.Edge, edge_id)
            if edge:
                return edge

        alias = self._command_str(command, "edge_alias", "id_alias", "alias")
        if alias and alias in context.edge_aliases:
            return context.edge_aliases[alias]

        extra = self._command_extra(command)
        for key in ("edge_ref", "applied_edge_ref"):
            ref = getattr(command, key, None) or extra.get(key)
            edge = self._resolve_edge_ref(ref, context)
            if edge:
                return edge
        return None

    def _resolve_edge_ref(
        self,
        ref: Any,
        context: CommandExecutionContext,
    ) -> Optional[models.Edge]:
        if ref is None:
            return None
        if isinstance(ref, dict):
            by = self._coerce_str(ref.get("by")) or ""
            value = ref.get("value")
            if by == "alias" and isinstance(value, str):
                alias = value.strip()
                edge = context.edge_aliases.get(alias)
                if edge:
                    return edge
                if alias.isdigit():
                    return self.session.get(models.Edge, int(alias))
            if by in {"id", "edge_id"}:
                edge_id = self._to_int(value)
                if edge_id is not None:
                    return self.session.get(models.Edge, edge_id)
            if isinstance(value, str) and value.strip().isdigit():
                return self.session.get(models.Edge, int(value.strip()))
        elif isinstance(ref, str):
            cleaned = ref.strip()
            if not cleaned:
                return None
            edge = context.edge_aliases.get(cleaned)
            if edge:
                return edge
            if cleaned.isdigit():
                return self.session.get(models.Edge, int(cleaned))
        elif isinstance(ref, int):
            return self.session.get(models.Edge, ref)
        return None

    def _resolve_candidate(self, command: ReviewAgentCommand) -> Optional[models.CandidateTriple]:
        candidate_id = self._command_int(command, "candidate_id", "id")
        if candidate_id is None:
            provenance = command.provenance or self._command_extra(command).get("provenance")
            if isinstance(provenance, dict):
                candidate_id = self._to_int(provenance.get("candidate_id"))
        if candidate_id is None:
            return None
        return self.candidates.get(candidate_id)

    def _collect_attribute_names(self, command: ReviewAgentCommand) -> List[str]:
        names: List[str] = []
        for source in (command.subject_attributes or []):
            if isinstance(source, dict):
                name = source.get("name") or source.get("label")
                cleaned = self._coerce_str(name)
                if cleaned:
                    names.append(cleaned)
        for source in self._command_list(command, "attributes", "subject_attributes"):
            if isinstance(source, dict):
                name = source.get("name") or source.get("label")
                cleaned = self._coerce_str(name)
                if cleaned:
                    names.append(cleaned)
            elif isinstance(source, str):
                cleaned = source.strip()
                if cleaned:
                    names.append(cleaned)
        single = self._command_str(command, "attribute", "attribute_name")
        if single:
            names.append(single)
        return names
