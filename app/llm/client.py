from __future__ import annotations

import functools
import logging
import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import orjson
import requests
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

from app.core.config import settings

from .schemas import (
    ConnectionRecommendationResponse,
    ExtractionContextUpdate,
    ExtractionResponse,
    OntologyParentSuggestion,
    OntologySuggestionResponse,
    QueryAnswerResponse,
    QueryPlanResponse,
    QueryRetrievalResponse,
    ReviewAgentResponse,
    NodeMergeDecision,
    RetrievalNode,
)

logger = logging.getLogger(__name__)

def _log_extraction_failure(request_payload: Dict[str, Any], response_data: Dict[str, Any], error: Exception) -> None:
    """Log detailed information about LLM extraction failures to a file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_data = {
        "timestamp": timestamp,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "request_payload": request_payload,
        "response_data": response_data,
        "chunk_preview": request_payload.get("messages", [])[-1].get("content", "")[:500] if request_payload.get("messages") else "",
    }
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Save to timestamped file
    failure_file = logs_dir / f"llm_extraction_failure_{timestamp}.json"
    try:
        with open(failure_file, "wb") as f:
            f.write(orjson.dumps(failure_data, option=orjson.OPT_INDENT_2))
        logger.error(f"LLM extraction failure logged to {failure_file}")
    except Exception as log_error:
        logger.error(f"Failed to log extraction failure: {log_error}")
        # Fallback to current directory
        fallback_file = f"llm_extraction_failure_{timestamp}.json"
        try:
            with open(fallback_file, "wb") as f:
                f.write(orjson.dumps(failure_data, option=orjson.OPT_INDENT_2))
            logger.error(f"LLM extraction failure logged to {fallback_file}")
        except Exception as fallback_error:
            logger.error(f"Failed to log extraction failure even to fallback location: {fallback_error}")

PROMPT_PATH = Path(__file__).resolve().parents[2] / "resources" / "prompts" / "extraction_system_prompt.txt"
EXTRACTION_SUMMARY_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "extraction_summary_prompt.txt"
)
ONTOLOGY_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "ontology_parent_prompt.txt"
)
QUERY_RETRIEVAL_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "query_retrieval_prompt.txt"
)
QUERY_PLAN_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "query_plan_prompt.txt"
)
QUERY_ANSWER_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "query_answer_prompt.txt"
)
CONNECTION_RECOMMENDATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "connection_recommendation_prompt.txt"
)
CANDIDATE_VALIDATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "candidate_validation_prompt.txt"
)
NODE_MERGE_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "node_merge_prompt.txt"
)
REVIEW_SUMMARY_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "review_summary_prompt.txt"
)
REVIEW_AGENT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "review_agent_prompt.txt"
)


@functools.lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("System prompt file %s missing; using fallback text", PROMPT_PATH)
        return "You extract subject-predicate-object triples from text and respond with JSON."


@functools.lru_cache(maxsize=1)
def _load_ontology_prompt() -> str:
    try:
        return ONTOLOGY_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("Ontology prompt file %s missing; using fallback text", ONTOLOGY_PROMPT_PATH)
        return (
            "You propose ontology parents for related nodes. "
            "Return JSON with a 'suggestion' object describing the parent."
        )


@functools.lru_cache(maxsize=1)
def _load_query_retrieval_prompt() -> str:
    try:
        return QUERY_RETRIEVAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Query retrieval prompt file %s missing; using fallback text",
            QUERY_RETRIEVAL_PROMPT_PATH,
        )
        return "Identify the most relevant graph nodes/predicates for the user question."


@functools.lru_cache(maxsize=1)
def _load_query_plan_prompt() -> str:
    try:
        return QUERY_PLAN_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Query plan prompt file %s missing; using fallback text",
            QUERY_PLAN_PROMPT_PATH,
        )
        return "Create a traversal plan over the knowledge graph in JSON."


@functools.lru_cache(maxsize=1)
def _load_query_answer_prompt() -> str:
    try:
        return QUERY_ANSWER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Query answer prompt file %s missing; using fallback text",
            QUERY_ANSWER_PROMPT_PATH,
        )
        return "Summarize the answer from the supplied triples."


@functools.lru_cache(maxsize=1)
def _load_connection_recommendation_prompt() -> str:
    try:
        return CONNECTION_RECOMMENDATION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Connection recommendation prompt file %s missing; using fallback text",
            CONNECTION_RECOMMENDATION_PROMPT_PATH,
        )
        return "Assess whether new nodes or edges are needed. Return empty lists when no changes are required."


@functools.lru_cache(maxsize=1)
def _load_candidate_validation_prompt() -> str:
    try:
        return CANDIDATE_VALIDATION_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Candidate validation prompt file %s missing; using fallback text",
            CANDIDATE_VALIDATION_PROMPT_PATH,
        )
        return "Ask a clarifying question about the candidate triple."


@functools.lru_cache(maxsize=1)
def _load_node_merge_prompt() -> str:
    try:
        return NODE_MERGE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Node merge prompt file %s missing; using fallback text",
            NODE_MERGE_PROMPT_PATH,
        )
        return (
            "Decide whether a candidate node should reuse an existing node label. "
            "Respond with JSON containing use_existing, preferred_label, and reason."
        )


@functools.lru_cache(maxsize=1)
def _load_review_summary_prompt() -> str:
    try:
        return REVIEW_SUMMARY_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Review summary prompt file %s missing; using fallback text",
            REVIEW_SUMMARY_PROMPT_PATH,
        )
        return "Summarize the candidate information and ask for clarifications if needed."


@functools.lru_cache(maxsize=1)
def _load_review_agent_prompt() -> str:
    try:
        return REVIEW_AGENT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Review agent prompt file %s missing; using fallback text",
            REVIEW_AGENT_PROMPT_PATH,
        )
        return (
            "You are a knowledge graph co-pilot. Respond with JSON containing a 'reply' string "
            "and a 'commands' list describing graph mutations to perform."
        )


@functools.lru_cache(maxsize=1)
def _load_extraction_summary_prompt() -> str:
    try:
        return EXTRACTION_SUMMARY_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning(
            "Extraction summary prompt file %s missing; using fallback text",
            EXTRACTION_SUMMARY_PROMPT_PATH,
        )
        return (
            "Maintain a concise running summary and entity list for the document. "
            "Respond with JSON containing 'summary' and 'entities'."
        )

class LLMClient:
    """Lightweight client for a chat-completions style LLM endpoint."""

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint = endpoint or settings.llm_endpoint
        self.api_key = api_key or settings.llm_api_key
        self._dry_run = self.endpoint is None

    @retry(wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(3))
    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.endpoint is None:
            logger.error("LLM endpoint missing; skipping API call")
            return {}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            if settings.model_source == "azure":
                headers["api-key"] = f"{self.api_key}"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Log request details for debugging
        logger.info("Calling LLM endpoint %s", self.endpoint)
        logger.debug("Request payload keys: %s", list(payload.keys()))
        if "messages" in payload:
            logger.debug("Message count: %d", len(payload["messages"]))
            # Log a preview of the user content for debugging (truncated)
            user_messages = [msg for msg in payload["messages"] if msg.get("role") == "user"]
            if user_messages:
                content_preview = user_messages[-1].get("content", "")[:200]
                logger.debug("User content preview: %s...", content_preview)
        
        try:
            response = requests.post(self.endpoint, headers=headers, data=orjson.dumps(payload))
            logger.info("Received response with status %s", response.status_code)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            logger.error("HTTP request failed: %s", exc)
            if hasattr(exc, 'response') and exc.response is not None:
                logger.error("Response status: %s", exc.response.status_code)
                logger.error("Response content: %s", exc.response.text[:500])
                with open("logs/llm_api_error_response.json", "wb") as f:
                    f.write(orjson.dumps({
                        "status_code": exc.response.status_code,
                        "content": exc.response.text,
                    }, option=orjson.OPT_INDENT_2))
            raise

    def extract_triples(
        self,
        *,
        chunk_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        canonical_context: Optional[List[Dict[str, Any]]] = None,
        predefined_ontology: Optional[Dict[str, Any]] = None,
        running_summary: Optional[str] = None,
        entity_memory: Optional[List[Dict[str, Any]]] = None,
        ontology_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResponse:
        """Call the configured LLM to extract triples from text."""

        if predefined_ontology is None:
            # Load the ontology from the JSON file if not provided
            ontology_path = Path(__file__).resolve().parents[2] / "resources" / "ontology" / "tad_ontology.json"
            predefined_ontology = orjson.loads(ontology_path.read_text(encoding="utf-8"))

        # Ensure predefined_ontology is a dictionary
        predefined_ontology = predefined_ontology or {}

        system_prompt = _load_system_prompt()
        ontology_payload = {
            "entities": predefined_ontology.get("entities", []),
            "relationships": predefined_ontology.get("relationships", []),
        }

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "system",
                "content": (
                    "The following ontology defines valid entities and relationships. "
                    "If an entity or relationship does not fit, propose an extension flagged as 'for-review'.\n"
                    f"Ontology: {orjson.dumps(ontology_payload).decode()}"
                ),
            },
        ]

        if canonical_context:
            canonical_payload = {"canonical_terms": canonical_context}
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Canonical vocabulary for this project (JSON). Use labels when aliases match.\n"
                        f"{orjson.dumps(canonical_payload).decode()}"
                    ),
                }
            )

        if ontology_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Ontology entries most relevant to this chunk (JSON). Prioritise these labels "
                        "for entity typing and predicate selection when evidence supports them.\n"
                        f"{orjson.dumps(ontology_context).decode()}"
                    ),
                }
            )

        if running_summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Condensed document summary carried over from earlier sections:\n"
                        f"{running_summary}"
                    ),
                }
            )

        if entity_memory:
            entity_payload = {"entity_memory": entity_memory[:40]}
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Previously identified entities (JSON). Use these labels or aliases when context matches.\n"
                        f"{orjson.dumps(entity_payload).decode()}"
                    ),
                }
            )

        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        "Extract knowledge triples from the following text. "
                        "Limit predicates to three words. Return JSON with triples array."
                    ),
                },
                {
                    "role": "user",
                    "content": chunk_text,
                },
            ]
        )

        payload = {
            "messages": messages,
            "temperature": settings.llm_temperature_extraction,
            "response_format": {"type": "json_object"},
        }
        if metadata and not settings.model_source == "azure":
            # Convert all metadata values to strings
            metadata = {k: str(v) for k, v in metadata.items()}
            payload["metadata"] = metadata

        try:
            raw = self._call_api(payload)
        except RetryError as exc:
            logger.exception("LLM triple extraction failed after retries")
            _log_extraction_failure(payload, {}, exc)
            raise RuntimeError("LLM triple extraction failed") from exc

        try:
            message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
            if isinstance(message_content, str):
                parsed_content = orjson.loads(message_content)
            elif isinstance(message_content, dict):
                parsed_content = message_content
            else:
                parsed_content = {}

            # Validate against predefined ontology
            triples = parsed_content.get("triples", [])
            for triple in triples:
                subject, predicate, obj = triple["subject"], triple["predicate"], triple["object"]
                if predefined_ontology and not self._fits_ontology(subject, predicate, obj, predefined_ontology):
                    triple["review_status"] = "for-review"
                else:
                    triple["review_status"] = "approved"

            parsed = ExtractionResponse.model_validate(
                {
                    "triples": triples,
                    "raw_response": raw,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse LLM extraction response")
            _log_extraction_failure(payload, raw or {}, exc)
            # Keep the old file for backward compatibility
            with open(".llm_extraction_error.json", "wb") as f:
                f.write(orjson.dumps(raw or {}, option=orjson.OPT_INDENT_2))
            raise RuntimeError("Invalid response from LLM extraction endpoint. Detailed failure logged to logs/ directory") from exc

        return parsed

    def update_extraction_context(
        self,
        *,
        document_title: str,
        previous_summary: str,
        existing_entities: List[Dict[str, Any]],
        chunk_order: int,
        chunk_page_label: Optional[str],
        chunk_text: str,
        extracted_triples: List[Dict[str, Any]],
    ) -> ExtractionContextUpdate:
        if self._dry_run:
            return ExtractionContextUpdate(
                summary=previous_summary or f"Document: {document_title}.",
                entities=[],
                raw_response={"error": "llm_endpoint_not_configured"},
            )

        summary_prompt = _load_extraction_summary_prompt()
        payload = {
            "document_title": document_title,
            "previous_summary": previous_summary,
            "existing_entities": existing_entities,
            "chunk": {
                "order": chunk_order,
                "page_label": chunk_page_label,
                "text": chunk_text,
            },
            "extracted_triples": extracted_triples[:30],
        }

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": orjson.dumps(payload).decode()},
        ]

        request_payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": settings.llm_temperature_extraction,
            "response_format": {"type": "json_object"},
        }

        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
        if isinstance(message_content, str):
            parsed_content = orjson.loads(message_content)
        elif isinstance(message_content, dict):
            parsed_content = message_content
        else:
            parsed_content = {}

        update = ExtractionContextUpdate.model_validate(
            {
                "summary": parsed_content.get("summary") or previous_summary,
                "entities": parsed_content.get("entities", []),
                "raw_response": raw,
            }
        )
        return update

    def _fits_ontology(self, subject: str, predicate: str, obj: str, ontology: Dict[str, Any]) -> bool:
        """Check if a triple fits into the predefined ontology."""
        # Example logic: Check if subject and object exist in ontology and predicate is valid
        if subject not in ontology.get("entities", []):
            return False
        if obj not in ontology.get("entities", []):
            return False
        if predicate not in ontology.get("relationships", []):
            return False
        return True

    def suggest_parent_node(
        self,
        *,
        child_labels: Sequence[str],
        existing_parent_labels: Optional[Sequence[str]] = None,
    ) -> OntologySuggestionResponse:
        if not child_labels:
            raise ValueError("child_labels must not be empty")

        if self._dry_run:
            parent_label = _heuristic_parent_label(child_labels)
            suggestion = OntologyParentSuggestion(
                parent_label=parent_label,
                relation="is_a",
                description=f"Heuristic grouping for {', '.join(child_labels)}",
                confidence=0.4,
                rationale="Generated without LLM endpoint; heuristic fallback used.",
                guardrail_flags=["llm_stubbed"],
            )
            return OntologySuggestionResponse(
                suggestion=suggestion,
                raw_response={"error": "llm_endpoint_not_configured"},
            )

        system_prompt = _load_ontology_prompt()
        payload: Dict[str, Any] = {
            "child_nodes": child_labels,
        }
        if existing_parent_labels:
            payload["existing_parents"] = list(existing_parent_labels)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Return a JSON object with a single 'suggestion' key containing the proposed parent node.\n"
                    f"{orjson.dumps(payload).decode()}"
                ),
            },
        ]

        request_payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": settings.llm_temperature_connections,
            "response_format": {"type": "json_object"},
        }

        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
        if isinstance(message_content, str):
            parsed_content = orjson.loads(message_content)
        elif isinstance(message_content, dict):
            parsed_content = message_content
        else:
            parsed_content = {}

        return OntologySuggestionResponse.model_validate(
            {
                "suggestion": parsed_content.get("suggestion"),
                "raw_response": raw,
            }
        )

    def analyze_query(
        self,
        *,
        question: str,
        context: Dict[str, Any],
    ) -> QueryRetrievalResponse:
        if self._dry_run:
            focus_nodes = [
                RetrievalNode(label=node["label"], node_id=node.get("id"))
                for node in context.get("nodes", [])[:3]
            ]
            return QueryRetrievalResponse(
                focus_nodes=focus_nodes,
                focus_predicates=[edge.get("predicate") for edge in context.get("edges", [])[:3] if edge.get("predicate")],
                intent="heuristic",
                notes="LLM retrieval stub",
                context_nodes=context.get("nodes", []),
                raw_response={"error": "llm_endpoint_not_configured"},
            )

        system_prompt = _load_query_retrieval_prompt()
        payload = {
            "question": question,
            "candidate_nodes": context.get("nodes", []),
            "candidate_edges": context.get("edges", []),
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode(),
            },
        ]
        request_payload = {
            "messages": messages,
            "temperature": settings.llm_temperature_retrieval,
            "response_format": {"type": "json_object"},
        }
        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
        if isinstance(message_content, str):
            parsed_content = orjson.loads(message_content)
        elif isinstance(message_content, dict):
            parsed_content = message_content
        else:
            parsed_content = {}
        response = QueryRetrievalResponse.model_validate(
            {
                "focus_nodes": parsed_content.get("focus_nodes", []),
                "focus_predicates": parsed_content.get("focus_predicates", []),
                "intent": parsed_content.get("intent"),
                "notes": parsed_content.get("notes"),
                "context_nodes": context.get("nodes", []),
                "raw_response": raw,
            }
        )
        return response

    def plan_query(
        self,
        *,
        question: str,
        retrieval: QueryRetrievalResponse,
        context: Dict[str, Any],
    ) -> QueryPlanResponse:
        if self._dry_run:
            focus_label = retrieval.focus_nodes[0].label if retrieval.focus_nodes else None
            plan = {
                "plan": {
                    "steps": [
                        {
                            "start_node": focus_label or (context.get("nodes", [{}])[0].get("label") if context.get("nodes") else question[:32]),
                            "direction": "outbound",
                            "predicates": retrieval.focus_predicates[:2],
                            "depth": 2,
                        }
                    ],
                    "max_hops": 2,
                }
            }
            return QueryPlanResponse.model_validate({**plan, "raw_response": plan})

        system_prompt = _load_query_plan_prompt()
        payload = {
            "question": question,
            "retrieval": retrieval.model_dump(),
            "context_nodes": context.get("nodes", []),
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode(),
            },
        ]
        request_payload = {
            "messages": messages,
            "temperature": settings.llm_temperature_plan,
            "response_format": {"type": "json_object"},
        }
        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
        if isinstance(message_content, str):
            parsed_content = orjson.loads(message_content)
        elif isinstance(message_content, dict):
            parsed_content = message_content
        else:
            parsed_content = {}
        return QueryPlanResponse.model_validate(
            {
                "plan": parsed_content.get("plan"),
                "warnings": parsed_content.get("warnings", []),
                "raw_response": raw,
            }
        )

    def answer_query(self, payload: Dict[str, Any]) -> QueryAnswerResponse:
        if self._dry_run:
            matches = payload.get("matches", [])
            text = "\n".join(
                f"- {item['subject']} {item['predicate']} {item['object']}"
                for item in matches[:5]
            )
            return QueryAnswerResponse(
                answers=["Stubbed answer", text],
                cited_edges=[m.get("edge_id") for m in matches[:5] if m.get("edge_id")],
                confidence=0.3,
                notes="LLM answer stub",
                raw_response={"error": "llm_endpoint_not_configured"},
            )

        system_prompt = _load_query_answer_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode(),
            },
        ]
        request_payload = {
            "messages": messages,
            "temperature": settings.llm_temperature_answer,
            "response_format": {"type": "json_object"},
        }
        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
        if isinstance(message_content, str):
            parsed_content = orjson.loads(message_content)
        elif isinstance(message_content, dict):
            parsed_content = message_content
        else:
            parsed_content = {}

        return QueryAnswerResponse.model_validate(
            {
                "answers": parsed_content.get("answers", []),
                "cited_edges": parsed_content.get("cited_edges", []),
                "confidence": parsed_content.get("confidence"),
                "notes": parsed_content.get("notes"),
                "raw_response": raw,
            }
        )

    def recommend_connections(self, payload: Dict[str, Any]) -> ConnectionRecommendationResponse:
        if self._dry_run:
            return ConnectionRecommendationResponse(
                new_nodes=[],
                new_edges=[],
                notes="LLM connection recommendation stub",
                raw_response={"error": "llm_endpoint_not_configured"},
            )

        system_prompt = _load_connection_recommendation_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode(),
            },
        ]
        request_payload = {
            "messages": messages,
            "temperature": settings.llm_temperature_connections,
            "response_format": {"type": "json_object"},
        }
        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
        if isinstance(message_content, str):
            parsed_content = orjson.loads(message_content)
        elif isinstance(message_content, dict):
            parsed_content = message_content
        else:
            parsed_content = {}

        return ConnectionRecommendationResponse.model_validate(
            {
                "new_nodes": parsed_content.get("new_nodes", []),
                "new_edges": parsed_content.get("new_edges", []),
                "notes": parsed_content.get("notes"),
                "raw_response": raw,
            }
        )

    def generate_validation_question(
        self,
        *,
        candidate_payload: Dict[str, Any],
        history: List[Dict[str, str]],
    ) -> str:
        if self._dry_run:
            return "Could you please confirm the key evidence supporting this relationship?"

        system_prompt = _load_candidate_validation_prompt()
        payload = {
            "candidate": candidate_payload,
            "history": history,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode(),
            },
        ]
        request_payload = {
            "messages": messages,
            "temperature": settings.llm_temperature_answer,
        }
        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(message_content, str):
            return message_content.strip()
        if isinstance(message_content, dict):
            return orjson.dumps(message_content).decode()
        return "Please provide additional evidence."

    def generate_review_summary(
        self,
        *,
        candidates: List[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> str:
        if self._dry_run:
            return "Here is what I've gathered so far..."

        system_prompt = _load_review_summary_prompt()
        payload = {
            "candidates": candidates,
            "history": history,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": orjson.dumps(payload).decode(),
            },
        ]
        request_payload = {
            "messages": messages,
            "temperature": settings.llm_temperature_answer,
        }
        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        if isinstance(message_content, str):
            return message_content.strip()
        if isinstance(message_content, dict):
            return orjson.dumps(message_content).decode()
        return "I’m ready when you are to confirm or provide corrections."

    def review_chat(
        self,
        *,
        messages: List[Dict[str, str]],
        graph_context: Dict[str, Any],
    ) -> ReviewAgentResponse:
        if self._dry_run:
            return ReviewAgentResponse(
                reply="LLM endpoint not configured; no changes applied.",
                commands=[],
                raw_response={"error": "llm_endpoint_not_configured"},
            )

        system_prompt = _load_review_agent_prompt()
        context_payload = orjson.dumps(graph_context or {}).decode()
        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": (
                    "Graph context in JSON for grounding. Use conservatively and confirm before mutations.\n"
                    f"{context_payload}"
                ),
            },
        ]

        for item in messages[-12:]:
            role = item.get("role", "user")
            content = str(item.get("content", ""))
            if not content:
                continue
            conversation.append({"role": role, "content": content})

        request_payload = {
            "messages": conversation,
            "temperature": settings.llm_temperature_review,
            "response_format": {"type": "json_object"},
        }

        raw = self._call_api(request_payload)
        message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
        if isinstance(message_content, str):
            try:
                parsed_content = orjson.loads(message_content)
            except orjson.JSONDecodeError:
                parsed_content = {"reply": message_content}
        elif isinstance(message_content, dict):
            parsed_content = message_content
        else:
            parsed_content = {}

        payload = {
            "reply": parsed_content.get("reply", "I could not produce a response."),
            "commands": parsed_content.get("commands", []),
            "raw_response": raw,
        }

        try:
            return ReviewAgentResponse.model_validate(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to parse review chat response; returning fallback reply")
            return ReviewAgentResponse(
                reply=str(parsed_content) if parsed_content else "I'm not sure how to help with that.",
                commands=[],
                raw_response=raw,
            )


def get_client() -> LLMClient:
    """Return a singleton LLM client using application settings."""
    return LLMClient()


def _heuristic_parent_label(child_labels: Sequence[str]) -> str:
    tokens_per_label = []
    for label in child_labels:
        tokens = {token for token in re.split(r"[^a-zA-Z0-9]+", label.lower()) if token}
        tokens_per_label.append(tokens)
    if not tokens_per_label:
        return "Related Concept"

    common_tokens = set.intersection(*tokens_per_label)
    if common_tokens:
        best = max(common_tokens, key=len)
        return best.title()

    # Fall back to the most frequent token across labels
    frequency: Dict[str, int] = {}
    for tokens in tokens_per_label:
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
    if frequency:
        best = max(frequency.items(), key=lambda item: (item[1], len(item[0])))[0]
        return best.title()

    return "Related Concept"
