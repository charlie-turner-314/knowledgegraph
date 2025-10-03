from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import orjson
import requests
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

from app.core.config import settings

from .schemas import (
    ExtractionResponse,
    OntologyParentSuggestion,
    OntologySuggestionResponse,
    ConnectionRecommendationResponse,
    QueryAnswerResponse,
    QueryPlanResponse,
    QueryRetrievalResponse,
    RetrievalNode,
)

logger = logging.getLogger(__name__)

PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "prompts" / "extraction_system_prompt.txt"
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
        response = requests.post(self.endpoint, headers=headers, data=orjson.dumps(payload))
        print(response.status_code)
        print(response.json())
        exit()
        response.raise_for_status()
        return response.json()

    def extract_triples(
        self,
        *,
        chunk_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        canonical_context: Optional[List[Dict[str, Any]]] = None,
    ) -> ExtractionResponse:
        """Call the configured LLM to extract triples from text."""

        if self._dry_run:
            logger.warning("LLM endpoint not configured; returning stubbed triple response")
            stub = ExtractionResponse(
                triples=[],
                raw_response={"error": "llm_endpoint_not_configured"},
            )
            return stub

        system_prompt = _load_system_prompt()
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": system_prompt,
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
            "temperature": settings.temperature,
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
            raise RuntimeError("LLM triple extraction failed") from exc

        try:
            message_content = raw.get("choices", [{}])[0].get("message", {}).get("content", {})
            if isinstance(message_content, str):
                parsed_content = orjson.loads(message_content)
            elif isinstance(message_content, dict):
                parsed_content = message_content
            else:
                parsed_content = {}
            parsed = ExtractionResponse.model_validate(
                {
                    "triples": parsed_content.get("triples", []),
                    "raw_response": raw,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse LLM extraction response")
            raise RuntimeError("Invalid response from LLM extraction endpoint") from exc

        return parsed

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
            "temperature": settings.temperature,
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
            "temperature": 0.0,
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
            "temperature": 0.0,
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
            "temperature": 0.3,
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
            "temperature": 0.0,
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
