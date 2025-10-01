from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import orjson
import requests
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

from app.core.config import settings

from .schemas import ExtractionResponse, ExtractionTriple

logger = logging.getLogger(__name__)


class GemmaClient:
    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint = endpoint or settings.gemma_endpoint
        self.api_key = api_key or settings.gemma_api_key
        self._dry_run = self.endpoint is None

    @retry(wait=wait_random_exponential(multiplier=1, max=10), stop=stop_after_attempt(3))
    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(self.endpoint, headers=headers, data=orjson.dumps(payload))
        response.raise_for_status()
        return response.json()

    def extract_triples(self, *, chunk_text: str, metadata: Optional[Dict[str, Any]] = None) -> ExtractionResponse:
        """Call Gemma to extract triples from a chunk of text."""

        if self._dry_run:
            logger.warning("Gemma endpoint not configured; returning stubbed triple response")
            stub = ExtractionResponse(
                triples=[],
                raw_response={"error": "gemma_endpoint_not_configured"},
            )
            return stub

        payload = {
            "model": "gemma-3-27b-instruct",  # placeholder
            "messages": [
                {
                    "role": "system",
                    "content": "You extract subject-predicate-object triples from text.",
                },
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
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if metadata:
            payload["metadata"] = metadata

        try:
            raw = self._call_api(payload)
        except RetryError as exc:
            logger.exception("Gemma triple extraction failed after retries")
            raise RuntimeError("Gemma triple extraction failed") from exc

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
            logger.exception("Failed to parse Gemma extraction response")
            raise RuntimeError("Invalid response from Gemma extraction endpoint") from exc

        return parsed


def get_client() -> GemmaClient:
    return GemmaClient()
