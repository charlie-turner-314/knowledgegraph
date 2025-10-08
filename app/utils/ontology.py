from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import orjson

ONTOLOGY_FILE = (
    Path(__file__).resolve().parents[2] / "resources" / "ontology" / "tad_ontology.json"
)


@dataclass(frozen=True)
class OntologyEvaluation:
    """Result of comparing a triple against the configured ontology."""

    is_match: bool
    predicate: str
    subject_type: Optional[str]
    object_type: Optional[str]
    suggested_mutations: List[Dict[str, Any]]


@lru_cache(maxsize=1)
def _load_raw_ontology() -> Dict[str, Any]:
    try:
        return orjson.loads(ONTOLOGY_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"entities": [], "relationships": []}


@lru_cache(maxsize=1)
def _ontology_sets() -> Dict[str, Any]:
    data = _load_raw_ontology()
    entities = data.get("entities", []) or []
    relationships = data.get("relationships", []) or []
    return {
        "entities": {str(item).strip().lower(): str(item).strip() for item in entities if str(item).strip()},
        "relationships": {
            str(item).strip().lower(): str(item).strip() for item in relationships if str(item).strip()
        },
    }


def get_ontology() -> Dict[str, Any]:
    """Return the ontology definition (entities, relationships)."""

    return {
        "entities": list(_ontology_sets()["entities"].values()),
        "relationships": list(_ontology_sets()["relationships"].values()),
    }


def _clean_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def infer_entity_type(attributes: Optional[Iterable[Dict[str, Any]]]) -> Optional[str]:
    """Best-effort extraction of an entity type label from attribute payloads."""

    if not attributes:
        return None

    preferred_keys = {"entity_type", "proposed_entity_type", "type", "class", "category"}
    for entry in attributes:
        if not isinstance(entry, dict):
            continue
        name = _clean_string(entry.get("name") or entry.get("label"))
        if not name or name.lower() not in preferred_keys:
            continue
        for key in ("value_text", "value", "value_label"):
            candidate = entry.get(key)
            if candidate is not None:
                inferred = _clean_string(candidate)
                if inferred:
                    return inferred
        if entry.get("value_boolean") is not None:
            return "boolean"
        if entry.get("value_number") is not None:
            return "number"
    return None


def evaluate_triple(
    *,
    subject_label: str,
    predicate: str,
    object_label: str,
    subject_attributes: Optional[Iterable[Dict[str, Any]]] = None,
    object_attributes: Optional[Iterable[Dict[str, Any]]] = None,
) -> OntologyEvaluation:
    """Return ontology compliance information for a triple."""

    sets = _ontology_sets()
    allowed_entities = sets["entities"]
    allowed_relationships = sets["relationships"]

    subject_label_clean = _clean_string(subject_label) or ""
    predicate_clean = _clean_string(predicate) or ""
    object_label_clean = _clean_string(object_label) or ""

    subject_type = _clean_string(infer_entity_type(subject_attributes))
    object_type = _clean_string(infer_entity_type(object_attributes))

    subject_type_key = subject_type.lower() if subject_type else None
    object_type_key = object_type.lower() if object_type else None
    predicate_key = predicate_clean.lower()

    subject_valid = subject_type_key in allowed_entities if subject_type_key else False
    object_valid = object_type_key in allowed_entities if object_type_key else False
    predicate_valid = predicate_key in allowed_relationships if predicate_key else False

    suggested_mutations: List[Dict[str, Any]] = []

    if not subject_valid:
        suggested_mutations.append(
            {
                "type": "entity_type",
                "role": "subject",
                "entity_label": subject_label_clean,
                "suggested_type": subject_type,
                "reason": (
                    "entity_type_not_in_ontology"
                    if subject_type
                    else "entity_type_missing"
                ),
            }
        )

    if not object_valid:
        suggested_mutations.append(
            {
                "type": "entity_type",
                "role": "object",
                "entity_label": object_label_clean,
                "suggested_type": object_type,
                "reason": (
                    "entity_type_not_in_ontology"
                    if object_type
                    else "entity_type_missing"
                ),
            }
        )

    if not predicate_valid:
        suggested_mutations.append(
            {
                "type": "relationship",
                "role": "predicate",
                "relationship": predicate_clean,
                "subject_type": subject_type,
                "object_type": object_type,
                "reason": "relationship_not_in_ontology",
            }
        )

    is_match = not suggested_mutations

    return OntologyEvaluation(
        is_match=is_match,
        predicate=predicate_clean,
        subject_type=subject_type,
        object_type=object_type,
        suggested_mutations=suggested_mutations,
    )


def summarize_mutations(mutations: Iterable[Dict[str, Any]]) -> List[str]:
    """Return human-friendly sentences describing ontology mutations."""

    summaries: List[str] = []
    for mutation in mutations:
        if not isinstance(mutation, dict):
            continue
        mtype = mutation.get("type")
        if mtype == "entity_type":
            role = mutation.get("role", "entity")
            entity_label = mutation.get("entity_label") or "(unknown entity)"
            suggested_type = mutation.get("suggested_type") or "unspecified class"
            summaries.append(
                f"Add or confirm entity type '{suggested_type}' for the {role} '{entity_label}'."
            )
        elif mtype == "relationship":
            relationship = mutation.get("relationship") or "(relationship)"
            subject_type = mutation.get("subject_type") or "subject"
            object_type = mutation.get("object_type") or "object"
            summaries.append(
                f"Add relationship '{relationship}' covering {subject_type} → {object_type}."
            )
    return summaries


def _write_ontology(data: Dict[str, Any]) -> None:
    ONTOLOGY_FILE.parent.mkdir(parents=True, exist_ok=True)
    serialized = orjson.dumps(
        {
            "entities": list(data.get("entities", []) or []),
            "relationships": list(data.get("relationships", []) or []),
        },
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
    )
    ONTOLOGY_FILE.write_bytes(serialized + b"\n")
    _load_raw_ontology.cache_clear()
    _ontology_sets.cache_clear()


def ensure_entity_label(label: str) -> bool:
    """Ensure ``label`` exists in the ontology entity list."""

    cleaned = _clean_string(label)
    if not cleaned:
        return False

    data = _load_raw_ontology().copy()
    entities = list(data.get("entities", []) or [])
    lowered = {str(item).strip().lower() for item in entities if str(item).strip()}
    if cleaned.lower() in lowered:
        return False

    entities.append(cleaned)
    entities = sorted(dict.fromkeys(entities), key=lambda item: str(item).lower())
    data["entities"] = entities
    _write_ontology(data)
    return True


def ensure_relationship_label(label: str) -> bool:
    """Ensure ``label`` exists in the ontology relationship list."""

    cleaned = _clean_string(label)
    if not cleaned:
        return False

    data = _load_raw_ontology().copy()
    relationships = list(data.get("relationships", []) or [])
    lowered = {str(item).strip().lower() for item in relationships if str(item).strip()}
    if cleaned.lower() in lowered:
        return False

    relationships.append(cleaned)
    relationships = sorted(dict.fromkeys(relationships), key=lambda item: str(item).lower())
    data["relationships"] = relationships
    _write_ontology(data)
    return True
