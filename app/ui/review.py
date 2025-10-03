from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import streamlit as st

from app.data.db import session_scope
from app.data.models import NodeAttributeType
from app.services.review_service import ReviewService, serialize_candidate


def render() -> None:
    """Render the SME review queue UI."""
    st.header("SME Review Queue")
    st.write("Approve, reject, or refine candidate triples before they enter the graph.")

    if flash := st.session_state.pop("review_flash", None):
        severity = flash.get("type", "info")
        message = flash.get("message", "")
        if severity == "success":
            st.success(message)
        elif severity == "error":
            st.error(message)
        else:
            st.info(message)

    pending = _load_pending_candidates()
    if not pending:
        st.info("No pending triples. Ingest a document to generate candidates.")
        return

    for candidate in pending:
        header = f"#{candidate['id']} — {candidate['subject']} / {candidate['predicate']} / {candidate['object']}"
        expander_col, approve_col, reject_col = st.columns([0.72, 0.14, 0.14])
        with expander_col:
            expander = st.expander(header, expanded=False)
        candidate_subject_attrs = candidate.get("subject_attributes") or []
        candidate_object_attrs = candidate.get("object_attributes") or []
        candidate_tags = candidate.get("tags") or []

        if approve_col.button(
            "Approve",
            key=f"quick_approve_{candidate['id']}",
            help="Approve with the current subject/predicate/object values.",
        ):
            _resolve_candidate(
                action="approve",
                candidate_id=candidate["id"],
                payload={
                    "subject": candidate["subject"],
                    "predicate": candidate["predicate"],
                    "object": candidate["object"],
                    "notes": None,
                    "subject_attributes": candidate_subject_attrs,
                    "object_attributes": candidate_object_attrs,
                    "tags": candidate_tags,
                },
            )
        if reject_col.button(
            "Reject",
            key=f"quick_reject_{candidate['id']}",
            help="Reject without opening the full detail view.",
        ):
            _resolve_candidate(
                action="reject",
                candidate_id=candidate["id"],
                payload={"reason": "Rejected via quick action"},
            )

        with expander:
            st.caption(
                f"Document: {candidate['document']}"
                + (f" — Page: {candidate['page_label']}" if candidate['page_label'] else "")
            )
            if candidate.get("is_duplicate"):
                duplicate_hint = (
                    f"Potential duplicate of candidate #{candidate['duplicate_of']}"
                    if candidate.get("duplicate_of")
                    else "Potential duplicate based on matching triple values"
                )
                st.warning(duplicate_hint)
            if candidate.get("chunk_text"):
                st.write(candidate["chunk_text"][:1000])

            subject_key = f"subject_{candidate['id']}"
            predicate_key = f"predicate_{candidate['id']}"
            object_key = f"object_{candidate['id']}"
            notes_key = f"notes_{candidate['id']}"

            subject_value = st.text_input("Subject", candidate["subject"], key=subject_key)
            predicate_value = st.text_input("Predicate", candidate["predicate"], key=predicate_key)
            object_value = st.text_input("Object", candidate["object"], key=object_key)
            notes_value = st.text_area("Notes", key=notes_key)

            subject_attr_key = f"subject_attrs_{candidate['id']}"
            object_attr_key = f"object_attrs_{candidate['id']}"
            tag_key = f"tags_{candidate['id']}"

            subject_attr_value = _format_attributes_for_input(candidate_subject_attrs)
            object_attr_value = _format_attributes_for_input(candidate_object_attrs)
            tag_value = ", ".join(candidate_tags)

            subject_attr_raw = st.text_area(
                "Subject attributes",
                value=subject_attr_value,
                key=subject_attr_key,
                help="Comma or newline separated key=value pairs. Prefix values with enum:, bool:, or number: to force a type.",
            )
            object_attr_raw = st.text_area(
                "Object attributes",
                value=object_attr_value,
                key=object_attr_key,
                help="Comma or newline separated key=value pairs. Prefix values with enum:, bool:, or number: to force a type.",
            )
            tag_raw = st.text_input(
                "Triple tags",
                value=tag_value,
                key=tag_key,
                help="Comma separated tags applied to this triple.",
            )

            col1, col2 = st.columns(2)
            if col1.button("Approve", key=f"approve_{candidate['id']}"):
                subject_attrs = _parse_attribute_input(subject_attr_raw)
                object_attrs = _parse_attribute_input(object_attr_raw)
                tags = _parse_tags(tag_raw)
                _resolve_candidate(
                    action="approve",
                    candidate_id=candidate["id"],
                    payload={
                        "subject": subject_value,
                        "predicate": predicate_value,
                        "object": object_value,
                        "notes": notes_value,
                        "subject_attributes": subject_attrs,
                        "object_attributes": object_attrs,
                        "tags": tags,
                    },
                )
            if col2.button("Reject", key=f"reject_{candidate['id']}"):
                _resolve_candidate(
                    action="reject",
                    candidate_id=candidate["id"],
                    payload={"reason": notes_value or "Rejected via UI"},
                )
            if candidate.get("is_duplicate"):
                if st.button(
                    "Dismiss duplicate",
                    key=f"dismiss_{candidate['id']}",
                    help="Mark this candidate as a duplicate and remove it from the queue.",
                ):
                    _resolve_candidate(
                        action="dismiss_duplicate",
                        candidate_id=candidate["id"],
                        payload={"reason": "Dismissed as duplicate"},
                    )


def _load_pending_candidates() -> List[Dict[str, Any]]:
    """Load pending candidate triples and serialise them for display."""
    with session_scope() as session:
        service = ReviewService(session)
        pending = service.list_pending(limit=25)
        serialized = [serialize_candidate(item) for item in pending]
    return serialized


def _resolve_candidate(*, action: str, candidate_id: int, payload: Dict[str, Any]) -> None:
    """Execute reviewer actions and update the review flash state."""
    try:
        with session_scope() as session:
            service = ReviewService(session)
            if action == "approve":
                service.approve_candidate(
                    candidate_id,
                    actor="sme-local",
                    subject_override=payload.get("subject"),
                    predicate_override=payload.get("predicate"),
                    object_override=payload.get("object"),
                    notes=payload.get("notes"),
                    subject_attributes=payload.get("subject_attributes"),
                    object_attributes=payload.get("object_attributes"),
                    tags=payload.get("tags"),
                )
            elif action == "reject":
                service.reject_candidate(candidate_id, actor="sme-local", reason=payload.get("reason"))
            elif action == "dismiss_duplicate":
                service.reject_candidate(candidate_id, actor="sme-local", reason=payload.get("reason"))
            else:
                raise ValueError(f"Unknown action {action}")
    except Exception as exc:  # noqa: BLE001
        st.session_state["review_flash"] = {"type": "error", "message": str(exc)}
    else:
        messages = {
            "approve": "Candidate approved",
            "reject": "Candidate rejected",
            "dismiss_duplicate": "Candidate dismissed as duplicate",
        }
        st.session_state["review_flash"] = {
            "type": "success",
            "message": messages.get(action, "Candidate updated"),
        }
    st.rerun()


def _parse_attribute_input(raw_value: Optional[str]) -> List[Dict[str, object]]:
    """Parse human-entered attributes into structured dictionaries."""
    if not raw_value:
        return []
    entries = re.split(r"[\n;,]+", raw_value)
    parsed: List[Dict[str, object]] = []
    for entry in entries:
        token = entry.strip()
        if not token or "=" not in token:
            continue
        name, value = token.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            continue

        data_type = NodeAttributeType.string
        value_text: Optional[str] = None
        value_number: Optional[float] = None
        value_boolean: Optional[bool] = None

        prefix_match = re.match(r"^(enum|bool|number):(.+)$", value, flags=re.IGNORECASE)
        if prefix_match:
            prefix = prefix_match.group(1).lower()
            value_body = prefix_match.group(2).strip()
            if prefix == "enum":
                data_type = NodeAttributeType.enum
                value_text = value_body
            elif prefix == "bool":
                data_type = NodeAttributeType.boolean
                lowered = value_body.lower()
                if lowered in {"true", "1", "yes"}:
                    value_boolean = True
                elif lowered in {"false", "0", "no"}:
                    value_boolean = False
                else:
                    value_boolean = None
                if value_boolean is None:
                    continue
            elif prefix == "number":
                data_type = NodeAttributeType.number
                try:
                    value_number = float(value_body)
                except ValueError:
                    continue
        else:
            lowered = value.lower()
            if lowered in {"true", "false", "yes", "no", "0", "1"}:
                data_type = NodeAttributeType.boolean
                value_boolean = lowered in {"true", "yes", "1"}
            else:
                try:
                    numeric = float(value)
                except ValueError:
                    if len(value.split()) <= 3 and value.isalpha():
                        data_type = NodeAttributeType.enum
                    value_text = value
                else:
                    data_type = NodeAttributeType.number
                    value_number = numeric

        if data_type == NodeAttributeType.string and value_text is None:
            value_text = value

        parsed.append(
            {
                "name": name,
                "data_type": data_type,
                "value_text": value_text,
                "value_number": value_number,
                "value_boolean": value_boolean,
            }
        )
    return parsed


def _parse_tags(raw_value: Optional[str]) -> List[str]:
    """Normalise a comma separated tag string."""
    if not raw_value:
        return []
    tags = [tag.strip().lower() for tag in raw_value.split(",") if tag.strip()]
    return sorted(set(tags))


def _format_attributes_for_input(attrs: Optional[List[Dict[str, object]]]) -> str:
    """Render attribute dictionaries into text suitable for the input textarea."""
    if not attrs:
        return ""
    formatted: List[str] = []
    for attr in attrs:
        name = attr.get("name")
        if not name:
            continue
        value = None
        if attr.get("value_text") is not None:
            value = attr["value_text"]
        elif attr.get("value_number") is not None:
            value = attr["value_number"]
        elif attr.get("value_boolean") is not None:
            value = attr["value_boolean"]
        elif attr.get("value") is not None:
            value = attr["value"]
        else:
            value = ""

        data_type = attr.get("data_type")
        prefix = ""
        if isinstance(data_type, NodeAttributeType):
            dt = data_type.value
        elif isinstance(data_type, str):
            dt = data_type
        else:
            dt = None
        if dt == NodeAttributeType.enum.value:
            prefix = "enum:"
        elif dt == NodeAttributeType.boolean.value:
            prefix = "bool:"
        elif dt == NodeAttributeType.number.value:
            prefix = "number:"
        formatted.append(f"{name}={prefix}{value}")
    return "\n".join(formatted)
