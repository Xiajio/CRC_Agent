from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_core.messages.utils import convert_to_messages

from backend.api.services.session_store import SessionMeta

CONTEXT_PAYLOAD_ALLOWLIST = {
    "fixture_case",
    "fixture_tick_delay_ms",
    "current_patient_id",
}
PATIENT_CONTEXT_REQUIRED_KEYS = {
    "patient_version",
    "projection_version",
    "medical_card_snapshot",
}


@dataclass(slots=True)
class PreparedGraphPayload:
    payload: dict[str, Any]
    drained_pending_context_messages: list[Any]


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _normalize_message(message: Any) -> BaseMessage:
    if isinstance(message, BaseMessage):
        return message
    if isinstance(message, Mapping) and {"type", "data"}.issubset(message):
        return messages_from_dict([dict(message)])[0]
    return convert_to_messages([message])[0]


def _current_turn_message(chat_request: Any) -> BaseMessage:
    for key in ("message", "current_turn", "content", "text"):
        value = _get_value(chat_request, key)
        if value is not None:
            return _normalize_message(value)
    raise ValueError("chat_request must include a current turn message")


def _snapshot_value(state_snapshot: Any, key: str, default: Any = None) -> Any:
    return _get_value(state_snapshot, key, default)


def _context_value(session_meta: SessionMeta, state_snapshot: Any, key: str, default: Any = None) -> Any:
    context_state = getattr(session_meta, "context_state", None)
    if isinstance(context_state, Mapping) and key in context_state:
        return context_state.get(key, default)
    return _snapshot_value(state_snapshot, key, default)


def _patient_context_cache(session_meta: SessionMeta) -> dict[str, Any] | None:
    context_state = getattr(session_meta, "context_state", None)
    if not isinstance(context_state, Mapping):
        return None
    cache = context_state.get("patient_context_cache")
    if not isinstance(cache, Mapping):
        return None
    if not PATIENT_CONTEXT_REQUIRED_KEYS.issubset(cache):
        return None
    if cache.get("patient_version") is None or cache.get("projection_version") is None:
        return None
    return deepcopy(dict(cache))


def build_graph_payload(
    chat_request: Any,
    session_meta: SessionMeta,
    state_snapshot: Any,
) -> PreparedGraphPayload:
    chat_request_context = _get_value(chat_request, "context", {})
    current_turn_message = _current_turn_message(chat_request)
    drained_pending_context_messages = list(session_meta.pending_context_messages)
    payload_context_messages = [
        _normalize_message(message)
        for message in drained_pending_context_messages
    ]

    payload_messages = payload_context_messages + [current_turn_message]
    patient_context = _patient_context_cache(session_meta)
    medical_card = (
        deepcopy(patient_context.get("medical_card_snapshot"))
        if patient_context is not None
        else _context_value(session_meta, state_snapshot, "medical_card")
    )

    payload: dict[str, Any] = {
        "messages": payload_messages,
        "patient_profile": _snapshot_value(state_snapshot, "patient_profile"),
        "clinical_stage": _snapshot_value(state_snapshot, "clinical_stage", "Assessment") or "Assessment",
        "findings": _snapshot_value(state_snapshot, "findings", {}),
        "assessment_draft": _snapshot_value(state_snapshot, "assessment_draft"),
        "decision_json": None,
        "medical_card": medical_card,
        "patient_context": patient_context,
        "roadmap": _snapshot_value(state_snapshot, "roadmap", []),
        "summary_memory": _context_value(session_meta, state_snapshot, "summary_memory"),
        "structured_summary": _context_value(session_meta, state_snapshot, "structured_summary"),
        "summary_memory_cursor": _context_value(session_meta, state_snapshot, "summary_memory_cursor", 0),
    }

    current_patient_id = _snapshot_value(state_snapshot, "current_patient_id")
    if current_patient_id is not None:
        payload["current_patient_id"] = current_patient_id

    if isinstance(chat_request_context, Mapping):
        for key in CONTEXT_PAYLOAD_ALLOWLIST:
            value = chat_request_context.get(key)
            if value is not None:
                payload[key] = value

    session_meta.pending_context_messages.clear()
    return PreparedGraphPayload(
        payload=payload,
        drained_pending_context_messages=drained_pending_context_messages,
    )


def restore_pending_context_messages(
    session_meta: SessionMeta,
    drained_messages: list[Any],
) -> None:
    session_meta.pending_context_messages.extend(list(drained_messages))
