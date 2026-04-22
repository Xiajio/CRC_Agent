from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages

from backend.api.adapters.card_payload_sanitizer import sanitize_card_payload, strip_binary
from backend.api.adapters.card_extractor import extract_cards
from backend.api.adapters.event_normalizer import _is_internal_ai_message
from backend.api.adapters.message_content import sanitize_user_visible_content
from backend.api.adapters.reference_normalizer import normalize_reference_list
from backend.api.schemas.events import CardUpsertEvent
from backend.api.schemas.responses import MessageHistoryPage, RecoverySnapshot, SessionMessage
from backend.api.services.session_store import SessionMeta
from src.services.patient_card_projector import project_patient_self_report_card

INLINE_CARD_TYPES = {
    "patient_card",
    "imaging_card",
    "tumor_detection_card",
    "radiomics_report_card",
    "triage_card",
    "triage_question_card",
    "decision_card",
}
INLINE_CARD_PRIORITY = {
    "patient_card": 1,
    "imaging_card": 2,
    "tumor_detection_card": 3,
    "radiomics_report_card": 4,
    "triage_card": 5,
    "triage_question_card": 5,
    "decision_card": 6,
}
TRIAGE_STATE_KEYS = (
    "encounter_track",
    "clinical_entry_reason",
    "entry_explanation_shown",
    "known_crc_signals",
    "triage_risk_level",
    "triage_disposition",
    "triage_suggested_tests",
    "triage_summary",
    "symptom_snapshot",
)


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else None
    return None


def _coerce_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    items: list[dict[str, Any]] = []
    for entry in value:
        mapped = _coerce_mapping(entry)
        if mapped is not None:
            items.append(mapped)
    return items


def _coerce_messages(messages: Sequence[Any] | None) -> list[Any]:
    if messages is None:
        return []
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        return list(messages)
    return []


def _is_user_visible_message(raw_message: Any) -> bool:
    message_type = _message_attr(raw_message, "type")
    if message_type in {"tool", "system"}:
        return False
    return True


def _is_internal_ai_control_message(raw_message: Any) -> bool:
    messages = _coerce_langchain_messages([raw_message])
    return any(_is_internal_ai_message(message) for message in messages)


def _coerce_langchain_messages(messages: Sequence[Any] | None) -> list[BaseMessage]:
    coerced: list[BaseMessage] = []

    for raw_message in _coerce_messages(messages):
        if isinstance(raw_message, BaseMessage):
            coerced.append(raw_message)
            continue
        if not isinstance(raw_message, Mapping):
            continue

        payload = dict(raw_message)
        if payload.get("type") == "tool" and not payload.get("tool_call_id"):
            payload["tool_call_id"] = "tool_call"

        try:
            message = convert_to_messages([payload])[0]
        except Exception:
            additional_kwargs = payload.get("additional_kwargs", {})
            if not isinstance(additional_kwargs, Mapping):
                additional_kwargs = {}

            content = payload.get("content", "")
            message_type = payload.get("type")
            if message_type == "ai":
                message = AIMessage(content=content, additional_kwargs=dict(additional_kwargs))
            elif message_type == "human":
                message = HumanMessage(content=content, additional_kwargs=dict(additional_kwargs))
            elif message_type == "system":
                message = SystemMessage(content=content, additional_kwargs=dict(additional_kwargs))
            elif message_type == "tool":
                message = ToolMessage(
                    content=str(content),
                    tool_call_id=str(payload.get("tool_call_id") or "tool_call"),
                    additional_kwargs=dict(additional_kwargs),
                )
            else:
                continue

        if isinstance(message, BaseMessage):
            coerced.append(message)

    return coerced


def _merge_triage_state_fields(state: Any, findings: dict[str, Any]) -> dict[str, Any]:
    merged = dict(findings)
    for key in TRIAGE_STATE_KEYS:
        has_value = False
        if isinstance(state, Mapping):
            has_value = key in state
        else:
            fields_set = getattr(state, "model_fields_set", None)
            if isinstance(fields_set, set):
                has_value = key in fields_set
        if has_value:
            merged[key] = _get_value(state, key)
    return merged


def _extract_asset_refs(raw_message: Any) -> list[dict[str, Any]]:
    candidates: list[Any] = []

    if isinstance(raw_message, BaseMessage):
        candidates.append(getattr(raw_message, "additional_kwargs", None))
        candidates.append(getattr(raw_message, "artifact", None))
    elif isinstance(raw_message, Mapping):
        candidates.append(raw_message.get("additional_kwargs"))
        candidates.append(raw_message.get("artifact"))

    refs: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            direct_refs = candidate.get("asset_refs")
            if isinstance(direct_refs, Sequence) and not isinstance(direct_refs, (str, bytes)):
                refs.extend(_coerce_mapping_list(direct_refs))

    return strip_binary(refs)


def _message_attr(raw_message: Any, name: str) -> Any:
    if isinstance(raw_message, BaseMessage):
        return getattr(raw_message, name, None)
    if isinstance(raw_message, Mapping):
        return raw_message.get(name)
    return None


def _prune_inline_cards(inline_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not inline_cards:
        return []

    highest_priority = max(
        INLINE_CARD_PRIORITY.get(str(card.get("card_type")), 0)
        for card in inline_cards
    )
    if highest_priority <= 0:
        return inline_cards

    return [
        card
        for card in inline_cards
        if INLINE_CARD_PRIORITY.get(str(card.get("card_type")), 0) == highest_priority
    ]


def _serialize_inline_cards(raw_message: Any) -> list[dict[str, Any]]:
    messages = _coerce_langchain_messages([raw_message])
    if not messages:
        return []

    inline_cards: list[dict[str, Any]] = []
    for event in _sanitize_card_events(extract_cards("message", {}, messages=messages)):
        if event.card_type not in INLINE_CARD_TYPES:
            continue

        inline_cards.append(
            {
                "card_type": event.card_type,
                "payload": event.payload,
            }
        )

    return _prune_inline_cards(inline_cards)


def _serialize_message(raw_message: Any, cursor: str) -> SessionMessage:
    return SessionMessage(
        cursor=cursor,
        type=str(_message_attr(raw_message, "type") or "unknown"),
        content=strip_binary(sanitize_user_visible_content(_message_attr(raw_message, "content"))),
        id=_message_attr(raw_message, "id"),
        name=_message_attr(raw_message, "name"),
        tool_call_id=_message_attr(raw_message, "tool_call_id"),
        status=_message_attr(raw_message, "status"),
        asset_refs=_extract_asset_refs(raw_message),
        inline_cards=_serialize_inline_cards(raw_message),
    )


def _paginate_messages(messages: list[Any], before: str | int | None, limit: int) -> tuple[list[Any], int, str | None, int]:
    total = len(messages)
    capped_limit = max(1, min(int(limit), 100))

    if before is None:
        end = total
    else:
        try:
            before_cursor = int(before)
        except (TypeError, ValueError):
            before_cursor = total
        end = max(0, min(before_cursor, total))

    start = max(0, end - capped_limit)
    next_before_cursor = str(start) if start > 0 else None
    return messages[start:end], total, next_before_cursor, start


def _sanitize_card_events(events: list[CardUpsertEvent]) -> list[CardUpsertEvent]:
    sanitized: list[CardUpsertEvent] = []
    for event in events:
        sanitized.append(
            CardUpsertEvent(
                card_type=event.card_type,
                payload=sanitize_card_payload(event.card_type, event.payload),
                source_channel=event.source_channel,
            )
        )
    return sanitized


def _inject_patient_self_report_card(
    *,
    cards: list[CardUpsertEvent],
    session_meta: SessionMeta,
    state: Mapping[str, Any],
) -> list[CardUpsertEvent]:
    if session_meta.scene != "patient":
        return cards

    recovery_state = dict(state)
    context_state = _coerce_mapping(getattr(session_meta, "context_state", None))
    if "medical_card" in context_state:
        recovery_state["medical_card"] = context_state["medical_card"]

    projected = project_patient_self_report_card(recovery_state)
    if projected is None:
        return cards

    replacement = CardUpsertEvent(
        card_type="patient_card",
        payload=sanitize_card_payload("patient_card", projected),
        source_channel="state",
    )
    first_self_report_index: int | None = None
    filtered: list[CardUpsertEvent] = []
    for index, event in enumerate(cards):
        payload_meta = _coerce_mapping(event.payload.get("card_meta")) or {}
        if (
            event.card_type == "patient_card"
            and isinstance(event.payload, Mapping)
            and payload_meta.get("source_mode") == "patient_self_report"
        ):
            if first_self_report_index is None:
                first_self_report_index = index
            continue
        filtered.append(event)

    insert_at = len(filtered) if first_self_report_index is None else min(first_self_report_index, len(filtered))
    filtered.insert(insert_at, replacement)
    return filtered


def build_message_history(
    agent_state: Mapping[str, Any] | None,
    before: str | int | None = None,
    limit: int = 50,
) -> MessageHistoryPage:
    state = agent_state if isinstance(agent_state, Mapping) else {}
    raw_messages = [
        message
        for message in _coerce_messages(_get_value(state, "messages", []))
        if _is_user_visible_message(message) and not _is_internal_ai_control_message(message)
    ]
    page_messages, total, next_before_cursor, start = _paginate_messages(raw_messages, before, limit)

    return MessageHistoryPage(
        messages_total=total,
        next_before_cursor=next_before_cursor,
        messages=[
            _serialize_message(message, cursor=str(start + index))
            for index, message in enumerate(page_messages)
        ],
    )


def build_recovery_snapshot(
    session_meta: SessionMeta,
    agent_state: Mapping[str, Any] | None,
    message_limit: int = 50,
) -> RecoverySnapshot:
    state = agent_state if isinstance(agent_state, Mapping) else {}
    history = build_message_history(state, limit=message_limit)

    messages = _coerce_messages(_get_value(state, "messages", []))
    cards = _sanitize_card_events(
        extract_cards(
            "snapshot",
            state,
            messages=_coerce_langchain_messages(messages),
        )
    )
    cards = _inject_patient_self_report_card(cards=cards, session_meta=session_meta, state=state)

    findings = _coerce_mapping(_get_value(state, "findings", {})) or {}
    findings = _merge_triage_state_fields(state, findings)
    current_patient_id = (
        _get_value(state, "current_patient_id")
        or findings.get("current_patient_id")
        or session_meta.patient_id
    )
    critic_verdict = _get_value(state, "critic_verdict")
    critic_feedback = _get_value(state, "critic_feedback")
    iteration_count = _get_value(state, "iteration_count")
    critic: dict[str, Any] | None = None
    if critic_verdict is not None or critic_feedback is not None:
        critic = {
            "verdict": critic_verdict,
            "feedback": critic_feedback,
        }
        if iteration_count is not None:
            critic["iteration_count"] = iteration_count

    safety_violation = _get_value(state, "safety_violation")
    safety_alert = (
        {"message": safety_violation, "blocking": True}
        if safety_violation
        else strip_binary(_coerce_mapping(_get_value(state, "safety_alert")))
    )
    raw_context_maintenance = _coerce_mapping(session_meta.context_maintenance) or {}
    context_maintenance = {
        key: raw_context_maintenance[key]
        for key in ("status", "message", "error")
        if raw_context_maintenance.get(key) is not None
    } or None
    context_state_payload = _coerce_mapping(session_meta.context_state) or {}
    if "summary_memory" not in context_state_payload:
        summary_memory = _get_value(state, "summary_memory")
        if summary_memory is not None:
            context_state_payload["summary_memory"] = summary_memory
    if "summary_memory_cursor" not in context_state_payload:
        summary_cursor = _get_value(state, "summary_memory_cursor")
        if summary_cursor is not None:
            context_state_payload["summary_memory_cursor"] = summary_cursor
    if "structured_summary" not in context_state_payload:
        structured_summary = _coerce_mapping(_get_value(state, "structured_summary"))
        if structured_summary is not None:
            context_state_payload["structured_summary"] = structured_summary
    context_state = strip_binary(context_state_payload) or None

    return RecoverySnapshot(
        snapshot_version=session_meta.snapshot_version,
        messages=history.messages,
        messages_total=history.messages_total,
        messages_next_before_cursor=history.next_before_cursor,
        cards=cards,
        roadmap=strip_binary(_coerce_mapping_list(_get_value(state, "roadmap", []))),
        findings=strip_binary(findings),
        patient_profile=strip_binary(_coerce_mapping(_get_value(state, "patient_profile"))),
        stage=_get_value(state, "stage") or _get_value(state, "clinical_stage"),
        assessment_draft=strip_binary(_get_value(state, "assessment_draft")),
        current_patient_id=current_patient_id,
        references=strip_binary(
            normalize_reference_list(
                _get_value(state, "retrieved_references", _get_value(state, "references", []))
            )
        ),
        plan=strip_binary(
            _coerce_mapping_list(_get_value(state, "current_plan", _get_value(state, "plan", [])))
        ),
        critic=strip_binary(critic),
        safety_alert=safety_alert,
        uploaded_assets=strip_binary(dict(session_meta.uploaded_assets)),
        context_maintenance=strip_binary(context_maintenance),
        context_state=context_state,
    )


def build_session_snapshot(
    agent_state: Mapping[str, Any] | None,
    session_meta: SessionMeta,
    message_limit: int = 50,
) -> RecoverySnapshot:
    return build_recovery_snapshot(
        session_meta=session_meta,
        agent_state=agent_state,
        message_limit=message_limit,
    )
