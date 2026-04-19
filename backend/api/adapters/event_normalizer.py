from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import convert_to_messages

from backend.api.adapters.card_payload_sanitizer import sanitize_card_payload, strip_binary
from backend.api.adapters.card_extractor import extract_cards
from backend.api.adapters.message_content import sanitize_user_visible_content, _split_inline_thinking
from backend.api.adapters.reference_normalizer import normalize_reference_list
from backend.api.schemas.events import (
    CardUpsertEvent,
    CriticVerdictEvent,
    FindingsPatchEvent,
    MessageDoneEvent,
    PatientProfileUpdateEvent,
    PlanUpdateEvent,
    ReferencesAppendEvent,
    RoadmapUpdateEvent,
    SafetyAlertEvent,
    StageUpdateEvent,
    StatusNodeEvent,
)

INTERNAL_MESSAGE_PREFIXES = (
    "[CT Staging Result]",
    "[CT Staging Skipped]",
    "[MRI Local Staging]",
    "[CT Distant Metastasis Screening]",
    "[WARNING]",
    "[Router]",
    "[Graph Router]",
    "[Intent]",
    "[Planner]",
    "[General Chat]",
    "[Staging Router]",
    "审核:",
    "✅",
    "[Critic]",
    "[Layered Summary]",
    "**知识检索完成**",
    "🔍 检索词:",
)
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


def _coerce_messages(messages: Sequence[Any] | None) -> list[BaseMessage]:
    if messages is None:
        return []

    coerced: list[BaseMessage] = []
    for raw_message in messages:
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


def _serialize_inline_cards_from_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
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


def _message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(value or "")


def _is_internal_ai_message(message: BaseMessage) -> bool:
    if not isinstance(message, AIMessage):
        return False

    content = _message_content(message.content).strip()
    if not content:
        return False

    for prefix in INTERNAL_MESSAGE_PREFIXES:
        if content.startswith(prefix):
            return True

    if content.startswith("[") and "Result]" in content:
        return True
    if content.startswith("[") and "Skipped]" in content:
        return True
    if "Agent is thinking" in content:
        return True
    if content.startswith("Profile:"):
        return True
    if set(content) == {"="}:
        return True

    return False


def _extract_references_from_tool_result(tool_name: str, content: str) -> list[dict[str, Any]]:
    if tool_name != "search_clinical_guidelines" or not content:
        return []

    references: list[dict[str, Any]] = []
    pattern = r"\[(\d+)\]\s*source=([^\n]+)\n([\s\S]*?)(?=\[\d+\]\s*source=|$)"
    for idx_text, source, snippet in re.findall(pattern, content):
        source_name = source.strip()
        if "/" in source_name:
            source_name = source_name.split("/")[-1]
        if "\\" in source_name:
            source_name = source_name.split("\\")[-1]

        snippet_clean = snippet.strip()
        if len(snippet_clean) > 200:
            snippet_clean = snippet_clean[:200] + "..."

        references.append(
            {
                "index": int(idx_text),
                "source": source_name,
                "snippet": snippet_clean,
            }
        )

    return references


def normalize_tick(
    node_name: str,
    node_output: Mapping[str, Any] | None,
    messages: Sequence[Any] | None = None,
) -> list[Any]:
    events: list[Any] = [StatusNodeEvent(node=node_name)]
    if node_output is None or not isinstance(node_output, Mapping):
        return events

    langchain_messages = _coerce_messages(messages if messages is not None else node_output.get("messages"))
    emitted_message_done = False

    stage = node_output.get("clinical_stage") or node_output.get("stage")
    if isinstance(stage, str) and stage:
        events.append(StageUpdateEvent(stage=stage))

    patient_profile = _coerce_mapping(node_output.get("patient_profile"))
    if patient_profile is not None:
        events.append(PatientProfileUpdateEvent(profile=strip_binary(patient_profile)))

    critic_verdict = node_output.get("critic_verdict")
    critic_feedback = node_output.get("critic_feedback")
    iteration_count = node_output.get("iteration_count")
    if critic_verdict is not None or critic_feedback is not None:
        events.append(
            CriticVerdictEvent(
                verdict=str(critic_verdict or ""),
                feedback=str(critic_feedback) if critic_feedback is not None else None,
                iteration_count=int(iteration_count) if iteration_count is not None else None,
            )
        )

    roadmap = _coerce_mapping_list(node_output.get("roadmap"))
    if roadmap:
        events.append(RoadmapUpdateEvent(roadmap=strip_binary(roadmap)))

    plan = _coerce_mapping_list(node_output.get("current_plan") or node_output.get("plan"))
    if plan:
        events.append(PlanUpdateEvent(plan=strip_binary(plan)))

    safety_violation = node_output.get("safety_violation")
    if isinstance(safety_violation, str) and safety_violation:
        events.append(SafetyAlertEvent(message=safety_violation))

    findings = _coerce_mapping(node_output.get("findings"))
    if findings is not None:
        events.append(FindingsPatchEvent(patch=strip_binary(findings)))

    references = normalize_reference_list(node_output.get("retrieved_references"))
    if references:
        events.append(ReferencesAppendEvent(items=strip_binary(references)))
    else:
        extracted_refs: list[dict[str, Any]] = []
        for message in langchain_messages:
            if not isinstance(message, ToolMessage):
                continue
            tool_name = str(getattr(message, "name", "") or "")
            extracted_refs.extend(_extract_references_from_tool_result(tool_name, _message_content(message.content)))
        if extracted_refs:
            events.append(ReferencesAppendEvent(items=strip_binary(normalize_reference_list(extracted_refs))))

    # 去重：同一条 AIMessage（按 message_id）在单次 normalize_tick 内只 emit 一次
    _seen_message_ids: set[str] = set()
    for message in langchain_messages:
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        if _is_internal_ai_message(message):
            continue

        content = _message_content(message.content).strip()
        if not content:
            continue

        msg_id = str(getattr(message, "id", "") or "")
        if msg_id and msg_id in _seen_message_ids:
            continue
        if msg_id:
            _seen_message_ids.add(msg_id)

        thinking_content = None
        additional_kwargs = getattr(message, "additional_kwargs", None) or {}
        raw_thinking = additional_kwargs.get("thinking_content") or additional_kwargs.get("reasoning_content") or additional_kwargs.get("thinking")
        if isinstance(raw_thinking, str) and raw_thinking.strip():
            thinking_content = raw_thinking.strip()

        sanitized_content = strip_binary(sanitize_user_visible_content(message.content))

        if not thinking_content and isinstance(sanitized_content, str):
            inline_thinking, split_response = _split_inline_thinking(sanitized_content)
            if inline_thinking:
                thinking_content = inline_thinking
                sanitized_content = split_response

        events.append(
            MessageDoneEvent(
                message_id=msg_id or None,
                content=sanitized_content,
                thinking=thinking_content,
                node=node_name,
                inline_cards=_serialize_inline_cards_from_messages([message]) or None,
            )
        )
        emitted_message_done = True

    assessment_draft = node_output.get("assessment_draft")
    if (
        node_name == "assessment"
        and not emitted_message_done
        and isinstance(assessment_draft, str)
        and assessment_draft.strip()
    ):
        events.append(
            MessageDoneEvent(
                message_id=f"{node_name}:assessment_draft",
                content=strip_binary(assessment_draft),
                node=node_name,
            )
        )

    events.extend(_sanitize_card_events(extract_cards(node_name, node_output, messages=langchain_messages)))
    return events
