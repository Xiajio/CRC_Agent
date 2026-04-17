from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langchain_core.messages import HumanMessage

from src.nodes.memory_nodes import (
    SUMMARY_TRIGGER_THRESHOLD,
    _merge_anchor_events,
    detect_key_events,
    incremental_summary,
    update_layered_summary,
)
from src.nodes.node_utils import _build_summary_memory, _format_messages_for_summary
from src.state import StructuredSummary, ensure_agent_state


CONTEXT_MAINTENANCE_RUNNING_MESSAGE = "答案已生成，后台正在整理上下文"
CONTEXT_MAINTENANCE_COMPLETED_MESSAGE = "上下文整理完成"
CONTEXT_MAINTENANCE_FAILED_MESSAGE = "上下文整理失败"
CONTEXT_STATE_KEYS = (
    "summary_memory",
    "structured_summary",
    "summary_memory_cursor",
)


def _merge_incremental_update(summary: StructuredSummary, incremental_update: dict[str, Any] | None) -> StructuredSummary:
    if not incremental_update:
        return summary

    field_changed = incremental_update.get("field_changed")
    field_name = incremental_update.get("field_name")
    new_value = incremental_update.get("new_value")

    if field_changed == "anchor_events":
        summary.anchor_events = _merge_anchor_events(
            summary.anchor_events,
            [{
                "type": field_name,
                "old_value": incremental_update.get("old_value"),
                "new_value": new_value,
                "reason": incremental_update.get("reason"),
            }],
        )
        return summary

    if field_changed == "immutable_info" and field_name:
        immutable = dict(summary.immutable_info or {})
        immutable[field_name] = new_value
        summary.immutable_info = immutable
        return summary

    if field_changed == "dynamic_info" and field_name:
        dynamic = dict(summary.dynamic_info or {})
        dynamic[field_name] = new_value
        summary.dynamic_info = dynamic

    return summary


class ContextMaintenanceService:
    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    def finalize(
        self,
        *,
        agent_state: Mapping[str, Any] | None,
        existing_context_state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged_state = dict(agent_state or {})
        for key in CONTEXT_STATE_KEYS:
            if isinstance(existing_context_state, Mapping) and key in existing_context_state:
                merged_state[key] = deepcopy(existing_context_state[key])

        state = ensure_agent_state(merged_state)
        messages = list(state.messages or [])
        if not messages:
            return {}

        structured_summary = state.structured_summary or StructuredSummary()
        cursor = max(0, state.summary_memory_cursor or 0)
        new_messages = messages[cursor:] if cursor < len(messages) else []

        if self._model is None:
            fallback_state = state.model_copy(update={
                "summary_memory": None,
                "structured_summary": structured_summary,
            })
            summary_text = _build_summary_memory(fallback_state)
            if summary_text and summary_text != "No summary available":
                structured_summary.text_summary = summary_text
            return {
                "summary_memory": summary_text,
                "summary_memory_cursor": len(messages),
                "structured_summary": structured_summary.model_dump(mode="json"),
            }

        if not new_messages:
            return {}

        new_dialogue = _format_messages_for_summary(new_messages, max_chars=1000)
        detected_events: list[dict[str, Any]] = []
        for message in new_messages:
            content = getattr(message, "content", "") or ""
            detected_events.extend(detect_key_events(content))

        if detected_events:
            structured_summary.anchor_events = _merge_anchor_events(
                structured_summary.anchor_events,
                detected_events,
            )

        incremental_update = incremental_summary(structured_summary, new_dialogue, self._model)
        if len(new_messages) < SUMMARY_TRIGGER_THRESHOLD and not incremental_update:
            return {}

        next_summary = update_layered_summary(structured_summary, new_dialogue, self._model)
        next_summary.anchor_events = _merge_anchor_events(
            next_summary.anchor_events,
            structured_summary.anchor_events,
        )
        next_summary = _merge_incremental_update(next_summary, incremental_update)
        next_summary.last_update_turn = max(
            next_summary.last_update_turn,
            len([message for message in messages if isinstance(message, HumanMessage)]),
        )

        return {
            "summary_memory": (
                (next_summary.text_summary or "").strip()
                or (structured_summary.text_summary or "").strip()
                or (state.summary_memory or "").strip()
            ),
            "summary_memory_cursor": len(messages),
            "structured_summary": next_summary.model_dump(mode="json"),
        }


def create_context_maintenance_service(settings: Any, *, runner_mode: str) -> ContextMaintenanceService:
    del settings, runner_mode
    return ContextMaintenanceService(model=None)
