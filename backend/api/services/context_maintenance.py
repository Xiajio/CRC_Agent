from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, model_validator

from src.nodes.memory_nodes import (
    SUMMARY_TRIGGER_THRESHOLD,
    _merge_anchor_events,
)
from src.nodes.node_utils import (
    _build_summary_memory,
    _format_messages_for_summary,
    _invoke_structured_with_recovery,
)
from src.services.llm_service import LLMService
from src.state import StructuredSummary, ensure_agent_state


CONTEXT_MAINTENANCE_RUNNING_MESSAGE = "答案已生成，后台正在整理上下文"
CONTEXT_MAINTENANCE_COMPLETED_MESSAGE = "上下文整理完成"
CONTEXT_MAINTENANCE_FAILED_MESSAGE = "上下文整理失败"
CONTEXT_STATE_KEYS = (
    "summary_memory",
    "structured_summary",
    "summary_memory_cursor",
)


class MemoryChangeDecision(BaseModel):
    has_change: bool = False
    field_changed: Literal["immutable_info", "dynamic_info", "anchor_events", "text_summary"] | None = None
    field_name: str | None = None
    old_value: Any | None = None
    new_value: Any | None = None
    importance: Literal["critical", "high", "medium", "low"] | None = None
    reason: str = ""
    text_summary: str = ""

    @model_validator(mode="after")
    def _normalize_no_change(self) -> "MemoryChangeDecision":
        if not self.has_change:
            self.field_changed = None
            self.field_name = None
            self.old_value = None
            self.new_value = None
            self.importance = None
        return self


MEMORY_CHANGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是肿瘤临床对话的长期记忆维护器。只记录有临床意义的新事实或变化，"
                "不要把普通寒暄、重复确认、计划中的假设当作变化。\n"
                "必须进行否定语义判断：'无转移'、'未进展'、'没有拒绝治疗'、'无过敏史'"
                "不得生成对应的正向锚点事件；可作为状态信息写入 dynamic_info 或 text_summary。\n"
                "'免疫组化'、'免疫功能'、'免疫力' 不等于免疫治疗方案调整。\n"
                "'诊断' 只有在确诊、诊断改变、诊断被排除时才算事件。\n"
                "如果没有临床意义变化，返回 has_change=false。"
            ),
        ),
        (
            "human",
            (
                "已有不变信息层：\n{immutable_info}\n\n"
                "已有动态信息层：\n{dynamic_info}\n\n"
                "已有锚点事件：\n{anchor_events}\n\n"
                "已有自然语言摘要：\n{text_summary}\n\n"
                "新增对话：\n{new_dialogue}\n\n"
                "请返回结构化 JSON，字段包括 has_change, field_changed, field_name, "
                "old_value, new_value, importance, reason, text_summary。"
            ),
        ),
    ]
)


def _apply_memory_change(summary: StructuredSummary, decision: MemoryChangeDecision) -> StructuredSummary:
    if not decision.has_change:
        return summary

    field_changed = decision.field_changed
    field_name = (decision.field_name or "").strip()
    new_value = decision.new_value

    if field_changed == "anchor_events":
        summary.anchor_events = _merge_anchor_events(
            summary.anchor_events,
            [{
                "type": field_name or "clinical_change",
                "old_value": decision.old_value,
                "new_value": new_value,
                "importance": decision.importance,
                "reason": decision.reason,
            }],
        )

    elif field_changed == "immutable_info" and field_name:
        immutable = dict(summary.immutable_info or {})
        immutable[field_name] = new_value
        summary.immutable_info = immutable

    elif field_changed == "dynamic_info" and field_name:
        dynamic = dict(summary.dynamic_info or {})
        dynamic[field_name] = new_value
        summary.dynamic_info = dynamic

    if decision.text_summary.strip():
        summary.text_summary = decision.text_summary.strip()

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

        if len(new_messages) < SUMMARY_TRIGGER_THRESHOLD:
            return {}

        new_dialogue = _format_messages_for_summary(new_messages, max_chars=1000)
        decision = _invoke_structured_with_recovery(
            prompt=MEMORY_CHANGE_PROMPT,
            model=self._model,
            schema=MemoryChangeDecision,
            payload={
                "immutable_info": structured_summary.immutable_info,
                "dynamic_info": structured_summary.dynamic_info,
                "anchor_events": structured_summary.anchor_events,
                "text_summary": structured_summary.text_summary,
                "new_dialogue": new_dialogue,
            },
            log_prefix="[Context Maintenance]",
            fallback_factory=lambda _payload, _err: MemoryChangeDecision(),
        )
        next_summary = _apply_memory_change(structured_summary.model_copy(deep=True), decision)
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
    if str(runner_mode or "").strip().lower() == "fixture":
        return ContextMaintenanceService(model=None)
    llm_settings = getattr(settings, "llm", None)
    if llm_settings is None:
        return ContextMaintenanceService(model=None)
    try:
        model = LLMService(llm_settings).create_chat_model()
    except Exception:
        model = None
    return ContextMaintenanceService(model=model)
