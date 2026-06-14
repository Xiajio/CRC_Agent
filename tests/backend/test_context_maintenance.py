from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from backend.api.services.context_maintenance import (
    ContextMaintenanceService,
    MemoryChangeDecision,
    create_context_maintenance_service,
)
import backend.api.services.context_maintenance as context_maintenance_module
from src.nodes.memory_nodes import SUMMARY_TRIGGER_THRESHOLD
from src.state import StructuredSummary


def _messages(count: int, text: str = "follow-up") -> list[HumanMessage | AIMessage]:
    return [HumanMessage(content=f"{text} {index}") for index in range(count)]


class _DecisionModel:
    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision
        self.invocations = 0
        self.prompts: list[str] = []

    def with_structured_output(self, schema: type[MemoryChangeDecision]):
        def _invoke(prompt_value: Any):
            self.invocations += 1
            self.prompts.append(str(prompt_value))
            return schema.model_validate(self.decision)

        return RunnableLambda(_invoke)


def test_finalize_below_threshold_skips_model_and_preserves_cursor() -> None:
    model = _DecisionModel({"has_change": True, "field_changed": "dynamic_info"})
    service = ContextMaintenanceService(model=model)

    result = service.finalize(
        agent_state={"messages": _messages(SUMMARY_TRIGGER_THRESHOLD - 1)},
        existing_context_state={"summary_memory_cursor": 0},
    )

    assert result == {}
    assert model.invocations == 0


def test_finalize_at_threshold_updates_dynamic_info_from_llm_decision() -> None:
    model = _DecisionModel(
        {
            "has_change": True,
            "field_changed": "dynamic_info",
            "field_name": "treatment_plan",
            "old_value": "FOLFOX",
            "new_value": "FOLFIRI",
            "importance": "high",
            "reason": "Current treatment changed.",
            "text_summary": "Treatment changed from FOLFOX to FOLFIRI.",
        }
    )
    service = ContextMaintenanceService(model=model)

    result = service.finalize(
        agent_state={
            "messages": _messages(SUMMARY_TRIGGER_THRESHOLD, "treatment update"),
            "structured_summary": StructuredSummary(
                dynamic_info={"treatment_plan": "FOLFOX"},
                text_summary="Initial treatment was FOLFOX.",
            ),
        },
        existing_context_state={"summary_memory_cursor": 0},
    )

    structured = result["structured_summary"]
    assert model.invocations == 1
    assert result["summary_memory_cursor"] == SUMMARY_TRIGGER_THRESHOLD
    assert result["summary_memory"] == "Treatment changed from FOLFOX to FOLFIRI."
    assert structured["dynamic_info"]["treatment_plan"] == "FOLFIRI"
    assert structured["text_summary"] == "Treatment changed from FOLFOX to FOLFIRI."


def test_finalize_negative_progression_does_not_create_anchor_event() -> None:
    model = _DecisionModel(
        {
            "has_change": False,
            "reason": "The message says there is no progression, so no positive progression event exists.",
            "text_summary": "",
        }
    )
    service = ContextMaintenanceService(model=model)

    result = service.finalize(
        agent_state={
            "messages": _messages(SUMMARY_TRIGGER_THRESHOLD, "未进展，无转移"),
            "structured_summary": StructuredSummary(
                anchor_events=[],
                dynamic_info={"disease_status": "stable"},
                text_summary="Disease status is stable.",
            ),
        },
        existing_context_state={"summary_memory_cursor": 0},
    )

    structured = result["structured_summary"]
    assert model.invocations == 1
    assert structured["anchor_events"] == []
    assert structured["dynamic_info"] == {"disease_status": "stable"}


def test_finalize_anchor_event_change_updates_summary_memory() -> None:
    model = _DecisionModel(
        {
            "has_change": True,
            "field_changed": "anchor_events",
            "field_name": "adverse_event",
            "old_value": None,
            "new_value": "grade 2 diarrhea after chemotherapy",
            "importance": "high",
            "reason": "New clinically relevant adverse event.",
            "text_summary": "Patient developed grade 2 diarrhea after chemotherapy.",
        }
    )
    service = ContextMaintenanceService(model=model)

    result = service.finalize(
        agent_state={
            "messages": _messages(SUMMARY_TRIGGER_THRESHOLD, "adverse event"),
            "structured_summary": StructuredSummary(text_summary="Existing summary."),
            "summary_memory": "Existing summary.",
        },
        existing_context_state={"summary_memory_cursor": 0},
    )

    structured = result["structured_summary"]
    assert result["summary_memory"] == "Patient developed grade 2 diarrhea after chemotherapy."
    assert structured["text_summary"] == "Patient developed grade 2 diarrhea after chemotherapy."
    assert structured["anchor_events"] == [
        {
            "type": "adverse_event",
            "old_value": None,
            "new_value": "grade 2 diarrhea after chemotherapy",
            "importance": "high",
            "reason": "New clinically relevant adverse event.",
        }
    ]


def test_finalize_prefers_existing_context_state_over_agent_state() -> None:
    model = _DecisionModel(
        {
            "has_change": False,
            "reason": "No change.",
            "text_summary": "",
        }
    )
    service = ContextMaintenanceService(model=model)

    service.finalize(
        agent_state={
            "messages": _messages(SUMMARY_TRIGGER_THRESHOLD),
            "structured_summary": StructuredSummary(
                dynamic_info={"stage": "stale-agent"},
                text_summary="Stale agent summary.",
            ),
        },
        existing_context_state={
            "summary_memory_cursor": 0,
            "structured_summary": {
                "dynamic_info": {"stage": "fresh-context"},
                "text_summary": "Fresh context summary.",
            },
        },
    )

    assert model.invocations == 1
    assert "fresh-context" in model.prompts[0]
    assert "stale-agent" not in model.prompts[0]


def test_finalize_no_change_advances_cursor_without_mutating_summary_layers() -> None:
    model = _DecisionModel(
        {
            "has_change": False,
            "reason": "Routine follow-up without clinically meaningful changes.",
            "text_summary": "",
        }
    )
    service = ContextMaintenanceService(model=model)
    existing = StructuredSummary(
        immutable_info={"diagnosis": "rectal cancer"},
        dynamic_info={"treatment_plan": "FOLFOX"},
        anchor_events=[{"type": "diagnosis", "content": "confirmed"}],
        text_summary="Existing summary.",
    )

    result = service.finalize(
        agent_state={
            "messages": _messages(SUMMARY_TRIGGER_THRESHOLD),
            "structured_summary": existing,
            "summary_memory": "Existing summary.",
        },
        existing_context_state={"summary_memory_cursor": 0},
    )

    structured = result["structured_summary"]
    assert model.invocations == 1
    assert result["summary_memory_cursor"] == SUMMARY_TRIGGER_THRESHOLD
    assert result["summary_memory"] == "Existing summary."
    assert structured["immutable_info"] == {"diagnosis": "rectal cancer"}
    assert structured["dynamic_info"] == {"treatment_plan": "FOLFOX"}
    assert structured["anchor_events"] == [{"type": "diagnosis", "content": "confirmed"}]


def test_create_context_maintenance_service_disables_model_in_fixture_mode(monkeypatch) -> None:
    def _unexpected_service(_settings: Any):
        raise AssertionError("fixture mode should not create an LLM")

    monkeypatch.setattr(context_maintenance_module, "LLMService", _unexpected_service)

    service = create_context_maintenance_service(object(), runner_mode="fixture")

    assert service._model is None


def test_create_context_maintenance_service_creates_model_in_real_mode(monkeypatch) -> None:
    model = object()

    class _FakeLLMService:
        def __init__(self, settings: Any) -> None:
            self.settings = settings

        def create_chat_model(self) -> object:
            return model

    monkeypatch.setattr(context_maintenance_module, "LLMService", _FakeLLMService)

    settings = type("_Settings", (), {"llm": object()})()
    service = create_context_maintenance_service(settings, runner_mode="real")

    assert service._model is model
