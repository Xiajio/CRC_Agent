from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from backend.api.adapters.event_normalizer import normalize_tick
from backend.api.adapters.state_snapshot import build_recovery_snapshot
from backend.api.services.session_store import SessionMeta
from src.nodes.citation_nodes import _fast_citation_report
from src.nodes.decision_nodes import _format_final_response, node_finalize
from src.nodes.node_utils import _invoke_structured_with_recovery
from src.nodes.tools_executor import node_tool_executor
from src.state import CRCAgentState, PlanStep, update_step_status


def test_update_step_status_does_not_mutate_original_plan_step() -> None:
    state = CRCAgentState(
        messages=[],
        current_plan=[
            PlanStep(
                id="step_1",
                description="Collect the requested context.",
                tool_needed="search",
                status="pending",
            )
        ],
    )

    original_step = state.current_plan[0]

    updated = update_step_status(state, "step_1", "completed", {"source": "test"})

    assert state.current_plan[0] is original_step
    assert state.current_plan[0].status == "pending"
    assert updated.current_plan[0].status == "completed"
    assert updated.current_plan[0] is not original_step
    assert updated.step_history[-1]["status"] == "completed"


def test_node_tool_executor_returns_error_for_empty_messages(monkeypatch) -> None:
    monkeypatch.setattr("src.nodes.tools_executor.list_all_tools", lambda: [])

    state = CRCAgentState(messages=[])

    result = node_tool_executor(state)

    assert result == {"error": "No tool calls found in the last message."}


def test_node_tool_executor_returns_error_for_non_ai_last_message(monkeypatch) -> None:
    monkeypatch.setattr("src.nodes.tools_executor.list_all_tools", lambda: [])

    state = CRCAgentState(messages=[HumanMessage(content="hello")])

    result = node_tool_executor(state)

    assert result == {"error": "No tool calls found in the last message."}


def test_invoke_structured_with_recovery_uses_raw_text_parser_after_structured_failure() -> None:
    class ReviewSchema(BaseModel):
        verdict: str = "APPROVED"
        feedback: str = ""

    class FailingStructuredModel:
        def invoke(self, payload):
            raise RuntimeError("structured unavailable")

    class RawModel:
        def with_structured_output(self, schema):
            return FailingStructuredModel()

        def invoke(self, payload):
            return "REJECTED: missing citation support"

    result = _invoke_structured_with_recovery(
        prompt=None,
        model=RawModel(),
        schema=ReviewSchema,
        payload={"decision": "x"},
        raw_text_parser=lambda text: {
            "verdict": "REJECTED" if "REJECTED" in text else "APPROVED",
            "feedback": text,
        },
    )

    assert result.verdict == "REJECTED"
    assert result.feedback == "REJECTED: missing citation support"


def test_rejected_critic_event_marks_human_review_required() -> None:
    events = normalize_tick(
        "critic",
        {
            "critic_verdict": "REJECTED",
            "critic_feedback": "missing neoadjuvant treatment rationale",
            "iteration_count": 1,
        },
    )

    critic_event = next(event for event in events if getattr(event, "type", None) == "critic.verdict")

    assert critic_event.requires_human_review is True


def test_rejected_critic_snapshot_marks_human_review_required() -> None:
    snapshot = build_recovery_snapshot(
        SessionMeta(session_id="sess-test", thread_id="thread-test"),
        {
            "critic_verdict": "REJECTED",
            "critic_feedback": "missing citation support",
            "iteration_count": 1,
        },
    )

    assert snapshot.critic is not None
    assert snapshot.critic["requires_human_review"] is True


def test_final_response_starts_with_human_review_warning_when_critic_rejects() -> None:
    text = _format_final_response(
        decision={
            "summary": "Synthetic rectal cancer case.",
            "treatment_plan": [
                {
                    "title": "Initial plan",
                    "content": "Discuss total neoadjuvant therapy before surgery.",
                }
            ],
        },
        verdict="REJECTED",
        feedback="missing citation support",
        references=[],
        citation_report={"coverage_score": 45, "missing_claims": ["insufficient_references"], "needs_more_sources": True},
        evaluation_report={"verdict": "PASS"},
    )

    assert text.startswith("> [!WARNING]\n> HUMAN_REVIEW_REQUIRED")
    assert "missing citation support" in text.splitlines()[1]


def test_finalize_emits_panel_data_and_human_review_flag_for_rejected_decision() -> None:
    state = CRCAgentState(
        messages=[],
        decision_json={
            "summary": "Synthetic rectal cancer case.",
            "treatment_plan": [
                {"title": "Treatment sequence", "content": "Consider TNT, restaging, then TME."},
                {"title": "Follow-up", "content": "Monitor toxicity and response."},
            ],
            "follow_up": ["MDT review"],
        },
        critic_verdict="REJECTED",
        critic_feedback="missing citation support",
    )

    result = node_finalize(show_thinking=False)(state)

    assert result["requires_human_review"] is True
    assert result["current_plan"]
    assert result["roadmap"]


def test_finalize_derives_panel_plan_from_step_rationale_decision_items() -> None:
    state = CRCAgentState(
        messages=[],
        decision_json={
            "summary": "Stage III low rectal adenocarcinoma, pMMR, cT3N1M0.",
            "treatment_plan": [
                {
                    "step": "Discuss total neoadjuvant therapy in multidisciplinary tumor board.",
                    "rationale": "cT3N1 low rectal cancer generally requires neoadjuvant treatment before surgery.",
                }
            ],
        },
        critic_verdict="REJECTED",
        critic_feedback="missing citation support",
    )

    result = node_finalize(show_thinking=False)(state)

    assert result["current_plan"][0].description == (
        "Discuss total neoadjuvant therapy in multidisciplinary tumor board."
    )
    assert result["current_plan"][0].reasoning == (
        "cT3N1 low rectal cancer generally requires neoadjuvant treatment before surgery."
    )


def test_template_fast_without_references_requires_more_sources() -> None:
    state = CRCAgentState(
        messages=[],
        findings={"decision_strategy": "template_fast"},
        decision_json={
            "summary": "Synthetic rectal cancer case.",
            "treatment_plan": [{"title": "Plan", "content": "Treatment recommendation."}],
        },
        retrieved_references=[],
    )

    report = _fast_citation_report(state)

    assert report.needs_more_sources is True
    assert "no_direct_references" in report.missing_claims
    assert "no_direct_references" in report.notes


def test_real_case_human_review_fixture_normalizes_review_plan_and_no_direct_references() -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "graph_ticks" / "real_case_human_review.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    events = []
    for tick in fixture["ticks"]:
        events.extend(normalize_tick(tick["node_name"], tick["node_output"]))

    critic_events = [event for event in events if getattr(event, "type", None) == "critic.verdict"]
    roadmap_events = [event for event in events if getattr(event, "type", None) == "roadmap.update"]
    plan_events = [event for event in events if getattr(event, "type", None) == "plan.update"]
    reference_events = [event for event in events if getattr(event, "type", None) == "references.append"]
    final_messages = [
        event
        for event in events
        if getattr(event, "type", None) == "message.done"
        and "HUMAN_REVIEW_REQUIRED" in str(getattr(event, "content", ""))
    ]

    assert critic_events[-1].requires_human_review is True
    assert any(
        any(step.get("status") == "blocked" for step in event.roadmap)
        for event in roadmap_events
    )
    assert any(
        any(step.get("status") == "blocked" for step in event.plan)
        for event in plan_events
    )
    assert reference_events == []
    assert final_messages
