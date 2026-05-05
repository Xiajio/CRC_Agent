from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from backend.api.adapters.event_normalizer import normalize_tick
from backend.api.adapters.state_snapshot import build_recovery_snapshot
from backend.api.services.session_store import SessionMeta
from src.nodes.citation_nodes import _fast_citation_report
from src.nodes.decision_nodes import _format_final_response, node_decision, node_finalize
from src.nodes.node_utils import _invoke_structured_with_recovery
from src.nodes.tools_executor import node_tool_executor
from src.state import CRCAgentState, PatientProfile, PlanStep, update_step_status


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


def test_invoke_structured_with_recovery_prefers_json_after_thinking_over_raw_parser() -> None:
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
            return (
                '<think>The critic considered {"verdict":"REJECTED","feedback":"wrong"} first.</think>\n'
                '{"verdict":"APPROVED","feedback":"需要补充 MMR/MSI 检测。"}'
            )

    result = _invoke_structured_with_recovery(
        prompt=None,
        model=RawModel(),
        schema=ReviewSchema,
        payload={"decision": "x"},
        raw_text_parser=lambda text: {
            "verdict": "APPROVED",
            "feedback": text,
        },
    )

    assert result.verdict == "APPROVED"
    assert result.feedback == "需要补充 MMR/MSI 检测。"


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


def test_internal_decision_and_critic_status_messages_are_not_emitted_as_chat() -> None:
    decision_events = normalize_tick(
        "decision",
        {
            "messages": [
                HumanMessage(content="进行诊断"),
                AIMessage(content="[Decision] template-fast 患者为结肠中，当前临床分期支持 cT4bN1cM0。"),
            ],
        },
    )
    critic_events = normalize_tick(
        "critic",
        {
            "critic_verdict": "REJECTED",
            "critic_feedback": "missing references",
            "messages": [
                AIMessage(content="❌ **诊断流程审核未通过** (Critic: REJECTED)"),
            ],
        },
    )

    assert all(getattr(event, "type", None) != "message.done" for event in decision_events)
    assert all(getattr(event, "type", None) != "message.done" for event in critic_events)
    assert any(getattr(event, "type", None) == "critic.verdict" for event in critic_events)


def test_current_diagnosis_status_messages_are_not_emitted_as_chat() -> None:
    events = normalize_tick(
        "decision",
        {
            "messages": [
                AIMessage(content="[Decision] template-fast \u60a3\u8005\u4e3a\u7ed3\u80a0\u4e2d\uff0c\u5f53\u524d\u4e34\u5e8a\u5206\u671f\u652f\u6301 cT4bN1cM0\u3002"),
                AIMessage(content="\u274c **\u8bca\u65ad\u6d41\u7a0b\u5ba1\u6838\u672a\u901a\u8fc7** (Critic: REJECTED)"),
                AIMessage(content="\U0001f4cb \u6cbb\u7597\u65b9\u6848\u5df2\u751f\u6210: \u4e34\u5e8a\u51b3\u7b56\u6458\u8981"),
            ],
        },
    )

    assert all(getattr(event, "type", None) != "message.done" for event in events)


@pytest.mark.asyncio
async def test_template_fast_retry_preserves_structured_plan_after_quality_rejection() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="\u5f00\u59cb\u8bca\u65ad\u6d41\u7a0b")],
        patient_profile=PatientProfile(
            age=31,
            gender="\u7537",
            ecog_score=0,
            pathology_confirmed=True,
            mmr_status="pMMR (MSS)",
            tnm_staging={"cT": "cT4b", "cN": "cN1c", "cM": "cM0"},
        ),
        findings={
            "fast_pass_mode": True,
            "pathology_confirmed": True,
            "tumor_location": "colon",
            "tumor_subsite": "\u4e2d",
            "histology_type": "\u7ed3\u80a0\u764c",
            "mmr_status": "pMMR (MSS)",
            "tnm_staging": {"cT": "cT4b", "cN": "cN1c", "cM": "cM0", "stage_group": "III"},
        },
        critic_verdict="REJECTED",
        critic_feedback='{"verdict":"APPROVED","feedback":"no_direct_references"}',
        iteration_count=1,
    )

    result = await node_decision(model=None, tools=[], show_thinking=False)(state)

    assert result["findings"]["decision_strategy"] == "template_fast"
    assert result["decision_json"]["summary"] != "\u4e34\u5e8a\u51b3\u7b56\u6458\u8981"
    assert len(result["decision_json"]["treatment_plan"]) >= 5


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


def test_final_response_renders_concise_quality_notice_when_critic_rejects() -> None:
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

    assert text.startswith("# 🏥 临床治疗建议")
    assert "### 质控提示" in text
    assert "需人工复核" in text
    assert "missing citation support" in text


def test_final_response_extracts_critic_json_feedback_and_does_not_show_raw_payload() -> None:
    raw_feedback = (
        '<think>{"verdict":"REJECTED","feedback":"wrong"}</think>\n'
        '{"verdict":"APPROVED","feedback":"需要补充 MMR/MSI 检测。"}'
    )

    text = _format_final_response(
        decision={
            "summary": "患者为结肠癌，当前临床分期支持 cT4bN1cM0。",
            "treatment_plan": [
                {"title": "手术方案", "content": "推荐结肠癌根治术。"},
            ],
            "follow_up": ["术后复查 CEA。"],
        },
        verdict="REJECTED",
        feedback=raw_feedback,
        references=[],
        citation_report={"coverage_score": 65, "missing_claims": ["no_direct_references"], "needs_more_sources": True},
        evaluation_report={"verdict": "FAIL", "feedback": "no_direct_references"},
    )

    assert "需要补充 MMR/MSI 检测。" not in text
    assert "HUMAN_REVIEW_REQUIRED" not in text
    assert '"verdict"' not in text
    assert "<think>" not in text
    assert "### 质控提示" in text
    assert "引用依据不足" in text
    assert "LLM-Judge" not in text


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
