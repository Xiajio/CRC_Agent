from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.nodes.general_nodes import node_general_chat
from src.state import CRCAgentState, PlanStep


def _completed_plan() -> list[PlanStep]:
    return [
        PlanStep(
            id="step_1",
            description="Collect the requested context.",
            tool_needed="search",
            status="completed",
        )
    ]


@pytest.mark.parametrize(
    ("name", "state_factory", "prepare_case"),
    [
        (
            "base",
            lambda: CRCAgentState(
                messages=[HumanMessage(content="请介绍一下你能提供哪些帮助？")],
                findings={"user_intent": "general_chat"},
            ),
            None,
        ),
        (
            "redirect",
            lambda: CRCAgentState(
                messages=[HumanMessage(content="我们聊点别的吧")],
                findings={"user_intent": "off_topic_redirect"},
            ),
            None,
        ),
        (
            "plan_followup",
            lambda: CRCAgentState(
                messages=[HumanMessage(content="继续解释上一步的计划")],
                findings={"user_intent": "general_chat", "plan_followup": True},
                decision_json={"summary": "keep going"},
            ),
            None,
        ),
        (
            "completed_plan_info_only",
            lambda: CRCAgentState(
                messages=[HumanMessage(content="把病例库里现成的信息整理一下")],
                findings={"user_intent": "case_database_query"},
                current_plan=_completed_plan(),
            ),
            None,
        ),
        (
            "completed_plan_simple_fact",
            lambda: CRCAgentState(
                messages=[HumanMessage(content="这个检查是什么")],
                findings={"user_intent": "general_chat"},
                current_plan=_completed_plan(),
            ),
            None,
        ),
        (
            "completed_plan_synthesis",
            lambda: CRCAgentState(
                messages=[HumanMessage(content="请综合前面收集的信息，给出完整说明")],
                findings={"user_intent": "general_chat"},
                current_plan=_completed_plan(),
            ),
            lambda monkeypatch: monkeypatch.setattr(
                "src.nodes.general_nodes._is_simple_fact_question",
                lambda _text: False,
            ),
        ),
    ],
)
def test_node_general_chat_passes_general_chat_node_name_to_streaming_invocations(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    state_factory: Callable[[], CRCAgentState],
    prepare_case: Callable[[pytest.MonkeyPatch], None] | None,
) -> None:
    del name
    captured_node_names: list[str | None] = []

    def fake_invoke_with_streaming(
        chain,
        context,
        streaming: bool = False,
        show_thinking: bool = True,
        node_name: str | None = None,
    ) -> AIMessage:
        del chain, context, streaming, show_thinking
        captured_node_names.append(node_name)
        return AIMessage(content="stub answer")

    monkeypatch.setattr(
        "src.nodes.general_nodes._invoke_with_streaming",
        fake_invoke_with_streaming,
    )
    if prepare_case is not None:
        prepare_case(monkeypatch)

    runnable = node_general_chat(
        model=RunnableLambda(lambda _input: AIMessage(content="unused")),
        streaming=True,
        show_thinking=False,
    )

    result = runnable(state_factory())

    assert captured_node_names == ["general_chat"]
    assert result["messages"][0].content == "stub answer"


def test_node_general_chat_uses_gaps_prompt_and_clears_stale_inquiry(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_contexts: list[dict[str, object]] = []

    def fake_invoke_with_streaming(
        chain,
        context,
        streaming: bool = False,
        show_thinking: bool = True,
        node_name: str | None = None,
    ) -> AIMessage:
        del chain, streaming, show_thinking
        assert node_name == "general_chat"
        captured_contexts.append(context)
        return AIMessage(content="general gaps answer")

    monkeypatch.setattr(
        "src.nodes.general_nodes._invoke_with_streaming",
        fake_invoke_with_streaming,
    )

    runnable = node_general_chat(
        model=RunnableLambda(lambda _input: AIMessage(content="unused")),
        streaming=True,
        show_thinking=False,
    )

    result = runnable(
        CRCAgentState(
            messages=[
                HumanMessage(content="Pathology shows colon adenocarcinoma."),
                AIMessage(content="Please provide TNM staging."),
                HumanMessage(content="Please draft a summary with what we have."),
            ],
            summary_memory="Known pathology: colon adenocarcinoma.",
            findings={
                "response_mode": "case_summary_with_gaps",
                "information_gaps": ["TNM Staging", "MMR/MSI Status"],
                "active_inquiry": True,
                "active_field": "tnm",
                "inquiry_message": "Please provide TNM staging.",
                "inquiry_type": "tnm_required",
            },
            missing_critical_data=["TNM Staging"],
        )
    )

    assert result["messages"][0].content == "general gaps answer"
    assert result["missing_critical_data"] == []
    assert result["findings"]["active_inquiry"] is False
    assert result["findings"]["active_field"] is None
    assert result["findings"]["inquiry_message"] is None
    assert result["findings"]["inquiry_type"] is None
    assert captured_contexts[0]["user_question"] == "Please draft a summary with what we have."
    assert captured_contexts[0]["information_gaps"] == ["TNM Staging", "MMR/MSI Status"]
    assert captured_contexts[0]["summary_memory"] == "Known pathology: colon adenocarcinoma."
    assert "pinned_context" in captured_contexts[0]
    assert "recent_conversation" in captured_contexts[0]
    assert "Pathology shows colon adenocarcinoma" in str(captured_contexts[0]["recent_conversation"])


def test_node_general_chat_clears_gap_state_before_later_general_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_modes: list[str] = []

    def fake_invoke_with_streaming(
        chain,
        context,
        streaming: bool = False,
        show_thinking: bool = True,
        node_name: str | None = None,
    ) -> AIMessage:
        del chain, streaming, show_thinking
        assert node_name == "general_chat"
        prompt_modes.append("gaps" if "information_gaps" in context else "base")
        return AIMessage(content=f"{prompt_modes[-1]} answer")

    monkeypatch.setattr(
        "src.nodes.general_nodes._invoke_with_streaming",
        fake_invoke_with_streaming,
    )

    runnable = node_general_chat(
        model=RunnableLambda(lambda _input: AIMessage(content="unused")),
        streaming=True,
        show_thinking=False,
    )
    first_state = CRCAgentState(
        messages=[HumanMessage(content="Please draft a summary with current gaps.")],
        findings={
            "user_intent": "general_chat",
            "response_mode": "case_summary_with_gaps",
            "information_gaps": ["TNM Staging"],
            "missing_info_policy": "answer_with_gaps",
            "active_inquiry": True,
            "active_field": "tnm",
            "inquiry_message": "Please provide TNM staging.",
            "inquiry_type": "tnm_required",
        },
        missing_critical_data=["TNM Staging"],
    )

    first_result = runnable(first_state)
    merged_findings = {**first_state.findings, **first_result["findings"]}
    second_state = CRCAgentState(
        messages=[HumanMessage(content="Tell me what you can help with today.")],
        findings=merged_findings,
    )

    second_result = runnable(second_state)

    assert first_result["findings"]["response_mode"] == "general_answer"
    assert first_result["findings"]["information_gaps"] == []
    assert first_result["findings"]["missing_info_policy"] == "none"
    assert second_result["messages"][0].content == "base answer"
    assert prompt_modes == ["gaps", "base"]
