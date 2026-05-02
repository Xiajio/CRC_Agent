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
