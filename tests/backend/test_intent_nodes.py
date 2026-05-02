from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.nodes.intent_nodes import node_intent_classifier
from src.state import CRCAgentState


class _StructuredFailureChain:
    def __init__(self, owner: "_MiniMaxRawFirstProbeModel") -> None:
        self._owner = owner

    def bind(self, **_kwargs):
        def _unexpected_invoke(_payload):
            self._owner.structured_invocations += 1
            raise AssertionError("minimax-compatible intent routing should not invoke structured output first.")

        return RunnableLambda(_unexpected_invoke)


class _MiniMaxRawFirstProbeModel:
    def __init__(self) -> None:
        self.model_name = "MiniMax-M2.7-highspeed"
        self.openai_api_base = "https://api.minimaxi.com/v1"
        self.structured_invocations = 0
        self.raw_invocations = 0

    def with_structured_output(self, _schema):
        return _StructuredFailureChain(self)

    def bind(self, **_kwargs):
        def _raw_invoke(_payload):
            self.raw_invocations += 1
            return AIMessage(
                content=(
                    '{"category":"knowledge_query","sub_tasks":null,'
                    '"requires_context":false,"correction_suggestion":null,'
                    '"reasoning":"raw-first"}'
                )
            )

        return RunnableLambda(_raw_invoke)


class _UnusedIntentModel:
    def __init__(self) -> None:
        self.structured_invocations = 0
        self.raw_invocations = 0

    def with_structured_output(self, _schema):
        class _UnexpectedStructuredChain:
            def __init__(self, owner: "_UnusedIntentModel") -> None:
                self._owner = owner

            def bind(self, **_kwargs):
                def _unexpected_invoke(_payload):
                    self._owner.structured_invocations += 1
                    raise AssertionError("meta intent fast path should not invoke structured output.")

                return RunnableLambda(_unexpected_invoke)

        return _UnexpectedStructuredChain(self)

    def bind(self, **_kwargs):
        def _unexpected_invoke(_payload):
            self.raw_invocations += 1
            raise AssertionError("meta intent fast path should not invoke raw model fallback.")

        return RunnableLambda(_unexpected_invoke)


def test_intent_classifier_uses_raw_first_for_minimax_compatible_provider() -> None:
    model = _MiniMaxRawFirstProbeModel()
    runnable = node_intent_classifier(model=model, streaming=False, show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="术后病理怎么看")],
            findings={},
        )
    )

    assert model.structured_invocations == 0
    assert model.raw_invocations == 1
    assert result["findings"]["user_intent"] == "knowledge_query"


def test_intent_classifier_fast_paths_meta_capability_queries_without_model_calls() -> None:
    model = _UnusedIntentModel()
    runnable = node_intent_classifier(model=model, streaming=False, show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content="你有什么用")],
            findings={},
        )
    )

    assert model.structured_invocations == 0
    assert model.raw_invocations == 0
    assert result["findings"]["user_intent"] == "general_chat"
    assert result["clinical_stage"] == "Intent"
    assert result["error"] is None
