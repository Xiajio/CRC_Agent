from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.nodes.intent_nodes import _META_CAPABILITY_QUERIES, _clinical_task_profile_from_text, node_intent_classifier
from src.nodes.router import route_after_intent
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


class _ParseFailureIntentModel:
    def with_structured_output(self, _schema):
        class _FailingStructuredChain:
            def bind(self, **_kwargs):
                def _raise(_payload):
                    raise ValueError("structured parse failed")

                return RunnableLambda(_raise)

        return _FailingStructuredChain()

    def bind(self, **_kwargs):
        def _raw_invoke(_payload):
            return AIMessage(content="not json")

        return RunnableLambda(_raw_invoke)


def test_clinical_task_profile_marks_report_draft_as_answer_with_gaps() -> None:
    draft_profile = _clinical_task_profile_from_text(
        "\u8bf7\u751f\u6210\u75c5\u4f8b\u6458\u8981\u8349\u7a3f\uff0c\u8d44\u6599\u7f3a\u5931\u4e5f\u5148\u8f93\u51fa\u3002",
        "general_chat",
    )
    summary_profile = _clinical_task_profile_from_text(
        "\u5e2e\u6211\u6574\u7406\u4e00\u4e0b\u75c5\u4f8b\u6458\u8981",
        "general_chat",
    )

    assert draft_profile["task_type"] == "document_draft"
    assert draft_profile["requires_complete_case"] is False
    assert draft_profile["missing_info_policy"] == "answer_with_gaps"
    assert draft_profile["response_mode"] != "decision_blocked"
    assert draft_profile["reason"].startswith("deterministic_rule:")
    assert summary_profile["task_type"] == "case_summary"
    assert summary_profile["requires_complete_case"] is False
    assert summary_profile["missing_info_policy"] == "answer_with_gaps"
    assert summary_profile["response_mode"] != "decision_blocked"


def test_clinical_task_profile_marks_treatment_next_step_and_staging_as_hard_inquiry() -> None:
    treatment = _clinical_task_profile_from_text("\u4e0b\u4e00\u6b65\u6cbb\u7597\u65b9\u6848\u600e\u4e48\u5b9a\uff1f", "treatment_decision")
    staging = _clinical_task_profile_from_text("\u8bf7\u8bc4\u4f30\u8fd9\u4e2a\u60a3\u8005\u7684\u5206\u671f", "clinical_assessment")

    assert treatment["requires_complete_case"] is True
    assert treatment["missing_info_policy"] == "hard_inquiry"
    assert treatment["response_mode"] == "decision_blocked"
    assert staging["task_type"] == "staging_assessment"
    assert staging["requires_complete_case"] is True
    assert staging["missing_info_policy"] == "hard_inquiry"


def test_clinical_task_profile_missing_info_guidance_wins_over_next_step_marker() -> None:
    profile = _clinical_task_profile_from_text("\u4e0b\u4e00\u6b65\u8fd8\u9700\u8981\u54ea\u4e9b\u8d44\u6599\uff1f", "general_chat")

    assert profile["task_type"] == "missing_info_guidance"
    assert profile["requires_complete_case"] is False
    assert profile["missing_info_policy"] == "guide_collection"
    assert profile["response_mode"] == "guided_collection"


def test_clinical_task_profile_explanation_wins_over_staging_marker() -> None:
    profile = _clinical_task_profile_from_text("\u5206\u671f\u662f\u4ec0\u4e48\u610f\u601d\uff1f", "general_chat")

    assert profile["task_type"] == "explain_existing_info"
    assert profile["requires_complete_case"] is False
    assert profile["missing_info_policy"] == "answer_with_gaps"
    assert profile["response_mode"] == "partial_explanation"


def test_clinical_task_profile_marks_missing_info_guidance_as_guided_collection() -> None:
    profile = _clinical_task_profile_from_text("\u8fd8\u7f3a\u4ec0\u4e48\u8d44\u6599\uff1f", "general_chat")

    assert profile["task_type"] == "missing_info_guidance"
    assert profile["requires_complete_case"] is False
    assert profile["missing_info_policy"] == "guide_collection"
    assert profile["response_mode"] == "guided_collection"


def test_clinical_task_profile_marks_explanation_as_partial_without_complete_case() -> None:
    profile = _clinical_task_profile_from_text("\u5e2e\u6211\u770b\u770b\u8fd9\u4e2a\u75c5\u7406\u62a5\u544a\u662f\u4ec0\u4e48\u610f\u601d", "general_chat")

    assert profile["task_type"] == "explain_existing_info"
    assert profile["requires_complete_case"] is False
    assert profile["missing_info_policy"] == "answer_with_gaps"
    assert profile["response_mode"] == "partial_explanation"


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


@pytest.mark.parametrize(
        "user_text",
        [
            "hello",
            next(iter(_META_CAPABILITY_QUERIES)),
            "\u8c22\u8c22",
        "show chat history",
        (
            "\u8bf7\u751f\u6210\u75c5\u4f8b\u6458\u8981\u8349\u7a3f\u3002"
            "\u5373\u4f7f\u8d44\u6599\u7f3a\u5931\uff0c\u4e5f\u5fc5\u987b\u5148\u8f93\u51fa"
            "\u75c5\u4f8b/\u62a5\u544a\u8349\u7a3f\u6a21\u677f\u3002"
        ),
    ],
)
def test_general_fast_paths_apply_top_level_runtime_resets(user_text: str) -> None:
    model = _UnusedIntentModel()
    runnable = node_intent_classifier(model=model, streaming=False, show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content=user_text)],
            encounter_track="crc_clinical",
            clinical_entry_reason="assessment",
            entry_explanation_shown=True,
            triage_risk_level="medium",
            triage_disposition="clinic",
            triage_suggested_tests=["cbc"],
            triage_summary="stale summary",
            symptom_snapshot={"pain": True},
            findings={"encounter_track": "crc_clinical"},
        )
    )

    assert result["encounter_track"] is None
    assert result["clinical_entry_reason"] is None
    assert result["entry_explanation_shown"] is False
    assert result["triage_risk_level"] is None
    assert result["triage_disposition"] is None
    assert result["triage_suggested_tests"] == []
    assert result["triage_summary"] is None
    assert result["symptom_snapshot"] == {}
    assert model.structured_invocations == 0
    assert model.raw_invocations == 0


def test_intent_classifier_routes_report_draft_requests_to_general_chat_without_model_calls() -> None:
    model = _UnusedIntentModel()
    runnable = node_intent_classifier(model=model, streaming=False, show_thinking=False)
    report_prompt = (
        "\u8bf7\u751f\u6210\u75c5\u4f8b\u6458\u8981\u8349\u7a3f\u3002"
        "\u5373\u4f7f\u8d44\u6599\u7f3a\u5931\uff0c\u4e5f\u5fc5\u987b\u5148\u8f93\u51fa"
        "\u75c5\u4f8b/\u62a5\u544a\u8349\u7a3f\u6a21\u677f\u3002"
    )

    result = runnable(
        CRCAgentState(
            messages=[HumanMessage(content=report_prompt)],
            registry_patient_id=128,
            findings={
                "encounter_track": "crc_clinical",
                "active_inquiry": True,
                "active_field": "pathology",
                "inquiry_message": "\u8bf7\u8865\u5145\u75c5\u7406\u62a5\u544a",
                "inquiry_type": "pathology_required",
            },
        )
    )

    assert model.structured_invocations == 0
    assert model.raw_invocations == 0
    assert result["findings"]["user_intent"] == "general_chat"
    assert result["findings"]["active_inquiry"] is False
    assert result["findings"]["active_field"] is None
    assert result["findings"]["inquiry_message"] is None
    assert result["findings"]["inquiry_type"] is None
    assert result["findings"]["clinical_task_profile"]["task_type"] == "document_draft"
    assert result["findings"]["requires_complete_case"] is False
    assert result["findings"]["missing_info_policy"] == "answer_with_gaps"
    assert result["findings"]["response_mode"] != "decision_blocked"
    assert result["clinical_stage"] == "Intent"
    assert result["error"] is None


def test_intent_classifier_clears_tnm_inquiry_when_report_draft_interrupts() -> None:
    model = _UnusedIntentModel()
    runnable = node_intent_classifier(model=model, streaming=False, show_thinking=False)

    result = runnable(
        CRCAgentState(
            messages=[
                HumanMessage(content="Pathology confirms colon adenocarcinoma."),
                AIMessage(content="Please provide TNM staging before treatment planning."),
                HumanMessage(content="Please draft a case summary with the information we have."),
            ],
            registry_patient_id=128,
            missing_critical_data=["TNM Staging"],
            findings={
                "encounter_track": "crc_clinical",
                "active_inquiry": True,
                "active_field": "tnm",
                "inquiry_message": "Please provide TNM staging before treatment planning.",
                "inquiry_type": "tnm_required",
                "missing_info_policy": "hard_inquiry",
                "response_mode": "decision_blocked",
            },
        )
    )

    merged_state = CRCAgentState(
        messages=[],
        findings=result["findings"],
        missing_critical_data=["TNM Staging"],
    )

    assert model.structured_invocations == 0
    assert model.raw_invocations == 0
    assert result["findings"]["user_intent"] == "general_chat"
    assert result["findings"]["active_inquiry"] is False
    assert result["findings"]["active_field"] is None
    assert result["findings"]["inquiry_message"] is None
    assert result["findings"]["inquiry_type"] is None
    assert result["findings"]["clinical_task_profile"]["task_type"] == "document_draft"
    assert result["findings"]["requires_complete_case"] is False
    assert result["findings"]["missing_info_policy"] == "answer_with_gaps"
    assert result["findings"]["response_mode"] == "case_summary_with_gaps"
    assert result["missing_critical_data"] == []
    assert route_after_intent(merged_state) == "general_chat"


def test_intent_classifier_parse_failure_long_text_defaults_to_soft_general_profile() -> None:
    model = _ParseFailureIntentModel()
    runnable = node_intent_classifier(model=model, streaming=False, show_thinking=False)
    long_text = (
        "\u8fd9\u91cc\u662f\u4e00\u6bb5\u5f88\u957f\u7684\u75c5\u7406\u62a5\u544a\u548c\u68c0\u67e5\u7ed3\u679c\u6587\u672c\uff0c"
        "\u8bf7\u5e2e\u6211\u770b\u770b\u91cc\u9762\u8bf4\u7684\u662f\u4ec0\u4e48\u610f\u601d\uff0c"
        "\u53ef\u4ee5\u5148\u505a\u666e\u901a\u89e3\u91ca\u548c\u8981\u70b9\u6574\u7406\u3002"
    )

    result = runnable(CRCAgentState(messages=[HumanMessage(content=long_text)], findings={}))

    assert result["findings"]["user_intent"] != "clinical_assessment"
    assert result["findings"]["clinical_task_profile"]["task_type"] == "explain_existing_info"
    assert result["findings"]["requires_complete_case"] is False
    assert result["findings"]["response_mode"] == "partial_explanation"
