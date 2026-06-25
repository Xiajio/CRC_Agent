from __future__ import annotations

from langchain_core.messages import HumanMessage

from src.nodes import planner
from src.state import CRCAgentState


def _state(findings: dict[str, object]) -> CRCAgentState:
    return CRCAgentState(messages=[HumanMessage(content="Please review this case.")], findings=findings)


def test_soft_clinical_assessment_does_not_prefill_missing_case_context() -> None:
    state = _state({"user_intent": "clinical_assessment"})

    assert planner._detect_missing_context(state) == {}


def test_treatment_decision_prefills_missing_case_context() -> None:
    state = _state({"user_intent": "treatment_decision"})

    assert planner._detect_missing_context(state) == {
        "pathology_confirmed": "case_database_query",
        "tnm_staging": "case_database_query",
    }


def test_explicit_complete_case_clinical_assessment_prefills_missing_context() -> None:
    state = _state(
        {
            "user_intent": "clinical_assessment",
            "clinical_task_profile": {
                "requires_complete_case": True,
                "missing_info_policy": "hard_inquiry",
            },
        }
    )

    assert planner._detect_missing_context(state) == {
        "pathology_confirmed": "case_database_query",
        "tnm_staging": "case_database_query",
    }
