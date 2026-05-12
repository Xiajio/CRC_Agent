from __future__ import annotations

from src.graph_builder import (
    route_after_assessment,
    route_after_patient_assessment,
    route_after_patient_chat_main,
)
from src.state import CRCAgentState


def _make_state(
    *,
    findings: dict[str, object] | None = None,
    encounter_track: str | None = None,
    missing_critical_data: list[str] | None = None,
) -> CRCAgentState:
    return CRCAgentState(
        messages=[],
        findings=findings or {},
        encounter_track=encounter_track,
        missing_critical_data=missing_critical_data or [],
    )


def test_patient_missing_data_followup_answer_reenters_assessment() -> None:
    missing_state = _make_state(
        findings={"user_intent": "clinical_assessment"},
        missing_critical_data=["gender", "age"],
    )
    asking_state = _make_state(
        findings={
            "user_intent": "clinical_assessment",
            "active_inquiry": True,
            "active_field": "gender",
        },
        missing_critical_data=["gender", "age"],
    )
    answered_last_followup_state = _make_state(
        findings={
            "user_intent": "clinical_assessment",
            "active_inquiry": False,
            "active_field": None,
        },
        missing_critical_data=["gender", "age"],
    )

    assert route_after_assessment(missing_state) == "chat_main"
    assert route_after_patient_assessment(missing_state) == "chat_main"
    assert route_after_patient_chat_main(asking_state) == "end"
    assert route_after_patient_chat_main(answered_last_followup_state) == "assessment"


def test_patient_assessment_preserves_outpatient_triage_end_turn() -> None:
    state = _make_state(encounter_track="outpatient_triage")

    assert route_after_assessment(state) == "end_turn"
    assert route_after_patient_assessment(state) == "end"
