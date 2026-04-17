from __future__ import annotations

from src.graph_builder import (
    route_after_assessment,
    route_after_doctor_clinical_entry,
    route_after_doctor_intent,
    route_after_doctor_planner,
    route_after_doctor_post_assessment,
    route_after_intent,
    route_after_patient_intent,
)
from src.nodes.router import dynamic_router
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


def test_doctor_intent_remaps_clinical_entry_to_assessment() -> None:
    state = _make_state(findings={"user_intent": "clinical_assessment"})

    assert route_after_intent(state) == "clinical_entry_resolver"
    assert route_after_doctor_intent(state) == "assessment"


def test_doctor_clinical_entry_can_never_route_to_outpatient_triage() -> None:
    state = _make_state(encounter_track="outpatient_triage")

    assert route_after_doctor_clinical_entry(state) == "assessment"


def test_patient_intent_keeps_case_database_blocking_intact() -> None:
    state = _make_state(findings={"user_intent": "case_database_query"})

    assert route_after_patient_intent(state) == "knowledge"


def test_doctor_planner_remaps_active_field_collection_from_chat_main_to_general_chat() -> None:
    state = _make_state(findings={"active_field": "age"})

    assert dynamic_router(state) == "chat_main"
    assert route_after_doctor_planner(state) == "general_chat"


def test_doctor_post_assessment_remaps_missing_data_followup_to_general_chat() -> None:
    state = _make_state(missing_critical_data=["age"])

    assert route_after_assessment(state) == "chat_main"
    assert route_after_doctor_post_assessment(state) == "general_chat"
