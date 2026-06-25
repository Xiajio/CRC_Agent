from __future__ import annotations

from src.nodes.router import dynamic_router, route_after_assessment, route_after_intent
from src.policies.routing_policy import decide_after_assessment
from src.policies.turn_facts import build_turn_facts, derive_routing_flags
from src.state import CRCAgentState, PatientProfile


def _state(findings: dict[str, object], **kwargs: object) -> CRCAgentState:
    return CRCAgentState(messages=[], findings=findings, **kwargs)


def test_treatment_decision_requires_complete_case_and_hard_inquiry() -> None:
    facts = build_turn_facts(
        _state(
            {
                "user_intent": "treatment_decision",
                "clinical_task_profile": {
                    "requires_complete_case": False,
                    "missing_info_policy": "soft_context",
                },
            }
        )
    )

    assert facts.requires_complete_case is True
    assert facts.response_mode == "clinical_answer"
    assert facts.missing_info_policy == "hard_inquiry"
    assert facts.needs_full_decision is True
    assert route_after_intent(_state({"user_intent": "treatment_decision"})) == "clinical_entry_resolver"


def test_default_clinical_assessment_is_soft_non_decision() -> None:
    state = _state(
        {"user_intent": "clinical_assessment"},
        pathology_confirmed=True,
    )

    facts = build_turn_facts(state)
    flags = derive_routing_flags(facts)

    assert facts.requires_complete_case is False
    assert facts.missing_info_policy == "soft_context"
    assert facts.needs_full_decision is False
    assert flags.can_fast_pass_decision is False
    assert route_after_intent(state) == "chat_main"
    assert dynamic_router(state) == "chat_main"


def test_clinical_assessment_can_opt_into_hard_case_completion() -> None:
    state = _state(
        {
            "user_intent": "clinical_assessment",
            "clinical_task_profile": {
                "requires_complete_case": True,
                "missing_info_policy": "hard_inquiry",
            },
        }
    )

    facts = build_turn_facts(state)

    assert facts.requires_complete_case is True
    assert facts.missing_info_policy == "hard_inquiry"
    assert route_after_intent(state) == "clinical_entry_resolver"
    assert dynamic_router(state) == "clinical_entry_resolver"


def test_dynamic_router_soft_clinical_active_inquiry_does_not_reenter_assessment() -> None:
    state = _state(
        {
            "user_intent": "clinical_assessment",
            "active_inquiry": True,
            "missing_info_policy": "soft_context",
        }
    )

    assert dynamic_router(state) == "chat_main"


def test_after_assessment_soft_clinical_does_not_fast_pass_to_decision() -> None:
    state = _state(
        {"user_intent": "clinical_assessment"},
        patient_profile=PatientProfile(
            pathology_confirmed=True,
            tnm_staging={"cT": "cT3", "cN": "cN1", "cM": "cM0"},
            is_locked=True,
        ),
    )

    facts = build_turn_facts(state)
    flags = derive_routing_flags(facts)

    assert flags.can_fast_pass_decision is False
    assert decide_after_assessment(facts, flags).target == "chat_main"
    assert route_after_assessment(state) == "chat_main"


def test_after_assessment_treatment_decision_fast_pass_still_routes_to_decision() -> None:
    state = _state(
        {"user_intent": "treatment_decision"},
        patient_profile=PatientProfile(
            pathology_confirmed=True,
            tnm_staging={"cT": "cT3", "cN": "cN1", "cM": "cM0"},
            is_locked=True,
        ),
    )

    facts = build_turn_facts(state)
    flags = derive_routing_flags(facts)

    assert facts.requires_complete_case is True
    assert flags.can_fast_pass_decision is True
    assert route_after_assessment(state) == "decision"
