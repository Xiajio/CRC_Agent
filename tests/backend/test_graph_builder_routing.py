from __future__ import annotations

from src.graph_builder import (
    route_after_assessment,
    route_after_doctor_clinical_entry,
    route_after_doctor_intent,
    route_after_doctor_planner,
    route_after_doctor_followup,
    route_after_doctor_post_assessment,
    route_after_intent,
    route_after_patient_chat_main,
    route_after_patient_intent,
)
from src.nodes import decision_nodes
from src.policies.types import ReviewDecision
from src.nodes.router import dynamic_router
from src.state import CRCAgentState, PlanStep


def _make_state(
    *,
    findings: dict[str, object] | None = None,
    encounter_track: str | None = None,
    missing_critical_data: list[str] | None = None,
    current_plan: list[PlanStep] | None = None,
) -> CRCAgentState:
    return CRCAgentState(
        messages=[],
        findings=findings or {},
        encounter_track=encounter_track,
        missing_critical_data=missing_critical_data or [],
        current_plan=current_plan or [],
    )


def test_doctor_intent_preserves_soft_clinical_chat_route() -> None:
    state = _make_state(findings={"user_intent": "clinical_assessment"})

    assert route_after_intent(state) == "chat_main"
    assert route_after_doctor_intent(state) == "chat_main"


def test_doctor_intent_remaps_hard_clinical_entry_to_assessment() -> None:
    state = _make_state(
        findings={
            "user_intent": "clinical_assessment",
            "requires_complete_case": True,
            "missing_info_policy": "hard_inquiry",
        }
    )

    assert route_after_intent(state) == "clinical_entry_resolver"
    assert route_after_doctor_intent(state) == "assessment"


def test_doctor_clinical_entry_can_never_route_to_outpatient_triage() -> None:
    state = _make_state(encounter_track="outpatient_triage")

    assert route_after_doctor_clinical_entry(state) == "assessment"


def test_patient_intent_keeps_case_database_blocking_intact() -> None:
    state = _make_state(findings={"user_intent": "case_database_query"})

    assert route_after_patient_intent(state) == "knowledge"


def test_doctor_planner_keeps_active_field_collection_on_chat_main() -> None:
    state = _make_state(findings={"active_field": "age"})

    assert dynamic_router(state) == "chat_main"
    assert route_after_doctor_planner(state) == "chat_main"


def test_doctor_planner_routes_calculator_pending_steps_to_tool_executor() -> None:
    state = _make_state(
        current_plan=[
            PlanStep(
                id="step_1",
                description="Calculate the risk score.",
                tool_needed="calculator",
                status="pending",
            )
        ]
    )

    assert route_after_doctor_planner(state) == "tool_executor"


def test_doctor_planner_routes_tool_executor_pending_steps_to_tool_executor() -> None:
    state = _make_state(
        current_plan=[
            PlanStep(
                id="step_1",
                description="Run the tool executor work item.",
                tool_needed="tool_executor",
                status="pending",
            )
        ]
    )

    assert route_after_doctor_planner(state) == "tool_executor"


def test_doctor_followup_routes_calculator_pending_steps_to_tool_executor() -> None:
    state = _make_state(
        current_plan=[
            PlanStep(
                id="step_1",
                description="Calculate the risk score.",
                tool_needed="calculator",
                status="pending",
            )
        ]
    )

    assert route_after_doctor_followup(state) == "tool_executor"


def test_doctor_followup_routes_tool_executor_pending_steps_to_tool_executor() -> None:
    state = _make_state(
        current_plan=[
            PlanStep(
                id="step_1",
                description="Run the tool executor work item.",
                tool_needed="tool_executor",
                status="pending",
            )
        ]
    )

    assert route_after_doctor_followup(state) == "tool_executor"


def test_doctor_post_assessment_routes_missing_data_followup_to_chat_main() -> None:
    state = _make_state(missing_critical_data=["age"])

    assert route_after_assessment(state) == "chat_main"
    assert route_after_doctor_post_assessment(state) == "chat_main"


def test_doctor_followup_keeps_ask_user_guard_for_chat_intents() -> None:
    state = _make_state(
        findings={"user_intent": "general_chat"},
        current_plan=[
            PlanStep(
                id="step_1",
                description="Ask the user for clarification.",
                tool_needed="ask_user",
                status="pending",
            )
        ],
    )

    assert route_after_doctor_followup(state) == "general_chat"


def test_doctor_followup_routes_valid_tool_aliases_through_classifier() -> None:
    assert route_after_doctor_followup(
        _make_state(
            current_plan=[
                PlanStep(
                    id="step_1",
                    description="Query the case database.",
                    tool_needed="case_database_query",
                    status="pending",
                )
            ]
        )
    ) == "case_database"
    assert route_after_doctor_followup(
        _make_state(
            current_plan=[
                PlanStep(
                    id="step_1",
                    description="Run tumor screening.",
                    tool_needed="tumor_screening",
                    status="pending",
                )
            ]
        )
    ) == "rad_agent"
    assert route_after_doctor_followup(
        _make_state(
            current_plan=[
                PlanStep(
                    id="step_1",
                    description="Run pathology analysis.",
                    tool_needed="pathology_analysis",
                    status="pending",
                )
            ]
        )
    ) == "path_agent"


def test_patient_chat_main_only_reenters_assessment_when_clinical_context_can_continue() -> None:
    assert route_after_patient_chat_main(
        _make_state(findings={"active_inquiry": True})
    ) == "end"
    assert route_after_patient_chat_main(
        _make_state(findings={"user_intent": "general_chat"})
    ) == "end"
    assert route_after_patient_chat_main(
        _make_state(findings={"user_intent": "clinical_assessment", "active_field": None})
    ) == "end"
    assert route_after_patient_chat_main(
        _make_state(
            findings={
                "user_intent": "clinical_assessment",
                "active_field": None,
                "requires_complete_case": True,
            }
        )
    ) == "assessment"


def test_route_by_critic_v2_uses_policy_route_over_legacy_action(monkeypatch) -> None:
    state = _make_state(findings={"decision_strategy": "template_fast"})

    calls: list[str] = []

    def fake_legacy_impl(_state: CRCAgentState) -> str:
        calls.append("legacy")
        return "finalize"

    def fake_policy_decision(_state: CRCAgentState) -> ReviewDecision:
        calls.append("policy")
        return ReviewDecision(action="retry_decision", rule_name="unit_test", rationale="policy path")

    monkeypatch.setattr(decision_nodes, "_route_by_critic_legacy_impl", fake_legacy_impl)
    monkeypatch.setattr(decision_nodes, "_route_by_critic_policy_decision", fake_policy_decision)
    monkeypatch.setattr(decision_nodes, "_log_review_shadow", lambda *args, **kwargs: None)

    assert decision_nodes.route_by_critic_v2(state) == "decision"
    assert calls == ["legacy", "policy"]
