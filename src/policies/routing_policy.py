"""Pure routing decisions derived from normalized turn facts."""

from __future__ import annotations

from .types import DerivedRoutingFlags, RouteDecision, TurnFacts


def _decision(target: str, rule_name: str, notes: str = "") -> RouteDecision:
    return RouteDecision(target=target, rule_name=rule_name, notes=notes)


def decide_after_intent(facts: TurnFacts, flags: DerivedRoutingFlags) -> RouteDecision:
    """Route immediately after intent classification without reading raw state."""
    if facts.encounter_track == "outpatient_triage" and facts.active_inquiry:
        if not facts.triage_switch_prompt_active or not facts.triage_explicit_switch_request:
            return _decision("clinical_entry_resolver", "outpatient_triage_active_inquiry")

    if flags.should_shortcut_to_general_chat:
        return _decision("general_chat", f"intent_{facts.user_intent or 'general_chat'}")

    if not facts.multi_task_mode and facts.user_intent == "knowledge_query":
        return _decision("knowledge", "intent_knowledge_query")

    if not facts.multi_task_mode and facts.user_intent == "clinical_assessment":
        return _decision("clinical_entry_resolver", "intent_clinical_assessment")

    if not facts.multi_task_mode and facts.user_intent in {"case_database_query", "imaging_query"}:
        return _decision("case_database", f"intent_{facts.user_intent}")

    if facts.user_intent in {"clinical_assessment", "treatment_decision"}:
        return _decision("clinical_entry_resolver", f"intent_{facts.user_intent}")

    return _decision("planner", "fallback_planner")

def decide_dynamic(facts: TurnFacts, flags: DerivedRoutingFlags) -> RouteDecision:
    """Route during plan execution while preserving planner-first semantics."""
    if facts.has_plan and facts.has_parallel_group:
        return _decision("parallel_subagents", "plan_parallel_group")

    if facts.has_plan and facts.pending_step_target:
        return _decision(facts.pending_step_target, "plan_pending_step_target")

    if facts.active_field or facts.pending_patient_data or facts.pending_patient_id:
        return _decision("chat_main", "active_field_collection")

    if not facts.multi_task_mode and facts.user_intent == "knowledge_query":
        return _decision("knowledge", "intent_knowledge_query")

    if flags.should_shortcut_to_general_chat:
        return _decision("general_chat", f"intent_{facts.user_intent or 'general_chat'}")

    if facts.active_inquiry and facts.user_intent in {"clinical_assessment", "treatment_decision"}:
        return _decision("assessment", "active_inquiry_assessment")

    if facts.user_intent == "imaging_analysis":
        return _decision("rad_agent", "intent_imaging_analysis")

    if facts.user_intent == "pathology_analysis":
        return _decision("path_agent", "intent_pathology_analysis")

    if facts.user_intent in {"case_database_query", "imaging_query"}:
        return _decision("case_database", f"intent_{facts.user_intent}")

    if facts.user_intent == "treatment_decision":
        if flags.can_fast_pass_decision:
            return _decision("decision", "intent_treatment_decision_fast_pass")
        return _decision("clinical_entry_resolver", "intent_treatment_decision")

    if facts.user_intent == "clinical_assessment":
        return _decision("clinical_entry_resolver", "intent_clinical_assessment")

    return _decision("assessment", "fallback_assessment")


def decide_after_assessment(facts: TurnFacts, flags: DerivedRoutingFlags) -> RouteDecision:
    """Route after assessment, including inquiry interruption and fast-pass behavior."""
    if facts.encounter_track == "outpatient_triage":
        return _decision("end_turn", "outpatient_triage_end_turn")

    if facts.has_missing_critical_data:
        if flags.should_end_turn_for_inquiry:
            return _decision("end_turn", "missing_critical_data_active_inquiry")
        return _decision("chat_main", "missing_critical_data")

    if facts.clinical_stage == "Inquiry_Pending":
        return _decision("end_turn", "clinical_stage_inquiry_pending")

    if flags.can_fast_pass_decision:
        return _decision("decision", "fast_pass_decision")

    return _decision("diagnosis", "assessment_complete")


__all__ = [
    "decide_after_intent",
    "decide_dynamic",
    "decide_after_assessment",
]
