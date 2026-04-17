from __future__ import annotations

from ..policies.routing_policy import decide_after_assessment, decide_after_intent, decide_dynamic
from ..policies.turn_facts import build_turn_facts, derive_routing_flags
from ..state import CRCAgentState


def _decision_target(state: CRCAgentState, decision_builder) -> str:
    facts = build_turn_facts(state)
    flags = derive_routing_flags(facts)
    return decision_builder(facts, flags).target


def route_after_intent(state: CRCAgentState) -> str:
    return _decision_target(state, decide_after_intent)


def dynamic_router(state: CRCAgentState) -> str:
    return _decision_target(state, decide_dynamic)


def route_after_assessment(state: CRCAgentState) -> str:
    return _decision_target(state, decide_after_assessment)


def route_after_clinical_entry(state: CRCAgentState) -> str:
    from .triage_nodes import route_after_clinical_entry as _triage_route_after_clinical_entry

    return _triage_route_after_clinical_entry(state)


__all__ = [
    "route_after_intent",
    "dynamic_router",
    "route_after_assessment",
    "route_after_clinical_entry",
]
