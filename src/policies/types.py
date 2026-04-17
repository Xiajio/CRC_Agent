"""Policy-layer types used by routing and review flows."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TurnFacts(BaseModel):
    """Stable, normalized facts captured for a single assistant turn."""

    user_intent: str = ""
    sub_tasks: tuple[str, ...] = ()
    multi_task_mode: bool = False
    has_plan: bool = False
    pending_step_tool: str = ""
    pending_step_target: str = ""
    has_parallel_group: bool = False
    active_inquiry: bool = False
    active_field: str = ""
    pending_patient_data: str = ""
    pending_patient_id: str = ""
    encounter_track: str = ""
    clinical_stage: str = ""
    triage_switch_prompt_active: bool = False
    triage_explicit_switch_request: bool = False
    has_missing_critical_data: bool = False
    missing_critical_data_count: int = 0
    pathology_confirmed: bool = False
    stage_complete: bool = False
    tumor_location: str = ""
    patient_profile_locked: bool = False
    needs_full_decision: bool = False
    decision_exists: bool = False
    decision_strategy: str = "full"
    iteration_count: int = 0
    rejection_count: int = 0
    evaluation_retry_count: int = 0
    critic_verdict: str = ""
    citation_coverage_score: int = 0
    citation_needs_more_sources: bool = False
    stable_guideline_rag_support: bool = False
    evaluator_verdict: str = ""
    evaluator_scores: dict[str, int] = Field(default_factory=dict)
    evaluator_actionable_retry: bool = False
    evaluator_degraded: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class DerivedRoutingFlags(BaseModel):
    """Stable, non-decisional booleans derived from turn facts."""

    is_degraded: bool = False
    has_guideline_support: bool = False
    has_inline_citations: bool = False
    can_fast_pass_decision: bool = False
    should_shortcut_to_general_chat: bool = False
    should_end_turn_for_inquiry: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")


class DegradedSignal(BaseModel):
    """Metadata for a fallback or degraded review signal."""

    is_degraded: bool = False
    reason: str = ""
    fallback_value: Any = None

    model_config = ConfigDict(frozen=True, extra="forbid")


class RouteDecision(BaseModel):
    """Routing outcome for the next policy step."""

    target: str = ""
    rule_name: str = ""
    should_retry: bool = False
    should_review: bool = False
    degraded: DegradedSignal | None = None
    notes: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def route(self) -> str:
        return self.target


class ReviewSignal(BaseModel):
    """Structured critic/evaluator signal consumed by the review policy."""

    source: str = ""
    verdict: str = ""
    retryable: bool = False
    reasons: tuple[str, ...] = ()
    degraded: DegradedSignal = Field(default_factory=DegradedSignal)
    feedback: str = ""
    scores: dict[str, int] = Field(default_factory=dict)
    coverage_score: int = 0
    inline_anchor_count: int = 0
    notes: str = ""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewDecision(BaseModel):
    """Policy decision for post-review control flow."""

    action: str = "finalize"
    rule_name: str = ""
    rationale: str = ""
    review_signal: ReviewSignal | None = None

    model_config = ConfigDict(frozen=True, extra="forbid")

    @property
    def route(self) -> str:
        if self.action == "retry_decision":
            return "decision"
        return "finalize"


__all__ = [
    "TurnFacts",
    "DerivedRoutingFlags",
    "DegradedSignal",
    "RouteDecision",
    "ReviewSignal",
    "ReviewDecision",
]