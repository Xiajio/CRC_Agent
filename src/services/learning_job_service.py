from __future__ import annotations

from typing import Callable

from backend.api.services.learning_job_store import LearningJobStore
from src.contracts.learning_job import (
    CandidatePatch,
    HarnessRequirement,
    HumanReviewRequirement,
    LearningJob,
    LearningSignal,
    make_candidate_patch_id,
    make_learning_job_id,
)


_STRONG_REASON_MARKERS = (
    "unsafe",
    "override",
    "failure",
    "regression",
    "evidence_delta",
    "evidence",
    "gap",
    "alert",
)
_STRONG_REASON_CODES = frozenset(
    {
        "unsafe_disposition",
        "citation_not_traceable",
        "evidence_conflict",
        "safety_signal",
        "harness_hard_fail",
        "missing_variable",
        "monitoring_alert",
    }
)
_NON_MUTATION_REASON = "shadow_learning_jobs_only"


class LearningJobValidationError(ValueError):
    """Raised when a learning job create request is invalid."""


class LearningJobService:
    def __init__(
        self,
        store: LearningJobStore,
        now: Callable[[], str] | None = None,
    ) -> None:
        self.store = store
        self._now = now or (lambda: "1970-01-01T00:00:00+00:00")

    def read_jobs(self) -> dict[str, object]:
        state = self.store.read_state()
        disabled_actions = _disabled_actions()
        return {
            "jobs": [job.to_dict() for job in state.jobs],
            "candidates": [candidate.to_dict() for candidate in state.candidates],
            "integrity": state.integrity,
            "disabled_actions": disabled_actions,
            "actions": _actions_compat(disabled_actions),
            "runtime": _runtime_metadata(),
        }

    def create_job(
        self,
        signals: list[LearningSignal],
        *,
        requested_by: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        _validate_create_inputs(
            signals,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
        )
        source_signal_ids = sorted(signal.signal_id for signal in signals)
        candidates = [
            self._build_candidate(signal, idempotency_key=idempotency_key)
            for signal in sorted(signals, key=lambda item: item.signal_id)
            if _is_strong_signal(signal)
        ]
        job = self._build_job(
            signals=signals,
            candidates=candidates,
            source_signal_ids=source_signal_ids,
            idempotency_key=idempotency_key,
        )

        self.store.write_job(job, candidates)
        disabled_actions = _disabled_actions()
        return {
            "job": job.to_dict(),
            "signals": [
                signal.to_dict()
                for signal in sorted(signals, key=lambda item: item.signal_id)
            ],
            "candidates": [candidate.to_dict() for candidate in candidates],
            "integrity": self.store.read_state().integrity,
            "disabled_actions": disabled_actions,
            "actions": _actions_compat(disabled_actions),
            "runtime": _runtime_metadata(),
        }

    def _build_candidate(
        self,
        signal: LearningSignal,
        *,
        idempotency_key: str,
    ) -> CandidatePatch:
        patch_type = signal.target_area
        patch_id = make_candidate_patch_id(
            patch_type,
            (
                f"{signal.signal_id}:{signal.signal_type}:"
                f"{signal.reason_code}:{idempotency_key}"
            ),
        )
        return CandidatePatch(
            patch_id=patch_id,
            patch_type=patch_type,
            target_ref={
                "kind": _target_kind(patch_type),
                "id": f"shadow_candidate_{patch_type}",
            },
            change_summary=(
                f"Shadow-only {patch_type} candidate from {signal.reason_code}."
            ),
            proposed_diff={
                "format": "structured_shadow_diff",
                "ops": [
                    {
                        "op": "propose_review",
                        "target_area": patch_type,
                        "reason_code": signal.reason_code,
                        "source_signal_id": signal.signal_id,
                    }
                ],
            },
            source_signal_ids=[signal.signal_id],
            status="candidate",
            applies_automatically=False,
        )

    def _build_job(
        self,
        *,
        signals: list[LearningSignal],
        candidates: list[CandidatePatch],
        source_signal_ids: list[str],
        idempotency_key: str,
    ) -> LearningJob:
        candidate_target_areas = {candidate.patch_type for candidate in candidates}
        job_type = _job_type_for_targets(candidate_target_areas)
        return LearningJob(
            job_id=make_learning_job_id(source_signal_ids, idempotency_key),
            job_type=job_type,
            status="shadow_only",
            created_at=self._now(),
            source_signal_ids=source_signal_ids,
            candidate_patch_ids=[candidate.patch_id for candidate in candidates],
            required_harness=_harness_requirement(candidate_target_areas),
            human_review=_human_review_requirement(candidate_target_areas),
            release_governance_ref=None,
            idempotency_key=idempotency_key,
        )


def _validate_create_inputs(
    signals: list[LearningSignal],
    *,
    requested_by: str,
    idempotency_key: str,
) -> None:
    if not isinstance(signals, list) or not signals:
        raise LearningJobValidationError("signals must not be empty")
    if not all(isinstance(signal, LearningSignal) for signal in signals):
        raise TypeError("signals must contain LearningSignal values")
    if not isinstance(requested_by, str) or not requested_by.strip():
        raise LearningJobValidationError("requested_by must be a non-empty string")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise LearningJobValidationError("idempotency_key must be a non-empty string")


def _is_strong_signal(signal: LearningSignal) -> bool:
    reason = signal.reason_code.lower()
    return reason in _STRONG_REASON_CODES or any(
        marker in reason for marker in _STRONG_REASON_MARKERS
    )


def _target_kind(patch_type: str) -> str:
    if patch_type == "evidence_ingest":
        return "evidence_ingest_candidate"
    return f"{patch_type}_candidate"


def _job_type_for_targets(target_areas: set[str]) -> str:
    if target_areas == {"test_case"}:
        return "candidate_test_case_generation"
    if "evidence_ingest" in target_areas:
        return "candidate_evidence_ingest"
    return "candidate_patch_generation"


def _harness_requirement(target_areas: set[str]) -> HarnessRequirement:
    levels = ["L0_L1", "shadow_replay"]
    if "evidence_ingest" in target_areas:
        levels.append("literature_shadow")
    else:
        levels.append("clinical_safety")
    return HarnessRequirement(
        case_pack_version="crc_mutation_pack_v0",
        required_levels=_unique(levels),
        hard_fail_policy="shadow_candidates_require_clean_harness_before_release_intent",
    )


def _human_review_requirement(target_areas: set[str]) -> HumanReviewRequirement:
    roles = ["release_manager"]
    if "evidence_ingest" in target_areas:
        roles.append("evidence_reviewer")
    if target_areas - {"evidence_ingest"}:
        roles.append("clinical_safety_reviewer")
    return HumanReviewRequirement(
        required=True,
        required_roles=_unique(roles),
        status="pending",
    )


def _disabled_actions() -> list[dict[str, object]]:
    return [
        {
            "id": "apply",
            "label": "Apply",
            "disabled": True,
            "reason": _NON_MUTATION_REASON,
        },
        {
            "id": "train",
            "label": "Train",
            "disabled": True,
            "reason": _NON_MUTATION_REASON,
        },
    ]


def _actions_compat(
    disabled_actions: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        str(action["id"]): {
            "enabled": not bool(action["disabled"]),
            "reason": action["reason"],
        }
        for action in disabled_actions
    }


def _runtime_metadata() -> dict[str, str]:
    return {
        "auth": "admin",
        "source": "reports/learning_jobs",
        "mode": "shadow_learning_jobs",
    }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["LearningJobService", "LearningJobValidationError"]
