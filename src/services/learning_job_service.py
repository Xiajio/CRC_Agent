from __future__ import annotations

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
_STRONG_SIGNAL_TYPES = (
    "doctor_action_trace",
    "evidence_delta",
    "harness_failure",
    "cohort_feasibility_gap",
    "release_monitoring_alert",
)
_NON_MUTATION_REASON = "shadow_learning_jobs_only"


class LearningJobService:
    def __init__(self, store: LearningJobStore) -> None:
        self.store = store

    def read_jobs(self) -> dict[str, object]:
        state = self.store.read_state()
        return {
            "jobs": [job.to_dict() for job in state.jobs],
            "candidates": [candidate.to_dict() for candidate in state.candidates],
            "integrity": state.integrity,
            "actions": _disabled_actions(),
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
        return {
            "jobs": [job.to_dict()],
            "candidates": [candidate.to_dict() for candidate in candidates],
            "integrity": self.store.read_state().integrity,
            "actions": _disabled_actions(),
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
        target_areas = {signal.target_area for signal in signals}
        job_type = _job_type_for_targets(target_areas)
        return LearningJob(
            job_id=make_learning_job_id(source_signal_ids, idempotency_key),
            job_type=job_type,
            status="shadow_only",
            created_at=min(signal.created_at for signal in signals),
            source_signal_ids=source_signal_ids,
            candidate_patch_ids=[candidate.patch_id for candidate in candidates],
            required_harness=_harness_requirement(target_areas),
            human_review=_human_review_requirement(target_areas),
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
        raise ValueError("signals must not be empty")
    if not all(isinstance(signal, LearningSignal) for signal in signals):
        raise TypeError("signals must contain LearningSignal values")
    if not isinstance(requested_by, str) or not requested_by.strip():
        raise ValueError("requested_by must be a non-empty string")
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise ValueError("idempotency_key must be a non-empty string")


def _is_strong_signal(signal: LearningSignal) -> bool:
    if signal.severity in {"high", "critical"}:
        return True
    reason = signal.reason_code.lower()
    if any(marker in reason for marker in _STRONG_REASON_MARKERS):
        return True
    return signal.signal_type in _STRONG_SIGNAL_TYPES and signal.severity != "low"


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
    levels = ["unit", "shadow_replay"]
    if "evidence_ingest" in target_areas:
        levels.extend(["literature", "evidence"])
    else:
        levels.append("clinical_safety")
    return HarnessRequirement(
        case_pack_version="learning_job_shadow_case_pack_v1",
        required_levels=_unique(levels),
        hard_fail_policy="shadow_candidates_require_clean_harness_before_release_intent",
    )


def _human_review_requirement(target_areas: set[str]) -> HumanReviewRequirement:
    if "evidence_ingest" in target_areas:
        roles = ["evidence_reviewer"]
    else:
        roles = ["clinical_safety_reviewer"]
    return HumanReviewRequirement(
        required=True,
        required_roles=roles,
        status="pending",
    )


def _disabled_actions() -> dict[str, dict[str, object]]:
    return {
        "apply": {"enabled": False, "reason": _NON_MUTATION_REASON},
        "train": {"enabled": False, "reason": _NON_MUTATION_REASON},
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


__all__ = ["LearningJobService"]
