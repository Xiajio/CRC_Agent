from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.learning_job_store import LearningJobStore
from src.contracts.learning_job import LearningSignal, make_learning_signal_id
from src.services.learning_job_service import (
    LearningJobService,
    LearningJobValidationError,
)


def make_signal(
    *,
    reason_code: str = "unsafe_disposition_override",
    target_area: str = "prompt",
    signal_type: str = "doctor_action_trace",
    severity: str = "high",
) -> LearningSignal:
    source_ref = {
        "kind": signal_type,
        "id": f"{signal_type}_{reason_code}_{target_area}",
        "projection": "aggregate_shadow_learning",
    }
    return LearningSignal(
        signal_id=make_learning_signal_id(source_ref),
        signal_type=signal_type,
        source_ref=source_ref,
        reason_code=reason_code,
        target_area=target_area,  # type: ignore[arg-type]
        severity=severity,
        summary="Aggregate deidentified shadow signal for learning job review.",
        deidentified=True,
        created_at="2026-07-09T10:00:00+08:00",
    )


def make_service(root: Path) -> LearningJobService:
    return LearningJobService(
        store=LearningJobStore(root),
        now=lambda: "2026-07-09T12:00:00+08:00",
    )


def test_read_jobs_returns_shadow_runtime_metadata_and_disabled_actions(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path / "reports" / "learning_jobs")

    payload = service.read_jobs()

    assert payload["jobs"] == []
    assert payload["candidates"] == []
    assert payload["integrity"] == {"status": "verified", "warnings": []}
    assert payload["disabled_actions"] == [
        {
            "id": "apply",
            "label": "Apply",
            "disabled": True,
            "reason": "shadow_learning_jobs_only",
        },
        {
            "id": "train",
            "label": "Train",
            "disabled": True,
            "reason": "shadow_learning_jobs_only",
        },
    ]
    assert payload["actions"] == {
        "apply": {"enabled": False, "reason": "shadow_learning_jobs_only"},
        "train": {"enabled": False, "reason": "shadow_learning_jobs_only"},
    }
    assert payload["runtime"] == {
        "auth": "admin",
        "source": "reports/learning_jobs",
        "mode": "shadow_learning_jobs",
    }


def test_create_job_writes_shadow_only_prompt_candidate(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    service = make_service(root)
    signal = make_signal(
        reason_code="unsafe_disposition_override",
        target_area="prompt",
    )

    result = service.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="learning-job-001",
    )

    assert "job" in result
    assert "signals" in result
    assert len(result["candidates"]) == 1
    job = result["job"]
    candidate = result["candidates"][0]
    assert result["signals"] == [signal.to_dict()]
    assert job["status"] == "shadow_only"
    assert job["created_at"] == "2026-07-09T12:00:00+08:00"
    assert job["release_governance_ref"] is None
    assert job["candidate_patch_ids"] == [candidate["patch_id"]]
    assert candidate["patch_type"] == "prompt"
    assert candidate["status"] == "candidate"
    assert candidate["applies_automatically"] is False
    assert job["human_review"]["required_roles"] == [
        "release_manager",
        "clinical_safety_reviewer",
    ]
    assert job["required_harness"]["case_pack_version"] == "crc_mutation_pack_v0"
    assert "L0_L1" in job["required_harness"]["required_levels"]
    state = LearningJobStore(root).read_state()
    assert [item.job_id for item in state.jobs] == [job["job_id"]]
    assert [item.patch_id for item in state.candidates] == [candidate["patch_id"]]


def test_create_job_for_weak_signal_writes_job_without_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    service = make_service(root)
    signal = make_signal(
        reason_code="low_confidence_observation",
        target_area="prompt",
        severity="low",
    )

    result = service.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="weak-learning-job-001",
    )

    assert result["candidates"] == []
    assert result["job"]["candidate_patch_ids"] == []
    assert result["job"]["status"] == "shadow_only"
    assert LearningJobStore(root).read_state().candidates == []


def test_documentation_note_review_required_template_signal_is_weak(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path / "reports" / "learning_jobs")
    signal = make_signal(
        reason_code="documentation_note",
        target_area="template",
        signal_type="doctor_action_trace",
        severity="review_required",
    )

    result = service.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="documentation-note-job-001",
    )

    assert result["candidates"] == []
    assert result["job"]["job_type"] == "candidate_patch_generation"
    assert result["job"]["candidate_patch_ids"] == []


def test_evidence_ingest_candidate_requires_evidence_reviewer_and_harness(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path / "reports" / "learning_jobs")
    signal = make_signal(
        reason_code="evidence_delta_detected",
        target_area="evidence_ingest",
        signal_type="evidence_delta",
    )

    result = service.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="evidence-learning-job-001",
    )

    job = result["job"]
    candidate = result["candidates"][0]
    assert candidate["patch_type"] == "evidence_ingest"
    assert job["human_review"]["required_roles"] == [
        "release_manager",
        "evidence_reviewer",
    ]
    assert "literature_shadow" in job["required_harness"]["required_levels"]


@pytest.mark.parametrize(
    "reason_code",
    [
        "unsafe_disposition",
        "citation_not_traceable",
        "evidence_conflict",
        "safety_signal",
        "harness_hard_fail",
        "missing_variable",
        "monitoring_alert",
    ],
)
def test_plan_strong_reason_codes_create_candidates(
    tmp_path: Path,
    reason_code: str,
) -> None:
    service = make_service(tmp_path / reason_code / "reports" / "learning_jobs")
    signal = make_signal(reason_code=reason_code, target_area="prompt")

    result = service.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key=f"{reason_code}-job-001",
    )

    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["source_signal_ids"] == [signal.signal_id]


@pytest.mark.parametrize("target_area", ["rubric", "route", "template", "test_case"])
def test_non_evidence_candidates_require_clinical_safety_reviewer(
    tmp_path: Path,
    target_area: str,
) -> None:
    service = make_service(tmp_path / f"reports_{target_area}" / "learning_jobs")
    signal = make_signal(
        reason_code="harness_failure_regression",
        target_area=target_area,
        signal_type="harness_failure",
    )

    result = service.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key=f"{target_area}-learning-job-001",
    )

    assert result["candidates"][0]["patch_type"] == target_area
    assert result["job"]["human_review"]["required_roles"] == [
        "release_manager",
        "clinical_safety_reviewer",
    ]


@pytest.mark.parametrize(
    ("signals", "requested_by", "idempotency_key", "match"),
    [
        ([], "admin_user", "idem-1", "signals must not be empty"),
        ([make_signal()], "", "idem-1", "requested_by must be a non-empty string"),
        ([make_signal()], "admin_user", "", "idempotency_key must be a non-empty string"),
    ],
)
def test_create_job_validates_required_inputs(
    tmp_path: Path,
    signals: list[LearningSignal],
    requested_by: str,
    idempotency_key: str,
    match: str,
) -> None:
    with pytest.raises(LearningJobValidationError, match=match):
        make_service(tmp_path / "reports" / "learning_jobs").create_job(
            signals,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
        )


def test_create_job_ids_are_deterministic_for_same_signals_and_key(
    tmp_path: Path,
) -> None:
    first = make_service(tmp_path / "first" / "reports" / "learning_jobs")
    second = make_service(tmp_path / "second" / "reports" / "learning_jobs")
    signal = make_signal()

    first_payload = first.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="stable-key",
    )
    second_payload = second.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="stable-key",
    )

    assert first_payload["job"]["job_id"] == second_payload["job"]["job_id"]
    assert (
        first_payload["candidates"][0]["patch_id"]
        == second_payload["candidates"][0]["patch_id"]
    )


def test_create_job_ids_change_with_idempotency_key(tmp_path: Path) -> None:
    first = make_service(tmp_path / "first" / "reports" / "learning_jobs")
    second = make_service(tmp_path / "second" / "reports" / "learning_jobs")
    signal = make_signal()

    first_payload = first.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="stable-key-1",
    )
    second_payload = second.create_job(
        [signal],
        requested_by="admin_user",
        idempotency_key="stable-key-2",
    )

    assert first_payload["job"]["job_id"] != second_payload["job"]["job_id"]
    assert (
        first_payload["candidates"][0]["patch_id"]
        != second_payload["candidates"][0]["patch_id"]
    )


def test_service_exposes_only_shadow_disabled_actions_after_create(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path / "reports" / "learning_jobs")
    service.create_job(
        [make_signal()],
        requested_by="admin_user",
        idempotency_key="learning-job-001",
    )

    payload = service.read_jobs()

    assert payload["disabled_actions"][0]["id"] == "apply"
    assert payload["disabled_actions"][0]["disabled"] is True
    assert payload["disabled_actions"][1]["id"] == "train"
    assert payload["disabled_actions"][1]["disabled"] is True
    assert payload["actions"]["apply"]["enabled"] is False
    assert payload["actions"]["train"]["enabled"] is False
    assert payload["runtime"]["mode"] == "shadow_learning_jobs"
    assert payload["jobs"][0]["status"] == "shadow_only"
