from __future__ import annotations

import pytest

from src.contracts.learning_job import (
    CandidatePatch,
    HarnessRequirement,
    HumanReviewRequirement,
    LearningJob,
    LearningSignal,
    canonical_learning_payload_hash,
    make_candidate_patch_id,
    make_learning_job_id,
    make_learning_signal_id,
)


def make_signal(**overrides: object) -> LearningSignal:
    source_ref = {
        "kind": "doctor_action_trace",
        "id": "doctor_action_trace_crc_shadow_001",
        "projection": "aggregate_disposition_pattern",
    }
    payload = {
        "signal_id": make_learning_signal_id(source_ref),
        "signal_type": "doctor_action_trace",
        "source_ref": source_ref,
        "target_area": "prompt",
        "summary": "Doctor overrode an unsafe low-acuity disposition in aggregate review.",
        "observed_at": "2026-07-09T10:00:00+08:00",
        "deidentified": True,
        "payload": {
            "action": "unsafe_disposition_override",
            "disposition": "urgent_review",
            "aggregate_count": 3,
        },
    }
    payload.update(overrides)
    return LearningSignal(**payload)


def make_patch(
    signal_id: str = "learning_signal_seed",
    **overrides: object,
) -> CandidatePatch:
    patch_id = make_candidate_patch_id("prompt", "doctor_action_trace_crc_shadow_001")
    payload = {
        "candidate_patch_id": patch_id,
        "patch_type": "prompt",
        "target_ref": {
            "kind": "prompt_template",
            "id": "crc_triage_disposition_prompt_v1",
        },
        "proposed_diff": {
            "format": "structured_diff",
            "ops": [
                {
                    "op": "add_guardrail",
                    "path": "/disposition/escalation",
                    "value": "Escalate unsafe low-acuity overrides to review.",
                }
            ],
        },
        "rationale": "Aggregate doctor overrides indicate a shadow candidate guardrail.",
        "source_signal_ids": [signal_id],
        "status": "candidate",
        "applies_automatically": False,
    }
    payload.update(overrides)
    return CandidatePatch(**payload)


def make_harness(**overrides: object) -> HarnessRequirement:
    payload = {
        "required": True,
        "case_pack_version": "crc_triage_harness_v2",
        "required_levels": ["unit", "scenario", "regression"],
        "hard_fail_policy": "any_clinical_safety_regression_blocks_release_intent",
    }
    payload.update(overrides)
    return HarnessRequirement(**payload)


def make_review(**overrides: object) -> HumanReviewRequirement:
    payload = {
        "required": True,
        "roles": ["clinical_safety_reviewer", "release_manager"],
        "status": "pending",
    }
    payload.update(overrides)
    return HumanReviewRequirement(**payload)


def make_job(
    source_signal_ids: list[str] | None = None,
    candidate_patch_ids: list[str] | None = None,
    **overrides: object,
) -> LearningJob:
    source_ids = (
        ["learning_signal_seed"] if source_signal_ids is None else source_signal_ids
    )
    candidate_ids = (
        ["candidate_patch_seed"]
        if candidate_patch_ids is None
        else candidate_patch_ids
    )
    idempotency_key = "learning-job-shadow-001"
    payload = {
        "learning_job_id": make_learning_job_id(source_ids, idempotency_key),
        "job_type": "candidate_patch_generation",
        "status": "shadow_only",
        "source_signal_ids": source_ids,
        "candidate_patch_ids": candidate_ids,
        "harness_requirement": make_harness(),
        "human_review_requirement": make_review(),
        "idempotency_key": idempotency_key,
        "created_at": "2026-07-09T10:05:00+08:00",
    }
    payload.update(overrides)
    return LearningJob(**payload)


def test_learning_job_contracts_round_trip() -> None:
    signal = make_signal()
    patch = make_patch(signal.signal_id)
    job = make_job(
        source_signal_ids=[signal.signal_id],
        candidate_patch_ids=[patch.candidate_patch_id],
    )

    assert signal.to_dict()["deidentified"] is True
    assert patch.to_dict()["applies_automatically"] is False
    assert job.to_dict()["status"] == "shadow_only"
    assert (
        job.to_dict()["harness_requirement"]["case_pack_version"]
        == "crc_triage_harness_v2"
    )


def test_signal_rejects_non_deidentified_input() -> None:
    with pytest.raises(ValueError, match="deidentified must be true"):
        make_signal(deidentified=False)


def test_signal_rejects_patient_level_rows() -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        make_signal(source_ref={"kind": "cohort_rows", "patient_id": "p-1"})


def test_candidate_rejects_automatic_application() -> None:
    payload = make_patch().to_dict()
    payload["applies_automatically"] = True

    with pytest.raises(ValueError, match="applies_automatically must be false"):
        CandidatePatch(**payload)


@pytest.mark.parametrize(
    "status",
    ["applied", "released", "trained", "clinical_rag_active"],
)
def test_job_rejects_active_statuses(status: str) -> None:
    with pytest.raises(ValueError, match="status must be one of"):
        make_job(status=status)


def test_canonical_hash_rejects_secret_content() -> None:
    with pytest.raises(ValueError, match="forbidden content"):
        canonical_learning_payload_hash({"note": "Bearer abcdef123456"})


def test_learning_job_allows_zero_candidate_ids_for_weak_signals() -> None:
    job = make_job(candidate_patch_ids=[])

    assert job.to_dict()["candidate_patch_ids"] == []


@pytest.mark.parametrize(
    "target_ref",
    [
        {"kind": "clinical_safety_policy", "id": "crc_prompt_v1"},
        {"kind": "prompt_template", "id": "clinical_safety_policy_crc_v1"},
    ],
)
def test_candidate_rejects_clinical_safety_policy_target(
    target_ref: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="clinical_safety_policy"):
        make_patch(target_ref=target_ref)


def test_payload_hash_is_deterministic_and_json_safe() -> None:
    left = canonical_learning_payload_hash({"b": [2, 1], "a": {"c": "safe"}})
    right = canonical_learning_payload_hash({"a": {"c": "safe"}, "b": [2, 1]})

    assert left == right
    assert left.startswith("sha256:")


def test_required_human_review_must_be_true() -> None:
    with pytest.raises(ValueError, match="required must be true"):
        make_review(required=False)
