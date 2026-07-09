from __future__ import annotations

import json

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
        "reason_code": "unsafe_disposition_override",
        "target_area": "prompt",
        "severity": "high",
        "summary": "Doctor overrode an unsafe low-acuity disposition in aggregate review.",
        "deidentified": True,
        "created_at": "2026-07-09T10:00:00+08:00",
    }
    payload.update(overrides)
    return LearningSignal(**payload)


def make_patch(
    signal_id: str = "learning_signal_seed",
    **overrides: object,
) -> CandidatePatch:
    patch_id = make_candidate_patch_id("prompt", "doctor_action_trace_crc_shadow_001")
    payload = {
        "patch_id": patch_id,
        "patch_type": "prompt",
        "target_ref": {
            "kind": "prompt_template",
            "id": "crc_triage_disposition_prompt_v1",
        },
        "change_summary": "Add a shadow-only escalation guardrail candidate.",
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
        "job_id": make_learning_job_id(source_ids, idempotency_key),
        "job_type": "candidate_patch_generation",
        "status": "shadow_only",
        "created_at": "2026-07-09T10:05:00+08:00",
        "source_signal_ids": source_ids,
        "candidate_patch_ids": candidate_ids,
        "required_harness": make_harness(),
        "human_review": make_review(),
        "release_governance_ref": {
            "kind": "release_governance_intent",
            "id": "release_governance_shadow_learning_v1",
        },
        "idempotency_key": idempotency_key,
    }
    payload.update(overrides)
    return LearningJob(**payload)


def test_learning_job_contracts_round_trip() -> None:
    signal = make_signal()
    patch = make_patch(signal.signal_id)
    job = make_job(
        source_signal_ids=[signal.signal_id],
        candidate_patch_ids=[patch.patch_id],
    )

    assert signal.to_dict()["deidentified"] is True
    assert signal.to_dict()["reason_code"] == "unsafe_disposition_override"
    assert patch.to_dict()["applies_automatically"] is False
    assert patch.to_dict()["patch_id"] == patch.patch_id
    assert job.to_dict()["status"] == "shadow_only"
    assert (
        job.to_dict()["required_harness"]["case_pack_version"]
        == "crc_triage_harness_v2"
    )
    assert (
        job.to_dict()["release_governance_ref"]["kind"]
        == "release_governance_intent"
    )
    json.dumps(signal.to_dict())
    json.dumps(patch.to_dict())
    json.dumps(job.to_dict())


def test_learning_job_contracts_serialize_plan_shaped_keys_only() -> None:
    signal = make_signal()
    patch = make_patch(signal.signal_id)
    job = make_job(
        source_signal_ids=[signal.signal_id],
        candidate_patch_ids=[patch.patch_id],
    )

    assert set(signal.to_dict()) == {
        "signal_id",
        "signal_type",
        "source_ref",
        "reason_code",
        "target_area",
        "severity",
        "summary",
        "deidentified",
        "created_at",
    }
    assert set(patch.to_dict()) == {
        "patch_id",
        "patch_type",
        "target_ref",
        "change_summary",
        "proposed_diff",
        "source_signal_ids",
        "status",
        "applies_automatically",
    }
    assert set(job.to_dict()) == {
        "job_id",
        "job_type",
        "status",
        "created_at",
        "source_signal_ids",
        "candidate_patch_ids",
        "required_harness",
        "human_review",
        "release_governance_ref",
        "idempotency_key",
    }


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


def test_signal_to_dict_rejects_post_construction_source_ref_mutation() -> None:
    signal = make_signal()
    signal.source_ref["patient_id"] = "p-1"

    with pytest.raises(ValueError, match="forbidden key"):
        signal.to_dict()


def test_candidate_to_dict_rejects_post_construction_target_ref_mutation() -> None:
    candidate = make_patch()
    candidate.target_ref["id"] = "clinical_safety_policy_crc_v1"

    with pytest.raises(ValueError, match="clinical_safety_policy"):
        candidate.to_dict()


def test_candidate_to_dict_rejects_post_construction_diff_mutation() -> None:
    candidate = make_patch()
    candidate.proposed_diff["patient_id"] = "p-1"

    with pytest.raises(ValueError, match="forbidden key"):
        candidate.to_dict()
