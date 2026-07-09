from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.api.services.learning_job_store import (
    LearningJobIntegrityError,
    LearningJobStore,
)
from src.contracts.learning_job import (
    CandidatePatch,
    HarnessRequirement,
    HumanReviewRequirement,
    LearningJob,
    make_candidate_patch_id,
    make_learning_job_id,
    make_learning_signal_id,
)


def make_candidate(signal_id: str = "learning_signal_seed") -> CandidatePatch:
    return CandidatePatch(
        patch_id=make_candidate_patch_id("prompt", signal_id),
        patch_type="prompt",
        target_ref={"kind": "prompt_template", "id": "crc_triage_disposition_prompt_v1"},
        change_summary="Add a shadow-only escalation guardrail candidate.",
        proposed_diff={
            "format": "structured_diff",
            "ops": [
                {
                    "op": "add_guardrail",
                    "path": "/disposition/escalation",
                    "value": "Escalate unsafe low-acuity overrides to review.",
                }
            ],
        },
        source_signal_ids=[signal_id],
        status="candidate",
        applies_automatically=False,
    )


def make_job(
    *,
    source_signal_ids: list[str] | None = None,
    candidate_patch_ids: list[str] | None = None,
    job_id: str | None = None,
) -> LearningJob:
    source_ref = {
        "kind": "doctor_action_trace",
        "id": "doctor_action_trace_crc_shadow_001",
    }
    source_ids = (
        [make_learning_signal_id(source_ref)]
        if source_signal_ids is None
        else source_signal_ids
    )
    candidate_ids = (
        [make_candidate(source_ids[0]).patch_id]
        if candidate_patch_ids is None
        else candidate_patch_ids
    )
    idempotency_key = "learning-job-shadow-001"
    return LearningJob(
        job_id=job_id or make_learning_job_id(source_ids, idempotency_key),
        job_type="candidate_patch_generation",
        status="shadow_only",
        created_at="2026-07-09T10:05:00+08:00",
        source_signal_ids=source_ids,
        candidate_patch_ids=candidate_ids,
        required_harness=HarnessRequirement(
            case_pack_version="crc_triage_harness_v2",
            required_levels=["unit", "scenario"],
            hard_fail_policy="any_clinical_safety_regression_blocks_release_intent",
        ),
        human_review=HumanReviewRequirement(
            required=True,
            required_roles=["clinical_safety_reviewer", "release_manager"],
            status="pending",
        ),
        release_governance_ref=None,
        idempotency_key=idempotency_key,
    )


def test_empty_store_read_is_verified_and_does_not_create_files(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    state = LearningJobStore(root).read_state()

    assert state.jobs == []
    assert state.candidates == []
    assert state.integrity == {"status": "verified", "warnings": []}
    assert not root.exists()


def test_write_job_creates_candidate_and_job_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    store = LearningJobStore(root)
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])

    store.write_job(job, [candidate])

    state = store.read_state()
    assert [item.job_id for item in state.jobs] == [job.job_id]
    assert [item.patch_id for item in state.candidates] == [candidate.patch_id]
    assert state.integrity == {"status": "verified", "warnings": []}
    assert (root / "jobs" / f"{job.job_id}.json").exists()
    assert (root / "candidates" / f"{candidate.patch_id}.json").exists()


def test_write_job_allows_weak_signal_job_with_no_candidates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    store = LearningJobStore(root)
    job = make_job(candidate_patch_ids=[])

    store.write_job(job, [])

    state = store.read_state()
    assert [item.job_id for item in state.jobs] == [job.job_id]
    assert state.candidates == []
    assert (root / "jobs" / f"{job.job_id}.json").exists()


def test_read_state_skips_malformed_json_and_invalid_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    jobs_dir = root / "jobs"
    candidates_dir = root / "candidates"
    jobs_dir.mkdir(parents=True)
    candidates_dir.mkdir(parents=True)
    (jobs_dir / "broken.json").write_text("{bad json", encoding="utf-8")
    (candidates_dir / "invalid.json").write_text(
        json.dumps({"patch_id": "candidate_patch_bad"}),
        encoding="utf-8",
    )

    state = LearningJobStore(root).read_state()

    assert state.jobs == []
    assert state.candidates == []
    assert state.integrity["status"] == "warning"
    assert any("not valid JSON" in warning for warning in state.integrity["warnings"])
    assert any("artifact is invalid" in warning for warning in state.integrity["warnings"])


def test_duplicate_job_or_candidate_ids_raise_without_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    store = LearningJobStore(root)
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])
    store.write_job(job, [candidate])
    original_job_text = (root / "jobs" / f"{job.job_id}.json").read_text(
        encoding="utf-8"
    )
    original_candidate_text = (
        root / "candidates" / f"{candidate.patch_id}.json"
    ).read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        store.write_job(job, [candidate])

    assert (root / "jobs" / f"{job.job_id}.json").read_text(
        encoding="utf-8"
    ) == original_job_text
    assert (root / "candidates" / f"{candidate.patch_id}.json").read_text(
        encoding="utf-8"
    ) == original_candidate_text


def test_duplicate_candidate_ids_in_same_write_raise_file_exists(
    tmp_path: Path,
) -> None:
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])

    with pytest.raises(FileExistsError):
        LearningJobStore(tmp_path / "reports" / "learning_jobs").write_job(
            job,
            [candidate, candidate],
        )


@pytest.mark.parametrize(
    ("job_id", "candidate_id"),
    [
        ("../outside", "candidate_patch_safe"),
        ("learning_job_safe", "..\\outside"),
        ("learning job bad", "candidate_patch_safe"),
    ],
)
def test_write_job_rejects_unsafe_artifact_ids(
    tmp_path: Path,
    job_id: str,
    candidate_id: str,
) -> None:
    candidate_payload = make_candidate().to_dict()
    candidate_payload["patch_id"] = candidate_id
    candidate = CandidatePatch(**candidate_payload)
    job = make_job(job_id=job_id, candidate_patch_ids=[candidate.patch_id])

    with pytest.raises(LearningJobIntegrityError):
        LearningJobStore(tmp_path / "reports" / "learning_jobs").write_job(
            job,
            [candidate],
        )


def test_write_job_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "outside_target"
    target.mkdir()
    root = tmp_path / "reports" / "learning_jobs"
    root.parent.mkdir(parents=True)
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is not available")
        raise
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])

    with pytest.raises(LearningJobIntegrityError, match="symlink"):
        LearningJobStore(root).write_job(job, [candidate])

    assert not (target / "jobs").exists()
    assert not (target / "candidates").exists()


@pytest.mark.parametrize("symlink_dir_name", ["jobs", "candidates"])
def test_write_job_rejects_symlink_artifact_directories(
    tmp_path: Path,
    symlink_dir_name: str,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    root.mkdir(parents=True)
    target = tmp_path / f"{symlink_dir_name}_outside_target"
    target.mkdir()
    try:
        (root / symlink_dir_name).symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is not available")
        raise
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])

    with pytest.raises(LearningJobIntegrityError, match="symlink"):
        LearningJobStore(root).write_job(job, [candidate])

    assert list(target.iterdir()) == []


def test_write_job_validates_candidate_ids_match_job_ids(tmp_path: Path) -> None:
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=["candidate_patch_missing"])

    with pytest.raises(ValueError, match="candidate_patch_ids"):
        LearningJobStore(tmp_path / "reports" / "learning_jobs").write_job(
            job,
            [candidate],
        )


def test_write_job_rejects_duplicate_candidate_refs_in_job(tmp_path: Path) -> None:
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])
    job.candidate_patch_ids.append(candidate.patch_id)

    with pytest.raises(ValueError, match="duplicate candidate_patch_ids"):
        LearningJobStore(tmp_path / "reports" / "learning_jobs").write_job(
            job,
            [candidate],
        )


def test_read_state_warns_on_job_with_duplicate_candidate_refs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    jobs_dir = root / "jobs"
    candidates_dir = root / "candidates"
    jobs_dir.mkdir(parents=True)
    candidates_dir.mkdir(parents=True)
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])
    job_payload = job.to_dict()
    job_payload["candidate_patch_ids"] = [candidate.patch_id, candidate.patch_id]
    (jobs_dir / f"{job.job_id}.json").write_text(
        json.dumps(job_payload, sort_keys=True),
        encoding="utf-8",
    )
    (candidates_dir / f"{candidate.patch_id}.json").write_text(
        json.dumps(candidate.to_dict(), sort_keys=True),
        encoding="utf-8",
    )

    state = LearningJobStore(root).read_state()

    assert state.integrity["status"] == "warning"
    assert any(
        "duplicate candidate_patch_ids" in warning
        for warning in state.integrity["warnings"]
    )


def test_write_job_rolls_back_files_written_in_same_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    store = LearningJobStore(root)
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])
    original_write_json_once = store._write_json_once
    calls = 0

    def fail_after_candidate(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated job write failure")
        original_write_json_once(path, payload)

    monkeypatch.setattr(store, "_write_json_once", fail_after_candidate)

    with pytest.raises(OSError, match="simulated job write failure"):
        store.write_job(job, [candidate])

    assert not (root / "candidates" / f"{candidate.patch_id}.json").exists()
    assert not (root / "jobs" / f"{job.job_id}.json").exists()


def test_write_json_once_removes_partial_file_when_dump_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    store = LearningJobStore(root)
    candidate = make_candidate()
    job = make_job(candidate_patch_ids=[candidate.patch_id])

    def fail_dump(*args: object, **kwargs: object) -> None:
        raise OSError("simulated dump failure")

    monkeypatch.setattr(
        "backend.api.services.learning_job_store.json.dump",
        fail_dump,
    )

    with pytest.raises(OSError, match="simulated dump failure"):
        store.write_job(job, [candidate])

    assert not (root / "candidates" / f"{candidate.patch_id}.json").exists()
    assert not (root / "jobs" / f"{job.job_id}.json").exists()


def test_write_job_does_not_delete_preexisting_artifacts_on_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reports" / "learning_jobs"
    store = LearningJobStore(root)
    preexisting = make_candidate("learning_signal_preexisting")
    preexisting_job = make_job(
        source_signal_ids=["learning_signal_preexisting"],
        candidate_patch_ids=[preexisting.patch_id],
    )
    store.write_job(preexisting_job, [preexisting])
    new_candidate = make_candidate("learning_signal_new")
    new_job = make_job(
        source_signal_ids=["learning_signal_new"],
        candidate_patch_ids=[new_candidate.patch_id],
    )
    original_write_json_once = store._write_json_once
    calls = 0

    def fail_after_candidate(path: Path, payload: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated job write failure")
        original_write_json_once(path, payload)

    monkeypatch.setattr(store, "_write_json_once", fail_after_candidate)

    with pytest.raises(OSError, match="simulated job write failure"):
        store.write_job(new_job, [new_candidate])

    assert (root / "candidates" / f"{preexisting.patch_id}.json").exists()
    assert (root / "jobs" / f"{preexisting_job.job_id}.json").exists()
    assert not (root / "candidates" / f"{new_candidate.patch_id}.json").exists()


def test_readme_documents_shadow_only_non_mutation_policy() -> None:
    readme = Path("reports/learning_jobs/README.md").read_text(encoding="utf-8")

    assert "shadow-only" in readme
    assert "do not apply" in readme
    for blocked_target in [
        "prompts",
        "rubrics",
        "routes",
        "templates",
        "RAG",
        "safety policy",
        "feature flags",
        "model training data",
    ]:
        assert blocked_target in readme
