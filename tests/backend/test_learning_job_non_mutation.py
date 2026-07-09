from __future__ import annotations

from pathlib import Path

from backend.api.services.learning_job_store import LearningJobStore
from src.contracts.learning_job import LearningSignal, make_learning_signal_id
from src.services.learning_job_service import LearningJobService


def _write_sentinel(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{label}: original shadow-boundary sentinel\n", encoding="utf-8")


def _snapshot(paths: dict[str, Path]) -> dict[str, str]:
    return {
        label: path.read_text(encoding="utf-8")
        for label, path in paths.items()
    }


def _all_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _learning_signal() -> LearningSignal:
    source_ref = {
        "kind": "doctor_action_trace",
        "id": "doctor_action_trace_crc_shadow_001",
        "projection": "aggregate_shadow_learning",
    }
    return LearningSignal(
        signal_id=make_learning_signal_id(source_ref),
        signal_type="doctor_action_trace",
        source_ref=source_ref,
        reason_code="unsafe_disposition",
        target_area="prompt",
        severity="high",
        summary="Aggregate deidentified shadow signal for learning job review.",
        deidentified=True,
        created_at="2026-07-09T10:00:00+08:00",
    )


def test_learning_job_create_writes_only_shadow_artifacts_and_preserves_runtime_state(
    tmp_path: Path,
) -> None:
    protected_paths = {
        "safety_policy": tmp_path / "config" / "clinical_safety_policy.yaml",
        "prompt": tmp_path / "src" / "prompts" / "crc_triage_prompt.md",
        "rubric": tmp_path / "src" / "rubrics" / "crc_triage_rubric.json",
        "route": tmp_path / "src" / "routes" / "crc_triage_routes.json",
        "template": tmp_path / "src" / "templates" / "crc_discharge_template.md",
        "rag_literature": tmp_path / "reports" / "literature" / "rag_index.json",
        "harness": tmp_path / "reports" / "harness" / "crc_mutation_pack_v0.json",
        "release_governance": (
            tmp_path / "reports" / "release_governance" / "active_intent.json"
        ),
        "release_feature_flags": (
            tmp_path / "reports" / "release_execution" / "feature_flags" / "current.json"
        ),
        "patient_state": tmp_path / "runtime" / "patient_state" / "current.json",
        "doctor_state": tmp_path / "runtime" / "doctor_state" / "current.json",
        "training_data": tmp_path / "training_data" / "crc_shadow_candidates.jsonl",
        "model_files": tmp_path / "models" / "crc_shadow_model.bin",
        "tool_manifests": tmp_path / "tools" / "crc_tool_manifest.json",
        "crc_client": tmp_path / "CRC-client" / "package.json",
    }
    for label, path in protected_paths.items():
        _write_sentinel(path, label)
    before_snapshot = _snapshot(protected_paths)
    before_files = _all_files(tmp_path)
    service = LearningJobService(
        store=LearningJobStore(tmp_path / "reports" / "learning_jobs"),
        now=lambda: "2026-07-09T12:00:00+08:00",
    )

    result = service.create_job(
        [_learning_signal()],
        requested_by="admin_user",
        idempotency_key="shadow-boundary-001",
    )

    assert result["job"]["status"] == "shadow_only"
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["applies_automatically"] is False
    assert _snapshot(protected_paths) == before_snapshot
    new_files = _all_files(tmp_path) - before_files
    assert new_files
    assert all(path.startswith("reports/learning_jobs/") for path in new_files)
