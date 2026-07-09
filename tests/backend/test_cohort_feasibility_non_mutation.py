from __future__ import annotations

from pathlib import Path

from src.contracts.research_asset import CohortFeasibilityRequest
from src.services.cohort_feasibility_service import CohortFeasibilityService


def _snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file():
            snapshot[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return snapshot


def test_cohort_feasibility_does_not_mutate_runtime_artifacts(tmp_path: Path) -> None:
    protected = [
        tmp_path / "config" / "safety_policy.yaml",
        tmp_path / "reports" / "literature" / "literature_harness.json",
        tmp_path / "reports" / "learning_jobs" / "sentinel.json",
        tmp_path / "src" / "prompts" / "decision_prompts.py",
        tmp_path / "src" / "routes" / "router.py",
    ]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}: original\n", encoding="utf-8")
    before = _snapshot(tmp_path)

    request = CohortFeasibilityRequest(
        request_id="cohort_request_crc_001",
        project_id="research_crc_001",
        question="Is there enough structured CRC triage data?",
        cohort_criteria={
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "required_features": ["rectal_bleeding"],
        },
        data_scope={
            "source": "patient_record_projection",
            "patient_level_export_requested": False,
            "deidentified_only": True,
        },
        version_refs={"projection_version": "patient_record_projection_v0"},
    )

    CohortFeasibilityService().evaluate(request=request, records=[])

    assert _snapshot(tmp_path) == before
