from __future__ import annotations

import json
from pathlib import Path

from backend.api.services.admin_release_dashboard import (
    ReleaseDashboardPaths,
    build_release_dashboard,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _harness_payload(hard_fail_count: int = 0) -> dict[str, object]:
    return {
        "run_id": "harness_test",
        "summary": {"total_cases": 1, "passed": 1, "hard_fail_count": hard_fail_count},
    }


def _release_payload(hard_fail_count: int = 0) -> dict[str, object]:
    return {
        "report_id": "release_test",
        "version_chain": {
            "agent_policy_version": "agent_policy_test",
            "clinical_safety_policy_version": "safety_test",
            "evidence_index_version": "evidence_test",
            "judge_rubric_version": "rubric_test",
        },
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_previous",
        "hard_fail_summary": {"count": hard_fail_count, "types": []},
    }


def _literature_payload(
    release_decision: str = "shadow_only",
    isolation_violations: int = 0,
    validation_errors: list[str] | None = None,
    include_validation_errors: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "literature_test",
        "release_decision": release_decision,
        "run_level": "L0_shadow",
        "summary": {
            "claims": 2,
            "negative_or_conflicting_claims": 1,
            "isolation_violations": isolation_violations,
        },
    }
    if include_validation_errors:
        payload["validation_errors"] = validation_errors or []
    return payload


def _tmp_paths(tmp_path: Path) -> ReleaseDashboardPaths:
    return ReleaseDashboardPaths(
        harness_report=tmp_path / "harness.json",
        release_safety_report=tmp_path / "release.json",
        literature_report=tmp_path / "literature.json",
    )


def test_build_release_dashboard_from_committed_reports() -> None:
    dashboard = build_release_dashboard()

    assert dashboard["version_chain"] == {
        "agent_policy_version": "agent_policy_20260629_0",
        "clinical_safety_policy_version": "crc_safety_policy_v0",
        "evidence_index_version": "rag_crc_guideline_20260620",
        "judge_rubric_version": "crc_rubric_v0",
    }
    assert dashboard["release_decision"] == "feature_flag_or_pass"
    assert dashboard["rollback_target"] == "agent_policy_20260624_0"
    assert dashboard["human_signoff"] == {
        "required": True,
        "status": "missing",
        "reason": "Step 11 is read-only; sign-off must be recorded by a later audited write path.",
    }
    assert dashboard["summary"]["hard_fail_count"] == 0
    assert dashboard["summary"]["p0_cases_total"] == 5
    assert dashboard["summary"]["p0_cases_passed"] == 5
    assert dashboard["summary"]["literature_claims"] == 3
    assert dashboard["summary"]["literature_isolation_violations"] == 0
    assert dashboard["summary"]["clinical_rag_ingest_enabled"] is False
    assert [run["run_id"] for run in dashboard["runs"]] == [
        "harness_20260629_001",
        "release_safety_20260629_001",
        "literature_harness_20260630_001",
    ]
    assert [run["status"] for run in dashboard["runs"]] == ["pass", "pass", "shadow_only"]
    assert {action["id"] for action in dashboard["disabled_actions"]} == {
        "record_human_signoff",
        "publish_feature_flag",
        "rollback_release",
    }
    assert dashboard["runtime"] == {
        "auth": "admin",
        "source": "reports/static_release_artifacts",
        "mode": "read_only",
    }


def test_missing_harness_report_marks_missing_without_hiding_literature() -> None:
    paths = ReleaseDashboardPaths(
        harness_report=Path("reports/harness/missing_harness.json"),
        release_safety_report=Path("reports/release_safety/release_safety_20260629_001.json"),
        literature_report=Path("reports/literature/literature_harness_20260630_001.json"),
    )

    dashboard = build_release_dashboard(paths=paths)

    harness_run = dashboard["runs"][0]
    literature_run = dashboard["runs"][2]
    assert harness_run["kind"] == "p0_crc_harness"
    assert harness_run["status"] == "missing"
    assert harness_run["run_id"] == "missing"
    assert dashboard["summary"]["p0_cases_total"] == 0
    assert dashboard["summary"]["p0_cases_passed"] == 0
    assert literature_run["run_id"] == "literature_harness_20260630_001"
    assert literature_run["status"] == "shadow_only"
    assert dashboard["summary"]["literature_claims"] == 3


def test_malformed_literature_report_marks_invalid_and_blocks_promotion(tmp_path: Path) -> None:
    malformed = tmp_path / "literature.json"
    malformed.write_text("{not valid json", encoding="utf-8")
    paths = ReleaseDashboardPaths(
        harness_report=Path("reports/harness/harness_20260629_001.json"),
        release_safety_report=Path("reports/release_safety/release_safety_20260629_001.json"),
        literature_report=malformed,
    )

    dashboard = build_release_dashboard(paths=paths)

    literature_run = dashboard["runs"][2]
    assert literature_run["kind"] == "literature_shadow_harness"
    assert literature_run["status"] == "invalid"
    assert literature_run["run_id"] == "invalid"
    assert dashboard["summary"]["literature_claims"] == 0
    assert dashboard["summary"]["literature_isolation_violations"] == 1
    assert any(gate["id"] == "no_literature_clinical_rag" for gate in dashboard["blocking_gates"])


def test_literature_block_decision_fails_and_blocks_clinical_rag(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    _write_json(paths.harness_report, _harness_payload())
    _write_json(paths.release_safety_report, _release_payload())
    _write_json(
        paths.literature_report,
        _literature_payload(release_decision="block", isolation_violations=0),
    )

    dashboard = build_release_dashboard(paths=paths)

    literature_run = dashboard["runs"][2]
    clinical_rag_gate = next(
        gate
        for gate in dashboard["blocking_gates"]
        if gate["id"] == "no_literature_clinical_rag"
    )
    assert literature_run["status"] == "fail"
    assert clinical_rag_gate["state"] == "blocked"


def test_literature_validation_errors_or_isolation_violations_fail(
    tmp_path: Path,
) -> None:
    paths = _tmp_paths(tmp_path)
    _write_json(paths.harness_report, _harness_payload())
    _write_json(paths.release_safety_report, _release_payload())
    _write_json(
        paths.literature_report,
        _literature_payload(validation_errors=["unexpected clinical write"]),
    )

    validation_error_dashboard = build_release_dashboard(paths=paths)

    _write_json(
        paths.literature_report,
        _literature_payload(isolation_violations=1, validation_errors=[]),
    )
    isolation_violation_dashboard = build_release_dashboard(paths=paths)

    assert validation_error_dashboard["runs"][2]["status"] == "fail"
    assert isolation_violation_dashboard["runs"][2]["status"] == "fail"
    assert all(
        gate["state"] == "blocked"
        for dashboard in (validation_error_dashboard, isolation_violation_dashboard)
        for gate in dashboard["blocking_gates"]
    )


def test_literature_missing_validation_errors_is_invalid(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    _write_json(paths.harness_report, _harness_payload())
    _write_json(paths.release_safety_report, _release_payload())
    _write_json(
        paths.literature_report,
        _literature_payload(
            release_decision="shadow_only",
            isolation_violations=0,
            include_validation_errors=False,
        ),
    )

    dashboard = build_release_dashboard(paths=paths)

    literature_run = dashboard["runs"][2]
    clinical_rag_gate = next(
        gate
        for gate in dashboard["blocking_gates"]
        if gate["id"] == "no_literature_clinical_rag"
    )
    assert literature_run["status"] == "invalid"
    assert literature_run["run_id"] == "invalid"
    assert dashboard["summary"]["literature_claims"] == 0
    assert dashboard["summary"]["literature_isolation_violations"] == 1
    assert clinical_rag_gate["state"] == "blocked"


def test_release_hard_fail_count_fails_only_release_run_without_writes(
    tmp_path: Path,
) -> None:
    paths = _tmp_paths(tmp_path)
    _write_json(paths.harness_report, _harness_payload())
    _write_json(paths.release_safety_report, _release_payload(hard_fail_count=2))
    _write_json(paths.literature_report, _literature_payload())
    before = {
        path: path.read_text(encoding="utf-8")
        for path in (
            paths.harness_report,
            paths.release_safety_report,
            paths.literature_report,
        )
    }

    dashboard = build_release_dashboard(paths=paths)

    assert dashboard["summary"]["hard_fail_count"] == 2
    assert dashboard["runs"][0]["status"] == "pass"
    assert dashboard["runs"][1]["status"] == "fail"
    assert {
        path: path.read_text(encoding="utf-8")
        for path in (
            paths.harness_report,
            paths.release_safety_report,
            paths.literature_report,
        )
    } == before


def test_harness_hard_fail_count_fails_only_harness_run(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    _write_json(paths.harness_report, _harness_payload(hard_fail_count=2))
    _write_json(paths.release_safety_report, _release_payload(hard_fail_count=0))
    _write_json(paths.literature_report, _literature_payload())

    dashboard = build_release_dashboard(paths=paths)

    assert dashboard["runs"][0]["status"] == "fail"
    assert dashboard["runs"][0]["hard_fail_count"] == 2
    assert dashboard["runs"][1]["status"] == "pass"


def test_empty_json_object_reports_are_invalid_for_each_run(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    _write_json(paths.harness_report, {})
    _write_json(paths.release_safety_report, _release_payload())
    _write_json(paths.literature_report, _literature_payload())

    harness_dashboard = build_release_dashboard(paths=paths)

    _write_json(paths.harness_report, _harness_payload())
    _write_json(paths.release_safety_report, {})
    release_dashboard = build_release_dashboard(paths=paths)

    _write_json(paths.release_safety_report, _release_payload())
    _write_json(paths.literature_report, {})
    literature_dashboard = build_release_dashboard(paths=paths)

    assert harness_dashboard["runs"][0]["status"] == "invalid"
    assert harness_dashboard["runs"][0]["run_id"] == "invalid"
    assert release_dashboard["runs"][1]["status"] == "invalid"
    assert release_dashboard["runs"][1]["run_id"] == "invalid"
    assert literature_dashboard["runs"][2]["status"] == "invalid"
    assert literature_dashboard["runs"][2]["run_id"] == "invalid"


def test_build_release_dashboard_does_not_write_report_files(tmp_path: Path) -> None:
    harness = tmp_path / "harness.json"
    release = tmp_path / "release.json"
    literature = tmp_path / "literature.json"
    harness_payload = {
        "run_id": "harness_test",
        "summary": {"total_cases": 1, "passed": 1, "hard_fail_count": 0},
    }
    release_payload = {
        "report_id": "release_test",
        "version_chain": {
            "agent_policy_version": "agent_policy_test",
            "clinical_safety_policy_version": "safety_test",
            "evidence_index_version": "evidence_test",
            "judge_rubric_version": "rubric_test",
        },
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_previous",
        "hard_fail_summary": {"count": 0, "types": []},
    }
    literature_payload = {
        "run_id": "literature_test",
        "release_decision": "shadow_only",
        "run_level": "L0_shadow",
        "summary": {"claims": 2, "negative_or_conflicting_claims": 1, "isolation_violations": 0},
        "validation_errors": [],
    }
    harness.write_text(json.dumps(harness_payload), encoding="utf-8")
    release.write_text(json.dumps(release_payload), encoding="utf-8")
    literature.write_text(json.dumps(literature_payload), encoding="utf-8")
    before = {
        harness: harness.read_text(encoding="utf-8"),
        release: release.read_text(encoding="utf-8"),
        literature: literature.read_text(encoding="utf-8"),
    }

    build_release_dashboard(
        paths=ReleaseDashboardPaths(
            harness_report=harness,
            release_safety_report=release,
            literature_report=literature,
        )
    )

    assert {
        harness: harness.read_text(encoding="utf-8"),
        release: release.read_text(encoding="utf-8"),
        literature: literature.read_text(encoding="utf-8"),
    } == before
