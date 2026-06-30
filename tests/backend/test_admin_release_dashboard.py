from __future__ import annotations

import json
from pathlib import Path

from backend.api.services.admin_release_dashboard import (
    ReleaseDashboardPaths,
    build_release_dashboard,
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
        "run_level": "L0_shadow",
        "summary": {"claims": 2, "negative_or_conflicting_claims": 1, "isolation_violations": 0},
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
