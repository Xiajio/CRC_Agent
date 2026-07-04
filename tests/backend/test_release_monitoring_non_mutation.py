from __future__ import annotations

from pathlib import Path

from backend.api.services.release_monitoring_store import ReleaseMonitoringStore
from src.services.release_monitoring import ReleaseMonitoringService


INTENT_ID = "release_intent_1"
EXECUTION_ID = "release_exec_1"
ROLLBACK_PLAN_ID = "rollback_plan_1"


def dashboard() -> dict[str, object]:
    return {
        "version_chain": {
            "agent_policy_version": "agent_policy_20260629_0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
            "evidence_index_version": "rag_crc_guideline_20260620",
            "judge_rubric_version": "crc_rubric_v0",
        },
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_20260624_0",
        "summary": {
            "hard_fail_count": 0,
            "literature_isolation_violations": 0,
            "clinical_rag_ingest_enabled": False,
        },
        "runs": [
            {
                "run_id": "release_safety_1",
                "kind": "release_safety",
                "status": "pass",
                "source_path": "reports/release_safety/sentinel.json",
                "hard_fail_count": 0,
            },
            {
                "run_id": "literature_harness_1",
                "kind": "literature_shadow_harness",
                "status": "shadow_only",
                "source_path": "reports/literature/sentinel.json",
                "hard_fail_count": 0,
            },
        ],
    }


def governance() -> dict[str, object]:
    current_dashboard = dashboard()
    return {
        "active_intent": {
            "intent_id": INTENT_ID,
            "target_scope": "feature_flag_candidate",
            "derived_status": "approved",
            "source_release_report_id": "release_safety_1",
            "release_decision_snapshot": "feature_flag_or_pass",
            "rollback_target": "agent_policy_20260624_0",
            "version_chain": current_dashboard["version_chain"],
        },
        "required_approvals": [
            {"role": "release_manager", "status": "approved"},
            {"role": "clinical_safety_reviewer", "status": "approved"},
            {"role": "evidence_reviewer", "status": "approved"},
        ],
        "rollback_plan": {
            "rollback_plan_id": ROLLBACK_PLAN_ID,
            "intent_id": INTENT_ID,
            "rollback_target": "agent_policy_20260624_0",
            "owner": "release_manager",
            "status": "accepted",
            "verification_steps": ["Confirm release report id.", "Run P0 harness."],
            "created_at": "2026-07-03T08:50:00+08:00",
        },
        "integrity": {"status": "verified", "warnings": []},
    }


def execution() -> dict[str, object]:
    flag_state = {
        "flag_name": "doctor_review_cockpit_v0",
        "enabled": True,
        "scope": "feature_flag_candidate",
        "source_intent_id": INTENT_ID,
        "source_execution_id": EXECUTION_ID,
        "rollback_target": "agent_policy_20260624_0",
        "updated_by": "release_manager",
        "updated_at": "2026-07-03T09:00:00+08:00",
    }
    release_result = {
        "result_id": "release_result_1",
        "execution_id": EXECUTION_ID,
        "intent_id": INTENT_ID,
        "action": "release",
        "status": "succeeded",
        "started_at": "2026-07-03T09:00:00+08:00",
        "finished_at": "2026-07-03T09:00:00+08:00",
        "actor": "release_manager",
        "previous_flag_state": None,
        "new_flag_state": flag_state,
        "failure_reason": None,
    }
    return {
        "feature_flag_state": flag_state,
        "results": [release_result],
        "integrity": {"status": "verified", "warnings": []},
    }


def snapshot_protected_files(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_path = relative.as_posix()
        parts = set(relative.parts)
        if relative_path.startswith("reports/release_monitoring/"):
            continue
        if ".git" in parts or "__pycache__" in parts:
            continue
        snapshot[relative_path] = path.read_text(encoding="utf-8")
    return snapshot


def test_monitoring_writes_only_to_monitoring_root(tmp_path: Path) -> None:
    protected_paths = {
        "execution": tmp_path / "reports" / "release_execution" / "sentinel.json",
        "governance": tmp_path / "reports" / "release_governance" / "sentinel.json",
        "harness": tmp_path / "reports" / "harness" / "sentinel.json",
        "release": tmp_path / "reports" / "release_safety" / "sentinel.json",
        "literature": tmp_path / "reports" / "literature" / "sentinel.json",
        "safety": tmp_path / "config" / "safety_policy.yaml",
        "monitoring_config": (
            tmp_path / "config" / "release_monitoring" / "sentinel.txt"
        ),
        "prompt": tmp_path / "src" / "prompts" / "decision_prompts.py",
        "other": tmp_path / "reports" / "other" / "sentinel.json",
    }
    for label, path in protected_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}: original\n", encoding="utf-8")
    before = snapshot_protected_files(tmp_path)
    assert "config/release_monitoring/sentinel.txt" in before
    monitor = ReleaseMonitoringService(
        store=ReleaseMonitoringStore(tmp_path / "reports" / "release_monitoring"),
        execution_loader=execution,
        governance_loader=governance,
        dashboard_loader=dashboard,
        now=lambda: "2026-07-03T11:00:00+08:00",
    )

    monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status="pass",
        observed_by="release_manager",
        summary="P0 harness replay passed after release execution.",
        evidence_refs=["reports/harness/sentinel.json"],
        metrics={"passed": 1, "failed": 0, "hard_fail_count": 0},
        idempotency_key="p0-harness-replay-pass-1",
    )

    assert snapshot_protected_files(tmp_path) == before
    assert (tmp_path / "reports" / "release_monitoring" / "checks").is_dir()
    written_files = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "reports" / "release_monitoring").rglob("*")
        if path.is_file()
    )
    assert written_files
    assert all(path.startswith("reports/release_monitoring/") for path in written_files)
