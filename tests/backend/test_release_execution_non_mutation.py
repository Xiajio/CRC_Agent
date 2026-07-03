from __future__ import annotations

from pathlib import Path

from backend.api.services.release_execution_store import ReleaseExecutionStore
from src.services.release_execution import ReleaseExecutionService

INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
ROLLBACK_PLAN_ID = "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"


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
                "run_id": "harness_20260629_001",
                "kind": "p0_crc_harness",
                "status": "pass",
                "source_path": "reports/harness/harness_20260629_001.json",
                "hard_fail_count": 0,
            },
            {
                "run_id": "release_safety_20260629_001",
                "kind": "release_safety",
                "status": "pass",
                "source_path": "reports/release_safety/release_safety_20260629_001.json",
                "hard_fail_count": 0,
            },
            {
                "run_id": "literature_harness_20260630_001",
                "kind": "literature_shadow_harness",
                "status": "shadow_only",
                "source_path": "reports/literature/literature_harness_20260630_001.json",
                "hard_fail_count": 0,
            },
        ],
    }


def governance() -> dict[str, object]:
    current_dashboard = dashboard()
    return {
        "dashboard_snapshot": {
            "version_chain": current_dashboard["version_chain"],
            "release_decision": "feature_flag_or_pass",
            "rollback_target": "agent_policy_20260624_0",
            "hard_fail_count": 0,
            "literature_status": "shadow_only",
        },
        "active_intent": {
            "intent_id": INTENT_ID,
            "target_scope": "feature_flag_candidate",
            "derived_status": "approved",
            "source_release_report_id": "release_safety_20260629_001",
            "release_decision_snapshot": "feature_flag_or_pass",
            "rollback_target": "agent_policy_20260624_0",
            "version_chain": current_dashboard["version_chain"],
        },
        "required_approvals": [
            {"role": "release_manager", "status": "approved", "latest_decision": "approve"},
            {"role": "clinical_safety_reviewer", "status": "approved", "latest_decision": "approve"},
            {"role": "evidence_reviewer", "status": "approved", "latest_decision": "approve"},
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


def read_if_exists(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def test_release_and_rollback_write_only_execution_root(tmp_path: Path) -> None:
    protected_paths = {
        "governance": tmp_path / "reports" / "release_governance" / "intents" / f"{INTENT_ID}.json",
        "harness": tmp_path / "reports" / "harness" / "harness_20260629_001.json",
        "release": tmp_path / "reports" / "release_safety" / "release_safety_20260629_001.json",
        "literature": tmp_path / "reports" / "literature" / "literature_harness_20260630_001.json",
        "safety": tmp_path / "config" / "safety_policy.yaml",
        "prompt": tmp_path / "src" / "prompts" / "decision_prompts.py",
    }
    for label, path in protected_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}: original\n", encoding="utf-8")
    before = {label: read_if_exists(path) for label, path in protected_paths.items()}
    app = ReleaseExecutionService(
        store=ReleaseExecutionStore(tmp_path / "reports" / "release_execution"),
        governance_loader=governance,
        dashboard_loader=dashboard,
        now=lambda: "2026-07-03T09:00:00+08:00",
    )

    app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    app.execute_rollback(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="rollback-1",
        reason="Rollback to approved target.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )

    assert {label: read_if_exists(path) for label, path in protected_paths.items()} == before
    written = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "reports" / "release_execution").rglob("*")
        if path.is_file()
    )
    assert all(path.startswith("reports/release_execution/") for path in written)
    assert "reports/release_execution/feature_flags/current.json" in written
