from __future__ import annotations

from pathlib import Path

from backend.api.services.release_governance_store import ReleaseGovernanceStore
from src.services.release_governance import ReleaseGovernanceService


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


def read_if_exists(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def test_governance_writes_only_to_governance_root(tmp_path: Path) -> None:
    protected_paths = {
        "safety": tmp_path / "config" / "safety_policy.yaml",
        "harness": tmp_path / "reports" / "harness" / "harness_20260629_001.json",
        "release": tmp_path / "reports" / "release_safety" / "release_safety_20260629_001.json",
        "literature": tmp_path / "reports" / "literature" / "literature_harness_20260630_001.json",
        "prompt": tmp_path / "src" / "prompts" / "decision_prompts.py",
    }
    for label, path in protected_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}: original\n", encoding="utf-8")
    before = {label: read_if_exists(path) for label, path in protected_paths.items()}
    governance = ReleaseGovernanceService(
        store=ReleaseGovernanceStore(tmp_path / "reports" / "release_governance"),
        dashboard_loader=dashboard,
        now=lambda: "2026-07-02T00:00:00+08:00",
    )

    model = governance.create_intent(
        requested_by="admin_operator",
        target_scope="shadow",
        status="pending_approval",
        reason="Prepare audited governance.",
    )
    intent_id = model["active_intent"]["intent_id"]
    governance.record_approval(
        intent_id=intent_id,
        approver_role="release_manager",
        decision="approve",
        reason="Release manager approval.",
        signed_by="release_admin",
    )
    governance.record_rollback_plan(
        intent_id=intent_id,
        owner="release_manager",
        status="accepted",
        verification_steps=[
            "Confirm release report id.",
            "Run P0 harness before future rollback execution.",
        ],
    )

    assert {label: read_if_exists(path) for label, path in protected_paths.items()} == before
    written_files = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "reports" / "release_governance").rglob("*")
        if path.is_file()
    )
    assert written_files == [
        "reports/release_governance/approvals/release_approval_release_intent_release_safety_20260629_001_a1661529_release_manager_23cda008.json",
        "reports/release_governance/audit/release_audit_20260702.jsonl",
        "reports/release_governance/intents/release_intent_release_safety_20260629_001_a1661529.json",
        "reports/release_governance/rollback_plans/rollback_plan_release_intent_release_safety_20260629_001_a1661529_1b00f364.json",
    ]
