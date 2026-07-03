from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.release_execution_store import ReleaseExecutionStore
from src.services.release_execution import (
    ReleaseExecutionPreflightError,
    ReleaseExecutionService,
)


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


def governance(
    target_scope: str = "feature_flag_candidate",
    approved: bool = True,
) -> dict[str, object]:
    required = [
        {
            "role": "release_manager",
            "status": "approved" if approved else "missing",
            "latest_decision": "approve" if approved else None,
        },
        {
            "role": "clinical_safety_reviewer",
            "status": "approved" if approved else "missing",
            "latest_decision": "approve" if approved else None,
        },
        {
            "role": "evidence_reviewer",
            "status": "approved" if approved else "missing",
            "latest_decision": "approve" if approved else None,
        },
    ]
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
            "target_scope": target_scope,
            "derived_status": "approved" if approved else "pending_approval",
            "source_release_report_id": "release_safety_20260629_001",
            "release_decision_snapshot": "feature_flag_or_pass",
            "rollback_target": "agent_policy_20260624_0",
            "version_chain": current_dashboard["version_chain"],
        },
        "required_approvals": required,
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


def service(
    tmp_path: Path,
    gov: dict[str, object] | None = None,
    dash: dict[str, object] | None = None,
) -> ReleaseExecutionService:
    return ReleaseExecutionService(
        store=ReleaseExecutionStore(tmp_path / "reports" / "release_execution"),
        governance_loader=lambda: gov if gov is not None else governance(),
        dashboard_loader=lambda: dash if dash is not None else dashboard(),
        now=lambda: "2026-07-03T09:00:00+08:00",
    )


def test_read_execution_returns_preflight_ready_for_approved_feature_flag_candidate(tmp_path: Path) -> None:
    model = service(tmp_path).read_execution()

    assert model["preflight"]["release"]["allowed"] is True
    assert model["preflight"]["release"]["reasons"] == []
    assert model["preflight"]["rollback"]["allowed"] is False


def test_release_blocks_shadow_scope(tmp_path: Path) -> None:
    app = service(tmp_path, gov=governance(target_scope="shadow"))

    with pytest.raises(ReleaseExecutionPreflightError, match="target_scope must be feature_flag_candidate"):
        app.execute_release(
            intent_id=INTENT_ID,
            requested_by="release_manager",
            idempotency_key="release-1",
            reason="Approved release.",
            expected_rollback_plan_id=ROLLBACK_PLAN_ID,
        )


def test_release_blocks_missing_approvals(tmp_path: Path) -> None:
    app = service(tmp_path, gov=governance(approved=False))

    with pytest.raises(ReleaseExecutionPreflightError, match="required approvals are incomplete"):
        app.execute_release(
            intent_id=INTENT_ID,
            requested_by="release_manager",
            idempotency_key="release-1",
            reason="Approved release.",
            expected_rollback_plan_id=ROLLBACK_PLAN_ID,
        )


def test_release_blocks_dashboard_drift(tmp_path: Path) -> None:
    drifted = dashboard()
    drifted["rollback_target"] = "agent_policy_20260620_0"
    app = service(tmp_path, dash=drifted)

    with pytest.raises(ReleaseExecutionPreflightError, match="dashboard rollback_target drifted"):
        app.execute_release(
            intent_id=INTENT_ID,
            requested_by="release_manager",
            idempotency_key="release-1",
            reason="Approved release.",
            expected_rollback_plan_id=ROLLBACK_PLAN_ID,
        )


def test_execute_release_then_rollback(tmp_path: Path) -> None:
    app = service(tmp_path)

    released = app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    assert released["feature_flag_state"]["enabled"] is True
    assert released["preflight"]["release"]["allowed"] is False
    assert released["preflight"]["rollback"]["allowed"] is True

    rolled_back = app.execute_rollback(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="rollback-1",
        reason="Rollback to approved target.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    assert rolled_back["feature_flag_state"]["enabled"] is False
    assert rolled_back["preflight"]["rollback"]["allowed"] is False


def test_release_is_idempotent_for_same_key(tmp_path: Path) -> None:
    app = service(tmp_path)
    first = app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    second = app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )

    assert second["results"] == first["results"]
