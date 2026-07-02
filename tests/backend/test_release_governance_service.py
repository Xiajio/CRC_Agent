from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.release_governance_store import ReleaseGovernanceStore
from src.services.release_governance import (
    GovernanceConflictError,
    GovernanceValidationError,
    ReleaseGovernanceService,
)


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
                "source_path": (
                    "reports/release_safety/release_safety_20260629_001.json"
                ),
                "hard_fail_count": 0,
            },
            {
                "run_id": "literature_harness_20260630_001",
                "kind": "literature_shadow_harness",
                "status": "shadow_only",
                "source_path": (
                    "reports/literature/literature_harness_20260630_001.json"
                ),
                "hard_fail_count": 0,
            },
        ],
    }


def service(
    tmp_path: Path,
    payload: dict[str, object] | None = None,
) -> ReleaseGovernanceService:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    return ReleaseGovernanceService(
        store=store,
        dashboard_loader=lambda: payload or dashboard(),
        now=lambda: "2026-07-02T00:00:00+08:00",
    )


def assert_audit_only_read_model(model: dict[str, object]) -> None:
    assert model["runtime"]["mode"] == "audit_only"
    assert "dashboard_snapshot" in model
    assert "active_intent" in model
    assert "audit_events" in model


def test_read_governance_before_writes_returns_empty_audit_only_model(
    tmp_path: Path,
) -> None:
    model = service(tmp_path).read_governance()

    assert model["active_intent"] is None
    assert model["required_approvals"][0]["role"] == "release_manager"
    assert model["integrity"] == {"status": "verified", "warnings": []}
    disabled_by_id = {
        action["id"]: action for action in model["disabled_execution_actions"]
    }
    assert disabled_by_id["execute_release"]["disabled"] is True
    assert disabled_by_id["execute_rollback"]["disabled"] is True
    assert "Step 12" in disabled_by_id["execute_release"]["reason"]
    assert "later execution-path" in disabled_by_id["execute_rollback"]["reason"]


def test_create_intent_from_dashboard_snapshot(tmp_path: Path) -> None:
    governance = service(tmp_path)

    model = governance.create_intent(
        requested_by="admin_operator",
        target_scope="shadow",
        status="pending_approval",
        reason="Prepare audited governance.",
    )

    assert_audit_only_read_model(model)
    assert (
        model["active_intent"]["source_release_report_id"]
        == "release_safety_20260629_001"
    )
    assert model["active_intent"]["rollback_target"] == "agent_policy_20260624_0"
    assert model["active_intent"]["blocking_summary"]["hard_fail_count"] == 0
    assert model["audit_events"][0]["event_type"] == "intent_created"


def test_hard_fail_blocks_pending_approval_intent(tmp_path: Path) -> None:
    payload = dashboard()
    payload["summary"] = {**payload["summary"], "hard_fail_count": 1}

    with pytest.raises(
        GovernanceValidationError,
        match="hard fails prevent pending approval",
    ):
        service(tmp_path, payload).create_intent(
            requested_by="admin_operator",
            target_scope="shadow",
            status="pending_approval",
            reason="Prepare audited governance.",
        )


def test_duplicate_active_intent_is_rejected(tmp_path: Path) -> None:
    governance = service(tmp_path)
    governance.create_intent(
        requested_by="admin_operator",
        target_scope="shadow",
        status="pending_approval",
        reason="Prepare audited governance.",
    )

    with pytest.raises(
        GovernanceConflictError,
        match="active intent already exists",
    ):
        governance.create_intent(
            requested_by="admin_operator",
            target_scope="shadow",
            status="pending_approval",
            reason="Prepare audited governance.",
        )


def test_feature_flag_candidate_requires_shadow_literature(
    tmp_path: Path,
) -> None:
    payload = dashboard()
    payload["runs"] = [
        run
        if run["kind"] != "literature_shadow_harness"
        else {**run, "status": "fail"}
        for run in payload["runs"]
    ]

    with pytest.raises(
        GovernanceValidationError,
        match="literature run must be shadow_only",
    ):
        service(tmp_path, payload).create_intent(
            requested_by="admin_operator",
            target_scope="feature_flag_candidate",
            status="pending_approval",
            reason="Prepare audited governance.",
        )


def test_record_approval_and_rollback_plan_derives_read_model(
    tmp_path: Path,
) -> None:
    governance = service(tmp_path)
    model = governance.create_intent(
        requested_by="admin_operator",
        target_scope="shadow",
        status="pending_approval",
        reason="Prepare audited governance.",
    )
    assert_audit_only_read_model(model)
    intent_id = model["active_intent"]["intent_id"]

    model = governance.record_approval(
        intent_id=intent_id,
        approver_role="release_manager",
        decision="approve",
        reason="P0 hard fails are zero.",
        signed_by="release_admin",
    )
    assert_audit_only_read_model(model)
    assert model["active_intent"]["intent_id"] == intent_id

    model = governance.record_approval(
        intent_id=model["active_intent"]["intent_id"],
        approver_role="clinical_safety_reviewer",
        decision="approve",
        reason="Clinical safety gates are locked.",
        signed_by="clinical_admin",
    )
    assert_audit_only_read_model(model)

    model = governance.record_rollback_plan(
        intent_id=model["active_intent"]["intent_id"],
        owner="release_manager",
        status="accepted",
        verification_steps=[
            "Confirm the active release report id.",
            "Run P0 harness before any future rollback execution.",
        ],
    )
    assert_audit_only_read_model(model)

    assert model["active_intent"]["derived_status"] == "approved"
    assert {item["status"] for item in model["required_approvals"]} == {
        "approved"
    }
    assert (
        model["rollback_plan"]["rollback_target"] == "agent_policy_20260624_0"
    )
    assert [event["event_type"] for event in model["audit_events"]] == [
        "intent_created",
        "approval_recorded",
        "approval_recorded",
        "rollback_plan_recorded",
    ]


def test_rejection_prevents_approved_derived_status(tmp_path: Path) -> None:
    governance = service(tmp_path)
    model = governance.create_intent(
        requested_by="admin_operator",
        target_scope="shadow",
        status="pending_approval",
        reason="Prepare audited governance.",
    )
    assert_audit_only_read_model(model)

    model = governance.record_approval(
        intent_id=model["active_intent"]["intent_id"],
        approver_role="release_manager",
        decision="reject",
        reason="Release window is closed.",
        signed_by="release_admin",
    )

    assert_audit_only_read_model(model)
    assert model["active_intent"]["derived_status"] == "rejected"


def test_cancel_intent_is_derived_without_deleting_records(
    tmp_path: Path,
) -> None:
    governance = service(tmp_path)
    model = governance.create_intent(
        requested_by="admin_operator",
        target_scope="shadow",
        status="pending_approval",
        reason="Prepare audited governance.",
    )
    assert_audit_only_read_model(model)

    model = governance.cancel_intent(
        intent_id=model["active_intent"]["intent_id"],
        actor="admin_operator",
        reason="Release window closed.",
    )

    assert_audit_only_read_model(model)
    assert model["active_intent"] is None
    assert model["intents"][0]["derived_status"] == "cancelled"
