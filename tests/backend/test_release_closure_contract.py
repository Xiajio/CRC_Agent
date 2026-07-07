from __future__ import annotations

import pytest

from src.contracts.release_closure import (
    GENESIS_CLOSURE_EVENT_HASH,
    ReleaseClosureAuditEvent,
    ReleaseClosureGate,
    ReleaseClosureGateCheck,
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    build_release_closure_audit_event,
    canonical_closure_payload_hash,
    make_release_closure_event_id,
    make_release_closure_id,
    make_release_evidence_package_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
RELEASE_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"


def make_closure() -> ReleaseClosureRecord:
    closure_id = make_release_closure_id(RELEASE_EXECUTION_ID, "close-1")
    return ReleaseClosureRecord(
        closure_id=closure_id,
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        rollback_execution_id=None,
        closure_status="accepted",
        closed_by="release_manager",
        closed_at="2026-07-07T10:00:00+08:00",
        rationale="Required checks passed and no active critical alerts remain.",
        monitoring_snapshot_hash="sha256:" + "a" * 64,
        dashboard_snapshot_hash="sha256:" + "b" * 64,
        governance_snapshot_hash="sha256:" + "c" * 64,
        execution_snapshot_hash="sha256:" + "d" * 64,
        required_check_ids=["release_monitor_check_1"],
        acknowledged_alert_ids=[],
        unresolved_alert_ids=[],
        rollback_trigger_candidate_id=None,
        evidence_package_id=make_release_evidence_package_id(closure_id),
        idempotency_key="close-1",
    )


def test_closure_contracts_round_trip_to_dict() -> None:
    closure = make_closure()
    package = ReleaseEvidencePackage(
        package_id=closure.evidence_package_id,
        closure_id=closure.closure_id,
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        rollback_execution_id=None,
        generated_by="release_manager",
        generated_at="2026-07-07T10:00:00+08:00",
        closure_status="accepted",
        summary="Release observation period closed after checks passed.",
        source_refs=[
            "GET /api/admin/release-dashboard",
            "GET /api/admin/release-governance",
            "GET /api/admin/release-execution",
            "GET /api/admin/release-monitoring",
        ],
        artifact_refs=[
            f"reports/release_closure/closures/{closure.closure_id}.json",
        ],
        snapshot_hashes={
            "dashboard": closure.dashboard_snapshot_hash,
            "governance": closure.governance_snapshot_hash,
            "execution": closure.execution_snapshot_hash,
            "monitoring": closure.monitoring_snapshot_hash,
        },
    )
    gate = ReleaseClosureGate(
        allowed=True,
        status="ready_to_close",
        reasons=[],
        checks=[
            ReleaseClosureGateCheck(
                name="required_monitoring_checks_complete",
                status="pass",
                reason="All Step 14 required checks are present.",
            )
        ],
    )
    event = build_release_closure_audit_event(
        event_id=make_release_closure_event_id(
            RELEASE_EXECUTION_ID,
            "closure_recorded",
            "2026-07-07T10:00:00+08:00",
        ),
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        event_type="closure_recorded",
        actor="release_manager",
        timestamp="2026-07-07T10:00:00+08:00",
        payload=closure.to_dict(),
        previous_event_hash=GENESIS_CLOSURE_EVENT_HASH,
    )

    assert closure.to_dict()["closure_status"] == "accepted"
    assert package.to_dict()["closure_id"] == closure.closure_id
    assert gate.to_dict()["status"] == "ready_to_close"
    assert event.to_dict()["event_hash"].startswith("sha256:")


@pytest.mark.parametrize("closure_status", ["pending", "failed", ""])
def test_closure_rejects_unknown_status(closure_status: str) -> None:
    payload = make_closure().to_dict()
    payload["closure_status"] = closure_status

    with pytest.raises(ValueError, match="closure_status must be one of"):
        ReleaseClosureRecord(**payload)


def test_accepted_closure_rejects_unresolved_alerts() -> None:
    payload = make_closure().to_dict()
    payload["unresolved_alert_ids"] = ["release_monitor_alert_1"]

    with pytest.raises(
        ValueError,
        match="accepted closure cannot contain unresolved alerts",
    ):
        ReleaseClosureRecord(**payload)


def test_rolled_back_closure_requires_rollback_execution_id() -> None:
    payload = make_closure().to_dict()
    payload["closure_status"] = "rolled_back"
    payload["rollback_execution_id"] = None

    with pytest.raises(ValueError, match="rollback_execution_id is required"):
        ReleaseClosureRecord(**payload)


def test_hash_rejects_forbidden_payload_keys() -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        canonical_closure_payload_hash({"patient_id": "p-1"})
