from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.release_closure_store import (
    ReleaseClosureIntegrityError,
    ReleaseClosureStore,
)
from src.contracts.release_closure import (
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    make_release_closure_id,
    make_release_evidence_package_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
RELEASE_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"


def make_pair(idempotency_key: str = "close-1") -> tuple[ReleaseClosureRecord, ReleaseEvidencePackage]:
    closure_id = make_release_closure_id(RELEASE_EXECUTION_ID, idempotency_key)
    package_id = make_release_evidence_package_id(closure_id)
    closure = ReleaseClosureRecord(
        closure_id=closure_id,
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        rollback_execution_id=None,
        closure_status="accepted",
        closed_by="release_manager",
        closed_at="2026-07-07T10:00:00+08:00",
        rationale="Required checks passed.",
        monitoring_snapshot_hash="sha256:" + "a" * 64,
        dashboard_snapshot_hash="sha256:" + "b" * 64,
        governance_snapshot_hash="sha256:" + "c" * 64,
        execution_snapshot_hash="sha256:" + "d" * 64,
        required_check_ids=["release_monitor_check_1"],
        acknowledged_alert_ids=[],
        unresolved_alert_ids=[],
        rollback_trigger_candidate_id=None,
        evidence_package_id=package_id,
        idempotency_key=idempotency_key,
    )
    package = ReleaseEvidencePackage(
        package_id=package_id,
        closure_id=closure_id,
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        rollback_execution_id=None,
        generated_by="release_manager",
        generated_at="2026-07-07T10:00:00+08:00",
        closure_status="accepted",
        summary="Release observation period closed.",
        source_refs=[
            "GET /api/admin/release-dashboard",
            "GET /api/admin/release-governance",
            "GET /api/admin/release-execution",
            "GET /api/admin/release-monitoring",
        ],
        artifact_refs=[f"reports/release_closure/closures/{closure_id}.json"],
        snapshot_hashes={
            "dashboard": closure.dashboard_snapshot_hash,
            "governance": closure.governance_snapshot_hash,
            "execution": closure.execution_snapshot_hash,
            "monitoring": closure.monitoring_snapshot_hash,
        },
    )
    return closure, package


def test_empty_store_read_is_verified_and_read_only(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)

    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    state = store.read_state()
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    assert state.integrity == {"status": "verified", "warnings": []}
    assert state.closures == []
    assert state.evidence_packages == []
    assert state.audit_events == []
    assert before == after


def test_write_closure_creates_closure_package_and_audit_events(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()

    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)
    state = store.read_state()

    assert [item.closure_id for item in state.closures] == [closure.closure_id]
    assert [item.package_id for item in state.evidence_packages] == [package.package_id]
    assert [event.event_type for event in state.audit_events] == [
        "closure_recorded",
        "evidence_package_generated",
    ]


def test_idempotent_replay_returns_existing_pair(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)

    match = store.find_closure_by_idempotency_key("close-1")

    assert match is not None
    assert match.closure.closure_id == closure.closure_id
    store.assert_idempotent_closure_matches(closure, package)


def test_idempotency_payload_mismatch_fails(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)
    changed, changed_package = make_pair()
    changed = ReleaseClosureRecord(**{**changed.to_dict(), "rationale": "Changed rationale."})

    with pytest.raises(FileExistsError, match="idempotency payload mismatch"):
        store.assert_idempotent_closure_matches(changed, changed_package)


def test_audit_tampering_blocks_writes(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)
    audit_file = next((tmp_path / "audit").glob("release_closure_*.jsonl"))
    audit_file.write_text(audit_file.read_text(encoding="utf-8").replace("closure_recorded", "closure_read"), encoding="utf-8")

    with pytest.raises(ReleaseClosureIntegrityError, match="release closure integrity failed"):
        store.write_closure_with_package(*make_pair("close-2"), timestamp="2026-07-07T10:05:00+08:00")
