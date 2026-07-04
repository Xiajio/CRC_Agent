from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.api.services.release_monitoring_store import (
    ReleaseMonitoringIntegrityError,
    ReleaseMonitoringStore,
)
from src.contracts.release_monitoring import (
    ReleaseMonitoringAcknowledgement,
    ReleaseMonitoringCheck,
    make_monitoring_acknowledgement_id,
    make_monitoring_alert_id,
    make_monitoring_check_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"


def make_check(key: str = "idem-1", status: str = "pass") -> ReleaseMonitoringCheck:
    return ReleaseMonitoringCheck(
        check_id=make_monitoring_check_id(EXECUTION_ID, "p0_harness_replay", key),
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status=status,
        observed_by="release_manager",
        observed_at="2026-07-03T11:00:00+08:00",
        summary="P0 harness replay completed after release execution.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"passed": 5, "failed": 0, "hard_fail_count": 0},
        idempotency_key=key,
    )


def make_ack(alert_id: str) -> ReleaseMonitoringAcknowledgement:
    return ReleaseMonitoringAcknowledgement(
        acknowledgement_id=make_monitoring_acknowledgement_id(alert_id, "ack-1"),
        alert_id=alert_id,
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        acknowledged_by="release_manager",
        acknowledged_at="2026-07-03T11:05:00+08:00",
        disposition="investigating",
        reason="Checking evidence before rollback execution.",
    )


def test_empty_store_read_is_verified_and_does_not_create_files(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "release_monitoring"
    store = ReleaseMonitoringStore(root)

    state = store.read_state()

    assert state.integrity == {"status": "verified", "warnings": []}
    assert state.checks == []
    assert state.acknowledgements == []
    assert state.audit_events == []
    assert not root.exists()


def test_write_check_creates_check_and_audit_event(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "release_monitoring"
    store = ReleaseMonitoringStore(root)
    check = make_check()

    store.write_check(check, timestamp=check.observed_at)

    state = store.read_state()
    assert [item.check_id for item in state.checks] == [check.check_id]
    assert [event.event_type for event in state.audit_events] == ["check_recorded"]
    assert (root / "checks" / f"{check.check_id}.json").exists()


def test_write_acknowledgement_creates_ack_and_audit_event(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "release_monitoring"
    store = ReleaseMonitoringStore(root)
    alert_id = make_monitoring_alert_id(EXECUTION_ID, "post_release_check_failed", "p0_harness_replay")
    acknowledgement = make_ack(alert_id)

    store.write_acknowledgement(acknowledgement, timestamp=acknowledgement.acknowledged_at)

    state = store.read_state()
    assert [item.acknowledgement_id for item in state.acknowledgements] == [acknowledgement.acknowledgement_id]
    assert [event.event_type for event in state.audit_events] == ["alert_acknowledged"]


def test_idempotent_check_replay_returns_existing_match(tmp_path: Path) -> None:
    store = ReleaseMonitoringStore(tmp_path / "reports" / "release_monitoring")
    check = make_check()
    store.write_check(check, timestamp=check.observed_at)

    match = store.find_check_by_idempotency_key(check.check_type, check.idempotency_key)

    assert match is not None
    assert match.check.check_id == check.check_id
    store.assert_idempotent_check_matches(check)


def test_idempotency_key_payload_mismatch_fails(tmp_path: Path) -> None:
    store = ReleaseMonitoringStore(tmp_path / "reports" / "release_monitoring")
    check = make_check()
    store.write_check(check, timestamp=check.observed_at)
    changed = ReleaseMonitoringCheck(**{**check.to_dict(), "summary": "Different summary."})

    with pytest.raises(ReleaseMonitoringIntegrityError, match="idempotency key payload mismatch"):
        store.assert_idempotent_check_matches(changed)


def test_tampered_check_blocks_writes(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "release_monitoring"
    store = ReleaseMonitoringStore(root)
    check = make_check()
    store.write_check(check, timestamp=check.observed_at)
    path = root / "checks" / f"{check.check_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "fail"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReleaseMonitoringIntegrityError, match="release monitoring integrity failed"):
        store.write_check(make_check("idem-2"), timestamp="2026-07-03T11:10:00+08:00")


def test_symlink_root_fails_integrity(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "reports" / "release_monitoring"
    root.parent.mkdir(parents=True)
    try:
        root.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is not available")
        raise

    state = ReleaseMonitoringStore(root).read_state()

    assert state.integrity["status"] == "failed"
    assert "symlink" in state.integrity["warnings"][0]
