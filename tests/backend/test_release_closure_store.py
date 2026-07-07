from __future__ import annotations

from pathlib import Path

import pytest

import backend.api.services.release_closure_store as release_closure_store_module
from backend.api.services.release_closure_store import (
    ReleaseClosureIntegrityError,
    ReleaseClosureStore,
)
from src.contracts.release_closure import (
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    build_release_closure_audit_event,
    make_release_closure_event_id,
    make_release_closure_id,
    make_release_evidence_package_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
RELEASE_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"


def make_pair(
    idempotency_key: str = "close-1",
    *,
    release_execution_id: str = RELEASE_EXECUTION_ID,
) -> tuple[ReleaseClosureRecord, ReleaseEvidencePackage]:
    closure_id = make_release_closure_id(release_execution_id, idempotency_key)
    package_id = make_release_evidence_package_id(closure_id)
    closure = ReleaseClosureRecord(
        closure_id=closure_id,
        intent_id=INTENT_ID,
        release_execution_id=release_execution_id,
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


def append_artifacts_with_audit(
    tmp_path: Path,
    closure: ReleaseClosureRecord,
    package: ReleaseEvidencePackage,
    *,
    timestamp: str,
) -> None:
    closures_dir = tmp_path / "closures"
    packages_dir = tmp_path / "packages"
    closures_dir.mkdir(parents=True, exist_ok=True)
    packages_dir.mkdir(parents=True, exist_ok=True)
    (closures_dir / f"{closure.closure_id}.json").write_text(
        release_closure_store_module.json.dumps(
            closure.to_dict(),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (packages_dir / f"{package.package_id}.json").write_text(
        release_closure_store_module.json.dumps(
            package.to_dict(),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    audit_dir = tmp_path / "audit"
    audit_path = audit_dir / "release_closure_20260707.jsonl"
    previous_event_hash = "sha256:GENESIS"
    if audit_path.exists():
        audit_lines = [
            release_closure_store_module.json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        previous_event_hash = audit_lines[-1]["event_hash"]
    closure_event = build_release_closure_audit_event(
        event_id=make_release_closure_event_id(
            closure.release_execution_id,
            "closure_recorded",
            f"{timestamp}#{closure.closure_id}",
        ),
        intent_id=closure.intent_id,
        release_execution_id=closure.release_execution_id,
        event_type="closure_recorded",
        actor=closure.closed_by,
        timestamp=timestamp,
        payload=closure.to_dict(),
        previous_event_hash=previous_event_hash,
    )
    package_event = build_release_closure_audit_event(
        event_id=make_release_closure_event_id(
            package.release_execution_id,
            "evidence_package_generated",
            f"{timestamp}#{package.package_id}",
        ),
        intent_id=package.intent_id,
        release_execution_id=package.release_execution_id,
        event_type="evidence_package_generated",
        actor=package.generated_by,
        timestamp=timestamp,
        payload=package.to_dict(),
        previous_event_hash=closure_event.event_hash,
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(
            release_closure_store_module.json.dumps(
                closure_event.to_dict(),
                sort_keys=True,
            )
        )
        handle.write("\n")
        handle.write(
            release_closure_store_module.json.dumps(
                package_event.to_dict(),
                sort_keys=True,
            )
        )
        handle.write("\n")


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


def test_write_rejects_second_closure_for_same_release_with_different_key(
    tmp_path: Path,
) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair("close-1")
    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)
    second_closure, second_package = make_pair("close-2")

    with pytest.raises(
        ReleaseClosureIntegrityError,
        match="release execution already has a closure",
    ):
        store.write_closure_with_package(
            second_closure,
            second_package,
            timestamp="2026-07-07T10:05:00+08:00",
        )

    state = store.read_state()
    assert [item.closure_id for item in state.closures] == [closure.closure_id]
    assert [item.package_id for item in state.evidence_packages] == [package.package_id]


def test_read_state_detects_multiple_closure_package_pairs_for_release_execution(
    tmp_path: Path,
) -> None:
    first_closure, first_package = make_pair("close-1")
    second_closure, second_package = make_pair("close-2")
    second_closure = ReleaseClosureRecord(
        **{
            **second_closure.to_dict(),
            "closed_at": "2026-07-07T10:05:00+08:00",
            "rationale": "Duplicate corrupted closure.",
        }
    )
    second_package = ReleaseEvidencePackage(
        **{
            **second_package.to_dict(),
            "generated_at": "2026-07-07T10:05:00+08:00",
            "summary": "Duplicate corrupted package.",
        }
    )
    append_artifacts_with_audit(
        tmp_path,
        first_closure,
        first_package,
        timestamp=first_closure.closed_at,
    )
    append_artifacts_with_audit(
        tmp_path,
        second_closure,
        second_package,
        timestamp=second_closure.closed_at,
    )

    state = ReleaseClosureStore(tmp_path).read_state()

    assert state.integrity["status"] == "failed"
    assert any(
        "multiple closure artifacts for release execution" in warning
        for warning in state.integrity["warnings"]
    )
    assert any(
        "multiple evidence package artifacts for release execution" in warning
        for warning in state.integrity["warnings"]
    )


def test_audit_tampering_blocks_writes(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)
    audit_file = next((tmp_path / "audit").glob("release_closure_*.jsonl"))
    audit_file.write_text(audit_file.read_text(encoding="utf-8").replace("closure_recorded", "closure_read"), encoding="utf-8")

    with pytest.raises(ReleaseClosureIntegrityError, match="release closure integrity failed"):
        store.write_closure_with_package(*make_pair("close-2"), timestamp="2026-07-07T10:05:00+08:00")


def test_failed_second_audit_append_rolls_back_without_poisoned_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    original_dumps = release_closure_store_module.json.dumps
    calls = 0

    def fail_after_first_event(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated second append failure")
        return original_dumps(*args, **kwargs)

    monkeypatch.setattr(release_closure_store_module.json, "dumps", fail_after_first_event)

    with pytest.raises(OSError, match="simulated second append failure"):
        store.write_closure_with_package(closure, package, timestamp=closure.closed_at)

    state = store.read_state()
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    assert state.integrity == {"status": "verified", "warnings": []}
    assert state.closures == []
    assert state.evidence_packages == []
    assert state.audit_events == []
    assert before == after


def test_read_state_detects_tampered_package_even_when_audit_hash_matches(
    tmp_path: Path,
) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)

    tampered_package_payload = package.to_dict()
    tampered_package_payload["snapshot_hashes"] = {
        **tampered_package_payload["snapshot_hashes"],
        "monitoring": "sha256:" + "e" * 64,
    }
    package_path = tmp_path / "packages" / f"{package.package_id}.json"
    package_path.write_text(
        release_closure_store_module.json.dumps(
            tampered_package_payload,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    audit_path = next((tmp_path / "audit").glob("release_closure_*.jsonl"))
    audit_lines = [
        release_closure_store_module.json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit_lines[1] = build_release_closure_audit_event(
        event_id=make_release_closure_event_id(
            package.release_execution_id,
            "evidence_package_generated",
            f"{closure.closed_at}#{package.package_id}",
        ),
        intent_id=package.intent_id,
        release_execution_id=package.release_execution_id,
        event_type="evidence_package_generated",
        actor=package.generated_by,
        timestamp=closure.closed_at,
        payload=tampered_package_payload,
        previous_event_hash=audit_lines[0]["event_hash"],
    ).to_dict()
    audit_path.write_text(
        "\n".join(
            release_closure_store_module.json.dumps(line, sort_keys=True)
            for line in audit_lines
        )
        + "\n",
        encoding="utf-8",
    )

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert any(
        "closure/package mismatch" in warning or "snapshot_hashes" in warning
        for warning in state.integrity["warnings"]
    )


@pytest.mark.parametrize(
    ("overrides", "artifact_refs"),
    [
        ({"closure_id": "release_closure_other"}, None),
        ({"package_id": "release_evidence_package_other"}, None),
        ({"intent_id": "release_intent_other"}, None),
        ({"release_execution_id": "release_exec_other"}, None),
        ({"rollback_execution_id": "rollback_exec_other"}, None),
        ({"closure_status": "accepted_with_observations"}, None),
        ({}, ["reports/release_closure/closures/release_closure_other.json"]),
    ],
)
def test_mismatched_closure_and_package_are_rejected_before_persistence(
    tmp_path: Path,
    overrides: dict[str, str | None],
    artifact_refs: list[str] | None,
) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    package_payload = {**package.to_dict(), **overrides}
    if artifact_refs is not None:
        package_payload["artifact_refs"] = artifact_refs
    mismatched_package = ReleaseEvidencePackage(**package_payload)

    with pytest.raises(ValueError, match="closure/package mismatch"):
        store.write_closure_with_package(
            closure,
            mismatched_package,
            timestamp=closure.closed_at,
        )

    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == []


def test_snapshot_hash_mismatch_is_rejected_before_persistence(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    mismatched_package = ReleaseEvidencePackage(
        **{
            **package.to_dict(),
            "snapshot_hashes": {
                **package.to_dict()["snapshot_hashes"],
                "monitoring": "sha256:" + "e" * 64,
            },
        }
    )

    with pytest.raises(ValueError, match="closure/package mismatch"):
        store.write_closure_with_package(
            closure,
            mismatched_package,
            timestamp=closure.closed_at,
        )

    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == []
