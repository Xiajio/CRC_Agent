from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.api.services.release_governance_store import (
    GovernanceIntegrityError,
    ReleaseGovernanceStore,
)
from src.contracts.release_governance import (
    GENESIS_EVENT_HASH,
    ReleaseApproval,
    ReleaseIntent,
    ReleaseRollbackPlan,
    build_audit_event,
    make_release_audit_event_id,
)


def make_intent() -> ReleaseIntent:
    return ReleaseIntent(
        intent_id="release_intent_release_safety_20260629_001_6da729a0",
        source_release_report_id="release_safety_20260629_001",
        source_report_path="reports/release_safety/release_safety_20260629_001.json",
        harness_run_ids=["harness_20260629_001"],
        literature_run_id="literature_harness_20260630_001",
        version_chain={
            "agent_policy_version": "agent_policy_20260629_0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
            "evidence_index_version": "rag_crc_guideline_20260620",
            "judge_rubric_version": "crc_rubric_v0",
        },
        release_decision_snapshot="feature_flag_or_pass",
        rollback_target="agent_policy_20260624_0",
        requested_by="admin_operator",
        requested_at="2026-07-02T00:00:00+08:00",
        target_scope="shadow",
        status="pending_approval",
        blocking_summary={
            "hard_fail_count": 0,
            "literature_isolation_violations": 0,
            "clinical_rag_ingest_enabled": False,
        },
    )


def make_second_intent() -> ReleaseIntent:
    return ReleaseIntent(
        **{
            **make_intent().to_dict(),
            "intent_id": "release_intent_release_safety_20260629_002_b9c4ad18",
            "source_release_report_id": "release_safety_20260629_002",
            "requested_at": "2026-07-02T00:00:00+08:00",
        }
    )


def make_approval(intent_id: str) -> ReleaseApproval:
    return ReleaseApproval(
        approval_id="release_approval_release_intent_release_manager_d79a98c1",
        intent_id=intent_id,
        approver_role="release_manager",
        decision="approve",
        reason="P0 hard fails are zero.",
        signed_by="release_admin",
        signed_at="2026-07-02T00:10:00+08:00",
        required=True,
    )


def make_rollback_plan(intent_id: str) -> ReleaseRollbackPlan:
    return ReleaseRollbackPlan(
        rollback_plan_id="rollback_plan_release_intent_1c338f15",
        intent_id=intent_id,
        rollback_target="agent_policy_20260624_0",
        owner="release_manager",
        status="accepted",
        verification_steps=[
            "Confirm the active release report id.",
            "Run P0 harness before any future rollback execution.",
        ],
        created_at="2026-07-02T00:15:00+08:00",
    )


def _write_audit_event(root: Path, *, event: object) -> None:
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True)
    audit_file = audit_dir / "release_audit_20260702.jsonl"
    audit_file.write_text(
        json.dumps(event.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_empty_store_reads_without_creating_files(tmp_path: Path) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")

    state = store.read_state()

    assert state.intents == []
    assert state.approvals == []
    assert state.rollback_plans == []
    assert state.audit_events == []
    assert state.integrity == {"status": "verified", "warnings": []}
    assert not (tmp_path / "release_governance").exists()


def test_write_intent_creates_intent_file_and_audit_event(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()

    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    state = store.read_state()

    assert [item.intent_id for item in state.intents] == [intent.intent_id]
    assert [event.event_type for event in state.audit_events] == [
        "intent_created"
    ]
    assert (
        tmp_path
        / "release_governance"
        / "intents"
        / f"{intent.intent_id}.json"
    ).exists()
    assert (
        tmp_path
        / "release_governance"
        / "audit"
        / "release_audit_20260702.jsonl"
    ).exists()


def test_store_rejects_overwriting_existing_intent(tmp_path: Path) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )

    with pytest.raises(FileExistsError):
        store.write_intent(
            intent,
            actor="admin_operator",
            timestamp="2026-07-02T00:01:00+08:00",
        )


def test_write_approval_requires_existing_intent_and_creates_no_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release_governance"
    store = ReleaseGovernanceStore(root)
    approval = make_approval(make_intent().intent_id)

    with pytest.raises(GovernanceIntegrityError, match="unknown release intent"):
        store.write_approval(
            approval,
            actor="release_admin",
            timestamp=approval.signed_at,
        )

    assert not (root / "approvals" / f"{approval.approval_id}.json").exists()
    assert not (root / "audit").exists()


def test_write_rollback_plan_requires_existing_intent_and_creates_no_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release_governance"
    store = ReleaseGovernanceStore(root)
    rollback_plan = make_rollback_plan(make_intent().intent_id)

    with pytest.raises(GovernanceIntegrityError, match="unknown release intent"):
        store.write_rollback_plan(
            rollback_plan,
            actor="release_manager",
            timestamp=rollback_plan.created_at,
        )

    assert not (
        root / "rollback_plans" / f"{rollback_plan.rollback_plan_id}.json"
    ).exists()
    assert not (root / "audit").exists()


def test_append_cancel_event_requires_existing_intent_and_creates_no_audit_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release_governance"
    store = ReleaseGovernanceStore(root)
    intent_id = make_intent().intent_id

    with pytest.raises(GovernanceIntegrityError, match="unknown release intent"):
        store.append_cancel_event(
            intent_id=intent_id,
            actor="admin_operator",
            reason="No such intent.",
            timestamp="2026-07-02T00:20:00+08:00",
        )

    assert not (root / "audit").exists()


def test_copied_intent_artifact_filename_mismatch_fails_integrity(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    intent_file = (
        tmp_path
        / "release_governance"
        / "intents"
        / f"{intent.intent_id}.json"
    )
    copy_file = tmp_path / "release_governance" / "intents" / "copy.json"
    copy_file.write_text(intent_file.read_text(encoding="utf-8"), encoding="utf-8")

    state = store.read_state()

    assert [item.intent_id for item in state.intents] == [intent.intent_id]
    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [intent.intent_id]
    assert any(
        "filename" in warning and "intent artifact" in warning
        for warning in state.integrity["warnings"]
    )


def test_symlinked_intents_directory_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "release_governance"
    escaped = tmp_path / "escaped_intents"
    escaped.mkdir()
    root.mkdir()
    try:
        (root / "intents").symlink_to(escaped, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink unsupported: {exc}")
    store = ReleaseGovernanceStore(root)
    intent = make_intent()

    with pytest.raises(GovernanceIntegrityError, match="outside governance root"):
        store.write_intent(
            intent,
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
        )

    assert not (escaped / f"{intent.intent_id}.json").exists()


def test_symlinked_audit_directory_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "release_governance"
    escaped = tmp_path / "escaped_audit"
    escaped.mkdir()
    root.mkdir()
    try:
        (root / "audit").symlink_to(escaped, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink unsupported: {exc}")
    store = ReleaseGovernanceStore(root)
    intent = make_intent()

    with pytest.raises(GovernanceIntegrityError, match="outside governance root"):
        store.write_intent(
            intent,
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
        )

    assert not (escaped / "release_audit_20260702.jsonl").exists()
    assert not (
        root / "intents" / f"{intent.intent_id}.json"
    ).exists()


def test_symlinked_existing_audit_file_cannot_escape_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release_governance"
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True)
    outside_file = tmp_path / "outside_audit.jsonl"
    outside_file.write_text("outside\n", encoding="utf-8")
    audit_file = audit_dir / "release_audit_20260702.jsonl"
    try:
        audit_file.symlink_to(outside_file)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlink unsupported: {exc}")
    store = ReleaseGovernanceStore(root)
    intent = make_intent()

    with pytest.raises(GovernanceIntegrityError, match="outside governance root|symlink"):
        store.write_intent(
            intent,
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
        )

    assert outside_file.read_text(encoding="utf-8") == "outside\n"
    assert not (
        root / "intents" / f"{intent.intent_id}.json"
    ).exists()


def test_write_approval_and_rollback_plan_append_to_audit_chain(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    approval = make_approval(intent.intent_id)
    rollback_plan = make_rollback_plan(intent.intent_id)

    store.write_approval(approval, actor="release_admin", timestamp=approval.signed_at)
    store.write_rollback_plan(
        rollback_plan,
        actor="release_manager",
        timestamp=rollback_plan.created_at,
    )
    state = store.read_state()

    assert len(state.approvals) == 1
    assert len(state.rollback_plans) == 1
    assert [event.event_type for event in state.audit_events] == [
        "intent_created",
        "approval_recorded",
        "rollback_plan_recorded",
    ]
    assert (
        state.audit_events[1].previous_event_hash
        == state.audit_events[0].event_hash
    )
    assert (
        state.audit_events[2].previous_event_hash
        == state.audit_events[1].event_hash
    )


def test_invalid_intent_audit_timestamp_does_not_leave_artifact(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()

    with pytest.raises(ValueError, match="timestamp must start"):
        store.write_intent(
            intent,
            actor="admin_operator",
            timestamp="not-a-date",
        )

    assert not (
        tmp_path
        / "release_governance"
        / "intents"
        / f"{intent.intent_id}.json"
    ).exists()


def test_invalid_approval_audit_timestamp_does_not_leave_artifact(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    approval = make_approval(intent.intent_id)

    with pytest.raises(ValueError, match="timestamp must start"):
        store.write_approval(
            approval,
            actor="release_admin",
            timestamp="not-a-date",
        )

    assert not (
        tmp_path
        / "release_governance"
        / "approvals"
        / f"{approval.approval_id}.json"
    ).exists()


def test_invalid_rollback_plan_audit_timestamp_does_not_leave_artifact(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    rollback_plan = make_rollback_plan(intent.intent_id)

    with pytest.raises(ValueError, match="timestamp must start"):
        store.write_rollback_plan(
            rollback_plan,
            actor="release_manager",
            timestamp="not-a-date",
        )

    assert not (
        tmp_path
        / "release_governance"
        / "rollback_plans"
        / f"{rollback_plan.rollback_plan_id}.json"
    ).exists()


def test_append_cancel_event_adds_cancel_audit_event(tmp_path: Path) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )

    store.append_cancel_event(
        intent_id=intent.intent_id,
        actor="admin_operator",
        reason="Release window closed.",
        timestamp="2026-07-02T00:20:00+08:00",
    )
    state = store.read_state()

    assert [event.event_type for event in state.audit_events] == [
        "intent_created",
        "intent_cancelled",
    ]
    assert (
        state.audit_events[1].previous_event_hash
        == state.audit_events[0].event_hash
    )


def test_non_monotonic_timestamps_append_chain_in_physical_order(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:10:00+08:00",
    )
    store.append_cancel_event(
        intent_id=intent.intent_id,
        actor="admin_operator",
        reason="First cancellation note.",
        timestamp="2026-07-02T00:05:00+08:00",
    )
    store.append_cancel_event(
        intent_id=intent.intent_id,
        actor="admin_operator",
        reason="Second cancellation note.",
        timestamp="2026-07-02T00:07:00+08:00",
    )

    state = store.read_state()

    assert state.integrity == {"status": "verified", "warnings": []}
    assert [event.timestamp for event in state.audit_events] == [
        "2026-07-02T00:10:00+08:00",
        "2026-07-02T00:05:00+08:00",
        "2026-07-02T00:07:00+08:00",
    ]
    assert (
        state.audit_events[2].previous_event_hash
        == state.audit_events[1].event_hash
    )


def test_reordered_audit_lines_return_integrity_warning(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    store.append_cancel_event(
        intent_id=intent.intent_id,
        actor="admin_operator",
        reason="Release window closed.",
        timestamp="2026-07-02T00:20:00+08:00",
    )
    audit_file = (
        tmp_path
        / "release_governance"
        / "audit"
        / "release_audit_20260702.jsonl"
    )
    lines = audit_file.read_text(encoding="utf-8").splitlines()
    audit_file.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert any(
        "previous_event_hash mismatch" in warning
        for warning in state.integrity["warnings"]
    )


def test_cross_day_backdated_append_is_rejected_without_audit_row(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-03T00:00:00+08:00",
    )

    with pytest.raises(GovernanceIntegrityError, match="backdated"):
        store.append_cancel_event(
            intent_id=intent.intent_id,
            actor="admin_operator",
            reason="Release window closed.",
            timestamp="2026-07-02T23:59:00+08:00",
        )

    assert not (
        tmp_path
        / "release_governance"
        / "audit"
        / "release_audit_20260702.jsonl"
    ).exists()
    state = store.read_state()
    assert state.integrity == {"status": "verified", "warnings": []}
    assert [event.event_type for event in state.audit_events] == [
        "intent_created"
    ]


def test_global_cross_day_backdated_new_intent_is_rejected_without_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release_governance"
    store = ReleaseGovernanceStore(root)
    intent_a = make_intent()
    intent_b = make_second_intent()
    store.write_intent(
        intent_a,
        actor="admin_operator",
        timestamp="2026-07-03T00:00:00+08:00",
    )

    with pytest.raises(GovernanceIntegrityError, match="backdated"):
        store.write_intent(
            intent_b,
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
        )

    assert not (root / "intents" / f"{intent_b.intent_id}.json").exists()
    assert not (root / "audit" / "release_audit_20260702.jsonl").exists()
    state = store.read_state()
    assert state.integrity == {"status": "verified", "warnings": []}
    assert [intent.intent_id for intent in state.intents] == [intent_a.intent_id]
    assert [event.event_type for event in state.audit_events] == [
        "intent_created"
    ]


def test_invalid_utf8_audit_file_returns_integrity_warning(
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "release_governance" / "audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "release_audit_20260702.jsonl").write_bytes(b"\xff\xfe\xfa")
    store = ReleaseGovernanceStore(tmp_path / "release_governance")

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["warnings"]


def test_malformed_intent_artifact_returns_integrity_warning_and_blocks_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release_governance"
    intents_dir = root / "intents"
    intents_dir.mkdir(parents=True)
    (intents_dir / "broken.json").write_text("{", encoding="utf-8")
    store = ReleaseGovernanceStore(root)

    state = store.read_state()

    assert state.intents == []
    assert state.integrity["status"] == "failed"
    assert state.integrity["warnings"]

    with pytest.raises(GovernanceIntegrityError):
        store.write_intent(
            make_intent(),
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
        )


def test_orphan_approval_artifact_and_audit_row_fails_integrity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release_governance"
    approvals_dir = root / "approvals"
    approvals_dir.mkdir(parents=True)
    missing_intent_id = "release_intent_missing_20260702_001"
    approval = make_approval(missing_intent_id)
    (approvals_dir / f"{approval.approval_id}.json").write_text(
        json.dumps(approval.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    event = build_audit_event(
        event_id=make_release_audit_event_id(
            missing_intent_id,
            "approval_recorded",
            approval.signed_at,
        ),
        intent_id=missing_intent_id,
        event_type="approval_recorded",
        actor="release_admin",
        timestamp=approval.signed_at,
        payload=approval.to_dict(),
        previous_event_hash=GENESIS_EVENT_HASH,
    )
    _write_audit_event(root, event=event)
    store = ReleaseGovernanceStore(root)

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [missing_intent_id]
    assert any("unknown intent" in warning for warning in state.integrity["warnings"])


def test_orphan_cancel_audit_row_fails_integrity(tmp_path: Path) -> None:
    root = tmp_path / "release_governance"
    missing_intent_id = "release_intent_missing_20260702_002"
    event = build_audit_event(
        event_id=make_release_audit_event_id(
            missing_intent_id,
            "intent_cancelled",
            "2026-07-02T00:20:00+08:00",
        ),
        intent_id=missing_intent_id,
        event_type="intent_cancelled",
        actor="admin_operator",
        timestamp="2026-07-02T00:20:00+08:00",
        payload={
            "intent_id": missing_intent_id,
            "actor": "admin_operator",
            "reason": "No such intent.",
        },
        previous_event_hash=GENESIS_EVENT_HASH,
    )
    _write_audit_event(root, event=event)
    store = ReleaseGovernanceStore(root)

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [missing_intent_id]
    assert any("unknown intent" in warning for warning in state.integrity["warnings"])


def test_schema_valid_approval_artifact_tampering_fails_integrity(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    approval = ReleaseApproval(
        **{
            **make_approval(intent.intent_id).to_dict(),
            "decision": "request_changes",
            "reason": "Need one more release note.",
        }
    )
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    store.write_approval(
        approval,
        actor="release_admin",
        timestamp=approval.signed_at,
    )
    approval_file = (
        tmp_path
        / "release_governance"
        / "approvals"
        / f"{approval.approval_id}.json"
    )
    tampered = approval.to_dict()
    tampered["decision"] = "approve"
    tampered["reason"] = "P0 hard fails are zero."
    approval_file.write_text(
        json.dumps(tampered, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [intent.intent_id]
    assert any("approval artifact" in warning for warning in state.integrity["warnings"])


def test_audited_rollback_plan_deletion_fails_integrity(tmp_path: Path) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    rollback_plan = make_rollback_plan(intent.intent_id)
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    store.write_rollback_plan(
        rollback_plan,
        actor="release_manager",
        timestamp=rollback_plan.created_at,
    )
    (
        tmp_path
        / "release_governance"
        / "rollback_plans"
        / f"{rollback_plan.rollback_plan_id}.json"
    ).unlink()

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [intent.intent_id]
    assert any(
        "rollback plan artifact" in warning
        for warning in state.integrity["warnings"]
    )


def test_schema_valid_intent_artifact_tampering_fails_integrity(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    intent_file = (
        tmp_path
        / "release_governance"
        / "intents"
        / f"{intent.intent_id}.json"
    )
    tampered = intent.to_dict()
    tampered["rollback_target"] = "agent_policy_20260620_0"
    intent_file.write_text(
        json.dumps(tampered, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [intent.intent_id]
    assert any("intent artifact" in warning for warning in state.integrity["warnings"])


def test_forbidden_key_in_intent_artifact_returns_integrity_warning_and_blocks_writes(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    intent_file = (
        tmp_path
        / "release_governance"
        / "intents"
        / f"{intent.intent_id}.json"
    )
    tampered = intent.to_dict()
    tampered["blocking_summary"]["patient_id"] = "MRN-1"
    intent_file.write_text(
        json.dumps(tampered, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [intent.intent_id]
    assert any(
        "intent artifact" in warning and intent.intent_id in warning
        for warning in state.integrity["warnings"]
    )
    with pytest.raises(GovernanceIntegrityError):
        store.append_cancel_event(
            intent_id=intent.intent_id,
            actor="admin_operator",
            reason="No writes while artifact integrity is failed.",
            timestamp="2026-07-02T00:20:00+08:00",
        )


def test_failed_audit_append_removes_newly_created_intent_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()

    def fail_append(_prepared_event: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_append_prepared_event", fail_append)

    with pytest.raises(OSError, match="disk full"):
        store.write_intent(
            intent,
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
        )

    assert not (
        tmp_path
        / "release_governance"
        / "intents"
        / f"{intent.intent_id}.json"
    ).exists()


def test_failed_append_cleanup_error_is_surfaced_and_orphan_fails_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()

    def fail_append(_prepared_event: object) -> None:
        raise OSError("disk full")

    def fail_cleanup(_path: Path) -> None:
        raise GovernanceIntegrityError("artifact cleanup failed")

    monkeypatch.setattr(store, "_append_prepared_event", fail_append)
    monkeypatch.setattr(store, "_remove_new_artifact", fail_cleanup)

    with pytest.raises(GovernanceIntegrityError, match="cleanup failed"):
        store.write_intent(
            intent,
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
        )

    monkeypatch.undo()
    state = ReleaseGovernanceStore(tmp_path / "release_governance").read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["affected_intent_ids"] == [intent.intent_id]
    assert any("intent artifact" in warning for warning in state.integrity["warnings"])


def test_chain_mismatch_returns_integrity_warning_and_rejects_write(
    tmp_path: Path,
) -> None:
    store = ReleaseGovernanceStore(tmp_path / "release_governance")
    intent = make_intent()
    store.write_intent(
        intent,
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
    )
    audit_file = (
        tmp_path
        / "release_governance"
        / "audit"
        / "release_audit_20260702.jsonl"
    )
    event = json.loads(audit_file.read_text(encoding="utf-8").splitlines()[0])
    event["event_hash"] = "sha256:broken"
    audit_file.write_text(json.dumps(event) + "\n", encoding="utf-8")

    state = store.read_state()

    assert state.integrity["status"] == "failed"
    assert state.integrity["warnings"]
    with pytest.raises(GovernanceIntegrityError):
        store.write_approval(
            ReleaseApproval(
                approval_id="release_approval_release_intent_release_manager_late",
                intent_id=intent.intent_id,
                approver_role="release_manager",
                decision="approve",
                reason="Late approval",
                signed_by="release_admin",
                signed_at="2026-07-02T00:20:00+08:00",
                required=True,
            ),
            actor="release_admin",
            timestamp="2026-07-02T00:20:00+08:00",
        )
