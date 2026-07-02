from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.api.services.release_governance_store import (
    GovernanceIntegrityError,
    ReleaseGovernanceStore,
)
from src.contracts.release_governance import (
    ReleaseApproval,
    ReleaseIntent,
    ReleaseRollbackPlan,
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
    approval = ReleaseApproval(
        approval_id="release_approval_release_intent_release_manager_d79a98c1",
        intent_id=intent.intent_id,
        approver_role="release_manager",
        decision="approve",
        reason="P0 hard fails are zero.",
        signed_by="release_admin",
        signed_at="2026-07-02T00:10:00+08:00",
        required=True,
    )
    rollback_plan = ReleaseRollbackPlan(
        rollback_plan_id="rollback_plan_release_intent_1c338f15",
        intent_id=intent.intent_id,
        rollback_target="agent_policy_20260624_0",
        owner="release_manager",
        status="accepted",
        verification_steps=[
            "Confirm the active release report id.",
            "Run P0 harness before any future rollback execution.",
        ],
        created_at="2026-07-02T00:15:00+08:00",
    )

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
