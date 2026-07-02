from __future__ import annotations

import pytest

from src.contracts.release_governance import (
    ReleaseApproval,
    ReleaseAuditEvent,
    ReleaseIntent,
    ReleaseRollbackPlan,
    build_audit_event,
    canonical_payload_hash,
    make_release_approval_id,
    make_release_audit_event_id,
    make_release_intent_id,
    make_release_rollback_plan_id,
)


VERSION_CHAIN = {
    "agent_policy_version": "agent_policy_20260629_0",
    "clinical_safety_policy_version": "crc_safety_policy_v0",
    "evidence_index_version": "rag_crc_guideline_20260620",
    "judge_rubric_version": "crc_rubric_v0",
}


def make_intent() -> ReleaseIntent:
    return ReleaseIntent(
        intent_id="release_intent_release_safety_20260629_001_6da729a0",
        source_release_report_id="release_safety_20260629_001",
        source_report_path="reports/release_safety/release_safety_20260629_001.json",
        harness_run_ids=["harness_20260629_001"],
        literature_run_id="literature_harness_20260630_001",
        version_chain=VERSION_CHAIN,
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


def test_release_governance_contracts_round_trip_to_dict() -> None:
    intent = make_intent()
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
    event = build_audit_event(
        event_id="release_audit_intent_created_27005abc",
        intent_id=intent.intent_id,
        event_type="intent_created",
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
        payload=intent.to_dict(),
        previous_event_hash="sha256:GENESIS",
    )

    assert intent.to_dict()["target_scope"] == "shadow"
    assert approval.to_dict()["decision"] == "approve"
    assert rollback_plan.to_dict()["verification_steps"] == [
        "Confirm the active release report id.",
        "Run P0 harness before any future rollback execution.",
    ]
    assert event.to_dict()["payload_hash"].startswith("sha256:")
    assert event.to_dict()["event_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("target_scope", "production", "target_scope must be one of"),
        ("status", "executed", "status must be one of"),
    ],
)
def test_release_intent_rejects_invalid_enums(
    field_name: str, value: str, expected: str
) -> None:
    payload = make_intent().to_dict()
    payload[field_name] = value

    with pytest.raises(ValueError, match=expected):
        ReleaseIntent(**payload)


def test_release_approval_rejects_invalid_decision_and_empty_reason() -> None:
    with pytest.raises(ValueError, match="decision must be one of"):
        ReleaseApproval(
            approval_id="release_approval_bad",
            intent_id="release_intent_1",
            approver_role="release_manager",
            decision="sign",
            reason="valid reason",
            signed_by="reviewer",
            signed_at="2026-07-02T00:10:00+08:00",
            required=True,
        )

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        ReleaseApproval(
            approval_id="release_approval_bad",
            intent_id="release_intent_1",
            approver_role="release_manager",
            decision="approve",
            reason=" ",
            signed_by="reviewer",
            signed_at="2026-07-02T00:10:00+08:00",
            required=True,
        )


def test_rollback_plan_requires_two_verification_steps() -> None:
    with pytest.raises(
        ValueError, match="verification_steps must contain at least two steps"
    ):
        ReleaseRollbackPlan(
            rollback_plan_id="rollback_plan_bad",
            intent_id="release_intent_1",
            rollback_target="agent_policy_20260624_0",
            owner="release_manager",
            status="accepted",
            verification_steps=["Only one check"],
            created_at="2026-07-02T00:15:00+08:00",
        )


def test_payload_hash_is_canonical_and_audit_chain_uses_previous_hash() -> None:
    left = canonical_payload_hash({"b": 2, "a": 1})
    right = canonical_payload_hash({"a": 1, "b": 2})
    first = build_audit_event(
        event_id="release_audit_1",
        intent_id="release_intent_1",
        event_type="intent_created",
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
        payload={"a": 1},
        previous_event_hash="sha256:GENESIS",
    )
    second = build_audit_event(
        event_id="release_audit_2",
        intent_id="release_intent_1",
        event_type="approval_recorded",
        actor="reviewer",
        timestamp="2026-07-02T00:01:00+08:00",
        payload={"b": 2},
        previous_event_hash=first.event_hash,
    )

    assert left == right
    assert first.previous_event_hash == "sha256:GENESIS"
    assert second.previous_event_hash == first.event_hash
    assert second.event_hash != first.event_hash


def test_id_helpers_are_stable_for_same_inputs() -> None:
    assert make_release_intent_id(
        "release_safety_20260629_001"
    ) == make_release_intent_id("release_safety_20260629_001")
    assert make_release_approval_id(
        "release_intent_1", "release_manager", "2026-07-02T00:00:00+08:00"
    )
    assert make_release_rollback_plan_id(
        "release_intent_1", "2026-07-02T00:00:00+08:00"
    )
    assert make_release_audit_event_id(
        "release_intent_1", "intent_created", "2026-07-02T00:00:00+08:00"
    )


def test_audit_event_rejects_secret_like_payload() -> None:
    with pytest.raises(ValueError, match="payload contains forbidden key"):
        build_audit_event(
            event_id="release_audit_bad",
            intent_id="release_intent_1",
            event_type="intent_created",
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
            payload={"api_key": "secret"},
            previous_event_hash="sha256:GENESIS",
        )
