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
    assert "payload" not in event.to_dict()
    assert event.to_dict()["payload_hash"].startswith("sha256:")
    assert event.to_dict()["event_hash"].startswith("sha256:")

    intent_dict = intent.to_dict()
    approval_dict = approval.to_dict()
    rollback_plan_dict = rollback_plan.to_dict()
    event_dict = event.to_dict()

    assert ReleaseIntent(**intent_dict).to_dict() == intent_dict
    assert ReleaseApproval(**approval_dict).to_dict() == approval_dict
    assert ReleaseRollbackPlan(**rollback_plan_dict).to_dict() == rollback_plan_dict
    assert ReleaseAuditEvent(**event_dict).to_dict() == event_dict


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


@pytest.mark.parametrize(
    "source_report_path",
    [
        "/reports/release_safety/release.json",
        "\\reports\\release_safety\\release.json",
        "C:/reports/release_safety/release.json",
        "C:reports/release_safety/release.json",
    ],
)
def test_release_intent_rejects_non_repo_relative_source_report_paths(
    source_report_path: str,
) -> None:
    payload = make_intent().to_dict()
    payload["source_report_path"] = source_report_path

    with pytest.raises(
        ValueError, match="source_report_path must be a repo-relative path"
    ):
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


def test_audit_event_omits_raw_payload_and_is_stable_after_source_payload_mutation() -> None:
    payload = {"nested": {"release_decision": "shadow"}}
    event = build_audit_event(
        event_id="release_audit_payload_immutability",
        intent_id="release_intent_1",
        event_type="intent_created",
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
        payload=payload,
        previous_event_hash="sha256:GENESIS",
    )
    event_dict = event.to_dict()

    payload["nested"]["release_decision"] = "tampered"
    payload["new_key"] = "new value"

    assert "payload" not in event_dict
    assert event.to_dict() == event_dict
    assert event.payload_hash == event_dict["payload_hash"]
    assert event.event_hash == event_dict["event_hash"]


def test_id_helpers_return_expected_stable_ids_and_change_with_inputs() -> None:
    intent_id = make_release_intent_id("release_safety_20260629_001")
    approval_id = make_release_approval_id(
        "release_intent_1", "release_manager", "2026-07-02T00:00:00+08:00"
    )
    rollback_plan_id = make_release_rollback_plan_id(
        "release_intent_1", "2026-07-02T00:00:00+08:00"
    )
    audit_event_id = make_release_audit_event_id(
        "release_intent_1", "intent_created", "2026-07-02T00:00:00+08:00"
    )

    assert intent_id == "release_intent_release_safety_20260629_001_a1661529"
    assert intent_id.startswith("release_intent_release_safety_20260629_001_")
    assert intent_id != make_release_intent_id("release_safety_20260629_002")

    assert approval_id == (
        "release_approval_release_intent_1_release_manager_09898625"
    )
    assert approval_id.startswith(
        "release_approval_release_intent_1_release_manager_"
    )
    assert approval_id != make_release_approval_id(
        "release_intent_1",
        "clinical_safety_reviewer",
        "2026-07-02T00:00:00+08:00",
    )

    assert rollback_plan_id == "rollback_plan_release_intent_1_b22fd55d"
    assert rollback_plan_id.startswith("rollback_plan_release_intent_1_")
    assert rollback_plan_id != make_release_rollback_plan_id(
        "release_intent_1", "2026-07-02T00:01:00+08:00"
    )

    assert audit_event_id == "release_audit_intent_created_59d22c26"
    assert audit_event_id.startswith("release_audit_intent_created_")
    assert audit_event_id != make_release_audit_event_id(
        "release_intent_1", "approval_recorded", "2026-07-02T00:00:00+08:00"
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


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": [{"headers": {"bearer-token": "secret"}}]},
        {"nested": {"deploymentCredential": "secret"}},
        {"nested": {"deployment credentials": "secret"}},
        {"nested": {"x-api-key": "secret"}},
        {"nested": {"openaiApiKey": "secret"}},
        {"nested": {"prompt": "system prompt"}},
        {"nested": {"hiddenReasoning": "private reasoning"}},
        {"nested": {"chainOfThought": "private reasoning"}},
        {"patient": {"id": "MRN-123"}},
        {"nested": {"patientIds": ["MRN-123"]}},
        {"nested": {"patient_mrn": "MRN-123"}},
        {"nested": {"patientId": "MRN-123"}},
        {"nested": {"patient_identifier": "MRN-123"}},
        {"nested": {"patientName": "Example Patient"}},
        {"nested": {"patient-number": "MRN-123"}},
        {"nested": {"mrn": "MRN-123"}},
        {"nested": {"medicalRecordNumber": "MRN-123"}},
        {"nested": {"clientSecret": "secret"}},
        {"nested": {"refreshToken": "secret"}},
        {"nested": {"sessionToken": "secret"}},
        {"nested": {"private key": "secret"}},
    ],
)
def test_audit_event_rejects_nested_secret_prompt_and_phi_payload_keys(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="payload contains forbidden key"):
        build_audit_event(
            event_id="release_audit_bad_nested_payload",
            intent_id="release_intent_1",
            event_type="intent_created",
            actor="admin_operator",
            timestamp="2026-07-02T00:00:00+08:00",
            payload=payload,
            previous_event_hash="sha256:GENESIS",
        )


def test_audit_event_allows_aggregate_patient_counts() -> None:
    event = build_audit_event(
        event_id="release_audit_aggregate_counts",
        intent_id="release_intent_1",
        event_type="intent_created",
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
        payload={"nested": {"patient_count": 12}},
        previous_event_hash="sha256:GENESIS",
    )

    assert event.payload_hash.startswith("sha256:")


def test_audit_event_reconstruction_rejects_hash_tampering() -> None:
    event_dict = build_audit_event(
        event_id="release_audit_hash_validation",
        intent_id="release_intent_1",
        event_type="intent_created",
        actor="admin_operator",
        timestamp="2026-07-02T00:00:00+08:00",
        payload={"status": "ready"},
        previous_event_hash="sha256:GENESIS",
    ).to_dict()

    mismatched_event_hash = dict(event_dict)
    mismatched_event_hash["event_hash"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="event_hash does not match"):
        ReleaseAuditEvent(**mismatched_event_hash)

    malformed_previous_hash = dict(event_dict)
    malformed_previous_hash["previous_event_hash"] = "sha256:not-valid"
    with pytest.raises(
        ValueError, match="previous_event_hash must be a sha256 hash"
    ):
        ReleaseAuditEvent(**malformed_previous_hash)

    stale_event_hash = dict(event_dict)
    stale_event_hash["payload_hash"] = "sha256:" + ("1" * 64)
    with pytest.raises(ValueError, match="payload_hash|event_hash"):
        ReleaseAuditEvent(**stale_event_hash)
