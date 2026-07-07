from __future__ import annotations

import pytest

from src.contracts.release_monitoring import (
    GENESIS_MONITORING_EVENT_HASH,
    MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS,
    MONITORING_ALERT_CATEGORIES,
    MONITORING_ALERT_STATUSES,
    MONITORING_RECOMMENDED_ACTIONS,
    ReleaseMonitoringAcknowledgement,
    ReleaseMonitoringAlert,
    ReleaseMonitoringCheck,
    ReleaseMonitoringAuditEvent,
    ReleaseRollbackTriggerCandidate,
    build_monitoring_audit_event,
    canonical_monitoring_payload_hash,
    make_monitoring_acknowledgement_id,
    make_monitoring_alert_id,
    make_monitoring_check_id,
    make_monitoring_event_id,
    make_rollback_trigger_candidate_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"
ROLLBACK_PLAN_ID = "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"


SPEC_ALERT_CATEGORIES = (
    "missing_required_check",
    "post_release_check_failed",
    "post_release_check_warning",
    "execution_integrity_failed",
    "governance_drift",
    "feature_flag_state_mismatch",
    "rollback_ready",
)
SPEC_RECOMMENDED_ACTIONS = (
    "observe",
    "investigate",
    "prepare_rollback",
    "execute_step13_rollback",
)
SPEC_ALERT_STATUSES = ("active", "acknowledged")
SPEC_ACKNOWLEDGEMENT_DISPOSITIONS = (
    "investigating",
    "accepted_risk",
    "rollback_started_elsewhere",
    "false_positive",
)


def make_check(status: str = "pass") -> ReleaseMonitoringCheck:
    return ReleaseMonitoringCheck(
        check_id=make_monitoring_check_id(EXECUTION_ID, "p0_harness_replay", "idem-1"),
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status=status,
        observed_by="release_manager",
        observed_at="2026-07-03T11:00:00+08:00",
        summary="P0 harness replay completed after release execution.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"passed": 5, "failed": 0, "hard_fail_count": 0},
        idempotency_key="idem-1",
    )


def make_candidate(action: str = "execute_step13_rollback") -> ReleaseRollbackTriggerCandidate:
    alert_id = make_monitoring_alert_id(
        EXECUTION_ID,
        "post_release_check_failed",
        "p0_harness_replay",
    )
    return ReleaseRollbackTriggerCandidate(
        candidate_id=make_rollback_trigger_candidate_id(EXECUTION_ID, [alert_id]),
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        source_alert_ids=[alert_id],
        recommended_action=action,
        rollback_plan_id=ROLLBACK_PLAN_ID,
        rollback_target="agent_policy_20260624_0",
        reason="A critical post-release check failed while the local feature flag remains enabled.",
        created_at="2026-07-03T11:00:00+08:00",
    )


def test_monitoring_enum_contracts_match_step14_spec() -> None:
    assert MONITORING_ALERT_CATEGORIES == SPEC_ALERT_CATEGORIES
    assert MONITORING_RECOMMENDED_ACTIONS == SPEC_RECOMMENDED_ACTIONS
    assert MONITORING_ALERT_STATUSES == SPEC_ALERT_STATUSES
    assert MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS == SPEC_ACKNOWLEDGEMENT_DISPOSITIONS


def test_monitoring_contracts_round_trip_to_dict() -> None:
    check = make_check()
    alert = ReleaseMonitoringAlert(
        alert_id=make_monitoring_alert_id(EXECUTION_ID, "post_release_check_failed", "p0_harness_replay"),
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        severity="critical",
        category="post_release_check_failed",
        status="active",
        message="P0 harness replay reported a hard fail after release execution.",
        source_check_ids=[check.check_id],
        recommended_action="execute_step13_rollback",
        created_at="2026-07-03T11:00:00+08:00",
    )
    candidate = ReleaseRollbackTriggerCandidate(
        candidate_id=make_rollback_trigger_candidate_id(EXECUTION_ID, [alert.alert_id]),
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        source_alert_ids=[alert.alert_id],
        recommended_action="execute_step13_rollback",
        rollback_plan_id=ROLLBACK_PLAN_ID,
        rollback_target="agent_policy_20260624_0",
        reason="A critical post-release check failed while the local feature flag remains enabled.",
        created_at="2026-07-03T11:00:00+08:00",
    )
    acknowledgement = ReleaseMonitoringAcknowledgement(
        acknowledgement_id=make_monitoring_acknowledgement_id(alert.alert_id, "ack-1"),
        alert_id=alert.alert_id,
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        acknowledged_by="release_manager",
        acknowledged_at="2026-07-03T11:05:00+08:00",
        disposition="investigating",
        reason="Checking harness evidence before rollback execution.",
    )
    event = build_monitoring_audit_event(
        event_id=make_monitoring_event_id(EXECUTION_ID, "check_recorded", "2026-07-03T11:00:00+08:00"),
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        event_type="check_recorded",
        actor="release_manager",
        timestamp="2026-07-03T11:00:00+08:00",
        payload=check.to_dict(),
        previous_event_hash=GENESIS_MONITORING_EVENT_HASH,
    )

    assert ReleaseMonitoringCheck(**check.to_dict()).to_dict() == check.to_dict()
    assert ReleaseMonitoringAlert(**alert.to_dict()).to_dict() == alert.to_dict()
    assert (
        ReleaseRollbackTriggerCandidate(**candidate.to_dict()).to_dict()
        == candidate.to_dict()
    )
    assert (
        ReleaseMonitoringAcknowledgement(**acknowledgement.to_dict()).to_dict()
        == acknowledgement.to_dict()
    )
    assert ReleaseMonitoringAuditEvent(**event.to_dict()).to_dict() == event.to_dict()


@pytest.mark.parametrize("action", ["observe", "investigate", "prepare_rollback"])
def test_rollback_trigger_candidate_requires_step13_rollback_action(action: str) -> None:
    payload = make_candidate().to_dict()
    payload["recommended_action"] = action

    with pytest.raises(
        ValueError,
        match="recommended_action must be execute_step13_rollback",
    ):
        ReleaseRollbackTriggerCandidate(**payload)


def test_rollback_trigger_candidate_requires_source_alert_ids() -> None:
    payload = make_candidate().to_dict()
    payload["source_alert_ids"] = []

    with pytest.raises(
        ValueError,
        match="source_alert_ids must contain at least one alert id",
    ):
        ReleaseRollbackTriggerCandidate(**payload)


def test_rollback_trigger_candidate_id_requires_alert_ids() -> None:
    with pytest.raises(
        ValueError,
        match="alert_ids must contain at least one alert id",
    ):
        make_rollback_trigger_candidate_id(EXECUTION_ID, [])


def test_alert_rejects_resolved_status() -> None:
    check = make_check()
    alert = ReleaseMonitoringAlert(
        alert_id=make_monitoring_alert_id(
            EXECUTION_ID,
            "post_release_check_failed",
            "p0_harness_replay",
        ),
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        severity="critical",
        category="post_release_check_failed",
        status="active",
        message="P0 harness replay reported a hard fail after release execution.",
        source_check_ids=[check.check_id],
        recommended_action="execute_step13_rollback",
        created_at="2026-07-03T11:00:00+08:00",
    ).to_dict()
    alert["status"] = "resolved"

    with pytest.raises(ValueError, match="status must be one of"):
        ReleaseMonitoringAlert(**alert)


@pytest.mark.parametrize("check_type", ["runtime_patient_scan", "scheduler_probe", ""])
def test_check_rejects_unknown_check_type(check_type: str) -> None:
    payload = make_check().to_dict()
    payload["check_type"] = check_type

    with pytest.raises(ValueError, match="check_type must be one of"):
        ReleaseMonitoringCheck(**payload)


@pytest.mark.parametrize("status", ["ok", "blocked", ""])
def test_check_rejects_unknown_status(status: str) -> None:
    payload = make_check().to_dict()
    payload["status"] = status

    with pytest.raises(ValueError, match="status must be one of"):
        ReleaseMonitoringCheck(**payload)


@pytest.mark.parametrize(
    "evidence_ref",
    [
        "D:/YiZhu_Agnet/LangG/reports/harness/harness_20260629_001.json",
        "D:foo",
        "/reports/harness/x.json",
        "reports/../secrets.json",
        "reports/./x.json",
        "reports//x.json",
        "https://example.com/x",
    ],
)
def test_evidence_refs_must_be_repo_relative_paths(evidence_ref: str) -> None:
    payload = make_check().to_dict()
    payload["evidence_refs"] = [evidence_ref]

    with pytest.raises(ValueError, match="evidence_refs must be repo-relative"):
        ReleaseMonitoringCheck(**payload)


def test_payload_hash_is_canonical_and_rejects_secrets() -> None:
    assert canonical_monitoring_payload_hash({"b": 2, "a": 1}) == canonical_monitoring_payload_hash(
        {"a": 1, "b": 2}
    )

    with pytest.raises(ValueError, match="payload contains forbidden key"):
        canonical_monitoring_payload_hash({"patient_id": "patient-123"})


def test_audit_event_hash_chain_uses_previous_hash() -> None:
    first = build_monitoring_audit_event(
        event_id="release_monitoring_audit_check_recorded_1",
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        event_type="check_recorded",
        actor="release_manager",
        timestamp="2026-07-03T11:00:00+08:00",
        payload={"check": "one"},
        previous_event_hash=GENESIS_MONITORING_EVENT_HASH,
    )
    second = build_monitoring_audit_event(
        event_id="release_monitoring_audit_alert_acknowledged_1",
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        event_type="alert_acknowledged",
        actor="release_manager",
        timestamp="2026-07-03T11:05:00+08:00",
        payload={"ack": "one"},
        previous_event_hash=first.event_hash,
    )

    assert second.previous_event_hash == first.event_hash
    assert second.event_hash != first.event_hash
