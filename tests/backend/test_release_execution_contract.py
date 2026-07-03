from __future__ import annotations

import pytest

from src.contracts.release_execution import (
    FEATURE_FLAG_NAME,
    FeatureFlagState,
    ReleaseExecutionRequest,
    ReleaseExecutionResult,
    build_execution_audit_event,
    canonical_execution_payload_hash,
    make_release_execution_event_id,
    make_release_execution_id,
    make_release_execution_result_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
ROLLBACK_PLAN_ID = "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"


def make_request(action: str = "release") -> ReleaseExecutionRequest:
    return ReleaseExecutionRequest(
        execution_id=make_release_execution_id(INTENT_ID, action, "idem-1"),
        intent_id=INTENT_ID,
        action=action,
        requested_by="release_manager",
        requested_at="2026-07-03T09:00:00+08:00",
        idempotency_key="idem-1",
        reason="All governance gates are complete.",
        expected_governance_hash="sha256:" + "a" * 64,
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
        target_flag_state={
            "flag_name": FEATURE_FLAG_NAME,
            "enabled": action == "release",
            "scope": "feature_flag_candidate",
        },
        rollback_target="agent_policy_20260624_0" if action == "rollback" else None,
    )


def test_release_execution_contracts_round_trip_to_dict() -> None:
    request = make_request()
    flag_state = FeatureFlagState(
        flag_name=FEATURE_FLAG_NAME,
        enabled=True,
        scope="feature_flag_candidate",
        source_intent_id=INTENT_ID,
        source_execution_id=request.execution_id,
        rollback_target="agent_policy_20260624_0",
        updated_by="release_manager",
        updated_at="2026-07-03T09:00:00+08:00",
    )
    result = ReleaseExecutionResult(
        result_id=make_release_execution_result_id(request.execution_id),
        execution_id=request.execution_id,
        intent_id=INTENT_ID,
        action="release",
        status="succeeded",
        started_at="2026-07-03T09:00:00+08:00",
        finished_at="2026-07-03T09:00:00+08:00",
        actor="release_manager",
        previous_flag_state=None,
        new_flag_state=flag_state.to_dict(),
        failure_reason=None,
    )
    event = build_execution_audit_event(
        event_id=make_release_execution_event_id(
            request.execution_id,
            "release_succeeded",
            "2026-07-03T09:00:00+08:00",
        ),
        execution_id=request.execution_id,
        intent_id=INTENT_ID,
        event_type="release_succeeded",
        actor="release_manager",
        timestamp="2026-07-03T09:00:00+08:00",
        payload=result.to_dict(),
        previous_event_hash="sha256:GENESIS",
    )

    assert request.to_dict()["target_flag_state"]["flag_name"] == FEATURE_FLAG_NAME
    assert result.to_dict()["status"] == "succeeded"
    assert event.to_dict()["event_hash"].startswith("sha256:")


@pytest.mark.parametrize("action", ["publish", "cancel", ""])
def test_request_rejects_unknown_action(action: str) -> None:
    payload = make_request().to_dict()
    payload["action"] = action

    with pytest.raises(ValueError, match="action must be one of"):
        ReleaseExecutionRequest(**payload)


def test_release_request_requires_enabled_target_flag() -> None:
    payload = make_request("release").to_dict()
    payload["target_flag_state"]["enabled"] = False

    with pytest.raises(ValueError, match="release target flag must be enabled"):
        ReleaseExecutionRequest(**payload)


def test_rollback_request_requires_rollback_target_and_disabled_flag() -> None:
    payload = make_request("rollback").to_dict()
    payload["rollback_target"] = None

    with pytest.raises(ValueError, match="rollback_target is required"):
        ReleaseExecutionRequest(**payload)

    payload = make_request("rollback").to_dict()
    payload["target_flag_state"]["enabled"] = True

    with pytest.raises(ValueError, match="rollback target flag must be disabled"):
        ReleaseExecutionRequest(**payload)


def test_result_requires_failure_reason_when_failed() -> None:
    request = make_request()

    with pytest.raises(ValueError, match="failure_reason is required"):
        ReleaseExecutionResult(
            result_id=make_release_execution_result_id(request.execution_id),
            execution_id=request.execution_id,
            intent_id=request.intent_id,
            action="release",
            status="failed",
            started_at="2026-07-03T09:00:00+08:00",
            finished_at="2026-07-03T09:00:00+08:00",
            actor="release_manager",
            previous_flag_state=None,
            new_flag_state=None,
            failure_reason=None,
        )


def test_payload_hash_is_canonical_and_rejects_secrets() -> None:
    assert canonical_execution_payload_hash({"b": 2, "a": 1}) == canonical_execution_payload_hash(
        {"a": 1, "b": 2}
    )

    with pytest.raises(ValueError, match="payload contains forbidden key"):
        canonical_execution_payload_hash({"deployment_credentials": "secret"})


def test_audit_event_hash_chain_uses_previous_hash() -> None:
    first = build_execution_audit_event(
        event_id="release_execution_audit_release_requested_1",
        execution_id="release_exec_1",
        intent_id=INTENT_ID,
        event_type="release_requested",
        actor="release_manager",
        timestamp="2026-07-03T09:00:00+08:00",
        payload={"a": 1},
        previous_event_hash="sha256:GENESIS",
    )
    second = build_execution_audit_event(
        event_id="release_execution_audit_release_succeeded_1",
        execution_id="release_exec_1",
        intent_id=INTENT_ID,
        event_type="release_succeeded",
        actor="release_manager",
        timestamp="2026-07-03T09:01:00+08:00",
        payload={"b": 2},
        previous_event_hash=first.event_hash,
    )

    assert second.previous_event_hash == first.event_hash
    assert second.event_hash != first.event_hash
