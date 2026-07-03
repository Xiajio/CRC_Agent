from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.release_execution_store import (
    ReleaseExecutionIntegrityError,
    ReleaseExecutionStore,
)
from src.contracts.release_execution import (
    FEATURE_FLAG_NAME,
    FeatureFlagState,
    ReleaseExecutionRequest,
    ReleaseExecutionResult,
    make_release_execution_id,
    make_release_execution_result_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
ROLLBACK_PLAN_ID = "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"


def request(action: str, key: str = "idem-1") -> ReleaseExecutionRequest:
    return ReleaseExecutionRequest(
        execution_id=make_release_execution_id(INTENT_ID, action, key),
        intent_id=INTENT_ID,
        action=action,
        requested_by="release_manager",
        requested_at="2026-07-03T09:00:00+08:00",
        idempotency_key=key,
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


def flag_state(execution_id: str, enabled: bool) -> FeatureFlagState:
    return FeatureFlagState(
        flag_name=FEATURE_FLAG_NAME,
        enabled=enabled,
        scope="feature_flag_candidate",
        source_intent_id=INTENT_ID,
        source_execution_id=execution_id,
        rollback_target="agent_policy_20260624_0",
        updated_by="release_manager",
        updated_at="2026-07-03T09:00:00+08:00",
    )


def result(req: ReleaseExecutionRequest, enabled: bool) -> ReleaseExecutionResult:
    return ReleaseExecutionResult(
        result_id=make_release_execution_result_id(req.execution_id),
        execution_id=req.execution_id,
        intent_id=req.intent_id,
        action=req.action,
        status="succeeded",
        started_at=req.requested_at,
        finished_at=req.requested_at,
        actor=req.requested_by,
        previous_flag_state=None,
        new_flag_state=flag_state(req.execution_id, enabled).to_dict(),
        failure_reason=None,
    )


def test_empty_execution_root_returns_verified_state_without_writes(tmp_path: Path) -> None:
    store = ReleaseExecutionStore(tmp_path / "reports" / "release_execution")

    state = store.read_state()

    assert state.integrity == {"status": "verified", "warnings": []}
    assert state.requests == []
    assert state.results == []
    assert state.feature_flag_state is None
    assert not (tmp_path / "reports" / "release_execution").exists()


def test_write_release_result_writes_only_execution_files(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "release_execution"
    store = ReleaseExecutionStore(root)
    req = request("release")
    res = result(req, enabled=True)
    current = flag_state(req.execution_id, enabled=True)

    store.write_successful_execution(req, res, current, timestamp=req.requested_at)

    written = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    assert written == [
        "audit/release_execution_20260703.jsonl",
        "feature_flags/current.json",
        f"feature_flags/history/{req.execution_id}.json",
        f"requests/{req.execution_id}.json",
        f"results/{res.result_id}.json",
    ]
    assert store.read_state().feature_flag_state["enabled"] is True


def test_same_idempotency_key_returns_existing_request_and_result(tmp_path: Path) -> None:
    store = ReleaseExecutionStore(tmp_path / "reports" / "release_execution")
    req = request("release", key="same-key")
    res = result(req, enabled=True)

    store.write_successful_execution(
        req,
        res,
        flag_state(req.execution_id, True),
        timestamp=req.requested_at,
    )

    match = store.find_by_idempotency_key("release", "same-key")

    assert match is not None
    assert match.request.execution_id == req.execution_id
    assert match.result is not None
    assert match.result.result_id == res.result_id


def test_idempotency_key_with_different_payload_fails(tmp_path: Path) -> None:
    store = ReleaseExecutionStore(tmp_path / "reports" / "release_execution")
    req = request("release", key="same-key")
    res = result(req, enabled=True)
    store.write_successful_execution(
        req,
        res,
        flag_state(req.execution_id, True),
        timestamp=req.requested_at,
    )

    changed = ReleaseExecutionRequest(**{**req.to_dict(), "reason": "Different reason."})

    with pytest.raises(ReleaseExecutionIntegrityError, match="idempotency key payload mismatch"):
        store.assert_idempotent_request_matches(changed)


def test_tampered_current_flag_blocks_writes(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "release_execution"
    store = ReleaseExecutionStore(root)
    req = request("release")
    res = result(req, enabled=True)
    store.write_successful_execution(
        req,
        res,
        flag_state(req.execution_id, True),
        timestamp=req.requested_at,
    )
    (root / "feature_flags" / "current.json").write_text("{bad json", encoding="utf-8")
    rollback_req = request("rollback", key="rollback-key")

    with pytest.raises(ReleaseExecutionIntegrityError, match="release execution integrity failed"):
        store.write_successful_execution(
            rollback_req,
            result(rollback_req, enabled=False),
            flag_state(rollback_req.execution_id, False),
            timestamp="2026-07-03T09:05:00+08:00",
        )
