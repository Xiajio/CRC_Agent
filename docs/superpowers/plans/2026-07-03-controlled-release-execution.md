# Controlled Release Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P2 Step 13 controlled local release execution so Agent Admin can execute and rollback a feature-flag release after Step 12 governance approval, with idempotency, preflight gates, audit events, and writes limited to `reports/release_execution/`.

**Architecture:** Add a new execution subsystem beside, not inside, Step 12 governance. Backend adds execution contracts, a file-backed execution store, a local feature flag executor, an execution service that gates against Step 12 governance and Step 11 dashboard state, and admin-only routes. Frontend extends the existing Agent Admin Release task with backend-derived execution readiness and release/rollback actions.

**Tech Stack:** Python 3.10, dataclasses, FastAPI, Pydantic v2, pytest, TypeScript, React, Vitest, Testing Library, existing Agent Admin UI components.

---

## Source Spec

Read before implementation:

- `docs/superpowers/specs/2026-07-03-controlled-release-execution-design.md`
- `docs/superpowers/specs/2026-07-02-controlled-release-governance-design.md`
- `docs/superpowers/specs/2026-06-30-agent-admin-release-dashboard-design.md`
- `docs/superpowers/specs/2026-06-30-evidenceclaim-literature-harness-design.md`
- `docs/superpowers/specs/2026-06-29-p1-clinical-review-loop-design.md`
- `docs/superpowers/specs/2026-06-29-p0-crc-safety-loop-design.md`

Step 13 may write only under `reports/release_execution/`. It must not mutate Step 12 governance files, release reports, harness reports, literature reports, safety policy, prompts, rubrics, routes, templates, RAG indexes, model files, tool manifests, patient/doctor state, deployment systems, or `CRC-client/`.

## File Structure

Backend contracts:

- Create `src/contracts/release_execution.py`
  - Dataclass contracts for `ReleaseExecutionRequest`, `ReleaseExecutionResult`, `FeatureFlagState`, and `ReleaseExecutionAuditEvent`.
  - Stable ID helpers, canonical hash helpers, and sensitive payload rejection.
- Create `tests/backend/test_release_execution_contract.py`
  - Contract validation and audit chain tests.

Backend persistence and execution:

- Create `backend/api/services/release_execution_store.py`
  - File-backed store rooted at `reports/release_execution/`.
  - Write-once requests/results/history, atomic current flag replace, append-only audit JSONL, integrity verification, idempotency lookup.
- Create `src/services/release_execution.py`
  - Preflight gates against `ReleaseGovernanceService.read_governance()` and Step 11 dashboard state.
  - Release and rollback orchestration.
- Create `tests/backend/test_release_execution_store.py`
  - Store integrity, idempotency, audit-chain, and atomic flag state tests.
- Create `tests/backend/test_release_execution_service.py`
  - Preflight and release/rollback behavior tests.

Backend API:

- Create `backend/api/schemas/release_execution.py`
  - Pydantic request/response schemas for execution routes.
- Modify `backend/api/routes/admin.py`
  - Add execution service factory and routes.
- Modify `backend/app.py`
  - Add execution routes to admin-token guard.
- Create `tests/backend/test_release_execution_api.py`
  - API route behavior and error mapping tests.
- Modify `tests/backend/test_auth_security.py`
  - Add release execution routes to the admin auth matrix.
- Create `tests/backend/test_release_execution_non_mutation.py`
  - Prove release and rollback execution mutate only `reports/release_execution/`.

Frontend API:

- Modify `frontend/src/app/api/types.ts`
  - Add execution response, request, result, flag state, preflight, and audit types.
- Modify `frontend/src/app/api/client.ts`
  - Add execution client methods.
- Modify `frontend/src/app/api/client.test.ts`
  - Add endpoint/header/body tests.
- Modify `frontend/src/test/test-utils.tsx`
  - Add default execution client stubs.

Frontend Agent Admin:

- Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`
  - Load execution state with the release task.
  - Add release/rollback mutation handlers.
- Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`
  - Add execution readiness, current flag state, release form, rollback form, and audit/result panels.
- Modify `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
  - Add execution loading, blocked, release success, rollback success, and error tests.
- Modify `frontend/src/styles/globals.css`
  - Add compact execution form/status styles if existing governance styles do not cover the new panel.

Docs:

- Create `reports/release_execution/README.md`
  - Documents local controlled execution artifacts.
- Do not commit generated runtime request/result/audit/flag files outside tests.

---

### Task 1: Release Execution Contracts

**Files:**
- Create: `src/contracts/release_execution.py`
- Create: `tests/backend/test_release_execution_contract.py`

- [ ] **Step 1: Write failing contract tests**

Create `tests/backend/test_release_execution_contract.py`:

```python
from __future__ import annotations

import pytest

from src.contracts.release_execution import (
    FEATURE_FLAG_NAME,
    FeatureFlagState,
    ReleaseExecutionAuditEvent,
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
```

- [ ] **Step 2: Run the contract tests to verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.contracts.release_execution'`.

- [ ] **Step 3: Implement the execution contracts**

Create `src/contracts/release_execution.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

FEATURE_FLAG_NAME = "doctor_review_cockpit_v0"
GENESIS_EXECUTION_EVENT_HASH = "sha256:GENESIS"
EXECUTION_ACTIONS = ("release", "rollback")
EXECUTION_RESULT_STATUSES = ("succeeded", "failed")
EXECUTION_EVENT_TYPES = (
    "release_requested",
    "release_succeeded",
    "release_failed",
    "rollback_requested",
    "rollback_succeeded",
    "rollback_failed",
    "execution_read",
)

ExecutionAction = Literal["release", "rollback"]
ExecutionResultStatus = Literal["succeeded", "failed"]
ExecutionAuditEventType = Literal[
    "release_requested",
    "release_succeeded",
    "release_failed",
    "rollback_requested",
    "rollback_succeeded",
    "rollback_failed",
    "execution_read",
]
```

Add frozen dataclasses with the fields from the design spec:

```python
@dataclass(frozen=True)
class ReleaseExecutionRequest:
    execution_id: str
    intent_id: str
    action: ExecutionAction
    requested_by: str
    requested_at: str
    idempotency_key: str
    reason: str
    expected_governance_hash: str
    expected_rollback_plan_id: str
    target_flag_state: dict[str, JsonValue]
    rollback_target: str | None = None


@dataclass(frozen=True)
class FeatureFlagState:
    flag_name: str
    enabled: bool
    scope: str
    source_intent_id: str
    source_execution_id: str
    rollback_target: str
    updated_by: str
    updated_at: str


@dataclass(frozen=True)
class ReleaseExecutionResult:
    result_id: str
    execution_id: str
    intent_id: str
    action: ExecutionAction
    status: ExecutionResultStatus
    started_at: str
    finished_at: str
    actor: str
    previous_flag_state: dict[str, JsonValue] | None
    new_flag_state: dict[str, JsonValue] | None
    failure_reason: str | None


@dataclass(frozen=True)
class ReleaseExecutionAuditEvent:
    event_id: str
    execution_id: str
    intent_id: str
    event_type: ExecutionAuditEventType
    actor: str
    timestamp: str
    payload_hash: str
    previous_event_hash: str
    event_hash: str
```

Validation rules:

- non-empty strings for IDs, actors, timestamps, reason, and idempotency key;
- action must be `release` or `rollback`;
- result status must be `succeeded` or `failed`;
- `failure_reason` is required for failed result and must be `None` for succeeded result;
- release target flag must be `{flag_name: FEATURE_FLAG_NAME, enabled: true, scope: "feature_flag_candidate"}`;
- rollback target flag must be `{flag_name: FEATURE_FLAG_NAME, enabled: false, scope: "feature_flag_candidate"}`;
- rollback request requires non-empty `rollback_target`;
- all hashes must match `sha256:[0-9a-f]{64}` except `sha256:GENESIS` where allowed.

Add helpers:

```python
def make_release_execution_id(intent_id: str, action: str, idempotency_key: str) -> str:
    _require_non_empty("intent_id", intent_id)
    _validate_choice("action", action, EXECUTION_ACTIONS)
    _require_non_empty("idempotency_key", idempotency_key)
    payload = {"intent_id": intent_id, "action": action, "idempotency_key": idempotency_key}
    return f"release_exec_{_slug(intent_id)}_{_slug(action)}_{_stable_hash(payload)}"


def make_release_execution_result_id(execution_id: str) -> str:
    _require_non_empty("execution_id", execution_id)
    return f"release_result_{_slug(execution_id)}"


def make_release_execution_event_id(execution_id: str, event_type: str, timestamp: str) -> str:
    _require_non_empty("execution_id", execution_id)
    _validate_choice("event_type", event_type, EXECUTION_EVENT_TYPES)
    _require_non_empty("timestamp", timestamp)
    payload = {"execution_id": execution_id, "event_type": event_type, "timestamp": timestamp}
    return f"release_execution_audit_{_slug(event_type)}_{_stable_hash(payload)}"


def canonical_execution_payload_hash(payload: JsonValue) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    stable_json = json.dumps(payload_copy, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()}"


def build_execution_audit_event(
    *,
    event_id: str,
    execution_id: str,
    intent_id: str,
    event_type: ExecutionAuditEventType,
    actor: str,
    timestamp: str,
    payload: JsonValue,
    previous_event_hash: str,
) -> ReleaseExecutionAuditEvent:
    payload_hash = canonical_execution_payload_hash(payload)
    event_payload: dict[str, JsonValue] = {
        "event_id": event_id,
        "execution_id": execution_id,
        "intent_id": intent_id,
        "event_type": event_type,
        "actor": actor,
        "timestamp": timestamp,
        "payload_hash": payload_hash,
        "previous_event_hash": previous_event_hash,
    }
    event_hash = canonical_execution_payload_hash(event_payload)
    return ReleaseExecutionAuditEvent(
        event_id=event_id,
        execution_id=execution_id,
        intent_id=intent_id,
        event_type=event_type,
        actor=actor,
        timestamp=timestamp,
        payload_hash=payload_hash,
        previous_event_hash=previous_event_hash,
        event_hash=event_hash,
    )


def validate_execution_audit_event_hash(event: ReleaseExecutionAuditEvent) -> bool:
    event_payload = event.to_dict()
    event_hash = event_payload.pop("event_hash")
    expected = canonical_execution_payload_hash(event_payload)
    if event_hash != expected:
        raise ValueError("event_hash does not match canonical execution audit event payload")
    return True
```

Use the Step 12 contract helper behavior as the exact validation model:

- stable JSON serialization with `sort_keys=True` and `separators=(",", ":")`;
- `_slug()` that keeps file-safe IDs;
- `_require_repo_relative_path()` if any future path field is added;
- recursive JSON safety validation;
- recursive forbidden payload key rejection including tokens, credentials, prompts, hidden reasoning, and patient identifiers.

- [ ] **Step 4: Run contract tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add src/contracts/release_execution.py tests/backend/test_release_execution_contract.py
git commit -m "feat: add release execution contracts"
```

---

### Task 2: Execution Store And Local Feature Flag Executor

**Files:**
- Create: `backend/api/services/release_execution_store.py`
- Create: `tests/backend/test_release_execution_store.py`
- Create: `reports/release_execution/README.md`

- [ ] **Step 1: Write failing store tests**

Create `tests/backend/test_release_execution_store.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.release_execution_store import (
    ReleaseExecutionStore,
    ReleaseExecutionIntegrityError,
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
        f"feature_flags/history/{req.execution_id}.json",
        "feature_flags/current.json",
        f"requests/{req.execution_id}.json",
        f"results/{res.result_id}.json",
    ]
    assert store.read_state().feature_flag_state["enabled"] is True


def test_same_idempotency_key_returns_existing_request_and_result(tmp_path: Path) -> None:
    store = ReleaseExecutionStore(tmp_path / "reports" / "release_execution")
    req = request("release", key="same-key")
    res = result(req, enabled=True)

    store.write_successful_execution(req, res, flag_state(req.execution_id, True), timestamp=req.requested_at)

    assert store.find_by_idempotency_key("release", "same-key").request.execution_id == req.execution_id
    assert store.find_by_idempotency_key("release", "same-key").result.result_id == res.result_id


def test_idempotency_key_with_different_payload_fails(tmp_path: Path) -> None:
    store = ReleaseExecutionStore(tmp_path / "reports" / "release_execution")
    req = request("release", key="same-key")
    res = result(req, enabled=True)
    store.write_successful_execution(req, res, flag_state(req.execution_id, True), timestamp=req.requested_at)

    changed = ReleaseExecutionRequest(**{**req.to_dict(), "reason": "Different reason."})

    with pytest.raises(ReleaseExecutionIntegrityError, match="idempotency key payload mismatch"):
        store.assert_idempotent_request_matches(changed)


def test_tampered_current_flag_blocks_writes(tmp_path: Path) -> None:
    root = tmp_path / "reports" / "release_execution"
    store = ReleaseExecutionStore(root)
    req = request("release")
    res = result(req, enabled=True)
    store.write_successful_execution(req, res, flag_state(req.execution_id, True), timestamp=req.requested_at)
    (root / "feature_flags" / "current.json").write_text("{bad json", encoding="utf-8")

    with pytest.raises(ReleaseExecutionIntegrityError, match="release execution integrity failed"):
        store.write_successful_execution(
            request("rollback", key="rollback-key"),
            result(request("rollback", key="rollback-key"), enabled=False),
            flag_state(request("rollback", key="rollback-key").execution_id, False),
            timestamp="2026-07-03T09:05:00+08:00",
        )
```

- [ ] **Step 2: Run store tests to verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_store.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `backend.api.services.release_execution_store`.

- [ ] **Step 3: Implement store and executor**

Create `backend/api/services/release_execution_store.py` with these public objects:

```python
class ReleaseExecutionIntegrityError(RuntimeError):
    """Raised when the release execution store is unsafe to append to."""


@dataclass(frozen=True)
class ReleaseExecutionState:
    requests: list[ReleaseExecutionRequest]
    results: list[ReleaseExecutionResult]
    feature_flag_state: dict[str, Any] | None
    audit_events: list[ReleaseExecutionAuditEvent]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class ReleaseExecutionIdempotencyMatch:
    request: ReleaseExecutionRequest
    result: ReleaseExecutionResult | None
```

Implement `ReleaseExecutionStore`:

```python
class ReleaseExecutionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.requests_dir = self.root / "requests"
        self.results_dir = self.root / "results"
        self.feature_flags_dir = self.root / "feature_flags"
        self.feature_flag_history_dir = self.feature_flags_dir / "history"
        self.audit_dir = self.root / "audit"

    def read_state(self) -> ReleaseExecutionState:
        return self._read_state_with_integrity()

    def find_by_idempotency_key(self, action: str, key: str) -> ReleaseExecutionIdempotencyMatch | None:
        return self._find_idempotency_match(action=action, key=key)

    def assert_idempotent_request_matches(self, request: ReleaseExecutionRequest) -> None:
        self._assert_idempotent_request_matches(request)

    def write_successful_execution(
        self,
        request: ReleaseExecutionRequest,
        result: ReleaseExecutionResult,
        feature_flag_state: FeatureFlagState,
        *,
        timestamp: str,
    ) -> None:
        self._write_successful_execution(
            request,
            result,
            feature_flag_state,
            timestamp=timestamp,
        )
```

Implementation requirements:

- Copy the Step 12 store path safety rules into this file for root, directory, file, symlink, and resolved path checks.
- Write request JSON with `open("x")`.
- Write result JSON with `open("x")`.
- Write feature flag history with `open("x")`.
- Replace `feature_flags/current.json` atomically using a temp file inside `feature_flags/` and `Path.replace()`.
- Append two audit events for successful release: `release_requested`, `release_succeeded`.
- Append two audit events for successful rollback: `rollback_requested`, `rollback_succeeded`.
- If any post-artifact audit append fails, remove newly written request/result/history files and keep current flag state unchanged.
- If current flag write fails after request/result write, remove newly written request/result/history files and do not append succeeded event.
- `read_state()` verifies artifact IDs match filenames, audit hashes match, event timestamps are monotonic per execution, and current flag JSON is valid when present.
- Integrity warnings produce `{"status": "failed", "warnings": ["warning message"]}`.

Create `reports/release_execution/README.md`:

```markdown
# Release Execution Artifacts

This directory is reserved for Step 13 controlled local release execution.

Runtime-generated files under `requests/`, `results/`, `feature_flags/`, and `audit/` are append-only execution evidence. They are created by admin-only release execution APIs and should not be edited manually.

Step 13 execution state is local and auditable. It does not call external deployment systems, store credentials, mutate clinical safety policy, mutate prompts, mutate RAG indexes, promote literature evidence, or change patient/doctor default paths.
```

- [ ] **Step 4: Run store tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_contract.py tests/backend/test_release_execution_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
git add backend/api/services/release_execution_store.py tests/backend/test_release_execution_store.py reports/release_execution/README.md
git commit -m "feat: add release execution store"
```

---

### Task 3: Release Execution Service

**Files:**
- Create: `src/services/release_execution.py`
- Create: `tests/backend/test_release_execution_service.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/backend/test_release_execution_service.py` with these fixtures and cases:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.release_execution_store import ReleaseExecutionStore
from src.services.release_execution import ReleaseExecutionPreflightError, ReleaseExecutionService


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
ROLLBACK_PLAN_ID = "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"


def dashboard() -> dict[str, object]:
    return {
        "version_chain": {
            "agent_policy_version": "agent_policy_20260629_0",
            "clinical_safety_policy_version": "crc_safety_policy_v0",
            "evidence_index_version": "rag_crc_guideline_20260620",
            "judge_rubric_version": "crc_rubric_v0",
        },
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_20260624_0",
        "summary": {
            "hard_fail_count": 0,
            "literature_isolation_violations": 0,
            "clinical_rag_ingest_enabled": False,
        },
        "runs": [
            {
                "run_id": "harness_20260629_001",
                "kind": "p0_crc_harness",
                "status": "pass",
                "source_path": "reports/harness/harness_20260629_001.json",
                "hard_fail_count": 0,
            },
            {
                "run_id": "release_safety_20260629_001",
                "kind": "release_safety",
                "status": "pass",
                "source_path": "reports/release_safety/release_safety_20260629_001.json",
                "hard_fail_count": 0,
            },
            {
                "run_id": "literature_harness_20260630_001",
                "kind": "literature_shadow_harness",
                "status": "shadow_only",
                "source_path": "reports/literature/literature_harness_20260630_001.json",
                "hard_fail_count": 0,
            },
        ],
    }


def governance(target_scope: str = "feature_flag_candidate", approved: bool = True) -> dict[str, object]:
    required = [
        {"role": "release_manager", "status": "approved" if approved else "missing", "latest_decision": "approve" if approved else None},
        {"role": "clinical_safety_reviewer", "status": "approved" if approved else "missing", "latest_decision": "approve" if approved else None},
        {"role": "evidence_reviewer", "status": "approved" if approved else "missing", "latest_decision": "approve" if approved else None},
    ]
    return {
        "dashboard_snapshot": {
            "version_chain": dashboard()["version_chain"],
            "release_decision": "feature_flag_or_pass",
            "rollback_target": "agent_policy_20260624_0",
            "hard_fail_count": 0,
            "literature_status": "shadow_only",
        },
        "active_intent": {
            "intent_id": INTENT_ID,
            "target_scope": target_scope,
            "derived_status": "approved" if approved else "pending_approval",
            "source_release_report_id": "release_safety_20260629_001",
            "release_decision_snapshot": "feature_flag_or_pass",
            "rollback_target": "agent_policy_20260624_0",
            "version_chain": dashboard()["version_chain"],
        },
        "required_approvals": required,
        "rollback_plan": {
            "rollback_plan_id": ROLLBACK_PLAN_ID,
            "intent_id": INTENT_ID,
            "rollback_target": "agent_policy_20260624_0",
            "owner": "release_manager",
            "status": "accepted",
            "verification_steps": ["Confirm release report id.", "Run P0 harness."],
            "created_at": "2026-07-03T08:50:00+08:00",
        },
        "integrity": {"status": "verified", "warnings": []},
    }


def service(tmp_path: Path, gov: dict[str, object] | None = None, dash: dict[str, object] | None = None) -> ReleaseExecutionService:
    return ReleaseExecutionService(
        store=ReleaseExecutionStore(tmp_path / "reports" / "release_execution"),
        governance_loader=lambda: gov if gov is not None else governance(),
        dashboard_loader=lambda: dash if dash is not None else dashboard(),
        now=lambda: "2026-07-03T09:00:00+08:00",
    )


def test_read_execution_returns_preflight_ready_for_approved_feature_flag_candidate(tmp_path: Path) -> None:
    model = service(tmp_path).read_execution()

    assert model["preflight"]["release"]["allowed"] is True
    assert model["preflight"]["release"]["reasons"] == []
    assert model["preflight"]["rollback"]["allowed"] is False


def test_release_blocks_shadow_scope(tmp_path: Path) -> None:
    app = service(tmp_path, gov=governance(target_scope="shadow"))

    with pytest.raises(ReleaseExecutionPreflightError, match="target_scope must be feature_flag_candidate"):
        app.execute_release(
            intent_id=INTENT_ID,
            requested_by="release_manager",
            idempotency_key="release-1",
            reason="Approved release.",
            expected_rollback_plan_id=ROLLBACK_PLAN_ID,
        )


def test_release_blocks_missing_approvals(tmp_path: Path) -> None:
    app = service(tmp_path, gov=governance(approved=False))

    with pytest.raises(ReleaseExecutionPreflightError, match="required approvals are incomplete"):
        app.execute_release(
            intent_id=INTENT_ID,
            requested_by="release_manager",
            idempotency_key="release-1",
            reason="Approved release.",
            expected_rollback_plan_id=ROLLBACK_PLAN_ID,
        )


def test_release_blocks_dashboard_drift(tmp_path: Path) -> None:
    drifted = dashboard()
    drifted["rollback_target"] = "agent_policy_20260620_0"
    app = service(tmp_path, dash=drifted)

    with pytest.raises(ReleaseExecutionPreflightError, match="dashboard rollback_target drifted"):
        app.execute_release(
            intent_id=INTENT_ID,
            requested_by="release_manager",
            idempotency_key="release-1",
            reason="Approved release.",
            expected_rollback_plan_id=ROLLBACK_PLAN_ID,
        )


def test_execute_release_then_rollback(tmp_path: Path) -> None:
    app = service(tmp_path)

    released = app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    assert released["feature_flag_state"]["enabled"] is True
    assert released["preflight"]["release"]["allowed"] is False
    assert released["preflight"]["rollback"]["allowed"] is True

    rolled_back = app.execute_rollback(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="rollback-1",
        reason="Rollback to approved target.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    assert rolled_back["feature_flag_state"]["enabled"] is False
    assert rolled_back["preflight"]["rollback"]["allowed"] is False


def test_release_is_idempotent_for_same_key(tmp_path: Path) -> None:
    app = service(tmp_path)
    first = app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    second = app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )

    assert second["results"] == first["results"]
```

- [ ] **Step 2: Run service tests to verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_service.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `src.services.release_execution`.

- [ ] **Step 3: Implement service**

Create `src/services/release_execution.py` with:

```python
class ReleaseExecutionPreflightError(ValueError):
    """Raised when controlled release execution gates are not satisfied."""


class ReleaseExecutionConflictError(ValueError):
    """Raised when execution conflicts with existing release state."""


class ReleaseExecutionService:
    def __init__(
        self,
        *,
        store: ReleaseExecutionStore,
        governance_loader: Callable[[], dict[str, Any]],
        dashboard_loader: Callable[[], dict[str, Any]],
        now: Callable[[], str],
    ) -> None:
        self._store = store
        self._governance_loader = governance_loader
        self._dashboard_loader = dashboard_loader
        self._now = now

    def read_execution(self) -> dict[str, Any]:
        return self._build_read_model()

    def execute_release(
        self,
        *,
        intent_id: str,
        requested_by: str,
        idempotency_key: str,
        reason: str,
        expected_rollback_plan_id: str,
    ) -> dict[str, Any]:
        return self._execute(
            action="release",
            intent_id=intent_id,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            reason=reason,
            expected_rollback_plan_id=expected_rollback_plan_id,
        )

    def execute_rollback(
        self,
        *,
        intent_id: str,
        requested_by: str,
        idempotency_key: str,
        reason: str,
        expected_rollback_plan_id: str,
    ) -> dict[str, Any]:
        return self._execute(
            action="rollback",
            intent_id=intent_id,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            reason=reason,
            expected_rollback_plan_id=expected_rollback_plan_id,
        )
```

Read model keys:

```python
{
    "governance": {
        "active_intent_id": "release_intent_release_safety_20260629_001_6da729a0",
        "derived_status": "approved",
        "required_approvals_complete": True,
        "rollback_plan_id": "rollback_plan_release_intent_release_safety_20260629_001_1b00f364",
    },
    "preflight": {
        "release": {"allowed": bool, "reasons": list[str]},
        "rollback": {"allowed": bool, "reasons": list[str]},
    },
    "feature_flag_state": dict | None,
    "requests": list[dict],
    "results": list[dict],
    "audit_events": list[dict],
    "integrity": dict,
    "runtime": {
        "auth": "admin",
        "source": "reports/release_execution",
        "mode": "controlled_local_execution",
    },
}
```

Release preflight reasons:

- no active governance intent;
- submitted intent does not match active intent;
- governance integrity is not verified;
- execution integrity is not verified;
- target_scope must be `feature_flag_candidate`;
- active intent is not approved;
- required approvals are incomplete;
- accepted rollback plan is missing;
- expected rollback plan id mismatch;
- dashboard release report drifted;
- dashboard release decision drifted;
- dashboard rollback_target drifted;
- dashboard version_chain drifted;
- hard_fail_count is not zero;
- literature status is not `shadow_only`;
- release already succeeded for this intent.

Rollback preflight reasons:

- no successful release execution exists for this intent;
- current feature flag is not enabled for this intent;
- accepted rollback plan is missing;
- expected rollback plan id mismatch;
- governance integrity is not verified;
- execution integrity is not verified.

Execution behavior:

- Build `expected_governance_hash` from active intent, required approvals, rollback plan, and dashboard snapshot using `canonical_execution_payload_hash`.
- Build `ReleaseExecutionRequest`.
- If store has same action/idempotency key and payload matches, return `read_execution()`.
- Release writes `FeatureFlagState(enabled=True)`.
- Rollback writes `FeatureFlagState(enabled=False)`.
- Use store `write_successful_execution()`.

- [ ] **Step 4: Run service tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_contract.py tests/backend/test_release_execution_store.py tests/backend/test_release_execution_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
git add src/services/release_execution.py tests/backend/test_release_execution_service.py
git commit -m "feat: add release execution service"
```

---

### Task 4: Admin Execution API And Auth

**Files:**
- Create: `backend/api/schemas/release_execution.py`
- Modify: `backend/api/routes/admin.py`
- Modify: `backend/app.py`
- Create: `tests/backend/test_release_execution_api.py`
- Modify: `tests/backend/test_auth_security.py`

- [ ] **Step 1: Write API and auth tests**

Create `tests/backend/test_release_execution_api.py`:

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app


def test_get_release_execution_delegates_to_service(monkeypatch) -> None:
    payload = {
        "governance": {"active_intent_id": None, "derived_status": None, "required_approvals_complete": False, "rollback_plan_id": None},
        "preflight": {"release": {"allowed": False, "reasons": ["no active governance intent"]}, "rollback": {"allowed": False, "reasons": ["no successful release execution exists"]}},
        "feature_flag_state": None,
        "requests": [],
        "results": [],
        "audit_events": [],
        "integrity": {"status": "verified", "warnings": []},
        "runtime": {"auth": "admin", "source": "reports/release_execution", "mode": "controlled_local_execution"},
    }

    class FakeService:
        def read_execution(self):
            return payload

    from backend.api.routes import admin

    monkeypatch.setattr(admin, "_release_execution_service", lambda: FakeService())
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/admin/release-execution")

    assert response.status_code == 200
    assert response.json() == payload


def test_post_release_execution_maps_preflight_conflict(monkeypatch) -> None:
    from backend.api.routes import admin
    from src.services.release_execution import ReleaseExecutionPreflightError

    class FakeService:
        def execute_release(self, **kwargs):
            raise ReleaseExecutionPreflightError("required approvals are incomplete")

    monkeypatch.setattr(admin, "_release_execution_service", lambda: FakeService())
    client = TestClient(create_app())

    response = client.post(
        "/api/admin/release-execution/release",
        json={
            "intent_id": "intent-1",
            "requested_by": "release_manager",
            "idempotency_key": "release-1",
            "reason": "Approved release.",
            "expected_rollback_plan_id": "rollback-1",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "required approvals are incomplete"


def test_release_execution_schema_rejects_extra_fields(monkeypatch) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/admin/release-execution/release",
        json={
            "intent_id": "intent-1",
            "requested_by": "release_manager",
            "idempotency_key": "release-1",
            "reason": "Approved release.",
            "expected_rollback_plan_id": "rollback-1",
            "deployment_credentials": "secret",
        },
    )

    assert response.status_code == 422
```

Extend `tests/backend/test_auth_security.py` route parameter lists with:

```python
("get", "/api/admin/release-execution"),
("post", "/api/admin/release-execution/release"),
("post", "/api/admin/release-execution/rollback"),
```

- [ ] **Step 2: Run API/auth tests to verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_api.py tests/backend/test_auth_security.py -q
```

Expected: FAIL because routes and guard are not implemented.

- [ ] **Step 3: Implement schemas**

Create `backend/api/schemas/release_execution.py`:

```python
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[str, Field(min_length=1)]


class ReleaseExecutionRequestPayload(BaseModel):
    intent_id: NonEmptyString
    requested_by: NonEmptyString
    idempotency_key: NonEmptyString
    reason: NonEmptyString
    expected_rollback_plan_id: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class ReleaseExecutionResponse(BaseModel):
    governance: dict[str, Any]
    preflight: dict[str, Any]
    feature_flag_state: dict[str, Any] | None
    requests: list[dict[str, Any]]
    results: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    integrity: dict[str, Any]
    runtime: dict[str, Any]

    model_config = ConfigDict(extra="forbid")
```

- [ ] **Step 4: Implement routes and auth**

Modify `backend/api/routes/admin.py`:

```python
from backend.api.schemas.release_execution import ReleaseExecutionRequestPayload
from backend.api.services.release_execution_store import ReleaseExecutionIntegrityError, ReleaseExecutionStore
from src.services.release_execution import (
    ReleaseExecutionConflictError,
    ReleaseExecutionPreflightError,
    ReleaseExecutionService,
)

_EXECUTION_STORE_ROOT = REPO_ROOT / "reports" / "release_execution"


def _release_execution_service() -> ReleaseExecutionService:
    return ReleaseExecutionService(
        store=ReleaseExecutionStore(_EXECUTION_STORE_ROOT),
        governance_loader=_release_governance_service().read_governance,
        dashboard_loader=build_release_dashboard,
        now=_governance_timestamp,
    )
```

Add error mapping:

```python
def _raise_execution_http_error(exc: Exception) -> None:
    if isinstance(exc, (ReleaseExecutionConflictError, ReleaseExecutionPreflightError, ReleaseExecutionIntegrityError, FileExistsError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, OSError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc
```

Add routes:

```python
@router.get("/release-execution")
async def get_admin_release_execution() -> dict[str, Any]:
    return _release_execution_service().read_execution()


@router.post("/release-execution/release")
async def execute_admin_release(payload: ReleaseExecutionRequestPayload) -> dict[str, Any]:
    try:
        return _release_execution_service().execute_release(**_model_dump(payload))
    except Exception as exc:
        _raise_execution_http_error(exc)


@router.post("/release-execution/rollback")
async def execute_admin_release_rollback(payload: ReleaseExecutionRequestPayload) -> dict[str, Any]:
    try:
        return _release_execution_service().execute_rollback(**_model_dump(payload))
    except Exception as exc:
        _raise_execution_http_error(exc)
```

Modify `backend/app.py`:

```python
if method == "GET" and path == "/api/admin/release-execution":
    return True
if method == "POST" and path.startswith("/api/admin/release-execution/"):
    return True
```

- [ ] **Step 5: Run API/auth tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_api.py tests/backend/test_auth_security.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add backend/api/schemas/release_execution.py backend/api/routes/admin.py backend/app.py tests/backend/test_release_execution_api.py tests/backend/test_auth_security.py
git commit -m "feat: add release execution admin api"
```

---

### Task 5: Frontend API Types And Client

**Files:**
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/test/test-utils.tsx`

- [ ] **Step 1: Write client tests**

Add to `frontend/src/app/api/client.test.ts`:

```ts
it("loads admin release execution with configured Authorization headers", async () => {
  const payload = {
    governance: { active_intent_id: null, derived_status: null, required_approvals_complete: false, rollback_plan_id: null },
    preflight: {
      release: { allowed: false, reasons: ["no active governance intent"] },
      rollback: { allowed: false, reasons: ["no successful release execution exists"] },
    },
    feature_flag_state: null,
    requests: [],
    results: [],
    audit_events: [],
    integrity: { status: "verified", warnings: [] },
    runtime: { auth: "admin", source: "reports/release_execution", mode: "controlled_local_execution" },
  };
  const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(payload));
  const client = createApiClient({
    baseUrl: "http://127.0.0.1:8000",
    fetchImpl,
    headers: { Authorization: "Bearer admin-token" },
  });

  await expect(client.getAdminReleaseExecution()).resolves.toEqual(payload);

  expect(fetchImpl).toHaveBeenCalledWith("http://127.0.0.1:8000/api/admin/release-execution", {
    headers: { Authorization: "Bearer admin-token" },
  });
});

it("executes admin release and rollback with JSON request bodies", async () => {
  const payload = {
    governance: { active_intent_id: "intent-1", derived_status: "approved", required_approvals_complete: true, rollback_plan_id: "rollback-1" },
    preflight: {
      release: { allowed: true, reasons: [] },
      rollback: { allowed: false, reasons: ["no successful release execution exists"] },
    },
    feature_flag_state: null,
    requests: [],
    results: [],
    audit_events: [],
    integrity: { status: "verified", warnings: [] },
    runtime: { auth: "admin", source: "reports/release_execution", mode: "controlled_local_execution" },
  };
  const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(payload));
  const client = createApiClient({
    baseUrl: "http://127.0.0.1:8000",
    fetchImpl,
    headers: { Authorization: "Bearer admin-token" },
  });
  const request = {
    intent_id: "intent-1",
    requested_by: "release_manager",
    idempotency_key: "release-1",
    reason: "Approved release.",
    expected_rollback_plan_id: "rollback-1",
  };

  await client.executeAdminRelease(request);
  const rollbackRequest = {
    intent_id: request.intent_id,
    requested_by: request.requested_by,
    idempotency_key: "rollback-1",
    reason: request.reason,
    expected_rollback_plan_id: request.expected_rollback_plan_id,
  };

  await client.executeAdminReleaseRollback(rollbackRequest);

  expect(fetchImpl).toHaveBeenNthCalledWith(1, "http://127.0.0.1:8000/api/admin/release-execution/release", {
    method: "POST",
    headers: expect.any(Headers),
    body: JSON.stringify(request),
  });
  expect(fetchImpl).toHaveBeenNthCalledWith(2, "http://127.0.0.1:8000/api/admin/release-execution/rollback", {
    method: "POST",
    headers: expect.any(Headers),
    body: JSON.stringify(rollbackRequest),
  });
});
```

- [ ] **Step 2: Run client tests to verify failure**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts
```

Expected: FAIL because execution client methods do not exist.

- [ ] **Step 3: Add frontend types**

Modify `frontend/src/app/api/types.ts` with:

```ts
export interface AdminReleaseExecutionPreflightAction {
  allowed: boolean;
  reasons: string[];
}

export interface AdminReleaseExecutionGovernanceSummary {
  active_intent_id: string | null;
  derived_status: AdminReleaseIntentStatus | null;
  required_approvals_complete: boolean;
  rollback_plan_id: string | null;
}

export interface AdminFeatureFlagState {
  flag_name: "doctor_review_cockpit_v0";
  enabled: boolean;
  scope: "feature_flag_candidate";
  source_intent_id: string;
  source_execution_id: string;
  rollback_target: string;
  updated_by: string;
  updated_at: string;
}

export interface AdminReleaseExecutionRequestRecord {
  execution_id: string;
  intent_id: string;
  action: "release" | "rollback";
  requested_by: string;
  requested_at: string;
  idempotency_key: string;
  reason: string;
  expected_governance_hash: string;
  expected_rollback_plan_id: string;
  target_flag_state: Record<string, JsonValue | unknown>;
  rollback_target: string | null;
}

export interface AdminReleaseExecutionResultRecord {
  result_id: string;
  execution_id: string;
  intent_id: string;
  action: "release" | "rollback";
  status: "succeeded" | "failed";
  started_at: string;
  finished_at: string;
  actor: string;
  previous_flag_state: AdminFeatureFlagState | null;
  new_flag_state: AdminFeatureFlagState | null;
  failure_reason: string | null;
}

export interface AdminReleaseExecutionAuditEvent {
  event_id: string;
  execution_id: string;
  intent_id: string;
  event_type:
    | "release_requested"
    | "release_succeeded"
    | "release_failed"
    | "rollback_requested"
    | "rollback_succeeded"
    | "rollback_failed"
    | "execution_read";
  actor: string;
  timestamp: string;
  payload_hash: string;
  previous_event_hash: string;
  event_hash: string;
}

export interface AdminReleaseExecutionResponse {
  governance: AdminReleaseExecutionGovernanceSummary;
  preflight: {
    release: AdminReleaseExecutionPreflightAction;
    rollback: AdminReleaseExecutionPreflightAction;
  };
  feature_flag_state: AdminFeatureFlagState | null;
  requests: AdminReleaseExecutionRequestRecord[];
  results: AdminReleaseExecutionResultRecord[];
  audit_events: AdminReleaseExecutionAuditEvent[];
  integrity: AdminReleaseGovernanceIntegrity;
  runtime: {
    auth: "admin";
    source: "reports/release_execution";
    mode: "controlled_local_execution";
  };
}

export interface AdminExecuteReleaseRequest {
  intent_id: string;
  requested_by: string;
  idempotency_key: string;
  reason: string;
  expected_rollback_plan_id: string;
}
```

- [ ] **Step 4: Add client methods**

Modify `frontend/src/app/api/client.ts`:

```ts
getAdminReleaseExecution(): Promise<AdminReleaseExecutionResponse>;
executeAdminRelease(request: AdminExecuteReleaseRequest): Promise<AdminReleaseExecutionResponse>;
executeAdminReleaseRollback(request: AdminExecuteReleaseRequest): Promise<AdminReleaseExecutionResponse>;
```

Implementation:

```ts
async getAdminReleaseExecution() {
  const response = await fetchImpl(buildUrl("/api/admin/release-execution", baseUrl), {
    headers: defaultHeaders,
  });
  return parseJsonResponse<AdminReleaseExecutionResponse>(response);
},

async executeAdminRelease(request) {
  const response = await fetchImpl(buildUrl("/api/admin/release-execution/release", baseUrl), {
    method: "POST",
    headers: buildJsonHeaders(defaultHeaders),
    body: JSON.stringify(request),
  });
  return parseJsonResponse<AdminReleaseExecutionResponse>(response);
},

async executeAdminReleaseRollback(request) {
  const response = await fetchImpl(buildUrl("/api/admin/release-execution/rollback", baseUrl), {
    method: "POST",
    headers: buildJsonHeaders(defaultHeaders),
    body: JSON.stringify(request),
  });
  return parseJsonResponse<AdminReleaseExecutionResponse>(response);
},
```

Add default stubs in `frontend/src/test/test-utils.tsx` so full API client mocks satisfy `ApiClient`.

- [ ] **Step 5: Run client tests and build**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts
cmd /c D:\anaconda3\envs\LangG\npm.cmd run build
```

Expected: client tests PASS; build PASS with any existing Vite chunk-size warning unchanged.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/test/test-utils.tsx
git commit -m "feat: add release execution frontend api"
```

---

### Task 6: Agent Admin Execution UI

**Files:**
- Modify: `frontend/src/features/agent-admin/agent-admin-view.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-pages.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
- Modify: `frontend/src/styles/globals.css` if layout needs new execution form styles.

- [ ] **Step 1: Write UI tests**

Add to `frontend/src/features/agent-admin/agent-admin-view.test.tsx`:

```ts
it("renders release execution preflight and blocked reasons", async () => {
  const apiClient = buildApiClientStub({
    getAdminReleaseExecution: vi.fn().mockResolvedValue({
      governance: { active_intent_id: null, derived_status: null, required_approvals_complete: false, rollback_plan_id: null },
      preflight: {
        release: { allowed: false, reasons: ["no active governance intent"] },
        rollback: { allowed: false, reasons: ["no successful release execution exists"] },
      },
      feature_flag_state: null,
      requests: [],
      results: [],
      audit_events: [],
      integrity: { status: "verified", warnings: [] },
      runtime: { auth: "admin", source: "reports/release_execution", mode: "controlled_local_execution" },
    }),
  });

  renderAgentAdmin({ apiClient });
  await openReleasePage();

  expect(await screen.findByText("Release execution")).toBeInTheDocument();
  expect(screen.getByText("no active governance intent")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Execute release/i })).toBeDisabled();
  expect(screen.getByRole("button", { name: /Execute rollback/i })).toBeDisabled();
});

it("executes release when backend preflight allows it", async () => {
  const executeAdminRelease = vi.fn().mockResolvedValue(executionResponse({ released: true }));
  const apiClient = buildApiClientStub({
    getAdminReleaseExecution: vi.fn().mockResolvedValue(executionResponse({ releaseAllowed: true })),
    executeAdminRelease,
  });

  renderAgentAdmin({ apiClient });
  await openReleasePage();
  await userEvent.type(screen.getByLabelText(/Execution actor/i), "release_manager");
  await userEvent.type(screen.getByLabelText(/Execution reason/i), "Approved release.");
  await userEvent.type(screen.getByLabelText(/Idempotency key/i), "release-1");
  await userEvent.type(screen.getByLabelText(/Expected rollback plan/i), "rollback-1");
  await userEvent.click(screen.getByRole("button", { name: /Execute release/i }));

  expect(executeAdminRelease).toHaveBeenCalledWith({
    intent_id: "intent-1",
    requested_by: "release_manager",
    idempotency_key: "release-1",
    reason: "Approved release.",
    expected_rollback_plan_id: "rollback-1",
  });
  expect(await screen.findByText("enabled / true")).toBeInTheDocument();
});
```

Use existing local helpers in the test file. Add a compact `executionResponse()` helper near existing release governance fixtures.

- [ ] **Step 2: Run UI tests to verify failure**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: FAIL because execution resource and UI do not exist.

- [ ] **Step 3: Load execution state and handlers**

Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`:

- Add `AgentAdminReleaseExecutionResource`.
- Add `AgentAdminReleaseExecutionActionState`.
- Load `apiClient.getAdminReleaseExecution()` when active task is `release`.
- Add `executeRelease()` and `executeRollback()` handlers that set action state, call client methods, and replace the execution resource with server response.
- Pass execution resource, action state, and action handlers into `AgentAdminTaskPages`.

Required handler shape:

```ts
export type AgentAdminReleaseExecutionActions = {
  executeRelease: (request: AdminExecuteReleaseRequest) => Promise<void>;
  executeRollback: (request: AdminExecuteReleaseRequest) => Promise<void>;
};
```

- [ ] **Step 4: Render execution panel**

Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`:

- Add execution props.
- Render `ReleaseExecutionPanel` below `ReleaseGovernanceForms`.
- Use `resource.data.preflight.release.allowed` and `resource.data.preflight.rollback.allowed` for button state.
- Show all backend preflight reasons.
- Show current flag state as:
  - `flag / doctor_review_cockpit_v0`
  - `enabled / true|false`
  - `intent / <source_intent_id>`
  - `updated / <updated_at>`
- Show latest results and audit events.

Use controlled inputs:

- actor default `release_manager`;
- reason empty and required;
- idempotency key empty and required;
- expected rollback plan id default from `execution.governance.rollback_plan_id ?? ""`.

- [ ] **Step 5: Add or adjust styles**

Modify `frontend/src/styles/globals.css` only if the new execution form needs layout support. Reuse existing `agent-admin-governance-form-grid` where possible.

- [ ] **Step 6: Run UI tests and build**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/agent-admin/agent-admin-view.test.tsx
cmd /c D:\anaconda3\envs\LangG\npm.cmd run build
```

Expected: frontend tests PASS; build PASS with any existing Vite chunk-size warning unchanged.

- [ ] **Step 7: Commit Task 6**

Run:

```powershell
git add frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx frontend/src/styles/globals.css
git commit -m "feat: add release execution admin ui"
```

---

### Task 7: Non-Mutation Tests And Final Backend Verification

**Files:**
- Create: `tests/backend/test_release_execution_non_mutation.py`
- Modify source files only if this task exposes a bug.

- [ ] **Step 1: Write non-mutation tests**

Create `tests/backend/test_release_execution_non_mutation.py`:

```python
from __future__ import annotations

from pathlib import Path

from backend.api.services.release_execution_store import ReleaseExecutionStore
from src.services.release_execution import ReleaseExecutionService

from tests.backend.test_release_execution_service import INTENT_ID, ROLLBACK_PLAN_ID, dashboard, governance


def read_if_exists(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def test_release_and_rollback_write_only_execution_root(tmp_path: Path) -> None:
    protected_paths = {
        "governance": tmp_path / "reports" / "release_governance" / "intents" / f"{INTENT_ID}.json",
        "harness": tmp_path / "reports" / "harness" / "harness_20260629_001.json",
        "release": tmp_path / "reports" / "release_safety" / "release_safety_20260629_001.json",
        "literature": tmp_path / "reports" / "literature" / "literature_harness_20260630_001.json",
        "safety": tmp_path / "config" / "safety_policy.yaml",
        "prompt": tmp_path / "src" / "prompts" / "decision_prompts.py",
    }
    for label, path in protected_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{label}: original\n", encoding="utf-8")
    before = {label: read_if_exists(path) for label, path in protected_paths.items()}
    app = ReleaseExecutionService(
        store=ReleaseExecutionStore(tmp_path / "reports" / "release_execution"),
        governance_loader=governance,
        dashboard_loader=dashboard,
        now=lambda: "2026-07-03T09:00:00+08:00",
    )

    app.execute_release(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="release-1",
        reason="Approved release.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )
    app.execute_rollback(
        intent_id=INTENT_ID,
        requested_by="release_manager",
        idempotency_key="rollback-1",
        reason="Rollback to approved target.",
        expected_rollback_plan_id=ROLLBACK_PLAN_ID,
    )

    assert {label: read_if_exists(path) for label, path in protected_paths.items()} == before
    written = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in (tmp_path / "reports" / "release_execution").rglob("*")
        if path.is_file()
    )
    assert all(path.startswith("reports/release_execution/") for path in written)
    assert "reports/release_execution/feature_flags/current.json" in written
```

- [ ] **Step 2: Run Step 13 backend tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_contract.py tests/backend/test_release_execution_store.py tests/backend/test_release_execution_service.py tests/backend/test_release_execution_api.py tests/backend/test_release_execution_non_mutation.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Step 12 regressions**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_governance_contract.py tests/backend/test_release_governance_store.py tests/backend/test_release_governance_service.py tests/backend/test_release_governance_api.py tests/backend/test_release_governance_non_mutation.py -q
```

Expected: PASS.

- [ ] **Step 4: Run Step 11 regressions**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_auth_security.py -q
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Run Step 10, P1, and P0 regressions**

Run Step 10:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
```

Expected: PASS.

Run P1:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
```

Expected: PASS.

Run P0:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py -q
```

Expected: PASS.

- [ ] **Step 6: Scope and generated-file check**

Run:

```powershell
git diff --check
git status --short
git diff --name-only
```

Expected:

- `git diff --check` has no output.
- No generated `reports/release_execution/requests/`, `results/`, `feature_flags/current.json`, `feature_flags/history/`, or `audit/*.jsonl` files are staged.
- `reports/release_execution/README.md` may be staged.
- No files under `CRC-client/` changed.
- No files under `reports/release_governance/`, `reports/harness/`, `reports/release_safety/`, or `reports/literature/` changed except intentional test fixtures.

- [ ] **Step 7: Commit Task 7 fixes if any**

If Task 7 exposes bugs, commit fixes:

```powershell
git add tests/backend/test_release_execution_non_mutation.py src/contracts/release_execution.py src/services/release_execution.py backend/api/services/release_execution_store.py backend/api/schemas/release_execution.py backend/api/routes/admin.py backend/app.py frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx frontend/src/app/api/client.test.ts frontend/src/test/test-utils.tsx frontend/src/styles/globals.css
git commit -m "fix: finalize release execution integration"
```

If Task 7 requires no fixes, do not create an empty commit.

## Final Handoff

When all tasks pass, report:

- Step 13 backend test results;
- Step 13 frontend test/build results;
- Step 12 governance regression results;
- Step 11 dashboard/auth/frontend regression results;
- Step 10/P1/P0 regression results;
- final changed file list;
- confirmation that Step 13 writes only under `reports/release_execution/`;
- confirmation that no external deployment, subprocess, network, safety policy, prompt, RAG, literature promotion, patient/doctor path, or `CRC-client/` mutation was added;
- current branch and whether it has been pushed.

Do not push to `origin/main` unless the user explicitly asks.

## Self-Review

Spec coverage: Task 1 covers execution contracts. Task 2 covers file-backed execution persistence and local flag state. Task 3 covers preflight and service orchestration. Task 4 covers API and admin auth. Task 5 covers frontend API. Task 6 covers Agent Admin UI. Task 7 covers non-mutation and regressions.

Marker scan: the plan contains no unresolved work markers and no unspecified file paths.

Type consistency: backend response keys match frontend interfaces: `governance`, `preflight`, `feature_flag_state`, `requests`, `results`, `audit_events`, `integrity`, and `runtime`. Client method names are consistently `getAdminReleaseExecution()`, `executeAdminRelease()`, and `executeAdminReleaseRollback()`.

Scope check: the plan implements controlled local execution only. It does not add external deployment, shell execution, network calls, real production credentials, safety policy edits, prompt edits, RAG writes, literature promotion, patient/doctor path mutation, or `CRC-client/` edits.
