# Post-Release Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P2 Step 14 post-release monitoring so Agent Admin can record audited post-release checks, see derived alerts, and receive rollback trigger recommendations after Step 13 controlled local release execution.

**Architecture:** Add a Step 14 monitoring subsystem beside, not inside, Step 13 execution. Backend adds monitoring contracts, a file-backed monitoring store, a service that derives alerts from execution/governance/dashboard/check state, and admin-only routes. Frontend extends the existing Agent Admin Release page with backend-derived monitoring status, check recording, alert acknowledgement, and rollback trigger recommendation.

**Tech Stack:** Python 3.10, dataclasses, FastAPI, Pydantic v2, pytest, TypeScript, React, Vitest, Testing Library, existing Agent Admin UI components.

---

## Source Spec

Read before implementation:

- `docs/superpowers/specs/2026-07-03-post-release-monitoring-design.md`
- `docs/superpowers/specs/2026-07-03-controlled-release-execution-design.md`
- `docs/superpowers/specs/2026-07-02-controlled-release-governance-design.md`
- `docs/superpowers/specs/2026-06-30-agent-admin-release-dashboard-design.md`
- `docs/superpowers/specs/2026-06-30-evidenceclaim-literature-harness-design.md`
- `docs/superpowers/specs/2026-06-29-p1-clinical-review-loop-design.md`
- `docs/superpowers/specs/2026-06-29-p0-crc-safety-loop-design.md`

Step 14 may write only under `reports/release_monitoring/`. It must not mutate Step 13 execution files, Step 12 governance files, release reports, harness reports, literature reports, safety policy, prompts, rubrics, routes, templates, RAG indexes, model files, tool manifests, patient/doctor state, deployment systems, or `CRC-client/`.

## File Structure

Backend contracts:

- Create `src/contracts/release_monitoring.py`
  - Dataclass contracts for `ReleaseMonitoringCheck`, `ReleaseMonitoringAlert`, `ReleaseRollbackTriggerCandidate`, `ReleaseMonitoringAcknowledgement`, and `ReleaseMonitoringAuditEvent`.
  - Stable ID helpers, canonical hash helpers, JSON safety validation, and forbidden payload key rejection.
- Create `tests/backend/test_release_monitoring_contract.py`
  - Contract validation, hash-chain, alert serialization, rollback candidate, and forbidden payload tests.

Backend persistence and monitoring:

- Create `backend/api/services/release_monitoring_store.py`
  - File-backed store rooted at `reports/release_monitoring/`.
  - Write-once checks and acknowledgements, append-only audit JSONL, integrity verification, and idempotency lookup.
- Create `src/services/release_monitoring.py`
  - Read Step 13 execution, Step 12 governance, and Step 11 dashboard state.
  - Derive monitoring status, required checks, alerts, and rollback trigger candidate.
- Create `tests/backend/test_release_monitoring_store.py`
  - Store integrity, idempotency, audit-chain, and read-only behavior tests.
- Create `tests/backend/test_release_monitoring_service.py`
  - Monitoring state, alert derivation, rollback trigger, and acknowledgement behavior tests.

Backend API:

- Create `backend/api/schemas/release_monitoring.py`
  - Pydantic request schemas for check recording and alert acknowledgement.
- Modify `backend/api/routes/admin.py`
  - Add monitoring service factory and routes.
- Modify `backend/app.py`
  - Add monitoring routes to admin-token guard.
- Create `tests/backend/test_release_monitoring_api.py`
  - API route behavior and error mapping tests.
- Modify `tests/backend/test_auth_security.py`
  - Add release monitoring routes to the admin auth matrix.
- Create `tests/backend/test_release_monitoring_non_mutation.py`
  - Prove Step 14 writes only under `reports/release_monitoring/`.
- Modify `.gitignore`
  - Add precise whitelist entries for new Step 14 backend tests.

Frontend API:

- Modify `frontend/src/app/api/types.ts`
  - Add monitoring response, check, alert, acknowledgement, rollback trigger, and request types.
- Modify `frontend/src/app/api/client.ts`
  - Add monitoring client methods.
- Modify `frontend/src/app/api/client.test.ts`
  - Add endpoint/header/body tests.
- Modify `frontend/src/test/test-utils.tsx`
  - Add default monitoring client stubs.

Frontend Agent Admin:

- Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`
  - Load monitoring state with the release task.
  - Add check recording and alert acknowledgement handlers.
- Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`
  - Add monitoring status, required checks, check form, alert list, acknowledgement form, and rollback trigger panel.
- Modify `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
  - Add monitoring loading, idle, active, alert, check submit, acknowledgement submit, and error tests.
- Modify `frontend/src/styles/globals.css`
  - Add compact monitoring status/form styles only when existing release styles are insufficient.

Docs:

- Create `reports/release_monitoring/README.md`
  - Documents local monitoring artifacts.
- Do not commit runtime-generated check, acknowledgement, or audit files outside tests.

---

### Task 1: Monitoring Contracts

**Files:**
- Create: `src/contracts/release_monitoring.py`
- Create: `tests/backend/test_release_monitoring_contract.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add test whitelist entries**

Modify `.gitignore` near the backend test whitelist:

```gitignore
!tests/backend/test_release_monitoring_contract.py
!tests/backend/test_release_monitoring_store.py
!tests/backend/test_release_monitoring_service.py
!tests/backend/test_release_monitoring_api.py
!tests/backend/test_release_monitoring_non_mutation.py
```

- [ ] **Step 2: Write failing contract tests**

Create `tests/backend/test_release_monitoring_contract.py`:

```python
from __future__ import annotations

import pytest

from src.contracts.release_monitoring import (
    GENESIS_MONITORING_EVENT_HASH,
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

    assert check.to_dict()["check_type"] == "p0_harness_replay"
    assert alert.to_dict()["recommended_action"] == "execute_step13_rollback"
    assert candidate.to_dict()["rollback_plan_id"] == ROLLBACK_PLAN_ID
    assert acknowledgement.to_dict()["disposition"] == "investigating"
    assert event.to_dict()["event_hash"].startswith("sha256:")


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


def test_evidence_refs_must_not_be_absolute_paths() -> None:
    payload = make_check().to_dict()
    payload["evidence_refs"] = ["D:/YiZhu_Agnet/LangG/reports/harness/harness_20260629_001.json"]

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
```

- [ ] **Step 3: Run RED contract tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_contract.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.contracts.release_monitoring'`.

- [ ] **Step 4: Implement monitoring contracts**

Create `src/contracts/release_monitoring.py` with these exported names and validation rules:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Sequence, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

GENESIS_MONITORING_EVENT_HASH = "sha256:GENESIS"

MonitoringCheckType = Literal[
    "execution_integrity",
    "governance_drift",
    "p0_harness_replay",
    "agent_admin_smoke",
    "doctor_review_smoke",
    "literature_isolation",
    "manual_operator_note",
]
MonitoringCheckStatus = Literal["pass", "warning", "fail"]
MonitoringAlertSeverity = Literal["info", "warning", "critical"]
MonitoringAlertCategory = Literal[
    "missing_required_check",
    "post_release_check_failed",
    "execution_integrity_failed",
    "governance_drift",
    "feature_flag_state_mismatch",
    "rollback_ready",
]
MonitoringAlertStatus = Literal["active", "acknowledged"]
MonitoringRecommendedAction = Literal["observe", "investigate", "prepare_rollback", "execute_step13_rollback"]
MonitoringAcknowledgementDisposition = Literal["investigating", "accepted_risk", "rollback_started_elsewhere", "false_positive"]
MonitoringAuditEventType = Literal["check_recorded", "alert_acknowledged", "monitoring_read"]

MONITORING_CHECK_TYPES: Sequence[str] = (
    "execution_integrity",
    "governance_drift",
    "p0_harness_replay",
    "agent_admin_smoke",
    "doctor_review_smoke",
    "literature_isolation",
    "manual_operator_note",
)
MONITORING_CHECK_STATUSES: Sequence[str] = ("pass", "warning", "fail")
MONITORING_ALERT_SEVERITIES: Sequence[str] = ("info", "warning", "critical")
MONITORING_ALERT_CATEGORIES: Sequence[str] = (
    "missing_required_check",
    "post_release_check_failed",
    "execution_integrity_failed",
    "governance_drift",
    "feature_flag_state_mismatch",
    "rollback_ready",
)
MONITORING_ALERT_STATUSES: Sequence[str] = ("active", "acknowledged")
MONITORING_RECOMMENDED_ACTIONS: Sequence[str] = ("observe", "investigate", "prepare_rollback", "execute_step13_rollback")
MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS: Sequence[str] = (
    "investigating",
    "accepted_risk",
    "rollback_started_elsewhere",
    "false_positive",
)
MONITORING_EVENT_TYPES: Sequence[str] = ("check_recorded", "alert_acknowledged", "monitoring_read")
FORBIDDEN_MONITORING_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "deployment_credentials",
        "hidden_reasoning",
        "chain_of_thought",
        "medical_record_number",
        "mrn",
        "patient_id",
        "patient_identifier",
        "patient_name",
        "patient_number",
        "password",
        "private_key",
        "prompt",
        "secret",
        "session_token",
        "token",
    }
)


@dataclass(frozen=True)
class ReleaseMonitoringCheck:
    check_id: str
    intent_id: str
    execution_id: str
    check_type: MonitoringCheckType
    status: MonitoringCheckStatus
    observed_by: str
    observed_at: str
    summary: str
    evidence_refs: list[str]
    metrics: dict[str, JsonValue]
    idempotency_key: str

    def __post_init__(self) -> None:
        _require_non_empty("check_id", self.check_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("execution_id", self.execution_id)
        _validate_choice("check_type", self.check_type, MONITORING_CHECK_TYPES)
        _validate_choice("status", self.status, MONITORING_CHECK_STATUSES)
        _require_non_empty("observed_by", self.observed_by)
        _require_non_empty("observed_at", self.observed_at)
        _require_non_empty("summary", self.summary)
        _require_non_empty("idempotency_key", self.idempotency_key)
        object.__setattr__(self, "evidence_refs", tuple(_validate_evidence_refs(self.evidence_refs)))
        object.__setattr__(self, "metrics", _freeze_json_safe(self.metrics, path="metrics"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "check_type": self.check_type,
            "status": self.status,
            "observed_by": self.observed_by,
            "observed_at": self.observed_at,
            "summary": self.summary,
            "evidence_refs": list(self.evidence_refs),
            "metrics": _copy_frozen_json_safe(self.metrics, path="metrics"),
            "idempotency_key": self.idempotency_key,
        }
```

Add dataclasses for `ReleaseMonitoringAlert`, `ReleaseRollbackTriggerCandidate`, `ReleaseMonitoringAcknowledgement`, and `ReleaseMonitoringAuditEvent` with the fields from the spec. Reuse the same patterns as Step 13:

```python
def make_monitoring_check_id(execution_id: str, check_type: str, idempotency_key: str) -> str:
    _require_non_empty("execution_id", execution_id)
    _validate_choice("check_type", check_type, MONITORING_CHECK_TYPES)
    _require_non_empty("idempotency_key", idempotency_key)
    payload = {"execution_id": execution_id, "check_type": check_type, "idempotency_key": idempotency_key}
    return f"release_monitor_check_{_slug(execution_id)}_{_slug(check_type)}_{_stable_hash(payload)}"


def make_monitoring_alert_id(execution_id: str, category: str, discriminator: str) -> str:
    _require_non_empty("execution_id", execution_id)
    _validate_choice("category", category, MONITORING_ALERT_CATEGORIES)
    _require_non_empty("discriminator", discriminator)
    payload = {"execution_id": execution_id, "category": category, "discriminator": discriminator}
    return f"release_monitor_alert_{_slug(execution_id)}_{_slug(category)}_{_stable_hash(payload)}"


def canonical_monitoring_payload_hash(payload: JsonValue) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    stable_json = json.dumps(payload_copy, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()}"
```

Also add:

- `make_rollback_trigger_candidate_id(execution_id, alert_ids)`
- `make_monitoring_acknowledgement_id(alert_id, acknowledgement_key)`
- `make_monitoring_event_id(execution_id, event_type, timestamp)`
- `build_monitoring_audit_event` with keyword-only event, actor, timestamp, payload, and previous-hash parameters
- `validate_monitoring_audit_event_hash(event)`
- JSON-safe helpers copied from Step 13 with monitoring-specific forbidden key rejection.

- [ ] **Step 5: Run contract tests GREEN**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add .gitignore src/contracts/release_monitoring.py tests/backend/test_release_monitoring_contract.py
git commit -m "feat: add release monitoring contracts"
```

### Task 2: Monitoring Store

**Files:**
- Create: `backend/api/services/release_monitoring_store.py`
- Create: `reports/release_monitoring/README.md`
- Create: `tests/backend/test_release_monitoring_store.py`

- [ ] **Step 1: Write failing store tests**

Create `tests/backend/test_release_monitoring_store.py`:

```python
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
    root.symlink_to(target, target_is_directory=True)

    state = ReleaseMonitoringStore(root).read_state()

    assert state.integrity["status"] == "failed"
    assert "symlink" in state.integrity["warnings"][0]
```

- [ ] **Step 2: Run RED store tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_store.py -q
```

Expected: FAIL with missing `backend.api.services.release_monitoring_store`.

- [ ] **Step 3: Implement store**

Create `backend/api/services/release_monitoring_store.py` using the Step 13 store pattern. Required public API:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any

from src.contracts.release_monitoring import (
    GENESIS_MONITORING_EVENT_HASH,
    ReleaseMonitoringAcknowledgement,
    ReleaseMonitoringAuditEvent,
    ReleaseMonitoringCheck,
    build_monitoring_audit_event,
    canonical_monitoring_payload_hash,
    make_monitoring_event_id,
)


class ReleaseMonitoringIntegrityError(RuntimeError):
    """Raised when the release monitoring store is unsafe to append to."""


@dataclass(frozen=True)
class ReleaseMonitoringState:
    checks: list[ReleaseMonitoringCheck]
    acknowledgements: list[ReleaseMonitoringAcknowledgement]
    audit_events: list[ReleaseMonitoringAuditEvent]
    integrity: dict[str, Any]


@dataclass(frozen=True)
class ReleaseMonitoringCheckIdempotencyMatch:
    check: ReleaseMonitoringCheck


class ReleaseMonitoringStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.checks_dir = self.root / "checks"
        self.acknowledgements_dir = self.root / "acknowledgements"
        self.audit_dir = self.root / "audit"

    def read_state(self) -> ReleaseMonitoringState:
        return self._read_state_with_integrity()

    def find_check_by_idempotency_key(
        self,
        check_type: str,
        key: str,
    ) -> ReleaseMonitoringCheckIdempotencyMatch | None:
        state = self.read_state()
        if state.integrity["status"] == "failed":
            raise ReleaseMonitoringIntegrityError("release monitoring integrity failed; refusing idempotency lookup")
        for check in state.checks:
            if check.check_type == check_type and check.idempotency_key == key:
                return ReleaseMonitoringCheckIdempotencyMatch(check=check)
        return None

    def assert_idempotent_check_matches(self, check: ReleaseMonitoringCheck) -> None:
        match = self.find_check_by_idempotency_key(check.check_type, check.idempotency_key)
        if match is None:
            return
        existing_hash = canonical_monitoring_payload_hash(match.check.to_dict())
        incoming_hash = canonical_monitoring_payload_hash(check.to_dict())
        if existing_hash != incoming_hash:
            raise ReleaseMonitoringIntegrityError("idempotency key payload mismatch")

    def write_check(self, check: ReleaseMonitoringCheck, *, timestamp: str) -> None:
        self._raise_if_integrity_failed()
        self.assert_idempotent_check_matches(check)
        event = build_monitoring_audit_event(
            event_id=make_monitoring_event_id(check.execution_id, "check_recorded", timestamp),
            intent_id=check.intent_id,
            execution_id=check.execution_id,
            event_type="check_recorded",
            actor=check.observed_by,
            timestamp=timestamp,
            payload=check.to_dict(),
            previous_event_hash=self._last_event_hash(check.execution_id),
        )
        self._write_json_once(self._artifact_path(self.checks_dir, check.check_id), check.to_dict())
        self._append_event(event, timestamp=timestamp)

    def write_acknowledgement(
        self,
        acknowledgement: ReleaseMonitoringAcknowledgement,
        *,
        timestamp: str,
    ) -> None:
        self._raise_if_integrity_failed()
        event = build_monitoring_audit_event(
            event_id=make_monitoring_event_id(acknowledgement.execution_id, "alert_acknowledged", timestamp),
            intent_id=acknowledgement.intent_id,
            execution_id=acknowledgement.execution_id,
            event_type="alert_acknowledged",
            actor=acknowledgement.acknowledged_by,
            timestamp=timestamp,
            payload=acknowledgement.to_dict(),
            previous_event_hash=self._last_event_hash(acknowledgement.execution_id),
        )
        self._write_json_once(
            self._artifact_path(self.acknowledgements_dir, acknowledgement.acknowledgement_id),
            acknowledgement.to_dict(),
        )
        self._append_event(event, timestamp=timestamp)
```

Implement private helpers equivalent to Step 13:

- `_read_state_with_integrity()`
- `_read_json_dir(directory, factory, artifact_name, id_field)`
- `_read_audit_events_with_integrity()`
- `_audit_chain_warnings(records)`
- `_artifact_consistency_warnings(checks, acknowledgements, audit_events)`
- `_artifact_path(directory, artifact_id)`
- `_root_layout_warning()`
- `_directory_layout_warning(path, label)`
- `_file_layout_warning(path, label)`
- `_raise_if_parent_outside_root(path)`
- `_raise_if_existing_write_target_outside_root(path)`
- `_write_json_once(path, payload)`
- `_append_event(event, timestamp)`
- `_last_event_hash(execution_id)`
- `_raise_if_integrity_failed()`
- `_audit_date(timestamp)`
- `_audit_file_date(path)`
- `_validate_artifact_id(artifact_id)`

The consistency warnings must verify:

- every check artifact has a matching `check_recorded` audit `payload_hash`;
- every acknowledgement artifact has a matching `alert_acknowledged` audit `payload_hash`;
- filenames match primary IDs.

- [ ] **Step 4: Add monitoring README**

Create `reports/release_monitoring/README.md`:

```markdown
# Release Monitoring Artifacts

This directory is reserved for Step 14 post-release monitoring.

Runtime-generated files under `checks/`, `acknowledgements/`, and `audit/` are append-only monitoring evidence. They are created by admin-only monitoring APIs and should not be edited manually.

Step 14 monitoring state is local and auditable. It does not call external alerting systems, execute rollback, store credentials, mutate release execution, mutate governance, mutate clinical safety policy, mutate prompts, mutate RAG indexes, promote literature evidence, or read patient/doctor runtime data.
```

- [ ] **Step 5: Run store tests GREEN**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_store.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add backend/api/services/release_monitoring_store.py reports/release_monitoring/README.md tests/backend/test_release_monitoring_store.py
git commit -m "feat: add release monitoring store"
```

### Task 3: Monitoring Service

**Files:**
- Create: `src/services/release_monitoring.py`
- Create: `tests/backend/test_release_monitoring_service.py`

- [ ] **Step 1: Write failing service tests**

Create `tests/backend/test_release_monitoring_service.py` with these tests:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.api.services.release_monitoring_store import ReleaseMonitoringStore
from src.services.release_monitoring import ReleaseMonitoringService


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"
ROLLBACK_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_rollback_10ca7caa"
ROLLBACK_PLAN_ID = "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"


def governance() -> dict[str, Any]:
    return {
        "active_intent": {
            "intent_id": INTENT_ID,
            "target_scope": "feature_flag_candidate",
            "derived_status": "approved",
            "rollback_target": "agent_policy_20260624_0",
            "source_release_report_id": "release_safety_20260629_001",
            "version_chain": {"agent_policy_version": "agent_policy_20260629_0"},
            "release_decision_snapshot": "feature_flag_or_pass",
        },
        "required_approvals": [{"role": "release_manager", "status": "approved"}],
        "rollback_plan": {
            "rollback_plan_id": ROLLBACK_PLAN_ID,
            "intent_id": INTENT_ID,
            "rollback_target": "agent_policy_20260624_0",
            "status": "accepted",
        },
        "integrity": {"status": "verified", "warnings": []},
    }


def dashboard() -> dict[str, Any]:
    return {
        "version_chain": {"agent_policy_version": "agent_policy_20260629_0"},
        "release_decision": "feature_flag_or_pass",
        "rollback_target": "agent_policy_20260624_0",
        "summary": {"hard_fail_count": 0},
        "runs": [{"kind": "literature_shadow_harness", "status": "shadow_only"}],
    }


def execution(released: bool = True, rolled_back: bool = False) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if released:
        results.append(
            {
                "result_id": "release_result_1",
                "execution_id": EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "release",
                "status": "succeeded",
                "started_at": "2026-07-03T09:00:00+08:00",
                "finished_at": "2026-07-03T09:00:00+08:00",
                "actor": "release_manager",
                "previous_flag_state": None,
                "new_flag_state": {
                    "flag_name": "doctor_review_cockpit_v0",
                    "enabled": True,
                    "scope": "feature_flag_candidate",
                    "source_intent_id": INTENT_ID,
                    "source_execution_id": EXECUTION_ID,
                    "rollback_target": "agent_policy_20260624_0",
                    "updated_by": "release_manager",
                    "updated_at": "2026-07-03T09:00:00+08:00",
                },
                "failure_reason": None,
            }
        )
    if rolled_back:
        results.append(
            {
                "result_id": "rollback_result_1",
                "execution_id": ROLLBACK_EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "rollback",
                "status": "succeeded",
                "started_at": "2026-07-03T12:00:00+08:00",
                "finished_at": "2026-07-03T12:00:00+08:00",
                "actor": "release_manager",
                "previous_flag_state": None,
                "new_flag_state": {
                    "flag_name": "doctor_review_cockpit_v0",
                    "enabled": False,
                    "scope": "feature_flag_candidate",
                    "source_intent_id": INTENT_ID,
                    "source_execution_id": ROLLBACK_EXECUTION_ID,
                    "rollback_target": "agent_policy_20260624_0",
                    "updated_by": "release_manager",
                    "updated_at": "2026-07-03T12:00:00+08:00",
                },
                "failure_reason": None,
            }
        )
    return {
        "feature_flag_state": results[-1]["new_flag_state"] if results else None,
        "results": results,
        "integrity": {"status": "verified", "warnings": []},
    }


def service(
    tmp_path: Path,
    exec_state: dict[str, Any] | None = None,
    gov_state: dict[str, Any] | None = None,
    dash_state: dict[str, Any] | None = None,
) -> ReleaseMonitoringService:
    return ReleaseMonitoringService(
        store=ReleaseMonitoringStore(tmp_path / "reports" / "release_monitoring"),
        execution_loader=lambda: exec_state if exec_state is not None else execution(),
        governance_loader=lambda: gov_state if gov_state is not None else governance(),
        dashboard_loader=lambda: dash_state if dash_state is not None else dashboard(),
        now=lambda: "2026-07-03T11:00:00+08:00",
    )


def test_monitoring_idle_before_successful_release(tmp_path: Path) -> None:
    model = service(tmp_path, exec_state=execution(released=False)).read_monitoring()

    assert model["status"] == "idle"
    assert model["alerts"] == []
    assert model["rollback_trigger_candidate"] is None


def test_missing_required_checks_create_warning_alerts(tmp_path: Path) -> None:
    model = service(tmp_path).read_monitoring()

    assert model["status"] == "monitoring"
    assert any(item["check_type"] == "p0_harness_replay" and item["status"] == "missing" for item in model["required_checks"])
    assert any(alert["category"] == "missing_required_check" for alert in model["alerts"])
    assert model["rollback_trigger_candidate"] is None


def test_failed_p0_check_creates_rollback_trigger_candidate(tmp_path: Path) -> None:
    monitor = service(tmp_path)
    model = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status="fail",
        observed_by="release_manager",
        summary="P0 hard fail after release.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"hard_fail_count": 1},
        idempotency_key="p0-fail-1",
    )

    assert any(alert["severity"] == "critical" for alert in model["alerts"])
    assert model["rollback_trigger_candidate"]["recommended_action"] == "execute_step13_rollback"


def test_false_positive_acknowledgement_removes_rollback_candidate(tmp_path: Path) -> None:
    monitor = service(tmp_path)
    model = monitor.record_check(
        intent_id=INTENT_ID,
        execution_id=EXECUTION_ID,
        check_type="p0_harness_replay",
        status="fail",
        observed_by="release_manager",
        summary="P0 hard fail after release.",
        evidence_refs=["reports/harness/harness_20260629_001.json"],
        metrics={"hard_fail_count": 1},
        idempotency_key="p0-fail-1",
    )
    alert_id = next(alert["alert_id"] for alert in model["alerts"] if alert["severity"] == "critical")

    model = monitor.acknowledge_alert(
        alert_id=alert_id,
        acknowledged_by="release_manager",
        disposition="false_positive",
        reason="Harness artifact was copied from a failed pre-release run.",
    )

    assert model["rollback_trigger_candidate"] is None


def test_successful_rollback_changes_monitoring_status(tmp_path: Path) -> None:
    model = service(tmp_path, exec_state=execution(rolled_back=True)).read_monitoring()

    assert model["status"] == "rolled_back"
    assert model["rollback_trigger_candidate"] is None
```

- [ ] **Step 2: Run RED service tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_service.py -q
```

Expected: FAIL with missing `src.services.release_monitoring`.

- [ ] **Step 3: Implement service**

Create `src/services/release_monitoring.py` with:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from backend.api.services.release_monitoring_store import ReleaseMonitoringStore
from src.contracts.release_monitoring import (
    ReleaseMonitoringAcknowledgement,
    ReleaseMonitoringAlert,
    ReleaseMonitoringCheck,
    ReleaseRollbackTriggerCandidate,
    make_monitoring_acknowledgement_id,
    make_monitoring_alert_id,
    make_monitoring_check_id,
    make_rollback_trigger_candidate_id,
)


class ReleaseMonitoringValidationError(ValueError):
    """Raised when monitoring input is invalid for the current release state."""


class ReleaseMonitoringConflictError(ValueError):
    """Raised when monitoring input conflicts with existing records."""


REQUIRED_CHECK_TYPES = (
    "execution_integrity",
    "governance_drift",
    "p0_harness_replay",
    "agent_admin_smoke",
    "doctor_review_smoke",
    "literature_isolation",
)


class ReleaseMonitoringService:
    def __init__(
        self,
        *,
        store: ReleaseMonitoringStore,
        execution_loader: Callable[[], dict[str, Any]],
        governance_loader: Callable[[], dict[str, Any]],
        dashboard_loader: Callable[[], dict[str, Any]],
        now: Callable[[], str],
    ) -> None:
        self._store = store
        self._execution_loader = execution_loader
        self._governance_loader = governance_loader
        self._dashboard_loader = dashboard_loader
        self._now = now

    def read_monitoring(self) -> dict[str, Any]:
        return self._build_read_model()

    def record_check(
        self,
        *,
        intent_id: str,
        execution_id: str,
        check_type: str,
        status: str,
        observed_by: str,
        summary: str,
        evidence_refs: list[str],
        metrics: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        execution = self._execution_loader()
        latest_release = self._latest_successful_release(execution)
        if latest_release is None or latest_release["execution_id"] != execution_id:
            raise ReleaseMonitoringValidationError("referenced release execution is not the latest successful release")
        if self._successful_rollback_exists(execution, intent_id) and check_type != "manual_operator_note":
            raise ReleaseMonitoringValidationError("rolled-back executions accept only manual_operator_note checks")
        timestamp = self._now()
        check = ReleaseMonitoringCheck(
            check_id=make_monitoring_check_id(execution_id, check_type, idempotency_key),
            intent_id=intent_id,
            execution_id=execution_id,
            check_type=check_type,
            status=status,
            observed_by=observed_by,
            observed_at=timestamp,
            summary=summary,
            evidence_refs=evidence_refs,
            metrics=metrics,
            idempotency_key=idempotency_key,
        )
        match = self._store.find_check_by_idempotency_key(check_type, idempotency_key)
        if match is not None:
            self._store.assert_idempotent_check_matches(check)
            return self._build_read_model()
        self._store.write_check(check, timestamp=timestamp)
        return self._build_read_model()

    def acknowledge_alert(
        self,
        *,
        alert_id: str,
        acknowledged_by: str,
        disposition: str,
        reason: str,
    ) -> dict[str, Any]:
        model = self._build_read_model()
        alert = next((item for item in model["alerts"] if item["alert_id"] == alert_id), None)
        if alert is None:
            raise ReleaseMonitoringValidationError("alert_id does not reference an active derived alert")
        timestamp = self._now()
        acknowledgement = ReleaseMonitoringAcknowledgement(
            acknowledgement_id=make_monitoring_acknowledgement_id(alert_id, f"{acknowledged_by}-{timestamp}"),
            alert_id=alert_id,
            intent_id=alert["intent_id"],
            execution_id=alert["execution_id"],
            acknowledged_by=acknowledged_by,
            acknowledged_at=timestamp,
            disposition=disposition,
            reason=reason,
        )
        self._store.write_acknowledgement(acknowledgement, timestamp=timestamp)
        return self._build_read_model()
```

Implement private methods:

- `_build_read_model()`
- `_latest_successful_release(execution)`
- `_successful_rollback_exists(execution, intent_id)`
- `_required_checks(latest_release, checks)`
- `_derive_alerts(latest_release, execution, governance, dashboard, checks, acknowledgements, monitoring_integrity)`
- `_derive_rollback_candidate(latest_release, governance, alerts)`
- `_latest_acknowledgement_by_alert(acknowledgements)`
- `_alert_status(alert_id, acknowledgements)`
- `_dashboard_drift_alerts(latest_release, governance, dashboard)`

Use deterministic alert IDs from `make_monitoring_alert_id`. A failed `p0_harness_replay` or `literature_isolation` check must produce a critical alert with `recommended_action: "execute_step13_rollback"`.

- [ ] **Step 4: Run service tests GREEN**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/services/release_monitoring.py tests/backend/test_release_monitoring_service.py
git commit -m "feat: derive release monitoring alerts"
```

### Task 4: Monitoring API And Auth

**Files:**
- Create: `backend/api/schemas/release_monitoring.py`
- Modify: `backend/api/routes/admin.py`
- Modify: `backend/app.py`
- Create: `tests/backend/test_release_monitoring_api.py`
- Modify: `tests/backend/test_auth_security.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/backend/test_release_monitoring_api.py`:

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import admin


def test_get_release_monitoring_returns_service_model(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(admin.router)
    expected = {
        "status": "idle",
        "latest_release": None,
        "required_checks": [],
        "checks": [],
        "alerts": [],
        "rollback_trigger_candidate": None,
        "acknowledgements": [],
        "integrity": {"status": "verified", "warnings": []},
        "runtime": {"auth": "admin", "source": "reports/release_monitoring", "mode": "post_release_monitoring"},
    }

    class StubService:
        def read_monitoring(self):
            return expected

    monkeypatch.setattr(admin, "_release_monitoring_service", lambda: StubService())

    response = TestClient(app).get("/api/admin/release-monitoring")

    assert response.status_code == 200
    assert response.json() == expected


def test_record_release_monitoring_check_returns_updated_model(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(admin.router)

    class StubService:
        def record_check(self, **payload):
            assert payload["check_type"] == "p0_harness_replay"
            return {"status": "monitoring", "alerts": [], "required_checks": [], "checks": [], "acknowledgements": [], "latest_release": None, "rollback_trigger_candidate": None, "integrity": {"status": "verified", "warnings": []}, "runtime": {"auth": "admin", "source": "reports/release_monitoring", "mode": "post_release_monitoring"}}

    monkeypatch.setattr(admin, "_release_monitoring_service", lambda: StubService())

    response = TestClient(app).post(
        "/api/admin/release-monitoring/checks",
        json={
            "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
            "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
            "check_type": "p0_harness_replay",
            "status": "pass",
            "observed_by": "release_manager",
            "summary": "P0 harness replay passed.",
            "evidence_refs": ["reports/harness/harness_20260629_001.json"],
            "metrics": {"hard_fail_count": 0},
            "idempotency_key": "p0-pass-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "monitoring"


def test_acknowledge_unknown_alert_maps_to_404(monkeypatch) -> None:
    from src.services.release_monitoring import ReleaseMonitoringValidationError

    app = FastAPI()
    app.include_router(admin.router)

    class StubService:
        def acknowledge_alert(self, **payload):
            raise ReleaseMonitoringValidationError("alert_id does not reference an active derived alert")

    monkeypatch.setattr(admin, "_release_monitoring_service", lambda: StubService())

    response = TestClient(app).post(
        "/api/admin/release-monitoring/alerts/release_monitor_alert_missing/acknowledge",
        json={
            "acknowledged_by": "release_manager",
            "disposition": "investigating",
            "reason": "Checking evidence.",
        },
    )

    assert response.status_code == 404
```

- [ ] **Step 2: Run RED API tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_api.py -q
```

Expected: FAIL because monitoring routes and schemas do not exist.

- [ ] **Step 3: Add schemas**

Create `backend/api/schemas/release_monitoring.py`:

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReleaseMonitoringCheckRequest(BaseModel):
    intent_id: str = Field(min_length=1)
    execution_id: str = Field(min_length=1)
    check_type: Literal[
        "execution_integrity",
        "governance_drift",
        "p0_harness_replay",
        "agent_admin_smoke",
        "doctor_review_smoke",
        "literature_isolation",
        "manual_operator_note",
    ]
    status: Literal["pass", "warning", "fail"]
    observed_by: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=1)


class ReleaseMonitoringAcknowledgeAlertRequest(BaseModel):
    acknowledged_by: str = Field(min_length=1)
    disposition: Literal["investigating", "accepted_risk", "rollback_started_elsewhere", "false_positive"]
    reason: str = Field(min_length=1)
```

- [ ] **Step 4: Add admin routes**

Modify `backend/api/routes/admin.py`:

```python
from backend.api.schemas.release_monitoring import (
    ReleaseMonitoringAcknowledgeAlertRequest,
    ReleaseMonitoringCheckRequest,
)
from backend.api.services.release_monitoring_store import (
    ReleaseMonitoringIntegrityError,
    ReleaseMonitoringStore,
)
from src.services.release_monitoring import (
    ReleaseMonitoringConflictError,
    ReleaseMonitoringService,
    ReleaseMonitoringValidationError,
)

_MONITORING_STORE_ROOT = REPO_ROOT / "reports" / "release_monitoring"


def _release_monitoring_service() -> ReleaseMonitoringService:
    return ReleaseMonitoringService(
        store=ReleaseMonitoringStore(_MONITORING_STORE_ROOT),
        execution_loader=_release_execution_service().read_execution,
        governance_loader=_release_governance_service().read_governance,
        dashboard_loader=build_release_dashboard,
        now=_governance_timestamp,
    )


def _raise_monitoring_http_error(exc: Exception) -> None:
    if isinstance(exc, ReleaseMonitoringValidationError) and "alert_id does not reference" in str(exc):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, (ReleaseMonitoringConflictError, ReleaseMonitoringIntegrityError, ReleaseMonitoringValidationError, FileExistsError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, OSError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


@router.get("/release-monitoring")
async def get_admin_release_monitoring() -> dict[str, Any]:
    return _release_monitoring_service().read_monitoring()


@router.post("/release-monitoring/checks")
async def record_admin_release_monitoring_check(
    payload: ReleaseMonitoringCheckRequest,
) -> dict[str, Any]:
    try:
        return _release_monitoring_service().record_check(**_model_dump(payload))
    except Exception as exc:
        _raise_monitoring_http_error(exc)


@router.post("/release-monitoring/alerts/{alert_id}/acknowledge")
async def acknowledge_admin_release_monitoring_alert(
    alert_id: str,
    payload: ReleaseMonitoringAcknowledgeAlertRequest,
) -> dict[str, Any]:
    try:
        return _release_monitoring_service().acknowledge_alert(
            alert_id=alert_id,
            **_model_dump(payload),
        )
    except Exception as exc:
        _raise_monitoring_http_error(exc)
```

- [ ] **Step 5: Extend auth guard**

Modify `backend/app.py::_requires_admin_token()`:

```python
if method == "GET" and path == "/api/admin/release-monitoring":
    return True
if method == "POST" and path.startswith("/api/admin/release-monitoring/"):
    return True
```

Update `tests/backend/test_auth_security.py` route matrix with:

```python
("get", "/api/admin/release-monitoring"),
("post", "/api/admin/release-monitoring/checks"),
("post", "/api/admin/release-monitoring/alerts/release_monitor_alert_1/acknowledge"),
```

- [ ] **Step 6: Run API and auth tests GREEN**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_api.py tests/backend/test_auth_security.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```powershell
git add backend/api/schemas/release_monitoring.py backend/api/routes/admin.py backend/app.py tests/backend/test_release_monitoring_api.py tests/backend/test_auth_security.py
git commit -m "feat: add release monitoring admin api"
```

### Task 5: Non-Mutation Coverage

**Files:**
- Create: `tests/backend/test_release_monitoring_non_mutation.py`

- [ ] **Step 1: Write non-mutation tests**

Create `tests/backend/test_release_monitoring_non_mutation.py`:

```python
from __future__ import annotations

from pathlib import Path

from backend.api.services.release_monitoring_store import ReleaseMonitoringStore
from src.services.release_monitoring import ReleaseMonitoringService


def snapshot_paths(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and "release_monitoring" not in path.parts
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    }


def test_monitoring_writes_only_monitoring_root(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    execution_root = reports / "release_execution"
    governance_root = reports / "release_governance"
    harness_root = reports / "harness"
    safety_root = reports / "release_safety"
    literature_root = reports / "literature"
    config_root = tmp_path / "config"
    for directory in [execution_root, governance_root, harness_root, safety_root, literature_root, config_root]:
        directory.mkdir(parents=True, exist_ok=True)
    (execution_root / "sentinel.json").write_text('{"keep": "execution"}\n', encoding="utf-8")
    (governance_root / "sentinel.json").write_text('{"keep": "governance"}\n', encoding="utf-8")
    (harness_root / "sentinel.json").write_text('{"keep": "harness"}\n', encoding="utf-8")
    (safety_root / "sentinel.json").write_text('{"keep": "safety"}\n', encoding="utf-8")
    (literature_root / "sentinel.json").write_text('{"keep": "literature"}\n', encoding="utf-8")
    (config_root / "safety_policy.yaml").write_text("policy_id: crc_safety_policy_v0\n", encoding="utf-8")
    before = snapshot_paths(tmp_path)

    service = ReleaseMonitoringService(
        store=ReleaseMonitoringStore(reports / "release_monitoring"),
        execution_loader=lambda: {
            "feature_flag_state": {
                "flag_name": "doctor_review_cockpit_v0",
                "enabled": True,
                "scope": "feature_flag_candidate",
                "source_intent_id": "release_intent_1",
                "source_execution_id": "release_exec_1",
                "rollback_target": "agent_policy_20260624_0",
                "updated_by": "release_manager",
                "updated_at": "2026-07-03T09:00:00+08:00",
            },
            "results": [
                {
                    "result_id": "release_result_1",
                    "execution_id": "release_exec_1",
                    "intent_id": "release_intent_1",
                    "action": "release",
                    "status": "succeeded",
                    "started_at": "2026-07-03T09:00:00+08:00",
                    "finished_at": "2026-07-03T09:00:00+08:00",
                    "actor": "release_manager",
                    "previous_flag_state": None,
                    "new_flag_state": None,
                    "failure_reason": None,
                }
            ],
            "integrity": {"status": "verified", "warnings": []},
        },
        governance_loader=lambda: {
            "active_intent": {"intent_id": "release_intent_1"},
            "rollback_plan": {"rollback_plan_id": "rollback_plan_1", "rollback_target": "agent_policy_20260624_0", "status": "accepted"},
            "integrity": {"status": "verified", "warnings": []},
        },
        dashboard_loader=lambda: {"summary": {"hard_fail_count": 0}, "runs": [{"kind": "literature_shadow_harness", "status": "shadow_only"}]},
        now=lambda: "2026-07-03T11:00:00+08:00",
    )

    service.record_check(
        intent_id="release_intent_1",
        execution_id="release_exec_1",
        check_type="p0_harness_replay",
        status="pass",
        observed_by="release_manager",
        summary="P0 harness replay passed.",
        evidence_refs=["reports/harness/sentinel.json"],
        metrics={"hard_fail_count": 0},
        idempotency_key="p0-pass-1",
    )

    after = snapshot_paths(tmp_path)
    assert after == before
    assert (reports / "release_monitoring" / "checks").exists()
```

- [ ] **Step 2: Run non-mutation test**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_non_mutation.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit Task 5**

```powershell
git add tests/backend/test_release_monitoring_non_mutation.py
git commit -m "test: prove release monitoring non mutation"
```

### Task 6: Frontend API Types And Client

**Files:**
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/test/test-utils.tsx`

- [ ] **Step 1: Write failing frontend API tests**

Add to `frontend/src/app/api/client.test.ts`:

```ts
it("gets admin release monitoring", async () => {
  const payload = releaseMonitoringResponse();
  fetchImpl.mockResolvedValue(jsonResponse(payload));
  const client = createApiClient({ baseUrl: "http://127.0.0.1:8000", fetchImpl });

  await expect(client.getAdminReleaseMonitoring()).resolves.toEqual(payload);

  expect(fetchImpl).toHaveBeenCalledWith("http://127.0.0.1:8000/api/admin/release-monitoring", {
    headers: expect.any(Headers),
  });
});

it("records admin release monitoring check and acknowledges alert", async () => {
  const payload = releaseMonitoringResponse();
  fetchImpl.mockResolvedValue(jsonResponse(payload));
  const client = createApiClient({ baseUrl: "http://127.0.0.1:8000", fetchImpl });
  const checkRequest = {
    intent_id: "release_intent_1",
    execution_id: "release_exec_1",
    check_type: "p0_harness_replay" as const,
    status: "pass" as const,
    observed_by: "release_manager",
    summary: "P0 harness replay passed.",
    evidence_refs: ["reports/harness/harness_20260629_001.json"],
    metrics: { hard_fail_count: 0 },
    idempotency_key: "p0-pass-1",
  };
  const acknowledgementRequest = {
    acknowledged_by: "release_manager",
    disposition: "investigating" as const,
    reason: "Checking evidence.",
  };

  await client.recordAdminReleaseMonitoringCheck(checkRequest);
  await client.acknowledgeAdminReleaseMonitoringAlert("release_monitor_alert_1", acknowledgementRequest);

  expect(fetchImpl).toHaveBeenNthCalledWith(1, "http://127.0.0.1:8000/api/admin/release-monitoring/checks", {
    method: "POST",
    headers: expect.any(Headers),
    body: JSON.stringify(checkRequest),
  });
  expect(fetchImpl).toHaveBeenNthCalledWith(2, "http://127.0.0.1:8000/api/admin/release-monitoring/alerts/release_monitor_alert_1/acknowledge", {
    method: "POST",
    headers: expect.any(Headers),
    body: JSON.stringify(acknowledgementRequest),
  });
});
```

Add helper:

```ts
function releaseMonitoringResponse(): AdminReleaseMonitoringResponse {
  return {
    status: "monitoring",
    latest_release: {
      intent_id: "release_intent_1",
      execution_id: "release_exec_1",
      released_at: "2026-07-03T09:00:00+08:00",
      flag_enabled: true,
      rollback_plan_id: "rollback_plan_1",
    },
    required_checks: [
      {
        check_type: "p0_harness_replay",
        status: "missing",
        latest_check_id: null,
        reason: "Record a post-release P0 harness replay check.",
      },
    ],
    checks: [],
    alerts: [],
    rollback_trigger_candidate: null,
    acknowledgements: [],
    integrity: { status: "verified", warnings: [] },
    runtime: { auth: "admin", source: "reports/release_monitoring", mode: "post_release_monitoring" },
  };
}
```

- [ ] **Step 2: Run RED frontend API tests**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts
```

Expected: FAIL because monitoring types and client methods do not exist.

- [ ] **Step 3: Add TypeScript types**

Add to `frontend/src/app/api/types.ts`:

```ts
export type AdminReleaseMonitoringStatus = "idle" | "monitoring" | "rolled_back";
export type AdminReleaseMonitoringCheckType =
  | "execution_integrity"
  | "governance_drift"
  | "p0_harness_replay"
  | "agent_admin_smoke"
  | "doctor_review_smoke"
  | "literature_isolation"
  | "manual_operator_note";
export type AdminReleaseMonitoringCheckStatus = "pass" | "warning" | "fail";
export type AdminReleaseMonitoringRequiredCheckStatus = AdminReleaseMonitoringCheckStatus | "missing";
export type AdminReleaseMonitoringAlertSeverity = "info" | "warning" | "critical";
export type AdminReleaseMonitoringRecommendedAction = "observe" | "investigate" | "prepare_rollback" | "execute_step13_rollback";
export type AdminReleaseMonitoringAcknowledgementDisposition =
  | "investigating"
  | "accepted_risk"
  | "rollback_started_elsewhere"
  | "false_positive";

export interface AdminReleaseMonitoringLatestRelease {
  intent_id: string;
  execution_id: string;
  released_at: string;
  flag_enabled: boolean;
  rollback_plan_id: string | null;
}

export interface AdminReleaseMonitoringRequiredCheck {
  check_type: AdminReleaseMonitoringCheckType;
  status: AdminReleaseMonitoringRequiredCheckStatus;
  latest_check_id: string | null;
  reason: string;
}

export interface AdminReleaseMonitoringCheckRecord {
  check_id: string;
  intent_id: string;
  execution_id: string;
  check_type: AdminReleaseMonitoringCheckType;
  status: AdminReleaseMonitoringCheckStatus;
  observed_by: string;
  observed_at: string;
  summary: string;
  evidence_refs: string[];
  metrics: Record<string, JsonValue | unknown>;
  idempotency_key: string;
}

export interface AdminReleaseMonitoringAlert {
  alert_id: string;
  intent_id: string;
  execution_id: string;
  severity: AdminReleaseMonitoringAlertSeverity;
  category: string;
  status: "active" | "acknowledged";
  message: string;
  source_check_ids: string[];
  recommended_action: AdminReleaseMonitoringRecommendedAction;
  created_at: string;
}

export interface AdminReleaseRollbackTriggerCandidate {
  candidate_id: string;
  intent_id: string;
  execution_id: string;
  source_alert_ids: string[];
  recommended_action: "execute_step13_rollback";
  rollback_plan_id: string;
  rollback_target: string;
  reason: string;
  created_at: string;
}

export interface AdminReleaseMonitoringAcknowledgement {
  acknowledgement_id: string;
  alert_id: string;
  intent_id: string;
  execution_id: string;
  acknowledged_by: string;
  acknowledged_at: string;
  disposition: AdminReleaseMonitoringAcknowledgementDisposition;
  reason: string;
}

export interface AdminReleaseMonitoringResponse {
  status: AdminReleaseMonitoringStatus;
  latest_release: AdminReleaseMonitoringLatestRelease | null;
  required_checks: AdminReleaseMonitoringRequiredCheck[];
  checks: AdminReleaseMonitoringCheckRecord[];
  alerts: AdminReleaseMonitoringAlert[];
  rollback_trigger_candidate: AdminReleaseRollbackTriggerCandidate | null;
  acknowledgements: AdminReleaseMonitoringAcknowledgement[];
  integrity: AdminReleaseGovernanceIntegrity;
  runtime: {
    auth: "admin";
    source: "reports/release_monitoring";
    mode: "post_release_monitoring";
  };
}

export interface AdminRecordReleaseMonitoringCheckRequest {
  intent_id: string;
  execution_id: string;
  check_type: AdminReleaseMonitoringCheckType;
  status: AdminReleaseMonitoringCheckStatus;
  observed_by: string;
  summary: string;
  evidence_refs: string[];
  metrics: Record<string, JsonValue | unknown>;
  idempotency_key: string;
}

export interface AdminAcknowledgeReleaseMonitoringAlertRequest {
  acknowledged_by: string;
  disposition: AdminReleaseMonitoringAcknowledgementDisposition;
  reason: string;
}
```

- [ ] **Step 4: Add client methods**

Add to `frontend/src/app/api/client.ts`:

```ts
async getAdminReleaseMonitoring() {
  const response = await fetchImpl(buildUrl("/api/admin/release-monitoring", baseUrl), {
    headers: buildHeaders(),
  });
  return parseJsonResponse<AdminReleaseMonitoringResponse>(response);
}

async recordAdminReleaseMonitoringCheck(request: AdminRecordReleaseMonitoringCheckRequest) {
  const response = await fetchImpl(buildUrl("/api/admin/release-monitoring/checks", baseUrl), {
    method: "POST",
    headers: buildHeaders({ json: true }),
    body: JSON.stringify(request),
  });
  return parseJsonResponse<AdminReleaseMonitoringResponse>(response);
}

async acknowledgeAdminReleaseMonitoringAlert(
  alertId: string,
  request: AdminAcknowledgeReleaseMonitoringAlertRequest,
) {
  const response = await fetchImpl(
    buildUrl(`/api/admin/release-monitoring/alerts/${encodeURIComponent(alertId)}/acknowledge`, baseUrl),
    {
      method: "POST",
      headers: buildHeaders({ json: true }),
      body: JSON.stringify(request),
    },
  );
  return parseJsonResponse<AdminReleaseMonitoringResponse>(response);
}
```

Update API interface exports and test utils stubs.

- [ ] **Step 5: Run frontend API tests GREEN**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```powershell
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/test/test-utils.tsx
git commit -m "feat: add release monitoring frontend api"
```

### Task 7: Agent Admin Monitoring UI

**Files:**
- Modify: `frontend/src/features/agent-admin/agent-admin-view.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-pages.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
- Modify: `frontend/src/styles/globals.css`

- [ ] **Step 1: Write failing Agent Admin tests**

Add tests to `frontend/src/features/agent-admin/agent-admin-view.test.tsx` that:

- provide `getAdminReleaseMonitoring` with idle response and assert "post-release monitoring" appears;
- provide active monitoring with missing checks and assert `p0_harness_replay` appears;
- provide a critical alert and rollback trigger and assert `execute_step13_rollback` appears;
- submit a check form and assert `recordAdminReleaseMonitoringCheck` is called;
- submit an acknowledgement form and assert `acknowledgeAdminReleaseMonitoringAlert` is called;
- reject monitoring API and assert dashboard/governance/execution panels remain rendered.

Use this response helper:

```ts
function monitoringResponse(overrides: Partial<AdminReleaseMonitoringResponse> = {}): AdminReleaseMonitoringResponse {
  const base: AdminReleaseMonitoringResponse = {
    status: "monitoring",
    latest_release: {
      intent_id: "release_intent_1",
      execution_id: "release_exec_1",
      released_at: "2026-07-03T09:00:00+08:00",
      flag_enabled: true,
      rollback_plan_id: "rollback_plan_1",
    },
    required_checks: [
      {
        check_type: "p0_harness_replay",
        status: "missing",
        latest_check_id: null,
        reason: "Record a post-release P0 harness replay check.",
      },
    ],
    checks: [],
    alerts: [],
    rollback_trigger_candidate: null,
    acknowledgements: [],
    integrity: { status: "verified", warnings: [] },
    runtime: { auth: "admin", source: "reports/release_monitoring", mode: "post_release_monitoring" },
  };
  return Object.assign(base, overrides);
}
```

- [ ] **Step 2: Run RED Agent Admin tests**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: FAIL because monitoring resource and UI do not exist.

- [ ] **Step 3: Add monitoring resource and actions**

Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`:

- include `getAdminReleaseMonitoring`, `recordAdminReleaseMonitoringCheck`, and `acknowledgeAdminReleaseMonitoringAlert` in the API client pick;
- add `AgentAdminReleaseMonitoringResource`;
- add `AgentAdminReleaseMonitoringActionState`;
- add `AgentAdminReleaseMonitoringActions`;
- load monitoring state when active task is `release`;
- pass monitoring resource and actions into `AgentAdminTaskPages`.

Use the same patterns as release governance and release execution:

```ts
export type AgentAdminReleaseMonitoringResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminReleaseMonitoringResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminReleaseMonitoringActions = {
  recordCheck: (request: AdminRecordReleaseMonitoringCheckRequest) => Promise<void>;
  acknowledgeAlert: (alertId: string, request: AdminAcknowledgeReleaseMonitoringAlertRequest) => Promise<void>;
};
```

- [ ] **Step 4: Add monitoring panel**

Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`:

- add monitoring props to `AgentAdminPagesProps`;
- render `ReleaseMonitoringPanel` below `ReleaseExecutionPanel`;
- implement `ReleaseMonitoringPanel` using existing `AgentAdminPanel`, `AgentAdminSplitWorkbench`, `AgentAdminStateIcon`, and form classes.

The panel must render:

- status and latest release;
- required checks;
- active alerts;
- rollback trigger candidate;
- check form;
- acknowledgement form;
- integrity warnings.

Button rules:

```ts
const canRecordCheck =
  monitoring.status === "monitoring" &&
  monitoring.latest_release !== null &&
  monitoring.integrity.status === "verified" &&
  !actionRunning;

const canAcknowledge =
  monitoring.alerts.length > 0 &&
  monitoring.integrity.status === "verified" &&
  !actionRunning;
```

- [ ] **Step 5: Add CSS only if needed**

If existing `.agent-admin-governance-form`, `.agent-admin-detail-list`, and `.agent-admin-timeline` styles cover the UI, do not add CSS. If a small addition is needed, add:

```css
.agent-admin-monitoring-alert-critical {
  border-color: rgba(220, 38, 38, 0.45);
}

.agent-admin-monitoring-trigger {
  border: 1px solid rgba(220, 38, 38, 0.35);
  background: rgba(254, 242, 242, 0.72);
}
```

- [ ] **Step 6: Run Agent Admin tests GREEN**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit Task 7**

```powershell
git add frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx frontend/src/styles/globals.css
git commit -m "feat: show release monitoring in agent admin"
```

### Task 8: Final Verification

**Files:**
- No new files unless verification exposes a defect.

- [ ] **Step 1: Backend focused verification**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_monitoring_contract.py tests/backend/test_release_monitoring_store.py tests/backend/test_release_monitoring_service.py tests/backend/test_release_monitoring_api.py tests/backend/test_release_monitoring_non_mutation.py -q
```

Expected: PASS.

- [ ] **Step 2: Release stack backend regression**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_contract.py tests/backend/test_release_execution_store.py tests/backend/test_release_execution_service.py tests/backend/test_release_execution_api.py tests/backend/test_release_execution_non_mutation.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_governance_api.py tests/backend/test_release_governance_non_mutation.py tests/backend/test_auth_security.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py -q
```

Expected: PASS.

- [ ] **Step 3: Clinical evidence regression**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py -q
```

Expected: PASS.

- [ ] **Step 4: Frontend focused verification**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/agent-admin/agent-admin-view.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Frontend build**

Run:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: PASS. Existing Vite chunk-size warnings are acceptable if unchanged.

- [ ] **Step 6: Diff and status check**

Run:

```powershell
git diff --check
git status --short
```

Expected: `git diff --check` has no output. `git status --short` lists only Step 14 files before final commit.

- [ ] **Step 7: Commit final fixes if needed**

If verification required edits, commit them:

```powershell
git add <changed-files>
git commit -m "test: stabilize release monitoring verification"
```

## Implementation Boundaries

- Do not edit `CRC-client/`.
- Do not execute release or rollback from monitoring code.
- Do not add deployment hooks.
- Do not run harness scripts from admin routes.
- Do not write under `reports/release_execution/` or `reports/release_governance/`.
- Do not mutate safety policy, prompt, rubric, route, template, RAG, model, tool, or report artifacts outside `reports/release_monitoring/`.
- Do not promote literature evidence.
- Do not add patient-level telemetry or research exports.
- Do not add live network or model calls.
- Do not introduce broad auth infrastructure in this slice.

## Plan Self-Review

Spec coverage: Tasks cover contracts, store, service, API/auth, non-mutation, frontend API, Agent Admin UI, docs, and verification. The rollback trigger is advisory only and never executes rollback.

Marker scan: no unresolved work markers remain.

Type consistency: backend response keys match frontend interfaces: `status`, `latest_release`, `required_checks`, `checks`, `alerts`, `rollback_trigger_candidate`, `acknowledgements`, `integrity`, and `runtime`. Client method names are consistently `getAdminReleaseMonitoring()`, `recordAdminReleaseMonitoringCheck()`, and `acknowledgeAdminReleaseMonitoringAlert()`.
