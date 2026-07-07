# Post-Release Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P2 Step 15 post-release closure so Agent Admin can close a controlled release or rollback observation period and generate a local JSON evidence package after Step 14 monitoring.

**Architecture:** Add a Step 15 closure subsystem beside, not inside, Step 14 monitoring. Backend adds closure contracts, a file-backed closure store, a service that derives closure readiness from dashboard, governance, execution, and monitoring state, and admin-only routes. Frontend extends the existing Agent Admin Release page with backend-derived closure gate state, closure recording, and evidence package summary.

**Tech Stack:** Python 3.10, dataclasses, FastAPI, Pydantic v2, pytest, TypeScript, React, Vitest, Testing Library, existing Agent Admin UI components.

## Global Constraints

- Step 15 may write only under `reports/release_closure/`.
- Step 15 must not mutate `reports/release_monitoring/`, `reports/release_execution/`, `reports/release_governance/`, release reports, harness reports, literature reports, safety policy, prompts, rubrics, routes, templates, RAG indexes, model files, tool manifests, patient/doctor state, deployment systems, or `CRC-client/`.
- Closure payloads must not contain API keys, bearer tokens, deployment credentials, model prompts, hidden reasoning, raw patient identifiers, patient records, doctor note text, or session transcripts.
- Step 15 does not execute release, execute rollback, acknowledge monitoring alerts, suppress monitoring alerts, generate rendered documents, send notifications, upload archives, or add role-based auth.
- Existing admin bearer auth behavior is reused.

---

## Source Spec

Read before implementation:

- `docs/superpowers/specs/2026-07-07-post-release-closure-design.md`
- `docs/superpowers/specs/2026-07-03-post-release-monitoring-design.md`
- `docs/superpowers/specs/2026-07-03-controlled-release-execution-design.md`
- `docs/superpowers/specs/2026-07-02-controlled-release-governance-design.md`
- `docs/superpowers/specs/2026-06-30-agent-admin-release-dashboard-design.md`
- `docs/superpowers/specs/2026-06-30-evidenceclaim-literature-harness-design.md`
- `docs/superpowers/specs/2026-06-29-p1-clinical-review-loop-design.md`
- `docs/superpowers/specs/2026-06-29-p0-crc-safety-loop-design.md`

## File Structure

Backend contracts:

- Create `src/contracts/release_closure.py`
  - Dataclass contracts for `ReleaseClosureRecord`, `ReleaseEvidencePackage`, `ReleaseClosureGate`, `ReleaseClosureGateCheck`, and `ReleaseClosureAuditEvent`.
  - Stable ID helpers, canonical hash helpers, JSON safety validation, and forbidden payload key rejection.
- Create `tests/backend/test_release_closure_contract.py`
  - Contract validation, audit hash-chain, closure gate, evidence package, and forbidden payload tests.

Backend persistence and closure:

- Create `backend/api/services/release_closure_store.py`
  - File-backed store rooted at `reports/release_closure/`.
  - Write-once closures and evidence packages, append-only audit JSONL, integrity verification, idempotency lookup, and atomic closure/package writes.
- Create `src/services/release_closure.py`
  - Read Step 11 dashboard, Step 12 governance, Step 13 execution, and Step 14 monitoring state.
  - Derive closure status, latest release summary, closure gate, and latest closure/package.
  - Record closure when the gate allows the requested closure status.
- Create `tests/backend/test_release_closure_store.py`
  - Store integrity, idempotency, audit-chain, atomicity, and read-only behavior tests.
- Create `tests/backend/test_release_closure_service.py`
  - Closure gate, blocked states, accepted closure, accepted-with-observations closure, rolled-back closure, and duplicate behavior tests.

Backend API:

- Create `backend/api/schemas/release_closure.py`
  - Pydantic request schema for closure recording.
- Modify `backend/api/routes/admin.py`
  - Add closure service factory and routes.
- Modify `backend/app.py`
  - Add closure routes to admin-token guard.
- Create `tests/backend/test_release_closure_api.py`
  - API route behavior and error mapping tests.
- Modify `tests/backend/test_auth_security.py`
  - Add release closure routes to the admin auth matrix.
- Create `tests/backend/test_release_closure_non_mutation.py`
  - Prove Step 15 writes only under `reports/release_closure/`.
- Modify `.gitignore`
  - Add precise whitelist entries for new Step 15 backend tests and the plan file.

Frontend API:

- Modify `frontend/src/app/api/types.ts`
  - Add closure response, gate, closure record, evidence package, audit, and request types.
- Modify `frontend/src/app/api/client.ts`
  - Add closure client methods.
- Modify `frontend/src/app/api/client.test.ts`
  - Add endpoint/header/body tests.
- Modify `frontend/src/test/test-utils.tsx`
  - Add default closure client stubs.

Frontend Agent Admin:

- Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`
  - Load closure state with the release task.
  - Add closure recording handler.
- Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`
  - Add closure gate, closure form, latest closure, and evidence package panel below monitoring.
- Modify `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
  - Add closure loading, idle, blocked, ready, closed, submit, and error tests.
- Modify `frontend/src/styles/globals.css`
  - Add compact closure gate/form styles only when existing release styles are insufficient.

Docs:

- Create `reports/release_closure/README.md`
  - Documents local closure artifacts.
- Do not commit runtime-generated closure, package, or audit files outside tests.

---

### Task 1: Closure Contracts

**Files:**
- Create: `src/contracts/release_closure.py`
- Create: `tests/backend/test_release_closure_contract.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `ReleaseClosureRecord`, `ReleaseEvidencePackage`, `ReleaseClosureGate`, `ReleaseClosureGateCheck`, `ReleaseClosureAuditEvent`.
- Produces: `canonical_closure_payload_hash(payload) -> str`.
- Produces: `make_release_closure_id(release_execution_id, idempotency_key) -> str`.
- Produces: `make_release_evidence_package_id(closure_id) -> str`.
- Produces: `build_release_closure_audit_event` keyword-only helper returning `ReleaseClosureAuditEvent`.

- [ ] **Step 1: Add test whitelist entries**

Modify `.gitignore` near the backend test whitelist:

```gitignore
!tests/backend/test_release_closure_contract.py
!tests/backend/test_release_closure_store.py
!tests/backend/test_release_closure_service.py
!tests/backend/test_release_closure_api.py
!tests/backend/test_release_closure_non_mutation.py
```

- [ ] **Step 2: Write failing contract tests**

Create `tests/backend/test_release_closure_contract.py`:

```python
from __future__ import annotations

import pytest

from src.contracts.release_closure import (
    GENESIS_CLOSURE_EVENT_HASH,
    ReleaseClosureAuditEvent,
    ReleaseClosureGate,
    ReleaseClosureGateCheck,
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    build_release_closure_audit_event,
    canonical_closure_payload_hash,
    make_release_closure_event_id,
    make_release_closure_id,
    make_release_evidence_package_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
RELEASE_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"


def make_closure() -> ReleaseClosureRecord:
    closure_id = make_release_closure_id(RELEASE_EXECUTION_ID, "close-1")
    return ReleaseClosureRecord(
        closure_id=closure_id,
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        rollback_execution_id=None,
        closure_status="accepted",
        closed_by="release_manager",
        closed_at="2026-07-07T10:00:00+08:00",
        rationale="Required checks passed and no active critical alerts remain.",
        monitoring_snapshot_hash="sha256:" + "a" * 64,
        dashboard_snapshot_hash="sha256:" + "b" * 64,
        governance_snapshot_hash="sha256:" + "c" * 64,
        execution_snapshot_hash="sha256:" + "d" * 64,
        required_check_ids=["release_monitor_check_1"],
        acknowledged_alert_ids=[],
        unresolved_alert_ids=[],
        rollback_trigger_candidate_id=None,
        evidence_package_id=make_release_evidence_package_id(closure_id),
        idempotency_key="close-1",
    )


def test_closure_contracts_round_trip_to_dict() -> None:
    closure = make_closure()
    package = ReleaseEvidencePackage(
        package_id=closure.evidence_package_id,
        closure_id=closure.closure_id,
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        rollback_execution_id=None,
        generated_by="release_manager",
        generated_at="2026-07-07T10:00:00+08:00",
        closure_status="accepted",
        summary="Release observation period closed after checks passed.",
        source_refs=[
            "GET /api/admin/release-dashboard",
            "GET /api/admin/release-governance",
            "GET /api/admin/release-execution",
            "GET /api/admin/release-monitoring",
        ],
        artifact_refs=[
            f"reports/release_closure/closures/{closure.closure_id}.json",
        ],
        snapshot_hashes={
            "dashboard": closure.dashboard_snapshot_hash,
            "governance": closure.governance_snapshot_hash,
            "execution": closure.execution_snapshot_hash,
            "monitoring": closure.monitoring_snapshot_hash,
        },
    )
    gate = ReleaseClosureGate(
        allowed=True,
        status="ready_to_close",
        reasons=[],
        checks=[
            ReleaseClosureGateCheck(
                name="required_monitoring_checks_complete",
                status="pass",
                reason="All Step 14 required checks are present.",
            )
        ],
    )
    event = build_release_closure_audit_event(
        event_id=make_release_closure_event_id(
            RELEASE_EXECUTION_ID,
            "closure_recorded",
            "2026-07-07T10:00:00+08:00",
        ),
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        event_type="closure_recorded",
        actor="release_manager",
        timestamp="2026-07-07T10:00:00+08:00",
        payload=closure.to_dict(),
        previous_event_hash=GENESIS_CLOSURE_EVENT_HASH,
    )

    assert closure.to_dict()["closure_status"] == "accepted"
    assert package.to_dict()["closure_id"] == closure.closure_id
    assert gate.to_dict()["status"] == "ready_to_close"
    assert event.to_dict()["event_hash"].startswith("sha256:")


@pytest.mark.parametrize("closure_status", ["pending", "failed", ""])
def test_closure_rejects_unknown_status(closure_status: str) -> None:
    payload = make_closure().to_dict()
    payload["closure_status"] = closure_status

    with pytest.raises(ValueError, match="closure_status must be one of"):
        ReleaseClosureRecord(**payload)


def test_accepted_closure_rejects_unresolved_alerts() -> None:
    payload = make_closure().to_dict()
    payload["unresolved_alert_ids"] = ["release_monitor_alert_1"]

    with pytest.raises(ValueError, match="accepted closure cannot contain unresolved alerts"):
        ReleaseClosureRecord(**payload)


def test_rolled_back_closure_requires_rollback_execution_id() -> None:
    payload = make_closure().to_dict()
    payload["closure_status"] = "rolled_back"
    payload["rollback_execution_id"] = None

    with pytest.raises(ValueError, match="rollback_execution_id is required"):
        ReleaseClosureRecord(**payload)


def test_hash_rejects_forbidden_payload_keys() -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        canonical_closure_payload_hash({"patient_id": "p-1"})
```

- [ ] **Step 3: Run RED contract tests**

Run:

```bash
pytest tests/backend/test_release_closure_contract.py -q
```

Expected: FAIL because `src.contracts.release_closure` does not exist.

- [ ] **Step 4: Implement closure contracts**

Create `src/contracts/release_closure.py` with:

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

GENESIS_CLOSURE_EVENT_HASH = "sha256:GENESIS"

ClosureStatus = Literal["accepted", "accepted_with_observations", "rolled_back"]
ClosureGateStatus = Literal["idle", "ready_to_close", "blocked", "closed", "rolled_back_closed"]
ClosureGateCheckStatus = Literal["pass", "warning", "fail"]
ClosureAuditEventType = Literal["closure_recorded", "evidence_package_generated", "closure_read"]

CLOSURE_STATUSES: Sequence[ClosureStatus] = ("accepted", "accepted_with_observations", "rolled_back")
CLOSURE_GATE_STATUSES: Sequence[ClosureGateStatus] = ("idle", "ready_to_close", "blocked", "closed", "rolled_back_closed")
CLOSURE_GATE_CHECK_STATUSES: Sequence[ClosureGateCheckStatus] = ("pass", "warning", "fail")
CLOSURE_EVENT_TYPES: Sequence[ClosureAuditEventType] = ("closure_recorded", "evidence_package_generated", "closure_read")

FORBIDDEN_CLOSURE_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "bearer_token",
        "chain_of_thought",
        "credential",
        "deployment_credentials",
        "doctor_note_text",
        "hidden_reasoning",
        "medical_record_number",
        "mrn",
        "password",
        "patient_id",
        "patient_identifier",
        "patient_name",
        "patient_record",
        "prompt",
        "raw_patient_identifier",
        "secret",
        "session_token",
        "token",
        "transcript",
    }
)
```

Add dataclasses matching the spec fields. Use the same validation style as `src/contracts/release_monitoring.py`:

- require non-empty IDs and actor labels;
- validate enum choices;
- validate sha256 hashes, with `sha256:GENESIS` allowed only for `previous_event_hash`;
- freeze JSON-safe dict/list values;
- reject forbidden payload keys recursively;
- reject absolute paths and URLs in `artifact_refs`;
- require `rolled_back` closures to include `rollback_execution_id`;
- reject `accepted` closures with unresolved alerts or rollback trigger candidate.

Add helpers:

```python
def make_release_closure_id(release_execution_id: str, idempotency_key: str) -> str:
    payload = {"release_execution_id": release_execution_id, "idempotency_key": idempotency_key}
    return f"release_closure_{_slug(release_execution_id)}_{_stable_hash(payload)}"


def make_release_evidence_package_id(closure_id: str) -> str:
    payload = {"closure_id": closure_id}
    return f"release_evidence_package_{_slug(closure_id)}_{_stable_hash(payload)}"


def make_release_closure_event_id(release_execution_id: str, event_type: str, timestamp: str) -> str:
    _validate_choice("event_type", event_type, CLOSURE_EVENT_TYPES)
    payload = {"release_execution_id": release_execution_id, "event_type": event_type, "timestamp": timestamp}
    return f"release_closure_audit_{_slug(event_type)}_{_stable_hash(payload)}"
```

Use `canonical_closure_payload_hash()` and `build_release_closure_audit_event()` equivalent to Step 14 monitoring helpers, but closure-specific.

- [ ] **Step 5: Run GREEN contract tests**

Run:

```bash
pytest tests/backend/test_release_closure_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add .gitignore src/contracts/release_closure.py tests/backend/test_release_closure_contract.py
git commit -m "feat: add release closure contracts"
```

### Task 2: Closure Store

**Files:**
- Create: `backend/api/services/release_closure_store.py`
- Create: `reports/release_closure/README.md`
- Create: `tests/backend/test_release_closure_store.py`

**Interfaces:**
- Consumes: Task 1 closure contracts and helpers.
- Produces: `ReleaseClosureStore(root: Path)`.
- Produces: `ReleaseClosureState(closures, evidence_packages, audit_events, integrity)`.
- Produces: `write_closure_with_package(closure, package, timestamp)`.
- Produces: `find_closure_by_idempotency_key(idempotency_key)`.
- Produces: `assert_idempotent_closure_matches(closure, package)`.

- [ ] **Step 1: Write failing store tests**

Create `tests/backend/test_release_closure_store.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from backend.api.services.release_closure_store import (
    ReleaseClosureIntegrityError,
    ReleaseClosureStore,
)
from src.contracts.release_closure import (
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    make_release_closure_id,
    make_release_evidence_package_id,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
RELEASE_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"


def make_pair(idempotency_key: str = "close-1") -> tuple[ReleaseClosureRecord, ReleaseEvidencePackage]:
    closure_id = make_release_closure_id(RELEASE_EXECUTION_ID, idempotency_key)
    package_id = make_release_evidence_package_id(closure_id)
    closure = ReleaseClosureRecord(
        closure_id=closure_id,
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
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


def test_audit_tampering_blocks_writes(tmp_path: Path) -> None:
    store = ReleaseClosureStore(tmp_path)
    closure, package = make_pair()
    store.write_closure_with_package(closure, package, timestamp=closure.closed_at)
    audit_file = next((tmp_path / "audit").glob("release_closure_*.jsonl"))
    audit_file.write_text(audit_file.read_text(encoding="utf-8").replace("closure_recorded", "closure_read"), encoding="utf-8")

    with pytest.raises(ReleaseClosureIntegrityError, match="release closure integrity failed"):
        store.write_closure_with_package(*make_pair("close-2"), timestamp="2026-07-07T10:05:00+08:00")
```

- [ ] **Step 2: Run RED store tests**

Run:

```bash
pytest tests/backend/test_release_closure_store.py -q
```

Expected: FAIL because `release_closure_store.py` does not exist.

- [ ] **Step 3: Implement closure store**

Create `backend/api/services/release_closure_store.py` following the Step 14 monitoring store structure:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Generic, TypeVar

from src.contracts.release_closure import (
    GENESIS_CLOSURE_EVENT_HASH,
    ReleaseClosureAuditEvent,
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    build_release_closure_audit_event,
    canonical_closure_payload_hash,
    make_release_closure_event_id,
)


class ReleaseClosureIntegrityError(RuntimeError):
    """Raised when the release closure store is unsafe to append to."""


@dataclass(frozen=True)
class ReleaseClosureState:
    closures: list[ReleaseClosureRecord]
    evidence_packages: list[ReleaseEvidencePackage]
    audit_events: list[ReleaseClosureAuditEvent]
    integrity: dict[str, object]


@dataclass(frozen=True)
class ReleaseClosureIdempotencyMatch:
    closure: ReleaseClosureRecord
    package: ReleaseEvidencePackage
```

Implement:

- `read_state()` that verifies closure/package/audit consistency and returns warnings in `integrity`.
- `find_closure_by_idempotency_key()` that refuses lookup when integrity is not verified.
- `assert_idempotent_closure_matches()` that compares canonical hashes of closure and package.
- `write_closure_with_package()` that writes closure and package, appends two audit events, and removes newly written artifacts if package or audit append fails.
- `_ensure_root()`, `_artifact_path()`, `_write_json_once()`, `_append_audit_event()`, `_read_json_dir()`, `_read_audit_events_with_integrity()`, and Windows reserved-name/path traversal guards matching the Step 14 pattern.

- [ ] **Step 4: Add release closure README**

Create `reports/release_closure/README.md`:

```markdown
# Release Closure Artifacts

This directory is reserved for Step 15 post-release closure.

Runtime-generated files under `closures/`, `packages/`, and `audit/` are append-only closure evidence. They are created by admin-only closure APIs and should not be edited manually.

Step 15 closure state is local and auditable. It does not execute release, execute rollback, suppress monitoring alerts, mutate monitoring, mutate execution, mutate governance, mutate clinical safety policy, mutate prompts, mutate RAG indexes, promote literature evidence, or read patient/doctor runtime data.
```

- [ ] **Step 5: Run GREEN store tests**

Run:

```bash
pytest tests/backend/test_release_closure_contract.py tests/backend/test_release_closure_store.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add backend/api/services/release_closure_store.py reports/release_closure/README.md tests/backend/test_release_closure_store.py
git commit -m "feat: add release closure store"
```

### Task 3: Closure Service

**Files:**
- Create: `src/services/release_closure.py`
- Create: `tests/backend/test_release_closure_service.py`

**Interfaces:**
- Consumes: `ReleaseClosureStore`.
- Consumes: loaders returning dashboard, governance, execution, and monitoring dictionaries.
- Produces: `ReleaseClosureService.read_closure() -> dict[str, object]`.
- Produces: `ReleaseClosureService.record_closure` with keyword-only request fields returning `dict[str, object]`.
- Produces: `ReleaseClosureValidationError`, `ReleaseClosureConflictError`.

- [ ] **Step 1: Write failing service tests**

Create `tests/backend/test_release_closure_service.py` with focused fixtures and tests:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.api.services.release_closure_store import ReleaseClosureStore
from src.services.release_closure import (
    ReleaseClosureConflictError,
    ReleaseClosureService,
)


INTENT_ID = "release_intent_release_safety_20260629_001_6da729a0"
RELEASE_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b"
ROLLBACK_EXECUTION_ID = "release_exec_release_intent_release_safety_20260629_001_6da729a0_rollback_10ca7caa"


def dashboard() -> dict[str, Any]:
    return {"summary": {"hard_fail_count": 0}, "integrity": {"status": "verified", "warnings": []}}


def governance() -> dict[str, Any]:
    return {
        "active_intent": {"intent_id": INTENT_ID, "rollback_target": "agent_policy_20260624_0"},
        "integrity": {"status": "verified", "warnings": []},
    }


def execution(*, rollback: bool = False) -> dict[str, Any]:
    results = [
        {
            "execution_id": RELEASE_EXECUTION_ID,
            "intent_id": INTENT_ID,
            "action": "release",
            "status": "succeeded",
            "finished_at": "2026-07-07T09:00:00+08:00",
        }
    ]
    if rollback:
        results.append(
            {
                "execution_id": ROLLBACK_EXECUTION_ID,
                "intent_id": INTENT_ID,
                "action": "rollback",
                "status": "succeeded",
                "finished_at": "2026-07-07T09:30:00+08:00",
            }
        )
    return {"results": results, "integrity": {"status": "verified", "warnings": []}}


def monitoring(*, missing: bool = False, critical: bool = False, warning_acknowledged: bool = False, rollback_candidate: bool = False) -> dict[str, Any]:
    required_checks = [
        {"check_type": "execution_integrity", "status": "pass", "latest_check_id": "check-execution", "reason": "ok"},
        {"check_type": "governance_drift", "status": "pass", "latest_check_id": "check-governance", "reason": "ok"},
    ]
    if missing:
        required_checks[1] = {"check_type": "governance_drift", "status": "missing", "latest_check_id": None, "reason": "missing"}
    alerts = []
    acknowledgements = []
    if critical:
        alerts.append(
            {
                "alert_id": "release_monitor_alert_critical",
                "intent_id": INTENT_ID,
                "execution_id": RELEASE_EXECUTION_ID,
                "severity": "critical",
                "status": "active",
                "recommended_action": "execute_step13_rollback",
            }
        )
    if warning_acknowledged:
        alerts.append(
            {
                "alert_id": "release_monitor_alert_warning",
                "intent_id": INTENT_ID,
                "execution_id": RELEASE_EXECUTION_ID,
                "severity": "warning",
                "status": "acknowledged",
                "recommended_action": "investigate",
            }
        )
        acknowledgements.append({"alert_id": "release_monitor_alert_warning", "disposition": "accepted_risk"})
    return {
        "status": "monitoring",
        "required_checks": required_checks,
        "checks": [{"check_id": item["latest_check_id"]} for item in required_checks if item["latest_check_id"]],
        "alerts": alerts,
        "acknowledgements": acknowledgements,
        "rollback_trigger_candidate": {"candidate_id": "candidate-1"} if rollback_candidate else None,
        "integrity": {"status": "verified", "warnings": []},
    }


def service(tmp_path: Path, *, execution_model: dict[str, Any] | None = None, monitoring_model: dict[str, Any] | None = None) -> ReleaseClosureService:
    return ReleaseClosureService(
        store=ReleaseClosureStore(tmp_path),
        dashboard_loader=dashboard,
        governance_loader=governance,
        execution_loader=lambda: execution_model if execution_model is not None else execution(),
        monitoring_loader=lambda: monitoring_model if monitoring_model is not None else monitoring(),
        now=lambda: "2026-07-07T10:00:00+08:00",
    )


def test_closure_is_ready_when_required_checks_pass(tmp_path: Path) -> None:
    model = service(tmp_path).read_closure()

    assert model["status"] == "ready_to_close"
    assert model["closure_gate"]["allowed"] is True


def test_closure_blocked_when_required_checks_missing(tmp_path: Path) -> None:
    model = service(tmp_path, monitoring_model=monitoring(missing=True)).read_closure()

    assert model["status"] == "blocked"
    assert "required monitoring checks are missing" in model["closure_gate"]["reasons"]


def test_record_accepted_closure_writes_package(tmp_path: Path) -> None:
    app = service(tmp_path)

    model = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="accepted",
        closed_by="release_manager",
        rationale="Required checks passed.",
        idempotency_key="close-1",
    )

    assert model["status"] == "closed"
    assert model["latest_closure"]["closure_status"] == "accepted"
    assert model["latest_evidence_package"]["closure_id"] == model["latest_closure"]["closure_id"]


def test_record_closure_rejects_active_critical_alert(tmp_path: Path) -> None:
    app = service(tmp_path, monitoring_model=monitoring(critical=True))

    with pytest.raises(ReleaseClosureConflictError, match="active critical monitoring alerts"):
        app.record_closure(
            intent_id=INTENT_ID,
            release_execution_id=RELEASE_EXECUTION_ID,
            closure_status="accepted",
            closed_by="release_manager",
            rationale="Close anyway.",
            idempotency_key="close-1",
        )


def test_rolled_back_closure_requires_successful_rollback(tmp_path: Path) -> None:
    app = service(tmp_path, execution_model=execution(rollback=True))

    model = app.record_closure(
        intent_id=INTENT_ID,
        release_execution_id=RELEASE_EXECUTION_ID,
        closure_status="rolled_back",
        closed_by="release_manager",
        rationale="Rollback completed after monitoring trigger.",
        idempotency_key="close-rollback-1",
    )

    assert model["status"] == "rolled_back_closed"
    assert model["latest_closure"]["rollback_execution_id"] == ROLLBACK_EXECUTION_ID
```

- [ ] **Step 2: Run RED service tests**

Run:

```bash
pytest tests/backend/test_release_closure_service.py -q
```

Expected: FAIL because `src.services.release_closure` does not exist.

- [ ] **Step 3: Implement closure service**

Create `src/services/release_closure.py` with:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from backend.api.services.release_closure_store import ReleaseClosureStore
from src.contracts.release_closure import (
    CLOSURE_STATUSES,
    ReleaseClosureGate,
    ReleaseClosureGateCheck,
    ReleaseClosureRecord,
    ReleaseEvidencePackage,
    canonical_closure_payload_hash,
    make_release_closure_id,
    make_release_evidence_package_id,
)


class ReleaseClosureValidationError(ValueError):
    """Raised when a closure request payload is invalid."""


class ReleaseClosureConflictError(ValueError):
    """Raised when closure is blocked by current release state."""
```

Implement `ReleaseClosureService`:

- `read_closure()` loads dashboard, governance, execution, monitoring, and closure store state.
- `record_closure()` validates status, latest release, requested IDs, idempotency, and gate.
- `_latest_successful_release(execution)` returns latest release result by finished timestamp.
- `_latest_successful_rollback(execution, intent_id)` returns latest rollback result for the same intent.
- `_derive_gate` returns `ReleaseClosureGate`.
- `_snapshot_hash(model)` uses `canonical_closure_payload_hash`.
- `_build_closure_record` creates stable closure payload.
- `_build_evidence_package` creates local JSON evidence package.
- `_status_from_state` returns `idle`, `ready_to_close`, `blocked`, `closed`, or `rolled_back_closed`.

Gate failures must include exact reason strings used by tests:

- `no successful release execution exists`
- `required monitoring checks are missing`
- `active critical monitoring alerts exist`
- `rollback trigger candidate exists`
- `successful rollback is required for rolled_back closure`
- `accepted closure is blocked after rollback`
- `release closure integrity failed`

- [ ] **Step 4: Run GREEN service tests**

Run:

```bash
pytest tests/backend/test_release_closure_contract.py tests/backend/test_release_closure_store.py tests/backend/test_release_closure_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/services/release_closure.py tests/backend/test_release_closure_service.py
git commit -m "feat: derive release closure gate"
```

### Task 4: Closure Admin API, Auth, And Non-Mutation

**Files:**
- Create: `backend/api/schemas/release_closure.py`
- Modify: `backend/api/routes/admin.py`
- Modify: `backend/app.py`
- Modify: `tests/backend/test_auth_security.py`
- Create: `tests/backend/test_release_closure_api.py`
- Create: `tests/backend/test_release_closure_non_mutation.py`

**Interfaces:**
- Consumes: `ReleaseClosureService`.
- Produces: `GET /api/admin/release-closure`.
- Produces: `POST /api/admin/release-closure/closures`.

- [ ] **Step 1: Write failing API and auth tests**

Create `tests/backend/test_release_closure_api.py` with route-level monkeypatching of the service factory:

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app
import backend.api.routes.admin as admin_routes
from src.services.release_closure import ReleaseClosureConflictError


def test_get_release_closure_requires_admin_token(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")
    monkeypatch.setattr(
        admin_routes,
        "_release_closure_service",
        lambda: type("Svc", (), {"read_closure": lambda self: {"status": "idle"}})(),
    )
    client = TestClient(create_app())

    response = client.get("/api/admin/release-closure", headers={"Authorization": "Bearer admin-token"})

    assert response.status_code == 200
    assert response.json()["status"] == "idle"


def test_record_closure_maps_gate_conflict(monkeypatch) -> None:
    monkeypatch.setenv("API_BEARER_TOKEN", "user-token")
    monkeypatch.setenv("API_ADMIN_BEARER_TOKEN", "admin-token")

    class Service:
        def record_closure(self, **kwargs):
            raise ReleaseClosureConflictError("active critical monitoring alerts exist")

    monkeypatch.setattr(admin_routes, "_release_closure_service", lambda: Service())
    client = TestClient(create_app())

    response = client.post(
        "/api/admin/release-closure/closures",
        headers={"Authorization": "Bearer admin-token"},
        json={
            "intent_id": "intent-1",
            "release_execution_id": "release-exec-1",
            "closure_status": "accepted",
            "closed_by": "release_manager",
            "rationale": "Close release.",
            "idempotency_key": "close-1",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "active critical monitoring alerts exist"
```

Modify `tests/backend/test_auth_security.py` to include:

```python
@app.get("/api/admin/release-closure")
async def release_closure_read():
    return {"ok": True}

@app.post("/api/admin/release-closure/closures")
async def release_closure_write():
    return {"ok": True}
```

- [ ] **Step 2: Run RED API tests**

Run:

```bash
pytest tests/backend/test_release_closure_api.py tests/backend/test_auth_security.py -q
```

Expected: FAIL because closure routes are missing from `admin.py` and `backend/app.py`.

- [ ] **Step 3: Add closure schema**

Create `backend/api/schemas/release_closure.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


NonEmptyString = str
ReleaseClosureStatus = Literal["accepted", "accepted_with_observations", "rolled_back"]


class ReleaseClosureRequest(BaseModel):
    intent_id: NonEmptyString = Field(min_length=1)
    release_execution_id: NonEmptyString = Field(min_length=1)
    closure_status: ReleaseClosureStatus
    closed_by: NonEmptyString = Field(min_length=1)
    rationale: NonEmptyString = Field(min_length=1)
    idempotency_key: NonEmptyString = Field(min_length=1)
```

- [ ] **Step 4: Add admin routes**

Modify `backend/api/routes/admin.py`:

```python
from backend.api.schemas.release_closure import ReleaseClosureRequest
from backend.api.services.release_closure_store import (
    ReleaseClosureIntegrityError,
    ReleaseClosureStore,
)
from src.services.release_closure import (
    ReleaseClosureConflictError,
    ReleaseClosureService,
    ReleaseClosureValidationError,
)

_CLOSURE_STORE_ROOT = REPO_ROOT / "reports" / "release_closure"


def _release_closure_service() -> ReleaseClosureService:
    return ReleaseClosureService(
        store=ReleaseClosureStore(_CLOSURE_STORE_ROOT),
        dashboard_loader=build_release_dashboard,
        governance_loader=_release_governance_service().read_governance,
        execution_loader=_release_execution_service().read_execution,
        monitoring_loader=_release_monitoring_service().read_monitoring,
        now=_governance_timestamp,
    )
```

Add error mapping:

```python
def _raise_closure_http_error(exc: Exception) -> None:
    if isinstance(exc, (ReleaseClosureConflictError, ReleaseClosureIntegrityError, FileExistsError)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (ReleaseClosureValidationError, TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, OSError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc
```

Add routes:

```python
@router.get("/release-closure")
async def get_admin_release_closure() -> dict[str, Any]:
    return _release_closure_service().read_closure()


@router.post("/release-closure/closures")
async def record_admin_release_closure(payload: ReleaseClosureRequest) -> dict[str, Any]:
    try:
        return _release_closure_service().record_closure(**_model_dump(payload))
    except Exception as exc:
        _raise_closure_http_error(exc)
```

- [ ] **Step 5: Add auth guard**

Modify `backend/app.py` in `_requires_admin_token()`:

```python
if method == "GET" and path == "/api/admin/release-closure":
    return True
if method == "POST" and path == "/api/admin/release-closure/closures":
    return True
```

- [ ] **Step 6: Add non-mutation test**

Create `tests/backend/test_release_closure_non_mutation.py` with a snapshot helper that excludes only `reports/release_closure/` and test temp/cache directories. Record a closure through the service and assert no paths outside the closure root changed. Include `reports/release_monitoring/`, `reports/release_execution/`, and `reports/release_governance/` fixture files in the snapshot so the test catches accidental mutation.

- [ ] **Step 7: Run GREEN API and non-mutation tests**

Run:

```bash
pytest tests/backend/test_release_closure_api.py tests/backend/test_auth_security.py tests/backend/test_release_closure_non_mutation.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add backend/api/schemas/release_closure.py backend/api/routes/admin.py backend/app.py tests/backend/test_auth_security.py tests/backend/test_release_closure_api.py tests/backend/test_release_closure_non_mutation.py
git commit -m "feat: add release closure admin api"
```

### Task 5: Frontend Closure API

**Files:**
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/test/test-utils.tsx`

**Interfaces:**
- Produces: `AdminReleaseClosureResponse`.
- Produces: `getAdminReleaseClosure()`.
- Produces: `recordAdminReleaseClosure(request)`.

- [ ] **Step 1: Write failing frontend client tests**

Modify `frontend/src/app/api/client.test.ts`:

```ts
it("gets admin release closure", async () => {
  const payload = releaseClosureResponse();
  const fetch = vi.fn().mockResolvedValue(jsonResponse(payload));
  const client = createApiClient({ baseUrl: "http://127.0.0.1:8000", fetchImpl: fetch, adminToken: "admin-token" });

  await expect(client.getAdminReleaseClosure()).resolves.toEqual(payload);

  expect(fetch).toHaveBeenCalledWith(
    "http://127.0.0.1:8000/api/admin/release-closure",
    expect.objectContaining({ headers: expect.objectContaining({ Authorization: "Bearer admin-token" }) }),
  );
});

it("records admin release closure with JSON request body", async () => {
  const payload = releaseClosureResponse();
  const fetch = vi.fn().mockResolvedValue(jsonResponse(payload));
  const client = createApiClient({ baseUrl: "http://127.0.0.1:8000", fetchImpl: fetch, adminToken: "admin-token" });
  const request = {
    intent_id: "intent-1",
    release_execution_id: "release-exec-1",
    closure_status: "accepted",
    closed_by: "release_manager",
    rationale: "Required checks passed.",
    idempotency_key: "close-1",
  } as const;

  await expect(client.recordAdminReleaseClosure(request)).resolves.toEqual(payload);

  expect(fetch).toHaveBeenCalledWith(
    "http://127.0.0.1:8000/api/admin/release-closure/closures",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify(request),
    }),
  );
});
```

- [ ] **Step 2: Run RED frontend client tests**

Run:

```bash
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run frontend/src/app/api/client.test.ts --reporter=verbose
```

Expected: FAIL because closure client types and methods are missing.

- [ ] **Step 3: Add frontend closure types**

Modify `frontend/src/app/api/types.ts`:

```ts
export type AdminReleaseClosureStatus = "idle" | "ready_to_close" | "blocked" | "closed" | "rolled_back_closed";
export type AdminReleaseClosureRecordStatus = "accepted" | "accepted_with_observations" | "rolled_back";
export type AdminReleaseClosureGateCheckStatus = "pass" | "warning" | "fail";

export interface AdminReleaseClosureGateCheck {
  name: string;
  status: AdminReleaseClosureGateCheckStatus;
  reason: string;
}

export interface AdminReleaseClosureGate {
  allowed: boolean;
  status: AdminReleaseClosureStatus;
  reasons: string[];
  checks: AdminReleaseClosureGateCheck[];
}

export interface AdminReleaseClosureLatestRelease {
  intent_id: string;
  release_execution_id: string;
  released_at: string | null;
  rollback_execution_id: string | null;
  rolled_back_at: string | null;
}

export interface AdminReleaseClosureRecord {
  closure_id: string;
  intent_id: string;
  release_execution_id: string;
  rollback_execution_id: string | null;
  closure_status: AdminReleaseClosureRecordStatus;
  closed_by: string;
  closed_at: string;
  rationale: string;
  evidence_package_id: string;
  idempotency_key: string;
}

export interface AdminReleaseEvidencePackage {
  package_id: string;
  closure_id: string;
  intent_id: string;
  release_execution_id: string;
  rollback_execution_id: string | null;
  generated_by: string;
  generated_at: string;
  closure_status: AdminReleaseClosureRecordStatus;
  summary: string;
  source_refs: string[];
  artifact_refs: string[];
  snapshot_hashes: Record<string, string>;
}

export interface AdminReleaseClosureResponse {
  status: AdminReleaseClosureStatus;
  latest_release: AdminReleaseClosureLatestRelease | null;
  closure_gate: AdminReleaseClosureGate;
  latest_closure: AdminReleaseClosureRecord | null;
  latest_evidence_package: AdminReleaseEvidencePackage | null;
  closures: AdminReleaseClosureRecord[];
  evidence_packages: AdminReleaseEvidencePackage[];
  integrity: { status: "verified" | "failed"; warnings: string[] };
  runtime: { auth: "admin"; source: "reports/release_closure"; mode: "post_release_closure" };
}

export interface AdminRecordReleaseClosureRequest {
  intent_id: string;
  release_execution_id: string;
  closure_status: AdminReleaseClosureRecordStatus;
  closed_by: string;
  rationale: string;
  idempotency_key: string;
}
```

- [ ] **Step 4: Add client methods and test stubs**

Modify `frontend/src/app/api/client.ts` to add methods:

```ts
getAdminReleaseClosure(): Promise<AdminReleaseClosureResponse>;
recordAdminReleaseClosure(request: AdminRecordReleaseClosureRequest): Promise<AdminReleaseClosureResponse>;
```

Implement:

```ts
async getAdminReleaseClosure() {
  const response = await fetchImpl(buildUrl("/api/admin/release-closure", baseUrl), {
    headers: adminHeaders(),
  });
  return parseJsonResponse<AdminReleaseClosureResponse>(response);
},
async recordAdminReleaseClosure(request) {
  const response = await fetchImpl(buildUrl("/api/admin/release-closure/closures", baseUrl), {
    method: "POST",
    headers: jsonAdminHeaders(),
    body: JSON.stringify(request),
  });
  return parseJsonResponse<AdminReleaseClosureResponse>(response);
},
```

Modify `frontend/src/test/test-utils.tsx` default client stub to include both methods returning an idle closure response.

- [ ] **Step 5: Run GREEN frontend client tests**

Run:

```bash
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run frontend/src/app/api/client.test.ts --reporter=verbose
cmd /c D:\anaconda3\envs\LangG\node.exe frontend\node_modules\typescript\bin\tsc -p frontend\tsconfig.json --noEmit
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/test/test-utils.tsx
git commit -m "feat: add release closure frontend api"
```

### Task 6: Agent Admin Closure UI

**Files:**
- Modify: `frontend/src/features/agent-admin/agent-admin-view.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-pages.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
- Modify: `frontend/src/styles/globals.css`

**Interfaces:**
- Consumes: Task 5 frontend closure API.
- Produces: closure resource loading in release task.
- Produces: closure panel below monitoring.
- Produces: closure form action handler.

- [ ] **Step 1: Write failing Agent Admin tests**

Modify `frontend/src/features/agent-admin/agent-admin-view.test.tsx`:

```ts
it("renders release closure gate and submits closure request", async () => {
  const api = createAgentAdminApiStub({
    getAdminReleaseClosure: vi.fn().mockResolvedValue(readyReleaseClosureResponse()),
    recordAdminReleaseClosure: vi.fn().mockResolvedValue(closedReleaseClosureResponse()),
  });

  render(<AgentAdminView api={api} initialTaskId="release" />);

  expect(await screen.findByText(/release closure/i)).toBeInTheDocument();
  expect(screen.getByText(/ready_to_close/i)).toBeInTheDocument();

  await userEvent.clear(screen.getByLabelText(/closure actor/i));
  await userEvent.type(screen.getByLabelText(/closure actor/i), "release_manager");
  await userEvent.clear(screen.getByLabelText(/closure rationale/i));
  await userEvent.type(screen.getByLabelText(/closure rationale/i), "Required checks passed.");
  await userEvent.clear(screen.getByLabelText(/closure idempotency key/i));
  await userEvent.type(screen.getByLabelText(/closure idempotency key/i), "close-1");
  await userEvent.click(screen.getByRole("button", { name: /record closure/i }));

  expect(api.recordAdminReleaseClosure).toHaveBeenCalledWith(
    expect.objectContaining({
      closure_status: "accepted",
      closed_by: "release_manager",
      idempotency_key: "close-1",
    }),
  );
});
```

Add tests for blocked gate reasons, closed package summary, and closure API error isolation.

- [ ] **Step 2: Run RED Agent Admin tests**

Run:

```bash
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run frontend/src/features/agent-admin/agent-admin-view.test.tsx --reporter=verbose
```

Expected: FAIL because closure resource and UI do not exist.

- [ ] **Step 3: Load closure resource and action**

Modify `frontend/src/features/agent-admin/agent-admin-view.tsx`:

- add closure resource state beside dashboard, governance, execution, and monitoring;
- load `api.getAdminReleaseClosure()` when release task is active;
- refresh closure after governance, execution, monitoring, and closure mutations;
- guard against request ordering the same way Step 14 monitoring refresh is guarded;
- add `recordReleaseClosure(request)` action that calls `api.recordAdminReleaseClosure(request)` and updates closure state.

- [ ] **Step 4: Render closure panel**

Modify `frontend/src/features/agent-admin/agent-admin-pages.tsx`:

- add `ReleaseClosureSection` below `ReleaseMonitoringSection`;
- render loading, error, idle, blocked, ready, closed, and rolled-back-closed states;
- render gate checks as compact status rows;
- render form fields:
  - `closure actor`;
  - `closure status`;
  - `closure rationale`;
  - `closure idempotency key`;
- disable submit when `closure_gate.allowed` is false;
- render latest closure and latest evidence package summary.

Use existing `AgentAdminPanel`, release form, and status styles before adding CSS.

- [ ] **Step 5: Add minimal styles only if needed**

Modify `frontend/src/styles/globals.css` only for classes that cannot reuse existing release styles:

```css
.agent-admin-release-closure-gate {
  display: grid;
  gap: 0.5rem;
}

.agent-admin-release-closure-package {
  display: grid;
  gap: 0.35rem;
  overflow-wrap: anywhere;
}
```

- [ ] **Step 6: Run GREEN Agent Admin tests and TypeScript**

Run:

```bash
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run frontend/src/features/agent-admin/agent-admin-view.test.tsx --reporter=verbose
cmd /c D:\anaconda3\envs\LangG\node.exe frontend\node_modules\typescript\bin\tsc -p frontend\tsconfig.json --noEmit
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx frontend/src/styles/globals.css
git commit -m "feat: show release closure in agent admin"
```

### Task 7: Final Verification And Regression

**Files:**
- Modify implementation files only if verification exposes a real Step 15 defect.

**Interfaces:**
- Produces: verified Step 15 branch.

- [ ] **Step 1: Run Step 15 backend focused verification**

Run:

```bash
pytest tests/backend/test_release_closure_contract.py tests/backend/test_release_closure_store.py tests/backend/test_release_closure_service.py tests/backend/test_release_closure_api.py tests/backend/test_release_closure_non_mutation.py -q
```

Expected: all Step 15 backend tests pass.

- [ ] **Step 2: Run release stack backend regressions**

Run:

```bash
pytest tests/backend/test_release_monitoring_contract.py tests/backend/test_release_monitoring_store.py tests/backend/test_release_monitoring_service.py tests/backend/test_release_monitoring_api.py tests/backend/test_release_monitoring_non_mutation.py tests/backend/test_release_execution_contract.py tests/backend/test_release_execution_store.py tests/backend/test_release_execution_service.py tests/backend/test_release_execution_api.py tests/backend/test_release_execution_non_mutation.py tests/backend/test_release_governance_api.py tests/backend/test_release_governance_non_mutation.py -q
```

Expected: Step 14, Step 13, and Step 12 regressions pass.

- [ ] **Step 3: Run clinical and evidence regressions**

Run:

```bash
pytest tests/backend/test_crc_harness_replay.py tests/backend/test_crc_triage_mutation_pack.py tests/backend/test_clinical_safety_policy.py tests/backend/test_intended_use_profiles.py tests/backend/test_patient_triage_protocol.py -q
pytest tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
pytest tests/backend/test_literature_harness.py tests/backend/test_evidence_claim_contract.py tests/backend/test_clinical_assertion_projection.py -q
```

Expected: P0, P1, Step 10, and clinical assertion regressions pass.

- [ ] **Step 4: Run frontend focused tests**

Run:

```bash
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run frontend/src/app/api/client.test.ts frontend/src/features/agent-admin/agent-admin-view.test.tsx --reporter=verbose
```

Expected: frontend client and Agent Admin tests pass.

- [ ] **Step 5: Run frontend build**

Run:

```bash
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: build passes. Existing Vite chunk-size warnings are acceptable if no new build failure appears.

- [ ] **Step 6: Check diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` has no output. `git status --short` lists only Step 15 files before final commit.

- [ ] **Step 7: Commit verification fixes if needed**

If verification required code changes, commit them:

```bash
git add <changed-step15-files>
git commit -m "test: stabilize release closure verification"
```

Expected: no commit is created when verification needs no changes.

## Implementation Handoff

Recommended execution mode: Subagent-Driven. Task boundaries are independent enough for one worker per task with review after each task:

1. Contracts.
2. Store.
3. Service.
4. API/auth/non-mutation.
5. Frontend API.
6. Agent Admin UI.
7. Final verification.

Do not execute release or rollback from closure code. Do not mutate monitoring acknowledgements or alerts. Do not copy patient-level data into closure packages.

## Plan Self-Review

Spec coverage: Tasks cover contracts, store, service, API/auth, non-mutation, frontend API, Agent Admin UI, docs, and verification. Closure evidence package generation is part of the store/service tasks and is generated atomically with closure recording.

Placeholder scan: no unresolved work markers remain.

Type consistency: backend response keys match frontend interfaces: `status`, `latest_release`, `closure_gate`, `latest_closure`, `latest_evidence_package`, `closures`, `evidence_packages`, `integrity`, and `runtime`. Client method names are consistently `getAdminReleaseClosure()` and `recordAdminReleaseClosure()`.

Plan complete and saved to `docs/superpowers/plans/2026-07-07-post-release-closure.md`. Two execution options:

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. Inline Execution - execute tasks in this session using executing-plans, batch execution with checkpoints.
