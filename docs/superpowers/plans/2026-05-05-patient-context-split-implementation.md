# Patient Context Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split historical case database sample context from real patient registry binding so chat-derived sample ids like `093` cannot trigger patient registry requests or contaminate clinical patient state.

**Architecture:** Add explicit `case_database_patient_id` and `registry_patient_id` fields through backend state, response schemas, snapshots, payload building, and frontend session state. Migrate case database, imaging, and pathology consumers to the sample field while making patient registry UI depend only on explicit registry binding. Keep `current_patient_id` as a read-only compatibility bridge during migration.

**Tech Stack:** Python, Pydantic, FastAPI response schemas, LangGraph state payloads, React, TypeScript, Vitest, pytest

---

## File Structure

### New Files To Create

- `tests/backend/test_patient_context_split_backend.py`
  - Backend contract tests for snapshot mapping, payload injection, and case-database sample extraction.

### Existing Files To Modify

- `src/state.py`
  - Add `case_database_patient_id` and `registry_patient_id` to `CRCAgentState`.
- `backend/api/schemas/responses.py`
  - Add new response fields to `RecoverySnapshot`.
- `backend/api/adapters/state_snapshot.py`
  - Populate `case_database_patient_id` and `registry_patient_id`.
  - Keep `current_patient_id` for compatibility only.
- `backend/api/services/payload_builder.py`
  - Inject new context fields into graph payloads.
  - Map legacy `current_patient_id` to `case_database_patient_id` only.
- `backend/api/routes/sessions.py`
  - Ensure explicit doctor binding flows surface `registry_patient_id`.
- `src/nodes/database_nodes.py`
  - Store case database sample ids under `case_database_patient_id`.
  - Stop writing `current_patient_id` for sample lookup branches.
- `src/nodes/radiology_nodes.py`
  - Resolve sample id from `case_database_patient_id` before legacy `current_patient_id`.
- `src/nodes/pathology_nodes.py`
  - Resolve sample id from `case_database_patient_id` before legacy `current_patient_id`.
- `src/nodes/planner.py`
  - Prefer `case_database_patient_id` for database-oriented planning context.
- `frontend/src/app/api/types.ts`
  - Add snake-case snapshot fields and camel-case session state fields.
- `frontend/src/app/store/stream-reducer.ts`
  - Hydrate and initialize the new fields.
- `frontend/src/app/store/stream-reducer.test.ts`
  - Cover hydration and legacy compatibility.
- `frontend/src/pages/workspace-page.tsx`
  - Drive `usePatientRegistry` from `registryPatientId`, not `currentPatientId`.
- `frontend/src/pages/workspace-page.test.tsx`
  - Cover no registry request when only case sample id exists.
- `frontend/src/features/patient-registry/use-patient-registry.ts`
  - Rename input option to `registryPatientId`.
  - Stop retrying same missing registry id until id changes.
- `frontend/src/features/patient-registry/use-patient-registry.test.tsx`
  - Cover registry fetch gating and missing-id retry prevention.
- `frontend/src/features/doctor/doctor-scene-shell.tsx`
  - Display separate registry patient and case sample labels.
- `frontend/src/features/doctor/doctor-scene-shell.test.tsx`
  - Cover the separate labels.
- `frontend/src/test/test-utils.tsx`
  - Extend default snapshot/session helpers with new fields.
- `tests/frontend/test-utils.tsx`
  - Mirror test helper changes for the secondary frontend test tree.

### Files To Inspect During Implementation

- `backend/api/services/session_store.py`
  - Confirm `SessionMeta.patient_id` remains the registry binding source.
- `backend/api/services/patient_context_resolver.py`
  - Confirm doctor registry context injection uses only `SessionMeta.patient_id`.
- `tests/backend/test_patient_context_resolver.py`
  - Reuse existing payload-builder test setup.
- `tests/backend/test_chat_main_database_integration.py`
  - Update old `current_patient_id` expectations to `case_database_patient_id`.

## Key Contracts

- `case_database_patient_id` is a string sample id, for example `"093"`.
- `registry_patient_id` is a numeric registry primary key, for example `7`.
- `current_patient_id` remains in responses temporarily but must not drive registry fetches.
- Chat text and case database extraction can produce only `case_database_patient_id`.
- Only explicit registry creation, selection, or bind actions can produce `registry_patient_id`.

## Task 1: Backend Schema And State Fields

**Files:**
- Modify: `src/state.py`
- Modify: `backend/api/schemas/responses.py`
- Modify: `frontend/src/app/api/types.ts`
- Test: `tests/backend/test_patient_context_split_backend.py`
- Test: `frontend/src/app/store/stream-reducer.test.ts`

- [ ] **Step 1: Write backend schema/state tests**

Create `tests/backend/test_patient_context_split_backend.py` with the imports and first two tests:

```python
from __future__ import annotations

from langchain_core.messages import HumanMessage

from backend.api.adapters.state_snapshot import build_recovery_snapshot
from backend.api.services.payload_builder import build_graph_payload
from backend.api.services.session_store import SessionMeta
from src.state import CRCAgentState


def test_agent_state_accepts_split_patient_context_fields() -> None:
    state = CRCAgentState(
        messages=[HumanMessage(content="query sample 093")],
        case_database_patient_id="093",
        registry_patient_id=7,
    )

    assert state.case_database_patient_id == "093"
    assert state.registry_patient_id == 7


def test_recovery_snapshot_exposes_split_patient_context_fields() -> None:
    snapshot = build_recovery_snapshot(
        SessionMeta(session_id="sess-test", thread_id="thread-test", patient_id=7),
        {
            "case_database_patient_id": "093",
            "registry_patient_id": 7,
            "current_patient_id": "093",
            "findings": {},
        },
    )

    assert snapshot.case_database_patient_id == "093"
    assert snapshot.registry_patient_id == 7
    assert snapshot.current_patient_id == "093"
```

- [ ] **Step 2: Run backend tests and verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_context_split_backend.py -q --basetemp=tmp\pytest-patient-context-split-red
```

Expected: FAIL because `CRCAgentState` and `RecoverySnapshot` do not yet expose the new fields.

- [ ] **Step 3: Add backend state and response fields**

In `src/state.py`, place these fields next to `current_patient_id`:

```python
    case_database_patient_id: Optional[str] = None
    registry_patient_id: Optional[int] = None
```

Keep the existing `current_patient_id: Optional[str] = None` field unchanged for compatibility.

In `backend/api/schemas/responses.py`, add fields to `RecoverySnapshot` next to `current_patient_id`:

```python
    case_database_patient_id: str | None = None
    registry_patient_id: int | None = None
    current_patient_id: int | str | None = None
```

- [ ] **Step 4: Write frontend type test**

Add this test to `frontend/src/app/store/stream-reducer.test.ts`:

```ts
it("hydrates split patient context fields from recovery snapshots", () => {
  const state = hydrateSessionState(createInitialSessionState(), {
    session_id: "sess",
    thread_id: "thread",
    scene: "doctor",
    patient_id: 7,
    snapshot_version: 1,
    runtime: { runner_mode: "real", fixture_case: null },
    snapshot: {
      snapshot_version: 1,
      messages: [],
      messages_total: 0,
      messages_next_before_cursor: null,
      cards: [],
      roadmap: [],
      findings: {},
      patient_profile: null,
      patient_identity: null,
      stage: null,
      assessment_draft: null,
      case_database_patient_id: "093",
      registry_patient_id: 7,
      current_patient_id: "093",
      references: [],
      plan: [],
      critic: null,
      safety_alert: null,
      uploaded_assets: {},
      context_maintenance: null,
      context_state: null,
    },
  });

  expect(state.caseDatabasePatientId).toBe("093");
  expect(state.registryPatientId).toBe(7);
  expect(state.currentPatientId).toBe("093");
});
```

- [ ] **Step 5: Run frontend test and verify failure**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/store/stream-reducer.test.ts --reporter=verbose
```

Expected: FAIL because `RecoverySnapshot` and `SessionState` do not yet define `case_database_patient_id`, `registry_patient_id`, `caseDatabasePatientId`, or `registryPatientId`.

If this fails with `spawn EPERM`, rerun the same command with sandbox escalation.

- [ ] **Step 6: Add frontend API and session state fields**

In `frontend/src/app/api/types.ts`, add to `RecoverySnapshot` next to `current_patient_id`:

```ts
  case_database_patient_id: string | null;
  registry_patient_id: number | null;
  current_patient_id: string | number | null;
```

Add to `SessionState` next to `currentPatientId`:

```ts
  caseDatabasePatientId: string | null;
  registryPatientId: number | null;
  currentPatientId: string | number | null;
```

In `frontend/src/app/store/stream-reducer.ts`, update `createInitialSessionState()`:

```ts
    caseDatabasePatientId: null,
    registryPatientId: null,
    currentPatientId: null,
```

Update `hydrateSessionState(...)`:

```ts
    caseDatabasePatientId: snapshot.case_database_patient_id ?? null,
    registryPatientId: snapshot.registry_patient_id ?? response.patient_id ?? null,
    currentPatientId: snapshot.current_patient_id,
```

- [ ] **Step 7: Run Task 1 tests and verify pass**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_context_split_backend.py -q --basetemp=tmp\pytest-patient-context-split-task1
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/store/stream-reducer.test.ts --reporter=verbose
```

Expected: backend test file passes; frontend reducer tests pass.

- [ ] **Step 8: Commit Task 1**

```powershell
git add src/state.py backend/api/schemas/responses.py frontend/src/app/api/types.ts frontend/src/app/store/stream-reducer.ts frontend/src/app/store/stream-reducer.test.ts tests/backend/test_patient_context_split_backend.py
git commit -m "feat: add split patient context fields"
```

## Task 2: Snapshot And Payload Builder Contract

**Files:**
- Modify: `backend/api/adapters/state_snapshot.py`
- Modify: `backend/api/services/payload_builder.py`
- Test: `tests/backend/test_patient_context_split_backend.py`
- Test: `tests/backend/test_patient_context_resolver.py`

- [ ] **Step 1: Extend backend contract tests for payload builder**

Append these tests to `tests/backend/test_patient_context_split_backend.py`:

```python
class _ChatRequest:
    def __init__(self, message: str, context: dict | None = None) -> None:
        self.message = HumanMessage(content=message)
        self.context = context or {}


def test_payload_builder_injects_split_context_fields() -> None:
    prepared = build_graph_payload(
        chat_request=_ChatRequest("continue sample work"),
        session_meta=SessionMeta(session_id="sess-test", thread_id="thread-test", patient_id=7),
        state_snapshot={
            "case_database_patient_id": "093",
            "registry_patient_id": 7,
            "current_patient_id": "093",
        },
    )

    assert prepared.payload["case_database_patient_id"] == "093"
    assert prepared.payload["registry_patient_id"] == 7
    assert "current_patient_id" not in prepared.payload


def test_payload_builder_maps_legacy_current_patient_id_to_case_sample_only() -> None:
    prepared = build_graph_payload(
        chat_request=_ChatRequest("continue sample work"),
        session_meta=SessionMeta(session_id="sess-test", thread_id="thread-test", patient_id=None),
        state_snapshot={"current_patient_id": "093"},
    )

    assert prepared.payload["case_database_patient_id"] == "093"
    assert "registry_patient_id" not in prepared.payload
    assert "current_patient_id" not in prepared.payload


def test_snapshot_maps_session_patient_to_registry_patient_id() -> None:
    snapshot = build_recovery_snapshot(
        SessionMeta(session_id="sess-test", thread_id="thread-test", patient_id=7),
        {"findings": {}},
    )

    assert snapshot.registry_patient_id == 7
    assert snapshot.case_database_patient_id is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_context_split_backend.py -q --basetemp=tmp\pytest-patient-context-split-task2-red
```

Expected: FAIL because snapshot builder and payload builder do not yet populate or inject the new fields.

- [ ] **Step 3: Update payload builder allowlist and legacy bridge**

In `backend/api/services/payload_builder.py`, replace the context allowlist with:

```python
CONTEXT_PAYLOAD_ALLOWLIST = {
    "fixture_case",
    "fixture_tick_delay_ms",
    "case_database_patient_id",
    "registry_patient_id",
}
```

Replace the existing `current_patient_id` injection block with:

```python
    case_database_patient_id = _snapshot_value(state_snapshot, "case_database_patient_id")
    legacy_current_patient_id = _snapshot_value(state_snapshot, "current_patient_id")
    if case_database_patient_id is None and legacy_current_patient_id is not None:
        case_database_patient_id = str(legacy_current_patient_id).zfill(3)
    if case_database_patient_id is not None:
        payload["case_database_patient_id"] = case_database_patient_id

    registry_patient_id = _snapshot_value(state_snapshot, "registry_patient_id")
    if registry_patient_id is None:
        registry_patient_id = getattr(session_meta, "patient_id", None)
    if registry_patient_id is not None:
        payload["registry_patient_id"] = registry_patient_id
```

Do not add `current_patient_id` to `payload`.

- [ ] **Step 4: Update snapshot builder**

In `backend/api/adapters/state_snapshot.py`, replace the `current_patient_id` calculation with:

```python
    findings = _coerce_mapping(_get_value(state, "findings", {})) or {}
    findings = _merge_triage_state_fields(state, findings)
    case_database_patient_id = (
        _get_value(state, "case_database_patient_id")
        or findings.get("case_database_patient_id")
        or _get_value(state, "current_patient_id")
        or findings.get("current_patient_id")
    )
    if case_database_patient_id is not None:
        case_database_patient_id = str(case_database_patient_id).zfill(3)
    registry_patient_id = (
        _get_value(state, "registry_patient_id")
        or findings.get("registry_patient_id")
        or session_meta.patient_id
    )
    current_patient_id = (
        _get_value(state, "current_patient_id")
        or findings.get("current_patient_id")
    )
```

When constructing `RecoverySnapshot`, add:

```python
        case_database_patient_id=case_database_patient_id,
        registry_patient_id=registry_patient_id,
        current_patient_id=current_patient_id,
```

- [ ] **Step 5: Run backend contract tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_context_split_backend.py tests\backend\test_patient_context_resolver.py -q --basetemp=tmp\pytest-patient-context-split-task2
```

Expected: all selected backend tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add backend/api/adapters/state_snapshot.py backend/api/services/payload_builder.py tests/backend/test_patient_context_split_backend.py tests/backend/test_patient_context_resolver.py
git commit -m "feat: split patient context in snapshots and payloads"
```

## Task 3: Move Case Database Writes To Sample Context

**Files:**
- Modify: `src/nodes/database_nodes.py`
- Modify: `tests/backend/test_chat_main_database_integration.py`
- Test: `tests/backend/test_patient_context_split_backend.py`

- [ ] **Step 1: Add case database node regression test**

Append this test to `tests/backend/test_patient_context_split_backend.py`:

```python
def test_case_database_result_uses_case_sample_id_not_registry_id() -> None:
    from src.nodes.database_nodes import _build_database_workbench_context

    context = _build_database_workbench_context(
        mode="detail",
        query_text="查看093号患者影像",
        filters={"patient_id": 93},
        selected_patient_id=93,
    )

    assert context["selected_patient_id"] == 93
```

This test pins the existing helper so Task 3 can focus on node return fields. The stronger behavior tests live in `tests/backend/test_chat_main_database_integration.py`.

Update existing assertions in `tests/backend/test_chat_main_database_integration.py`:

```python
assert result["case_database_patient_id"] == "093"
assert result["findings"]["case_database_patient_id"] == "093"
assert "current_patient_id" not in result or result["current_patient_id"] is None
```

Replace old assertions:

```python
assert result["current_patient_id"] == "093"
assert result["findings"]["pending_patient_id"] == "93"
```

with:

```python
assert result["case_database_patient_id"] == "093"
assert result["findings"]["case_database_patient_id"] == "093"
assert result["findings"]["pending_patient_id"] == "93"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_chat_main_database_integration.py tests\backend\test_patient_context_split_backend.py -q --basetemp=tmp\pytest-patient-context-split-task3-red
```

Expected: FAIL because `node_case_database` still returns `current_patient_id` for case database branches.

- [ ] **Step 3: Add local helper inside `node_case_database`**

In `src/nodes/database_nodes.py`, inside the deterministic tool-calling block where `current_patient_id = active_patient_id` is currently used, add:

```python
            def _normalize_case_database_patient_id(value: Any) -> str | None:
                if value is None:
                    return None
                text = str(value).strip()
                if not text:
                    return None
                return text.zfill(3) if text.isdigit() else text
```

Then replace:

```python
            current_patient_id = active_patient_id
```

with:

```python
            case_database_patient_id = _normalize_case_database_patient_id(active_patient_id)
            current_patient_id = active_patient_id
```

Keep the local `current_patient_id` variable for tool calls in this task, but it must no longer be returned as graph registry context.

- [ ] **Step 4: Replace return fields for deterministic case database branches**

For every return dictionary in `src/nodes/database_nodes.py` that currently contains:

```python
                    "current_patient_id": str(current_patient_id).zfill(3) if current_patient_id else None,
```

replace it with:

```python
                    "case_database_patient_id": case_database_patient_id,
```

For every `findings_updates["current_patient_id"] = ...` assignment inside case database sample query branches, replace it with:

```python
                    findings_updates["case_database_patient_id"] = case_database_patient_id
```

Do not change explicit patient-record edit branches that save real patient records until their tests are inspected. If a branch is only reading historical sample data, it writes the sample field.

- [ ] **Step 5: Run focused database tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_chat_main_database_integration.py tests\backend\test_patient_context_split_backend.py -q --basetemp=tmp\pytest-patient-context-split-task3
```

Expected: selected tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src/nodes/database_nodes.py tests/backend/test_chat_main_database_integration.py tests/backend/test_patient_context_split_backend.py
git commit -m "feat: store case database ids separately"
```

## Task 4: Migrate Imaging, Pathology, And Planning Consumers

**Files:**
- Modify: `src/nodes/radiology_nodes.py`
- Modify: `src/nodes/pathology_nodes.py`
- Modify: `src/nodes/planner.py`
- Modify: `src/nodes/node_utils.py`
- Modify: `src/nodes/knowledge_utils.py`
- Test: `tests/backend/test_patient_context_split_backend.py`

- [ ] **Step 1: Add shared resolver expectation to backend test**

Append this test to `tests/backend/test_patient_context_split_backend.py`:

```python
def test_sample_context_is_available_for_follow_up_payloads() -> None:
    prepared = build_graph_payload(
        chat_request=_ChatRequest("继续查看影像"),
        session_meta=SessionMeta(session_id="sess-test", thread_id="thread-test", patient_id=None),
        state_snapshot={
            "case_database_patient_id": "093",
            "findings": {"case_database_patient_id": "093"},
        },
    )

    assert prepared.payload["case_database_patient_id"] == "093"
    assert "registry_patient_id" not in prepared.payload
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_context_split_backend.py -q --basetemp=tmp\pytest-patient-context-split-task4-red
```

Expected: pass if Task 2 is complete. This confirms the input contract before migrating consumers.

- [ ] **Step 3: Add resolver helper in each migrated node**

In `src/nodes/radiology_nodes.py` and `src/nodes/pathology_nodes.py`, use this local pattern where patient/sample id is currently read from `state.current_patient_id`:

```python
def _resolve_case_database_patient_id(state: CRCAgentState, user_text: str | None = None) -> str | None:
    extracted = _extract_patient_id(user_text or "") if user_text else None
    findings = state.findings or {}
    candidate = (
        extracted
        or getattr(state, "case_database_patient_id", None)
        or findings.get("case_database_patient_id")
        or getattr(state, "current_patient_id", None)
        or findings.get("current_patient_id")
    )
    if candidate is None:
        return None
    text = str(candidate).strip()
    return text.zfill(3) if text.isdigit() else text
```

Then replace reads like:

```python
patient_id = state.current_patient_id
```

or:

```python
patient_id = state.current_patient_id or _extract_patient_id(user_text)
```

with:

```python
patient_id = _resolve_case_database_patient_id(state, user_text)
```

When these nodes return a selected sample id, return:

```python
"case_database_patient_id": patient_id,
```

and place in findings:

```python
findings_update["case_database_patient_id"] = patient_id
```

Do not return `current_patient_id` from imaging/pathology sample flows.

- [ ] **Step 4: Update planning and utility consumers**

In `src/nodes/planner.py`, replace database-oriented reads:

```python
patient_id = findings.get("db_query_patient_id") or findings.get("current_patient_id")
```

with:

```python
patient_id = (
    findings.get("db_query_patient_id")
    or findings.get("case_database_patient_id")
    or getattr(state, "case_database_patient_id", None)
    or findings.get("current_patient_id")
)
```

In `src/nodes/node_utils.py`, where state prompt context uses `current_patient_id`, prefer registry then sample:

```python
patient_id = (
    getattr(state, "registry_patient_id", None)
    or getattr(state, "case_database_patient_id", None)
    or getattr(state, "current_patient_id", None)
)
```

In `src/nodes/knowledge_utils.py`, change the current-patient display to:

```python
patient_id = (
    getattr(state, "registry_patient_id", None)
    or getattr(state, "case_database_patient_id", None)
    or state.current_patient_id
)
if not patient_id:
    return ""
status_parts = [f"**Current Context ID: {patient_id}**:"]
```

- [ ] **Step 5: Run backend regression set**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_context_split_backend.py tests\backend\test_chat_main_database_integration.py tests\backend\test_graph_service_streaming.py -q --basetemp=tmp\pytest-patient-context-split-task4
```

Expected: selected tests pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/nodes/radiology_nodes.py src/nodes/pathology_nodes.py src/nodes/planner.py src/nodes/node_utils.py src/nodes/knowledge_utils.py tests/backend/test_patient_context_split_backend.py
git commit -m "feat: route sample context to imaging and pathology"
```

## Task 5: Frontend Registry Binding Uses Only Registry Id

**Files:**
- Modify: `frontend/src/features/patient-registry/use-patient-registry.ts`
- Modify: `frontend/src/features/patient-registry/use-patient-registry.test.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Modify: `frontend/src/pages/workspace-page.test.tsx`
- Modify: `frontend/src/test/test-utils.tsx`
- Modify: `tests/frontend/test-utils.tsx`

- [ ] **Step 1: Update hook tests first**

In `frontend/src/features/patient-registry/use-patient-registry.test.tsx`, rename props from `currentPatientId` to `registryPatientId`:

```ts
function renderPatientRegistryHook(
  initialProps = { enabled: true, registryPatientId: 1 as number | null },
) {
  return renderHook(
    (props: { enabled: boolean; registryPatientId: number | null }) => usePatientRegistry(props),
    { initialProps },
  );
}
```

Add this test:

```ts
it("does not load registry data when only a case database sample id exists elsewhere", async () => {
  renderPatientRegistryHook({ enabled: true, registryPatientId: null });

  await waitFor(() => {
    expect(mockApiClient.getPatientRegistryDetail).not.toHaveBeenCalled();
    expect(mockApiClient.getPatientRecords).not.toHaveBeenCalled();
    expect(mockApiClient.getPatientRegistryAlerts).not.toHaveBeenCalled();
  });
});
```

Add this retry-prevention test:

```ts
it("does not retry the same missing registry id until the id changes", async () => {
  mockApiClient.getPatientRegistryDetail.mockRejectedValueOnce(new ApiClientError("missing", 404));
  const { rerender } = renderPatientRegistryHook({ enabled: true, registryPatientId: 93 });

  await waitFor(() => expect(mockApiClient.getPatientRegistryDetail).toHaveBeenCalledTimes(1));

  rerender({ enabled: true, registryPatientId: 93 });
  await waitFor(() => expect(mockApiClient.getPatientRegistryDetail).toHaveBeenCalledTimes(1));

  rerender({ enabled: true, registryPatientId: 7 });
  await waitFor(() => expect(mockApiClient.getPatientRegistryDetail).toHaveBeenCalledTimes(2));
});
```

- [ ] **Step 2: Run hook tests and verify failure**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/patient-registry/use-patient-registry.test.tsx --reporter=verbose
```

Expected: FAIL because the hook still accepts `currentPatientId`.

- [ ] **Step 3: Rename hook option and add missing-id guard**

In `frontend/src/features/patient-registry/use-patient-registry.ts`, change the signature to:

```ts
export function usePatientRegistry(options: { enabled: boolean; registryPatientId: number | null }) {
  const { enabled, registryPatientId } = options;
```

Add a ref:

```ts
  const missingRegistryIdRef = useRef<number | null>(null);
```

In `loadBoundPatient`, clear or set the ref:

```ts
    try {
      const [detailResponse, recordsResponse, alertsResponse] = await Promise.all([
        apiClient.getPatientRegistryDetail(patientId),
        apiClient.getPatientRecords(patientId),
        apiClient.getPatientRegistryAlerts(patientId),
      ]);
      missingRegistryIdRef.current = null;
```

In `catch`:

```ts
      if (error instanceof ApiClientError && error.status === 404) {
        missingRegistryIdRef.current = patientId;
      }
      setError(readErrorMessage(error));
```

In the effect, replace `currentPatientId` with:

```ts
    if (registryPatientId === null) {
      loadRequestIdRef.current += 1;
      missingRegistryIdRef.current = null;
      setBoundPatientDetail(null);
      setBoundPatientRecords([]);
      setBoundPatientAlerts([]);
      setIsLoadingBoundPatient(false);
      return;
    }
    if (missingRegistryIdRef.current === registryPatientId) {
      return;
    }
    setBoundPatientDetail(null);
    setBoundPatientRecords([]);
    setBoundPatientAlerts([]);
    void loadBoundPatient(registryPatientId);
  }, [enabled, registryPatientId]);
```

- [ ] **Step 4: Update workspace page usage**

In `frontend/src/pages/workspace-page.tsx`, replace:

```ts
  const doctorPatientId = readFiniteNumber(doctor.state.currentPatientId);
```

with:

```ts
  const registryPatientId = readFiniteNumber(doctor.state.registryPatientId);
```

Replace:

```ts
    currentPatientId: doctorPatientId,
```

with:

```ts
    registryPatientId,
```

For `DoctorSceneShell`, pass both ids:

```tsx
        registryPatientId={registryPatientId}
        caseDatabasePatientId={doctor.state.caseDatabasePatientId}
```

Keep the old `currentPatientId` prop only if needed during Task 6; otherwise remove it when Task 6 updates the component.

- [ ] **Step 5: Update workspace tests and helpers**

In `frontend/src/test/test-utils.tsx` and `tests/frontend/test-utils.tsx`, add defaults:

```ts
      case_database_patient_id: overrides.snapshot?.case_database_patient_id ?? null,
      registry_patient_id: overrides.snapshot?.registry_patient_id ?? patientId,
      current_patient_id: overrides.snapshot?.current_patient_id ?? patientId,
```

In `frontend/src/pages/workspace-page.test.tsx`, add a test where doctor session has only case sample id:

```ts
it("does not bind patient registry from case database sample context", async () => {
  mockSceneSessions.doctor.state = {
    ...mockSceneSessions.doctor.state,
    caseDatabasePatientId: "093",
    registryPatientId: null,
    currentPatientId: "093",
  };

  renderWorkspacePage();

  expect(mockUsePatientRegistry).toHaveBeenCalledWith(
    expect.objectContaining({ registryPatientId: null }),
  );
});
```

- [ ] **Step 6: Run frontend focused tests**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/patient-registry/use-patient-registry.test.tsx src/pages/workspace-page.test.tsx src/app/store/stream-reducer.test.ts --reporter=verbose
```

Expected: selected frontend tests pass.

- [ ] **Step 7: Commit Task 5**

```powershell
git add frontend/src/features/patient-registry/use-patient-registry.ts frontend/src/features/patient-registry/use-patient-registry.test.tsx frontend/src/pages/workspace-page.tsx frontend/src/pages/workspace-page.test.tsx frontend/src/test/test-utils.tsx tests/frontend/test-utils.tsx
git commit -m "feat: gate registry loads on explicit registry id"
```

## Task 6: Doctor UI Separates Registry Patient And Case Sample

**Files:**
- Modify: `frontend/src/features/doctor/doctor-scene-shell.tsx`
- Modify: `frontend/src/features/doctor/doctor-scene-shell.test.tsx`
- Modify: `frontend/src/features/doctor/doctor-database-view.tsx`
- Modify: `frontend/src/features/doctor/doctor-database-view.test.tsx`
- Modify: `frontend/src/features/patient-registry/registry-browser-pane.tsx`
- Modify: `frontend/src/features/patient-registry/registry-browser-pane.test.tsx`

- [ ] **Step 1: Update doctor scene component tests**

In `frontend/src/features/doctor/doctor-scene-shell.test.tsx`, update render helpers to pass:

```tsx
registryPatientId={1024}
caseDatabasePatientId="093"
```

Add assertions:

```ts
expect(screen.getByText("P-1024")).toBeInTheDocument();
expect(screen.getByText("093")).toBeInTheDocument();
expect(screen.getByText(/Registry patient/i)).toBeInTheDocument();
expect(screen.getByText(/Case sample/i)).toBeInTheDocument();
```

- [ ] **Step 2: Run doctor scene tests and verify failure**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/doctor/doctor-scene-shell.test.tsx --reporter=verbose
```

Expected: FAIL because the component has only `currentPatientId`.

- [ ] **Step 3: Rename props in doctor shell**

In `frontend/src/features/doctor/doctor-scene-shell.tsx`, replace prop fields:

```ts
  currentPatientId: number | null;
```

with:

```ts
  registryPatientId: number | null;
  caseDatabasePatientId: string | null;
```

Update the summary rows to:

```ts
const contextRows: Array<[string, string]> = [
  ["Registry patient:", registryPatientId !== null ? `P-${registryPatientId}` : "Unbound"],
  ["Case sample:", caseDatabasePatientId ?? "None"],
];
```

Pass `registryPatientId` to registry/browser components that compare current registry binding.

- [ ] **Step 4: Rename registry browser prop**

In `frontend/src/features/doctor/doctor-database-view.tsx` and `frontend/src/features/patient-registry/registry-browser-pane.tsx`, keep the UI semantics but rename props from `currentPatientId` to `registryPatientId` where the value means active registry binding.

Use this prop shape:

```ts
  registryPatientId: number | null;
```

Replace comparisons:

```ts
const isCurrent = item.patient_id === currentPatientId;
```

with:

```ts
const isCurrent = item.patient_id === registryPatientId;
```

- [ ] **Step 5: Run doctor and registry component tests**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/doctor/doctor-scene-shell.test.tsx src/features/doctor/doctor-database-view.test.tsx src/features/patient-registry/registry-browser-pane.test.tsx --reporter=verbose
```

Expected: selected tests pass.

- [ ] **Step 6: Commit Task 6**

```powershell
git add frontend/src/features/doctor/doctor-scene-shell.tsx frontend/src/features/doctor/doctor-scene-shell.test.tsx frontend/src/features/doctor/doctor-database-view.tsx frontend/src/features/doctor/doctor-database-view.test.tsx frontend/src/features/patient-registry/registry-browser-pane.tsx frontend/src/features/patient-registry/registry-browser-pane.test.tsx
git commit -m "feat: show separate registry and sample context"
```

## Task 7: End-To-End Regression And Cleanup

**Files:**
- Modify: `frontend/src/pages/workspace-page.test.tsx`
- Modify: `tests/backend/test_patient_context_split_backend.py`
- Inspect: all files changed in previous tasks

- [ ] **Step 1: Add final frontend leak-path test**

In `frontend/src/pages/workspace-page.test.tsx`, add:

```ts
it("does not request registry detail for legacy currentPatientId when registryPatientId is null", () => {
  mockSceneSessions.doctor.state = {
    ...mockSceneSessions.doctor.state,
    caseDatabasePatientId: "093",
    registryPatientId: null,
    currentPatientId: "093",
  };

  renderWorkspacePage();

  expect(mockUsePatientRegistry).toHaveBeenCalledWith(
    expect.objectContaining({ registryPatientId: null }),
  );
});
```

- [ ] **Step 2: Add final backend leak-path test**

In `tests/backend/test_patient_context_split_backend.py`, add:

```python
def test_legacy_current_patient_id_never_becomes_registry_id_in_snapshot_or_payload() -> None:
    meta = SessionMeta(session_id="sess-test", thread_id="thread-test", patient_id=None)
    snapshot = build_recovery_snapshot(meta, {"current_patient_id": "093", "findings": {}})

    assert snapshot.case_database_patient_id == "093"
    assert snapshot.registry_patient_id is None

    prepared = build_graph_payload(
        chat_request=_ChatRequest("continue"),
        session_meta=meta,
        state_snapshot=snapshot.model_dump(),
    )

    assert prepared.payload["case_database_patient_id"] == "093"
    assert "registry_patient_id" not in prepared.payload
    assert "current_patient_id" not in prepared.payload
```

- [ ] **Step 3: Run backend regression suite**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_context_split_backend.py tests\backend\test_chat_main_database_integration.py tests\backend\test_patient_context_resolver.py tests\backend\test_graph_service_streaming.py tests\backend\test_state_tools_executor_regressions.py -q --basetemp=tmp\pytest-patient-context-split-final
```

Expected: all selected backend tests pass.

- [ ] **Step 4: Run frontend regression suite**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/store/stream-reducer.test.ts src/pages/workspace-page.test.tsx src/features/patient-registry/use-patient-registry.test.tsx src/features/doctor/doctor-scene-shell.test.tsx src/features/doctor/doctor-database-view.test.tsx src/features/patient-registry/registry-browser-pane.test.tsx --reporter=verbose
```

Expected: all selected frontend tests pass.

- [ ] **Step 5: Run frontend build**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: TypeScript and Vite build pass.

- [ ] **Step 6: Search for remaining unsafe registry consumers**

Run:

```powershell
rg -n "usePatientRegistry\\(|currentPatientId|current_patient_id|registryPatientId|registry_patient_id|caseDatabasePatientId|case_database_patient_id" frontend\src src backend tests
```

Expected:

- `usePatientRegistry` call sites pass `registryPatientId`.
- `current_patient_id` remains only in compatibility reads, tests, or deprecated snapshot fields.
- database/imaging/pathology sample flows use `case_database_patient_id`.

- [ ] **Step 7: Run diff check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Existing line-ending warnings can be noted but must not hide real whitespace errors.

- [ ] **Step 8: Commit final cleanup**

```powershell
git add frontend/src/pages/workspace-page.test.tsx tests/backend/test_patient_context_split_backend.py
git commit -m "test: cover split patient context leak path"
```

## Final Verification Checklist

- [ ] `093` case database queries do not trigger patient registry requests.
- [ ] Explicit registry binding still triggers patient registry requests.
- [ ] Case database cards still render using sample id.
- [ ] Imaging/pathology follow-up queries can reuse `case_database_patient_id`.
- [ ] Old `current_patient_id="093"` snapshots are treated as sample context only.
- [ ] Doctor UI shows registry patient and case sample as separate labels.
- [ ] Backend selected regression tests pass.
- [ ] Frontend selected regression tests pass.
- [ ] Frontend build passes.

## Suggested Execution Order

Execute tasks in order. Do not begin Task 3 until Task 2 is green, because Task 3 depends on the new snapshot/payload contract. Do not begin Task 5 until Task 1 is green, because frontend session state needs the new fields before registry fetch gating can be changed safely.
