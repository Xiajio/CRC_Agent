# Scene-Driven Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 1 of the scene-driven workspace so patient and doctor scenes use separate sessions, patient uploads persist into a shared SQLite patient registry, and doctor scene can bind/query those patients from the same single-page workspace.

**Architecture:** Add a SQLite-backed patient registry behind new backend services and routes, make sessions scene-aware with patient binding, split graph runtime into independent patient/doctor services, and refactor the React workspace into a scene shell with separate session bootstrap plus an always-on patient-registry panel for doctor mode. Keep historical case search on the existing Excel-backed database service and integrate it through scoped frontend controllers instead of merging the data models.

**Tech Stack:** Python, FastAPI, SQLite, LangGraph, React, TypeScript, Vitest, Vite, pytest

---

**Repository note:** This workspace is not currently a git repository, so commit steps are written as checkpoints but cannot be executed until repository metadata is restored.

## File Structure

### Backend files to create

- `D:\YiZhu_Agnet\LangG\backend\api\services\patient_registry_service.py`
  - SQLite schema bootstrap, draft-patient creation, patient snapshot merge, asset dedup, record read APIs, and doctor-side summary loading.
- `D:\YiZhu_Agnet\LangG\backend\api\schemas\patient_registry.py`
  - Request/response models for recent patients, patient search, patient detail, patient records, and doctor-session bind payloads.
- `D:\YiZhu_Agnet\LangG\backend\api\routes\patient_registry.py`
  - Patient-registry list/search/detail/records endpoints for the doctor scene.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_service.py`
  - Service-level coverage for draft creation, merge rules, asset dedup, and patient summary reads.
- `D:\YiZhu_Agnet\LangG\tests\backend\conftest.py`
  - Shared backend test harness for `TestClient` app wiring, registry/runtime stubs, and reusable graph/upload fakes once route and graph tests start sharing helpers.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_scene_sessions.py`
  - Session route coverage for scene-aware creation, doctor bind semantics, and response payload shape.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_routes.py`
  - API coverage for recent-patient, search, detail, and record-list routes.
- `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`
  - Scene bootstrap, scene switching, bind-patient, recent-patient panel, and scoped workbench coverage.

### Backend files to modify

- `D:\YiZhu_Agnet\LangG\backend\app.py`
  - Initialize patient registry service, dual graph services, and runtime wiring for new routes.
- `D:\YiZhu_Agnet\LangG\backend\api\adapters\state_snapshot.py`
  - Expose scene and patient id in session snapshots.
- `D:\YiZhu_Agnet\LangG\backend\api\routes\sessions.py`
  - Accept `scene` on create, add doctor bind endpoint, and return scene-aware session responses.
- `D:\YiZhu_Agnet\LangG\backend\api\routes\chat.py`
  - Resolve the correct graph service from the runtime scene router.
- `D:\YiZhu_Agnet\LangG\backend\api\routes\uploads.py`
  - Pass registry service into the upload rewrite path.
- `D:\YiZhu_Agnet\LangG\backend\api\schemas\responses.py`
  - Extend `SessionResponse` and snapshot metadata with `scene` and `patient_id`.
- `D:\YiZhu_Agnet\LangG\backend\api\services\graph_factory.py`
  - Cache separate patient and doctor compiled graphs.
- `D:\YiZhu_Agnet\LangG\backend\api\services\graph_service.py`
  - Define `PatientGraphService`, `DoctorGraphService`, and a thin `SceneGraphRouter`.
- `D:\YiZhu_Agnet\LangG\backend\api\services\session_store.py`
  - Add `scene` and `patient_id` to `SessionMeta`, plus helpers for scene-aware create/bind operations.
- `D:\YiZhu_Agnet\LangG\backend\api\services\upload_service.py`
  - Replace session-first JSON injection with registry-first write plus lightweight context reference.
- `D:\YiZhu_Agnet\LangG\src\graph_builder.py`
  - Add `build_patient_graph()` and `build_doctor_graph()` without physically splitting files.

### Frontend files to create

- `D:\YiZhu_Agnet\LangG\frontend\src\features\workspace\use-scene-sessions.ts`
  - Dual-session bootstrap, persistence, recovery, active-scene switching, and stream abort coordination.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-patient-registry.ts`
  - Recent-patient loading, patient search/detail fetch, and doctor-session bind actions.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\recent-patients-panel.tsx`
  - Always-on doctor-side recent-patient entry panel.
- `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`
  - Shared frontend test harness for API client stubs, response builders, render helpers, providers, and stream-abort utilities used by scene-shell tests.

### Frontend files to modify

- `D:\YiZhu_Agnet\LangG\frontend\src\app\api\client.ts`
  - Add scene-aware session create, bind-patient, and patient-registry API methods.
- `D:\YiZhu_Agnet\LangG\frontend\src\app\api\types.ts`
  - Add scene/patient session fields and patient-registry response models.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\database\use-database-workbench.ts`
  - Support `scope` and re-bootstrap on scope changes.
- `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
  - Convert the current page into the scene shell and compose patient/doctor views from separate state sources.

## Test Harness Preconditions

- No existing `tests/backend/conftest.py` or `tests/frontend/test-utils.tsx` were found in this workspace.
- Backend pytest commands should use `D:\anaconda3\envs\LangG\python.exe -m pytest ...`; the system Python does not have the project FastAPI stack.
- On this machine, OS temp directories may be permission-restricted. If `tmp_path` setup fails, switch the affected test to a workspace-local directory under `runtime/`.
- Task 2 may stay self-contained with a local `_build_client()` helper, but by Task 5 shared backend helpers should move into `tests/backend/conftest.py`.
- Before Task 6, create `tests/frontend/test-utils.tsx` so `buildApiClientStub`, `buildStreamingApiClientStub`, `buildAppWrapper`, `renderWorkspaceWithSceneSessions`, `makeSessionResponse`, `makeDatabaseStatsResponse`, and `makeDatabaseSearchResponse` are not redefined per test file.
- Before relying on a focused Vitest path, verify discovery once with:

```bash
cmd /c npm --prefix frontend run test -- --run --reporter=verbose
```

If `tests/frontend/workspace-scenes.test.tsx` is not discovered, move the file to the frontend-recognized test root or update the Vitest include pattern before continuing.

## Task 1: Build the SQLite Patient Registry Service

**Files:**
- Create: `D:\YiZhu_Agnet\LangG\backend\api\services\patient_registry_service.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_service.py`

- [ ] **Step 1: Write a failing service test for draft creation**

```python
from pathlib import Path

from backend.api.services.patient_registry_service import PatientRegistryService


def test_create_draft_patient_persists_created_by_session_id(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")

    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    detail = service.get_patient_detail(patient_id)
    assert detail["patient_id"] == patient_id
    assert detail["status"] == "draft"
    assert detail["created_by_session_id"] == "sess_patient_1"
```

- [ ] **Step 2: Add a failing test for non-empty snapshot merge**

```python
def test_write_record_merges_non_empty_snapshot_fields(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "ct-1.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-1",
            "storage_path": "runtime/assets/ct-1.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={
            "tumor_location": "rectum",
            "mmr_status": "not_provided",
            "clinical_stage": "cT3N1M0",
        },
        record_payload={"document_type": "ct_report"},
        summary_text="cT3N1M0 rectal lesion",
        record_type="medical_card",
    )
    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "pathology-1.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-2",
            "storage_path": "runtime/assets/pathology-1.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={
            "tumor_location": "not_provided",
            "mmr_status": "dMMR",
        },
        record_payload={"document_type": "pathology_report"},
        summary_text="dMMR confirmed",
        record_type="medical_card",
    )

    detail = service.get_patient_detail(patient_id)
    assert detail["tumor_location"] == "rectum"
    assert detail["mmr_status"] == "dMMR"
    assert detail["clinical_stage"] == "cT3N1M0"
```

- [ ] **Step 3: Add a failing safety test proving `treatment_draft` never lands in the patient snapshot**

```python
def test_write_record_does_not_propagate_treatment_draft_into_patient_snapshot(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_1")

    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "treatment.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-treatment",
            "storage_path": "runtime/assets/treatment.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={
            "tumor_location": "rectum",
            "treatment_draft": [{"step": 1, "name": "手术"}],
        },
        record_payload={
            "document_type": "report",
            "treatment_draft": [{"step": 1, "name": "手术"}],
        },
        summary_text="rectal lesion",
        record_type="medical_card",
    )

    detail = service.get_patient_detail(patient_id)
    assert "treatment_draft" not in detail
    assert detail["tumor_location"] == "rectum"
```

- [ ] **Step 4: Add a failing test for asset dedup by `patient_id + sha256`**

```python
def test_write_record_reuses_existing_asset_for_same_patient_and_sha256(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    first = service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "report.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-1",
            "storage_path": "runtime/assets/report.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"tumor_location": "rectum"},
        record_payload={"document_type": "report"},
        summary_text="rectal lesion",
        record_type="medical_card",
    )
    second = service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "report.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-1",
            "storage_path": "runtime/assets/report.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"tumor_location": "rectum"},
        record_payload={"document_type": "report"},
        summary_text="rectal lesion",
        record_type="medical_card",
    )

    assert second["asset_id"] == first["asset_id"]
    assert second["reused"] is True
```

- [ ] **Step 5: Run the focused backend service tests and verify failure**

Run:

```bash
pytest tests/backend/test_patient_registry_service.py -v
```

Expected: FAIL because the registry service does not exist yet.

- [ ] **Step 6: Implement `PatientRegistryService` with schema bootstrap and focused write/read helpers**

Create a single service with:

- database initialization for `patients`, `patient_assets`, and `patient_records`
- `create_draft_patient(created_by_session_id: str) -> int`
- `write_medical_card_record(...) -> dict`
- `get_patient_detail(patient_id: int) -> dict`
- `list_patient_records(patient_id: int) -> list[dict]`
- `list_recent_patients(limit: int) -> list[dict]`
- `search_patients(...) -> dict`
- `get_patient_summary_message(patient_id: int) -> HumanMessage | None`

Snapshot-field extraction rule:

- whitelist only patient snapshot fields that are safe to flatten into `patients`
- keep `treatment_draft` only inside `patient_records.normalized_payload_json`

Core merge helper:

```python
EMPTY_VALUES = {"not_provided", "Unknown", "pending_evaluation", None, ""}

def _merge_snapshot(existing: dict[str, object], incoming: dict[str, object]) -> dict[str, object]:
    merged = dict(existing)
    for field, value in incoming.items():
        if merged.get(field) in EMPTY_VALUES and value not in EMPTY_VALUES:
            merged[field] = value
    return merged
```

- [ ] **Step 7: Re-run the focused backend service tests**

Run:

```bash
pytest tests/backend/test_patient_registry_service.py -v
```

Expected: PASS

- [ ] **Step 8: Checkpoint**

```bash
git add backend/api/services/patient_registry_service.py tests/backend/test_patient_registry_service.py
git commit -m "feat: add sqlite patient registry service"
```

## Task 2: Make Sessions Scene-Aware and Draft-Patient-Aware

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\session_store.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\schemas\responses.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\adapters\state_snapshot.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\routes\sessions.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\app.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_scene_sessions.py`

- [ ] **Step 1: Add failing route tests for patient-scene create and doctor bind semantics**

```python
from fastapi.testclient import TestClient


def test_create_patient_scene_returns_sqlite_backed_patient_id(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"scene": "patient"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["scene"] == "patient"
    assert isinstance(payload["patient_id"], int)
```

- [ ] **Step 2: Add a failing doctor-scene test for null patient id and bind semantics**

```python
def test_create_doctor_scene_returns_null_patient_id(client: TestClient) -> None:
    response = client.post("/api/sessions", json={"scene": "doctor"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["scene"] == "doctor"
    assert payload["patient_id"] is None


def test_bind_patient_rejects_rebinding_without_reset(client: TestClient) -> None:
    doctor = client.post("/api/sessions", json={"scene": "doctor"}).json()
    first = client.patch(
        f"/api/sessions/{doctor['session_id']}",
        json={"patient_id": 101},
    )
    second = client.patch(
        f"/api/sessions/{doctor['session_id']}",
        json={"patient_id": 102},
    )

    assert first.status_code == 200
    assert second.status_code == 409
```

- [ ] **Step 3: Run the focused session route tests and verify failure**

Run:

```bash
pytest tests/backend/test_scene_sessions.py -v
```

Expected: FAIL because sessions are not scene-aware and `PATCH /api/sessions/{id}` does not exist.

- [ ] **Step 4: Implement scene-aware session metadata, schema changes, and route wiring**

Required behavior:

- `SessionMeta` gains `scene` and `patient_id`
- `InMemorySessionStore.create_session(scene: str = "doctor", patient_id: int | None = None)`
- `InMemorySessionStore.bind_patient(session_id: str, patient_id: int) -> SessionMeta`
- `SessionResponse` gains `scene` and `patient_id`
- patient scene create calls `PatientRegistryService.create_draft_patient(...)`
- doctor scene create leaves `patient_id = null`
- `PATCH /api/sessions/{id}` is bind-only for doctor sessions

Compatibility rule:

- keep `create_session()` default parameters so existing direct callers in graph/upload tests do not break during the refactor

- [ ] **Step 5: Re-run the focused session route tests**

Run:

```bash
pytest tests/backend/test_scene_sessions.py -v
```

Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add backend/api/services/session_store.py backend/api/schemas/responses.py backend/api/adapters/state_snapshot.py backend/api/routes/sessions.py backend/app.py tests/backend/test_scene_sessions.py
git commit -m "feat: add scene-aware sessions and doctor bind semantics"
```

## Task 3: Rewrite Upload Persistence to Be Registry-First

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\upload_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\routes\uploads.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_service.py`

- [ ] **Step 1: Add a failing upload test for lightweight context references and registry writeback**

```python
from pathlib import Path

from langchain_core.messages import HumanMessage


def test_store_session_upload_writes_registry_and_enqueues_lightweight_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from backend.api.services.session_store import InMemorySessionStore
    from backend.api.services.upload_service import store_session_upload

    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="patient", patient_id=12)

    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: FakeMedicalVisualizationCard(
            patient_summary={"chief_complaint": "rectal bleeding"},
            staging_block={"clinical_stage": "cT3N1M0", "t_stage": "T3", "n_stage": "N1", "m_stage": "M0"},
            key_findings=[{"finding": "rectal wall thickening"}],
        ),
    )

    response = store_session_upload(
        session_store=session_store,
        patient_registry=build_test_registry(tmp_path / "patient_registry.db"),
        assets_root=tmp_path / "assets",
        session_id=meta.session_id,
        filename="report.pdf",
        content_type="application/pdf",
        file_bytes=b"fake",
        reserved_run_id="run_upload_1",
    )

    assert response["derived"]["record_id"] is not None
    pending = session_store.drain_context_messages(meta.session_id)
    assert len(pending) == 1
    assert isinstance(pending[0], HumanMessage)
    assert "record_id=" in pending[0].content
    assert "rectal wall thickening" in pending[0].content
```

Test helper rule:

- define `FakeMedicalVisualizationCard` with `data.patient_summary`, `data.staging_block`, and `data.key_findings`
- define `build_test_registry(db_path)` to return an initialized `PatientRegistryService`
- keep both helpers in the test module initially; move them into `tests/backend/conftest.py` once Task 5 introduces more shared backend fakes

- [ ] **Step 2: Run the focused upload test and verify failure**

Run:

```bash
pytest tests/backend/test_patient_registry_service.py -v
```

Expected: FAIL because uploads still inject full JSON into session context and do not write through the registry.

- [ ] **Step 3: Replace session-first upload handling with mapper plus registry writer**

Implementation rules:

- read `session_meta.patient_id` and reject patient-scene uploads without one
- keep `convert_uploaded_file()` as the normalization step
- add a small pure mapper inside `upload_service.py` or `patient_registry_service.py`:

```python
def flatten_medical_card(card: MedicalVisualizationCard) -> tuple[dict[str, object], dict[str, object], str]:
    ...
```

- call `patient_registry.write_medical_card_record(...)`
- enqueue only a lightweight `HumanMessage` reference
- downgrade `uploaded_assets` to a recent-upload view containing:
  - `asset_id`
  - `record_id`
  - `patient_id`
  - `filename`
  - `reused`

- [ ] **Step 4: Re-run the focused upload test**

Run:

```bash
pytest tests/backend/test_patient_registry_service.py -v
```

Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add backend/api/services/upload_service.py backend/api/routes/uploads.py tests/backend/test_patient_registry_service.py
git commit -m "feat: rewrite uploads to registry-first persistence"
```

## Task 4: Split the Graph Runtime into Patient and Doctor Services

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\src\graph_builder.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\graph_factory.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\graph_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\routes\chat.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\app.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`

- [ ] **Step 1: Add failing graph-service tests for scene routing and patient context-finalizer disablement**

```python
def test_scene_router_returns_patient_service_for_patient_session():
    session_store = InMemorySessionStore()
    session_store.create_session(scene="patient", patient_id=10)

    router = SceneGraphRouter(
        patient_service=PatientGraphService(compiled_graph=FakeGraph(), session_store=session_store),
        doctor_service=DoctorGraphService(compiled_graph=FakeGraph(), session_store=session_store),
        session_store=session_store,
    )

    service = router.for_session(next(iter(session_store._sessions.keys())))
    assert isinstance(service, PatientGraphService)


def test_patient_graph_service_never_emits_context_maintenance_running():
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="patient", patient_id=10)
    service = PatientGraphService(compiled_graph=FakeStreamingGraph(), session_store=session_store)

    events = collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("hello")))
    assert all(event["type"] != "context.maintenance" for event in events)
```

- [ ] **Step 2: Run the focused graph-service tests and verify failure**

Run:

```bash
pytest tests/backend/test_graph_service_streaming.py -v
```

Expected: FAIL because only one graph service and one graph factory entry point exist today.

- [ ] **Step 3: Implement dual graph builders, dual graph caches, and a thin scene router**

Required changes:

- `src.graph_builder` exports:
  - `build_patient_graph(settings)`
  - `build_doctor_graph(settings)`
- `graph_factory.py` exports:
  - `get_patient_graph(...)`
  - `get_doctor_graph(...)`
- `graph_service.py` contains:
  - `PatientGraphService`
  - `DoctorGraphService`
  - `SceneGraphRouter`
- `backend.app` initializes:
  - patient graph service with `context_finalizer=None`
  - doctor graph service with the existing context finalizer
- `chat.py` resolves the service via `request.app.state.runtime.scene_router.for_session(session_id)`

Graph composition rule:

- `build_patient_graph(settings)` keeps the patient-scene path only:
  - intent / entry routing
  - clinical entry resolver
  - outpatient triage
  - knowledge / general chat path
- `build_doctor_graph(settings)` keeps the current doctor-heavy path:
  - database / historical case path
  - radiology path
  - pathology path
  - decision / citation path

Test helper rule:

- define `FakeGraph`, `FakeStreamingGraph`, `make_chat_request`, and `collect_sse_events` inside `tests/backend/test_graph_service_streaming.py` first
- once both Task 4 and Task 5 need them, consolidate them into `tests/backend/conftest.py`

- [ ] **Step 4: Re-run the focused graph-service tests**

Run:

```bash
pytest tests/backend/test_graph_service_streaming.py -v
```

Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/graph_builder.py backend/api/services/graph_factory.py backend/api/services/graph_service.py backend/api/routes/chat.py backend/app.py tests/backend/test_graph_service_streaming.py
git commit -m "feat: split patient and doctor graph runtime"
```

## Task 5: Add Doctor Lazy Patient Injection and Patient Registry Read Routes

**Files:**
- Create: `D:\YiZhu_Agnet\LangG\backend\api\schemas\patient_registry.py`
- Create: `D:\YiZhu_Agnet\LangG\backend\api\routes\patient_registry.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\patient_registry_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\graph_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\app.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_routes.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`

- [ ] **Step 1: Add failing tests for recent-patient routes and lazy doctor injection**

```python
def test_recent_patients_route_returns_registry_rows(client: TestClient) -> None:
    service = client.app.state.runtime.patient_registry_service
    first = service.create_draft_patient(created_by_session_id="sess_a")
    second = service.create_draft_patient(created_by_session_id="sess_b")

    response = client.get("/api/patient-registry/patients/recent?limit=5")

    payload = response.json()
    assert response.status_code == 200
    assert {item["patient_id"] for item in payload["items"]} == {first, second}


def test_doctor_graph_service_injects_patient_summary_when_patient_is_newly_bound():
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    registry = FakePatientRegistry(summary_message=HumanMessage(content="patient_id=33 summary"))
    service = DoctorGraphService(
        compiled_graph=FakeStreamingGraph(),
        session_store=session_store,
        patient_registry=registry,
    )

    list(collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("next"))))

    pending = session_store.drain_context_messages(meta.session_id)
    assert len(pending) == 1
    assert "patient_id=33" in pending[0].content


def test_doctor_graph_service_does_not_reinject_patient_summary_when_already_bound():
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=33)
    registry = FakePatientRegistry(summary_message=HumanMessage(content="patient_id=33 summary"))
    service = DoctorGraphService(
        compiled_graph=FakeStreamingGraph(),
        session_store=session_store,
        patient_registry=registry,
    )

    list(collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("hello"))))
    session_store.drain_context_messages(meta.session_id)
    session_store.merge_context_state(meta.session_id, {"bound_patient_id": 33})

    list(collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("next"))))
    pending = session_store.drain_context_messages(meta.session_id)
    assert len(pending) == 0
```

Test helper rule:

- define `FakePatientRegistry` with a `get_patient_summary_message(patient_id)` method in `tests/backend/test_graph_service_streaming.py`
- migrate it to `tests/backend/conftest.py` when Task 5 merges backend graph and route test helpers

- [ ] **Step 2: Run the focused route and graph tests and verify failure**

Run:

```bash
pytest tests/backend/test_patient_registry_routes.py tests/backend/test_graph_service_streaming.py -v
```

Expected: FAIL because the patient-registry routes do not exist and doctor lazy injection is not implemented.

- [ ] **Step 3: Implement read routes and doctor lazy-injection logic**

Route surface:

- `GET /api/patient-registry/patients/recent`
- `POST /api/patient-registry/patients/search`
- `GET /api/patient-registry/patients/{patient_id}`
- `GET /api/patient-registry/patients/{patient_id}/records`

Doctor graph logic:

- in `DoctorGraphService.stream_turn()`, before `build_graph_payload(...)`:
  - read `meta.patient_id`
  - compare against `context_state.bound_patient_id`
  - if changed or snapshot stale, load a summary `HumanMessage`
  - enqueue it, then proceed
  - update `context_state` with `bound_patient_id` and `bound_patient_snapshot_version`

- [ ] **Step 4: Re-run the focused route and graph tests**

Run:

```bash
pytest tests/backend/test_patient_registry_routes.py tests/backend/test_graph_service_streaming.py -v
```

Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add backend/api/schemas/patient_registry.py backend/api/routes/patient_registry.py backend/api/services/patient_registry_service.py backend/api/services/graph_service.py backend/app.py tests/backend/test_patient_registry_routes.py tests/backend/test_graph_service_streaming.py
git commit -m "feat: add patient registry routes and doctor lazy patient context"
```

## Task 6: Add Scene-Aware Frontend API Types and Session Bootstrap

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\app\api\types.ts`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\app\api\client.ts`
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\workspace\use-scene-sessions.ts`
- Create: `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`
- Test: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Add failing frontend tests for dual-session bootstrap and 404 recovery**

```tsx
test("boots patient and doctor sessions and recreates both when either stored id is stale", async () => {
  const getSession = vi
    .fn()
    .mockRejectedValueOnce(new ApiClientError(404, "missing", { detail: "missing" }))
    .mockResolvedValueOnce(makeSessionResponse({ scene: "patient", patient_id: 11 }))
    .mockResolvedValueOnce(makeSessionResponse({ scene: "doctor", patient_id: null }));

  const apiClient = {
    getSession,
    createSession: vi
      .fn()
      .mockResolvedValueOnce(makeSessionResponse({ scene: "patient", patient_id: 11 }))
      .mockResolvedValueOnce(makeSessionResponse({ scene: "doctor", patient_id: null })),
  } as unknown as ApiClient;

  const { result } = renderHook(() => useSceneSessions({ apiClient }), {
    wrapper: buildAppWrapper(apiClient),
  });

  await waitFor(() => expect(result.current.bootstrapStatus).toBe("ready"));
  expect(apiClient.createSession).toHaveBeenCalledTimes(2);
});
```

Test helper rule:

- `buildAppWrapper(apiClient)` must live in `tests/frontend/test-utils.tsx`
- `makeSessionResponse(...)` must live in `tests/frontend/test-utils.tsx`
- keep provider wiring there instead of redefining it in every scene-shell test file

- [ ] **Step 2: Run the focused frontend test and verify failure**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run --reporter=verbose
```

Expected: confirm the planned test file path is discovered by Vitest; if not, move the test file or adjust config before the focused run.

- [ ] **Step 3: Run the focused frontend test and verify failure**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run ../tests/frontend/workspace-scenes.test.tsx
```

Expected: FAIL because scene-aware API types and `useSceneSessions()` do not exist.

- [ ] **Step 4: Extend API types/client and implement `useSceneSessions()`**

Required type changes:

- `SessionResponse.scene`
- `SessionResponse.patient_id`
- patient-registry list/search/detail models

Required client methods:

- `createSession(scene)`
- `bindPatient(sessionId, patientId)`
- `getRecentPatients(limit?)`
- `searchPatientRegistry(request)`
- `getPatientRegistryDetail(patientId)`
- `getPatientRecords(patientId)`

Required `useSceneSessions()` behavior:

- persist two keys:
  - `langg.workspace.patient-session-id`
  - `langg.workspace.doctor-session-id`
- bootstrap both sessions together
- if either persisted session returns `404`, clear both and recreate both
- own `activeScene`
- abort active stream on scene switch

- [ ] **Step 5: Re-run the focused frontend test**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run ../tests/frontend/workspace-scenes.test.tsx
```

Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/features/workspace/use-scene-sessions.ts tests/frontend/workspace-scenes.test.tsx
git commit -m "feat: add scene-aware frontend sessions"
```

## Task 7: Build Doctor Patient-Registry Hooks and the Scene Shell

**Files:**
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-patient-registry.ts`
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\recent-patients-panel.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`
- Test: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Add failing workspace tests for scene switching, recent patients, and bind-patient flow**

```tsx
test("doctor scene loads recent patients without waiting for findings and binds selected patient", async () => {
  const apiClient = buildApiClientStub({
    getRecentPatients: vi.fn().mockResolvedValue({
      items: [
        { patient_id: 21, status: "draft", updated_at: "2026-04-16T10:00:00" },
      ],
    }),
    bindPatient: vi.fn().mockResolvedValue(
      makeSessionResponse({ scene: "doctor", patient_id: 21 }),
    ),
  });

  renderWorkspaceWithSceneSessions(apiClient);
  const tabs = await screen.findAllByRole("tab");
  fireEvent.click(tabs[1]);

  expect(await screen.findByText(/#21/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /#21/ }));

  await waitFor(() => expect(apiClient.bindPatient).toHaveBeenCalledWith(expect.any(String), 21));
});


test("switching scenes aborts the active stream", async () => {
  const abortSpy = vi.fn();
  renderWorkspaceWithSceneSessions(buildStreamingApiClientStub({ onAbort: abortSpy }));

  fireEvent.click(screen.getByRole("button", { name: /send/i }));
  const tabs = await screen.findAllByRole("tab");
  fireEvent.click(tabs[1]);

  expect(abortSpy).toHaveBeenCalledTimes(1);
});
```

- [ ] **Step 2: Run the focused workspace tests and verify failure**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run ../tests/frontend/workspace-scenes.test.tsx
```

Expected: FAIL because the workspace still assumes one session and has no patient-registry hook or panel.

- [ ] **Step 3: Record the current wider frontend baseline before touching `workspace-page.tsx`**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run
```

Expected: capture the current baseline so the scene-shell refactor does not silently widen regressions.

- [ ] **Step 4: Implement `usePatientRegistry()` and convert `workspace-page.tsx` into the scene shell**

Required `usePatientRegistry()` outputs:

- `recentPatients`
- `selectedPatientId`
- `detail`
- `records`
- `isLoadingRecent`
- `isBinding`
- `loadRecentPatients()`
- `bindPatient(patientId)`
- `loadPatientDetail(patientId)`

Required workspace shell behavior:

- render a scene switcher for `patient` and `doctor`
- maintain separate submit/upload handlers by scene session id
- keep patient uploads on patient session only
- show `RecentPatientsPanel` in doctor scene immediately on entry
- keep doctor recent-patient panel independent of `findings`
- keep shared frontend helpers in `tests/frontend/test-utils.tsx`, including:
  - `buildApiClientStub`
  - `buildStreamingApiClientStub`
  - `makeSessionResponse`
  - `makeDatabaseStatsResponse`
  - `makeDatabaseSearchResponse`
  - `renderWorkspaceWithSceneSessions`

- [ ] **Step 5: Re-run the focused workspace tests**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run ../tests/frontend/workspace-scenes.test.tsx
```

Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/features/patient-registry/use-patient-registry.ts frontend/src/features/patient-registry/recent-patients-panel.tsx frontend/src/pages/workspace-page.tsx tests/frontend/workspace-scenes.test.tsx
git commit -m "feat: add scene shell and doctor patient registry panel"
```

## Task 8: Scope the Database Workbench and Finish the Single-Page Doctor View

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\database\use-database-workbench.ts`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
- Test: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Add a failing test for workbench scope re-bootstrap in doctor scene**

```tsx
test("changing workbench scope reboots the doctor data source", async () => {
  const apiClient = buildApiClientStub({
    getDatabaseStats: vi.fn().mockResolvedValue(makeDatabaseStatsResponse()),
    searchDatabaseCases: vi.fn().mockResolvedValue(makeDatabaseSearchResponse()),
    searchPatientRegistry: vi.fn().mockResolvedValue({ items: [], total: 0 }),
  });

  renderWorkspaceWithSceneSessions(apiClient, { activeScene: "doctor" });

  fireEvent.click(await screen.findByRole("button", { name: /historical/i }));
  await waitFor(() => expect(apiClient.getDatabaseStats).toHaveBeenCalledTimes(1));

  fireEvent.click(screen.getByRole("button", { name: /patient registry/i }));
  await waitFor(() => expect(apiClient.searchPatientRegistry).toHaveBeenCalledTimes(1));
});
```

- [ ] **Step 2: Run the focused workspace tests and verify failure**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run ../tests/frontend/workspace-scenes.test.tsx
```

Expected: FAIL because `useDatabaseWorkbench()` does not accept `scope` and doctor view is not yet split cleanly.

- [ ] **Step 3: Add `scope` to the shared workbench controller and finish doctor-only composition**

Required behavior:

- `useDatabaseWorkbench({ scope, autoBootstrap, bootstrapKey, ... })`
- include `scope` in the bootstrap lifecycle key
- route API calls by `scope`
- in `workspace-page.tsx`, doctor scene shows:
  - conversation
  - recent patients panel
  - bound patient summary
  - historical case workbench
  - cards / roadmap / execution plan
- patient scene excludes doctor-only registry/workbench chrome

- [ ] **Step 4: Re-run the focused workspace tests**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run ../tests/frontend/workspace-scenes.test.tsx
```

Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add frontend/src/features/database/use-database-workbench.ts frontend/src/pages/workspace-page.tsx tests/frontend/workspace-scenes.test.tsx
git commit -m "feat: scope doctor workbench for historical and patient registry data"
```
## Task 9: Regression Sweep and Manual Verification

**Files:**
- Modify as needed based on failures
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_service.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_scene_sessions.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_routes.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`
- Test: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Run the focused backend regression suite**

Run:

```bash
pytest tests/backend/test_patient_registry_service.py tests/backend/test_scene_sessions.py tests/backend/test_patient_registry_routes.py tests/backend/test_graph_service_streaming.py -v
```

Expected: PASS

- [ ] **Step 2: Run the focused frontend regression suite**

Run:

```bash
cmd /c npm --prefix frontend run test -- --run ../tests/frontend/workspace-scenes.test.tsx
```

Expected: PASS

- [ ] **Step 3: Manually verify the critical flows**

Verify:

1. Open the workspace and confirm patient and doctor tabs both bootstrap.
2. In patient scene, upload a report and confirm the upload succeeds.
3. Switch to doctor scene, open recent patients, bind the uploaded patient, and verify the summary appears.
4. Send a doctor-scene message after binding and confirm the doctor graph receives patient context without patient chat history.
5. Switch scenes while a stream is active and confirm the prior stream is aborted.
6. Use doctor historical-case workbench and confirm the old Excel-backed search still works.

- [ ] **Step 4: Record environment blockers instead of overstating success**

Examples:

- Python dependencies missing for backend tests
- frontend `npm` test runner unavailable
- local app not running for manual verification

- [ ] **Step 5: Final checkpoint**

```bash
git add backend frontend tests
git commit -m "feat: implement phase 1 scene-driven workspace"
```

## Execution Notes

- Keep `payload_builder.py` pure. Do not pull SQLite reads into it.
- Keep `src/graph_builder.py` as the only graph-builder file in Phase 1; split assembly, not file layout.
- `PatientGraphService` must not start context maintenance.
- `PATCH /api/sessions/{id}` is bind-only, not generic session update.
- Treat `runtime/patient_registry.db` as the only write target for draft/current patients.
- Do not write draft/current patient data back into `classification.xlsx`.
- Do not collapse patient registry and historical case base into one query model in Phase 1.
