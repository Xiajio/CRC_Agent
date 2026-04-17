# Streamlit to React BFF Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit UI with a separated React/Vite frontend and Python BFF while preserving the existing LangGraph core in `src/`.

**Architecture:** Keep `src/` as the single source of truth for agent orchestration and clinical logic. Add a new `backend/` FastAPI BFF that owns session management, SSE streaming, uploads, assets, and event normalization. Add a new `frontend/` React app that consumes typed stream/snapshot APIs and does not depend on Streamlit state conventions.

**Tech Stack:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, LangGraph, pytest, httpx, React `18.3.1`, React DOM `18.3.1`, React Router DOM `6.28.0`, Vite `5.4.14`, TypeScript `5.6.3`, TailwindCSS `3.4.17`, PostCSS `8.4.49`, Autoprefixer `10.4.20`, GSAP `3.12.7`, Vitest `2.1.8`, React Testing Library `16.1.0`, Playwright `1.49.1`

---

**Repository note:** This workspace is currently **not** a git repository. Replace each `Checkpoint` step with `git add` / `git commit` once the repo is initialized. Until then, use the listed changed files as the manual checkpoint boundary.

## Planned File Structure

**Backend**

- Create: `backend/app.py`
- Create: `backend/api/routes/sessions.py`
- Create: `backend/api/routes/chat.py`
- Create: `backend/api/routes/uploads.py`
- Create: `backend/api/routes/assets.py`
- Create: `backend/api/adapters/event_normalizer.py`
- Create: `backend/api/adapters/card_extractor.py`
- Create: `backend/api/adapters/state_snapshot.py`
- Create: `backend/api/schemas/requests.py`
- Create: `backend/api/schemas/events.py`
- Create: `backend/api/schemas/responses.py`
- Create: `backend/api/services/graph_factory.py`
- Create: `backend/api/services/graph_service.py`
- Create: `backend/api/services/payload_builder.py`
- Create: `backend/api/services/session_store.py`
- Create: `backend/api/services/upload_service.py`
- Create: `backend/api/services/asset_service.py`
- Create: `backend/api/services/fixture_graph_runner.py`

**Frontend**

- Create: `frontend/package.json`
- Create: `frontend/playwright.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/app/router.tsx`
- Create: `frontend/src/app/providers.tsx`
- Create: `frontend/src/app/store/session-store.ts`
- Create: `frontend/src/app/store/stream-reducer.ts`
- Create: `frontend/src/app/store/ui-store.ts`
- Create: `frontend/src/app/api/client.ts`
- Create: `frontend/src/app/api/stream.ts`
- Create: `frontend/src/app/api/types.ts`
- Create: `frontend/src/pages/workspace-page.tsx`
- Create: `frontend/src/features/chat/*`
- Create: `frontend/src/features/cards/*`
- Create: `frontend/src/features/roadmap/*`
- Create: `frontend/src/features/uploads/*`
- Create: `frontend/src/features/patient-profile/*`
- Create: `frontend/src/features/execution-plan/*`
- Create: `frontend/src/components/layout/*`
- Create: `frontend/src/components/primitives/*`
- Create: `frontend/src/components/motion/*`
- Create: `frontend/src/styles/globals.css`
- Create: `frontend/src/styles/tokens.css`

**Tests**

- Create: `tests/backend/test_state_model_source.py`
- Create: `tests/backend/test_session_store.py`
- Create: `tests/backend/test_payload_builder.py`
- Create: `tests/backend/test_card_extractor.py`
- Create: `tests/backend/test_event_normalizer.py`
- Create: `tests/backend/test_state_snapshot.py`
- Create: `tests/backend/test_sessions_routes.py`
- Create: `tests/backend/test_app_lifespan.py`
- Create: `tests/backend/test_chat_stream_route.py`
- Create: `tests/backend/test_uploads_routes.py`
- Create: `tests/fixtures/graph_ticks/README.md`
- Create: `tests/frontend/setup.ts`
- Create: `tests/frontend/stream-reducer.test.ts`
- Create: `tests/frontend/session-store.test.ts`
- Create: `tests/frontend/workspace-page.test.tsx`
- Create: `tests/e2e/workspace.spec.ts`

**Scripts**

- Create: `scripts/capture_graph_fixtures.py`
- Create: `scripts/start_backend_fixture.ps1`
- Create: `scripts/start_backend_real.ps1`
- Create: `scripts/start_frontend.ps1`

**Existing files to modify**

- Modify: `pyproject.toml`
- Modify: `src/state.py`
- Modify: `src/config.py` only if backend-facing settings reuse is cleaner than parallel settings
- Modify: `app/run_app.py`
- Modify: `app/run_app_int.py`
- Modify: `app/README_STREAMLIT.md`
- Modify: `USER_GUIDE.md`

## Task 1: Stabilize `CRCAgentState` Before Migration

**Files:**
- Modify: `src/state.py`
- Test: `tests/backend/test_state_model_source.py`

- [ ] **Step 1: Write the failing test for duplicate field declarations**

```python
from pathlib import Path


def test_crc_agent_state_has_no_duplicate_sensitive_fields():
    source = Path("src/state.py").read_text(encoding="utf-8")
    assert source.count("retrieved_references: List[RetrievedReference]") == 1
    assert source.count("subagent_reports: Annotated[List[Dict[str, Any]], append_list]") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_state_model_source.py -v`  
Expected: FAIL because `src/state.py` currently contains duplicate declarations.

- [ ] **Step 3: Remove duplicate field declarations and keep a single canonical definition**

```python
retrieved_references: List[RetrievedReference] = Field(default_factory=list)
subagent_reports: Annotated[List[Dict[str, Any]], append_list] = Field(default_factory=list)
```

Keep only one copy of each field and preserve existing downstream semantics for `decision_with_evidence`, `roadmap`, `current_plan`, and audit fields.

- [ ] **Step 4: Run regression checks**

Run: `pytest tests/backend/test_state_model_source.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint changed files**

Files to verify:
- `src/state.py`
- `tests/backend/test_state_model_source.py`

## Task 2: Add Backend Runtime and Test Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `backend/app.py`
- Create: `backend/api/__init__.py`
- Create: `backend/api/routes/__init__.py`
- Create: `backend/api/adapters/__init__.py`
- Create: `backend/api/schemas/__init__.py`
- Create: `backend/api/services/__init__.py`
- Test: `tests/backend/test_sessions_routes.py`

- [ ] **Step 1: Write the failing route smoke test**

```python
import httpx
from backend.app import app


async def test_healthlike_session_route_smoke():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/sessions")
    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_sessions_routes.py::test_healthlike_session_route_smoke -v`  
Expected: FAIL because `backend.app` and `/api/sessions` do not exist.

- [ ] **Step 3: Add backend dependencies and app skeleton**

Update `pyproject.toml` with:

```toml
dependencies = [
  # existing deps...
  "fastapi>=0.116.0",
  "uvicorn>=0.35.0",
  "python-multipart>=0.0.20",
]
```

Create minimal app:

```python
from fastapi import FastAPI

app = FastAPI(title="LangG BFF")
```

- [ ] **Step 4: Add placeholder `/api/sessions` route**

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("")
async def create_session():
    return {"session_id": "placeholder", "thread_id": "placeholder", "snapshot": {}}
```

- [ ] **Step 5: Run smoke test**

Run: `pytest tests/backend/test_sessions_routes.py::test_healthlike_session_route_smoke -v`  
Expected: PASS

- [ ] **Step 6: Checkpoint changed files**

Files to verify:
- `pyproject.toml`
- `backend/app.py`
- `backend/api/routes/__init__.py`
- `backend/api/routes/sessions.py`
- `tests/backend/test_sessions_routes.py`

## Task 3: Implement Session Store, Session Routes, and Pending Context Queue

**Files:**
- Create: `backend/api/services/session_store.py`
- Create: `backend/api/schemas/responses.py`
- Modify: `backend/api/routes/sessions.py`
- Test: `tests/backend/test_session_store.py`
- Test: `tests/backend/test_sessions_routes.py`

- [ ] **Step 1: Write failing tests for session lifecycle, busy-lock behavior, and pending context messages**

```python
from backend.api.services.session_store import InMemorySessionStore


def test_create_session_returns_session_and_thread_ids():
    store = InMemorySessionStore()
    meta = store.create_session()
    assert meta.session_id.startswith("sess_")
    assert meta.thread_id.startswith("thread_")


def test_run_lock_is_exclusive_per_session():
    store = InMemorySessionStore()
    meta = store.create_session()
    assert store.try_acquire_run_lock(meta.session_id, "run_1") is True
    assert store.try_acquire_run_lock(meta.session_id, "run_2") is False


def test_pending_context_messages_can_be_enqueued_and_drained():
    store = InMemorySessionStore()
    meta = store.create_session()
    store.enqueue_context_message(meta.session_id, {"role": "user", "content": "context"})
    drained = store.drain_context_messages(meta.session_id)
    assert len(drained) == 1
    assert store.drain_context_messages(meta.session_id) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_session_store.py tests/backend/test_sessions_routes.py -v`  
Expected: FAIL because the session store and typed session responses are not implemented.

- [ ] **Step 3: Implement `SessionMeta` and in-memory store**

```python
@dataclass
class SessionMeta:
    session_id: str
    thread_id: str
    snapshot_version: int = 0
    uploaded_assets: dict[str, "AssetRecord"] = field(default_factory=dict)
    processed_files: dict[str, "ProcessedFileRecord"] = field(default_factory=dict)
    pending_context_messages: list[dict[str, Any]] = field(default_factory=list)
    active_run_id: str | None = None
```

Methods:

- `create_session()`
- `get_session()`
- `rotate_thread()`
- `try_acquire_run_lock()`
- `release_run_lock()`
- `enqueue_context_message()`
- `drain_context_messages()`
- `restore_context_messages()`
- `bump_snapshot_version()`

- [ ] **Step 4: Implement `POST /api/sessions`, `GET /api/sessions/{session_id}`, and `POST /api/sessions/{session_id}/reset`**

Return placeholder snapshots now; real snapshot generation arrives in Task 6.

- [ ] **Step 5: Run tests**

Run: `pytest tests/backend/test_session_store.py tests/backend/test_sessions_routes.py -v`  
Expected: PASS

- [ ] **Step 6: Checkpoint changed files**

Files to verify:
- `backend/api/services/session_store.py`
- `backend/api/schemas/responses.py`
- `backend/api/routes/sessions.py`
- `tests/backend/test_session_store.py`
- `tests/backend/test_sessions_routes.py`

## Task 4: Capture Golden Fixtures and Lock Payload Migration Semantics

**Files:**
- Create: `scripts/capture_graph_fixtures.py`
- Create: `tests/fixtures/graph_ticks/README.md`
- Create: `backend/api/services/payload_builder.py`
- Test: `tests/backend/test_payload_builder.py`

- [ ] **Step 1: Capture representative golden fixtures from real `graph.astream()` runs**

Create a capture script that records sanitized `node_name -> node_output` ticks into:

- `tests/fixtures/graph_ticks/database_case.json`
- `tests/fixtures/graph_ticks/decision_case.json`
- `tests/fixtures/graph_ticks/safety_case.json`

Fixture requirements:

- strip secrets and machine-local absolute paths
- keep enough structure to cover cards, findings, stage, plan, safety, and references
- document the capture command and input prompts in `tests/fixtures/graph_ticks/README.md`

- [ ] **Step 2: Write failing payload-builder tests for `prepare_payload()` parity**

```python
from backend.api.services.payload_builder import build_graph_payload


def test_build_graph_payload_uses_current_turn_and_pending_context_only():
    payload = build_graph_payload(
        chat_request={"message": {"role": "user", "content": "问诊"}},
        session_meta=...,
        state_snapshot=...,
    )
    assert [m["content"] for m in payload["messages"]] == ["upload-context", "问诊"]
```

Add failing cases for:

- `graph_input_cursor` is not rebuilt in BFF
- sticky fields from legacy `prepare_payload()` remain present:
  - `patient_profile`
  - `clinical_stage`
  - `findings`
  - `assessment_draft`
  - `medical_card`
  - `roadmap`
  - `current_patient_id`
- pending context messages are drained once and can be restored on failure

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/backend/test_payload_builder.py -v`  
Expected: FAIL because the payload builder and fixture docs do not exist.

- [ ] **Step 4: Implement the migration strategy for `graph_input_cursor`**

Explicit strategy:

- do **not** rebuild `graph_input_cursor` in BFF
- React sends only the current turn
- BFF prepends any queued upload-derived context messages from `pending_context_messages`
- LangGraph checkpointer retains prior conversation state
- if graph execution aborts before the turn completes, drained pending context messages must be restorable

- [ ] **Step 5: Implement `payload_builder.py`**

`build_graph_payload()` should be the BFF replacement for Streamlit `prepare_payload()` and preserve:

- `medical_card` injection
- `current_patient_id` injection
- sticky contextual state fields
- queued upload context messages

- [ ] **Step 6: Run tests**

Run: `pytest tests/backend/test_payload_builder.py -v`  
Expected: PASS

- [ ] **Step 7: Checkpoint changed files**

Files to verify:
- `scripts/capture_graph_fixtures.py`
- `tests/fixtures/graph_ticks/README.md`
- golden fixture JSON files
- `backend/api/services/payload_builder.py`
- `tests/backend/test_payload_builder.py`

## Task 5: Implement Typed Event Schemas and Card Extraction

**Files:**
- Create: `backend/api/schemas/events.py`
- Create: `backend/api/adapters/card_extractor.py`
- Test: `tests/backend/test_card_extractor.py`

- [ ] **Step 1: Write failing tests for three-channel card extraction**

```python
from backend.api.adapters.card_extractor import extract_cards


def test_extract_cards_prefers_top_level_over_findings_and_kwargs():
    node_output = {
        "patient_card": {"type": "patient_card", "name": "top"},
        "findings": {"patient_card": {"type": "patient_card", "name": "findings"}},
    }
    messages = [type("Msg", (), {"additional_kwargs": {"patient_card": {"type": "patient_card", "name": "kwargs"}}})()]
    cards = extract_cards("database", node_output, messages)
    assert len(cards) == 1
    assert cards[0].payload["name"] == "top"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_card_extractor.py -v`  
Expected: FAIL because extractor and card event schema do not exist.

- [ ] **Step 3: Implement typed `CardUpsertEvent` and extractor**

```python
class CardUpsertEvent(BaseModel):
    type: Literal["card.upsert"] = "card.upsert"
    card_type: str
    payload: dict[str, Any]
    source_channel: Literal["state", "findings", "message_kwargs"]
```

Extractor rules:

- top-level state
- findings embedded cards
- `AIMessage.additional_kwargs`
- priority: state > findings > kwargs
- dedupe: `(card_type, payload_hash)`

- [ ] **Step 4: Add tests for decision card, medical card, and pathology/radiomics card families**

Cover:

- `decision_json -> decision_card`
- `medical_card`
- `pathology_card`
- `pathology_slide_card`
- `radiomics_report_card`

- [ ] **Step 5: Run tests**

Run: `pytest tests/backend/test_card_extractor.py -v`  
Expected: PASS

- [ ] **Step 6: Checkpoint changed files**

Files to verify:
- `backend/api/schemas/events.py`
- `backend/api/adapters/card_extractor.py`
- `tests/backend/test_card_extractor.py`

## Task 6: Implement Event Normalizer and Snapshot Builder

**Files:**
- Create: `backend/api/adapters/event_normalizer.py`
- Create: `backend/api/adapters/state_snapshot.py`
- Modify: `backend/api/routes/sessions.py`
- Modify: `backend/api/schemas/responses.py`
- Test: `tests/backend/test_event_normalizer.py`
- Test: `tests/backend/test_state_snapshot.py`
- Test: `tests/backend/test_sessions_routes.py`

- [ ] **Step 1: Write failing tests for handler-to-event mapping**

```python
from backend.api.adapters.event_normalizer import normalize_tick


def test_stage_and_profile_become_typed_events():
    events = list(normalize_tick(
        node_name="staging",
        node_output={"clinical_stage": "Diagnosis", "patient_profile": {"age": 62}},
        new_messages=[],
        ctx=None,
    ))
    assert any(event.type == "stage.update" for event in events)
    assert any(event.type == "patient_profile.update" for event in events)
```

Add failing cases for:

- `critic.verdict`
- `roadmap.update`
- `safety.alert`
- `findings.patch`
- `references.append`
- fixture-backed `status.node` + card extraction compatibility

- [ ] **Step 2: Write failing snapshot test for bounded recovery payload**

```python
from backend.api.adapters.state_snapshot import build_session_snapshot


def test_snapshot_limits_messages_and_keeps_asset_refs_only():
    snapshot = build_session_snapshot(agent_state=..., session_meta=..., message_limit=2)
    assert len(snapshot["messages"]) == 2
    assert "image_base64" not in str(snapshot)
```

Add a second failing case ensuring `medical_card`, `assessment_draft`, and `current_patient_id` survive into recovery payloads.

Add a third failing route test ensuring `GET /api/sessions/{session_id}/messages?before=<cursor>&limit=<n>` returns bounded history pages without embedding asset blobs in the response.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py -v`  
Expected: FAIL because normalizer and snapshot builder are not implemented.

- [ ] **Step 4: Implement normalizer with the Streamlit mapping table**

Map:

- `handle_critic_feedback -> critic.verdict`
- `handle_node_messages -> message.done/references.append`
- `handle_findings_update -> findings.patch`
- `handle_stage_update -> stage.update`
- `handle_patient_profile_update -> patient_profile.update`
- `handle_safety_update -> safety.alert`
- `handle_roadmap_update -> roadmap.update`
- `handle_medical_card_update -> card.upsert("medical_card")`
- other card handlers -> `card_extractor`
- `assessment_draft`, `decision_json`, `current_patient_id` -> snapshot model fields

- [ ] **Step 5: Implement snapshot builder**

Snapshot fields:

- `messages`
- `messages_total`
- `messages_next_before_cursor`
- `cards`
- `roadmap`
- `findings`
- `patient_profile`
- `stage`
- `assessment_draft`
- `current_patient_id`
- `references`
- `plan`
- `critic`
- `safety_alert`
- `uploaded_assets`

- [ ] **Step 5.5: Extend session routes for bounded recovery and history pagination**

Add:

- `GET /api/sessions/{session_id}` -> bounded recovery snapshot
- `GET /api/sessions/{session_id}/messages` -> paginated message history using `before` + `limit`

The pagination route must return message cursors plus asset references only; no inline binary payloads or base64 blobs.

- [ ] **Step 6: Run tests**

Run: `pytest tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py tests/backend/test_sessions_routes.py -v`  
Expected: PASS

- [ ] **Step 7: Checkpoint changed files**

Files to verify:
- `backend/api/adapters/event_normalizer.py`
- `backend/api/adapters/state_snapshot.py`
- `backend/api/routes/sessions.py`
- `backend/api/schemas/responses.py`
- `tests/backend/test_event_normalizer.py`
- `tests/backend/test_state_snapshot.py`
- `tests/backend/test_sessions_routes.py`

## Task 7: Implement FastAPI Lifespan, Graph Service, Fixture Runner, and SSE Chat Route

**Files:**
- Create: `backend/api/services/graph_factory.py`
- Create: `backend/api/services/graph_service.py`
- Create: `backend/api/services/fixture_graph_runner.py`
- Create: `backend/api/routes/chat.py`
- Modify: `backend/app.py`
- Test: `tests/backend/test_app_lifespan.py`
- Test: `tests/backend/test_chat_stream_route.py`

- [ ] **Step 1: Write failing tests for lifespan startup, SSE order, and exception paths**

```python
async def test_lifespan_compiles_graph_once_and_sets_app_state():
    ...


async def test_stream_route_emits_status_then_done(async_client, session_id):
    ...


async def test_stream_route_rejects_concurrent_run(async_client, session_id, session_store):
    ...


async def test_stream_route_emits_error_and_done_on_graph_failure(async_client, session_id):
    ...


async def test_stream_disconnect_releases_run_lock(async_client, session_id, session_store):
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_app_lifespan.py tests/backend/test_chat_stream_route.py -v`  
Expected: FAIL because lifespan, fixture runner, graph service, and chat route do not exist.

- [ ] **Step 3: Implement FastAPI lifespan**

Startup must:

- load settings
- create session store
- compile graph singleton
- create runtime asset directories
- optionally warm RAG on startup when `RAG_WARMUP=true`

Shutdown must release any open resources cleanly.

Phase 1 concurrency model is explicitly:

- single process
- one Uvicorn worker
- asyncio coroutines inside that worker
- `session_id` scoped write locks enforced in-process only

Do not claim multi-worker safety in this phase. Multi-worker deployment is deferred until both the session store and checkpointer are externalized.

- [ ] **Step 4: Implement graph singleton factory and fixture runner**

```python
from src.graph_builder import build_graph


def get_compiled_graph():
    # compile once per process
    ...
```

Reuse `src/checkpoint.py` and existing settings.

Modes:

- `GRAPH_RUNNER_MODE=real` -> compiled LangGraph
- `GRAPH_RUNNER_MODE=fixture` -> deterministic replay from golden fixtures

Use fixture mode for Playwright and deterministic backend tests; use real mode for manual smoke validation.

- [ ] **Step 5: Implement `GraphService.stream_turn()`**

Requirements:

- resolve `session_id -> thread_id`
- build payload via `payload_builder.py`
- acquire session write lock
- call `graph.astream(...)` with `recursion_limit=200`
- emit `status.node`
- normalize each tick
- send `: ping` heartbeats every 15s
- emit `done` in both success and recoverable error flows
- restore pending context if the turn aborts before successful completion
- release lock and clear `active_run_id` in `finally`

- [ ] **Step 6: Implement `POST /api/sessions/{session_id}/messages/stream`**

Use `StreamingResponse` with `text/event-stream`.

- [ ] **Step 7: Run tests**

Run: `pytest tests/backend/test_app_lifespan.py tests/backend/test_chat_stream_route.py -v`  
Expected: PASS

- [ ] **Step 8: Checkpoint changed files**

Files to verify:
- `backend/api/services/graph_factory.py`
- `backend/api/services/graph_service.py`
- `backend/api/services/fixture_graph_runner.py`
- `backend/api/routes/chat.py`
- `backend/app.py`
- `tests/backend/test_app_lifespan.py`
- `tests/backend/test_chat_stream_route.py`

## Task 8: Implement Upload and Asset Services, Including Medical Card Injection

**Files:**
- Create: `backend/api/services/upload_service.py`
- Create: `backend/api/services/asset_service.py`
- Create: `backend/api/routes/uploads.py`
- Create: `backend/api/routes/assets.py`
- Test: `tests/backend/test_uploads_routes.py`

- [ ] **Step 1: Write failing tests for file dedupe, medical card injection, and cleanup**

```python
async def test_upload_reuses_processed_file_by_sha(async_client, session_id):
    ...


async def test_upload_enqueues_context_message_for_next_turn(async_client, session_id):
    ...


async def test_upload_failure_does_not_leave_partial_asset_state(async_client, session_id):
    ...
```

Cover:

- SHA-based dedupe
- `medical_card` derivation from document conversion
- synthetic context message creation mirroring legacy `create_context_message_from_card()`
- no orphan asset record / no stale `processed_files` / no stale pending context on failure

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_uploads_routes.py -v`  
Expected: FAIL because upload and asset routes do not exist.

- [ ] **Step 3: Implement runtime asset layout and services**

Target layout:

```text
runtime/
  assets/
    <session_id>/
      <asset_id>/
        original
        derived/
```

Services must:

- compute `sha256`
- dedupe on `processed_files`
- return `asset_id`
- integrate current text extraction / document conversion helpers
- enqueue upload-derived context for the next graph turn

- [ ] **Step 4: Implement upload and asset routes**

Routes:

- `POST /api/sessions/{session_id}/uploads`
- `GET /api/assets/{asset_id}`

- [ ] **Step 5: Run tests**

Run: `pytest tests/backend/test_uploads_routes.py -v`  
Expected: PASS

- [ ] **Step 6: Checkpoint changed files**

Files to verify:
- `backend/api/services/upload_service.py`
- `backend/api/services/asset_service.py`
- `backend/api/routes/uploads.py`
- `backend/api/routes/assets.py`
- `tests/backend/test_uploads_routes.py`

## Task 9: Add Authentication, CORS, and Runtime Settings

**Files:**
- Modify: `backend/app.py`
- Modify: `src/config.py` or create `backend/api/services/settings.py`
- Test: `tests/backend/test_sessions_routes.py`
- Test: `tests/backend/test_chat_stream_route.py`

- [ ] **Step 1: Write failing tests for bearer auth and origin allowlist**

```python
async def test_bearer_auth_rejects_missing_token(async_client):
    response = await async_client.post("/api/sessions")
    assert response.status_code == 401
```

Add a second test that a disallowed `Origin` header is rejected or not granted CORS headers.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_sessions_routes.py tests/backend/test_chat_stream_route.py -v`  
Expected: FAIL because auth/CORS middleware is not wired.

- [ ] **Step 3: Implement environment-driven auth mode and CORS allowlist**

Config:

- `AUTH_MODE=none|bearer`
- `API_BEARER_TOKEN`
- `FRONTEND_ORIGINS`
- `GRAPH_RUNNER_MODE=fixture|real`
- `RAG_WARMUP=true|false`

- [ ] **Step 4: Run tests**

Run: `pytest tests/backend/test_sessions_routes.py tests/backend/test_chat_stream_route.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint changed files**

Files to verify:
- `backend/app.py`
- `src/config.py` or `backend/api/services/settings.py`
- updated backend route tests

## Task 10: Scaffold the React Frontend and Lock the Toolchain

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/playwright.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/styles/globals.css`
- Create: `frontend/src/styles/tokens.css`
- Create: `tests/frontend/setup.ts`
- Test: `tests/frontend/workspace-page.test.tsx`

- [ ] **Step 1: Write the failing frontend smoke test**

```tsx
import { render, screen } from "@testing-library/react";
import { WorkspacePage } from "../../frontend/src/pages/workspace-page";


test("renders the workspace shell", () => {
  render(<WorkspacePage />);
  expect(screen.getByText("Clinical Workspace")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: FAIL because the frontend app does not exist yet.

- [ ] **Step 3: Create Vite/React/Tailwind/GSAP skeleton with pinned versions**

Pin at minimum:

- `react@18.3.1`
- `react-dom@18.3.1`
- `react-router-dom@6.28.0`
- `vite@5.4.14`
- `typescript@5.6.3`
- `tailwindcss@3.4.17`
- `postcss@8.4.49`
- `autoprefixer@10.4.20`
- `gsap@3.12.7`
- `vitest@2.1.8`
- `@testing-library/react@16.1.0`
- `@testing-library/jest-dom@6.6.3`
- `playwright@1.49.1`

Use Tailwind v3 config conventions intentionally; do not adopt Tailwind v4 API in this migration.

- [ ] **Step 4: Create minimal page shell and testing setup**

```tsx
export function WorkspacePage() {
  return <div>Clinical Workspace</div>;
}
```

- [ ] **Step 5: Run test**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: PASS

- [ ] **Step 6: Checkpoint changed files**

Files to verify:
- `frontend/package.json`
- `frontend/playwright.config.ts`
- `frontend/tsconfig.json`
- `frontend/vite.config.ts`
- `frontend/tailwind.config.ts`
- `frontend/postcss.config.js`
- `frontend/src/main.tsx`
- `frontend/src/pages/workspace-page.tsx`
- `frontend/src/styles/globals.css`
- `frontend/src/styles/tokens.css`
- `tests/frontend/setup.ts`
- `tests/frontend/workspace-page.test.tsx`

## Task 11: Implement Frontend API Client, Stream Parser, and Stores

**Files:**
- Create: `frontend/src/app/api/types.ts`
- Create: `frontend/src/app/api/client.ts`
- Create: `frontend/src/app/api/stream.ts`
- Create: `frontend/src/app/store/session-store.ts`
- Create: `frontend/src/app/store/stream-reducer.ts`
- Create: `frontend/src/app/store/ui-store.ts`
- Test: `tests/frontend/stream-reducer.test.ts`
- Test: `tests/frontend/session-store.test.ts`

- [ ] **Step 1: Write failing reducer tests for typed events**

```ts
import { reduceStreamEvent } from "../../frontend/src/app/store/stream-reducer";


test("card.upsert replaces cards by card_type", () => {
  const next = reduceStreamEvent(
    { cards: {} } as any,
    { type: "card.upsert", card_type: "patient_card", payload: { name: "A" } } as any,
  );
  expect(next.cards.patient_card.name).toBe("A");
});
```

Add failing tests for:

- `stage.update`
- `patient_profile.update`
- `plan.update`
- `safety.alert`
- `message.done`
- paginated snapshot hydration

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run test -- --run tests/frontend/stream-reducer.test.ts tests/frontend/session-store.test.ts`  
Expected: FAIL because reducers and stores do not exist.

- [ ] **Step 3: Implement typed event models, reducer, and fetch-based SSE parser**

State must include:

- `sessionId`
- `threadId`
- `messages`
- `cards`
- `roadmap`
- `findings`
- `patientProfile`
- `stage`
- `references`
- `plan`
- `critic`
- `safetyAlert`

- [ ] **Step 4: Implement API client methods**

Requirements:

- `createSession()`
- `getSession(sessionId, messageLimit?)`
- `getMessages(sessionId, before?, limit?)`
- `streamTurn(sessionId, request, onEvent)`
- `uploadFile(sessionId, file)`
- `resetSession(sessionId)`

- [ ] **Step 5: Run tests**

Run: `npm run test -- --run tests/frontend/stream-reducer.test.ts tests/frontend/session-store.test.ts`  
Expected: PASS

- [ ] **Step 6: Checkpoint changed files**

Files to verify:
- `frontend/src/app/api/types.ts`
- `frontend/src/app/api/client.ts`
- `frontend/src/app/api/stream.ts`
- `frontend/src/app/store/session-store.ts`
- `frontend/src/app/store/stream-reducer.ts`
- `frontend/src/app/store/ui-store.ts`
- `tests/frontend/stream-reducer.test.ts`
- `tests/frontend/session-store.test.ts`

## Task 12: Build the Workspace Shell, Routing, and Session Bootstrap

**Files:**
- Create: `frontend/src/app/router.tsx`
- Create: `frontend/src/app/providers.tsx`
- Create: `frontend/src/components/layout/workspace-layout.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Test: `tests/frontend/workspace-page.test.tsx`

- [ ] **Step 1: Extend the failing UI tests to cover the 3-column shell**

```tsx
test("renders left center and right workspace regions", () => {
  render(<WorkspacePage />);
  expect(screen.getByTestId("left-rail")).toBeInTheDocument();
  expect(screen.getByTestId("center-workspace")).toBeInTheDocument();
  expect(screen.getByTestId("right-inspector")).toBeInTheDocument();
});
```

Add failing tests for:

- automatic session creation on first load
- snapshot bootstrap rendering
- loading and error placeholders

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: FAIL because routing, providers, and bootstrap are not implemented.

- [ ] **Step 3: Implement routing, providers, and shell layout**

Desktop layout:

- left rail
- center workspace
- right inspector

Bootstrap flow:

- create session on first load
- render snapshot

- [ ] **Step 4: Run tests**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: PASS

- [ ] **Step 5: Checkpoint changed files**

Files to verify:
- `frontend/src/app/router.tsx`
- `frontend/src/app/providers.tsx`
- `frontend/src/components/layout/workspace-layout.tsx`
- `frontend/src/pages/workspace-page.tsx`
- `tests/frontend/workspace-page.test.tsx`

## Task 13: Build Domain Panels and Structured UI Surfaces

**Files:**
- Create: `frontend/src/features/chat/chat-timeline.tsx`
- Create: `frontend/src/features/chat/composer.tsx`
- Create: `frontend/src/features/chat/execution-status.tsx`
- Create: `frontend/src/features/cards/card-dock.tsx`
- Create: `frontend/src/features/cards/card-detail.tsx`
- Create: `frontend/src/features/roadmap/roadmap-panel.tsx`
- Create: `frontend/src/features/uploads/upload-panel.tsx`
- Create: `frontend/src/features/patient-profile/patient-profile-panel.tsx`
- Create: `frontend/src/features/execution-plan/plan-panel.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Test: `tests/frontend/workspace-page.test.tsx`

- [ ] **Step 1: Extend failing UI tests for domain panels**

Add tests for:

- safety banner rendering
- card dock rendering after `card.upsert`
- patient profile rendering after snapshot load
- roadmap stage highlighting
- plan panel rendering
- upload panel rendering

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: FAIL because feature components do not exist.

- [ ] **Step 3: Implement the domain panels**

Panels:

- chat timeline
- composer
- execution status
- card dock/detail
- roadmap
- upload
- patient profile
- execution plan / references

- [ ] **Step 4: Run tests**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: PASS

- [ ] **Step 5: Checkpoint changed files**

Files to verify:
- domain feature files
- `frontend/src/pages/workspace-page.tsx`
- `tests/frontend/workspace-page.test.tsx`

## Task 14: Integrate Streaming Interactions, GSAP Motion, and Responsive Behavior

**Files:**
- Create: `frontend/src/components/motion/*`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Modify: `frontend/src/features/chat/*`
- Modify: `frontend/src/features/cards/*`
- Test: `tests/frontend/workspace-page.test.tsx`

- [ ] **Step 1: Extend failing tests for interactions and resilience**

Add tests for:

- submit message -> consume stream -> update reducer
- upload file -> reflect queued card/context state
- responsive panel toggle state
- disconnect/error UI path

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: FAIL because interaction wiring is not complete.

- [ ] **Step 3: Wire interactions and GSAP**

Requirements:

- create session on first load only once
- submit messages via `streamTurn()`
- consume node-level stream events
- show blocking safety alerts
- keep GSAP limited to shell reveal, card highlight, roadmap progression, and viewer transitions
- preserve usability under `prefers-reduced-motion`

- [ ] **Step 4: Run tests**

Run: `npm run test -- --run tests/frontend/workspace-page.test.tsx`  
Expected: PASS

- [ ] **Step 5: Checkpoint changed files**

Files to verify:
- `frontend/src/components/motion/*`
- updated feature files
- `frontend/src/pages/workspace-page.tsx`
- `tests/frontend/workspace-page.test.tsx`

## Task 15: Add E2E Environment, Full Verification, and Retire Streamlit Entrypoints

**Files:**
- Create: `tests/e2e/workspace.spec.ts`
- Create: `scripts/start_backend_fixture.ps1`
- Create: `scripts/start_backend_real.ps1`
- Create: `scripts/start_frontend.ps1`
- Modify: `frontend/package.json`
- Modify: `USER_GUIDE.md`
- Modify: `app/README_STREAMLIT.md`
- Modify: `app/run_app.py`
- Modify: `app/run_app_int.py`

- [ ] **Step 1: Write the failing E2E spec**

```ts
import { test, expect } from "@playwright/test";


test("user can create a session, send a message, and see cards", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Clinical Workspace")).toBeVisible();
  await page.getByRole("textbox").fill("Analyze treatment plan");
  await page.getByRole("button", { name: /send/i }).click();
  await expect(page.getByText(/status/i)).toBeVisible();
});
```

- [ ] **Step 2: Run the E2E spec to verify it fails**

Run: `npm run test:e2e -- --project=chromium tests/e2e/workspace.spec.ts`  
Expected: FAIL until frontend and backend are both wired.

- [ ] **Step 3: Create the E2E startup scripts and runtime modes**

Implement:

- `scripts/start_backend_fixture.ps1`
- `scripts/start_backend_real.ps1`
- `scripts/start_frontend.ps1`

Requirements:

- fixture backend script starts FastAPI with `GRAPH_RUNNER_MODE=fixture`
- real backend script starts FastAPI with `GRAPH_RUNNER_MODE=real`
- real backend script may opt into `RAG_WARMUP=true`
- frontend script starts Vite on a fixed local port used by Playwright
- `frontend/package.json` defines `test:e2e` and any helper scripts required by Playwright

Default automation:

- backend tests -> fixture mode
- Playwright -> fixture mode
- manual smoke -> real mode

- [ ] **Step 4: Finish parity verification checklist**

Verify:

- create session
- upload file
- send message
- see `status.node`
- see `message.done`
- see `card.upsert` reflected in UI
- refresh page and recover from snapshot
- open older history through paginated message API
- reset session and get a new thread
- disconnect stream and recover from snapshot

- [ ] **Step 5: Deprecate Streamlit runtime entrypoints**

At this stage:

- mark Streamlit docs as legacy
- redirect primary user guide to backend/frontend startup
- keep old Streamlit files in place temporarily if rollback is needed
- remove Streamlit from the default run path

- [ ] **Step 6: Run the full verification suite**

Backend:

Run: `pytest tests/backend -v`  
Expected: PASS

Frontend unit:

Run: `npm run test -- --run`  
Expected: PASS

Frontend E2E:

Run: `npm run test:e2e -- --project=chromium`  
Expected: PASS

Manual real-graph smoke:

Run backend with `scripts/start_backend_real.ps1` and verify one end-to-end turn against the live compiled graph, including upload-derived context and snapshot recovery.

- [ ] **Step 7: Checkpoint changed files**

Files to verify:
- `tests/e2e/workspace.spec.ts`
- `scripts/start_backend_fixture.ps1`
- `scripts/start_backend_real.ps1`
- `scripts/start_frontend.ps1`
- `frontend/package.json`
- `USER_GUIDE.md`
- `app/README_STREAMLIT.md`
- `app/run_app.py`
- `app/run_app_int.py`

## Execution Notes

- Implement in the exact task order above. Do not skip Task 1; the state model cleanup is a prerequisite for stable snapshot work.
- Phase 1 deployment is single-process, single-worker FastAPI with asyncio; `session_id` write locks are only guaranteed within that process.
- Keep `src/` business logic changes minimal. Most migration work belongs in `backend/`, `frontend/`, and tests.
- `payload_builder.py` is the BFF replacement for Streamlit `prepare_payload()`.
- `graph_input_cursor` is removed, not rebuilt. The BFF sends only the current user turn plus any queued upload-derived context messages, while the LangGraph checkpointer carries prior conversation state.
- Upload-derived `medical_card` data and the synthetic context message must survive until the next turn and must be restorable if execution fails before completion.
- Do not reintroduce Streamlit-style popup booleans into React.
- Treat `card_extractor.py` as the single source of truth for both streaming and snapshot card derivation.
- Treat `GET /api/sessions/{session_id}` and `GET /api/sessions/{session_id}/messages` as distinct contracts: snapshot recovery vs. message pagination.
- FastAPI lifespan must compile the graph once per process and optionally warm RAG only when `RAG_WARMUP=true`.
- Default automated verification uses fixture graph mode. Real graph execution is reserved for manual smoke validation and golden fixture capture.
- Client reconnect in v1 is snapshot-based rather than `Last-Event-ID` resume: on disconnect, release the run lock, reload the latest snapshot, then fetch older messages if needed.
- First release acceptance is based on **node-level streaming**, not token-level streaming.
- Exception paths are required test coverage, not optional hardening: graph failure, upload failure, client disconnect, and partial-state cleanup must all be verified.

## Manual Review Checklist

- `src/state.py` no longer contains duplicate sensitive fields.
- `backend/` has no imports from `streamlit`.
- `payload_builder.py` preserves legacy `prepare_payload()` sticky state semantics without rebuilding `graph_input_cursor`.
- `frontend/` does not inspect raw LangGraph message objects.
- `card.upsert` is the only card update path exposed to the frontend.
- upload-derived `medical_card` plus its synthetic context message are preserved for the next turn and cleaned up correctly on failure.
- auth/CORS are enabled in deployed mode.
- assets are served by `asset_id`, not raw local paths.
- recovery snapshots are bounded and older messages come from `GET /api/sessions/{session_id}/messages`.
- FastAPI lifespan compiles the graph once and only warms RAG when configured.
- session write locks reject concurrent `stream`, `upload`, and `reset`.
- fixture-mode E2E is deterministic, and a separate real-graph smoke path is documented.