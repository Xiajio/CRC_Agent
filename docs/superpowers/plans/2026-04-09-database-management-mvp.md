# Database Management MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read/write virtual database console for the React + FastAPI runtime so users can search, inspect, and update multiple patient records without being limited to single-patient chat turns.

**Architecture:** Add explicit FastAPI database APIs on top of the existing virtual case database and card formatter, then build a dedicated `/database` React page that reuses the same patient/imaging/pathology card renderers already used in the workspace. Add an agentic natural-language query bar that converts constrained medical search requests into supported filter fields instead of generating free-form SQL.

**Tech Stack:** FastAPI, Pydantic, existing `src/services/virtual_database_service.py`, existing `src/tools/card_formatter.py`, React, React Router, existing API client/provider, Vitest + React Testing Library, pytest

---

## MVP Scope

**In scope**
- Read-only stats, faceted search, pagination, sorting, row selection
- Patient detail drawer/panel with reusable cards
- Single-record create/update (`upsert`) against the current Excel-backed virtual database
- Natural-language query parser that maps user text into supported filter fields
- Shared navigation between workspace and database console

**Out of scope**
- Batch import wizard
- Hard delete
- Auth/RBAC changes
- Arbitrary NL2SQL or schema-free querying
- Replacing Excel with PostgreSQL in this iteration

## File Structure

**Backend**
- Create: `backend/api/routes/database.py`
  - Database-specific HTTP routes: stats, search, detail, upsert, intent parse
- Create: `backend/api/schemas/database.py`
  - Pydantic request/response models for database APIs
- Create: `backend/api/services/database_service.py`
  - Application service that wraps virtual DB reads/writes and builds reusable detail payloads
- Create: `backend/api/services/database_intent_service.py`
  - Constrained natural-language parser that returns supported filter fields and unsupported terms
- Modify: `backend/app.py`
  - Register the new router
- Modify: `src/services/virtual_database_service.py`
  - Add stable helpers for paginated search/facets if needed by the new backend service
- Modify: `src/tools/card_formatter.py`
  - Only if needed to expose a small helper for assembling a patient detail card bundle

**Frontend**
- Create: `frontend/src/pages/database-page.tsx`
  - Database console page composition and local page state
- Create: `frontend/src/features/database/database-natural-query-bar.tsx`
  - NL query input, parse/apply button, parse status, unsupported-term notice
- Create: `frontend/src/features/database/database-filters-panel.tsx`
  - Structured filters with clear/apply controls
- Create: `frontend/src/features/database/database-results-table.tsx`
  - Search results table, sort headers, pagination controls, row selection
- Create: `frontend/src/features/database/database-detail-panel.tsx`
  - Detail panel that reuses `cardTitle` + `renderCardContent`
- Create: `frontend/src/features/database/database-edit-form.tsx`
  - Minimal patient update form for the current virtual schema
- Modify: `frontend/src/app/router.tsx`
  - Add `/database`
- Modify: `frontend/src/app/api/types.ts`
  - Add database request/response types
- Modify: `frontend/src/app/api/client.ts`
  - Add database API methods
- Modify: `frontend/src/pages/workspace-page.tsx`
  - Add entry link/button to the database console
- Modify: `frontend/src/styles/globals.css`
  - Add database page layout styles that match the existing visual language

**Tests**
- Create: `tests/backend/test_database_routes.py`
- Create: `tests/backend/test_database_intent_service.py`
- Create: `tests/frontend/database-page.test.tsx`
- Modify: `tests/frontend/workspace-page.test.tsx`

## API Contracts

### `GET /api/database/stats`

Response:

```json
{
  "total_cases": 123,
  "gender_distribution": {"male": 70, "female": 53},
  "age_statistics": {"min": 28, "max": 84, "mean": 58.4},
  "tumor_location_distribution": {"横结肠": 10},
  "ct_stage_distribution": {"4b": 12},
  "mmr_status_distribution": {"pMMR_MSS": 98, "dMMR_MSI_H": 25},
  "cea_statistics": {"min": 0.9, "max": 88.2, "mean": 12.6}
}
```

### `POST /api/database/cases/search`

Request:

```json
{
  "filters": {
    "patient_id": null,
    "tumor_location": ["横结肠"],
    "ct_stage": ["4b"],
    "cn_stage": ["1c"],
    "histology_type": [],
    "mmr_status": ["pMMR_MSS"],
    "age_min": 30,
    "age_max": 40,
    "cea_max": 20
  },
  "pagination": {"page": 1, "page_size": 20},
  "sort": {"field": "patient_id", "direction": "asc"}
}
```

Response:

```json
{
  "items": [
    {
      "patient_id": 93,
      "gender": "男",
      "age": 31,
      "tumor_location": "横结肠",
      "histology_type": "中分化",
      "ct_stage": "4b",
      "cn_stage": "1c",
      "clinical_stage": "III期",
      "cea_level": 10.29,
      "mmr_status": "pMMR (MSS)"
    }
  ],
  "total": 1,
  "page": 1,
  "page_size": 20,
  "applied_filters": {},
  "warnings": []
}
```

### `GET /api/database/cases/{patient_id}`

Response:

```json
{
  "patient_id": "093",
  "case_record": {},
  "available_data": {
    "case_info": true,
    "imaging": true,
    "pathology_slides": false
  },
  "cards": {
    "patient_card": {},
    "imaging_card": {}
  }
}
```

### `POST /api/database/cases/upsert`

Request:

```json
{
  "record": {
    "patient_id": 93,
    "age": 32,
    "tumor_location": "横结肠",
    "ct_stage": "4b",
    "cn_stage": "1c"
  }
}
```

### `POST /api/database/query-intent`

Request:

```json
{
  "query": "帮我找出 30 到 40 岁、横结肠、pMMR 的患者"
}
```

Response:

```json
{
  "query": "帮我找出 30 到 40 岁、横结肠、pMMR 的患者",
  "normalized_query": "30-40岁 横结肠 pMMR",
  "filters": {
    "age_min": 30,
    "age_max": 40,
    "tumor_location": ["横结肠"],
    "mmr_status": ["pMMR_MSS"]
  },
  "unsupported_terms": [],
  "warnings": []
}
```

If the user asks for unsupported concepts such as liver metastasis and the schema does not contain that field, return:

```json
{
  "filters": {},
  "unsupported_terms": ["肝转移"],
  "warnings": ["当前虚拟数据库不包含转移器官字段，未执行该条件筛选。"]
}
```

## UI Flow

1. User enters `/database` from the workspace header.
2. Page loads `stats` once and executes a default `search` for page 1.
3. User can refine results in two ways:
   - Structured filters panel
   - Natural-language query bar
4. Natural-language parse fills the structured form first, then triggers `search`.
5. Clicking a result row loads `GET /api/database/cases/{patient_id}`.
6. Detail panel renders the returned cards via existing `cardTitle` + `renderCardContent`.
7. User edits supported fields in the detail panel and submits `upsert`.
8. After save, the page refreshes the detail payload and current search result row.

## Reuse Rules

- Do not duplicate patient/imaging/pathology card markup.
- `database-detail-panel.tsx` must import and reuse:
  - `cardTitle`
  - `renderCardContent`
  - existing card payloads returned by backend
- Detail payload must therefore be card-ready on the backend.

## Task 1: Backend Read-Only Database API

**Files:**
- Create: `backend/api/routes/database.py`
- Create: `backend/api/schemas/database.py`
- Create: `backend/api/services/database_service.py`
- Modify: `backend/app.py`
- Test: `tests/backend/test_database_routes.py`

- [ ] **Step 1: Write failing backend route tests for stats, search, and detail**

```python
def test_get_database_stats(client):
    response = client.get("/api/database/stats")
    assert response.status_code == 200
    assert "total_cases" in response.json()

def test_search_database_cases(client):
    response = client.post("/api/database/cases/search", json={
        "filters": {"tumor_location": ["横结肠"]},
        "pagination": {"page": 1, "page_size": 20},
        "sort": {"field": "patient_id", "direction": "asc"},
    })
    assert response.status_code == 200
    assert "items" in response.json()

def test_get_case_detail_returns_cards(client):
    response = client.get("/api/database/cases/93")
    assert response.status_code == 200
    body = response.json()
    assert "cards" in body
    assert "patient_card" in body["cards"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_database_routes.py -v`
Expected: FAIL because `/api/database/*` routes do not exist.

- [ ] **Step 3: Implement database schemas and service**

Implementation notes:
- `database_service.py` should expose:
  - `get_database_stats()`
  - `search_database_cases(filters, pagination, sort)`
  - `get_database_case_detail(patient_id)`
- `get_database_case_detail()` must build `cards` using existing card formatter outputs.
- Keep search field set constrained to the currently supported virtual DB schema.

- [ ] **Step 4: Add routes and register router**

Implementation notes:
- `database.py` defines:
  - `GET /api/database/stats`
  - `POST /api/database/cases/search`
  - `GET /api/database/cases/{patient_id}`
- Register `database_routes.router` in `backend/app.py`.

- [ ] **Step 5: Run backend route tests**

Run: `pytest tests/backend/test_database_routes.py -v`
Expected: PASS for stats, search, and detail.

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/api/routes/database.py backend/api/schemas/database.py backend/api/services/database_service.py tests/backend/test_database_routes.py
git commit -m "feat: add database read APIs"
```

## Task 2: Backend Natural-Language Query Parser

**Files:**
- Create: `backend/api/services/database_intent_service.py`
- Modify: `backend/api/routes/database.py`
- Modify: `backend/api/schemas/database.py`
- Test: `tests/backend/test_database_intent_service.py`
- Test: `tests/backend/test_database_routes.py`

- [ ] **Step 1: Write failing tests for supported and unsupported NL queries**

```python
def test_parse_supported_database_query():
    result = parse_database_query("30到40岁 横结肠 pMMR")
    assert result.filters["age_min"] == 30
    assert result.filters["age_max"] == 40
    assert result.filters["tumor_location"] == ["横结肠"]

def test_parse_unsupported_term_reports_warning():
    result = parse_database_query("30到40岁 有肝转移")
    assert "肝转移" in result.unsupported_terms
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_database_intent_service.py -v`
Expected: FAIL because parser service does not exist.

- [ ] **Step 3: Implement constrained parser**

Implementation notes:
- Parse only supported fields:
  - `patient_id`
  - `tumor_location`
  - `ct_stage`
  - `cn_stage`
  - `histology_type`
  - `mmr_status`
  - `age_min`
  - `age_max`
  - `cea_max`
- Prefer deterministic extraction first.
- Optional LLM assist can be used only to map user text into this fixed schema.
- Unsupported medical concepts must be returned explicitly, never silently dropped.

- [ ] **Step 4: Expose `POST /api/database/query-intent`**

Implementation notes:
- Route returns normalized query, parsed filters, unsupported terms, warnings.
- Frontend will apply filters from this response and then call search.

- [ ] **Step 5: Run parser and route tests**

Run: `pytest tests/backend/test_database_intent_service.py tests/backend/test_database_routes.py -v`
Expected: PASS with stable parser output.

- [ ] **Step 6: Commit**

```bash
git add backend/api/routes/database.py backend/api/schemas/database.py backend/api/services/database_intent_service.py tests/backend/test_database_intent_service.py tests/backend/test_database_routes.py
git commit -m "feat: add constrained database query intent parser"
```

## Task 3: Backend Upsert API

**Files:**
- Modify: `backend/api/routes/database.py`
- Modify: `backend/api/schemas/database.py`
- Modify: `backend/api/services/database_service.py`
- Modify: `src/services/virtual_database_service.py`
- Test: `tests/backend/test_database_routes.py`
- Test: `tests/backend/test_case_excel_service.py`

- [ ] **Step 1: Write failing tests for upsert save and detail refresh**

```python
def test_upsert_database_case_returns_updated_detail(client):
    response = client.post("/api/database/cases/upsert", json={
        "record": {
            "patient_id": 93,
            "age": 32,
            "tumor_location": "横结肠"
        }
    })
    assert response.status_code == 200
    assert response.json()["case_record"]["age"] == 32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/backend/test_database_routes.py::test_upsert_database_case_returns_updated_detail -v`
Expected: FAIL because route is missing.

- [ ] **Step 3: Implement upsert route and service**

Implementation notes:
- Route: `POST /api/database/cases/upsert`
- Service should validate payload through schema first.
- After save, return the refreshed detail payload so frontend can reuse existing cards immediately.
- Reuse current Excel writer instead of introducing a new persistence mechanism in MVP.

- [ ] **Step 4: Run backend tests**

Run: `pytest tests/backend/test_database_routes.py tests/backend/test_case_excel_service.py -v`
Expected: PASS without regressing Excel-backed persistence.

- [ ] **Step 5: Commit**

```bash
git add backend/api/routes/database.py backend/api/schemas/database.py backend/api/services/database_service.py src/services/virtual_database_service.py tests/backend/test_database_routes.py
git commit -m "feat: add database upsert API"
```

## Task 4: Frontend Database API Layer and Routing

**Files:**
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/router.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Test: `tests/frontend/workspace-page.test.tsx`

- [ ] **Step 1: Write failing frontend tests for navigation entry and client methods**

```tsx
test("workspace header links to database console", () => {
  render(<WorkspacePage />);
  expect(screen.getByRole("link", { name: /数据库/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/workspace-page.test.tsx`
Expected: FAIL because no database navigation exists.

- [ ] **Step 3: Add API types and client methods**

Implementation notes:
- Add client methods:
  - `getDatabaseStats()`
  - `searchDatabaseCases()`
  - `getDatabaseCaseDetail()`
  - `upsertDatabaseCase()`
  - `parseDatabaseQueryIntent()`
- Add corresponding request/response types in `types.ts`.

- [ ] **Step 4: Add `/database` route and workspace entry**

Implementation notes:
- Keep `/` as workspace.
- Add a visible route entry in the workspace header.
- Add a “返回工作台” link on the database page.

- [ ] **Step 5: Run frontend tests**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/workspace-page.test.tsx`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/router.tsx frontend/src/pages/workspace-page.tsx tests/frontend/workspace-page.test.tsx
git commit -m "feat: add database console routing and API client"
```

## Task 5: Database Console Page UI

**Files:**
- Create: `frontend/src/pages/database-page.tsx`
- Create: `frontend/src/features/database/database-natural-query-bar.tsx`
- Create: `frontend/src/features/database/database-filters-panel.tsx`
- Create: `frontend/src/features/database/database-results-table.tsx`
- Modify: `frontend/src/styles/globals.css`
- Test: `tests/frontend/database-page.test.tsx`

- [ ] **Step 1: Write failing page tests for stats load, search load, and row selection**

```tsx
test("database page loads stats and results", async () => {
  render(<DatabasePage />);
  expect(await screen.findByText(/总病例数/i)).toBeInTheDocument();
  expect(await screen.findByRole("table")).toBeInTheDocument();
});

test("clicking a row selects a patient", async () => {
  render(<DatabasePage />);
  const row = await screen.findByRole("row", { name: /93/ });
  fireEvent.click(row);
  expect(await screen.findByText(/患者 #93/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: FAIL because page and components do not exist.

- [ ] **Step 3: Build page composition**

Implementation notes:
- Page layout:
  - top: natural-language query bar
  - left: structured filters
  - center: stats + results table
  - right: patient detail panel
- Use local page state for:
  - filters
  - parse state
  - search result
  - selected patient id
  - selected patient detail

- [ ] **Step 4: Add results table, sort, and pagination**

Implementation notes:
- Table columns:
  - patient_id
  - gender
  - age
  - tumor_location
  - histology_type
  - clinical_stage
  - cea_level
  - mmr_status
- Row click loads detail.

- [ ] **Step 5: Run tests**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: PASS for page shell, stats, table, and selection behavior.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/database-page.tsx frontend/src/features/database/database-natural-query-bar.tsx frontend/src/features/database/database-filters-panel.tsx frontend/src/features/database/database-results-table.tsx frontend/src/styles/globals.css tests/frontend/database-page.test.tsx
git commit -m "feat: add database console page"
```

## Task 6: Detail Panel Card Reuse

**Files:**
- Create: `frontend/src/features/database/database-detail-panel.tsx`
- Modify: `frontend/src/features/cards/card-renderers.tsx` (only if a tiny export adjustment is needed)
- Test: `tests/frontend/database-page.test.tsx`

- [ ] **Step 1: Write failing test for detail panel card reuse**

```tsx
test("database detail panel reuses patient card renderer", async () => {
  render(<DatabasePage />);
  fireEvent.click(await screen.findByRole("row", { name: /93/ }));
  expect(await screen.findByText(/诊断信息/i)).toBeInTheDocument();
  expect(screen.getByText(/查看原始数据/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: FAIL because detail panel is not rendering cards.

- [ ] **Step 3: Implement detail panel by reusing existing card renderers**

Implementation notes:
- Import and reuse:
  - `cardTitle`
  - `renderCardContent`
- Do not rebuild patient/imaging/pathology UI in database feature files.
- Iterate `detail.cards` and pass backend payloads directly into `renderCardContent`.

- [ ] **Step 4: Run tests**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: PASS with patient detail visible through existing card UI.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/database/database-detail-panel.tsx tests/frontend/database-page.test.tsx
git commit -m "feat: reuse card renderers in database detail panel"
```

## Task 7: Edit and Save Flow

**Files:**
- Create: `frontend/src/features/database/database-edit-form.tsx`
- Modify: `frontend/src/pages/database-page.tsx`
- Test: `tests/frontend/database-page.test.tsx`

- [ ] **Step 1: Write failing test for inline edit/save**

```tsx
test("editing a patient and saving refreshes detail", async () => {
  render(<DatabasePage />);
  fireEvent.click(await screen.findByRole("row", { name: /93/ }));
  fireEvent.change(await screen.findByLabelText(/年龄/i), { target: { value: "32" } });
  fireEvent.click(screen.getByRole("button", { name: /保存/i }));
  expect(await screen.findByDisplayValue("32")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: FAIL because edit form does not exist.

- [ ] **Step 3: Implement minimal edit form**

Implementation notes:
- Editable MVP fields:
  - age
  - gender
  - ecog_score
  - histology_type
  - tumor_location
  - ct_stage
  - cn_stage
  - clinical_stage
  - cea_level
  - mmr_status
- On save:
  - call `upsertDatabaseCase()`
  - replace local detail payload
  - update current result row

- [ ] **Step 4: Run tests**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: PASS for edit/save flow.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/features/database/database-edit-form.tsx frontend/src/pages/database-page.tsx tests/frontend/database-page.test.tsx
git commit -m "feat: add database case editing"
```

## Task 8: Agentic UI Wiring

**Files:**
- Modify: `frontend/src/pages/database-page.tsx`
- Modify: `frontend/src/features/database/database-natural-query-bar.tsx`
- Test: `tests/frontend/database-page.test.tsx`

- [ ] **Step 1: Write failing test for NL parse -> filter sync -> auto-search**

```tsx
test("natural-language query fills filters and reruns search", async () => {
  render(<DatabasePage />);
  fireEvent.change(screen.getByPlaceholderText(/自然语言查询/i), {
    target: { value: "30到40岁 横结肠 pMMR" },
  });
  fireEvent.click(screen.getByRole("button", { name: /解析并筛选/i }));
  expect(await screen.findByDisplayValue("30")).toBeInTheDocument();
  expect(await screen.findByDisplayValue("40")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: FAIL because NL parse flow is not wired.

- [ ] **Step 3: Implement parse/apply UX**

Implementation notes:
- Parse button calls `parseDatabaseQueryIntent()`.
- Apply parsed filters into the structured form state.
- Show unsupported terms inline.
- Trigger search immediately after parse success.
- Keep structured filters as the single source of truth after parse.

- [ ] **Step 4: Run tests**

Run: `cd frontend`
Run: `npm run test -- --run ../tests/frontend/database-page.test.tsx`
Expected: PASS for parse/apply behavior.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/database-page.tsx frontend/src/features/database/database-natural-query-bar.tsx tests/frontend/database-page.test.tsx
git commit -m "feat: add agentic natural-language database filtering"
```

## Verification Checklist

- Backend:
  - `pytest tests/backend/test_database_routes.py -v`
  - `pytest tests/backend/test_database_intent_service.py -v`
  - `pytest tests/backend/test_case_excel_service.py -v`
- Frontend:
  - `cd frontend`
  - `npm run test -- --run ../tests/frontend/database-page.test.tsx`
  - `npm run test -- --run ../tests/frontend/workspace-page.test.tsx`
- Manual:
  - start backend
  - start frontend
  - open `/database`
  - verify stats load
  - run a structured search
  - run an NL query
  - click a row and confirm patient/imaging/pathology cards reuse existing UI
  - edit one field and confirm search result + detail both refresh

## Acceptance Criteria

- Users can browse more than one patient at a time without using the chat turn flow.
- `/database` results are filterable and pageable.
- Detail view uses the same patient/imaging/pathology cards as the workspace.
- NL query bar converts supported natural-language requests into visible structured filters.
- Unsupported medical concepts are surfaced as warnings instead of being silently ignored.
- Single-record save works against the current virtual Excel-backed database.
