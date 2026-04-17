# Chat Workspace Database Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an auto-opening database workbench to the chat workspace that supports overall stats, filtered search, patient detail inspection, and explicit form-confirmed writeback while keeping the existing `/database` page working.

**Architecture:** Extract the current `/database` page state machine into a shared frontend controller hook and render its center content through a reusable embedded workbench panel. Extend `node_case_database` so database-related chat turns patch `findings.database_workbench`, allowing `WorkspacePage` to open the shared workbench, preload search/detail state, and route patient detail plus edit form into the right inspector without introducing any new REST endpoints.

**Tech Stack:** Python, FastAPI, LangGraph, React, TypeScript, Vitest, Vite, pytest

---

**Repository note:** This workspace is not currently a git repository, so commit steps are written as checkpoints but cannot be executed until repository metadata is restored.

## File Structure

### Frontend files to create

- `D:\亿铸智能体\LangG_New\frontend\src\features\database\use-database-workbench.ts`
  - Shared stateful controller for stats, search, detail loading, editable draft state, natural-language parsing, save, and refresh behavior.
- `D:\亿铸智能体\LangG_New\frontend\src\features\database\database-workbench-panel.tsx`
  - Reusable center-panel UI for workbench header, natural query bar, stats cards, and search results table.

### Frontend files to modify

- `D:\亿铸智能体\LangG_New\frontend\src\app\api\types.ts`
  - Add typed `DatabaseWorkbenchContext` and any supporting mode/type aliases shared by the workspace page and database feature.
- `D:\亿铸智能体\LangG_New\frontend\src\pages\database-page.tsx`
  - Refactor to use the shared hook and shared workbench center panel while preserving the left structured filters and right detail/edit inspector.
- `D:\亿铸智能体\LangG_New\frontend\src\pages\workspace-page.tsx`
  - Render the embedded database workbench below the conversation panel, manage close/reopen rules, and swap the right inspector into database detail/edit mode when active.

### Backend files to modify

- `D:\亿铸智能体\LangG_New\src\prompts\database_prompts.py`
  - Remove prompt guidance that forces `upsert_patient_info` during chat-side edit requests and replace it with workbench-opening behavior.
- `D:\亿铸智能体\LangG_New\src\nodes\database_nodes.py`
  - Add helpers that derive `findings.database_workbench` from deterministic branches and tool calls, and ensure edit requests no longer auto-write to the database.

### Tests to modify

- `D:\亿铸智能体\LangG_New\tests\frontend\database-page.test.tsx`
  - Keep `/database` page behavior stable after the state extraction and add save-refresh coverage.
- `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`
  - Add chat-workspace workbench visibility, detail/edit, save-refresh, close/reopen, and duplicate-card suppression coverage.
- `D:\亿铸智能体\LangG_New\tests\backend\test_database_node.py`
  - Verify the database node emits the right workbench context for stats/search/detail/edit flows and does not auto-upsert on edit intent.

## Task 1: Extract a Shared Database Workbench Controller from `/database`

**Files:**
- Create: `D:\亿铸智能体\LangG_New\frontend\src\features\database\use-database-workbench.ts`
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\pages\database-page.tsx`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\database-page.test.tsx`

- [ ] **Step 1: Add a failing `/database` save-refresh regression test**

```tsx
test("database page refreshes stats and results after saving a case", async () => {
  const { DatabasePage } = await import("../../frontend/src/pages/database-page");
  const apiClient = createDatabaseApiClientStub();

  render(
    <AppProviders apiClient={apiClient}>
      <DatabasePage />
    </AppProviders>,
  );

  await screen.findByText("总病例数");
  const initialStatsCalls = apiClient.getDatabaseStats.mock.calls.length;
  const initialSearchCalls = apiClient.searchDatabaseCases.mock.calls.length;

  fireEvent.click(await screen.findByRole("button", { name: /查看 93/i }));
  await waitFor(() => expect(apiClient.getDatabaseCaseDetail).toHaveBeenCalledWith(93));

  fireEvent.change(await screen.findByLabelText("年龄"), {
    target: { value: "32" },
  });
  fireEvent.click(screen.getByRole("button", { name: "保存记录" }));

  await waitFor(() => expect(apiClient.upsertDatabaseCase).toHaveBeenCalled());
  await waitFor(() => expect(apiClient.getDatabaseStats).toHaveBeenCalledTimes(initialStatsCalls + 1));
  await waitFor(() => expect(apiClient.searchDatabaseCases).toHaveBeenCalledTimes(initialSearchCalls + 1));
});
```

- [ ] **Step 2: Run the focused `/database` page tests and verify failure**

Run:

```bash
npm run test -- --run ../tests/frontend/database-page.test.tsx
```

Expected: FAIL because the current save flow only refreshes the case detail and search results, not database stats.

- [ ] **Step 3: Implement `use-database-workbench.ts` with the current database-page state machine**

Create a focused hook that owns:

- `stats`
- `searchRequest`
- `searchResponse`
- `selectedPatientId`
- `detail`
- `editRecord`
- `naturalQuery`
- `intentWarnings`
- `unsupportedTerms`
- `pageError`
- loading flags for bootstrap, search, detail, parse, save

Core API sketch:

```ts
export function useDatabaseWorkbench(options?: {
  autoBootstrap?: boolean;
  initialRequest?: DatabaseSearchRequest;
  initialNaturalQuery?: string;
  initialSelectedPatientId?: number | null;
}) {
  return {
    stats,
    searchRequest,
    searchResponse,
    selectedPatientId,
    detail,
    editRecord,
    naturalQuery,
    intentWarnings,
    unsupportedTerms,
    pageError,
    isBootstrapping,
    isSearching,
    isLoadingDetail,
    isParsing,
    isSaving,
    setNaturalQuery,
    setFilters,
    runSearch,
    loadCaseDetail,
    saveRecord,
    resetWorkbench,
  };
}
```

Implementation requirements:

- keep helper functions such as `createDefaultFilters`, `createDefaultSearchRequest`, `normalizeIntentFilters`, `normalizeRecordForUpsert`, and `readPatientId` inside the new hook module unless a later step proves they should be split out
- after a successful save, refresh:
  - `getDatabaseStats()`
  - `searchDatabaseCases(searchRequest)`
  - selected patient detail if the save response does not already contain everything needed
- preserve the currently selected patient unless the refreshed result set no longer contains that patient

- [ ] **Step 4: Refactor `database-page.tsx` to consume the shared hook**

Keep the page layout the same:

- left rail: `DatabaseFiltersPanel`
- center: natural query bar, stats cards, results table
- right inspector: `DatabaseDetailPanel`, `DatabaseEditForm`

But remove duplicated local state and replace it with hook outputs/actions.

- [ ] **Step 5: Re-run the `/database` page tests**

Run:

```bash
npm run test -- --run ../tests/frontend/database-page.test.tsx
```

Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/features/database/use-database-workbench.ts frontend/src/pages/database-page.tsx tests/frontend/database-page.test.tsx
git commit -m "refactor: extract shared database workbench controller"
```

## Task 2: Emit `database_workbench` Context from the Database Node and Stop Auto-Upserting Chat Edits

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\prompts\database_prompts.py`
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\database_nodes.py`
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\app\api\types.ts`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_database_node.py`

- [ ] **Step 1: Add failing backend tests for stats/detail/search/edit workbench context**

```python
def test_case_database_sets_stats_workbench_context_from_statistics_tool():
    run_node = node_case_database(
        model=_ToolCallingModel(
            tool_name="get_database_statistics",
            tool_args={},
            final_content="数据库共有 2 例。",
        ),
        show_thinking=False,
    )

    result = run_node(
        CRCAgentState(
            messages=[HumanMessage(content="查看数据库整体统计")],
            findings={"user_intent": "case_database_query"},
        )
    )

    context = result["findings"]["database_workbench"]
    assert context["visible"] is True
    assert context["mode"] == "stats"
    assert context["selected_patient_id"] is None


def test_case_database_sets_detail_workbench_context_for_patient_query(monkeypatch):
    calls = []
    monkeypatch.setattr("src.nodes.database_nodes.ATOMIC_DATABASE_TOOLS", [
        _RecordingTool("get_patient_case_info", {"patient_id": 93, "gender": "男"}, calls),
    ])

    run_node = node_case_database(model=_StubModel(), show_thinking=False)
    result = run_node(
        CRCAgentState(
            messages=[HumanMessage(content="查看 93 号患者信息")],
            findings={"user_intent": "case_database_query"},
        )
    )

    context = result["findings"]["database_workbench"]
    assert context["mode"] == "detail"
    assert context["selected_patient_id"] == 93


def test_case_database_opens_edit_mode_without_calling_upsert(monkeypatch):
    calls = []
    monkeypatch.setattr("src.nodes.database_nodes.ATOMIC_DATABASE_TOOLS", [
        _RecordingTool("upsert_patient_info", {"message": "saved"}, calls),
        _RecordingTool("get_patient_case_info", {"patient_id": 93, "gender": "男"}, calls),
    ])

    run_node = node_case_database(model=_StubModel(), show_thinking=False)
    result = run_node(
        CRCAgentState(
            messages=[HumanMessage(content="我要修改 93 号患者的年龄")],
            findings={"user_intent": "case_database_query"},
        )
    )

    context = result["findings"]["database_workbench"]
    assert context["mode"] == "edit"
    assert context["selected_patient_id"] == 93
    assert all(name != "upsert_patient_info" for name, _payload in calls)
```

If the current node needs a tool-calling model stub for the `search_cases` path, add a fourth test that returns a `tool_calls` payload for `search_cases(...)` and assert `mode == "search"` plus the parsed filters.

- [ ] **Step 2: Run the focused backend database-node tests and verify failure**

Run:

```bash
pytest tests/backend/test_database_node.py -v
```

Expected: FAIL because the node does not yet write `findings.database_workbench`, and the edit branch still allows direct chat-side `upsert_patient_info` behavior.

- [ ] **Step 3: Add a typed frontend model for workbench context in `api/types.ts`**

Introduce minimal shared types:

```ts
export type DatabaseWorkbenchMode = "stats" | "search" | "detail" | "edit";

export interface DatabaseWorkbenchContext {
  visible: boolean;
  mode: DatabaseWorkbenchMode;
  query_text?: string | null;
  filters?: Partial<DatabaseFilters> | null;
  selected_patient_id?: number | null;
}
```

Use these only for typing and parsing. Do not change the stream protocol itself.

- [ ] **Step 4: Implement backend helpers in `database_nodes.py`**

Add small focused helpers, for example:

```python
def _build_database_workbench_context(
    *,
    mode: str,
    query_text: str | None,
    filters: dict | None = None,
    selected_patient_id: int | None = None,
) -> dict:
    return {
        "visible": True,
        "mode": mode,
        "query_text": query_text,
        "filters": filters,
        "selected_patient_id": selected_patient_id,
    }


def _merge_database_workbench_findings(
    findings: dict | None,
    context: dict,
) -> dict:
    next_findings = dict(findings or {})
    next_findings["database_workbench"] = context
    return next_findings
```

Apply these helpers in:

- deterministic patient detail branch
- deterministic edit-intent branch
- tool-handling branches for:
  - `get_database_statistics`
  - `search_cases`
  - `get_patient_case_info`

Required behavioral changes:

- edit/update chat requests should open workbench edit mode, not call `upsert_patient_info`
- if no valid patient id is available for an edit request, fall back to `mode="search"` and keep the assistant response explicit about needing patient selection
- keep existing patient-card/image-card behavior intact where already present

- [ ] **Step 5: Update `database_prompts.py` so the LLM no longer auto-writes on edit requests**

Replace the current “must call `upsert_patient_info`” guidance with something equivalent to:

```text
When the user wants to add or edit database records in chat, gather or confirm the target patient if needed, answer normally, and let the frontend workbench handle the actual save after explicit human confirmation.
```

Do not remove legitimate tool guidance for stats/search/detail queries.

- [ ] **Step 6: Re-run the backend database-node tests**

Run:

```bash
pytest tests/backend/test_database_node.py -v
```

Expected: PASS

- [ ] **Step 7: Checkpoint**

```bash
git add src/prompts/database_prompts.py src/nodes/database_nodes.py frontend/src/app/api/types.ts tests/backend/test_database_node.py
git commit -m "feat: emit database workbench context for chat database turns"
```

## Task 3: Embed the Shared Workbench Below the Conversation Panel

**Files:**
- Create: `D:\亿铸智能体\LangG_New\frontend\src\features\database\database-workbench-panel.tsx`
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\pages\workspace-page.tsx`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`

- [ ] **Step 1: Add a failing workspace bootstrap test for embedded workbench visibility**

```tsx
test("opens the embedded database workbench from bootstrap findings context", async () => {
  const apiClient = renderWorkspace(async () =>
    makeSessionResponse({
      snapshot: makeSnapshot({
        findings: {
          database_workbench: {
            visible: true,
            mode: "search",
            query_text: "帮我找出横结肠病例",
            filters: { tumor_location: ["横"] },
            selected_patient_id: null,
          },
        },
      }),
    }),
  );

  expect(await screen.findByText("数据库工作台")).toBeInTheDocument();
  await waitFor(() => expect(apiClient.getDatabaseStats).toHaveBeenCalledTimes(1));
  await waitFor(() =>
    expect(apiClient.searchDatabaseCases).toHaveBeenCalledWith(
      expect.objectContaining({
        filters: expect.objectContaining({ tumor_location: ["横"] }),
      }),
    ),
  );
});
```

- [ ] **Step 2: Run the focused workspace tests and verify failure**

Run:

```bash
npm run test -- --run ../tests/frontend/workspace-page.test.tsx
```

Expected: FAIL because `WorkspacePage` does not yet render any database workbench or bootstrap from `findings.database_workbench`.

- [ ] **Step 3: Implement `database-workbench-panel.tsx` as a reusable center-panel component**

Render:

- compact header with title, mode badge, and close button
- `DatabaseNaturalQueryBar`
- stats cards
- `DatabaseResultsTable`

API sketch:

```tsx
export function DatabaseWorkbenchPanel(props: {
  title?: string;
  mode: DatabaseWorkbenchMode;
  naturalQuery: string;
  stats: DatabaseStatsResponse | null;
  searchRequest: DatabaseSearchRequest;
  searchResponse: DatabaseSearchResponse | null;
  selectedPatientId: number | null;
  isParsing: boolean;
  isSearching: boolean;
  isBootstrapping: boolean;
  warnings: string[];
  unsupportedTerms: string[];
  error: string | null;
  onNaturalQueryChange: (value: string) => void;
  onNaturalQuerySubmit: () => void;
  onSelectPatient: (patientId: number) => void;
  onSortChange: (field: DatabaseSortField) => void;
  onPageChange: (page: number) => void;
  onClose?: () => void;
}) {
  ...
}
```

Keep this component presentation-focused. Data fetching stays in the hook.

- [ ] **Step 4: Render the workbench below `ConversationPanel` in `workspace-page.tsx`**

Implementation outline:

- read `database_workbench` from `sessionState.findings`
- create a derived initial request from the context filters
- instantiate `useDatabaseWorkbench({ autoBootstrap: workbenchVisible, ... })`
- mount the panel only when a database workbench is visible for the current thread

Suggested close/reopen pattern:

```ts
const contextSignature = JSON.stringify(databaseWorkbenchContext ?? null);
const [dismissedSignature, setDismissedSignature] = useState<string | null>(null);

const isWorkbenchVisible =
  databaseWorkbenchContext?.visible === true &&
  contextSignature !== dismissedSignature;
```

When the user clicks close, store the current signature in `dismissedSignature`. When a later database turn updates the context, the signature changes and the panel reopens automatically.

- [ ] **Step 5: Re-run the focused workspace tests**

Run:

```bash
npm run test -- --run ../tests/frontend/workspace-page.test.tsx
```

Expected: PASS for the new bootstrap-open scenario and any unaffected existing workspace tests.

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/features/database/database-workbench-panel.tsx frontend/src/pages/workspace-page.tsx tests/frontend/workspace-page.test.tsx
git commit -m "feat: embed database workbench in workspace center"
```

## Task 4: Load Patient Detail and Save Edits from the Workspace Right Inspector

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\pages\workspace-page.tsx`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`

- [ ] **Step 1: Add a failing workspace test for patient detail and save-writeback**

```tsx
test("workspace workbench loads patient detail and saves edits from the right inspector", async () => {
  const apiClient = renderWorkspace(async () =>
    makeSessionResponse({
      snapshot: makeSnapshot({
        findings: {
          database_workbench: {
            visible: true,
            mode: "edit",
            query_text: "修改 93 号患者",
            filters: { patient_id: 93 },
            selected_patient_id: 93,
          },
        },
      }),
    }),
  );

  await waitFor(() => expect(apiClient.getDatabaseCaseDetail).toHaveBeenCalledWith(93));
  expect(await screen.findByText("诊断信息")).toBeInTheDocument();

  fireEvent.change(await screen.findByLabelText("年龄"), {
    target: { value: "32" },
  });
  fireEvent.click(screen.getByRole("button", { name: "保存记录" }));

  await waitFor(() =>
    expect(apiClient.upsertDatabaseCase).toHaveBeenCalledWith(
      expect.objectContaining({
        record: expect.objectContaining({
          patient_id: 93,
          age: 32,
        }),
      }),
    ),
  );
  await waitFor(() => expect(apiClient.getDatabaseStats).toHaveBeenCalledTimes(2));
});
```

- [ ] **Step 2: Run the focused workspace tests and verify failure**

Run:

```bash
npm run test -- --run ../tests/frontend/workspace-page.test.tsx
```

Expected: FAIL because the workspace inspector does not yet swap into database detail/edit mode or connect save behavior to the shared workbench controller.

- [ ] **Step 3: Route the right inspector into database detail/edit mode**

When the workbench is active:

- if `selectedPatientId` exists, render:
  - `DatabaseDetailPanel`
  - `DatabaseEditForm`
- otherwise keep the current inspector stack

Use the shared hook actions:

- `loadCaseDetail(patientId)` when a row is selected
- `saveRecord()` for the form save button
- `setEditRecord(...)` or an equivalent field-change handler from the hook

Keep the workbench center panel and right inspector in the same controller instance so list/detail/save refreshes stay synchronized.

- [ ] **Step 4: Re-run the focused workspace tests**

Run:

```bash
npm run test -- --run ../tests/frontend/workspace-page.test.tsx
```

Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add frontend/src/pages/workspace-page.tsx tests/frontend/workspace-page.test.tsx
git commit -m "feat: support patient detail editing in workspace inspector"
```

## Task 5: Prevent Duplicate Case Cards and Preserve Close/Reopen Semantics

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\pages\workspace-page.tsx`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`

- [ ] **Step 1: Add failing workspace tests for duplicate-card suppression and reopen behavior**

```tsx
test("filters overlapping clinical cards when database detail is active", async () => {
  renderWorkspace(async () =>
    makeSessionResponse({
      snapshot: makeSnapshot({
        findings: {
          database_workbench: {
            visible: true,
            mode: "detail",
            selected_patient_id: 93,
          },
        },
        cards: [
          {
            type: "card.upsert",
            card_type: "patient_card",
            payload: {
              type: "patient_card",
              patient_id: 93,
              data: { patient_info: { age: 31 } },
            },
            source_channel: "state",
          },
        ],
      }),
    }),
  );

  await screen.findByText("数据库工作台");
  expect(screen.queryByTestId("clinical-cards-panel")).not.toHaveTextContent("病人#93");
});


test("reopens the database workbench when a later findings patch carries a new context", async () => {
  const apiClient = renderWorkspace(
    async () => makeSessionResponse(),
    {
      streamTurn: vi.fn(async (_sessionId, _request, onEvent) => {
        onEvent({
          type: "findings.patch",
          patch: {
            database_workbench: {
              visible: true,
              mode: "search",
              query_text: "查 pMMR 患者",
              filters: { mmr_status: ["pMMR_MSS"] },
            },
          },
        });
        onEvent({
          type: "done",
          thread_id: "thread_1",
          run_id: "run_db_workbench_reopen",
          snapshot_version: 1,
        });
      }),
    },
  );

  await waitFor(() => expect(apiClient.createSession).toHaveBeenCalledTimes(1));
  fireEvent.change(screen.getByPlaceholderText("请输入病例问题、报告内容或诊疗需求"), {
    target: { value: "帮我查 pMMR 患者" },
  });
  fireEvent.click(screen.getByRole("button", { name: /发送/ }));

  expect(await screen.findByText("数据库工作台")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the focused workspace tests and verify failure**

Run:

```bash
npm run test -- --run ../tests/frontend/workspace-page.test.tsx
```

Expected: FAIL because the page still shows duplicate patient-card content and does not yet treat `findings.patch.database_workbench` as a reopen signal after local dismissal.

- [ ] **Step 3: Filter overlapping cards only while a database patient detail is active**

In `workspace-page.tsx`, derive a filtered card set before rendering `ClinicalCardsPanel`:

```ts
const overlappingCardTypes = new Set(["patient_card", "imaging_card", "pathology_slide_card"]);
const shouldFilterCaseCards = isWorkbenchVisible && selectedPatientId !== null;

const visibleCards = shouldFilterCaseCards
  ? Object.fromEntries(
      Object.entries(sessionState.cards).filter(([cardType]) => !overlappingCardTypes.has(cardType)),
    )
  : sessionState.cards;
```

Also derive `selectedCardType` from `visibleCards`, not from the unfiltered `sessionState.cards`.

- [ ] **Step 4: Keep close/reopen behavior local to the current thread**

Rules to implement:

- close hides the panel locally without mutating `sessionState.findings`
- reset thread clears the local dismissal state
- any later database workbench context with a different signature reopens the panel

Use local state only. Do not add new backend endpoints or persistence for dismissal.

- [ ] **Step 5: Re-run the focused workspace tests**

Run:

```bash
npm run test -- --run ../tests/frontend/workspace-page.test.tsx
```

Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/pages/workspace-page.tsx tests/frontend/workspace-page.test.tsx
git commit -m "feat: polish workspace database workbench visibility rules"
```

## Task 6: Focused Regression Sweep and Manual End-to-End Retest

**Files:**
- Modify as needed based on failures
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_database_node.py`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\database-page.test.tsx`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`

- [ ] **Step 1: Run the focused backend suite**

Run:

```bash
pytest tests/backend/test_database_node.py -v
```

Expected: PASS

- [ ] **Step 2: Run the focused frontend suite**

Run:

```bash
npm run test -- --run ../tests/frontend/database-page.test.tsx ../tests/frontend/workspace-page.test.tsx
```

Expected: PASS

- [ ] **Step 3: Manually retest the workspace flow against the local app**

Verify these end-to-end cases manually:

1. Ask a database stats question in `/` and confirm the embedded workbench auto-opens with stats + list.
2. Ask for a single patient in `/` and confirm the right inspector shows patient detail + editable form.
3. Modify a field, click `保存记录`, and confirm the updated value persists after refresh.
4. Close the workbench, then ask another database-related question and confirm it reopens.
5. Open `/database` and confirm the page still supports search, detail, and save exactly as before.

- [ ] **Step 4: Record any environment blockers instead of claiming success without evidence**

Examples:

- frontend tests blocked by missing Node dependencies
- backend tests blocked by missing Python packages
- manual app retest blocked by the local backend/frontend not running

- [ ] **Step 5: Final checkpoint**

```bash
git add src frontend tests
git commit -m "feat: add workspace database workbench with confirmed writeback"
```

## Execution Notes

- Keep the backend change narrowly scoped to `findings.database_workbench`; do not invent a new stream event type unless a concrete reducer limitation appears.
- The shared frontend hook is the main leverage point. Do not copy `/database` page state into `WorkspacePage`.
- Preserve the existing `/database` page layout and behavior while refactoring; the page should become a consumer of shared logic, not a rewritten experience.
- The writeback rule is strict: chat can open edit mode, but only the visible form save button may call `upsertDatabaseCase`.
- If tests show that `findings.patch` merging already behaves correctly, avoid unnecessary edits to `stream-reducer.ts`.
