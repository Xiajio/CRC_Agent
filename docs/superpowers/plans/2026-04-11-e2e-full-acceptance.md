# E2E Full Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable full-acceptance harness that exercises the real frontend and backend against controlled fixture graph/data behavior, captures evidence, and supports one 8-hour release gate.

**Architecture:** Keep the product surfaces real, but make the acceptance environment deterministic. The backend should accept per-turn fixture-case overrides for graph replay, boot against a seeded acceptance database, and use a fixture upload converter for stable derived cards. The frontend acceptance suite should run through a dedicated Playwright config with shared drivers/evidence helpers, while manual review uses lightweight docs and templates under `docs/superpowers/acceptance/`.

**Tech Stack:** TypeScript, Playwright, Vite, React, Python, FastAPI, pytest, PowerShell

---

**Repository note:** This workspace is not currently a git repository, so commit steps are written as checkpoints but cannot be executed until repository metadata is restored.

## File Structure

### Backend and fixture files to create

- `D:\亿铸智能体\LangG_New\backend\api\services\testing\acceptance_case_db.py`
  - Build a deterministic virtual case database tree from a small JSON seed so the database page can run against controlled data.
- `D:\亿铸智能体\LangG_New\backend\api\services\upload_fixture_cards.py`
  - Load deterministic derived medical-card payloads for known acceptance upload samples.
- `D:\亿铸智能体\LangG_New\scripts\prepare_acceptance_case_db.py`
  - CLI wrapper that materializes the acceptance case database into a runtime folder.
- `D:\亿铸智能体\LangG_New\scripts\start_backend_acceptance_fixture.ps1`
  - Start the backend in fixture graph mode with the seeded acceptance database and fixture upload conversion enabled.
- `D:\亿铸智能体\LangG_New\scripts\run_e2e_full_acceptance.ps1`
  - One-command launcher for the L0-L2 automated acceptance pack.
- `D:\亿铸智能体\LangG_New\tests\fixtures\acceptance_case_db\seed.json`
  - Source-of-truth patient rows and asset references for the acceptance database.
- `D:\亿铸智能体\LangG_New\tests\fixtures\acceptance_case_db\imaging\093\I1000000.png`
  - Stable imaging sample copied into the generated acceptance database.
- `D:\亿铸智能体\LangG_New\tests\fixtures\uploads\acceptance-note.txt`
  - Deterministic upload sample used for upload and upload-follow-up E2E coverage.
- `D:\亿铸智能体\LangG_New\tests\fixtures\uploads\acceptance-summary.txt`
  - Second deterministic upload sample used for duplicate/reuse and follow-up assertions.
- `D:\亿铸智能体\LangG_New\tests\fixtures\upload_cards\acceptance-note.json`
  - Fixture derived-card payload loaded when `UPLOAD_CONVERTER_MODE=fixture`.
- `D:\亿铸智能体\LangG_New\tests\fixtures\upload_cards\acceptance-summary.json`
  - Additional deterministic derived-card payload for a second upload sample.
- `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\knowledge_case.json`
  - Golden replay for the knowledge-query acceptance path.
- `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\offtopic_date_case.json`
  - Golden replay for single-turn off-topic redirect without hidden-reasoning leakage.
- `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\offtopic_date_after_plan_case.json`
  - Golden replay for the multi-turn 'decision then date question' hidden-reasoning regression.
- `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\upload_followup_case.json`
  - Golden replay for the 'upload then ask follow-up' graph turn.
- `D:\亿铸智能体\LangG_New\tests\backend\test_acceptance_case_db.py`
  - Tests for acceptance database materialization.
- `D:\亿铸智能体\LangG_New\tests\frontend\acceptance-fixtures.test.ts`
  - Fast unit coverage for fixture metadata used by Playwright support helpers.

### Backend and fixture files to modify

- `D:\亿铸智能体\LangG_New\backend\api\services\payload_builder.py`
  - Pass through an allowlisted subset of `ChatTurnRequest.context` into the graph payload so Playwright can choose fixture cases per turn.
- `D:\亿铸智能体\LangG_New\backend\api\services\upload_service.py`
  - Switch to deterministic fixture-card conversion when acceptance mode is enabled.
- `D:\亿铸智能体\LangG_New\scripts\capture_graph_fixtures.py`
  - Add the acceptance fixture presets required by the full suite.
- `D:\亿铸智能体\LangG_New\tests\backend\test_payload_builder.py`
  - Lock context allowlisting and override behavior.
- `D:\亿铸智能体\LangG_New\tests\backend\test_capture_graph_fixtures.py`
  - Lock the fixture preset catalog and prompt inventory.
- `D:\亿铸智能体\LangG_New\tests\backend\test_uploads_routes.py`
  - Lock fixture upload conversion behavior end-to-end through the route.
- `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\README.md`
  - Document the full acceptance fixture inventory and capture commands.

### Frontend and E2E files to create

- `D:\亿铸智能体\LangG_New\frontend\playwright.acceptance.config.ts`
  - Dedicated Playwright config for the full acceptance pack with separate output and backend startup.
- `D:\亿铸智能体\LangG_New\tests\e2e\support\acceptance-fixtures.ts`
  - Typed map of fixture-case names, prompts, expected markers, and forbidden leakage phrases.
- `D:\亿铸智能体\LangG_New\tests\e2e\support\backend-session.ts`
  - Helpers to fetch session snapshots and message history from the BFF during E2E runs.
- `D:\亿铸智能体\LangG_New\tests\e2e\support\evidence.ts`
  - Helpers that attach screenshots and JSON evidence bundles to Playwright test output.
- `D:\亿铸智能体\LangG_New\tests\e2e\support\workspace-driver.ts`
  - Workspace selectors, prompt submission helpers, and request interception for per-turn fixture selection.
- `D:\亿铸智能体\LangG_New\tests\e2e\support\database-driver.ts`
  - Database-page selectors and common actions for search, detail, and save flows.
- `D:\亿铸智能体\LangG_New\tests\e2e\support\upload-driver.ts`
  - Upload panel selectors and asset verification helpers.
- `D:\亿铸智能体\LangG_New\tests\e2e\acceptance\workspace-core.spec.ts`
  - L1 automated acceptance for workspace bootstrap, patient query, decision, knowledge, off-topic redirect, safety, reset, and error recovery.
- `D:\亿铸智能体\LangG_New\tests\e2e\acceptance\database-console.spec.ts`
  - L2 automated acceptance for database NL query, structured filtering, detail view, and upsert.
- `D:\亿铸智能体\LangG_New\tests\e2e\acceptance\uploads-and-cards.spec.ts`
  - L2 automated acceptance for upload, upload-follow-up, card prompt actions, and persisted asset evidence.

### Frontend and E2E files to modify

- `D:\亿铸智能体\LangG_New\frontend\package.json`
  - Add a dedicated acceptance script so the full pack can be invoked consistently.

### Acceptance docs to create

- `D:\亿铸智能体\LangG_New\docs\superpowers\acceptance\e2e-full-acceptance-runbook.md`
  - Operator runbook for the 8-hour acceptance execution.
- `D:\亿铸智能体\LangG_New\docs\superpowers\acceptance\e2e-manual-review-checklist.md`
  - Manual sign-off checklist for medical wording, visual quality, and trust/safety review.
- `D:\亿铸智能体\LangG_New\docs\superpowers\acceptance\e2e-release-report-template.md`
  - Final PASS / PASS WITH CONDITIONS / FAIL report template.

## Task 1: Plumb Per-Turn Fixture Selection Through Graph Payloads

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\backend\api\services\payload_builder.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_payload_builder.py`

- [ ] **Step 1: Write the failing payload-builder override test**

```python
def test_build_graph_payload_merges_allowlisted_context_fields():
    session_meta = _make_session_meta()

    result = build_graph_payload(
        chat_request={
            "message": {"role": "user", "content": "current turn"},
            "context": {
                "fixture_case": "decision_case",
                "fixture_tick_delay_ms": 25,
                "current_patient_id": "093",
                "ignored": "do-not-pass-through",
            },
        },
        session_meta=session_meta,
        state_snapshot={"current_patient_id": "001"},
    )

    assert result.payload["fixture_case"] == "decision_case"
    assert result.payload["fixture_tick_delay_ms"] == 25
    assert result.payload["current_patient_id"] == "093"
    assert "ignored" not in result.payload
```

- [ ] **Step 2: Run the payload-builder test to verify it fails**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_payload_builder.py -k allowlisted_context_fields -v`  
Expected: FAIL because `build_graph_payload()` currently ignores `chat_request["context"]`.

- [ ] **Step 3: Implement an allowlisted context merge in `payload_builder.py`**

Keep the merge narrow and explicit:

```python
CONTEXT_PAYLOAD_ALLOWLIST = {
    "fixture_case",
    "fixture_tick_delay_ms",
    "current_patient_id",
}
```

Only copy keys in that allowlist. Do not pass arbitrary `context` through to the graph payload.

- [ ] **Step 4: Re-run the payload-builder tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_payload_builder.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add backend/api/services/payload_builder.py tests/backend/test_payload_builder.py
git commit -m "feat: allow acceptance fixture overrides in graph payloads"
```

## Task 2: Expand the Graph Fixture Catalog for Full Acceptance Coverage

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\scripts\capture_graph_fixtures.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_capture_graph_fixtures.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\README.md`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\knowledge_case.json`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\offtopic_date_case.json`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\offtopic_date_after_plan_case.json`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\graph_ticks\upload_followup_case.json`

- [ ] **Step 1: Write the failing fixture-preset inventory test**

```python
from scripts.capture_graph_fixtures import CASE_PRESETS


def test_capture_script_exposes_acceptance_fixture_presets():
    assert "knowledge_case" in CASE_PRESETS
    assert "offtopic_date_case" in CASE_PRESETS
    assert "offtopic_date_after_plan_case" in CASE_PRESETS
    assert "upload_followup_case" in CASE_PRESETS
    assert CASE_PRESETS["offtopic_date_after_plan_case"]["prompt"]
```

- [ ] **Step 2: Run the capture-script test to verify it fails**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_capture_graph_fixtures.py -k acceptance_fixture_presets -v`  
Expected: FAIL because the acceptance presets do not exist yet.

- [ ] **Step 3: Add the new fixture presets to `capture_graph_fixtures.py`**

Use prompts aligned with the acceptance design:

```python
"knowledge_case": {
    "thread_id": "capture-knowledge-case",
    "prompt": "What does cT3 mean in colorectal cancer staging?",
    ...
}
"offtopic_date_case": {
    "thread_id": "capture-offtopic-date-case",
    "prompt": "What date is it today?",
    ...
}
"offtopic_date_after_plan_case": {
    "thread_id": "capture-offtopic-after-plan-case",
    "prompt": "Can you check what date it is today?",
    "state": {
        "current_patient_id": "093",
        "clinical_stage": "Decision",
        "assessment_draft": "Existing CRC assessment draft.",
        "findings": {"pathology_confirmed": True},
    },
}
"upload_followup_case": {
    "thread_id": "capture-upload-followup-case",
    "prompt": "Please use the uploaded note as context and summarize the CRC-relevant findings.",
    ...
}
```

The `offtopic_date_after_plan_case` must represent the second-turn state after a prior decision-oriented interaction. Do not capture it as a fresh empty session.

- [ ] **Step 4: Update the fixture README before capturing**

Document:

- the new supported case names
- the exact preset prompts
- which acceptance case each fixture serves
- that `offtopic_date_after_plan_case` is a second-turn regression capture

- [ ] **Step 5: Capture the new golden fixtures from real runs**

Run each command separately with UTF-8 enabled:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
D:\anaconda3\envs\LangG\python.exe scripts\capture_graph_fixtures.py --case knowledge_case --output tests/fixtures/graph_ticks/knowledge_case.json
```

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
D:\anaconda3\envs\LangG\python.exe scripts\capture_graph_fixtures.py --case offtopic_date_case --output tests/fixtures/graph_ticks/offtopic_date_case.json
```

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
D:\anaconda3\envs\LangG\python.exe scripts\capture_graph_fixtures.py --case offtopic_date_after_plan_case --output tests/fixtures/graph_ticks/offtopic_date_after_plan_case.json
```

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
D:\anaconda3\envs\LangG\python.exe scripts\capture_graph_fixtures.py --case upload_followup_case --output tests/fixtures/graph_ticks/upload_followup_case.json
```

Expected: each command writes a UTF-8 JSON file under `tests/fixtures/graph_ticks/`.

- [ ] **Step 6: Re-run the capture-script tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_capture_graph_fixtures.py -v`  
Expected: PASS

- [ ] **Step 7: Checkpoint**

```bash
git add scripts/capture_graph_fixtures.py tests/backend/test_capture_graph_fixtures.py tests/fixtures/graph_ticks/README.md tests/fixtures/graph_ticks/knowledge_case.json tests/fixtures/graph_ticks/offtopic_date_case.json tests/fixtures/graph_ticks/offtopic_date_after_plan_case.json tests/fixtures/graph_ticks/upload_followup_case.json
git commit -m "test: add full acceptance graph fixtures"
```

## Task 3: Materialize a Deterministic Acceptance Database

**Files:**
- Create: `D:\亿铸智能体\LangG_New\backend\api\services\testing\acceptance_case_db.py`
- Create: `D:\亿铸智能体\LangG_New\scripts\prepare_acceptance_case_db.py`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\acceptance_case_db\seed.json`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\acceptance_case_db\imaging\093\I1000000.png`
- Create: `D:\亿铸智能体\LangG_New\tests\backend\test_acceptance_case_db.py`

- [ ] **Step 1: Write the failing acceptance-database materialization test**

```python
def test_materialize_acceptance_case_db_creates_workbook_and_imaging_tree(tmp_path: Path):
    output_root = tmp_path / "acceptance-db"

    result = materialize_acceptance_case_db(
        seed_path=Path("tests/fixtures/acceptance_case_db/seed.json"),
        output_root=output_root,
    )

    assert result == output_root
    assert (output_root / "Clinical Case" / "classification.xlsx").exists()
    assert (output_root / "Radiographic Imaging" / "093" / "I1000000.png").exists()
```

- [ ] **Step 2: Run the materialization test to verify it fails**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_acceptance_case_db.py -v`  
Expected: FAIL because the helper module and seed fixture do not exist yet.

- [ ] **Step 3: Implement the acceptance database materializer**

Create a focused helper with an interface like:

```python
def materialize_acceptance_case_db(*, seed_path: Path, output_root: Path) -> Path:
    ...
```

Use the same workbook shape already exercised in `tests/backend/test_database_routes.py`. Keep the seed human-editable in JSON; do not hand-maintain `.xlsx` binaries in the repository.

- [ ] **Step 4: Add the CLI wrapper script**

`prepare_acceptance_case_db.py` should accept:

```text
--seed
--output
```

and call `materialize_acceptance_case_db()`.

- [ ] **Step 5: Re-run the acceptance database test**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_acceptance_case_db.py -v`  
Expected: PASS

- [ ] **Step 6: Run the materializer once by hand**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe scripts\prepare_acceptance_case_db.py --seed tests/fixtures/acceptance_case_db/seed.json --output runtime/acceptance_case_db
```

Expected: `runtime/acceptance_case_db/Clinical Case/classification.xlsx` and the imaging subtree exist.

- [ ] **Step 7: Checkpoint**

```bash
git add backend/api/services/testing/acceptance_case_db.py scripts/prepare_acceptance_case_db.py tests/fixtures/acceptance_case_db/seed.json tests/fixtures/acceptance_case_db/imaging/093/I1000000.png tests/backend/test_acceptance_case_db.py
git commit -m "test: add deterministic acceptance database materializer"
```

## Task 4: Add Deterministic Upload Conversion for Acceptance Mode

**Files:**
- Create: `D:\亿铸智能体\LangG_New\backend\api\services\upload_fixture_cards.py`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\uploads\acceptance-note.txt`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\uploads\acceptance-summary.txt`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\upload_cards\acceptance-note.json`
- Create: `D:\亿铸智能体\LangG_New\tests\fixtures\upload_cards\acceptance-summary.json`
- Modify: `D:\亿铸智能体\LangG_New\backend\api\services\upload_service.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_uploads_routes.py`

- [ ] **Step 1: Write the failing fixture-upload conversion test**

```python
def test_convert_upload_to_medical_card_uses_fixture_cards_when_acceptance_mode_enabled(monkeypatch):
    monkeypatch.setenv("UPLOAD_CONVERTER_MODE", "fixture")

    card = convert_upload_to_medical_card(
        b"Acceptance note",
        "acceptance-note.txt",
        "text/plain",
    )

    assert card["type"] == "medical_visualization_card"
    assert card["data"]["patient_summary"]["name"] == "Acceptance Patient 093"
```

- [ ] **Step 2: Run the upload-route test to verify it fails**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_uploads_routes.py -k fixture_cards_when_acceptance_mode_enabled -v`  
Expected: FAIL because upload conversion still goes through the live document converter path.

- [ ] **Step 3: Implement fixture-card loading in `upload_service.py`**

Keep the switch explicit:

```python
if os.getenv("UPLOAD_CONVERTER_MODE", "").strip().lower() == "fixture":
    return load_fixture_upload_card(filename)
```

`load_fixture_upload_card()` should:

- resolve the sanitized filename
- load the matching JSON under `tests/fixtures/upload_cards/`
- raise a clear error if the sample is missing

Do not overload the live converter with acceptance-only branching beyond this small mode switch.

- [ ] **Step 4: Re-run the upload-route tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_uploads_routes.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add backend/api/services/upload_fixture_cards.py backend/api/services/upload_service.py tests/fixtures/uploads/acceptance-note.txt tests/fixtures/uploads/acceptance-summary.txt tests/fixtures/upload_cards/acceptance-note.json tests/fixtures/upload_cards/acceptance-summary.json tests/backend/test_uploads_routes.py
git commit -m "test: add deterministic acceptance upload conversion"
```

## Task 5: Add the Acceptance Playwright Harness and Shared Support Helpers

**Files:**
- Create: `D:\亿铸智能体\LangG_New\frontend\playwright.acceptance.config.ts`
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\support\acceptance-fixtures.ts`
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\support\backend-session.ts`
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\support\evidence.ts`
- Modify: `D:\亿铸智能体\LangG_New\frontend\package.json`
- Create: `D:\亿铸智能体\LangG_New\tests\frontend\acceptance-fixtures.test.ts`
- Create: `D:\亿铸智能体\LangG_New\scripts\start_backend_acceptance_fixture.ps1`

- [ ] **Step 1: Write the failing fixture-metadata unit test**

```ts
import { describe, expect, it } from "vitest";

import { ACCEPTANCE_FIXTURES } from "../e2e/support/acceptance-fixtures";

describe("ACCEPTANCE_FIXTURES", () => {
  it("includes the full workspace acceptance inventory", () => {
    expect(ACCEPTANCE_FIXTURES.knowledge_case.prompt).toMatch(/staging/i);
    expect(ACCEPTANCE_FIXTURES.offtopic_date_after_plan_case.forbiddenPhrases).toContain("根据我的指导原则");
    expect(ACCEPTANCE_FIXTURES.safety_case.expectedEvents).toContain("safety.alert");
  });
});
```

- [ ] **Step 2: Run the unit test to verify it fails**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run ../tests/frontend/acceptance-fixtures.test.ts`  
Expected: FAIL because the support file does not exist yet.

- [ ] **Step 3: Create the acceptance support modules**

`acceptance-fixtures.ts` should export a typed map like:

```ts
export const ACCEPTANCE_FIXTURES = {
  database_case: { ... },
  decision_case: { ... },
  knowledge_case: { ... },
  safety_case: { ... },
  offtopic_date_case: { ... },
  offtopic_date_after_plan_case: { ... },
  upload_followup_case: { ... },
} as const;
```

`backend-session.ts` should expose helpers to fetch:

- `/api/sessions/:id`
- `/api/sessions/:id/messages`

`evidence.ts` should attach:

- screenshot
- session snapshot JSON
- message history JSON

to the current Playwright `testInfo`.

- [ ] **Step 4: Add the acceptance Playwright config and backend startup script**

`playwright.acceptance.config.ts` should:

- point `testDir` to `../tests/e2e/acceptance`
- write output under `../output/acceptance/`
- use `scripts/start_backend_acceptance_fixture.ps1`
- keep `scripts/start_frontend.ps1` as the frontend server

`start_backend_acceptance_fixture.ps1` should:

- materialize `runtime/acceptance_case_db`
- set `GRAPH_RUNNER_MODE=fixture`
- set `GRAPH_FIXTURE_CASE=database_case`
- set `UPLOAD_CONVERTER_MODE=fixture`
- set `CASE_DATABASE_PATH` to the generated acceptance database
- start `uvicorn backend.app:app`

- [ ] **Step 5: Add a dedicated package script**

Add:

```json
"test:e2e:acceptance": "playwright test --config playwright.acceptance.config.ts"
```

- [ ] **Step 6: Re-run the fixture-metadata unit test**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run ../tests/frontend/acceptance-fixtures.test.ts`  
Expected: PASS

- [ ] **Step 7: Checkpoint**

```bash
git add frontend/playwright.acceptance.config.ts frontend/package.json tests/e2e/support/acceptance-fixtures.ts tests/e2e/support/backend-session.ts tests/e2e/support/evidence.ts tests/frontend/acceptance-fixtures.test.ts scripts/start_backend_acceptance_fixture.ps1
git commit -m "test: add acceptance playwright harness"
```

## Task 6: Implement the Core Workspace Acceptance Pack

**Files:**
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\support\workspace-driver.ts`
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\acceptance\workspace-core.spec.ts`
- Modify: `D:\亿铸智能体\LangG_New\tests\e2e\support\backend-session.ts`
- Modify: `D:\亿铸智能体\LangG_New\tests\e2e\support\evidence.ts`

- [ ] **Step 1: Write the failing core acceptance spec**

Cover these cases in `workspace-core.spec.ts`:

```ts
test("workspace bootstrap and patient query stay consistent with persisted history", async ({ page }) => {
  ...
});

test("decision flow emits plan and references without leaking reasoning", async ({ page }) => {
  ...
});

test("knowledge query routes to knowledge fixture instead of patient database output", async ({ page }) => {
  ...
});

test("off-topic redirect after a decision turn does not leak reasoning text", async ({ page }) => {
  ...
});

test("safety flow surfaces a blocking safety alert", async ({ page }) => {
  ...
});

test("stream failure recovers without ghost assistant messages", async ({ page }) => {
  ...
});
```

- [ ] **Step 2: Run the core spec to verify it fails**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/workspace-core.spec.ts`  
Expected: FAIL because the workspace driver and acceptance assertions do not exist yet.

- [ ] **Step 3: Implement the workspace driver**

The driver should:

- wrap workspace selectors
- submit prompts through the real UI
- intercept `POST /api/sessions/*/messages/stream` and inject `context.fixture_case`
- allow changing the requested fixture case between turns

Implementation shape:

```ts
export async function installFixtureCaseOverride(page: Page) {
  let activeFixtureCase = "database_case";
  await page.route("**/api/sessions/*/messages/stream", async (route) => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    const context = typeof body.context === "object" && body.context ? { ...(body.context as Record<string, unknown>) } : {};
    context.fixture_case = activeFixtureCase;
    await route.continue({ postData: JSON.stringify({ ...body, context }) });
  });
  return {
    setFixtureCase(next: AcceptanceFixtureCase) {
      activeFixtureCase = next;
    },
  };
}
```

- [ ] **Step 4: Add persisted-history and leakage assertions**

For each core case:

- assert the visible assistant text
- fetch `/api/sessions/:id/messages?limit=100`
- assert the last stored assistant message matches the visible result
- assert forbidden reasoning phrases are absent from both UI and persisted history

For the multi-turn regression, use:

1. `decision_case`
2. `offtopic_date_after_plan_case`

in the same browser session.

- [ ] **Step 5: Re-run the core spec**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/workspace-core.spec.ts`  
Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add tests/e2e/support/workspace-driver.ts tests/e2e/acceptance/workspace-core.spec.ts tests/e2e/support/backend-session.ts tests/e2e/support/evidence.ts
git commit -m "test: add core workspace acceptance coverage"
```

## Task 7: Implement the Database Console Acceptance Pack

**Files:**
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\support\database-driver.ts`
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\acceptance\database-console.spec.ts`

- [ ] **Step 1: Write the failing database-console acceptance spec**

Cover these cases:

```ts
test("database page loads seeded statistics and result table", async ({ page }) => {
  ...
});

test("database natural-language query populates filters and results", async ({ page }) => {
  ...
});

test("database structured filters and detail view stay stable", async ({ page }) => {
  ...
});

test("database upsert persists after reload", async ({ page }) => {
  ...
});
```

- [ ] **Step 2: Run the database spec to verify it fails**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/database-console.spec.ts`  
Expected: FAIL because the database driver and seeded assertions do not exist yet.

- [ ] **Step 3: Implement the database driver**

The driver should expose:

- open database page
- submit natural-language query
- apply a structured filter
- open patient detail
- edit and save a field
- reload and re-check the saved value

Keep selectors centralized in the driver so the acceptance spec stays scenario-focused.

- [ ] **Step 4: Re-run the database spec**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/database-console.spec.ts`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add tests/e2e/support/database-driver.ts tests/e2e/acceptance/database-console.spec.ts
git commit -m "test: add database console acceptance coverage"
```

## Task 8: Implement the Upload and Card-Follow-Up Acceptance Pack

**Files:**
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\support\upload-driver.ts`
- Create: `D:\亿铸智能体\LangG_New\tests\e2e\acceptance\uploads-and-cards.spec.ts`
- Modify: `D:\亿铸智能体\LangG_New\tests\e2e\support\workspace-driver.ts`

- [ ] **Step 1: Write the failing upload/card acceptance spec**

Cover these cases:

```ts
test("upload stores an asset and exposes deterministic derived-card context", async ({ page }) => {
  ...
});

test("post-upload follow-up uses the upload followup fixture without corrupting the session", async ({ page }) => {
  ...
});

test("card prompt actions trigger the expected follow-up turn", async ({ page }) => {
  ...
});
```

- [ ] **Step 2: Run the upload/card spec to verify it fails**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/uploads-and-cards.spec.ts`  
Expected: FAIL because the upload driver and deterministic assertions do not exist yet.

- [ ] **Step 3: Implement the upload driver**

The driver should:

- attach files from `tests/fixtures/uploads/`
- wait for upload status completion
- fetch the session snapshot and uploaded assets
- fetch `/api/assets/:asset_id` when needed

Reuse the core workspace evidence helpers instead of duplicating JSON attachment logic.

- [ ] **Step 4: Add the upload-follow-up fixture-case switch**

After upload completes, switch the workspace fixture override to `upload_followup_case` before submitting the follow-up prompt. Assert:

- the uploaded asset is still present in the snapshot
- the follow-up assistant message is visible
- no reasoning text leaks

- [ ] **Step 5: Re-run the upload/card spec**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/uploads-and-cards.spec.ts`  
Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add tests/e2e/support/upload-driver.ts tests/e2e/acceptance/uploads-and-cards.spec.ts tests/e2e/support/workspace-driver.ts
git commit -m "test: add upload and card follow-up acceptance coverage"
```

## Task 9: Add the Full-Acceptance Runner and Manual Review Artifacts

**Files:**
- Create: `D:\亿铸智能体\LangG_New\scripts\run_e2e_full_acceptance.ps1`
- Create: `D:\亿铸智能体\LangG_New\docs\superpowers\acceptance\e2e-full-acceptance-runbook.md`
- Create: `D:\亿铸智能体\LangG_New\docs\superpowers\acceptance\e2e-manual-review-checklist.md`
- Create: `D:\亿铸智能体\LangG_New\docs\superpowers\acceptance\e2e-release-report-template.md`

- [ ] **Step 1: Run the missing runner script to confirm the gap**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -ListOnly`  
Expected: FAIL because the runner script does not exist yet.

- [ ] **Step 2: Implement the acceptance runner**

`run_e2e_full_acceptance.ps1` should support:

- `-ListOnly`
- `-CoreOnly`
- `-SkipManualDocs`

At minimum it must:

1. run the focused backend acceptance-support tests
2. run `npm run test:e2e:acceptance`
3. print where Playwright evidence and docs live

- [ ] **Step 3: Write the runbook and checklists**

The runbook should contain:

- L0 prerequisites and environment checks
- exact commands for L1 and L2
- evidence locations
- manual review handoff order
- blocker policy

The manual checklist should contain flat pass/fail/note lines for:

- medical wording
- visual quality
- card semantics
- image/pathology preview usability
- trust and safety presentation

The release report template should contain:

- environment metadata
- automation summary
- manual sign-offs
- blocker list
- final decision

- [ ] **Step 4: Verify the runner help path**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -ListOnly`  
Expected: PASS and prints the backend test command, the Playwright command, and the docs paths without executing the full suite.

- [ ] **Step 5: Checkpoint**

```bash
git add scripts/run_e2e_full_acceptance.ps1 docs/superpowers/acceptance/e2e-full-acceptance-runbook.md docs/superpowers/acceptance/e2e-manual-review-checklist.md docs/superpowers/acceptance/e2e-release-report-template.md
git commit -m "docs: add full acceptance runbook and launcher"
```

## Task 10: Run the Automated Acceptance Pack and Record Handoff Notes

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\docs\superpowers\plans\2026-04-11-e2e-full-acceptance.md`

- [ ] **Step 1: Run the backend acceptance-support regression suite**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest `
  tests/backend/test_payload_builder.py `
  tests/backend/test_capture_graph_fixtures.py `
  tests/backend/test_acceptance_case_db.py `
  tests/backend/test_uploads_routes.py `
  -v
```

Expected: PASS

- [ ] **Step 2: Run the frontend acceptance support test**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run ../tests/frontend/acceptance-fixtures.test.ts`  
Expected: PASS

- [ ] **Step 3: Run the full automated acceptance pack**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1`  
Expected: PASS, with Playwright evidence written under `output/acceptance/`.

- [ ] **Step 4: Record handoff notes in this plan**

Add a short completion note under this task covering:

- which fixture cases were captured and frozen
- whether the per-turn fixture override remained test-only and allowlisted
- where the acceptance evidence bundle was written
- whether any L2 cases were downgraded to conditional pass

**Completion Note (YYYY-MM-DD):**

- Frozen fixture cases: `database_case`, `decision_case`, `knowledge_case`, `safety_case`, `offtopic_date_case`, `offtopic_date_after_plan_case`, and `upload_followup_case`.
- The backend still only passes through the allowlisted context keys `fixture_case`, `fixture_tick_delay_ms`, and `current_patient_id`; arbitrary `context` fields remain blocked.
- Automated evidence bundle location: `output/acceptance/`.
- Conditional-pass exceptions: `<none or list explicitly>`.

**Execution Update (2026-04-13):**

- Task 2 fixture generation was converted to a deterministic mixed model: `database_case` remains live-captured, while `decision_case`, `safety_case`, `knowledge_case`, `offtopic_date_case`, `offtopic_date_after_plan_case`, and `upload_followup_case` are now script-generated template fixtures validated against `normalize_tick`.
- The template fixtures now persist public assistant messages as fixture `messages`, not just `assessment_draft`, so UI output and `GET /api/sessions/.../messages` history stay consistent in acceptance mode.
- Backend acceptance-support regression suite passed: `34 passed`, with only the pre-existing `src/services/document_converter.py` Pydantic deprecation warnings.
- Frontend acceptance fixture unit test passed: `tests/frontend/acceptance-fixtures.test.ts`.
- The previously skipped Task 2 workspace and upload acceptance cases were promoted to active coverage and now pass:
  - `workspace-core.spec.ts`: decision, safety, knowledge, and off-topic-after-plan
  - `uploads-and-cards.spec.ts`: upload follow-up and card prompt actions
- Full launcher passed after stabilizing the database acceptance driver to wait for `POST /api/database/cases/search` after structured-filter apply, in addition to the existing wait for `POST /api/database/cases/upsert` before reload.
- Automated evidence bundle location: `output/acceptance/`.
- Conditional-pass exceptions: none.

- [ ] **Step 5: Checkpoint**

```bash
git add docs/superpowers/plans/2026-04-11-e2e-full-acceptance.md
git commit -m "docs: record full acceptance execution handoff"
```

## Implementation Notes

- Use `@superpowers:test-driven-development` discipline inside every task. Keep tests small and focused before expanding scenarios.
- Use `@playwright` for the E2E tasks rather than ad hoc browser scripting.
- Use `@superpowers:verification-before-completion` before claiming the acceptance harness is complete.
- Do not hand-author graph tick JSON. Every `tests/fixtures/graph_ticks/*.json` file in this plan must come from `scripts/capture_graph_fixtures.py`.
- Do not let the backend pass arbitrary `context` keys into the graph payload. The acceptance suite needs deterministic fixture selection, not an open-ended backdoor.
- Keep the acceptance database seed human-readable in JSON and regenerate workbook artifacts at runtime. Do not check in mutable `.xlsx` fixture outputs.
