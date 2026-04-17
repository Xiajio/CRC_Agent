# Doctor Workspace Database Tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the doctor scene into two secondary tabs, `Consultation Workspace` and `Database`, while preserving doctor-side general clinical consultation without a bound patient and patient-bound consultation when a current patient exists.

**Architecture:** Keep the existing patient/doctor scene split intact, but make two targeted backend behavior adjustments before the doctor-tab UI refactor: remap doctor-side routing away from patient triage, and introduce scene-aware assessment entry points or wrappers so patient and doctor assessment policies can diverge cleanly. Then move doctor-only tab state and long-lived doctor browsing hooks into a dedicated `DoctorSceneShell`, narrow `useDatabaseWorkbench()` back to historical-case responsibilities, add a dedicated `useRegistryBrowser()` for patient-registry preview, and treat `Set As Current Patient` as the only action that mutates the doctor session. This plan intentionally absorbs the intended frontend behavior from the PDF-intake credible-closure plan instead of implementing that older frontend task separately.

**Tech Stack:** Python, LangGraph, FastAPI, React, TypeScript, Vitest, Vite

---

**Repository note:** This workspace is not currently a git repository, so commit steps are checkpoints only and cannot be executed until repository metadata is restored.

**Execution dependency:** Before starting this plan, the backend-facing registry surfaces from the PDF-intake credibility pass must already exist and pass locally:

- `GET /api/patient-registry/patients/{id}`
- `GET /api/patient-registry/patients/{id}/records`
- `GET /api/patient-registry/patients/{id}/alerts`
- `PATCH /api/sessions/{doctorSessionId}`

If those backend prerequisites are not green, stop and finish the PDF-intake backend phase first. Do **not** separately execute the older PDF-intake frontend Task 5 after this plan begins.

## Current Frontend Baseline

The current workspace already contains a substantial portion of the doctor-tab frontend refactor. Treat these as the starting baseline, not future tasks:

- `DoctorSceneShell` already exists and is rendered from `workspace-page.tsx` for the doctor scene.
- `DoctorDatabaseView` already exists and already hosts:
  - `Historical Case Base`
  - `Patient Registry`
- `useRegistryBrowser()` already exists and already owns:
  - recent-patient discovery
  - registry search
  - preview detail / alerts / records
- `usePatientRegistry()` is already narrowed to current bound-patient detail / records / alerts plus `bindPatient()`.
- Registry preview is already separate from bind.
- `Set As Current Patient` already exists and already returns the UI to the consultation tab.
- Existing frontend tests already cover:
  - doctor secondary tabs
  - registry preview vs bind
  - registry-preview state preservation
  - historical-case workbench still being available

Because that frontend baseline is already landed, this plan should not re-implement those structures. The remaining work is to align the backend routing semantics and upgrade the consultation tab from the old no-patient empty state to `General Clinical Mode`, then refresh the tests around that new behavior.

## File Structure

### Frontend files to create

No new frontend files are required by the remaining delta in this plan.

- None.

### Frontend files to modify

- `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
  - Keep rendering `DoctorSceneShell` when the active scene is `doctor`; only update props and behavior if the consultation-mode refinement needs it.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\database\use-database-workbench.ts`
  - Preserve the historical-only workbench role; clean up any remaining legacy compatibility shim only if it still blocks the refined consultation/database split.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-patient-registry.ts`
  - Preserve the current bound-patient-only role while adapting consultation mode behavior if needed.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\recent-patients-panel.tsx`
  - Preserve preview-oriented registry discovery semantics; only adjust if the database-tab UX needs cleanup.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-scene-shell.tsx`
  - Keep the doctor secondary-tab shell, but update it so the consultation tab no longer falls back to the old no-patient empty state.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-consultation-view.tsx`
  - Upgrade the consultation tab from the current empty-state behavior to explicit `General Clinical Mode` vs `Patient-Bound Mode`.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-database-view.tsx`
  - Preserve the existing database-tab structure and preview/bind flow while adjusting only if consultation-mode changes require prop cleanup.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\use-doctor-view-state.ts`
  - Keep the existing doctor-tab UI state unless implementation discovers a missing state transition for the refined consultation mode.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-registry-browser.ts`
  - Preserve current registry-browsing behavior; no structural rewrite is expected.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\registry-browser-pane.tsx`
  - Preserve current registry preview / explicit bind behavior; only touch if the consultation-mode CTA or copy needs alignment.
- `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`
  - Extend only if the refined consultation-mode tests need additional stubs.
- `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`
  - Update the existing doctor-scene tests so they reflect the current frontend baseline plus the new `General Clinical Mode` behavior.

### Backend files to modify

- `D:\YiZhu_Agnet\LangG\src\graph_builder.py`
  - Add a doctor-side route remap so doctor turns cannot enter patient outpatient-triage flow, while leaving patient-side routing intact.
- `D:\YiZhu_Agnet\LangG\src\nodes\assessment_nodes.py`
  - Introduce scene-aware assessment entry points or wrappers so patient-side assessment remains inquiry-oriented and doctor-side assessment keeps full clinical reasoning.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`
  - Extend doctor-scene coverage around unbound-patient graph behavior and current-patient injection.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_scene_sessions.py`
  - Keep session behavior coverage stable while the doctor scene gains general-clinical consultation mode.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_builder_routing.py`
  - Add focused routing tests for doctor remap and patient database-query blocking.

### Existing files intentionally not modified

- `D:\YiZhu_Agnet\LangG\frontend\src\app\api\client.ts`
  - Existing registry and bind client methods are sufficient.
- `D:\YiZhu_Agnet\LangG\frontend\src\app\api\types.ts`
  - Existing registry and session types are sufficient unless implementation discovers a missing UI-only type alias worth adding locally.

## Test Harness Notes

- Frontend tests should continue to use `D:\anaconda3\pkgs\nodejs-25.8.2-h80d1838_0\npm.cmd --prefix frontend run test -- --run --reporter=verbose`.
- Backend tests should continue to use `D:\anaconda3\envs\LangG\python.exe -m pytest ...`.
- Keep the new doctor-tab tests in `tests/frontend/workspace-scenes.test.tsx` unless the file becomes hard to reason about; only split into a new test file if the existing file loses readability.
- Preserve TDD order: failing test, verify fail, minimal implementation, verify pass, then broaden regression.
- Because tab-state preservation is central to the redesign, avoid mounting stateful hooks inside leaf views that unmount during tab switches.

## Task 1: Remap Doctor Routing Away From Patient Triage

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\src\graph_builder.py`
- Create: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_builder_routing.py`

- [ ] **Step 1: Write a failing routing test for doctor-side symptom prompts**

```python
def make_state(*, findings: dict[str, object] | None = None, current_plan: list[object] | None = None):
    return CRCAgentState(
        messages=[],
        findings=findings or {},
        current_plan=current_plan or [],
    )


def test_doctor_route_after_intent_remaps_clinical_entry_to_assessment():
    state = make_state(
        findings={"user_intent": "general_chat"},
        current_plan=[
            SimpleNamespace(
                status="pending",
                parallel_group=None,
                id="ask-1",
                description="Clarify the presenting symptom",
                tool_needed="ask_user",
                retry_count=0,
            )
        ],
    )

    assert route_after_intent(state) == "clinical_entry_resolver"
    assert route_after_doctor_intent(state) == "assessment"
```

- [ ] **Step 2: Verify the failing backend test**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_graph_builder_routing.py -v`

Expected: FAIL because the doctor graph still exposes the patient triage branch.

- [ ] **Step 3: Add a doctor-specific intent remap in `graph_builder.py`**

Implementation target:

- keep patient-side routing unchanged
- on the doctor side, remap `clinical_entry_resolver` to `assessment`
- do not allow doctor-side routing to reach `outpatient_triage`

- [ ] **Step 4: Re-run the focused backend test**

Expected: PASS, and patient-side `route_after_patient_intent()` behavior remains unchanged.

- [ ] **Step 5: Checkpoint**

```bash
git add src/graph_builder.py tests/backend/test_graph_builder_routing.py
git commit -m "fix: remap doctor routing away from outpatient triage"
```

## Task 2: Introduce Scene-Aware Assessment Entry Points

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\src\nodes\assessment_nodes.py`
- Modify: `D:\YiZhu_Agnet\LangG\src\graph_builder.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`

- [ ] **Step 1: Write failing coverage for patient-vs-doctor assessment semantics**

```python
@pytest.mark.asyncio
async def test_doctor_assessment_runs_without_bound_patient_context() -> None:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="doctor", patient_id=None)
    graph = FakeStreamingGraph()
    service = DoctorGraphService(
        compiled_graph=graph,
        session_store=session_store,
        patient_registry=FakePatientRegistry(summary_message=HumanMessage(content="Bound patient summary: patient_id=33")),
        heartbeat_interval_seconds=0,
    )

    await collect_sse_events(service.stream_turn(meta.session_id, make_chat_request("请给出局部进展期直肠癌治疗思路")))

    payload_messages = graph.last_payload["messages"]
    assert not any(
        isinstance(message, HumanMessage) and "Bound patient summary:" in str(message.content)
        for message in payload_messages
    )


def test_patient_assessment_wrapper_is_distinct_from_doctor_assessment_wrapper():
    patient_entry = node_patient_assessment(model=StubModel(), tools=[], streaming=False, show_thinking=False)
    doctor_entry = node_doctor_assessment(model=StubModel(), tools=[], streaming=False, show_thinking=False)

    assert patient_entry is not doctor_entry
```

- [ ] **Step 2: Verify the failing backend tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_graph_service_streaming.py -v`

Expected: FAIL because both scenes still share the same assessment entry.

- [ ] **Step 3: Add scene-aware wrappers or entry points**

Implementation target:

- preserve one assessment core where possible
- introduce `doctor_assessment` and `patient_assessment` wrappers or prompt variants
- keep doctor-side full clinical reasoning available without `patient_id`
- keep patient-side assessment inquiry-oriented
- preferred implementation path: keep the assessment core scene-agnostic and add two thin wrapper entry points in `assessment_nodes.py`; wire those different wrapper functions from `graph_builder.py` instead of adding scene checks throughout the full assessment core

- [ ] **Step 4: Wire doctor and patient graphs to the new entry points**

- [ ] **Step 5: Re-run the focused backend tests**

Expected: PASS, with doctor general-clinical mode intact and patient behavior constrained.

- [ ] **Step 6: Checkpoint**

```bash
git add src/nodes/assessment_nodes.py src/graph_builder.py tests/backend/test_graph_service_streaming.py
git commit -m "refactor: add scene-aware assessment entry points"
```

## Task 3: Refine the Existing Doctor Secondary Tabs and Shell

**Files:**
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\use-doctor-view-state.ts`
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-scene-shell.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Verify the existing doctor-scene tab coverage still reflects the current baseline**

```tsx
test("doctor scene renders Consultation Workspace and Database tabs", async () => {
  ...
});
```

- [ ] **Step 2: Re-run the existing focused frontend test**

Run: `D:\anaconda3\pkgs\nodejs-25.8.2-h80d1838_0\npm.cmd --prefix frontend run test -- --run --reporter=verbose ../tests/frontend/workspace-scenes.test.tsx`

- [ ] **Step 3: Adjust `DoctorSceneShell` / `useDoctorViewState()` only if the current implementation is missing a state transition needed by the refined consultation mode**

- [ ] **Step 4: Keep `workspace-page.tsx` aligned with the existing doctor shell contract**

- [ ] **Step 5: Re-run the focused frontend test**

Expected: PASS, confirming the existing doctor secondary-tab baseline before changing consultation behavior.

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/features/doctor/use-doctor-view-state.ts frontend/src/features/doctor/doctor-scene-shell.tsx frontend/src/pages/workspace-page.tsx tests/frontend/workspace-scenes.test.tsx
git commit -m "test: confirm doctor secondary-tab baseline"
```

## Task 4: Replace the Current Consultation Empty State With General Clinical Mode

**Files:**
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-consultation-view.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-scene-shell.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-patient-registry.ts`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Write failing tests for consultation modes**

```tsx
test("doctor consultation shows General Clinical Mode when no patient is bound", async () => {
  const apiClient = buildApiClientStub({
    createSession: vi.fn(async (scene: "patient" | "doctor") =>
      makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : null })),
  });

  renderWorkspaceWithSceneSessions(apiClient);
  fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));

  expect(await screen.findByText(/general clinical mode/i)).toBeInTheDocument();
  expect(screen.getByRole("textbox")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /open database > patient registry/i })).toBeInTheDocument();
});

test("doctor consultation shows bound-patient context when a patient is bound", async () => {
  const apiClient = buildApiClientStub({
    createSession: vi.fn(async (scene: "patient" | "doctor") =>
      makeSessionResponse({ scene, session_id: `sess-${scene}`, patient_id: scene === "patient" ? 201 : 33 })),
    getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
    getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
    getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
  });

  renderWorkspaceWithSceneSessions(apiClient);
  fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));

  expect(await screen.findByText(/patient #33/i)).toBeInTheDocument();
  expect(screen.queryByText(/general clinical mode/i)).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Verify the failing frontend tests**

- [ ] **Step 3: Implement `DoctorConsultationView` with two modes**

Implementation target:

- when `currentPatientId === null`, render:
  - `General Clinical Mode`
  - doctor conversation panel
  - roadmap / cards / plan
  - CTA to open `Database > Patient Registry`
- when `currentPatientId !== null`, render:
  - current patient summary
  - patient alerts / records
  - the same conversation panel

Current frontend note:

- the existing frontend already has doctor secondary tabs and registry preview/bind
- the remaining gap is that the consultation tab still falls back to `ConsultationEmptyState` when no patient is bound
- this task should convert that existing empty-state branch into the new general-clinical consultation mode rather than reworking the whole doctor tab structure

- [ ] **Step 4: Keep `usePatientRegistry()` focused on bound-patient context only**

- [ ] **Step 5: Re-run the focused frontend tests**

Expected: PASS for both consultation modes.

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/features/doctor/doctor-consultation-view.tsx frontend/src/features/doctor/doctor-scene-shell.tsx frontend/src/features/patient-registry/use-patient-registry.ts tests/frontend/workspace-scenes.test.tsx
git commit -m "feat: add doctor general and bound consultation modes"
```

## Task 5: Confirm Historical Browsing and Registry Preview Flow Against the Current Frontend Baseline

**Files:**
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-database-view.tsx`
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-registry-browser.ts`
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\registry-browser-pane.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\database\use-database-workbench.ts`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\recent-patients-panel.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Write failing tests for database-tab source ownership and preview-only registry behavior**

```tsx
test("database source switching lives inside the Database tab", async () => {
  const apiClient = buildApiClientStub({
    getDatabaseStats: vi.fn(async () => makeDatabaseStatsResponse()),
    searchDatabaseCases: vi.fn(async () => makeDatabaseSearchResponse()),
    getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
  });

  renderWorkspaceWithSceneSessions(apiClient);
  fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
  fireEvent.click(screen.getByRole("button", { name: /^database$/i }));

  expect(await screen.findByRole("button", { name: /historical case base/i })).toBeInTheDocument();
  expect(await screen.findByRole("button", { name: /patient registry/i })).toBeInTheDocument();
});

test("registry preview does not bind the doctor session", async () => {
  const bindPatient = vi.fn();
  const apiClient = buildApiClientStub({
    getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
    getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
    getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
    getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
    bindPatient,
  });

  renderWorkspaceWithSceneSessions(apiClient);
  fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
  fireEvent.click(screen.getByRole("button", { name: /^database$/i }));
  fireEvent.click(await screen.findByRole("button", { name: /patient registry/i }));
  fireEvent.click(await screen.findByRole("button", { name: /preview patient 33/i }));

  expect(await screen.findByText(/registry preview: patient #33/i)).toBeInTheDocument();
  expect(bindPatient).not.toHaveBeenCalled();
});
```

- [ ] **Step 2: Verify the failing frontend tests**

- [ ] **Step 3: Verify `useDatabaseWorkbench()` remains historical-only in practice**

- [ ] **Step 4: Verify `useRegistryBrowser()` / `RegistryBrowserPane` still own registry preview behavior**

Implementation target:

- preserve recent-patient discovery
- preserve registry search
- preserve preview detail / alerts / records
- preserve no doctor-session mutation on row click

- [ ] **Step 4a: Update or replace the legacy scope-switching test**

The existing frontend test named `"changing workbench scope reboots the doctor data source"` should not survive unchanged.  
Its behavior now belongs to `DoctorDatabaseView` source switching, not `useDatabaseWorkbench()`. Replace it with doctor-database-tab coverage instead of carrying the old `scope` contract forward.

- [ ] **Step 5: Verify old toolbar-level historical/registry scope buttons are gone and do not reappear during consultation-mode changes**

- [ ] **Step 6: Re-run the focused frontend tests**

Expected: PASS, confirming the current database-tab ownership before and after consultation-mode work.

- [ ] **Step 7: Checkpoint**

```bash
git add frontend/src/features/doctor/doctor-database-view.tsx frontend/src/features/patient-registry/use-registry-browser.ts frontend/src/features/patient-registry/registry-browser-pane.tsx frontend/src/features/database/use-database-workbench.ts frontend/src/features/patient-registry/recent-patients-panel.tsx frontend/src/pages/workspace-page.tsx tests/frontend/test-utils.tsx tests/frontend/workspace-scenes.test.tsx
git commit -m "test: lock in doctor database tab baseline"
```

## Task 6: Keep Explicit Bind and Database-State Preservation Intact While Refining Consultation Mode

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-scene-shell.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\doctor\doctor-database-view.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\registry-browser-pane.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Write failing tests for `Set As Current Patient` and state preservation**

```tsx
test("Set As Current Patient binds and returns to Consultation Workspace", async () => {
  const bindPatient = vi.fn(async (sessionId: string, patientId: number) =>
    makeSessionResponse({
      scene: "doctor",
      session_id: sessionId,
      patient_id: patientId,
      snapshot: { current_patient_id: patientId },
    }));
  const apiClient = buildApiClientStub({
    bindPatient,
    getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
    getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
    getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
    getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
  });

  renderWorkspaceWithSceneSessions(apiClient);
  fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
  fireEvent.click(screen.getByRole("button", { name: /^database$/i }));
  fireEvent.click(await screen.findByRole("button", { name: /patient registry/i }));
  fireEvent.click(await screen.findByRole("button", { name: /preview patient 33/i }));
  fireEvent.click(await screen.findByRole("button", { name: /set current patient 33/i }));

  await waitFor(() => expect(bindPatient).toHaveBeenCalledWith("sess-doctor", 33));
  expect(await screen.findByRole("button", { name: /consultation workspace/i })).toBeInTheDocument();
  expect(await screen.findByText(/patient #33/i)).toBeInTheDocument();
});

test("database tab preserves registry preview state after bind and return", async () => {
  const apiClient = buildApiClientStub({
    getRecentPatients: vi.fn(async () => makePatientRegistryListResponse()),
    getPatientRegistryDetail: vi.fn(async () => makePatientRegistryDetail({ patient_id: 33 })),
    getPatientRecords: vi.fn(async () => makePatientRegistryRecordsResponse()),
    getPatientRegistryAlerts: vi.fn(async () => makePatientRegistryAlertsResponse()),
  });

  renderWorkspaceWithSceneSessions(apiClient);
  fireEvent.click(await screen.findByRole("button", { name: /doctor scene/i }));
  fireEvent.click(screen.getByRole("button", { name: /^database$/i }));
  fireEvent.click(await screen.findByRole("button", { name: /patient registry/i }));
  fireEvent.click(await screen.findByRole("button", { name: /preview patient 33/i }));
  fireEvent.click(screen.getByRole("button", { name: /consultation workspace/i }));
  fireEvent.click(screen.getByRole("button", { name: /^database$/i }));

  expect(await screen.findByText(/registry preview: patient #33/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Verify the failing frontend tests**

- [ ] **Step 3: Wire `Set As Current Patient` through the existing bind endpoint**

- [ ] **Step 4: Preserve the current `DoctorSceneShell` hook-mounting pattern so state survives tab switches**

- [ ] **Step 5: Re-run the focused frontend tests**

Expected: PASS, proving the current preview→bind flow still survives the consultation-mode refinement.

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/features/doctor/doctor-scene-shell.tsx frontend/src/features/doctor/doctor-database-view.tsx frontend/src/features/patient-registry/registry-browser-pane.tsx tests/frontend/workspace-scenes.test.tsx
git commit -m "feat: bind registry preview into doctor consultation flow"
```

## Task 7: Regression and Manual Doctor Flow Verification

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_builder_routing.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`

- [ ] **Step 1: Add final backend and frontend regression assertions**

Coverage targets:

- patient-side `case_database_query` remains blocked
- doctor-side unbound consultation still runs
- doctor-side routing no longer reaches outpatient triage
- historical-case browsing still works after the tab refactor

- [ ] **Step 2: Run the full backend suite relevant to this refactor**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_graph_builder_routing.py tests/backend/test_graph_service_streaming.py tests/backend/test_scene_sessions.py -v`

- [ ] **Step 3: Run the full frontend suite**

Run: `D:\anaconda3\pkgs\nodejs-25.8.2-h80d1838_0\npm.cmd --prefix frontend run test -- --run --reporter=verbose`

- [ ] **Step 4: Perform manual browser verification**

Verify this exact flow:

1. Open the doctor scene and confirm it defaults to `Consultation Workspace`.
2. Confirm `General Clinical Mode` appears if no patient is bound, and that the doctor conversation panel is still visible.
3. Ask a non-patient-specific doctor question and confirm the doctor flow still responds.
4. Use the CTA to reach `Database > Patient Registry`.
5. Preview a registry patient without changing the current doctor session.
6. Click `Set As Current Patient` and confirm the view returns to `Consultation Workspace`.
7. Confirm the bound patient summary, alerts, and records refresh.
8. Switch back to `Database` and confirm the previous registry preview still exists.
9. Switch to `Historical Case Base` and confirm historical-case search still works.

- [ ] **Step 5: Checkpoint**

```bash
git add tests/backend/test_graph_builder_routing.py tests/backend/test_graph_service_streaming.py tests/backend/test_scene_sessions.py tests/frontend/workspace-scenes.test.tsx tests/frontend/test-utils.tsx
git commit -m "test: cover doctor general and bound consultation modes"
```

## Handoff Notes

- This plan supersedes the intended doctor-facing frontend work in `D:\YiZhu_Agnet\LangG\docs\superpowers\plans\2026-04-17-pdf-intake-credible-closure-implementation.md`.
- Do not reintroduce registry browsing through `useDatabaseWorkbench()` after `useRegistryBrowser()` lands.
- Do not leave the old doctor-toolbar `Historical Case Base` / `Patient Registry` buttons in place once `DoctorDatabaseView` owns source switching.
- Do not let doctor-side routing reach `outpatient_triage` after the backend remap lands.
