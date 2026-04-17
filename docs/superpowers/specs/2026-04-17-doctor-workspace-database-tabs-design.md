# Doctor Workspace Database Tabs Design

**Date:** 2026-04-17  
**Status:** Draft for review  
**Goal:** Simplify the doctor scene by separating current-patient consultation work from database browsing, while preserving both doctor general-clinical consultation without a bound patient and patient-bound consultation inside one doctor scene.

## 1. Context

The current scene-driven workspace already has the following baseline:

- patient and doctor run in separate sessions
- the doctor scene can bind a current patient from the shared SQLite registry
- the doctor page already renders patient summary, alerts, records, chat, and a database workbench
- the database workbench currently supports two data sources:
  - `historical_case_base`
  - `patient_registry`

This means the doctor scene is functionally capable, but it is no longer cleanly scoped. The same page is trying to serve two very different workflows:

- active consultation for the current patient
- database browsing for reference and lookup

That overlap makes the doctor scene heavier than necessary and weakens the conceptual boundary between "the patient I am seeing now" and "the data I am browsing."

This redesign also depends on ongoing PDF-intake credibility work.

The doctor consultation surfaces in this document assume that the following backend-facing registry capabilities already exist and are stable:

- patient-registry detail normalization
- patient-registry records
- patient-registry alerts

Those capabilities come from the PDF-intake credibility pass. The execution order therefore matters:

1. Complete the PDF-intake backend hardening phase (`Task 1` through `Task 4`).
2. Apply the doctor-workspace tab refactor from this document.
3. Do **not** separately implement the PDF-intake frontend `Task 5` as originally written.

The original PDF-intake frontend `Task 5` assumed registry browsing would continue to live inside `useDatabaseWorkbench()`. That direction conflicts with this design. Its doctor-facing registry UI work should instead be absorbed into:

- `DoctorDatabaseView`
- `RegistryBrowserPane`
- `useRegistryBrowser()`

## 2. Problem Statement

The current doctor page places current-patient consultation and database browsing side by side in one continuous layout. This causes four product problems:

1. The consultation view is visually overloaded.
2. The registry browser and historical workbench compete for space with core doctor actions.
3. The product does not clearly distinguish between:
   - a patient being previewed in the database
   - the patient currently bound to the doctor session
4. The consultation tab currently implies that a doctor cannot begin meaningful clinical dialogue until a patient is bound, even though the doctor graph already supports general clinical reasoning without patient context.

The system already has the backend primitives needed to separate those concepts. The missing piece is a cleaner doctor-side information architecture.

## 3. Design Principles

- Keep the doctor scene as a single top-level page.
- Separate "current patient care" from "database browsing" at the doctor-scene level.
- Preserve the existing registry and historical backends as distinct data planes.
- Make preview and bind explicit, separate actions.
- Preserve database browsing state when switching between doctor tabs.
- Preserve the original doctor clinical flow even when no patient is bound.
- Keep symptom-triage behavior isolated from the doctor route.
- Avoid unnecessary backend redesign when existing APIs are already sufficient.

## 4. Target Outcome

The doctor scene should become a two-tab workspace:

- `Consultation Workspace`
- `Database`

The expected behavior is:

1. The doctor enters the doctor scene and lands on `Consultation Workspace`.
2. If no patient is bound, the consultation tab renders `General Clinical Mode`:
   - the doctor still has the conversation panel
   - the doctor can ask general clinical, knowledge, and database questions
   - the doctor is told that no patient-specific context is active
3. If a patient is bound, the consultation tab renders `Patient-Bound Mode`:
   - current-patient summary
   - current-patient alerts and records
   - the same doctor conversation panel, now with patient context
4. The doctor can switch to `Database` to browse:
   - `Historical Case Base`
   - `Patient Registry`
5. In the registry browser, clicking a patient previews that patient only.
6. The doctor must explicitly click `Set As Current Patient` to bind that patient into the doctor session.
7. After binding, the doctor consultation tab updates to the new current patient.
8. Database browsing state remains intact when returning to the database tab.

## 5. Information Architecture

### 5.1 Doctor Scene Structure

The doctor scene remains one page, but its internal structure changes to:

- `DoctorSceneShell`
  - `Consultation Workspace`
  - `Database`

### 5.2 Consultation Workspace

This tab is responsible for the current doctor session. It has two runtime modes.

#### General Clinical Mode

This mode applies when `currentPatientId = null`.

It includes:

- a visible mode indicator such as `General Clinical Mode`
- the doctor conversation panel
- roadmap / execution plan / clinical cards
- a CTA that opens `Database > Patient Registry`
- guidance that responses are not using patient-specific context

It explicitly does **not** block the doctor from using the original consultation flow.  
The doctor may still perform:

- general chat
- knowledge query
- case-database query
- general clinical reasoning
- full doctor-side clinical reasoning without patient context

It does **not** include a current-patient summary, alerts, or records because no patient is bound.

#### Patient-Bound Mode

This mode applies when `currentPatientId` is present.

It includes:

- current bound patient summary
- registry alerts for the current patient
- registry records for the current patient
- doctor conversation panel
- roadmap / execution plan / clinical cards

The conversation panel is the same consultation surface as in general mode, but the doctor graph now receives bound-patient context.

The consultation tab explicitly does **not** include the database workbench.

It also does **not** include a browsing-oriented recent-patients panel.  
That panel belongs to the database side of the doctor experience because it is part of patient discovery, not active consultation.

The previous "empty consultation workspace" concept is no longer the target behavior.

### 5.3 Database Tab

This tab is responsible only for database browsing and lookup.

It includes a second-level source switch:

- `Historical Case Base`
- `Patient Registry`

The database tab supports:

- historical-case search and detail browsing
- recent-patient discovery inside the registry browser
- patient-registry search and preview
- explicit promotion of a previewed registry patient into the current doctor session

## 6. Interaction Model

### 6.1 Consultation vs Database

The consultation tab is where the doctor acts on the currently bound patient.  
The database tab is where the doctor searches and previews data.

These two responsibilities must not share one undifferentiated content region.

### 6.2 Registry Preview vs Current Patient

Within the `Patient Registry` database source:

- clicking a patient row updates the preview only
- previewing a patient does not mutate the doctor session
- only the explicit `Set As Current Patient` action triggers session binding

This preserves a clear distinction between:

- `previewedRegistryPatientId`
- `doctorSession.currentPatientId`

### 6.3 Binding Behavior

When the doctor clicks `Set As Current Patient`:

- call the existing doctor-session bind endpoint
- update the doctor session state
- refresh the consultation view data
- preserve database browsing state
- default behavior should switch back to `Consultation Workspace`

The automatic return to the consultation tab is recommended because the bind action represents an explicit intent to begin working on that patient.

## 7. State Model

The redesign requires three distinct state groups.

### 7.1 Doctor Session State

Existing session-backed doctor state remains the source of truth for:

- current bound patient id
- doctor messages
- graph findings
- roadmap / plan / cards

### 7.2 Doctor View State

Add a lightweight UI-only state model for doctor tab selection:

- `activeDoctorTab = consultation | database`
- `activeDatabaseSource = historical | registry`

This state is not session-backed.

### 7.3 Doctor Database Browsing State

This state is independent from the doctor session and must survive tab switching inside the current page lifetime.

It includes:

- historical workbench state
- registry search text
- registry filters
- registry pagination
- `previewedRegistryPatientId`
- preview detail / alerts / records

The doctor can leave the database tab and return without losing this state.

## 8. Frontend Component Design

### 8.1 New Shell Layer

Introduce a doctor-scene shell component:

- `DoctorSceneShell`

Recommended file location:

- `frontend/src/features/doctor/doctor-scene-shell.tsx`

Responsibilities:

- render doctor-side secondary tabs
- hold `doctorViewState`
- delegate to the correct child view

`workspace-page.tsx` remains the top-level scene shell for `patient | doctor`. When the active scene is `doctor`, it should render `DoctorSceneShell` rather than directly owning the doctor-specific secondary-tab layout.

### 8.2 Consultation View

Introduce:

- `DoctorConsultationView`

Responsibilities:

- render current-patient consultation context only
- consume current bound-patient data
- avoid any embedded database browsing UI

### 8.3 Database View

Introduce:

- `DoctorDatabaseView`

Responsibilities:

- render database source switch
- host either:
  - `HistoricalDatabasePane`
  - `RegistryBrowserPane`

### 8.4 Registry Browser Pane

Introduce:

- `RegistryBrowserPane`

Responsibilities:

- recent-patient discovery
- patient-registry search
- registry results list
- preview detail
- preview alerts / records
- explicit `Set As Current Patient` action

This pane must not bind the doctor session automatically when a row is selected.

## 9. Hook Responsibilities

### 9.1 Keep `usePatientRegistry()` Focused on Current Patient Context

`usePatientRegistry()` should remain the hook for:

- bound patient detail
- bound patient records (backed by `GET /api/patient-registry/patients/{id}/records`)
- bound patient alerts
- `bindPatient()`

This hook serves the consultation tab.

### 9.2 Narrow `useDatabaseWorkbench()` Back to Historical Workbench

`useDatabaseWorkbench()` should no longer try to be a shared abstraction for both historical and registry browsing.

Its responsibility should narrow to:

- historical search
- historical detail
- historical stats
- historical edit flow

This keeps the hook semantically aligned with the actual workbench role.

### 9.3 Introduce `useRegistryBrowser()`

Add a dedicated hook for doctor-side patient-registry browsing.

It should manage:

- registry search request
- registry search response
- previewed patient id
- preview detail
- preview alerts
- preview records
- loading and error state

It should not mutate the doctor session unless the caller explicitly invokes the bind action.

## 10. Backend and API Impact

This redesign does **not** require major backend API expansion.

The existing APIs are already sufficient:

- patient registry recent
- patient registry search
- patient registry detail
- patient registry records
- patient registry alerts
- doctor session bind
- historical database search / detail / stats APIs

The key rule is not to change API semantics:

- preview remains a registry read operation
- bind remains a doctor-session mutation

No new preview-specific endpoint is required in this phase.

However, the doctor graph needs two behavior-level adjustments so the consultation tab semantics stay coherent:

### 10.1 Doctor Route Remap Away From Patient Triage

The doctor scene should not route into patient-style outpatient triage.

The current code still registers `clinical_entry_resolver` and `outpatient_triage` in the doctor graph.  
That is acceptable as an intermediate topology, but doctor-side intent routing must not send doctor turns into that branch.

Recommended rule:

- on the doctor side, any doctor turn that would have targeted `clinical_entry_resolver` should instead enter doctor-side clinical assessment
- the doctor graph must not enter `outpatient_triage`

This keeps symptom-based patient intake isolated to the patient scene without reopening the entire doctor graph architecture.

### 10.2 Scene-Aware Assessment Entry

`node_assessment` is currently shared by both scenes, but the product semantics are no longer identical.

Recommended direction:

- keep one shared assessment core if possible
- introduce scene-aware wrappers or prompts:
  - `patient_assessment`
  - `doctor_assessment`
- ensure patient assessment remains inquiry-oriented
- ensure doctor assessment continues to support full clinical reasoning

This avoids duplicating the full reasoning engine while still separating policy and prompt behavior by scene.

## 11. Behavioral Rules

### 11.1 Database State Preservation

When the doctor leaves `Database` and returns:

- database source stays the same
- search terms stay the same
- filters stay the same
- pagination stays the same
- previewed patient stays the same

This persistence is required at least for the current page lifecycle.

### 11.2 Reset Semantics

`Reset Active Scene` should continue to reset the doctor session.

It should **not** implicitly clear database browsing state unless the product later decides that database reset should be explicit as well.

### 11.3 Route and Data-Plane Separation

The redesign must preserve the hard separation between:

- registry patients
- historical cases

Registry preview must not trigger historical detail loading.  
Historical case browsing must not be implicitly rebound into the current doctor session.

### 11.4 Legacy Toolbar Scope Removal

The current doctor toolbar already exposes `Historical Case Base` and `Patient Registry` scope buttons.

After this redesign:

- those toolbar-level scope buttons must be removed
- database-source switching must exist only inside the `Database` tab
- the consultation tab must not expose historical/registry source controls

### 11.5 Doctor Consultation Modes

The consultation tab must support both:

- `General Clinical Mode`
- `Patient-Bound Mode`

The difference between them is context injection and surrounding UI, not whether the doctor can open the conversation panel.

## 12. Minimal Implementation Sequence

1. Complete the PDF-intake backend phase through alerts/records support.
2. Add a doctor-side route remap so doctor turns do not enter outpatient triage.
3. Introduce scene-aware assessment entry points or wrappers.
4. Add `DoctorSceneShell` and doctor secondary tabs.
5. Remove the old toolbar-level historical/registry scope buttons.
6. Extract current doctor content into `DoctorConsultationView`.
7. Replace the old no-patient empty state with `General Clinical Mode`.
8. Extract database content into `DoctorDatabaseView`.
9. Move historical browsing responsibility fully under `useDatabaseWorkbench()`.
10. Add `useRegistryBrowser()` and `RegistryBrowserPane`.
11. Move recent-patient discovery into the registry browser instead of the consultation tab.
12. Add explicit `Set As Current Patient`.
13. Preserve database state across doctor-tab switching.
14. Fold the intended PDF-intake frontend registry behavior into this database-tab implementation instead of shipping it separately.
15. Run regression checks for doctor routing, session binding, registry preview, and historical-case browsing.

## 13. Acceptance Criteria

The redesign is complete when all of the following are true:

### 13.1 Doctor Information Architecture

- the doctor scene defaults to `Consultation Workspace`
- the doctor scene exposes secondary tabs:
  - `Consultation Workspace`
  - `Database`
- the consultation tab no longer renders the database workbench
- the consultation tab no longer renders a browsing-oriented recent-patients list
- the database tab renders both:
  - `Historical Case Base`
  - `Patient Registry`

### 13.2 State Separation

- selecting a registry patient for preview does not change `currentPatientId`
- only `Set As Current Patient` updates the doctor session
- switching back to the database tab preserves prior browsing state

### 13.3 Consultation Modes

- when no current patient is bound, the consultation tab still renders the doctor conversation panel
- when no current patient is bound, the consultation tab clearly indicates `General Clinical Mode`
- that mode includes a CTA to open `Database > Patient Registry`
- when a patient is bound, the consultation tab switches to `Patient-Bound Mode`

### 13.4 Binding Flow

- binding a previewed registry patient updates the doctor session
- the consultation tab refreshes with the newly bound patient
- the database tab state remains intact after the bind action

### 13.5 Historical / Registry Boundary

- registry preview never calls historical-case detail APIs
- historical search and detail still work normally
- patient-registry browsing still works normally

### 13.6 Routing and Clinical Safety

- doctor-side symptom prompts do not enter `outpatient_triage`
- doctor-side full clinical reasoning still works without a bound patient
- patient-side `case_database_query` remains blocked
- patient-side assessment behavior remains inquiry-oriented after any assessment-wrapper change

### 13.7 Regression Safety

- patient-side upload flow remains unaffected
- doctor-side context injection still uses the bound patient summary
- current-patient alerts and records remain visible in consultation mode
- the old toolbar-level historical/registry scope buttons are gone

## 14. Risks

- If the doctor shell is introduced without separating state ownership, the code may still look cleaner while remaining behaviorally tangled.
- If `useDatabaseWorkbench()` keeps both registry and historical semantics, the new tab structure will still rest on a confused abstraction.
- If doctor routing still reaches outpatient triage, the consultation tab will blur doctor and patient behaviors again.
- If preview and bind are not clearly split in the registry pane, the redesign will not solve the core product ambiguity.

## 15. Recommendation

Proceed with a light backend-plus-frontend refactor that:

- keeps doctor general-clinical consultation available without a bound patient
- remaps doctor-side routing away from patient triage
- introduces scene-aware assessment entry
- introduces doctor-scene secondary tabs
- narrows the historical workbench abstraction
- introduces a dedicated registry browser abstraction
- keeps backend APIs stable

This yields a cleaner doctor experience without removing the original doctor-side clinical flow.
