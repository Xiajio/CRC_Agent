# Patient CRC Triage Subpage Design

Date: 2026-06-24
Written: 2026-06-25
Status: Draft for user review
Scope: Add a CRC-specific patient triage subpage while preserving the existing patient assistant, profile, upload, session, registry, and doctor workflows.

## Context

The current LangG patient side already has a patient workspace with:

- a default patient assistant entry
- patient identity capture
- patient profile and background cards
- report upload and parsing
- patient session recovery
- registry-backed patient identity and records
- event-sourced patient writes through `PatientCommandService`
- existing outpatient triage and patient self-report card behavior

The `CRC-client` project provides a separate CRC-focused patient triage prototype. Its valuable parts are the clinical interaction model, not its application shell. It contains a structured patient intake flow, staged triage questions, risk stratification, mid-risk follow-up behavior, endoscopy-information blocking, patient record summaries, and material export ideas.

Directly merging `CRC-client` as a child app would introduce a second frontend framework path, a second API style, local CSV persistence, local patient numbering, and browser storage behavior that does not match the current LangG patient registry architecture.

The recommended direction is to keep the existing patient functionality intact and add a new patient-side subpage for CRC-specific triage. The new page should extract and adapt the useful protocol behavior from `CRC-client`, then land it on LangG's existing patient session, registry, event-sourcing, and card-rendering foundation.

## Goal

Add a patient-side "CRC special triage" subpage that supports structured colorectal-cancer-related pre-consultation intake without disrupting the existing patient assistant.

The new subpage should:

- preserve the current patient assistant as the default patient entry
- add a separate patient navigation item for CRC-specific triage
- reuse the current patient session and registry patient identity
- guide patients through structured CRC intake and follow-up questions
- produce a structured assessment summary that can be persisted as a patient record
- make the resulting patient-side information available to later doctor review and report workflows
- support a staged rollout, starting with page isolation and protocol entry before richer history, follow-up cards, and exports

## Non-Goals

This design does not include:

- directly moving `CRC-client` pages into the main frontend
- embedding the `CRC-client` Next.js app under the current React app
- using `CRC-client` CSV files as a runtime source of truth
- replacing LangG patient registry identity with `CRC-client` 10-digit patient IDs
- replacing the existing patient assistant
- redesigning the doctor cockpit
- replacing the current upload flow
- implementing PDF or Markdown export in the first stage
- building a full patient portal with notification, account, or scheduling features
- adding a separate persistence mechanism for CRC triage

## Recommended Approach

Use a new tab inside the existing patient workspace:

- `问助手` remains the default patient tab.
- `专项问诊` becomes the CRC triage subpage.
- `我的资料` remains the patient identity and background area.
- `上传报告` remains the upload and parsing area.

This is preferred over an independent route because the current patient experience is already organized around a workspace surface, scene sessions, patient navigation tabs, patient cards, and session recovery. A new tab keeps the state model simple and avoids splitting patient recovery behavior across unrelated routes.

This is also preferred over direct code migration from `CRC-client` because the useful asset is the protocol and patient experience, not the original framework boundary.

## Alternatives Considered

### Option 1: Patient Workspace Tab

Add a new patient workspace tab for CRC-specific triage.

Benefits:

- lowest risk to existing flows
- reuses current patient session and patient registry
- keeps all patient functionality under one workspace
- easy to hide, gate, or roll out gradually
- aligns with existing `usePatientWorkspaceNav` behavior

Tradeoff:

- the existing `WorkspacePage` is already broad, so implementation should avoid making it larger by extracting CRC-specific UI into feature components.

Decision: Recommended.

### Option 2: Independent Patient Route

Add a route such as `/patient/crc-triage`.

Benefits:

- clearer URL boundary
- easier deep-linking later

Tradeoff:

- adds routing, recovery, and patient session coordination work
- may duplicate workspace layout decisions
- more surface area for patient identity and upload navigation

Decision: Defer until deep-linking becomes a real requirement.

### Option 3: Direct CRC-client Migration

Move or embed the `CRC-client` app as-is.

Benefits:

- fastest visual reuse if the app were standalone

Tradeoff:

- introduces duplicated frontend and backend patterns
- conflicts with registry and event-sourcing source-of-truth rules
- keeps CSV/local-storage behavior that should not become production patient data infrastructure
- increases long-term maintenance cost

Decision: Reject.

## User Experience

### Navigation

Patient navigation should become:

- `问助手`
- `专项问诊`
- `我的资料`
- `上传报告`

The patient workspace must still default to `问助手`. The CRC triage subpage is opt-in.

Suggested user-facing names:

- Navigation label: `专项问诊`
- Page title: `CRC 专项预问诊`
- Internal flow key: `crc_triage`

### Page States

The CRC triage subpage has four primary states.

### 1. Not Started

The page explains that this is a structured CRC-related pre-consultation intake. It provides a clear start action and indicates that the user can still use the normal patient assistant.

Required behavior:

- show the current patient identity context when available
- show a start button
- show a short safety note that urgent symptoms should seek timely medical care
- provide a secondary path to upload a report if the patient wants to provide existing materials first

### 2. In Progress

The page guides the patient through structured questions.

Required behavior:

- show the current question
- support option-based answers when a stable schema exists
- allow free-text supplement at all times
- show collected information in a compact summary
- show missing or pending information when available
- keep the normal patient assistant separate

### 3. Needs More Information

When critical information is missing, the page should ask for it before presenting a final summary.

Examples:

- endoscopy was mentioned but key result details are missing
- red-flag symptoms need clarification
- duration, bleeding, bowel habit change, weight loss, or fever information is incomplete
- uploaded report information is absent or not yet parsed

Required behavior:

- state what information is missing in patient-facing language
- allow continued free-text answer
- provide a navigation action to `上传报告` when report upload is the right next step

### 4. Completed

The page presents a structured summary and persistence status.

Required behavior:

- show chief complaint and symptom summary
- show risk level and disposition language
- show suggested next action
- show missing-information caveats
- show whether the assessment was saved into patient records
- allow the user to start a new CRC triage session without deleting the saved record

## Interaction Rules

The CRC triage subpage should be structured-first but not form-only.

Rules:

- Prefer buttons or option cards for common triage follow-up fields.
- Keep free-text input available for every question.
- Use existing patient chat transport where possible.
- Do not make button answers a separate source of truth.
- Convert button answers into natural-language user turns that the backend can still parse.
- Attach structured interaction metadata only as auxiliary context.
- After repeated non-answers, show a clearer option-style prompt.
- If red-flag symptoms are detected, prioritize urgent-care guidance over completing a routine triage flow.
- If the user changes topics, keep the CRC triage state recoverable but do not pollute the normal patient assistant's behavior.

## Architecture

### Frontend Placement

Add the new subpage under the existing patient workspace.

Expected boundaries:

- Extend patient workspace navigation with a `crc_triage` tab.
- Keep the default tab as `assistant`.
- Render the new page only when patient scene is active and selected tab is `crc_triage`.
- Keep doctor scene untouched.
- Keep profile and upload tabs untouched.

The main workspace page should only select the active panel. CRC-specific layout and behavior should live in feature components under a patient CRC triage feature area, rather than expanding the workspace page with protocol details.

Suggested component boundaries:

- `PatientCrcTriagePanel`
- `CrcTriageStartView`
- `CrcTriageConversation`
- `CrcTriageProgressPanel`
- `CrcTriageResultSummary`
- `CrcTriageRecordList` for a later phase

### Backend Placement

Do not port `CRC-client/src/app/api/chat/route.ts`.

The backend should reuse the current FastAPI, session, and LangGraph architecture:

- patient session remains the request context
- patient ID remains the registry patient ID
- CRC triage is selected through explicit context such as `patient_subflow = "crc_triage"`
- the existing outpatient triage logic remains the base where it already overlaps
- CRC-client protocol rules are translated into backend-owned Python rules, nodes, or service helpers
- final records are written through `PatientCommandService`

### Flow Selection

The patient side should distinguish normal patient assistant turns from CRC triage turns.

Recommended request context:

```json
{
  "patient_subflow": "crc_triage",
  "crc_triage": {
    "interaction_source": "patient_crc_triage_tab",
    "question_id": "optional-question-id",
    "selected_option_ids": ["optional-option-id"],
    "free_text": "optional user supplement"
  }
}
```

Phase 1 should still primarily trust natural-language content. Structured context is useful for logging, validation, UI state, and future migration to stricter structured submissions.

## Data Model

### Patient Identity

The new subpage must reuse the patient created for the existing patient session.

Rules:

- Do not allocate a CRC-client 10-digit ID.
- Do not use browser local storage as the patient identity source.
- Do not create a second registry patient for the CRC triage session.
- If patient identity is incomplete, allow the triage to proceed but make the missing identity visible.

### Triage Record

When a CRC triage assessment is completed, persist a patient record.

Recommended record type:

```text
crc_triage_assessment
```

Minimum record payload:

```json
{
  "record_type": "crc_triage_assessment",
  "chief_complaint": "...",
  "symptom_group": "...",
  "risk_level": "...",
  "disposition": "...",
  "red_flags": [],
  "known_crc_signals": {},
  "suggested_tests": [],
  "missing_information": [],
  "qa_summary": [],
  "patient_summary": "...",
  "next_step": "...",
  "source_session_id": "...",
  "source_subflow": "crc_triage"
}
```

The payload can be expanded later, but the first version must be stable enough for doctor review and future migration.

### Events

The first implementation may persist only the completed assessment as a patient record if that is materially simpler.

If event granularity is introduced, use event types like:

- `patient.crc_triage_started`
- `patient.crc_triage_answered`
- `patient.crc_triage_assessed`
- `patient.followup_card_created`

All persisted patient facts must flow through `PatientCommandService`. Session state and frontend state may cache derived views only.

## Relationship To Existing Features

### Patient Assistant

The normal patient assistant remains default and unchanged. It continues to support:

- general patient questions
- report explanation
- symptom supplement
- treatment-option explanation
- current patient chat card behavior

The CRC triage subpage may reuse conversation UI primitives but should not merge its protocol state into the default assistant view.

### Patient Profile

The CRC triage subpage can read patient identity and background context. It should not replace the identity form.

If identity is missing, the subpage can show a soft prompt to complete `我的资料`.

### Upload Report

The CRC triage subpage can route the patient to upload reports when needed.

Rules:

- Upload remains owned by the existing upload flow.
- Parsed report data remains written through current registry/event pathways.
- CRC triage can reference uploaded record summaries after they are available.

### Doctor Side

Doctor workflows should not change in the first stage.

The doctor side benefits later by seeing CRC triage records in patient context, records, or generated summaries. It should not need a new doctor-only workflow for the first release.

## Error Handling

### Session Expiration

Use existing patient session recovery behavior. If a patient session expires, the CRC triage page should follow the same recovery model as the rest of the patient workspace.

### Backend Failure

If a CRC triage turn fails:

- preserve user input
- show retry affordance
- do not mark the question complete
- do not write partial final records as completed assessments

### Persistence Failure

If the assessment is completed but persistence fails:

- show that the assessment is complete but not saved
- allow retrying the save
- do not silently discard the result

### Stale Question Submission

If a user submits an answer for an older question:

- do not hard-fail the turn
- process the natural-language content as a normal answer where safe
- use structured metadata only for logging or soft validation

### Missing Report Data

If report data is missing:

- do not block the entire page
- explain what is missing
- allow upload navigation
- include missing-information caveats in the final summary

## Safety Requirements

The CRC triage subpage must avoid presenting itself as a diagnostic authority.

Requirements:

- red-flag symptoms must trigger urgent-care guidance
- incomplete information must remain visible in the result
- the result should use triage and next-step language, not final diagnosis language
- the page should not produce final treatment plans
- the page should not suppress warnings because a patient wants to continue routine triage
- saved records must identify source session and subflow

## Testing Strategy

### Frontend Tests

Cover:

- patient navigation includes `专项问诊`
- patient workspace still defaults to `问助手`
- existing profile and upload tabs remain reachable
- CRC triage tab renders the start state
- start action moves to in-progress state
- upload navigation from CRC triage selects the upload tab
- completed state displays summary and save status
- doctor scene is unaffected

### Backend Tests

Cover:

- CRC subflow context routes to CRC triage behavior
- normal patient assistant turns still route normally
- red-flag inputs prioritize urgent-care guidance
- missing endoscopy details produce a missing-information state
- completed assessment can be persisted as a patient record
- persisted record contains patient ID, source session, source subflow, and assessment payload

### Integration Tests

Cover:

- patient starts CRC triage, answers structured questions, completes summary
- completed assessment appears in patient records or patient context
- session recovery does not lose persisted completed assessment
- upload flow remains independent but accessible from CRC triage
- existing patient assistant regression tests remain stable

## Rollout Plan

### Phase 1: Page Isolation

Add the patient navigation item and a CRC triage page shell.

Deliverables:

- `专项问诊` tab
- not-started state
- start action
- upload navigation action
- no changes to default assistant behavior

### Phase 2: Protocol Entry

Connect the page to backend CRC triage context.

Deliverables:

- patient subflow request context
- structured question display
- free-text fallback
- initial CRC triage state management
- reuse of existing outpatient triage where appropriate

### Phase 3: Assessment Summary And Persistence

Save completed CRC triage assessments as patient records.

Deliverables:

- completed summary view
- record payload shape
- persistence through patient command service
- save success and save failure states

### Phase 4: History And Follow-Up

Expose previous CRC triage records and follow-up cards.

Deliverables:

- CRC triage record list
- per-assessment summary cards
- follow-up card display
- patient-facing timeline if needed

### Phase 5: Export And Migration

Evaluate material export and CRC-client historical data import.

Deliverables:

- PDF/Markdown export if required
- CSV import tool for old CRC-client records if needed
- migration validation report

## Acceptance Criteria

The first complete implementation is acceptable when:

- patient workspace still defaults to `问助手`
- original patient assistant, profile, and upload functions still work
- `专项问诊` exists as an isolated patient subpage
- CRC triage uses the current patient session and registry patient ID
- CRC triage does not allocate a separate local patient ID
- completed CRC triage can produce a structured assessment summary
- completed summary can be saved as a patient record
- saved records include source session and subflow
- doctor workflows do not regress
- tests cover default tab, new tab, routing isolation, persistence, and key safety states

## Implementation Notes

Keep the implementation staged and conservative:

- avoid expanding `WorkspacePage` with protocol-specific logic
- keep CRC-specific UI in feature components
- keep backend protocol rules owned by backend code, not copied from the TypeScript route handler
- treat `CRC-client` as a reference for behavior and language, not as a code dependency
- introduce stricter structured submissions only after natural-language compatibility is stable
- defer export and migration until the core subpage and persistence path are reliable

## Open Decisions

The following decisions should be confirmed before implementation planning:

- whether the navigation label should be `专项问诊`, `CRC预问诊`, or another patient-facing name
- whether Phase 1 should include a static protocol preview or only a start state
- whether completed assessments should appear immediately in `我的资料` or only in doctor-facing records for the first release

## Recommended Decision

Proceed with the patient workspace tab approach.

This preserves the existing patient-side product, avoids direct `CRC-client` code migration, uses the current patient data architecture, and creates a clear path to gradually absorb the new CRC triage capabilities without destabilizing current patient and doctor workflows.
