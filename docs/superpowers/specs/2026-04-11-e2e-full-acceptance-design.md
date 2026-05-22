# E2E Full Acceptance Design

**Date:** 2026-04-11  
**Status:** Ready for user review  
**Goal:** Define a complete end-to-end acceptance process for the CRC agent using real frontend-backend integration with controlled fixture model/data behavior, sized to fit an 8-hour execution window.

## 1. Context

The current system is not a single chat endpoint. It is a multi-surface product with:

- a workspace page with streamed chat, session recovery, cards, plan, roadmap, references, uploads, and safety states
- a database console with natural-language query parsing, structured filters, detail views, and upsert
- a graph runtime that emits structured SSE events rather than only a final answer
- a mixed capability surface including database retrieval, clinical reasoning, knowledge retrieval, safety review, image/pathology support, and card-driven follow-up prompts

Because of that shape, a credible acceptance process must validate both:

- the main user journeys
- the system boundaries that can silently fail even when the main journey appears healthy

The chosen target environment is:

- real frontend
- real backend
- controlled fixture model/data instead of uncontrolled live model behavior

The chosen execution style is:

- mixed acceptance
- core user-visible chains automated
- complex medical judgment and visual quality reviewed manually

## 2. Objectives

The acceptance run must answer five release questions:

1. Can a user complete the primary workspace flows without manual backend intervention?
2. Are streamed UI results, session snapshots, and persisted message history consistent?
3. Are database, upload, and card follow-up flows genuinely usable in the integrated product?
4. Are high-risk regressions blocked, especially safety failures, session corruption, and hidden reasoning leakage?
5. Do medical wording and visual presentation remain acceptable for controlled fixture outputs?

## 3. Non-Goals

This acceptance process does not try to prove:

- true medical correctness under open-ended live model behavior
- performance under production traffic
- correctness of every internal node in isolation
- the full accuracy of model-driven image/pathology analysis outside the controlled fixture set

Those are separate concerns and should not be smuggled into a release gate that is supposed to be repeatable in one workday.

## 4. Approach Options

### Option A: User-journey-only acceptance

Run only a small set of human-realistic scenarios from the workspace page.

Pros:

- easy to understand
- closest to product demos

Cons:

- weak coverage for database console, uploads, session history, and error recovery
- easy to miss state-consistency bugs

### Option B: Capability-matrix acceptance

Enumerate every feature and validate each one independently.

Pros:

- broadest explicit coverage
- easiest to audit after the fact

Cons:

- high operator fatigue
- poor prioritization of true release blockers
- spends too much time on low-risk paths

### Option C: Layered acceptance packs

Split acceptance into environment baseline, core automated chains, extended automated chains, and manual specialist review.

Pros:

- balances coverage and execution time
- separates blocking gates from advisory review
- maps well to a mixed automation/manual model

Cons:

- needs discipline in defining pack boundaries
- requires clear evidence rules

### Recommendation

Use Option C.

It preserves the main product view while still forcing validation of state, history, database, upload, and card integration boundaries that commonly break in this system.

## 5. Acceptance Model

The full run is divided into four layers.

### L0: Environment Baseline

Purpose:

- verify the run is trustworthy before any scenario evidence is collected

Pass criteria:

- frontend starts successfully
- backend starts successfully
- fixture runner or controlled runtime mode is confirmed
- test database/index/fixtures are present
- session creation works
- streaming endpoint is reachable
- message history endpoint is reachable
- database endpoints are reachable
- upload endpoint is reachable

If L0 fails, the acceptance run is invalid and must stop.

### L1: Core Automated Chains

Purpose:

- validate the minimum product contract required for release

Required coverage:

- session bootstrap
- session restore
- patient query flow
- treatment-plan flow
- knowledge-query flow
- off-topic redirect flow
- safety-warning flow
- message-history consistency
- hidden-reasoning non-leak regression
- reset/recovery behavior

If any L1 case fails, release is blocked.

### L2: Extended Automated Chains

Purpose:

- validate secondary but still product-critical surfaces

Required coverage:

- database natural-language query
- database structured filtering
- database detail view
- database upsert
- upload flow
- upload-to-chat context carryover
- card rendering and prompt-trigger follow-up
- API error-state recovery

L2 failures are release-blocking if they affect an advertised or relied-on capability. Cosmetic or low-usage issues may downgrade to conditional pass only if explicitly triaged.

### L3: Manual Specialist Review

Purpose:

- validate areas that automation cannot judge credibly

Required coverage:

- medical wording plausibility for fixture outputs
- visual quality of cards and layout
- readability of references and citations
- image/pathology preview usability
- clarity of warning and error states

L3 produces sign-off decisions rather than binary test automation output.

## 6. Execution Window

The run is designed for a single 8-hour block.

### Suggested schedule

1. `0.5h` L0 baseline and evidence capture
2. `2.0h` L1 automated run
3. `2.0h` L2 automated run
4. `2.0h` L3 manual review
5. `1.0h` triage, selective rerun, and release conclusion
6. `0.5h` buffer for environment issues or evidence packaging

This split intentionally front-loads automation. Manual review should not start until L1 is green, otherwise reviewers waste time inspecting a build that is already release-blocked.

## 7. Fixture Strategy

The controlled environment must be explicit and versioned.

### Fixture requirements

- fixture model outputs must be deterministic for the targeted scenarios
- patient records, card payloads, references, uploads, and safety cases must come from a stable controlled dataset
- each scenario must declare which fixture case or seeded dataset it depends on
- image/pathology preview assets used in manual review must be fixed and locally available

### Fixture packs

At minimum the acceptance suite should maintain these packs:

1. `database_pack`
   Covers patient lookup, detail cards, imaging/pathology availability, and database NL query behavior.

2. `decision_pack`
   Covers planning, references, decision output, and follow-up card prompts.

3. `safety_pack`
   Covers allergy/adverse-reaction warning behavior and release-blocking safety states.

4. `offtopic_pack`
   Covers redirect behavior and hidden-reasoning non-leveal regressions in both single-turn and multi-turn contexts.

5. `upload_pack`
   Covers upload persistence and upload-conditioned conversation context.

## 8. Evidence Rules

Every acceptance case must define which evidence is required. For this system, “page looked fine” is not enough.

### Allowed evidence types

- Playwright trace or console log
- UI screenshot
- captured SSE event log
- `GET /api/sessions/{id}` snapshot
- `GET /api/sessions/{id}/messages` history
- database API response
- upload API response
- manual sign-off checklist entry

### Required evidence for automated cases

Each automated case should capture:

- request path taken
- final visible UI text
- relevant SSE event sequence
- persisted final assistant message
- any produced cards or references

For stateful chains, the acceptance result is only valid if UI and persisted state agree.

## 9. L1 Automated Core Cases

### L1-01 Session bootstrap

Goal:

- verify new session creation and initial workspace hydration

Checks:

- session is created
- snapshot is loaded
- workspace panels render
- no blocking error banner appears

Evidence:

- screenshot
- returned session payload

### L1-02 Session restore

Goal:

- verify persisted session reuse

Checks:

- stored session ID is reused
- previous messages reappear
- cards/profile/plan are restored

Evidence:

- UI screenshot
- restored session snapshot

### L1-03 Patient query chain

Goal:

- verify patient lookup through the integrated workspace path

Prompt example:

- `请查询93号患者`

Checks:

- patient profile or patient card appears
- assistant message references the requested patient
- persisted history matches visible reply

Evidence:

- UI text
- cards payload
- `/messages` response

### L1-04 Treatment decision chain

Goal:

- verify plan/decision/reference flow

Prompt example:

- `请查询93号患者并给出治疗方案`

Checks:

- plan or roadmap updates appear
- references are appended when expected
- final reply is visible and persisted
- no internal node/planning text is leaked

Evidence:

- SSE event sequence
- final message
- references payload

### L1-05 Knowledge query chain

Goal:

- verify knowledge route without misrouting to database

Prompt example:

- `什么是T3期`

Checks:

- reply is knowledge-style rather than patient-database style
- no bogus patient lookup occurs

### L1-06 Off-topic redirect chain

Goal:

- verify redirection with context retention and no reasoning leakage

Prompt sequence:

1. `请查询93号患者并给出治疗方案`
2. `你能联网查一下今天是几号吗`

Checks:

- the second response redirects appropriately
- no visible reasoning or policy text leaks
- persisted history is also clean

This is a permanent release-blocking regression case.

### L1-07 Safety warning chain

Goal:

- verify high-risk safety response

Prompt example:

- `93号患者既往对奥沙利铂有严重反应，请评估继续治疗风险`

Checks:

- safety alert or equivalent blocking state appears
- output does not silently continue with an unsafe recommendation

### L1-08 Message/history consistency

Goal:

- verify visible final reply equals stored reply

Checks:

- last UI assistant message equals or normalizes exactly to last stored assistant message
- no hidden reasoning survives in persistence

### L1-09 Session reset/recovery

Goal:

- verify reset creates a clean thread without corrupting the client state

Checks:

- new thread ID or rotated session state
- previous in-flight context is cleared
- UI remains usable

### L1-10 Error recovery baseline

Goal:

- verify recoverable stream or API failure yields a user-visible error without corrupting the session

Checks:

- error event is surfaced
- session remains reloadable
- no duplicate ghost messages appear

## 10. L2 Automated Extended Cases

### L2-01 Database natural-language search

Input example:

- `查找 40 到 60 岁、MSI-H、横结肠病例`

Checks:

- NL query is parsed into filters
- result table updates
- warnings/unsupported terms are surfaced if needed

### L2-02 Database structured filter search

Checks:

- filters apply correctly
- pagination and sorting remain stable

### L2-03 Database detail view

Checks:

- selecting a patient loads detail
- available data flags and cards are consistent

### L2-04 Database upsert

Checks:

- edit form saves
- saved values survive reload
- type normalization is correct for numeric fields

### L2-05 Upload flow

Checks:

- file upload succeeds
- asset metadata is returned
- uploaded asset is visible in session state

### L2-06 Upload-to-chat carryover

Checks:

- after upload, follow-up prompt can reference the uploaded content
- session is not corrupted by upload locking or race conditions

### L2-07 Card-driven prompt triggers

Checks:

- clicking card action buttons submits the intended prompt
- follow-up chain completes

### L2-08 Reference display fidelity

Checks:

- title/snippet/page metadata display correctly
- selecting a reference shows the expected preview content

## 11. L3 Manual Review Pack

Manual review must use a checklist with explicit pass/fail/notes fields.

### M1 Medical wording

Reviewers inspect fixture outputs for:

- clinically plausible wording
- no unsafe silent omissions
- appropriate uncertainty language
- alignment between recommendation and provided fixture facts

### M2 Visual quality

Reviewers inspect:

- panel layout
- long-text wrapping
- card readability
- empty states
- loading and error states
- mobile or narrow-width sanity if that is in scope for release

### M3 Card semantics

Reviewers inspect:

- card titles
- metadata labeling
- action-button wording
- consistency between card summary and underlying raw payload

### M4 Image/pathology preview usability

Reviewers inspect:

- preview images render
- thumbnail selection works
- text fallback appears when preview is absent

### M5 Trust and safety presentation

Reviewers inspect:

- safety alerts are prominent
- references are understandable
- no hidden reasoning, prompt fragments, or internal policy text are visible

## 12. Blocking Policy

### Hard blockers

The run fails immediately if any of these occur:

- L0 environment invalid
- any L1 case fails
- safety warning chain fails
- hidden reasoning leak reproduces
- visible reply and persisted history diverge
- session reset or restore corrupts state
- upload breaks session usability

### Conditional-pass defects

These may be tolerated only with explicit sign-off:

- minor visual spacing issues
- non-blocking copy inconsistencies
- low-value extended cases with a clear workaround

Conditional pass is forbidden if the defect affects:

- safety
- trust
- state consistency
- primary workflow completion

## 13. Roles

Recommended run ownership:

- QA or developer: L0, L1, L2 execution
- product/design: visual sign-off
- clinical reviewer or domain owner: medical wording sign-off
- release owner: final go/no-go call

One person can perform multiple roles, but the sign-off dimensions should remain separated in the report.

## 14. Deliverables

A complete acceptance run must produce:

1. `run sheet`
   Environment, build, fixture pack, executors, start/end time.

2. `automation evidence bundle`
   Logs, screenshots, event captures, persisted-state snapshots.

3. `manual review checklist`
   Pass/fail/notes by reviewer.

4. `defect list`
   Blocking vs conditional defects.

5. `release conclusion`
   One of:
   - `PASS`
   - `PASS WITH CONDITIONS`
   - `FAIL`

## 15. Proposed Next Step

After user approval of this design, the implementation plan should produce:

- a repeatable acceptance runbook
- case IDs and fixture mappings
- Playwright/API automation scope
- manual checklist templates
- final report template

That plan should optimize for one-command automation of L0-L2 and a lightweight operator checklist for L3.
