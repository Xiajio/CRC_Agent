# Agent Development And Validation Balance Design

Date: 2026-06-29
Status: Draft for review
Scope: Balance product functionality, agent effectiveness validation, and frontier experimentation for the current LangG CRC-agent codebase.

## Goal

Create a practical development strategy for LangG that keeps feature delivery moving while proving whether the medical agent is effective, safe enough for controlled use, and technically competitive.

The strategy is grounded in the current repository state:

- current branch: `crc-client-integration-verification`
- current product focus: CRC patient triage integration, patient records, care cards, and patient/doctor workspace continuity
- current runtime architecture: FastAPI BFF, POST SSE streaming, patient and doctor LangGraph graphs, registry-backed patient data, RAG/tooling, and React workspace UI
- current validation assets: pytest, Vitest, Playwright acceptance suites, fixture graph runner, `critic`, `evaluator`, `node_timings`, `rag_trace`, references, and Agent Admin designs

## Current Repository Reading

The repo already has the ingredients needed for balanced development. The missing piece is an explicit operating model tying them together.

### Product Functionality

The active implementation direction is CRC triage inside the existing patient workspace rather than embedding the separate `CRC-client` app. This direction is consistent with `docs/superpowers/specs/2026-06-24-patient-crc-triage-subpage-design.md`.

Relevant current files include:

- `frontend/src/features/patient-crc-triage/patient-crc-triage-panel.tsx`
- `frontend/src/features/patient-crc-triage/crc-triage-context.ts`
- `backend/api/routes/crc_triage.py`
- `backend/api/services/patient_commands.py`
- `backend/api/services/patient_care_cards.py`
- `src/services/crc_triage_flow.py`
- `src/services/patient_triage_protocol.py`
- `src/nodes/triage_nodes.py`

The working tree currently contains many modified and untracked files in this area. Treat those files as active WIP and avoid unrelated rewrites.

### Agent Runtime

The runtime is not a single chat endpoint. It is a graph product with observable state and structured events:

- `backend/app.py` wires FastAPI, auth, runtime services, patient/doctor graph services, and routers.
- `src/graph_builder.py` builds the doctor and patient graphs.
- Doctor graph includes planner, knowledge, case database, radiology, pathology, web search, parallel subagents, assessment, diagnosis, staging, decision, critic, citation, evaluator, finalize, and memory.
- Patient graph is intentionally narrower and includes intent, planner, clinical entry, outpatient triage, knowledge, assessment, chat, and general chat.
- Frontend consumes SSE through `frontend/src/app/api/stream.ts` and reduces structured events through `frontend/src/app/store/stream-reducer.ts`.

### Existing Validation Surface

The repo already has several validation layers:

- backend tests under `tests/backend/`
- route and adapter tests under `backend/api/**/test_*.py`
- frontend unit/component tests under `frontend/src/**/*.test.tsx`
- Playwright acceptance and visual suites under `tests/e2e/` and `frontend/playwright*.config.ts`
- controlled demo and fixture runner scripts under `scripts/`
- existing acceptance model in `docs/superpowers/specs/2026-04-11-e2e-full-acceptance-design.md`
- RAG evidence contract in `docs/superpowers/specs/2026-04-29-rag-evidence-contract-design.md`
- Agent Admin observation design in `docs/superpowers/specs/2026-06-14-agent-admin-phase-one-design.md`

This means the best next step is not to invent a separate evaluation platform. The best next step is to make every feature land with its own evidence pack.

## Problem

LangG has three competing pressures:

1. Build visible software: CRC triage, patient records, care cards, doctor review, upload continuity, and admin observation.
2. Prove the agent works: routing correctness, medical safety signals, state consistency, RAG grounding, persistence, and UI behavior.
3. Stay frontier-aware: compare new models, prompting methods, graph policies, RAG profiles, judge methods, and autonomous tool strategies.

If functionality always wins, the project becomes demo-rich but scientifically weak. If evaluation always wins, the product becomes a lab harness without user value. If frontier experiments always win, the codebase becomes unstable and difficult to ship.

The balancing design is to use one vertical product slice as the shared carrier for all three pressures.

## Recommended Strategy

Use CRC patient triage as the first balanced vertical slice.

Every CRC triage increment must ship with four artifacts:

1. Product behavior
   - User-visible UI or API behavior.
   - Clear ownership in existing patient session and patient registry boundaries.

2. State and persistence proof
   - Session snapshot, patient record, care card, or doctor-visible context evidence.
   - No second identity source, no browser-local patient source of truth, and no direct `CRC-client` runtime dependency.

3. Agent evaluation evidence
   - Deterministic case fixtures or golden tests covering routing, risk classification, missing information, persistence, and visible wording.
   - Evidence captured through pytest, Vitest, Playwright, or API snapshot checks.

4. Observation hook
   - Trace, timing, reference, record, or admin-readable state that explains what happened.
   - The observation layer must not expose hidden reasoning or raw credentials.

This turns validation into part of feature delivery rather than a later audit.

## Strategy Options

### Option A: Functionality First

Build CRC triage, patient records, and doctor review as quickly as possible. Add tests only for breakage.

Benefits:

- fastest visible demo progress
- lower short-term coordination cost
- useful while UI and API boundaries are still moving

Costs:

- weak proof that the agent is clinically useful
- hidden regressions in routing, persistence, and safety are likely
- difficult to justify frontier claims

Use this only for throwaway prototypes, not for the main branch.

### Option B: Evaluation First

Create benchmark packs, model comparison reports, and evaluation dashboards before extending the product.

Benefits:

- strongest scientific discipline
- easier to compare agent variants
- clearer evidence for technical claims

Costs:

- product delivery slows down
- benchmark cases may not match real UI/session behavior
- high risk of building a separate harness that the product does not actually use

Use this for later research-quality reports, not for the current integration branch.

### Option C: Vertical Slice With Embedded Validation

Build CRC triage as a complete patient-to-record-to-doctor-review slice, and attach validation gates to each step.

Benefits:

- product and validation reinforce each other
- tests use real code paths and real state contracts
- frontier experiments can run in shadow mode against the same cases
- matches the existing repo architecture

Costs:

- each feature needs slightly more upfront definition
- test fixture hygiene matters
- requires discipline to avoid over-expanding the slice

Recommendation: choose Option C.

## Operating Model

Use a phase-dependent investment ratio.

### Current Phase: Integration Stabilization

Recommended ratio: `60 / 25 / 15`

- 60% functionality: CRC triage UI, backend save route, patient records, care cards, workspace continuity.
- 25% validation: golden cases, route tests, frontend component tests, Playwright/API acceptance evidence.
- 15% frontier experiments: model/prompt/RAG/policy shadow comparisons only.

This ratio fits the current branch because the active WIP is still integrating `CRC-client` concepts into LangG.

### Next Phase: Evidence Hardening

Recommended ratio: `45 / 35 / 20`

- functionality remains active, but feature scope narrows
- validation expands to case packs and acceptance reports
- frontier experiments compare candidate agent variants against controlled cases

Enter this phase after the CRC triage slice can persist a completed assessment and show it in patient records.

### Later Phase: Productized Intelligence

Recommended ratio: `40 / 35 / 25`

- functionality focuses on doctor-facing review and operational workflow
- validation includes longitudinal patient state, RAG evidence quality, and manual clinical review
- frontier work can include autonomous literature search, alternate judge models, and stronger retrieval/routing policies

Enter this phase only when the basic CRC triage workflow is reliable under fixture and local real-mode runs.

## Development Gate For Every Feature

Each feature should pass four gates before it is treated as done.

### Gate 1: Product Contract

The feature has a clear user-visible or API-visible behavior.

Examples:

- patient can answer a CRC triage question
- assessment can be saved through `/api/sessions/{session_id}/crc-triage/assessments`
- latest CRC triage assessment can produce patient care cards
- doctor context can read the saved patient record

### Gate 2: State Contract

The feature updates the intended state boundary and no other source of truth.

For CRC triage:

- patient identity comes from the LangG session and patient registry
- completed assessment flows through `PatientCommandService`
- session snapshot version is bumped after record persistence
- care cards derive from patient records, not from independent frontend storage

### Gate 3: Evaluation Contract

The feature has at least one deterministic verification path.

Acceptable forms:

- pure rule tests for protocol helpers
- backend API tests for route and persistence behavior
- frontend component tests for state rendering
- Playwright acceptance tests for integrated UI flow
- fixture graph tests for controlled SSE/event behavior

### Gate 4: Observation Contract

The feature leaves enough evidence for debugging and later evaluation.

Examples:

- `node_timings` for graph nodes
- `rag_trace` and `retrieved_evidence` for RAG calls
- normalized patient record payloads
- SSE event logs and session snapshots in acceptance runs
- Agent Admin state panels for plan, trace, evidence, tools, and memory

## CRC Triage Case Pack

The first balanced validation pack should contain deterministic CRC triage cases.

### Case 1: Low-Risk Routine Triage

Purpose:

- verify the normal structured flow completes without urgent warnings

Expected evidence:

- current question advances correctly
- final assessment is produced
- risk is low or medium-low according to protocol
- suggested tests are present when appropriate
- no emergency disposition is emitted

### Case 2: Red-Flag High Risk

Purpose:

- verify urgent symptoms interrupt routine collection

Expected evidence:

- red flag count or fatal-risk signal is detected
- disposition is urgent or emergency according to protocol
- routine completion does not hide the urgent recommendation
- care card includes urgent follow-up language

### Case 3: Mid-Risk Backfill

Purpose:

- verify one red flag or uncertain signal triggers targeted follow-up rather than premature finalization

Expected evidence:

- missing or unasked relevant flags are listed
- the flow asks for the missing high-value item
- final assessment preserves the caveat if the user does not answer

### Case 4: Endoscopy Mentioned But Key Finding Missing

Purpose:

- verify the flow blocks final archive when the user mentions an endoscopy but does not give the key result

Expected evidence:

- `missing_information` includes endoscopy result details
- final assessment says information is insufficient
- upload navigation or report supplement path remains available

### Case 5: Topic Switch During Triage

Purpose:

- verify CRC triage state remains recoverable when the user changes topic

Expected evidence:

- normal patient assistant is not polluted by stale CRC triage prompts
- CRC triage state can resume
- off-topic or treatment-plan intent routes away cleanly

### Case 6: Saved Assessment To Patient Records

Purpose:

- verify product state and persistence agree

Expected evidence:

- API save route returns patient id, patient version, projection version, event ids, and record id
- patient records include `crc_triage_assessment`
- care cards derive from the saved payload
- session snapshot version increments

## Measuring Agent Effectiveness

Agent effectiveness should be measured with a small number of dimensions that map to product risk.

### Task Completion

Question:

- did the user complete the intended flow without backend intervention?

Signals:

- successful final assessment
- saved patient record
- visible confirmation in UI
- no blocking errors

### Clinical Safety Behavior

Question:

- did the system correctly escalate red flags and avoid unsafe closure?

Signals:

- fatal-risk and red-flag protocol outputs
- missing critical information
- urgent disposition language
- manual review of high-risk case wording

### State Consistency

Question:

- do UI state, session snapshot, patient record, and derived cards agree?

Signals:

- `GET /api/sessions/{id}` snapshot
- patient records API response
- care card payload
- frontend rendered summary

### Groundedness And Evidence

Question:

- when medical guidance requires evidence, does the agent preserve source and retrieval metadata?

Signals:

- `retrieved_evidence`
- `retrieved_references`
- `rag_trace`
- citation coverage
- evaluator and critic signals

### Interaction Quality

Question:

- does the agent ask useful next questions instead of asking too much, too little, or the wrong thing?

Signals:

- question order
- repeated non-answer behavior
- clear missing-information prompts
- topic-switch handling

### Latency And Operational Reliability

Question:

- is the flow responsive enough for local/demo use and diagnosable when slow?

Signals:

- `node_timings`
- frontend latency trace
- stream completion events
- retry/error recovery behavior

## Frontier Experiment Policy

Frontier work is valuable only when it is comparable and reversible.

Allowed frontier experiments in the current phase:

- prompt variants for triage clarification
- alternate model providers for triage summarization
- stricter or looser routing policies for topic switching
- RAG profile changes for evidence-heavy doctor review
- alternate LLM-judge prompts for evaluator/critic review
- shadow-mode autonomous literature search for future Agent Admin learning readiness

Rules:

- do not put a frontier experiment directly on the main user path without a baseline comparison
- do not change persistence contracts to support an experiment
- do not use one-off subjective output quality as proof
- compare candidate behavior on the same CRC case pack
- promote only if it improves a named metric without breaking state consistency or safety behavior

Recommended promotion path:

```text
idea -> shadow fixture comparison -> focused tests -> optional feature flag -> default path
```

## Relationship To Agent Admin

Agent Admin should be treated as the observation surface, not as the first dependency of CRC triage.

Short-term:

- use existing local state, tests, and API snapshots as evidence
- keep Agent Admin read-only
- show available trace, evidence, memory, plan, and tool metadata when available

Medium-term:

- expose sanitized `retrieved_evidence`, `rag_trace`, `node_timings`, critic/evaluator signals, and patient context through admin APIs
- render partial data clearly
- keep hidden reasoning and secrets out of admin payloads

Do not block CRC triage MVP on a full Agent Admin implementation.

## Relationship To Production Readiness

Production hardening is important but should not displace the current balance goal.

Do now:

- preserve current auth and session boundaries
- avoid adding new source-of-truth stores
- keep tests deterministic
- document single-worker and fixture assumptions when they matter

Defer unless directly blocking the CRC slice:

- Redis distributed run locks
- SSE resume protocol
- OAuth/OIDC migration
- full deployment metrics stack

Those topics already have a roadmap in `docs/superpowers/specs/2026-06-13-production-readiness-roadmap-design.md`.

## Immediate Next Steps

### Step 1: Stabilize The Current CRC Triage WIP

Finish the active WIP without broadening scope.

Focus files:

- `frontend/src/features/patient-crc-triage/*`
- `frontend/src/features/patient-records/*`
- `backend/api/routes/crc_triage.py`
- `backend/api/services/patient_commands.py`
- `backend/api/services/patient_care_cards.py`
- `src/services/crc_triage_flow.py`
- `src/services/patient_triage_protocol.py`
- `src/nodes/triage_nodes.py`

Expected result:

- patient can complete a CRC triage assessment
- assessment can be saved
- patient record and care cards are consistent
- normal patient assistant remains separate

### Step 2: Build The CRC Case Pack

Add or consolidate tests for the six case pack scenarios.

Minimum test types:

- pure protocol tests for `patient_triage_protocol`
- flow-state tests for `crc_triage_flow`
- API tests for save route and patient records
- frontend tests for panel state and saved summary rendering

Expected result:

- every new triage feature has deterministic proof
- regressions identify whether the failure is protocol, persistence, frontend, or graph routing

### Step 3: Add Integrated Acceptance Evidence

Create one Playwright or scripted acceptance path using fixture mode.

Expected result:

- captured UI screenshot or trace
- captured final session snapshot
- captured patient record response
- visible record/care-card consistency

### Step 4: Start Shadow Frontier Comparison

Use the same case pack to compare one candidate improvement at a time.

Good first candidate:

- compare current deterministic protocol summary against an LLM-assisted summary that must preserve the same structured fields

Promotion criteria:

- field completeness is equal or better
- urgent/missing-information behavior is unchanged or safer
- patient-facing wording is clearer under manual review
- latency remains acceptable for the flow

## Risks

### Risk: Evaluation Becomes A Separate Product

Mitigation:

- tie every evaluation artifact to a product feature and a real code path
- use fixture mode and API snapshots before creating dashboards

### Risk: Frontier Experiments Destabilize The User Path

Mitigation:

- require shadow comparison first
- keep experiment output compatible with existing structured payloads
- promote through feature flags or isolated route changes

### Risk: CRC-client Concepts Reintroduce A Second Architecture

Mitigation:

- do not use CRC-client local IDs as LangG patient identity
- do not use CSV/local storage as runtime source of truth
- translate useful protocol rules into backend-owned Python services

### Risk: Tests Prove Too Little

Mitigation:

- require state consistency evidence, not only visible UI text
- pair frontend tests with backend/API checks for saved records
- use acceptance evidence rules from the existing E2E acceptance design

## Acceptance Criteria For This Strategy

This balance strategy is adopted when:

- current CRC triage work is treated as the primary vertical slice
- each new feature is required to pass product, state, evaluation, and observation gates
- CRC triage has a named deterministic case pack
- frontier experiments run in shadow mode against the same case pack
- Agent Admin remains an observation surface and does not block the CRC MVP
- production-readiness work is scoped to non-disruptive boundaries until the CRC slice is stable

## Review Questions

The main decision to review is whether CRC triage should be the first balanced vertical slice. If yes, implementation planning should focus on stabilizing the current branch's CRC triage WIP and adding the case pack. If no, choose a different vertical slice and reuse the same four-gate model.
