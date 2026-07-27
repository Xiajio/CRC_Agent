# Admin Shadow Research Console Design

> Version: 2026-07-13
> Scope: Agent Admin 「自动科研 / Learning」子任务产品化（1 → 2）
> Depends on: Literature Harness (L0 shadow), LearningJob candidate pipeline, Cohort Feasibility Admin API, Release Dashboard literature aggregation, Agent Admin Live Observability
> Goal: Turn the Agent Admin Learning page from a roadmap placeholder into a shadow-only research observation console, then add explicit manual triggers for existing Admin shadow APIs — without entering patient/doctor default clinical paths.

## 1. Background

LangG already has several **shadow research / learning** substrates:

- `LiteratureHarnessRun` (L0 shadow) via `scripts/run_literature_harness.py` → `reports/literature/`
- `LearningJob` / inert candidate patches via `GET|POST /api/admin/learning-jobs` → `reports/learning_jobs/`
- Cohort feasibility via `POST /api/admin/research/cohort-feasibility` (shadow mode)
- Release Dashboard aggregation of literature runs via `GET /api/admin/release-dashboard`

The Agent Admin **学习** page still presents a fake pipeline (roadmap badges, disabled scheduler actions). Operators cannot see live shadow artifacts or manually create shadow jobs from the UI.

Product direction agreed with stakeholders:

```text
Phase 1 — Read-only observation
  wire existing Admin APIs / dashboard literature slices into Learning page

Phase 2 — Manual shadow triggers
  enable explicit UI actions that call existing Admin APIs
  results remain shadow / inert
```

Out of scope for this design:

- Daily scheduler / cron
- Promoting literature into clinical RAG, prompts, or training data
- Injecting `search_latest_research` into patient/doctor graphs by default
- Auto-applying LearningJob candidate patches
- New third workspace surface (keep Agent Admin)

## 2. Current Project Context

### Backend (already exists)

| Capability | Entry | Notes |
|---|---|---|
| Literature harness replay | `scripts/run_literature_harness.py`, `src/services/literature_harness.py` | Offline; writes `reports/literature/` |
| Literature in release board | `GET /api/admin/release-dashboard` | Aggregates literature shadow runs |
| Learning jobs list/create | `GET|POST /api/admin/learning-jobs` | Shadow store under `reports/learning_jobs/` |
| Cohort feasibility | `POST /api/admin/research/cohort-feasibility` | Shadow evaluation over registry projection |
| Tool candidate | `search_latest_research` in `src/tools/manifest.py` | `executor_only` / `candidate`; not in clinical web-search set |

### Frontend (gap)

| Surface | Today |
|---|---|
| Agent Admin Learning page | Roadmap placeholder; disabled Run now / Write KB / Enable scheduler |
| Agent Admin Evidence page | Session `references` only — not literature harness ledger |
| Agent Admin Release page | Already shows literature harness summary for release gates |
| API client | Has release dashboard; **missing** `getAdminLearningJobs` / `createAdminLearningJob` / cohort feasibility client methods |

### Related specs

- `docs/superpowers/specs/2026-06-30-evidenceclaim-literature-harness-design.md`
- `docs/superpowers/specs/2026-07-08-learningjob-candidate-pipeline-design.md`
- `docs/superpowers/specs/2026-07-08-crc-cohort-feasibility-design.md`
- `docs/superpowers/specs/2026-06-30-agent-admin-release-dashboard-design.md`
- `docs/superpowers/plans/2026-07-12-agent-admin-live-observability.md` (honesty badges / live vs catalog)

## 3. Design Decision

### Recommended approach: Upgrade Learning page in two phases

Reuse the existing Agent Admin shell and **Learning** task id (`learning`). Rename copy to **自动科研 / Learning（影子）** without adding a new top-level surface.

**Phase 1** binds read APIs and release-dashboard literature slices.
**Phase 2** enables manual POSTs that already exist on the backend.

Benefits:

- Small blast radius; reuses audited Admin auth and shadow stores
- Release page stays the release-gate surface; Learning becomes the research-shadow workbench
- Matches existing honesty badges (`runtime-api` / `roadmap`)

### Rejected approaches

| Approach | Why rejected |
|---|---|
| New `/research` route outside Agent Admin | Splits admin auth/session context; Phase 1 design already chose third surface = Agent Admin only |
| Wire literature into Evidence page | Evidence is session citation pool; mixing harness ledgers confuses operators |
| Auto-run harness on page load | Side effects and long jobs; Phase 2 must be explicit user intent |
| Enable `search_latest_research` in doctor graph | Out of scope; separate product decision after shadow console exists |

## 4. Hard Boundaries (non-negotiable)

Every panel and action on this page MUST communicate:

1. **Shadow only** — outputs are review material, not clinical facts.
2. **No clinical default-path mutation** — patient graph / doctor graph behavior unchanged.
3. **No RAG / prompt / safety_policy writes** from this console.
4. **No scheduler** — Phase 2 is manual click only.
5. **LearningJob candidates remain inert** until a separate human-reviewed release governance path promotes them.
6. **Literature claims** require human sign-off before any promotion (already enforced by literature harness contracts).
7. **Structured model transport** — manual Auto Research and experimental Diagram generation require `LLM_MODE=API` with a function-calling compatible endpoint; in-process Local HF/VLLM backends fail closed with `503` until they implement the same structured-output contract.

UI must keep disabled (or omit) controls for: Enable scheduler, Write knowledge base, Promote to RAG, Apply candidate patch.

## 5. Information Architecture

### Page chrome

```text
[AgentAdminSourceBadge] runtime-api | roadmap (when empty/unwired sections remain)
Title: 自动科研（影子）
Subtitle: Literature harness · LearningJob · Cohort feasibility — not clinical default path
```

### Layout

```text
┌─────────────────────────────────────────────────────────────┐
│ Metric strip                                                 │
│  literature runs | isolation violations | learning jobs |    │
│  candidates (inert)                                          │
├──────────────────────────┬──────────────────────────────────┤
│ Primary                  │ Secondary                         │
│ · Literature harness     │ · Selected item inspector         │
│   ledger (from release   │ · Shadow action panel (Phase 2)   │
│   dashboard slice)       │ · Hard-boundary copy              │
│ · LearningJob list       │                                   │
├──────────────────────────┴──────────────────────────────────┤
│ Cohort feasibility strip (Phase 1: empty/help; Phase 2: form)│
└─────────────────────────────────────────────────────────────┘
```

Release page continues to own version-chain / blocking-gates / sign-off readiness. This page deep-links to Release when an operator needs gate context (`onNavigateTask("release")`).

## 6. Phase 1 — Read-only Observation

### 6.1 Data sources

| Panel | Client call | Render |
|---|---|---|
| Literature ledger | Reuse `getAdminReleaseDashboard()`; filter `runs` where `kind === "literature_shadow_harness"` (and summary fields `literature_*`) | run id, status (`shadow_only` / fail / missing), isolation violations, source path |
| Learning jobs | **New** `getAdminLearningJobs()` → `GET /api/admin/learning-jobs` | job id, status, signal count, candidate count, requested_by, updated_at |
| Metrics | Derived from the two payloads above | Honest `n/a` when resource error/idle |
| Empty states | When reports missing or API 404/empty | 「尚无影子报告 / 尚无 LearningJob」+ how to generate offline (`run_literature_harness.py`) |

### 6.2 Frontend modules

- `frontend/src/app/api/types.ts` — `AdminLearningJobsResponse` (mirror backend `read_jobs()` shape)
- `frontend/src/app/api/client.ts` — `getAdminLearningJobs()`
- `frontend/src/features/agent-admin/agent-admin-view.tsx` — load learning jobs (+ optional refresh of release dashboard) when `activeTaskId === "learning"`
- `frontend/src/features/agent-admin/agent-admin-pages.tsx` — replace `LearningPage` placeholder body with live panels
- `frontend/src/features/agent-admin/agent-admin-model.ts` — helpers to project dashboard/literature + learning-job rows; keep `buildLearningReadiness` only if still useful as boundary copy, or retire fake readiness metrics

### 6.3 Honesty

- Success → `AgentAdminSourceBadge source="runtime-api"`
- Fallback / error with no data → `catalog` or `unavailable` with explicit message (do not invent job rows)
- Remove fake “first two pipeline stages success” entirely (already partially done; Phase 1 must not reintroduce)

### 6.4 Phase 1 acceptance

- With fixture/report data present, Learning page shows real literature run ids and/or learning job ids
- Without data, empty state is honest
- No POST actions yet
- Patient/doctor graphs unchanged
- Vitest covers: loading/success/error for learning jobs; literature slice rendering from dashboard fixture; absence of Enable scheduler / Write KB executable buttons

## 7. Phase 2 — Manual Shadow Triggers

### 7.1 Actions (explicit user intent)

| Action | API | UI |
|---|---|---|
| Create LearningJob | `POST /api/admin/learning-jobs` | Form: `requested_by`, `idempotency_key`, minimal signal list (reuse `CreateLearningJobRequest` fields) |
| Run cohort feasibility | `POST /api/admin/research/cohort-feasibility` | Form: minimal `CohortFeasibilityRequest` fields already accepted by backend |
| Refresh literature view | `GET /api/admin/release-dashboard` | Button「刷新文献 harness」— does **not** spawn offline script |

### 7.2 Literature harness “run” policy

Phase 2 **does not** shell out to `python scripts/run_literature_harness.py` from the browser by default.

Rationale: script execution needs process isolation, path sandboxing, and long-running job UX not yet present.

If a later slice adds “trigger literature replay”, it must be a **new Admin endpoint** that:

- requires admin token
- writes only under `reports/literature/`
- returns the new run id for refresh
- never mutates RAG / prompts

That endpoint is **optional follow-on**, not required to close Phase 2.

### 7.3 Action UX rules

- Buttons labeled with `shadow` / `inert` / `不进入默认流`
- On success: refresh list + show job/result id
- On 409/422: show API detail without retry loops that create duplicates (respect idempotency_key)
- Keep Apply / Promote / Scheduler controls disabled or absent

### 7.4 Client additions

- `createAdminLearningJob(request)`
- `evaluateAdminCohortFeasibility(request)`
- Types for request/response payloads matching backend schemas

### 7.5 Phase 2 acceptance

- Creating a LearningJob from UI appears in the list after refresh and remains inert
- Cohort feasibility POST returns shadow result and does not write clinical sessions
- No path from UI to “write knowledge base”
- Contract tests + frontend tests for success/error action states
- Non-mutation sentinels (existing learning-job / cohort tests) remain green

## 8. Data Flow

```text
Operator opens Agent Admin → Learning
        │
        ├─ GET /api/admin/release-dashboard ──► literature run slice
        ├─ GET /api/admin/learning-jobs ──────► job + candidate ledger
        │
        └─ (Phase 2 only)
              ├─ POST /api/admin/learning-jobs
              └─ POST /api/admin/research/cohort-feasibility
                        │
                        ▼
              reports/learning_jobs/ or shadow JSON response
                        │
                        ▼
              UI refresh — still shadow / inert
```

No SSE / graph turn is required for this console. It does not depend on patient/doctor `SessionState` except optional context strip already shown by Agent Admin chrome.

## 9. Auth and Safety

- All endpoints already behind Admin bearer (`_requires_admin_token` / admin routes). Frontend continues using existing `ApiClient` admin token configuration.
- Do not introduce patient-identifying fields into LearningJob UI beyond what the API already allows; prefer opaque refs already used by learning-job contracts.
- Cohort feasibility may use registry research projection — keep UI copy that patient-level export remains gated by existing service ethics checks.

## 10. Testing Strategy

### Phase 1

- Frontend unit/integration: Learning page with mocked `getAdminLearningJobs` + `getAdminReleaseDashboard`
- Assert literature kind filtering and empty states
- Assert disabled/absent promote/scheduler controls

### Phase 2

- Frontend: form submit → create job → list refresh
- Frontend: cohort form submit → result panel
- Backend: existing `tests/backend/test_learning_jobs_api.py`, `test_research_api.py`, non-mutation suites must still pass
- Optional e2e smoke later; not required for first delivery

## 11. Documentation

- Update README Agent Admin / 学习 bullet: Phase 1 observation + Phase 2 manual shadow triggers; still not scheduled auto-research
- Keep `reports/literature/README.md` and `reports/learning_jobs/README.md` as source-of-truth for artifact semantics

## 12. Delivery Order

1. **Phase 1 implementation plan** — client + Learning page read-only wiring
2. **Phase 1 verification** — fixtures / empty honesty
3. **Phase 2 implementation plan** — create job + cohort feasibility forms
4. **Optional follow-on** — Admin-triggered literature harness replay endpoint

Recommended first ship: **Phase 1 alone** is valuable and mergeable; Phase 2 can follow immediately after.

## 13. Open Follow-ups (explicitly deferred)

- Scheduler / daily paper ingestion
- Promoting `search_latest_research` into clinical tool sets
- Literature harness process runner API
- Cross-linking DoctorActionTrace signals into LearningJob create form autofill
- Dedicated EvidenceClaim card gallery beyond release-dashboard summary fields

## 14. Spec Self-Check

- Covers Phase 1 read-only and Phase 2 manual triggers as agreed (1 → 2)
- Hard boundaries prevent clinical default-path leakage
- Reuses existing APIs; no invented third graph
- Literature “run script from UI” deferred with rationale
- Acceptance criteria listed per phase
- No placeholder TBD sections for in-scope work
