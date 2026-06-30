# Agent Admin Release Dashboard Design

> Version: 2026-06-30
> Scope: P1.5 Step 11
> Source plan: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`
> Depends on: P0 `HarnessRun` / `ReleaseSafetyReport`, P1 `ClinicalAssertion` / `DoctorActionTrace`, and P1.5 Step 10 `LiteratureHarnessRun`
> Goal: Add a read-only Agent Admin release dashboard that shows version chain, harness runs, hard fails, rollback target, human sign-off readiness, and Step 10 shadow evidence status without publishing, mutating policy, or promoting candidate literature.

## 1. Background

P0 created the deterministic CRC safety loop:

- intended-use and safety-policy artifacts;
- CRC mutation pack replay;
- `HarnessRun` JSON in `reports/harness/`;
- `ReleaseSafetyReport` JSON in `reports/release_safety/`;
- safety and persistence regression tests.

P1 created the clinical review substrate:

- `ClinicalAssertion` projection;
- doctor-only review read model;
- append-only `DoctorActionTrace`;
- frontend Doctor Review Cockpit behind a feature flag.

P1.5 Step 10 created the literature evidence substrate:

- `EvidenceClaim`, `EvidenceDelta`, and `LiteratureHarnessRun`;
- fixed literature claim pack fixture;
- deterministic shadow replay report in `reports/literature/`;
- tests proving unreviewed literature stays out of patient, doctor-default, and clinical RAG paths.

Step 11 should make those release artifacts visible to an operator in Agent Admin. It is an observation surface, not a release executor. The dashboard must make readiness and blocking gates obvious while keeping all release, rollback, sign-off, policy editing, RAG ingest, and learning-job controls disabled or absent.

## 2. Current Project Context

Relevant existing files:

- `backend/api/routes/admin.py`: existing read-only admin route for `/api/admin/tools`.
- `backend/app.py`: bearer auth middleware and `_requires_admin_token()` admin path guard.
- `frontend/src/app/api/client.ts`: frontend API client with `getAdminTools()`.
- `frontend/src/app/api/types.ts`: frontend admin response types.
- `frontend/src/features/agent-admin/agent-admin-view.tsx`: Agent Admin shell, task navigation, and tools API loading.
- `frontend/src/features/agent-admin/agent-admin-model.ts`: Agent Admin task catalog and model helpers.
- `frontend/src/features/agent-admin/agent-admin-pages.tsx`: Agent Admin task page renderer.
- `frontend/src/features/agent-admin/agent-admin-view.test.tsx`: Agent Admin behavior tests.
- `tests/backend/test_auth_security.py`: admin bearer-token protection tests.
- `reports/harness/harness_20260629_001.json`: P0 harness evidence.
- `reports/release_safety/release_safety_20260629_001.json`: P0 release safety evidence.
- `reports/literature/literature_harness_20260630_001.json`: Step 10 literature shadow evidence.

The current Agent Admin surface already has a read-only shell, subtask rail, admin theme, and runtime tools manifest page. Step 11 should extend that surface rather than creating a standalone route or a new admin application.

## 3. Design Decision

Use the approved approach: **read-only release dashboard API plus Agent Admin page**.

The backend adds one admin-protected read-only endpoint:

```text
GET /api/admin/release-dashboard
```

The endpoint reads committed report JSON files, normalizes them into a small release dashboard response, and returns only safe metadata. It must not run harnesses, execute scripts, call models, access the network, write files, write databases, change feature flags, update safety policy, update RAG stores, or record human sign-off.

The frontend adds one Agent Admin task page:

```text
Release Dashboard
```

The page calls `getAdminReleaseDashboard()`, renders real backend data when available, and shows explicit loading, error, and empty states. Disabled release, sign-off, and rollback affordances may be shown only as non-interactive read-only boundary indicators with reasons.

### Rejected Approach: Frontend Static Page Only

A frontend-only page would be fast, but it would not prove that the backend can safely read committed release artifacts or that the endpoint is protected by the admin bearer token. It would also leave the page vulnerable to stale hardcoded data.

### Rejected Approach: Full Release Console

Adding human sign-off submission, feature-flag release, monitoring activation, rollback, or policy editing would cross the Step 11 boundary. Those actions require a later write-path design with audit, authorization, rollback, and production-token handling.

## 4. Scope

Step 11 includes:

1. A backend normalization layer for release dashboard data.
2. A read-only admin endpoint for the normalized dashboard payload.
3. Admin bearer-token protection for the new endpoint.
4. Frontend API types and client method for the dashboard.
5. A new Agent Admin task entry and page.
6. UI sections for:
   - version chain;
   - harness runs;
   - hard fail summary;
   - rollback target;
   - release decision;
   - human sign-off readiness;
   - Step 10 literature shadow status;
   - disabled mutation controls with explicit reasons.
7. Backend tests for normalization, auth, missing-report handling, and read-only behavior.
8. Frontend tests for API client behavior, page rendering, loading/error states, and disabled controls.

Step 11 excludes:

- Creating, editing, or approving release reports.
- Writing human sign-off records.
- Toggling feature flags.
- Running harness scripts from the API.
- Publishing or rolling back a release.
- Promoting `EvidenceClaim` records to Project Evidence Pool.
- Ingesting literature claims into clinical RAG.
- Updating prompt, rubric, route, template, or safety policy files.
- LearningJob candidate generation.
- P2 research cohort feasibility.
- Any edit to `CRC-client/`.

## 5. Architecture

```text
reports/harness/harness_20260629_001.json
reports/release_safety/release_safety_20260629_001.json
reports/literature/literature_harness_20260630_001.json
  -> backend release dashboard reader / normalizer
    -> GET /api/admin/release-dashboard
      -> frontend API client
        -> Agent Admin Release Dashboard task page
```

The backend owns all interpretation of report files. The frontend should render the API response, not recalculate release readiness from raw JSON. This keeps safety and release semantics in one place and avoids drift between backend tests and the admin page.

The dashboard is intentionally shallow. It reads the latest committed static artifacts and reports what they say. It does not inspect live runtime state or infer deployment health.

## 6. Backend Data Model

The API response should be stable, small, and explicit:

```json
{
  "version_chain": {
    "agent_policy_version": "agent_policy_20260629_0",
    "clinical_safety_policy_version": "crc_safety_policy_v0",
    "evidence_index_version": "rag_crc_guideline_20260620",
    "judge_rubric_version": "crc_rubric_v0"
  },
  "release_decision": "feature_flag_or_pass",
  "rollback_target": "agent_policy_20260624_0",
  "human_signoff": {
    "required": true,
    "status": "missing",
    "reason": "Step 11 is read-only; sign-off must be recorded by a later audited write path."
  },
  "summary": {
    "hard_fail_count": 0,
    "p0_cases_total": 5,
    "p0_cases_passed": 5,
    "literature_claims": 3,
    "literature_isolation_violations": 0,
    "clinical_rag_ingest_enabled": false
  },
  "runs": [
    {
      "run_id": "harness_20260629_001",
      "kind": "p0_crc_harness",
      "status": "pass",
      "source_path": "reports/harness/harness_20260629_001.json",
      "hard_fail_count": 0
    },
    {
      "run_id": "release_safety_20260629_001",
      "kind": "release_safety",
      "status": "pass",
      "source_path": "reports/release_safety/release_safety_20260629_001.json",
      "hard_fail_count": 0
    },
    {
      "run_id": "literature_harness_20260630_001",
      "kind": "literature_shadow_harness",
      "status": "shadow_only",
      "source_path": "reports/literature/literature_harness_20260630_001.json",
      "hard_fail_count": 0
    }
  ],
  "blocking_gates": [
    {
      "id": "no_literature_patient_default",
      "label": "Unreviewed literature stays out of patient default path",
      "state": "locked",
      "reason": "Step 10 report has 0 isolation violations."
    },
    {
      "id": "no_literature_clinical_rag",
      "label": "Unreviewed literature stays out of clinical RAG",
      "state": "locked",
      "reason": "Clinical RAG ingest is disabled in Step 11."
    }
  ],
  "disabled_actions": [
    {
      "id": "record_human_signoff",
      "label": "Record human sign-off",
      "reason": "Requires a later audited write-path design."
    },
    {
      "id": "publish_feature_flag",
      "label": "Publish feature flag release",
      "reason": "Step 11 observes readiness only."
    },
    {
      "id": "rollback_release",
      "label": "Rollback release",
      "reason": "Rollback execution is outside this read-only slice."
    }
  ],
  "runtime": {
    "auth": "admin",
    "source": "reports/static_release_artifacts",
    "mode": "read_only"
  }
}
```

Allowed status values:

- run `status`: `pass`, `fail`, `shadow_only`, `missing`, `invalid`;
- gate `state`: `pass`, `locked`, `warning`, `blocked`, `missing`;
- human sign-off `status`: `missing`, `recorded_elsewhere`, `not_required`.

If a report is missing or invalid, the endpoint should still return a dashboard payload with the affected run marked `missing` or `invalid`; it should not crash the entire admin page. The only exception is an unexpected filesystem or JSON parsing error that prevents all dashboard construction, which can return a normal API error.

## 7. Normalization Rules

The backend normalizer should apply deterministic rules:

1. Prefer the release safety report for `version_chain`, `release_decision`, and `rollback_target`.
2. Use the P0 harness report for P0 case totals and hard fail details.
3. Use the Step 10 literature report for `literature_claims`, `negative_or_conflicting_claims`, and `literature_isolation_violations`.
4. Treat literature run status as `shadow_only` when:
   - the report exists;
   - its release decision or run-level semantics do not approve clinical RAG ingest;
   - `isolation_violations == 0`.
5. Mark clinical RAG ingest as disabled for all Step 11 responses.
6. Mark human sign-off as required and missing because Step 11 has no write path.
7. Mark publish, rollback, and sign-off actions as disabled even when all read-only gates pass.
8. Do not include raw claim text beyond short display snippets if the frontend later needs claim detail; the first dashboard can show counts and status only.

These rules should be unit-tested directly. Tests should not depend on external time, network, model output, or generated files.

## 8. Admin API

Extend `backend/api/routes/admin.py` with:

```text
GET /api/admin/release-dashboard
```

The route should:

- use existing FastAPI route patterns;
- call a pure helper or service that reads and normalizes reports;
- return JSON-serializable dictionaries or Pydantic models consistent with existing backend style;
- not mutate `request.app.state`;
- not execute any harness script;
- not create or update report files.

Extend `backend/app.py::_requires_admin_token()`:

```python
if method == "GET" and path == "/api/admin/release-dashboard":
    return True
```

The endpoint should reuse the existing bearer auth behavior:

- distinct admin token: user token receives `403`;
- admin token succeeds;
- no separate admin token: user token succeeds through existing fallback;
- invalid or missing token receives `401`.

## 9. Frontend Integration

Add frontend types in `frontend/src/app/api/types.ts` for the release dashboard response. Use string literal unions for known statuses and allow safe forward-compatible strings only where the backend intentionally may add values later.

Add `getAdminReleaseDashboard()` to `frontend/src/app/api/client.ts`. It should call:

```text
/api/admin/release-dashboard
```

with existing default headers. It must not introduce a new auth token path, local storage key, admin login flow, or browser-held admin secret.

Extend `AgentAdminView` to load release dashboard data when the active task is `release`. This should mirror the existing `tools` resource pattern:

- idle before the task is selected;
- loading while the API call is active;
- success with normalized data;
- error with status and message.

The page should keep the rest of Agent Admin functional if the release dashboard endpoint fails.

## 10. Frontend Page Design

Add a new Agent Admin task:

```text
id: "release"
label: "Release"
detailTitle: "Release Dashboard"
description: "version chain / harness runs / rollback target / sign-off readiness"
status: "read-only"
```

Recommended layout:

- KPI strip:
  - hard fails;
  - P0 pass count;
  - literature claim count;
  - isolation violations;
  - release decision.
- Version chain:
  - agent policy version;
  - clinical safety policy version;
  - evidence index version;
  - judge rubric version.
- Runs table:
  - run id;
  - kind;
  - status;
  - source path;
  - hard fail count.
- Blocking gates panel:
  - patient default path isolation;
  - doctor default path isolation;
  - clinical RAG ingest disabled;
  - human sign-off readiness.
- Disabled actions panel:
  - record human sign-off;
  - publish feature flag;
  - rollback release.

The page should reuse existing Agent Admin components such as `AgentAdminMetricStrip`, `AgentAdminPanel`, `AgentAdminStatusChip`, `AgentAdminDisabledAction`, and existing CSS classes where possible. Add new CSS only for layout pieces that cannot be expressed with current primitives.

The design should remain operational and dense, not marketing-like. It should use stable dimensions for tables and controls so status strings do not shift layout. It must remain readable on mobile and desktop.

## 11. Security and Safety Boundaries

Step 11 must preserve these boundaries:

- `/api/admin/release-dashboard` is admin-token protected when bearer auth is enabled.
- The endpoint does not return secrets, API keys, hidden reasoning, raw token values, filesystem absolute paths, or model prompts.
- The endpoint may return repo-relative report paths such as `reports/harness/harness_20260629_001.json`.
- The endpoint does not expose patient-level records or private patient identifiers.
- The endpoint does not promote candidate literature.
- The endpoint does not write sign-off, release, rollback, RAG, policy, or prompt changes.
- Frontend disabled controls must be non-interactive and covered by tests.

## 12. Error Handling

Backend:

- Missing report file: return a dashboard with the related run marked `missing`.
- Malformed report JSON: return a dashboard with the related run marked `invalid`.
- Missing optional fields: use explicit `missing` status or `null` fields rather than inventing success.
- Unknown extra report fields: ignore unless they affect known safety gates.

Frontend:

- Loading state: show that release artifacts are being read.
- Error state: show `release dashboard unavailable` with status/message.
- Empty or missing report state: render the rest of the dashboard and mark affected panels as missing.
- Disabled action state: show why each mutation is disabled.

## 13. Testing Strategy

Backend tests:

- Normalization from the three committed report fixtures produces the expected version chain, run rows, hard fail summary, rollback target, and Step 10 shadow status.
- Missing P0 harness report marks P0 run `missing` without hiding Step 10 data.
- Malformed literature report marks literature run `invalid` and blocks promotion.
- `/api/admin/release-dashboard` is protected by admin token in the same auth cases as `/api/admin/tools`.
- The route is read-only: tests should verify no harness script execution and no report file writes.

Frontend tests:

- `getAdminReleaseDashboard()` calls `/api/admin/release-dashboard` and preserves configured headers.
- Agent Admin nav includes the release task.
- Selecting the release task triggers exactly one dashboard request.
- Success state renders version chain, run rows, hard fail count, rollback target, and Step 10 shadow status.
- Loading and error states render scoped messages without breaking other Agent Admin pages.
- Disabled sign-off, publish, and rollback controls are visible as disabled/non-clickable controls with reasons.

Manual verification:

- Run focused backend admin/release tests.
- Run focused frontend API and Agent Admin tests.
- Run P0/P1/Step10 backend regressions after implementation because Step 11 reads their artifacts.
- If frontend UI changes are material, inspect the Agent Admin release page in browser at desktop and mobile widths.

## 14. Acceptance Criteria

Step 11 is complete when:

1. `GET /api/admin/release-dashboard` returns a normalized read-only release dashboard payload from committed static reports.
2. The endpoint is protected by admin bearer-token rules.
3. Agent Admin has a `Release` task page that renders the dashboard.
4. The dashboard displays version chain, harness runs, hard fails, rollback target, human sign-off readiness, and Step 10 literature shadow status.
5. Release, sign-off, and rollback actions are absent or visibly disabled with reasons.
6. Missing or invalid report files produce explicit missing/invalid states.
7. Tests cover backend normalization, auth, frontend client, release page rendering, and disabled controls.
8. No patient, doctor-default, clinical RAG, safety policy, prompt, rubric, route, template, training, or `CRC-client/` path is modified by Step 11.

## 15. Implementation Notes For The Next Plan

The implementation plan should likely split work into these tasks:

1. Backend release dashboard normalizer and tests.
2. Admin API endpoint and auth tests.
3. Frontend API types/client and tests.
4. Agent Admin task/page model and rendering tests.
5. Final integration verification, including P0/P1/Step10 backend regressions.

The implementation should use an isolated worktree or branch from current `main`. The current `main` already contains Step 10 and is synchronized with `origin/main`.

## 16. Self-Review

Placeholder scan: no unresolved placeholder text remains.

Internal consistency: the backend owns report interpretation, the frontend renders the normalized API payload, and every release-related action stays read-only.

Scope check: this is one coherent Step 11 subsystem. It adds a release observability slice to the existing Agent Admin surface and does not include release execution, research cohort feasibility, or LearningJob automation.

Ambiguity check: missing and malformed report behavior, human sign-off status, clinical RAG ingest status, disabled actions, and admin auth behavior are explicitly defined.
