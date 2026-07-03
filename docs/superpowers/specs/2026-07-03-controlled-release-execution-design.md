# Controlled Release Execution Design

> Version: 2026-07-03
> Scope: P2 Step 13
> Source context: Step 12 `docs/superpowers/specs/2026-07-02-controlled-release-governance-design.md`
> Goal: Add a controlled, audited execution path for feature-flag release and rollback after Step 12 governance approval, without integrating external deployment systems or mutating clinical safety artifacts.

## 1. Background

Step 11 created a read-only Agent Admin Release Dashboard.

Step 12 created audit-only release governance:

- `ReleaseIntent`, `ReleaseApproval`, `ReleaseRollbackPlan`, and `ReleaseAuditEvent` contracts.
- `ReleaseGovernanceStore` under `reports/release_governance/`.
- `ReleaseGovernanceService` that derives approval state from committed governance records.
- Admin-protected governance APIs.
- Agent Admin UI forms for intent, approval, rollback plan, and cancel.
- Disabled `Execute release` and `Execute rollback` controls.

Step 13 moves one step further. It adds execution records and a bounded local feature-flag state artifact. It still avoids production deployment credentials, subprocess execution, network calls, policy edits, prompt edits, RAG mutations, literature promotion, and patient or doctor path changes.

## 2. Current Project Context

Relevant current files:

- `src/contracts/release_governance.py`: Step 12 governance contracts and hash helpers.
- `backend/api/services/release_governance_store.py`: file-backed governance store with integrity checks.
- `src/services/release_governance.py`: governance read model and disabled execution actions.
- `backend/api/routes/admin.py`: admin release dashboard and governance routes.
- `backend/api/schemas/release_governance.py`: governance request schemas.
- `backend/app.py`: bearer auth and admin route protection.
- `frontend/src/app/api/types.ts`: release dashboard and governance types.
- `frontend/src/app/api/client.ts`: admin governance client methods.
- `frontend/src/features/agent-admin/agent-admin-pages.tsx`: release governance UI panels.
- `tests/backend/test_release_governance_*.py`: Step 12 contract, store, service, API, and non-mutation coverage.

Important observed boundary:

- There is no existing runtime feature-flag config file in production code.
- `doctor_review_cockpit_v0` exists as a response label, not as a general writable flag framework.
- `config/safety_policy.yaml` and `config/intended_use_profiles.yaml` must not be used as the Step 13 execution target.

## 3. Design Options

### Option A: External Deployment Integration

The admin API could call a deployment system, CI workflow, or feature flag service.

Rejected for Step 13. The repo has no deployment integration contract, credential handling, environment isolation, or production ownership model. Adding that now would mix application governance with external operational infrastructure.

### Option B: Direct Config Mutation

The admin API could mutate a source config or safety policy file to represent the release.

Rejected. That would violate the Step 12 non-mutation boundary and risk changing clinical behavior without a dedicated runtime rollout design.

### Option C: Local File-Backed Execution Ledger And Flag State

The admin API writes execution requests, results, audit events, and a local feature-flag state under `reports/release_execution/`.

Recommended. It gives operators a real controlled execution path with idempotency, rollback, audit, and preflight gates while keeping all writes inside a dedicated execution ledger. The state is local and inspectable. Runtime clinical paths do not consume it in Step 13.

## 4. Scope

Step 13 includes:

1. Execution contracts for release requests, rollback requests, execution results, local feature-flag state, and execution audit events.
2. A file-backed execution store under `reports/release_execution/`.
3. A local feature-flag executor that writes only `reports/release_execution/feature_flags/current.json` and immutable history snapshots.
4. A release execution service that reads Step 12 governance and Step 11 dashboard state before executing.
5. Admin-protected read and execute APIs.
6. Agent Admin execution UI that enables release/rollback only when the backend preflight allows it.
7. Idempotency handling for repeated execute requests.
8. Rollback from the latest successful Step 13 release execution to the Step 12 rollback target.
9. Tests proving Step 13 writes only under `reports/release_execution/`.
10. Regression tests for Step 12, Step 11, Step 10, P1, and P0.

Step 13 excludes:

- External deployment providers.
- CI workflow dispatch.
- Shell scripts or subprocess execution from admin routes.
- Network calls.
- Production credentials.
- Editing `config/safety_policy.yaml`.
- Editing prompts, rubrics, routes, templates, RAG indexes, model weights, or tool manifests.
- Promoting literature evidence.
- Writing to clinical RAG.
- Changing patient or doctor default flows.
- Editing `CRC-client/`.
- General role-based auth beyond existing admin bearer-token protection.

## 5. Architecture

```text
Step 11 release dashboard
  + Step 12 release governance read model
    -> Step 13 release execution service
      -> preflight gates
      -> idempotency check
      -> local feature flag executor
      -> release execution store
        -> requests/*.json
        -> results/*.json
        -> feature_flags/current.json
        -> feature_flags/history/*.json
        -> audit/release_execution_YYYYMMDD.jsonl
          -> GET /api/admin/release-execution
          -> POST /api/admin/release-execution/release
          -> POST /api/admin/release-execution/rollback
            -> Agent Admin Release page execution panel
```

The execution service owns all gates. The frontend renders backend readiness and submits explicit execution commands. The frontend must not infer readiness from raw governance records.

## 6. Storage Layout

Use a new dedicated directory:

```text
reports/release_execution/
  README.md
  requests/
    release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b.json
  results/
    release_result_release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b.json
  feature_flags/
    current.json
    history/
      release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b.json
  audit/
    release_execution_20260703.jsonl
```

Rules:

- Reads must not create files.
- Writes may create only files under `reports/release_execution/`.
- Existing request, result, and history files are write-once.
- `feature_flags/current.json` may be replaced atomically by release or rollback execution.
- Every write is paired with an append-only execution audit event.
- Store integrity failure prevents new writes.
- Store path checks must reject symlinks, path traversal, reserved Windows device names, and resolved paths outside the execution root.

## 7. Execution Contracts

### ReleaseExecutionRequest

```json
{
  "execution_id": "release_exec_release_intent_abc12345",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "action": "release",
  "requested_by": "release_manager",
  "requested_at": "2026-07-03T09:00:00+08:00",
  "idempotency_key": "release-intent-001-release-20260703",
  "reason": "All required governance approvals are complete.",
  "expected_governance_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "expected_rollback_plan_id": "rollback_plan_release_intent_abc12345",
  "target_flag_state": {
    "flag_name": "doctor_review_cockpit_v0",
    "enabled": true,
    "scope": "feature_flag_candidate"
  }
}
```

Rules:

- `action` is `release` or `rollback`.
- `intent_id`, `requested_by`, `requested_at`, `idempotency_key`, and `reason` are required.
- `expected_governance_hash` is a canonical hash of the active governance intent, required approvals, rollback plan, and dashboard snapshot used for preflight.
- `expected_rollback_plan_id` must match the active accepted rollback plan.
- `target_flag_state.flag_name` is `doctor_review_cockpit_v0` for Step 13.
- Release requests set `enabled: true`.
- Rollback requests set `enabled: false` and include `rollback_target`.

### ReleaseExecutionResult

```json
{
  "result_id": "release_result_release_exec_abc12345",
  "execution_id": "release_exec_release_intent_abc12345",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "action": "release",
  "status": "succeeded",
  "started_at": "2026-07-03T09:00:00+08:00",
  "finished_at": "2026-07-03T09:00:00+08:00",
  "actor": "release_manager",
  "previous_flag_state": null,
  "new_flag_state": {
    "flag_name": "doctor_review_cockpit_v0",
    "enabled": true,
    "scope": "feature_flag_candidate",
    "source_intent_id": "release_intent_release_safety_20260629_001_6da729a0",
    "rollback_target": "agent_policy_20260624_0",
    "updated_at": "2026-07-03T09:00:00+08:00"
  },
  "failure_reason": null
}
```

Rules:

- `status` is `succeeded` or `failed`.
- Failed results must include `failure_reason`.
- A result must reference an existing request.
- Replaying the same `idempotency_key` returns the existing result.
- Reusing an idempotency key with different payload returns conflict.

### FeatureFlagState

```json
{
  "flag_name": "doctor_review_cockpit_v0",
  "enabled": true,
  "scope": "feature_flag_candidate",
  "source_intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "source_execution_id": "release_exec_release_intent_abc12345",
  "rollback_target": "agent_policy_20260624_0",
  "updated_by": "release_manager",
  "updated_at": "2026-07-03T09:00:00+08:00"
}
```

Rules:

- Step 13 writes this as local execution state only.
- Runtime clinical paths do not consume it in Step 13.
- Rollback sets `enabled: false` and keeps source execution provenance.

### ReleaseExecutionAuditEvent

```json
{
  "event_id": "release_execution_audit_release_requested_abc12345",
  "execution_id": "release_exec_release_intent_abc12345",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "event_type": "release_succeeded",
  "actor": "release_manager",
  "timestamp": "2026-07-03T09:00:00+08:00",
  "payload_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "previous_event_hash": "sha256:GENESIS",
  "event_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
}
```

Allowed event types:

- `release_requested`
- `release_succeeded`
- `release_failed`
- `rollback_requested`
- `rollback_succeeded`
- `rollback_failed`
- `execution_read`

Audit rules mirror Step 12:

- canonical JSON payload hash;
- per-execution hash chain;
- no secrets, tokens, prompts, hidden reasoning, or patient identifiers;
- chain verification on read;
- failed chain blocks writes for the affected execution.

## 8. Preflight Gates

Release execution requires all of these:

1. Step 12 governance integrity is `verified`.
2. Active intent exists and matches the submitted `intent_id`.
3. Active intent `target_scope` is `feature_flag_candidate`.
4. Active intent derived status is `approved`.
5. Required approvals are all `approved`.
6. Latest rollback plan exists and has `status: accepted`.
7. Current Step 11 dashboard still matches the active intent snapshot:
   - release report id;
   - release decision;
   - rollback target;
   - version chain;
   - hard fail count;
   - literature status.
8. Current Step 11 dashboard has `hard_fail_count: 0`.
9. Current Step 11 dashboard release decision is `feature_flag_or_pass`.
10. Current literature status is `shadow_only`.
11. No successful release execution is already active for the same intent unless the request uses the same idempotency key.
12. Execution store integrity is `verified`.

Rollback execution requires all of these:

1. Step 12 governance integrity is `verified`.
2. Active intent exists and matches the submitted `intent_id`, or the submitted intent has a successful prior release execution.
3. Accepted rollback plan exists and matches `expected_rollback_plan_id`.
4. A successful release execution exists for the same intent.
5. Current feature flag state is enabled for the same `source_intent_id`.
6. Execution store integrity is `verified`.
7. The rollback request idempotency key is new or matches an existing rollback request with identical payload.

## 9. Read Model

`GET /api/admin/release-execution` returns:

```json
{
  "governance": {
    "active_intent_id": "release_intent_release_safety_20260629_001_6da729a0",
    "derived_status": "approved",
    "required_approvals_complete": true,
    "rollback_plan_id": "rollback_plan_release_intent_abc12345"
  },
  "preflight": {
    "release": {
      "allowed": true,
      "reasons": []
    },
    "rollback": {
      "allowed": false,
      "reasons": ["No successful release execution exists for this intent."]
    }
  },
  "feature_flag_state": null,
  "requests": [],
  "results": [],
  "audit_events": [],
  "integrity": {
    "status": "verified",
    "warnings": []
  },
  "runtime": {
    "auth": "admin",
    "source": "reports/release_execution",
    "mode": "controlled_local_execution"
  }
}
```

Rules:

- Read combines governance read model and execution store state.
- Read does not write files.
- Read lists disallowed reasons explicitly.
- Frontend action availability is derived from this response.

## 10. Admin API

Add routes:

```text
GET  /api/admin/release-execution
POST /api/admin/release-execution/release
POST /api/admin/release-execution/rollback
```

### Release Request

```json
{
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "requested_by": "release_manager",
  "idempotency_key": "release-intent-001-release-20260703",
  "reason": "All required governance approvals are complete.",
  "expected_rollback_plan_id": "rollback_plan_release_intent_abc12345"
}
```

Behavior:

- Requires admin token.
- Runs preflight.
- Writes a request artifact.
- Writes local feature flag state.
- Writes a result artifact.
- Appends execution audit events.
- Returns updated execution read model.

### Rollback Request

```json
{
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "requested_by": "release_manager",
  "idempotency_key": "release-intent-001-rollback-20260703",
  "reason": "Rollback to the Step 12 rollback target.",
  "expected_rollback_plan_id": "rollback_plan_release_intent_abc12345"
}
```

Behavior:

- Requires admin token.
- Runs rollback preflight.
- Writes a rollback request artifact.
- Writes local feature flag state with `enabled: false`.
- Writes a rollback result artifact.
- Appends execution audit events.
- Returns updated execution read model.

## 11. Auth

Extend `_requires_admin_token()`:

```python
if method == "GET" and path == "/api/admin/release-execution":
    return True
if method == "POST" and path.startswith("/api/admin/release-execution/"):
    return True
```

Rules:

- Existing bearer auth behavior is reused.
- No browser-stored admin secret is added.
- No role-based auth is added in Step 13.
- `requested_by` remains an audit label, not an authentication claim.

## 12. Frontend

Extend the Release task page with an execution section below the Step 12 governance panels.

UI states:

- Loading execution state.
- Preflight blocked with reason list.
- Release ready.
- Release succeeded with current flag state.
- Rollback ready after successful release.
- Rollback succeeded with disabled flag state.
- Integrity failed, with writes disabled.
- API error, while dashboard and governance panels remain visible.

Rules:

- The UI uses backend `preflight.release.allowed` and `preflight.rollback.allowed`.
- The UI must not infer allowed actions from local form state.
- Release and rollback forms require actor, reason, idempotency key, and expected rollback plan id.
- Buttons remain disabled when backend preflight disallows the action.
- Existing governance forms remain available according to Step 12 behavior.

## 13. Error Handling

Backend:

- Invalid request payload: `422`.
- Governance preflight failure: `409`.
- Execution integrity failure: `409`.
- Idempotency payload mismatch: `409`.
- Filesystem write failure: `500`.
- Auth failure: existing `401` / `403`.

Frontend:

- Render preflight reasons near execution controls.
- Preserve release dashboard and governance state when execution API fails.
- Show execution action error without resetting form inputs.

## 14. Safety And Security Boundaries

Step 13 writes only:

- `reports/release_execution/requests/*.json`
- `reports/release_execution/results/*.json`
- `reports/release_execution/feature_flags/current.json`
- `reports/release_execution/feature_flags/history/*.json`
- `reports/release_execution/audit/*.jsonl`
- `reports/release_execution/README.md`

Step 13 must not write:

- `reports/release_governance/`
- `reports/harness/`
- `reports/release_safety/`
- `reports/literature/`
- `config/`
- prompt, rubric, route, template, RAG, model, or tool files
- patient registry or session data
- `CRC-client/`

Execution payloads must not contain:

- API keys;
- bearer tokens;
- deployment credentials;
- model prompts;
- hidden reasoning;
- raw patient identifiers;
- unrelated patient records.

## 15. Testing Strategy

Backend contract tests:

- release and rollback request validation;
- feature flag state validation;
- execution result validation;
- audit hash chain validation;
- forbidden payload key rejection.

Store and executor tests:

- empty execution root returns verified empty state;
- release writes request, result, flag current, flag history, and audit;
- rollback writes request, result, flag current, flag history, and audit;
- reads do not write;
- duplicate idempotency key with identical payload returns existing result;
- duplicate idempotency key with different payload fails;
- audit or artifact tampering blocks writes;
- symlink/path traversal/root containment protections.

Service tests:

- release blocked when no active governance intent exists;
- release blocked when target scope is `shadow`;
- release blocked when required approvals are missing;
- release blocked when rollback plan is missing or not accepted;
- release blocked when dashboard drifts from intent snapshot;
- release succeeds when governance is approved and preflight is clean;
- rollback blocked before successful release;
- rollback succeeds after successful release;
- rollback blocked when current flag state belongs to another intent.

API/auth tests:

- all execution routes require admin token;
- user token receives `403` when admin token differs;
- missing or invalid token receives `401`;
- release and rollback endpoints map validation/preflight/idempotency errors.

Frontend tests:

- API client uses correct endpoints, headers, and JSON bodies;
- Release page loads execution state;
- blocked preflight reasons render;
- release button enables only when backend says allowed;
- rollback button enables only after backend says allowed;
- execution API errors do not break dashboard or governance panels.

Non-mutation tests:

- release and rollback execution do not mutate Step 12 governance files;
- release and rollback execution do not mutate Step 11/P0/P1/Step10 reports;
- release and rollback execution do not mutate config, prompts, RAG, tool manifests, patient/doctor paths, or `CRC-client/`.

Regression tests:

- Step 12 governance backend and frontend tests pass.
- Step 11 admin release dashboard tests pass.
- Step 10 literature harness tests pass.
- P1 doctor review tests pass.
- P0 safety loop tests pass.

## 16. Acceptance Criteria

Step 13 is complete when:

1. Execution contracts exist and are tested.
2. File-backed execution store exists under `reports/release_execution/`.
3. Local feature flag state can be released and rolled back through admin-only APIs.
4. Release execution is blocked unless Step 12 governance is approved, integrity is verified, target scope is `feature_flag_candidate`, and rollback plan is accepted.
5. Rollback execution is blocked unless a successful release execution exists for the same intent.
6. Execution requests are idempotent.
7. Execution audit events are append-only and hash chained.
8. Agent Admin Release page shows execution preflight, current local flag state, and release/rollback actions.
9. Tests prove Step 13 writes only under `reports/release_execution/`.
10. Step 12, Step 11, Step 10, P1, and P0 regressions pass.

## 17. Future Work

Later specs may cover:

- connecting local execution state to a real runtime feature-flag reader;
- production deployment provider integration;
- operator identity and role-based authorization;
- execution monitoring and alerting;
- evidence promotion governance;
- research cohort feasibility and ethics gates.

These remain outside Step 13.

## 18. Spec Self-Review

Marker scan: no unresolved work markers remain.

Internal consistency: execution writes are scoped to `reports/release_execution/`, while governance remains Step 12 and audit-only. The preflight gates, API, UI, and tests all enforce that separation.

Scope check: this is one coherent subsystem: controlled local release/rollback execution after governance approval. External deployment, runtime consumption, clinical config mutation, and evidence promotion remain separate future work.

Ambiguity check: storage paths, allowed writes, forbidden writes, preflight gates, idempotency, rollback conditions, auth, and acceptance criteria are explicit.
