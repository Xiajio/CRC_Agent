# Controlled Release Governance Design

> Version: 2026-07-02
> Scope: P2 Step 12
> Source plan: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`
> Supersedes next-step ordering: the source plan labels Step 12 as CRC cohort feasibility. This design intentionally inserts controlled release governance first because Step 11 exposed release readiness but left all write paths locked.
> Depends on: P0 `HarnessRun` / `ReleaseSafetyReport`, P1 `ClinicalAssertion` / `DoctorActionTrace`, P1.5 Step 10 `LiteratureHarnessRun`, and P1.5 Step 11 Agent Admin Release Dashboard.
> Goal: Move Agent Admin release management from read-only observation to audited governance writes: release intent, human approval, rollback plan, and append-only audit events. Step 12 does not execute release, rollback, policy mutation, RAG ingest, model update, tool deployment, or evidence promotion.

## 1. Background

P0 established the deterministic CRC safety loop:

- intended-use and safety-policy artifacts;
- CRC mutation pack replay;
- `HarnessRun` JSON in `reports/harness/`;
- `ReleaseSafetyReport` JSON in `reports/release_safety/`;
- release decisions and rollback targets.

P1 established the clinical review substrate:

- `ClinicalAssertion` projection;
- doctor-only review read model;
- append-only `DoctorActionTrace`;
- Doctor Review Cockpit behind a feature flag.

P1.5 Step 10 established the shadow literature substrate:

- `EvidenceClaim`, `EvidenceDelta`, and `LiteratureHarnessRun`;
- deterministic literature replay report in `reports/literature/`;
- tests proving unreviewed literature stays out of patient advice, doctor-default paths, and clinical RAG.

P1.5 Step 11 added the Agent Admin Release Dashboard:

- read-only `GET /api/admin/release-dashboard`;
- normalized version chain, harness runs, hard fails, rollback target, sign-off readiness, and literature shadow status;
- disabled release, sign-off, and rollback controls with explicit reasons.

The current gap is not another clinical feature. The gap is a controlled write path for release governance records. Operators need to record intent, approval, and rollback readiness before later execution work can be considered. Step 12 creates that governance layer without adding the execution layer.

## 2. Current Project Context

Relevant existing files:

- `backend/api/services/admin_release_dashboard.py`: reads static reports and normalizes release dashboard state.
- `backend/api/routes/admin.py`: exposes `/api/admin/tools` and `/api/admin/release-dashboard`.
- `backend/app.py`: bearer auth middleware and admin-token path guard.
- `frontend/src/app/api/client.ts`: frontend API client.
- `frontend/src/app/api/types.ts`: frontend admin response types.
- `frontend/src/features/agent-admin/agent-admin-view.tsx`: Agent Admin shell and dashboard resource loading.
- `frontend/src/features/agent-admin/agent-admin-pages.tsx`: Agent Admin task page rendering.
- `frontend/src/features/agent-admin/agent-admin-model.ts`: task catalog and navigation.
- `reports/harness/harness_20260629_001.json`: P0 safety replay evidence.
- `reports/release_safety/release_safety_20260629_001.json`: P0 release safety evidence.
- `reports/literature/literature_harness_20260630_001.json`: Step 10 shadow literature evidence.
- `tests/backend/test_admin_release_dashboard.py`: dashboard normalization tests.
- `tests/backend/test_admin_release_dashboard_api.py`: dashboard route tests.
- `tests/backend/test_auth_security.py`: admin route auth matrix.
- `frontend/src/features/agent-admin/agent-admin-view.test.tsx`: Agent Admin page tests.

Missing Step 12 artifacts:

- `src/contracts/release_governance.py`
- `src/services/release_governance.py`
- `backend/api/schemas/release_governance.py`
- `backend/api/services/release_governance_store.py`
- `reports/release_governance/README.md`
- append-only governance files under `reports/release_governance/`
- backend tests for governance contracts, store, service, API, auth, and non-mutation boundaries
- frontend API types/client methods and Agent Admin governance UI tests

At spec creation time `main` is synchronized with `origin/main` at `a557e0d`.

## 3. Design Decision

Use **audit-only controlled release governance**.

Step 12 adds admin-protected write APIs that create governance records and append audit events. It deliberately stops before any mechanism that changes live runtime behavior. The write path is real, but its writes are constrained to governance artifacts.

### Recommended Approach: Audit-Only Governance

The backend creates:

- release intent records;
- human approval records;
- rollback plan records;
- append-only audit events with a hash chain.

The frontend adds governance controls to the existing Release Dashboard page. These controls submit governance records only. Release and rollback execution buttons remain disabled.

Benefits:

- Creates the first safe write path after the read-only dashboard.
- Gives release managers a real audit trail before execution exists.
- Keeps safety policy, prompts, RAG indexes, feature flags, and model assets untouched.
- Creates a stable contract for a later execution layer.
- Is testable without network access, deployment credentials, model calls, or live infrastructure.

Trade-off:

- Operators still cannot execute release or rollback from the UI. This is intentional; execution needs a separate design with stronger authorization, failure recovery, production environment isolation, and operational ownership.

### Rejected Approach: Local Feature Flag Release

Allowing Step 12 to write feature flag or config state would make governance and execution indistinguishable. It would also require stronger recovery behavior and environment targeting than the current system has.

### Rejected Approach: Execution Hooks

Triggering release or rollback scripts from the admin API would cross into deployment automation. That needs a dedicated execution spec with rate limits, idempotency keys, locks, privileged credentials, failure rollback, observability, and production approval policy.

### Rejected Approach: Evidence Promotion

Promoting Step 10 literature evidence to Project Evidence Pool or clinical RAG during release approval would violate the shadow-only evidence boundary. Literature evidence remains candidate/shadow until a separate evidence governance flow exists.

## 4. Scope

Step 12 includes:

1. `ReleaseIntent`, `ReleaseApproval`, `ReleaseRollbackPlan`, and `ReleaseAuditEvent` contracts.
2. Append-only governance storage under `reports/release_governance/`.
3. A governance service that reads Step 11 dashboard state and writes governance records.
4. Admin-protected read/write APIs for governance records.
5. Validation that only eligible release reports can produce an approval-ready intent.
6. Audit-event hash chaining.
7. Agent Admin governance UI embedded in the existing Release Dashboard task.
8. Tests proving writes are limited to governance artifacts.
9. Tests proving release, rollback, feature flag, RAG, prompt, rubric, route, template, safety policy, and evidence promotion paths are not mutated.

Step 12 excludes:

- Executing release.
- Executing rollback.
- Toggling real feature flags.
- Editing `config/safety_policy.yaml`.
- Editing prompts, rubrics, routes, templates, or tool manifests.
- Running harness scripts from the admin API.
- Running live web search or model calls.
- Writing to clinical RAG indexes.
- Promoting `EvidenceClaim` records to Project Evidence Pool.
- Promoting literature claims to `approved_for_clinical_rag`.
- LearningJob candidate generation.
- CRC cohort feasibility, ethics review, or patient-level research export.
- Broad role-based auth infrastructure beyond the existing admin token.
- External deployment integrations.
- Any edit to `CRC-client/`.

## 5. Architecture

```text
reports/harness/*.json
reports/release_safety/*.json
reports/literature/*.json
  -> Step 11 release dashboard normalizer
    -> Step 12 release governance service
      -> ReleaseIntent / ReleaseApproval / ReleaseRollbackPlan
      -> ReleaseAuditEvent JSONL hash chain
        -> GET /api/admin/release-governance
        -> POST /api/admin/release-governance/intents
        -> POST /api/admin/release-governance/intents/{intent_id}/approvals
        -> POST /api/admin/release-governance/intents/{intent_id}/rollback-plan
          -> Agent Admin Release Dashboard governance panel
```

The backend owns governance validation and persistence. The frontend renders backend state and submits explicit forms. The frontend must not infer approval readiness from raw report JSON.

The governance store is intentionally local and file-backed for this slice. It can be replaced by a database later because all write operations go through a service boundary.

## 6. Storage Layout

Use a dedicated governance directory:

```text
reports/release_governance/
  README.md
  intents/
    release_intent_20260702_001.json
  approvals/
    release_approval_20260702_001.json
  rollback_plans/
    rollback_plan_20260702_001.json
  audit/
    release_audit_20260702.jsonl
```

Rules:

- Every JSON object is UTF-8, deterministic in key order where practical, and JSON-serializable.
- The store may write new files and append audit JSONL.
- The store must not modify existing `reports/harness/`, `reports/release_safety/`, or `reports/literature/` artifacts.
- Existing governance records must not be overwritten. State transitions are represented by new records and audit events.
- The service should tolerate missing governance directory by creating it during a governance write or returning an empty governance state for reads.
- The service should not create governance files during read-only `GET`.

## 7. ReleaseIntent Contract

Minimum object:

```json
{
  "intent_id": "release_intent_20260702_001",
  "source_release_report_id": "release_safety_20260629_001",
  "source_report_path": "reports/release_safety/release_safety_20260629_001.json",
  "harness_run_ids": ["harness_20260629_001"],
  "literature_run_id": "literature_harness_20260630_001",
  "version_chain": {
    "agent_policy_version": "agent_policy_20260629_0",
    "clinical_safety_policy_version": "crc_safety_policy_v0",
    "evidence_index_version": "rag_crc_guideline_20260620",
    "judge_rubric_version": "crc_rubric_v0"
  },
  "release_decision_snapshot": "feature_flag_or_pass",
  "rollback_target": "agent_policy_20260624_0",
  "requested_by": "admin_operator",
  "requested_at": "2026-07-02T00:00:00+08:00",
  "target_scope": "shadow",
  "status": "pending_approval",
  "blocking_summary": {
    "hard_fail_count": 0,
    "literature_isolation_violations": 0,
    "clinical_rag_ingest_enabled": false
  }
}
```

Required fields:

- `intent_id`
- `source_release_report_id`
- `source_report_path`
- `harness_run_ids`
- `version_chain`
- `release_decision_snapshot`
- `rollback_target`
- `requested_by`
- `requested_at`
- `target_scope`
- `status`
- `blocking_summary`

Allowed `target_scope` values:

- `shadow`
- `feature_flag_candidate`

Allowed `status` values:

- `draft`
- `pending_approval`
- `approved`
- `rejected`
- `cancelled`

Rules:

- Step 12 may create `draft` or `pending_approval` intents.
- Step 12 may derive `approved` or `rejected` as a read-model state from approval records.
- Step 12 must not convert an intent into executed release state.
- `feature_flag_candidate` means "candidate for a later feature-flag execution step"; it does not toggle a flag.
- `source_report_path` must be repo-relative.
- `source_release_report_id` must match the current normalized release dashboard source report.
- `rollback_target` must match the Step 11 dashboard rollback target.
- `hard_fail_count > 0` prevents `pending_approval`.
- missing or invalid release safety reports prevent intent creation.
- missing or invalid literature reports prevent `feature_flag_candidate` scope and may still allow `shadow` only if the release manager explicitly records that limitation.

## 8. ReleaseApproval Contract

Minimum object:

```json
{
  "approval_id": "release_approval_20260702_001",
  "intent_id": "release_intent_20260702_001",
  "approver_role": "clinical_safety_reviewer",
  "decision": "approve",
  "reason": "P0 hard fails are zero and Step 10 literature remains shadow-only.",
  "signed_by": "reviewer_admin",
  "signed_at": "2026-07-02T00:10:00+08:00",
  "required": true
}
```

Required fields:

- `approval_id`
- `intent_id`
- `approver_role`
- `decision`
- `reason`
- `signed_by`
- `signed_at`
- `required`

Allowed `approver_role` values:

- `release_manager`
- `clinical_safety_reviewer`
- `evidence_reviewer`

Allowed `decision` values:

- `approve`
- `reject`
- `request_changes`

Approval rules:

- Every approval must reference an existing intent.
- Duplicate approvals for the same `intent_id` and `approver_role` are allowed only as new audit events; the latest decision wins in the derived read model.
- A `reject` or `request_changes` decision prevents derived intent state from becoming `approved`.
- For `target_scope: shadow`, required roles are `release_manager` and `clinical_safety_reviewer`.
- For `target_scope: feature_flag_candidate`, required roles are `release_manager`, `clinical_safety_reviewer`, and `evidence_reviewer`.
- `evidence_reviewer` approval acknowledges that Step 10 literature remains shadow-only. It must not promote literature.
- `reason` must be non-empty.
- Approval writes must append an audit event.

## 9. ReleaseRollbackPlan Contract

Minimum object:

```json
{
  "rollback_plan_id": "rollback_plan_20260702_001",
  "intent_id": "release_intent_20260702_001",
  "rollback_target": "agent_policy_20260624_0",
  "owner": "release_manager",
  "status": "accepted",
  "verification_steps": [
    "Confirm the active release report is release_safety_20260629_001.",
    "Confirm the target policy label is agent_policy_20260624_0.",
    "Run the P0 harness before any future rollback execution.",
    "Confirm Step 10 literature remains shadow-only after rollback."
  ],
  "created_at": "2026-07-02T00:15:00+08:00"
}
```

Required fields:

- `rollback_plan_id`
- `intent_id`
- `rollback_target`
- `owner`
- `status`
- `verification_steps`
- `created_at`

Allowed `status` values:

- `proposed`
- `accepted`

Rules:

- Every rollback plan must reference an existing intent.
- `rollback_target` must match the intent rollback target.
- `verification_steps` must contain at least two explicit human-checkable steps.
- The rollback plan must not execute rollback.
- The rollback plan must not mutate config, policy, prompts, RAG, feature flags, deployment state, or report artifacts.
- Rollback plan writes must append an audit event.

## 10. ReleaseAuditEvent Contract

Minimum object:

```json
{
  "event_id": "release_audit_20260702_001",
  "intent_id": "release_intent_20260702_001",
  "event_type": "approval_recorded",
  "actor": "reviewer_admin",
  "timestamp": "2026-07-02T00:10:00+08:00",
  "payload_hash": "sha256:...",
  "previous_event_hash": "sha256:...",
  "event_hash": "sha256:..."
}
```

Required fields:

- `event_id`
- `intent_id`
- `event_type`
- `actor`
- `timestamp`
- `payload_hash`
- `previous_event_hash`
- `event_hash`

Allowed `event_type` values:

- `intent_created`
- `approval_recorded`
- `rollback_plan_recorded`
- `intent_cancelled`
- `governance_read`

Hash rules:

- `payload_hash` is a SHA-256 hash of the canonical JSON payload that caused the event.
- `previous_event_hash` is the prior event hash for the same `intent_id`, or `sha256:GENESIS` for the first event.
- `event_hash` is a SHA-256 hash of the event content excluding `event_hash`.
- The service must verify the chain when reading governance state.
- Chain verification failure returns an explicit governance integrity warning and prevents new writes for the affected intent.

Privacy rules:

- Audit events must not contain hidden chain-of-thought.
- Audit events must not contain API keys, bearer tokens, deployment credentials, model prompts, raw patient identifiers, or unrelated patient records.
- Actor identifiers should be operational labels, not secrets.

## 11. Governance Read Model

`GET /api/admin/release-governance` returns a derived state:

```json
{
  "dashboard_snapshot": {
    "release_decision": "feature_flag_or_pass",
    "rollback_target": "agent_policy_20260624_0",
    "hard_fail_count": 0,
    "literature_status": "shadow_only"
  },
  "intents": [],
  "active_intent": null,
  "required_approvals": [
    {
      "role": "release_manager",
      "status": "missing",
      "latest_decision": null
    },
    {
      "role": "clinical_safety_reviewer",
      "status": "missing",
      "latest_decision": null
    }
  ],
  "rollback_plan": null,
  "audit_events": [],
  "integrity": {
    "status": "verified",
    "warnings": []
  },
  "disabled_execution_actions": [
    {
      "id": "execute_release",
      "label": "Execute release",
      "reason": "Step 12 records governance only."
    },
    {
      "id": "execute_rollback",
      "label": "Execute rollback",
      "reason": "Rollback execution requires a later execution-path design."
    }
  ],
  "runtime": {
    "auth": "admin",
    "source": "reports/release_governance",
    "mode": "audit_only"
  }
}
```

Rules:

- Reads may combine Step 11 dashboard state with governance records.
- Reads must not create files or append audit unless explicitly configured to audit reads. If read auditing is enabled later, it must be documented and tested separately.
- The read model should prefer the latest non-cancelled intent for `active_intent`.
- The read model must show missing approvals explicitly.
- Execution actions remain disabled even when all required approvals are recorded.

## 12. Admin API

Add endpoints under the existing admin router:

```text
GET  /api/admin/release-governance
POST /api/admin/release-governance/intents
POST /api/admin/release-governance/intents/{intent_id}/approvals
POST /api/admin/release-governance/intents/{intent_id}/rollback-plan
POST /api/admin/release-governance/intents/{intent_id}/cancel
```

### `GET /api/admin/release-governance`

Returns the governance read model.

Validation and behavior:

- Requires admin token.
- Reads release dashboard state through the backend service layer.
- Reads governance files.
- Verifies audit chain.
- Does not write files.

### `POST /api/admin/release-governance/intents`

Request:

```json
{
  "requested_by": "admin_operator",
  "target_scope": "shadow",
  "status": "pending_approval",
  "reason": "Prepare audited governance before controlled release execution exists."
}
```

Behavior:

- Requires admin token.
- Reads current Step 11 release dashboard state.
- Validates eligibility.
- Creates a `ReleaseIntent` file.
- Appends `intent_created` audit event.
- Returns the created intent and updated governance read model.

### `POST /api/admin/release-governance/intents/{intent_id}/approvals`

Request:

```json
{
  "approver_role": "clinical_safety_reviewer",
  "decision": "approve",
  "reason": "Safety harness and release dashboard gates are clear.",
  "signed_by": "reviewer_admin"
}
```

Behavior:

- Requires admin token.
- Validates intent exists and audit chain is intact.
- Validates role and decision enum.
- Writes an approval record.
- Appends `approval_recorded` audit event.
- Returns updated governance read model.

### `POST /api/admin/release-governance/intents/{intent_id}/rollback-plan`

Request:

```json
{
  "owner": "release_manager",
  "status": "accepted",
  "verification_steps": [
    "Confirm release report id before rollback execution.",
    "Run P0 harness after rollback execution in a later step."
  ]
}
```

Behavior:

- Requires admin token.
- Validates intent exists and audit chain is intact.
- Uses the intent rollback target.
- Rejects a request that tries to provide a different rollback target.
- Writes a rollback plan record.
- Appends `rollback_plan_recorded` audit event.
- Returns updated governance read model.

### `POST /api/admin/release-governance/intents/{intent_id}/cancel`

Request:

```json
{
  "actor": "release_manager",
  "reason": "Superseded by a later release report."
}
```

Behavior:

- Requires admin token.
- Appends an audit event and derives the intent as cancelled.
- Does not delete existing records.

## 13. Auth And Permissions

Extend `_requires_admin_token()`:

```python
if path == "/api/admin/release-governance" and method == "GET":
    return True
if path.startswith("/api/admin/release-governance/") and method == "POST":
    return True
```

Rules:

- Existing bearer auth behavior is reused.
- There is no new browser-stored admin secret.
- There is no role-based auth implementation in Step 12.
- `approver_role` is a governance field, not an authentication claim.
- Tests must cover distinct admin token, user token, missing token, invalid token, and no separate admin token fallback.

## 14. Frontend Integration

Add frontend types for:

- `AdminReleaseGovernanceResponse`
- `AdminReleaseIntent`
- `AdminReleaseApproval`
- `AdminReleaseRollbackPlan`
- `AdminReleaseAuditEvent`
- request payloads for intent, approval, rollback plan, and cancel.

Add API client methods:

```ts
getAdminReleaseGovernance()
createAdminReleaseIntent(payload)
recordAdminReleaseApproval(intentId, payload)
recordAdminReleaseRollbackPlan(intentId, payload)
cancelAdminReleaseIntent(intentId, payload)
```

Rules:

- Use the existing API client and headers.
- Do not introduce an admin login flow.
- Do not store admin credentials in the browser.
- Do not calculate approval readiness in frontend-only state.
- Optimistic UI is not required for Step 12; prefer server-confirmed state.

## 15. Agent Admin UI Design

Extend the existing `Release` task page with a governance section.

Recommended layout:

- Governance status strip:
  - active intent status;
  - required approvals complete/missing;
  - rollback plan status;
  - audit integrity status.
- Release intent panel:
  - source release report;
  - target scope;
  - version chain;
  - rollback target;
  - create intent action.
- Required approvals panel:
  - release manager;
  - clinical safety reviewer;
  - evidence reviewer when required;
  - decision, signer, reason, timestamp.
- Rollback plan panel:
  - rollback target;
  - verification steps;
  - owner;
  - status.
- Audit trail panel:
  - event type;
  - actor;
  - timestamp;
  - event hash suffix;
  - integrity warnings.
- Disabled execution controls:
  - Execute release;
  - Execute rollback.

UI action rules:

- Creating intent, recording approval, recording rollback plan, and cancelling intent may be enabled when validation allows.
- Execute release and execute rollback remain disabled in all Step 12 states.
- Approval forms must require role, decision, reason, and signed-by label.
- Rollback plan form must require at least two verification steps.
- Error states must be scoped to governance panels and must not break the release dashboard.
- Mobile and desktop layouts should preserve readable governance status without overlapping text.

## 16. Validation Rules

Intent creation:

- Reject if release dashboard has no valid release safety report.
- Reject `pending_approval` if `hard_fail_count > 0`.
- Reject `pending_approval` if release decision is `block`.
- Reject `feature_flag_candidate` if literature run is missing, invalid, failed, or not `shadow_only`.
- Reject if rollback target is missing.
- Reject if an active intent already exists for the same source release report unless the caller cancels the prior intent first.

Approval:

- Reject unknown role.
- Reject unknown decision.
- Reject empty reason.
- Reject if intent does not exist.
- Reject if audit chain fails verification.
- Reject approval for cancelled intents.
- Allow `request_changes` as a non-final decision that keeps the intent pending.

Rollback plan:

- Reject if intent does not exist.
- Reject if rollback target is missing.
- Reject if caller attempts to override the intent rollback target.
- Reject fewer than two verification steps.
- Reject empty owner.
- Reject unknown status.
- Reject if audit chain fails verification.

Cancellation:

- Reject empty reason.
- Reject if audit chain fails verification.
- Do not delete records.

## 17. Error Handling

Backend:

- Missing governance directory on read: return empty governance state.
- Unparseable governance file: return integrity warning and prevent new writes for the affected intent.
- Audit chain mismatch: return `integrity.status = "failed"` and reject writes for that intent.
- Duplicate active intent: return a typed conflict error.
- Invalid request payload: return `422`.
- Auth failure: keep existing `401` and `403` behavior.
- Filesystem write failure: return a typed API error without pretending the governance action succeeded.

Frontend:

- Loading state: show governance records are loading.
- Empty state: show no release intent has been created.
- Validation error: show field-specific message where possible.
- Integrity failure: show audit chain warning and disable governance writes.
- API error: keep Release Dashboard read-only data visible.
- Execution controls: remain disabled with reasons regardless of governance state.

## 18. Safety And Security Boundaries

Step 12 must preserve these boundaries:

- Governance writes are admin-token protected.
- Governance records do not include secrets, API keys, bearer tokens, deployment credentials, hidden reasoning, model prompts, or raw patient identifiers.
- Governance records may include repo-relative report paths and version labels.
- Governance writes do not mutate release safety reports, harness reports, literature reports, safety policy config, prompts, rubrics, routes, templates, feature flags, RAG indexes, model weights, or tool manifests.
- Governance writes do not change patient, doctor, or clinical default paths.
- Step 10 literature remains shadow-only. Approval acknowledges the shadow boundary; it does not promote claims.
- Doctor feedback remains review signal. It does not automatically change release governance state.
- No live network, deployment system, or model call is required for governance writes.
- `CRC-client/` remains out of scope.

## 19. Interaction With Existing Reports

Release governance records snapshot report state at intent creation time:

- release safety report id;
- release safety report path;
- harness run ids;
- literature run id;
- version chain;
- release decision;
- rollback target;
- hard fail count;
- literature isolation violations;
- clinical RAG ingest disabled flag.

Rules:

- Governance records reference committed report artifacts; they do not rewrite them.
- If a newer release safety report appears later, create a new intent instead of mutating the old one.
- If reports disappear or become invalid after intent creation, the read model should mark dashboard current-state drift and preserve the original intent snapshot.
- Drift does not execute rollback or release; it only informs operators.

## 20. Testing Strategy

Backend contract tests:

- Validate required fields and enum values for `ReleaseIntent`, `ReleaseApproval`, `ReleaseRollbackPlan`, and `ReleaseAuditEvent`.
- Verify deterministic ID generation where implemented.
- Verify audit payload hashing and event hash chaining.
- Reject malformed objects.

Backend store/service tests:

- Empty governance directory returns empty read model.
- Creating an intent writes exactly one intent and one audit event.
- Duplicate active intent for same release report is rejected.
- Hard fail release dashboard blocks `pending_approval`.
- Approval writes an approval and audit event.
- Latest approval per role wins in read model.
- Rejection prevents derived approved state.
- Rollback plan writes a plan and audit event.
- Audit chain mismatch prevents further writes.
- Read operations do not write files.

Backend API/auth tests:

- `GET /api/admin/release-governance` requires admin token.
- Every POST governance route requires admin token.
- User token receives `403` when admin token is distinct.
- Missing or invalid token receives `401`.
- No separate admin token falls back to user token per existing behavior.

Non-mutation tests:

- Governance POST routes do not modify:
  - `config/safety_policy.yaml`;
  - `reports/harness/*`;
  - `reports/release_safety/*`;
  - `reports/literature/*`;
  - prompt/rubric/route/template files;
  - clinical RAG index paths;
  - frontend feature flag config;
  - `CRC-client/`.

Frontend tests:

- API client calls the new endpoints with existing headers.
- Release task loads governance state after release dashboard data.
- Empty governance state renders explicit no-intent message.
- Create intent action renders server-confirmed intent.
- Approval form requires role, decision, reason, and signed-by.
- Rollback plan form requires at least two verification steps.
- Audit trail renders event types and integrity status.
- Execute release and execute rollback controls remain disabled after all approvals.
- Governance API errors do not break the release dashboard.

Regression tests:

- Step 11 admin release dashboard tests still pass.
- Step 10 literature harness tests still pass.
- P1 clinical assertion and doctor review tests still pass.
- P0 safety loop and harness replay tests still pass.

Suggested verification commands:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_governance_contract.py tests/backend/test_release_governance_service.py tests/backend/test_release_governance_api.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_auth_security.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py -q
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/agent-admin/agent-admin-view.test.tsx
```

## 21. Rollout Plan

1. Add release governance contracts and unit tests.
2. Add file-backed governance store with append-only audit events.
3. Add governance service that validates Step 11 dashboard state.
4. Add admin schemas and API routes.
5. Extend admin auth guard for governance routes.
6. Add frontend API types and client methods.
7. Extend Agent Admin Release page with governance panels and forms.
8. Add frontend tests for governance states and disabled execution controls.
9. Run Step 11, Step 10, P1, and P0 regressions.
10. Keep release/rollback execution disabled and document the next execution-path design separately.

## 22. Acceptance Criteria

Step 12 is complete when:

1. `ReleaseIntent`, `ReleaseApproval`, `ReleaseRollbackPlan`, and `ReleaseAuditEvent` contracts exist and are tested.
2. Governance records are written only under `reports/release_governance/`.
3. Audit events are append-only and hash chained.
4. `GET /api/admin/release-governance` returns current dashboard snapshot, intents, approvals, rollback plan, audit trail, integrity state, and disabled execution actions.
5. Admin POST routes can create intent, approval, rollback plan, and cancellation records.
6. Admin auth protects every governance route.
7. Hard fails, block decisions, missing rollback target, and invalid report state prevent approval-ready intent creation.
8. Agent Admin Release Dashboard shows governance state and lets operators record governance actions.
9. Execute release and execute rollback remain disabled in all Step 12 states.
10. Tests prove governance writes do not modify safety policy, prompts, rubrics, routes, templates, RAG indexes, feature flags, release reports, harness reports, literature reports, patient paths, doctor paths, or `CRC-client/`.
11. P0, P1, Step 10, and Step 11 regressions pass.

## 23. Implementation Boundaries

Implementation must preserve these boundaries:

- Do not edit `CRC-client/`.
- Do not execute release or rollback.
- Do not add deployment hooks.
- Do not write feature flag state.
- Do not mutate safety policy, prompt, rubric, route, template, RAG, model, tool, or report artifacts outside `reports/release_governance/`.
- Do not promote literature evidence.
- Do not add patient-level research exports.
- Do not add live network or model calls.
- Do not replace the Step 11 release dashboard normalizer with frontend-only logic.
- Do not introduce broad auth infrastructure in this slice.

## 24. Future Work

After Step 12, later specs may cover:

- Step 13 controlled execution path for feature flag release and rollback.
- Research cohort feasibility and ethics gate from the original roadmap.
- Evidence governance for Project Evidence Pool and clinical RAG promotion.
- LearningJob candidate pipeline with human review and release governance integration.
- Stronger operator identity and role-based access control.
- External deployment system integration.

Each of these must remain separate from the Step 12 audit-only governance design.

## 25. Spec Self-Review

Placeholder scan: no placeholder sections remain.

Internal consistency: the scope, contracts, API, storage, UI, validation, tests, and acceptance criteria all target audit-only release governance. Execution remains explicitly disabled.

Scope check: this is one coherent subsystem. It adds controlled governance writes after the read-only Step 11 dashboard and defers release execution, rollback execution, evidence promotion, cohort feasibility, LearningJob automation, and production integrations.

Ambiguity check: allowed writes, forbidden mutations, approval roles, rollback plan behavior, audit hash chaining, admin auth, and non-mutation requirements are explicitly defined.
