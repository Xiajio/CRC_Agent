# Post-Release Closure And Evidence Package Design

> Version: 2026-07-07
> Scope: P2 Step 15
> Source context: Step 14 `docs/superpowers/specs/2026-07-03-post-release-monitoring-design.md`
> Goal: Add an audited post-release closure layer that lets Agent Admin close a controlled release or rollback observation period and generate a local evidence package, without changing runtime feature flag behavior, suppressing monitoring alerts, or exporting patient-level data.

## 1. Background

Step 11 added a read-only Agent Admin Release Dashboard.

Step 12 added audit-only release governance:

- release intent creation;
- role-based approval records;
- rollback plan records;
- governance audit events under `reports/release_governance/`.

Step 13 added controlled local release execution:

- release and rollback execution requests;
- local feature flag state artifacts;
- execution results and audit events under `reports/release_execution/`.

Step 14 added post-release monitoring:

- explicit monitoring check records;
- derived alerts;
- rollback trigger candidates;
- alert acknowledgements under `reports/release_monitoring/`.

The remaining operational gap is closure. After monitoring checks are recorded and alerts are resolved, acknowledged, or followed by rollback, there is no single audited record that says the release observation period is closed, why it is closed, what evidence was reviewed, and what package should be handed to an operator or reviewer. Step 15 adds that closure layer. It is a local ledger and read model, not a runtime release mechanism.

## 2. Current Project Context

Relevant current files:

- `src/contracts/release_governance.py`: Step 12 governance contracts and hash helpers.
- `backend/api/services/release_governance_store.py`: Step 12 file-backed governance store.
- `src/services/release_governance.py`: Step 12 governance service and active intent derivation.
- `src/contracts/release_execution.py`: Step 13 execution contracts, feature flag state, and audit hashing.
- `backend/api/services/release_execution_store.py`: Step 13 execution store.
- `src/services/release_execution.py`: Step 13 preflight and local execution orchestration.
- `src/contracts/release_monitoring.py`: Step 14 monitoring contracts, alerts, acknowledgements, and audit hashing.
- `backend/api/services/release_monitoring_store.py`: Step 14 monitoring store.
- `src/services/release_monitoring.py`: Step 14 monitoring read model, required checks, alerts, and rollback trigger derivation.
- `backend/api/routes/admin.py`: admin release dashboard, governance, execution, and monitoring routes.
- `backend/app.py`: bearer auth and admin route protection.
- `frontend/src/app/api/types.ts`: admin release dashboard, governance, execution, and monitoring types.
- `frontend/src/app/api/client.ts`: admin release API client methods.
- `frontend/src/features/agent-admin/agent-admin-view.tsx`: Agent Admin release resource loading and mutation handlers.
- `frontend/src/features/agent-admin/agent-admin-pages.tsx`: Agent Admin Release page panels.

Important current boundary:

- Step 14 acknowledgements do not close a release.
- Step 14 rollback trigger candidates do not execute rollback.
- Step 13 local feature flag state is still not consumed by patient or doctor runtime paths.
- No external deployment provider, alert provider, signing authority, or archival system exists.
- Closure must not read patient-level runtime data, transcripts, raw patient identifiers, doctor draft text, or hidden reasoning.

## 3. Design Options

### Option A: Close Releases By Mutating Governance Or Monitoring Records

Step 15 could write a final status back into Step 12 governance or Step 14 monitoring records.

Rejected. Step 12 and Step 14 are append-only ledgers with their own meanings. Mutating them would blur approvals, monitoring evidence, and closure outcome. Closure should be a separate ledger that references those artifacts.

### Option B: Export A PDF Or Send Evidence To An External Archive

Step 15 could generate a PDF package, push it to WPS, send email, or upload to a compliance archive.

Rejected for this slice. The repo has no archive provider contract, delivery guarantee, operator identity model, or document rendering dependency in the release stack. A later spec can add rendered exports after the JSON evidence package is stable.

### Option C: Local Closure Ledger And Evidence Package

Step 15 creates `reports/release_closure/` with write-once closure records, write-once evidence packages, and append-only audit events. A closure service reads Step 11 dashboard state, Step 12 governance state, Step 13 execution state, and Step 14 monitoring state to derive closure readiness. Admin APIs expose the read model and record closure decisions. The frontend renders a closure panel below monitoring.

Recommended. It completes the local release lifecycle while preserving all existing boundaries. Closure is explicit, auditable, and inspectable, but it does not mutate governance, execution, monitoring, or clinical runtime paths.

## 4. Scope

Step 15 includes:

1. Contracts for `ReleaseClosureRecord`, `ReleaseEvidencePackage`, `ReleaseClosureGate`, and `ReleaseClosureAuditEvent`.
2. A file-backed closure store under `reports/release_closure/`.
3. A closure service that reads Step 11, Step 12, Step 13, Step 14 state and derives closure readiness.
4. Admin-protected APIs to read closure state and record a closure.
5. Automatic local JSON evidence package generation when a closure is recorded.
6. Agent Admin Release page closure panel below Step 14 monitoring.
7. Tests proving closure writes only under `reports/release_closure/`.
8. Regression tests for Step 14 monitoring, Step 13 execution, Step 12 governance, Step 11 dashboard, Step 10 literature, P1 doctor review, and P0 safety loop.

Step 15 excludes:

- Executing release or rollback.
- Automatically acknowledging monitoring alerts.
- Suppressing monitoring alerts.
- Mutating Step 14 monitoring files.
- Mutating Step 13 execution files or feature flag state.
- Mutating Step 12 governance files.
- Generating PDFs, Word documents, or WPS artifacts.
- Sending email, Slack, PagerDuty, or webhook notifications.
- External archive upload.
- Scheduled closure reminders.
- Role-based authentication beyond existing admin bearer auth.
- Runtime consumption of local feature flag state.
- Patient-level telemetry, raw patient identifiers, session transcripts, doctor note content, or hidden reasoning.
- Evidence promotion into clinical RAG.
- Research cohort export or ethics review.
- Editing safety policy, prompts, rubrics, routes, templates, RAG indexes, model weights, tool manifests, or `CRC-client/`.

## 5. Architecture

```text
Step 11 release dashboard
  + Step 12 release governance read model
  + Step 13 release execution read model
  + Step 14 release monitoring read model
    -> Step 15 release closure service
      -> closure gate
      -> closure record
      -> evidence package
      -> closure audit events
        -> reports/release_closure/
          -> GET /api/admin/release-closure
          -> POST /api/admin/release-closure/closures
            -> Agent Admin Release page closure panel
```

The closure service owns readiness derivation. The frontend renders backend-derived gate state and submits an explicit closure request. The frontend must not decide that a release is closeable from form state.

## 6. Storage Layout

Use a new dedicated directory:

```text
reports/release_closure/
  README.md
  closures/
    release_closure_release_exec_abc12345_8d6e5f11.json
  packages/
    release_evidence_package_release_closure_release_exec_abc12345_8d6e5f11.json
  audit/
    release_closure_20260707.jsonl
```

Rules:

- Reads must not create files.
- Writes may create only files under `reports/release_closure/`.
- Closure records are write-once.
- Evidence packages are write-once.
- Audit JSONL is append-only.
- Each closure write creates one closure record, one evidence package, and two audit events: `closure_recorded` and `evidence_package_generated`.
- Store integrity failure prevents new writes.
- Store path checks must reject symlinks, path traversal, reserved Windows device names, and resolved paths outside the closure root.
- Closure artifacts must reference release intent IDs, release execution IDs, optional rollback execution IDs, monitoring check IDs, alert IDs, and acknowledgement IDs. They must not contain patient identifiers.

## 7. Contracts

### ReleaseClosureRecord

```json
{
  "closure_id": "release_closure_release_exec_abc12345_8d6e5f11",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "release_execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "rollback_execution_id": null,
  "closure_status": "accepted",
  "closed_by": "release_manager",
  "closed_at": "2026-07-07T10:00:00+08:00",
  "rationale": "Required post-release checks passed and no active critical monitoring alerts remain.",
  "monitoring_snapshot_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "dashboard_snapshot_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "governance_snapshot_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "execution_snapshot_hash": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "required_check_ids": [
    "release_monitor_check_release_exec_abc12345_p0_harness_replay_291a1a2b"
  ],
  "acknowledged_alert_ids": [],
  "unresolved_alert_ids": [],
  "rollback_trigger_candidate_id": null,
  "evidence_package_id": "release_evidence_package_release_closure_release_exec_abc12345_8d6e5f11",
  "idempotency_key": "close-release-20260707-001"
}
```

Allowed `closure_status` values:

- `accepted`
- `accepted_with_observations`
- `rolled_back`

Rules:

- `accepted` requires all required Step 14 checks to be present with `pass`, no active warning or critical alerts, no rollback trigger candidate, and verified dashboard, governance, execution, monitoring, and closure integrity.
- `accepted_with_observations` allows warning checks and acknowledged warning alerts. It still rejects active critical alerts and rollback trigger candidates.
- `rolled_back` requires a successful Step 13 rollback after the latest successful release for the same intent.
- A closure record never changes Step 13 feature flag state.
- Reusing the same `idempotency_key` with an identical payload returns the existing closure.
- Reusing the same `idempotency_key` with a different payload returns conflict.

### ReleaseEvidencePackage

```json
{
  "package_id": "release_evidence_package_release_closure_release_exec_abc12345_8d6e5f11",
  "closure_id": "release_closure_release_exec_abc12345_8d6e5f11",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "release_execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "rollback_execution_id": null,
  "generated_by": "release_manager",
  "generated_at": "2026-07-07T10:00:00+08:00",
  "closure_status": "accepted",
  "summary": "Release observation period closed after required checks passed.",
  "source_refs": [
    "GET /api/admin/release-dashboard",
    "GET /api/admin/release-governance",
    "GET /api/admin/release-execution",
    "GET /api/admin/release-monitoring"
  ],
  "artifact_refs": [
    "reports/release_closure/closures/release_closure_release_exec_abc12345_8d6e5f11.json"
  ],
  "snapshot_hashes": {
    "dashboard": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "governance": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "execution": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "monitoring": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }
}
```

Rules:

- The package is JSON only.
- The package contains references and hashes, not copied patient records or raw runtime transcripts.
- The package is generated atomically with the closure record.
- Package generation failure rolls back the closure artifact write.

### ReleaseClosureGate

`GET /api/admin/release-closure` returns a derived gate:

```json
{
  "allowed": true,
  "status": "ready_to_close",
  "reasons": [],
  "checks": [
    {
      "name": "required_monitoring_checks_complete",
      "status": "pass",
      "reason": "All Step 14 required checks are present."
    }
  ]
}
```

Allowed gate status values:

- `idle`
- `ready_to_close`
- `blocked`
- `closed`
- `rolled_back_closed`

Gate rules:

1. If no successful Step 13 release exists, status is `idle`.
2. If a closure already exists for the latest release execution, status is `closed` or `rolled_back_closed`.
3. If Step 12, Step 13, Step 14, or Step 15 integrity is not verified, status is `blocked`.
4. If required Step 14 checks are missing, status is `blocked`.
5. If a Step 14 rollback trigger candidate exists, status is `blocked` unless a successful Step 13 rollback exists and the requested closure status is `rolled_back`.
6. If active critical alerts exist, status is `blocked`.
7. If active warning alerts exist, only `accepted_with_observations` may be recorded, and warning alerts must be acknowledged.
8. If a successful rollback exists for the latest release intent, only `rolled_back` may be recorded.

### ReleaseClosureAuditEvent

```json
{
  "event_id": "release_closure_audit_closure_recorded_abc12345",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "release_execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "event_type": "closure_recorded",
  "actor": "release_manager",
  "timestamp": "2026-07-07T10:00:00+08:00",
  "payload_hash": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "previous_event_hash": "sha256:GENESIS",
  "event_hash": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
}
```

Allowed `event_type` values:

- `closure_recorded`
- `evidence_package_generated`
- `closure_read`

Audit rules mirror Step 14:

- canonical JSON payload hash;
- per-release-execution hash chain;
- no secrets, tokens, prompts, hidden reasoning, or patient identifiers;
- chain verification on read;
- failed chain blocks writes for the affected release execution.

## 8. Closure Evaluation Rules

The service derives one closure state for the latest successful release execution.

If no successful Step 13 release exists:

- closure status is `idle`;
- closure gate status is `idle`;
- latest closure is `null`;
- latest package is `null`.

If a successful release exists and no closure exists:

- closure status is `ready_to_close` when every gate passes;
- closure status is `blocked` when any gate fails;
- the service returns exact reasons.

If a successful closure exists for the latest release execution:

- closure status is `closed` for `accepted` or `accepted_with_observations`;
- closure status is `rolled_back_closed` for `rolled_back`;
- new closure writes for the same release execution are blocked unless they are idempotent replays of the same closure payload.

If a successful Step 13 rollback exists for the same intent:

- accepted closure statuses are blocked;
- `rolled_back` closure is allowed when integrity is verified and the rollback execution is successful.

Derived gate checks:

1. `successful_release_exists`
2. `closure_not_already_recorded`
3. `dashboard_integrity_verified`
4. `governance_integrity_verified`
5. `execution_integrity_verified`
6. `monitoring_integrity_verified`
7. `closure_integrity_verified`
8. `required_monitoring_checks_complete`
9. `no_active_critical_monitoring_alerts`
10. `warning_alerts_acknowledged_for_observed_acceptance`
11. `rollback_trigger_absent_or_rollback_succeeded`
12. `requested_status_matches_release_outcome`

## 9. Read Model

`GET /api/admin/release-closure` returns:

```json
{
  "status": "ready_to_close",
  "latest_release": {
    "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
    "release_execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
    "released_at": "2026-07-03T09:00:00+08:00",
    "rollback_execution_id": null,
    "rolled_back_at": null
  },
  "closure_gate": {
    "allowed": true,
    "status": "ready_to_close",
    "reasons": [],
    "checks": []
  },
  "latest_closure": null,
  "latest_evidence_package": null,
  "closures": [],
  "evidence_packages": [],
  "integrity": {
    "status": "verified",
    "warnings": []
  },
  "runtime": {
    "auth": "admin",
    "source": "reports/release_closure",
    "mode": "post_release_closure"
  }
}
```

Rules:

- Read combines Step 11 dashboard, Step 12 governance, Step 13 execution, Step 14 monitoring, and Step 15 closure store state.
- Read does not write files.
- Closure gate derivation happens server-side.
- The frontend action availability is derived from this response.

## 10. Admin API

Add routes:

```text
GET  /api/admin/release-closure
POST /api/admin/release-closure/closures
```

### Record Closure Request

```json
{
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "release_execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "closure_status": "accepted",
  "closed_by": "release_manager",
  "rationale": "Required post-release checks passed and no active critical monitoring alerts remain.",
  "idempotency_key": "close-release-20260707-001"
}
```

Behavior:

- Requires admin token.
- Validates the referenced release execution exists and succeeded.
- Validates the requested closure status against the current closure gate.
- Creates a closure record and evidence package atomically.
- Appends closure audit events.
- Returns the updated closure read model.

## 11. Auth

Extend `_requires_admin_token()`:

```python
if method == "GET" and path == "/api/admin/release-closure":
    return True
if method == "POST" and path == "/api/admin/release-closure/closures":
    return True
```

Rules:

- Existing bearer auth behavior is reused.
- No browser-stored admin secret is added.
- No role-based auth is added in Step 15.
- `closed_by` remains an audit label, not an authentication claim.

## 12. Frontend

Extend the Release task page with a closure section below Step 14 monitoring.

UI states:

- Idle before release execution.
- Blocked with gate reasons.
- Ready to close after monitoring checks satisfy the gate.
- Closed with latest closure and evidence package summary.
- Rolled-back closed after Step 13 rollback is successful and closure is recorded.
- API error while dashboard, governance, execution, and monitoring panels remain visible.

Rules:

- The UI uses backend `status`, `closure_gate`, `latest_closure`, and `latest_evidence_package`.
- The UI must not infer closeability from local form state.
- Closure form requires actor, closure status, rationale, and idempotency key.
- The form disables `accepted` when backend gate requires `rolled_back`.
- Step 15 does not add a one-click rollback button. Operators still use the Step 13 rollback form.
- Step 15 does not add alert acknowledgement controls beyond Step 14 controls.

## 13. Error Handling

Backend:

- Invalid request payload: `422`.
- Referenced release missing or not successful: `409`.
- Closure gate blocked: `409`.
- Closure integrity failure: `409`.
- Idempotency payload mismatch: `409`.
- Filesystem write failure: `500`.
- Auth failure: existing `401` and `403`.

Frontend:

- Render closure API errors inside the closure panel.
- Preserve release dashboard, governance, execution, and monitoring panels when closure fails.
- Show closure action errors without resetting form inputs.

## 14. Safety And Security Boundaries

Step 15 writes only:

- `reports/release_closure/closures/*.json`
- `reports/release_closure/packages/*.json`
- `reports/release_closure/audit/*.jsonl`
- `reports/release_closure/README.md`

Step 15 must not write:

- `reports/release_monitoring/`
- `reports/release_execution/`
- `reports/release_governance/`
- `reports/harness/`
- `reports/release_safety/`
- `reports/literature/`
- `config/`
- prompt, rubric, route, template, RAG, model, or tool files
- patient registry or session data
- `CRC-client/`

Closure payloads and packages must not contain:

- API keys;
- bearer tokens;
- deployment credentials;
- model prompts;
- hidden reasoning;
- raw patient identifiers;
- patient records;
- doctor note text;
- session transcripts.

## 15. Testing Strategy

Backend contract tests:

- closure record validation;
- evidence package validation;
- closure gate serialization;
- audit hash validation;
- forbidden payload key rejection.

Store tests:

- empty closure root returns verified empty state;
- read does not write;
- closure write creates one closure, one package, and two audit events;
- idempotent closure replay returns existing closure;
- idempotency payload mismatch fails;
- audit tampering blocks writes;
- closure write rolls back if package write fails;
- path traversal, symlink, and reserved filename protections.

Service tests:

- closure is idle before successful release;
- closure is blocked when Step 14 required checks are missing;
- closure is blocked when active critical alerts exist;
- closure is blocked when rollback trigger candidate exists before rollback execution;
- accepted closure is allowed after required checks pass and no active alerts exist;
- accepted with observations is allowed when warnings are acknowledged;
- rolled-back closure is allowed after successful Step 13 rollback;
- duplicate non-idempotent closure for the same release is blocked;
- reads do not write.

API/auth tests:

- all closure routes require admin token;
- user token receives `403` when admin token differs;
- missing or invalid token receives `401`;
- record closure maps validation, gate, idempotency, and integrity errors.

Frontend tests:

- API client uses correct endpoints, headers, and JSON bodies;
- Release page loads closure state with release task;
- idle state renders before release;
- blocked gate reasons render;
- ready-to-close form submits a closure request;
- closed state renders closure and package summary;
- closure API errors do not break dashboard, governance, execution, or monitoring panels.

Non-mutation tests:

- record closure does not mutate Step 14 monitoring files;
- record closure does not mutate Step 13 execution files;
- record closure does not mutate Step 12 governance files;
- record closure does not mutate Step 11/P0/P1/Step10 reports;
- record closure does not mutate config, prompts, RAG, tool manifests, patient/doctor paths, or `CRC-client/`.

Regression tests:

- Step 14 monitoring backend and frontend tests pass.
- Step 13 execution backend and frontend tests pass.
- Step 12 governance backend and frontend tests pass.
- Step 11 admin release dashboard tests pass.
- Step 10 literature harness tests pass.
- P1 doctor review tests pass.
- P0 safety loop tests pass.

## 16. Acceptance Criteria

Step 15 is complete when:

1. Closure contracts exist and are tested.
2. File-backed closure store exists under `reports/release_closure/`.
3. Admin-only closure APIs read state and record closure.
4. Closure read model derives readiness from Step 11 through Step 14 state.
5. Closure writes create a closure record, evidence package, and audit events atomically.
6. Accepted closure is blocked while required checks are missing, active critical alerts exist, or rollback trigger candidate exists.
7. Rolled-back closure is allowed only after Step 13 rollback succeeds.
8. Agent Admin Release page shows closure gate, closure form, latest closure, and evidence package summary.
9. Tests prove Step 15 writes only under `reports/release_closure/`.
10. Step 14, Step 13, Step 12, Step 11, Step 10, P1, and P0 regressions pass.

## 17. Future Work

Later specs may cover:

- rendered PDF or Word release evidence packages;
- external archive delivery;
- scheduled closure reminders;
- stronger operator identity and role-based authorization;
- external alert delivery;
- real production feature flag provider integration;
- runtime consumption of release flag state;
- evidence promotion governance;
- research cohort feasibility and ethics gates.

These remain outside Step 15.

## 18. Spec Self-Review

Marker scan: no unresolved work markers remain.

Internal consistency: Step 15 reads Step 11 through Step 14 state, writes only `reports/release_closure/`, creates closure records and evidence packages atomically, and never executes release or rollback.

Scope check: this is one coherent subsystem: post-release closure and local evidence packaging after monitoring. External archive delivery, document rendering, scheduled reminders, runtime flag consumption, evidence promotion, and research cohort workflows remain separate future work.

Ambiguity check: storage paths, allowed writes, forbidden writes, contracts, gate rules, API behavior, frontend behavior, and acceptance criteria are explicit.
