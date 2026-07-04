# Post-Release Monitoring And Rollback Trigger Design

> Version: 2026-07-03
> Scope: P2 Step 14
> Source context: Step 13 `docs/superpowers/specs/2026-07-03-controlled-release-execution-design.md`
> Goal: Add audited post-release monitoring and rollback trigger recommendations after controlled local release execution, without executing rollback automatically or connecting local feature flag state to clinical runtime behavior.

## 1. Background

Step 11 added a read-only Agent Admin Release Dashboard.

Step 12 added audit-only release governance:

- `ReleaseIntent`, `ReleaseApproval`, `ReleaseRollbackPlan`, and `ReleaseAuditEvent`.
- `reports/release_governance/` append-only governance records.
- Admin-protected governance APIs and UI forms.

Step 13 added controlled local release execution:

- `ReleaseExecutionRequest`, `ReleaseExecutionResult`, `FeatureFlagState`, and `ReleaseExecutionAuditEvent`.
- `reports/release_execution/` request, result, local flag, history, and audit artifacts.
- Admin-protected release and rollback APIs.
- Agent Admin execution UI.

The remaining gap is post-release observation. Once a local release execution has succeeded, operators need a structured way to record post-release checks, see derived alerts, and know when the Step 13 rollback path should be considered. Step 14 adds that monitoring layer. It remains local, explicit, and auditable.

## 2. Current Project Context

Relevant current files:

- `src/contracts/release_execution.py`: Step 13 execution contracts and audit hashing.
- `backend/api/services/release_execution_store.py`: file-backed Step 13 execution store.
- `src/services/release_execution.py`: Step 13 preflight and execution orchestration.
- `backend/api/routes/admin.py`: release dashboard, governance, and execution routes.
- `backend/app.py`: bearer auth and admin route protection.
- `frontend/src/app/api/types.ts`: release dashboard, governance, and execution types.
- `frontend/src/app/api/client.ts`: admin release API methods.
- `frontend/src/features/agent-admin/agent-admin-view.tsx`: release resources and action handlers.
- `frontend/src/features/agent-admin/agent-admin-pages.tsx`: release dashboard, governance, and execution panels.
- `reports/release_execution/README.md`: Step 13 execution artifact boundary.
- `tests/backend/test_release_execution_*.py`: Step 13 backend coverage.

Important observed boundary:

- Step 13 local `FeatureFlagState` is not consumed by patient or doctor runtime paths.
- Step 13 writes are limited to `reports/release_execution/`.
- There is no external deployment, alerting, scheduler, or production telemetry integration.
- Monitoring must not invent live clinical health signals from patient sessions.

## 3. Design Options

### Option A: External Observability Integration

Step 14 could push metrics to Prometheus, Slack, email, PagerDuty, or a deployment provider.

Rejected. The project has no provider contracts, credentials, operator identity model, delivery guarantees, rate limits, or production environment ownership. Adding that now would mix release governance with external operations infrastructure.

### Option B: Runtime Clinical Telemetry Monitor

Step 14 could read patient and doctor sessions after a release to infer safety regressions.

Rejected. That would pull patient-level runtime data into the release system and would blur clinical workflow monitoring with release evidence. This slice must not read patient identifiers, raw session transcripts, or doctor drafts.

### Option C: Local File-Backed Monitoring Ledger And Derived Alerts

Step 14 writes explicit monitoring check records and acknowledgements under `reports/release_monitoring/`. A service reads Step 11 dashboard state, Step 12 governance state, Step 13 execution state, and Step 14 check records to derive alerts and rollback trigger candidates.

Recommended. It gives operators a concrete post-release monitoring surface while keeping writes local, auditable, and separate from execution. Alerts are recommendations, not actions.

## 4. Scope

Step 14 includes:

1. Contracts for `ReleaseMonitoringCheck`, `ReleaseMonitoringAlert`, `ReleaseRollbackTriggerCandidate`, `ReleaseMonitoringAcknowledgement`, and `ReleaseMonitoringAuditEvent`.
2. A file-backed monitoring store under `reports/release_monitoring/`.
3. A monitoring service that reads Step 13 execution state and derives monitoring status.
4. Admin-protected APIs to read monitoring state, record a monitoring check, and acknowledge a derived alert.
5. Agent Admin Release page monitoring panel below Step 13 execution.
6. Deterministic alert derivation for missing required checks, failed checks, execution integrity failures, dashboard drift, and rollback-ready conditions.
7. Tests proving Step 14 writes only under `reports/release_monitoring/`.
8. Regression tests for Step 13 execution, Step 12 governance, Step 11 dashboard, Step 10 literature, P1 doctor review, and P0 safety loop.

Step 14 excludes:

- Automatic rollback execution.
- Automatic release execution.
- Polling loops, schedulers, cron jobs, or background workers.
- Running harness scripts from an admin API.
- Network calls.
- External alert providers.
- Deployment providers.
- Production credentials.
- Runtime consumption of Step 13 local feature flag state.
- Patient-level telemetry, session transcripts, raw patient identifiers, or doctor note content.
- Promoting literature evidence.
- Editing safety policy, prompts, rubrics, routes, templates, RAG indexes, model weights, tool manifests, or `CRC-client/`.

## 5. Architecture

```text
Step 11 release dashboard
  + Step 12 release governance read model
  + Step 13 release execution read model
    -> Step 14 release monitoring service
      -> monitoring check records
      -> derived alert rules
      -> rollback trigger candidate
      -> acknowledgement records
      -> monitoring audit events
        -> reports/release_monitoring/
          -> GET /api/admin/release-monitoring
          -> POST /api/admin/release-monitoring/checks
          -> POST /api/admin/release-monitoring/alerts/{alert_id}/acknowledge
            -> Agent Admin Release page monitoring panel
```

The monitoring service owns alert derivation. The frontend renders backend results and submits explicit monitoring checks or acknowledgements. The frontend must not derive rollback trigger state on its own.

## 6. Storage Layout

Use a new dedicated directory:

```text
reports/release_monitoring/
  README.md
  checks/
    release_monitor_check_release_exec_abc12345_p0_harness_replay_291a1a2b.json
  acknowledgements/
    release_monitor_ack_release_monitor_alert_abc12345_57aa8e34.json
  audit/
    release_monitoring_20260703.jsonl
```

Rules:

- Reads must not create files.
- Writes may create only files under `reports/release_monitoring/`.
- Check and acknowledgement artifacts are write-once.
- Audit JSONL is append-only.
- Store integrity failure prevents new writes.
- Store path checks must reject symlinks, path traversal, reserved Windows device names, and resolved paths outside the monitoring root.
- Monitoring artifacts must reference Step 13 execution IDs and Step 12 intent IDs, not patient identifiers.

## 7. Contracts

### ReleaseMonitoringCheck

```json
{
  "check_id": "release_monitor_check_release_exec_abc12345_p0_harness_replay_291a1a2b",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "check_type": "p0_harness_replay",
  "status": "pass",
  "observed_by": "release_manager",
  "observed_at": "2026-07-03T11:00:00+08:00",
  "summary": "P0 harness replay passed after controlled release execution.",
  "evidence_refs": [
    "reports/harness/harness_20260629_001.json"
  ],
  "metrics": {
    "passed": 5,
    "failed": 0,
    "hard_fail_count": 0
  },
  "idempotency_key": "release-exec-abc12345-p0-harness-20260703"
}
```

Allowed `check_type` values:

- `execution_integrity`
- `governance_drift`
- `p0_harness_replay`
- `agent_admin_smoke`
- `doctor_review_smoke`
- `literature_isolation`
- `manual_operator_note`

Allowed `status` values:

- `pass`
- `warning`
- `fail`

Rules:

- `intent_id`, `execution_id`, `check_type`, `status`, `observed_by`, `observed_at`, `summary`, and `idempotency_key` are required.
- `evidence_refs` must be repo-relative paths or command labels, not absolute paths.
- `metrics` must be JSON-safe and must not contain secrets, prompts, hidden reasoning, or patient identifiers.
- Reusing the same `idempotency_key` with an identical payload returns the existing check.
- Reusing the same `idempotency_key` with a different payload returns conflict.

### ReleaseMonitoringAlert

Alerts are derived by the monitoring service and are not written directly by clients.

```json
{
  "alert_id": "release_monitor_alert_release_exec_abc12345_p0_failed_2c4f8b11",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "severity": "critical",
  "category": "post_release_check_failed",
  "status": "active",
  "message": "P0 harness replay reported a hard fail after release execution.",
  "source_check_ids": [
    "release_monitor_check_release_exec_abc12345_p0_harness_replay_291a1a2b"
  ],
  "recommended_action": "execute_step13_rollback",
  "created_at": "2026-07-03T11:00:00+08:00"
}
```

Allowed `severity` values:

- `info`
- `warning`
- `critical`

Allowed `category` values:

- `missing_required_check`
- `post_release_check_failed`
- `execution_integrity_failed`
- `governance_drift`
- `feature_flag_state_mismatch`
- `rollback_ready`

Allowed `recommended_action` values:

- `observe`
- `investigate`
- `prepare_rollback`
- `execute_step13_rollback`

Rules:

- Alerts are deterministic from execution state and check records.
- Acknowledged alerts remain visible with acknowledgement metadata.
- A critical alert never executes rollback. It only recommends using the Step 13 rollback path.

### ReleaseRollbackTriggerCandidate

```json
{
  "candidate_id": "release_rollback_trigger_release_exec_abc12345_8ad31f20",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "source_alert_ids": [
    "release_monitor_alert_release_exec_abc12345_p0_failed_2c4f8b11"
  ],
  "recommended_action": "execute_step13_rollback",
  "rollback_plan_id": "rollback_plan_release_intent_release_safety_20260629_001_1b00f364",
  "rollback_target": "agent_policy_20260624_0",
  "reason": "A critical post-release check failed while the local feature flag remains enabled.",
  "created_at": "2026-07-03T11:00:00+08:00"
}
```

Rules:

- A candidate is derived when at least one active critical alert recommends `execute_step13_rollback`.
- A candidate requires an accepted Step 12 rollback plan and a successful Step 13 release execution.
- A candidate is hidden when Step 13 rollback has already succeeded for the same intent.
- A candidate never invokes `execute_rollback`.

### ReleaseMonitoringAcknowledgement

```json
{
  "acknowledgement_id": "release_monitor_ack_release_monitor_alert_abc12345_57aa8e34",
  "alert_id": "release_monitor_alert_release_exec_abc12345_p0_failed_2c4f8b11",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "acknowledged_by": "release_manager",
  "acknowledged_at": "2026-07-03T11:05:00+08:00",
  "disposition": "investigating",
  "reason": "Release manager is checking the harness artifact before rollback execution."
}
```

Allowed `disposition` values:

- `investigating`
- `accepted_risk`
- `rollback_started_elsewhere`
- `false_positive`

Rules:

- Acknowledgement records are append-only.
- Latest acknowledgement per `alert_id` controls derived alert status.
- Acknowledgement does not suppress rollback trigger derivation when a critical alert remains actionable, unless the disposition is `false_positive`.

### ReleaseMonitoringAuditEvent

```json
{
  "event_id": "release_monitoring_audit_check_recorded_abc12345",
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "event_type": "check_recorded",
  "actor": "release_manager",
  "timestamp": "2026-07-03T11:00:00+08:00",
  "payload_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "previous_event_hash": "sha256:GENESIS",
  "event_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
}
```

Allowed `event_type` values:

- `check_recorded`
- `alert_acknowledged`
- `monitoring_read`

Audit rules mirror Step 13:

- canonical JSON payload hash;
- per-execution hash chain;
- no secrets, tokens, prompts, hidden reasoning, or patient identifiers;
- chain verification on read;
- failed chain blocks writes for the affected execution.

## 8. Monitoring Evaluation Rules

The service derives one monitoring state per latest successful release execution.

If no successful Step 13 release exists:

- monitoring status is `idle`;
- required checks are empty;
- alerts are empty;
- rollback trigger candidate is `null`.

If a successful Step 13 rollback exists for the same intent:

- monitoring status is `rolled_back`;
- new rollback trigger candidate is `null`;
- historical checks and alerts remain visible.

If a successful release exists and no successful rollback exists:

- monitoring status is `monitoring`;
- required checks are:
  - `execution_integrity`;
  - `governance_drift`;
  - `p0_harness_replay`;
  - `agent_admin_smoke`;
  - `doctor_review_smoke`;
  - `literature_isolation`.

Derived alert rules:

1. Execution store integrity failure creates a critical `execution_integrity_failed` alert.
2. Governance integrity failure creates a critical `governance_drift` alert.
3. Dashboard release report, rollback target, version chain, hard fail count, or literature status drift creates a warning or critical `governance_drift` alert.
4. Missing required checks create warning `missing_required_check` alerts.
5. A failed `p0_harness_replay` check creates a critical `post_release_check_failed` alert with recommended action `execute_step13_rollback`.
6. A failed `doctor_review_smoke` check creates a critical `post_release_check_failed` alert with recommended action `prepare_rollback`.
7. A failed `literature_isolation` check creates a critical `post_release_check_failed` alert with recommended action `execute_step13_rollback`.
8. A current local feature flag enabled for an execution without a matching successful release result creates a critical `feature_flag_state_mismatch` alert.
9. A current local feature flag disabled after successful rollback clears rollback trigger candidate derivation.

## 9. Read Model

`GET /api/admin/release-monitoring` returns:

```json
{
  "status": "monitoring",
  "latest_release": {
    "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
    "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
    "released_at": "2026-07-03T09:00:00+08:00",
    "flag_enabled": true,
    "rollback_plan_id": "rollback_plan_release_intent_release_safety_20260629_001_1b00f364"
  },
  "required_checks": [
    {
      "check_type": "p0_harness_replay",
      "status": "missing",
      "latest_check_id": null,
      "reason": "Record a post-release P0 harness replay check."
    }
  ],
  "checks": [],
  "alerts": [],
  "rollback_trigger_candidate": null,
  "acknowledgements": [],
  "integrity": {
    "status": "verified",
    "warnings": []
  },
  "runtime": {
    "auth": "admin",
    "source": "reports/release_monitoring",
    "mode": "post_release_monitoring"
  }
}
```

Rules:

- Read combines Step 13 execution state, Step 12 governance state, Step 11 dashboard state, and Step 14 monitoring store state.
- Read does not write files.
- Alert and rollback trigger derivation happen server-side.
- The frontend action availability is derived from this response.

## 10. Admin API

Add routes:

```text
GET  /api/admin/release-monitoring
POST /api/admin/release-monitoring/checks
POST /api/admin/release-monitoring/alerts/{alert_id}/acknowledge
```

### Record Check Request

```json
{
  "intent_id": "release_intent_release_safety_20260629_001_6da729a0",
  "execution_id": "release_exec_release_intent_release_safety_20260629_001_6da729a0_release_291a1a2b",
  "check_type": "p0_harness_replay",
  "status": "pass",
  "observed_by": "release_manager",
  "summary": "P0 harness replay passed after release execution.",
  "evidence_refs": [
    "reports/harness/harness_20260629_001.json"
  ],
  "metrics": {
    "passed": 5,
    "failed": 0,
    "hard_fail_count": 0
  },
  "idempotency_key": "release-exec-abc12345-p0-harness-20260703"
}
```

Behavior:

- Requires admin token.
- Validates the referenced release execution exists and succeeded.
- Rejects checks for rolled-back executions unless `check_type` is `manual_operator_note`.
- Writes a check artifact.
- Appends monitoring audit event.
- Returns updated monitoring read model.

### Acknowledge Alert Request

```json
{
  "acknowledged_by": "release_manager",
  "disposition": "investigating",
  "reason": "Checking the referenced harness artifact before rollback execution."
}
```

Behavior:

- Requires admin token.
- Validates the alert currently exists in the derived read model.
- Writes an acknowledgement artifact.
- Appends monitoring audit event.
- Returns updated monitoring read model.

## 11. Auth

Extend `_requires_admin_token()`:

```python
if method == "GET" and path == "/api/admin/release-monitoring":
    return True
if method == "POST" and path.startswith("/api/admin/release-monitoring/"):
    return True
```

Rules:

- Existing bearer auth behavior is reused.
- No browser-stored admin secret is added.
- No role-based auth is added in Step 14.
- `observed_by` and `acknowledged_by` remain audit labels, not authentication claims.

## 12. Frontend

Extend the Release task page with a monitoring section below Step 13 execution.

UI states:

- Monitoring idle before release execution.
- Monitoring active after release execution.
- Required checks with missing, pass, warning, and fail states.
- Active alerts with severity and recommended action.
- Rollback trigger candidate with explicit "use Step 13 rollback execution" copy.
- Acknowledged alerts.
- Integrity failed, with writes disabled.
- API error while dashboard, governance, and execution panels remain visible.

Rules:

- The UI uses backend `status`, `required_checks`, `alerts`, and `rollback_trigger_candidate`.
- The UI must not infer rollback trigger state from form state.
- Check form requires actor, check type, status, summary, and idempotency key.
- Acknowledgement form requires actor, disposition, and reason.
- Step 14 does not add a one-click rollback button. Operators still use the Step 13 rollback form.

## 13. Error Handling

Backend:

- Invalid request payload: `422`.
- Referenced execution missing or not successful: `409`.
- Monitoring integrity failure: `409`.
- Idempotency payload mismatch: `409`.
- Unknown derived alert id: `404`.
- Filesystem write failure: `500`.
- Auth failure: existing `401` and `403`.

Frontend:

- Render monitoring API errors inside the monitoring panel.
- Preserve release dashboard, governance, and execution panels when monitoring fails.
- Show check and acknowledgement action errors without resetting form inputs.

## 14. Safety And Security Boundaries

Step 14 writes only:

- `reports/release_monitoring/checks/*.json`
- `reports/release_monitoring/acknowledgements/*.json`
- `reports/release_monitoring/audit/*.jsonl`
- `reports/release_monitoring/README.md`

Step 14 must not write:

- `reports/release_execution/`
- `reports/release_governance/`
- `reports/harness/`
- `reports/release_safety/`
- `reports/literature/`
- `config/`
- prompt, rubric, route, template, RAG, model, or tool files
- patient registry or session data
- `CRC-client/`

Monitoring payloads must not contain:

- API keys;
- bearer tokens;
- deployment credentials;
- model prompts;
- hidden reasoning;
- raw patient identifiers;
- patient records;
- doctor note text.

## 15. Testing Strategy

Backend contract tests:

- check validation;
- acknowledgement validation;
- audit hash validation;
- alert and rollback candidate serialization;
- forbidden payload key rejection.

Store tests:

- empty monitoring root returns verified empty state;
- read does not write;
- check write creates one check and one audit event;
- acknowledgement write creates one acknowledgement and one audit event;
- idempotent check replay returns existing check;
- idempotency payload mismatch fails;
- audit tampering blocks writes;
- path traversal, symlink, and reserved filename protections.

Service tests:

- monitoring is idle before successful release;
- monitoring is active after successful release;
- rollback success changes status to `rolled_back`;
- missing required checks produce warning alerts;
- failed P0 harness check produces critical rollback trigger candidate;
- failed literature isolation check produces critical rollback trigger candidate;
- acknowledged false positive alert does not produce rollback trigger candidate;
- execution integrity failure produces critical alert;
- dashboard drift produces drift alert;
- read operations do not write.

API/auth tests:

- all monitoring routes require admin token;
- user token receives `403` when admin token differs;
- missing or invalid token receives `401`;
- record check maps validation, preflight, idempotency, and integrity errors;
- acknowledge alert maps unknown alert id to `404`.

Frontend tests:

- API client uses correct endpoints, headers, and JSON bodies;
- Release page loads monitoring state with release task;
- idle state renders before release;
- missing required checks render;
- failed check alert renders with rollback trigger candidate;
- check form submits record request;
- acknowledgement form submits acknowledgement request;
- monitoring API errors do not break dashboard, governance, or execution panels.

Non-mutation tests:

- record check and acknowledge alert do not mutate Step 13 execution files;
- record check and acknowledge alert do not mutate Step 12 governance files;
- record check and acknowledge alert do not mutate Step 11/P0/P1/Step10 reports;
- record check and acknowledge alert do not mutate config, prompts, RAG, tool manifests, patient/doctor paths, or `CRC-client/`.

Regression tests:

- Step 13 execution backend and frontend tests pass.
- Step 12 governance backend and frontend tests pass.
- Step 11 admin release dashboard tests pass.
- Step 10 literature harness tests pass.
- P1 doctor review tests pass.
- P0 safety loop tests pass.

## 16. Acceptance Criteria

Step 14 is complete when:

1. Monitoring contracts exist and are tested.
2. File-backed monitoring store exists under `reports/release_monitoring/`.
3. Admin-only monitoring APIs read state, record checks, and acknowledge alerts.
4. Monitoring read model derives required checks, active alerts, and rollback trigger candidate.
5. Critical failed post-release checks recommend Step 13 rollback without executing it.
6. Monitoring check and acknowledgement writes are idempotent and audit chained.
7. Agent Admin Release page shows monitoring status, required checks, alerts, acknowledgements, and rollback trigger recommendation.
8. Tests prove Step 14 writes only under `reports/release_monitoring/`.
9. Step 13, Step 12, Step 11, Step 10, P1, and P0 regressions pass.

## 17. Future Work

Later specs may cover:

- scheduled monitoring jobs;
- external alert delivery;
- real production feature flag provider integration;
- runtime consumption of release flag state;
- stronger operator identity and role-based authorization;
- evidence promotion governance;
- research cohort feasibility and ethics gates.

These remain outside Step 14.

## 18. Spec Self-Review

Marker scan: no unresolved work markers remain.

Internal consistency: Step 14 reads Step 11 through Step 13 state, writes only `reports/release_monitoring/`, derives alerts server-side, and never executes rollback.

Scope check: this is one coherent subsystem: post-release monitoring and rollback trigger recommendation after controlled local execution. Scheduling, external alerting, production flag providers, runtime clinical telemetry, and evidence promotion remain separate future work.

Ambiguity check: storage paths, allowed writes, forbidden writes, contracts, alert rules, API behavior, frontend behavior, and acceptance criteria are explicit.
