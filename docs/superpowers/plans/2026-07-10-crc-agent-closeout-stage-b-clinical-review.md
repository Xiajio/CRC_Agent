# CRC Agent Closeout Stage B Clinical Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the P1 clinical-review gate by making one immutable, sanitized DoctorDraftVersion the source for both report and review views, completing all six authorized review actions, and proving that review does not mutate patient facts or safety configuration.

**Architecture:** Stage B consumes Stage A's approved integrity, AuthContext, sanitizer, structured care-card, runtime-version, and StageGate contracts. A dedicated restricted SQLite DoctorReviewStore persists immutable draft versions, exact section provenance, action traces, a SHA-256 audit chain, and server-only HMAC integrity records in one transaction. The graph captures only the filtered visible report output; both frontend surfaces fetch the same ClinicalVersionProjection, while the backend retains the internal ClinicalVersionRef and derives allowed actions and authorization.

**Tech Stack:** Python 3.10, dataclasses, SQLite, HMAC, FastAPI 0.135, Pydantic 2.12, React 18, TypeScript 5.6, Vitest 2.1, Playwright, pytest.

## Global Constraints

- Start from the merged Stage A commit only after its post-merge StageGateReport and StageGateApprovalAttestation validate against the latest approved requirement manifest. Before Task 1, build the Stage B `StagePlanApprovalSubject` from this plan's tracked blob/source commit and validate its policy-required `PlanApprovalAttestation` with Stage A's shared CLI.
- Consume `VersionRef`, internal `ClinicalVersionRef`/`ClinicalIntegrityRecord`, public `ClinicalVersionProjection`, and audit helpers from `src/contracts/integrity.py`; consume `src/contracts/auth_context.py`, `src/services/write_boundary_sanitizer.py`, `src/services/runtime_version_registry.py`, and Stage A's closeout runner without redefining them.
- DoctorDraftVersion is patient-scoped. Public APIs expose `ClinicalVersionProjection`, never the internal ClinicalVersionRef's restricted handle, integrity MAC, key version, or a patient-linked hash.
- The report view and Review Cockpit must resolve the same persisted draft ID/version and section versions.
- Request bodies may carry a display label but never grant reviewer roles. Actor principal, credential, roles, scopes, and correlation ID come only from AuthContext.
- Only `edit` may create a superseding draft. The other five actions append trace/audit records and do not change drafts or clinical facts.
- Neither patient nor doctor streams, drafts, traces, logs, nor UI may contain hidden chain-of-thought, credentials, prompt secrets, or unsanitized identifiers.
- Missing evidence provenance remains an explicit unverified warning; never fabricate a runtime evidence-index label.
- Preserve unrelated user files, do not modify `CRC-client/`, and stage only files named by the active task.

## Pre-Implementation Plan Authorization

- [ ] Resolve the exact tracked plan blob, publish its subject, collect the two policy-required external approvals, and exact-stage the resulting evidence:

```powershell
$planPath = "docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-b-clinical-review.md"
$planSourceCommit = (git log -1 --format=%H -- $planPath).Trim()
git cat-file -e "${planSourceCommit}:$planPath"
$trackedPlanBlob = (git rev-parse "${planSourceCommit}:$planPath").Trim()
$workingPlanBlob = (git hash-object -- $planPath).Trim()
if ($trackedPlanBlob -ne $workingPlanBlob) { throw "Stage B plan blob is not tracked" }
$planSubjectPath = "reports/closeout/plan_subjects/stage_b_plan_20260710_001.json"
$planAttestationPath = "reports/closeout/attestations/stage_b_plan_approval_20260710_001.json"
$planEvidencePaths = "output/closeout/stage-b-plan-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py build-plan-subject --plan $planPath --source-commit $planSourceCommit --author-credential-env LANGG_CLOSEOUT_PUBLISHER_TOKEN --output $planSubjectPath
$planSubjectHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field sha256).Trim()
$planSubjectVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-b-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$planSubjectHash-stage-b-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_plan --subject-path $planSubjectPath --output $planAttestationPath --path-list-output $planEvidencePaths --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_plan --subject-path $planSubjectPath --attestation-path $planAttestationPath
git add --pathspec-from-file=$planEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $planEvidencePaths --subject-path $planSubjectPath --attestation-path $planAttestationPath
git commit -m "evidence(closeout): approve Stage B plan"
```

- [ ] **Hard stop:** do not start Task 1 until the exact staged-set verifier and commit succeed and the subject plan ref/hash/version, approval-policy ref, author exclusion, quorum, ledger head, and latest-plan selection all validate.

## Source Design

- `docs/superpowers/specs/2026-07-10-crc-agent-closeout-program-design.md`, especially Sections 5, 7, 10-14, 16-18.
- `docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-a-safety-persistence.md` for inherited contracts and gate behavior.
- Existing doctor-report, DoctorReview, patient-registry, graph-stream, and frontend tests are migration inputs; static placeholder draft behavior and doctor thinking disclosure are explicitly retired.

## File Structure

- `src/contracts/doctor_draft.py`: immutable draft, section, provenance, and action-state contracts.
- `src/contracts/doctor_action_trace.py`: exact-target commands/results and sanitized trace contract.
- `src/contracts/clinical_assertion.py`: multi-source assertion provenance with safe opaque refs.
- `backend/api/services/doctor_review_store.py`: restricted SQLite drafts, traces, audit events, idempotency, and ledger heads.
- `backend/api/services/doctor_review_service.py`: authorization, target resolution, sanitization, and atomic commands.
- `backend/api/routes/doctor_review.py`: same-source read APIs and secured action write API.
- `frontend/src/features/doctor/doctor-report-draft-view.tsx`: persisted report source.
- `frontend/src/features/doctor/doctor-review-cockpit.tsx`: backend-allowed six-action UI.
- `config/closeout_stage_suites.yaml`: inherited runner registration for Stage B.

---

### Task 1: DoctorDraftVersion And Action-State Domain Contracts

**Files:**

- Create: `src/contracts/doctor_draft.py`
- Create: `src/contracts/doctor_action_trace.py`
- Create: `tests/backend/test_doctor_draft_contract.py`
- Create: `tests/backend/test_doctor_action_state_contract.py`

**Core contract:**

```python
ALL_DOCTOR_ACTIONS = (
    "accept",
    "edit",
    "reject",
    "escalate",
    "request_evidence",
    "mark_unsafe",
)
ACTION_STATE_CONTRACT_VERSION = "doctor_action_state_v1"

def allowed_actions_for(
    *,
    target_kind: str,
    is_latest_draft: bool,
    section_exists: bool,
) -> tuple[str, ...]: ...
```

`DoctorDraftVersion` contains opaque draft ID, monotonically increasing version, patient/session binding, graph-run ID, visible source-message refs, structured sections, section versions, per-section provenance, `runtime_snapshot_ref: VersionRef`, exact `runtime_version_bindings: tuple[RuntimeVersionBinding, ...]`, exact `sanitizer_ref: VersionRef`, schema version, supersedes ref, creation actor, and an internal `ClinicalIntegrityRecord` association.

- [ ] Add failing tests for stable opaque IDs, monotonic versions, supersedes linkage, unique section IDs, section-version increments, immutable runtime/sanitizer refs, and public serialization through `ClinicalVersionProjection` with rejection of restricted handles/MAC/key versions.
- [ ] Add failing action-state tests: an exact latest existing section exposes all six actions; stale/superseded/missing sections expose none; unknown target kinds are rejected; the order is deterministic.
- [ ] Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_draft_contract.py tests/backend/test_doctor_action_state_contract.py -q -p no:cacheprovider
```

Expected: collection fails because the new contracts do not exist.

- [ ] Implement frozen dataclasses and pure validators. Use `VersionRef`, `ClinicalVersionRef`, `ClinicalIntegrityRecord`, and canonical hashing from Stage A.
- [ ] Model provenance for assertion/assessment/record/care-card/citation/evidence refs plus confidence, review status, and missing-provenance warnings. Do not store free-form hidden reasoning.
- [ ] Re-run the tests; require all pass.
- [ ] Commit only Task 1 paths:

```powershell
git add src/contracts/doctor_draft.py src/contracts/doctor_action_trace.py tests/backend/test_doctor_draft_contract.py tests/backend/test_doctor_action_state_contract.py
git commit -m "feat: add versioned doctor review contracts"
```

---

### Task 2: Multi-Source ClinicalAssertion Projection

**Files:**

- Modify: `src/contracts/clinical_assertion.py`
- Modify: `src/services/clinical_assertion_projection.py`
- Modify: `backend/api/schemas/patient_registry.py`
- Modify: `tests/backend/test_clinical_assertion_projection.py`
- Modify: `tests/backend/test_patient_registry_record_assertions.py`

**Projection entry points:**

```python
def project_clinical_assertions_from_record(record: Mapping[str, object]) -> list[ClinicalAssertion]: ...
def project_clinical_assertions_from_care_card(card: PatientCareCardProjection) -> list[ClinicalAssertion]: ...
def project_clinical_assertions_from_reviewed_evidence(entry: ReviewedEvidenceProjection) -> list[ClinicalAssertion]: ...
def project_clinical_assertions_from_draft(draft: DoctorDraftVersion) -> list[ClinicalAssertion]: ...
def project_clinical_assertions_from_records(records: Iterable[Mapping[str, object]]) -> list[ClinicalAssertion]: ...
```

- [ ] Add failing table-driven tests for six approved sources: CRC triage/structured self-report, uploaded structured report summary, clinician-note `structured_facts`, Stage A structured care card, reviewed evidence, and model draft.
- [ ] Assert arbitrary clinician free text is never parsed; evidence candidates cannot appear as reviewed evidence; draft assertions are always `model_generated_unverified`.
- [ ] Add regression tests proving numeric patient ID never contributes to an assertion ID or public hash. Patient-scoped assertions use random opaque IDs, internal ClinicalVersionRef binding, and server-only integrity; the authorized API exposes ClinicalVersionProjection without a content hash. A separately de-identified cross-domain artifact receives a new unrelated ID/VersionRef.
- [ ] Add compatibility tests: legacy records without a valid ref yield an empty or partial projection with a warning, not a crash or fabricated provenance.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_patient_registry_record_assertions.py -q -p no:cacheprovider`; observe failures from the current CRC-only projection and low-entropy ID hashing.
- [ ] Implement explicit source adapters and deterministic projection ordering. Sanitize display text before the server-only integrity record is calculated; the patient-scoped public `ClinicalVersionProjection` contains no content hash.
- [ ] Re-run and require all pass.
- [ ] Commit only Task 2 paths:

```powershell
git add src/contracts/clinical_assertion.py src/services/clinical_assertion_projection.py backend/api/schemas/patient_registry.py tests/backend/test_clinical_assertion_projection.py tests/backend/test_patient_registry_record_assertions.py
git commit -m "feat: expand clinical assertion provenance sources"
```

---

### Task 3: Restricted DoctorReviewStore And Service

**Files:**

- Create: `backend/api/services/doctor_review_store.py`
- Create: `backend/api/services/doctor_review_service.py`
- Create: `src/contracts/clinical_retention.py`
- Create: `config/clinical_retention.yaml`
- Create: `tests/backend/test_doctor_review_store.py`
- Create: `tests/backend/test_doctor_draft_service.py`
- Create: `tests/backend/test_doctor_review_retention.py`
- Create: `tests/backend/test_doctor_review_retention_api.py`
- Modify: `backend/api/services/settings.py`
- Modify: `backend/app.py`
- Modify: `backend/api/services/patient_registry_service.py`
- Modify: `backend/api/routes/patient_registry.py`
- Modify: `backend/api/schemas/patient_registry.py`

**Tables:** `doctor_draft_versions`, `doctor_draft_sections`, `doctor_action_traces`, `doctor_review_idempotency`, `doctor_review_audit_events`, `doctor_review_ledger_heads`, `clinical_retention_events`, `clinical_retention_idempotency`, and `patient_asset_deletion_outbox`.

**Interfaces:**

```python
class DoctorReviewStore:
    def get_latest_draft(self, *, patient_id: int, session_id: str) -> DoctorDraftVersion | None: ...
    def get_draft(self, ref: ClinicalVersionRef) -> DoctorDraftVersion: ...
    def verify_integrity(self, *, subject_id: str) -> None: ...
    def write_generated_draft(
        self,
        draft: DoctorDraftVersion,
        integrity: ClinicalIntegrityRecord,
        audit_event: AuditEvent,
        idempotency_key: str,
        payload_sha256: str,
    ) -> DoctorDraftVersion: ...
    def write_action(
        self,
        plan: DoctorActionWritePlan,
        audit_event: AuditEvent,
        idempotency_key: str,
        payload_sha256: str,
    ) -> DoctorActionWriteResult: ...
    def delete_patient_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        patient_id: int,
        retention_event_id: str,
    ) -> None: ...
    def clear_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        retention_event_id: str,
    ) -> None: ...

class DoctorReviewService:
    def create_generated_draft(
        self,
        *,
        patient_ref: ClinicalVersionRef,
        session_id: str,
        graph_run_id: str,
        source_message_ref: ClinicalVersionRef,
        visible_text: str,
        auth_context: AuthContext,
        idempotency_key: str,
    ) -> DoctorDraftVersion: ...

    def record_action(
        self,
        *,
        patient_id: int,
        session_id: str,
        command: DoctorActionCommand,
        auth_context: AuthContext,
    ) -> DoctorActionWriteResult: ...

class PatientRegistryService:
    def delete_patient(
        self,
        patient_id: int,
        *,
        auth_context: AuthContext,
        idempotency_key: str,
    ) -> PatientDeletionResult: ...
    def clear_registry(
        self,
        *,
        auth_context: AuthContext,
        idempotency_key: str,
    ) -> PatientDeletionResult: ...
```

`DoctorReviewService` receives `RuntimeVersionRegistry` as a server-side dependency. Immediately before sanitization and persistence it calls `snapshot()` and persists that snapshot ref plus labelled bindings. Stage B entry requires `intended_use:patient_crc_triage`, `sanitizer:write_boundary`, and `safety_policy:patient_crc_triage:active`. Task 4 activates and then requires the exact `prompt:doctor_report:active` and `rubric:doctor_report:active` refs through the Stage A expected-current protocol. `clinical_rag:crc_guideline:active` is explicitly optional before Stage C: absence produces a typed `missing_binding` provenance warning and never a fabricated label; once Stage C activates it, the Stage C adapter and all later gates require it. Patient/doctor release targets are validated through the release-target registry, not invented as runtime-version slots. Routes, graph payloads, and callers cannot submit runtime refs or slot labels.

- [ ] Add failing store tests for create-once versions, latest selection, exact reads, chain verification, tamper detection, transaction rollback, same-key/same-payload replay, and same-key/different-payload conflict. Add real `DELETE /api/patient-registry/patients/{patient_id}` and `DELETE /api/patient-registry/patients` route tests requiring server-derived `AuthContext`, `require_project_scope(auth, "closeout:crc", "patient_data_admin")`, and an `Idempotency-Key` header. Cover failure at every patient/review table boundary, retry after rollback, response loss after a successful commit, process restart, same-key/same-canonical-request replay, same-key/different patient/operation/scope conflict, concurrent duplicate requests, and no orphaned rows.
- [ ] Add service tests for `require_project_scope` enforcement, server-derived actors, patient/session binding, exact sanitizer-ref persistence, sanitizer rejection, and missing runtime-evidence warnings.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_review_store.py tests/backend/test_doctor_draft_service.py tests/backend/test_doctor_review_retention.py tests/backend/test_doctor_review_retention_api.py -q -p no:cacheprovider`; expect missing store/service/transactional-cascade failures.
- [ ] Implement the dedicated store over the same restricted SQLite database and injected transaction connection used by `PatientRegistryService`; it owns separate review tables but not a second database file. In one transaction write draft/sections or trace, clinical integrity record, idempotency result, audit event, and ledger head. Validate the existing chain and head before every write.
- [ ] Configure `CLINICAL_INTEGRITY_HMAC_KEY` plus key version only in backend settings. When the review write feature is enabled, absence is a startup failure; never provide an insecure default.
- [ ] Implement one explicit retention policy in `config/clinical_retention.yaml`: the authorized patient delete/clear route canonicalizes operation kind, route schema, internal patient selector, and project scope, then computes server-keyed HMACs for both the idempotency key namespace and canonical request. Raw keys, patient selectors, actor/credential/correlation values, and canonical payloads are never persisted. Before any delete it checks a global restricted `clinical_retention_idempotency` row: same key-HMAC/same request-HMAC returns the stored sanitized result after restart; same key-HMAC/different request-HMAC returns 409 without writes. The first execution reserves and completes that row in the same SQLite transaction as patient/review deletion, `clinical_retention_events`, and a durable `patient_asset_deletion_outbox`. The retained idempotency result contains only random operation/event ID, state, response schema, and aggregate deleted-count fields needed for deterministic compatibility; it contains no patient/ref/hash, asset path, ID list, actor, credential, scope, timestamp, or reversible mapping. The response reconstructs the delete path parameter from the authenticated retry request and always suppresses legacy patient/record/asset ID/path lists; clear responses suppress all ID/path lists on both first execution and replay. The outbox temporarily holds encrypted asset paths until idempotent cleanup succeeds, then deletes those rows; a cleanup crash leaves a restart-safe pending state rather than a best-effort post-commit gap. Failure injection at every table/outbox/cleanup boundary proves rollback or resumable cleanup, stable replay, no duplicate event, and no linkable tombstone fields.
- [ ] Configure `CLINICAL_RETENTION_IDEMPOTENCY_HMAC_KEY` and key version with no insecure default. Tests inspect the database to prove raw idempotency keys, patient IDs, canonical payloads, asset paths, and reversible hashes are absent; key rotation preserves replay for existing keyed rows through the configured verification key ring.
- [ ] Do not extend `PatientCommandService.record_doctor_action_trace`; route Stage B writes through the dedicated store so patient snapshot versions do not change.
- [ ] Re-run that exact command and require PASS, then commit only this task:

```powershell
git add backend/api/services/doctor_review_store.py backend/api/services/doctor_review_service.py src/contracts/clinical_retention.py config/clinical_retention.yaml tests/backend/test_doctor_review_store.py tests/backend/test_doctor_draft_service.py tests/backend/test_doctor_review_retention.py tests/backend/test_doctor_review_retention_api.py backend/api/services/settings.py backend/app.py backend/api/services/patient_registry_service.py backend/api/routes/patient_registry.py backend/api/schemas/patient_registry.py
git commit -m "feat: persist immutable doctor draft versions"
```

---

### Task 4: Capture Real Sanitized Graph Output And Block Thinking

**Files:**

- Modify: `backend/api/services/payload_builder.py`
- Modify: `backend/api/services/graph_service.py`
- Modify: `backend/api/schemas/events.py`
- Modify: `backend/app.py`
- Modify: `tests/backend/test_graph_service_streaming.py`
- Create: `tests/backend/test_doctor_draft_graph_capture.py`

**Stream signature:**

```python
def stream_turn(
    self,
    session_id: str,
    chat_request: Any,
    *,
    auth_context: AuthContext | None = None,
) -> AsyncIterator[str]: ...
```

- [ ] Add failing tests for a versioned `doctor_report_draft_action`, capture of the final filtered visible `MessageDoneEvent`, exact graph-run/source-message/sanitizer refs, one draft per idempotency key, and no draft on failed/cancelled runs.
- [ ] Keep Stage A's patient/doctor SSE thinking-removal tests as inherited GREEN regressions; add new RED assertions that filtered output, credentials, prompt secrets, and direct identifiers cannot enter draft storage, logs, or error payloads.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_graph_service_streaming.py tests/backend/test_doctor_draft_graph_capture.py -q -p no:cacheprovider`. The inherited transport assertions must remain green; new draft-capture/storage assertions fail until this task is implemented.
- [ ] Extend the report action context, capture the filtered visible output only after a successful doctor report run, sanitize using the Stage A `doctor_free_text` scope, perform independent de-identification validation, persist the draft, then emit the completion event with its opaque ref.
- [ ] Resolve the actual doctor-report prompt/rubric artifacts to VersionRefs and activate `prompt:doctor_report:active` and `rubric:doctor_report:active` with expected-current, protected AuthContext, and idempotency keys before enabling graph draft capture. Tests block stale/unauthorized activation and prove every captured draft carries those labelled bindings.
- [ ] If an exact runtime evidence-index VersionRef is unavailable, create the draft with an explicit unverified/missing-provenance warning. Never insert `rag_crc_guideline_20260620` or another static label as evidence.
- [ ] Re-run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_graph_service_streaming.py tests/backend/test_doctor_draft_graph_capture.py -q -p no:cacheprovider
```

Expected: all pass and no raw sensitive value appears in captured output.

- [ ] Commit only Task 4 paths:

```powershell
git add backend/api/services/payload_builder.py backend/api/services/graph_service.py backend/api/schemas/events.py backend/app.py tests/backend/test_graph_service_streaming.py tests/backend/test_doctor_draft_graph_capture.py
git commit -m "feat: capture real sanitized doctor report drafts"
```

---

### Task 5: Same-Source Doctor Draft And Review Read APIs

**Files:**

- Modify: `backend/api/schemas/doctor_review.py`
- Modify: `backend/api/routes/doctor_review.py`
- Modify: `tests/backend/test_doctor_review_api.py`
- Create: `tests/backend/test_doctor_draft_api.py`

**Endpoints:**

```text
GET /api/sessions/{session_id}/doctor-drafts/latest
GET /api/sessions/{session_id}/doctor-review
```

- [ ] Add failing API tests asserting both endpoints return the exact same public `ClinicalVersionProjection`, sections/text/version, section versions, runtime snapshot ref, labelled runtime bindings, and sanitizer ref; reject any client attempt to inject runtime refs or slot labels.
- [ ] Assert no draft returns typed `null`/empty state and never calls `_build_draft()` to synthesize placeholder content.
- [ ] Assert each section returns `action_state_contract_version`, backend-derived `allowed_actions`, citation confidence, EvidenceClaim review status, care-card provenance, and missing-provenance warnings.
- [ ] Add role tests: clinical read roles work; a legacy shared/admin token is read-only; cross-project/session access returns a non-enumerating response.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_review_api.py tests/backend/test_doctor_draft_api.py -q -p no:cacheprovider`; expect static `_build_draft()` and schema failures.
- [ ] Replace placeholder construction with `DoctorReviewStore.get_latest_draft()` in both endpoints and serialize only the approved clinical projection.
- [ ] Re-run both API test files, require PASS, and commit only Task 5 paths:

```powershell
git add backend/api/schemas/doctor_review.py backend/api/routes/doctor_review.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_draft_api.py
git commit -m "feat: serve same-source doctor draft review data"
```

---

### Task 6: Secure DoctorActionTrace Write API

**Files:**

- Modify: `backend/api/schemas/doctor_action_trace.py`
- Modify: `backend/api/routes/doctor_review.py`
- Modify: `backend/api/services/doctor_review_service.py`
- Modify: `backend/api/services/doctor_review_store.py`
- Modify: `tests/backend/test_doctor_action_trace.py`
- Create: `tests/backend/test_doctor_action_trace_security.py`
- Create: `tests/backend/test_doctor_action_trace_idempotency.py`

**Required request fields:** `idempotency_key`, exact draft ID/version, exact section ID/version, action type, optional typed assertion/assessment/record/care-card/citation refs, edit expected-before/after, and an allowlisted reason code.

- [ ] Add failing tests for the validation order: role/scope, audit integrity, idempotency, exact latest draft/section, backend allowed action, target resolution, sanitizer/de-identification, then atomic write.
- [ ] Cover all six actions, stale/superseded refs, unknown action, missing target, cross-patient/cross-session target, non-existent citation evidence, client reviewer-role spoofing, and generic non-enumerating conflicts.
- [ ] Assert `edit` atomically creates exactly one superseding draft plus one trace/audit event; the other five create no draft and change no clinical data.
- [ ] Assert response actor data comes only from AuthContext and trace uses `content_sanitized=true`, `free_text_deidentified=true`, and `patient_linked=true`; retire the misleading blanket `deidentified=true` claim.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_action_trace.py tests/backend/test_doctor_action_trace_security.py tests/backend/test_doctor_action_trace_idempotency.py -q -p no:cacheprovider`; observe current incomplete action/target/idempotency failures.
- [ ] Implement target resolvers for assertion, assessment, record, care card, draft/section, citation, and evidence object. Validate actual existence and context before writing.
- [ ] Persist exact before/after only for edit after sanitizer and de-identification validation. Reason codes are allowlisted enums; no unbounded reviewer narrative enters the trace.
- [ ] Re-run the three trace test files, require PASS, and commit only Task 6 paths:

```powershell
git add backend/api/schemas/doctor_action_trace.py backend/api/routes/doctor_review.py backend/api/services/doctor_review_service.py backend/api/services/doctor_review_store.py tests/backend/test_doctor_action_trace.py tests/backend/test_doctor_action_trace_security.py tests/backend/test_doctor_action_trace_idempotency.py
git commit -m "feat: secure doctor action trace writes"
```

---

### Task 7: Report Frontend Uses The Persisted Draft

**Files:**

- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/test/test-utils.tsx`
- Modify: `frontend/src/features/doctor/doctor-report-draft-utils.ts`
- Modify: `frontend/src/features/doctor/doctor-report-draft-utils.test.ts`
- Modify: `frontend/src/features/doctor/doctor-report-draft-view.tsx`
- Modify: `frontend/src/features/doctor/doctor-report-draft-view.test.tsx`
- Modify: `frontend/src/features/doctor/doctor-scene-shell.tsx`
- Modify: `frontend/src/features/doctor/doctor-scene-shell.test.tsx`

- [ ] Add failing tests proving ordinary AI messages cannot replace a persisted report draft, the generation request sends `doctor_report_draft_action`, loading/empty/error states are typed, and the displayed ref/version matches the API.
- [ ] Run the exact focused suite before changing implementation:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/doctor/doctor-report-draft-utils.test.ts src/features/doctor/doctor-report-draft-view.test.tsx src/features/doctor/doctor-scene-shell.test.tsx src/app/api/client.test.ts --reporter=verbose
```

Expected RED: the new persisted-draft source/ref/loading assertions fail against the message-derived implementation.

- [ ] Remove `latestReportDraftFromMessages()` as the authoritative report source. It may remain only as a migration display helper if no write/action path consumes it; otherwise delete it and update tests.
- [ ] Fetch `/doctor-drafts/latest` by `sessionId`, render sections/provenance, and refresh only on a successful generation result carrying a new draft ref.
- [ ] Re-run the exact focused command above. Expected GREEN: all tests pass.

- [ ] Commit only Task 7 paths:

```powershell
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/test/test-utils.tsx frontend/src/features/doctor/doctor-report-draft-utils.ts frontend/src/features/doctor/doctor-report-draft-utils.test.ts frontend/src/features/doctor/doctor-report-draft-view.tsx frontend/src/features/doctor/doctor-report-draft-view.test.tsx frontend/src/features/doctor/doctor-scene-shell.tsx frontend/src/features/doctor/doctor-scene-shell.test.tsx
git commit -m "feat: render persisted doctor drafts in reports"
```

---

### Task 8: Review Cockpit Renders All Backend-Allowed Actions

**Files:**

- Modify: `frontend/src/features/doctor/doctor-review-events.ts`
- Modify: `frontend/src/features/doctor/doctor-review-events.test.ts`
- Modify: `frontend/src/features/doctor/doctor-review-cockpit.tsx`
- Modify: `frontend/src/features/doctor/doctor-review-cockpit.test.tsx`
- Create: `tests/e2e/acceptance/doctor-review-six-actions.spec.ts`

**Builder:**

```typescript
buildDoctorActionTrace({
  actionType,
  draftRef,
  sectionId,
  sectionVersion,
  targetRefs,
  beforeAfter,
  reasonCode,
  idempotencyKey,
})
```

- [ ] Replace the existing test fixtures with failing typed-builder and UI assertions for all six actions, hidden disallowed actions, stale response handling, edit ref refresh, duplicate-submit reuse of the same key, and absence of authoritative reviewer roles in requests. Add a failing real Playwright UI→API→restricted-store spec that generates one persisted draft and exercises all six backend-allowed actions.
- [ ] Run before changing implementation:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/doctor/doctor-review-events.test.ts src/features/doctor/doctor-review-cockpit.test.tsx --reporter=verbose
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/doctor-review-six-actions.spec.ts
```

Expected RED: typed-builder/all-six-action assertions and the end-to-end restricted-store flow fail against the incomplete cockpit.

- [ ] Implement one typed builder, render only `section.allowed_actions`, disable the section during an in-flight command, and surface allowlisted conflict reasons without echoing submitted sensitive text.
- [ ] For edit, send exact expected before/after and update the UI to the returned superseding draft ref. For other actions, retain the same draft ref. The Playwright assertions require six successful audit traces, only one superseding edit draft, stale replay conflict, server-derived actor identity, and unchanged patient/policy/RAG/flag snapshots.
- [ ] Re-run both exact commands above. Expected GREEN: both exit 0 before commit.
- [ ] Commit only Task 8 paths:

```powershell
git add frontend/src/features/doctor/doctor-review-events.ts frontend/src/features/doctor/doctor-review-events.test.ts frontend/src/features/doctor/doctor-review-cockpit.tsx frontend/src/features/doctor/doctor-review-cockpit.test.tsx tests/e2e/acceptance/doctor-review-six-actions.spec.ts
git commit -m "feat: render versioned doctor review actions"
```

---

### Task 9: Hidden-Reasoning Removal, Non-Mutation, And Failure Injection

**Files:**

- Modify: `frontend/src/features/doctor/doctor-consultation-view.tsx`
- Modify: `frontend/src/features/chat/conversation-panel.tsx`
- Modify: `frontend/src/features/doctor/doctor-scene-shell.test.tsx`
- Create: `tests/backend/test_doctor_review_non_mutation.py`
- Create: `tests/backend/test_doctor_review_failure_injection.py`

- [ ] Treat Stage A's E2E absence of raw thinking, `<think>` content, and `.clinical-thinking-disclosure` as an inherited regression. Add UI-specific failing tests proving no dormant disclosure component can render hidden-reasoning fields if malformed local test data supplies them.
- [ ] Add failing non-mutation snapshots covering patient facts/records/snapshot/care cards, ClinicalSafetyPolicyVersion, prompt/rubric/route/template, RAG/evidence index, feature flags, and model/training data. For each non-edit action, assert snapshots and draft-version count are unchanged; for edit, assert only one superseding draft plus its trace/audit/idempotency rows.
- [ ] Add failing injection cases after draft/trace insert, after audit insert, and before ledger-head update. Require full rollback, same-key retry, and no partial row.
- [ ] Run before the implementation/removal work:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_review_non_mutation.py tests/backend/test_doctor_review_failure_injection.py tests/backend/test_auth_security.py -q -p no:cacheprovider
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/frontend-regression-contracts.spec.ts
```

Expected RED: the new failure points/non-mutation assertions and malformed hidden-reasoning UI case fail before the production paths are hardened.

- [ ] Remove every dormant hidden-reasoning render path, implement the missing transaction rollback boundaries, and constrain action mutations to the exact states above.
- [ ] Re-run both exact commands above. Expected GREEN: backend tests pass; Playwright reports all tests passed and no thinking disclosure exists.

- [ ] Commit only Task 9 paths:

```powershell
git add frontend/src/features/doctor/doctor-consultation-view.tsx frontend/src/features/chat/conversation-panel.tsx frontend/src/features/doctor/doctor-scene-shell.test.tsx tests/backend/test_doctor_review_non_mutation.py tests/backend/test_doctor_review_failure_injection.py
git commit -m "test: enforce doctor review safety boundaries"
```

---

### Task 10: Register, Merge, And Approve The Stage B Gate

**Files:**

- Modify: `config/closeout_stage_suites.yaml`
- Create: `tests/backend/test_stage_b_clinical_review_gate.py`
- Create after merge: `reports/closeout/stages/stage_b.<merge-sha12>.json`
- Create after report validation: `reports/closeout/attestations/stage_b_approval.<report-hash12>.json`

- [ ] Add failing exact-set Stage B gate tests without changing the suite manifest.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_stage_b_clinical_review_gate.py -q -p no:cacheprovider`. Expected RED: Stage B is unregistered and required owned entries/artifacts are absent.
- [ ] Register Stage B owned requirement IDs, fixed cwd/argv commands, required artifact kinds, zero-skip policy, and `inherits: [A]` in the existing suite manifest. The inherited runner must execute and record the complete expanded Stage A suite, not a sampled subset. Do not fork the Stage A runner.
- [ ] Run the complete backend verification:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_draft_contract.py tests/backend/test_doctor_action_state_contract.py tests/backend/test_clinical_assertion_projection.py tests/backend/test_patient_registry_record_assertions.py tests/backend/test_doctor_review_store.py tests/backend/test_doctor_draft_service.py tests/backend/test_doctor_review_retention.py tests/backend/test_doctor_review_retention_api.py tests/backend/test_doctor_draft_graph_capture.py tests/backend/test_doctor_draft_api.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py tests/backend/test_doctor_action_trace_security.py tests/backend/test_doctor_action_trace_idempotency.py tests/backend/test_doctor_review_non_mutation.py tests/backend/test_doctor_review_failure_injection.py tests/backend/test_stage_b_clinical_review_gate.py tests/backend/test_auth_security.py -q -p no:cacheprovider
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py tests/backend/test_graph_service_streaming.py -q -p no:cacheprovider
```

Expected: all required tests pass with zero skips.

- [ ] Run the complete frontend verification:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/doctor/doctor-report-draft-utils.test.ts src/features/doctor/doctor-report-draft-view.test.tsx src/features/doctor/doctor-review-events.test.ts src/features/doctor/doctor-review-cockpit.test.tsx src/features/doctor/doctor-scene-shell.test.tsx src/app/api/client.test.ts --reporter=verbose
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/doctor-review-six-actions.spec.ts
```

Expected: Vitest passes and Vite/TypeScript exits 0.

- [ ] Commit the Stage B suite/test, then run branch advisory against that exact commit:

```powershell
git add config/closeout_stage_suites.yaml tests/backend/test_stage_b_clinical_review_gate.py
git commit -m "test: register stage b closeout suite"
$branchHead = (git rev-parse HEAD).Trim()
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage A --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage A merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage A merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $branchHead
if ($LASTEXITCODE -ne 0) { throw "Stage B branch does not descend from approved Stage A" }
$manifestAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind closeout_manifest --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts\run_closeout_stage_gate.py --stage B --mode branch-advisory --base-sha $stageBase --head-sha $branchHead --tested-content-sha $branchHead --plan docs\superpowers\plans\2026-07-10-crc-agent-closeout-stage-b-clinical-review.md --plan-subject reports\closeout\plan_subjects\stage_b_plan_20260710_001.json --plan-attestation reports\closeout\attestations\stage_b_plan_approval_20260710_001.json --manifest-attestation $manifestAttestationPath --suite config\closeout_stage_suites.yaml --output-root output\closeout-advisory
```

Expected: every Stage B owned required row passes; inherited Stage A regressions remain green.

- [ ] Merge through the protected workflow. From a clean checkout, rerun the complete verification above, bind the actual merge SHA, and publish the post-merge report:

```powershell
$actualMergeSha = $env:LANGG_STAGE_B_MERGE_SHA
$checkoutSha = (git rev-parse HEAD).Trim()
if (($actualMergeSha -notmatch '^[0-9a-f]{40}$') -or ($checkoutSha -ne $actualMergeSha)) { throw "checkout is not the recorded protected Stage B merge" }
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage A --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage A merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage A merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $actualMergeSha
if ($LASTEXITCODE -ne 0) { throw "Stage B merge does not descend from approved Stage A" }
$mergeSha12 = $actualMergeSha.Substring(0, 12)
$manifestAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind closeout_manifest --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_stage_gate.py --stage B --mode post-merge --base-sha $stageBase --head-sha $actualMergeSha --tested-content-sha $actualMergeSha --merged-sha $actualMergeSha --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-b-clinical-review.md --plan-subject reports/closeout/plan_subjects/stage_b_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_b_plan_approval_20260710_001.json --manifest-attestation $manifestAttestationPath --suite config/closeout_stage_suites.yaml --publish
```

- [ ] Reuse Stage A's stage-gate subcommands, collect the two Stage B approvals against the exact report hash/version, and exact-stage only the report/attestation/event chain:

```powershell
$stageReportPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-published --subject-kind stage_gate --stage B --field subject_path).Trim()
$reportHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field sha256).Trim()
$reportVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field subject_version).Trim()
$reportHash12 = $reportHash.Substring(0, 12)
$attestationPath = "reports/closeout/attestations/stage_b_approval.$reportHash12.json"
$gateEvidencePaths = "output/closeout/stage-b-gate-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$reportHash-stage-b-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$reportHash-stage-b-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_gate --subject-path $stageReportPath --output $attestationPath --path-list-output $gateEvidencePaths --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_gate --subject-path $stageReportPath --attestation-path $attestationPath
git add --pathspec-from-file=$gateEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $gateEvidencePaths --subject-path $stageReportPath --attestation-path $attestationPath
git commit -m "evidence(stage-b): record post-merge clinical review gate"
```
- [ ] Open Stage C only after the merged report, approval, latest manifest, and all inherited chains validate. If shared Stage A contracts changed, revalidate or revoke Stage A explicitly.

## Plan Self-Review Checklist

- [ ] Every Stage B design requirement maps to a task and owned manifest row.
- [ ] The report and Review Cockpit have one persisted draft source and identical public ClinicalVersionProjections backed by the same internal ClinicalVersionRef.
- [ ] All six actions have backend state, authorization, target-resolution, idempotency, frontend, and non-mutation tests.
- [ ] The plan never trusts a client reviewer role or exposes restricted clinical integrity data.
- [ ] Doctor thinking disclosure and static placeholder-draft behavior are explicitly removed with regression tests.
- [ ] Stage A shared types and runners are consumed by exact path and not duplicated.
- [ ] Every red/green command has an expected result; failure injection proves atomic rollback and retry.
- [ ] `git diff --check` passes, code fences close, and task commits stage only named files.
