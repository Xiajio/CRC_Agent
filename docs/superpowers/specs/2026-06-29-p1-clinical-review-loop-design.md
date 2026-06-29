# P1 Clinical Review Loop Design

> Version: 2026-06-29  
> Scope: P1 Step 7-9  
> Source plan: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`  
> Depends on: `docs/superpowers/specs/2026-06-29-p0-crc-safety-loop-design.md` and P0 commits `29b6a66`, `c15ba37`  
> Goal: Let the doctor side safely read CRC assessment evidence, review provenance, and record structured review actions without replacing clinician judgment or automatically modifying safety policy, prompts, RAG, or training data.

## 1. Background

P0 converged the CRC triage safety loop. The system now has intended-use profiles, deterministic `ClinicalSafetyPolicyVersion`, mutation fixtures, safety metadata in CRC assessments, persistence traceability, and static `HarnessRun` / `ReleaseSafetyReport` artifacts.

P1 should not expand the patient triage behavior again. Its job is to build the first clinician-facing review loop on top of P0 evidence:

1. Convert persisted patient facts and CRC assessments into canonical, traceable `ClinicalAssertion` objects.
2. Add a feature-flagged Doctor Review Cockpit that reads those assertions and provenance without replacing the existing doctor flow.
3. Record field-level `DoctorActionTrace` events as review/distillation candidates, not as automatic truth.

The core product direction is: **patient facts become traceable assertions; doctor review reads and annotates them; doctor feedback becomes auditable candidate signal only.**

## 2. Current Project Context

Relevant P0 files already exist:

- `config/intended_use_profiles.yaml`
- `config/safety_policy.yaml`
- `src/services/clinical_safety_policy.py`
- `src/services/crc_triage_flow.py`
- `backend/api/routes/crc_triage.py`
- `backend/api/services/patient_commands.py`
- `backend/api/services/patient_care_cards.py`
- `src/contracts/harness.py`
- `src/contracts/release_safety_report.py`
- `tests/backend/test_crc_triage_save.py`
- `tests/backend/test_crc_harness_replay.py`

Relevant patient record and projection files:

- `backend/api/services/patient_registry_service.py`
- `backend/api/services/patient_commands.py`
- `backend/api/schemas/patient_registry.py`
- `backend/api/routes/patient_registry.py`
- `frontend/src/features/patient-records/patient-records-panel.tsx`
- `frontend/src/features/patient-registry/registry-browser-pane.tsx`

Relevant doctor-side files:

- `backend/api/services/graph_service.py`
- `backend/api/services/patient_context_resolver.py`
- `src/graph_builder.py`
- `src/nodes/assessment_nodes.py`
- `src/nodes/decision_nodes.py`
- `frontend/src/features/doctor/doctor-scene-shell.tsx`
- `frontend/src/features/doctor/doctor-consultation-view.tsx`
- `frontend/src/features/doctor/doctor-report-draft-view.tsx`
- `frontend/src/features/doctor/use-doctor-view-state.ts`

Missing P1 artifacts:

- `src/contracts/clinical_assertion.py`
- `src/services/clinical_assertion_projection.py`
- `backend/api/schemas/doctor_review.py`
- `backend/api/routes/doctor_review.py`
- `backend/api/schemas/doctor_action_trace.py`
- `frontend/src/features/doctor/doctor-review-cockpit.tsx`
- `frontend/src/features/doctor/doctor-review-events.ts`
- `tests/backend/test_clinical_assertion_projection.py`
- `tests/backend/test_doctor_review_api.py`
- `tests/backend/test_doctor_action_trace.py`
- focused frontend tests for the feature-flagged cockpit and action event payloads

At spec creation time the branch is `main`. The working tree has an existing modified `CRC-client` entry that is outside this P1 scope and must not be touched.

## 3. Design Decision

Use a layered full-P1 design.

### Recommended Approach: Layered P1

P1 is implemented in three layers:

1. `ClinicalAssertion` projection.
2. Feature-flagged Doctor Review Cockpit.
3. `DoctorActionTrace` capture.

Benefits:

- Covers all P1 steps while keeping each layer independently testable.
- Keeps doctor review read-only until action capture is explicitly added.
- Prevents doctor edits from silently becoming model truth or safety-rule changes.
- Gives P1.5/P2 a stable assertion/provenance substrate later.

Trade-off:

- Requires careful route/schema design before the UI becomes useful.

### Rejected Approach: Backend Contracts Only

Defining only schemas and backend projections would be lower risk, but it would not complete the P1 requirement that doctors can review CRC assessments and provenance in a usable surface.

### Rejected Approach: Full Cockpit Replacement

Replacing the existing doctor flow with a new cockpit would create unnecessary regression risk. P1 should add an experimental panel or tab controlled by a feature flag, not replace the existing doctor consultation, report draft, or database views.

## 4. Scope

P1 includes:

1. `ClinicalAssertion` contract and projection from patient records, CRC assessments, uploaded report facts, and safety metadata.
2. Assertion references on patient record projections, without requiring migration of old records.
3. Doctor Review Cockpit MVP behind a feature flag.
4. Provenance view that distinguishes patient facts, uploaded records, CRC safety policy metadata, RAG/evidence refs when present, and model-generated unverified text.
5. `DoctorActionTrace` schema and API for accept, edit, reject, escalate, request evidence, and mark unsafe actions.
6. `reason_code` enum for field-level review signal.
7. Tests proving P1 does not change P0 safety policy behavior or the existing doctor flow by default.

P1 excludes:

- Changing `ClinicalSafetyPolicyVersion` from doctor feedback.
- Automatic SFT/DPO dataset creation.
- Automatic prompt, rubric, route, template, or RAG patching.
- Literature `EvidenceClaim`, `EvidenceDelta`, `IngestPreview`, or clinical RAG ingestion.
- Agent Admin release dashboard.
- Research cohort feasibility or patient-level research export.
- Replacing existing doctor UI screens.
- New authentication, OIDC, role-based permission system, Redis locks, SSE resume, or FHIR server.
- Direct modification of `CRC-client`.

## 5. Architecture

P1 builds on P0 in this order:

```text
persisted CRC assessment / patient records
  -> ClinicalAssertion projection
    -> doctor review read model
      -> feature-flagged Doctor Review Cockpit
        -> DoctorActionTrace event capture
          -> review/distillation candidate signal
```

### Layer 1: ClinicalAssertion Projection

Files:

- `src/contracts/clinical_assertion.py`
- `src/services/clinical_assertion_projection.py`
- `backend/api/services/patient_registry_service.py`
- `backend/api/services/patient_commands.py`
- `tests/backend/test_clinical_assertion_projection.py`

Responsibility:

- Normalize patient facts into traceable assertions.
- Preserve source evidence refs, record IDs, assessment IDs, safety policy version, and review status.
- Generate deterministic assertion IDs so the same record projection produces stable references.
- Keep old records compatible by producing zero assertions or best-effort assertions without failing the patient record API.

### Layer 2: Doctor Review Read Model

Files:

- `backend/api/schemas/doctor_review.py`
- `backend/api/routes/doctor_review.py`
- `backend/api/services/patient_registry_service.py`
- `backend/api/services/graph_service.py`
- `tests/backend/test_doctor_review_api.py`

Responsibility:

- Expose a read-only review payload for a doctor session bound to a patient.
- Include patient timeline items, CRC assessment summary, assertion refs, care cards, safety metadata, and provenance.
- Mark missing provenance as `model_generated_unverified`.
- Avoid invoking model calls, network calls, or RAG ingestion when building the read model.

### Layer 3: Feature-Flagged Doctor Review Cockpit

Files:

- `frontend/src/features/doctor/doctor-review-cockpit.tsx`
- `frontend/src/features/doctor/doctor-review-cockpit.test.tsx`
- `frontend/src/features/doctor/use-doctor-view-state.ts`
- `frontend/src/features/doctor/doctor-scene-shell.tsx`
- `frontend/src/features/doctor/doctor-review-events.ts`

Responsibility:

- Add an experimental cockpit panel or tab when `doctor_review_cockpit_v0` is enabled.
- Display a patient fact timeline, agent draft/provenance, and available doctor actions.
- Do not remove or rename existing doctor views.
- Do not show unverified model text as guideline fact.

### Layer 4: DoctorActionTrace

Files:

- `backend/api/schemas/doctor_action_trace.py`
- `backend/api/routes/doctor_review.py`
- `backend/api/services/patient_commands.py`
- `frontend/src/features/doctor/doctor-review-events.ts`
- `tests/backend/test_doctor_action_trace.py`

Responsibility:

- Record field-level doctor actions as auditable patient events or review events.
- Keep action payloads de-identified by default.
- Link actions to `assertion_id`, `assessment_id`, `record_id`, `care_card_id`, `citation_id`, or `draft` target paths.
- Store review signal without modifying patient facts, safety policy, prompts, RAG, or model training data.

## 6. ClinicalAssertion Contract

Minimum object:

```json
{
  "assertion_id": "assertion_crc_assessment_abc123_rectal_bleeding",
  "patient_id": "33",
  "session_id": "sess_patient_001",
  "source": "triage",
  "source_record_id": "record_42",
  "source_assessment_id": "crc_assessment_abc123",
  "normalized_fact": {
    "type": "condition_signal",
    "name": "rectal_bleeding",
    "value": true
  },
  "evidence_refs": [
    {
      "kind": "patient_record",
      "id": "record_42",
      "field": "payload.known_crc_signals.rectal_bleeding"
    }
  ],
  "confidence": "structured_user_report",
  "reviewed_status": "unreviewed",
  "safety_policy_version": "crc_safety_policy_v0",
  "created_from_projection_version": "patient_record_projection_v0"
}
```

Required fields:

- `assertion_id`
- `patient_id`
- `source`
- `normalized_fact`
- `evidence_refs`
- `confidence`
- `reviewed_status`

Allowed `source` values:

- `triage`
- `patient_upload`
- `doctor_note`
- `database_snapshot`
- `care_card`
- `model_draft`

Allowed `normalized_fact.type` values for P1:

- `condition_signal`
- `symptom`
- `risk_disposition`
- `missing_information`
- `test_status`
- `safety_rule_match`
- `document_fact`

Allowed `reviewed_status` values:

- `unreviewed`
- `accepted`
- `edited`
- `rejected`
- `needs_evidence`
- `unsafe`

Assertion ID rule:

```text
assertion_<source>_<source_object_id>_<normalized_fact.name>_<stable_hash_8>
```

The stable hash must be computed from patient id, source object id, normalized fact, and evidence refs. It must not use current time, random UUID, model output, or frontend state.

## 7. ClinicalAssertion Projection Rules

Projection from CRC triage assessment:

- `known_crc_signals.rectal_bleeding = true` becomes `condition_signal: rectal_bleeding`.
- `red_flags.weight_loss = true` becomes `symptom: weight_loss`.
- `disposition` or `risk_class` becomes `risk_disposition`.
- `missing_information` becomes one assertion per missing field.
- `matched_rules` becomes one `safety_rule_match` assertion per rule.
- `safety_policy_version` is copied into each derived assertion.
- `assessment_id` is copied into `source_assessment_id`.

Projection from uploaded records:

- Medical card snapshot fields become `document_fact` assertions only when already accepted into the patient record projection.
- Upload parse failures do not create clinical assertions.
- Unreviewed upload summaries may create assertions with `confidence: extracted_report_candidate`, but those assertions must remain `reviewed_status: unreviewed`.

Projection from doctor notes:

- P1 may define the source value and target shape, but it must not auto-promote doctor free text into patient truth.
- Doctor-created assertions require explicit accept or edit action in `DoctorActionTrace`.

Compatibility:

- Old records without `assessment_id`, safety metadata, or assertion refs must not fail projection.
- The projection should return an empty list or partial assertions with evidence refs that point to available record IDs.
- Existing `PatientRegistryRecord` response fields remain backward compatible.

## 8. Patient Record Projection Requirements

Patient detail and record APIs may add optional fields:

```json
{
  "assertion_refs": [
    "assertion_triage_record_42_rectal_bleeding_a1b2c3d4"
  ],
  "clinical_assertions": [
    {
      "assertion_id": "assertion_triage_record_42_rectal_bleeding_a1b2c3d4",
      "source": "triage",
      "normalized_fact": {
        "type": "condition_signal",
        "name": "rectal_bleeding",
        "value": true
      },
      "reviewed_status": "unreviewed"
    }
  ]
}
```

Rules:

- New fields are optional and additive.
- Existing frontend record panels must continue to render if these fields are missing.
- Assertion refs should be derived server-side from records/projection, not stored in frontend local state as a second source.
- P1 should not require a destructive database migration. If persistence is needed, use additive columns or derived read-time projection.

## 9. Doctor Review Read Model

The read model returned by the doctor review API should minimally include:

```json
{
  "patient_id": 33,
  "session_id": "sess_doctor_001",
  "feature_flag": "doctor_review_cockpit_v0",
  "timeline": [
    {
      "item_id": "record_42",
      "kind": "crc_triage_assessment",
      "title": "CRC专项预问诊",
      "created_at": "2026-06-29T10:00:00+08:00",
      "assertion_refs": ["assertion_triage_record_42_rectal_bleeding_a1b2c3d4"]
    }
  ],
  "assertions": [],
  "draft": {
    "draft_id": "draft_crc_review_33_latest",
    "sections": [
      {
        "section_id": "risk_summary",
        "text": "患者报告便血，年龄超过50岁，建议尽快临床复核。",
        "provenance": [
          {
            "kind": "patient_fact",
            "assertion_id": "assertion_triage_record_42_rectal_bleeding_a1b2c3d4"
          }
        ],
        "verification_status": "traceable"
      }
    ]
  },
  "available_actions": [
    "accept",
    "edit",
    "reject",
    "escalate",
    "request_evidence",
    "mark_unsafe"
  ]
}
```

Read model rules:

- It must require a doctor session bound to a patient.
- It must not be available to patient sessions.
- It must not invoke a model just to render the review payload.
- If a draft sentence has no patient fact or evidence ref, `verification_status` must be `model_generated_unverified`.
- Candidate literature evidence must not be shown as clinical evidence in P1 unless it is already part of an existing approved source path.

## 10. Doctor Review Cockpit MVP

The cockpit should be an experimental doctor-side panel or tab with four regions:

1. Patient fact timeline
   - CRC assessment.
   - Patient records.
   - Care cards.
   - Upload summaries.

2. Agent draft
   - Summary.
   - Risk points.
   - Suggested follow-up questions.
   - Report draft excerpt when available.

3. Provenance and evidence
   - Patient fact refs.
   - Source record refs.
   - CRC safety policy refs.
   - Citation/evidence refs only when present.
   - `model_generated_unverified` label for unsupported text.

4. Doctor actions
   - Accept.
   - Edit.
   - Reject.
   - Escalate.
   - Request evidence.
   - Mark unsafe.

Feature flag:

```text
doctor_review_cockpit_v0
```

Feature-flag behavior:

- Disabled: existing doctor flow is unchanged.
- Enabled: cockpit appears as an experimental panel/tab.
- No route should force the user into the cockpit by default during P1.

## 11. DoctorActionTrace Contract

Minimum object:

```json
{
  "trace_id": "doctor_trace_01JABC",
  "patient_id": 33,
  "session_id": "sess_doctor_001",
  "action_type": "edit",
  "target_object": "draft.risk_summary",
  "target_refs": {
    "draft_id": "draft_crc_review_33_latest",
    "assertion_id": "assertion_triage_record_42_rectal_bleeding_a1b2c3d4",
    "assessment_id": "crc_assessment_abc123",
    "record_id": "record_42"
  },
  "before_after": {
    "before": "建议观察",
    "after": "建议尽快线下临床评估"
  },
  "reason_code": "unsafe_disposition",
  "reviewer_role": "physician_reviewer",
  "deidentified": true,
  "timestamp": "2026-06-29T15:00:00+08:00"
}
```

Allowed `action_type` values:

- `accept`
- `edit`
- `reject`
- `escalate`
- `request_evidence`
- `mark_unsafe`

Allowed `reason_code` values:

- `fact_wrong`
- `missing_red_flag`
- `unsupported_claim`
- `bad_tone`
- `workflow_mismatch`
- `citation_not_traceable`
- `missing_information`
- `unsafe_disposition`
- `evidence_conflict`
- `template_mismatch`

Targeting rules:

- Every trace must target at least one stable object: `draft_id`, `assertion_id`, `assessment_id`, `record_id`, `care_card_id`, `citation_id`, or `target_object`.
- `edit` requires `before_after.before` and `before_after.after`.
- `request_evidence` requires a target and `reason_code: citation_not_traceable`, `unsupported_claim`, `missing_information`, or `evidence_conflict`.
- `mark_unsafe` requires a safety-related target and `reason_code: unsafe_disposition`, `missing_red_flag`, or `unsupported_claim`.

Storage rule:

- P1 should store traces as append-only review events.
- Traces may use the existing patient event infrastructure if implemented without mutating clinical snapshots.
- Traces must not directly update `ClinicalAssertion.reviewed_status` unless the implementation explicitly treats the status update as a derived read model from action events.

Privacy rule:

- Do not store hidden chain-of-thought.
- Do not store prompt secrets, API keys, or model debug payloads.
- Default traces must be de-identified enough for review/distillation workflows. When `before_after` contains free text, it should be stored only because the doctor explicitly edited that visible text.

## 12. Data Flow

```text
P0 CRC assessment is saved
  -> patient record contains assessment_id and safety metadata
  -> ClinicalAssertion projector derives assertion refs
  -> doctor session binds patient
  -> doctor review API builds read-only review payload
  -> frontend cockpit renders timeline, draft, provenance, and actions
  -> doctor clicks an action
  -> DoctorActionTrace API validates target and reason_code
  -> append-only review event is stored
  -> read model can show action status without changing safety policy
```

## 13. Error Handling

ClinicalAssertion projection errors:

- Invalid record payload: skip that record and include a projection warning in tests/logs, not in patient-facing UI.
- Missing `assessment_id`: use record ID as fallback source object ID.
- Missing safety metadata: create clinical facts that can be traced to records, but omit safety-policy-specific fields.
- Duplicate facts: produce deterministic assertion IDs and de-dupe by `assertion_id`.

Doctor review API errors:

- Patient session attempts doctor review: return 409 or 403.
- Doctor session is not bound to a patient: return explicit "patient binding required" error.
- Patient projection missing: return 404 or a typed empty-state response; do not call the model to fill gaps.
- Feature flag disabled: return 404/disabled response for cockpit read model, or omit cockpit UI route.

DoctorActionTrace errors:

- Unknown `action_type`: reject request.
- Unknown `reason_code`: reject request.
- Missing target object: reject request.
- `edit` without `before_after`: reject request.
- Target refers to another patient: reject request.
- Storage conflict: preserve append-only semantics and return explicit failure.

## 14. Testing Strategy

Backend tests:

- `tests/backend/test_clinical_assertion_projection.py`
  - Projects CRC rectal bleeding, weight loss, disposition, missing information, and matched rules from a saved P0 assessment.
  - Produces stable assertion IDs across repeated projection.
  - Handles old records without P0 metadata.
  - De-dupes duplicate facts.

- `tests/backend/test_doctor_review_api.py`
  - Rejects patient sessions.
  - Requires a doctor session bound to a patient.
  - Returns timeline, assertions, draft sections, provenance, and available actions.
  - Marks unsupported draft text as `model_generated_unverified`.
  - Does not invoke model or network dependencies.

- `tests/backend/test_doctor_action_trace.py`
  - Accepts all allowed action types with valid targets.
  - Rejects unknown reason codes.
  - Rejects edit without before/after.
  - Stores append-only trace events.
  - Does not mutate `ClinicalSafetyPolicyVersion`, patient snapshot facts, or RAG indexes.

- Existing P0 regression tests remain required:
  - `tests/backend/test_clinical_safety_policy.py`
  - `tests/backend/test_crc_triage_flow.py`
  - `tests/backend/test_crc_triage_save.py`
  - `tests/backend/test_crc_harness_replay.py`

Frontend tests:

- `frontend/src/features/doctor/doctor-review-cockpit.test.tsx`
  - Renders timeline, assertion summary, provenance tags, and action buttons.
  - Hides cockpit when `doctor_review_cockpit_v0` is disabled.
  - Shows `model_generated_unverified` for unsupported draft sections.
  - Emits valid `DoctorActionTrace` payloads through `doctor-review-events.ts`.

- Existing doctor tests remain required:
  - `frontend/src/features/doctor/doctor-scene-shell.test.tsx`
  - `frontend/src/features/doctor/doctor-report-draft-view.test.tsx`
  - `frontend/src/features/doctor/doctor-database-view.test.tsx`

Suggested verification commands:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_review_api.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_action_trace.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py -q
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose
```

## 15. Release Gates

P1 is allowed to progress when:

- `ClinicalAssertion` projection is deterministic.
- Old patient records remain readable.
- Existing doctor flow passes its current tests with the cockpit disabled.
- Doctor Review Cockpit is feature-flagged and not default.
- Provenance gaps are explicitly marked unverified.
- `DoctorActionTrace` validates action type, target, and reason code.
- Doctor actions are append-only and do not mutate P0 safety policy or patient facts automatically.
- P0 harness/release report tests still pass.

P1 is blocked when:

- Doctor feedback automatically changes `ClinicalSafetyPolicyVersion`.
- Doctor feedback automatically enters training data.
- Unsupported model text is displayed as guideline or clinical fact.
- Patient sessions can access doctor review APIs.
- Existing doctor report draft or consultation flow regresses when the flag is disabled.
- Action traces contain hidden chain-of-thought, prompt secrets, API keys, or unrelated sensitive payloads.

## 16. Rollout Plan

1. Add `ClinicalAssertion` contract and projector tests.
2. Implement pure projection from existing patient records and P0 CRC assessment payloads.
3. Add optional assertion refs to backend read models.
4. Add doctor review read API with feature flag disabled by default.
5. Add cockpit component behind `doctor_review_cockpit_v0`.
6. Add `DoctorActionTrace` schema validation and append-only storage.
7. Wire cockpit action buttons to trace payload creation.
8. Run P1 backend/frontend tests and P0 regression tests.

## 17. Acceptance Criteria

P1 is complete when:

- `src/contracts/clinical_assertion.py` defines the canonical assertion contract.
- Patient record projection can produce assertion refs for P0 CRC assessment records.
- Old records without assertion refs still render and remain API-compatible.
- Doctor review API returns a read-only payload for a bound doctor session.
- Doctor Review Cockpit can display patient facts, draft sections, provenance, and action controls behind a feature flag.
- Every unverified draft section is labeled `model_generated_unverified`.
- `DoctorActionTrace` schema/API records accept, edit, reject, escalate, request evidence, and mark unsafe.
- `reason_code` enum is enforced.
- Trace storage is append-only and does not mutate safety policy, patient facts, prompts, RAG, or model training data.
- P0 safety loop tests still pass after P1 changes.

## 18. Implementation Boundaries

Implementation must preserve these boundaries:

- Do not edit `CRC-client`.
- Do not replace existing doctor flow.
- Do not make cockpit default during P1.
- Do not store doctor feedback as training data.
- Do not auto-update safety policy or prompt templates.
- Do not introduce literature evidence ingestion.
- Do not add broad auth infrastructure.
- Do not rely on model calls for assertion projection or review read model construction.
- Do not use frontend local state as the source of medical facts.

## 19. Spec Self-Review

Placeholder scan: no placeholder sections remain.

Internal consistency: the scope, architecture, contracts, data flow, tests, and acceptance criteria all target P1 Step 7-9.

Scope check: this spec is intentionally larger than P0 because it covers all P1, but it remains one coherent subsystem: clinical review loop. P1.5 literature evidence and P2 research workflows are explicitly excluded.

Ambiguity check: doctor feedback handling, provenance requirements, feature flag behavior, action trace validation, and safety boundaries are explicit.
