# LearningJob Candidate Pipeline Design

> Version: 2026-07-08
> Scope: P2 Step 13
> Source plan: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`
> Depends on: Step 11 release dashboard, Step 12 cohort feasibility / ethics gate, P1 `DoctorActionTrace`, P1.5 `EvidenceDelta`, P0/P1.5 harness reports, and controlled release governance/execution/monitoring surfaces.
> Goal: Add a shadow-only `LearningJob` pipeline that turns reviewed signals into candidate prompt, rubric, route, template, or evidence-ingest patches without automatically changing production behavior, training data, safety policy, or clinical RAG.

## 1. Background

P0 created deterministic CRC safety and release evidence. P1 created doctor review and append-only feedback traces. P1.5 created claim-level literature deltas and an admin release dashboard. Later release-governance modules add audited intent, approval, execution, monitoring, and closure boundaries.

Step 13 should make improvement candidates visible without making them active. Its job is to collect safe improvement signals and produce candidate patches that can be reviewed, tested through harnesses, and routed through human release governance later.

It must not automatically learn from doctor edits, literature candidates, failed harness cases, or cohort feasibility gaps. It must not update prompts, rubrics, routes, templates, RAG indexes, safety policy, feature flags, or model training data.

## 2. Current Project Context

Relevant existing files:

- `backend/api/schemas/doctor_action_trace.py`: doctor action types and reason codes.
- `backend/api/routes/doctor_review.py`: append-only doctor action recording.
- `src/contracts/evidence_claim.py`: `EvidenceDelta` and literature evidence status.
- `src/contracts/harness.py`: P0 harness run contract.
- `src/contracts/release_safety_report.py`: release safety report contract.
- `src/services/literature_harness.py`: literature shadow harness.
- `src/services/release_governance.py`: release intent and approval service.
- `src/services/release_execution.py`: audited feature flag execution service.
- `src/services/release_monitoring.py`: post-release monitoring service.
- `src/services/release_closure.py`: post-release closure service.
- `backend/api/routes/admin.py`: release governance/execution/monitoring/closure endpoints.
- `docs/superpowers/specs/2026-07-08-crc-cohort-feasibility-design.md`: Step 12 design.

Missing Step 13 artifacts:

- `src/contracts/learning_job.py`
- `src/services/learning_job_service.py`
- `backend/api/routes/learning_jobs.py`
- `reports/learning_jobs/README.md`
- `tests/backend/test_learning_job_contract.py`
- `tests/backend/test_learning_job_service.py`
- `tests/backend/test_learning_jobs_api.py`
- Optional Agent Admin learning-job page.

## 3. Design Decision

Use a **candidate-only LearningJob pipeline**.

The pipeline reads approved or reviewable signals, creates candidate patches, links each candidate to required harness evidence and human review, and stores immutable job artifacts. It never applies the patch.

### Recommended Approach: Signal Ledger To Candidate Patch

Create a small contract and service that accept explicit source signals:

- `DoctorActionTrace` reason codes;
- `EvidenceDelta` conflict or safety signals;
- failed `HarnessRun` cases;
- Step 12 missing-variable or low-coverage findings;
- release monitoring alerts.

The service normalizes signals, groups them by target area, creates candidate patch payloads, and returns `LearningJob` records with `shadow_only` status.

Benefits:

- Keeps automated improvement separate from runtime behavior.
- Creates auditable candidate artifacts for release governance.
- Makes repeated failure patterns visible.
- Supports future automation without bypassing human review.

Trade-off:

- It does not improve the live system by itself. That is intentional; release governance must decide whether to promote a candidate.

### Rejected Approach: Automatic Prompt Or RAG Update

Directly editing prompt, rubric, route, template, or RAG files from signals would bypass the harness and human-review gates. Step 13 must not write active runtime artifacts.

### Rejected Approach: Training Dataset Builder

Doctor edits and feedback are not automatically ground truth. Step 13 may store deidentified candidate signals, but it must not produce SFT, DPO, or evaluation training sets.

## 4. Scope

Step 13 includes:

1. `LearningJob` contract.
2. `LearningSignal` contract for normalized input signals.
3. `CandidatePromptPatch`, `CandidateRubricPatch`, `CandidateRoutePatch`, `CandidateTemplatePatch`, and `CandidateEvidenceIngest` contracts.
4. Deterministic candidate ID generation.
5. Candidate status machine from `draft` to `ready_for_harness` to `awaiting_human_review`.
6. Harness requirement mapping for each candidate type.
7. Human review and release governance references.
8. Storage of shadow job artifacts under `reports/learning_jobs/`.
9. Backend tests proving no runtime files are modified.
10. Optional read-only Agent Admin surface for candidate jobs.

Step 13 excludes:

- Applying patches.
- Editing prompts, rubrics, routes, templates, safety policy, or RAG indexes.
- Feature flag release.
- Automatic model training.
- Automatic SFT/DPO/eval dataset creation.
- Automatic conversion of doctor feedback to truth.
- Automatic promotion of literature claims.
- Patient or doctor default UI changes.
- Any edit to `CRC-client/`.

## 5. Architecture

```text
DoctorActionTrace / EvidenceDelta / HarnessRun failure / Step 12 feasibility gap
  -> LearningSignal normalization
    -> LearningJobService
      -> CandidatePatch objects
      -> required HarnessRun references
      -> human review requirements
      -> reports/learning_jobs/*.json
        -> read-only Agent Admin learning candidates
          -> later release governance intent
```

All writes are artifact writes to a learning-job report store. No active runtime artifact is changed.

## 6. LearningSignal Contract

Minimum object:

```json
{
  "signal_id": "learning_signal_doctor_trace_001",
  "signal_type": "doctor_action_trace",
  "source_ref": {
    "kind": "doctor_action_trace",
    "id": "doctor_trace_abc123"
  },
  "reason_code": "unsafe_disposition",
  "target_area": "prompt",
  "severity": "review_required",
  "summary": "Doctor marked risk summary unsafe because disposition was too low.",
  "deidentified": true,
  "created_at": "2026-07-08T00:00:00+08:00"
}
```

Allowed `signal_type` values:

- `doctor_action_trace`
- `evidence_delta`
- `harness_failure`
- `cohort_feasibility_gap`
- `release_monitoring_alert`

Allowed `target_area` values:

- `prompt`
- `rubric`
- `route`
- `template`
- `evidence_ingest`
- `test_case`

Rules:

- Signals must be deidentified.
- Signals must keep source references.
- Signals must not include hidden chain-of-thought, API keys, prompts with secrets, or patient-level rows.
- Signals are not training data.

## 7. LearningJob Contract

Minimum object:

```json
{
  "job_id": "learning_job_crc_20260708_001",
  "job_type": "candidate_patch_generation",
  "status": "shadow_only",
  "created_at": "2026-07-08T00:00:00+08:00",
  "source_signal_ids": [
    "learning_signal_doctor_trace_001"
  ],
  "candidate_patch_ids": [
    "candidate_prompt_patch_crc_001"
  ],
  "required_harness": {
    "case_pack_version": "crc_mutation_pack_v0",
    "required_levels": ["L0_L1"],
    "hard_fail_policy": "block_on_any_hard_fail"
  },
  "human_review": {
    "required": true,
    "required_roles": [
      "clinical_safety_reviewer",
      "release_manager"
    ],
    "status": "pending"
  },
  "release_governance_ref": null
}
```

Allowed `job_type` values:

- `candidate_patch_generation`
- `candidate_evidence_ingest`
- `candidate_test_case_generation`

Allowed `status` values:

- `draft`
- `shadow_only`
- `ready_for_harness`
- `harness_failed`
- `awaiting_human_review`
- `rejected`
- `approved_for_release_intent`
- `archived`

Step 13 must not emit `applied`, `released`, `trained`, or `clinical_rag_active`.

## 8. Candidate Patch Contracts

### CandidatePromptPatch

```json
{
  "patch_id": "candidate_prompt_patch_crc_001",
  "patch_type": "prompt",
  "target_ref": {
    "kind": "prompt",
    "id": "assessment_prompt_crc_triage"
  },
  "change_summary": "Add explicit instruction not to down-rank rectal bleeding in older patients.",
  "proposed_diff": {
    "format": "unified_diff",
    "content": "--- current\n+++ candidate\n@@ ..."
  },
  "source_signal_ids": ["learning_signal_doctor_trace_001"],
  "status": "candidate",
  "applies_automatically": false
}
```

### CandidateRubricPatch

Targets evaluator or judge rubrics. It may propose scoring rules, but it must not change active evaluation files.

### CandidateRoutePatch

Targets routing rules. It must never bypass `ClinicalSafetyPolicyVersion`.

### CandidateTemplatePatch

Targets patient or doctor wording templates. It must keep intended-use disclaimers.

### CandidateEvidenceIngest

Targets future evidence review. It may reference reviewed `EvidenceClaim` IDs, but it must not ingest into clinical RAG in Step 13.

Shared rules:

- `applies_automatically` must be `false`.
- `status` must be `candidate`, `needs_harness`, `needs_human_review`, `rejected`, or `approved_for_release_intent`.
- Candidate patches must be content-addressed or deterministically identified.
- Candidate patches must reference source signals.

## 9. Signal To Patch Mapping

| Signal | Candidate target | Required gate |
|---|---|---|
| `DoctorActionTrace.reason_code == unsafe_disposition` | prompt, rubric, or test case | CRC mutation harness and clinical safety review |
| `DoctorActionTrace.reason_code == citation_not_traceable` | template or evidence-ingest candidate | evidence review and provenance check |
| `EvidenceDelta.delta_type == conflict` | evidence-ingest candidate or review queue | evidence reviewer approval |
| `EvidenceDelta.delta_type == safety_signal` | rubric or test case | clinical safety review |
| `HarnessRun` hard fail | prompt/rubric/route/test case candidate | harness replay must pass before release intent |
| Step 12 missing variable | template or data capture candidate | research ethics/data governance review |
| release monitoring alert | route/template/test case candidate | monitoring review and rollback analysis |

The service can create zero candidates when signals are weak, duplicate, or outside scope.

## 10. Harness Requirements

Every candidate patch must declare required verification before release intent:

- Prompt patch: P0 CRC mutation pack plus affected workflow tests.
- Rubric patch: evaluator/rubric tests plus P0 hard-fail replay.
- Route patch: routing tests plus safety policy non-regression.
- Template patch: intended-use copy tests and no clinical advice regression.
- Evidence ingest candidate: literature harness plus isolation checks.
- Test case candidate: harness dry run proving the new case fails before patch and passes after a reviewed patch.

Step 13 may only attach required harness metadata. It does not need to run all harnesses automatically in the first implementation.

## 11. Human Review And Release Governance

Human review is mandatory before any candidate can become a release intent.

Rules:

- Clinical safety reviewer required for prompt, route, rubric, and safety-related test cases.
- Evidence reviewer required for evidence-ingest candidates.
- Release manager required before `approved_for_release_intent`.
- Governance intent must be created through existing release governance APIs, not by mutating feature flags.
- Release execution remains a later controlled action.

## 12. Storage Boundary

LearningJob artifacts may be written under:

```text
reports/learning_jobs/
  README.md
  jobs/<job_id>.json
  candidates/<patch_id>.json
```

Rules:

- Writes must be append-only.
- Existing job artifacts must not be overwritten.
- Artifacts must not include patient-level rows.
- Artifacts must not include hidden chain-of-thought, prompt secrets, API keys, or credentials.
- Artifacts must be safe for code review.

## 13. API Boundary

Optional Step 13 API:

```text
GET /api/admin/learning-jobs
POST /api/admin/learning-jobs
POST /api/admin/learning-jobs/{job_id}/archive
```

The API should be admin-protected. In the first implementation it may support:

- reading current jobs;
- creating a shadow job from explicit source signal IDs;
- archiving a job.

It must not apply a candidate patch, run release execution, write active config, or ingest evidence into clinical RAG.

## 14. Frontend Boundary

Allowed UI:

- Read-only Agent Admin learning-job list.
- Candidate detail view.
- Source signal summary.
- Required harness and human review checklist.
- Disabled "Apply" and "Release" actions with reasons.

Forbidden UI:

- Active patch editor that writes runtime files.
- One-click publish.
- Training dataset export.
- Patient-level row display.
- Clinical RAG ingest button.

## 15. Error Handling

Validation errors:

- Unknown source signal: reject job creation.
- Non-deidentified signal: reject job creation.
- Unsupported target area: reject candidate.
- Candidate has `applies_automatically: true`: reject candidate.
- Candidate attempts to target `ClinicalSafetyPolicyVersion`: reject candidate unless a future explicit safety-policy workflow is approved.
- Candidate includes patient-level rows: reject candidate.

Runtime errors:

- Report store unavailable: return explicit storage error.
- Duplicate job ID: reject write.
- Malformed existing job artifact: mark store integrity warning and do not load that artifact as active.

## 16. Testing Strategy

Backend tests:

- `tests/backend/test_learning_job_contract.py`
  - validates enums and required fields;
  - rejects active/applied statuses;
  - rejects non-deidentified signals;
  - rejects automatic patch application.
- `tests/backend/test_learning_job_service.py`
  - normalizes doctor action, evidence delta, harness failure, and cohort gap signals;
  - creates candidate patches with source refs;
  - writes append-only artifacts;
  - does not modify runtime prompt/rubric/route/template/RAG files;
  - requires harness and human review metadata.
- `tests/backend/test_learning_jobs_api.py`
  - verifies admin auth;
  - creates shadow jobs only;
  - rejects apply/release behavior.

Regression tests:

- `tests/backend/test_doctor_action_trace.py`
- `tests/backend/test_literature_harness.py`
- `tests/backend/test_crc_harness_replay.py`
- `tests/backend/test_release_governance_service.py`
- `tests/backend/test_release_execution_service.py`

Frontend tests, if UI is added:

- renders learning jobs read-only;
- shows source signals and required gates;
- disabled apply/release controls are non-interactive;
- no patient-level data appears.

Suggested verification commands:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_contract.py tests/backend/test_learning_job_service.py tests/backend/test_learning_jobs_api.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_doctor_action_trace.py tests/backend/test_literature_harness.py tests/backend/test_crc_harness_replay.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py -q
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose
```

## 17. Release Gates

Step 13 is allowed to progress when:

- Learning jobs are shadow-only.
- Candidate patches do not modify runtime files.
- All candidates require harness evidence and human review.
- Source signals are deidentified and traceable.
- Release governance is referenced but not bypassed.
- Tests prove no prompt/rubric/route/template/RAG/policy file is changed.

Step 13 is blocked when:

- Any candidate applies automatically.
- Any job writes active runtime config.
- Any candidate changes `ClinicalSafetyPolicyVersion`.
- Doctor feedback becomes training data or patient truth automatically.
- Unreviewed literature is promoted to clinical RAG.
- A feature flag is toggled without release governance.
- Patient-level rows are stored in learning artifacts.

## 18. Acceptance Criteria

Step 13 is complete when:

1. `src/contracts/learning_job.py` defines learning signals, jobs, and candidate patch contracts.
2. `src/services/learning_job_service.py` creates candidate-only jobs from explicit source signals.
3. Learning job artifacts are append-only under `reports/learning_jobs/`.
4. Each candidate patch declares required harness and human review gates.
5. Optional admin API can create and read shadow jobs without applying them.
6. Tests prove active prompt, rubric, route, template, RAG, policy, and feature-flag files are unchanged.
7. Existing P0, P1, P1.5, Step 11, Step 12, and release-governance tests still pass.

## 19. Implementation Notes For The Next Plan

The implementation plan should likely split work into:

1. Learning contract dataclasses and tests.
2. Signal normalizers from doctor action, evidence delta, harness failure, and cohort gap inputs.
3. Candidate patch builders.
4. Append-only learning-job store.
5. Optional admin API.
6. Optional read-only Agent Admin page.
7. Regression verification.

## 20. Self-Review

Placeholder scan: no placeholder sections remain.

Internal consistency: the pipeline creates candidate patches only, attaches harness and human review gates, and keeps active runtime artifacts unchanged.

Scope check: this is one coherent Step 13 subsystem. It depends on Step 12 and release governance but does not implement release execution, training, or automatic policy updates.

Ambiguity check: automatic application, training data creation, clinical RAG ingest, safety-policy mutation, feature-flag release, and patient-level row storage are explicitly forbidden.
