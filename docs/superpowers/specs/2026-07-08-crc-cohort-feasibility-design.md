# CRC Cohort Feasibility Design

> Version: 2026-07-08
> Scope: P2 Step 12
> Source plan: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`
> Depends on: P0 `ClinicalSafetyPolicyVersion` / `HarnessRun` / `ReleaseSafetyReport`, P1 `ClinicalAssertion` / `DoctorActionTrace`, P1.5 `EvidenceClaim` / `LiteratureHarnessRun`, and Step 11 Agent Admin release dashboard.
> Goal: Add a shadow-only research cohort feasibility loop that estimates whether CRC research questions have enough traceable patient-record variables, coverage, and governance readiness without exporting patient-level data or producing clinical recommendations.

## 1. Background

P0 established the patient-facing CRC safety loop. P1 established traceable clinical review through `ClinicalAssertion` and append-only `DoctorActionTrace`. P1.5 established claim-level literature evidence and a release dashboard that keeps unreviewed literature isolated from patient and doctor default paths.

Step 12 starts the AI4Science path conservatively. It should not generate research conclusions, papers, protocols, treatment advice, or patient-level datasets. It should answer a narrower operational question:

```text
Given a research question and a small set of required variables,
does the current patient-record projection contain enough traceable,
reviewable, and ethically gated data to justify later research review?
```

The output is a feasibility candidate for a research workspace or Agent Admin panel. It is not a clinical fact, a diagnosis, a cohort export, or a publication-ready analysis.

## 2. Current Project Context

Relevant existing files:

- `config/intended_use_profiles.yaml`: includes `research_workspace` boundaries.
- `src/contracts/clinical_assertion.py`: canonical assertion contract.
- `src/services/clinical_assertion_projection.py`: deterministic assertion projection from patient records.
- `backend/api/services/patient_registry_service.py`: patient and record storage boundary.
- `backend/api/services/patient_commands.py`: append-only patient event write boundary.
- `src/contracts/evidence_claim.py`: claim-level literature candidate contract.
- `src/services/literature_harness.py`: shadow literature evidence harness.
- `backend/api/services/admin_release_dashboard.py`: read-only release dashboard normalizer.
- `frontend/src/features/agent-admin/*`: existing admin surface.
- `frontend/src/app/api/types.ts`: typed frontend API contracts.

Missing Step 12 artifacts:

- `src/contracts/research_asset.py`
- `src/services/cohort_feasibility_service.py`
- `backend/api/routes/research.py`
- `tests/backend/test_research_asset_contract.py`
- `tests/backend/test_cohort_feasibility_service.py`
- `tests/backend/test_research_api.py`
- Optional read-only frontend research or Agent Admin surface for feasibility cards.

## 3. Design Decision

Use a **shadow-only, aggregate-only cohort feasibility service**.

The service reads already-projected patient records and `ClinicalAssertion` data, computes aggregate counts and variable coverage, emits governance review items, and returns a feasibility result. It does not export row-level patient data, call models, call external services, change patient records, update RAG, or modify clinical workflows.

### Recommended Approach: Deterministic Aggregate Feasibility

The first implementation should be a deterministic Python service with small contracts. It should accept a `CohortFeasibilityRequest`, read a bounded in-memory or registry-backed record projection, map required variables to `ClinicalAssertion` facts, and return a `CohortFeasibilityResult`.

Benefits:

- Reuses P1 assertion provenance instead of creating a new research truth source.
- Keeps patient-level data inside the registry boundary.
- Produces reviewable aggregate metrics for research planning.
- Provides a safe input for Step 13 learning jobs.
- Is testable without network, model calls, or live browser state.

Trade-off:

- It cannot answer complex epidemiology questions yet. It only estimates availability, coverage, and governance readiness.

### Rejected Approach: Direct Dataset Export First

Exporting patient-level rows early would cross the ethics and governance boundary before the system has review queues, de-identification policy, and dataset hashing. Step 12 must not add export buttons, CSV downloads, or dataset materialization.

### Rejected Approach: LLM Research Analyst First

Generating hypotheses, protocols, or literature-backed conclusions before feasibility and ethics gates would bypass the safety order in the source plan. Step 12 may store a research question, but it must not present conclusions as science.

## 4. Scope

Step 12 includes:

1. `ResearchAsset`, `CohortFeasibilityRequest`, `CohortFeasibilityResult`, `VariableCoverage`, and `ReviewQueueItem` contracts.
2. Deterministic variable mapping from required features to `ClinicalAssertion.normalized_fact`.
3. Aggregate estimated count and variable coverage.
4. Missing key variable detection.
5. Bias and data-quality warnings for low coverage, conflicting assertions, and unreviewed clinical facts.
6. Ethics review queue item generation when patient-level data is used to estimate feasibility.
7. Read-only API for feasibility preview.
8. Backend tests proving aggregate-only output and no patient-level export.
9. Optional read-only frontend display in Agent Admin or a research workspace.

Step 12 excludes:

- Patient-level dataset export.
- Dataset version materialization.
- Dataset hash generation for exported data.
- Live model analysis.
- Publication, protocol, grant, or patent drafting.
- Clinical recommendations.
- Patient or doctor default UI changes.
- Clinical RAG ingestion.
- EvidenceClaim promotion.
- LearningJob candidate patch generation.
- Feature-flag release execution.
- Any edit to `CRC-client/`.

## 5. Architecture

```text
patient records / patient snapshots
  -> ClinicalAssertion projection
    -> CohortFeasibilityRequest
      -> CohortFeasibilityService
        -> CohortFeasibilityResult
        -> ReviewQueueItem(research_ethics_review)
          -> read-only research/admin surface
```

The service must use existing projections as its read model. It must not query session memory directly. It must not introduce a second patient-record source.

## 6. ResearchAsset Contract

`ResearchAsset` represents a research workspace object. In Step 12 it is metadata only.

Minimum object:

```json
{
  "asset_id": "research_asset_crc_20260708_001",
  "asset_type": "cohort_feasibility",
  "title": "CRC triage risk cohort feasibility",
  "status": "candidate",
  "created_by": "research_workspace",
  "created_at": "2026-07-08T00:00:00+08:00",
  "source_refs": [
    {
      "kind": "clinical_assertion_projection",
      "id": "patient_record_projection_v0"
    }
  ],
  "governance_refs": [
    "review_queue_research_ethics_001"
  ]
}
```

Allowed `asset_type` values for Step 12:

- `cohort_feasibility`
- `ethics_review_item`

Allowed `status` values:

- `candidate`
- `needs_review`
- `blocked`
- `reviewed`

Step 12 must not emit `approved_dataset`, `published`, or `clinical_rag_ready` statuses.

## 7. CohortFeasibilityRequest Contract

Minimum object:

```json
{
  "request_id": "cohort_request_crc_001",
  "project_id": "research_crc_001",
  "question": "Is there enough structured CRC triage data to study rectal bleeding risk escalation?",
  "cohort_criteria": {
    "condition": "colorectal_cancer_or_crc_triage_risk",
    "age_min": 50,
    "required_features": [
      "rectal_bleeding",
      "colonoscopy_status",
      "pathology_result"
    ]
  },
  "data_scope": {
    "source": "patient_record_projection",
    "patient_level_export_requested": false,
    "deidentified_only": true
  },
  "version_refs": {
    "projection_version": "patient_record_projection_v0",
    "clinical_safety_policy_version": "crc_safety_policy_v0",
    "evidence_index_version": "rag_crc_guideline_20260620"
  }
}
```

Rules:

- `required_features` must be explicit.
- `patient_level_export_requested` must be `false` in Step 12.
- `data_scope.source` must be `patient_record_projection`.
- The request cannot include raw patient identifiers.
- The request cannot include hidden prompts or model-generated reasoning.

## 8. Variable Mapping

Feature names map to assertion facts:

| Required feature | Matching assertion examples |
|---|---|
| `rectal_bleeding` | `normalized_fact.type == condition_signal` and `name == rectal_bleeding` |
| `weight_loss` | `condition_signal.weight_loss` or `symptom.weight_loss` |
| `disposition` | `risk_disposition.disposition` |
| `matched_safety_rule` | `safety_rule_match.*` |
| `colonoscopy_status` | `test_status.colonoscopy_status` or uploaded record facts when available |
| `pathology_result` | `document_fact.pathology_result` when projected from uploads |

If a feature has no mapper, the service should include it in `unmapped_required_features` and treat coverage as `0`.

## 9. CohortFeasibilityResult Contract

Minimum object:

```json
{
  "result_id": "cohort_feasibility_crc_001",
  "request_id": "cohort_request_crc_001",
  "project_id": "research_crc_001",
  "status": "needs_review",
  "estimated_count": 42,
  "variable_coverage": {
    "rectal_bleeding": {
      "covered_count": 39,
      "coverage_ratio": 0.92,
      "source_fact_types": ["condition_signal"],
      "reviewed_status_mix": {
        "unreviewed": 39
      }
    },
    "colonoscopy_status": {
      "covered_count": 24,
      "coverage_ratio": 0.58,
      "source_fact_types": ["test_status"],
      "reviewed_status_mix": {
        "unreviewed": 24
      }
    },
    "pathology_result": {
      "covered_count": 13,
      "coverage_ratio": 0.31,
      "source_fact_types": ["document_fact"],
      "reviewed_status_mix": {
        "unreviewed": 13
      }
    }
  },
  "missing_key_variables": ["pathology_result"],
  "unmapped_required_features": [],
  "bias_warnings": [
    "pathology_result coverage is below 0.5"
  ],
  "requires_review": true,
  "review_queue_items": ["review_queue_research_ethics_001"],
  "patient_level_rows_returned": false
}
```

Allowed `status` values:

- `feasible_for_review`
- `needs_review`
- `insufficient_data`
- `blocked_by_governance`

Step 12 should default to `needs_review` whenever patient-level source data was inspected, even if only aggregate values are returned.

## 10. ReviewQueueItem Contract

Minimum object:

```json
{
  "review_item_id": "review_queue_research_ethics_001",
  "review_type": "research_ethics_review",
  "status": "pending",
  "trigger": "patient_level_data_used_for_cohort_feasibility",
  "scope": {
    "project_id": "research_crc_001",
    "request_id": "cohort_request_crc_001",
    "data_minimization": "aggregate_only",
    "patient_level_export_requested": false
  },
  "required_checks": [
    "authorization_basis",
    "deidentification_strategy",
    "data_minimization",
    "irb_or_local_policy_need"
  ]
}
```

Allowed `review_type` values:

- `research_ethics_review`
- `pi_review`
- `data_governance_review`

Step 12 only creates review items. It does not approve them.

## 11. Data Flow

```text
Receive feasibility request
  -> validate intended-use profile is research_workspace
  -> load patient records through registry/service boundary
  -> project or read ClinicalAssertion objects
  -> map requested features to assertions
  -> compute aggregate count and variable coverage
  -> detect missing and unmapped features
  -> create research ethics review item
  -> return aggregate CohortFeasibilityResult
```

The service should be deterministic for the same input records and request.

## 12. Ethics And Governance Gate

Step 12 must create a `research_ethics_review` item when feasibility uses patient-level records or assertions, even when the response is aggregate-only.

Gate rules:

- If `patient_level_export_requested` is `true`, return `blocked_by_governance`.
- If no ethics review item can be created, return `blocked_by_governance`.
- If any required feature coverage is below a configured threshold, return `needs_review` or `insufficient_data`.
- If patient-level rows would be returned by any code path, fail the request.
- If the request asks for clinical advice or treatment recommendation, reject it as outside `research_workspace`.

## 13. API Boundary

Optional Step 12 API:

```text
POST /api/research/cohort-feasibility
```

The route should:

- require a research/admin context when auth is enabled;
- call a deterministic service;
- return aggregate feasibility only;
- not write patient records;
- not export files;
- not call models or external services.

The first implementation may keep the API admin-only if a research workspace auth model does not exist yet.

## 14. Frontend Boundary

Allowed UI:

- Read-only Agent Admin or Research Workspace panel.
- Feasibility result cards.
- Variable coverage table.
- Ethics review pending indicator.
- Missing variable list.

Forbidden UI:

- Patient-level table.
- CSV or dataset export.
- Approve ethics review.
- Publish or release controls.
- Clinical recommendation wording.
- Any display in patient or doctor default workflows.

## 15. Error Handling

Validation errors:

- Missing `project_id`: reject request.
- Empty `required_features`: reject request.
- Unknown feature mapper: include in `unmapped_required_features`, do not crash.
- Invalid age or criterion values: reject request.
- Request includes patient IDs: reject request.
- Request includes `patient_level_export_requested: true`: return `blocked_by_governance`.

Runtime errors:

- Patient registry unavailable: return an explicit unavailable error.
- Projection failure for one record: skip that record and include a data-quality warning.
- No matching records: return `insufficient_data` with zero coverage.

## 16. Testing Strategy

Backend tests:

- `tests/backend/test_research_asset_contract.py`
  - validates contract enums and JSON-safe serialization;
  - rejects clinical-RAG-ready or dataset-approved statuses in Step 12.
- `tests/backend/test_cohort_feasibility_service.py`
  - computes aggregate estimated count;
  - computes per-feature coverage;
  - flags missing and unmapped features;
  - creates ethics review item;
  - returns no patient-level rows;
  - blocks export requests.
- `tests/backend/test_research_api.py`
  - validates route response shape;
  - rejects non-research requests;
  - proves API is read-only for patient records.

Regression tests:

- `tests/backend/test_clinical_assertion_projection.py`
- `tests/backend/test_evidence_claim_contract.py`
- `tests/backend/test_admin_release_dashboard.py`

Frontend tests, if UI is added:

- renders aggregate coverage only;
- hides patient-level rows;
- shows ethics review pending state;
- does not show export or approve buttons.

Suggested verification commands:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_asset_contract.py tests/backend/test_cohort_feasibility_service.py tests/backend/test_research_api.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_evidence_claim_contract.py tests/backend/test_admin_release_dashboard.py -q
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose
```

## 17. Release Gates

Step 12 is allowed to progress when:

- Feasibility output is aggregate-only.
- Every patient-level source use creates a review queue item.
- The service reads patient records through existing registry/projection boundaries.
- No patient, doctor, RAG, prompt, policy, or release state is mutated.
- Low-coverage and unmapped variables are explicit.
- Tests prove no patient-level rows are returned.

Step 12 is blocked when:

- Any endpoint returns patient-level rows.
- Any feature writes dataset exports.
- Any result is presented as a clinical recommendation.
- Any unreviewed `EvidenceClaim` becomes clinical evidence.
- The service reads session memory directly.
- The workflow bypasses ethics or data-governance review.

## 18. Acceptance Criteria

Step 12 is complete when:

1. `src/contracts/research_asset.py` defines Step 12 research and feasibility contracts.
2. `src/services/cohort_feasibility_service.py` computes deterministic aggregate feasibility from patient record projections and clinical assertions.
3. The service emits `research_ethics_review` queue items.
4. Optional API returns aggregate-only feasibility results.
5. Tests prove patient-level data is not exported.
6. Tests prove governance blocks export requests.
7. Existing P0, P1, P1.5, and Step 11 regression tests still pass.

## 19. Implementation Notes For The Next Plan

The implementation plan should likely split work into:

1. Research contract dataclasses and tests.
2. Feature-to-assertion mapper and tests.
3. Cohort feasibility aggregate service and fixtures.
4. Review queue item creation.
5. Optional API endpoint.
6. Optional read-only frontend panel.
7. Regression verification.

## 20. Self-Review

Placeholder scan: no placeholder sections remain.

Internal consistency: the design uses existing patient record and assertion projections, returns aggregate-only feasibility, and treats ethics review as required before any export.

Scope check: this is one coherent Step 12 subsystem. It does not implement LearningJob, dataset export, publication drafting, or clinical recommendations.

Ambiguity check: patient-level rows, dataset export, clinical advice, RAG ingest, and automatic approval are explicitly forbidden.
