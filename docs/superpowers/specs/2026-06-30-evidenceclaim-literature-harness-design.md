# EvidenceClaim Literature Harness Design

> Version: 2026-06-30  
> Scope: P1.5 Step 10  
> Source plan: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`  
> Depends on: P0 `HarnessRun` / `ReleaseSafetyReport` and P1 `ClinicalAssertion` / `DoctorActionTrace`  
> Goal: Upgrade literature evidence handling from paper-level summaries to claim-level, auditable `EvidenceClaim` artifacts while keeping every unreviewed literature result out of patient advice, doctor default workflows, and the clinical RAG index.

## 1. Background

P0 established deterministic CRC safety policy evaluation, mutation replay, persistence traceability, and static release evidence. P1 established a doctor-side clinical review loop using deterministic `ClinicalAssertion` projection and append-only `DoctorActionTrace` events.

Step 10 should create the first literature evidence substrate for P1.5. Its job is not to make literature search clinically active. Its job is to convert literature candidates into structured, reviewable, claim-level evidence cards and run a deterministic shadow harness that proves unreviewed evidence remains isolated.

The product direction is:

```text
paper candidate
  -> claim-level extraction
    -> EvidenceClaim candidate
      -> EvidenceDelta / conflict report
        -> LiteratureHarnessRun
          -> Agent Admin / research read-only candidate view
```

No Step 10 output may become a patient recommendation, clinician guideline fact, prompt patch, RAG index chunk, or training signal.

## 2. Current Project Context

Relevant existing files:

- `src/contracts/harness.py`: P0 harness summary and release decision contract.
- `src/contracts/release_safety_report.py`: P0 release report contract.
- `src/contracts/clinical_assertion.py`: P1 clinical assertion contract.
- `src/rag/evidence.py`: existing RAG evidence normalization support.
- `src/services/web_search_service.py`: live web search service boundary.
- `src/tools/web_search_tools.py`: tool-facing search wrapper.
- `src/tools/manifest.py`: tool manifest exposed to admin/runtime surfaces.
- `reports/harness/*`: committed P0 replay evidence.
- `reports/release_safety/*`: committed P0 release evidence.
- `frontend/src/features/agent-admin/*`: read-only admin surface.

Missing Step 10 artifacts:

- `src/contracts/evidence_claim.py`
- `src/services/literature_harness.py`
- `tests/fixtures/literature_claim_pack_v0.json`
- `tests/backend/test_evidence_claim_contract.py`
- `tests/backend/test_literature_harness.py`
- `reports/literature/README.md`
- A generated shadow run JSON such as `reports/literature/literature_harness_20260630_001.json`
- Optional frontend research/admin type file for rendering candidate cards later.

At spec creation time the branch is `main`, and `origin/main` already contains P0 and P1. The local `CRC-client/` directory is ignored and outside this scope.

## 3. Design Decision

Use a shadow-only literature harness.

Step 10 creates contract, fixture, deterministic extraction helpers, conflict/delta reporting, and a replay script. It does not run live external search, does not mutate RAG stores, and does not surface candidates in patient or doctor default paths.

### Recommended Approach: Contract Plus Deterministic Harness

This approach creates a small `EvidenceClaim` contract and a local replay harness that reads fixed `PaperCandidate` fixtures. It outputs claim-level cards, negative/conflicting evidence, and isolation checks into a committed report.

Benefits:

- Gives Step 11 a stable dashboard input.
- Gives P2 a research evidence substrate later.
- Keeps unreviewed literature out of clinical workflows.
- Is testable without network access or model calls.
- Reuses P0's release-gate style without expanding runtime blast radius.

Trade-off:

- It does not provide a live literature search UI yet. That is deliberate; live search can feed the same contract after the shadow harness is stable.

### Rejected Approach: Live Literature Search Integration First

Directly wiring `web_search_service.py` into the harness would make tests depend on network, credentials, model behavior, and changing external pages. That would undermine the release-gate purpose of Step 10.

### Rejected Approach: Clinical RAG Ingest Preview In Step 10

Creating `IngestPreview` and RAG index update flows now would blur Step 10 with Step 11 and later release workflow. Step 10 may define the isolation boundary, but it must not ingest unreviewed claims into clinical RAG.

## 4. Scope

Step 10 includes:

1. `PaperCandidate`, `EvidenceClaim`, `EvidenceDelta`, and `LiteratureHarnessRun` contracts.
2. Deterministic claim ID generation.
3. Fixture-based literature claim pack.
4. Claim-level extraction from fixed candidates.
5. Preservation of negative and conflicting evidence.
6. A harness report showing candidate counts, conflict counts, isolation checks, and release decision.
7. Tests proving candidates remain shadow-only and do not enter clinical RAG or patient/doctor default data.
8. Optional TypeScript types for future read-only Agent Admin / research rendering.

Step 10 excludes:

- Live external web search execution.
- Model-based literature summarization.
- Clinical RAG ingestion.
- Patient-facing or doctor-default UI changes.
- Prompt, rubric, route, template, or policy patching.
- Automatic sign-off or evidence promotion.
- Research cohort feasibility.
- LearningJob candidate patch generation.
- Any edit to `CRC-client/`.

## 5. Architecture

```text
tests/fixtures/literature_claim_pack_v0.json
  -> src/contracts/evidence_claim.py
  -> src/services/literature_harness.py
  -> scripts/run_literature_harness.py
  -> reports/literature/literature_harness_*.json
  -> Step 11 Agent Admin release dashboard input
```

The harness is pure and local:

- Inputs are committed JSON fixtures.
- Outputs are deterministic JSON reports.
- IDs are content-addressed.
- Release decisions are derived from isolation and conflict checks.
- No network, model, database, or RAG index write is allowed.

## 6. EvidenceClaim Contract

Minimum `EvidenceClaim` object:

```json
{
  "claim_id": "claim_crc_0001_a1b2c3d4",
  "source_id": "paper_crc_2026_001",
  "claim_text": "Intervention X improved outcome Y in adults with colorectal cancer.",
  "population": "adults with colorectal cancer",
  "intervention": "Intervention X",
  "comparator": "standard of care",
  "outcome": "overall_survival",
  "effect_direction": "benefit",
  "effect_size": "HR 0.82",
  "uncertainty": "95% CI 0.70-0.96",
  "evidence_grade": "rct",
  "study_design": "randomized_controlled_trial",
  "sample_size": 820,
  "risk_of_bias": "moderate",
  "source_quality": {
    "is_guideline": false,
    "is_systematic_review": false,
    "is_preprint": false,
    "is_retracted": false
  },
  "local_guideline_conflict": "none",
  "applicability_to_crc_context": "partial",
  "source_span": {
    "page": 4,
    "section": "Results",
    "quote": "short extracted span"
  },
  "review_status": "candidate",
  "created_from": "literature_claim_pack_v0"
}
```

Required fields:

- `claim_id`
- `source_id`
- `claim_text`
- `population`
- `outcome`
- `effect_direction`
- `evidence_grade`
- `study_design`
- `risk_of_bias`
- `source_quality`
- `local_guideline_conflict`
- `applicability_to_crc_context`
- `source_span`
- `review_status`

Allowed `effect_direction` values:

- `benefit`
- `harm`
- `neutral`
- `inconclusive`
- `conflicting`

Allowed `evidence_grade` values:

- `guideline`
- `systematic_review`
- `rct`
- `observational`
- `case_series`
- `preclinical`
- `expert_opinion`
- `unknown`

Allowed `review_status` values:

- `candidate`
- `needs_review`
- `rejected`
- `approved_for_project_pool`
- `approved_for_clinical_rag`

Step 10 may only emit `candidate`, `needs_review`, or `rejected`. It must not emit `approved_for_project_pool` or `approved_for_clinical_rag`; those statuses require later human sign-off workflows.

## 7. PaperCandidate Contract

Minimum `PaperCandidate` object:

```json
{
  "source_id": "paper_crc_2026_001",
  "title": "Trial of Intervention X in metastatic colorectal cancer",
  "url": "https://example.org/paper_crc_2026_001",
  "publication_year": 2026,
  "venue": "Example Oncology Journal",
  "candidate_summary": "The paper reports improved overall survival with Intervention X.",
  "retrieval_query": "colorectal cancer intervention x overall survival",
  "retrieval_timestamp": "2026-06-30T00:00:00+08:00",
  "source_quality": {
    "is_guideline": false,
    "is_systematic_review": false,
    "is_preprint": false,
    "is_retracted": false
  },
  "extracted_claims": []
}
```

Rules:

- `PaperCandidate` represents an external-search candidate only.
- It is not clinical evidence.
- It may contain retrieval logs and candidate summaries.
- It must preserve URL/source metadata for traceability.
- It must not contain hidden chain-of-thought, credentials, prompt payloads, or unrelated patient data.

## 8. EvidenceDelta Contract

`EvidenceDelta` compares candidate claims against existing project evidence assumptions or against another candidate claim.

Minimum object:

```json
{
  "delta_id": "delta_claim_crc_0001_claim_crc_0002_a1b2c3d4",
  "claim_id": "claim_crc_0001_a1b2c3d4",
  "related_claim_id": "claim_crc_0002_b2c3d4e5",
  "delta_type": "conflict",
  "summary": "One candidate reports benefit while another reports no significant improvement.",
  "severity": "review_required",
  "recommended_action": "human_evidence_review"
}
```

Allowed `delta_type` values:

- `new_claim`
- `supporting`
- `conflict`
- `negative_evidence`
- `safety_signal`
- `retraction_or_quality_warning`

Allowed `severity` values:

- `info`
- `review_required`
- `block_promotion`

Rules:

- Negative or conflicting evidence must be preserved.
- Retractions and preprints must be flagged.
- Any `harm`, `conflicting`, `is_retracted`, or `block_promotion` signal prevents promotion beyond shadow candidate status.

## 9. Claim ID Rule

Claim IDs must be deterministic:

```text
claim_<source_id>_<outcome>_<stable_hash_8>
```

The stable hash is computed from:

- `source_id`
- normalized `claim_text`
- `population`
- `intervention`
- `comparator`
- `outcome`
- `effect_direction`
- `source_span`

The stable hash must not use current time, random UUID, network response order, model output metadata, or frontend state.

## 10. LiteratureHarnessRun

Minimum harness report:

```json
{
  "run_id": "literature_harness_20260630_001",
  "run_level": "L0_shadow",
  "claim_pack_version": "literature_claim_pack_v0",
  "evidence_index_version": "rag_crc_guideline_20260620",
  "summary": {
    "paper_candidates": 3,
    "claims": 5,
    "deltas": 2,
    "negative_or_conflicting_claims": 2,
    "isolation_violations": 0
  },
  "claims": [],
  "deltas": [],
  "isolation_checks": [
    {
      "check_id": "no_candidate_in_clinical_rag",
      "passed": true
    },
    {
      "check_id": "no_candidate_in_patient_or_doctor_default_path",
      "passed": true
    }
  ],
  "release_decision": "shadow_only"
}
```

Allowed `release_decision` values:

- `block`
- `shadow_only`
- `candidate_ready_for_human_review`

Step 10 should normally output `shadow_only`. It outputs `block` if isolation checks fail, a retracted source is not flagged, or negative/conflicting evidence is dropped.

## 11. Three-Zone Isolation

Step 10 must enforce this boundary:

| Zone | Allowed content | Forbidden behavior | Promotion condition |
|---|---|---|---|
| External literature search zone | `PaperCandidate`, unreviewed summary, URL, retrieval log | Used as patient or doctor advice | Source traceability and human initial review |
| Project Evidence Pool | Reviewed `EvidenceClaim`, `EvidenceDelta`, conflict report | Automatically shown as guideline fact | PI/doctor sign-off and conflict handling |
| Clinical RAG Index | Versioned, rollback-capable clinical evidence chunks | Any unapproved candidate claim | `IngestPreview` approval and passing harness |

Step 10 implements the first zone and the candidate output needed for later review. It must not implement promotion to the second or third zone.

## 12. Data Flow

```text
Load literature claim pack fixture
  -> validate PaperCandidate objects
  -> normalize extracted claims
  -> assign deterministic claim IDs
  -> classify source quality and evidence grade
  -> preserve negative and conflicting claims
  -> build EvidenceDelta items
  -> run isolation checks
  -> write LiteratureHarnessRun JSON
```

The flow must be deterministic for a fixed fixture. Running the harness twice with the same input should produce the same claims, deltas, counts, and release decision.

## 13. Error Handling

Fixture validation errors:

- Missing `source_id`: reject the candidate and include a harness validation error.
- Missing `claim_text`: reject the claim and include a validation error.
- Unknown enum value: reject the claim and include the field path.
- Invalid `sample_size`: reject non-positive or non-integer values when provided.

Evidence quality warnings:

- `is_retracted: true`: keep the claim, mark `review_status: rejected`, add `retraction_or_quality_warning`, and force `release_decision: block` unless the fixture explicitly expects rejection.
- `is_preprint: true`: keep the claim, add `review_required`.
- `risk_of_bias: high`: keep the claim, add `review_required`.

Isolation failures:

- Candidate appears in clinical RAG fixture or runtime index manifest: fail `no_candidate_in_clinical_rag`.
- Candidate appears in patient or doctor default payload fixture: fail `no_candidate_in_patient_or_doctor_default_path`.
- Negative/conflicting evidence count is lower than expected: fail `negative_evidence_preserved`.

## 14. Frontend / Admin Boundary

Step 10 may add TypeScript types or a simple read-only data mapper for future Agent Admin / research display. It should not add a full UI unless the implementation plan keeps it read-only and fixture-backed.

Allowed frontend artifacts:

- `frontend/src/features/research/evidence-claim-types.ts`
- `frontend/src/features/research/evidence-claim-types.test.ts`

Forbidden frontend behavior:

- Showing candidates in patient workspace.
- Showing candidates in default doctor consultation or review cockpit as clinical facts.
- Calling live search or backend model services from the browser.
- Providing approve/promote buttons.

## 15. Interaction With Web Search Tools

The source plan lists `src/tools/web_search_tools.py`, `src/services/web_search_service.py`, and `src/tools/manifest.py` as possible Step 10 files. For this Step 10 design, those files are not required for the initial deterministic harness.

If implementation touches them, changes must be limited to one of these safe actions:

- Add a pure mapper from a web-search result dictionary to `PaperCandidate`.
- Add manifest metadata that labels literature candidates as `research_shadow_only`.
- Add tests proving the existing clinical web search tool behavior is unchanged.

Forbidden changes:

- Running live web search inside the harness.
- Automatically turning web search summaries into clinical evidence.
- Adding a planner route that sends literature candidates into patient or doctor default flow.

## 16. Testing Strategy

Backend tests:

- `tests/backend/test_evidence_claim_contract.py`
  - Validates required fields.
  - Verifies deterministic claim IDs.
  - Rejects invalid enum values.
  - Preserves JSON-safe payloads.

- `tests/backend/test_literature_harness.py`
  - Loads `tests/fixtures/literature_claim_pack_v0.json`.
  - Produces claim-level `EvidenceClaim` cards.
  - Preserves negative/conflicting claims.
  - Produces `EvidenceDelta` conflict and safety-signal entries.
  - Emits `shadow_only` for clean candidate runs.
  - Emits `block` when isolation checks fail.

- Existing regression tests:
  - `tests/backend/test_clinical_assertion_projection.py`
  - `tests/backend/test_doctor_review_api.py`
  - `tests/backend/test_doctor_action_trace.py`
  - `tests/backend/test_clinical_safety_policy.py`
  - `tests/backend/test_crc_harness_replay.py`

Frontend type tests, if TypeScript research types are added:

- `frontend/src/features/research/evidence-claim-types.test.ts`
  - Ensures candidate status is not treated as approved clinical evidence.
  - Ensures `review_status: candidate` maps to read-only display metadata.

Suggested verification commands:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_harness_replay.py -q
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose
```

## 17. Release Gates

Step 10 is allowed to progress when:

- `EvidenceClaim` IDs are deterministic.
- Harness output is stable for the same fixture.
- Negative and conflicting evidence is preserved.
- Retracted/preprint/high-bias sources are flagged.
- Unreviewed claims remain `candidate`, `needs_review`, or `rejected`.
- No unreviewed claim enters clinical RAG, patient default payloads, or doctor default payloads.
- P0 and P1 regression tests still pass.

Step 10 is blocked when:

- Any candidate claim is marked `approved_for_clinical_rag`.
- A literature claim changes `ClinicalSafetyPolicyVersion`.
- A literature claim changes a prompt, rubric, route, template, or RAG index.
- A literature claim is shown as guideline fact in patient or doctor default UI.
- Negative, harmful, conflicting, or retracted evidence is dropped.
- Harness requires network access, model calls, or local browser state.

## 18. Acceptance Criteria

Step 10 is complete when:

- `src/contracts/evidence_claim.py` defines `PaperCandidate`, `EvidenceClaim`, `EvidenceDelta`, and `LiteratureHarnessRun`.
- `tests/fixtures/literature_claim_pack_v0.json` contains at least one benefit claim, one negative or neutral claim, and one conflicting or quality-warning claim.
- `src/services/literature_harness.py` converts fixture candidates into claim-level cards and deltas.
- `scripts/run_literature_harness.py` writes a deterministic report under `reports/literature/`.
- The report includes isolation checks and a release decision.
- Backend tests prove unreviewed claims remain out of clinical RAG and default patient/doctor paths.
- Existing P0/P1 tests still pass.

## 19. Implementation Boundaries

Implementation must preserve these boundaries:

- Do not edit `CRC-client/`.
- Do not add live search to tests or harness.
- Do not add model calls.
- Do not write to `chroma_db/`, `bm25_index/`, or any RAG index path.
- Do not edit `config/safety_policy.yaml`.
- Do not edit patient or doctor default UI to show literature candidates.
- Do not create approve/promote controls.
- Do not store hidden chain-of-thought, prompt secrets, API keys, or patient-level research exports.

## 20. Spec Self-Review

Placeholder scan: no placeholder sections remain.

Internal consistency: scope, contracts, harness flow, isolation rules, tests, gates, and acceptance criteria all target P1.5 Step 10 only.

Scope check: this is one coherent subsystem: deterministic claim-level literature evidence harness. Agent Admin release dashboard is left to Step 11, and research cohort feasibility is left to P2 Step 12.

Ambiguity check: unreviewed literature cannot enter clinical RAG, patient default paths, doctor default paths, prompt/rubric/route/template patches, or training data. Promotion statuses are explicitly forbidden in Step 10.
