# P0/P1/P1.5 To P2 Readiness Handoff

## Purpose

This handoff closes the CRC P0, P1, and P1.5 work as a substrate for later P2 research workflows. It records the completed artifacts, verification evidence, integration state, and boundaries that P2 must preserve.

P2 should start from a clean, merged `main`, not from an unmerged worktree branch.

## Current Integration State

- Main workspace: clean at `main` / `origin/main` commit `6c8c834`.
- Step 10 worktree: clean at branch `step10-evidenceclaim-literature-harness`, commit `89518da`.
- Step 10 merge base with current `main`: `47df20a`.
- Current `main` contains a later README-only documentation commit that Step 10 did not touch.
- `git merge-tree` did not show conflict markers between current `main` and the Step 10 branch.

Recommended before P2:

1. Integrate `step10-evidenceclaim-literature-harness` into latest `main` by merge or rebase.
2. Re-run the backend verification set on merged `main`.
3. Restore/install frontend dependencies before any P2 frontend work; the current frontend test command cannot find `vitest`.

## Completed Substrates

### P0 CRC Safety Loop

Available substrate:

- Intended-use boundaries: `docs/safety/intended_use.md`
- Profiles: `config/intended_use_profiles.yaml`
- Deterministic safety policy: `config/safety_policy.yaml`
- Policy evaluator: `src/services/clinical_safety_policy.py`
- Harness/release artifacts: `reports/harness/*`, `reports/release_safety/*`
- Regression coverage:
  - `tests/backend/test_clinical_safety_policy.py`
  - `tests/backend/test_crc_triage_flow.py`
  - `tests/backend/test_crc_triage_save.py`
  - `tests/backend/test_crc_harness_replay.py`

P2 boundary inherited from P0:

- Research workspace may use research/intended-use boundaries.
- No patient-level research export before ethics/data-governance checks.
- Research outputs must not become patient advice or clinical decisions.

### P1 Clinical Review Loop

Available substrate:

- Clinical assertion contract: `src/contracts/clinical_assertion.py`
- Assertion projection: `src/services/clinical_assertion_projection.py`
- Doctor review API/action traces covered by:
  - `tests/backend/test_clinical_assertion_projection.py`
  - `tests/backend/test_doctor_review_api.py`
  - `tests/backend/test_doctor_action_trace.py`

P2 boundary inherited from P1:

- Doctor feedback is append-only unless a later release gate explicitly promotes it.
- Doctor edits must not automatically mutate P0 safety policy or patient facts.
- P2 should reuse assertion provenance rather than adding hidden model truth.

### P1.5 Step 10 Literature Evidence Harness

Available substrate on `step10-evidenceclaim-literature-harness`:

- Evidence contract: `src/contracts/evidence_claim.py`
- Literature harness service: `src/services/literature_harness.py`
- Fixed claim pack fixture: `tests/fixtures/literature_claim_pack_v0.json`
- Replay script: `scripts/run_literature_harness.py`
- Shadow report: `reports/literature/literature_harness_20260630_001.json`
- Report README: `reports/literature/README.md`
- Regression coverage:
  - `tests/backend/test_evidence_claim_contract.py`
  - `tests/backend/test_literature_harness.py`

P2 boundary inherited from Step 10:

- Literature claims are `candidate`, `needs_review`, or `rejected` only.
- Literature claims must not enter clinical RAG, patient default paths, doctor default paths, prompt/rubric/route/template patches, or training data.
- Claim payload attempts to provide `review_status` are rejected and block the harness run.
- The committed report is deterministic and shadow-only.

## Latest Verification Evidence

Run from `D:\YiZhu_Agnet\LangG\.worktrees\step10-evidenceclaim-literature-harness`:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py -q
```

Result: `52 passed`.

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_assertion_projection.py tests/backend/test_doctor_review_api.py tests/backend/test_doctor_action_trace.py -q
```

Result: `34 passed`.

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_save.py tests/backend/test_crc_harness_replay.py -q
```

Result: `29 passed`.

```powershell
D:\anaconda3\envs\LangG\python.exe scripts\run_literature_harness.py
git diff -- reports\literature\literature_harness_20260630_001.json
git ls-files --eol reports\literature\literature_harness_20260630_001.json
```

Result:

- Replay wrote `reports\literature\literature_harness_20260630_001.json`.
- Report diff was empty.
- EOL was `i/lf w/lf attr/text eol=lf`.

Frontend broad regression status:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose
```

Result: failed because `vitest` is not recognized and `frontend\node_modules\.bin\vitest.cmd` is absent in the Step 10 worktree. No frontend files are changed by Step 10.

## P2 Entry Checklist

Before writing a P2 spec or implementation plan:

- Merge or rebase Step 10 onto latest `main`.
- Confirm `git status --short --branch` is clean in the P2 starting workspace.
- Re-run backend verification after integration:
  - Step 10 focused tests
  - P1 backend tests
  - P0 backend tests
- Restore frontend dependencies if P2 touches frontend:
  - `frontend/node_modules/.bin/vitest.cmd` must exist.
  - Frontend tests should run before any frontend P2 acceptance claim.
- Decide whether P2 depends on Step 11 Agent Admin dashboard first. Step 10 provides a stable report input for Step 11, while P2 research cohort feasibility is still a separate later scope.
- Keep P2 read-only until an explicit data-governance/release gate exists.

## P2 Safe Starting Points

P2 can safely consume:

- P0 intended-use profile boundaries.
- P1 `ClinicalAssertion` and `DoctorActionTrace` provenance.
- P1.5 `EvidenceClaim`, `EvidenceDelta`, `LiteratureHarnessRun`, and the deterministic literature report.

P2 should not consume:

- Unreviewed literature as clinical recommendations.
- Candidate claims as clinical RAG chunks.
- Doctor action traces as automatic patient truth.
- Patient-level research export without a governance gate.
- Prompt/rubric/route/template patches from evidence deltas.

## Open Items For P2 Planning

- Define whether Step 11 Agent Admin read-only review dashboard comes before P2 research workspace.
- Define the governance model for any patient-level research export.
- Define whether P2 uses only committed reports/fixtures first, or introduces a new research query layer.
- Define a read-only data model for research cohort feasibility that references assertions and evidence claims without promoting them.
- Fix or reinstall frontend dependencies before any P2 frontend UI work.
