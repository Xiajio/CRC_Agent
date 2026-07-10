# CRC Agent Closeout Stage C Evidence Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the P1.5 evidence-promotion gate with immutable literature evidence, an audited ProjectEvidencePool, scope-bound IngestPreview approval, real-store isolation checks, versioned release reports, and read-only evidence views.

**Architecture:** Stage C consumes the accepted Stage B draft/evidence provenance plus Stage A's public integrity, AuthContext, sanitizer, atomic artifact, runtime-version, and StageGate contracts. Literature candidates remain separate from a reviewed append-only ProjectEvidencePool. An immutable IngestPreview binds exact claim, target index, review, sanitizer, governance, and harness versions; its approved output is only a ClinicalRagPromotionCandidate. Dedicated adapters prove candidate/pool/RAG/default-path isolation against real stores and runtime VersionRefs. Release dashboards discover immutable report trains dynamically and block on the newest invalid candidate.

**Tech Stack:** Python 3.10, dataclasses, JSON/SHA-256 audit chains, server-only HMAC for restricted clinical integrity, FastAPI 0.135, Pydantic 2.12, React 18, TypeScript 5.6, Vitest 2.1, pytest.

## Global Constraints

- Start from the merged Stage B commit only after its post-merge StageGateReport and StageGateApprovalAttestation validate against the latest approved manifest. Before Task 1, build the Stage C `StagePlanApprovalSubject` from this plan's tracked blob/source commit and validate its policy-required `PlanApprovalAttestation` with Stage A's shared CLI.
- Consume `src/contracts/integrity.py`, `src/contracts/auth_context.py`, `src/services/atomic_artifact_store.py`, `src/services/write_boundary_sanitizer.py`, `src/services/runtime_version_registry.py`, and the inherited StageGate runner.
- LiteratureCandidateStore, ProjectEvidencePoolStore, ClinicalRagManifest, and patient/doctor default-path usage are distinct versioned adapters. Empty fixture ID lists are not isolation evidence.
- Missing, unreadable, unhashed, ambiguous, or drifted adapter state blocks isolation and promotion.
- ProjectEvidencePool and all review/approval events are append-only. Authors cannot self-approve.
- IngestPreview is immutable, expiring, supersedable, and scope-bound. Approval never carries to a superseding preview.
- Stage C ends at `ClinicalRagPromotionCandidate`; no Stage C route, service, test helper, dashboard, or LearningJob writes Clinical RAG.
- A P0 harness pass cannot substitute for literature/evidence review, preview approval, isolation, or RAG target validation.
- Release-report composition has two explicit, mutually exclusive bases: the normal Stage C flow consumes an approved Stage A gate; the later final-closeout flow consumes a currently approved final-manifest attestation plus exact same-commit typed HarnessRun, P0 ReleaseSafetyReport, and LiteratureHarnessRun refs and can publish only an unauthorized candidate. The latter exists solely to avoid a candidate-to-refreshed-gate dependency cycle; it grants no release or preview approval.
- EvidenceClaim views are read-only and may expose only sanitized research-safe fields, never patient content, hidden reasoning, credentials, or prompts.
- Preserve unrelated user files, do not modify `CRC-client/`, and stage only files named by the active task.

## Pre-Implementation Plan Authorization

- [ ] Resolve the exact tracked plan blob, publish its subject, collect the three policy-required external approvals, and exact-stage the resulting evidence:

```powershell
$planPath = "docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-c-evidence-promotion.md"
$planSourceCommit = (git log -1 --format=%H -- $planPath).Trim()
git cat-file -e "${planSourceCommit}:$planPath"
$trackedPlanBlob = (git rev-parse "${planSourceCommit}:$planPath").Trim()
$workingPlanBlob = (git hash-object -- $planPath).Trim()
if ($trackedPlanBlob -ne $workingPlanBlob) { throw "Stage C plan blob is not tracked" }
$planSubjectPath = "reports/closeout/plan_subjects/stage_c_plan_20260710_001.json"
$planAttestationPath = "reports/closeout/attestations/stage_c_plan_approval_20260710_001.json"
$planEvidencePaths = "output/closeout/stage-c-plan-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py build-plan-subject --plan $planPath --source-commit $planSourceCommit --author-credential-env LANGG_CLOSEOUT_PUBLISHER_TOKEN --output $planSubjectPath
$planSubjectHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field sha256).Trim()
$planSubjectVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-c-evidence-reviewer-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-c-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$planSubjectHash-stage-c-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_plan --subject-path $planSubjectPath --output $planAttestationPath --path-list-output $planEvidencePaths --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_plan --subject-path $planSubjectPath --attestation-path $planAttestationPath
git add --pathspec-from-file=$planEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $planEvidencePaths --subject-path $planSubjectPath --attestation-path $planAttestationPath
git commit -m "evidence(closeout): approve Stage C plan"
```

- [ ] **Hard stop:** do not start Task 1 until the exact staged-set verifier and commit succeed and the subject plan ref/hash/version, approval-policy ref, author exclusion, quorum, ledger head, and latest-plan selection all validate.

## Source Design

- `docs/superpowers/specs/2026-07-10-crc-agent-closeout-program-design.md`, especially Sections 5, 8, and 10-18.
- `docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-a-safety-persistence.md` and `docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-b-clinical-review.md` for inherited contracts.
- Existing literature harness, EvidenceClaim, release-safety, admin dashboard, research API, and frontend workspace behavior are migration inputs; static report paths and fixture-only isolation are explicitly insufficient.

## File Structure

- `src/contracts/literature_harness.py`: versioned literature run and isolation result.
- `src/contracts/evidence_promotion.py`: evidence-pool review, IngestPreview, approval, invalidation, and ClinicalRagPromotionCandidate contracts.
- `backend/api/services/literature_candidate_store.py`: read-only validated literature candidates.
- `backend/api/services/project_evidence_pool_store.py`: append-only reviewed evidence and lifecycle events.
- `backend/api/services/ingest_preview_store.py`: immutable previews, approvals, invalidations, and supersession.
- `backend/api/services/evidence_isolation_adapters.py`: real candidate/pool/RAG/default-path adapters.
- `src/services/evidence_promotion.py`: review quorum, preview validation, and candidate creation.
- `src/contracts/release_report.py`, `backend/api/services/release_report_catalog.py`: versioned release-report trains and immutable latest selection.
- `backend/api/services/release_state_snapshot.py`: non-recursive dynamic dashboard snapshot.
- `src/services/research_evidence_view.py`: sanitized read-only EvidenceClaim projection.
- `frontend/src/features/research/*`: read-only evidence workspace.

---

### Task 1: Version EvidenceClaim And LiteratureHarnessRun

**Files:**

- Create: `src/contracts/literature_harness.py`
- Create: `tests/backend/test_literature_harness_contract.py`
- Modify: `src/contracts/evidence_claim.py`
- Modify: `src/services/literature_harness.py`
- Modify: `tests/backend/test_evidence_claim_contract.py`
- Modify: `tests/backend/test_literature_harness.py`

**LiteratureHarnessRun fields:** schema/run IDs, creation time, `claim_pack_ref`, `evidence_index_ref`, `judge_rubric_ref`, `case_catalog_ref`, claims, deltas, case results, isolation report, compliance status, release disposition, validation errors, source commit, and toolchain ref.

- [ ] Add failing EvidenceClaim tests for stable claim ID, claim/delta relationship, positive/negative/conflict zones, review/retraction status, citation confidence, exact `sanitizer_ref: VersionRef`, and deterministic sanitized hash.
- [ ] Add failing LiteratureHarnessRun tests for complete VersionRefs, exact declared-case comparison, isolation adapter results, compliance/disposition separation, acyclic hashing, and rejection of bare version labels.
- [ ] Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness_contract.py tests/backend/test_literature_harness.py -q -p no:cacheprovider
```

Expected: missing contract or incomplete-version assertions fail.

- [ ] Extend EvidenceClaim additively and implement frozen LiteratureHarnessRun contracts using Stage A canonical hashing. Volatile timestamps are excluded only from the documented semantic hash payload.
- [ ] Refactor the literature harness to return the typed run and reject missing/unhashed refs before evaluation.
- [ ] Re-run the exact Task 1 command, require PASS, and commit only Task 1 paths:

```powershell
git add src/contracts/literature_harness.py tests/backend/test_literature_harness_contract.py src/contracts/evidence_claim.py src/services/literature_harness.py tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness.py
git commit -m "feat: version literature evidence contracts"
```

---

### Task 2: Append-Only ProjectEvidencePool And Authorized Review Chain

**Files:**

- Create: `src/contracts/evidence_promotion.py`
- Create: `backend/api/services/literature_candidate_store.py`
- Create: `backend/api/services/project_evidence_pool_store.py`
- Create: `scripts/manage_project_evidence_pool.py`
- Create: `src/services/evidence_promotion.py`
- Create: `backend/api/schemas/evidence.py`
- Create: `backend/api/routes/evidence.py`
- Create: `reports/evidence_pool/README.md`
- Create: `tests/backend/test_project_evidence_pool_contract.py`
- Create: `tests/backend/test_project_evidence_pool_store.py`
- Create: `tests/backend/test_evidence_promotion_service.py`
- Create: `tests/backend/test_evidence_promotion_non_mutation.py`
- Modify: `backend/app.py`

**Event path:** `reports/evidence_pool/events/{sequence:020d}.json`.

- [ ] Add failing contracts for candidate ref, review request, review decision, pool entry, supersession/retraction, audit event, and pool-head VersionRef.
- [ ] Add store tests for path bounds, sequence continuity, previous-event hashes, immutable entries, idempotent same-payload writes, conflicting replays, corrupted heads, and recovery by a higher-sequence event rather than overwrite.
- [ ] Add authorization tests: project-scoped reviewer roles, distinct principal/credential quorum, author self-approval rejection, revoked credential behavior, server-derived actor data, allowlisted reason codes, and sanitized bounded reason text with exact sanitizer ref.
- [ ] Add non-mutation snapshots proving review/pool promotion cannot change Clinical RAG, patient/doctor default runtime refs, patient records, safety policy, prompts, feature flags, or models.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_project_evidence_pool_contract.py tests/backend/test_project_evidence_pool_store.py tests/backend/test_evidence_promotion_service.py tests/backend/test_evidence_promotion_non_mutation.py -q -p no:cacheprovider`; expect missing module failures.
- [ ] Implement a read-only candidate adapter and append-only pool store through `AtomicJsonArtifactPublisher`. Validate candidate schema/hash and exact sanitizer `VersionRef` before opening review.
- [ ] Implement the exact operator CLI below. Each decision invocation resolves one protected credential into AuthContext, binds expected claim/pool versions and an idempotency key, rejects the candidate author, and emits the verified pool-entry VersionRef to a sanitized ignored context file:

```text
open-review --literature-run PATH --claim-id ID --output PATH
inspect-subject --subject PATH --field sha256|subject_version
record-decision --subject PATH --expected-sha256 SHA --expected-version VERSION --decision approve|reject --reason-code CODE --credential-env ENV_NAME --idempotency-key KEY
verify-entry --subject PATH --output PATH
```
- [ ] Implement evidence review routes with AuthContext authorization. A reviewed pool entry references the original candidate and all approval events; it never copies unsanitized source text.
- [ ] Re-run the four new test files, require PASS, and commit only Task 2 paths:

```powershell
git add src/contracts/evidence_promotion.py backend/api/services/literature_candidate_store.py backend/api/services/project_evidence_pool_store.py scripts/manage_project_evidence_pool.py src/services/evidence_promotion.py backend/api/schemas/evidence.py backend/api/routes/evidence.py reports/evidence_pool/README.md tests/backend/test_project_evidence_pool_contract.py tests/backend/test_project_evidence_pool_store.py tests/backend/test_evidence_promotion_service.py tests/backend/test_evidence_promotion_non_mutation.py backend/app.py
git commit -m "feat: add audited project evidence pool"
```

---

### Task 3: Immutable IngestPreview, Approval, Invalidation, And Candidate Boundary

**Files:**

- Create: `backend/api/services/ingest_preview_store.py`
- Create: `scripts/manage_ingest_preview.py`
- Create: `tests/backend/test_ingest_preview_contract.py`
- Create: `tests/backend/test_ingest_preview_store.py`
- Modify: `src/contracts/evidence_promotion.py`
- Modify: `src/services/evidence_promotion.py`
- Modify: `backend/api/schemas/evidence.py`
- Modify: `backend/api/routes/evidence.py`
- Modify: `tests/backend/test_evidence_promotion_service.py`
- Modify: `tests/backend/test_evidence_promotion_non_mutation.py`

**IngestPreview fields:** preview ID/hash, created/expires timestamps, supersedes ref, claim/delta refs, target evidence-index ref, normalized proposed chunks with source spans, duplicate/conflict findings, validation warnings, expected retrieval changes, upstream review refs present at creation, literature-harness ref, exact sanitizer/schema/governance refs, project ID, and creator principal.

`scripts/manage_ingest_preview.py` exposes the following exact non-interactive surface:

```text
create --preview-id ID --sequence N --pool-entry-ref PATH --target-index PATH --literature-run PATH --expires-at UTC --output PATH [--supersedes PATH]
inspect-subject --preview PATH --field sha256|subject_version
resolve-latest --target-index PATH --project-id ID --output PATH [--require-effective-approval]
inspect-context --context PATH --field preview_path|sha256|subject_version|sequence|next_sequence|literature_run_path
record-review --preview PATH --expected-sha256 SHA --expected-version VERSION --decision approve|reject --reason-code CODE --credential-env ENV_NAME --idempotency-key KEY
invalidate --preview PATH --expected-sha256 SHA --expected-version VERSION --reason-code CODE --credential-env ENV_NAME --idempotency-key KEY
verify-ready --preview PATH --output PATH
write-evidence-path-list --pool-entry-context PATH --preview PATH --output PATH
verify-staged-evidence --path-list PATH --pool-entry-context PATH --preview PATH
```

`record-review` resolves exactly one server-side AuthContext and never accepts a role/principal argument. `create` requires a monotonically increasing sequence; an initial preview forbids `--supersedes`, while every later sequence requires it to point to the exact current latest preview and derives a create-once ID containing the zero-padded sequence plus a 12-character source/prior hash. For a superseder, `--pool-entry-ref` may be the ignored `resolve-latest` context; `create` extracts and revalidates its exact pool-entry ref and rejects a race where the prior preview is no longer latest. `resolve-latest` validates the bounded target/project chain and writes one ignored context containing the exact preview path/ref/hash/version, sequence/next-sequence, literature-run path, and pool-entry ref; `--require-effective-approval` additionally rejects expired, revoked, invalidated, drifted, or unapproved state. `inspect-context` prints only one allowlisted field. `verify-ready` derives quorum from the event ledger, revalidates all refs/expiry/isolation/governance, and writes an ignored sanitized context containing the aggregate approval-event VersionRef and embedded ClinicalRagPromotionCandidate ref. `write-evidence-path-list` includes only newly created pool/preview/event evidence in the current commit; already tracked pool entries, prior previews, and prior events are rehashed and validated as dependencies but excluded from the staged set. `verify-staged-evidence` requires the staged path set to equal that list and rejects older/unrelated uncommitted events or source/config files.

- [ ] Add failing tests for immutable content/hash over warnings and expected retrieval changes, chunk source spans, TTL, target/project scope, monotonic sequence plus SHA-derived ID, required exact supersession, approval/invalidation events, effective/latest resolution, and preview path safety.
- [ ] Add table-driven service tests requiring distinct `evidence_reviewer` and `clinical_safety_reviewer` for every approval; requiring `release_manager` only when the active governance contract says so; requiring `pi_reviewer` only for a research-governed source; and covering author exclusion, expiry, target drift, review revocation, isolation failure, and idempotency.
- [ ] Assert a superseding preview inherits no approval. Assert a P0 harness run cannot substitute for a LiteratureHarnessRun or preview-specific approval.
- [ ] Assert the final approval event embeds exactly one immutable `ClinicalRagPromotionCandidate` snapshot in the same create-once aggregate event artifact; there is no second-file partial commit. Every Clinical RAG adapter method remains read-only.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_ingest_preview_contract.py tests/backend/test_ingest_preview_store.py tests/backend/test_evidence_promotion_service.py tests/backend/test_evidence_promotion_non_mutation.py -q -p no:cacheprovider`; expect missing preview behavior.
- [ ] Implement preview create/read/supersede/invalidate/approve operations as append-only artifacts/events. Validate auth and every bound VersionRef immediately before approval.
- [ ] Return only the promotion candidate boundary; do not add an ingest/apply endpoint or write helper.
- [ ] Re-run the focused Task 3 tests, require PASS, and commit only Task 3 paths:

```powershell
git add backend/api/services/ingest_preview_store.py scripts/manage_ingest_preview.py tests/backend/test_ingest_preview_contract.py tests/backend/test_ingest_preview_store.py src/contracts/evidence_promotion.py src/services/evidence_promotion.py backend/api/schemas/evidence.py backend/api/routes/evidence.py tests/backend/test_evidence_promotion_service.py tests/backend/test_evidence_promotion_non_mutation.py
git commit -m "feat: add immutable ingest preview governance"
```

---

### Task 4: Real-Store Isolation And Dedicated Literature Replay

**Files:**

- Create: `backend/api/services/evidence_isolation_adapters.py`
- Create: `config/evidence_indexes/rag_crc_guideline_20260620.json`
- Create: `config/literature_judge_rubrics/crc_evidence_closeout_v1.json`
- Create: `tests/fixtures/literature_claim_pack_closeout_v1.json`
- Create: `tests/fixtures/literature_harness_case_catalog_v1.json`
- Create: `tests/backend/test_evidence_isolation_adapters.py`
- Create after the implementation commit: `reports/literature/literature_harness_stage_c_<source-sha12>.json`
- Modify: `src/services/literature_harness.py`
- Modify: `scripts/run_literature_harness.py`
- Modify: `src/services/runtime_version_registry.py`
- Modify: `tests/backend/test_literature_harness.py`

**Adapters:** `LiteratureCandidateStoreAdapter`, `ProjectEvidencePoolStoreAdapter`, `ClinicalRagManifestAdapter`, `PatientDefaultUsageAdapter`, and `DoctorDefaultUsageAdapter`.

- [ ] Add failing adapter tests against real temporary stores/manifests for exact refs, hashes, source commits, missing/unreadable/malformed/unhashed states, ambiguous matches, and drift between registry/API usage and manifest.
- [ ] Add integration tests proving candidate IDs exist only in the candidate/pool domains unless an approved candidate explicitly references them; no candidate/pool ID or hash appears in Clinical RAG or patient/doctor effective runtime bindings.
- [ ] Put the stable claim ID `stage_c_governed_claim_001` in the claim pack and case catalog. Its declared closeout governance requires distinct evidence-reviewer, clinical-safety, and release-manager approvals for the preview and is not a research-governed source, so the Task 9 CLI sequence is deterministic and does not require PI review.
- [ ] Ensure patient/doctor adapters load and validate the actual immutable runtime snapshot refs plus labelled `RuntimeInfo.version_bindings` and `DoctorDraftVersion.runtime_version_bindings`, not fixture-provided empty ref lists. They must prove slot-to-ref equality and block duplicate, missing, unknown, active/shadow-swapped, or registry-ledger-drifted bindings.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_isolation_adapters.py tests/backend/test_literature_harness.py -q -p no:cacheprovider`; expect current fixture-only isolation to fail.
- [ ] Implement versioned adapters and a composite isolation report whose missing/invalid/ambiguous result is `compliance_status=block`.
- [ ] Activate the exact Clinical RAG manifest `VersionRef` in the Stage A registry's `clinical_rag:crc_guideline:active` slot through `activate(slot, ref, expected_current_ref, auth, idempotency_key)`. Add tests for stale expected-current refs, unauthorized/project-mismatched AuthContext, idempotent retry, and append-only activation history; never overwrite or silently re-register a slot. The manifest file must enumerate content artifacts and hashes, not only a version label.
- [ ] Replace the literature script's current execute-on-any-argument/hard-coded-overwrite behavior with argparse. `--help` exits without running; publication requires explicit `--run-id`, `--claim-pack`, `--case-catalog`, `--evidence-index`, `--judge-rubric`, `--source-commit`, `--output-root`, and `--publish`; `--verify-against` uses a temporary root and does not mutate committed evidence.
- [ ] Re-run the exact adapter/literature test command and require GREEN, then commit the Task 4 source/config/test paths before generating evidence:

```powershell
git add backend/api/services/evidence_isolation_adapters.py config/evidence_indexes/rag_crc_guideline_20260620.json config/literature_judge_rubrics/crc_evidence_closeout_v1.json tests/fixtures/literature_claim_pack_closeout_v1.json tests/fixtures/literature_harness_case_catalog_v1.json tests/backend/test_evidence_isolation_adapters.py src/services/literature_harness.py scripts/run_literature_harness.py src/services/runtime_version_registry.py tests/backend/test_literature_harness.py
git commit -m "feat: validate literature evidence against real stores"
```

- [ ] Publish the dedicated replay from that exact source commit, prove create-once behavior, and commit the one evidence path separately:

```powershell
$literatureSourceCommit = (git rev-parse HEAD).Trim()
$literatureSha12 = $literatureSourceCommit.Substring(0, 12)
$literatureRunId = "literature_harness_stage_c_$literatureSha12"
$literatureRunPath = "reports/literature/$literatureRunId.json"
D:\anaconda3\envs\LangG\python.exe scripts/run_literature_harness.py --run-id $literatureRunId --claim-pack tests/fixtures/literature_claim_pack_closeout_v1.json --case-catalog tests/fixtures/literature_harness_case_catalog_v1.json --evidence-index config/evidence_indexes/rag_crc_guideline_20260620.json --judge-rubric config/literature_judge_rubrics/crc_evidence_closeout_v1.json --source-commit $literatureSourceCommit --output-root reports --publish
$publishedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $literatureRunPath).Hash
D:\anaconda3\envs\LangG\python.exe scripts/run_literature_harness.py --run-id $literatureRunId --claim-pack tests/fixtures/literature_claim_pack_closeout_v1.json --case-catalog tests/fixtures/literature_harness_case_catalog_v1.json --evidence-index config/evidence_indexes/rag_crc_guideline_20260620.json --judge-rubric config/literature_judge_rubrics/crc_evidence_closeout_v1.json --source-commit $literatureSourceCommit --output-root reports --publish
if ($LASTEXITCODE -eq 0) { throw "create-once replay unexpectedly overwrote evidence" }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $literatureRunPath).Hash -ne $publishedHash) { throw "create-once replay changed evidence" }
git add -- $literatureRunPath
git diff --cached --check
$stagedPaths = @(git diff --cached --name-only)
if (($stagedPaths.Count -ne 1) -or ($stagedPaths[0] -ne $literatureRunPath)) { throw "unexpected Stage C literature evidence staging set" }
git commit -m "evidence(stage-c): publish literature replay"
```

The CLI resolves and binds exact VersionRefs for both config inputs. A second publication must fail without changing the artifact hash.

---

### Task 5: Versioned Release-Report Trains And Immutable Catalog

**Files:**

- Create: `src/contracts/release_report.py`
- Create: `backend/api/services/release_report_catalog.py`
- Create: `scripts/publish_release_report_candidate.py`
- Create: `reports/release_safety/candidates/README.md`
- Create: `tests/backend/test_release_report_contract.py`
- Create: `tests/backend/test_release_report_catalog.py`
- Create: `tests/backend/test_publish_release_report_candidate.py`
- Create after the implementation commit: `reports/release_safety/candidates/release_safety_stage_c_<source-sha12>.json`

**Header:** schema version, report ID, train ID, monotonically increasing sequence, creation time, parent report ref, optional superseded report ref, composition mode/basis ref, authorization status, payload hash, source commit, and toolchain ref.

**Typed final-closeout interface:**

```python
@dataclass(frozen=True, slots=True)
class FinalCloseoutCandidateCompositionRequest:
    report_id: str
    train_id: str
    sequence: int
    source_commit: str
    final_manifest_ref: VersionRef
    final_manifest_attestation_ref: VersionRef
    publisher_event_ref: VersionRef
    harness_run_ref: VersionRef
    p0_release_safety_report_ref: VersionRef
    literature_harness_run_ref: VersionRef
    parent_report_ref: VersionRef | None


def compose_final_closeout_candidate(
    request: FinalCloseoutCandidateCompositionRequest,
    *,
    publisher_auth: AuthContext,
) -> ReleaseReportCandidate: ...
```

- [ ] Add failing tests for train/sequence uniqueness, parent linkage, newest selection, last-valid diagnostic selection, immutable malformed candidates, supersession by a higher sequence, path bounds, and legacy normalization. Add mode-exclusivity tests plus final-closeout cases for missing/stale/revoked/wrong-source manifest attestation, non-final manifest, caller-supplied principal or role, non-`closeout_publisher`/wrong-project AuthContext, mismatched source commits, swapped object kinds, a P0 report whose embedded harness ref differs, a path/ref/hash mismatch, accidental `approved` status, and any attempt to treat the unauthorized candidate as a gate attestation. Add restart tests for a crash after the content-free publisher event but before candidate publication, unrelated ledger advancement, same-principal credential rotation, different-principal replacement, revoked fresh credentials, and an event ref/hash that differs from the fsynced request: the first three must reuse the original fixed event/request where authorization remains valid, while the latter cases block without creating a second candidate.
- [ ] Assert the newest candidate controls release state: if it is invalid, the train blocks even when an older valid report exists; the older report is diagnostic only.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_report_contract.py tests/backend/test_release_report_catalog.py tests/backend/test_publish_release_report_candidate.py -q -p no:cacheprovider`; expect missing catalog/publisher failures.
- [ ] Implement the catalog over `reports/release_safety/candidates/` with create-once publication. Keep legacy reports untouched and visible as migration history.
- [ ] Implement one release-report composition service used by `publish_release_report_candidate.py`. It consumes immutable typed P0 HarnessRun/ReleaseSafetyReport and LiteratureHarnessRun refs, normalizes legacy values only at this boundary, and validates all linked policy, runtime, environment, and evidence refs. Do not modify or reuse `run_crc_harness_replay.py` as the literature runner and do not allow either typed harness to substitute for the other. Its two mutually exclusive publication surfaces are:

```text
publish_release_report_candidate.py --composition-mode stage-a-approved --report-id ID --train-id ID --sequence N --stage-a-gate-report PATH --stage-a-gate-attestation PATH --literature-run PATH --source-commit SHA --output PATH --publish [--parent-report PATH]
publish_release_report_candidate.py --composition-mode final-closeout-unapproved --report-id ID --train-id ID --sequence N --final-manifest PATH --final-manifest-attestation PATH --harness-run PATH --release-safety-report PATH --literature-run PATH --source-commit SHA --publisher-credential-env LANGG_CLOSEOUT_PUBLISHER_TOKEN --publisher-idempotency-key KEY --phase-context PATH --output PATH --publish [--parent-report PATH]
```

In `stage-a-approved` mode, the publisher validates the Stage A report/attestation chain and resolves exactly one typed P0 HarnessRun plus one typed ReleaseSafetyReport from its artifact refs; callers cannot supply substitute object kinds. In `final-closeout-unapproved` mode, it instead requires the shared typed fields `manifest_phase="final"`, a lowercase full-SHA `frozen_release_content_commit`, and a valid `supersedes_manifest_ref` to the exact prior approved manifest; a bootstrap/unknown phase, null/wrong commit, filename-derived phase, or non-current superseder blocks. It validates that exact final CloseoutRequirementManifest and its current external attestation, loads all three explicit evidence paths without discovery, verifies path/ref/canonical-hash equality, requires those three evidence refs plus the candidate's declared source commit to equal the typed frozen commit and the CLI `--source-commit`, and requires the P0 report's embedded harness ref to equal the supplied harness ref. The manifest attestation may be published in a later evidence-only commit; it is validated by its bound subject ref/hash, policy, ledger, and approval chain rather than by pretending its own creation commit is the release-content commit. Before any public artifact, the service resolves a fresh project-scoped `closeout_publisher`, deterministically derives the content-free publisher event from the caller-supplied idempotency key, puts that expected event ref into `FinalCloseoutCandidateCompositionRequest`, and fsyncs the complete request/expected hashes to the ignored path-bounded phase context. It then publishes or reuses that exact event and only then publishes the candidate. `compose_final_closeout_candidate()` loads the declared event by ref, checks its kind/hash/request-digest binding to the source commit, manifest/attestation, report ID and three evidence refs, resolves its historical credential record, and verifies the event's principal, credential, `closeout_publisher` role and project scope against fresh authorization. On restart, fresh authorization must be active and resolve to the event's same principal/project; same-principal credential rotation and unrelated later ledger events do not change the fixed event ref, while a different principal, mismatched event, or invalid fresh credential blocks. The candidate persists the server-derived author principal loaded from that event plus its fixed ref; the phase context stores neither identity nor credential data. This mode always emits `composition_mode=final_closeout_unapproved` and `authorization_status=not_approved`; it accepts no Stage A gate arguments and cannot produce or stand in for an approval. The later refreshed Stage A and C gates must validate the same exact refs before the candidate can contribute to a final decision.
- [ ] Re-run the exact Task 5 pytest command and require GREEN, then commit source/tests before generating the candidate:

```powershell
git add src/contracts/release_report.py backend/api/services/release_report_catalog.py scripts/publish_release_report_candidate.py reports/release_safety/candidates/README.md tests/backend/test_release_report_contract.py tests/backend/test_release_report_catalog.py tests/backend/test_publish_release_report_candidate.py
git commit -m "feat: add versioned release report catalog"
```

- [ ] Generate the first candidate from the exact validated Stage A gate and Task 4 literature evidence, then commit the one candidate path separately:

```powershell
$candidateSourceCommit = (git rev-parse HEAD).Trim()
$candidateSha12 = $candidateSourceCommit.Substring(0, 12)
$candidateId = "release_safety_stage_c_$candidateSha12"
$candidatePath = "reports/release_safety/candidates/$candidateId.json"
$literatureSourceCommit = (git log -1 --format=%H -- scripts/run_literature_harness.py).Trim()
$literatureSha12 = $literatureSourceCommit.Substring(0, 12)
$literatureRunPath = "reports/literature/literature_harness_stage_c_$literatureSha12.json"
$stageAGateReportPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage A --field subject_path).Trim()
$stageAGateAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage A --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/publish_release_report_candidate.py --composition-mode stage-a-approved --report-id $candidateId --train-id crc_closeout_stage_c --sequence 1 --stage-a-gate-report $stageAGateReportPath --stage-a-gate-attestation $stageAGateAttestationPath --literature-run $literatureRunPath --source-commit $candidateSourceCommit --output $candidatePath --publish
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_report_contract.py tests/backend/test_release_report_catalog.py tests/backend/test_publish_release_report_candidate.py -q -p no:cacheprovider
git add -- $candidatePath
git diff --cached --check
$stagedPaths = @(git diff --cached --name-only)
if (($stagedPaths.Count -ne 1) -or ($stagedPaths[0] -ne $candidatePath)) { throw "unexpected Stage C release candidate staging set" }
git commit -m "evidence(stage-c): publish release report candidate"
```

---

### Task 6: Dynamic Release Dashboard And Flag-Drift Detection

**Files:**

- Create: `backend/api/services/release_state_snapshot.py`
- Create: `tests/backend/test_release_dashboard_flag_drift.py`
- Modify: `backend/api/services/admin_release_dashboard.py`
- Modify: `backend/api/routes/admin.py`
- Modify: `src/services/release_governance.py`
- Modify: `src/services/release_execution.py`
- Modify: `tests/backend/test_admin_release_dashboard.py`
- Modify: `tests/backend/test_admin_release_dashboard_api.py`
- Modify: `tests/backend/test_release_governance_contract.py`
- Modify: `tests/backend/test_release_governance_service.py`
- Modify: `tests/backend/test_release_execution_service.py`
- Modify: `tests/backend/test_release_monitoring_service.py`
- Modify: `tests/backend/test_release_closure_service.py`

- [ ] Add failing tests for multi-train dynamic discovery, latest/last-valid distinction, hard/non-hard case rows, artifact links, intended versus effective flags, drift, isolation readiness, preview readiness, and newest-invalid blocking.
- [ ] Add regression mapping legacy `feature_flag_or_pass` to `feature_flag`; never surface the legacy value as a current decision.
- [ ] Avoid dependency recursion: `ReleaseStateSnapshot` reads stores/adapters directly; release governance/execution and dashboard consume the snapshot, but the snapshot never calls those services.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_release_dashboard_flag_drift.py tests/backend/test_release_governance_contract.py tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py tests/backend/test_release_monitoring_service.py tests/backend/test_release_closure_service.py -q -p no:cacheprovider`; expect static report-path and flag-state failures.
- [ ] Implement a sanitized typed snapshot. Intended flag state comes only from the active release-governance intent store, effective state comes from the real feature-flag store, and report/candidate data supplies the technical ceiling/evidence rather than intent. Test intent/report/effective three-way agreement and each independent drift; any disagreement blocks execution.
- [ ] Re-run dashboard/governance/execution tests, require PASS, and commit only Task 6 paths:

```powershell
git add backend/api/services/release_state_snapshot.py tests/backend/test_release_dashboard_flag_drift.py backend/api/services/admin_release_dashboard.py backend/api/routes/admin.py src/services/release_governance.py src/services/release_execution.py tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_release_governance_contract.py tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py tests/backend/test_release_monitoring_service.py tests/backend/test_release_closure_service.py
git commit -m "feat: harden release dashboard evidence"
```

---

### Task 7: Read-Only EvidenceClaim Research API

**Files:**

- Create: `src/services/research_evidence_view.py`
- Create: `tests/backend/test_research_evidence_view.py`
- Create: `tests/backend/test_research_evidence_api.py`
- Modify: `backend/api/schemas/research.py`
- Modify: `backend/api/routes/research.py`
- Modify: `backend/app.py`

**Endpoint:** `GET /api/research/evidence-claims`.

- [ ] Add failing service tests for every card field: claim text, population, outcome/effect, effect size and uncertainty, evidence grade, study design, sample size, guideline/systematic-review/preprint/retraction quality flags, bias, local-guideline conflict, CRC applicability, source span, conflict/negative-evidence state, review history, current isolation zone, claim/delta refs, citation confidence, deterministic sorting/filtering, and explicit source VersionRefs.
- [ ] Add authorization tests for project-scoped research/evidence readers, legacy read-only compatibility, and non-enumerating cross-project behavior.
- [ ] Add negative tests proving no action URL, approval mutation, patient data, direct identifier, hidden reasoning, prompt, credential, or unreviewed raw source text enters the projection.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_evidence_view.py tests/backend/test_research_evidence_api.py -q -p no:cacheprovider`; expect missing service/route failures.
- [ ] Implement a pure read projection over validated candidate/pool artifacts and add the GET route only.
- [ ] Re-run the two Task 7 tests, require PASS, and commit only Task 7 paths:

```powershell
git add src/services/research_evidence_view.py tests/backend/test_research_evidence_view.py tests/backend/test_research_evidence_api.py backend/api/schemas/research.py backend/api/routes/research.py backend/app.py
git commit -m "feat: expose research evidence review cards"
```

---

### Task 8: Read-Only Research Workspace And Admin Drilldown

**Files:**

- Create: `frontend/src/features/research/evidence-claim-types.ts`
- Create: `frontend/src/features/research/evidence-claim-types.test.ts`
- Create: `frontend/src/features/research/research-evidence-view.tsx`
- Create: `frontend/src/features/research/research-evidence-view.test.tsx`
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/features/workspace/workspace-surface-switcher.tsx`
- Modify: `frontend/src/features/workspace/workspace-surface-switcher.test.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Modify: `frontend/src/pages/workspace-page.test.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-pages.tsx`
- Modify: `frontend/src/styles/globals.css`
- Modify: `frontend/src/styles/style-architecture-contract.test.ts`

- [ ] Add failing navigation/render tests for a `"research"` WorkspaceSurface and failing type guards for the complete Task 7 field list, all four isolation zones, and missing/malformed refs or quality fields.
- [ ] Run the exact focused test command before changing production TypeScript/React:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/research/evidence-claim-types.test.ts src/features/research/research-evidence-view.test.tsx src/features/workspace/workspace-surface-switcher.test.tsx src/pages/workspace-page.test.tsx src/features/agent-admin/agent-admin-view.test.tsx src/styles/style-architecture-contract.test.ts --reporter=verbose
```

Expected RED: the research surface, strict type guard, read-only cards, and drilldown assertions fail against the current UI.

- [ ] Extend `WorkspaceSurface` with `"research"`. Render read-only evidence cards with claim text, population, outcome/effect, effect-size uncertainty, grade, study design/sample size, every quality/bias/applicability flag, source span, conflict/negative/retraction state, review history, all four isolation zones, and exact refs. Assert no POST request, review button, promotion button, or action builder exists.
- [ ] Add Admin drilldown for release evidence, trains, case rows, artifacts, isolation/preview readiness, and flag drift. Admin must not duplicate evidence-review controls.
- [ ] Re-run the exact Vitest command above, then run `cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build`. Expected GREEN: all tests pass and Vite/TypeScript exits 0.

- [ ] Commit only Task 8 paths:

```powershell
git add frontend/src/features/research/evidence-claim-types.ts frontend/src/features/research/evidence-claim-types.test.ts frontend/src/features/research/research-evidence-view.tsx frontend/src/features/research/research-evidence-view.test.tsx frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/features/workspace/workspace-surface-switcher.tsx frontend/src/features/workspace/workspace-surface-switcher.test.tsx frontend/src/pages/workspace-page.tsx frontend/src/pages/workspace-page.test.tsx frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/styles/globals.css frontend/src/styles/style-architecture-contract.test.ts
git commit -m "feat: add research evidence workspace"
```

---

### Task 9: Register, Merge, And Approve The Stage C Gate

**Files:**

- Create: `tests/backend/test_stage_c_evidence_gate.py`
- Modify: `config/closeout_stage_suites.yaml`
- Create before branch gate: `reports/evidence_pool/previews/ingest_preview_closeout_20260710_<sequence>_<source-sha12>.json`
- Create through append-only approval: `reports/evidence_pool/events/<assigned-sequence>.json`
- Create after merge: `reports/closeout/stages/stage_c.<merge-sha12>.<artifact-binding-hash12>.json`
- Create after report validation: `reports/closeout/attestations/stage_c_approval.<report-hash12>.json`

- [ ] Add failing gate tests for exact owned-entry sets, newest invalid release candidate, isolation adapters, approved/non-expired preview, preview expiry after report publication/between approval events, hash-suffixed single-parent report supersession, no Clinical RAG writes, read-only APIs/UI, inherited A/B regressions, and source/manifest drift, without registering Stage C yet.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_stage_c_evidence_gate.py -q -p no:cacheprovider`. Expected RED: Stage C, its required artifacts, and its exact owned-entry set are absent.
- [ ] Register Stage C directly in `config/closeout_stage_suites.yaml` with fixed commands, required artifact kinds, and immediate predecessor `inherits: [B]`; use `scripts/run_closeout_stage_gate.py --stage C` as the only CLI. Recursive deterministic expansion must execute and record the complete A+B+C suite exactly once.
- [ ] Re-run the exact focused gate test with its isolated positive fixtures and require GREEN, then commit the gate test/suite implementation before generating repository evidence:

```powershell
git add tests/backend/test_stage_c_evidence_gate.py config/closeout_stage_suites.yaml
git commit -m "test: register stage c evidence promotion gate"
```

- [ ] Open and approve the deterministic closeout claim, create/approve its real preview, exact-stage the full append-only chain, and commit it before the branch gate:

```powershell
$literatureSourceCommit = (git log -1 --format=%H -- scripts/run_literature_harness.py).Trim()
$literatureSha12 = $literatureSourceCommit.Substring(0, 12)
$literatureRunPath = "reports/literature/literature_harness_stage_c_$literatureSha12.json"
$poolReviewContext = "output/closeout/stage-c-pool-review.json"
$poolEntryContext = "output/closeout/stage-c-pool-entry-ref.json"
D:\anaconda3\envs\LangG\python.exe scripts/manage_project_evidence_pool.py open-review --literature-run $literatureRunPath --claim-id stage_c_governed_claim_001 --output $poolReviewContext
$poolReviewHash = (D:\anaconda3\envs\LangG\python.exe scripts/manage_project_evidence_pool.py inspect-subject --subject $poolReviewContext --field sha256).Trim()
$poolReviewVersion = (D:\anaconda3\envs\LangG\python.exe scripts/manage_project_evidence_pool.py inspect-subject --subject $poolReviewContext --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/manage_project_evidence_pool.py record-decision --subject $poolReviewContext --expected-sha256 $poolReviewHash --expected-version $poolReviewVersion --decision approve --reason-code closeout_candidate_validated --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$poolReviewHash-pool-evidence-review-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_project_evidence_pool.py verify-entry --subject $poolReviewContext --output $poolEntryContext
$previewId = "ingest_preview_closeout_20260710_001_$literatureSha12"
$previewPath = "reports/evidence_pool/previews/$previewId.json"
$previewExpiresAt = $env:LANGG_STAGE_C_PREVIEW_EXPIRES_AT
if ([string]::IsNullOrWhiteSpace($previewExpiresAt)) { throw "LANGG_STAGE_C_PREVIEW_EXPIRES_AT must be an operator-approved future UTC instant" }
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py create --preview-id $previewId --sequence 1 --pool-entry-ref $poolEntryContext --target-index config/evidence_indexes/rag_crc_guideline_20260620.json --literature-run $literatureRunPath --expires-at $previewExpiresAt --output $previewPath
$previewHash = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $previewPath --field sha256).Trim()
$previewVersion = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $previewPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $previewPath --expected-sha256 $previewHash --expected-version $previewVersion --decision approve --reason-code evidence_quality_verified --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$previewHash-preview-evidence-review-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $previewPath --expected-sha256 $previewHash --expected-version $previewVersion --decision approve --reason-code clinical_safety_verified --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$previewHash-preview-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $previewPath --expected-sha256 $previewHash --expected-version $previewVersion --decision approve --reason-code closeout_release_governance_verified --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$previewHash-preview-release-manager-v1"
$previewReadyContext = "output/closeout/stage-c-preview-ready.json"
$previewEvidencePaths = "output/closeout/stage-c-preview-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py verify-ready --preview $previewPath --output $previewReadyContext
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py write-evidence-path-list --pool-entry-context $poolEntryContext --preview $previewPath --output $previewEvidencePaths
git add --pathspec-from-file=$previewEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py verify-staged-evidence --path-list $previewEvidencePaths --pool-entry-context $poolEntryContext --preview $previewPath
git commit -m "evidence(stage-c): record governed ingest preview"
```

The scripts resolve exact claim/delta/rubric/sanitizer/governance VersionRefs; a bare label cannot satisfy validation. `verify-ready` must produce one unexpired ClinicalRagPromotionCandidate embedded in the final event.

- [ ] If effective approval expires, is revoked, or is invalidated before branch/post-merge verification, create a sequence/hash-derived superseder and collect a completely new approval chain. This block is self-contained across the hard stop and never overwrites the prior path:

```powershell
$latestPreviewContext = "output/closeout/stage-c-latest-preview.json"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py resolve-latest --target-index config/evidence_indexes/rag_crc_guideline_20260620.json --project-id closeout:crc --output $latestPreviewContext
$priorPreviewPath = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $latestPreviewContext --field preview_path).Trim()
$priorPreviewHash = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $latestPreviewContext --field sha256).Trim()
$priorLiteratureRunPath = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $latestPreviewContext --field literature_run_path).Trim()
$nextSequence = [int](D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $latestPreviewContext --field next_sequence).Trim()
$nextPreviewId = "ingest_preview_closeout_20260710_{0:D3}_{1}" -f $nextSequence, $priorPreviewHash.Substring(0, 12)
$nextPreviewPath = "reports/evidence_pool/previews/$nextPreviewId.json"
$previewExpiresAt = $env:LANGG_STAGE_C_PREVIEW_EXPIRES_AT
if ([string]::IsNullOrWhiteSpace($previewExpiresAt)) { throw "LANGG_STAGE_C_PREVIEW_EXPIRES_AT must be an operator-approved future UTC instant" }
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py create --preview-id $nextPreviewId --sequence $nextSequence --pool-entry-ref $latestPreviewContext --target-index config/evidence_indexes/rag_crc_guideline_20260620.json --literature-run $priorLiteratureRunPath --expires-at $previewExpiresAt --supersedes $priorPreviewPath --output $nextPreviewPath
$nextPreviewHash = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $nextPreviewPath --field sha256).Trim()
$nextPreviewVersion = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $nextPreviewPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $nextPreviewPath --expected-sha256 $nextPreviewHash --expected-version $nextPreviewVersion --decision approve --reason-code evidence_quality_reverified --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$nextPreviewHash-preview-evidence-review-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $nextPreviewPath --expected-sha256 $nextPreviewHash --expected-version $nextPreviewVersion --decision approve --reason-code clinical_safety_reverified --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$nextPreviewHash-preview-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $nextPreviewPath --expected-sha256 $nextPreviewHash --expected-version $nextPreviewVersion --decision approve --reason-code closeout_release_governance_reverified --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$nextPreviewHash-preview-release-manager-v1"
$nextReadyContext = "output/closeout/stage-c-next-preview-ready.json"
$nextEvidencePaths = "output/closeout/stage-c-next-preview-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py verify-ready --preview $nextPreviewPath --output $nextReadyContext
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py write-evidence-path-list --pool-entry-context $latestPreviewContext --preview $nextPreviewPath --output $nextEvidencePaths
git add --pathspec-from-file=$nextEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py verify-staged-evidence --path-list $nextEvidencePaths --pool-entry-context $latestPreviewContext --preview $nextPreviewPath
git commit -m "evidence(stage-c): supersede governed ingest preview"
```

If the target, literature run, claim, pool entry, sanitizer, rubric, schema, or governance VersionRef itself drifted, do not reuse this expiry-only flow: restart the owning Task 2-4 review/publication step, create a superseder against those new exact refs, and rerun the complete Stage C gate. Branch and post-merge commands always resolve the one currently effective approved preview path; they never retain the original fixed filename.
- [ ] Run the complete Stage C backend suite:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_evidence_claim_contract.py tests/backend/test_literature_harness_contract.py tests/backend/test_project_evidence_pool_contract.py tests/backend/test_project_evidence_pool_store.py tests/backend/test_evidence_promotion_service.py tests/backend/test_ingest_preview_contract.py tests/backend/test_ingest_preview_store.py tests/backend/test_evidence_isolation_adapters.py tests/backend/test_literature_harness.py tests/backend/test_release_report_contract.py tests/backend/test_release_report_catalog.py tests/backend/test_publish_release_report_candidate.py tests/backend/test_admin_release_dashboard.py tests/backend/test_admin_release_dashboard_api.py tests/backend/test_release_dashboard_flag_drift.py tests/backend/test_research_evidence_view.py tests/backend/test_research_evidence_api.py tests/backend/test_evidence_promotion_non_mutation.py tests/backend/test_stage_c_evidence_gate.py -q -p no:cacheprovider
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_governance_contract.py tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py tests/backend/test_release_monitoring_service.py tests/backend/test_release_closure_service.py tests/backend/test_auth_security.py -q -p no:cacheprovider
```

Expected: all required tests pass with zero skips and no Clinical RAG mutation.

- [ ] Run the exact frontend suite and build; require all pass:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/research/evidence-claim-types.test.ts src/features/research/research-evidence-view.test.tsx src/features/workspace/workspace-surface-switcher.test.tsx src/pages/workspace-page.test.tsx src/features/agent-admin/agent-admin-view.test.tsx src/styles/style-architecture-contract.test.ts --reporter=verbose
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```
- [ ] Resolve the last suite-manifest content commit and the evidence-only descendant now at HEAD, then run branch advisory against that exact pair:

```powershell
$stageCImplementationHead = (git log -1 --format=%H -- config/closeout_stage_suites.yaml).Trim()
$branchHead = (git rev-parse HEAD).Trim()
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage B --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage B merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage B merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $stageCImplementationHead
if ($LASTEXITCODE -ne 0) { throw "Stage C implementation does not descend from approved Stage B" }
$literatureSourceCommit = (git log -1 --format=%H -- scripts/run_literature_harness.py).Trim()
$literatureRunPath = "reports/literature/literature_harness_stage_c_$($literatureSourceCommit.Substring(0, 12)).json"
$candidateSourceCommit = (git log -1 --format=%H -- scripts/publish_release_report_candidate.py).Trim()
$candidatePath = "reports/release_safety/candidates/release_safety_stage_c_$($candidateSourceCommit.Substring(0, 12)).json"
$previewContext = "output/closeout/stage-c-effective-preview.json"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py resolve-latest --target-index config/evidence_indexes/rag_crc_guideline_20260620.json --project-id closeout:crc --output $previewContext --require-effective-approval
$previewPath = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $previewContext --field preview_path).Trim()
$manifestAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind closeout_manifest --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts\run_closeout_stage_gate.py --stage C --mode branch-advisory --base-sha $stageBase --head-sha $branchHead --tested-content-sha $stageCImplementationHead --plan docs\superpowers\plans\2026-07-10-crc-agent-closeout-stage-c-evidence-promotion.md --plan-subject reports\closeout\plan_subjects\stage_c_plan_20260710_001.json --plan-attestation reports\closeout\attestations\stage_c_plan_approval_20260710_001.json --manifest-attestation $manifestAttestationPath --suite config\closeout_stage_suites.yaml --artifact $literatureRunPath --artifact $candidatePath --artifact $previewPath --output-root output\closeout-advisory
```

Expected: every Stage C owned required row passes and inherited Stage A/B reports remain valid. The runner proves every change from `$stageCImplementationHead` to `$branchHead` is one of the exact validated preview/event evidence paths; any source/runtime/config change in that interval blocks.

- [ ] Merge through the protected workflow. From a clean checkout rerun the full Stage C verification, re-resolve the three committed artifact paths as above, and publish against the actual merge SHA:

```powershell
$actualMergeSha = $env:LANGG_STAGE_C_MERGE_SHA
if ($actualMergeSha -notmatch '^[0-9a-f]{40}$') { throw "LANGG_STAGE_C_MERGE_SHA must be the protected workflow's exact merge SHA" }
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage B --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage B merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage B merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $actualMergeSha
if ($LASTEXITCODE -ne 0) { throw "Stage C merge does not descend from approved Stage B" }
git merge-base --is-ancestor $actualMergeSha HEAD
if ($LASTEXITCODE -ne 0) { throw "Stage C merge SHA is not an ancestor of the current evidence head" }
$postEvidenceHead = (git rev-parse HEAD).Trim()
$mergeSha12 = $actualMergeSha.Substring(0, 12)
$literatureSourceCommit = (git log -1 --format=%H -- scripts/run_literature_harness.py).Trim()
$literatureRunPath = "reports/literature/literature_harness_stage_c_$($literatureSourceCommit.Substring(0, 12)).json"
$candidateSourceCommit = (git log -1 --format=%H -- scripts/publish_release_report_candidate.py).Trim()
$candidatePath = "reports/release_safety/candidates/release_safety_stage_c_$($candidateSourceCommit.Substring(0, 12)).json"
$previewContext = "output/closeout/stage-c-post-merge-effective-preview.json"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py resolve-latest --target-index config/evidence_indexes/rag_crc_guideline_20260620.json --project-id closeout:crc --output $previewContext --require-effective-approval
$previewPath = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $previewContext --field preview_path).Trim()
$manifestAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind closeout_manifest --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_stage_gate.py --stage C --mode post-merge --base-sha $stageBase --head-sha $postEvidenceHead --tested-content-sha $actualMergeSha --merged-sha $actualMergeSha --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-c-evidence-promotion.md --plan-subject reports/closeout/plan_subjects/stage_c_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_c_plan_approval_20260710_001.json --manifest-attestation $manifestAttestationPath --suite config/closeout_stage_suites.yaml --artifact $literatureRunPath --artifact $candidatePath --artifact $previewPath --publish
```

The post-merge report ID includes the canonical ordered hash of the literature/candidate/preview path-ref bindings. Every `record-approval` call below revalidates preview liveness and the report's artifact-binding hash. If the preview becomes ineffective after report publication but before gate approval, the call blocks; run the exact preview-superseder workflow above, then publish a report successor without overwriting the first report:

```powershell
$actualMergeSha = $env:LANGG_STAGE_C_MERGE_SHA
if ($actualMergeSha -notmatch '^[0-9a-f]{40}$') { throw "LANGG_STAGE_C_MERGE_SHA must be the protected workflow's exact merge SHA" }
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage B --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage B merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage B merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $actualMergeSha
if ($LASTEXITCODE -ne 0) { throw "Stage C merge does not descend from approved Stage B" }
$postEvidenceHead = (git rev-parse HEAD).Trim()
$priorStageReportPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-published --subject-kind stage_gate --stage C --field subject_path).Trim()
$literatureSourceCommit = (git log -1 --format=%H -- scripts/run_literature_harness.py).Trim()
$literatureRunPath = "reports/literature/literature_harness_stage_c_$($literatureSourceCommit.Substring(0, 12)).json"
$candidateSourceCommit = (git log -1 --format=%H -- scripts/publish_release_report_candidate.py).Trim()
$candidatePath = "reports/release_safety/candidates/release_safety_stage_c_$($candidateSourceCommit.Substring(0, 12)).json"
$previewContext = "output/closeout/stage-c-post-merge-effective-preview.json"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py resolve-latest --target-index config/evidence_indexes/rag_crc_guideline_20260620.json --project-id closeout:crc --output $previewContext --require-effective-approval
$previewPath = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $previewContext --field preview_path).Trim()
$manifestAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind closeout_manifest --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_stage_gate.py --stage C --mode post-merge --base-sha $stageBase --head-sha $postEvidenceHead --tested-content-sha $actualMergeSha --merged-sha $actualMergeSha --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-c-evidence-promotion.md --plan-subject reports/closeout/plan_subjects/stage_c_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_c_plan_approval_20260710_001.json --manifest-attestation $manifestAttestationPath --suite config/closeout_stage_suites.yaml --artifact $literatureRunPath --artifact $candidatePath --artifact $previewPath --supersedes-report $priorStageReportPath --publish
```

The successor keeps merged/tested content fixed, changes the artifact-binding hash, points to the exact prior report, and receives fresh hash-based approvals. Repeat if liveness changes again; old reports/approval attempts remain immutable but cannot satisfy latest-published/latest-approved selection.

- [ ] Reuse Stage A's stage-gate subcommands, collect the three Stage C approvals against the exact report hash/version, and exact-stage only the report/attestation/event chain:

```powershell
$stageReportPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-published --subject-kind stage_gate --stage C --field subject_path).Trim()
$reportHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field sha256).Trim()
$reportVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field subject_version).Trim()
$reportHash12 = $reportHash.Substring(0, 12)
$attestationPath = "reports/closeout/attestations/stage_c_approval.$reportHash12.json"
$gateEvidencePaths = "output/closeout/stage-c-gate-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$reportHash-stage-c-evidence-reviewer-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$reportHash-stage-c-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$reportHash-stage-c-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_gate --subject-path $stageReportPath --output $attestationPath --path-list-output $gateEvidencePaths --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_gate --subject-path $stageReportPath --attestation-path $attestationPath
git add --pathspec-from-file=$gateEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $gateEvidencePaths --subject-path $stageReportPath --attestation-path $attestationPath
git commit -m "evidence(stage-c): record post-merge evidence promotion gate"
```
- [ ] Open Stage D only after the post-merge report, approval, exact manifest counts, source hashes, newest report train, and inherited gates all validate.

## Plan Self-Review Checklist

- [ ] Every Stage C design requirement maps to a task and owned manifest row.
- [ ] Candidate, pool, Clinical RAG, patient default, and doctor default use distinct real-store adapters with blocking failure semantics.
- [ ] The only Stage C output toward Clinical RAG is an immutable `ClinicalRagPromotionCandidate`; no apply/write path exists.
- [ ] IngestPreview expiration, drift, invalidation, supersession, quorum, and non-inheritance are explicitly tested.
- [ ] Release-report discovery is dynamic and the newest invalid candidate blocks despite an older valid report.
- [ ] EvidenceClaim API/workspace and Admin drilldown are read-only and expose no sensitive or action-capable data.
- [ ] Shared Stage A/B types are consumed by exact path; the final-closeout direct-ref composer remains unauthorized until refreshed gates and breaks the candidate/gate cycle without a circular service dependency.
- [ ] Red/green commands and post-merge gate behavior are exact; `git diff --check` and code-fence checks pass.
