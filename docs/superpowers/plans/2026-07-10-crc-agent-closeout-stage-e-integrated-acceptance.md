# CRC Agent Closeout Stage E Integrated Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the non-bypassable final evidence, rollback, acceptance, and authorization layer that decides whether patient and doctor default paths may be released.

**Architecture:** Stage E consumes approved Stage A-D gate payloads and attestations, freezes one release-content commit, generates a fixed HarnessRun, distinct P0 ReleaseSafetyReport, LiteratureHarnessRun, composed ReleaseReportCandidate, final governed preview, and ReleaseBundle, runs full acceptance and rollback rehearsal, and then creates an immutable CloseoutDecisionPayload. FinalApprovalAttestation is created only after the decision hash exists, and existing governance/execution revalidates every bound hash immediately before any feature-flag mutation.

**Tech Stack:** Python 3.10, dataclasses, JSON/SHA-256/HMAC, FastAPI 0.135, Pydantic 2.12, PowerShell, pytest, React 18, TypeScript 5.6, Vitest, Playwright.

## Global Constraints

- Start only after Stage A-D post-merge StageGateReports have valid approval attestations and no compliance block.
- This pre-authored Stage E blueprint is not implementation authorization. After Stage D approval, re-read the actual A-D contracts/artifacts, amend any drift, commit the final Stage E plan blob, build its `StagePlanApprovalSubject`, validate the policy-required `PlanApprovalAttestation` with Stage A's `stage_plan` subcommands, and bind both refs in StageEVerificationReport before Task 1 begins.
- Consume `src/contracts/integrity.py`, `src/contracts/auth_context.py`, `src/services/atomic_artifact_store.py`, `src/services/write_boundary_sanitizer.py`, `src/services/runtime_version_registry.py`, and the inherited StageGate runner; do not redefine shared contracts.
- The release-content commit is immutable during acceptance; later commits may contain path-bounded evidence only.
- Compliance status is `pass|block`; release disposition is `pass|feature_flag|shadow_only|block`, aggregated independently per scope.
- Emergency, policy activation, persistence, privacy/security, authorization, deterministic replay, and rollback requirements allow only `pass|block`.
- A FinalApprovalAttestation is an external post-payload authorization condition, never a self-referential matrix row.
- A report never embeds the commit SHA that first contains that report.
- A ReleaseBundle contains deployable/runtime VersionRefs only; evidence artifacts point to the bundle, never the reverse.
- Required tests may not be skipped. Non-critical skips require a manifest-authorized, expiring waiver and yield at most `feature_flag`.
- No hidden chain-of-thought, secret, credential, direct patient identifier, or matched sensitive value appears in logs or evidence.
- Use `D:\anaconda3\envs\LangG\python.exe` and `D:\anaconda3\envs\LangG\npm.cmd`; preserve unrelated user files and do not modify `CRC-client/`.

## Pre-Implementation Plan Authorization

- [ ] After the Stage D re-read/amendment, commit this exact final plan blob, publish its subject, collect the three policy-required external approvals, and exact-stage the resulting evidence:

```powershell
$planPath = "docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-e-integrated-acceptance.md"
$planSourceCommit = (git log -1 --format=%H -- $planPath).Trim()
git cat-file -e "${planSourceCommit}:$planPath"
$trackedPlanBlob = (git rev-parse "${planSourceCommit}:$planPath").Trim()
$workingPlanBlob = (git hash-object -- $planPath).Trim()
if ($trackedPlanBlob -ne $workingPlanBlob) { throw "Stage E plan blob is not tracked" }
$planSubjectPath = "reports/closeout/plan_subjects/stage_e_plan_20260710_001.json"
$planAttestationPath = "reports/closeout/attestations/stage_e_plan_approval_20260710_001.json"
$planEvidencePaths = "output/closeout/stage-e-plan-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py build-plan-subject --plan $planPath --source-commit $planSourceCommit --author-credential-env LANGG_CLOSEOUT_PUBLISHER_TOKEN --output $planSubjectPath
$planSubjectHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field sha256).Trim()
$planSubjectVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-e-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_DATA_GOVERNANCE_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-e-data-governance-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$planSubjectHash-stage-e-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_plan --subject-path $planSubjectPath --output $planAttestationPath --path-list-output $planEvidencePaths --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_plan --subject-path $planSubjectPath --attestation-path $planAttestationPath
git add --pathspec-from-file=$planEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $planEvidencePaths --subject-path $planSubjectPath --attestation-path $planAttestationPath
git commit -m "evidence(closeout): approve Stage E plan"
```

- [ ] **Hard stop:** do not start Task 1 until the exact staged-set verifier and commit succeed and the subject plan ref/hash/version, approval-policy ref, author exclusion, quorum, ledger head, and latest-plan selection all validate.

## Source Design

- `docs/superpowers/specs/2026-07-10-crc-agent-closeout-program-design.md`, especially Sections 12-18.
- Existing release designs/plans dated 2026-07-02, 2026-07-03, and 2026-07-07 under `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## File Structure

- `src/contracts/closeout.py`: extend Stage A contracts with stage/final payloads, attestations, scope graph, and release bundle.
- `backend/api/services/closeout_store.py`: path-bounded, atomic, append-only closeout artifact store.
- `src/services/closeout_gate.py`: exact-set validation and per-scope aggregation.
- `src/contracts/release_execution.py`: backward-compatible typed ReleaseTarget and multi-target execution results.
- `src/services/release_governance.py`, `src/services/release_execution.py`: final decision/attestation/bundle preflight.
- `scripts/run_closeout_sensitive_artifact_scan.py`: sanitized policy-driven scanner.
- `config/closeout_sensitive_scan.yaml`: scan scopes, rules, and exact fixture exclusions.
- `config/closeout_acceptance_manifest.yaml`: canonical required commands and test paths.
- `config/release_scope_dependencies.yaml`: versioned scope nodes, dependency edges, propagation rules, schema, and hash.
- `scripts/run_e2e_full_acceptance.ps1`: repaired path-preflighting canonical runner.
- `scripts/run_stage_e_closeout.py`: evidence orchestration without self-referential hashes.
- `backend/api/routes/admin.py`, `frontend/src/features/agent-admin/closeout-gate-panel.tsx`: read-only final status.
- `reports/closeout/*`, `reports/release_governance/release_bundles/*`: generated evidence roots.

---

### Task 1: Final Closeout Contracts And Acyclic Hashing

**Files:**
- Modify: `src/contracts/closeout.py`
- Create: `tests/backend/test_closeout_contract.py`
- Create: `tests/backend/test_closeout_hash_order.py`
- Create: `config/release_scope_dependencies.yaml`

**Interfaces:**
- Consumes: `VersionRef`, canonicalization, and `AuditEvent` from `src/contracts/integrity.py`, plus the Stage A manifest and artifact-ID contracts.
- Consumes the complete Stage A `StageGateReport` and `StageGateApprovalAttestation` types without redefining them. Produces `StageArtifactInputBinding`, `StageArtifactBinding`, `ReleaseScopeDependencyManifest`, `StageEVerificationReport`, `StageEVerificationApprovalAttestation`, `CloseoutDecisionPayload`, `FinalApprovalAttestation`, and `EvidenceCommitAttestation`; ReleaseBundle is produced by Task 3.

- [ ] **Step 1: Write failing acyclic-hash and exact-enum tests**

```python
def test_decision_hash_does_not_include_later_approval() -> None:
    payload = CloseoutDecisionPayload.from_rows(valid_header(), [passing_row()])
    decision_hash = payload.sha256
    approval = FinalApprovalAttestation.from_validated_ledger(
        payload.to_version_ref(),
        ledger_with_distinct_authorized_events(payload.to_version_ref()),
        approval_policy=bound_closeout_approval_policy(),
    )
    assert payload.sha256 == decision_hash
    assert approval.decision_payload_sha256 == decision_hash


def test_stage_report_never_contains_its_verification_evidence_commit() -> None:
    report = StageEVerificationReport(**valid_report_fields())
    assert "evidence_commit" not in report.to_dict()
    assert report.tested_release_content_commit == FULL_GIT_SHA
```

- [ ] Add exact-field tests for source-document hashes, expected/actual required-entry counts, Stage E verification attestation ref, final merged commit, release train ID, ReleaseBundle ID/hash, separate P0 ReleaseSafetyReport and composed ReleaseReportCandidate refs, scope-graph VersionRef, duplicate/unknown/missing requirements, and source/count/bundle drift. Reject using a generic `stage_gate` subject kind for a Stage E verification approval, a `stage_e_verification` event against any non-StageEVerificationReport object kind, and any report-ref/hash/version mismatch.

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_contract.py tests/backend/test_closeout_hash_order.py -q -p no:cacheprovider
```

Expected: FAIL because final contracts are absent.

- [ ] **Step 3: Implement immutable payloads with one-way references**

```python
@dataclass(frozen=True)
class StageArtifactInputBinding:
    object_kind: str
    repository_path: str
    artifact_ref: VersionRef


@dataclass(frozen=True)
class StageArtifactBinding:
    stage_id: Literal["A", "B", "C", "D"]
    input_artifacts: tuple[StageArtifactInputBinding, ...]
    collector_id: str | None
    collector_output_path: str | None
    collector_output_ref: VersionRef | None


@dataclass(frozen=True)
class StageEVerificationReport:
    report_id: str
    schema_version: str
    subject_version: str
    author_principal_id: str
    publisher_event_ref: VersionRef
    project_scope: str
    approval_policy_ref: VersionRef
    stage_plan_ref: VersionRef
    stage_plan_subject_ref: VersionRef
    stage_plan_attestation_ref: VersionRef
    manifest_ref: VersionRef
    manifest_attestation_ref: VersionRef
    source_document_refs: tuple[VersionRef, ...]
    source_document_hashes: Mapping[str, str]
    expected_required_entry_count: int
    actual_required_entry_count: int
    tested_release_content_commit: str
    prerequisite_artifact_commit_sha: str
    refreshed_stage_gate_refs: tuple[VersionRef, ...]
    refreshed_stage_gate_attestation_refs: tuple[VersionRef, ...]
    refreshed_stage_artifact_bindings: Mapping[str, StageArtifactBinding]
    candidate_release_inputs_ref: VersionRef
    candidate_release_inputs_sha256: str
    command_results: tuple[StageCommandResult, ...]
    artifact_refs: tuple[VersionRef, ...]
    release_bundle_ref: VersionRef
    release_bundle_id: str
    release_bundle_sha256: str
    rollback_evidence_ref: VersionRef
    rows: tuple[StageGateRow, ...]
    compliance_status: Literal["pass", "block"]
    per_scope_technical_dispositions: Mapping[str, str]
    pre_approval_ledger_head: str
    report_sha256: str


@dataclass(frozen=True)
class EvidenceCommitAttestation:
    attestation_id: str
    stage_e_verification_report_ref: VersionRef
    tested_release_content_commit: str
    evidence_commit_sha: str
    evidence_commit_parent_sha: str
    committed_path_hashes: Mapping[str, str]
    attestation_sha256: str


@dataclass(frozen=True)
class CloseoutDecisionPayload:
    decision_id: str
    schema_version: str
    subject_version: str
    author_principal_id: str
    publisher_event_ref: VersionRef
    project_scope: str
    approval_policy_ref: VersionRef
    manifest_ref: VersionRef
    manifest_attestation_ref: VersionRef
    source_document_refs: tuple[VersionRef, ...]
    source_document_hashes: Mapping[str, str]
    expected_required_entry_count: int
    actual_required_entry_count: int
    stage_gate_refs: tuple[VersionRef, ...]
    stage_gate_attestation_refs: tuple[VersionRef, ...]
    stage_e_verification_ref: VersionRef
    stage_e_verification_attestation_ref: VersionRef
    stage_e_evidence_commit_attestation_ref: VersionRef
    candidate_release_inputs_ref: VersionRef
    candidate_release_inputs_sha256: str
    release_bundle_ref: VersionRef
    release_bundle_id: str
    release_bundle_sha256: str
    release_scope_graph_ref: VersionRef
    rows: tuple[CloseoutGateRow, ...]
    technical_dispositions: Mapping[str, str]
    final_merged_commit: str
    release_train_id: str
    sha256: str


@dataclass(frozen=True)
class FinalApprovalAttestation:
    attestation_id: str
    decision_payload_ref: VersionRef
    decision_payload_sha256: str
    approval_policy_ref: VersionRef
    approver_event_refs: tuple[VersionRef, ...]
    approver_principal_ids: tuple[str, ...]
    approver_credential_ids: tuple[str, ...]
    post_approval_ledger_head: str
    attestation_sha256: str


@dataclass(frozen=True)
class StageEVerificationApprovalAttestation:
    attestation_id: str
    stage_e_verification_report_ref: VersionRef
    approval_policy_ref: VersionRef
    approver_event_refs: tuple[VersionRef, ...]
    approver_principal_ids: tuple[str, ...]
    approver_credential_ids: tuple[str, ...]
    post_approval_ledger_head: str
    attestation_sha256: str
```

`entry_hash = H(canonical entry)`, `manifest_hash = H(header + ordered entry hashes)`, and every approval/commit attestation is calculated only after the referenced payload hash/commit exists.

`config/release_scope_dependencies.yaml` contains exactly `patient_default`, `doctor_default`, `clinical_rag`, `research_workspace`, and `learning_pipeline`, directed edges, propagation rules, schema version, and semantic hash. Unknown scopes, cycles, duplicate edges, illegal propagation, or hash mismatch block before matrix aggregation.

- [ ] **Step 4: Run tests and confirm GREEN**

Run Step 2 command. Expected: PASS, including invalid enum, duplicate scope, cycle, self-reference, same-principal quorum, and unknown schema tests.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/contracts/closeout.py tests/backend/test_closeout_contract.py tests/backend/test_closeout_hash_order.py config/release_scope_dependencies.yaml
git commit -m "feat(closeout): define final release evidence contracts"
```

### Task 2: Closeout Store And Exact-Set Gate Validator

**Files:**
- Create: `backend/api/services/closeout_store.py`
- Modify: `src/services/closeout_gate.py`
- Create: `tests/backend/test_closeout_store.py`
- Create: `tests/backend/test_closeout_gate.py`
- Create: `reports/closeout/README.md`

**Interfaces:**
- Consumes: approved requirement manifest/attestation and Stage A-D gate payloads/attestations.
- Produces: `CloseoutStore.read_latest_approved_manifest()`, `CloseoutGateService.validate_stage_gate()`, `build_decision_payload()`.

- [ ] **Step 1: Write failing missing-row, stale-manifest, and tamper tests**

```python
def test_gate_blocks_missing_required_id(tmp_path: Path) -> None:
    service = closeout_gate(tmp_path, manifest=manifest_with("P0.SAFETY.ACTIVE"))
    with pytest.raises(CloseoutGateBlocked, match="missing requirement: P0.SAFETY.ACTIVE"):
        service.build_decision_payload(
            rows=[],
            technical_inputs=valid_technical_inputs(),
            gate_inputs=valid_final_gate_inputs(),
            publisher_auth=closeout_publisher(),
        )


def test_newer_manifest_invalidates_old_stage_report(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    store.publish_manifest(new_manifest(), new_manifest_attestation())
    with pytest.raises(CloseoutGateBlocked, match="stage report uses stale manifest"):
        closeout_gate_from(store).validate_stage_gate(old_stage_report())


def test_gate_rejects_duplicate_rows_before_ordering() -> None:
    duplicate = passing_row(requirement_id="P0.SAFETY.ACTIVE")
    with pytest.raises(CloseoutGateBlocked, match="duplicate requirement rows"):
        valid_gate_service().build_decision_payload(
            rows=[duplicate, duplicate],
            technical_inputs=valid_technical_inputs(),
            gate_inputs=valid_final_gate_inputs(),
            publisher_auth=closeout_publisher(),
        )


def test_final_unapproved_candidate_requires_matching_refreshed_a_and_c_gates() -> None:
    inputs = final_closeout_inputs_with_unapproved_composed_candidate()
    validate_exact_final_gate_inputs(inputs.manifest, inputs.gate_inputs)
    for broken in (
        inputs.without_refreshed_stage("A"),
        inputs.without_refreshed_stage("C"),
        inputs.with_wrong_candidate_inputs_hash("A"),
        inputs.with_wrong_candidate_inputs_hash("C"),
        inputs.with_ordinary_unapproved_candidate(),
    ):
        with pytest.raises(CloseoutGateBlocked, match="unapproved final candidate gate pair"):
            validate_exact_final_gate_inputs(broken.manifest, broken.gate_inputs)
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_store.py tests/backend/test_closeout_gate.py -q -p no:cacheprovider
```

Expected: FAIL because the store and validator do not exist.

- [ ] **Step 3: Implement atomic roots and exact-set validation**

```python
class CloseoutGateService:
    def build_decision_payload(
        self,
        rows: Sequence[CloseoutGateRow],
        technical_inputs: FinalTechnicalInputs,
        gate_inputs: FinalGateInputs,
        publisher_auth: AuthContext,
    ) -> CloseoutDecisionPayload:
        manifest, attestation = self._store.read_latest_approved_manifest()
        validate_exact_final_gate_inputs(manifest, gate_inputs)
        required_entries = tuple(
            entry for entry in manifest.entries if entry.disposition == "required"
        )
        required = {entry.requirement_id: entry for entry in required_entries}
        actual = {row.requirement_id: row for row in rows}
        if len(rows) != len(actual):
            raise CloseoutGateBlocked("duplicate requirement rows")
        if set(actual) != set(required):
            raise CloseoutGateBlocked(exact_set_error(required, actual))
        ordered_rows = tuple(actual[entry.requirement_id] for entry in required_entries)
        for entry, row in zip(required_entries, ordered_rows, strict=True):
            validate_row_against_entry(row, entry)
        validate_fixed_technical_input_refs(technical_inputs)
        source_hashes = exact_source_hash_map(manifest.source_refs)
        matrix = aggregate_by_scope(manifest, ordered_rows, self._scope_graph())
        technical_dispositions = apply_most_restrictive_per_scope(
            matrix=matrix,
            harness_dispositions=technical_inputs.harness_dispositions,
            release_safety_dispositions=technical_inputs.release_safety_dispositions,
            rollback_preflight=technical_inputs.rollback_preflight,
            target_preflight=technical_inputs.target_preflight,
        )
        stage_order = ("A", "B", "C", "D")
        return CloseoutDecisionPayload.create(
            decision_id=self._ids.next_decision_id(),
            schema_version=CLOSEOUT_DECISION_SCHEMA_VERSION,
            subject_version=self._versions.next_subject_version("closeout_decision"),
            author_principal_id=publisher_auth.principal_id,
            publisher_event_ref=self._record_publisher_event(publisher_auth),
            project_scope="closeout:crc",
            approval_policy_ref=manifest.approval_policy_ref,
            manifest_ref=manifest.to_version_ref(),
            manifest_attestation_ref=attestation.to_version_ref(),
            source_document_refs=manifest.source_refs,
            source_document_hashes=source_hashes,
            expected_required_entry_count=len(required),
            actual_required_entry_count=len(ordered_rows),
            stage_gate_refs=tuple(
                gate_inputs.stage_reports[stage].to_version_ref() for stage in stage_order
            ),
            stage_gate_attestation_refs=tuple(
                gate_inputs.stage_attestations[stage].to_version_ref() for stage in stage_order
            ),
            stage_e_verification_ref=gate_inputs.stage_e_report.to_version_ref(),
            stage_e_verification_attestation_ref=gate_inputs.stage_e_attestation.to_version_ref(),
            stage_e_evidence_commit_attestation_ref=(
                gate_inputs.evidence_commit_attestation.to_version_ref()
            ),
            candidate_release_inputs_ref=(
                gate_inputs.stage_e_report.candidate_release_inputs_ref
            ),
            candidate_release_inputs_sha256=(
                gate_inputs.stage_e_report.candidate_release_inputs_sha256
            ),
            release_bundle_ref=technical_inputs.release_bundle.to_version_ref(),
            release_bundle_id=technical_inputs.release_bundle.bundle_id,
            release_bundle_sha256=technical_inputs.release_bundle.sha256,
            release_scope_graph_ref=self._scope_graph().to_version_ref(),
            rows=ordered_rows,
            technical_dispositions=technical_dispositions,
            final_merged_commit=gate_inputs.stage_e_report.tested_release_content_commit,
            release_train_id=technical_inputs.release_bundle.release_train_id,
        )
```

```python
@dataclass(frozen=True)
class FinalGateInputs:
    stage_reports: Mapping[Literal["A", "B", "C", "D"], StageGateReport]
    stage_attestations: Mapping[Literal["A", "B", "C", "D"], StageGateApprovalAttestation]
    stage_artifact_bindings: Mapping[Literal["A", "B", "C", "D"], StageArtifactBinding]
    stage_e_report: StageEVerificationReport
    stage_e_attestation: StageEVerificationApprovalAttestation
    evidence_commit_attestation: EvidenceCommitAttestation
```

`validate_exact_final_gate_inputs()` requires exactly the A/B/C/D keys once each across reports, attestations, and artifact bindings; validates every report's stage ID, latest manifest/plan/policy/source refs, merged commit, report hash, artifact refs, orchestration-input ref, combined gate-input hash, and no-block status; and requires each attestation's report ref to equal its paired report ref. The supplied binding for each stage must equal `StageEVerificationReport.refreshed_stage_artifact_bindings[stage]`, every refreshed report's orchestration input must equal the Stage E report's CandidateReleaseInputs ref, and the report's artifact set must equal that binding's validated input refs plus its declared collector output refs. It then validates the Stage E report/approval pair and EvidenceCommitAttestation against the same frozen content, manifest, ReleaseBundle, report hash, and verification-evidence commit. Missing, duplicate, wrong-stage, cross-paired, stale, superseded, expired-preview, artifact-path/ref mismatch, or policy-drifted input blocks. `exact_source_hash_map()` derives a deterministic locator key and SHA-256 from each `manifest.source_refs` VersionRef and rejects duplicate locators; it does not rely on an undeclared manifest field. `build_decision_payload()` orders unique rows by `manifest.entries`, copies the resulting exact VersionRefs into `CloseoutDecisionPayload`, and rejects duplicate input rows before aggregation; rows alone can never stand in for gate artifacts.

There is one narrow treatment for the deliberately unauthorized composed candidate. `authorization_status=not_approved` does not lower the technical disposition only when all of these hold simultaneously: `composition_mode=final_closeout_unapproved`; its final manifest/attestation, exact three typed evidence refs, source commit, publisher event, path, and hash equal the effective CandidateReleaseInputs mapping; the candidate ref is the exact Stage C bound input; refreshed Stage A validates the same HarnessRun/P0 pair; refreshed Stage C validates the same LiteratureHarnessRun/candidate/final preview; both A and C reports are current, no-block, externally attested, and bind the same CandidateReleaseInputs ref/hash and frozen release-content commit. In that case the matching refreshed A+C attestations are the external authorization supplied after composition. Any ordinary-train `not_approved`, missing or stale A/C pair, differing candidate-input hash, wrong artifact binding, or attempt to reuse an earlier gate blocks. The candidate is never rewritten to `approved`, and neither A nor C alone is sufficient.

`FinalTechnicalInputs` binds the fixed HarnessRun, P0 ReleaseSafetyReport, LiteratureHarnessRun, composed ReleaseReportCandidate (including composition/authorization status and exact basis refs), rollback rehearsal, ReleaseBundle target/kill-switch/health preflight, and their VersionRefs/hashes for every applicable scope. The P0 safety report and composed candidate are distinct object kinds and paths; neither may substitute for the other. Add parameterized tests where exactly one input is tightened to each of `feature_flag`, `shadow_only`, or `block`; the final scope result must tighten even when all matrix rows pass. Missing, stale, mismatched, or cross-bundle input refs block. Add exact-set tests for every `FinalGateInputs` failure listed above, including a Stage C preview path swapped after refresh, a D collector output ref absent from its binding, and every failed condition in the narrow final-unapproved-candidate rule.

`CloseoutStore` composes Stage A's `AtomicJsonArtifactPublisher` and artifact-ID validator; it must not fork their path or publication logic. A logical multi-object mutation is one aggregate event artifact or a durable SQLite outbox operation—two independent file publishes are never described as atomic. Reads block on unknown schema, duplicate sequence, broken chain, reparse escape, or unreadable candidate.

- [ ] **Step 4: Run tests and confirm GREEN**

Run Step 2 command. Expected: PASS, including source drift, illegal waiver, critical skip, duplicate ID, unknown scope, cyclic scope graph, and audit-chain break tests.

- [ ] **Step 5: Commit Task 2**

```powershell
git add backend/api/services/closeout_store.py src/services/closeout_gate.py tests/backend/test_closeout_store.py tests/backend/test_closeout_gate.py reports/closeout/README.md
git commit -m "feat(closeout): validate exact release evidence sets"
```

### Task 3: ReleaseBundle And Typed ReleaseTarget Registry

**Files:**
- Modify: `src/contracts/closeout.py`
- Modify: `src/contracts/release_execution.py`
- Create: `backend/api/services/release_bundle_store.py`
- Modify: `backend/api/services/release_execution_store.py`
- Modify: `src/services/release_execution.py`
- Modify: `tests/backend/test_release_execution_contract.py`
- Modify: `tests/backend/test_release_execution_store.py`
- Modify: `tests/backend/test_release_execution_service.py`
- Create: `tests/backend/test_release_bundle_store.py`
- Create: `reports/release_governance/release_bundles/README.md`

**Interfaces:**
- Consumes: deployable/runtime VersionRefs, the immutable Stage A `RuntimeVersionSnapshot` with labelled bindings, explicit ReleaseTargets, and their health/kill-switch state refs. It does not consume a gate result, candidate authorization, or final technical disposition.
- Produces: immutable ReleaseBundle, `ReleaseTargetRegistry`, multi-target preflight/kill switch, and execution result states `not_applied|applied|unknown_or_partial`.

- [ ] **Step 1: Write failing multi-target and rollback rehearsal tests**

```python
def test_bundle_rejects_target_without_kill_switch() -> None:
    with pytest.raises(ValueError, match="verified kill switch is required"):
        ReleaseBundle.create(components(), targets=[target(kill_switch_ref=None)])


def test_partial_execution_freezes_scope_and_runs_kill_switch(tmp_path: Path) -> None:
    executor = execution_service(tmp_path, adapter=partial_adapter())
    result = executor.execute_release(valid_request())
    assert result.observed_status == "unknown_or_partial"
    assert result.release_disposition == "block"
    assert executor.registry.read("doctor_review_cockpit_v0").enabled is False
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_execution_contract.py tests/backend/test_release_bundle_store.py tests/backend/test_release_execution_store.py tests/backend/test_release_execution_service.py -q -p no:cacheprovider
```

Expected: FAIL because execution is hard-coded to one feature flag and two result states.

- [ ] **Step 3: Implement backward-compatible typed targets**

```python
@dataclass(frozen=True)
class ReleaseBundle:
    bundle_id: str
    schema_version: str
    release_train_id: str
    code_commit: str
    environment_lock_ref: VersionRef
    schema_compatibility_range: str
    policy_refs: tuple[VersionRef, ...]
    evidence_index_refs: tuple[VersionRef, ...]
    prompt_rubric_refs: tuple[VersionRef, ...]
    runtime_snapshot_ref: VersionRef
    runtime_version_bindings: tuple[RuntimeVersionBinding, ...]
    feature_flag_snapshot_ref: VersionRef
    artifact_manifest: tuple[VersionRef, ...]
    targets: tuple[ReleaseTarget, ...]
    sha256: str


@dataclass(frozen=True)
class ReleaseTarget:
    target_id: str
    target_type: Literal["feature_flag"]
    scope: str
    state_ref: VersionRef
    health_check_ref: VersionRef
    kill_switch_ref: VersionRef
    required_runtime_slots: tuple[str, ...]


class ReleaseTargetRegistry:
    def apply(
        self,
        target: ReleaseTarget,
        enabled: bool,
        expected_state_ref: VersionRef,
        idempotency_key: str,
    ) -> TargetExecutionObservation:
        adapter = self._adapters[target.target_id]
        return adapter.set_enabled(
            enabled=enabled,
            expected_state_ref=expected_state_ref,
            idempotency_key=idempotency_key,
        )
```

Every ReleaseBundle field is deployable/runtime state; it does not reference the requirement manifest, StageGateReport, StageEVerificationReport, decision, approval, technical disposition, or other evidence kinds. The bundle is frozen before refreshed gates; later technical inputs, gate reports, and the final decision point one-way to that exact bundle and may tighten disposition, never the reverse. Add a contract test that constructs the bundle before any final gate/decision exists and rejects any attempted evidence/disposition field. `ReleaseBundleStore` uses the shared create-once publisher under `reports/release_governance/release_bundles/` and validates every component hash/source commit before publication. It also validates the runtime snapshot hash/ledger head, requires exactly one labelled binding for every target-required slot, and rejects unlabelled, duplicate, unknown, missing, or active/shadow-swapped refs. Register `doctor_review_cockpit_v0` through the new adapter so existing behavior stays compatible; additional targets must be explicitly registered and preflighted.

- [ ] **Step 4: Run tests and confirm GREEN**

Run Step 2 command. Expected: PASS, including duplicate execution, partial write, failed kill switch, rollback retry, external recovery, no-evidence-in-bundle, runtime-slot drift, snapshot-ledger drift, and active/shadow swap tests.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/contracts/closeout.py src/contracts/release_execution.py backend/api/services/release_bundle_store.py backend/api/services/release_execution_store.py src/services/release_execution.py tests/backend/test_release_execution_contract.py tests/backend/test_release_execution_store.py tests/backend/test_release_execution_service.py tests/backend/test_release_bundle_store.py reports/release_governance/release_bundles/README.md
git commit -m "feat(release): add rollback-capable release bundles"
```

### Task 4: Final Governance And Execution Preflight

**Files:**
- Modify: `src/contracts/release_governance.py`
- Modify: `src/services/release_governance.py`
- Modify: `src/services/release_execution.py`
- Modify: `backend/api/schemas/release_governance.py`
- Modify: `backend/api/schemas/release_execution.py`
- Modify: `backend/api/routes/admin.py`
- Modify: `tests/backend/test_release_governance_service.py`
- Modify: `tests/backend/test_release_execution_service.py`
- Modify: `tests/backend/test_release_governance_api.py`
- Modify: `tests/backend/test_release_execution_api.py`
- Modify: `tests/backend/test_auth_security.py`

**Interfaces:**
- Consumes: latest CloseoutDecisionPayload, FinalApprovalAttestation, highest non-superseded CandidateReleaseInputs and its final IngestPreview approval chain, ReleaseBundle, current RuntimeVersionRegistry snapshot, AuthContext, monitoring state, and an injected UTC clock.
- Produces: create-intent/execute preflight that rejects Stage A-D evidence and cannot raise the technical disposition ceiling.

- [ ] **Step 1: Write failing final-authorization tests**

```python
def test_governance_rejects_stage_report_without_final_payload() -> None:
    with pytest.raises(GovernanceValidationError, match="final closeout decision is required"):
        governance_service(stage_only_store()).create_intent(
            target_scope="doctor_default",
            requested_disposition="feature_flag",
            status="pending_approval",
            reason="release",
            auth=release_manager(),
            expected_governance_version="governance-v3",
            idempotency_key="intent-doctor-default-001",
        )


def test_execution_revalidates_attestation_immediately_before_write() -> None:
    service = execution_service_with_revoked_attestation()
    with pytest.raises(ReleaseExecutionPreflightError, match="final approval attestation is invalid"):
        service.execute_release(**valid_execute_args())
    assert service.target_state("doctor_review_cockpit_v0").enabled is False


def test_execution_blocks_active_slot_drift_after_final_approval() -> None:
    service = execution_service_with_runtime_slot_drift("safety_policy:patient_crc_triage:active")
    with pytest.raises(ReleaseExecutionPreflightError, match="runtime slot binding drift"):
        service.execute_release(**valid_execute_args())
    assert service.target_state("doctor_review_cockpit_v0").enabled is False


@pytest.mark.parametrize("preview_state", ["expired", "revoked", "superseded"])
def test_intent_and_execution_revalidate_final_preview_liveness_without_write(
    preview_state: str,
) -> None:
    service = release_service_with_final_preview_state(preview_state)
    with pytest.raises(GovernanceValidationError, match="final preview is not effective"):
        service.create_intent(**valid_intent_args())
    with pytest.raises(ReleaseExecutionPreflightError, match="final preview is not effective"):
        service.execute_release(**valid_execute_args())
    assert service.intent_store_writes == []
    assert service.target_adapter_writes == []


def test_preview_revocation_between_target_writes_blocks_and_rolls_back() -> None:
    service = multi_target_service(revoke_preview_before_target=2)
    result = service.execute_release(**valid_execute_args())
    assert result.release_disposition == "block"
    assert service.target_adapter_writes[1].operation == "kill_switch"
    assert all(target.enabled is False for target in service.current_target_states())
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py tests/backend/test_release_governance_api.py tests/backend/test_release_execution_api.py tests/backend/test_auth_security.py -q -p no:cacheprovider
```

Expected: FAIL because current preflight only checks P0/literature dashboard state and trusts client identity fields.

- [ ] **Step 3: Implement final decision and server-principal enforcement**

```python
def create_intent(
    self,
    *,
    target_scope: ReleaseScope,
    requested_disposition: ReleaseDisposition,
    status: str,
    reason: str,
    auth: AuthContext,
    expected_governance_version: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_roles(auth, "release_manager")
    decision, attestation = self._closeout_loader()
    validate_final_authorization(decision, attestation, auth)
    technical_ceiling = decision.technical_disposition(target_scope)
    if disposition_rank(requested_disposition) < disposition_rank(technical_ceiling):
        raise GovernanceValidationError("governance intent exceeds technical disposition")
    return self._create_from_final_decision(
        decision,
        auth,
        target_scope,
        requested_disposition,
        status,
        reason,
        expected_governance_version,
        idempotency_key,
    )
```

`disposition_rank` uses `block=3`, `shadow_only=2`, `feature_flag=1`, and `pass=0`; a request is illegal when it is less restrictive than the technical ceiling. Tests cover less/equal/more restrictive requests for every scope. API schema tests reject `requested_by`, `approver_role`, and other authority-bearing client fields rather than passing them to the service. Same idempotency key/same payload returns the original intent; a changed payload or stale expected version conflicts before any write. Approvals use distinct `principal_id` and `credential_id`; shared/legacy admin credentials cannot satisfy quorum.

`validate_final_authorization()` recursively reloads the highest non-superseded CandidateReleaseInputs from the bounded closeout store, requires its ref/hash to equal the decision and StageEVerificationReport bindings, then resolves the exact final Stage C IngestPreview and revalidates its expiry against the injected current UTC time, revocation/supersession chain, target/literature/promotion refs, approval policy, approval-event ledger head, and A/C refreshed-gate bindings. It never trusts the snapshot embedded in an old decision as current liveness. Call this validation immediately before intent persistence, at the start of `execute_release` before any adapter call, and again immediately before each individual target write together with the final decision/attestation, monitoring state, bundle hash, and fresh runtime snapshot. An expired, revoked, superseded, missing, ambiguous, or hash-drifted preview blocks with zero writes when observed at intent/start preflight and requires a new preview, CandidateReleaseInputs sequence, refreshed A-D/Stage E evidence, decision, and approvals. If liveness changes between target writes, do not write the next target; invoke the declared kill switches/rollback for any already applied target, persist the blocked/unknown observation, and leave every target disabled before retry.

Immediately before intent creation and again immediately before every target write, resolve a fresh `RuntimeVersionSnapshot` and require exact slot/ref/activation-event equality with the ReleaseBundle bindings and snapshot ledger lineage; any post-approval activation, rollback, missing slot, or active/shadow swap blocks without mutation and requires a new bundle/verification/decision chain.

- [ ] **Step 4: Run tests and confirm GREEN**

Run Step 2 command. Expected: PASS, including stale manifest, changed bundle, monitoring block, flag drift, same-principal double-sign, expired/revoked/superseded final preview, between-target liveness loss with safe rollback, and no-write-on-start-preflight-failure cases.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/contracts/release_governance.py src/services/release_governance.py src/services/release_execution.py backend/api/schemas/release_governance.py backend/api/schemas/release_execution.py backend/api/routes/admin.py tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py tests/backend/test_release_governance_api.py tests/backend/test_release_execution_api.py tests/backend/test_auth_security.py
git commit -m "feat(release): require final closeout authorization"
```

### Task 5: Sensitive Artifact Scanner

**Files:**
- Create: `config/closeout_sensitive_scan.yaml`
- Create: `scripts/run_closeout_sensitive_artifact_scan.py`
- Create: `tests/backend/test_closeout_sensitive_artifact_scan.py`

**Interfaces:**
- Consumes: production paths, exact synthetic-fixture path/hash exclusions, and the shared sanitizer/scan-policy VersionRefs.
- Produces: a VersionRef-bound sanitized scan result containing only rule IDs, counts, and safe locations.

- [ ] **Step 1: Write failing production-hit and fixture-exclusion tests**

```python
def test_scanner_blocks_secret_without_echoing_value(tmp_path: Path) -> None:
    secret = "SYNTHETIC_SECRET_VALUE"
    (tmp_path / "reports" / "bad.json").write_text(json.dumps({"token": secret}))
    result = scan(tmp_path, policy())
    assert result.status == "block"
    assert secret not in json.dumps(result.to_dict())
    assert result.findings[0].rule_id == "credential_value"


def test_exact_synthetic_fixture_hash_is_excluded_but_import_is_not(tmp_path: Path) -> None:
    fixture = write_approved_fixture(tmp_path)
    assert scan(tmp_path, policy_for(fixture)).status == "pass"
    copy_fixture_into_production(fixture, tmp_path / "src" / "copied.py")
    assert scan(tmp_path, policy_for(fixture)).status == "block"
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_sensitive_artifact_scan.py -q -p no:cacheprovider
```

Expected: FAIL because the scanner does not exist.

- [ ] **Step 3: Implement policy-driven sanitized scanning**

```python
def scan_file(path: Path, policy: ScanPolicy) -> list[SanitizedFinding]:
    if policy.is_exact_fixture_exclusion(path, sha256_file(path)):
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        SanitizedFinding(rule_id=rule.rule_id, relative_path=relative(path), line=line_number)
        for rule in policy.rules
        for line_number in matching_lines(text, rule.pattern)
    ]
```

Mandatory roots include production code/config, persisted review/research/learning/release artifacts, and generated closeout reports. The scanner must never emit match text or low-entropy hashes.

- [ ] **Step 4: Run tests and scan the current tree**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_sensitive_artifact_scan.py -q -p no:cacheprovider
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_sensitive_artifact_scan.py --policy config/closeout_sensitive_scan.yaml --output output/closeout/sensitive-scan.json
```

Expected: test PASS; scanner exits 0 or reports only explicitly classified current findings that must be fixed before committing this task.

- [ ] **Step 5: Commit Task 5**

```powershell
git add config/closeout_sensitive_scan.yaml scripts/run_closeout_sensitive_artifact_scan.py tests/backend/test_closeout_sensitive_artifact_scan.py
git commit -m "test(closeout): add sanitized sensitive artifact scan"
```

### Task 6: Repair The Canonical Acceptance Runner

**Files:**
- Modify: `.gitignore`
- Create: `config/closeout_acceptance_manifest.yaml`
- Create: `scripts/validate_closeout_acceptance_manifest.py`
- Create: `scripts/run_closeout_deterministic_replay.py`
- Create: `scripts/validate_closeout_git_hygiene.py`
- Create: `scripts/run_stage_e_closeout.py`
- Modify: `scripts/run_e2e_full_acceptance.ps1`
- Create: `docs/superpowers/acceptance/e2e-full-acceptance-runbook.md`
- Create: `tests/backend/test_closeout_acceptance_manifest.py`
- Create: `tests/backend/test_closeout_deterministic_replay.py`
- Create: `tests/backend/test_closeout_git_hygiene.py`
- Create: `tests/backend/test_stage_e_closeout.py`
- Create: `tests/e2e/acceptance/patient-crc-closeout.spec.ts`
- Create: `tests/e2e/acceptance/doctor-review-closeout.spec.ts`
- Create: `tests/e2e/acceptance/evidence-admin-closeout.spec.ts`
- Create: `tests/e2e/acceptance/research-learning-closeout.spec.ts`

**Interfaces:**
- Consumes: final requirement manifest and existing test/build/harness/scanner entry points.
- Produces: path-preflighted runner; `-ListOnly` remains diagnostic and cannot generate pass evidence.

- [ ] **Step 1: Write failing stale-path validation tests**

```python
def test_current_legacy_paths_are_rejected() -> None:
    errors = validate_paths(repo_root(), legacy_runner_manifest())
    assert "tests/backend/test_payload_builder.py" in errors
    assert "tests/e2e/acceptance/workspace-core.spec.ts" in errors


def test_final_manifest_paths_exist() -> None:
    assert validate_paths(repo_root(), load_manifest("config/closeout_acceptance_manifest.yaml")) == []


def test_canonical_runbook_has_exact_unignore_contract() -> None:
    runbook = "docs/superpowers/acceptance/e2e-full-acceptance-runbook.md"
    assert (repo_root() / runbook).is_file()
    assert git_check_ignore(repo_root(), runbook).returncode == 1
    assert exact_acceptance_runbook_exceptions(repo_root() / ".gitignore")
```

- [ ] **Step 2: Run test and old runner preflight to confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_acceptance_manifest.py -q -p no:cacheprovider
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -ListOnly
```

Expected: pytest FAIL because validator/manifest do not exist; old runner output names nonexistent paths and is not acceptance evidence.

- [ ] **Step 3: Implement manifest preflight and real commands**

```yaml
schema_version: closeout_acceptance_v1
required_commands:
  - id: backend_full
    executable: D:\anaconda3\envs\LangG\python.exe
    args: [-m, pytest, -q, -p, no:cacheprovider]
  - id: frontend_full
    executable: D:\anaconda3\envs\LangG\npm.cmd
    args: [--prefix, frontend, run, test, --, --run]
  - id: frontend_build
    executable: D:\anaconda3\envs\LangG\npm.cmd
    args: [--prefix, frontend, run, build]
  - id: patient_crc_e2e
    executable: D:\anaconda3\envs\LangG\npm.cmd
    args: [--prefix, frontend, run, test:e2e:acceptance, --, ../tests/e2e/acceptance/patient-crc-closeout.spec.ts]
  - id: doctor_review_e2e
    executable: D:\anaconda3\envs\LangG\npm.cmd
    args: [--prefix, frontend, run, test:e2e:acceptance, --, ../tests/e2e/acceptance/doctor-review-closeout.spec.ts]
  - id: evidence_admin_e2e
    executable: D:\anaconda3\envs\LangG\npm.cmd
    args: [--prefix, frontend, run, test:e2e:acceptance, --, ../tests/e2e/acceptance/evidence-admin-closeout.spec.ts]
  - id: research_learning_e2e
    executable: D:\anaconda3\envs\LangG\npm.cmd
    args: [--prefix, frontend, run, test:e2e:acceptance, --, ../tests/e2e/acceptance/research-learning-closeout.spec.ts]
  - id: frontend_regression_e2e
    executable: D:\anaconda3\envs\LangG\npm.cmd
    args: [--prefix, frontend, run, test:e2e:acceptance, --, ../tests/e2e/acceptance/frontend-regression-contracts.spec.ts]
  - id: sensitive_scan
    executable: D:\anaconda3\envs\LangG\python.exe
    args: [scripts/run_closeout_sensitive_artifact_scan.py, --policy, config/closeout_sensitive_scan.yaml, --output, output/closeout/sensitive-scan.json]
  - id: deterministic_replay
    executable: D:\anaconda3\envs\LangG\python.exe
    args: [scripts/run_closeout_deterministic_replay.py, --context, output/closeout/frozen-inputs.json, --output-root, output/closeout/replay]
  - id: rollback_rehearsal
    executable: D:\anaconda3\envs\LangG\python.exe
    args: [scripts/run_stage_e_closeout.py, rehearse-rollback, --context, output/closeout/frozen-inputs.json]
  - id: diff_check
    executable: D:\anaconda3\envs\LangG\python.exe
    args: [scripts/validate_closeout_git_hygiene.py, --context, output/closeout/frozen-inputs.json]
required_paths:
  - tests/e2e/acceptance/frontend-regression-contracts.spec.ts
  - tests/e2e/acceptance/patient-crc-closeout.spec.ts
  - tests/e2e/acceptance/doctor-review-closeout.spec.ts
  - tests/e2e/acceptance/evidence-admin-closeout.spec.ts
  - tests/e2e/acceptance/research-learning-closeout.spec.ts
```

`frontend_build` is the required TypeScript (`tsc --noEmit`) plus production Vite build because the checked-in package script composes both. `frozen-inputs.json` is generated by Stage E from validated VersionRefs and contains no secret or patient data. The deterministic replay command regenerates the P0 HarnessRun/ReleaseSafetyReport and LiteratureHarnessRun into a temporary root, rebuilds the composed ReleaseReportCandidate from those exact refs, validates all schemas, compares semantic hashes to the frozen inputs, and never overwrites committed evidence. The rollback command rehearses the exact ReleaseBundle target and verifies recovery state. The hygiene validator executes `git diff --check "$programBase..$releaseContentCommit"` after resolving both full SHAs from the validated baseline/frozen context, checks staged/unstaged state, and permits only the baseline record's exact user-owned exclusions; it does not accept an arbitrary ignore pattern.

The PowerShell runner must call the validator first, stop on missing paths, execute all four E2E flows separately, capture exit code/timestamps/tool versions/sanitized output hashes, reject `-ListOnly` when `-WriteEvidence` is requested, reject every required skip, and run the scanner/replays/report validation/rollback/diff checks. It also runs `git status --porcelain=v1 --untracked-files=all` and permits only the baseline-manifest's exact user-owned exclusions; any other staged, modified, or untracked path blocks.

Add exactly these narrow `.gitignore` rules in Task 6 so the canonical runbook and its parent become visible only in the same task that stages them; this avoids exposing a pre-existing ignored working copy to earlier clean-tree gates and does not expose unrelated documentation:

```gitignore
!docs/superpowers/acceptance/
docs/superpowers/acceptance/*
!docs/superpowers/acceptance/e2e-full-acceptance-runbook.md
```

The contract test rejects a broad `docs/superpowers/**` exception, requires the exact file to exist and be non-ignored, and the post-commit check below proves the path is retrievable from the new commit rather than merely present in one worktree.

- [ ] **Step 4: Run manifest/unit validation and list-only preflight**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_acceptance_manifest.py -q -p no:cacheprovider
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_deterministic_replay.py tests/backend/test_closeout_git_hygiene.py tests/backend/test_stage_e_closeout.py -q -p no:cacheprovider
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -ListOnly
```

Expected: schemas/unit tests pass and list-only preflights every executable/static path without running acceptance or requiring `frozen-inputs.json`. Task 8 is the sole `-WriteEvidence` execution point after the frozen context and prerequisite evidence commit exist.

- [ ] **Step 5: Commit Task 6**

```powershell
git add .gitignore config/closeout_acceptance_manifest.yaml scripts/validate_closeout_acceptance_manifest.py scripts/run_closeout_deterministic_replay.py scripts/validate_closeout_git_hygiene.py scripts/run_stage_e_closeout.py scripts/run_e2e_full_acceptance.ps1 docs/superpowers/acceptance/e2e-full-acceptance-runbook.md tests/backend/test_closeout_acceptance_manifest.py tests/backend/test_closeout_deterministic_replay.py tests/backend/test_closeout_git_hygiene.py tests/backend/test_stage_e_closeout.py tests/e2e/acceptance/patient-crc-closeout.spec.ts tests/e2e/acceptance/doctor-review-closeout.spec.ts tests/e2e/acceptance/evidence-admin-closeout.spec.ts tests/e2e/acceptance/research-learning-closeout.spec.ts
git commit -m "test(closeout): repair canonical full acceptance runner"
$task6Commit = (git rev-parse HEAD).Trim()
git cat-file -e "${task6Commit}:docs/superpowers/acceptance/e2e-full-acceptance-runbook.md"
if ($LASTEXITCODE -ne 0) { throw "canonical acceptance runbook is not tracked in Task 6 commit" }
```

### Task 7: Final Closeout API And Read-Only Admin Panel

**Files:**
- Create: `backend/api/schemas/closeout.py`
- Modify: `backend/api/routes/admin.py`
- Modify: `backend/app.py`
- Create: `tests/backend/test_closeout_api.py`
- Modify: `tests/backend/test_auth_security.py`
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Create: `frontend/src/features/agent-admin/closeout-gate-panel.tsx`
- Create: `frontend/src/features/agent-admin/closeout-gate-panel.test.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-pages.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.test.tsx`

**Interfaces:**
- Consumes: validated latest decision payload/attestation and ReleaseBundle.
- Produces: `GET /api/admin/closeout`, `ApiClient.getAdminCloseout()`, and a read-only panel with per-scope decisions and evidence drift.

- [ ] **Step 1: Write failing API/UI tests**

```python
def test_closeout_api_blocks_invalid_latest_payload() -> None:
    response = closeout_client(tampered_store()).get("/api/admin/closeout", headers=admin_headers())
    assert response.status_code == 409
    assert response.json()["detail"] == "closeout evidence integrity failed"
```

```tsx
it("shows per-scope technical and authorization status", async () => {
  render(<CloseoutGatePanel resource={closeoutResource()} />);
  expect(screen.getByText("patient_default: feature_flag")).toBeInTheDocument();
  expect(screen.getByText("authorization: not authorized")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /release/i })).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_api.py tests/backend/test_auth_security.py -q -p no:cacheprovider
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/agent-admin/closeout-gate-panel.test.tsx src/features/agent-admin/agent-admin-view.test.tsx src/app/api/client.test.ts
```

Expected: FAIL because the endpoint/types/panel do not exist.

- [ ] **Step 3: Implement read-only API and UI**

```python
@router.get("/closeout", response_model=AdminCloseoutResponse)
async def read_closeout(request: Request) -> dict[str, Any]:
    require_roles(request.state.auth_context, "release_manager")
    return closeout_service().read_validated_status()
```

```tsx
type CloseoutResource =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: AdminCloseoutResponse };

interface CloseoutGatePanelProps {
  resource: CloseoutResource;
}

export function CloseoutGatePanel({ resource }: CloseoutGatePanelProps) {
  if (resource.status === "loading") return <p role="status">正在加载收口状态</p>;
  if (resource.status === "error") return <p role="alert">{resource.message}</p>;
  return (
    <table aria-label="按范围的收口决策">
      <tbody>
        {resource.data.scope_decisions.map((decision) => (
          <tr key={decision.scope}>
            <th>{decision.scope}</th>
            <td>{decision.technical_disposition}</td>
            <td>{decision.authorization_status}</td>
            <td>{decision.drift_status}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 4: Run tests and build**

Run Step 2 commands, then:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: tests PASS and build exits 0.

- [ ] **Step 5: Commit Task 7**

```powershell
git add backend/api/schemas/closeout.py backend/api/routes/admin.py backend/app.py tests/backend/test_closeout_api.py tests/backend/test_auth_security.py frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/features/agent-admin/closeout-gate-panel.tsx frontend/src/features/agent-admin/closeout-gate-panel.test.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx
git commit -m "feat(admin): expose final closeout gate status"
```

### Task 8: Stage E Orchestration, Source Reconciliation, And Final Evidence

**Files:**
- Modify: `src/contracts/closeout.py`
- Modify: `src/services/closeout_stage_runner.py`
- Modify: `scripts/run_stage_e_closeout.py`
- Modify: `tests/backend/test_stage_e_closeout.py`
- Modify during the Stage E branch before freeze: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`
- Create after final-manifest publication: `reports/closeout/requirements/closeout_requirements_final_<release-sha12>.json`
- Create after final-manifest approval: `reports/closeout/attestations/manifest_approval.<manifest-hash12>.json`
- Create after candidate publication: `reports/harness/harness_closeout_final_<release-sha12>.json`
- Create after candidate publication: `reports/literature/literature_harness_closeout_final_<release-sha12>.json`
- Create after P0 replay: `reports/release_safety/release_safety_closeout_final_<release-sha12>.json`
- Create after composed-candidate publication: `reports/release_safety/candidates/release_report_closeout_final_<release-sha12>.json`
- Create after candidate publication: `reports/release_governance/release_bundles/release_bundle_closeout_final_<release-sha12>.json`
- Create after final literature publication: `reports/evidence_pool/previews/ingest_preview_closeout_final_<sequence>_<literature-sha12>.json`
- Create through final-preview approval: `reports/evidence_pool/events/<assigned-sequence>.json`
- Create before refreshed gates: `reports/closeout/stage_inputs/stage_d_evidence.<release-sha12>.json`
- Create before refreshed gates: `reports/closeout/candidate_inputs/candidate_release_inputs.<sequence>.<inputs-hash12>.json`
- Create after refreshed-gate publication: `reports/closeout/stages/stage_a.<release-sha12>.<candidate-inputs-hash12>.json`
- Create after refreshed-gate publication: `reports/closeout/stages/stage_b.<release-sha12>.<candidate-inputs-hash12>.json`
- Create after refreshed-gate publication: `reports/closeout/stages/stage_c.<release-sha12>.<candidate-inputs-hash12>.json`
- Create after refreshed-gate publication: `reports/closeout/stages/stage_d.<release-sha12>.<candidate-inputs-hash12>.json`
- Create after refreshed-gate approval: `reports/closeout/attestations/stage_a_approval.<report-hash12>.json`
- Create after refreshed-gate approval: `reports/closeout/attestations/stage_b_approval.<report-hash12>.json`
- Create after refreshed-gate approval: `reports/closeout/attestations/stage_c_approval.<report-hash12>.json`
- Create after refreshed-gate approval: `reports/closeout/attestations/stage_d_approval.<report-hash12>.json`
- Create after Stage E verification: `reports/closeout/stages/stage_e.<release-sha12>.<candidate-inputs-hash12>.json`
- Create after verification-evidence commit: `reports/closeout/attestations/evidence_commit.<stage-e-report-hash12>.json`
- Create after Stage E approval: `reports/closeout/attestations/stage_e_approval.<stage-e-report-hash12>.json`
- Create after decision publication: `reports/closeout/decisions/closeout_decision_20260710_<candidate-inputs-hash12>.json`
- Create after final authorization: `reports/closeout/attestations/final_approval.<decision-hash12>.json`
- Generate after content freeze in acyclic phases: final manifest then external attestation; P0 HarnessRun/ReleaseSafetyReport, LiteratureHarnessRun, composed ReleaseReportCandidate, ReleaseBundle; final preview then external preview approvals; frozen CandidateReleaseInputs and Stage D evidence; refreshed Stage A-D reports then external attestations; prerequisite evidence-only commit; acceptance and StageEVerificationReport; verification-evidence commit and EvidenceCommitAttestation; external Stage E report attestation; CloseoutDecisionPayload; external FinalApprovalAttestation; final evidence-only commit.

**Interfaces:**
- Consumes: frozen release-content commit, all prior plans/evidence, canonical runner output, and Stage C's already-implemented `FinalCloseoutCandidateCompositionRequest` / `compose_final_closeout_candidate()` unauthorized direct-ref interface.
- Produces: fixed-hash evidence in the acyclic order required by the design.

- [ ] **Step 1: Write failing orchestration-order tests**

```python
def test_orchestrator_requires_bundle_before_rollback_rehearsal(tmp_path: Path) -> None:
    events = dry_run_stage_e(tmp_path)
    assert events.index("implementation_branch_committed") < events.index("protected_merge_verified")
    assert events.index("protected_merge_verified") < events.index("release_content_frozen")
    assert events.index("release_content_frozen") < events.index("manifest_published")
    assert events.index("manifest_published") < events.index("manifest_approved")
    assert events.index("manifest_approved") < events.index("release_bundle_published")
    assert events.index("release_bundle_published") < events.index("final_stage_c_preview_published")
    assert events.index("final_stage_c_preview_published") < events.index("final_stage_c_preview_approved")
    assert events.index("final_stage_c_preview_approved") < events.index("stage_d_evidence_collected")
    assert events.index("stage_d_evidence_collected") < events.index("candidate_inputs_frozen")
    assert events.index("candidate_inputs_frozen") < events.index("refreshed_stage_gates_published")
    assert events.index("refreshed_stage_gates_published") < events.index("refreshed_stage_gates_approved")
    assert events.index("release_bundle_published") < events.index("prerequisite_evidence_commit_recorded")
    assert events.index("refreshed_stage_gates_approved") < events.index("prerequisite_evidence_commit_recorded")
    assert events.index("prerequisite_evidence_commit_recorded") < events.index("rollback_rehearsal_started")
    assert events.index("rollback_rehearsal_started") < events.index("acceptance_started")
    assert events.index("acceptance_completed") < events.index("stage_e_report_published")
    assert events.index("stage_e_report_published") < events.index("verification_evidence_commit_recorded")
    assert events.index("verification_evidence_commit_recorded") < events.index("evidence_commit_attested")
    assert events.index("evidence_commit_attested") < events.index("stage_e_report_approved")
    assert events.index("stage_e_report_approved") < events.index("decision_payload_published")
    assert events.index("decision_payload_published") < events.index("final_attestation_published")
    assert events.index("final_attestation_published") < events.index("final_evidence_commit_recorded")


def test_bound_source_change_after_decision_invalidates_authorization(tmp_path: Path) -> None:
    run = completed_stage_e(tmp_path)
    mutate_source_plan(run.repo)
    assert run.preflight().technical_disposition == "block"


def test_final_candidate_composition_precedes_and_does_not_require_refreshed_gate(
    tmp_path: Path,
) -> None:
    run = stage_e_with_approved_final_manifest(tmp_path)
    artifacts = run.publish_candidate_artifacts()
    assert artifacts.release_report_candidate.authorization_status == "not_approved"
    assert artifacts.release_report_candidate.composition_mode == "final_closeout_unapproved"
    assert run.events.index("release_report_candidate_published") < run.events.index(
        "candidate_inputs_frozen"
    )
    assert run.events.index("candidate_inputs_frozen") < run.events.index(
        "refreshed_stage_gates_published"
    )


@pytest.mark.parametrize(
    "mutation",
    ["bootstrap_phase", "null_frozen_commit", "wrong_frozen_commit", "stale_superseder"],
)
def test_final_manifest_requires_typed_phase_commit_and_current_parent(
    tmp_path: Path,
    mutation: str,
) -> None:
    run = stage_e_after_final_manifest_build(tmp_path)
    manifest = mutate_final_manifest(run.final_manifest, mutation)
    with pytest.raises(CloseoutGateBlocked, match="final manifest binding"):
        run.validate_final_manifest(manifest)


def test_candidate_inputs_path_list_contains_exact_uncommitted_chain(tmp_path: Path) -> None:
    run = stage_e_through_candidate_inputs(tmp_path)
    expected = (
        run.final_manifest_chain_paths
        | run.all_new_referenced_publisher_event_paths
        | run.p0_literature_candidate_bundle_paths
        | run.final_preview_event_paths
        | {run.stage_d_summary_path}
        | set(run.candidate_inputs_chain_paths)
    ) - run.tracked_dependency_paths
    assert set(run.write_evidence_path_list("candidate-inputs")) == expected
    assert run.final_candidate_publisher_event_path in expected
    assert run.phase_context_paths.isdisjoint(expected)
    run.stage_exact("candidate-inputs")
    assert run.cached_paths() == expected


@pytest.mark.parametrize(
    "failure_point",
    ["after_stage_e_report_published", "after_decision_published"],
)
def test_late_preview_race_preserves_exact_history_and_cleans_tree(
    tmp_path: Path,
    failure_point: str,
) -> None:
    run = stage_e_until(tmp_path, failure_point)
    run.expire_final_preview()
    recovery = run.supersede_downstream(reason_code="stage_c_preview_not_effective")
    expected = recovery.validated_untracked_attempt_paths | recovery.supersession_event_paths
    assert set(recovery.write_evidence_path_list("recovery-history")) == expected
    recovery.stage_exact("recovery-history")
    assert recovery.cached_paths() == expected
    recovery.verify_staged("recovery-history")
    recovery.commit_history()
    assert recovery.status_paths() == recovery.baseline_exclusions
```

- [ ] Add source-reconciliation tests that map every normalized requirement and source-plan checklist row to its owner/evidence/disposition, reject contradictory legacy wording or unaccounted rows, and route any changed requirement back to Stage A-D before freeze rather than marking it passed in Stage E.
- [ ] Add exact A/B/C/D artifact-binding tests for the required kind sets, canonical repository paths, path/ref/hash equality, top-level candidate-ref equality, final-preview literature binding and effective approval, Stage D collector ID/output path/ref, and `head_sha == tested_content_sha == merged_sha == release_content_commit`. Reject scans/globs/latest selection, duplicate or cross-stage paths, a prior/expired/superseded preview, missing promotion-event chain, an extra B artifact, an unallowlisted collector, or any refreshed report whose artifact refs differ from the frozen mapping.
- [ ] Add a real registered-runner integration in `test_stage_e_closeout.py` that invokes the production `run_closeout_stage_gate.py --stage C` path and `config/closeout_stage_suites.yaml`, not an orchestration mock. With final CandidateReleaseInputs, prove an exact `final_closeout_unapproved` candidate whose manifest, input hash, LiteratureHarnessRun, final preview and publisher-event refs all match yields a no-block technical refreshed C report while the candidate itself remains immutable `not_approved`; the report is unauthorized until its external attestation exists. Parameterize ordinary-train `not_approved`, missing CandidateReleaseInputs, wrong manifest/input hash, wrong candidate/literature/preview ref, and absent/mismatched publisher event; each must block before report publication.
- [ ] Add parser/dispatch tests that enumerate every Stage E subcommand below, run top-level and subcommand `--help` without mutation, require every declared flag, reject unknown commands/flags/inspect fields/evidence phases, enforce mutually exclusive recovery options, and prove inspector stdout contains exactly one sanitized scalar. Exercise each parser path through its real handler boundary; a command may not exist only in prose or a PowerShell example.
- [ ] Add restart/failure-injection tests after every final HarnessRun, P0 safety report, LiteratureHarnessRun, composed candidate, ReleaseBundle, final preview, Stage D summary, CandidateReleaseInputs, A-D refreshed report, approval-event, attestation, and ignored phase-context update. Include the exact crash windows after the fixed publisher event but before candidate publication and after the candidate's create-once publish but before its completed-step context update; restart must reuse the fsynced event ref and original train/sequence/parent/request rather than the current ledger/catalog. Test same-principal credential rotation, different-principal replacement, revocation, and unrelated ledger advancement. Every rerun must validate and mark exact existing steps `already_complete`, continue missing steps, and block any mismatch without overwrite.
- [ ] Add liveness-race tests expiring/revoking the final preview after C report publication, after one A-D approval, after the prerequisite commit, during acceptance, after StageEVerificationReport publication, and after decision publication. Every case must supersede downstream subjects, advance CandidateReleaseInputs, use new hash-suffixed report/decision IDs and idempotency keys, and rerun the required phase; no old approval or EvidenceCommitAttestation may carry forward. The after-report and after-decision cases must additionally prove that `recovery-history` stages the exact validated untracked set, preserves already tracked predecessors by hash without restaging them, commits immutable abandoned history, and restores status to the baseline exclusions before the next hygiene-gated acceptance.

- [ ] **Step 2: Run test and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_stage_e_closeout.py -q -p no:cacheprovider
```

Expected: FAIL because the orchestrator does not exist.

- [ ] **Step 3: Implement explicit phase ordering**

```python
@dataclass(frozen=True)
class StagePlanGateBinding:
    stage_id: Literal["A", "B", "C", "D"]
    plan_path: str
    plan_subject_path: str
    plan_attestation_path: str
    plan_ref: VersionRef
    plan_subject_ref: VersionRef
    plan_attestation_ref: VersionRef


@dataclass(frozen=True)
class CandidateArtifactSet:
    harness_run: StageArtifactInputBinding
    literature_harness_run: StageArtifactInputBinding
    p0_release_safety_report: StageArtifactInputBinding
    release_report_candidate: StageArtifactInputBinding
    release_bundle_path: str
    release_bundle_ref: VersionRef


@dataclass(frozen=True)
class CandidateReleaseInputs:
    inputs_id: str
    schema_version: str
    subject_version: str
    sequence: int
    supersedes_ref: VersionRef | None
    source_commit: str
    author_principal_id: str
    publisher_event_ref: VersionRef
    manifest_ref: VersionRef
    manifest_attestation_ref: VersionRef
    harness_run_ref: VersionRef
    literature_harness_run_ref: VersionRef
    release_safety_report_ref: VersionRef
    release_report_candidate_ref: VersionRef
    release_bundle_ref: VersionRef
    runtime_snapshot_ref: VersionRef
    stage_plan_bindings: Mapping[Literal["A", "B", "C", "D"], StagePlanGateBinding]
    stage_artifact_bindings: Mapping[Literal["A", "B", "C", "D"], StageArtifactBinding]
    inputs_sha256: str


def publish_final_manifest(context: StageEContext) -> CloseoutRequirementManifest:
    require_clean_frozen_release_content(context)
    previous = context.manifest_service.read_latest_approved_manifest()
    manifest = context.manifest_service.publish_final(
        manifest_phase="final",
        frozen_release_content_commit=context.release_content_commit,
        supersedes_manifest_ref=previous.to_version_ref(),
    )
    require_final_manifest_binding(
        manifest,
        frozen_commit=context.release_content_commit,
        expected_parent=previous.to_version_ref(),
    )
    return manifest


@dataclass(frozen=True)
class FinalCandidatePhaseContext:
    request_digest: str
    fixed_timestamp_seed: str
    ordered_target_paths: tuple[str, ...]
    expected_canonical_hashes: tuple[str, ...]
    publisher_event_idempotency_key: str
    publisher_event_ref: VersionRef
    final_candidate_request: FinalCloseoutCandidateCompositionRequest
    release_bundle_request: ReleaseBundleRequest


def publish_candidate_artifacts(
    context: StageEContext,
    manifest_attestation: ManifestApprovalAttestation,
) -> CandidateArtifactSet:
    manifest = validate_final_manifest_attestation(manifest_attestation)
    publisher_auth = context.publisher_auth_resolver.resolve_from_env(
        "LANGG_CLOSEOUT_PUBLISHER_TOKEN"
    )
    phase = context.candidate_service.load_or_create_fsynced_phase_context(
        manifest=manifest,
        manifest_attestation=manifest_attestation,
        source_commit=context.release_content_commit,
        report_id=final_candidate_id(context.release_content_commit),
        train_id="crc_closeout_final",
        publisher_auth=publisher_auth,
        publisher_idempotency_key=final_candidate_publisher_key(
            context.release_content_commit,
            manifest.to_version_ref(),
        ),
    )
    publisher_event = context.candidate_service.ensure_publisher_event(
        phase,
        fresh_auth=publisher_auth,
    )
    require_equal(
        phase.publisher_event_idempotency_key,
        final_candidate_publisher_key(
            context.release_content_commit,
            manifest.to_version_ref(),
        ),
    )
    require_exact_ref(publisher_event, phase.publisher_event_ref)
    require_exact_ref(publisher_event, phase.final_candidate_request.publisher_event_ref)
    p0, literature = context.candidate_service.publish_typed_replays(phase)
    request = phase.final_candidate_request
    candidate = compose_final_closeout_candidate(
        request,
        publisher_auth=publisher_auth,
    )
    require_unapproved_final_composition(candidate, request)
    bundle = context.candidate_service.publish_release_bundle(
        phase.release_bundle_request,
        after_candidate_ref=candidate.to_version_ref(),
    )
    return CandidateArtifactSet.from_exact(p0, literature, candidate, bundle)


def freeze_candidate_inputs(
    context: StageEContext,
    artifacts: CandidateArtifactSet,
    stage_c_preview_ready_context: Path,
) -> CandidateReleaseInputs:
    preview = validate_explicit_stage_c_preview_context(stage_c_preview_ready_context)
    stage_d_evidence = context.stage_runner.collect_stage_d_evidence(
        source_commit=context.release_content_commit,
        output_path=stage_d_output_path(context.release_content_commit),
    )
    return context.candidate_service.freeze_exact_inputs(
        artifacts,
        preview,
        stage_d_evidence=stage_d_evidence,
    )


def publish_refreshed_stage_gates(
    context: StageEContext,
    inputs: CandidateReleaseInputs,
) -> tuple[StageGateReport, ...]:
    validate_exact_stage_plan_bindings(inputs.stage_plan_bindings)
    validate_exact_stage_artifact_bindings(inputs.stage_artifact_bindings)
    return context.stage_runner.revalidate_a_to_d(
        inputs,
        plan_bindings=inputs.stage_plan_bindings,
        artifact_bindings=inputs.stage_artifact_bindings,
        orchestration_input_path=Path(inputs.to_version_ref().canonical_path),
        head_sha=context.release_content_commit,
        tested_content_sha=context.release_content_commit,
        merged_sha=context.release_content_commit,
    )


def publish_stage_e_verification(
    context: StageEContext,
    inputs: CandidateReleaseInputs,
    gate_attestations: tuple[StageGateApprovalAttestation, ...],
) -> StageEVerificationReport:
    validate_all_refreshed_gate_attestations(gate_attestations)
    validate_exact_stage_artifact_bindings(inputs.stage_artifact_bindings)
    validate_refreshed_reports_match_candidate_inputs(inputs, gate_attestations)
    release_bundle = context.bundle_store.load_and_validate(inputs.release_bundle_ref)
    validate_runtime_snapshot_binding(
        candidate_snapshot_ref=inputs.runtime_snapshot_ref,
        bundle_snapshot_ref=release_bundle.runtime_snapshot_ref,
        bundle_bindings=release_bundle.runtime_version_bindings,
    )
    rollback = context.rollback_service.rehearse(release_bundle)
    return context.acceptance_service.run(
        inputs,
        release_bundle,
        rollback,
        candidate_inputs_ref=inputs.to_version_ref(),
        stage_artifact_bindings=inputs.stage_artifact_bindings,
    )


def record_approval_event(
    subject_kind: Literal["closeout_manifest", "stage_gate", "stage_e_verification", "closeout_decision"],
    subject_path: Path,
    expected_subject_sha256: str,
    expected_subject_version: str,
    auth: AuthContext,
    idempotency_key: str,
) -> ApprovalLedgerProjection:
    subject = load_and_validate_approval_subject(
        subject_kind=subject_kind,
        subject_path=subject_path,
        expected_sha256=expected_subject_sha256,
        expected_subject_version=expected_subject_version,
    )
    policy = load_bound_approval_policy(subject.approval_policy_ref)
    return approval_ledger().append_one_authorized_event(
        subject_ref=subject.to_version_ref(),
        expected_subject_version=expected_subject_version,
        policy=policy,
        author_principal_id=subject.author_principal_id,
        auth=auth,
        idempotency_key=idempotency_key,
    )


def publish_decision(
    context: StageEContext,
    rows: Sequence[CloseoutGateRow],
    technical_inputs: FinalTechnicalInputs,
    gate_inputs: FinalGateInputs,
) -> CloseoutDecisionPayload:
    validate_stage_e_authorization(
        gate_inputs.stage_e_attestation,
        gate_inputs.evidence_commit_attestation,
    )
    return context.decision_service.build_and_publish_payload(
        rows=rows,
        technical_inputs=technical_inputs,
        gate_inputs=gate_inputs,
        publisher_auth=context.publisher_auth_resolver.resolve_from_env(
            "LANGG_CLOSEOUT_PUBLISHER_TOKEN"
        ),
    )
```

`frozen-inputs.json` serializes only the exact public VersionRefs/paths/hashes above plus release/base commits and test parameters; it contains no AuthContext. `stage_plan_bindings` and `stage_artifact_bindings` must each have exactly A/B/C/D once. Every plan binding carries the three explicit repository paths plus their refs. Every artifact input carries one repository-relative path and one matching VersionRef; paths are canonicalized, bounded, create-once, and rehashed before use. Filename scanning, glob selection, and implicit “latest” discovery are forbidden.

CandidateReleaseInputs is itself create-once and hash chained by monotonically increasing `sequence`/`supersedes_ref`. For Stage E refreshes the runner validates that all explicit artifact bindings are exactly those embedded in this orchestration input, then uses CandidateReleaseInputs SHA-256 as the combined gate-input-binding hash. A refreshed report ID includes both the frozen release-content SHA prefix and that hash prefix, so a new preview/input mapping never collides with or overwrites an earlier report for the same release commit. Only the highest valid non-superseded input sequence and its four matching report/attestation pairs are effective; approvals never carry across input or report hashes.

The artifact mapping is an exact stage contract: A has exactly the final HarnessRun and distinct P0 ReleaseSafetyReport; B has no input artifact and no collector; C has exactly the final LiteratureHarnessRun, the composed ReleaseReportCandidate that consumes the A P0 pair plus that literature run, and one newly approved/unexpired IngestPreview whose embedded promotion candidate binds that literature run; D has no ordinary input artifact and exactly the allowlisted `stage_d_evidence_v1` collector with `reports/closeout/stage_inputs/stage_d_evidence.<release-sha12>.json` plus its pre-collected VersionRef. Before freezing CandidateReleaseInputs, Stage E invokes the production Stage D collector once against `release_content_commit`, publishes create-once, and binds that exact output. The refreshed D gate deterministically replays/compares it and cannot discover or replace it.

Every refresh builds `StageGateRunRequest.plan_path`, `plan_subject_path`, `plan_attestation_path`, `artifact_paths`, and `orchestration_input_path` solely from those mappings and the exact CandidateReleaseInputs artifact, verifies the full plan-subject-attestation and artifact path/ref/hash chains, and sets `head_sha == tested_content_sha == merged_sha == release_content_commit`. On CandidateReleaseInputs sequence greater than one, the frozen context supplies each exact prior report path as `supersedes_report_path`; the runner validates a single-parent same-stage/content chain and publishes new hash-suffixed reports. Missing, duplicate, cross-stage, stale-plan, expired/revoked preview, wrong orchestration input, wrong object kind, unexpected collector, policy drift, blob-unretrievable path, or any report artifact set unequal to its binding blocks before or after the suite. `StageEVerificationReport.refreshed_stage_artifact_bindings` is the same immutable mapping, not a rediscovered copy.

Candidate/runtime tests also require `CandidateReleaseInputs.runtime_snapshot_ref == ReleaseBundle.runtime_snapshot_ref`, exact binding/ledger-head equality, and equality with the freshly resolved registry state; a candidate snapshot A paired with bundle snapshot B blocks before rollback or acceptance. They also require the A harness/P0-safety refs and C literature/composed-candidate refs in the stage mappings to equal the corresponding four top-level CandidateReleaseInputs refs, and require the composed candidate's embedded inputs to equal those exact A/C refs.

No orchestration function creates an approval event without a protected operator credential. Each `record_approval_event` invocation loads one declared subject path, validates the canonical kind/hash/explicit subject version and bound approval policy, then accepts exactly one server-derived AuthContext and idempotency key. Implement `record-approval`, `derive-attestation`, and `verify-attestation` dispatch for all four listed subject kinds; `closeout_manifest` and `stage_gate` delegate to Stage A's shared subject loaders, ledger, and attestation builders, while the two Stage E kinds extend that same registry rather than forking it. Derivation emits the kind-specific aggregate attestation only after the versioned policy's required role groups, counts, project scope, distinct principal/credential rules, author exclusion, and current ledger head validate. Callers cannot pass an arbitrary tuple of AuthContexts, client role labels, quorum, or policy overrides.

Each hard stop is resumed by one invocation per distinct credential. The parser grammar is below; the phase-specific PowerShell blocks later in this task supply every exact subject path/hash/version, role-specific Stage A environment variable, and hash-derived key:

```text
validate-implementation-branch --base-sha SHA --head-sha SHA --plan PATH --plan-subject PATH --plan-attestation PATH --source-plan PATH --output-root PATH
verify-frozen-merge --release-content-commit SHA --expected-merged-sha SHA
publish-manifest --release-content-commit SHA
inspect-context --context PATH --field final_manifest_path|final_manifest_sha256|final_manifest_subject_version|preview_path|candidate_inputs_sha256|untracked_recovery_history_count|stage_e_report_path|stage_e_report_sha256|stage_e_report_subject_version|decision_path|decision_sha256|decision_subject_version
verify-manifest-approval --context PATH
publish-candidate-artifacts --context PATH
prepare-final-stage-c-preview --context PATH --expires-at UTC --output PATH [--supersede-current]
freeze-candidate-inputs --context PATH --stage-c-preview-ready-context PATH --preview-evidence-path-list PATH [--supersede-current]
refresh-stage-gates --context PATH
record-refreshed-stage-approval --stage A|B|C|D --context PATH --credential-env ENV_NAME --idempotency-key KEY
derive-refreshed-stage-attestation --stage A|B|C|D --context PATH
verify-refreshed-stage-gates --context PATH
assert-candidate-inputs-live --context PATH
rehearse-rollback --context PATH
publish-stage-e-verification --context PATH
record-evidence-commit --commit SHA --context PATH
build-decision --context PATH
verify-decision-ready --decision PATH
verify --decision PATH
record-approval --subject-kind closeout_manifest|stage_gate|stage_e_verification|closeout_decision --subject-path PATH --expected-sha256 SHA --expected-version VERSION --credential-env ENV_NAME --idempotency-key KEY
derive-attestation --subject-kind closeout_manifest|stage_gate|stage_e_verification|closeout_decision --subject-path PATH --output PATH
verify-attestation --subject-kind closeout_manifest|stage_gate|stage_e_verification|closeout_decision --subject-path PATH --attestation-path PATH
write-evidence-path-list --phase candidate-inputs|prerequisites|verification|recovery-history|final-authorization --context PATH --output PATH
verify-staged-evidence --phase candidate-inputs|prerequisites|verification|recovery-history|final-authorization --path-list PATH --context PATH
record-phase-commit --phase prerequisites|recovery-history|final-authorization --commit SHA --context PATH
supersede-downstream --context PATH --reason-code stage_c_preview_not_effective
verify-recovery-clean --context PATH
```

The CLI parser registers exactly this grammar before any Step 4 invocation. SHA arguments are lowercase full commit IDs and are verified with `git cat-file`; paths are normalized repository-relative or within the command's declared ignored output root, and unknown/reparse/out-of-root paths block before writes. `--supersede-current` is legal only when an effective current subject exists and is mutually exclusive with initial creation. `inspect-context` validates the context schema/hash and selected subject before printing exactly one LF-terminated sanitized path, lowercase hash, explicit subject version, or non-negative integer to stdout; diagnostics go to stderr and an unavailable field is a nonzero error. Mutation commands return nonzero on missing approval, stale liveness, drift, collision, or partial validation and never report success from a diagnostic/list-only path.

The parser defines the evidence-phase enum once and dispatches every path-list, exact-stage verifier, and phase-commit recorder through the same validator; an unknown phase or a commit command for a non-committing phase is rejected. `record-evidence-commit` is the verification-phase-specific commit binder and rejects a commit whose exact path set/tree does not match the staged verification phase. `recovery-history` is a first-class resumable phase whose context records the abandoned candidate-input hash, ordered present targets, expected hashes, supersession-event refs, tracked dependency refs, and optional history commit. The approval CLI resolves the token through the Stage A credential mapping, checks revocation/project scope/required role, and writes only a content-free event. It never accepts principal ID, credential ID, role, or policy as an authority-bearing argument. Separately, every `publish-*`, `refresh-*`, and `build-decision` invocation resolves its author fresh from protected `LANGG_CLOSEOUT_PUBLISHER_TOKEN`; `StageEContext` and `output/closeout/frozen-inputs.json` never carry a token, principal, credential, role, scope, or client-supplied author. Published subjects persist only the server-derived author principal needed for author-exclusion validation plus a content-free publisher audit-event ref. Tests reject context/CLI author injection, revoked publisher credentials, publisher/approval credential reuse where policy forbids it, and author drift between phases.

Implement `validate-implementation-branch` in this CLI. It verifies the explicit base/head diff, exact committed Stage E plan subject/attestation, source-plan reconciliation result, no unstaged implementation path, required unit results, and list-only acceptance preflight; it writes advisory output only under ignored `output/closeout-advisory/`. Before the protected merge, reconcile `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`, run the source-reconciliation tests, and commit that document with the Stage E orchestration changes. Merge all Stage E runtime/config/source-plan changes first. The actual protected merge SHA becomes the frozen release-content commit; the source plan must not be staged in any later evidence-only commit.

- [ ] **Step 4: Verify, commit, advisory-check, and merge the Stage E implementation**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_stage_e_closeout.py -q -p no:cacheprovider
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -ListOnly
git diff --check
git add src/contracts/closeout.py src/services/closeout_stage_runner.py scripts/run_stage_e_closeout.py tests/backend/test_stage_e_closeout.py langg_crc_agent_stepwise_modification_plan_2026-06-29.md
git commit -m "feat(closeout): orchestrate frozen Stage E verification"
$branchHead = (git rev-parse HEAD).Trim()
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage D --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage D merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage D merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $branchHead
if ($LASTEXITCODE -ne 0) { throw "Stage E branch does not descend from approved Stage D" }
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py validate-implementation-branch --base-sha $stageBase --head-sha $branchHead --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-e-integrated-acceptance.md --plan-subject reports/closeout/plan_subjects/stage_e_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_e_plan_approval_20260710_001.json --source-plan langg_crc_agent_stepwise_modification_plan_2026-06-29.md --output-root output/closeout-advisory
```

Expected: tests and list-only pass, the advisory proves the supplied head contains every orchestration/test/source-plan change and the approved Stage E plan blob, and the working tree contains only baseline exclusions. Merge through the protected workflow. From a clean checkout at the actual merge commit, rerun the two verification commands above and record that exact merge SHA; do not substitute the branch head.

- [ ] **Step 5: Freeze the merged content and publish only prerequisites whose approvals already exist**

```powershell
$releaseContentCommit = (git rev-parse HEAD).Trim()
$expectedMergedSha = $env:LANGG_STAGE_E_MERGE_SHA
if (($expectedMergedSha -notmatch '^[0-9a-f]{40}$') -or ($releaseContentCommit -ne $expectedMergedSha)) { throw "checkout is not the recorded protected Stage E merge" }
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage D --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage D merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage D merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $releaseContentCommit
if ($LASTEXITCODE -ne 0) { throw "Stage E merge does not descend from approved Stage D" }
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-frozen-merge --release-content-commit $releaseContentCommit --expected-merged-sha $expectedMergedSha
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py publish-manifest --release-content-commit $releaseContentCommit
```

Expected: the final manifest is published with authorization `not_approved`, and the next phase refuses to run. **Hard stop:** approve the exact manifest through three distinct protected credentials, then publish the distinct P0/composed candidate artifacts:

```powershell
$contextPath = "output/closeout/frozen-inputs.json"
$manifestPath = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field final_manifest_path).Trim()
$manifestHash = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field final_manifest_sha256).Trim()
$manifestVersion = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field final_manifest_subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind closeout_manifest --subject-path $manifestPath --expected-sha256 $manifestHash --expected-version $manifestVersion --credential-env LANGG_REQUIREMENT_OWNER_TOKEN --idempotency-key "$manifestHash-final-requirement-owner-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind closeout_manifest --subject-path $manifestPath --expected-sha256 $manifestHash --expected-version $manifestVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$manifestHash-final-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind closeout_manifest --subject-path $manifestPath --expected-sha256 $manifestHash --expected-version $manifestVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$manifestHash-final-release-manager-v1"
$manifestAttestationPath = "reports/closeout/attestations/manifest_approval.$($manifestHash.Substring(0, 12)).json"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py derive-attestation --subject-kind closeout_manifest --subject-path $manifestPath --output $manifestAttestationPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-manifest-approval --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py publish-candidate-artifacts --context $contextPath
```

`publish-candidate-artifacts` derives every ID from the frozen release-content SHA. Before publishing any target, it resolves the protected publisher credential fresh and derives one stable source/manifest-bound publisher-event idempotency key. `load_or_create_fsynced_phase_context` uses that AuthContext only in memory to deterministically build the expected content-free publisher event, P0/literature/bundle payloads and refs, reads the report catalog exactly once to choose the prior parent and next sequence, puts the expected publisher-event ref into the complete final-candidate request, builds the expected candidate payload, and fsyncs the request digest, fixed timestamp seed, ordered paths, expected hashes, train, sequence, parent, report ID, source commit, manifest refs, publisher-event ref, and three typed evidence refs. The context never stores a token, credential ID, principal ID, role, or scope. The publisher event is then the first ordered create-once target. A restart resolves an active credential again and reuses the stored event after validating that event's principal/project against the fresh AuthContext; same-principal credential rotation and unrelated ledger advancement do not alter it, while a different principal, revoked/unauthorized fresh credential, or event mismatch blocks. It loads the catalog request verbatim and never calls live `next_sequence()` or `current_ref()` again. Concurrent catalog advancement or any mismatched context blocks rather than silently choosing a new sequence.

The phase then runs the final P0 replay, publishes `harness_closeout_final_<release-sha12>.json` plus the distinct non-catalog P0 `release_safety_closeout_final_<release-sha12>.json`, runs `literature_harness_closeout_final_<release-sha12>.json`, and calls Stage C's typed `compose_final_closeout_candidate()` to create `candidates/release_report_closeout_final_<release-sha12>.json` from the context's exact three refs plus the approved final-manifest/attestation refs. It does not call the normal Stage A-gate-backed CLI mode and does not fabricate a preliminary gate. The composer revalidates same-commit path/ref/hash/P0-linkage and protected `closeout_publisher` identity, and the result remains `not_approved`; only the later refreshed A and C reports/attestations can validate it for final closeout. Only after the candidate validates does the phase publish the SHA-derived ReleaseBundle; `after_candidate_ref` is an ordering precondition and is not embedded in the deployable-only bundle. This is a Stage A shared resumable phase, not a multi-file atomic claim: before every step it recomputes the expected payload from the fsynced context; an existing path is accepted only through full schema/ref/source/canonical-hash equality and recorded as `already_complete`, a mismatch blocks, and an absent path is published before continuing. The ignored context records the ordered completed set, so a crash after candidate publication but before its completion marker reuses the same request and returns `already_complete`. A P0/composed-kind substitution blocks; a new protected content merge receives new IDs rather than colliding with prior evidence.

- [ ] Prepare a new Stage C preview that binds the final literature run, hard-stop for its three external reviews, freeze CandidateReleaseInputs and the Stage D collector output, then exact-stage the cumulative candidate-input chain before any refreshed gate runs:

```powershell
$contextPath = "output/closeout/frozen-inputs.json"
$previewExpiresAt = $env:LANGG_STAGE_E_PREVIEW_EXPIRES_AT
if ([string]::IsNullOrWhiteSpace($previewExpiresAt)) { throw "LANGG_STAGE_E_PREVIEW_EXPIRES_AT must cover the approved Stage E completion window" }
$finalPreviewContext = "output/closeout/final-stage-c-preview.json"
$finalPreviewReadyContext = "output/closeout/final-stage-c-preview-ready.json"
$finalPreviewEvidencePaths = "output/closeout/final-stage-c-preview-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py prepare-final-stage-c-preview --context $contextPath --expires-at $previewExpiresAt --output $finalPreviewContext
$finalPreviewPath = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $finalPreviewContext --field preview_path).Trim()
$finalPreviewHash = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $finalPreviewPath --field sha256).Trim()
$finalPreviewVersion = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $finalPreviewPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $finalPreviewPath --expected-sha256 $finalPreviewHash --expected-version $finalPreviewVersion --decision approve --reason-code final_evidence_quality_verified --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$finalPreviewHash-final-evidence-review-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $finalPreviewPath --expected-sha256 $finalPreviewHash --expected-version $finalPreviewVersion --decision approve --reason-code final_clinical_safety_verified --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$finalPreviewHash-final-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $finalPreviewPath --expected-sha256 $finalPreviewHash --expected-version $finalPreviewVersion --decision approve --reason-code final_release_governance_verified --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$finalPreviewHash-final-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py verify-ready --preview $finalPreviewPath --output $finalPreviewReadyContext
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py write-evidence-path-list --pool-entry-context $finalPreviewContext --preview $finalPreviewPath --output $finalPreviewEvidencePaths
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py freeze-candidate-inputs --context $contextPath --stage-c-preview-ready-context $finalPreviewReadyContext --preview-evidence-path-list $finalPreviewEvidencePaths
$candidateInputsEvidencePaths = "output/closeout/candidate-inputs-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py write-evidence-path-list --phase candidate-inputs --context $contextPath --output $candidateInputsEvidencePaths
git add --pathspec-from-file=$candidateInputsEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-staged-evidence --phase candidate-inputs --path-list $candidateInputsEvidencePaths --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py refresh-stage-gates --context $contextPath
```

`prepare-final-stage-c-preview` reuses Stage C's store/service and the exact accepted Stage C pool-entry ref already frozen in context; it creates a sequence plus final-literature-hash-derived superseder, never an approval. If the final literature claim no longer matches that pool entry, it blocks and requires a new Stage C pool review. `freeze-candidate-inputs` revalidates the ready preview, constructs the exact A/B/C/D mapping above, runs the allowlisted Stage D collector at the frozen commit, and persists its exact output path/ref. Stage D summary then CandidateReleaseInputs are another declared resumable phase: exact existing outputs are revalidated and reused, mismatches block, and missing later output continues after restart. Every refreshed `StageGateRunRequest` then uses only that mapping and the frozen commit for head/tested/merged.

- [ ] **Hard stop:** record one credential-specific approval event for every refreshed gate, then derive all four attestations. These wrappers resolve the exact report path/hash/version only from frozen context and still append exactly one event per invocation:

```powershell
$contextPath = "output/closeout/frozen-inputs.json"
$candidateInputsHash = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field candidate_inputs_sha256).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage A --context $contextPath --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$candidateInputsHash-refresh-a-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage A --context $contextPath --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$candidateInputsHash-refresh-a-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage B --context $contextPath --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$candidateInputsHash-refresh-b-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage B --context $contextPath --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$candidateInputsHash-refresh-b-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage C --context $contextPath --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$candidateInputsHash-refresh-c-evidence-reviewer-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage C --context $contextPath --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$candidateInputsHash-refresh-c-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage C --context $contextPath --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$candidateInputsHash-refresh-c-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage D --context $contextPath --credential-env LANGG_PI_REVIEWER_TOKEN --idempotency-key "$candidateInputsHash-refresh-d-pi-reviewer-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage D --context $contextPath --credential-env LANGG_DATA_GOVERNANCE_REVIEWER_TOKEN --idempotency-key "$candidateInputsHash-refresh-d-data-governance-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-refreshed-stage-approval --stage D --context $contextPath --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$candidateInputsHash-refresh-d-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py derive-refreshed-stage-attestation --stage A --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py derive-refreshed-stage-attestation --stage B --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py derive-refreshed-stage-attestation --stage C --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py derive-refreshed-stage-attestation --stage D --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-refreshed-stage-gates --context $contextPath
```

- [ ] If the final preview expires, is revoked, or is invalidated before the prerequisite commit, verification must block with `stage_c_preview_not_effective`. Recover without changing release content or overwriting reports by superseding the preview and CandidateReleaseInputs:

```powershell
$contextPath = "output/closeout/frozen-inputs.json"
$previewExpiresAt = $env:LANGG_STAGE_E_PREVIEW_EXPIRES_AT
if ([string]::IsNullOrWhiteSpace($previewExpiresAt)) { throw "LANGG_STAGE_E_PREVIEW_EXPIRES_AT must cover the renewed completion window" }
$recoveryPreviewContext = "output/closeout/final-stage-c-preview-recovery.json"
$recoveryPreviewReadyContext = "output/closeout/final-stage-c-preview-recovery-ready.json"
$recoveryPreviewEvidencePaths = "output/closeout/final-stage-c-preview-recovery-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py prepare-final-stage-c-preview --context $contextPath --expires-at $previewExpiresAt --supersede-current --output $recoveryPreviewContext
$recoveryPreviewPath = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-context --context $recoveryPreviewContext --field preview_path).Trim()
$recoveryPreviewHash = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $recoveryPreviewPath --field sha256).Trim()
$recoveryPreviewVersion = (D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py inspect-subject --preview $recoveryPreviewPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $recoveryPreviewPath --expected-sha256 $recoveryPreviewHash --expected-version $recoveryPreviewVersion --decision approve --reason-code final_evidence_quality_reverified --credential-env LANGG_EVIDENCE_REVIEWER_TOKEN --idempotency-key "$recoveryPreviewHash-final-evidence-review-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $recoveryPreviewPath --expected-sha256 $recoveryPreviewHash --expected-version $recoveryPreviewVersion --decision approve --reason-code final_clinical_safety_reverified --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$recoveryPreviewHash-final-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py record-review --preview $recoveryPreviewPath --expected-sha256 $recoveryPreviewHash --expected-version $recoveryPreviewVersion --decision approve --reason-code final_release_governance_reverified --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$recoveryPreviewHash-final-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py verify-ready --preview $recoveryPreviewPath --output $recoveryPreviewReadyContext
D:\anaconda3\envs\LangG\python.exe scripts/manage_ingest_preview.py write-evidence-path-list --pool-entry-context $recoveryPreviewContext --preview $recoveryPreviewPath --output $recoveryPreviewEvidencePaths
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py freeze-candidate-inputs --context $contextPath --stage-c-preview-ready-context $recoveryPreviewReadyContext --preview-evidence-path-list $recoveryPreviewEvidencePaths --supersede-current
$candidateInputsEvidencePaths = "output/closeout/candidate-inputs-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py write-evidence-path-list --phase candidate-inputs --context $contextPath --output $candidateInputsEvidencePaths
git add --pathspec-from-file=$candidateInputsEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-staged-evidence --phase candidate-inputs --path-list $candidateInputsEvidencePaths --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py refresh-stage-gates --context $contextPath
```

The new CandidateReleaseInputs sequence supersedes the prior mapping, and refreshed report paths use its new hash suffix. Old reports/approval events remain immutable history but are ineffective for the new subjects. Re-run the immediately preceding ten credential-specific `record-refreshed-stage-approval` commands and four derivations; because each key is based on the newly read `$candidateInputsHash`, they cannot collide with prior subjects. Repeat recovery if liveness changes again. Any non-preview artifact drift instead invalidates the frozen release and requires a new protected content merge.

Each command must fail if a prerequisite external attestation is missing or stale. Verification validates all report hashes, exact manifest rows/counts, source refs, distinct signers, post-approval ledger heads, and exact artifact mappings. Then create the prerequisite evidence-only commit before any hygiene-gated acceptance command:

```powershell
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-refreshed-stage-gates --context output/closeout/frozen-inputs.json
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py write-evidence-path-list --phase prerequisites --context output/closeout/frozen-inputs.json --output output/closeout/prerequisite-evidence-paths.txt
git add --pathspec-from-file=output/closeout/prerequisite-evidence-paths.txt
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-staged-evidence --phase prerequisites --path-list output/closeout/prerequisite-evidence-paths.txt --context output/closeout/frozen-inputs.json
git commit -m "evidence(closeout): freeze approved verification prerequisites"
$prerequisiteEvidenceCommit = git rev-parse HEAD
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-phase-commit --phase prerequisites --commit $prerequisiteEvidenceCommit --context output/closeout/frozen-inputs.json
```

`write-evidence-path-list` emits repository-relative, newline-delimited, exact paths only after rejecting duplicate, missing, symlinked, out-of-repository, source/runtime/config, and baseline-exclusion paths. For the initial `candidate-inputs` phase, the exact cumulative uncommitted set is: the final manifest plus its three approval events and attestation; the final-candidate content-free publisher event; the P0 HarnessRun and distinct ReleaseSafetyReport; the LiteratureHarnessRun; the composed ReleaseReportCandidate; the ReleaseBundle; the final preview plus promotion/review/supersession event chain; the Stage D summary; and every CandidateReleaseInputs publisher/sequence/supersession artifact. It recursively follows only declared publisher/audit-event VersionRefs from those subjects and includes every newly created referenced event (not merely the candidate event), without scanning directories. Ignored phase/ready/path-list contexts are always excluded, as are dependencies already tracked after full ref/hash validation. After recovery, the same rule includes all still-uncommitted historical attempts plus the new effective preview/CandidateReleaseInputs sequence while excluding already committed predecessors. `verify-staged-evidence` requires the staged path set to equal the selected phase list, explicitly requires the candidate publisher event whenever its candidate is newly staged, validates every artifact and approval ledger head, and refuses mixed source/evidence commits. The later `prerequisites` list is the cumulative still-uncommitted candidate-input set plus refreshed A-D reports, approval events and attestations. `record-phase-commit` verifies the committed path set and tree before recording the SHA in the ignored, path-bounded frozen context. Expected: the repository is clean except for the baseline manifest's exact user-owned exclusions; ignored `output/closeout/frozen-inputs.json` is runtime context, not evidence and not repository dirt.

- [ ] **Step 6: Run acceptance, publish Stage E verification, and create the verification-evidence commit**

After the prerequisite evidence commit is recorded, run:

```powershell
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py assert-candidate-inputs-live --context output/closeout/frozen-inputs.json
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_stage_e_closeout.py tests/backend/test_closeout_contract.py tests/backend/test_closeout_hash_order.py tests/backend/test_closeout_store.py tests/backend/test_closeout_gate.py tests/backend/test_closeout_sensitive_artifact_scan.py tests/backend/test_closeout_acceptance_manifest.py tests/backend/test_closeout_deterministic_replay.py tests/backend/test_closeout_git_hygiene.py tests/backend/test_closeout_api.py tests/backend/test_release_governance_service.py tests/backend/test_release_execution_service.py -q -p no:cacheprovider
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -WriteEvidence
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py assert-candidate-inputs-live --context output/closeout/frozen-inputs.json
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py publish-stage-e-verification --context output/closeout/frozen-inputs.json
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py write-evidence-path-list --phase verification --context output/closeout/frozen-inputs.json --output output/closeout/verification-evidence-paths.txt
git add --pathspec-from-file=output/closeout/verification-evidence-paths.txt
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-staged-evidence --phase verification --path-list output/closeout/verification-evidence-paths.txt --context output/closeout/frozen-inputs.json
git commit -m "evidence(closeout): record integrated verification results"
$verificationEvidenceCommit = git rev-parse HEAD
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-evidence-commit --commit $verificationEvidenceCommit --context output/closeout/frozen-inputs.json
```

The acceptance runner performs its hygiene check before writing any verification artifact. Expected: required tests/scans/replays/rollback/diff/hygiene all pass with no required skip; the verification commit contains only the Stage E report and its declared acceptance artifacts; EvidenceCommitAttestation binds the StageEVerificationReport hash to that commit and does not claim the SHA of its own later commit.

Candidate-input liveness is rechecked before and after acceptance, before every Stage E/decision approval event, before payload publication, and in final `verify`. If any check fails after a prerequisite commit, first freeze the abandoned attempt as immutable history before executing the Step 5 preview-recovery block:

```powershell
$contextPath = "output/closeout/frozen-inputs.json"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py supersede-downstream --context $contextPath --reason-code stage_c_preview_not_effective
$recoveryHistoryCount = [int](D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field untracked_recovery_history_count).Trim()
if ($recoveryHistoryCount -gt 0) {
    $recoveryHistoryPaths = "output/closeout/recovery-history-evidence-paths.txt"
    D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py write-evidence-path-list --phase recovery-history --context $contextPath --output $recoveryHistoryPaths
    git add --pathspec-from-file=$recoveryHistoryPaths
    git diff --cached --check
    D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-staged-evidence --phase recovery-history --path-list $recoveryHistoryPaths --context $contextPath
    git commit -m "evidence(closeout): preserve superseded Stage E attempt"
    $recoveryHistoryCommit = (git rev-parse HEAD).Trim()
    D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-phase-commit --phase recovery-history --commit $recoveryHistoryCommit --context $contextPath
}
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-recovery-clean --context $contextPath
```

`supersede-downstream` validates every present path against the fsynced attempt context and appends create-once supersession events for the acceptance attempt plus any current StageEVerificationReport, EvidenceCommitAttestation, Stage E approval ledger/attestation, CloseoutDecisionPayload, decision approval ledger, or FinalApprovalAttestation; it never rewrites or deletes a partial artifact. `recovery-history` contains exactly the present, validated, untracked acceptance outputs, report, commit attestation, approval events/attestations, decision, final approval events/attestation, and their supersession events. Already tracked predecessors are rehashed and required as dependencies but excluded from the path list; absent not-yet-published targets are not invented. Its own ignored phase context applies the shared resumable multi-artifact protocol, so a crash during history publication or staging resumes without collision. `verify-recovery-clean` proves the index is empty and working-tree status equals only the recorded baseline exclusions.

After that history commit (or the verified zero-history case), execute the Step 5 preview-recovery block, all refreshed-gate approvals, and a new prerequisite evidence-only commit; rerun all of Step 6 to publish `stage_e.<release-sha12>.<new-inputs-hash12>.json` plus a new verification evidence commit/attestation. Step 7 must then build a new hash-suffixed decision and new approvals. No previous Stage E report, EvidenceCommitAttestation, decision payload, or approval carries across CandidateReleaseInputs hashes. Any liveness failure inside Step 7 returns to this same recovery-history block before any final-authorization staging.

- [ ] **Step 7: Approve Stage E, build/approve the decision, and create the final evidence-only commit**

**Hard stop:** resolve the current hash-suffixed StageEVerificationReport from context, record its three credential-specific approval events, derive/verify its attestation, and only then build the immutable, still-unauthorized decision payload:

```powershell
$contextPath = "output/closeout/frozen-inputs.json"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py assert-candidate-inputs-live --context $contextPath
$stageEReportPath = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field stage_e_report_path).Trim()
$stageEReportHash = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field stage_e_report_sha256).Trim()
$stageEReportVersion = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field stage_e_report_subject_version).Trim()
$stageEAttestationPath = "reports/closeout/attestations/stage_e_approval.$($stageEReportHash.Substring(0, 12)).json"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind stage_e_verification --subject-path $stageEReportPath --expected-sha256 $stageEReportHash --expected-version $stageEReportVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$stageEReportHash-stage-e-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind stage_e_verification --subject-path $stageEReportPath --expected-sha256 $stageEReportHash --expected-version $stageEReportVersion --credential-env LANGG_DATA_GOVERNANCE_REVIEWER_TOKEN --idempotency-key "$stageEReportHash-stage-e-data-governance-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind stage_e_verification --subject-path $stageEReportPath --expected-sha256 $stageEReportHash --expected-version $stageEReportVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$stageEReportHash-stage-e-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py derive-attestation --subject-kind stage_e_verification --subject-path $stageEReportPath --output $stageEAttestationPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-attestation --subject-kind stage_e_verification --subject-path $stageEReportPath --attestation-path $stageEAttestationPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py assert-candidate-inputs-live --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py build-decision --context $contextPath
```

Expected: the hash-suffixed payload validates technically but reports `not_authorized`. **Hard stop again:** in a self-contained session resolve that exact decision, record its three new approval events, and derive/verify a hash-derived FinalApprovalAttestation before exact staging:

```powershell
$contextPath = "output/closeout/frozen-inputs.json"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py assert-candidate-inputs-live --context $contextPath
$decisionPath = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field decision_path).Trim()
$decisionHash = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field decision_sha256).Trim()
$decisionVersion = (D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py inspect-context --context $contextPath --field decision_subject_version).Trim()
$finalAttestationPath = "reports/closeout/attestations/final_approval.$($decisionHash.Substring(0, 12)).json"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind closeout_decision --subject-path $decisionPath --expected-sha256 $decisionHash --expected-version $decisionVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$decisionHash-final-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind closeout_decision --subject-path $decisionPath --expected-sha256 $decisionHash --expected-version $decisionVersion --credential-env LANGG_DATA_GOVERNANCE_REVIEWER_TOKEN --idempotency-key "$decisionHash-final-data-governance-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-approval --subject-kind closeout_decision --subject-path $decisionPath --expected-sha256 $decisionHash --expected-version $decisionVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$decisionHash-final-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py derive-attestation --subject-kind closeout_decision --subject-path $decisionPath --output $finalAttestationPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-attestation --subject-kind closeout_decision --subject-path $decisionPath --attestation-path $finalAttestationPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py assert-candidate-inputs-live --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-decision-ready --decision $decisionPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py write-evidence-path-list --phase final-authorization --context $contextPath --output output/closeout/final-evidence-paths.txt
git add --pathspec-from-file=output/closeout/final-evidence-paths.txt
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify-staged-evidence --phase final-authorization --path-list output/closeout/final-evidence-paths.txt --context $contextPath
git commit -m "evidence(closeout): record final decision authorization"
$finalEvidenceCommit = git rev-parse HEAD
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py record-phase-commit --phase final-authorization --commit $finalEvidenceCommit --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py assert-candidate-inputs-live --context $contextPath
D:\anaconda3\envs\LangG\python.exe scripts/run_stage_e_closeout.py verify --decision $decisionPath
```

Expected: verify exits 0 and reports each scope's technical ceiling plus external authorization. It revalidates the latest manifest, highest non-superseded CandidateReleaseInputs sequence/hash, exact per-stage artifact mappings, all matching report/attestation hashes and ledger heads, effective final preview, P0/composed report separation, Stage D collector output, EvidenceCommitAttestation, ReleaseBundle/targets, monitoring state, source hashes, counts, and frozen commit. Do not execute a release or push unless the user separately authorizes it.

## Plan Self-Review Checklist

- [ ] Acyclic manifest/report/decision/attestation hashing: Tasks 1, 2, and 8.
- [ ] Exact manifest-to-row coverage and per-scope aggregation: Task 2.
- [ ] Multi-target ReleaseBundle, kill switch, rollback rehearsal/recovery: Tasks 3 and 8.
- [ ] Final governance/execution preflight and distinct principals: Task 4.
- [ ] Secret/PII/CoT scan without value leakage: Task 5.
- [ ] Real path-preflighted full acceptance runner: Task 6.
- [ ] Read-only final status: Task 7.
- [ ] Source-plan reconciliation before freeze and fixed evidence order: Task 8.
- [ ] Exact A/B/C/D artifact mappings, P0/composed report separation, preview-liveness recovery, and hash-suffixed downstream subjects: Tasks 1, 2, and 8.
