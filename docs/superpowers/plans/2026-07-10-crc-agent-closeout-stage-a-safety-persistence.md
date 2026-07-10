# CRC Agent Closeout Stage A Safety & Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the P0 safety and persistence gate with an approved requirement baseline, versioned intended-use and policy lifecycles, deterministic mutation replay, atomic CRC persistence, reproducible environments, and a reusable post-merge StageGate.

**Architecture:** Stage A first bootstraps shared integrity, authentication, sanitization, immutable artifact, baseline, and requirement-manifest contracts. After an externally authorized manifest attestation exists, the runtime resolves only active intended-use and safety-policy versions, executes real mutation cases, persists one atomic provenance chain, and publishes release evidence through a reusable gate runner. Public artifacts use SHA-256 `VersionRef`; patient-scoped artifacts use opaque `ClinicalVersionRef` with server-only HMAC integrity.

**Tech Stack:** Python 3.10, dataclasses, JSON/SHA-256/HMAC, SQLite, FastAPI 0.135, Pydantic 2.12, React 18, TypeScript 5.6, Vitest 2.1, Playwright, pytest, uv, npm, GitHub Actions.

## Global Constraints

- Begin from the approved design/plan-doc commit and, before changing runtime code, persist the actual program-base SHA plus sanitized starting status under ignored `output/closeout/`; every later baseline command must replay that recorded commit rather than the current implementation HEAD.
- Tasks 1-3 are bootstrap work. Stop after Task 3 until valid `PlanApprovalAttestation` and `ManifestApprovalAttestation` objects from policy-required distinct authorized principals approve the committed Stage A plan subject and finished manifest hash.
- Never expose patient identifiers, patient-linked hashes, restricted store handles, integrity MACs, HMAC key versions, credentials, secrets, prompts, or hidden chain-of-thought in public artifacts or APIs.
- Public `VersionRef` has exactly one locator: `canonical_path` xor `store_adapter`.
- Filesystem writers are create-once and path-bounded. Failed or invalid artifacts remain auditable; no in-place repair or overwrite is allowed.
- Request bodies never grant authority. All authorization comes from server-derived `AuthContext`.
- Emergency, policy activation, persistence consistency, privacy/security, sanitization, rollback, and required replay requirements allow only `pass|block`.
- Preserve compatibility additively unless this plan explicitly retires a known unsafe behavior.
- Use `D:\anaconda3\envs\LangG\python.exe`, `D:\anaconda3\Scripts\uv.exe`, and `D:\anaconda3\envs\LangG\npm.cmd` for verification.
- Do not modify `CRC-client/`. Preserve the user-owned untracked paths `scripts/generate_langg_hospital_strategy_pdf.py` and `work_reports_tmp/`.
- Stage only files named by the current task and use the commit message specified by that task.

## Source Design

- `docs/superpowers/specs/2026-07-10-crc-agent-closeout-program-design.md`, especially Sections 4-6 and 10-18.
- `langg_crc_agent_stepwise_modification_plan_2026-06-29.md` and every source document enumerated by `config/closeout_requirement_sources.yaml`.
- Existing CRC runtime, persistence, harness, release-safety, and acceptance tests are the behavioral baseline; legacy artifacts remain immutable migration inputs, never release authorization.

## File Structure

- `src/contracts/integrity.py`: `VersionRef`, `ClinicalVersionRef`, server-only integrity record, canonical hashing, and audit-chain primitives.
- `src/contracts/auth_context.py`: server-derived principal, credential, role, project-scope, and correlation identity.
- `src/contracts/closeout.py`: baseline, requirement manifest, approval attestation, and StageGate contracts.
- `src/services/atomic_artifact_store.py`: path-bounded create-once JSON publication.
- `src/services/write_boundary_sanitizer.py`: shared reject/redact sanitizer.
- `src/services/runtime_version_registry.py`: one registry for runtime-visible public VersionRefs.
- `src/services/closeout_manifest.py`: manifest construction, exact-set validation, and latest-approved selection.
- `src/services/closeout_gate.py`, `src/services/closeout_stage_runner.py`: reusable branch-advisory and post-merge gates.
- `src/services/intended_use.py`, `src/services/safety_policy_store.py`: active-version resolution and lifecycle authorization.
- `src/services/crc_mutation_replay.py`: real runtime mutation executor and field-by-field comparator.
- `config/closeout_*.yaml`, `config/intended_use_*.yaml`, `config/safety_*.yaml`: versioned inputs.
- `reports/closeout/*`: immutable baseline, requirement, stage, and attestation artifacts.

---

### Task 1: Shared Integrity, Sanitization, And Immutable Artifact Boundaries

**Files:**

- Create: `src/contracts/integrity.py`
- Create: `src/services/atomic_artifact_store.py`
- Create: `src/services/write_boundary_sanitizer.py`
- Create: `config/write_boundary_sanitizer.yaml`
- Create: `tests/backend/test_integrity_contracts.py`
- Create: `tests/backend/test_atomic_artifact_store.py`
- Create: `tests/backend/test_write_boundary_sanitizer.py`
- Modify: `.gitignore`
- Modify: `tests/backend/test_crc_safety_gitignore_contract.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class VersionRef:
    object_kind: str
    stable_id: str
    schema_version: str
    canonical_path: str | None
    store_adapter: str | None
    sha256: str
    source_git_commit: str

@dataclass(frozen=True, slots=True)
class ClinicalVersionRef:
    object_kind: str
    opaque_id: str
    version: str
    schema_version: str
    restricted_store_handle: str

@dataclass(frozen=True, slots=True)
class ClinicalVersionProjection:
    object_kind: str
    opaque_id: str
    version: str
    schema_version: str

@dataclass(frozen=True, slots=True)
class ClinicalIntegrityRecord:
    ref: ClinicalVersionRef
    integrity_mac: str
    integrity_key_version: str

@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_id: str
    subject_id: str
    subject_version: str
    sequence: int
    previous_event_hash: str
    event_hash: str
    schema_version: str
    actor_principal_id: str
    actor_credential_id: str
    actor_roles: tuple[str, ...]
    idempotency_key: str
    occurred_at: str
```

Also expose `canonical_json_bytes()`, `canonical_sha256()`, `build_audit_event()`, and `validate_audit_chain()`. `ClinicalVersionRef` and `ClinicalIntegrityRecord` are internal store types. Authorized APIs serialize only the random opaque ID/version/schema in `ClinicalVersionProjection`; they never expose a content hash, restricted handle, MAC, key version, or other patient-linkable integrity value. A genuinely de-identified cross-domain export receives a new unrelated stable ID and public VersionRef after separate validation.

- [ ] Before writing the first failing test, run `git rev-parse HEAD` and `git status --porcelain=v1 --untracked-files=all`; persist the SHA to `output/closeout/program-base.txt` and the sanitized status to `output/closeout/program-start-status.txt`. Validate that the only starting dirt is the baseline's exact user-owned exclusions. Do not update either file after Task 1 begins.
- [ ] Add failing tests for locator xor validation, lowercase 64-character SHA-256, Windows-safe artifact IDs, reserved device names, traversal, symlink/reparse escape, canonical ordering, event sequence/hash linkage, and tamper detection.
- [ ] Add failing store tests proving create-once publication, same-root temporary files, flush/validation before publish, ignored temporary names, collision failure, preservation of an invalid existing candidate, and `ensure_exact_artifact()` recovery that reuses only a fully validated byte/canonical-hash/source-identical existing target and otherwise blocks.
- [ ] Add failing sanitizer tests for direct identifiers, credentials, prompt secrets, `<think>`/reasoning fields, nested field paths, reject mode, redact mode, exact `sanitizer_ref: VersionRef`, and sanitized error details containing only rule IDs/count/field locations.
- [ ] Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_integrity_contracts.py tests/backend/test_atomic_artifact_store.py tests/backend/test_write_boundary_sanitizer.py -q -p no:cacheprovider
```

Expected: collection fails because the three new modules do not exist.

- [ ] Implement frozen dataclasses and canonical hashing. Exclude volatile timestamps only when the calling contract explicitly defines a semantic hash payload; do not make the canonicalizer silently omit fields.
- [ ] Implement `AtomicJsonArtifactPublisher.publish(root, artifact_id, payload, validate) -> VersionRef` with a same-root non-candidate temporary name, file flush/fsync, schema/hash validation, exclusive final creation, and cleanup of only the writer-owned temporary file. Add `ensure_exact_artifact(root, artifact_id, expected_payload, validate) -> ArtifactEnsureResult`: absent targets publish; existing targets are opened without following links, fully schema/ref/source/canonical-hash validated against the recomputed expected payload, and returned as `already_complete`; any mismatch is a collision block and is never overwritten.
- [ ] Implement `sanitize_text(text, *, scope, mode='reject')` and `sanitize_payload(payload, *, scope, mode='reject')`; return `SanitizedValue` with sanitized value, rule IDs, count, field locations, and the exact sanitizer-config `VersionRef`. Register that ref in `RuntimeVersionRegistry` when the registry is introduced in Task 4.
- [ ] Define the multi-object atomicity rule: one logical filesystem mutation embeds its complete immutable result in one aggregate event artifact; a restricted SQLite mutation that also publishes a public artifact uses a durable outbox committed in the same database transaction and an idempotent publisher. A declared multi-artifact evidence phase is not atomic: after all payloads are deterministically computed, it first fsyncs an ignored validated phase context containing request digest, fixed timestamp seed, ordered target paths, and expected canonical hashes; only then does it use `ensure_exact_artifact()` and resume-from-first-missing semantics. A crash before that context cannot leave a final artifact; a crash after a publish can recompute byte-identical payloads from the context. No later stage may claim two independent file renames are one transaction.
- [ ] Install the program's ignore-policy bootstrap in Task 1. The five frozen plans are already tracked by the approved plan-doc source commit, but the current repository deliberately ignores every other new `tests/backend/*` file and most `tests/fixtures/*`. Add only the exact exceptions below now so later tasks can use their ordinary path-bounded `git add`; do not add a broad subtree exception. The canonical Stage E runbook is intentionally excluded from this bootstrap because an ignored legacy working copy may already exist: Stage E Task 6 owns its parent/exact exception and stages `.gitignore` plus the runbook atomically, so no earlier clean-tree gate sees newly exposed dirt.

```gitignore
!tests/backend/test_atomic_artifact_store.py
!tests/backend/test_auth_context.py
!tests/backend/test_closeout_acceptance_manifest.py
!tests/backend/test_closeout_api.py
!tests/backend/test_closeout_contract.py
!tests/backend/test_closeout_deterministic_replay.py
!tests/backend/test_closeout_gate.py
!tests/backend/test_closeout_gate_contracts.py
!tests/backend/test_closeout_git_hygiene.py
!tests/backend/test_closeout_hash_order.py
!tests/backend/test_closeout_manifest_contracts.py
!tests/backend/test_closeout_manifest_validation.py
!tests/backend/test_closeout_sensitive_artifact_scan.py
!tests/backend/test_closeout_stage_runner.py
!tests/backend/test_closeout_store.py
!tests/backend/test_doctor_action_state_contract.py
!tests/backend/test_doctor_action_trace_idempotency.py
!tests/backend/test_doctor_action_trace_security.py
!tests/backend/test_doctor_draft_api.py
!tests/backend/test_doctor_draft_contract.py
!tests/backend/test_doctor_draft_graph_capture.py
!tests/backend/test_doctor_draft_service.py
!tests/backend/test_doctor_review_failure_injection.py
!tests/backend/test_doctor_review_non_mutation.py
!tests/backend/test_doctor_review_retention.py
!tests/backend/test_doctor_review_retention_api.py
!tests/backend/test_doctor_review_store.py
!tests/backend/test_environment_lock.py
!tests/backend/test_evidence_isolation_adapters.py
!tests/backend/test_evidence_promotion_non_mutation.py
!tests/backend/test_evidence_promotion_service.py
!tests/backend/test_ingest_preview_contract.py
!tests/backend/test_ingest_preview_store.py
!tests/backend/test_integrity_contracts.py
!tests/backend/test_intended_use_resolver.py
!tests/backend/test_learning_signal_store.py
!tests/backend/test_literature_harness_contract.py
!tests/backend/test_project_evidence_pool_contract.py
!tests/backend/test_project_evidence_pool_store.py
!tests/backend/test_publish_release_report_candidate.py
!tests/backend/test_release_bundle_store.py
!tests/backend/test_release_dashboard_flag_drift.py
!tests/backend/test_release_report_catalog.py
!tests/backend/test_release_report_contract.py
!tests/backend/test_research_approval_adapters.py
!tests/backend/test_research_evidence_api.py
!tests/backend/test_research_evidence_view.py
!tests/backend/test_research_governance_contract.py
!tests/backend/test_research_governance_service.py
!tests/backend/test_research_governance_store.py
!tests/backend/test_research_metadata_catalog.py
!tests/backend/test_research_query_ledger.py
!tests/backend/test_research_review_api.py
!tests/backend/test_runtime_version_registry.py
!tests/backend/test_safety_change_gate.py
!tests/backend/test_safety_policy_api.py
!tests/backend/test_safety_policy_lifecycle.py
!tests/backend/test_stage_b_clinical_review_gate.py
!tests/backend/test_stage_c_evidence_gate.py
!tests/backend/test_stage_d_evidence_contract.py
!tests/backend/test_stage_d_research_learning_gate.py
!tests/backend/test_stage_e_closeout.py
!tests/backend/test_write_boundary_sanitizer.py
!tests/fixtures/crc_mutation_pack_v1.json
!tests/fixtures/literature_claim_pack_closeout_v1.json
!tests/fixtures/literature_harness_case_catalog_v1.json
!tests/fixtures/stage_d_gate_case_catalog_v1.json
!docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-a-safety-persistence.md
!docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-b-clinical-review.md
!docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-c-evidence-promotion.md
!docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-d-research-learning.md
!docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-e-integrated-acceptance.md
```

`test_crc_safety_gitignore_contract.py` must parse the frozen A-E `Files: Create` inventories, enumerate every planned path that the pre-bootstrap rules ignore, and prove that the exact set partitions into (a) these Task 1 file exceptions and (b) the one delayed Stage E runbook exception paired with a same-task `.gitignore` modification and exact staging command. It must also run `git check-ignore`/`git cat-file` assertions proving every plan is tracked and retrievable at its source commit, the four fixtures become non-ignored, already permitted E2E `*.ts` behavior remains unchanged, and fixed closeout report roots are non-ignored. The runbook assertion is deliberately dual-state and is tested with both repository fixtures: when `HEAD` has no runbook blob, the path must remain ignored and none of its three delayed rules may exist; when `HEAD` contains the Stage E Task 6 blob, those exact three rules must exist, the path must be non-ignored/tracked, and `git cat-file` must retrieve the same blob. A missing exception, stale extra exception, force-added-but-later-untracked plan revision, prematurely exposed runbook, post-Task6 ignored/unretrievable runbook, or broad `tests/backend/**`, `tests/fixtures/**`, or `docs/superpowers/**` exception fails.
- [ ] Re-run the focused tests plus:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_crc_safety_gitignore_contract.py -q -p no:cacheprovider
```

Expected: all tests pass; no failure output contains a matched secret or raw sensitive value.

- [ ] Commit only Task 1 paths:

```powershell
git add src/contracts/integrity.py src/services/atomic_artifact_store.py src/services/write_boundary_sanitizer.py config/write_boundary_sanitizer.yaml tests/backend/test_integrity_contracts.py tests/backend/test_atomic_artifact_store.py tests/backend/test_write_boundary_sanitizer.py .gitignore tests/backend/test_crc_safety_gitignore_contract.py
git commit -m "feat(closeout): add shared integrity and artifact boundaries"
```

---

### Task 2: Server-Derived AuthContext And Bearer Principal Mapping

**Files:**

- Create: `src/contracts/auth_context.py`
- Create: `tests/backend/test_auth_context.py`
- Modify: `backend/api/services/settings.py`
- Modify: `backend/app.py`
- Modify: `tests/backend/test_auth_security.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class AuthContext:
    principal_id: str
    credential_id: str
    roles: frozenset[str]
    project_scopes: frozenset[str]
    correlation_id: str

def has_role(auth: AuthContext, role: str) -> bool: ...
def require_roles(auth: AuthContext, *roles: str) -> None: ...
def require_project_scope(auth: AuthContext, project_id: str, *roles: str) -> None: ...
```

- [ ] Add failing tests for token-to-principal mapping, `require_project_scope`, immutable historical identity, revoked credentials, missing context, client-supplied role rejection, and distinct principal plus credential quorum. The allowlist includes `closeout_publisher`, `requirement_owner`, `evidence_reviewer`, `clinical_safety_reviewer`, `researcher`, `ethics_reviewer`, `irb_reviewer`, `pi_reviewer`, `data_governance_reviewer`, `publication_reviewer`, `release_manager`, `patient_data_admin`, `migration_admin`, and read-only roles; unknown closeout or destructive-data roles are a startup/config error. `closeout_publisher` may author immutable subjects but satisfies no reviewer quorum.
- [ ] Add a regression proving one shared legacy admin token cannot satisfy two approval roles and is restricted to `migration_admin` or read-only endpoints.
- [ ] Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_auth_context.py tests/backend/test_auth_security.py -q -p no:cacheprovider
```

Expected: new imports or assertions fail.

- [ ] Add backward-compatible credential mappings to settings, including protected `LANGG_CLOSEOUT_PUBLISHER_TOKEN`, `LANGG_REQUIREMENT_OWNER_TOKEN`, `LANGG_EVIDENCE_REVIEWER_TOKEN`, `LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN`, `LANGG_RESEARCHER_TOKEN`, `LANGG_ETHICS_REVIEWER_TOKEN`, `LANGG_IRB_REVIEWER_TOKEN`, `LANGG_PI_REVIEWER_TOKEN`, `LANGG_DATA_GOVERNANCE_REVIEWER_TOKEN`, `LANGG_PUBLICATION_REVIEWER_TOKEN`, `LANGG_RELEASE_MANAGER_TOKEN`, and `LANGG_PATIENT_DATA_ADMIN_TOKEN` variables. `LANGG_CLOSEOUT_PUBLISHER_TOKEN` maps only to a project-scoped `closeout_publisher`; every other variable maps to its namesake role. Each variable resolves to one immutable server-side credential record; never log token values, and validation errors may identify only a configured credential ID.
- [ ] Update `BearerAuthMiddleware.dispatch` to attach `request.state.auth_context`; create the correlation ID server-side when no trusted inbound ID exists.
- [ ] Re-run the tests and verify all existing authenticated read paths still work with their intended compatibility token.
- [ ] Commit only Task 2 paths:

```powershell
git add src/contracts/auth_context.py tests/backend/test_auth_context.py backend/api/services/settings.py backend/app.py tests/backend/test_auth_security.py
git commit -m "feat(auth): derive closeout authorization context from bearer credentials"
```

---

### Task 3: Baseline, Requirement Manifest, And External Approval Contract

**Files:**

- Create: `src/contracts/closeout.py`
- Create: `src/services/closeout_manifest.py`
- Create: `config/closeout_requirement_sources.yaml`
- Create: `config/closeout_approval_policy.yaml`
- Create: `scripts/build_closeout_baseline.py`
- Create: `scripts/build_closeout_requirement_manifest.py`
- Create: `scripts/record_closeout_attestation.py`
- Create: `tests/backend/test_closeout_manifest_contracts.py`
- Create: `tests/backend/test_closeout_manifest_validation.py`
- Create: `reports/closeout/baselines/closeout_baseline_20260710_001.json`
- Create: `reports/closeout/requirements/closeout_requirements_20260710_001.json`
- Create after bootstrap commit: `reports/closeout/plan_subjects/stage_a_plan_20260710_001.json`
- Create after external authorization: `reports/closeout/attestations/stage_a_plan_approval_20260710_001.json`
- Create after external authorization: `reports/closeout/attestations/manifest_approval_20260710_001.json`

**Contracts:**

```python
@dataclass(frozen=True, slots=True)
class CloseoutBaselineRecord:
    baseline_id: str
    base_commit: str
    branch: str
    branch_topology: tuple[str, ...]
    clean_checkout: bool
    excluded_user_paths: tuple[str, ...]
    tool_versions: dict[str, str]
    command_results: tuple[dict[str, object], ...]
    known_failures: tuple[dict[str, object], ...]

@dataclass(frozen=True, slots=True)
class CloseoutRequirementManifest:
    manifest_id: str
    schema_version: str
    subject_version: str
    manifest_phase: Literal["bootstrap", "final"]
    frozen_release_content_commit: str | None
    author_principal_id: str
    publisher_event_ref: VersionRef
    project_scope: str
    approval_policy_ref: VersionRef
    previous_manifest_id: str | None
    previous_manifest_hash: str | None
    supersedes_manifest_ref: VersionRef | None
    source_refs: tuple[VersionRef, ...]
    entries: tuple[CloseoutRequirementEntry, ...]
    manifest_hash: str

@dataclass(frozen=True, slots=True)
class ManifestApprovalAttestation:
    attestation_id: str
    baseline_ref: VersionRef
    manifest_ref: VersionRef
    approval_policy_ref: VersionRef
    disposition_approvals: tuple[dict[str, object], ...]
    approver_events: tuple[AuditEvent, ...]
    post_approval_ledger_head: str

@dataclass(frozen=True, slots=True)
class StagePlanApprovalSubject:
    subject_id: str
    subject_version: str
    plan_ref: VersionRef
    author_principal_id: str
    publisher_event_ref: VersionRef
    project_scope: str
    approval_policy_ref: VersionRef
    subject_sha256: str

@dataclass(frozen=True, slots=True)
class PlanApprovalAttestation:
    attestation_id: str
    plan_subject_ref: VersionRef
    approval_policy_ref: VersionRef
    approver_event_refs: tuple[VersionRef, ...]
    approver_principal_ids: tuple[str, ...]
    approver_credential_ids: tuple[str, ...]
    post_approval_ledger_head: str
    attestation_sha256: str
```

`config/closeout_approval_policy.yaml` is a canonically hashed, versioned policy for `stage_plan`, `closeout_manifest`, `stage_gate`, `stage_e_verification`, and `closeout_decision`. Every row below requires one signer for each named role, all principal IDs and credential IDs distinct, none equal to the subject author, and exact project scope `closeout:crc`:

| Subject | Required roles |
|---|---|
| Stage A/B plan or gate | `clinical_safety_reviewer`, `release_manager` |
| Stage C plan or gate | `evidence_reviewer`, `clinical_safety_reviewer`, `release_manager` |
| Stage D plan or gate | `pi_reviewer`, `data_governance_reviewer`, `release_manager` |
| Stage E plan, `stage_e_verification`, or `closeout_decision` | `clinical_safety_reviewer`, `data_governance_reviewer`, `release_manager` |
| `closeout_manifest` | `requirement_owner`, `clinical_safety_reviewer`, `release_manager` |

Manifest phase is a typed invariant, not a filename convention. The initial Stage A artifact has `manifest_phase="bootstrap"`, `frozen_release_content_commit=None`, and no superseder. A `manifest_phase="final"` artifact requires one lowercase 40-hex `frozen_release_content_commit`, must set `supersedes_manifest_ref` to the exact currently approved manifest, and must make `previous_manifest_id`/`previous_manifest_hash` equal that ref's subject identity/hash. Every source ref and normalized source blob hash is resolved from that frozen commit; a missing/unretrievable source, a ref from another commit, a non-current parent, or a final-to-self/cyclic supersession blocks. Bootstrap with a commit, final without one, unknown phase, or caller inference from path/ID is invalid. These fields and the superseder ref are inside the manifest hash.

The policy fixes those role groups/counts, author exclusion, project-scope derivation, and stage selection; no “owning reviewer” placeholder or caller-supplied condition exists. Every approvable subject binds `subject_version`, server-derived `author_principal_id`, `project_scope`, and the exact approval-policy VersionRef. Policy drift invalidates readiness and requires a superseding subject; request arguments cannot redefine quorum.

`scripts/record_closeout_attestation.py` exposes three non-interactive subcommands shared by Stages A-D:

```text
build-plan-subject --plan PATH --source-commit SHA --author-credential-env ENV_NAME --output PATH
inspect-baseline --baseline PATH --field base_commit
inspect-subject --subject-kind stage_plan|closeout_manifest|stage_gate --subject-path PATH --field sha256|subject_version
resolve-latest-published --subject-kind stage_gate --stage A|B|C|D --field subject_path|sha256|subject_version|merged_sha
resolve-latest-approved --subject-kind closeout_manifest|stage_gate [--stage A|B|C|D] --field subject_path|attestation_path|sha256|subject_version|merged_sha
record-approval --subject-kind stage_plan|closeout_manifest|stage_gate --subject-path PATH --expected-sha256 SHA --expected-version VERSION --credential-env ENV_NAME --idempotency-key KEY
derive-attestation --subject-kind stage_plan|closeout_manifest|stage_gate --subject-path PATH --output PATH [--path-list-output PATH] [--include-subject] [--include-evidence PATH ...]
verify-attestation --subject-kind stage_plan|closeout_manifest|stage_gate --subject-path PATH --attestation-path PATH
verify-staged-evidence --path-list PATH --subject-path PATH --attestation-path PATH
```

`build-plan-subject` requires the plan path to be tracked in the supplied commit and verifies its blob hash before binding the plan VersionRef and server-derived author. `inspect-baseline` validates the exact committed baseline schema/hash and returns only its full-SHA `base_commit`. `resolve-latest-published` searches only the bounded stage-report directory, accepts only validated post-merge reports for the named stage, applies report supersession/source-commit ordering, and prints one requested value; zero, ambiguous, branch-advisory, malformed, or wrong-stage candidates block. It does not claim approval. `resolve-latest-approved` searches only the subject kind's bounded repository directories, validates every candidate hash/version/attestation/policy/ledger chain, applies supersession and revocation, requires `--stage` for `stage_gate` and forbids it for `closeout_manifest`, then prints exactly one sanitized requested value; `merged_sha` is allowed only for a validated post-merge `stage_gate`. Zero or ambiguous latest candidates block. `record-approval` resolves exactly one credential from the named protected environment variable into server-side AuthContext, loads and validates the selected subject kind, compares expected hash and explicit subject version, revalidates every bound stage artifact's current status, checks the bound policy's revocation/project scope/required role/author exclusion, and appends one content-free event. Same-key/same-subject restart returns the original event; a changed subject conflicts. It never accepts principal, credential, role, or quorum arguments. `derive-attestation` succeeds only after the policy-required distinct principal/credential quorum exists; exact existing output with the same subject/policy/ledger head is validated and returned as `already_complete`, while any mismatch blocks. When requested, it writes an ignored newline-delimited exact path list containing the derived attestation, referenced approval-event artifacts, explicitly validated evidence paths, and the subject only when `--include-subject` is present. For a report successor it also includes every uncommitted superseded-report/approval-attempt artifact required by the append-only chain; already tracked predecessors are rehashed but excluded. Use `--include-subject` for a newly generated plan subject or gate report that must enter the same evidence commit. Omit it for an already tracked immutable manifest: the verifier then requires that subject to be unchanged at its bound source commit and excludes it from the staged set. `verify-staged-evidence` requires the staged path set to equal that list and rejects source/runtime/config paths, unrelated older events, missing refs, or mixed commits. `verify-attestation` checks subject kind/hash/version, approval-policy ref, author exclusion, sequence chain, pre/post ledger heads, and latest-subject selection.

- [ ] Write failing contract tests for baseline-ref/hash binding, tracked plan-blob binding, acyclic entry/manifest hashing, ordered exact evidence sets, source-document hashes, required-entry counts, append-only version linkage, bootstrap/final phase and frozen-commit invariants, exact current-parent supersession, explicit subject-version extraction for every supported kind, bounded latest-published report resolution (including zero/ambiguous/advisory/wrong-stage candidates), bounded latest-approved resolution (including zero/ambiguous/revoked candidates), policy-defined quorum, distinct approvers, author exclusion, policy drift, and ledger-head verification.
- [ ] Write failing validation tests proving missing, unknown, duplicated, downgraded, stale, or source-drifted requirements block. Safety/privacy/authorization/hard-fail/persistence/sanitization/rollback/replay rows cannot be downgraded.
- [ ] Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_manifest_contracts.py tests/backend/test_closeout_manifest_validation.py -q -p no:cacheprovider
```

Expected: missing closeout contract/service failures.

- [ ] Implement the contracts and validators. An entry hash covers the canonical entry without its hash; the manifest hash covers the canonical header without its hash plus ordered entry hashes and source hashes. Approval data is external to the manifest hash. The manifest builder resolves author/project metadata only from the protected credential named by `--author-credential-env`; it rejects principal, role, project, policy, or author values supplied elsewhere.
- [ ] Populate `config/closeout_requirement_sources.yaml` with the approved design, the original modification plan, and every source plan/spec used to normalize requirements. Each source entry has an exact repository path and required document state. Populate `config/closeout_approval_policy.yaml` with the exact role/count/scope rules above and publish its VersionRef before building plan/manifest approval subjects.
- [ ] Build the baseline by loading the immutable SHA from `output/closeout/program-base.txt`, verifying that commit exists and contains the approved plan/design blobs, and running baseline commands in a clean detached temporary worktree at that commit. The script accepts explicit `--base-sha`, `--workspace-status-file`, and repeated `--excluded-user-path` arguments; it rejects the current implementation HEAD, a changed start-status record, extra dirt, or a base that lacks the source blobs. Record `main`, command/results/toolchain, known failures, and the two exact user-owned excluded paths. Remove only the script-owned temporary worktree after capture.
- [ ] Build the canonical manifest and validate that every baseline failure/exclusion maps to at least one requirement row. The sequencing-only historical issue is explicitly marked superseded by the approved baseline procedure, not silently passed.
- [ ] Run:

```powershell
$programBase = (Get-Content output\closeout\program-base.txt -Raw).Trim()
D:\anaconda3\envs\LangG\python.exe scripts\build_closeout_baseline.py --base-sha $programBase --workspace-status-file output\closeout\program-start-status.txt --excluded-user-path scripts/generate_langg_hospital_strategy_pdf.py --excluded-user-path work_reports_tmp/ --output reports\closeout\baselines\closeout_baseline_20260710_001.json
D:\anaconda3\envs\LangG\python.exe scripts\build_closeout_requirement_manifest.py --manifest-phase bootstrap --sources config\closeout_requirement_sources.yaml --baseline reports\closeout\baselines\closeout_baseline_20260710_001.json --author-credential-env LANGG_CLOSEOUT_PUBLISHER_TOKEN --output reports\closeout\requirements\closeout_requirements_20260710_001.json
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_manifest_contracts.py tests/backend/test_closeout_manifest_validation.py -q -p no:cacheprovider
```

Expected: scripts exit 0, immutable artifacts validate, and tests pass.

- [ ] Commit the bootstrap implementation and pre-approval artifacts only:

```powershell
git add src/contracts/closeout.py src/services/closeout_manifest.py config/closeout_requirement_sources.yaml config/closeout_approval_policy.yaml scripts/build_closeout_baseline.py scripts/build_closeout_requirement_manifest.py scripts/record_closeout_attestation.py tests/backend/test_closeout_manifest_contracts.py tests/backend/test_closeout_manifest_validation.py reports/closeout/baselines/closeout_baseline_20260710_001.json reports/closeout/requirements/closeout_requirements_20260710_001.json
git commit -m "feat(closeout): bootstrap baseline and requirement manifest"
```
- [ ] **Hard stop:** build, externally approve, exact-stage, and commit the Stage A plan authorization evidence:

```powershell
$planPath = "docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-a-safety-persistence.md"
$planSourceCommit = (git log -1 --format=%H -- $planPath).Trim()
git cat-file -e "${planSourceCommit}:$planPath"
$trackedPlanBlob = (git rev-parse "${planSourceCommit}:$planPath").Trim()
$workingPlanBlob = (git hash-object -- $planPath).Trim()
if ($trackedPlanBlob -ne $workingPlanBlob) { throw "Stage A plan blob is not the tracked authorized source" }
$planSubjectPath = "reports/closeout/plan_subjects/stage_a_plan_20260710_001.json"
$planAttestationPath = "reports/closeout/attestations/stage_a_plan_approval_20260710_001.json"
$planEvidencePaths = "output/closeout/stage-a-plan-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py build-plan-subject --plan $planPath --source-commit $planSourceCommit --author-credential-env LANGG_CLOSEOUT_PUBLISHER_TOKEN --output $planSubjectPath
$planSubjectHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field sha256).Trim()
$planSubjectVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-a-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$planSubjectHash-stage-a-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_plan --subject-path $planSubjectPath --output $planAttestationPath --path-list-output $planEvidencePaths --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_plan --subject-path $planSubjectPath --attestation-path $planAttestationPath
git add --pathspec-from-file=$planEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $planEvidencePaths --subject-path $planSubjectPath --attestation-path $planAttestationPath
git commit -m "evidence(closeout): approve Stage A plan"
```

- [ ] Separately approve the immutable requirement manifest with its three policy-required distinct credentials, then exact-stage and commit only its attestation chain:

```powershell
$manifestPath = "reports/closeout/requirements/closeout_requirements_20260710_001.json"
$manifestAttestationPath = "reports/closeout/attestations/manifest_approval_20260710_001.json"
$manifestEvidencePaths = "output/closeout/manifest-approval-evidence-paths.txt"
$manifestHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind closeout_manifest --subject-path $manifestPath --field sha256).Trim()
$manifestVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind closeout_manifest --subject-path $manifestPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind closeout_manifest --subject-path $manifestPath --expected-sha256 $manifestHash --expected-version $manifestVersion --credential-env LANGG_REQUIREMENT_OWNER_TOKEN --idempotency-key "$manifestHash-requirement-owner-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind closeout_manifest --subject-path $manifestPath --expected-sha256 $manifestHash --expected-version $manifestVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$manifestHash-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind closeout_manifest --subject-path $manifestPath --expected-sha256 $manifestHash --expected-version $manifestVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$manifestHash-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind closeout_manifest --subject-path $manifestPath --output $manifestAttestationPath --path-list-output $manifestEvidencePaths
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind closeout_manifest --subject-path $manifestPath --attestation-path $manifestAttestationPath
git add --pathspec-from-file=$manifestEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $manifestEvidencePaths --subject-path $manifestPath --attestation-path $manifestAttestationPath
git commit -m "evidence(closeout): approve requirement manifest"
```

Do not start Task 4 from a self-approved, policy-drifted, untracked-plan, unapproved-manifest, or mixed source/evidence commit.

---

### Task 4: Intended-Use Resolver, Runtime Metadata, And Version Registry

**Files:**

- Create: `config/intended_use_disclaimers.yaml`
- Create: `src/services/intended_use.py`
- Create: `src/services/runtime_version_registry.py`
- Create: `tests/backend/test_intended_use_resolver.py`
- Create: `tests/backend/test_runtime_version_registry.py`
- Modify: `config/intended_use_profiles.yaml`
- Modify: `src/services/crc_triage_flow.py`
- Modify: `backend/api/routes/crc_triage.py`
- Modify: `backend/api/schemas/responses.py`
- Modify: `backend/app.py`
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/features/patient-crc-triage/crc-triage-context.ts`
- Modify: `frontend/src/features/patient-crc-triage/patient-crc-triage-panel.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Modify: `backend/api/services/graph_service.py`
- Modify: `tests/backend/test_graph_service_streaming.py`
- Modify: `tests/e2e/acceptance/frontend-regression-contracts.spec.ts`
- Modify: `tests/backend/test_intended_use_profiles.py`
- Modify: `tests/backend/test_crc_triage_flow.py`
- Modify: `backend/api/routes/test_crc_triage_api.py`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/features/patient-crc-triage/crc-triage-context.test.ts`
- Modify: `frontend/src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx`
- Modify: `frontend/src/pages/workspace-page.test.tsx`

**Interfaces:**

```python
def resolve_intended_use(profile_id: str, locale: str) -> IntendedUseMetadata: ...

class RuntimeVersionRegistry:
    def resolve_active(self, slot: str) -> VersionRef: ...
    def activate(
        self,
        slot: str,
        ref: VersionRef,
        expected_current_ref: VersionRef | None,
        auth: AuthContext,
        idempotency_key: str,
    ) -> RuntimeVersionActivationEvent: ...
    def snapshot(self) -> RuntimeVersionSnapshot: ...
```

```python
@dataclass(frozen=True, slots=True)
class RuntimeVersionBinding:
    slot: str
    ref: VersionRef
    activation_event_ref: VersionRef


@dataclass(frozen=True, slots=True)
class RuntimeVersionSnapshot:
    snapshot_id: str
    schema_version: str
    bindings: tuple[RuntimeVersionBinding, ...]
    registry_ledger_head: str
    source_git_commit: str
    snapshot_sha256: str

    def to_version_ref(self) -> VersionRef: ...
```

Slots are namespaced logical identities such as `intended_use:patient_crc_triage`, `sanitizer:write_boundary`, `safety_policy:patient_crc_triage:active`, `safety_policy:patient_crc_triage:shadow`, and `clinical_rag:crc_guideline:active`. Activation appends one authorized, expected-current event; same-key replay is idempotent, a stale expected ref conflicts, and history is never overwritten. `snapshot()` returns an immutable, canonically hashed snapshot with exactly one `(slot, ref, activation_event_ref)` binding per effective slot in deterministic slot order; historical refs are available only through the append-only activation ledger. Runtime/API consumers persist or expose the snapshot VersionRef plus bindings, never an unlabelled tuple whose active/shadow meaning must be guessed.

- [ ] Add failing tests for supported locale/profile resolution, fallback behavior, unknown profiles, disclaimer visibility, assessment persistence of the exact intended-use ref, one active ref per logical slot, expected-current conflicts, activation/rollback history, idempotent replay, stable current-snapshot ordering, snapshot hash tamper, duplicate/missing slots, and active/shadow refs with the same object kind remaining distinguishable by slot.
- [ ] Add API/frontend tests for `GET /api/sessions/{session_id}/crc-triage/metadata?locale=zh-CN`, additive `RuntimeInfo.version_bindings`/`runtime_snapshot_ref`, and compatibility-only `version_refs` derived from those bindings.
- [ ] Add backend/E2E tests proving both patient and doctor streams remove `thinking`, hidden-reasoning keys, and `<think>` blocks before persistence or transport and never render `.clinical-thinking-disclosure`.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_intended_use_profiles.py tests/backend/test_intended_use_resolver.py tests/backend/test_runtime_version_registry.py tests/backend/test_crc_triage_flow.py tests/backend/test_graph_service_streaming.py backend/api/routes/test_crc_triage_api.py -q -p no:cacheprovider` plus the Vitest command below; expect missing resolver/metadata/snapshot failures.
- [ ] Implement the resolver and append-only activation registry. Activate intended-use catalog/profile plus the sanitizer config ref into their named slots. Stage C will activate the exact Clinical RAG manifest metadata ref through the same expected-current protocol without writing RAG content.
- [ ] Update `_build_final_assessment` to persist the resolved intended-use reference. Render the localized intended-use statement before triage starts and beside the final assessment. Task 5 adds active-policy resolution and registration; Task 4 must not fabricate an active policy ref before that store exists.
- [ ] Update `GraphService._filter_scene_event` only now, after the manifest hard stop, and keep the filtered visible message as the sole downstream payload.
- [ ] Re-run focused tests and build:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_intended_use_profiles.py tests/backend/test_intended_use_resolver.py tests/backend/test_runtime_version_registry.py tests/backend/test_crc_triage_flow.py tests/backend/test_graph_service_streaming.py backend/api/routes/test_crc_triage_api.py -q -p no:cacheprovider
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/patient-crc-triage/crc-triage-context.test.ts src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx src/pages/workspace-page.test.tsx --reporter=verbose
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/frontend-regression-contracts.spec.ts
```

Expected: all tests pass and Vite/TypeScript exits 0.

- [ ] Commit only Task 4 paths:

```powershell
git add config/intended_use_disclaimers.yaml src/services/intended_use.py src/services/runtime_version_registry.py tests/backend/test_intended_use_resolver.py tests/backend/test_runtime_version_registry.py config/intended_use_profiles.yaml src/services/crc_triage_flow.py backend/api/routes/crc_triage.py backend/api/schemas/responses.py backend/app.py frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/features/patient-crc-triage/crc-triage-context.ts frontend/src/features/patient-crc-triage/patient-crc-triage-panel.tsx frontend/src/pages/workspace-page.tsx backend/api/services/graph_service.py tests/backend/test_graph_service_streaming.py tests/e2e/acceptance/frontend-regression-contracts.spec.ts tests/backend/test_intended_use_profiles.py tests/backend/test_crc_triage_flow.py backend/api/routes/test_crc_triage_api.py frontend/src/app/api/client.test.ts frontend/src/features/patient-crc-triage/crc-triage-context.test.ts frontend/src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx frontend/src/pages/workspace-page.test.tsx
git commit -m "feat(crc): expose resolved intended-use metadata"
```

---

### Task 5: Authorized Safety-Policy Lifecycle And Safe Stop

**Files:**

- Create: `src/contracts/safety_policy_lifecycle.py`
- Create: `src/services/safety_policy_store.py`
- Create: `backend/api/schemas/safety_policies.py`
- Create: `backend/api/routes/safety_policies.py`
- Create: `tests/backend/test_safety_policy_lifecycle.py`
- Create: `tests/backend/test_safety_policy_api.py`
- Modify: `config/safety_policy.yaml`
- Modify: `src/services/clinical_safety_policy.py`
- Modify: `src/services/crc_triage_flow.py`
- Modify: `backend/app.py`
- Modify: `tests/backend/test_clinical_safety_policy.py`
- Modify: `tests/backend/test_crc_triage_flow.py`

**Interfaces:**

```python
class SafetyPolicyStore:
    def resolve_active(self, profile_id: str) -> ActivePolicy: ...
    def record_approval(self, policy_ref: VersionRef, auth: AuthContext, idempotency_key: str) -> AuditEvent: ...
    def activate(self, policy_ref: VersionRef, approval_refs: tuple[VersionRef, ...], auth: AuthContext, idempotency_key: str) -> ActivePolicy: ...
    def rollback(self, target_ref: VersionRef, approval_refs: tuple[VersionRef, ...], auth: AuthContext, idempotency_key: str) -> ActivePolicy: ...
```

`PolicyEvaluationOutcome` contains `workflow_status`, nullable `pre_policy_disposition`, nullable `pre_policy_ref`, nullable final `clinical_disposition`, allowlisted `policy_failure_reason_code`, `patient_message_key`, `assessment_save_allowed`, `automated_closure_allowed`, and active/shadow refs.

- [ ] Add failing tests proving draft/shadow/revoked policies never become active by file presence; activation and rollback require authorized distinct approvals and an intact audit chain.
- [ ] Add failure tests for missing/corrupt/expired active policy, unknown rule, evaluation exception, and drift between resolved/evaluated refs. Safe stop must preserve a validated deterministic pre-policy disposition/ref as the minimum urgency and may never lower it; without a valid pre-policy ref it returns no specific clinical disposition. Both cases block save and automated closure and use only an allowlisted patient message.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_safety_policy_lifecycle.py tests/backend/test_safety_policy_api.py tests/backend/test_crc_triage_flow.py -q -p no:cacheprovider`; expect lifecycle/import or active-policy assertions to fail.
- [ ] Implement immutable policy versions and append-only lifecycle events. Preserve `ClinicalSafetyPolicy`, `load_clinical_safety_policy`, `evaluate_clinical_safety_policy`, and `merge_policy_disposition` behind the active resolver.
- [ ] Route all CRC evaluation through `resolve_active`; remove the current unconditional consumption of a `status: draft` file. Register the exact active/shadow policy and policy-config refs in `RuntimeVersionRegistry` and persist the active ref in the assessment.
- [ ] Re-run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_clinical_safety_policy.py tests/backend/test_safety_policy_lifecycle.py tests/backend/test_safety_policy_api.py tests/backend/test_crc_triage_flow.py -q -p no:cacheprovider
```

Expected: all pass, including failure injection and idempotent lifecycle replay.

- [ ] Commit only Task 5 paths:

```powershell
git add src/contracts/safety_policy_lifecycle.py src/services/safety_policy_store.py backend/api/schemas/safety_policies.py backend/api/routes/safety_policies.py tests/backend/test_safety_policy_lifecycle.py tests/backend/test_safety_policy_api.py config/safety_policy.yaml src/services/clinical_safety_policy.py src/services/crc_triage_flow.py backend/app.py tests/backend/test_clinical_safety_policy.py tests/backend/test_crc_triage_flow.py
git commit -m "feat(safety): enforce authorized policy lifecycle and safe stop"
```

---

### Task 6: Complete Mutation Pack Through The Real CRC Runtime

**Files:**

- Create: `config/crc_hard_fail_catalog.yaml`
- Create: `tests/fixtures/crc_mutation_pack_v1.json`
- Create: `src/services/crc_mutation_replay.py`
- Modify: `src/services/crc_triage_flow.py`
- Modify: `src/nodes/triage_nodes.py`
- Modify: `tests/backend/test_crc_triage_mutation_pack.py`
- Modify: `tests/backend/test_crc_triage_flow.py`
- Modify: `tests/backend/test_outpatient_triage_gating.py`

- [ ] Replace the metadata-only topic-switch test with failing cases that execute age, family history, rectal bleeding, weight loss, obstruction, missing tests, topic switch, and topic return through `_run_crc_triage_protocol_turn` and the actual flow.
- [ ] Encode declared expected fields per case: visible question/answer state, collected facts, unresolved fields, disposition constraints, hard-fail codes, workflow status, and isolation assertions.
- [ ] Add metamorphic tests: family-history mutation cannot reduce the baseline disposition; missing required inputs cannot increase certainty; off-topic content cannot mutate CRC answers; return resumes the exact pending CRC question. Do not invent clinical conclusions not present in the approved policy.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_crc_triage_mutation_pack.py tests/backend/test_crc_triage_flow.py tests/backend/test_outpatient_triage_gating.py -q -p no:cacheprovider`; expect the current synthetic topic-switch behavior or incomplete comparisons to fail.
- [ ] Implement a deterministic replay service that executes every case, compares every declared expected field, records the exact runtime snapshot ref plus labelled slot bindings, and rejects undeclared actual fields that affect a safety decision.
- [ ] Re-run that exact pytest command and require PASS with zero skips.
- [ ] Commit only Task 6 paths:

```powershell
git add config/crc_hard_fail_catalog.yaml tests/fixtures/crc_mutation_pack_v1.json src/services/crc_mutation_replay.py src/services/crc_triage_flow.py src/nodes/triage_nodes.py tests/backend/test_crc_triage_mutation_pack.py tests/backend/test_crc_triage_flow.py tests/backend/test_outpatient_triage_gating.py
git commit -m "test(safety): replay complete CRC mutations through runtime"
```

---

### Task 7: HarnessRun And ReleaseSafetyReport Decision Propagation

**Files:**

- Modify: `src/contracts/harness.py`
- Modify: `src/contracts/release_safety_report.py`
- Modify: `scripts/run_crc_harness_replay.py`
- Modify: `tests/backend/test_crc_harness_replay.py`
- Modify: `reports/harness/README.md`
- Modify: `reports/release_safety/README.md`

**Decision order:** `block > shadow_only > feature_flag > pass`.

- [ ] Add failing tests for real topic-switch execution, every expected-field comparison, version-chain inclusion, hard-fail propagation, non-hard-fail `shadow_only`, and legacy `feature_flag_or_pass` normalization to `feature_flag`.
- [ ] Delete the test that protects `_actual_for_case` metadata synthesis for arbitrary case IDs.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_crc_harness_replay.py -q -p no:cacheprovider`; expect failures from the current synthetic actuals and hard-fail-only report aggregation.
- [ ] Make `HarnessRun` bind the mutation pack, hard-fail catalog, intended-use, active policy, immutable runtime snapshot ref plus labelled slot bindings, environment lock, exact command, and case results.
- [ ] Make `ReleaseSafetyReport.release_decision` use the most restrictive input disposition. Never overwrite the 2026-06-29 legacy reports; publish new IDs through `AtomicJsonArtifactPublisher`.
- [ ] Replace the script's current execute-on-any-argument/hard-coded-overwrite behavior with argparse. `--help` exits without running; publication requires explicit `--run-id`, `--report-id`, `--phase branch|post_merge|final`, `--mutation-pack`, `--environment-lock`, `--source-commit`, `--output-root`, and `--publish`. It first deterministically builds and validates both payloads, fsyncs their request/timestamp/hash phase context, then runs a resumable two-step phase: an exact existing HarnessRun or ReleaseSafetyReport is `already_complete`, a mismatched existing target is a collision block, and a missing second target is published after validating the first. The ignored phase context records ordered expected paths/refs and completed steps but is not evidence. Add failure injection before the initial context fsync, after each publish, and before/after each context update, including process restart. Run/report IDs are derived from the tested content SHA plus the declared phase, so a legitimate later source commit never collides with stale create-once evidence. A `--verify-against` mode writes only to a temporary output root and compares semantic hashes without mutating committed reports.
- [ ] Re-run that exact pytest command using an injected valid environment-lock fixture ref and require PASS. Do not publish repository candidate artifacts yet; create the real lock in Task 10, finish safety/release code in Tasks 11-12, and generate SHA-derived branch/post-merge evidence in Task 13.
- [ ] Commit only Task 7 paths:

```powershell
git add src/contracts/harness.py src/contracts/release_safety_report.py scripts/run_crc_harness_replay.py tests/backend/test_crc_harness_replay.py reports/harness/README.md reports/release_safety/README.md
git commit -m "fix(harness): propagate deterministic safety decisions"
```

---

### Task 8: Atomic CRC Persistence, Explicit Idempotency, And Provenance

**Files:**

- Modify: `backend/api/services/patient_registry_service.py`
- Modify: `backend/api/services/patient_commands.py`
- Modify: `backend/api/routes/crc_triage.py`
- Modify: `backend/api/routes/sessions.py`
- Modify: `backend/api/schemas/patient_registry.py`
- Modify: `backend/api/services/patient_care_cards.py`
- Modify: `backend/api/services/settings.py`
- Modify: `tests/backend/test_crc_triage_patient_commands.py`
- Modify: `tests/backend/test_crc_triage_save.py`
- Modify: `tests/backend/test_patient_event_sourcing.py`
- Modify: `tests/backend/test_patient_care_cards.py`
- Modify: `backend/api/routes/test_crc_triage_api.py`
- Modify: `backend/api/routes/test_session_patient_records_api.py`

**Additive response:**

```python
class PatientCommandResult:
    event_id: str
    assessment_ref: ClinicalVersionProjection
    record_ref: ClinicalVersionProjection
    snapshot_ref: ClinicalVersionProjection
    care_card_refs: tuple[ClinicalVersionProjection, ...]
```

Preserve existing plural IDs as deprecated compatibility projections. `SaveCrcTriageAssessmentRequest.idempotency_key` is required; the server separately computes the canonical payload hash.

- [ ] Add failing tests for same key/same payload replay, same key/different payload 409, opaque version refs, one transaction, and care-card provenance fields `derived_from_record_id`, `derived_from_event_id`, `derived_from_assessment_id`, and `safety_policy_version`.
- [ ] Add failure injection after event insert, after record insert, and before snapshot update. Each failure must leave no event, record, snapshot, ledger head, or idempotency row, must produce no care card at read time, and must allow a clean retry.
- [ ] Add audit tamper tests; public responses must not include the restricted handle, patient identifier hash, MAC, or key version.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_crc_triage_patient_commands.py tests/backend/test_crc_triage_save.py tests/backend/test_patient_event_sourcing.py tests/backend/test_patient_care_cards.py backend/api/routes/test_crc_triage_api.py backend/api/routes/test_session_patient_records_api.py -q -p no:cacheprovider`; observe current partial-write/idempotency/provenance failures.
- [ ] Refactor `record_crc_triage_assessment` into one SQLite transaction containing idempotency reservation, event, assessment/record/snapshot, clinical integrity rows, and audit head. Do not add a persisted care-card truth table.
- [ ] Make structured care-card content a deterministic read-time projection of committed records/events/assessments. Allocate a random public opaque card ID in a restricted `patient_care_card_projection_ids` identity-map row inside the same patient-write transaction; its internal logical-card key is a server-only keyed token over the patient-scoped committed refs and policy version. The map stores no card content and is not a second clinical truth. Reads reuse the mapping to produce a stable `ClinicalVersionProjection`; neither logical key, key version, patient binding, hash, handle, nor MAC crosses the API. Key rotation preserves existing random IDs and uses the new key only for new mappings; patient delete/clear removes the map transactionally. Test same logical card stability, cross-patient/cross-domain unlinkability, collision handling, rotation, deletion, and absence of public integrity fields. Preserve legacy string projections only as deprecated views of the same deterministic content.
- [ ] Re-run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_crc_triage_patient_commands.py tests/backend/test_crc_triage_save.py tests/backend/test_patient_event_sourcing.py tests/backend/test_patient_care_cards.py backend/api/routes/test_crc_triage_api.py backend/api/routes/test_session_patient_records_api.py -q -p no:cacheprovider
```

Expected: all tests pass with deterministic replay and zero residue after every injected failure.

- [ ] Commit only Task 8 paths:

```powershell
git add backend/api/services/patient_registry_service.py backend/api/services/patient_commands.py backend/api/routes/crc_triage.py backend/api/routes/sessions.py backend/api/schemas/patient_registry.py backend/api/services/patient_care_cards.py backend/api/services/settings.py tests/backend/test_crc_triage_patient_commands.py tests/backend/test_crc_triage_save.py tests/backend/test_patient_event_sourcing.py tests/backend/test_patient_care_cards.py backend/api/routes/test_crc_triage_api.py backend/api/routes/test_session_patient_records_api.py
git commit -m "fix(persistence): make CRC assessment provenance atomic"
```

---

### Task 9: Frontend Save Trace And Structured Care Cards

**Files:**

- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/features/patient-crc-triage/crc-triage-context.ts`
- Modify: `frontend/src/features/patient-crc-triage/patient-crc-triage-panel.tsx`
- Modify: `frontend/src/features/patient-records/patient-care-cards.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/features/patient-crc-triage/crc-triage-context.test.ts`
- Modify: `frontend/src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx`
- Modify: `frontend/src/features/patient-records/patient-care-cards.test.tsx`
- Modify: `frontend/src/pages/workspace-page.test.tsx`

- [ ] Add failing tests showing the context retains `crcTriageSaveTrace: SaveCrcTriageAssessmentResponse | null`, resends the same idempotency key on retry, and renders structured card provenance without exposing restricted fields.
- [ ] Run `cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/patient-crc-triage/crc-triage-context.test.ts src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx src/features/patient-records/patient-care-cards.test.tsx src/pages/workspace-page.test.tsx --reporter=verbose`; require the new assertions to FAIL for the missing retained trace/retry/provenance behavior before changing production code.
- [ ] Update API types/client, retain the save result for navigation/traceability, and render category/text/source version from structured cards.
- [ ] Run `cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/patient-crc-triage/crc-triage-context.test.ts src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx src/features/patient-records/patient-care-cards.test.tsx src/pages/workspace-page.test.tsx --reporter=verbose`; require PASS.
- [ ] Run `cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build`; require exit 0 with no type narrowing or compatibility errors.
- [ ] Commit only Task 9 paths:

```powershell
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/features/patient-crc-triage/crc-triage-context.ts frontend/src/features/patient-crc-triage/patient-crc-triage-panel.tsx frontend/src/features/patient-records/patient-care-cards.tsx frontend/src/pages/workspace-page.tsx frontend/src/app/api/client.test.ts frontend/src/features/patient-crc-triage/crc-triage-context.test.ts frontend/src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx frontend/src/features/patient-records/patient-care-cards.test.tsx frontend/src/pages/workspace-page.test.tsx
git commit -m "feat(frontend): retain CRC save provenance"
```

---

### Task 10: Reproducible Python And Frontend Environments

**Files:**

- Create: `uv.lock`
- Create: `config/environment-lock.json`
- Create: `scripts/validate_environment_lock.py`
- Create: `tests/backend/test_environment_lock.py`
- Modify: `pyproject.toml`
- Modify: `frontend/package.json`
- Modify mechanically: `frontend/package-lock.json`

- [ ] Add failing tests that require every resolved Python/frontend package version, source, integrity hash, Python/Node/npm/uv version, platform constraint, and environment manifest `VersionRef`.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_environment_lock.py -q -p no:cacheprovider`; expect missing lock/validator assertions to fail before generating either lock.
- [ ] Record the intended environment constraints explicitly. Use the observed bootstrap values only if still true: Python 3.10.19, Node 25.8.2, npm 11.11.1, and Windows x86_64.
- [ ] Generate `uv.lock` from `pyproject.toml`; generate/update `frontend/package-lock.json` only through npm.
- [ ] Build `config/environment-lock.json` from both lockfiles and verify its content hash.
- [ ] Validate the Task 7 runner against an injected environment-lock fixture under `output/` only. Do not publish the final HarnessRun/ReleaseSafetyReport in Task 10: Tasks 11-12 still change safety/release-contract paths, so Task 13 derives final branch IDs from their committed implementation SHA and publishes fresh evidence there.
- [ ] Run:

```powershell
D:\anaconda3\Scripts\uv.exe lock --check
D:\anaconda3\Scripts\uv.exe sync --frozen --python D:\anaconda3\envs\LangG\python.exe
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend ci --ignore-scripts
D:\anaconda3\envs\LangG\python.exe scripts\validate_environment_lock.py --manifest config\environment-lock.json
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_environment_lock.py -q -p no:cacheprovider
```

Expected: all commands exit 0 without modifying either lockfile.

- [ ] Commit only Task 10 environment/lock paths:

```powershell
git add uv.lock config/environment-lock.json scripts/validate_environment_lock.py tests/backend/test_environment_lock.py pyproject.toml frontend/package.json frontend/package-lock.json
git commit -m "build: lock Stage A Python and frontend environments"
```

---

### Task 11: Safety-Relevant Path Gate And Required CI

**Files:**

- Create: `config/safety_relevant_paths.yaml`
- Create: `scripts/validate_safety_change_gate.py`
- Create: `tests/backend/test_safety_change_gate.py`
- Create: `.github/workflows/safety-change-gate.yml`

- [ ] Add failing tests for explicit base/head SHAs, rename-aware diffs, unknown/renamed/generated/ambiguous paths, permanent safety categories, manifest expansion-only behavior, stale evidence, missing changed-artifact hash linkage, and invalid HarnessRun/ReleaseSafetyReport.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_safety_change_gate.py -q -p no:cacheprovider`; expect missing classifier/validator failures.
- [ ] Implement the classifier and validator. Runtime prompt, model, RAG/evidence index, tool, clinical policy, CRC runtime, persistence, and release-contract paths are permanently safety-relevant.
- [ ] Add a workflow covering `pull_request`, `merge_group`, and main/release pushes with `fetch-depth: 0` and one stable required-check name.
- [ ] Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_safety_change_gate.py -q -p no:cacheprovider
D:\anaconda3\envs\LangG\python.exe scripts\validate_safety_change_gate.py --manifest config\safety_relevant_paths.yaml --mode config-check --fixture tests\fixtures\crc_mutation_pack_v1.json
```

Expected: tests pass and config-check exits 0 without claiming current-branch release evidence. Task 13 is the only blocking invocation: it runs after Tasks 11-12 are committed and after SHA-derived fresh HarnessRun/ReleaseSafetyReport publication.

- [ ] Record repository-host evidence that the named check is required by branch protection or equivalent merge policy. The workflow file alone is insufficient evidence.
- [ ] Commit only Task 11 paths:

```powershell
git add config/safety_relevant_paths.yaml scripts/validate_safety_change_gate.py tests/backend/test_safety_change_gate.py .github/workflows/safety-change-gate.yml
git commit -m "ci: enforce safety-relevant change evidence"
```

---

### Task 12: Reusable StageGate Runner

**Files:**

- Create: `src/services/closeout_gate.py`
- Create: `src/services/closeout_stage_runner.py`
- Modify: `src/contracts/closeout.py`
- Create: `config/closeout_stage_suites.yaml`
- Create: `scripts/run_closeout_stage_gate.py`
- Create: `tests/backend/test_closeout_gate_contracts.py`
- Create: `tests/backend/test_closeout_stage_runner.py`

**Required StageGate bindings:** stage-plan VersionRef; latest approved manifest ID/hash/count and owned entry hashes; diff base, branch head, actual merged SHA when post-merge, and changed-path manifest hash; exact commands/results/toolchain/times/sanitized output refs; implementation/test/review/HarnessRun/ReleaseSafetyReport refs; ordered artifact-binding hash, optional orchestration-input ref, combined gate-input-binding hash, and optional superseded-report ref; compliance/disposition per row and aggregate per scope; applicable ReleaseBundle ref; prerequisite review-event refs; the ledger head before report approval; and report hash. Approval is a separate `StageGateApprovalAttestation` with distinct authorized principals/credentials and the post-approval ledger head.

```python
@dataclass(frozen=True, slots=True)
class StageGateReport:
    report_id: str
    stage_id: str
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
    manifest_required_entry_count: int
    owned_entry_hashes: tuple[str, ...]
    diff_base_sha: str
    branch_head_sha: str
    tested_content_sha: str
    merged_commit_sha: str | None
    changed_path_manifest_ref: VersionRef
    command_results: tuple[StageCommandResult, ...]
    artifact_refs: tuple[VersionRef, ...]
    artifact_binding_sha256: str
    orchestration_input_ref: VersionRef | None
    gate_input_binding_sha256: str
    supersedes_report_ref: VersionRef | None
    rows: tuple[StageGateRow, ...]
    compliance_status: Literal["pass", "block"]
    per_scope_dispositions: Mapping[str, str]
    release_bundle_ref: VersionRef | None
    prerequisite_review_event_refs: tuple[VersionRef, ...]
    pre_approval_ledger_head: str
    report_sha256: str


@dataclass(frozen=True, slots=True)
class StageGateApprovalAttestation:
    attestation_id: str
    stage_gate_report_ref: VersionRef
    approval_policy_ref: VersionRef
    approver_event_refs: tuple[VersionRef, ...]
    approver_principal_ids: tuple[str, ...]
    approver_credential_ids: tuple[str, ...]
    post_approval_ledger_head: str
    attestation_sha256: str
```

**Reusable runner interface:**

```python
@dataclass(frozen=True, slots=True)
class StageGateRunRequest:
    stage_id: str
    mode: Literal["branch-advisory", "post-merge"]
    base_sha: str
    head_sha: str
    tested_content_sha: str
    merged_sha: str | None
    plan_path: Path
    plan_subject_path: Path
    plan_attestation_path: Path
    manifest_attestation_path: Path
    suite_path: Path
    artifact_paths: tuple[Path, ...]
    orchestration_input_path: Path | None
    supersedes_report_path: Path | None
    output_root: Path
    publish: bool
    publisher_auth: AuthContext

def run_registered_stage_gate(request: StageGateRunRequest) -> StageGateRunResult: ...
```

The CLI resolves `publisher_auth` only from the protected `LANGG_CLOSEOUT_PUBLISHER_TOKEN` environment variable through the Stage A credential mapping; no command-line principal/role value can author a report.
Before running commands, it verifies `git cat-file` retrievability and blob SHA for `stage_plan_ref`, then requires `PlanApprovalAttestation.plan_subject_ref -> StagePlanApprovalSubject.plan_ref == StageGateReport.stage_plan_ref` with the latest approval-policy ref and ledger head. A plan path without that exact chain blocks branch-advisory and post-merge modes. `tested_content_sha` is the implementation or merged code/config SHA actually exercised; `head_sha` may be a descendant only when every intervening change is an allowlisted, validated evidence-only path. The runner rejects runtime/config drift between those SHAs and binds both values in the report. It canonically hashes ordered `(object_kind, repository_path, VersionRef)` artifact bindings, validates an optional explicit orchestration-input artifact such as CandidateReleaseInputs, and hashes both into `gate_input_binding_sha256`. A superseding report must explicitly name the exact prior report path, keep stage/merged/tested content fixed, change the combined gate-input-binding hash, and receive a new create-once ID containing that hash prefix; otherwise report overwrite, forked supersession, or approval carryover blocks.

- [ ] Add failing tests for all required StageGate fields, tracked/retrievable stage-plan blob and hash, exact plan-subject/plan-attestation chain, explicit subject version, server-derived report author, bound approval policy/quorum, author self-approval rejection, exact owned-entry sets, duplicate/unknown/missing IDs, source drift, stale manifest, explicit HarnessRun/ReleaseSafetyReport artifact paths, ordered artifact and orchestration-input binding hashes, valid single-parent report supersession, evidence-only head versus tested-content validation, branch-advisory versus post-merge mode, inherited regression, illegal dispositions, acyclic report/attestation hashing, and restart after every collector/report publish or phase-context boundary.
- [ ] Run `D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_gate_contracts.py tests/backend/test_closeout_stage_runner.py -q -p no:cacheprovider`; expect missing runner/contract failures before implementation.
- [ ] Implement fixed-cwd/fixed-argv suite execution with captured exit code, sanitized stdout/stderr digest, duration, skip policy, and produced artifact refs. Do not use shell-evaluated command strings from a manifest. Collector outputs and the final StageGateReport form a declared resumable phase: recompute all expected payloads from the immutable request, reuse an existing output only through `ensure_exact_artifact()`, continue from the first missing path, and record ordered progress in ignored output-root context. A crash after a StageDEvidenceSummary or report publish must restart successfully; any source/ref/hash mismatch blocks without overwrite.
- [ ] Define `config/closeout_stage_suites.yaml` with exact test/artifact IDs and an immediate-predecessor `inherits` list. Register Stage A commands and owned requirement IDs; recursively compute a stage-ID transitive closure in deterministic ancestor-first order, reject unknown/cyclic/repeated stage IDs and duplicate requirement/test/artifact IDs across the expanded closure, and record the complete expanded suite. Use the single chain B `inherits: [A]`, C `inherits: [B]`, and D `inherits: [C]`; do not flatten ancestors into multiple inheritance paths.
- [ ] Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_closeout_gate_contracts.py tests/backend/test_closeout_stage_runner.py -q -p no:cacheprovider
```

Expected: all contract and failure-injection tests pass.

- [ ] Commit the runner/suite implementation. Do not run branch advisory yet; Task 13 must first publish evidence derived from this final implementation commit:

```powershell
git add src/services/closeout_gate.py src/services/closeout_stage_runner.py src/contracts/closeout.py config/closeout_stage_suites.yaml scripts/run_closeout_stage_gate.py tests/backend/test_closeout_gate_contracts.py tests/backend/test_closeout_stage_runner.py
git commit -m "feat(closeout): add reusable stage gate runner"
```

Expected: commit succeeds; the worktree has no new implementation/config changes beyond the exact baseline exclusions.

---

### Task 13: Merge, Post-Merge Stage A Gate, And Approval Evidence

**Files:**

- Create before branch advisory: `reports/harness/harness_stage_a_branch_<content-sha12>.json`
- Create before branch advisory: `reports/release_safety/release_safety_stage_a_branch_<content-sha12>.json`
- Create after merge: `reports/harness/harness_stage_a_post_merge_<merge-sha12>.json`
- Create after merge: `reports/release_safety/release_safety_stage_a_post_merge_<merge-sha12>.json`
- Create after merge: `reports/closeout/stages/stage_a.<merge-sha12>.json`
- Create after report validation: `reports/closeout/attestations/stage_a_approval.<report-hash12>.json`

- [ ] Before merging, run the complete Stage A focused suites:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_integrity_contracts.py tests/backend/test_atomic_artifact_store.py tests/backend/test_write_boundary_sanitizer.py tests/backend/test_auth_context.py tests/backend/test_closeout_manifest_contracts.py tests/backend/test_closeout_manifest_validation.py tests/backend/test_clinical_safety_policy.py tests/backend/test_intended_use_profiles.py tests/backend/test_intended_use_resolver.py tests/backend/test_runtime_version_registry.py tests/backend/test_safety_policy_lifecycle.py tests/backend/test_safety_policy_api.py tests/backend/test_crc_triage_flow.py tests/backend/test_crc_triage_mutation_pack.py tests/backend/test_crc_harness_replay.py tests/backend/test_crc_triage_patient_commands.py tests/backend/test_crc_triage_save.py tests/backend/test_patient_event_sourcing.py tests/backend/test_patient_care_cards.py tests/backend/test_environment_lock.py tests/backend/test_safety_change_gate.py tests/backend/test_closeout_gate_contracts.py tests/backend/test_closeout_stage_runner.py backend/api/routes/test_crc_triage_api.py backend/api/routes/test_session_patient_records_api.py -q -p no:cacheprovider
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/app/api/client.test.ts src/features/patient-crc-triage/crc-triage-context.test.ts src/features/patient-crc-triage/patient-crc-triage-panel.test.tsx src/features/patient-records/patient-care-cards.test.tsx src/pages/workspace-page.test.tsx --reporter=verbose
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: all required tests pass, zero required skips, build exits 0.

- [ ] Publish SHA-derived branch evidence only after every Task 1-12 implementation commit, commit that evidence, run the sole blocking safety validator, and then run branch advisory:

```powershell
$implementationHead = git rev-parse HEAD
$implementationSha12 = $implementationHead.Substring(0, 12)
$branchRunId = "harness_stage_a_branch_$implementationSha12"
$branchReportId = "release_safety_stage_a_branch_$implementationSha12"
$branchHarnessPath = "reports/harness/$branchRunId.json"
$branchSafetyPath = "reports/release_safety/$branchReportId.json"
D:\anaconda3\envs\LangG\python.exe scripts/run_crc_harness_replay.py --run-id $branchRunId --report-id $branchReportId --phase branch --mutation-pack tests/fixtures/crc_mutation_pack_v1.json --environment-lock config/environment-lock.json --source-commit $implementationHead --output-root reports --publish
git add -- $branchHarnessPath $branchSafetyPath
git diff --cached --check
git commit -m "evidence(stage-a): publish branch safety replay"
$branchEvidenceHead = git rev-parse HEAD
$programBase = (Get-Content output/closeout/program-base.txt -Raw).Trim()
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-baseline --baseline reports/closeout/baselines/closeout_baseline_20260710_001.json --field base_commit).Trim()
if (($stageBase -notmatch '^[0-9a-f]{40}$') -or ($stageBase -ne $programBase)) { throw "Stage A base does not match the committed baseline" }
git cat-file -e "$stageBase`^{commit}"
if ($LASTEXITCODE -ne 0) { throw "Stage A base commit is unavailable" }
git merge-base --is-ancestor $stageBase $implementationHead
if ($LASTEXITCODE -ne 0) { throw "Stage A base is not an ancestor of implementation HEAD" }
D:\anaconda3\envs\LangG\python.exe scripts/validate_safety_change_gate.py --base-sha $programBase --head-sha $implementationHead --evidence-head-sha $branchEvidenceHead --manifest config/safety_relevant_paths.yaml --harness-run $branchHarnessPath --release-safety-report $branchSafetyPath --mode blocking
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_stage_gate.py --stage A --mode branch-advisory --base-sha $stageBase --head-sha $branchEvidenceHead --tested-content-sha $implementationHead --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-a-safety-persistence.md --plan-subject reports/closeout/plan_subjects/stage_a_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_a_plan_approval_20260710_001.json --manifest-attestation reports/closeout/attestations/manifest_approval_20260710_001.json --suite config/closeout_stage_suites.yaml --artifact $branchHarnessPath --artifact $branchSafetyPath --output-root output/closeout-advisory
```

Expected: the evidence IDs bind the exact final implementation SHA; the evidence commit contains only the two declared report paths; blocking validation and advisory pass. Any implementation/config change after `$implementationHead` invalidates both artifacts and requires new IDs derived from the new SHA before retry.

- [ ] Merge the branch through the repository's protected workflow. Capture the actual merge SHA; never substitute the branch/evidence head. From a clean checkout at that merge, publish and commit a fresh post-merge replay before the post-merge StageGate:

```powershell
$mergedSha = (git rev-parse HEAD).Trim()
$expectedMergedSha = $env:LANGG_STAGE_A_MERGE_SHA
if (($expectedMergedSha -notmatch '^[0-9a-f]{40}$') -or ($mergedSha -ne $expectedMergedSha)) { throw "checkout is not the recorded protected Stage A merge" }
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-baseline --baseline reports/closeout/baselines/closeout_baseline_20260710_001.json --field base_commit).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid Stage A baseline commit" }
git cat-file -e "$stageBase`^{commit}"
if ($LASTEXITCODE -ne 0) { throw "Stage A base commit is unavailable" }
git merge-base --is-ancestor $stageBase $mergedSha
if ($LASTEXITCODE -ne 0) { throw "Stage A merge does not descend from its recorded base" }
$mergeSha12 = $mergedSha.Substring(0, 12)
$postRunId = "harness_stage_a_post_merge_$mergeSha12"
$postReportId = "release_safety_stage_a_post_merge_$mergeSha12"
$postHarnessPath = "reports/harness/$postRunId.json"
$postSafetyPath = "reports/release_safety/$postReportId.json"
D:\anaconda3\envs\LangG\python.exe scripts/run_crc_harness_replay.py --run-id $postRunId --report-id $postReportId --phase post_merge --mutation-pack tests/fixtures/crc_mutation_pack_v1.json --environment-lock config/environment-lock.json --source-commit $mergedSha --output-root reports --publish
git add -- $postHarnessPath $postSafetyPath
git diff --cached --check
git commit -m "evidence(stage-a): publish post-merge safety replay"
$postEvidenceHead = git rev-parse HEAD
D:\anaconda3\envs\LangG\python.exe scripts/validate_safety_change_gate.py --base-sha $stageBase --head-sha $mergedSha --evidence-head-sha $postEvidenceHead --manifest config/safety_relevant_paths.yaml --harness-run $postHarnessPath --release-safety-report $postSafetyPath --mode blocking
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_stage_gate.py --stage A --mode post-merge --base-sha $stageBase --head-sha $postEvidenceHead --tested-content-sha $mergedSha --merged-sha $mergedSha --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-a-safety-persistence.md --plan-subject reports/closeout/plan_subjects/stage_a_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_a_plan_approval_20260710_001.json --manifest-attestation reports/closeout/attestations/manifest_approval_20260710_001.json --suite config/closeout_stage_suites.yaml --artifact $postHarnessPath --artifact $postSafetyPath --publish
```

Expected: post-merge evidence IDs bind the actual merge SHA, both evidence commits are path-bounded/runtime-neutral, and the registered runner repeats the complete Stage A suite. It publishes one immutable report under `reports/closeout/stages/`; every owned required row passes, source hashes and exact counts match, and the report does not embed its later approval.

- [ ] **Hard stop:** use `record_closeout_attestation.py record-approval --subject-kind stage_gate` once per policy-required distinct authorized credential with the published report path/hash/version. After quorum, derive, stage, verify, and commit only the report/attestation/events:

```powershell
$stageReportPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-published --subject-kind stage_gate --stage A --field subject_path).Trim()
$reportHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field sha256).Trim()
$reportVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field subject_version).Trim()
$reportHash12 = $reportHash.Substring(0, 12)
$attestationPath = "reports/closeout/attestations/stage_a_approval.$reportHash12.json"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_CLINICAL_SAFETY_REVIEWER_TOKEN --idempotency-key "$reportHash-stage-a-clinical-safety-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$reportHash-stage-a-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_gate --subject-path $stageReportPath --output $attestationPath --path-list-output output/closeout/stage-a-gate-evidence-paths.txt --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_gate --subject-path $stageReportPath --attestation-path $attestationPath
git add --pathspec-from-file=output/closeout/stage-a-gate-evidence-paths.txt
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list output/closeout/stage-a-gate-evidence-paths.txt --subject-path $stageReportPath --attestation-path $attestationPath
git commit -m "evidence(stage-a): record post-merge safety gate"
```

- [ ] Open Stage B only after the merged report plus attestation validate. A later shared-contract change must revalidate or revoke Stage A.

## Plan Self-Review Checklist

- [ ] Every Stage A design requirement maps to a task and a manifest-owned requirement row.
- [ ] Every created/modified path is exact, and generated artifacts have deterministic IDs or ID derivation rules.
- [ ] Shared types use `src/contracts/integrity.py` and `src/contracts/auth_context.py`; later stages consume rather than redefine them.
- [ ] Red/green commands use existing runners and state an expected failure/success condition.
- [ ] Public/patient-scoped integrity boundaries, path safety, sanitizer behavior, and failure rollback have explicit tests.
- [ ] Manifest, report, and approval hashes are acyclic; external approvals are never embedded in the payload they approve.
- [ ] The plan contains no placeholder implementation decisions; angle-bracket CLI values are runtime-derived SHAs/paths, not unresolved design work.
- [ ] `git diff --check` passes, all fenced code blocks close, and only this plan's files are staged per task.
