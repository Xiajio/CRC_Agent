# CRC Agent Closeout Stage D Research & Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the P2 gate by making cohort feasibility criteria real, persisting research governance, adding disclosure-controlled research artifacts, and turning LearningJob into a non-applying, auditable lifecycle.

**Architecture:** Stage D consumes the accepted Stage C evidence boundary and the shared integrity/AuthContext/audit contracts from `src/contracts/integrity.py` and `src/contracts/auth_context.py`. Research authorizes and reserves a query against sanitized projection metadata before loading one bounded patient-record snapshot, applies allowlisted criteria, and emits aggregate-only results. Patient-row snapshots remain internal `ClinicalVersionRef` objects; only `ClinicalVersionProjection` and sanitized public manifests cross the API boundary. LearningJob consumes opaque aggregate receipts, records state transitions in an append-only ledger, and delegates any feature-flag mutation to existing release governance/execution.

**Tech Stack:** Python 3.10, dataclasses, SQLite, FastAPI 0.135, Pydantic 2.12, React 18, TypeScript 5.6, Vitest 2.1, pytest.

## Global Constraints

- Start from the merged and accepted Stage C commit; do not implement this plan in parallel with Stages A-C. Before Task 1, build the Stage D `StagePlanApprovalSubject` from this plan's tracked blob/source commit and validate its policy-required `PlanApprovalAttestation` with Stage A's shared CLI.
- Consume `src/contracts/integrity.py`, `src/contracts/auth_context.py`, `src/services/atomic_artifact_store.py`, `src/services/write_boundary_sanitizer.py`, `src/services/runtime_version_registry.py`, and the inherited StageGate runner; do not redefine them.
- Research reads `patient_record_projection`, never session memory, and returns no patient-level row or resolvable patient reference.
- Unsupported criteria block before registry reads; supported criteria must materially change selection.
- Minimum-cell suppression and differencing protection happen before serialization.
- Patient-level export remains prohibited even after governance review.
- LearningJob never applies prompt, rubric, route, template, evidence, RAG, policy, model, training, or feature-flag changes directly.
- No hidden chain-of-thought, secret, credential, direct identifier, patient-linked hash, or doctor free text enters research or LearningJob artifacts.
- Every sanitized artifact stores the exact `sanitizer_ref: VersionRef`; every clinical-policy lineage stores an exact policy VersionRef. Bare sanitizer/policy labels are invalid evidence.
- Use `D:\anaconda3\envs\LangG\python.exe` and `D:\anaconda3\envs\LangG\npm.cmd` for verification on Windows.
- Do not modify `CRC-client/`, and stage only files named by the current task.

## Pre-Implementation Plan Authorization

- [ ] Resolve the exact tracked plan blob, publish its subject, collect the three policy-required external approvals, and exact-stage the resulting evidence:

```powershell
$planPath = "docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-d-research-learning.md"
$planSourceCommit = (git log -1 --format=%H -- $planPath).Trim()
git cat-file -e "${planSourceCommit}:$planPath"
$trackedPlanBlob = (git rev-parse "${planSourceCommit}:$planPath").Trim()
$workingPlanBlob = (git hash-object -- $planPath).Trim()
if ($trackedPlanBlob -ne $workingPlanBlob) { throw "Stage D plan blob is not tracked" }
$planSubjectPath = "reports/closeout/plan_subjects/stage_d_plan_20260710_001.json"
$planAttestationPath = "reports/closeout/attestations/stage_d_plan_approval_20260710_001.json"
$planEvidencePaths = "output/closeout/stage-d-plan-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py build-plan-subject --plan $planPath --source-commit $planSourceCommit --author-credential-env LANGG_CLOSEOUT_PUBLISHER_TOKEN --output $planSubjectPath
$planSubjectHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field sha256).Trim()
$planSubjectVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_plan --subject-path $planSubjectPath --field subject_version).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_PI_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-d-pi-reviewer-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_DATA_GOVERNANCE_REVIEWER_TOKEN --idempotency-key "$planSubjectHash-stage-d-data-governance-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_plan --subject-path $planSubjectPath --expected-sha256 $planSubjectHash --expected-version $planSubjectVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$planSubjectHash-stage-d-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_plan --subject-path $planSubjectPath --output $planAttestationPath --path-list-output $planEvidencePaths --include-subject
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_plan --subject-path $planSubjectPath --attestation-path $planAttestationPath
git add --pathspec-from-file=$planEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $planEvidencePaths --subject-path $planSubjectPath --attestation-path $planAttestationPath
git commit -m "evidence(closeout): approve Stage D plan"
```

- [ ] **Hard stop:** do not start Task 1 until the exact staged-set verifier and commit succeed and the subject plan ref/hash/version, approval-policy ref, author exclusion, quorum, ledger head, and latest-plan selection all validate.

## Source Design

- `docs/superpowers/specs/2026-07-10-crc-agent-closeout-program-design.md`, especially Sections 9-14.
- Existing baselines: `docs/superpowers/plans/2026-07-09-crc-cohort-feasibility.md` and `docs/superpowers/plans/2026-07-09-learningjob-candidate-pipeline.md`.

## File Structure

- `src/contracts/research_asset.py`: typed criteria, projection snapshot, aggregate result, and backward-compatible research contracts.
- `src/contracts/research_governance.py`: immutable review, dataset, analysis, hypothesis, protocol, and publication contracts.
- `src/contracts/learning_job_transition.py`: v2 lifecycle and transition guards.
- `backend/api/services/patient_registry_service.py`: complete, bounded projection snapshot reads.
- `backend/api/services/research_governance_store.py`: append-only review and research-asset store.
- `backend/api/services/research_query_ledger.py`: persistent HMAC query budget/differencing ledger.
- `backend/api/services/learning_signal_store.py`: restricted clinical lineage to opaque learning-signal receipts.
- `backend/api/services/learning_job_store.py`: creation snapshots plus transition ledger.
- `src/services/cohort_feasibility_service.py`: criteria filtering and disclosure-controlled aggregates.
- `src/services/research_governance_service.py`: review triggers and artifact guards.
- `src/services/learning_job_service.py`: v2 migration, transition, and target eligibility.
- `backend/api/routes/research.py`, `backend/api/routes/learning_jobs.py`: authorized APIs.
- `frontend/src/features/research/*`: aggregate-only Research workspace.
- `frontend/src/features/agent-admin/learning-job-panel.tsx`: read-only LearningJob lifecycle.
- `reports/research_governance/README.md`, `reports/learning_jobs/README.md`: path and integrity contracts.

---

### Task 1: Typed Cohort Criteria And Immutable Projection Snapshot

**Files:**
- Modify: `src/contracts/research_asset.py`
- Modify: `backend/api/services/patient_registry_service.py`
- Modify: `tests/backend/test_research_asset_contract.py`
- Modify: `tests/backend/test_cohort_feasibility_service.py`

**Interfaces:**
- Consumes: `VersionRef`, `ClinicalVersionRef`, and canonical hashing from `src/contracts/integrity.py`.
- Produces: `CohortCriteria.from_dict()`, `ResearchProjectionMetadata`, `ResearchProjectionSnapshot`, `PatientRegistryService.read_research_projection_metadata()`, and `PatientRegistryService.read_research_projection_snapshot(expected_ref=...)`.

- [ ] **Step 1: Write failing contract and snapshot tests**

```python
def test_criteria_reject_unknown_key_before_records_are_read() -> None:
    with pytest.raises(ValueError, match="unsupported cohort criteria: free_text"):
        CohortCriteria.from_dict({
            "condition": "colorectal_cancer_or_crc_triage_risk",
            "required_features": ["rectal_bleeding"],
            "free_text": "anything",
        })


def test_snapshot_reads_all_rows_at_one_max_record_id(tmp_path: Path) -> None:
    registry = seeded_registry(tmp_path, patient_count=1005)
    metadata = registry.read_research_projection_metadata()
    snapshot = registry.read_research_projection_snapshot(expected_ref=metadata.snapshot_ref)
    assert snapshot.total_records == 1005
    assert len(snapshot.records) == 1005
    assert snapshot.max_record_id == max(row["record_id"] for row in snapshot.records)
```

- [ ] Add criteria tests for allowlisted clinical stages, configured sites, inclusive date range, contradictory dates/bounds, duplicate values, unsupported site/stage/date mapping, filter limits/operators/types, and material effect on the selected aggregate. Every invalid/unmapped case must prove zero metadata and patient-row reads.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_asset_contract.py tests/backend/test_cohort_feasibility_service.py -q -p no:cacheprovider
```

Expected: FAIL because `CohortCriteria` and `read_research_projection_snapshot` do not exist and the current API truncates at 1000 rows.

- [ ] **Step 3: Implement the typed contract and bounded snapshot read**

```python
@dataclass(frozen=True)
class CohortCriteria:
    condition: str
    age_min: int | None
    age_max: int | None
    clinical_stages: tuple[str, ...]
    site_codes: tuple[str, ...]
    date_from: date | None
    date_to: date | None
    required_features: tuple[str, ...]
    reviewed_statuses: tuple[str, ...]
    filters: tuple[StructuredFilter, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, JsonValue]) -> "CohortCriteria":
        unknown = sorted(set(value) - ALLOWED_CRITERIA_KEYS)
        if unknown:
            raise ValueError(f"unsupported cohort criteria: {', '.join(unknown)}")
        return cls(
            condition=_required_condition(value.get("condition")),
            age_min=_optional_age("age_min", value.get("age_min")),
            age_max=_optional_age("age_max", value.get("age_max")),
            clinical_stages=tuple(_allowlisted_stages(value.get("clinical_stages", []))),
            site_codes=tuple(_allowlisted_sites(value.get("site_codes", []))),
            date_from=_optional_date("date_from", value.get("date_from")),
            date_to=_optional_date("date_to", value.get("date_to")),
            required_features=tuple(_required_features(value.get("required_features"))),
            reviewed_statuses=tuple(_reviewed_statuses(value.get("reviewed_statuses", []))),
            filters=tuple(_structured_filters(value.get("filters", []))),
        )


def read_research_projection_metadata(self) -> ResearchProjectionMetadata:
    return self._read_projection_metadata_without_patient_rows()


def read_research_projection_snapshot(self, *, expected_ref: ClinicalVersionRef) -> ResearchProjectionSnapshot:
    with self._connect() as connection:
        connection.execute("BEGIN")
        max_record_id = int(connection.execute(
            "SELECT COALESCE(MAX(record_id), 0) FROM patient_records"
        ).fetchone()[0])
        rows = connection.execute(RESEARCH_PROJECTION_SQL, (max_record_id,)).fetchall()
    return ResearchProjectionSnapshot.from_rows(expected_ref, max_record_id, [dict(row) for row in rows])
```

`ResearchProjectionMetadata` contains an internal `snapshot_ref: ClinicalVersionRef` and public sanitized-manifest `VersionRef` without patient rows. `RESEARCH_PROJECTION_SQL` must join `patient_snapshots` for age/stage/site, apply the date bound, select only allowlisted fields, use `record_id <= ?`, and contain no `LIMIT 1000`. Unsupported/unmapped stage, site, date, or filter values block before metadata or row reads.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the Step 2 command. Expected: all selected tests PASS; the 1005-row case proves there is no page cap.

- [ ] **Step 5: Commit Task 1**

```powershell
git add src/contracts/research_asset.py backend/api/services/patient_registry_service.py tests/backend/test_research_asset_contract.py tests/backend/test_cohort_feasibility_service.py
git commit -m "feat(research): apply typed cohort criteria to complete snapshots"
```

### Task 2: Persistent Research Review Queue

**Files:**
- Create: `src/contracts/research_governance.py`
- Create: `backend/api/services/research_governance_store.py`
- Create: `backend/api/services/research_approval_adapters.py`
- Create: `src/services/research_governance_service.py`
- Create: `tests/backend/test_research_governance_contract.py`
- Create: `tests/backend/test_research_governance_store.py`
- Create: `tests/backend/test_research_governance_service.py`
- Create: `tests/backend/test_research_approval_adapters.py`
- Create: `tests/backend/test_research_review_api.py`
- Create: `reports/research_governance/README.md`
- Modify: `backend/api/schemas/research.py`
- Modify: `backend/api/routes/research.py`
- Modify: `backend/app.py`

**Interfaces:**
- Consumes: `AuthContext` from `src/contracts/auth_context.py`, audit event/hash-chain helpers from `src/contracts/integrity.py`, and Stage A artifact ID/path validation.
- Produces: immutable `ReviewRequest`, append-only `ReviewDecisionEvent`/`ReviewRevocationEvent`/`ReviewExpiryEvent`/`ReviewDependencyRevocationEvent`, automatically created pending successors, validated upstream-approval adapters, `ResearchGovernanceStore`, and `ResearchGovernanceService.require_effective_approval()`.

- [ ] **Step 1: Write failing lifecycle, expiry, and separation-of-duties tests**

```python
def test_review_approval_requires_distinct_authorized_principal(tmp_path: Path) -> None:
    service = governance_service(tmp_path)
    request = service.ensure_review_request(scope=ethics_scope(), author=researcher())
    with pytest.raises(PermissionError, match="author cannot provide final approval"):
        service.record_decision(request.request_id, "approved", auth=researcher())


def test_revocation_blocks_bound_assets_and_creates_pending_successor(tmp_path: Path) -> None:
    service, approved = approved_ethics_review(tmp_path)
    successor = service.revoke(approved.request_id, reason="policy retired", auth=ethics_reviewer())
    assert successor.status == "pending"
    assert service.effective_status(approved.request_id) == "blocked"


def test_expiry_appends_event_and_one_pending_successor(tmp_path: Path) -> None:
    service, approved = approved_ethics_review(tmp_path, valid_until="2026-07-10T10:00:00Z")
    successor = service.reconcile_effective_reviews(now="2026-07-10T10:00:01Z").only_successor()
    assert service.events(approved.request_id)[-1].event_kind == "review_expired"
    assert successor.supersedes_request_id == approved.request_id
    assert service.reconcile_effective_reviews(now="2026-07-10T10:00:02Z").created == ()


def test_upstream_irb_revocation_propagates_to_bound_review(tmp_path: Path) -> None:
    service, approved = approved_review_bound_to_irb(tmp_path)
    revoke_irb_in_authoritative_store(approved.irb_approval_ref)
    successor = service.reconcile_effective_reviews(now=NOW).only_successor()
    assert service.events(approved.request_id)[-1].event_kind == "dependency_revoked"
    assert successor.status == "pending"
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_governance_contract.py tests/backend/test_research_governance_store.py tests/backend/test_research_governance_service.py tests/backend/test_research_approval_adapters.py tests/backend/test_research_review_api.py -q -p no:cacheprovider
```

Expected: FAIL because the contracts and store do not exist.

- [ ] **Step 3: Implement event-sourced review state**

```python
@dataclass(frozen=True)
class ReviewRequest:
    request_id: str
    review_type: ReviewType
    subject_ref: VersionRef | ClinicalVersionRef
    scope_hash: str
    required_role: str
    author_principal_id: str
    created_at: str
    valid_until: str | None
    supersedes_request_id: str | None


class ResearchGovernanceService:
    def reconcile_effective_reviews(self, *, now: str) -> ReviewReconciliationResult: ...

    def require_effective_approval(self, *, review_type: str, scope_hash: str) -> ReviewRequest:
        self.reconcile_effective_reviews(now=self._now())
        request = self._store.latest_for_scope(review_type, scope_hash)
        if request is None or self._store.effective_status(request.request_id, self._now()) != "approved":
            raise ResearchGovernanceBlocked(f"{review_type} approval is required")
        return request
```

Patient-projection review requests live in the restricted governance store and serialize only `ClinicalVersionProjection`; public-artifact subjects use VersionRef. Store writes must use same-key/same-payload idempotency, expected subject version, one aggregate event artifact (or a restricted SQLite transaction plus durable outbox), and hash-chain validation on every read. `reconcile_effective_reviews()` reads authorization-policy, de-identification-policy, ethics, institutional IRB, data-governance, and publication approvals through explicit authoritative adapters. For every elapsed `valid_until` it atomically appends exactly one `ReviewExpiryEvent` plus one linked pending successor; for every revoked/retired/mismatched upstream approval it appends exactly one `ReviewDependencyRevocationEvent` plus one successor. A deterministic reconciliation idempotency key prevents duplicate events after restart. Passive wall-clock calculation alone is not an allowed terminal-state implementation.

The server-side policy maps `research_ethics_review -> ethics_reviewer`, `pi_review -> pi_reviewer`, `data_governance_review -> data_governance_reviewer`, and `publication_review -> publication_reviewer`; authoritative institutional IRB decisions require the separate `irb_reviewer` credential. Artifact authors use `researcher` and cannot satisfy any of those reviewer roles for their own subject. All six roles resolve through the Stage A credential mapping and exact project scope; a caller-supplied role string is ignored and rejected when authority-bearing.

Add `POST /api/admin/research/reviews/{request_id}/decisions`, `POST /api/admin/research/reviews/{request_id}/revocations`, and an idempotent protected `POST /api/admin/research/reviews/reconcile` operator endpoint. Requests require expected request version, idempotency key, allowlisted decision/reason code, sanitized bounded reason, valid-until and IRB fields when applicable. The route derives AuthContext, calls `require_project_scope` for the request's project/required role, rejects author self-approval/cross-project/stale/terminal writes, and returns the derived queue projection. API tests cover approve/reject/manual revoke/automatic expiry/upstream revocation propagation, same-key replay, conflicting payload, audit tamper, restart reconciliation, and zero patient-row reads before effective ethics approval.

- [ ] **Step 4: Run tests and confirm GREEN**

Run the Step 2 command. Expected: PASS, including stale-version, duplicate-key, expiry, revocation, and chain-tamper cases.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/contracts/research_governance.py backend/api/services/research_governance_store.py backend/api/services/research_approval_adapters.py src/services/research_governance_service.py backend/api/schemas/research.py backend/api/routes/research.py backend/app.py tests/backend/test_research_governance_contract.py tests/backend/test_research_governance_store.py tests/backend/test_research_governance_service.py tests/backend/test_research_approval_adapters.py tests/backend/test_research_review_api.py reports/research_governance/README.md
git commit -m "feat(research): persist governed review decisions"
```

### Task 3: Query Ledger, Criteria Filtering, And Disclosure Control

**Files:**
- Create: `backend/api/services/research_query_ledger.py`
- Modify: `src/services/cohort_feasibility_service.py`
- Modify: `backend/api/services/settings.py`
- Modify: `tests/backend/test_cohort_feasibility_service.py`
- Create: `tests/backend/test_research_query_ledger.py`

**Interfaces:**
- Consumes: `CohortCriteria`, `ResearchProjectionMetadata`, internal `ResearchProjectionSnapshot`, effective ethics approval, and `AuthContext`.
- Produces: `ResearchQueryPermit` and `CohortFeasibilityService.evaluate(request, snapshot, permit)`.

- [ ] **Step 1: Write failing selection, suppression, and ledger tests**

The tests also assert the permit/result bind the exact effective approval VersionRef, revocation between reservation/read/finalization blocks without serialization, and the public result exposes only safe review refs—not query HMAC, key version, or restricted handles.

```python
def test_age_and_condition_criteria_change_estimated_count() -> None:
    snapshot = snapshot_for_ages([42, 55, 67])
    result = service().evaluate(request(age_min=50), snapshot, permit())
    assert result.estimated_count == 2
    assert result.applied_criteria["age_min"] == 50


def test_small_cell_is_suppressed_before_serialization() -> None:
    result = service(minimum_cell_size=5).evaluate(request(), snapshot_for_count(3), permit())
    assert result.estimated_count is None
    assert result.count_disclosure == "suppressed"
    assert "3" not in json.dumps(result.to_dict())


def test_ledger_unavailable_blocks_before_snapshot_loader(tmp_path: Path) -> None:
    ledger = ResearchQueryLedger(tmp_path / "ledger.db", hmac_key=None)
    with pytest.raises(ResearchQueryBlocked, match="query ledger unavailable"):
        ledger.reserve(
            auth_context(),
            request(),
            projection_metadata(),
            approval_ref=ethics_approval_ref(),
            idempotency_key="query-ledger-unavailable-001",
        )
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_query_ledger.py tests/backend/test_cohort_feasibility_service.py -q -p no:cacheprovider
```

Expected: FAIL because the ledger, applied criteria, and suppression fields do not exist.

- [ ] **Step 3: Implement atomic HMAC ledger and filtered aggregation**

```python
class ResearchQueryLedger:
    def reserve(
        self,
        auth: AuthContext,
        request: CohortFeasibilityRequest,
        metadata: ResearchProjectionMetadata,
        approval_ref: VersionRef,
        idempotency_key: str,
    ) -> ResearchQueryPermit:
        scope = canonical_query_scope(request, metadata.sanitized_manifest_ref)
        query_hmac = hmac.new(self._key, scope, hashlib.sha256).hexdigest()
        with self._transaction() as connection:
            self._enforce_budget_and_differencing(connection, auth, request.project_id, query_hmac)
            values = build_query_ledger_row(
                auth=auth,
                request=request,
                projection_ref=metadata.snapshot_ref,
                approval_ref=approval_ref,
                query_hmac=query_hmac,
                key_version=self._key_version,
                idempotency_key=idempotency_key,
            )
            connection.execute(INSERT_QUERY_SQL, values)
        return ResearchQueryPermit.from_restricted_ledger(values)

    def finalize_disclosure(
        self,
        permit: ResearchQueryPermit,
        disclosure_fingerprint: str,
        idempotency_key: str,
    ) -> None:
        self._atomically_record_disclosed_bucket_or_block_overlap(
            permit,
            disclosure_fingerprint,
            idempotency_key,
        )


def _eligible_patients(criteria: CohortCriteria, observations: Iterable[_Observation]) -> set[str]:
    return {
        row.patient_id for row in observations
        if _condition_matches(criteria.condition, row)
        and _age_matches(criteria.age_min, criteria.age_max, row.age)
        and _stage_matches(criteria.clinical_stages, row.clinical_stage)
        and _site_matches(criteria.site_codes, row.site_code)
        and _date_matches(criteria.date_from, criteria.date_to, row.recorded_at)
        and _required_features_match(criteria.required_features, row.features)
        and _review_status_matches(criteria.reviewed_statuses, row.reviewed_status)
        and _filters_match(criteria.filters, row)
    }
```

`ResearchQueryPermit` binds the exact effective ethics-approval VersionRef, project/purpose/criteria scope, and snapshot ref. The service revalidates that approval immediately before loading rows and again before disclosure finalization. Query HMAC, key version, and disclosure fingerprint are restricted internal values and never serialize into DatasetVersion, aggregate results, logs, or APIs. The aggregate result returns only safe review VersionRefs. Reservation blocks before row reads; finalization runs after suppression/bucketing but before serialization and atomically records the disclosed bucket so overlapping-query reconstruction is restart-safe. Finalization failure returns no result. `RuntimeSettings` gains `research_query_hmac_key`, `research_query_hmac_key_version`, `research_minimum_cell_size`, and `research_query_budget`; startup rejects missing keys when the research route is enabled.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the Step 2 command. Expected: PASS, including restart persistence, concurrent quota, overlap/differencing, and no-raw-query persistence tests.

- [ ] **Step 5: Commit Task 3**

```powershell
git add backend/api/services/research_query_ledger.py src/services/cohort_feasibility_service.py backend/api/services/settings.py tests/backend/test_research_query_ledger.py tests/backend/test_cohort_feasibility_service.py
git commit -m "feat(research): enforce disclosure-controlled cohort queries"
```

### Task 4: Authorized Research API And Aggregate Workspace

**Files:**
- Modify: `backend/api/schemas/research.py`
- Modify: `backend/api/routes/research.py`
- Modify: `backend/app.py`
- Modify: `tests/backend/test_research_api.py`
- Modify: `tests/backend/test_auth_security.py`
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Modify: `frontend/src/features/workspace/workspace-surface-switcher.tsx`
- Modify: `frontend/src/features/workspace/workspace-surface-switcher.test.tsx`
- Modify: `frontend/src/pages/workspace-page.tsx`
- Modify: `frontend/src/pages/workspace-page.test.tsx`
- Create: `frontend/src/features/research/research-workspace.tsx`
- Create: `frontend/src/features/research/research-workspace.test.tsx`
- Create: `frontend/src/features/research/cohort-feasibility-panel.tsx`
- Create: `frontend/src/features/research/cohort-feasibility-panel.test.tsx`
- Modify: `frontend/src/features/research/research-evidence-view.tsx`
- Modify: `frontend/src/features/research/research-evidence-view.test.tsx`
- Modify: `frontend/src/styles/globals.css`

**Interfaces:**
- Consumes: research services, `request.state.auth_context`, and Stage C's read-only `ResearchEvidenceView`.
- Produces: `POST /api/admin/research/cohort-feasibility`, `GET /api/admin/research/reviews`, `ApiClient.evaluateCohortFeasibility()`, and a `research` workspace wrapper that preserves EvidenceClaim cards while adding cohort/review panels.

- [ ] **Step 1: Write failing API and UI tests**

```python
def test_api_blocks_registry_read_until_ethics_scope_is_approved() -> None:
    response = client().post("/api/admin/research/cohort-feasibility", json=payload(), headers=research_headers())
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "research_ethics_review_required"
    request_ref = VersionRef.from_dict(detail["review_request_ref"])
    persisted = governance_store().load_request(request_ref)
    assert persisted.review_type == "research_ethics_review"
    assert persisted.scope_hash == expected_ethics_scope_hash(payload())
    assert registry_stub.patient_row_read_count == 0
```

```tsx
it("renders aggregate feasibility without patient rows", async () => {
  render(<ResearchWorkspace apiClient={apiClientWithFeasibility()} />);
  await userEvent.click(screen.getByRole("button", { name: "运行可行性评估" }));
  expect(await screen.findByText("样本量：已抑制")).toBeInTheDocument();
  expect(screen.queryByText(/patient_id|record_id/)).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_api.py tests/backend/test_auth_security.py -q -p no:cacheprovider
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/research/research-evidence-view.test.tsx src/features/research/cohort-feasibility-panel.test.tsx src/features/research/research-workspace.test.tsx src/features/workspace/workspace-surface-switcher.test.tsx src/pages/workspace-page.test.tsx src/app/api/client.test.ts
```

Expected: backend and frontend FAIL because the scoped endpoints/types/surface do not exist.

- [ ] **Step 3: Implement the API and workspace**

```python
@router.post("/cohort-feasibility")
async def evaluate_cohort_feasibility(payload: CohortFeasibilityRequestPayload, request: Request) -> dict[str, Any]:
    validated = CohortFeasibilityRequest.from_payload(payload.model_dump())
    auth = request.state.auth_context
    require_project_scope(auth, validated.project_id, "researcher")
    metadata = registry_service(request).read_research_projection_metadata()
    approval = governance_service(request).require_ethics_approval(validated, metadata)
    permit = query_ledger(request).reserve(
        auth,
        validated,
        metadata,
        approval_ref=approval.version_ref,
        idempotency_key=validated.idempotency_key,
    )
    governance_service(request).revalidate_effective_approval(permit.approval_ref)
    snapshot = registry_service(request).read_research_projection_snapshot(
        expected_ref=metadata.snapshot_ref,
    )
    result = feasibility_service(request).evaluate(validated, snapshot, permit)
    governance_service(request).revalidate_effective_approval(permit.approval_ref)
    query_ledger(request).finalize_disclosure(
        permit,
        result.restricted_disclosure_fingerprint,
        idempotency_key=validated.idempotency_key,
    )
    return result.to_dict()
```

`CohortFeasibilityRequestPayload.idempotency_key` is required. Unknown/unsupported criteria validation occurs before metadata access. Projection metadata contains no patient rows. On the first valid scan request, `require_ethics_approval()` atomically calls `ensure_review_request()`, persists a real `research_ethics_review` request bound to exact snapshot/policy/purpose/criteria scope, and returns its sanitized VersionRef in the blocking 409 before any patient row read. Same-key/same-payload retry returns the same request; scope drift creates a successor. Authorization, request persistence, approval validation, budget/differencing reservation, and projection-ref binding all occur before patient-row reads.

```tsx
export type WorkspaceSurface = "patient" | "doctor" | "research" | "agent-admin";

if (activeSurface === "research") {
  return <ResearchWorkspace apiClient={apiClient} surfaceSwitcher={workspaceSurfaceSwitcher} />;
}
```

`ResearchWorkspace` renders `<ResearchEvidenceView>` and `<CohortFeasibilityPanel>` as sibling read-only/governed sections. Its tests require the full Stage C EvidenceClaim card fields and no evidence POST controls to remain visible while cohort criteria/results/review state render below; Stage D must not replace or duplicate the Stage C view.

- [ ] **Step 4: Run backend/frontend tests and build**

Run Step 2 commands, then:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: tests PASS and build exits 0.

- [ ] **Step 5: Commit Task 4**

```powershell
git add backend/api/schemas/research.py backend/api/routes/research.py backend/app.py tests/backend/test_research_api.py tests/backend/test_auth_security.py frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/features/workspace/workspace-surface-switcher.tsx frontend/src/features/workspace/workspace-surface-switcher.test.tsx frontend/src/pages/workspace-page.tsx frontend/src/pages/workspace-page.test.tsx frontend/src/features/research/research-workspace.tsx frontend/src/features/research/research-workspace.test.tsx frontend/src/features/research/cohort-feasibility-panel.tsx frontend/src/features/research/cohort-feasibility-panel.test.tsx frontend/src/features/research/research-evidence-view.tsx frontend/src/features/research/research-evidence-view.test.tsx frontend/src/styles/globals.css
git commit -m "feat(research): add governed aggregate research workspace"
```

### Task 5: Dataset, Analysis, Hypothesis, Protocol, And Publication Artifacts

**Files:**
- Create: `config/research/codebook.yaml`
- Create: `config/research/data_dictionary.yaml`
- Create: `config/research/deidentification_policy.yaml`
- Create: `config/research/access_policy.yaml`
- Create: `src/services/research_metadata_catalog.py`
- Create: `tests/backend/test_research_metadata_catalog.py`
- Modify: `src/contracts/research_governance.py`
- Modify: `backend/api/services/research_governance_store.py`
- Modify: `src/services/research_governance_service.py`
- Modify: `backend/api/schemas/research.py`
- Modify: `backend/api/routes/research.py`
- Modify: `tests/backend/test_research_governance_contract.py`
- Modify: `tests/backend/test_research_governance_service.py`
- Modify: `tests/backend/test_research_api.py`

**Interfaces:**
- Consumes: approved review requests, disclosure-controlled cohort result, and validated VersionRefs from `ResearchMetadataCatalog`.
- Produces: `DatasetVersion`, `AnalysisRun`, `HypothesisDraft`, `ProtocolDraft`, and `PublicationIntent` APIs without patient rows.

- [ ] **Step 1: Write failing artifact guard tests**

```python
def test_dataset_version_contains_only_disclosed_metadata() -> None:
    dataset = service().create_dataset_version(result=suppressed_result(), auth=data_governance_reviewer())
    payload = dataset.to_dict()
    assert payload["row_count"] is None
    assert payload["count_disclosure"] == "suppressed"
    assert "patient_id" not in json.dumps(payload)


def test_irb_required_blocks_analysis_without_matching_approval() -> None:
    with pytest.raises(ResearchGovernanceBlocked, match="IRB approval is required"):
        service().create_analysis_run(dataset_ref(), protocol_with_irb_required(), auth=researcher())


@pytest.mark.parametrize(
    ("operation", "review_type"),
    [
        ("create_hypothesis", "pi_review"),
        ("create_protocol", "pi_review"),
        ("create_dataset", "data_governance_review"),
        ("create_analysis", "data_governance_review"),
        ("create_publication_intent", "publication_review"),
    ],
)
def test_artifact_creation_persists_mandatory_review_request(operation: str, review_type: str) -> None:
    blocked = invoke_artifact_operation(operation)
    request = governance_store().load_request(blocked.review_request_ref)
    assert request.review_type == review_type
    assert request.subject_ref == blocked.trigger_subject_ref
    assert request.status == "pending"
```

- [ ] Add contract/API tests for every DatasetVersion/AnalysisRun/HypothesisDraft/ProtocolDraft/PublicationIntent field listed below, public omission of query HMAC/key version/restricted handles, codebook and missingness disclosure control, artifact idempotency conflicts, stale expected versions, draft-only status, and absence of patient rows or clinical recommendations. For each mandatory trigger, assert the request is persisted before the operation blocks, the request ref is returned, its subject ref equals the triggering draft/metadata ref, pending/rejected/expired/revoked state blocks use, and only an exact effective approval releases the allowed next step.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_metadata_catalog.py tests/backend/test_research_governance_contract.py tests/backend/test_research_governance_service.py tests/backend/test_research_api.py -q -p no:cacheprovider
```

Expected: FAIL because the artifact classes/endpoints do not exist.

- [ ] **Step 3: Implement immutable artifact contracts and guards**

```python
@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    version: str
    opaque_query_id: str
    query_ledger_receipt_ref: ClinicalVersionRef
    sanitized_criteria_sha256: str
    criteria_manifest_ref: VersionRef
    field_manifest: tuple[DatasetFieldManifestEntry, ...]
    codebook_ref: VersionRef
    data_dictionary_ref: VersionRef
    completeness_summary: Mapping[str, CountBucket]
    missingness_summary: Mapping[str, CountBucket]
    row_count: int | None
    count_disclosure: Literal["exact", "bucketed", "suppressed"]
    projection_ref: ClinicalVersionRef
    projection_manifest_ref: VersionRef
    deidentification_policy_ref: VersionRef
    access_policy_ref: VersionRef
    governance_review_refs: tuple[VersionRef, ...]
    provenance_refs: tuple[VersionRef, ...]
    sanitized_manifest_sha256: str
```

The restricted query-ledger receipt contains the keyed query HMAC and key version; neither value is emitted by `DatasetVersion.to_public_dict()` or any API. The public projection uses `ClinicalVersionProjection`, the opaque query ID, sanitized criteria/field metadata, bucketed completeness/missingness, and public policy/provenance VersionRefs. Codebook/data-dictionary entries contain definitions only, never rare values or row examples.

`ResearchMetadataCatalog` loads only the four fixed YAML paths above, validates schema/version/canonical semantic hash, rejects unknown fields and patient/example values, and returns retrievable `VersionRef` objects bound to the source commit. Dataset creation obtains `codebook_ref`, `data_dictionary_ref`, `deidentification_policy_ref`, and `access_policy_ref` from this server-side catalog; the API cannot accept them from a request. Tests cover missing/malformed/unhashed/drifted catalogs, rare-value/example leakage, source-commit mismatch, and deterministic refs.

`AnalysisRun` must declare run/version, approved DatasetVersion projection, ProtocolDraft ref, aggregate method/toolchain VersionRefs, sanitized aggregate-output refs, governance state, idempotency/audit refs, and no executable notebook or patient rows. `HypothesisDraft` and `ProtocolDraft` must each declare research question, falsification conditions, EvidenceClaim/other evidence refs, bias considerations, required review types, draft-only status, author/project binding, supersedes ref, schema/sanitizer refs, and exact idempotency/version metadata; ProtocolDraft additionally declares population/exposure/comparator/outcome, aggregate analysis plan, IRB determination/ref, and disclosure constraints. `PublicationIntent` declares intent/version, paper|grant|patent kind, source/citation-check refs, contribution statement, privacy/de-identification attestation ref, institution-policy ref/decision, project/author, draft-only status, supersedes/idempotency/audit refs, and no externalized content. They never emit clinical recommendations.

```python
def create_protocol_draft(self, hypothesis: HypothesisDraft, decision: ReviewDecisionEvent) -> ProtocolDraft:
    if decision.irb_determination == "uncertain":
        raise ResearchGovernanceBlocked("IRB determination is unresolved")
    if decision.irb_determination == "required":
        self._require_matching_irb_approval(decision)
    return ProtocolDraft.from_hypothesis(hypothesis, decision)
```

Creating a HypothesisDraft or ProtocolDraft persists a `pi_review` request and leaves the artifact draft-only; creating DatasetVersion metadata or AnalysisRun persists a `data_governance_review` request and suppresses restricted metadata while pending; creating PublicationIntent persists a `publication_review` request and blocks every export/externalization. Publication approval requires source VersionRefs/citation check, contribution statement, privacy attestation, and institution-policy VersionRef. The returned artifact/request refs are stored together in the same transaction/outbox boundary, and no response-construction shortcut may fabricate a pending request.

Every artifact-creation POST requires `idempotency_key` and the expected current subject/version where it supersedes another artifact. Same key/same canonical payload returns the original object; the same key with a different payload or a stale expected version conflicts without an event. Restricted metadata, public projection/outbox record, audit event, and idempotency result commit in one SQLite transaction; the public artifact publisher drains the durable outbox idempotently.

- [ ] **Step 4: Run tests and confirm GREEN**

Run Step 2 command. Expected: PASS, including pending/rejected/expired/revoked approval and no-patient-row serialization tests.

- [ ] **Step 5: Commit Task 5**

```powershell
git add config/research/codebook.yaml config/research/data_dictionary.yaml config/research/deidentification_policy.yaml config/research/access_policy.yaml src/services/research_metadata_catalog.py tests/backend/test_research_metadata_catalog.py src/contracts/research_governance.py backend/api/services/research_governance_store.py src/services/research_governance_service.py backend/api/schemas/research.py backend/api/routes/research.py tests/backend/test_research_governance_contract.py tests/backend/test_research_governance_service.py tests/backend/test_research_api.py
git commit -m "feat(research): add governed research draft artifacts"
```

### Task 6: Opaque Learning Signal Receipts

**Files:**
- Create: `backend/api/services/learning_signal_store.py`
- Modify: `src/contracts/learning_job.py`
- Modify: `src/services/learning_job_service.py`
- Create: `tests/backend/test_learning_signal_store.py`
- Modify: `tests/backend/test_learning_job_contract.py`
- Modify: `tests/backend/test_learning_job_service.py`

**Interfaces:**
- Consumes four bounded source adapters: threshold-satisfying DoctorActionTrace aggregate batches, Stage C EvidenceDelta aggregate summaries, sanitized Harness failure summaries, and disclosure-controlled cohort-gap summaries.
- Produces: opaque `LearningSignalReceipt`, durable public-receipt outbox, and restricted lineage mapping; LearningJob sees no source event/patient/query ID or source free text.

- [ ] **Step 1: Write failing privacy-boundary tests**

```python
def test_job_receipt_does_not_expose_trace_or_patient_ids(tmp_path: Path) -> None:
    receipt = store(tmp_path).create_receipt(
        trace_batch(),
        auth=clinical_lineage_reviewer(),
        idempotency_key="receipt-doctor-trace-001",
    )
    payload = receipt.to_dict()
    assert payload["learning_signal_id"].startswith("learning_signal_")
    assert "source_event_id" not in json.dumps(payload)
    assert "patient_id" not in json.dumps(payload)


def test_free_text_is_rejected_before_receipt_creation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="free text is forbidden"):
        store(tmp_path).create_receipt(
            {"doctor_text": "patient story"},
            auth=clinical_lineage_reviewer(),
            idempotency_key="receipt-invalid-001",
        )
```

- [ ] Add one adapter/receipt test for each of the four source types, plus threshold, sanitizer, non-linkability, cross-project rejection, same-key replay, conflicting-key payload, crash-before-outbox-drain, and idempotent outbox recovery tests.

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_signal_store.py tests/backend/test_learning_job_contract.py tests/backend/test_learning_job_service.py -q -p no:cacheprovider
```

Expected: FAIL because the receipt store does not exist and current LearningSignal accepts source refs directly.

- [ ] **Step 3: Implement restricted lineage and public receipt**

```python
@dataclass(frozen=True)
class LearningSignalReceipt:
    learning_signal_id: str
    project_scope_ref: VersionRef
    signal_type: str
    summary_counts: dict[str, int]
    sanitizer_ref: VersionRef
    source_policy_ref: VersionRef
    threshold_attestation: VersionRef


def create_receipt(
    self,
    source: LearningSignalSource,
    auth: AuthContext,
    idempotency_key: str,
) -> LearningSignalReceipt:
    require_project_scope(auth, source.project_id, "clinical_lineage_reviewer")
    _reject_identifiers_text_and_linkable_hashes(source)
    receipt, restricted_lineage = _build_receipt_and_lineage(source)
    return self._commit_lineage_receipt_and_outbox(
        receipt,
        restricted_lineage,
        idempotency_key=idempotency_key,
    )
```

Every LearningSignalSource carries an internal project ID; the public receipt emits only its access-policy/project-scope VersionRef. Cross-project source/auth combinations are rejected before reads. `_commit_lineage_receipt_and_outbox` writes restricted lineage, receipt payload, audit event, idempotency result, and durable outbox row in one SQLite transaction. `AtomicJsonArtifactPublisher` drains the outbox create-once; a crash or collision is retried from the committed outbox and never leaves a LearningJob-visible receipt without its restricted lineage. Public receipts use exact sanitizer/policy VersionRefs, not bare version labels.

- [ ] **Step 4: Run tests and confirm GREEN**

Run Step 2 command. Expected: PASS; serialized job/receipt output contains no patient-resolvable value.

- [ ] **Step 5: Commit Task 6**

```powershell
git add backend/api/services/learning_signal_store.py src/contracts/learning_job.py src/services/learning_job_service.py tests/backend/test_learning_signal_store.py tests/backend/test_learning_job_contract.py tests/backend/test_learning_job_service.py
git commit -m "feat(learning): isolate clinical lineage from learning jobs"
```

### Task 7: LearningJob V2 Transition Ledger And Migration

**Files:**
- Create: `src/contracts/learning_job_transition.py`
- Modify: `backend/api/services/learning_job_store.py`
- Modify: `src/services/learning_job_service.py`
- Modify: `backend/api/schemas/learning_jobs.py`
- Modify: `backend/api/routes/learning_jobs.py`
- Modify: `tests/backend/test_learning_job_store.py`
- Modify: `tests/backend/test_learning_job_service.py`
- Modify: `tests/backend/test_learning_jobs_api.py`

**Interfaces:**
- Consumes: immutable job snapshot, opaque receipts, HarnessRun/review/governance/execution/monitoring/rollback VersionRefs.
- Produces: `LearningJobTransitionEvent`, `LearningJobProjection`, `POST /api/admin/learning-jobs/{job_id}/transitions`.

- [ ] **Step 1: Write failing state, migration, and partial-execution tests**

Include cross-project create/read/transition rejection tests and prove the store is not touched before project-scope authorization succeeds.

```python
def test_transition_rejects_missing_harness_without_state_change(tmp_path: Path) -> None:
    service = job_service(tmp_path)
    job = service.create_job(valid_request())
    with pytest.raises(LearningJobTransitionError, match="HarnessRun reference is required"):
        service.transition(
            job.job_id,
            "awaiting_human_review",
            refs={},
            auth=reviewer(),
            expected_subject_version=job.version,
            idempotency_key="transition-missing-harness-001",
        )
    assert service.get(job.job_id).status == "ready_for_harness"


def test_unknown_partial_execution_enters_rollback_candidate(tmp_path: Path) -> None:
    projected = job_service(tmp_path).record_execution(job_id(), result("unknown_or_partial"))
    assert projected.status == "rollback_candidate"
    assert projected.release_disposition == "block"


def test_legacy_approved_job_projects_shadow_until_validated(tmp_path: Path) -> None:
    write_legacy_job(tmp_path, status="approved_for_release_intent")
    projection = job_store(tmp_path).read_state().jobs[0]
    assert projection.effective_status == "shadow_only"
    assert projection.migration_required is True
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_learning_job_store.py tests/backend/test_learning_job_service.py tests/backend/test_learning_jobs_api.py -q -p no:cacheprovider
```

Expected: FAIL because transitions and v2 projection are absent.

- [ ] **Step 3: Implement v2 events and guarded transitions**

```python
ALLOWED_TRANSITIONS = {
    "shadow_only": {"ready_for_harness"},
    "ready_for_harness": {"harness_failed", "awaiting_human_review"},
    "awaiting_human_review": {"review_rejected", "governance_candidate"},
    "governance_candidate": {"governance_rejected", "release_candidate"},
    "release_candidate": {"execution_failed", "monitoring", "rollback_candidate"},
    "monitoring": {"closed", "rollback_candidate"},
    "rollback_candidate": {"rolled_back", "externally_recovered"},
}


def transition(
    self,
    job_id: str,
    target: str,
    refs: TransitionRefs,
    auth: AuthContext,
    expected_subject_version: str,
    idempotency_key: str,
) -> LearningJobProjection:
    current = self.get(job_id)
    require_project_scope(auth, current.project_id, required_role_for_transition(target))
    validate_transition(current, target, refs, auth)
    self._store.append_transition(
        build_transition_event(
            current,
            target,
            refs,
            auth,
            expected_subject_version=expected_subject_version,
            idempotency_key=idempotency_key,
        )
    )
    return self.get(job_id)
```

LearningJob creation snapshots carry the internal project ID and public project-scope VersionRef; create/migration/transition APIs reject cross-project AuthContext before reads. Job creation and migration-validation events also require idempotency keys and expected source versions. `LearningJobTransitionEvent` records canonical payload hash, expected/previous/result versions, actor principal/credential/roles, correlation ID, required refs, idempotency key, sequence, previous-event hash, and event hash in the same transaction as the new ledger head. Evidence-ingest and test-case targets must stop at their design boundaries; ClinicalSafetyPolicyVersion targets remain rejected.

Implement and table-test this target eligibility matrix before `governance_candidate -> release_candidate`:

| Target type | Maximum lifecycle boundary | Required proof |
|---|---|---|
| `prompt`, `rubric`, `route`, `template` | feature-flag release candidate | changed component is already deployed behind a registered ReleaseTarget; deployed component VersionRef, candidate patch ref, target registry ref, and effective flag ref all match |
| `evidence_ingest` | Stage C promotion candidate | approved ClinicalRagPromotionCandidate only; no Clinical RAG write or release execution |
| `test_case` | reviewed harness artifact | reviewed case catalog/ref only; no runtime mutation |
| `clinical_safety_policy` | rejected from LearningJob | independent Stage A activation/rollback lifecycle |
| `model`, `training_data` | rejected/out of scope | model training is outside this closeout |

Missing, undeployed, unregistered, stale, disabled-without-approved-intent, or VersionRef-mismatched prompt/rubric/route/template targets cannot enter `release_candidate` and append no transition.

- [ ] **Step 4: Run tests and confirm GREEN**

Run Step 2 command. Expected: PASS, including audit-chain tamper, idempotency mismatch, terminal retry/supersession, failed rollback attempt, and external recovery evidence cases.

- [ ] **Step 5: Commit Task 7**

```powershell
git add src/contracts/learning_job_transition.py backend/api/services/learning_job_store.py src/services/learning_job_service.py backend/api/schemas/learning_jobs.py backend/api/routes/learning_jobs.py tests/backend/test_learning_job_store.py tests/backend/test_learning_job_service.py tests/backend/test_learning_jobs_api.py
git commit -m "feat(learning): add audited learning job lifecycle"
```

### Task 8: Read-Only LearningJob Admin View

**Files:**
- Modify: `frontend/src/app/api/types.ts`
- Modify: `frontend/src/app/api/client.ts`
- Modify: `frontend/src/app/api/client.test.ts`
- Create: `frontend/src/features/agent-admin/learning-job-panel.tsx`
- Create: `frontend/src/features/agent-admin/learning-job-panel.test.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-pages.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.tsx`
- Modify: `frontend/src/features/agent-admin/agent-admin-view.test.tsx`
- Modify: `frontend/src/styles/globals.css`

**Interfaces:**
- Consumes: LearningJob read API only.
- Produces: candidate hash, transition history, evidence refs, supersession, and rollback state display with no apply/train/ingest/release control.

- [ ] **Step 1: Write the failing UI/client tests**

```tsx
it("renders lifecycle evidence without mutation shortcuts", async () => {
  render(<LearningJobPanel resource={successfulLearningJobResource()} />);
  expect(await screen.findByText("awaiting_human_review")).toBeInTheDocument();
  expect(screen.getByText("harness_20260710_001")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /apply|train|ingest|release/i })).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run frontend tests and confirm RED**

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/agent-admin/learning-job-panel.test.tsx src/features/agent-admin/agent-admin-view.test.tsx src/app/api/client.test.ts
```

Expected: FAIL because the panel and client types are missing.

- [ ] **Step 3: Implement the read-only panel**

```tsx
type LearningJobResource =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: LearningJobsResponse };

interface LearningJobPanelProps {
  resource: LearningJobResource;
}

export function LearningJobPanel({ resource }: LearningJobPanelProps) {
  if (resource.status === "loading") return <p role="status">正在加载 LearningJob</p>;
  if (resource.status === "error") return <p role="alert">{resource.message}</p>;
  return (
    <section aria-label="LearningJob 生命周期">
      {resource.data.jobs.map((job) => (
        <article key={job.job_id}>
          <strong>{job.effective_status}</strong>
          <span>{job.candidate_patch_sha256}</span>
          <ul aria-label={`${job.job_id} evidence refs`}>
            {job.evidence_refs.map((ref) => <li key={ref.sha256}>{ref.object_kind}: {ref.stable_id}</li>)}
          </ul>
        </article>
      ))}
    </section>
  );
}
```

- [ ] **Step 4: Run tests and build**

Run Step 2, then:

```powershell
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: tests PASS and build exits 0.

- [ ] **Step 5: Commit Task 8**

```powershell
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/app/api/client.test.ts frontend/src/features/agent-admin/learning-job-panel.tsx frontend/src/features/agent-admin/learning-job-panel.test.tsx frontend/src/features/agent-admin/agent-admin-pages.tsx frontend/src/features/agent-admin/agent-admin-view.tsx frontend/src/features/agent-admin/agent-admin-view.test.tsx frontend/src/styles/globals.css
git commit -m "feat(admin): show learning job lifecycle evidence"
```

### Task 9: Stage D Gate, Non-Mutation, And Post-Merge Evidence

**Files:**
- Create: `src/contracts/stage_d_evidence.py`
- Create: `src/services/stage_d_evidence.py`
- Create: `tests/fixtures/stage_d_gate_case_catalog_v1.json`
- Create: `tests/backend/test_stage_d_evidence_contract.py`
- Create: `tests/backend/test_stage_d_research_learning_gate.py`
- Modify: `src/services/closeout_stage_runner.py`
- Modify: `scripts/run_closeout_stage_gate.py`
- Modify: `tests/backend/test_closeout_stage_runner.py`
- Modify: `tests/backend/test_cohort_feasibility_non_mutation.py`
- Modify: `tests/backend/test_learning_job_non_mutation.py`
- Modify: `config/closeout_stage_suites.yaml`
- Create during branch advisory only: `output/closeout-advisory/stage_inputs/stage_d_evidence.<branch-head-sha12>.json`
- Create during post-merge run: `reports/closeout/stage_inputs/stage_d_evidence.<merge-sha12>.json`
- Create after the merged-commit run: `reports/closeout/stages/stage_d.<merge-sha12>.json`
- Create after report validation: `reports/closeout/attestations/stage_d_approval.<report-hash12>.json`

**Interfaces:**
- Consumes: final Stage D services, Stage C gate report/attestation, latest approved requirement manifest/attestation, `config/closeout_stage_suites.yaml`, and Stage A's `StageGateRunRequest`/`run_registered_stage_gate()` through the sole CLI `scripts/run_closeout_stage_gate.py`.
- Produces: a resolvable `StageDEvidenceSummary` from fixed synthetic acceptance cases and a StageGateReport payload bound to the actual merged Stage D commit; approval remains a separate `StageGateApprovalAttestation`.

- [ ] **Step 1: Write the failing Stage D gate test**

```python
def test_stage_d_gate_blocks_patient_rows_and_unreviewed_transitions(tmp_path: Path) -> None:
    result = run_registered_stage_gate(stage_d_test_request(repo_root(), output_root=tmp_path))
    assert result.report.compliance_status == "pass"
    summary_ref = only_artifact_ref(result.report, object_kind="stage_d_evidence_summary")
    summary = load_stage_d_evidence_summary(summary_ref)
    assert summary.patient_rows_exported == 0
    assert summary.criteria_effect_cases_passed == summary.criteria_effect_cases_total
    assert summary.learning_jobs_applied == 0
    assert summary.required_review_triggers == {
        "research_ethics_review", "pi_review", "data_governance_review", "publication_review"
    }
    assert {case.review_type for case in summary.review_trigger_case_refs} == summary.required_review_triggers
    for case in summary.review_trigger_case_refs:
        request = resolve_review_request(case.request_ref)
        assert request.review_type == case.review_type
        assert request.subject_ref == case.trigger_subject_ref
        if case.review_type == "research_ethics_review":
            assert is_sanitized_research_projection_manifest_ref(case.trigger_subject_ref)
        assert resolve_enforcement(case.blocking_result_ref).status == "blocked"
        assert resolve_enforcement(case.permitted_result_ref).approval_event_ref == case.approval_event_ref
    assert required_case_kinds(summary.artifact_case_refs) == {
        "review_request", "review_decision", "review_revocation", "review_expiry",
        "dependency_revocation", "pending_successor", "dataset_version", "analysis_run",
        "hypothesis_draft", "protocol_draft", "publication_intent",
    }
    assert all(resolve_and_validate_case_ref(ref) for ref in summary.artifact_case_refs)
```

`StageDEvidenceSummary` is an immutable public artifact with schema/source commit/toolchain/sanitizer/case-catalog refs, named aggregate counters only, per-criterion effect case refs, non-mutation snapshot refs, typed `artifact_case_refs`, exact `review_trigger_case_refs`, review-trigger set, and `learning_jobs_applied=0`. Each trigger case binds one of the four required review types to the triggering subject, persisted ReviewRequest, pre-approval blocking result, exact approval event, and post-approval permitted result. The synthetic gate's ethics case first publishes a sanitized, patient-row-free `ResearchProjectionManifest VersionRef` and uses that public ref as both `trigger_subject_ref` and the persisted request subject. Operational patient-projection reviews may retain a `ClinicalVersionRef` only inside the restricted governance store and serialize only `ClinicalVersionProjection`; neither the internal ref nor a patient-linked derivative may enter this public summary. Each artifact case binds case ID, artifact kind, resolvable sanitized VersionRef, source commit, and expected terminal/effective state. It contains no patient/query/source-event values. The shared runner exposes it only through the standard `artifact_refs` and requirement rows.

- [ ] **Step 2: Run the gate test and confirm RED**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_stage_d_research_learning_gate.py -q -p no:cacheprovider
```

Expected: FAIL because Stage D is not registered in the shared suite manifest and its required evidence collectors are absent.

- [ ] **Step 3: Register Stage D in the shared runner**

```python
@dataclass(frozen=True)
class StageDArtifactCaseRef:
    case_id: str
    artifact_kind: str
    ref: VersionRef
    expected_state: str


@dataclass(frozen=True)
class StageDReviewTriggerCaseRef:
    case_id: str
    review_type: Literal[
        "research_ethics_review",
        "pi_review",
        "data_governance_review",
        "publication_review",
    ]
    trigger_subject_ref: VersionRef
    request_ref: VersionRef
    blocking_result_ref: VersionRef
    approval_event_ref: VersionRef
    permitted_result_ref: VersionRef


@dataclass(frozen=True)
class StageDEvidenceSummary:
    summary_id: str
    schema_version: str
    source_commit: str
    toolchain_ref: VersionRef
    sanitizer_ref: VersionRef
    case_catalog_ref: VersionRef
    criteria_effect_case_refs: tuple[VersionRef, ...]
    non_mutation_snapshot_refs: tuple[VersionRef, ...]
    artifact_case_refs: tuple[StageDArtifactCaseRef, ...]
    review_trigger_case_refs: tuple[StageDReviewTriggerCaseRef, ...]
    required_review_triggers: frozenset[str]
    patient_rows_exported: int
    criteria_effect_cases_passed: int
    criteria_effect_cases_total: int
    learning_jobs_applied: int
    summary_sha256: str


def collect_stage_d_evidence(request: StageDCollectionRequest) -> VersionRef: ...
```

Implement `collect_stage_d_evidence()` in `src/services/stage_d_evidence.py`. It runs the fixed `stage_d_gate_case_catalog_v1.json` against real temporary governance/query/learning stores. The catalog requires exactly one independently executed trigger case for `research_ethics_review`, `pi_review`, `data_governance_review`, and `publication_review`; every case must persist a request whose subject matches its trigger, prove a blocked attempt before approval, record the exact authorized decision event, and prove only the corresponding allowed operation after approval. The ethics acceptance case must construct its sanitized public ResearchProjectionManifest before the attempted registry read and bind that manifest VersionRef throughout the case; the collector rejects a ClinicalVersionRef, ClinicalVersionProjection, patient-linked hash, or synthetic label in the public trigger slot. It also covers revocation/expiry/upstream-revocation/successor and all five research artifact kinds, validates every resulting sanitized ref through its production adapter, captures pre/post non-mutation snapshots, and publishes one create-once aggregate summary. Static trigger-name strings, one generic request reused for multiple types, missing/duplicate/fabricated/unresolvable/wrong-kind/wrong-state/stale-source/catalog-unaccounted refs, or request/subject/enforcement mismatch block. The collector never copies patient rows, query HMACs, actor values, or operational source events. Publication uses the Stage A shared resumable phase: an exact pre-existing summary is validated and reused after a crash, a mismatch blocks, and the missing StageGateReport is then published. Failure-injection tests cover interruption after summary publication and before/after runner context updates.

Add the collector ID to an explicit Python allowlist in `scripts/run_closeout_stage_gate.py`/`closeout_stage_runner.py`; the YAML may select that ID but cannot import or execute an arbitrary dotted path. Register the Stage D owned requirement IDs, exact fixed-cwd/fixed-argv commands, required artifact, immediate predecessor `inherits: [C]`, zero-skip policy, non-mutation assertions, and the allowed `shadow_only` scope ceiling in `config/closeout_stage_suites.yaml`. Recursive deterministic expansion must execute and record the complete A+B+C+D suite exactly once; do not add another CLI or report engine.

- [ ] **Step 4: Run full Stage D verification**

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_research_asset_contract.py tests/backend/test_cohort_feasibility_service.py tests/backend/test_research_query_ledger.py tests/backend/test_research_governance_contract.py tests/backend/test_research_governance_store.py tests/backend/test_research_governance_service.py tests/backend/test_research_approval_adapters.py tests/backend/test_research_review_api.py tests/backend/test_research_metadata_catalog.py tests/backend/test_research_api.py tests/backend/test_learning_signal_store.py tests/backend/test_learning_job_contract.py tests/backend/test_learning_job_store.py tests/backend/test_learning_job_service.py tests/backend/test_learning_jobs_api.py tests/backend/test_cohort_feasibility_non_mutation.py tests/backend/test_learning_job_non_mutation.py tests/backend/test_stage_d_evidence_contract.py tests/backend/test_stage_d_research_learning_gate.py tests/backend/test_closeout_stage_runner.py -q -p no:cacheprovider
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/features/research/research-evidence-view.test.tsx src/features/research/cohort-feasibility-panel.test.tsx src/features/research/research-workspace.test.tsx src/features/agent-admin/learning-job-panel.test.tsx src/features/agent-admin/agent-admin-view.test.tsx src/pages/workspace-page.test.tsx src/features/workspace/workspace-surface-switcher.test.tsx src/app/api/client.test.ts
cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
git diff --check
```

Expected: all commands exit 0; no required skip; `git diff --check` returns no output.

- [ ] **Step 5: Commit the Stage D gate implementation, then run branch advisory against that commit**

Run with the actual branch SHAs:

```powershell
git add src/contracts/stage_d_evidence.py src/services/stage_d_evidence.py tests/fixtures/stage_d_gate_case_catalog_v1.json tests/backend/test_stage_d_evidence_contract.py tests/backend/test_stage_d_research_learning_gate.py tests/backend/test_cohort_feasibility_non_mutation.py tests/backend/test_learning_job_non_mutation.py src/services/closeout_stage_runner.py scripts/run_closeout_stage_gate.py tests/backend/test_closeout_stage_runner.py config/closeout_stage_suites.yaml
git commit -m "test(closeout): register stage D research learning gate"
$branchHead = (git rev-parse HEAD).Trim()
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage C --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage C merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage C merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $branchHead
if ($LASTEXITCODE -ne 0) { throw "Stage D branch does not descend from approved Stage C" }
$manifestAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind closeout_manifest --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_stage_gate.py --stage D --mode branch-advisory --base-sha $stageBase --head-sha $branchHead --tested-content-sha $branchHead --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-d-research-learning.md --plan-subject reports/closeout/plan_subjects/stage_d_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_d_plan_approval_20260710_001.json --manifest-attestation $manifestAttestationPath --suite config/closeout_stage_suites.yaml --output-root output/closeout-advisory
```

Expected: every owned required row has `compliance_status=pass`; research/learning scope may remain `shadow_only` only where the approved manifest permits it.

- [ ] **Step 6: Merge, rerun at the actual merge commit, and approve evidence**

Merge through the protected workflow. From a clean checkout at the actual merged commit, rerun Step 4, then run:

```powershell
$actualMergeSha = $env:LANGG_STAGE_D_MERGE_SHA
$checkoutSha = (git rev-parse HEAD).Trim()
if (($actualMergeSha -notmatch '^[0-9a-f]{40}$') -or ($checkoutSha -ne $actualMergeSha)) { throw "checkout is not the recorded protected Stage D merge" }
$stageBase = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind stage_gate --stage C --field merged_sha).Trim()
if ($stageBase -notmatch '^[0-9a-f]{40}$') { throw "invalid approved Stage C merge SHA" }
git cat-file -e "${stageBase}^{commit}"
if ($LASTEXITCODE -ne 0) { throw "approved Stage C merge commit is unavailable" }
git merge-base --is-ancestor $stageBase $actualMergeSha
if ($LASTEXITCODE -ne 0) { throw "Stage D merge does not descend from approved Stage C" }
$mergeSha12 = $actualMergeSha.Substring(0, 12)
$manifestAttestationPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-approved --subject-kind closeout_manifest --field attestation_path).Trim()
D:\anaconda3\envs\LangG\python.exe scripts/run_closeout_stage_gate.py --stage D --mode post-merge --base-sha $stageBase --head-sha $actualMergeSha --tested-content-sha $actualMergeSha --merged-sha $actualMergeSha --plan docs/superpowers/plans/2026-07-10-crc-agent-closeout-stage-d-research-learning.md --plan-subject reports/closeout/plan_subjects/stage_d_plan_20260710_001.json --plan-attestation reports/closeout/attestations/stage_d_plan_approval_20260710_001.json --manifest-attestation $manifestAttestationPath --suite config/closeout_stage_suites.yaml --publish
```

Expected: the inherited runner publishes one immutable StageDEvidenceSummary plus StageGateReport, exact manifest counts/source hashes match, and the report does not embed its later approval.

- [ ] Collect the three Stage D approvals against the exact report hash/version, include the referenced StageDEvidenceSummary in the exact path list, and commit only the summary/report/attestation/event chain:

```powershell
$stageReportPath = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-published --subject-kind stage_gate --stage D --field subject_path).Trim()
$reportMergedSha = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py resolve-latest-published --subject-kind stage_gate --stage D --field merged_sha).Trim()
$reportMergeSha12 = $reportMergedSha.Substring(0, 12)
$stageDEvidencePath = "reports/closeout/stage_inputs/stage_d_evidence.$reportMergeSha12.json"
$reportHash = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field sha256).Trim()
$reportVersion = (D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py inspect-subject --subject-kind stage_gate --subject-path $stageReportPath --field subject_version).Trim()
$reportHash12 = $reportHash.Substring(0, 12)
$attestationPath = "reports/closeout/attestations/stage_d_approval.$reportHash12.json"
$gateEvidencePaths = "output/closeout/stage-d-gate-evidence-paths.txt"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_PI_REVIEWER_TOKEN --idempotency-key "$reportHash-stage-d-pi-reviewer-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_DATA_GOVERNANCE_REVIEWER_TOKEN --idempotency-key "$reportHash-stage-d-data-governance-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py record-approval --subject-kind stage_gate --subject-path $stageReportPath --expected-sha256 $reportHash --expected-version $reportVersion --credential-env LANGG_RELEASE_MANAGER_TOKEN --idempotency-key "$reportHash-stage-d-release-manager-v1"
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py derive-attestation --subject-kind stage_gate --subject-path $stageReportPath --output $attestationPath --path-list-output $gateEvidencePaths --include-subject --include-evidence $stageDEvidencePath
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-attestation --subject-kind stage_gate --subject-path $stageReportPath --attestation-path $attestationPath
git add --pathspec-from-file=$gateEvidencePaths
git diff --cached --check
D:\anaconda3\envs\LangG\python.exe scripts/record_closeout_attestation.py verify-staged-evidence --path-list $gateEvidencePaths --subject-path $stageReportPath --attestation-path $attestationPath
git commit -m "evidence(stage-d): record post-merge research learning gate"
```

## Plan Self-Review Checklist

- [ ] Criteria validation and filtering: Tasks 1 and 3.
- [ ] Complete immutable projection scan: Task 1.
- [ ] Ethics/PI/data-governance/publication triggers and revocation: Tasks 2 and 5.
- [ ] Disclosure control, query HMAC, restart-safe ledger: Task 3.
- [ ] Aggregate-only Research workspace: Task 4.
- [ ] DatasetVersion, AnalysisRun, HypothesisDraft, ProtocolDraft, PublicationIntent: Task 5.
- [ ] Opaque signal boundary: Task 6.
- [ ] LearningJob v2 state machine, migration, execution/rollback evidence: Task 7.
- [ ] Read-only lifecycle UI: Task 8.
- [ ] Non-mutation, inherited runner reuse, post-merge report, and separate approval: Task 9.
