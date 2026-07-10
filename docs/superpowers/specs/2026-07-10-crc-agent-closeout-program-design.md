# CRC Agent Closeout Program Design

> Date: 2026-07-10
>
> Baseline: `main@7bc83989788d6b40025c32e6f74d7e7c08612ea6`
>
> Source plan: `langg_crc_agent_stepwise_modification_plan_2026-06-29.md`
>
> Goal: Close the remaining P0, P1, P1.5, and P2 gaps so the patient and doctor default paths can be released with versioned, replayable, auditable, and rollback-capable evidence.

## 1. Background

The original CRC agent stepwise plan established a safety-first sequence:

1. P0 CRC safety and persistence;
2. P1 doctor review and field-level feedback;
3. P1.5 claim-level literature evidence and release observability;
4. P2 cohort feasibility and shadow learning jobs.

The implementation now contains substantial code for all four phases. The P2 implementation branch was fast-forwarded into `main` at `7bc8398`. The merge session observed `288 passed, 3 skipped` in the focused backend verification and a successful frontend production build, but those results do not yet have committed run artifacts; the CloseoutBaselineRecord must rerun and capture them.

The remaining work is not another broad feature expansion. It is a closeout program that converts partially satisfied contracts into enforceable release gates. The closeout must repair runtime gaps, add missing promotion and governance objects, generate fresh release evidence, and reconcile the source plan with the later narrowed subsystem specifications.

## 2. Current Project Context

The closeout builds on these existing specifications:

- `docs/superpowers/specs/2026-06-29-p0-crc-safety-loop-design.md`
- `docs/superpowers/specs/2026-06-29-p1-clinical-review-loop-design.md`
- `docs/superpowers/specs/2026-06-30-evidenceclaim-literature-harness-design.md`
- `docs/superpowers/specs/2026-06-30-agent-admin-release-dashboard-design.md`
- `docs/superpowers/specs/2026-07-02-controlled-release-governance-design.md`
- `docs/superpowers/specs/2026-07-03-controlled-release-execution-design.md`
- `docs/superpowers/specs/2026-07-03-post-release-monitoring-design.md`
- `docs/superpowers/specs/2026-07-07-post-release-closure-design.md`
- `docs/superpowers/specs/2026-07-08-crc-cohort-feasibility-design.md`
- `docs/superpowers/specs/2026-07-08-learningjob-candidate-pipeline-design.md`

The implementation already provides:

- deterministic CRC safety-policy evaluation;
- CRC mutation fixtures and static HarnessRun/ReleaseSafetyReport artifacts;
- CRC assessment persistence and patient record projection;
- ClinicalAssertion projection for CRC triage records;
- a feature-flagged Doctor Review Cockpit and append-only DoctorActionTrace events;
- EvidenceClaim, EvidenceDelta, LiteratureHarnessRun, and shadow isolation checks;
- a read-only release dashboard plus release governance, execution, monitoring, and closure subsystems;
- aggregate cohort feasibility contracts and API;
- append-only shadow LearningJob contracts, store, service, and admin API.

The audit identified release-blocking gaps in runtime intended use, policy activation, mutation replay fidelity, persistence traceability, doctor feedback sanitization, evidence promotion, cohort criteria application, research governance persistence, and LearningJob lifecycle closure.

## 3. Design Decision

Use a sequential, gate-driven closeout program. Each stage starts from the latest merged `main`, receives its own implementation plan and branch, and must receive an approved non-blocking post-merge StageGateReport before the next stage begins.

```text
P0 Safety & Persistence
  -> P1 Clinical Review Safety
    -> P1.5 Evidence Promotion
      -> P2 Research & Learning
        -> Integrated Release Acceptance
```

This ordering preserves the dependency direction in the source plan:

- patient safety and persistence precede doctor review;
- doctor review provenance precedes evidence promotion and research reuse;
- evidence isolation precedes Clinical RAG promotion;
- research and learning remain shadow-only until release evidence and human review are linked;
- integrated release acceptance is the only stage allowed to issue a final default-path release decision.

### Rejected Approach: One Large Closeout Branch

A single branch would reduce handoffs but mix CRC runtime, doctor review, evidence promotion, research governance, Agent Admin, and release reporting. It would weaken rollback boundaries and make failures difficult to attribute to one gate.

### Rejected Approach: Parallel Closeout Tracks

Parallel tracks would reduce calendar time but contend on shared API types, `backend/app.py`, Agent Admin pages, report contracts, test whitelists, and release artifacts. The final integration risk would exceed the scheduling benefit.

## 4. Program Scope

The program includes five independently testable closeout stages.

### Stage A: P0 Safety & Persistence Gate Closure

Includes:

- runtime intended-use display and API metadata;
- explicit safety-policy activation rules;
- complete mutation coverage and real topic-switch replay;
- end-to-end safety metadata preservation;
- structured care-card provenance;
- deterministic HarnessRun and ReleaseSafetyReport decision propagation;
- an enforceable release check for safety-relevant changes;
- reproducible Python/frontend dependency locks or an equivalent immutable environment manifest with package versions, sources, and integrity hashes.

Excludes:

- doctor review behavior;
- literature evidence promotion;
- research and LearningJob behavior.

### Stage B: P1 Clinical Review Safety Closure

Includes:

- multi-source ClinicalAssertion projection;
- real doctor-draft provenance;
- complete doctor action controls;
- target-reference validation;
- PII, hidden-reasoning, credential, and prompt-secret sanitization;
- proof that doctor feedback does not mutate patient facts, safety policy, prompts, RAG, or training data.

Excludes:

- evidence-pool approval;
- Clinical RAG ingest;
- model training or automatic distillation.

### Stage C: P1.5 Evidence Promotion Closure

Includes:

- a Project Evidence Pool boundary;
- review decisions and sign-off records;
- IngestPreview with deterministic hash and source spans;
- conflict, duplicate, negative-evidence, and review-state visibility;
- a read-only EvidenceClaim review view that makes candidate, evidence-pool, and Clinical RAG boundaries visible;
- real-store isolation checks;
- dynamic release-report discovery and hard-fail case drill-down.

Excludes:

- automatic Clinical RAG mutation;
- automatic guideline promotion;
- patient or doctor default-path use of candidate claims.

### Stage D: P2 Research & Learning Closure

Includes:

- actual application of cohort criteria;
- persistent ethics, PI, data-governance, and publication review items;
- DatasetVersion and dataset hash metadata without patient-level export;
- metadata-only AnalysisRun with disclosure-controlled aggregate outputs;
- HypothesisDraft and ProtocolDraft contracts;
- LearningJob validation, human-review, release, monitoring, and rollback references;
- an explicit LearningJob state machine that remains non-applying by default.
- a Research workspace aggregate-feasibility view and a read-only Agent Admin LearningJob lifecycle view.

Excludes:

- automatic research conclusions;
- patient-level dataset download;
- automatic patch application, training, RAG ingest, or feature-flag release.

### Stage E: Integrated Release Evidence & Acceptance

Includes:

- full inherited regression;
- patient, doctor, evidence/admin, research, and LearningJob end-to-end acceptance;
- deterministic replay of release artifacts;
- a versioned CloseoutRequirementManifest derived from the source plan and approved subsystem specifications;
- CloseoutGateMatrix generation;
- final release decision and rollback target;
- reconciliation of the source plan and subsystem specifications.

Excludes:

- unrelated production infrastructure such as distributed locks, OIDC migration, complete FHIR infrastructure, or new model training.

## 5. Architecture

The closeout preserves existing subsystem boundaries and adds only the contracts needed to make promotion explicit.

```text
Patient input
  -> Intended-use boundary
  -> ClinicalSafetyPolicyVersion
  -> CRC Assessment
  -> Patient Record / Care Card
  -> ClinicalAssertion
  -> Doctor Review / DoctorActionTrace

PaperCandidate
  -> EvidenceClaim / EvidenceDelta
  -> Project Evidence Pool
  -> IngestPreview
  -> Reviewed Clinical RAG candidate

DoctorActionTrace + EvidenceDelta + Harness failure + Cohort gap
  -> LearningJob candidate
  -> Harness validation
  -> Human review
  -> Release governance
  -> Feature flag
  -> Monitoring / rollback
```

All promotion is one-way, explicit, versioned, and reviewable. No downstream subsystem may infer approval from object existence alone.

Three shared integrity contracts apply to every stage:

1. Public/cross-domain `VersionRef` contains object kind, stable ID, schema version, canonical path or store adapter, SHA-256, and source git commit. Patient-scoped content instead uses `ClinicalVersionRef`: kind, opaque ID, version, schema, and restricted store handle, with keyed integrity MAC and key version retained server-side. Authorized clinical APIs may expose the opaque ID/version plus a SHA-256 of a separately sanitized projection, but never the integrity MAC or a public hash of identifiers. A bare version label is never sufficient release or rollback evidence.
2. Every append-only lifecycle uses the existing audit-chain pattern or a compatible `AuditEvent` with event ID, subject ID/version, monotonically increasing sequence, previous-event hash, canonical event hash, schema version, server-derived actor, idempotency key, and timestamp. Appends and ledger-head updates are atomic. A chain break, duplicate sequence, unknown schema, unreadable ledger, or derived-state mismatch blocks the owning gate.
3. `AuthContext` contains server-derived principal ID, credential identity, authenticated roles, project scopes, and correlation ID. The existing bearer-token configuration receives a backward-compatible token-to-principal/role/scope mapping for closeout roles; request bodies may carry display labels but cannot grant authority. Approval quorum requires distinct principal and credential identities. A shared or legacy admin token cannot satisfy two roles and is limited to migration administration or read-only use. Token rotation or revocation does not rewrite historical approvals, which retain the immutable principal identity. This minimal authorization layer is in scope, while a broader OIDC migration remains out of scope.

Filesystem artifacts use fixed path-bounded roots: `reports/closeout/baselines/`, `reports/closeout/requirements/`, `reports/closeout/stages/`, `reports/closeout/decisions/`, `reports/closeout/attestations/`, `reports/release_governance/release_bundles/`, and an `events/` child under each subsystem's existing report root for lifecycle events. Artifact IDs match a Windows-safe lowercase alphanumeric/dot/underscore/hyphen grammar, contain no separator or reserved device name, and resolve beneath the configured root without symlink/reparse-point escape. Writers create a non-candidate temporary file in the same root, flush it, validate schema/hash/chain, then atomically publish it; readers ignore temporary names and never repair artifacts in place.

## 6. Stage A Design: P0 Safety & Persistence

### 6.1 Intended Use

The patient CRC page and saved assessment API metadata must expose the `patient_crc_triage` profile boundary. The visible copy must state that the workflow is assistive and does not provide final diagnosis, treatment decisions, or screening conclusions.

`config/intended_use_profiles.yaml` remains the profile registry, and a machine-readable `config/intended_use_disclaimers.yaml` becomes the canonical key-to-copy catalog with catalog version, locale, visible text, and content hash. The backend resolver validates profile, key, catalog version, and locale, returns the resolved text plus VersionRef in assessment metadata, and persists that reference. The frontend renders this API-resolved copy. Missing key, unsupported locale without a configured default, or hash mismatch blocks the CRC workflow rather than falling back to unrelated hard-coded text.

### 6.2 Safety Policy Activation

ClinicalSafetyPolicyVersion is immutable. Its effective lifecycle status is derived from append-only, server-authorized activation and retirement events:

- `draft`: test and shadow evaluation only;
- `active`: eligible for default-path evaluation;
- `retired`: readable for replay but not eligible for new assessments.

For each intended-use profile, exactly one version may be active. Activation atomically retires the previous active version and activates the approved replacement, with distinct clinical-safety and release-governance approvals. Zero or multiple active versions is an integrity failure and follows the fixed safe-stop behavior below.

Policy rollback is a new server-authorized PolicyRollbackActivationEvent that atomically reactivates a verified prior version and retires the faulty active version; it does not delete or rewrite either policy or its earlier lifecycle events.

Default-path and patient-visible decisions must use an `active` policy. An explicit safety-policy feature flag may evaluate a `draft` policy in shadow mode, but the draft result may not lower, replace, or silently become the patient-visible decision. A draft must pass harness, clinical review, and release-governance approval in a separate activation record before its status can become `active`; a feature flag cannot substitute for activation.

Every policy-load, schema, or rule-conflict failure enters a fixed safe-stop state, blocks automated closure, emits a non-specific safe patient message, and requires human review. When a deterministic pre-policy disposition has a validated VersionRef, it is preserved as the minimum urgency and cannot be lowered; without one, the safe-stop emits no specific clinical conclusion. Failure handling must not invent a more specific conclusion merely to fail closed.

The safe-stop contract sets `workflow_status=blocked_pending_human_review`, nullable `clinical_disposition` from the validated pre-policy result only, an allowlisted `policy_failure_reason_code`, a safe `patient_message_key`, `assessment_save_allowed=false`, and `automated_closure_allowed=false`. A content-free integrity/audit event may still record the failure class and correlation ID.

### 6.3 Mutation Replay

The mutation pack must cover:

- age escalation;
- family history;
- rectal bleeding;
- weight loss;
- bowel obstruction;
- missing colonoscopy/test information;
- topic switch and return to CRC state.

The harness must execute topic-switch and return messages through the real CRC flow. Metadata-only expectations may not be synthesized as `true`.

Harness comparison must evaluate every declared expected field, including missing-information requirements, minimum dispositions, prohibited closure states, hard-fail semantics, and assistant-state isolation.

### 6.4 Persistence Traceability

The frontend CRC assessment request type and normalization path must preserve:

- `assessment_id`;
- `session_id` or conversation identifier used by the persistence contract;
- `safety_policy_version`;
- `matched_rules`;
- `hard_fail_flags`;
- `patient_message_key`;
- all safety metadata returned by assessment.

The save response must expose `record_id`, `event_id`, and snapshot/version references, and the frontend must retain them for navigation and traceability.

Care cards become structured records with stable `card_id`, `session_id`, `derived_from_record_id`, `derived_from_event_id`, `derived_from_assessment_id`, and `safety_policy_version` fields. The backend remains the only source of care-card derivation.

Assessment event, patient record, and patient snapshot writes must remain atomic. Care cards remain a deterministic read-time projection of committed records rather than a second independently persisted truth. Failure injection must prove that no partial event, record, or snapshot remains after rollback and that a failed assessment cannot appear in the care-card projection.

### 6.5 Release Decision

Release decision precedence is fixed:

```text
block > shadow_only > feature_flag > pass
```

ReleaseSafetyReport must inherit the most restrictive HarnessRun decision. A non-hard-fail case failure cannot become `pass` or `feature_flag` merely because `hard_fail_count` is zero. The legacy ambiguous value `feature_flag_or_pass` maps fail-closed to `feature_flag` during migration; only new explicit evidence can produce `pass`.

### 6.6 Safety-Relevant Change Gate

A versioned change-classification manifest defines safety-relevant paths for policy, prompts, models, RAG/evidence, tools, CRC runtime, persistence, and release contracts. Runtime prompt, model, RAG/evidence-index, tool, clinical-policy, and release-contract categories are permanently safety-relevant; the manifest may expand but never downgrade them. A repository check compares an explicit merge-base SHA and head SHA with that manifest and records both SHAs and the manifest hash. Unknown, renamed, generated, or ambiguously classified paths are treated as safety-relevant until an approved manifest update classifies them.

On main and release branches, any safety-relevant change requires a fresh deterministic HarnessRun and ReleaseSafetyReport whose version chain includes the changed artifact hash. Missing, stale, or inconsistent evidence blocks the check. Stage A adds a local validator and a checked-in required CI workflow; branch protection or the repository's equivalent merge policy requires that named check on pull requests and merge-queue commits. Feature branches may run it in advisory mode, but merge acceptance may not bypass the protected blocking result.

Because the current repository lacks complete Python and frontend dependency locks, Stage A also commits reproducible lock artifacts or one equivalent immutable environment manifest that records every resolved package/version, source, integrity hash, Python/Node/npm version, and platform constraint. StageGateReports and ReleaseBundle reference this VersionRef; ambient tool-version logging alone is not deterministic replay evidence.

## 7. Stage B Design: P1 Clinical Review Safety

### 7.1 ClinicalAssertion Sources

ClinicalAssertion projection expands from CRC triage to these bounded sources:

- patient triage and structured self-report;
- patient-uploaded report summary;
- clinician-authored note;
- backend-derived care card;
- reviewed evidence reference;
- model-generated draft marked unverified.

Each assertion must carry source, evidence references, confidence, review status, patient binding, and version metadata. Old records remain readable when assertion references are absent.

### 7.2 Real Draft Provenance

Doctor Review must consume the same report draft that the doctor sees in the report workflow. Static placeholder sections cannot satisfy provenance acceptance.

A patient-scoped, immutable `DoctorDraftVersion` is the single source for both the report view and Review Cockpit. It contains draft ID/version, patient and session binding, graph-run and visible source-message references, structured sections, per-section provenance, schema/sanitizer versions, and restricted integrity MAC. The backend creates it after removing internal-reasoning fields and exposes the same ClinicalVersionRef through both APIs. A new generation or edit creates a superseding version; it never overwrites an earlier draft. DoctorActionTrace targets the exact draft and section version.

Within the restricted clinical store, draft integrity uses a keyed HMAC with key version over the full canonical patient-scoped content, including authorized identifiers; that MAC is not exposed cross-domain. Any evidence, learning, harness, or release reference uses a separately sanitized projection and public SHA-256. Low-entropy patient identifiers are never protected or exposed through a bare public hash.

Each material draft section must contain either:

- patient fact references;
- RAG/evidence references with source spans;
- or `model_generated_unverified`.

The cockpit must expose citation confidence, EvidenceClaim candidate status, care-card provenance, and missing-provenance warnings.

### 7.3 DoctorActionTrace Safety

All six actions remain supported:

- accept;
- edit;
- reject;
- escalate;
- request evidence;
- mark unsafe.

For each DoctorDraftVersion/section, the backend returns `allowed_actions` from one versioned action-state contract, and the frontend renders only that set. The write endpoint re-evaluates the same rules with the expected draft/section version. Unknown, superseded, stale, or disallowed targets return a conflict and append no trace.

The backend must resolve every target reference. Patient-scoped assertion, assessment, record, and care-card targets must belong to the same patient; draft and citation targets must belong to the same doctor-review context and the citation's referenced evidence object must exist. `edit` atomically creates a superseding DoctorDraftVersion plus its trace while preserving the prior draft; the other five actions append review traces without rewriting the draft or clinical facts.

Free text must pass a shared sanitizer before persistence. The sanitizer removes or rejects:

- all direct patient identifiers in free text; patient binding remains in authorized structured target references;
- hidden-reasoning tags and content;
- API keys, bearer tokens, and credentials;
- prompt-secret markers;
- unrelated sensitive payloads.

Sanitizer success sets `content_sanitized=true`. A separate post-sanitization check may set `free_text_deidentified=true` only when no direct or quasi-identifier remains. DoctorActionTrace itself remains `patient_linked=true` because its structured target references bind it to a patient; it must not claim object-level de-identification. Sanitization or free-text de-identification failure rejects the trace write. Any patient-scoped draft content that must retain an identifier stays in the patient-authorized draft store and is not copied into DoctorActionTrace. A separately exported de-identified artifact must remove all patient-linkable references and receive a new ID/hash.

Reviewer identity and role are derived from the authenticated server context, not trusted from request fields. A rejected write may create a content-free security audit event containing request ID, reviewer ID, target type, and rejection class, but neither the raw rejected payload nor extracted secret or PII values may be logged.

### 7.4 Non-Mutation Boundary

DoctorActionTrace remains an append-only review signal. Recording a trace must not mutate:

- the underlying patient fact;
- ClinicalSafetyPolicyVersion;
- prompt, rubric, route, or template files;
- RAG indexes;
- feature flags;
- model or training data.

## 8. Stage C Design: P1.5 Evidence Promotion

### 8.1 Project Evidence Pool

The evidence pool stores reviewed claims separately from external candidates and the Clinical RAG index. Promotion requires an append-only review decision containing reviewer, server-authorized role, decision, reason, timestamp, source claim hash, and target evidence-pool version. Candidate authors cannot provide their own final promotion approval. Clinical RAG promotion readiness requires distinct `evidence_reviewer` and `clinical_safety_reviewer` sign-offs, plus `release_manager` when required by the active governance contract. PI review is an additional requirement only when the source is governed by a research project; client-supplied role labels have no authority.

Negative, conflicting, harmful, retracted, and quality-warning claims remain visible even when excluded from promotion.

### 8.2 IngestPreview

IngestPreview is an immutable proposal snapshot and contains:

- preview ID and stable hash;
- creation time, fixed expiration time, and optional `supersedes_preview_id`;
- EvidenceClaim and EvidenceDelta references;
- target evidence-index version;
- proposed chunks and source spans;
- duplicate and conflict findings;
- review and sign-off references present at creation;
- validation LiteratureHarnessRun references bound to the literature claim pack, evidence-index VersionRef, judge-rubric VersionRef, and negative/conflict/retraction/isolation case catalog;
- sanitizer and schema versions.

Upstream claim-review references are content inputs to the preview. Preview-specific approvals and invalidations are separate append-only `IngestPreviewReviewEvent` and `IngestPreviewInvalidationEvent` records that reference the exact preview hash. A derived projection computes current readiness without rewriting the preview. Source claim, upstream review, target index, chunk, harness, schema, sanitizer, expiration, or approval-revocation changes invalidate the old preview and require a new version linked through `supersedes_preview_id`; recording a preview-specific approval does not invalidate the content it approves, and approvals never carry forward to a superseding preview.

Approval creates a promotion candidate only when the required evidence and clinical sign-offs are distinct and authorized, any conditionally required PI/release-manager sign-off exists, the referenced LiteratureHarnessRun completed with a passing compliance result and no blocking release disposition, the preview is unexpired, and all source and target hashes still match. A P0 CRC mutation HarnessRun cannot substitute for the literature run. A failed or missing literature harness can never yield an approved promotion candidate. The closeout does not add automatic RAG writes.

### 8.3 Real Isolation Checks

Isolation verification uses explicit, versioned adapters: the LiteratureCandidateStore adapter for validated artifacts under `reports/literature/`; the append-only ProjectEvidencePoolStore under `reports/evidence_pool/`; a read-only ClinicalRagManifest adapter backed by versioned manifests under `config/evidence_indexes/`; and patient/doctor default-path usage adapters that inspect the exact runtime VersionRefs exposed by their APIs. Empty fixture-provided ID lists are not sufficient evidence. A missing, unreadable, unhashed, or ambiguous adapter/version blocks isolation acceptance.

The isolation report must list inspected versions, discovered claim IDs, violations, and the decision applied to each violation.

### 8.4 Release Dashboard

Each release report declares schema version, release train ID, monotonically increasing sequence, creation time, parent report ID/hash, and optional superseded report ID. The dashboard enumerates every path-bounded candidate report, groups by release train, and selects the greatest sequence; duplicate sequences, parent-chain forks, timestamp/sequence disagreement, or unknown schemas block. It never skips a newer malformed report to silently fall back to an older valid one. If the newest report is invalid, the dashboard blocks release, displays that failure, and may separately label the last known valid report for diagnosis only. It exposes:

- version chain;
- HarnessRun and ReleaseSafetyReport status;
- hard-fail and non-hard-fail cases;
- case ID, expected, actual, and artifact references;
- ReleaseBundle rollback target;
- intended and effective feature-flag snapshots plus any drift from the active release intent;
- human sign-off readiness;
- evidence isolation and IngestPreview status.

Generators write outside the candidate namespace, validate schema/hash/manifest links, then atomically publish the completed report. Malformed or incomplete candidate reports remain immutable and visible as invalid; recovery publishes a higher sequence with `superseded_report_id` and preserves the failure audit rather than deleting or rewriting it.

### 8.5 Evidence Review View

The Research workspace exposes read-only EvidenceClaim cards with claim text, population, outcome/effect, effect size and uncertainty, evidence grade, study design, sample size, source quality flags including guideline/systematic-review/preprint/retraction state, bias, local-guideline conflict, CRC applicability, source span, conflict/negative-evidence status, review history, and current isolation zone. The UI must distinguish external candidate, Project Evidence Pool, ingest candidate, and Clinical RAG states; it may not provide a direct ingest control. Agent Admin continues to host release evidence and isolation status, not duplicate claim-review controls.

## 9. Stage D Design: P2 Research & Learning

### 9.1 Cohort Criteria

The cohort service applies supported criteria before computing estimated count:

- allowlisted CRC condition codes or triage-risk classes;
- integer `age_min` and `age_max` within the contract range;
- allowlisted required-feature names;
- allowlisted reviewed-status values;
- at most the configured number of structured filters using only `eq`, `in`, `gte`, and `lte` on allowlisted fields and typed values.

Unknown keys/operators, duplicate or contradictory bounds, excessive filters, and free-form expressions are invalid. The service evaluates against one immutable patient-record projection snapshot VersionRef and uses complete server-side pagination or an equivalent database aggregate; the existing 1000-record page limit cannot truncate the count. Snapshot drift, mixed versions, incomplete pagination, or an unavailable snapshot blocks before returning a result.

Unsupported or unmapped criteria produce a validation-only blocked response before patient-record reads. That response may list warnings and supported alternatives, but it contains no cohort count or feasibility result. Missing variables discovered only after applying otherwise supported criteria remain visible as feasibility warnings; no criterion may be silently ignored.

Results remain aggregate-only and include the applied criteria, record/projection version, estimated count, coverage, missing variables, bias warnings, and review references.

Aggregate output is disclosure-controlled. The service enforces a configured minimum cohort size, suppresses or buckets small cells, rejects unsupported high-cardinality filters, records a keyed HMAC of the normalized query rather than a dictionary-attackable plain hash, and applies per-project authorization, rate limits, and differencing-query detection. The restricted query ledger atomically checks and records actor, project, scope, purpose, data/projection version, and query HMAC across restarts. Ledger unavailability, HMAC failure, or concurrent quota conflict blocks before patient-record reads. A suppressed result cannot be reconstructed through repeated overlapping queries, and suppression occurs before serialization.

### 9.2 Persistent Review Queue

The persistent review queue separates an immutable `ReviewRequest` from append-only `ReviewDecisionEvent` and `ReviewTransitionEvent` records. A derived `ReviewQueueItem` projection exposes current state without rewriting prior requests or decisions. It supports four review types:

- `research_ethics_review`;
- `pi_review`;
- `data_governance_review`;
- `publication_review`.

Each request records subject artifact, required role, creation reason, creation time, and optional superseded request. Each event records actor, server-authorized role, decision or transition, reason, time, expected request version, and idempotency key. Allowed historical state is `pending -> approved | rejected | expired | superseded`; terminal records cannot be rewritten, and an approval includes fixed `valid_until`. Scope changes create a new request. Duplicate same-key/same-payload events return the original result, while stale versions or conflicting payloads are rejected. Review state must not be inferred from response construction alone.

Effective validity is derived separately. When `valid_until` passes, a bound policy is retired, or an ethics, IRB, data-governance, publication, authorization, or de-identification approval is revoked, an append-only expiry/revocation event preserves the historical terminal decision but sets `effective_status=blocked` and immediately creates a superseding `pending` request. Every bound cohort result, DatasetVersion, AnalysisRun, ProtocolDraft, and PublicationIntent becomes invalid for further use until the new request is approved.

The authenticated principal must hold the required server-side project role. Researcher, PI, data-governance, publication, clinical-evidence, and release-governance permissions are distinct; the final reviewer for a governed artifact cannot be the same principal that authored its candidate unless an explicit break-glass policy records justification and secondary approval.

Trigger behavior is mandatory:

- a project's first patient-record projection scan creates `research_ethics_review` and blocks the scan until it is approved for the named project/version, purpose, criteria/field minimization scope, exact projection snapshot, authorization-policy VersionRef, de-identification/disclosure-policy VersionRef, and expiration period; any material purpose, scope, policy, snapshot, or expiry change creates a superseding request and blocks before reads;
- creating a HypothesisDraft or ProtocolDraft creates `pi_review`, and the artifact remains draft-only while review is pending; approval requires an explicit `irb_determination` of `required`, `not_required`, or `uncertain`, rationale, and linked ethics request. `uncertain` remains blocked. `required` also requires a matching, unexpired institutional IRB approval VersionRef before patient reads, AnalysisRun, or publication. `not_required` must be decided by an authorized ethics/IRB principal distinct from the author, not self-declared by the PI or requester;
- creating DatasetVersion or AnalysisRun metadata, or requesting any export, creates `data_governance_review`; pending metadata remains a restricted draft with suppressed counts, and patient-level export remains prohibited during this closeout even after review;
- an explicit `PublicationIntent` for a paper, grant, or patent draft creates `publication_review` and blocks any externalization or export artifact while pending or rejected; approval requires source VersionRefs and citation check, a contribution statement, privacy/de-identification attestation, and institution-policy VersionRef/decision.

### 9.3 Dataset and Hypothesis Contracts

DatasetVersion records disclosure-controlled metadata only:

- dataset ID and version;
- opaque query ID and keyed criteria HMAC with key version;
- field manifest;
- suppressed, bucketed, or exact aggregate row count as permitted by the cohort disclosure policy;
- source projection versions;
- de-identification policy version;
- access-policy VersionRef containing authorized project/roles, purpose, expiry, and export policy;
- governance review references;
- sanitized manifest hash; it never hashes or serializes patient rows into an exportable artifact.

The field manifest describes approved field names and disclosure-controlled coverage only; it contains no rare values or row-level identifiers. Its hash covers only already-suppressed metadata. The closeout does not materialize or export patient-level rows.

AnalysisRun is metadata-only: run ID/version, approved DatasetVersion reference, protocol reference, aggregate method/version, sanitized aggregate-output references, governance state, and audit events. It contains no executable notebook payload, patient rows, or unsuppressed cell output in this closeout.

HypothesisDraft and ProtocolDraft record research questions, falsification conditions, evidence references, bias considerations, review requirements, and draft-only status. They do not produce clinical recommendations.

### 9.4 LearningJob Lifecycle

LearningJob uses an explicit state machine:

```text
shadow_only -> ready_for_harness
ready_for_harness -> harness_failed | awaiting_human_review
awaiting_human_review -> review_rejected | governance_candidate
governance_candidate -> governance_rejected | release_candidate
release_candidate -> execution_failed | monitoring | rollback_candidate
monitoring -> closed | rollback_candidate
rollback_candidate -> rolled_back | externally_recovered
```

The existing write-once job JSON remains the immutable creation snapshot. State changes are append-only LearningJobTransitionEvents in a separate versioned transition ledger, and the current-state projection validates the full hash chain before use.

`harness_failed`, `review_rejected`, `governance_rejected`, `execution_failed`, `closed`, `rolled_back`, and `externally_recovered` are terminal for that candidate version. Retrying a failed candidate creates a new version linked by `supersedes_job_id`; it does not rewrite the terminal job. `rollback_candidate` reaches `rolled_back` only after a successful rollback execution result and post-rollback verification. A failed rollback attempt appends an immutable RollbackAttempt failure event without leaving `rollback_candidate`, invokes the required preflighted kill switch, blocks the effective release, and permits another authorized attempt against the same release. An incident commander may instead record authorized ExternalRecoveryEvidence plus post-recovery verification to reach `externally_recovered`; that status does not authorize another release without a fresh gate.

Every transition requires the relevant immutable reference:

- candidate patch hash;
- HarnessRun and validation result;
- human-review decision;
- release-governance intent;
- execution result;
- monitoring status;
- rollback or closure record.

The required reference is validated before a transition is appended. A missing, stale, or mismatched reference rejects the attempted transition and leaves the current state unchanged; it never resets an in-progress job to `shadow_only`.

New jobs use `status_schema_version=2`. A read adapter preserves existing records without rewriting them and returns `original_status`, `legacy_read_only=true`, `migration_required=true`, and an effective v2 state no higher than `shadow_only`; legacy `rejected` may project to terminal `review_rejected`, but `approved_for_release_intent` and `archived` do not inherit v2 approval or closure meaning. Only an append-only MigrationValidationEvent that proves every v2 reference may create a v2 successor linked to the legacy job.

No state transition applies the patch directly. Existing release governance and execution services remain the only feature-flag write path.

Target eligibility is explicit. Prompt, rubric, route, and template candidates may reach feature-flag execution only when the changed code/config is already deployed behind a registered ReleaseTarget. Evidence-ingest candidates stop at the Stage C promotion-candidate boundary and cannot write Clinical RAG. Test-case candidates stop as reviewed harness artifacts. ClinicalSafetyPolicyVersion changes use the independent activation lifecycle and are never applied through LearningJob.

Release execution returns `not_applied`, `applied`, or `unknown_or_partial` plus the observed final target state. `not_applied` may enter `execution_failed`; `applied` may enter `monitoring`; `unknown_or_partial` immediately freezes expansion, invokes the target kill switch, enters `rollback_candidate`, and requires reconciliation before any retry.

LearningJob signal inputs are opaque, bounded summaries, not clinical payload copies. A restricted clinical-lineage store converts eligible DoctorActionTrace batches into an unresolvable `learning_signal_id` plus an attestation containing action/reason enums, threshold-satisfying aggregate counts, and sanitizer/policy versions. Only the opaque signal and attestation enter LearningJob; the event-to-signal mapping remains in the patient-authorized store under separate RBAC, audit, and retention policy. Doctor free text, source event IDs, patient/session/record identifiers, patient-linked hashes, and row-level clinical narratives are forbidden. Invalid signal content is rejected before job creation and may emit only a content-free security audit event.

### 9.5 Research and Learning Views

The Research workspace displays submitted criteria, applied criteria, aggregate count, coverage, missing variables, bias warnings, projection versions, and review-queue state. It never renders patient rows or row-level identifiers.

Agent Admin displays LearningJob candidate hashes, state-transition history, harness and review references, release/monitoring/rollback evidence, and supersession links. The view is read-only during this closeout and exposes no apply, train, ingest, or release shortcut.

## 10. Error Handling

All closeout stages fail closed.

| Failure | Required behavior |
|---|---|
| Safety policy missing, invalid, or conflicting | Always enter fixed safe-stop, block closure, emit a non-specific safe message, and require human review; preserve a validated deterministic pre-policy disposition only as the minimum urgency |
| Assessment persistence failure | Roll back the complete transaction; leave no partial event, record, or snapshot, and expose no derived care card for the failed assessment |
| DoctorActionTrace sanitization failure | Reject the write; do not claim de-identification |
| Missing or cross-patient provenance target | Reject validation and append no trace |
| Evidence conflict, missing review, or isolation violation | Missing review leaves the claim candidate with review pending; conflict records a visible blocking conflict decision; isolation violation records a blocking isolation decision and blocks the affected release scope; none creates an ingest candidate |
| IngestPreview hash/version/sign-off mismatch | Invalidate the preview and require regeneration |
| Invalid cohort criteria | Reject before patient-record reads |
| Unauthorized patient-level export | Return `blocked_by_governance` and create no export artifact |
| LearningJob transition missing required references | Reject the attempted transition, leave current state unchanged, and create no release intent |
| Missing, malformed, or inconsistent release report | Force `block` |

Expected domain validation uses existing API conventions: validation errors are explicit client errors, missing resources are not found, state conflicts are conflicts, and unexpected storage or integrity failures do not return partial success.

All mutating endpoints introduced or modified by Stages A-E, plus the in-scope assessment-save, DoctorActionTrace, evidence review/preview, research review/query-ledger, LearningJob transition, release governance/execution/rollback, and closeout-artifact write paths, require an idempotency key and a server-computed canonical payload hash. Unmodified unrelated application writes are outside this closeout. Repeating the same key and payload returns the original result; reusing a key with a different payload returns a conflict. Same-store mutations use one transaction. Cross-store side effects use a transactional outbox or an explicitly modeled saga with replay-safe compensations. Timeouts and restarts reconcile pending operations before accepting a conflicting retry. Failure-injection tests cover process interruption, duplicate delivery, timeout, rollback, outbox replay, and reconciliation.

## 11. Security and Privacy Boundaries

- No hidden chain-of-thought is stored, displayed, exported, or included in reports. The server omits internal-reasoning fields before serializing both patient and doctor responses; a separately generated concise clinical rationale is allowed only as visible, provenance-bearing output.
- No API key, bearer token, prompt secret, or credential is stored in review or release artifacts.
- Patient-scoped clinical stores may retain only the identifiers needed by the existing authorized care workflow. DoctorActionTrace free text removes or rejects all direct identifiers; patient binding uses structured target references. `content_sanitized`, `free_text_deidentified`, and `patient_linked` are separate assertions; a patient-linked trace never claims object-level de-identification.
- A shared write-boundary sanitizer covers doctor drafts, trace text, EvidenceClaim text and source metadata, review reasons, harness expected/actual values, monitoring/error details, and closeout artifacts before persistence or logging. Patient-authorized clinical content that cannot be deidentified stays only in its patient-scoped store and is never copied to evidence, learning, harness, monitoring, or release artifacts.
- Research APIs return aggregate data only.
- Aggregate research responses enforce small-cell suppression, bounded filters, query auditing, rate limits, and differencing-query protection.
- Patient-level rows are not written to LearningJob or evidence artifacts.
- Doctor, evidence, research, and release actions use server-derived identities and scoped authorization; client-supplied role strings never grant permission.
- Final evidence promotion and feature-flag release obey separation of duties through existing release governance.
- Admin write paths remain authenticated, authorized, idempotent, append-only, and path-bounded.
- Cross-domain stable hashes use documented UTF-8 canonical JSON over the sanitized semantic payload and SHA-256. Each hash records schema version, canonicalizer version, source commit, and toolchain; volatile timestamps, random ordering, secrets, direct identifiers, and hidden model metadata are excluded. Patient-scoped integrity uses non-exported keyed HMACs as defined above. Trusted CI attests public manifest/hash associations before release.
- Secret, PII, and hidden-reasoning scans are defense in depth. Scan reports expose only rule IDs, counts, and sanitized locations; they never echo matched values or low-entropy value hashes.

The existing regression contract `tests/e2e/acceptance/frontend-regression-contracts.spec.ts` currently requires doctor-visible internal reasoning. That assertion is intentionally superseded by this safety boundary and must be changed to prove that neither patient nor doctor responses render `.clinical-thinking-disclosure` or receive raw thinking payloads.

## 12. Rollback Strategy

- New runtime behavior starts behind a feature flag or in shadow mode.
- New fields are additive and backward compatible.
- Destructive schema migration is outside this closeout.
- Previous policy, evidence-index, prompt/rubric, feature-flag, and artifact versions remain replayable.
- A rollback target is an immutable, hashed ReleaseBundle manifest, not a free-form policy label. It identifies the code commit, dependency/environment-lock VersionRef, schema-compatibility range, policy, evidence-index, prompt/rubric, feature-flag snapshot, artifact manifest, and their hashes.
- The ReleaseBundle artifact manifest contains deployable/runtime components only. StageGateReports, StageEVerificationReport, CloseoutDecisionPayload, and approval attestations reference the bundle but are never included inside it, preserving one-way hash dependencies.
- ReleaseBundle components are VersionRefs that resolve to retrievable artifacts. The current single `doctor_review_cockpit_v0` execution contract is extended backward-compatibly to a registry of typed ReleaseTargets with per-target enable/disable state, authorization policy, health check, and idempotent kill switch; unknown or unsupported targets cannot enter a releasable bundle.
- Before release, rollback preflight proves the bundle is retrievable, internally consistent, schema-compatible, authorized, and replayable. Execution and rollback use idempotency keys and append immutable ReleaseExecution, RollbackAttempt, and RollbackEvidence records.
- A verified kill switch is mandatory for every mutable release target. Missing or failed kill-switch preflight keeps the scope `shadow_only` or `block`; it cannot reach `release_candidate` or `monitoring`.
- Stage E performs an isolated or shadow-environment rollback rehearsal, including duplicate execution and partial-failure reconciliation, then records hashed RollbackEvidence. A missing or failed rehearsal blocks the affected scope.
- Critical monitoring alerts immediately freeze rollout expansion and set the affected release scope to `block`. Actual rollback continues through release governance and execution. Failure keeps the incident open, invokes the verified mandatory kill switch, and escalates without treating the old release as recovered.
- Post-rollback verification reruns the affected harness, inherited regression, persistence reconciliation, and health checks before recording success.
- A failed stage reverts its own runtime/code/config changes and unpublished candidate artifacts, but never deletes or rewrites audit events, failed reports, attestations, or rollback evidence. It appends a failure/rollback record. If a shared dependency change or inherited regression invalidates evidence for an earlier accepted gate, that gate is revoked and reopened; prior acceptance is never preserved against contradictory evidence.

## 13. Testing Strategy

Each stage uses the same verification layers:

1. contract and unit tests;
2. API and service integration tests;
3. frontend component and API-client tests;
4. non-mutation boundary tests;
5. inherited regression from all earlier stages;
6. TypeScript and frontend production build;
7. deterministic stage harness and report validation.

Implementation follows test-driven development. Each behavior change begins with a focused failing test, followed by the minimum implementation and a passing focused test before broader regression.

### Stage A Gate

- intended-use boundary is visible in the patient path;
- only an active policy can affect the default or patient-visible path; a flagged draft remains shadow-only;
- the required mutation set executes through real runtime behavior;
- the versioned hard-fail catalog plus declared mutation/property coverage contains zero emergency false negatives within its recorded coverage boundary;
- assessment, event, record, snapshot, session, and derived care-card provenance forms a complete validated chain;
- the protected changed-path check treats policy/prompt/model/RAG/tool categories as always relevant and blocks stale, missing, or mismatched harness evidence on main and release merges;
- a clean environment can restore the locked dependency set and reproduce the focused safety replay;
- all failures produce `block` or `shadow_only` as required.

### Stage B Gate

- the existing doctor flow does not regress;
- the real draft carries provenance or explicit unverified status;
- all six doctor actions work end to end;
- PII, hidden reasoning, and credentials cannot enter traces;
- doctor feedback causes no prohibited mutation.

### Stage C Gate

- claim-level fields are complete;
- negative and conflicting evidence is preserved;
- unreviewed claims cannot enter Clinical RAG or default paths;
- IngestPreview is immutable, event-reviewed, versioned, and invalidated on mismatch;
- promotion candidates require distinct authorized sign-offs and a passing LiteratureHarnessRun covering claim, negative/conflict/retraction, and real-store isolation cases;
- the dashboard shows complete failure evidence, blocks on the newest malformed or incomplete report, and labels any older valid report as diagnostic only.
- the dashboard reports effective feature-flag state and blocks unexplained drift from the active release intent.

### Stage D Gate

- supported cohort criteria affect selection;
- no patient-level rows leave the registry boundary;
- each defined ethics, PI, data-governance, and publication trigger creates and enforces its persisted review request;
- review queue, disclosure-controlled DatasetVersion/AnalysisRun metadata, and hypothesis drafts are traceable;
- candidate patches remain non-applying;
- LearningJob transitions require harness, review, release, monitoring, and rollback evidence.

### Stage E Gate

- the latest approved manifest, all source hashes, and every Stage A-D post-merge GateReport validate as one unbroken chain;
- the complete backend, frontend, build, end-to-end, security/privacy, deterministic replay, and working-tree checks pass on the frozen release-content commit;
- the canonical acceptance runner contains only existing manifest-required test paths and records exact results;
- every scope has a retrievable ReleaseBundle, mandatory kill-switch preflight, and successful isolated rollback rehearsal;
- the CloseoutDecisionPayload and FinalApprovalAttestation contracts/validators pass; the actual distinct-principal attestation is created only after the immutable payload hash exists and is verified by release preflight rather than by a self-referential matrix row;
- release-governance and execution preflight reject Stage A-D reports, stale manifests, or missing Stage E authorization.

## 14. Closeout Requirement, Stage Gate, and Final Matrix Contracts

### 14.1 Closeout Baseline Record

At program start, a committed CloseoutBaselineRecord captures the base commit, branch topology, clean-checkout status, tool/runtime versions, baseline commands and results, known failures or work-in-progress, and explicitly excluded user-owned working-tree paths. Every known failure or exclusion maps to a requirement row and proves whether it affects build, runtime, tests, or release evidence; it cannot be normalized as harmless merely by listing it.

This record closes the evidentiary gap in the original Step 0. It does not rewrite the historical fact that the first implementation preceded its original design commit. The reconciliation report marks that sequencing-only requirement as superseded by the approved closeout baseline procedure.

### 14.2 CloseoutRequirementManifest

Before Stage A implementation begins, the program creates a canonical JSON manifest payload under `reports/closeout/requirements/`. Its top level records manifest ID, schema version, previous manifest ID/hash, creation time, and every source document as a VersionRef containing path, document SHA-256, and git commit. Approval events are external attestations that reference the finished manifest hash; they are not embedded in the hash input.

Each entry records:

- stable requirement ID and normalized requirement-text hash;
- source VersionRef and section anchor;
- owning stage and release scope;
- criticality, required test IDs, required artifact kinds, required review roles, replay evidence, skip policy, and whether rollback evidence is required;
- allowed compliance outcomes and fixed failure release policy;
- disposition: `required`, `superseded`, or `out_of_scope`;
- rationale and required approval roles for a non-required disposition.

Hash construction is acyclic: each `entry_hash` is SHA-256 over its canonical entry without a hash field; `manifest_hash` is SHA-256 over the canonical header without a hash field plus ordered entry hashes and source-document hashes. A separate ManifestApprovalAttestation records manifest hash, entry-specific disposition approvals, distinct approvers, and post-approval ledger head. Validators compare each row's evidence exact set against its manifest entry, not merely the presence of a row.

Safety, privacy, authorization, hard-fail, persistence consistency, sanitization, rollback, and required replay requirements are always `required`; the manifest cannot downgrade them. Other non-required dispositions need distinct server-authorized approvals from the requirement owner and release manager, plus clinical-safety approval when the requirement can affect clinical behavior. Authors cannot provide final approval.

The manifest is append-only after approval. A source-document, entry, or disposition change creates a new manifest version, invalidates affected StageGateReports, and requires a new ManifestApprovalAttestation. The latest approved manifest is selected by a valid attestation and append-only approval ledger, not by filename. Its source hashes and exact required-entry count are authoritative.

### 14.3 Per-Stage Gate Reports

Stages A through D each commit an immutable StageGateReport under `reports/closeout/stages/`. Stage E commits an immutable StageEVerificationReport with the same evidence fields but does not reference the final matrix, avoiding a hash cycle. A report binds:

- stage plan VersionRef;
- latest approved manifest ID, hash, required-entry count, and owned entry hashes;
- diff base SHA, branch head SHA, actual merged commit SHA, and changed-path manifest hash;
- exact commands, working directories, environment/toolchain versions, start/end times, exit codes, result counts, and sanitized output artifacts;
- implementation, test, review, HarnessRun, ReleaseSafetyReport, and other artifact VersionRefs and hashes;
- compliance result and per-scope release disposition;
- ReleaseBundle reference when the stage can affect a releasable scope;
- prerequisite review-event references and the ledger head that existed before that report's own approval.

A separate StageGateApprovalAttestation references the StageGateReport hash, records distinct authorized approvers, and records the post-approval ledger head. The report never embeds its own later approval. The report plus a valid attestation is the non-blocking gate artifact.

Branch verification is advisory until the stage is merged. The required gate run executes again at the actual merged commit. Only a post-merge report with a valid StageGateApprovalAttestation opens the next stage. A later shared-dependency change, source/manifest change, audit-chain failure, or inherited regression revokes and reopens every affected earlier gate.

### 14.4 CloseoutGateMatrix

After the StageEVerificationReport is committed and approved, Stage E creates a canonical CloseoutDecisionPayload under `reports/closeout/decisions/`. The payload contains the gate matrix and binds the latest approved manifest/attestation, source-document hashes, expected and actual required-entry counts, Stage A-D GateReport/attestation VersionRefs, StageEVerificationReport/attestation VersionRefs, final merged commit, release train ID, and ReleaseBundle ID/hash. It does not embed its later approvals or their ledger head.

A separate append-only FinalApprovalAttestation references the CloseoutDecisionPayload hash, records distinct authorized approvers, and records the post-approval audit-ledger head. The payload plus a valid attestation is the final matrix authorization. Stale manifests, count mismatch, duplicate/unknown/missing IDs, source drift, a non-latest manifest, or an invalid attestation force `block`.

Each required row contains requirement/entry hash, owning stage, release scope, `compliance_status`, `release_disposition`, decision reason and owner, evidence commit, implementation/test VersionRefs, verification-run results, artifact hashes, review-event references, skip dispositions, and the applicable ReleaseBundle reference.

`compliance_status` is only `pass` or `block`. `release_disposition` is one of:

```text
block > shadow_only > feature_flag > pass
```

The manifest fixes which release dispositions are legal for each requirement. Emergency hard-fail, policy activation, persistence consistency, privacy/security, and rollback requirements allow only `pass` or `block`; they cannot be waived to `feature_flag` or `shadow_only`. An illegal row value is itself blocking.

Aggregation occurs independently for `patient_default`, `doctor_default`, `clinical_rag`, `research_workspace`, and `learning_pipeline`, then follows a versioned ReleaseScopeDependencyManifest containing scope nodes, directed dependency edges, propagation rules, schema version, and hash. Cycles, unknown scopes, or manifest/hash mismatch block. A research or learning object that correctly remains shadow can have `compliance_status=pass` without downgrading an unrelated patient scope. Runtime object lifecycle states never substitute for requirement compliance.

The requirement manifest distinguishes technical compliance from post-payload authorization. It requires tests for the FinalApprovalAttestation schema, signer quorum, hash binding, ledger validation, and execution-preflight rejection behavior, but the actual attestation for a specific CloseoutDecisionPayload is not a matrix requirement row. It is created afterward and remains a mandatory external authorization condition.

For each release scope, the CloseoutDecisionPayload computes a technical disposition as the most restrictive of the applicable HarnessRun, ReleaseSafetyReport, matrix rows, and rollback/target preflight. A later governance intent cannot raise that ceiling. At execution time, authorization is the most restrictive of the technical disposition, current monitoring/preflight state, and active release-governance intent; absence of an approved intent means “not authorized” rather than mutating the immutable technical payload. Any identifier, hash, version, scope, or decision disagreement forces `block`. The legacy ambiguous value `feature_flag_or_pass` maps to `feature_flag` until explicit new evidence proves `pass`.

Required safety, privacy, security, authorization, deterministic replay, and release tests may not be skipped or waived; any skip blocks. A non-critical skip can be considered only with reason, owner, risk, scope, approval events, and expiry, and can yield at most `feature_flag`.

The existing `reports/harness/harness_20260629_001.json` has SHA-256 `56789d67fdc71d6b533830652253a67d4475dfbb3f9d3c80d1c3da50b66de9e1` and may anchor the baseline record. It does not authorize release. Its legacy rollback label is migration metadata only and cannot satisfy the ReleaseBundle requirement.

## 15. Integrated Acceptance

The final stage must run and record:

- the complete backend test suite;
- the complete frontend Vitest suite;
- TypeScript checking and production build;
- patient CRC end-to-end acceptance;
- doctor review and action-trace end-to-end acceptance;
- evidence promotion and Agent Admin acceptance;
- research and LearningJob acceptance;
- secret, PII, and hidden-reasoning scans;
- deterministic HarnessRun and report replay;
- `git diff --check` and working-tree hygiene checks.

Stage E repairs or replaces `scripts/run_e2e_full_acceptance.ps1` as the canonical runner. Its current references to four nonexistent backend tests and `tests/e2e/acceptance/workspace-core.spec.ts` cannot be treated as evidence. The repaired runner preflights every manifest-required path, rejects missing files, and records actual commands/results; list-only output is never acceptance evidence.

Stage E also adds a dedicated sensitive-artifact scanner with a versioned rule/scope manifest. Production code, runtime configuration, generated reports, and persisted artifacts are mandatory scan scope. Synthetic secret and `<think>` fixtures may be excluded only by exact path plus fixture-purpose hash; importing an excluded fixture into production scope blocks. The scanner reports sanitized rule IDs/counts/locations and is not a raw repository-wide text search.

Required safety/privacy/security/replay/release tests may not be skipped. Non-critical skips require the manifest-defined waiver evidence. A raw skipped count or an explained skip without the allowed waiver contract is not acceptable.

The release-content commit is frozen before acceptance. Generated StageEVerificationReport, decision payload, and attestation may be committed afterward only as evidence-only descendants whose diff is proven path-bounded and runtime-neutral. A report payload binds the tested release-content commit and its own content hash, never the unknown commit that will first contain it. After that first evidence commit exists, a separate EvidenceCommitAttestation may bind report hash to that commit SHA; the attestation does not claim the SHA of the commit containing itself. Any runtime/config change after the freeze requires a new Stage E run.

Existing release-governance preflight is extended backward-compatibly to require a validated latest CloseoutDecisionPayload and FinalApprovalAttestation for any patient/doctor default or feature-flag intent. Stage A-D reports are advisory/stage evidence only and cannot authorize default release. Execution revalidates manifest, payload, ReleaseBundle, target state, and attestation immediately before mutation.

## 16. Rollout Plan

1. Create and approve the CloseoutBaselineRecord and CloseoutRequirementManifest.
2. Write and approve the Stage A implementation plan.
3. Implement Stage A on an isolated branch or worktree, merge it, rerun at the merged commit, and approve its StageGateReport.
4. Repeat the plan, implementation, post-merge verification, report, and approval cycle for Stages B, C, and D.
5. Write and approve the Stage E implementation and acceptance plan after all four earlier StageGateReports contain no compliance `block`.
6. On the isolated Stage E branch, implement validators, canonical runner, scanner, matrix/attestation contracts, and release-governance preflight, and reconcile the source-plan checklist. A changed normalized requirement returns work to its owning stage.
7. Merge Stage E and freeze the release-content commit. In an evidence-only descendant, generate and approve the final manifest bound to the frozen source-document blobs/commit, then generate the candidate HarnessRun, ReleaseSafetyReport, and immutable ReleaseBundle against that exact commit and manifest before testing; any later change to one invalidates the run.
8. Revalidate the affected Stage A-D gates against the final manifest, execute integrated acceptance and the rollback rehearsal against that exact ReleaseBundle, then commit and approve the refreshed gate evidence and StageEVerificationReport as evidence-only descendants.
9. Assemble and validate the CloseoutDecisionPayload and per-scope decisions from those fixed hashes, then record the distinct-principal FinalApprovalAttestation. No bound source, runtime, manifest, harness, report, or bundle may change afterward.
10. Push or release patient and doctor default paths only after the final per-scope decision is `pass` or an explicitly approved `feature_flag` and execution preflight reconfirms the evidence.

A stage may advance only when its StageGateReport has `compliance_status=pass` for every owned required entry and its StageGateApprovalAttestation is valid. A scope-level `shadow_only` disposition can be merged for downstream integration, but that scope cannot satisfy final default-path release until a later valid gate yields `pass` or an approved `feature_flag`.

## 17. Acceptance Criteria

The closeout program is complete when:

1. the baseline record and requirement manifest account for every source-plan requirement;
2. all five stage plans have been executed and merged in order;
3. the P0, P1, P1.5, and P2 gates have no blocking requirement;
4. patient and doctor default paths satisfy their intended-use and safety boundaries;
5. unreviewed literature cannot enter Clinical RAG or default clinical paths;
6. research APIs remain aggregate-only and governance-gated;
7. LearningJob candidates cannot apply themselves and have complete validation and review references;
8. the final CloseoutDecisionPayload gate matrix has no prohibited skip, missing evidence, duplicate/unknown requirement, or invalid version chain;
9. the final patient/doctor release decision is `pass` or an explicitly approved `feature_flag`, with a concrete ReleaseBundle rollback target;
10. the final decision and rollback target are reproducible from committed artifacts;
11. the source plan and later subsystem specifications no longer contradict the implemented release boundary.

## 18. Implementation Boundaries

This program must not:

- introduce automatic diagnosis, treatment, or screening conclusions;
- train or fine-tune a model;
- automatically apply prompt, rubric, route, template, evidence, or RAG patches;
- export patient-level research datasets;
- replace existing release governance, execution, monitoring, or closure services;
- add unrelated authentication, distributed locking, SSE-resume, complete FHIR, or infrastructure rewrites;
- modify `CRC-client/`.

The authentication exclusion does not prohibit the minimal AuthContext, scoped role mapping, distinct-principal approval checks, or project authorization required by this design; it excludes a broader identity-platform replacement.

## 19. Spec Self-Review

- Scope is intentionally decomposed into five sequential implementation plans.
- Each stage has a distinct runtime boundary, failure policy, and acceptance gate.
- Promotion and release decisions use one consistent precedence order.
- Data flow never allows downstream object existence to imply approval.
- Existing governance and rollback subsystems are reused rather than replaced.
- No unresolved placeholder, deferred implementation marker, or ambiguous automatic-action permission remains in this specification.
