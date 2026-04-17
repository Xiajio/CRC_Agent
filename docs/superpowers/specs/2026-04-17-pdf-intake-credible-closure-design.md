# PDF Intake Credible-Closure Design

**Date:** 2026-04-17  
**Status:** Draft for review  
**Goal:** Upgrade the current patient-side PDF upload flow from a technically working registry-first pipeline into a credible end-to-end intake flow that protects the patient snapshot, keeps doctor-side consumption semantically correct, and preserves the historical case workbench boundary.

## 1. Context

The scene-driven workspace baseline is already in place:

- patient and doctor run in separate sessions
- both scenes can access the shared SQLite patient registry at `runtime/patient_registry.db`
- patient-side uploads are persisted into the registry instead of being injected as large JSON blobs into session state
- doctor-side binding can pull shared patient summary into doctor chat context without sharing patient chat history

This means the technical bridge exists. The remaining problem is not transport or storage. It is credibility.

Manual verification exposed two concrete blockers and one broader product gap:

- `GET /api/patient-registry/patients/{id}` can fail with `500` when a registry field such as `age` contains a dirty value that cannot satisfy the response schema
- after a doctor binds a registry patient, parts of the doctor workbench still try to load historical-case detail through `/api/database/cases/{id}`, which is the wrong data source
- the upload pipeline still treats arbitrary PDFs as if they were patient facts, which allows guideline or education material to flow into the patient registry path and generate misleading snapshot content

The design in this document addresses those gaps without changing the confirmed dual-session scene architecture.

## 2. Problem Statement

The current upload flow is technically successful but semantically unsafe.

Today the product can already do the following:

- accept a patient PDF upload
- parse it into a `MedicalVisualizationCard`
- write assets and extracted records into the shared registry path
- let the doctor bind the resulting patient and receive a summary in doctor chat

However, the product still lacks clear rules for:

- which uploaded documents are allowed to affect the patient snapshot
- which extracted values are trustworthy enough to become current patient facts
- how the doctor should distinguish trusted patient snapshot data from raw extraction records
- how the registry patient view and the historical case workbench remain separate inside one page

Without those rules, "upload success" can still produce bad patient data.

## 3. Design Principles

- Separate asset acceptance from snapshot mutation.
- Treat every extraction result as candidate evidence before it becomes a patient fact.
- Keep `patient_records` append-only and preserve evidence, even when evidence is low-confidence or conflicting.
- Keep `patients` conservative and stable; do not let placeholder or parser-failure values enter the snapshot.
- Make doctor-side consumption explicit: trusted snapshot first, record trail second, warnings visible.
- Preserve the existing split between `patient_registry` and `historical_case_base`; do not unify them through shared ids or shared detail routes.
- Fix credibility through data semantics and UI clarity before investing in heavier extraction or orchestration work.

## 4. Scope

### In scope

- document-type gating for uploaded PDFs
- stricter snapshot eligibility rules
- dirty-field normalization for patient-registry detail responses
- explicit doctor-side registry consumption model
- explicit separation between registry patient detail and historical case detail
- warning surfaces for low-confidence and conflict states
- updated acceptance criteria for the PDF intake loop

### Out of scope

- asynchronous extraction jobs or queue workers
- human review workflows
- timeline visualization beyond a basic records list
- full conflict-resolution UI
- replacing the current document converter model
- changing the dual-session scene architecture

## 5. Target Outcome

The target product behavior is:

1. A patient uploads a PDF.
2. The system always stores the asset.
3. The system classifies the document type before deciding whether it can affect the patient snapshot.
4. The system preserves structured extraction results as records.
5. Only trusted document types and trusted field values can update the patient snapshot.
6. The doctor can bind the patient, see trusted snapshot data, inspect the upload-derived records, and understand whether any data is low-confidence or conflicting.
7. The historical case workbench continues to function as a separate reference tool.

The target is not "more extraction." The target is "credible intake."

## 6. Current-State Gaps

### 6.1 Registry Detail Fragility

The registry detail endpoint currently assumes that stored values already match the response schema. When a stored field is dirty, the endpoint can fail entirely.

This is unacceptable for doctor-side consumption. A single bad field must degrade to `null`, not break the whole patient detail view.

### 6.2 Registry/Historical Boundary Leakage

The doctor workspace currently still contains code paths that treat a bound registry `patient_id` as if it were a historical-case id. This leaks across the boundary between:

- current patient data in the shared registry
- read-only historical case data backed by the existing Excel path

This must be explicitly corrected in both BFF and frontend behavior.

### 6.3 Snapshot Pollution Risk

The current registry-first upload path can still over-trust extraction results. A guideline PDF or poorly parsed document can produce placeholder fields such as:

- `parse_failed_text`
- `not_provided`
- `pending_assessment`

Those values are acceptable as extraction evidence inside records. They are not acceptable as current patient facts inside the snapshot.

## 7. Credible Intake Flow

The upload pipeline should be interpreted as five stages:

`asset accepted -> document classified -> structured extraction -> ingest decision -> lightweight context emit`

### 7.1 Asset Accepted

Every uploaded PDF should still produce a persisted asset entry. Upload acceptance only means:

- the file was received
- the file was persisted
- the file is tied to a draft or active patient id

It does not mean the file is suitable for patient snapshot mutation.

### 7.2 Document Classified

Before snapshot logic runs, the system must classify the file into at least one of:

- `patient_report`
- `patient_summary`
- `pathology_report`
- `imaging_report`
- `guideline_or_education`
- `unknown`
- `parse_failed`

This classification becomes the first gate for downstream registry behavior.

### 7.3 Structured Extraction

The existing `DocumentConverter` output remains useful and should continue to produce a normalized `MedicalVisualizationCard`. The extracted card is not a patient snapshot. It is candidate structured evidence.

### 7.4 Ingest Decision

Each upload should produce one of three ingest outcomes:

- `asset_only`
- `record_only`
- `record_and_snapshot`

Recommended mapping:

- `guideline_or_education` -> `asset_only` or `record_only`
- `unknown` -> `record_only`
- `parse_failed` -> `asset_only`
- `patient_report` / `patient_summary` / `pathology_report` / `imaging_report` -> may allow `record_and_snapshot`

### 7.5 Lightweight Context Emit

The patient session should only receive a lightweight context message indicating:

- the filename
- the classified document type
- the ingest outcome
- the `record_id`
- a short summary of accepted findings if any

The chat context should not include the full extracted JSON payload.

## 8. Snapshot Eligibility Rules

The patient snapshot must only accept a whitelist of fields:

- `chief_complaint`
- `age`
- `gender`
- `tumor_location`
- `mmr_status`
- `clinical_stage`
- `t_stage`
- `n_stage`
- `m_stage`

The following must never enter the snapshot:

- `treatment_draft`
- long free-text explanations
- model-generated treatment suggestions
- parser failure text
- placeholder text such as `not_provided`, `pending_assessment`, `Unknown`, empty string, or `parse_failed_text`

This means the extraction record may contain more information than the patient snapshot, which is expected and correct.

## 9. Snapshot Update Rules

`patient_records` and `patients` serve different purposes:

- `patient_records` is an append-only record of upload-derived evidence
- `patients` is the current trusted patient snapshot

### 9.1 Fill Before Overwrite

If a snapshot field is currently empty and a trustworthy new value appears, the field may be filled.

If a snapshot field already contains a good value, the new value must not automatically replace it unless the new evidence comes from an equal-or-higher authority source and is itself valid.

### 9.2 Bad Values Never Replace Good Values

Invalid or placeholder values may remain visible in records, but they must never overwrite an existing trusted snapshot value.

### 9.3 Source Priority

Recommended source ranking:

1. `doctor_curated`
2. `pathology_report`
3. `imaging_report`
4. `patient_summary`
5. `generic_patient_report`
6. `unknown`
7. `guideline_or_education`

Lower-priority evidence should not override higher-priority snapshot values.

### 9.4 Conflict Handling

If two non-empty values conflict:

- keep the current snapshot value
- store the new value inside `patient_records`
- mark the field or record as conflicted
- expose the conflict to the doctor UI as a visible warning

This is safer than auto-overwriting clinical facts in Phase 1.

### 9.5 Draft vs Active Patient State

Patient registry status should remain meaningful:

- `draft`: only assets, low-confidence extraction, or no snapshot-eligible evidence yet
- `active`: at least one accepted document has successfully updated snapshot fields

This prevents unrelated PDFs from making a patient appear more complete than they are.

## 10. Doctor-Side Consumption Model

The doctor scene should consume registry upload results through three explicit surfaces.

### 10.1 Bound Patient Summary

This surface shows the trusted patient snapshot and should be the primary data source for doctor chat context.

The response must be tolerant:

- fields are normalized
- invalid scalar values become `null`
- a single bad field never causes a `500`

### 10.2 Patient Records List

This surface shows upload-derived records with at least:

- filename
- `document_type`
- `ingest_decision`
- `created_at`
- `source`
- `record_id`
- extraction summary

The purpose is to help the doctor answer:

- what was uploaded
- what kind of document was it
- did it update the snapshot
- if it did not, why not

### 10.3 Registry Alerts

Registry alerts should be surfaced separately from the summary and records list. Minimum alert types:

- `low_confidence`
- `conflict_detected`
- `not_snapshot_eligible`

Warnings should not be hidden in long paragraphs. They should be short, visible, and tied to a reason.

## 11. Doctor Chat Context Rules

Doctor chat should consume only:

- the current trusted snapshot
- the latest one or two accepted evidence summaries
- conflict or low-confidence warnings when relevant

Doctor chat should not consume:

- patient-side chat history
- full extraction payloads
- the full records trail

The doctor should receive enough context to reason about the patient, not enough context to inherit all upstream noise.

## 12. Registry vs Historical Case Boundary

The application must continue to support both data domains on one page, but their responsibilities must remain separate.

### 12.1 Patient Registry

Use for:

- current bound patient summary
- upload-derived records
- warning surfaces
- doctor-side patient-specific context

### 12.2 Historical Case Base

Use for:

- statistical analysis
- natural-language case search
- similar-case exploration
- reference-only case detail

### 12.3 Boundary Rule

Once a doctor binds a registry patient:

- registry summary and records must load only through registry endpoints
- the UI must not call `/api/database/cases/{id}` for that bound registry patient

The historical case workbench remains available, but only as a separate scope and separate detail flow.

## 13. Data and API Implications

This design does not require a storage rewrite, but it does require stricter behavior around existing registry structures.

### 13.1 Patient Registry Detail Normalization

The patient-registry detail service and schema should normalize dirty scalar values before serialization. Typical examples:

- integer-like strings for `age` may coerce to integers
- invalid values fall back to `null`
- malformed values do not crash the whole detail response

### 13.2 Record Metadata Requirements

Each upload-derived record should preserve enough metadata to support the doctor UI and downstream auditability:

- `document_type`
- `ingest_decision`
- `source`
- `created_at`
- `record_id`
- whether the record contributed to the snapshot
- whether any field conflict was detected

### 13.3 Snapshot Metadata

Phase 1 may optionally store lightweight snapshot provenance metadata, such as:

- last contributing `record_id` per field
- source type per field
- last update time per field
- conflict markers

This can live in a lightweight metadata blob rather than many new columns.

## 14. UI Implications

The doctor workspace should present the registry patient in an order that matches clinical reasoning:

1. Who is the current patient
2. Which fields are trusted
3. Which documents contributed evidence
4. Are there warnings or conflicts
5. If needed, what does the historical case base suggest

This means the minimal doctor view should clearly separate:

- trusted snapshot
- upload-derived records
- registry warnings
- historical case workbench

The UI does not need a full redesign. It does need stronger visual separation and stronger route separation.

## 15. Minimal Implementation Sequence

This design is intentionally scoped as a small corrective layer on top of the existing scene-driven baseline.

1. Fix registry detail normalization so doctor-side summary loads reliably.
2. Fix doctor-side route separation so bound registry patients never use historical-case detail routes.
3. Add document-type gating to upload ingest.
4. Upgrade snapshot merge from non-empty merge to trusted merge.
5. Add doctor-side records and warning surfaces.
6. Add regression coverage for the new safety rules.

## 16. Acceptance Criteria

### 16.1 Upload Classification

- uploading a guideline PDF saves the asset and optionally a record, but does not mutate the patient snapshot
- uploading a valid patient report creates a record and may update the patient snapshot if snapshot-eligible fields pass validation

### 16.2 Snapshot Safety

- `patients` never contains placeholder values such as `parse_failed_text`, `not_provided`, `pending_assessment`, or `Unknown`
- `treatment_draft` never appears in the patient snapshot
- a second upload for the same patient does not let bad values override existing good values
- lower-priority evidence does not override higher-priority snapshot values

### 16.3 Doctor Consumption

- doctor-side patient detail returns `200` even when one stored field is dirty
- dirty snapshot values degrade to `null` instead of causing a `500`
- doctor chat receives trusted patient summary context without inheriting patient chat history

### 16.4 Boundary Preservation

- binding a registry patient never causes the UI to request `/api/database/cases/{id}` for that patient
- the historical case workbench still supports stats, search, and case detail independently

### 16.5 Failure Visibility

- uploads classified as `unknown`, `guideline_or_education`, or `parse_failed` are visible as such in the doctor-facing record view
- snapshot conflicts are visible as warnings and traceable back to specific records

### 16.6 End-to-End Demo Path

The following manual path should pass:

1. Upload a real patient report PDF and observe a trusted snapshot update.
2. Upload a guideline PDF and observe asset persistence without snapshot pollution.
3. Bind the patient in the doctor scene.
4. View the trusted snapshot, recent records, and warnings.
5. Ask a doctor-side question and observe that the response uses registry context, not patient chat history.
6. Open the historical case workbench and verify it still works independently.

## 17. Recommendation

The recommended next move is not a broader extraction rewrite. It is a credibility hardening pass centered on:

- document-type gating
- trusted snapshot merge rules
- registry detail tolerance
- doctor-side separation between trusted snapshot, record evidence, and historical case reference

This sequence directly addresses the real failures already observed in manual verification and should be treated as the next focused design increment after the scene-driven workspace baseline.
