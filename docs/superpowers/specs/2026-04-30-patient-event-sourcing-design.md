# Patient Event Sourcing Design

Date: 2026-04-30
Status: Draft for user review
Scope: Patient data consistency, registry source of truth, upload transaction flow, doctor context refresh, and session cache downgrade.

## Background

The current patient data path can place related facts in multiple locations:

- `InMemorySessionStore.context_state`
- `runtime/patient_registry.db`
- `runtime/assets/`
- LangGraph checkpoint state
- frontend `localStorage` session ids
- RAG references, cards, and messages derived from patient context

This creates a source-of-truth risk. For example, upload flow can write `medical_card` into session context and also write registry data, while doctor graph preparation reads registry summary and alerts into session context. If patient data changes after doctor context was injected, the graph may continue with a stale patient summary or stale medical card.

This design makes patient facts event-sourced. The event log becomes the only write-side source of patient facts. Registry tables become projections/read models. Session context becomes temporary state and versioned cache only.

## Goals

- Define one authoritative patient fact source.
- Ensure every upload has a registry asset record, including parse failures.
- Add a monotonic `patient_version` consistency contract.
- Prevent graph payloads from using stale session-cached patient facts.
- Refresh doctor context when patient facts change.
- Preserve auditability: a doctor run must identify the patient version it used.
- Keep implementation staged so the current product can keep moving.

## Non-Goals

- Replacing the session store with Redis/Postgres in this spec.
- Implementing distributed run locks in this spec.
- RAG index version governance in this spec.
- Full async projection infrastructure in the first implementation stage.
- Broad authorization and ownership enforcement, except where fields must carry actor/source metadata for later enforcement.

These remain important but should be handled as separate specs or PRs to avoid stacking unrelated risk.

## Core Invariant

Patient facts are written only through `PatientCommandService`, which appends patient events and updates projections in the same transaction.

The following are patient facts:

- identity
- medical card snapshot
- uploaded assets
- parsed clinical records
- alerts
- patient summary/snapshot

The following are not patient fact sources:

- session context
- graph checkpoint
- frontend local storage
- cards/messages/references
- copied doctor context messages

Those locations may contain derived views, references, or versioned caches only.

## Patient Version Contract

Each patient has a monotonic integer `patient_version`.

Every fact-changing event increments `patient_version` by exactly 1. This includes identity changes, upload-received events, record ingestion, alert changes, snapshot correction, and snapshot projection events when they represent a persisted patient fact change.

Consumers must treat cached patient context as valid only when its `patient_version` matches the registry projection version being read.

## Event Model

Add `patient_events` as an append-only table.

Required fields:

- `event_id`
- `patient_id`
- `patient_version`
- `event_type`
- `event_payload_json`
- `actor_type`
- `actor_id`
- `source_session_id`
- `idempotency_key`
- `causation_id`
- `correlation_id`
- `created_at`

Required constraints:

- unique `(patient_id, patient_version)`
- unique `(patient_id, idempotency_key)` when `idempotency_key` is present

Initial event types:

- `patient.created`
- `patient.identity_set`
- `patient.upload_received`
- `patient.upload_alias_added`
- `patient.upload_parse_failed`
- `patient.medical_card_extracted`
- `patient.record_ingested`
- `patient.alert_created`
- `patient.alert_resolved`
- `patient.snapshot_projected`
- `patient.snapshot_corrected`

## Projection Model

Registry tables become projections generated from events.

### `patient_assets`

Tracks uploaded files, file storage state, and parse state.

Key fields:

- `asset_id`
- `patient_id`
- `upload_event_id`
- `sha256`
- `storage_path`
- `storage_status`
- `original_filename`
- `mime_type`
- `size_bytes`
- `parse_status`
- `parse_error_code`
- `parse_error_message`
- `record_ids`
- `patient_version`

### `patient_records`

Tracks clinical records produced by uploads, manual entry, or extraction.

Key fields:

- `record_id`
- `patient_id`
- `record_type`
- `record_payload_json`
- `source_event_id`
- `asset_id`
- `patient_version`
- `created_at`

### `patient_alerts`

Tracks active and resolved alerts.

Key fields:

- `alert_id`
- `patient_id`
- `status`
- `severity`
- `message`
- `source_event_id`
- `created_patient_version`
- `resolved_patient_version`

### `patient_snapshots`

Tracks the current patient read model for graph payloads and doctor views.

Key fields:

- `patient_id`
- `patient_version`
- `projection_version`
- `medical_card_snapshot_json`
- `summary_json`
- `active_alerts_json`
- `record_refs_json`
- `asset_refs_json`
- `source_event_ids_json`
- `updated_at`

### `patient_projection_state`

Tracks projector state and future projection upgrades.

Key fields:

- `patient_id`
- `projector_name`
- `projector_schema_version`
- `last_projected_patient_version`
- `projection_version`
- `updated_at`

First implementation keeps projection synchronous and in the same SQLite transaction as event append. File-system moves are not part of the SQLite transaction, so asset projections must expose `storage_status` and callers must not treat an asset as usable until storage is confirmed. If projection later becomes async, APIs must expose projection lag explicitly instead of silently serving stale facts.

## Write Path

All patient writes go through `PatientCommandService`.

Callers:

- upload route/service
- patient identity route
- future record/alert mutation routes
- migration importer

The command service:

1. validates the command
2. calculates the next `patient_version`
3. appends one or more patient events
4. invokes `PatientProjector.apply(...)`
5. updates projections in the same database transaction
6. returns the resulting patient/projection version and created ids

Response shape:

```json
{
  "patient_id": "...",
  "patient_version": 12,
  "projection_version": 12,
  "event_ids": ["..."],
  "asset_id": "...",
  "record_id": "...",
  "alerts_changed": true,
  "snapshot_changed": true
}
```

## Upload Flow

Every upload must create a registry asset index entry, including parse failures.

Flow:

1. API receives file and writes it to a staging path.
2. Upload service computes `sha256`, MIME, size, and original filename.
3. Upload service computes the stable content-addressed path:

```text
runtime/assets/{patient_id}/{sha256}/original/{filename}
```

4. The service moves the staging file to the stable path.
5. Upload service calls `PatientCommandService.record_upload_received(...)`.
6. The command appends `patient.upload_received`.
7. The projector creates or updates `patient_assets` with `storage_status = available`.
8. If DB append or projection fails after the file move, the service removes the stable file before returning failure.
9. Session context stores only references such as `asset_id`, `event_id`, and `patient_version`.

### Parse Success

Parsing success appends events instead of directly writing session or registry snapshot fields:

- `patient.medical_card_extracted`
- `patient.record_ingested`
- `patient.alert_created`
- `patient.alert_resolved`
- `patient.snapshot_projected`

The projector updates assets, records, alerts, and snapshot projections.

### Parse Failure

Parsing failure appends `patient.upload_parse_failed`.

The projector updates:

- `patient_assets.parse_status = failed`
- `patient_assets.parse_error_code`
- `patient_assets.parse_error_message`

Optionally, it may also create a `parse_failed` record so doctor and audit views can show that a file exists but did not produce trusted clinical content.

### Idempotency

Use `(patient_id, sha256)` as the base upload idempotency key.

- Same patient uploads the same file again: do not duplicate clinical records.
- Record a `patient.upload_alias_added` event for a new session, filename, or upload source if needed.
- Different patients uploading the same file still produce separate patient asset records.

### Upload Failure Boundaries

- File write succeeds but DB event append fails: delete the staging file and return upload failure.
- DB event append succeeds but parsing fails: keep the asset and write parse failure state.
- Projection update fails: roll back the DB transaction so event/projection do not diverge.
- Stable file move fails before DB event append: no patient event is written; clean staging and return upload failure.
- DB transaction fails after stable file move: delete the stable file; if deletion fails, log an orphan cleanup task and do not expose an asset id to the session.

## Session Context Rules

Session context is a temporary graph/session state container and versioned cache.

Allowed:

```json
{
  "patient_id": "...",
  "patient_context_cache": {
    "patient_version": 12,
    "projection_version": 12,
    "medical_card_snapshot": {},
    "summary": {},
    "alerts": [],
    "record_refs": [],
    "asset_refs": [],
    "source_event_ids": [],
    "cached_at": "..."
  }
}
```

Not allowed as authoritative long-lived facts:

- top-level `context_state.medical_card`
- copied identity fields
- copied record bodies
- copied alert bodies
- doctor-injected patient summaries without version checks

New code must not write `context_state.medical_card`.

## Patient Context Resolver

Add `PatientContextResolver` as the single read gateway for graph payload patient context.

Used by:

- `payload_builder`
- `DoctorGraphService`
- patient/doctor session preparation
- upload completion refresh logic

Resolver behavior:

1. Load session meta.
2. Find bound `patient_id`.
3. Load current patient projection and version.
4. If session cache exists and `patient_version` matches, return the cache.
5. If cache is missing or stale, rebuild it from projection and update session meta.
6. If projection cannot be loaded, return an explicit stale/error result.

Graph payload building must not silently fall back to old session patient facts. If fresh context cannot be resolved, the graph should fail closed or emit a clear `PATIENT_CONTEXT_STALE` error.

## Doctor Context Refresh

Doctor sessions must refresh patient context whenever the patient version changes.

Doctor session metadata should track:

```json
{
  "bound_patient_id": "...",
  "bound_patient_version": 12,
  "last_injected_patient_version": 12,
  "patient_context_cache": {
    "patient_version": 12,
    "projection_version": 12
  }
}
```

Before every doctor graph run:

1. Resolve the current patient projection.
2. Compare `current_patient_version` with `last_injected_patient_version`.
3. If versions match, use the existing versioned cache.
4. If current version is newer, refresh cache from projection.
5. Inject a new doctor context message that includes the new version and delta.
6. Update `last_injected_patient_version`.

Doctor context injection should include:

- patient summary
- active alerts
- recent record references
- recent asset references
- delta since the previous injected version
- `patient_version`
- `source_event_ids`

It should not inject full raw uploaded documents by default.

## Graph Run Versioning

Each graph run should pin the patient version used at run start.

Rules:

- A run uses one `patient_version_used` for its full duration.
- If the patient changes while a run is active, do not mix new facts into that run.
- The next turn refreshes context before running.
- SSE `done` metadata should include `run_id`, `patient_version_used`, and `patient_context_stale`.

This makes it possible to answer which patient facts informed a doctor response.

## Error Codes

Add explicit patient consistency errors:

- `PATIENT_CONTEXT_STALE`: session cache is stale and refresh failed.
- `PATIENT_PROJECTION_LAGGED`: events exist but projection has not caught up.
- `PATIENT_UPLOAD_PARSE_FAILED`: upload succeeded but parsing failed.
- `PATIENT_EVENT_CONFLICT`: patient version or idempotency conflict.
- `PATIENT_ASSET_ORPHANED`: file and asset projection are inconsistent.

Graph payload construction should fail closed on patient context staleness for clinical outputs.

## Migration Strategy

Migration creates event history from existing registry data and then rebuilds projections.

Steps:

1. Back up `runtime/patient_registry.db`.
2. Back up `runtime/assets/`.
3. Read current patients, records, alerts, assets, and snapshots.
4. Generate legacy bootstrap events per patient:
   - `patient.created`
   - `patient.identity_set`
   - `patient.record_ingested`
   - `patient.alert_created`
   - `patient.snapshot_projected`
5. Rebuild projections from events.
6. Compare rebuilt projections with old registry reads.
7. Mark migrated sessions so they use `patient_context_cache` instead of legacy fields.

Legacy session handling:

- If a session has `patient_id`, refresh cache from projection.
- If a session has `context_state.medical_card` but no registry patient, import once as a legacy event.
- After import, session stores only `patient_context_cache` and references.

If migration fails, stop the service and restore backup. Do not continue in a half-migrated state.

## Implementation Phases

Phase 1: Event schema and command service.

- Add `patient_events`.
- Add projection tables/state where missing.
- Implement `PatientCommandService`.
- Implement synchronous `PatientProjector`.

Phase 2: Upload write path.

- Route upload writes through command service.
- Ensure all uploads create asset projection records.
- Persist parse failures.
- Stop writing new `context_state.medical_card`.

Phase 3: Context resolution.

- Add `PatientContextResolver`.
- Update `payload_builder` to use resolver output only.
- Add stale-context error behavior.

Phase 4: Doctor refresh.

- Update `DoctorGraphService` to compare patient versions every turn.
- Inject refreshed context when versions change.
- Pin `patient_version_used` per run.

Phase 5: Migration and cleanup.

- Generate bootstrap events from existing registry data.
- Rebuild and compare projections.
- Migrate legacy session medical card data.
- Remove or block old direct-write registry paths.

Phase 6: SSE metadata.

- Include `run_id`.
- Include `patient_version_used`.
- Include `patient_context_stale`.
- Consider `event_seq` in a separate SSE reliability change if this PR becomes too large.

## Test Plan

Unit tests:

- appending events increments `patient_version` monotonically
- duplicate upload idempotency does not duplicate clinical records
- parse failure produces asset projection with failed parse state
- projector rebuild from event log is deterministic
- projector failure rolls back the transaction
- context resolver refreshes stale session cache
- context resolver fails closed when projection is unavailable

Integration tests:

- upload success writes events, assets, records, snapshot, and session references
- upload parse failure still appears in registry asset view
- `payload_builder` does not use legacy `context_state.medical_card`
- patient upload after doctor binding causes doctor context refresh on next turn
- doctor run pins one `patient_version_used`
- migration rebuilds projection equivalent to old registry data

Failure tests:

- file staging succeeds but DB event append fails, staging file is cleaned up
- DB event append and projection update stay transactional
- duplicate command idempotency returns existing result or a clear conflict

## Acceptance Criteria

- There is no new write path that stores patient facts only in session context.
- All uploads, including parse failures, produce patient events and asset projections.
- `patient_version` changes on every patient fact mutation.
- Graph payload patient facts come from version-validated projection/cache only.
- Doctor sessions refresh context when patient data changes.
- A graph run exposes the patient version it used.
- Legacy session medical cards are either imported once or ignored as stale cache, never silently used as current truth.
