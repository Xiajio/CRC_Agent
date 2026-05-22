# Patient Context Split Design

Date: 2026-05-05
Status: Draft for user review
Scope: Separate case database sample context from patient registry context across graph state, snapshots, payload building, and frontend registry binding.

## Background

The doctor workspace currently allows a case database sample id, such as `093`, to flow through the same field that the frontend uses as the patient registry id. This creates a patient context contamination risk:

- Prompt and tool examples frequently mention `93` or `093`.
- `node_case_database` can extract a broad one-to-three digit number from user text or historical user messages.
- That value can be written to `current_patient_id`.
- `state_snapshot.py` exposes `current_patient_id`.
- `payload_builder.py` injects `current_patient_id` into later graph turns.
- The frontend parses `"093"` as number `93` and uses it to call patient registry routes.

The visible symptom is repeated `/api/patient-registry/patients/93` 404 responses when no registry patient exists. The larger issue is that a historical case database sample id can become active patient state for later clinical context.

## Goals

- Make case database sample ids and patient registry ids separate concepts.
- Prevent sample ids from automatically binding the doctor workspace to a registry patient.
- Preserve case database, imaging, pathology, and card workflows for queries like `view sample 093 imaging`.
- Keep explicit registry binding available through create, select, or bind actions.
- Prevent old `current_patient_id="093"` snapshots from causing registry requests.
- Keep migration incremental and testable.

## Non-Goals

- Replacing patient event sourcing or registry storage.
- Removing every `93` or `093` example from prompts and tool docs in this design.
- Reworking the entire intent classifier.
- Adding patient-number based auto-binding.
- Solving authorization or multi-user ownership.

Prompt cleanup can still be useful, but it is not the primary safety mechanism.

## Core Invariant

The system must never infer a registry patient id from a case database sample id.

`case_database_patient_id` and `registry_patient_id` are different fields with different sources, consumers, and validation rules.

## Field Semantics

### `case_database_patient_id`

Type: `str | None`

Format:

- Preserve case database folder/sample format.
- Normalize numeric input to three digits when used for case database lookup, for example `93` becomes `"093"`.

Allowed sources:

- User asks for case database, imaging, pathology, radiomics, or historical sample data.
- Case database tools return a selected historical sample id.
- Legacy graph state has `current_patient_id`, and no explicit registry patient is bound.

Allowed consumers:

- Case database tools.
- Imaging sample cards.
- Pathology sample cards.
- Radiomics/sample report cards.
- Follow-up case database turns.

Forbidden consumers:

- Patient registry detail, records, alerts, identity, and binding routes.
- Registry patient context injection.

### `registry_patient_id`

Type: `int | None`

Format:

- Registry primary key.
- No leading-zero string semantics.

Allowed sources:

- Patient session creation.
- Explicit doctor bind action.
- Explicit doctor selection from patient registry.
- Registry create/import command that returns a real registry patient id.

Allowed consumers:

- Patient registry detail, records, alerts, identity.
- Doctor patient context resolver.
- Registry-bound medical card summaries.

Forbidden sources:

- Numeric extraction from chat text.
- Case database sample lookup.
- Imaging/pathology tool calls.
- Legacy `current_patient_id` unless a registry lookup has already proven it exists and the user explicitly asked to bind.

### `current_patient_id`

Type: deprecated compatibility field

Rules:

- Do not use it for new writes.
- Do not use it to drive frontend registry requests.
- During migration, snapshots may still include it for old consumers, but new code reads `case_database_patient_id` or `registry_patient_id`.
- If old checkpoint state only has `current_patient_id="093"`, treat it as `case_database_patient_id="093"` for case database continuity, not as `registry_patient_id=93`.

## Backend Design

### State Model

Add fields to `CRCAgentState`:

- `case_database_patient_id: Optional[str] = None`
- `registry_patient_id: Optional[int] = None`

Keep `current_patient_id` temporarily for compatibility. New logic should not write it.

### Database Node

`node_case_database` should write:

- `case_database_patient_id`
- `findings["case_database_patient_id"]`

It must stop writing:

- `current_patient_id` for case database sample discovery
- `findings["current_patient_id"]` for case database sample discovery

The broad fallback extractor can remain only for case database sample ids, but its result must not become registry context.

The history fallback should only inspect recent user messages when the active route is a case database route. It should not backfill registry context.

### Radiology and Pathology Nodes

Radiology and pathology flows should resolve sample ids in this order:

1. Explicit id in the current user request.
2. `state.case_database_patient_id`.
3. `findings["case_database_patient_id"]`.
4. Legacy `state.current_patient_id`, imported as a case database id only.

They should not read `registry_patient_id` unless a future feature explicitly maps registry patients to case samples.

### Payload Builder

`payload_builder.py` should allowlist and inject:

- `case_database_patient_id`
- `registry_patient_id`

It should stop injecting `current_patient_id` into new graph payloads except for a short migration bridge that maps legacy `current_patient_id` to `case_database_patient_id` when no new field is present.

The migration bridge must not set `registry_patient_id`.

### Snapshot Builder

`state_snapshot.py` should expose:

- `case_database_patient_id`
- `registry_patient_id`
- deprecated `current_patient_id`

Snapshot precedence:

1. `registry_patient_id` comes from session meta or explicit graph state only.
2. `case_database_patient_id` comes from graph state/findings/legacy `current_patient_id`.
3. `current_patient_id` may be populated for compatibility, but frontend registry code must ignore it.

### Session Store and Binding

`SessionMeta.patient_id` remains the authoritative registry binding for doctor sessions.

Explicit binding through `/api/sessions/{session_id}` should update:

- `SessionMeta.patient_id`
- snapshot `registry_patient_id`

Reset for doctor sessions clears:

- `SessionMeta.patient_id`
- `context_state`
- graph thread id

Because reset rotates the graph thread, legacy checkpoint values should not be available after reset.

### Patient Context Resolver

Doctor patient context injection must depend only on `SessionMeta.patient_id` or explicit `registry_patient_id`.

It must not read `case_database_patient_id` or legacy `current_patient_id`.

## Frontend Design

### API Types and Store

Add snapshot fields:

- `case_database_patient_id: string | null`
- `registry_patient_id: number | null`

Add session state fields:

- `caseDatabasePatientId: string | null`
- `registryPatientId: number | null`

Keep `currentPatientId` temporarily for compatibility displays only. It must not drive registry fetches.

### Workspace Page

Replace:

- `doctorPatientId = readFiniteNumber(doctor.state.currentPatientId)`

With:

- `registryPatientId = readFiniteNumber(doctor.state.registryPatientId)`

`usePatientRegistry` receives `registryPatientId`, not `currentPatientId`.

When only `caseDatabasePatientId="093"` is present:

- registry panel remains unbound
- no patient registry detail/records/alerts requests run
- case database cards and imaging/pathology sample views still render

### Doctor Scene UI

Show separate context labels:

- `Registry patient`: `P-7` or unbound
- `Case sample`: `093` or none

This makes it visible when the doctor is browsing historical samples without binding a real registry patient.

### Registry Browser

Selecting or binding a registry patient updates only `registryPatientId`.

It does not overwrite `caseDatabasePatientId`.

### Cards

Patient registry cards should show `registry_patient_id`.

Case database, imaging, pathology, and radiomics cards should show `case_database_patient_id` or their own payload-specific sample id.

## Error Handling

### Missing Case Database Sample

If `case_database_patient_id="093"` has no case data:

- show a case database error card or assistant message
- do not clear or mutate `registry_patient_id`
- do not call patient registry endpoints

### Missing Registry Patient

This should only happen after explicit binding or stale session data.

Frontend behavior:

- show registry binding error
- stop retrying the same id until the id changes or the user retries
- keep case database sample context unchanged

Backend behavior:

- session `GET` should leave `patient_identity` null when identity lookup fails
- explicit bind should validate the registry patient exists before accepting the binding

## Migration Plan

### Stage 1: Add Fields and Snapshot Contract

- Add state fields.
- Add response schema fields.
- Add snapshot builder mapping.
- Add payload builder allowlist.
- Keep old `current_patient_id` behavior in place for existing tests.

### Stage 2: Move Case Database Writes

- Update `node_case_database` to write `case_database_patient_id`.
- Update imaging/pathology/radiology nodes to read sample id from the new field.
- Add legacy import from `current_patient_id` to `case_database_patient_id` only.

### Stage 3: Move Frontend Registry Binding

- Hydrate `caseDatabasePatientId` and `registryPatientId`.
- Make `usePatientRegistry` depend only on `registryPatientId`.
- Update doctor UI labels and tests.
- Ensure old `current_patient_id="093"` does not trigger registry fetch.

### Stage 4: Tighten Compatibility

- Stop writing `current_patient_id` from graph nodes.
- Keep read-only snapshot compatibility for one release window.
- Remove compatibility reads after tests and callers no longer depend on it.

## Test Plan

### Backend Unit Tests

- `node_case_database` extracts `093` into `case_database_patient_id`.
- `node_case_database` does not return `current_patient_id` for sample queries.
- Payload builder injects `case_database_patient_id` and `registry_patient_id`.
- Payload builder maps legacy `current_patient_id="093"` to `case_database_patient_id`, not `registry_patient_id`.
- Snapshot builder returns both new fields.
- Patient context resolver ignores `case_database_patient_id`.

### Frontend Unit Tests

- Snapshot hydration stores `caseDatabasePatientId` and `registryPatientId`.
- `usePatientRegistry` is not called when only `caseDatabasePatientId` is present.
- `usePatientRegistry` is called when `registryPatientId` is present.
- Old `currentPatientId="093"` does not cause registry fetch.
- Doctor scene displays separate registry patient and case sample labels.

### Integration Tests

- User asks for `093` imaging: case database and imaging cards render; registry routes are not requested.
- User binds registry patient `7`: registry detail/records/alerts are requested for `7`.
- User resets doctor session: both registry and case sample context clear.
- User asks another case database question after sample `093`: graph reuses `case_database_patient_id`, not registry id.

## Acceptance Criteria

- No chat-derived case database id can trigger `/api/patient-registry/patients/{id}`.
- Explicit registry binding still works.
- Case database sample browsing still works.
- Old snapshots with `current_patient_id="093"` do not produce registry requests.
- The UI clearly distinguishes registry patient context from case sample context.
- Tests cover the full leak path from extraction to frontend request prevention.

## Risks and Mitigations

- Risk: Existing nodes still read `current_patient_id`.
  Mitigation: Add compatibility helper functions and tests for all known consumers.

- Risk: Frontend or tests still assume one patient id field.
  Mitigation: Update API types and store first, then migrate consumers one by one.

- Risk: Users expect `093` to bind a registry patient.
  Mitigation: Show separate labels and require explicit registry selection or binding.

- Risk: Too many files change at once.
  Mitigation: Implement in staged commits with backend contract tests before frontend migration.

## Implementation Boundary

This spec authorizes the architecture and migration shape only. Implementation should be planned separately before editing production code.
