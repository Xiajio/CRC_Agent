# PDF Intake Credible-Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the current PDF upload loop so only trusted uploads can update the shared patient snapshot, doctor-side registry detail stays resilient to dirty data, and the doctor workspace cleanly separates registry patients from historical cases.

**Architecture:** Keep the current scene-driven dual-session baseline intact, then add a credibility layer on top of it: normalize registry detail output, classify uploaded documents before snapshot mutation, upgrade patient snapshot merge from non-empty merge to trusted merge with conflict markers, and expose doctor-facing records plus alerts through registry-specific APIs and UI surfaces. Fix the frontend so a bound registry patient never falls through to historical-case detail routes.

**Tech Stack:** Python, FastAPI, SQLite, LangGraph, React, TypeScript, Vitest, Vite, pytest

---

**Repository note:** This workspace is not currently a git repository, so commit steps are checkpoints only and cannot be executed until repository metadata is restored.

## File Structure

### Backend files to modify

- `D:\YiZhu_Agnet\LangG\backend\api\services\patient_registry_service.py`
  - Add trusted snapshot merge, dirty-value normalization, conflict metadata, and doctor-facing record/alert read helpers.
- `D:\YiZhu_Agnet\LangG\backend\api\services\upload_service.py`
  - Add document classification, ingest-decision gating, and registry-first upload behavior that updates the snapshot only when allowed.
- `D:\YiZhu_Agnet\LangG\backend\api\services\graph_service.py`
  - Expand doctor-side lazy injection so the summary can include accepted evidence and warning text, not just a bare snapshot.
- `D:\YiZhu_Agnet\LangG\backend\api\schemas\patient_registry.py`
  - Add tolerant detail serialization and explicit response models for record metadata and alerts.
- `D:\YiZhu_Agnet\LangG\backend\api\routes\patient_registry.py`
  - Expose registry detail, records, and alerts through stable doctor-facing endpoints.
- `D:\YiZhu_Agnet\LangG\backend\api\routes\uploads.py`
  - Return upload metadata that reflects document classification and ingest outcome.

### Backend files to create

- `D:\YiZhu_Agnet\LangG\tests\backend\test_upload_registry_intake.py`
  - Focused tests for upload classification, ingest-decision gating, and upload response shape.

### Backend files to modify for tests

- `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_service.py`
  - Add trusted merge, bad-value rejection, and conflict-flag coverage.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_routes.py`
  - Add detail normalization and new records/alerts route coverage.
- `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`
  - Verify doctor-side lazy injection uses trusted registry context plus warnings without repeating unnecessary noise.

### Frontend files to create

- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\patient-records-panel.tsx`
  - Show upload-derived records for the currently bound registry patient.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\patient-registry-alerts.tsx`
  - Show low-confidence, not-snapshot-eligible, and conflict warnings.

### Frontend files to modify

- `D:\YiZhu_Agnet\LangG\frontend\src\app\api\types.ts`
  - Add registry record/alert models and upload response metadata for document classification and ingest outcome.
- `D:\YiZhu_Agnet\LangG\frontend\src\app\api\client.ts`
  - Add registry records/alerts fetchers and align upload response typing.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-patient-registry.ts`
  - Fetch detail, records, and alerts for a bound registry patient.
- `D:\YiZhu_Agnet\LangG\frontend\src\features\database\use-database-workbench.ts`
  - Prevent historical detail bootstrap from hijacking a bound registry patient flow.
- `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
  - Render the trusted snapshot, record list, and alerts separately inside doctor mode and keep registry/historical queries scoped.
- `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`
  - Extend API stubs and response builders for registry records, alerts, and upload metadata.
- `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`
  - Verify registry detail resilience, registry/historical route separation, and doctor-side records/alerts rendering.

## Test Harness Notes

- Backend tests should continue to use `D:\anaconda3\envs\LangG\python.exe -m pytest ...`.
- Frontend tests should continue to use `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose`.
- Existing route tests already have working `TestClient` wiring through the current backend test harness; extend those helpers instead of creating a second app bootstrap path.
- Keep document classification deterministic in Phase 1. Do not add another LLM call just to classify uploads.

## Task 1: Harden Patient Registry Detail Serialization

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\patient_registry_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\schemas\patient_registry.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\routes\patient_registry.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_routes.py`

- [ ] **Step 1: Write a failing service test for dirty scalar normalization**

```python
def test_get_patient_detail_normalizes_dirty_age_to_none(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")
    service._connection.execute(
        "UPDATE patients SET age = ?, chief_complaint = ? WHERE id = ?",
        ("age??", "rectal bleeding", patient_id),
    )
    service._connection.commit()

    detail = service.get_patient_detail(patient_id)

    assert detail["patient_id"] == patient_id
    assert detail["age"] is None
    assert detail["chief_complaint"] == "rectal bleeding"
```

- [ ] **Step 2: Write a failing route test proving the detail endpoint returns `200` instead of `500`**

```python
def test_patient_registry_detail_route_tolerates_dirty_scalar_values(client: TestClient) -> None:
    patient_id = seed_registry_patient(client, age="age??")

    response = client.get(f"/api/patient-registry/patients/{patient_id}")

    assert response.status_code == 200
    assert response.json()["age"] is None
```

- [ ] **Step 3: Implement minimal normalization helpers**

```python
def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
```

- [ ] **Step 4: Thread the normalized values through `get_patient_detail()` and `PatientRegistryDetailResponse` construction**

```python
return PatientRegistryDetailResponse(
    patient_id=int(row["id"]),
    age=_normalize_optional_int(row.get("age")),
    ...
)
```

- [ ] **Step 5: Run focused backend tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_registry_service.py tests\backend\test_patient_registry_routes.py -v`

Expected: PASS, including the new dirty-field normalization coverage.

- [ ] **Step 6: Checkpoint**

```bash
git add backend/api/services/patient_registry_service.py backend/api/schemas/patient_registry.py backend/api/routes/patient_registry.py tests/backend/test_patient_registry_service.py tests/backend/test_patient_registry_routes.py
git commit -m "fix: normalize dirty registry detail values"
```

## Task 2: Add Deterministic Document Classification and Ingest Decisions

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\upload_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\routes\uploads.py`
- Create: `D:\YiZhu_Agnet\LangG\tests\backend\test_upload_registry_intake.py`

- [ ] **Step 1: Create a failing upload-intake test for guideline PDFs**

```python
def test_guideline_upload_is_record_only_and_does_not_update_snapshot(...) -> None:
    response = store_session_upload(..., filename="crc-guideline.pdf", file_bytes=b"%PDF-guideline")

    assert response["derived"]["document_type"] == "guideline_or_education"
    assert response["derived"]["ingest_decision"] in {"asset_only", "record_only"}
    detail = patient_registry.get_patient_detail(patient_id)
    assert detail["tumor_location"] is None
    assert detail["clinical_stage"] is None
```

- [ ] **Step 2: Add a failing upload-intake test for a valid patient report**

```python
def test_patient_report_upload_is_snapshot_eligible(...) -> None:
    response = store_session_upload(..., filename="patient-report.pdf", file_bytes=b"%PDF-report")

    assert response["derived"]["document_type"] == "patient_report"
    assert response["derived"]["ingest_decision"] == "record_and_snapshot"
```

- [ ] **Step 3: Implement a deterministic document classifier**

```python
def classify_upload_document(filename: str, card_payload: dict[str, Any]) -> str:
    lowered = filename.lower()
    if "guideline" in lowered or "consensus" in lowered:
        return "guideline_or_education"
    if has_clinical_snapshot_fields(card_payload):
        return "patient_report"
    return "unknown"
```

- [ ] **Step 4: Gate snapshot mutation behind `document_type` and emit `ingest_decision`**

```python
if document_type in {"guideline_or_education", "unknown", "parse_failed"}:
    patient_snapshot = {}
    ingest_decision = "record_only"
else:
    ingest_decision = "record_and_snapshot"
```

- [ ] **Step 5: Return the new metadata from the upload route**

```python
"derived": {
    "sqlite_record_id": record_id,
    "document_type": document_type,
    "ingest_decision": ingest_decision,
}
```

- [ ] **Step 6: Run focused backend tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_upload_registry_intake.py tests\backend\test_patient_registry_service.py -v`

Expected: PASS, with guideline uploads no longer mutating the snapshot.

- [ ] **Step 7: Checkpoint**

```bash
git add backend/api/services/upload_service.py backend/api/routes/uploads.py tests/backend/test_upload_registry_intake.py
git commit -m "feat: classify uploads before snapshot ingest"
```

## Task 3: Upgrade Snapshot Merge from Non-Empty Merge to Trusted Merge

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\patient_registry_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_service.py`

- [ ] **Step 1: Write a failing test proving bad values never override good values**

```python
def test_trusted_merge_rejects_placeholder_overwrite(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_1")
    write_snapshot(service, patient_id, {"tumor_location": "rectum"}, document_type="pathology_report")
    write_snapshot(service, patient_id, {"tumor_location": "not_provided"}, document_type="patient_report")

    detail = service.get_patient_detail(patient_id)
    assert detail["tumor_location"] == "rectum"
```

- [ ] **Step 2: Write a failing test for lower-priority overwrite rejection and conflict marking**

```python
def test_trusted_merge_keeps_high_priority_value_and_marks_conflict(tmp_path: Path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_1")
    write_snapshot(service, patient_id, {"clinical_stage": "cT3N1M0"}, document_type="pathology_report")
    write_snapshot(service, patient_id, {"clinical_stage": "cT2N0M0"}, document_type="patient_report")

    detail = service.get_patient_detail(patient_id)
    records = service.list_patient_records(patient_id)
    assert detail["clinical_stage"] == "cT3N1M0"
    assert any(record.get("conflict_detected") for record in records)
```

- [ ] **Step 3: Implement trusted-merge helpers inside `PatientRegistryService`**

```python
SOURCE_PRIORITY = {
    "doctor_curated": 100,
    "pathology_report": 80,
    "imaging_report": 70,
    "patient_summary": 60,
    "patient_report": 50,
    "unknown": 10,
    "guideline_or_education": 0,
}
```

- [ ] **Step 4: Store enough metadata to explain snapshot provenance**

```python
snapshot_meta[field_name] = {
    "record_id": record_id,
    "document_type": document_type,
    "priority": SOURCE_PRIORITY[document_type],
    "conflict_detected": conflict_detected,
}
```

- [ ] **Step 5: Run focused backend tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_registry_service.py -v`

Expected: PASS, including placeholder rejection and conflict coverage.

- [ ] **Step 6: Checkpoint**

```bash
git add backend/api/services/patient_registry_service.py tests/backend/test_patient_registry_service.py
git commit -m "feat: apply trusted merge to patient snapshot"
```

## Task 4: Expose Registry Records and Alerts, Then Use Them in Doctor Context

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\patient_registry_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\schemas\patient_registry.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\routes\patient_registry.py`
- Modify: `D:\YiZhu_Agnet\LangG\backend\api\services\graph_service.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_patient_registry_routes.py`
- Modify: `D:\YiZhu_Agnet\LangG\tests\backend\test_graph_service_streaming.py`

- [ ] **Step 1: Write a failing route test for registry alerts**

```python
def test_patient_registry_alerts_route_returns_conflict_and_low_confidence(client: TestClient) -> None:
    patient_id = seed_conflicted_registry_patient(client)

    response = client.get(f"/api/patient-registry/patients/{patient_id}/alerts")

    assert response.status_code == 200
    assert response.json()["items"][0]["kind"] in {"conflict_detected", "low_confidence"}
```

- [ ] **Step 2: Write a failing graph-service test proving doctor injection includes warning text**

```python
def test_doctor_graph_service_injects_registry_summary_with_alerts() -> None:
    registry = FakePatientRegistry(
        summary_message="Bound patient summary: patient_id=33, clinical_stage=cT3N1M0. Warning: conflict_detected on mmr_status."
    )
    ...
    assert "conflict_detected" in drained_messages[0].content
```

- [ ] **Step 3: Add alert models and route handlers**

```python
class PatientRegistryAlert(BaseModel):
    kind: str
    message: str
    record_id: int | None = None
```

- [ ] **Step 4: Implement alert derivation and evidence summary helpers in the service**

```python
def list_patient_alerts(self, patient_id: int) -> list[dict[str, Any]]:
    ...

def get_patient_summary_message(self, patient_id: int) -> str:
    ...
```

- [ ] **Step 5: Update `DoctorGraphService` to inject trusted summary plus warnings**

```python
summary_message = self._patient_registry.get_patient_summary_message(meta.patient_id)
self._session_store.enqueue_context_message(session_id, HumanMessage(content=summary_message))
```

- [ ] **Step 6: Run focused backend tests**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_registry_routes.py tests\backend\test_graph_service_streaming.py -v`

Expected: PASS, including the new alerts route and doctor-context warning coverage.

- [ ] **Step 7: Checkpoint**

```bash
git add backend/api/services/patient_registry_service.py backend/api/schemas/patient_registry.py backend/api/routes/patient_registry.py backend/api/services/graph_service.py tests/backend/test_patient_registry_routes.py tests/backend/test_graph_service_streaming.py
git commit -m "feat: expose registry alerts and trusted doctor context"
```

## Task 5: Fix Frontend Registry/Historical Separation and Render Registry Evidence

**Files:**
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\patient-records-panel.tsx`
- Create: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\patient-registry-alerts.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\app\api\types.ts`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\app\api\client.ts`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\patient-registry\use-patient-registry.ts`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\features\database\use-database-workbench.ts`
- Modify: `D:\YiZhu_Agnet\LangG\frontend\src\pages\workspace-page.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\test-utils.tsx`
- Modify: `D:\YiZhu_Agnet\LangG\tests\frontend\workspace-scenes.test.tsx`

- [ ] **Step 1: Write a failing frontend test proving a bound registry patient does not trigger historical-case detail fetch**

```tsx
it("does not request historical case detail for a bound registry patient", async () => {
  const api = buildApiClientStub({
    getPatientRegistryDetail: vi.fn().mockResolvedValue(makePatientRegistryDetail()),
    getDatabaseCaseDetail: vi.fn(),
  });

  renderWorkspaceWithSceneSessions({ apiClient: api, activeScene: "doctor", doctorPatientId: 1 });

  await screen.findByText("Bound Patient 1");
  expect(api.getDatabaseCaseDetail).not.toHaveBeenCalled();
});
```

- [ ] **Step 2: Write a failing frontend test for records and alerts rendering**

```tsx
it("renders registry records and alerts for the bound patient", async () => {
  const api = buildApiClientStub({
    getPatientRegistryDetail: vi.fn().mockResolvedValue(makePatientRegistryDetail()),
    getPatientRegistryRecords: vi.fn().mockResolvedValue(makePatientRegistryRecordsResponse()),
    getPatientRegistryAlerts: vi.fn().mockResolvedValue(makePatientRegistryAlertsResponse()),
  });

  renderWorkspaceWithSceneSessions({ apiClient: api, activeScene: "doctor", doctorPatientId: 1 });

  expect(await screen.findByText("Registry Alerts")).toBeInTheDocument();
  expect(await screen.findByText("Patient Records")).toBeInTheDocument();
});
```

- [ ] **Step 3: Extend the API types and client methods**

```ts
export interface PatientRegistryAlert {
  kind: string;
  message: string;
  record_id: number | null;
}
```

- [ ] **Step 4: Update `usePatientRegistry` to fetch detail, records, and alerts together**

```ts
const [detail, records, alerts] = await Promise.all([
  apiClient.getPatientRegistryDetail(patientId),
  apiClient.getPatientRegistryRecords(patientId),
  apiClient.getPatientRegistryAlerts(patientId),
]);
```

- [ ] **Step 5: Keep registry and historical scopes separate in `workspace-page.tsx` and `useDatabaseWorkbench`**

```ts
const shouldLoadHistoricalDetail =
  scope === "historical_case_base" && boundPatientSource !== "patient_registry";
```

- [ ] **Step 6: Run focused frontend tests**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose`

Expected: PASS, including registry records/alerts rendering and no historical detail fetch for bound registry patients.

- [ ] **Step 7: Checkpoint**

```bash
git add frontend/src/app/api/types.ts frontend/src/app/api/client.ts frontend/src/features/patient-registry/use-patient-registry.ts frontend/src/features/patient-registry/patient-records-panel.tsx frontend/src/features/patient-registry/patient-registry-alerts.tsx frontend/src/features/database/use-database-workbench.ts frontend/src/pages/workspace-page.tsx tests/frontend/test-utils.tsx tests/frontend/workspace-scenes.test.tsx
git commit -m "feat: render trusted registry evidence in doctor workspace"
```

## Task 6: Regression Sweep and Manual Verification

**Files:**
- Modify: `D:\YiZhu_Agnet\LangG\docs\superpowers\specs\2026-04-17-pdf-intake-credible-closure-design.md` (only if implementation requires a spec clarification)

- [ ] **Step 1: Run backend regression relevant to the credibility pass**

Run: `D:\anaconda3\envs\LangG\python.exe -m pytest tests\backend\test_patient_registry_service.py tests\backend\test_patient_registry_routes.py tests\backend\test_upload_registry_intake.py tests\backend\test_graph_service_streaming.py -v`

Expected: PASS

- [ ] **Step 2: Run frontend regression**

Run: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run --reporter=verbose`

Expected: PASS

- [ ] **Step 3: Perform manual registry credibility checks**

Manual checklist:

1. Upload a real patient report PDF and verify snapshot fields update.
2. Upload a guideline PDF and verify asset persistence without snapshot mutation.
3. Bind the patient in doctor mode and verify detail returns successfully.
4. Confirm records and alerts load.
5. Send one doctor-side message and verify the response uses registry context, not patient chat history.
6. Open the historical case workbench and verify it still works independently.

- [ ] **Step 4: Record any acceptance deltas**

If behavior differs from the approved spec, update the spec before claiming completion.

- [ ] **Step 5: Final checkpoint**

```bash
git add .
git commit -m "test: complete pdf intake credible-closure regression sweep"
```
