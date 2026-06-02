from __future__ import annotations

from backend.api.services.patient_registry_service import PatientRegistryService
from scripts.backfill_tumor_region_codes import backfill_tumor_region_codes


def test_backfill_tumor_region_codes_updates_legacy_patient_rows(tmp_path) -> None:
    service = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_backfill")
    with service._connect() as connection:
        connection.execute(
            "UPDATE patients SET tumor_location = ?, tumor_region_code = NULL, tumor_region_codes_json = ? WHERE id = ?",
            ("rectosigmoid junction", "[]", patient_id),
        )

    result = backfill_tumor_region_codes(service.db_path)

    assert result == {"updated": 1, "scanned": 1}
    detail = service.get_patient_detail(patient_id)
    assert detail["tumor_region_code"] == "rectosigmoid"
    assert detail["tumor_region_codes"] == ["rectosigmoid"]
