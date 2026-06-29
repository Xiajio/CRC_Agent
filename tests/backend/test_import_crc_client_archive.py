from __future__ import annotations

from pathlib import Path

from backend.api.services.patient_registry_service import PatientRegistryService
from scripts.import_crc_client_archive import import_crc_archive


def test_import_crc_archive_creates_patient_and_crc_triage_record(tmp_path: Path) -> None:
    csv_path = tmp_path / "assessment-archive-v4.csv"
    csv_path.write_text(
        "患者ID,用户主诉,风险等级,转诊意见,诊断意见,关键线索,就诊建议,诊前观察要点,问题1,本次就诊时间\n"
        "2606250001,反复便血,中危,消化专科,结直肠病变待排,便血,预约门诊,记录便血颜色,Q: 是否便血 | A: 是,2026-06-25T08:00:00Z\n",
        encoding="utf-8",
    )
    registry = PatientRegistryService(tmp_path / "patient_registry.db")

    result = import_crc_archive(csv_path=csv_path, registry=registry, source_session_id="import_test")

    assert result == {"imported_records": 1, "skipped_rows": 0}
    patient = registry.search_patients(limit=10)["items"][0]
    identity = registry.get_patient_identity(patient["patient_id"])
    assert identity["patient_number"] == "2606250001"
    records = registry.list_patient_records(patient["patient_id"])
    assert records[0]["record_type"] == "crc_triage_assessment"
    assert records[0]["summary_text"] == "反复便血；中危；消化专科"


def test_import_crc_archive_reuses_existing_patient_number_and_is_idempotent(tmp_path: Path) -> None:
    csv_path = tmp_path / "assessment-archive-v4.csv"
    csv_path.write_text(
        "患者ID,用户主诉,风险等级,转诊意见,诊断意见,关键线索,就诊建议,诊前观察要点\n"
        "2606250001,反复便血,中危,消化专科,结直肠病变待排,便血,预约门诊,记录便血颜色\n",
        encoding="utf-8",
    )
    registry = PatientRegistryService(tmp_path / "patient_registry.db")

    first = import_crc_archive(csv_path=csv_path, registry=registry, source_session_id="import_test")
    second = import_crc_archive(csv_path=csv_path, registry=registry, source_session_id="import_test")

    assert first["imported_records"] == 1
    assert second["imported_records"] == 1
    patient = registry.search_patients(limit=10)["items"][0]
    records = registry.list_patient_records(patient["patient_id"])
    assert len(records) == 1
