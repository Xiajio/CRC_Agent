from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.upload_service import store_session_upload


def _build_root() -> Path:
    root = Path("runtime") / "test-patient-registry" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_service(db_path: Path | None = None) -> PatientRegistryService:
    if db_path is None:
        db_path = _build_root() / "patient_registry.db"
    return PatientRegistryService(db_path)


def _build_medical_card() -> dict[str, object]:
    return {
        "type": "medical_visualization_card",
        "thinking_process": "stub",
        "data": {
            "patient_summary": {
                "age": "52",
                "gender": "female",
                "chief_complaint": "rectal bleeding",
            },
            "diagnosis_block": {
                "location": "rectum",
                "mmr_status": "not_provided",
            },
            "staging_block": {
                "clinical_stage": "cT3N1M0",
                "t_stage": "T3",
                "n_stage": "N1",
                "m_stage": "M0",
            },
            "key_findings": [
                {
                    "finding": "rectal wall thickening",
                }
            ],
            "treatment_draft": [
                {
                    "step": 1,
                    "name": "surgery",
                }
            ],
        },
    }


def _write_snapshot(
    service: PatientRegistryService,
    *,
    patient_id: int,
    snapshot: dict[str, object],
    document_type: str,
    filename: str,
) -> dict[str, object]:
    return service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": filename,
            "content_type": "application/pdf",
            "sha256": f"sha-{filename}",
            "storage_path": f"runtime/assets/{filename}",
            "source": "patient_generated",
        },
        patient_snapshot=snapshot,
        record_payload={"document_type": document_type},
        summary_text=document_type,
        record_type="medical_card",
    )


def test_create_draft_patient_persists_created_by_session_id() -> None:
    service = _build_service()

    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    detail = service.get_patient_detail(patient_id)
    assert detail["patient_id"] == patient_id
    assert detail["status"] == "draft"
    assert detail["created_by_session_id"] == "sess_patient_1"


def test_get_patient_detail_normalizes_dirty_age_to_none() -> None:
    service = _build_service()
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")
    with service._connect() as connection:
        connection.execute(
            "UPDATE patients SET age = ?, chief_complaint = ? WHERE id = ?",
            ("age??", "rectal bleeding", patient_id),
        )

    detail = service.get_patient_detail(patient_id)

    assert detail["patient_id"] == patient_id
    assert detail["age"] is None
    assert detail["chief_complaint"] == "rectal bleeding"


def test_write_record_merges_non_empty_snapshot_fields() -> None:
    service = _build_service()
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "ct-1.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-1",
            "storage_path": "runtime/assets/ct-1.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={
            "tumor_location": "rectum",
            "mmr_status": "not_provided",
            "clinical_stage": "cT3N1M0",
        },
        record_payload={"document_type": "ct_report"},
        summary_text="cT3N1M0 rectal lesion",
        record_type="medical_card",
    )
    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "pathology-1.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-2",
            "storage_path": "runtime/assets/pathology-1.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={
            "tumor_location": "not_provided",
            "mmr_status": "dMMR",
        },
        record_payload={"document_type": "pathology_report"},
        summary_text="dMMR confirmed",
        record_type="medical_card",
    )

    detail = service.get_patient_detail(patient_id)
    assert detail["tumor_location"] == "rectum"
    assert detail["mmr_status"] == "dMMR"
    assert detail["clinical_stage"] == "cT3N1M0"


def test_trusted_merge_prefers_higher_priority_value() -> None:
    service = _build_service()
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"tumor_location": "rectum"},
        document_type="patient_report",
        filename="patient-report-1.pdf",
    )
    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"tumor_location": "rectosigmoid"},
        document_type="pathology_report",
        filename="pathology-report-1.pdf",
    )

    detail = service.get_patient_detail(patient_id)

    assert detail["tumor_location"] == "rectosigmoid"


def test_trusted_merge_rejects_placeholder_values_and_flags_alerts() -> None:
    service = _build_service()
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"mmr_status": "dMMR"},
        document_type="pathology_report",
        filename="pathology-report-2.pdf",
    )
    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"mmr_status": "not_provided"},
        document_type="patient_report",
        filename="patient-report-2.pdf",
    )

    detail = service.get_patient_detail(patient_id)
    alerts = service.list_patient_alerts(patient_id)

    assert detail["mmr_status"] == "dMMR"
    assert any(alert["kind"] == "not_snapshot_eligible" for alert in alerts)


def test_write_record_does_not_propagate_treatment_draft_into_patient_snapshot() -> None:
    service = _build_service()
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "treatment.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-treatment",
            "storage_path": "runtime/assets/treatment.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={
            "tumor_location": "rectum",
            "treatment_draft": [{"step": 1, "name": "surgery"}],
        },
        record_payload={"document_type": "report", "treatment_draft": [{"step": 1, "name": "surgery"}]},
        summary_text="rectal lesion",
        record_type="medical_card",
    )

    detail = service.get_patient_detail(patient_id)
    assert "treatment_draft" not in detail
    assert detail["tumor_location"] == "rectum"


def test_write_record_reuses_existing_asset_for_same_patient_and_sha256() -> None:
    service = _build_service()
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    first = service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "report.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-1",
            "storage_path": "runtime/assets/report.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"tumor_location": "rectum"},
        record_payload={"document_type": "report"},
        summary_text="rectal lesion",
        record_type="medical_card",
    )
    second = service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "report.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-1",
            "storage_path": "runtime/assets/report.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"tumor_location": "rectum"},
        record_payload={"document_type": "report"},
        summary_text="rectal lesion",
        record_type="medical_card",
    )

    assert second["asset_id"] == first["asset_id"]
    assert second["reused"] is True


def test_store_session_upload_writes_registry_and_enqueues_lightweight_reference(monkeypatch) -> None:
    root = _build_root()
    registry = _build_service(root / "patient_registry.db")
    patient_id = registry.create_draft_patient(created_by_session_id="sess_patient_1")
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="patient", patient_id=patient_id)

    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _build_medical_card(),
    )

    response = store_session_upload(
        session_store=session_store,
        patient_registry=registry,
        assets_root=root / "assets",
        session_id=meta.session_id,
        filename="report.pdf",
        content_type="application/pdf",
        file_bytes=b"fake",
    )

    assert response["derived"]["record_id"] is not None
    pending = session_store.drain_context_messages(meta.session_id)
    assert len(pending) == 1
    assert isinstance(pending[0], HumanMessage)
    assert "record_id=" in pending[0].content
    assert "rectal wall thickening" in pending[0].content
    assert "Treat the following derived medical card JSON as context" not in pending[0].content


def test_delete_patient_removes_registry_rows_and_runtime_assets() -> None:
    root = _build_root()
    service = _build_service(root / "patient_registry.db")
    patient_id = service.create_draft_patient(created_by_session_id="sess_delete_one")
    asset_path = root / "assets" / "delete-one.pdf"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"delete-me")

    write_result = service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "delete-one.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-delete-one",
            "storage_path": str(asset_path),
            "source": "patient_generated",
        },
        patient_snapshot={"tumor_location": "rectum"},
        record_payload={"document_type": "patient_report"},
        summary_text="delete one",
        record_type="medical_card",
    )

    deleted = service.delete_patient(patient_id)

    assert deleted["patient_id"] == patient_id
    assert deleted["deleted_records"] == 1
    assert deleted["deleted_assets"] == 1
    assert write_result["record_id"] in deleted["record_ids"]
    assert asset_path.exists() is False
    with pytest.raises(KeyError):
        service.get_patient_detail(patient_id)


def test_clear_registry_removes_all_patients_rows_and_runtime_assets() -> None:
    root = _build_root()
    service = _build_service(root / "patient_registry.db")
    first_patient_id = service.create_draft_patient(created_by_session_id="sess_clear_a")
    second_patient_id = service.create_draft_patient(created_by_session_id="sess_clear_b")

    for patient_id, filename, sha256 in (
        (first_patient_id, "clear-a.pdf", "sha-clear-a"),
        (second_patient_id, "clear-b.pdf", "sha-clear-b"),
    ):
        asset_path = root / "assets" / filename
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(filename.encode("utf-8"))
        service.write_medical_card_record(
            patient_id=patient_id,
            asset_row={
                "filename": filename,
                "content_type": "application/pdf",
                "sha256": sha256,
                "storage_path": str(asset_path),
                "source": "patient_generated",
            },
            patient_snapshot={"tumor_location": "rectum"},
            record_payload={"document_type": "patient_report"},
            summary_text=filename,
            record_type="medical_card",
        )

    cleared = service.clear_registry()

    assert cleared["deleted_patients"] == 2
    assert cleared["deleted_records"] == 2
    assert cleared["deleted_assets"] == 2
    assert service.list_recent_patients(limit=10) == []
    assert not any((root / "assets").rglob("*"))
