from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import patient_registry as patient_registry_routes
from backend.api.services.patient_registry_service import PatientRegistryService


def _build_service(tmp_path: Path) -> PatientRegistryService:
    return PatientRegistryService(tmp_path / "patient_registry.db")


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


def _build_client(tmp_path: Path) -> TestClient:
    app = FastAPI()
    app.state.runtime = SimpleNamespace(patient_registry_service=_build_service(tmp_path))
    app.include_router(patient_registry_routes.router)
    return TestClient(app)


def test_get_patient_detail_derives_tumor_region_codes_for_legacy_rows(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")
    with service._connect() as connection:
        connection.execute(
            "UPDATE patients SET tumor_location = ?, tumor_region_code = NULL, tumor_region_codes_json = ? WHERE id = ?",
            ("\u4e59\u72b6\u7ed3\u80a0", "[]", patient_id),
        )

    detail = service.get_patient_detail(patient_id)

    assert detail["tumor_region_code"] == "sigmoid_colon"
    assert detail["tumor_region_codes"] == ["sigmoid_colon"]


def test_write_record_persists_tumor_region_codes_with_provenance(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")

    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"tumor_location": "rectum"},
        document_type="pathology_report",
        filename="rectum.pdf",
    )

    detail = service.get_patient_detail(patient_id)
    assert detail["tumor_region_code"] == "rectum"
    assert detail["tumor_region_codes"] == ["rectum"]
    with service._connect() as connection:
        row = connection.execute(
            """
            SELECT tumor_region_code, tumor_region_codes_json, snapshot_provenance_json
            FROM patients
            WHERE id = ?
            """,
            (patient_id,),
        ).fetchone()

    assert row["tumor_region_code"] == "rectum"
    assert row["tumor_region_codes_json"] == '["rectum"]'
    provenance = json.loads(row["snapshot_provenance_json"])
    assert provenance["tumor_region_code"]["derived_from"] == "tumor_location"
    assert provenance["tumor_region_code"]["priority"] == 80
    assert provenance["tumor_region_codes"]["derived_from"] == "tumor_location"


def test_region_codes_follow_higher_priority_location_conflicts(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")
    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"tumor_location": "rectum"},
        document_type="patient_report",
        filename="patient-report.pdf",
    )
    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"tumor_location": "rectosigmoid"},
        document_type="pathology_report",
        filename="pathology-report.pdf",
    )

    detail = service.get_patient_detail(patient_id)

    assert detail["tumor_location"] == "rectosigmoid"
    assert detail["tumor_region_code"] == "rectosigmoid"
    assert detail["tumor_region_codes"] == ["rectosigmoid"]
    with service._connect() as connection:
        row = connection.execute(
            "SELECT snapshot_provenance_json FROM patients WHERE id = ?",
            (patient_id,),
        ).fetchone()
    provenance = json.loads(row["snapshot_provenance_json"])
    assert provenance["tumor_region_code"]["derived_from"] == "tumor_location"
    assert provenance["tumor_region_code"]["conflict_detected"] is True


def test_search_patients_filters_by_tumor_region_code(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    first = service.create_draft_patient(created_by_session_id="sess_first")
    second = service.create_draft_patient(created_by_session_id="sess_second")
    _write_snapshot(
        service,
        patient_id=first,
        snapshot={"tumor_location": "rectum"},
        document_type="patient_report",
        filename="first-region.pdf",
    )
    _write_snapshot(
        service,
        patient_id=second,
        snapshot={"tumor_location": "colon"},
        document_type="patient_report",
        filename="second-region.pdf",
    )

    result = service.search_patients(tumor_region_code="rectum", limit=10)

    assert result["total"] == 1
    assert result["items"][0]["patient_id"] == first
    assert result["items"][0]["tumor_region_code"] == "rectum"


def test_search_patients_returns_no_rows_for_unknown_tumor_region_code(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    patient_id = service.create_draft_patient(created_by_session_id="sess_unknown_region")
    _write_snapshot(
        service,
        patient_id=patient_id,
        snapshot={"tumor_location": "rectum"},
        document_type="patient_report",
        filename="unknown-region.pdf",
    )

    result = service.search_patients(tumor_region_code="not_a_region", limit=10)

    assert result == {"items": [], "total": 0}


def test_patient_detail_route_returns_structured_tumor_region_fields(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        service = client.app.state.runtime.patient_registry_service
        patient_id = service.create_draft_patient(created_by_session_id="sess_a")
        service.write_medical_card_record(
            patient_id=patient_id,
            asset_row={
                "filename": "detail.pdf",
                "content_type": "application/pdf",
                "sha256": "sha-detail",
                "storage_path": "runtime/assets/detail.pdf",
                "source": "patient_generated",
            },
            patient_snapshot={"tumor_location": "rectum", "clinical_stage": "cT3N1M0"},
            record_payload={"document_type": "report"},
            summary_text="detail",
            record_type="medical_card",
        )

        response = client.get(f"/api/patient-registry/patients/{patient_id}")

    payload = response.json()
    assert response.status_code == 200
    assert payload["tumor_region_code"] == "rectum"
    assert payload["tumor_region_codes"] == ["rectum"]


def test_search_patients_route_accepts_tumor_region_code(tmp_path: Path) -> None:
    with _build_client(tmp_path) as client:
        service = client.app.state.runtime.patient_registry_service
        first = service.create_draft_patient(created_by_session_id="sess_first")
        second = service.create_draft_patient(created_by_session_id="sess_second")
        _write_snapshot(
            service,
            patient_id=first,
            snapshot={"tumor_location": "rectum"},
            document_type="patient_report",
            filename="route-first.pdf",
        )
        _write_snapshot(
            service,
            patient_id=second,
            snapshot={"tumor_location": "colon"},
            document_type="patient_report",
            filename="route-second.pdf",
        )

        response = client.post(
            "/api/patient-registry/patients/search",
            json={"tumor_region_code": "rectum", "limit": 10},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["total"] == 1
    assert payload["items"][0]["patient_id"] == first
    assert payload["items"][0]["tumor_region_code"] == "rectum"
