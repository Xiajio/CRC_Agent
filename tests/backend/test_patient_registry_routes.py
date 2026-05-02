from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import patient_registry as patient_registry_routes
from backend.api.services.patient_registry_service import PatientRegistryService


def _build_root() -> Path:
    root = Path("runtime") / "test-patient-registry-routes" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def client() -> TestClient:
    root = _build_root()
    service = PatientRegistryService(root / "patient_registry.db")
    app = FastAPI()
    app.state.runtime = SimpleNamespace(patient_registry_service=service)
    app.include_router(patient_registry_routes.router)
    return TestClient(app)


def seed_registry_patient(client: TestClient, **snapshot: object) -> int:
    service = client.app.state.runtime.patient_registry_service
    patient_id = service.create_draft_patient(created_by_session_id="sess_seed")
    if snapshot:
        service.write_medical_card_record(
            patient_id=patient_id,
            asset_row={
                "filename": "seed.pdf",
                "content_type": "application/pdf",
                "sha256": "sha-seed",
                "storage_path": "runtime/assets/seed.pdf",
                "source": "patient_generated",
            },
            patient_snapshot=dict(snapshot),
            record_payload={"document_type": "patient_report"},
            summary_text="seed",
            record_type="medical_card",
        )
    return patient_id


def test_recent_patients_route_returns_registry_rows(client: TestClient) -> None:
    service = client.app.state.runtime.patient_registry_service
    first = service.create_draft_patient(created_by_session_id="sess_a")
    second = service.create_draft_patient(created_by_session_id="sess_b")

    response = client.get("/api/patient-registry/patients/recent?limit=5")

    payload = response.json()
    assert response.status_code == 200
    assert {item["patient_id"] for item in payload["items"]} == {first, second}


def test_patient_detail_route_tolerates_dirty_scalar_values(client: TestClient) -> None:
    patient_id = seed_registry_patient(client, age="age??")
    service = client.app.state.runtime.patient_registry_service
    with service._connect() as connection:
        connection.execute("UPDATE patients SET age = ? WHERE id = ?", ("age??", patient_id))

    response = client.get(f"/api/patient-registry/patients/{patient_id}")

    assert response.status_code == 200
    assert response.json()["age"] is None


def test_search_patients_route_returns_filtered_rows(client: TestClient) -> None:
    service = client.app.state.runtime.patient_registry_service
    first = service.create_draft_patient(created_by_session_id="sess_a")
    second = service.create_draft_patient(created_by_session_id="sess_b")
    service.write_medical_card_record(
        patient_id=first,
        asset_row={
            "filename": "first.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-first",
            "storage_path": "runtime/assets/first.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"tumor_location": "rectum", "clinical_stage": "cT3N1M0"},
        record_payload={"document_type": "report"},
        summary_text="first",
        record_type="medical_card",
    )
    service.write_medical_card_record(
        patient_id=second,
        asset_row={
            "filename": "second.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-second",
            "storage_path": "runtime/assets/second.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"tumor_location": "colon", "clinical_stage": "cT2N0M0"},
        record_payload={"document_type": "report"},
        summary_text="second",
        record_type="medical_card",
    )

    response = client.post(
        "/api/patient-registry/patients/search",
        json={"tumor_location": "rectum", "limit": 10},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["total"] == 1
    assert payload["items"][0]["patient_id"] == first


def test_patient_detail_and_records_routes_return_registry_data(client: TestClient) -> None:
    service = client.app.state.runtime.patient_registry_service
    patient_id = service.create_draft_patient(created_by_session_id="sess_a")
    write_result = service.write_medical_card_record(
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

    detail_response = client.get(f"/api/patient-registry/patients/{patient_id}")
    records_response = client.get(f"/api/patient-registry/patients/{patient_id}/records")

    detail_payload = detail_response.json()
    records_payload = records_response.json()
    assert detail_response.status_code == 200
    assert detail_payload["patient_id"] == patient_id
    assert detail_payload["tumor_location"] == "rectum"
    assert records_response.status_code == 200
    assert records_payload["items"][0]["record_id"] == write_result["record_id"]
    assert records_payload["items"][0]["document_type"] == "report"


def test_patient_registry_records_and_alerts_routes_expose_conflicts(client: TestClient) -> None:
    service = client.app.state.runtime.patient_registry_service
    patient_id = service.create_draft_patient(created_by_session_id="sess_a")
    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "pathology.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-pathology",
            "storage_path": "runtime/assets/pathology.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"clinical_stage": "cT3N1M0"},
        record_payload={"document_type": "pathology_report"},
        summary_text="pathology",
        record_type="medical_card",
    )
    service.write_medical_card_record(
        patient_id=patient_id,
        asset_row={
            "filename": "patient.pdf",
            "content_type": "application/pdf",
            "sha256": "sha-patient",
            "storage_path": "runtime/assets/patient.pdf",
            "source": "patient_generated",
        },
        patient_snapshot={"clinical_stage": "cT2N0M0"},
        record_payload={"document_type": "patient_report"},
        summary_text="patient",
        record_type="medical_card",
    )

    records_response = client.get(f"/api/patient-registry/patients/{patient_id}/records")
    alerts_response = client.get(f"/api/patient-registry/patients/{patient_id}/alerts")

    assert records_response.status_code == 200
    assert alerts_response.status_code == 200
    assert any(item["conflict_detected"] for item in records_response.json()["items"])
    assert alerts_response.json()["items"]
    assert alerts_response.json()["items"][0]["kind"] in {"conflict_detected", "low_confidence"}


def test_delete_patient_route_removes_one_registry_patient(client: TestClient) -> None:
    patient_id = seed_registry_patient(client, tumor_location="rectum")

    response = client.delete(f"/api/patient-registry/patients/{patient_id}")

    assert response.status_code == 200
    assert response.json()["patient_id"] == patient_id
    missing = client.get(f"/api/patient-registry/patients/{patient_id}")
    assert missing.status_code == 404


def test_clear_registry_route_removes_all_registry_patients(client: TestClient) -> None:
    seed_registry_patient(client, tumor_location="rectum")
    seed_registry_patient(client, tumor_location="colon")

    response = client.delete("/api/patient-registry/patients")

    assert response.status_code == 200
    assert response.json()["deleted_patients"] == 2
    recent = client.get("/api/patient-registry/patients/recent?limit=10")
    assert recent.status_code == 200
    assert recent.json()["items"] == []
