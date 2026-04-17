from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import uploads as upload_routes
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.upload_service import store_session_upload


def _build_root() -> Path:
    root = Path("runtime") / "test-upload-registry-intake" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_service(root: Path) -> PatientRegistryService:
    return PatientRegistryService(root / "patient_registry.db")


def _build_session_store(*, patient_id: int) -> tuple[InMemorySessionStore, str]:
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="patient", patient_id=patient_id)
    return session_store, meta.session_id


def _patient_report_card() -> dict[str, object]:
    return {
        "type": "medical_visualization_card",
        "data": {
            "patient_summary": {
                "age": "52",
                "gender": "female",
                "chief_complaint": "rectal bleeding",
            },
            "diagnosis_block": {
                "location": "rectum",
                "mmr_status": "dMMR",
            },
            "staging_block": {
                "clinical_stage": "cT3N1M0",
                "t_stage": "T3",
                "n_stage": "N1",
                "m_stage": "M0",
            },
            "key_findings": [
                {"finding": "rectal wall thickening"},
            ],
        },
    }


def _guideline_card() -> dict[str, object]:
    return {
        "type": "medical_visualization_card",
        "data": {
            "patient_summary": {
                "chief_complaint": "clinical guideline",
            },
            "key_findings": [
                {"finding": "education material"},
            ],
        },
    }


def _unknown_card() -> dict[str, object]:
    return {
        "type": "medical_visualization_card",
        "data": {},
    }


def _build_route_client(root: Path) -> TestClient:
    service = _build_service(root)
    app = FastAPI()
    app.state.runtime = SimpleNamespace(
        session_store=InMemorySessionStore(),
        assets_root=root / "assets",
        patient_registry_service=service,
    )
    app.include_router(upload_routes.router)
    return TestClient(app)


def test_guideline_upload_is_record_only_and_does_not_update_snapshot(monkeypatch) -> None:
    root = _build_root()
    registry = _build_service(root)
    patient_id = registry.create_draft_patient(created_by_session_id="sess_patient_1")
    session_store, session_id = _build_session_store(patient_id=patient_id)

    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _guideline_card(),
    )

    response = store_session_upload(
        session_store=session_store,
        patient_registry=registry,
        assets_root=root / "assets",
        session_id=session_id,
        filename="crc-guideline.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF-guideline",
    )

    detail = registry.get_patient_detail(patient_id)
    assert response["derived"]["document_type"] == "guideline_or_education"
    assert response["derived"]["ingest_decision"] == "record_only"
    assert response["derived"]["record_id"] is not None
    assert detail["chief_complaint"] is None
    assert detail["tumor_location"] is None
    assert detail["clinical_stage"] is None
    assert detail["mmr_status"] is None


def test_patient_report_upload_is_record_and_snapshot_eligible(monkeypatch) -> None:
    root = _build_root()
    registry = _build_service(root)
    patient_id = registry.create_draft_patient(created_by_session_id="sess_patient_1")
    session_store, session_id = _build_session_store(patient_id=patient_id)

    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _patient_report_card(),
    )

    response = store_session_upload(
        session_store=session_store,
        patient_registry=registry,
        assets_root=root / "assets",
        session_id=session_id,
        filename="patient-report.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF-report",
    )

    detail = registry.get_patient_detail(patient_id)
    assert response["derived"]["document_type"] == "patient_report"
    assert response["derived"]["ingest_decision"] == "record_and_snapshot"
    assert detail["chief_complaint"] == "rectal bleeding"
    assert detail["tumor_location"] == "rectum"
    assert detail["clinical_stage"] == "cT3N1M0"
    assert detail["mmr_status"] == "dMMR"


def test_parse_failed_upload_is_asset_only_and_does_not_update_snapshot(monkeypatch) -> None:
    root = _build_root()
    registry = _build_service(root)
    patient_id = registry.create_draft_patient(created_by_session_id="sess_patient_1")
    session_store, session_id = _build_session_store(patient_id=patient_id)

    def _raise_converter(**_: object) -> dict[str, object]:
        raise ValueError("parse failed")

    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        _raise_converter,
    )

    response = store_session_upload(
        session_store=session_store,
        patient_registry=registry,
        assets_root=root / "assets",
        session_id=session_id,
        filename="broken.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF-broken",
    )

    detail = registry.get_patient_detail(patient_id)
    assert response["derived"]["document_type"] == "parse_failed"
    assert response["derived"]["ingest_decision"] == "asset_only"
    assert response["derived"].get("record_id") is None
    assert detail["chief_complaint"] is None
    assert detail["tumor_location"] is None
    assert detail["clinical_stage"] is None
    assert detail["mmr_status"] is None


def test_unknown_upload_is_record_only_and_does_not_update_snapshot(monkeypatch) -> None:
    root = _build_root()
    registry = _build_service(root)
    patient_id = registry.create_draft_patient(created_by_session_id="sess_patient_1")
    session_store, session_id = _build_session_store(patient_id=patient_id)

    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _unknown_card(),
    )

    response = store_session_upload(
        session_store=session_store,
        patient_registry=registry,
        assets_root=root / "assets",
        session_id=session_id,
        filename="misc.pdf",
        content_type="application/pdf",
        file_bytes=b"%PDF-misc",
    )

    detail = registry.get_patient_detail(patient_id)
    assert response["derived"]["document_type"] == "unknown"
    assert response["derived"]["ingest_decision"] == "record_only"
    assert detail["chief_complaint"] is None
    assert detail["tumor_location"] is None
    assert detail["clinical_stage"] is None
    assert detail["mmr_status"] is None


def test_upload_route_returns_document_type_and_ingest_decision(monkeypatch) -> None:
    root = _build_root()
    client = _build_route_client(root)
    service = client.app.state.runtime.patient_registry_service
    patient_id = service.create_draft_patient(created_by_session_id="sess_patient_1")
    session_store = client.app.state.runtime.session_store
    meta = session_store.create_session(scene="patient", patient_id=patient_id)

    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _patient_report_card(),
    )

    response = client.post(
        f"/api/sessions/{meta.session_id}/uploads",
        files={"file": ("patient-report.pdf", b"%PDF-report", "application/pdf")},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["derived"]["document_type"] == "patient_report"
    assert payload["derived"]["ingest_decision"] == "record_and_snapshot"
