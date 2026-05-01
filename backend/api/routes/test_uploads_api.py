from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import uploads as upload_routes
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore


def _build_root() -> Path:
    root = Path("runtime") / "test-uploads-api" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_route_client(root: Path) -> TestClient:
    app = FastAPI()
    patient_registry = PatientRegistryService(root / "patient_registry.db")
    app.state.runtime = SimpleNamespace(
        session_store=InMemorySessionStore(),
        assets_root=root / "assets",
        patient_registry_service=patient_registry,
        patient_command_service=PatientCommandService(patient_registry),
    )
    app.include_router(upload_routes.router)
    return TestClient(app)


def _patient_report_card() -> dict[str, object]:
    return {
        "type": "medical_visualization_card",
        "data": {
            "patient_summary": {
                "chief_complaint": "rectal bleeding",
            },
            "diagnosis_block": {
                "location": "rectum",
                "mmr_status": "dMMR",
            },
            "staging_block": {
                "clinical_stage": "cT3N1M0",
            },
            "key_findings": [
                {"finding": "rectal wall thickening"},
            ],
        },
    }


def test_upload_route_accepts_file_within_limit(monkeypatch) -> None:
    root = _build_root()
    client = _build_route_client(root)
    commands = client.app.state.runtime.patient_command_service
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    session_store = client.app.state.runtime.session_store
    meta = session_store.create_session(scene="patient", patient_id=patient.patient_id)

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
    assert payload["filename"] == "patient-report.pdf"
    assert payload["size"] == len(b"%PDF-report")
    assert payload["derived"]["document_type"] == "patient_report"
    assert payload["derived"]["ingest_decision"] == "record_and_snapshot"


def test_upload_route_returns_413_when_file_exceeds_configured_max(monkeypatch) -> None:
    root = _build_root()
    client = _build_route_client(root)
    commands = client.app.state.runtime.patient_command_service
    patient = commands.create_patient(created_by_session_id="sess_patient_1")
    session_store = client.app.state.runtime.session_store
    meta = session_store.create_session(scene="patient", patient_id=patient.patient_id)

    monkeypatch.setattr(upload_routes, "MAX_UPLOAD_BYTES", 3)
    monkeypatch.setattr(upload_routes, "UPLOAD_CHUNK_SIZE", 2)
    monkeypatch.setattr(
        upload_routes,
        "store_session_upload",
        lambda **_: (_ for _ in ()).throw(AssertionError("store_session_upload should not be called")),
    )

    response = client.post(
        f"/api/sessions/{meta.session_id}/uploads",
        files={"file": ("too-large.pdf", b"1234", "application/pdf")},
    )

    refreshed = session_store.get_session(meta.session_id)
    assert response.status_code == 413
    assert response.json()["detail"] == "UPLOAD_TOO_LARGE: maximum size is 3 bytes"
    assert refreshed is not None
    assert refreshed.active_run_id is None
