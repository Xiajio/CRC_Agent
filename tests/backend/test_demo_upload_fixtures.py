from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import assets as asset_routes
from backend.api.routes import uploads as upload_routes
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore
from backend.app import BearerAuthMiddleware
from backend.api.services.settings import RuntimeSettings
from backend.api.services.upload_fixture_cards import load_fixture_upload_card


def test_demo_upload_fixture_files_are_paired():
    upload_root = Path("tests/fixtures/demo_uploads")
    card_root = Path("tests/fixtures/upload_cards")

    for stem in ("demo_colonoscopy_report", "demo_pathology_report"):
        assert (upload_root / f"{stem}.pdf").is_file()
        assert (card_root / f"{stem}.json").is_file()


def test_demo_upload_cards_load_by_uploaded_filename():
    colonoscopy_card = load_fixture_upload_card("demo_colonoscopy_report.pdf")
    pathology_card = load_fixture_upload_card("demo_pathology_report.pdf")

    assert colonoscopy_card["type"] == "medical_visualization_card"
    assert pathology_card["type"] == "medical_visualization_card"


def test_demo_profile_upload_asset_url_can_be_downloaded_without_bearer(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_CONVERTER_MODE", "fixture")
    patient_registry = PatientRegistryService(tmp_path / "patient_registry.db")
    patient_commands = PatientCommandService(patient_registry)
    session_store = InMemorySessionStore()
    patient = patient_commands.create_patient(created_by_session_id=f"sess_{uuid4().hex}")
    session = session_store.create_session(scene="patient", patient_id=patient.patient_id)

    app = FastAPI()
    app.state.runtime = SimpleNamespace(
        session_store=session_store,
        assets_root=tmp_path / "assets",
        patient_command_service=patient_commands,
    )
    app.add_middleware(
        BearerAuthMiddleware,
        settings=RuntimeSettings(auth_mode="none"),
    )
    app.include_router(upload_routes.router)
    app.include_router(asset_routes.router)

    upload_path = Path("tests/fixtures/demo_uploads/demo_colonoscopy_report.pdf")
    upload_bytes = upload_path.read_bytes()
    with TestClient(app) as client:
        upload_response = client.post(
            f"/api/sessions/{session.session_id}/uploads",
            files={"file": (upload_path.name, upload_bytes, "application/pdf")},
        )
        assert upload_response.status_code == 200
        upload_payload = upload_response.json()
        assert upload_payload["filename"] == upload_path.name
        assert upload_payload["asset_url"] == (
            f"/api/sessions/{session.session_id}/assets/{upload_payload['asset_id']}"
        )

        asset_response = client.get(upload_payload["asset_url"])

    assert asset_response.status_code == 200
    assert asset_response.content == upload_bytes
