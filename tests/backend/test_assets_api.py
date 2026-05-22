from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routes import assets as asset_routes
from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.upload_service import store_session_upload


def _patient_report_card() -> dict[str, object]:
    return {
        "type": "medical_visualization_card",
        "data": {
            "patient_summary": {"chief_complaint": "rectal bleeding"},
            "diagnosis_block": {"location": "rectum", "mmr_status": "dMMR"},
            "staging_block": {"clinical_stage": "cT3N1M0"},
            "key_findings": [{"finding": "rectal wall thickening"}],
        },
    }


def _build_asset_client(monkeypatch) -> tuple[TestClient, InMemorySessionStore, PatientCommandService, Path]:
    root = Path("runtime") / "test-assets-api" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    patient_registry = PatientRegistryService(root / "patient_registry.db")
    session_store = InMemorySessionStore()
    commands = PatientCommandService(patient_registry)
    app = FastAPI()
    app.state.runtime = SimpleNamespace(
        session_store=session_store,
        assets_root=root / "assets",
        patient_command_service=commands,
    )
    app.include_router(asset_routes.router)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _patient_report_card(),
    )
    return TestClient(app), session_store, commands, root


def _upload_asset(
    *,
    session_store: InMemorySessionStore,
    commands: PatientCommandService,
    root: Path,
    filename: str,
    content: bytes,
) -> tuple[str, str]:
    patient = commands.create_patient(created_by_session_id=f"sess_{uuid4().hex}")
    meta = session_store.create_session(scene="patient", patient_id=patient.patient_id)
    upload = store_session_upload(
        session_store=session_store,
        assets_root=root / "assets",
        session_id=meta.session_id,
        filename=filename,
        content_type="application/pdf",
        file_bytes=content,
        patient_commands=commands,
    )
    return meta.session_id, str(upload["asset_id"])


def test_session_asset_route_returns_content_for_own_session(monkeypatch) -> None:
    client, session_store, commands, root = _build_asset_client(monkeypatch)
    try:
        session_id, asset_id = _upload_asset(
            session_store=session_store,
            commands=commands,
            root=root,
            filename="patient-report.pdf",
            content=b"%PDF-own-session",
        )

        response = client.get(f"/api/sessions/{session_id}/assets/{asset_id}")

        assert response.status_code == 200
        assert response.content == b"%PDF-own-session"
        assert response.headers["x-content-type-options"] == "nosniff"
    finally:
        client.close()


def test_session_asset_route_does_not_scan_other_sessions(monkeypatch) -> None:
    client, session_store, commands, root = _build_asset_client(monkeypatch)
    try:
        _, asset_id = _upload_asset(
            session_store=session_store,
            commands=commands,
            root=root,
            filename="patient-report.pdf",
            content=b"%PDF-first-session",
        )
        other_patient = commands.create_patient(created_by_session_id="sess_other")
        other_session = session_store.create_session(scene="patient", patient_id=other_patient.patient_id)

        response = client.get(f"/api/sessions/{other_session.session_id}/assets/{asset_id}")

        assert response.status_code == 404
    finally:
        client.close()


def test_global_asset_route_is_not_available(monkeypatch) -> None:
    client, session_store, commands, root = _build_asset_client(monkeypatch)
    try:
        _, asset_id = _upload_asset(
            session_store=session_store,
            commands=commands,
            root=root,
            filename="patient-report.pdf",
            content=b"%PDF-global-route",
        )

        response = client.get(f"/api/assets/{asset_id}")

        assert response.status_code == 404
    finally:
        client.close()
