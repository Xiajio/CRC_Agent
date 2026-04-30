from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.upload_service import store_session_upload


def _build_root() -> Path:
    root = Path("runtime") / "test-upload-registry-intake" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def _patient_report_card() -> dict[str, object]:
    return {
        "type": "medical_visualization_card",
        "data": {
            "patient_summary": {
                "chief_complaint": "rectal bleeding",
                "age": "58 years",
                "gender": "female",
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


def _create_patient_session(
    root: Path,
) -> tuple[PatientRegistryService, PatientCommandService, InMemorySessionStore, str, int]:
    registry = PatientRegistryService(root / "patient_registry.db")
    commands = PatientCommandService(registry)
    created = commands.create_patient(created_by_session_id="sess_patient_1")
    session_store = InMemorySessionStore()
    meta = session_store.create_session(scene="patient", patient_id=created.patient_id)
    return registry, commands, session_store, meta.session_id, created.patient_id


def _store_upload(
    *,
    session_store: InMemorySessionStore,
    commands: PatientCommandService,
    assets_root: Path,
    session_id: str,
    filename: str,
    file_bytes: bytes,
) -> dict[str, object]:
    return store_session_upload(
        session_store=session_store,
        patient_commands=commands,
        assets_root=assets_root,
        session_id=session_id,
        filename=filename,
        content_type="application/pdf",
        file_bytes=file_bytes,
    )


def test_parse_failed_upload_is_asset_only_and_does_not_update_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    registry, commands, session_store, session_id, patient_id = _create_patient_session(root)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )

    _store_upload(
        session_store=session_store,
        commands=commands,
        assets_root=root / "assets",
        session_id=session_id,
        filename="broken.pdf",
        file_bytes=b"%PDF-broken",
    )

    with registry._connect() as connection:
        assets = connection.execute(
            "SELECT parse_status, parse_error_message FROM patient_assets WHERE patient_id = ?",
            (patient_id,),
        ).fetchall()
        events = connection.execute(
            "SELECT event_type FROM patient_events WHERE patient_id = ? ORDER BY patient_version",
            (patient_id,),
        ).fetchall()

    assert len(assets) == 1
    assert assets[0]["parse_status"] == "failed"
    assert "parse failed" in assets[0]["parse_error_message"]
    assert [row["event_type"] for row in events] == [
        "patient.created",
        "patient.upload_received",
        "patient.upload_parse_failed",
    ]
    assert "medical_card" not in session_store.get_session(session_id).context_state


def test_patient_report_upload_is_record_and_snapshot_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    registry, commands, session_store, session_id, patient_id = _create_patient_session(root)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _patient_report_card(),
    )

    _store_upload(
        session_store=session_store,
        commands=commands,
        assets_root=root / "assets",
        session_id=session_id,
        filename="patient-report.pdf",
        file_bytes=b"%PDF-report",
    )

    with registry._connect() as connection:
        events = connection.execute(
            "SELECT event_type FROM patient_events WHERE patient_id = ? ORDER BY patient_version",
            (patient_id,),
        ).fetchall()

    assert [row["event_type"] for row in events] == [
        "patient.created",
        "patient.upload_received",
        "patient.medical_card_extracted",
    ]
    assert "medical_card" not in session_store.get_session(session_id).context_state
