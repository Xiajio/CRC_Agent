from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from backend.api.services.patient_commands import PatientCommandService
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.session_store import InMemorySessionStore
from backend.api.services.upload_service import UploadProcessingError, compute_file_sha256, store_session_upload


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


def test_extraction_command_failure_marks_asset_failed_and_removes_derived_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    registry, commands, session_store, session_id, patient_id = _create_patient_session(root)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: _patient_report_card(),
    )
    monkeypatch.setattr(
        commands,
        "record_medical_card_extracted",
        lambda **_: (_ for _ in ()).throw(RuntimeError("extract command failed")),
    )

    with pytest.raises(UploadProcessingError, match="extract command failed"):
        _store_upload(
            session_store=session_store,
            commands=commands,
            assets_root=root / "assets",
            session_id=session_id,
            filename="patient-report.pdf",
            file_bytes=b"%PDF-report-command-fails",
        )

    sha256 = compute_file_sha256(b"%PDF-report-command-fails")
    with registry._connect() as connection:
        asset = connection.execute(
            """
            SELECT parse_status, parse_error_code, parse_error_message
            FROM patient_assets
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchone()
        events = connection.execute(
            "SELECT event_type FROM patient_events WHERE patient_id = ? ORDER BY patient_version",
            (patient_id,),
        ).fetchall()

    assert asset["parse_status"] == "failed"
    assert asset["parse_error_code"] == "UPLOAD_COMMAND_ERROR"
    assert "extract command failed" in asset["parse_error_message"]
    assert [row["event_type"] for row in events] == [
        "patient.created",
        "patient.upload_received",
        "patient.upload_parse_failed",
    ]
    assert not (root / "assets" / str(patient_id) / sha256 / "derived" / "medical_card.json").exists()
    assert "medical_card" not in session_store.get_session(session_id).context_state


def test_parse_failed_command_failure_marks_asset_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    registry, commands, session_store, session_id, patient_id = _create_patient_session(root)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )
    monkeypatch.setattr(
        commands,
        "record_upload_parse_failed",
        lambda **_: (_ for _ in ()).throw(RuntimeError("parse command failed")),
    )

    with pytest.raises(UploadProcessingError, match="parse command failed"):
        _store_upload(
            session_store=session_store,
            commands=commands,
            assets_root=root / "assets",
            session_id=session_id,
            filename="broken.pdf",
            file_bytes=b"%PDF-parse-command-fails",
        )

    with registry._connect() as connection:
        asset = connection.execute(
            """
            SELECT parse_status, parse_error_code, parse_error_message
            FROM patient_assets
            WHERE patient_id = ?
            """,
            (patient_id,),
        ).fetchone()
        events = connection.execute(
            "SELECT event_type FROM patient_events WHERE patient_id = ? ORDER BY patient_version",
            (patient_id,),
        ).fetchall()

    assert asset["parse_status"] == "failed"
    assert asset["parse_error_code"] == "UPLOAD_COMMAND_ERROR"
    assert "parse command failed" in asset["parse_error_message"]
    assert [row["event_type"] for row in events] == [
        "patient.created",
        "patient.upload_received",
        "patient.upload_parse_failed",
    ]


def test_duplicate_parse_failure_reports_command_level_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _build_root()
    registry, commands, session_store, first_session_id, patient_id = _create_patient_session(root)
    second_meta = session_store.create_session(scene="patient", patient_id=patient_id)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_uploaded_file",
        lambda **_: (_ for _ in ()).throw(RuntimeError("parse failed")),
    )

    _store_upload(
        session_store=session_store,
        commands=commands,
        assets_root=root / "assets",
        session_id=first_session_id,
        filename="broken.pdf",
        file_bytes=b"%PDF-duplicate-broken",
    )
    second = _store_upload(
        session_store=session_store,
        commands=commands,
        assets_root=root / "assets",
        session_id=second_meta.session_id,
        filename="broken-copy.pdf",
        file_bytes=b"%PDF-duplicate-broken",
    )

    with registry._connect() as connection:
        event_count = connection.execute(
            "SELECT COUNT(*) AS count FROM patient_events WHERE patient_id = ?",
            (patient_id,),
        ).fetchone()

    assert second["reused"] is True
    assert int(event_count["count"]) == 3
