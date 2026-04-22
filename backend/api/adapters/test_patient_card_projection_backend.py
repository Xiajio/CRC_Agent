from __future__ import annotations

import gc
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from backend.api.adapters.state_snapshot import build_recovery_snapshot
from backend.api.services.patient_registry_service import PatientRegistryService
from backend.api.services.payload_builder import build_graph_payload
from backend.api.services.session_store import InMemorySessionStore, SessionMeta
from backend.api.services.upload_service import UploadProcessingError, store_session_upload
from src.services.patient_card_projector import (
    project_patient_card,
    project_patient_card_for_updates,
    project_patient_self_report_card,
)
from src.state import CRCAgentState, PatientProfile


def _nested_meta(card: dict, section: str, field: str) -> dict:
    return card["field_meta"][section][field]


def test_project_patient_card_emits_full_spec_skeleton_and_metadata() -> None:
    card = project_patient_card(
        patient_id="current",
        patient_profile=PatientProfile(
            age=52,
            gender="female",
            chief_complaint="Rectal bleeding",
            pathology_confirmed=True,
            mmr_status="pMMR",
            ecog_score=1,
        ),
        findings={
            "tumor_location": "rectum",
            "tnm_staging": {"cT": "cT2", "cN": "cN0", "cM": "cM0"},
            "cea_level": 3.2,
            "family_history": True,
            "family_history_details": "father had CRC",
            "risk_factors": ["smoking", "obesity"],
            "biopsy_details": "adenocarcinoma",
        },
        symptom_snapshot={"duration": "2 weeks"},
        medical_card=None,
    )

    assert card["type"] == "patient_card"
    assert card["patient_id"] == "current"
    assert card["card_meta"]["source_mode"] == "patient_self_report"
    assert card["card_meta"]["projection_version"]
    assert card["card_meta"]["completion_ratio"] > 0
    assert card["card_meta"]["conflict_count"] == 0
    assert card["data"] == {
        "patient_info": {
            "gender": "female",
            "age": 52,
            "ecog": 1,
            "cea": 3.2,
        },
        "diagnosis_block": {
            "confirmed": True,
            "primary_site": "rectum",
            "mmr_status": "pMMR",
        },
        "staging_block": {
            "clinical_stage": None,
            "ct_stage": "cT2",
            "cn_stage": "cN0",
            "cm_stage": "cM0",
        },
        "history_block": {
            "chief_complaint": "Rectal bleeding",
            "symptom_duration": "2 weeks",
            "family_history": True,
            "family_history_details": "father had CRC",
            "biopsy_confirmed": True,
            "biopsy_details": "adenocarcinoma",
            "risk_factors": ["obesity", "smoking"],
        },
    }
    assert _nested_meta(card, "patient_info", "age") == {"status": "confirmed", "display": "52"}
    assert _nested_meta(card, "history_block", "symptom_duration") == {"status": "confirmed", "display": "2 weeks"}
    assert card["source_candidates"]["patient_info.age"][0]["source_type"] == "patient_profile"
    assert card["source_candidates"]["patient_info.age"][0]["source_path"] == "patient_profile.age"


def test_project_patient_card_conflicts_serialize_null_and_chinese_display() -> None:
    card = project_patient_card(
        patient_profile=PatientProfile(age=52),
        findings={"family_history": True},
        symptom_snapshot={},
        medical_card={
            "data": {
                "patient_summary": {"age": 61},
                "history_block": {"family_history": False},
            }
        },
        patient_id="current",
    )

    assert card["data"]["patient_info"]["age"] is None
    assert card["data"]["history_block"]["family_history"] is None
    assert _nested_meta(card, "patient_info", "age") == {"status": "conflict", "display": "待确认（来源不一致）"}
    assert _nested_meta(card, "history_block", "family_history") == {"status": "conflict", "display": "待确认（来源不一致）"}
    assert card["card_meta"]["conflict_count"] >= 2


def test_project_patient_card_pending_fields_use_chinese_display() -> None:
    card = project_patient_card(
        patient_profile=PatientProfile(age=None, gender=None, chief_complaint=None, mmr_status="Unknown"),
        findings={"tumor_location": "Unknown"},
        symptom_snapshot={},
        medical_card=None,
        patient_id="current",
    )

    assert card["data"]["patient_info"]["age"] is None
    assert card["data"]["diagnosis_block"]["primary_site"] is None
    assert _nested_meta(card, "patient_info", "age") == {"status": "pending", "display": "待确认"}
    assert _nested_meta(card, "diagnosis_block", "primary_site") == {"status": "pending", "display": "待确认"}


def test_project_patient_card_normalizes_lists_and_equivalent_values() -> None:
    card = project_patient_card(
        patient_profile=PatientProfile(chief_complaint="  Bleeding "),
        findings={"risk_factors": ["Smoking", "obesity", "smoking"]},
        symptom_snapshot={},
        medical_card={
            "data": {
                "patient_summary": {"chief_complaint": "bleeding"},
                "history_block": {"risk_factors": ["obesity", "smoking"]},
            }
        },
        patient_id="current",
    )
    update_payload = project_patient_card_for_updates(
        patient_profile=PatientProfile(chief_complaint="Bleeding"),
        findings={"risk_factors": ["smoking", "obesity"]},
        symptom_snapshot={},
        medical_card=None,
    )

    assert card["data"]["history_block"]["risk_factors"] == ["obesity", "smoking"]
    assert _nested_meta(card, "history_block", "chief_complaint")["status"] == "confirmed"
    assert _nested_meta(card, "history_block", "risk_factors")["status"] == "confirmed"
    assert update_payload["data"]["history_block"]["risk_factors"] == ["obesity", "smoking"]


def test_project_patient_card_prefers_latest_triage_snapshot_for_self_report_fields() -> None:
    card = project_patient_card(
        patient_profile=PatientProfile(chief_complaint="Old complaint"),
        findings={"risk_factors": ["smoking"]},
        symptom_snapshot={"chief_symptoms": "New triage complaint", "duration": "3 days"},
        medical_card={
            "data": {
                "patient_summary": {"chief_complaint": "Older upload complaint"},
                "history_block": {"symptom_duration": "2 months"},
            }
        },
        patient_id="current",
    )

    assert card["data"]["history_block"]["chief_complaint"] == "New triage complaint"
    assert card["data"]["history_block"]["symptom_duration"] == "3 days"
    assert _nested_meta(card, "history_block", "chief_complaint") == {"status": "confirmed", "display": "New triage complaint"}


def test_build_graph_payload_prefers_fresher_context_state_medical_card() -> None:
    session_meta = SessionMeta(
        session_id="sess_test",
        thread_id="thread_test",
        scene="patient",
        context_state={"medical_card": {"type": "medical_card", "version": "fresh"}},
    )
    state_snapshot = {"medical_card": {"type": "medical_card", "version": "stale"}}

    prepared = build_graph_payload(
        chat_request={"message": HumanMessage(content="hello")},
        session_meta=session_meta,
        state_snapshot=state_snapshot,
    )

    assert prepared.payload["medical_card"] == {"type": "medical_card", "version": "fresh"}


def test_store_session_upload_persists_medical_card_into_context_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_root = Path("backend/api/adapters/.tmp_patient_card_projection") / uuid4().hex
    scratch_root.mkdir(parents=True, exist_ok=False)
    session_store = InMemorySessionStore()
    session_meta = session_store.create_session(scene="patient", patient_id=None)
    registry = PatientRegistryService(scratch_root / "registry.sqlite3")
    patient_id = registry.create_draft_patient(created_by_session_id=session_meta.session_id)
    session_store.set_patient_id(session_meta.session_id, patient_id)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_upload_to_medical_card",
        lambda *_args, **_kwargs: {
            "type": "medical_card",
            "data": {
                "patient_summary": {"age": 61, "gender": "male"},
                "diagnosis_block": {"location": "colon", "mmr_status": "pMMR"},
                "staging_block": {"clinical_stage": "III", "t_stage": "T3", "n_stage": "N1", "m_stage": "M0"},
                "history_block": {"risk_factors": ["smoking"]},
            },
        },
    )

    try:
        store_session_upload(
            session_store=session_store,
            patient_registry=registry,
            assets_root=scratch_root / "assets",
            session_id=session_meta.session_id,
            filename="report.pdf",
            content_type="application/pdf",
            file_bytes=b"pdf",
        )

        saved = session_store.get_session(session_meta.session_id).context_state["medical_card"]
        assert saved["type"] == "medical_card"
        assert saved["data"]["patient_summary"]["age"] == 61
    finally:
        del registry
        gc.collect()
        shutil.rmtree(scratch_root, ignore_errors=True)


def test_store_session_upload_rolls_back_context_medical_card_when_registry_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_root = Path("backend/api/adapters/.tmp_patient_card_projection") / uuid4().hex
    scratch_root.mkdir(parents=True, exist_ok=False)
    session_store = InMemorySessionStore()
    session_meta = session_store.create_session(scene="patient", patient_id=None)
    registry = PatientRegistryService(scratch_root / "registry.sqlite3")
    patient_id = registry.create_draft_patient(created_by_session_id=session_meta.session_id)
    session_store.set_patient_id(session_meta.session_id, patient_id)
    monkeypatch.setattr(
        "backend.api.services.upload_service.convert_upload_to_medical_card",
        lambda *_args, **_kwargs: {
            "type": "medical_card",
            "data": {
                "patient_summary": {"age": 61, "gender": "male"},
                "diagnosis_block": {"location": "colon", "mmr_status": "pMMR"},
                "staging_block": {"clinical_stage": "III", "t_stage": "T3", "n_stage": "N1", "m_stage": "M0"},
            },
        },
    )
    monkeypatch.setattr(
        registry,
        "write_medical_card_record",
        lambda **_: (_ for _ in ()).throw(RuntimeError("registry write failed")),
    )

    try:
        with pytest.raises(UploadProcessingError, match="registry write failed"):
            store_session_upload(
                session_store=session_store,
                patient_registry=registry,
                assets_root=scratch_root / "assets",
                session_id=session_meta.session_id,
                filename="report.pdf",
                content_type="application/pdf",
                file_bytes=b"pdf",
            )

        refreshed = session_store.get_session(session_meta.session_id)
        assert refreshed is not None
        assert "medical_card" not in refreshed.context_state
    finally:
        del registry
        gc.collect()
        shutil.rmtree(scratch_root, ignore_errors=True)


def test_recovery_snapshot_uses_fresh_context_medical_card_when_recomputing() -> None:
    session_meta = SessionMeta(
        session_id="sess_test",
        thread_id="thread_test",
        scene="patient",
        context_state={"medical_card": {"data": {"patient_summary": {"age": 64}}}},
    )
    state = CRCAgentState(
        messages=[AIMessage(content="latest answer")],
        patient_profile=PatientProfile(age=44, gender="female", chief_complaint="abdominal pain"),
        medical_card={"data": {"patient_summary": {"age": 31}}},
        findings={"encounter_track": "patient_assessment", "tumor_location": "colon"},
    )

    snapshot = build_recovery_snapshot(session_meta, state.model_dump())

    patient_cards = [card for card in snapshot.cards if card.card_type == "patient_card" and card.payload.get("card_meta", {}).get("source_mode") == "patient_self_report"]
    assert len(patient_cards) == 1
    assert patient_cards[0].payload["data"]["patient_info"]["age"] is None
    assert patient_cards[0].payload["source_candidates"]["patient_info.age"][1]["value"] == 64


def test_recovery_snapshot_replaces_only_self_report_card_and_preserves_patient_scene_legacy_card() -> None:
    session_meta = SessionMeta(session_id="sess_test", thread_id="thread_test", scene="patient")
    stale_self_report = {
        "type": "patient_card",
        "card_meta": {"source_mode": "patient_self_report"},
        "data": {"patient_info": {"age": 30}},
    }
    legacy_database_card = {
        "type": "patient_card",
        "patient_id": 9,
        "data": {"patient_info": {"age": 68}},
        "text_summary": "legacy database card",
    }
    state = CRCAgentState(
        messages=[
            AIMessage(content="self report", additional_kwargs={"patient_card": stale_self_report}),
            AIMessage(content="legacy", additional_kwargs={"patient_card": legacy_database_card}),
        ],
        patient_profile=PatientProfile(age=45, gender="female", chief_complaint="updated complaint"),
        findings={"encounter_track": "patient_assessment"},
    )

    snapshot = build_recovery_snapshot(session_meta, state.model_dump())

    patient_cards = [card.payload for card in snapshot.cards if card.card_type == "patient_card"]
    assert len(patient_cards) == 2
    assert any(card.get("card_meta", {}).get("source_mode") == "patient_self_report" and card["data"]["patient_info"]["age"] == 45 for card in patient_cards)
    assert legacy_database_card in patient_cards


def test_recovery_snapshot_preserves_legacy_doctor_scene_patient_card_payload() -> None:
    session_meta = SessionMeta(session_id="sess_test", thread_id="thread_test", scene="doctor")
    legacy_card = {
        "type": "patient_card",
        "patient_id": 7,
        "data": {"patient_info": {"age": 68}},
        "text_summary": "legacy database card",
    }
    state = {"messages": [AIMessage(content="db result", additional_kwargs={"patient_card": legacy_card})]}

    snapshot = build_recovery_snapshot(session_meta, state)

    patient_cards = [card.payload for card in snapshot.cards if card.card_type == "patient_card"]
    assert patient_cards == [legacy_card]


def test_project_patient_self_report_card_uses_state_wrapper() -> None:
    state = CRCAgentState(
        patient_profile=PatientProfile(age=52, gender="female", chief_complaint="Rectal bleeding"),
        findings={"tumor_location": "rectum"},
        symptom_snapshot={"duration": "2 weeks"},
    )

    card = project_patient_self_report_card(state)

    assert card["data"]["patient_info"]["age"] == 52
