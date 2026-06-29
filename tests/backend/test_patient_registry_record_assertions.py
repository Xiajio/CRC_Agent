from __future__ import annotations

import json

from backend.api.schemas.patient_registry import PatientRegistryRecord


def _record_row(
    *,
    record_type: str,
    normalized_payload_json: object,
) -> dict[str, object]:
    return {
        "record_id": 42,
        "patient_id": 33,
        "asset_id": 7,
        "record_type": record_type,
        "document_type": record_type,
        "ingest_decision": "record_only",
        "snapshot_contributed": 0,
        "conflict_detected": 0,
        "normalized_payload_json": normalized_payload_json,
        "summary_text": "summary",
        "source": "triage",
        "snapshot_meta_json": "{}",
        "created_at": "2026-06-29T00:00:00+00:00",
    }


def test_patient_registry_record_from_row_derives_crc_assertions_from_json_payload() -> None:
    record = PatientRegistryRecord.from_row(
        _record_row(
            record_type="crc_triage_assessment",
            normalized_payload_json=json.dumps(
                {
                    "known_crc_signals": {"rectal_bleeding": True},
                }
            ),
        )
    )

    assert record.clinical_assertions[0]["normalized_fact"] == {
        "type": "condition_signal",
        "name": "rectal_bleeding",
        "value": True,
    }
    assert record.clinical_assertion_refs == [
        assertion["assertion_id"] for assertion in record.clinical_assertions
    ]


def test_patient_registry_record_from_row_uses_empty_assertion_arrays_for_non_crc_records() -> None:
    record = PatientRegistryRecord.from_row(
        _record_row(
            record_type="medical_card",
            normalized_payload_json={"known_crc_signals": {"rectal_bleeding": True}},
        )
    )

    assert record.clinical_assertions == []
    assert record.clinical_assertion_refs == []
