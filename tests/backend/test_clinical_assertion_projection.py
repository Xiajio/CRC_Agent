from __future__ import annotations

import json
from typing import get_type_hints

import pytest

from src.contracts.clinical_assertion import (
    ClinicalAssertion,
    EvidenceRef,
    NormalizedFact,
    make_assertion_id,
)


def test_make_assertion_id_is_stable_and_content_addressed() -> None:
    fact = NormalizedFact(
        type="condition_signal",
        name="rectal_bleeding",
        value=True,
    )
    refs = [
        EvidenceRef(
            kind="patient_record",
            id="record_42",
            field="payload.known_crc_signals.rectal_bleeding",
        )
    ]

    first = make_assertion_id(
        source="triage",
        patient_id=33,
        source_object_id="crc_assessment_abc123",
        normalized_fact=fact,
        evidence_refs=refs,
    )
    second = make_assertion_id(
        source="triage",
        patient_id=33,
        source_object_id="crc_assessment_abc123",
        normalized_fact=fact,
        evidence_refs=refs,
    )

    assert first == second
    assert first.startswith("assertion_triage_crc_assessment_abc123_rectal_bleeding_")
    assert len(first.rsplit("_", 1)[-1]) == 8

    positional = make_assertion_id(
        "triage",
        33,
        "crc_assessment_abc123",
        fact,
        refs,
    )

    assert positional == first

    changed_fact = make_assertion_id(
        source="triage",
        patient_id=33,
        source_object_id="crc_assessment_abc123",
        normalized_fact=NormalizedFact(
            type="condition_signal",
            name="rectal_bleeding",
            value=False,
        ),
        evidence_refs=refs,
    )
    changed_evidence = make_assertion_id(
        source="triage",
        patient_id=33,
        source_object_id="crc_assessment_abc123",
        normalized_fact=fact,
        evidence_refs=[
            EvidenceRef(
                kind="patient_record",
                id="record_43",
                field="payload.known_crc_signals.rectal_bleeding",
            )
        ],
    )

    assert changed_fact != first
    assert changed_evidence != first


def test_make_assertion_id_canonicalizes_evidence_ref_order() -> None:
    fact = NormalizedFact(
        type="condition_signal",
        name="rectal_bleeding",
        value=True,
    )
    first_refs = [
        EvidenceRef(
            kind="patient_record",
            id="record_42",
            field="payload.known_crc_signals.rectal_bleeding",
        ),
        EvidenceRef(
            kind="patient_record",
            id="record_17",
            field="payload.known_crc_signals.family_history_crc",
        ),
    ]
    reversed_refs = list(reversed(first_refs))

    first = make_assertion_id(
        "triage",
        33,
        "crc_assessment_abc123",
        fact,
        first_refs,
    )
    second = make_assertion_id(
        "triage",
        33,
        "crc_assessment_abc123",
        fact,
        reversed_refs,
    )

    assert first == second


def test_normalized_fact_rejects_non_json_value() -> None:
    fact = NormalizedFact(
        type="condition_signal",
        name="rectal_bleeding",
        value={"not_json_safe"},
    )

    with pytest.raises(TypeError, match="JSON"):
        fact.to_dict()


def test_clinical_assertion_serializes_to_json_safe_dict() -> None:
    assertion = ClinicalAssertion(
        assertion_id="assertion_triage_record_42_rectal_bleeding_a1b2c3d4",
        patient_id="33",
        session_id="sess_patient_001",
        source="triage",
        source_record_id="record_42",
        source_assessment_id="crc_assessment_abc123",
        normalized_fact=NormalizedFact(
            type="condition_signal",
            name="rectal_bleeding",
            value=True,
        ),
        evidence_refs=[
            EvidenceRef(
                kind="patient_record",
                id="record_42",
                field="payload.known_crc_signals.rectal_bleeding",
            )
        ],
        confidence="structured_user_report",
        reviewed_status="unreviewed",
        safety_policy_version="crc_safety_policy_v0",
        created_from_projection_version="patient_record_projection_v0",
    )

    payload = assertion.to_dict()

    assert payload["assertion_id"] == "assertion_triage_record_42_rectal_bleeding_a1b2c3d4"
    assert payload["normalized_fact"] == {
        "type": "condition_signal",
        "name": "rectal_bleeding",
        "value": True,
    }
    assert payload["evidence_refs"] == [
        {
            "kind": "patient_record",
            "id": "record_42",
            "field": "payload.known_crc_signals.rectal_bleeding",
        }
    ]
    assert payload["reviewed_status"] == "unreviewed"

    json.dumps(payload, ensure_ascii=False)


def test_clinical_assertion_omits_optional_none_values() -> None:
    assertion = ClinicalAssertion(
        assertion_id="assertion_triage_record_42_rectal_bleeding_a1b2c3d4",
        patient_id="33",
        session_id=None,
        source="triage",
        normalized_fact=NormalizedFact(
            type="condition_signal",
            name="rectal_bleeding",
            value=True,
        ),
        evidence_refs=[
            EvidenceRef(
                kind="patient_record",
                id="record_42",
            )
        ],
        confidence="structured_user_report",
        reviewed_status="unreviewed",
        safety_policy_version=None,
        created_from_projection_version=None,
    )

    payload = assertion.to_dict()

    json.dumps(payload, ensure_ascii=False)
    assert "session_id" not in payload
    assert "source_record_id" not in payload
    assert "source_assessment_id" not in payload
    assert "safety_policy_version" not in payload
    assert "created_from_projection_version" not in payload
    assert payload["evidence_refs"] == [
        {
            "kind": "patient_record",
            "id": "record_42",
        }
    ]


def test_clinical_assertion_optional_fields_are_typed_as_optional() -> None:
    hints = get_type_hints(ClinicalAssertion)

    assert hints["session_id"] == str | None
    assert hints["source_record_id"] == str | None
    assert hints["source_assessment_id"] == str | None
    assert hints["safety_policy_version"] == str | None
    assert hints["created_from_projection_version"] == str | None
