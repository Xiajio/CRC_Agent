from __future__ import annotations

import json
import math
from typing import get_args, get_type_hints

import pytest

from src.contracts.clinical_assertion import (
    ClinicalAssertion,
    ClinicalAssertionSource,
    ClinicalFactType,
    EvidenceRef,
    NormalizedFact,
    ReviewedStatus,
    make_assertion_id,
)
from src.services.clinical_assertion_projection import (
    project_clinical_assertions_from_records,
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


def test_normalized_fact_rejects_nan_value() -> None:
    fact = NormalizedFact(
        type="symptom",
        name="bad",
        value=math.nan,
    )

    with pytest.raises(TypeError, match="JSON"):
        fact.to_dict()


def test_normalized_fact_rejects_infinite_value() -> None:
    fact = NormalizedFact(
        type="symptom",
        name="bad",
        value=math.inf,
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


def test_clinical_assertion_literals_match_plan_contract() -> None:
    sources = set(get_args(ClinicalAssertionSource))
    fact_types = set(get_args(ClinicalFactType))
    reviewed_statuses = set(get_args(ReviewedStatus))

    assert sources == {
        "triage",
        "patient_upload",
        "doctor_note",
        "database_snapshot",
        "care_card",
        "model_draft",
    }
    assert fact_types == {
        "condition_signal",
        "symptom",
        "risk_disposition",
        "missing_information",
        "test_status",
        "safety_rule_match",
        "document_fact",
    }
    assert reviewed_statuses == {
        "unreviewed",
        "accepted",
        "edited",
        "rejected",
        "needs_evidence",
        "unsafe",
    }


def _crc_record() -> dict[str, object]:
    return {
        "record_id": 42,
        "patient_id": 33,
        "record_type": "crc_triage_assessment",
        "document_type": "crc_triage_assessment",
        "normalized_payload_json": {
            "record_type": "crc_triage_assessment",
            "assessment_id": "crc_assessment_abc123",
            "source_session_id": "sess_patient_001",
            "known_crc_signals": {"rectal_bleeding": True},
            "red_flags": ["weight_loss"],
            "disposition": "urgent_gi_clinic",
            "missing_information": ["family_history"],
            "matched_rules": ["rectal_bleeding_age_escalation"],
            "safety_policy_version": "crc_safety_policy_v0",
        },
    }


def test_project_crc_triage_record_to_assertions() -> None:
    assertions = project_clinical_assertions_from_records([_crc_record()])
    payloads = [assertion.to_dict() for assertion in assertions]
    names = {
        item["normalized_fact"]["name"]: item["normalized_fact"]["type"]
        for item in payloads
    }

    assert names["rectal_bleeding"] == "condition_signal"
    assert names["weight_loss"] == "symptom"
    assert names["disposition"] == "risk_disposition"
    assert names["family_history"] == "missing_information"
    assert names["rectal_bleeding_age_escalation"] == "safety_rule_match"
    assert all(item["source_assessment_id"] == "crc_assessment_abc123" for item in payloads)
    assert all(item["source_record_id"] == "42" for item in payloads)
    assert all(item["safety_policy_version"] == "crc_safety_policy_v0" for item in payloads)


def test_project_crc_triage_record_is_stable_and_deduped() -> None:
    first = project_clinical_assertions_from_records([_crc_record(), _crc_record()])
    second = project_clinical_assertions_from_records([_crc_record()])

    assert [item.assertion_id for item in first] == [item.assertion_id for item in second]


def test_project_old_record_without_p0_metadata_does_not_fail() -> None:
    assertions = project_clinical_assertions_from_records(
        [
            {
                "record_id": 9,
                "patient_id": 33,
                "record_type": "crc_triage_assessment",
                "normalized_payload_json": {
                    "known_crc_signals": {"rectal_bleeding": True},
                },
            }
        ]
    )

    assert len(assertions) == 1
    payload = assertions[0].to_dict()
    assert payload["source_record_id"] == "9"
    assert "source_assessment_id" not in payload


def test_project_legacy_risk_class_uses_risk_class_evidence_field() -> None:
    assertions = project_clinical_assertions_from_records(
        [
            {
                "record_id": 10,
                "patient_id": 33,
                "record_type": "crc_triage_assessment",
                "normalized_payload_json": {
                    "risk_class": "urgent_gi_clinic",
                },
            }
        ]
    )

    risk_assertion = next(
        assertion
        for assertion in assertions
        if assertion.normalized_fact.type == "risk_disposition"
    )

    assert risk_assertion.evidence_refs[0].field == "payload.risk_class"
