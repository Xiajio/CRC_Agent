from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from src.contracts.clinical_assertion import (
    ClinicalAssertion,
    EvidenceRef,
    NormalizedFact,
    make_assertion_id,
)


PROJECTION_VERSION = "patient_record_projection_v0"
CRC_TRIAGE_ASSESSMENT = "crc_triage_assessment"


def project_clinical_assertions_from_record(
    record: Mapping[str, Any],
) -> list[ClinicalAssertion]:
    if record.get("record_type") != CRC_TRIAGE_ASSESSMENT:
        return []

    payload = _load_payload(record.get("normalized_payload_json"))
    record_id = _record_id(record)
    patient_id = _patient_id(record)
    assertions: list[ClinicalAssertion] = []

    for name, value in _bool_signal_items(payload.get("known_crc_signals")):
        assertions.append(
            _make_assertion(
                record=record,
                payload=payload,
                patient_id=patient_id,
                record_id=record_id,
                normalized_fact=NormalizedFact(
                    type="condition_signal",
                    name=name,
                    value=value,
                ),
                field=f"payload.known_crc_signals.{name}",
                confidence="structured_user_report",
            )
        )

    for name in _string_list(payload.get("red_flags")):
        assertions.append(
            _make_assertion(
                record=record,
                payload=payload,
                patient_id=patient_id,
                record_id=record_id,
                normalized_fact=NormalizedFact(
                    type="symptom",
                    name=name,
                    value=True,
                ),
                field="payload.red_flags",
                confidence="structured_user_report",
            )
        )

    disposition = _optional_string(payload.get("disposition"))
    disposition_field = "payload.disposition"
    if disposition is None:
        disposition = _optional_string(payload.get("risk_class"))
        disposition_field = "payload.risk_class"
    if disposition is not None:
        assertions.append(
            _make_assertion(
                record=record,
                payload=payload,
                patient_id=patient_id,
                record_id=record_id,
                normalized_fact=NormalizedFact(
                    type="risk_disposition",
                    name="disposition",
                    value=disposition,
                ),
                field=disposition_field,
                confidence="deterministic_policy_or_protocol",
            )
        )

    for name in _string_list(payload.get("missing_information")):
        assertions.append(
            _make_assertion(
                record=record,
                payload=payload,
                patient_id=patient_id,
                record_id=record_id,
                normalized_fact=NormalizedFact(
                    type="missing_information",
                    name=name,
                    value=True,
                ),
                field="payload.missing_information",
                confidence="structured_user_report",
            )
        )

    for name in _string_list(payload.get("matched_rules")):
        assertions.append(
            _make_assertion(
                record=record,
                payload=payload,
                patient_id=patient_id,
                record_id=record_id,
                normalized_fact=NormalizedFact(
                    type="safety_rule_match",
                    name=name,
                    value=True,
                ),
                field="payload.matched_rules",
                confidence="deterministic_safety_policy",
            )
        )

    return assertions


def project_clinical_assertions_from_records(
    records: Iterable[Mapping[str, Any]],
) -> list[ClinicalAssertion]:
    assertions_by_id: dict[str, ClinicalAssertion] = {}
    for record in records:
        for assertion in project_clinical_assertions_from_record(record):
            assertions_by_id.setdefault(assertion.assertion_id, assertion)
    return list(assertions_by_id.values())


def assertion_refs(assertions: Iterable[ClinicalAssertion]) -> list[str]:
    return [assertion.assertion_id for assertion in assertions]


def _load_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            items.append(stripped)
    return items


def _bool_signal_items(value: Any) -> list[tuple[str, bool]]:
    if not isinstance(value, Mapping):
        return []
    items: list[tuple[str, bool]] = []
    for name, signal_value in value.items():
        if signal_value is True:
            stripped_name = str(name).strip()
            if stripped_name:
                items.append((stripped_name, True))
    return items


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("record_id")
    if value is None:
        return ""
    return str(value)


def _patient_id(record: Mapping[str, Any]) -> str:
    value = record.get("patient_id")
    if value is None:
        return ""
    return str(value)


def _make_assertion(
    *,
    record: Mapping[str, Any],
    payload: Mapping[str, Any],
    patient_id: str,
    record_id: str,
    normalized_fact: NormalizedFact,
    field: str,
    confidence: str,
) -> ClinicalAssertion:
    evidence_refs = [
        EvidenceRef(
            kind="patient_record",
            id=record_id,
            field=field,
        )
    ]
    assessment_id = _optional_string(payload.get("assessment_id"))
    source_object_id = assessment_id if assessment_id is not None else record_id
    assertion_id = make_assertion_id(
        source="triage",
        patient_id=patient_id,
        source_object_id=source_object_id,
        normalized_fact=normalized_fact,
        evidence_refs=evidence_refs,
    )

    return ClinicalAssertion(
        assertion_id=assertion_id,
        patient_id=patient_id,
        session_id=_optional_string(payload.get("source_session_id")),
        source="triage",
        source_record_id=record_id,
        source_assessment_id=assessment_id,
        normalized_fact=normalized_fact,
        evidence_refs=evidence_refs,
        confidence=confidence,
        reviewed_status="unreviewed",
        safety_policy_version=_optional_string(payload.get("safety_policy_version")),
        created_from_projection_version=PROJECTION_VERSION,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return stripped
