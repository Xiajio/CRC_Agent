from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal, TypeAlias


ClinicalAssertionSource = Literal["triage", "clinical_review", "manual"]
ClinicalFactType = Literal[
    "condition_signal",
    "clinical_fact",
    "symptom",
    "risk_disposition",
    "missing_information",
    "safety_rule_match",
]
ReviewedStatus = Literal["unreviewed", "reviewed", "accepted", "rejected"]
JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class NormalizedFact:
    type: ClinicalFactType
    name: str
    value: JsonValue

    def to_dict(self) -> dict[str, Any]:
        _validate_json_value(self.value)
        return _omit_none(
            {
                "type": self.type,
                "name": self.name,
                "value": self.value,
            }
        )


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    id: str
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "kind": self.kind,
                "id": self.id,
                "field": self.field,
            }
        )


@dataclass(frozen=True)
class ClinicalAssertion:
    assertion_id: str
    patient_id: str
    session_id: str | None
    source: ClinicalAssertionSource
    normalized_fact: NormalizedFact
    evidence_refs: list[EvidenceRef]
    confidence: str
    reviewed_status: ReviewedStatus
    safety_policy_version: str | None
    created_from_projection_version: str | None
    source_record_id: str | None = None
    source_assessment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _omit_none(
            {
                "assertion_id": self.assertion_id,
                "patient_id": self.patient_id,
                "session_id": self.session_id,
                "source": self.source,
                "source_record_id": self.source_record_id,
                "source_assessment_id": self.source_assessment_id,
                "normalized_fact": self.normalized_fact.to_dict(),
                "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
                "confidence": self.confidence,
                "reviewed_status": self.reviewed_status,
                "safety_policy_version": self.safety_policy_version,
                "created_from_projection_version": self.created_from_projection_version,
            }
        )


def make_assertion_id(
    source: ClinicalAssertionSource,
    patient_id: str | int,
    source_object_id: str,
    normalized_fact: NormalizedFact,
    evidence_refs: list[EvidenceRef],
) -> str:
    hash_payload = {
        "source": source,
        "patient_id": str(patient_id),
        "source_object_id": source_object_id,
        "normalized_fact": normalized_fact.to_dict(),
        "evidence_refs": _canonical_evidence_refs(evidence_refs),
    }
    stable_json = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
    stable_hash = hashlib.sha256(stable_json.encode("utf-8")).hexdigest()[:8]
    return f"assertion_{source}_{source_object_id}_{normalized_fact.name}_{stable_hash}"


def _omit_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _canonical_evidence_refs(evidence_refs: list[EvidenceRef]) -> list[dict[str, Any]]:
    refs = [ref.to_dict() for ref in evidence_refs]
    return sorted(
        refs,
        key=lambda ref: json.dumps(ref, sort_keys=True, separators=(",", ":")),
    )


def _validate_json_value(value: JsonValue) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("NormalizedFact.value must be JSON-safe")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("NormalizedFact.value must be JSON-safe")
            _validate_json_value(item)
        return
    raise TypeError("NormalizedFact.value must be JSON-safe")
