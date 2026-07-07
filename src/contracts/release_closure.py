from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Literal, TypeAlias


JsonValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)

GENESIS_CLOSURE_EVENT_HASH = "sha256:GENESIS"

ClosureStatus = Literal["accepted", "accepted_with_observations", "rolled_back"]
ClosureGateStatus = Literal[
    "idle",
    "ready_to_close",
    "blocked",
    "closed",
    "rolled_back_closed",
]
ClosureGateCheckStatus = Literal["pass", "warning", "fail"]
ClosureAuditEventType = Literal[
    "closure_recorded",
    "evidence_package_generated",
    "closure_read",
]

CLOSURE_STATUSES: tuple[ClosureStatus, ...] = (
    "accepted",
    "accepted_with_observations",
    "rolled_back",
)
CLOSURE_GATE_STATUSES: tuple[ClosureGateStatus, ...] = (
    "idle",
    "ready_to_close",
    "blocked",
    "closed",
    "rolled_back_closed",
)
CLOSURE_GATE_CHECK_STATUSES: tuple[ClosureGateCheckStatus, ...] = (
    "pass",
    "warning",
    "fail",
)
CLOSURE_EVENT_TYPES: tuple[ClosureAuditEventType, ...] = (
    "closure_recorded",
    "evidence_package_generated",
    "closure_read",
)
FORBIDDEN_CLOSURE_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "bearer_token",
        "chain_of_thought",
        "cookie",
        "credential",
        "credentials",
        "deployment_credential",
        "deployment_credentials",
        "doctor_note_text",
        "hidden_reasoning",
        "medical_record_number",
        "mrn",
        "password",
        "patient_id",
        "patient_identifier",
        "patient_identifiers",
        "patient_name",
        "patient_record",
        "patient_records",
        "private_key",
        "prompt",
        "raw_patient_identifier",
        "raw_patient_identifiers",
        "refresh_token",
        "secret",
        "session_token",
        "token",
        "transcript",
        "transcripts",
    }
)
_SENSITIVE_PAYLOAD_TOKENS = frozenset(
    {
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "prompt",
        "secret",
        "token",
        "tokens",
    }
)
_PATIENT_CONTEXT_TOKENS = frozenset({"patient", "patients"})
_PATIENT_IDENTIFIER_TOKENS = frozenset(
    {
        "id",
        "ids",
        "identifier",
        "identifiers",
        "medical",
        "mrn",
        "mrns",
        "name",
        "names",
        "number",
        "numbers",
        "record",
        "records",
    }
)
_PATIENT_IDENTIFIER_SUFFIXES = frozenset(
    {
        "id",
        "ids",
        "identifier",
        "identifiers",
        "medicalrecordnumber",
        "medicalrecordnumbers",
        "mrn",
        "mrns",
        "name",
        "names",
        "number",
        "numbers",
        "record",
        "records",
    }
)


@dataclass(frozen=True)
class ReleaseClosureRecord:
    closure_id: str
    intent_id: str
    release_execution_id: str
    rollback_execution_id: str | None
    closure_status: ClosureStatus
    closed_by: str
    closed_at: str
    rationale: str
    monitoring_snapshot_hash: str
    dashboard_snapshot_hash: str
    governance_snapshot_hash: str
    execution_snapshot_hash: str
    required_check_ids: list[str]
    acknowledged_alert_ids: list[str]
    unresolved_alert_ids: list[str]
    rollback_trigger_candidate_id: str | None
    evidence_package_id: str
    idempotency_key: str

    def __post_init__(self) -> None:
        _require_non_empty("closure_id", self.closure_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("release_execution_id", self.release_execution_id)
        _require_optional_string("rollback_execution_id", self.rollback_execution_id)
        _validate_choice("closure_status", self.closure_status, CLOSURE_STATUSES)
        _require_non_empty("closed_by", self.closed_by)
        _require_non_empty("closed_at", self.closed_at)
        _require_non_empty("rationale", self.rationale)
        _require_hash("monitoring_snapshot_hash", self.monitoring_snapshot_hash)
        _require_hash("dashboard_snapshot_hash", self.dashboard_snapshot_hash)
        _require_hash("governance_snapshot_hash", self.governance_snapshot_hash)
        _require_hash("execution_snapshot_hash", self.execution_snapshot_hash)
        _require_optional_string(
            "rollback_trigger_candidate_id", self.rollback_trigger_candidate_id
        )
        _require_non_empty("evidence_package_id", self.evidence_package_id)
        _require_non_empty("idempotency_key", self.idempotency_key)
        object.__setattr__(
            self,
            "required_check_ids",
            tuple(_require_string_list("required_check_ids", self.required_check_ids)),
        )
        object.__setattr__(
            self,
            "acknowledged_alert_ids",
            tuple(
                _require_string_list(
                    "acknowledged_alert_ids", self.acknowledged_alert_ids
                )
            ),
        )
        object.__setattr__(
            self,
            "unresolved_alert_ids",
            tuple(
                _require_string_list(
                    "unresolved_alert_ids", self.unresolved_alert_ids
                )
            ),
        )
        if self.closure_status == "accepted":
            if self.unresolved_alert_ids:
                raise ValueError(
                    "accepted closure cannot contain unresolved alerts"
                )
            if self.rollback_trigger_candidate_id is not None:
                raise ValueError(
                    "accepted closure cannot contain rollback trigger candidate"
                )
        if self.closure_status == "rolled_back":
            _require_required_string(
                "rollback_execution_id", self.rollback_execution_id
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "closure_id": self.closure_id,
            "intent_id": self.intent_id,
            "release_execution_id": self.release_execution_id,
            "rollback_execution_id": self.rollback_execution_id,
            "closure_status": self.closure_status,
            "closed_by": self.closed_by,
            "closed_at": self.closed_at,
            "rationale": self.rationale,
            "monitoring_snapshot_hash": self.monitoring_snapshot_hash,
            "dashboard_snapshot_hash": self.dashboard_snapshot_hash,
            "governance_snapshot_hash": self.governance_snapshot_hash,
            "execution_snapshot_hash": self.execution_snapshot_hash,
            "required_check_ids": list(self.required_check_ids),
            "acknowledged_alert_ids": list(self.acknowledged_alert_ids),
            "unresolved_alert_ids": list(self.unresolved_alert_ids),
            "rollback_trigger_candidate_id": self.rollback_trigger_candidate_id,
            "evidence_package_id": self.evidence_package_id,
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class ReleaseEvidencePackage:
    package_id: str
    closure_id: str
    intent_id: str
    release_execution_id: str
    rollback_execution_id: str | None
    generated_by: str
    generated_at: str
    closure_status: ClosureStatus
    summary: str
    source_refs: list[str]
    artifact_refs: list[str]
    snapshot_hashes: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _require_non_empty("package_id", self.package_id)
        _require_non_empty("closure_id", self.closure_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("release_execution_id", self.release_execution_id)
        _require_optional_string("rollback_execution_id", self.rollback_execution_id)
        _require_non_empty("generated_by", self.generated_by)
        _require_non_empty("generated_at", self.generated_at)
        _validate_choice("closure_status", self.closure_status, CLOSURE_STATUSES)
        _require_non_empty("summary", self.summary)
        object.__setattr__(
            self,
            "source_refs",
            tuple(_require_string_list("source_refs", self.source_refs)),
        )
        object.__setattr__(
            self,
            "artifact_refs",
            tuple(_validate_artifact_refs("artifact_refs", self.artifact_refs)),
        )
        if type(self.snapshot_hashes) is not dict:
            raise TypeError("snapshot_hashes must be a dictionary")
        snapshot_hashes = _copy_json_safe(self.snapshot_hashes, path="snapshot_hashes")
        _reject_forbidden_payload_keys(snapshot_hashes)
        _validate_snapshot_hashes(snapshot_hashes, path="snapshot_hashes")
        object.__setattr__(
            self,
            "snapshot_hashes",
            _freeze_json_safe(snapshot_hashes, path="snapshot_hashes"),
        )
        if self.closure_status == "rolled_back":
            _require_required_string(
                "rollback_execution_id", self.rollback_execution_id
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "closure_id": self.closure_id,
            "intent_id": self.intent_id,
            "release_execution_id": self.release_execution_id,
            "rollback_execution_id": self.rollback_execution_id,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "closure_status": self.closure_status,
            "summary": self.summary,
            "source_refs": list(self.source_refs),
            "artifact_refs": list(self.artifact_refs),
            "snapshot_hashes": _copy_frozen_json_safe(
                self.snapshot_hashes,
                path="snapshot_hashes",
            ),
        }


@dataclass(frozen=True)
class ReleaseClosureGateCheck:
    name: str
    status: ClosureGateCheckStatus
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _validate_choice("status", self.status, CLOSURE_GATE_CHECK_STATUSES)
        _require_non_empty("reason", self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReleaseClosureGate:
    allowed: bool
    status: ClosureGateStatus
    reasons: list[str]
    checks: list[ReleaseClosureGateCheck]

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be bool")
        _validate_choice("status", self.status, CLOSURE_GATE_STATUSES)
        object.__setattr__(
            self,
            "reasons",
            tuple(_require_string_list("reasons", self.reasons)),
        )
        object.__setattr__(
            self,
            "checks",
            tuple(_require_gate_checks("checks", self.checks)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reasons": list(self.reasons),
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class ReleaseClosureAuditEvent:
    event_id: str
    intent_id: str
    release_execution_id: str
    event_type: ClosureAuditEventType
    actor: str
    timestamp: str
    payload_hash: str
    previous_event_hash: str
    event_hash: str

    def __post_init__(self) -> None:
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("release_execution_id", self.release_execution_id)
        _validate_choice("event_type", self.event_type, CLOSURE_EVENT_TYPES)
        _require_non_empty("actor", self.actor)
        _require_non_empty("timestamp", self.timestamp)
        _require_hash("payload_hash", self.payload_hash)
        _require_hash(
            "previous_event_hash",
            self.previous_event_hash,
            allow_genesis=True,
        )
        _require_hash("event_hash", self.event_hash)
        validate_release_closure_audit_event_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "intent_id": self.intent_id,
            "release_execution_id": self.release_execution_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "payload_hash": self.payload_hash,
            "previous_event_hash": self.previous_event_hash,
            "event_hash": self.event_hash,
        }


def make_release_closure_id(
    release_execution_id: str,
    idempotency_key: str,
) -> str:
    _require_non_empty("release_execution_id", release_execution_id)
    _require_non_empty("idempotency_key", idempotency_key)
    payload = {
        "release_execution_id": release_execution_id,
        "idempotency_key": idempotency_key,
    }
    return f"release_closure_{_slug(release_execution_id)}_{_stable_hash(payload)}"


def make_release_evidence_package_id(closure_id: str) -> str:
    _require_non_empty("closure_id", closure_id)
    payload = {"closure_id": closure_id}
    return f"release_evidence_package_{_slug(closure_id)}_{_stable_hash(payload)}"


def make_release_closure_event_id(
    release_execution_id: str,
    event_type: str,
    timestamp: str,
) -> str:
    _require_non_empty("release_execution_id", release_execution_id)
    _validate_choice("event_type", event_type, CLOSURE_EVENT_TYPES)
    _require_non_empty("timestamp", timestamp)
    payload = {
        "release_execution_id": release_execution_id,
        "event_type": event_type,
        "timestamp": timestamp,
    }
    return f"release_closure_audit_{_slug(event_type)}_{_stable_hash(payload)}"


def canonical_closure_payload_hash(payload: JsonValue) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    _reject_forbidden_payload_values(payload_copy)
    stable_json = json.dumps(
        payload_copy,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()}"


def build_release_closure_audit_event(
    *,
    event_id: str,
    intent_id: str,
    release_execution_id: str,
    event_type: ClosureAuditEventType,
    actor: str,
    timestamp: str,
    payload: JsonValue,
    previous_event_hash: str,
) -> ReleaseClosureAuditEvent:
    payload_hash = canonical_closure_payload_hash(payload)
    event_payload: dict[str, JsonValue] = {
        "event_id": event_id,
        "intent_id": intent_id,
        "release_execution_id": release_execution_id,
        "event_type": event_type,
        "actor": actor,
        "timestamp": timestamp,
        "payload_hash": payload_hash,
        "previous_event_hash": previous_event_hash,
    }
    event_hash = canonical_closure_payload_hash(event_payload)
    return ReleaseClosureAuditEvent(
        event_id=event_id,
        intent_id=intent_id,
        release_execution_id=release_execution_id,
        event_type=event_type,
        actor=actor,
        timestamp=timestamp,
        payload_hash=payload_hash,
        previous_event_hash=previous_event_hash,
        event_hash=event_hash,
    )


def validate_release_closure_audit_event_hash(
    event: ReleaseClosureAuditEvent,
) -> bool:
    if not isinstance(event, ReleaseClosureAuditEvent):
        raise TypeError("event must be ReleaseClosureAuditEvent")
    event_payload = event.to_dict()
    event_hash = event_payload.pop("event_hash")
    expected_event_hash = canonical_closure_payload_hash(event_payload)
    if event_hash != expected_event_hash:
        raise ValueError(
            "event_hash does not match canonical closure audit event payload"
        )
    return True


def validate_json_safe(value: JsonValue, *, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError(f"{path} must be JSON-safe")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} must contain only string keys")
            validate_json_safe(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} must be JSON-safe")


def _copy_json_safe(value: JsonValue, *, path: str) -> JsonValue:
    validate_json_safe(value, path=path)
    if isinstance(value, list):
        return [
            _copy_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _copy_json_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    return value


def _freeze_json_safe(value: JsonValue, *, path: str) -> object:
    validate_json_safe(value, path=path)
    if isinstance(value, list):
        return tuple(
            _freeze_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        return MappingProxyType(
            {
                key: _freeze_json_safe(item, path=f"{path}.{key}")
                for key, item in value.items()
            }
        )
    return value


def _copy_frozen_json_safe(value: object, *, path: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError(f"{path} must be JSON-safe")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [
            _copy_frozen_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, MappingProxyType):
        return {
            key: _copy_frozen_json_safe(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    raise TypeError(f"{path} must be JSON-safe")


def _validate_artifact_refs(field_name: str, value: list[str]) -> list[str]:
    refs = _require_string_list(field_name, value)
    for ref in refs:
        normalized_ref = ref.replace("\\", "/")
        windows_path = PureWindowsPath(ref)
        if (
            PurePosixPath(ref).is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or "://" in normalized_ref
        ):
            raise ValueError(f"{field_name} must be repo-relative")
        parts = normalized_ref.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"{field_name} must be repo-relative")
        if not normalized_ref.startswith("reports/release_closure/"):
            raise ValueError(
                "artifact_refs must be under reports/release_closure"
            )
    return refs


def _validate_snapshot_hashes(
    value: dict[str, JsonValue],
    *,
    path: str,
) -> None:
    for key, item in value.items():
        item_path = f"{path}.{key}"
        if not isinstance(item, str):
            raise ValueError(
                "snapshot_hashes values must be direct sha256 hash strings"
            )
        _require_hash(item_path, item)


def _reject_forbidden_payload_values(
    value: JsonValue,
    *,
    path: str = "payload",
) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_payload_values(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_forbidden_payload_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, str):
        marker = _find_forbidden_payload_value_marker(value)
        if marker is not None:
            raise ValueError(
                f"payload contains forbidden content at {path}: {marker}"
            )


def _find_forbidden_payload_value_marker(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", value).strip().lower()
    patterns = (
        (r"\bbearer\b", "bearer"),
        (r"\bapi[\s_-]*key\b", "api key"),
        (r"\bapikey\b", "apikey"),
        (r"\bhidden[\s_-]*reasoning\b", "hidden reasoning"),
        (r"\bchain[\s_-]*of[\s_-]*thought\b", "chain of thought"),
        (r"\bprompt\b", "prompt"),
        (r"\bsession[\s_-]*transcript\b", "session transcript"),
        (r"\btranscript\b", "transcript"),
        (
            r"\braw[\s_-]*patient[\s_-]*identifier(?:s)?\b",
            "raw patient identifier",
        ),
    )
    for pattern, marker in patterns:
        if re.search(pattern, normalized) is not None:
            return marker
    return None


def _require_gate_checks(
    field_name: str,
    value: list[ReleaseClosureGateCheck | dict[str, Any]],
) -> list[ReleaseClosureGateCheck]:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list")
    checks: list[ReleaseClosureGateCheck] = []
    for item in value:
        if isinstance(item, ReleaseClosureGateCheck):
            checks.append(item)
            continue
        if type(item) is dict:
            checks.append(ReleaseClosureGateCheck(**item))
            continue
        raise TypeError(f"{field_name} must contain ReleaseClosureGateCheck")
    return checks


def _require_string_list(field_name: str, value: list[str]) -> list[str]:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list")
    for item in value:
        _require_non_empty(field_name, item)
    return list(value)


def _stable_hash(payload: dict[str, JsonValue]) -> str:
    validate_json_safe(payload)
    stable_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_json.encode("utf-8")).hexdigest()[:8]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return slug or "unknown"


def _require_non_empty(field_name: str, value: str | None) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_required_string(field_name: str, value: str | None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")


def _require_optional_string(field_name: str, value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")


def _validate_choice(
    field_name: str,
    value: str,
    allowed_values: tuple[str, ...],
) -> None:
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{field_name} must be one of: {allowed}")


def _require_hash(
    field_name: str,
    value: str,
    *,
    allow_genesis: bool = False,
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if allow_genesis and value == GENESIS_CLOSURE_EVENT_HASH:
        return
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a sha256 hash")


def _reject_forbidden_payload_keys(
    value: JsonValue,
    *,
    ancestor_key_tokens: tuple[tuple[str, ...], ...] = (),
) -> None:
    if isinstance(value, list):
        for item in value:
            _reject_forbidden_payload_keys(
                item,
                ancestor_key_tokens=ancestor_key_tokens,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = _normalize_payload_key(key)
            key_tokens = tuple(normalized_key.split("_")) if normalized_key else ()
            if _is_forbidden_payload_key(
                normalized_key,
                key_tokens,
                ancestor_key_tokens=ancestor_key_tokens,
            ):
                raise ValueError(f"payload contains forbidden key: {key}")
            _reject_forbidden_payload_keys(
                item,
                ancestor_key_tokens=ancestor_key_tokens + (key_tokens,),
            )


def _is_forbidden_payload_key(
    normalized_key: str,
    key_tokens: tuple[str, ...],
    *,
    ancestor_key_tokens: tuple[tuple[str, ...], ...],
) -> bool:
    token_set = set(key_tokens)
    path_token_set = set(key_tokens)
    for ancestor_tokens in ancestor_key_tokens:
        path_token_set.update(ancestor_tokens)
    compact_key = normalized_key.replace("_", "")
    if normalized_key in FORBIDDEN_CLOSURE_PAYLOAD_KEYS:
        return True
    if "apikey" in compact_key:
        return True
    if {"api", "key"}.issubset(path_token_set):
        return True
    if {"private", "key"}.issubset(path_token_set):
        return True
    if {"deployment", "credential"}.issubset(path_token_set):
        return True
    if {"deployment", "credentials"}.issubset(path_token_set):
        return True
    if {"hidden", "reasoning"}.issubset(path_token_set):
        return True
    if {"chain", "of", "thought"}.issubset(path_token_set):
        return True
    if token_set & _SENSITIVE_PAYLOAD_TOKENS:
        return True
    if _is_patient_identifier_compound(compact_key):
        return True
    if (
        token_set & _PATIENT_CONTEXT_TOKENS
        and token_set & _PATIENT_IDENTIFIER_TOKENS
    ):
        return True
    if any(
        set(tokens) & _PATIENT_CONTEXT_TOKENS for tokens in ancestor_key_tokens
    ):
        return bool(token_set & _PATIENT_IDENTIFIER_TOKENS)
    return False


def _is_patient_identifier_compound(compact_key: str) -> bool:
    for patient_prefix in ("patients", "patient"):
        if compact_key.startswith(patient_prefix):
            suffix = compact_key.removeprefix(patient_prefix)
            return suffix in _PATIENT_IDENTIFIER_SUFFIXES
    return False


def _normalize_payload_key(key: str) -> str:
    key_with_word_boundaries = re.sub(
        r"(?<=[A-Z])(?=[A-Z][a-z])",
        "_",
        key.strip(),
    )
    key_with_word_boundaries = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        key_with_word_boundaries,
    )
    return (
        re.sub(r"[^a-zA-Z0-9]+", "_", key_with_word_boundaries)
        .strip("_")
        .lower()
    )


__all__ = [
    "CLOSURE_EVENT_TYPES",
    "CLOSURE_GATE_CHECK_STATUSES",
    "CLOSURE_GATE_STATUSES",
    "CLOSURE_STATUSES",
    "ClosureAuditEventType",
    "ClosureGateCheckStatus",
    "ClosureGateStatus",
    "ClosureStatus",
    "FORBIDDEN_CLOSURE_PAYLOAD_KEYS",
    "GENESIS_CLOSURE_EVENT_HASH",
    "JsonValue",
    "ReleaseClosureAuditEvent",
    "ReleaseClosureGate",
    "ReleaseClosureGateCheck",
    "ReleaseClosureRecord",
    "ReleaseEvidencePackage",
    "build_release_closure_audit_event",
    "canonical_closure_payload_hash",
    "make_release_closure_event_id",
    "make_release_closure_id",
    "make_release_evidence_package_id",
    "validate_json_safe",
    "validate_release_closure_audit_event_hash",
]
