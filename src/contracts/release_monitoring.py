from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
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

GENESIS_MONITORING_EVENT_HASH = "sha256:GENESIS"

MonitoringCheckType = Literal[
    "execution_integrity",
    "governance_drift",
    "p0_harness_replay",
    "agent_admin_smoke",
    "doctor_review_smoke",
    "literature_isolation",
    "manual_operator_note",
]
MonitoringCheckStatus = Literal["pass", "warning", "fail"]
MonitoringAlertSeverity = Literal["info", "warning", "critical"]
MonitoringAlertCategory = Literal[
    "missing_required_check",
    "post_release_check_failed",
    "execution_integrity_failed",
    "governance_drift",
    "feature_flag_state_mismatch",
    "rollback_ready",
]
MonitoringAlertStatus = Literal["active", "acknowledged"]
MonitoringRecommendedAction = Literal[
    "observe",
    "prepare_rollback",
    "investigate",
    "execute_step13_rollback",
]
MonitoringAcknowledgementDisposition = Literal[
    "investigating",
    "false_positive",
    "accepted_risk",
    "rollback_started_elsewhere",
]
MonitoringAuditEventType = Literal[
    "check_recorded",
    "alert_acknowledged",
    "monitoring_read",
]

MONITORING_CHECK_TYPES: tuple[MonitoringCheckType, ...] = (
    "execution_integrity",
    "governance_drift",
    "p0_harness_replay",
    "agent_admin_smoke",
    "doctor_review_smoke",
    "literature_isolation",
    "manual_operator_note",
)
MONITORING_CHECK_STATUSES: tuple[MonitoringCheckStatus, ...] = (
    "pass",
    "warning",
    "fail",
)
MONITORING_ALERT_SEVERITIES: tuple[MonitoringAlertSeverity, ...] = (
    "info",
    "warning",
    "critical",
)
MONITORING_ALERT_CATEGORIES: tuple[MonitoringAlertCategory, ...] = (
    "missing_required_check",
    "post_release_check_failed",
    "execution_integrity_failed",
    "governance_drift",
    "feature_flag_state_mismatch",
    "rollback_ready",
)
MONITORING_ALERT_STATUSES: tuple[MonitoringAlertStatus, ...] = (
    "active",
    "acknowledged",
)
MONITORING_RECOMMENDED_ACTIONS: tuple[MonitoringRecommendedAction, ...] = (
    "observe",
    "prepare_rollback",
    "investigate",
    "execute_step13_rollback",
)
MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS: tuple[
    MonitoringAcknowledgementDisposition, ...
] = (
    "investigating",
    "false_positive",
    "accepted_risk",
    "rollback_started_elsewhere",
)
MONITORING_EVENT_TYPES: tuple[MonitoringAuditEventType, ...] = (
    "check_recorded",
    "alert_acknowledged",
    "monitoring_read",
)
FORBIDDEN_MONITORING_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "bearer_token",
        "chain_of_thought",
        "client_secret",
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
        "patient_name",
        "patient_number",
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
class ReleaseMonitoringCheck:
    check_id: str
    intent_id: str
    execution_id: str
    check_type: MonitoringCheckType
    status: MonitoringCheckStatus
    observed_by: str
    observed_at: str
    summary: str
    evidence_refs: list[str]
    metrics: dict[str, JsonValue]
    idempotency_key: str

    def __post_init__(self) -> None:
        _require_non_empty("check_id", self.check_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("execution_id", self.execution_id)
        _validate_choice("check_type", self.check_type, MONITORING_CHECK_TYPES)
        _validate_choice("status", self.status, MONITORING_CHECK_STATUSES)
        _require_non_empty("observed_by", self.observed_by)
        _require_non_empty("observed_at", self.observed_at)
        _require_non_empty("summary", self.summary)
        _require_non_empty("idempotency_key", self.idempotency_key)
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(_validate_evidence_refs(self.evidence_refs)),
        )
        if type(self.metrics) is not dict:
            raise TypeError("metrics must be a dictionary")
        metrics = _copy_json_safe(self.metrics, path="metrics")
        _reject_forbidden_payload_keys(metrics)
        object.__setattr__(self, "metrics", _freeze_json_safe(metrics, path="metrics"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "check_type": self.check_type,
            "status": self.status,
            "observed_by": self.observed_by,
            "observed_at": self.observed_at,
            "summary": self.summary,
            "evidence_refs": list(self.evidence_refs),
            "metrics": _copy_frozen_json_safe(self.metrics, path="metrics"),
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True)
class ReleaseMonitoringAlert:
    alert_id: str
    intent_id: str
    execution_id: str
    severity: MonitoringAlertSeverity
    category: MonitoringAlertCategory
    status: MonitoringAlertStatus
    message: str
    source_check_ids: list[str]
    recommended_action: MonitoringRecommendedAction
    created_at: str

    def __post_init__(self) -> None:
        _require_non_empty("alert_id", self.alert_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("execution_id", self.execution_id)
        _validate_choice("severity", self.severity, MONITORING_ALERT_SEVERITIES)
        _validate_choice("category", self.category, MONITORING_ALERT_CATEGORIES)
        _validate_choice("status", self.status, MONITORING_ALERT_STATUSES)
        _require_non_empty("message", self.message)
        object.__setattr__(
            self,
            "source_check_ids",
            tuple(_require_string_list("source_check_ids", self.source_check_ids)),
        )
        _validate_choice(
            "recommended_action",
            self.recommended_action,
            MONITORING_RECOMMENDED_ACTIONS,
        )
        _require_non_empty("created_at", self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "severity": self.severity,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "source_check_ids": list(self.source_check_ids),
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ReleaseRollbackTriggerCandidate:
    candidate_id: str
    intent_id: str
    execution_id: str
    source_alert_ids: list[str]
    recommended_action: MonitoringRecommendedAction
    rollback_plan_id: str
    rollback_target: str
    reason: str
    created_at: str

    def __post_init__(self) -> None:
        _require_non_empty("candidate_id", self.candidate_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("execution_id", self.execution_id)
        object.__setattr__(
            self,
            "source_alert_ids",
            tuple(_require_alert_id_list("source_alert_ids", self.source_alert_ids)),
        )
        if self.recommended_action != "execute_step13_rollback":
            raise ValueError("recommended_action must be execute_step13_rollback")
        _require_non_empty("rollback_plan_id", self.rollback_plan_id)
        _require_non_empty("rollback_target", self.rollback_target)
        _require_non_empty("reason", self.reason)
        _require_non_empty("created_at", self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "source_alert_ids": list(self.source_alert_ids),
            "recommended_action": self.recommended_action,
            "rollback_plan_id": self.rollback_plan_id,
            "rollback_target": self.rollback_target,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ReleaseMonitoringAcknowledgement:
    acknowledgement_id: str
    alert_id: str
    intent_id: str
    execution_id: str
    acknowledged_by: str
    acknowledged_at: str
    disposition: MonitoringAcknowledgementDisposition
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty("acknowledgement_id", self.acknowledgement_id)
        _require_non_empty("alert_id", self.alert_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("execution_id", self.execution_id)
        _require_non_empty("acknowledged_by", self.acknowledged_by)
        _require_non_empty("acknowledged_at", self.acknowledged_at)
        _validate_choice(
            "disposition",
            self.disposition,
            MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS,
        )
        _require_non_empty("reason", self.reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acknowledgement_id": self.acknowledgement_id,
            "alert_id": self.alert_id,
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "disposition": self.disposition,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ReleaseMonitoringAuditEvent:
    event_id: str
    intent_id: str
    execution_id: str
    event_type: MonitoringAuditEventType
    actor: str
    timestamp: str
    payload_hash: str
    previous_event_hash: str
    event_hash: str

    def __post_init__(self) -> None:
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("execution_id", self.execution_id)
        _validate_choice("event_type", self.event_type, MONITORING_EVENT_TYPES)
        _require_non_empty("actor", self.actor)
        _require_non_empty("timestamp", self.timestamp)
        _require_hash("payload_hash", self.payload_hash)
        _require_hash(
            "previous_event_hash",
            self.previous_event_hash,
            allow_genesis=True,
        )
        _require_hash("event_hash", self.event_hash)
        validate_monitoring_audit_event_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "payload_hash": self.payload_hash,
            "previous_event_hash": self.previous_event_hash,
            "event_hash": self.event_hash,
        }


def make_monitoring_check_id(
    execution_id: str,
    check_type: str,
    idempotency_key: str,
) -> str:
    _require_non_empty("execution_id", execution_id)
    _validate_choice("check_type", check_type, MONITORING_CHECK_TYPES)
    _require_non_empty("idempotency_key", idempotency_key)
    payload = {
        "execution_id": execution_id,
        "check_type": check_type,
        "idempotency_key": idempotency_key,
    }
    return f"release_monitor_check_{_slug(execution_id)}_{_slug(check_type)}_{_stable_hash(payload)}"


def make_monitoring_alert_id(
    execution_id: str,
    category: str,
    discriminator: str,
) -> str:
    _require_non_empty("execution_id", execution_id)
    _validate_choice("category", category, MONITORING_ALERT_CATEGORIES)
    _require_non_empty("discriminator", discriminator)
    payload = {
        "execution_id": execution_id,
        "category": category,
        "discriminator": discriminator,
    }
    return f"release_monitor_alert_{_slug(execution_id)}_{_slug(category)}_{_stable_hash(payload)}"


def make_rollback_trigger_candidate_id(
    execution_id: str,
    alert_ids: list[str],
) -> str:
    _require_non_empty("execution_id", execution_id)
    ordered_alert_ids = _require_alert_id_list("alert_ids", alert_ids)
    payload = {"execution_id": execution_id, "alert_ids": sorted(ordered_alert_ids)}
    return f"release_monitor_rollback_candidate_{_slug(execution_id)}_{_stable_hash(payload)}"


def make_monitoring_acknowledgement_id(
    alert_id: str,
    acknowledgement_key: str,
) -> str:
    _require_non_empty("alert_id", alert_id)
    _require_non_empty("acknowledgement_key", acknowledgement_key)
    payload = {"alert_id": alert_id, "acknowledgement_key": acknowledgement_key}
    return f"release_monitor_ack_{_slug(alert_id)}_{_stable_hash(payload)}"


def make_monitoring_event_id(
    execution_id: str,
    event_type: str,
    timestamp: str,
) -> str:
    _require_non_empty("execution_id", execution_id)
    _validate_choice("event_type", event_type, MONITORING_EVENT_TYPES)
    _require_non_empty("timestamp", timestamp)
    payload = {
        "execution_id": execution_id,
        "event_type": event_type,
        "timestamp": timestamp,
    }
    return f"release_monitoring_audit_{_slug(event_type)}_{_stable_hash(payload)}"


def canonical_monitoring_payload_hash(payload: JsonValue) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    stable_json = json.dumps(
        payload_copy,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()}"


def build_monitoring_audit_event(
    *,
    event_id: str,
    intent_id: str,
    execution_id: str,
    event_type: MonitoringAuditEventType,
    actor: str,
    timestamp: str,
    payload: JsonValue,
    previous_event_hash: str,
) -> ReleaseMonitoringAuditEvent:
    payload_hash = canonical_monitoring_payload_hash(payload)
    event_payload: dict[str, JsonValue] = {
        "event_id": event_id,
        "intent_id": intent_id,
        "execution_id": execution_id,
        "event_type": event_type,
        "actor": actor,
        "timestamp": timestamp,
        "payload_hash": payload_hash,
        "previous_event_hash": previous_event_hash,
    }
    event_hash = canonical_monitoring_payload_hash(event_payload)
    return ReleaseMonitoringAuditEvent(
        event_id=event_id,
        intent_id=intent_id,
        execution_id=execution_id,
        event_type=event_type,
        actor=actor,
        timestamp=timestamp,
        payload_hash=payload_hash,
        previous_event_hash=previous_event_hash,
        event_hash=event_hash,
    )


def validate_monitoring_audit_event_hash(
    event: ReleaseMonitoringAuditEvent,
) -> bool:
    if not isinstance(event, ReleaseMonitoringAuditEvent):
        raise TypeError("event must be ReleaseMonitoringAuditEvent")
    event_payload = event.to_dict()
    event_hash = event_payload.pop("event_hash")
    expected_event_hash = canonical_monitoring_payload_hash(event_payload)
    if event_hash != expected_event_hash:
        raise ValueError(
            "event_hash does not match canonical monitoring audit event payload"
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


def _validate_evidence_refs(evidence_refs: list[str]) -> list[str]:
    refs = _require_string_list("evidence_refs", evidence_refs)
    for ref in refs:
        normalized_ref = ref.replace("\\", "/")
        if re.match(r"^[a-zA-Z]:", normalized_ref) is not None:
            raise ValueError("evidence_refs must be repo-relative")
        if normalized_ref.startswith("/"):
            raise ValueError("evidence_refs must be repo-relative")
        if "://" in normalized_ref:
            raise ValueError("evidence_refs must be repo-relative")
        parts = normalized_ref.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("evidence_refs must be repo-relative")
    return refs


def _require_string_list(field_name: str, value: list[str]) -> list[str]:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list")
    for item in value:
        _require_non_empty(field_name, item)
    return list(value)


def _require_alert_id_list(field_name: str, value: list[str]) -> list[str]:
    values = _require_string_list(field_name, value)
    if not values:
        raise ValueError(f"{field_name} must contain at least one alert id")
    return values


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
    if allow_genesis and value == GENESIS_MONITORING_EVENT_HASH:
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
    if normalized_key in FORBIDDEN_MONITORING_PAYLOAD_KEYS:
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
    "FORBIDDEN_MONITORING_PAYLOAD_KEYS",
    "GENESIS_MONITORING_EVENT_HASH",
    "JsonValue",
    "MONITORING_ACKNOWLEDGEMENT_DISPOSITIONS",
    "MONITORING_ALERT_CATEGORIES",
    "MONITORING_ALERT_SEVERITIES",
    "MONITORING_ALERT_STATUSES",
    "MONITORING_CHECK_STATUSES",
    "MONITORING_CHECK_TYPES",
    "MONITORING_EVENT_TYPES",
    "MONITORING_RECOMMENDED_ACTIONS",
    "MonitoringAcknowledgementDisposition",
    "MonitoringAlertCategory",
    "MonitoringAlertSeverity",
    "MonitoringAlertStatus",
    "MonitoringAuditEventType",
    "MonitoringCheckStatus",
    "MonitoringCheckType",
    "MonitoringRecommendedAction",
    "ReleaseMonitoringAcknowledgement",
    "ReleaseMonitoringAlert",
    "ReleaseMonitoringAuditEvent",
    "ReleaseMonitoringCheck",
    "ReleaseRollbackTriggerCandidate",
    "build_monitoring_audit_event",
    "canonical_monitoring_payload_hash",
    "make_monitoring_acknowledgement_id",
    "make_monitoring_alert_id",
    "make_monitoring_check_id",
    "make_monitoring_event_id",
    "make_rollback_trigger_candidate_id",
    "validate_json_safe",
    "validate_monitoring_audit_event_hash",
]
