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

FEATURE_FLAG_NAME = "doctor_review_cockpit_v0"
FEATURE_FLAG_SCOPE = "feature_flag_candidate"
GENESIS_EXECUTION_EVENT_HASH = "sha256:GENESIS"

ExecutionAction = Literal["release", "rollback"]
ExecutionResultStatus = Literal["succeeded", "failed"]
ExecutionAuditEventType = Literal[
    "release_requested",
    "release_succeeded",
    "release_failed",
    "rollback_requested",
    "rollback_succeeded",
    "rollback_failed",
    "execution_read",
]

EXECUTION_ACTIONS: tuple[ExecutionAction, ...] = ("release", "rollback")
EXECUTION_RESULT_STATUSES: tuple[ExecutionResultStatus, ...] = (
    "succeeded",
    "failed",
)
EXECUTION_EVENT_TYPES: tuple[ExecutionAuditEventType, ...] = (
    "release_requested",
    "release_succeeded",
    "release_failed",
    "rollback_requested",
    "rollback_succeeded",
    "rollback_failed",
    "execution_read",
)
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "deployment_credential",
        "deployment_credentials",
        "hidden_reasoning",
        "chain_of_thought",
        "medical_record_number",
        "mrn",
        "password",
        "patient_id",
        "patient_identifier",
        "patient_name",
        "patient_number",
        "private_key",
        "prompt",
        "refresh_token",
        "secret",
        "session_token",
        "token",
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
    }
)


@dataclass(frozen=True)
class ReleaseExecutionRequest:
    execution_id: str
    intent_id: str
    action: ExecutionAction
    requested_by: str
    requested_at: str
    idempotency_key: str
    reason: str
    expected_governance_hash: str
    expected_rollback_plan_id: str
    target_flag_state: dict[str, JsonValue]
    rollback_target: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty("execution_id", self.execution_id)
        _require_non_empty("intent_id", self.intent_id)
        _validate_choice("action", self.action, EXECUTION_ACTIONS)
        _require_non_empty("requested_by", self.requested_by)
        _require_non_empty("requested_at", self.requested_at)
        _require_non_empty("idempotency_key", self.idempotency_key)
        _require_non_empty("reason", self.reason)
        _require_hash("expected_governance_hash", self.expected_governance_hash)
        _require_non_empty(
            "expected_rollback_plan_id",
            self.expected_rollback_plan_id,
        )
        if type(self.target_flag_state) is not dict:
            raise TypeError("target_flag_state must be a dictionary")
        _validate_target_flag_state(self.action, self.target_flag_state)
        if self.action == "rollback":
            _require_required_string("rollback_target", self.rollback_target)
        elif self.rollback_target is not None:
            _require_non_empty("rollback_target", self.rollback_target)
        object.__setattr__(
            self,
            "target_flag_state",
            _freeze_json_safe(
                self.target_flag_state,
                path="target_flag_state",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "action": self.action,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "idempotency_key": self.idempotency_key,
            "reason": self.reason,
            "expected_governance_hash": self.expected_governance_hash,
            "expected_rollback_plan_id": self.expected_rollback_plan_id,
            "target_flag_state": _copy_frozen_json_safe(
                self.target_flag_state,
                path="target_flag_state",
            ),
            "rollback_target": self.rollback_target,
        }


@dataclass(frozen=True)
class FeatureFlagState:
    flag_name: str
    enabled: bool
    scope: str
    source_intent_id: str
    source_execution_id: str
    rollback_target: str
    updated_by: str
    updated_at: str

    def __post_init__(self) -> None:
        if self.flag_name != FEATURE_FLAG_NAME:
            raise ValueError(f"flag_name must be {FEATURE_FLAG_NAME}")
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be bool")
        if self.scope != FEATURE_FLAG_SCOPE:
            raise ValueError(f"scope must be {FEATURE_FLAG_SCOPE}")
        _require_non_empty("source_intent_id", self.source_intent_id)
        _require_non_empty("source_execution_id", self.source_execution_id)
        _require_non_empty("rollback_target", self.rollback_target)
        _require_non_empty("updated_by", self.updated_by)
        _require_non_empty("updated_at", self.updated_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag_name": self.flag_name,
            "enabled": self.enabled,
            "scope": self.scope,
            "source_intent_id": self.source_intent_id,
            "source_execution_id": self.source_execution_id,
            "rollback_target": self.rollback_target,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ReleaseExecutionResult:
    result_id: str
    execution_id: str
    intent_id: str
    action: ExecutionAction
    status: ExecutionResultStatus
    started_at: str
    finished_at: str
    actor: str
    previous_flag_state: dict[str, JsonValue] | None
    new_flag_state: dict[str, JsonValue] | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        _require_non_empty("result_id", self.result_id)
        _require_non_empty("execution_id", self.execution_id)
        _require_non_empty("intent_id", self.intent_id)
        _validate_choice("action", self.action, EXECUTION_ACTIONS)
        _validate_choice("status", self.status, EXECUTION_RESULT_STATUSES)
        _require_non_empty("started_at", self.started_at)
        _require_non_empty("finished_at", self.finished_at)
        _require_non_empty("actor", self.actor)
        if self.previous_flag_state is not None:
            if type(self.previous_flag_state) is not dict:
                raise TypeError("previous_flag_state must be a dictionary or None")
            object.__setattr__(
                self,
                "previous_flag_state",
                _freeze_json_safe(
                    self.previous_flag_state,
                    path="previous_flag_state",
                ),
            )
        if self.new_flag_state is not None:
            if type(self.new_flag_state) is not dict:
                raise TypeError("new_flag_state must be a dictionary or None")
            object.__setattr__(
                self,
                "new_flag_state",
                _freeze_json_safe(self.new_flag_state, path="new_flag_state"),
            )
        if self.status == "failed":
            _require_required_string("failure_reason", self.failure_reason)
        else:
            if self.failure_reason is not None:
                raise ValueError("failure_reason must be None when status succeeded")
            if self.new_flag_state is None:
                raise ValueError("new_flag_state is required when status succeeded")

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "action": self.action,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "actor": self.actor,
            "previous_flag_state": (
                _copy_frozen_json_safe(
                    self.previous_flag_state,
                    path="previous_flag_state",
                )
                if self.previous_flag_state is not None
                else None
            ),
            "new_flag_state": (
                _copy_frozen_json_safe(
                    self.new_flag_state,
                    path="new_flag_state",
                )
                if self.new_flag_state is not None
                else None
            ),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ReleaseExecutionAuditEvent:
    event_id: str
    execution_id: str
    intent_id: str
    event_type: ExecutionAuditEventType
    actor: str
    timestamp: str
    payload_hash: str
    previous_event_hash: str
    event_hash: str

    def __post_init__(self) -> None:
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("execution_id", self.execution_id)
        _require_non_empty("intent_id", self.intent_id)
        _validate_choice("event_type", self.event_type, EXECUTION_EVENT_TYPES)
        _require_non_empty("actor", self.actor)
        _require_non_empty("timestamp", self.timestamp)
        _require_hash("payload_hash", self.payload_hash)
        _require_hash(
            "previous_event_hash",
            self.previous_event_hash,
            allow_genesis=True,
        )
        _require_hash("event_hash", self.event_hash)
        validate_execution_audit_event_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "payload_hash": self.payload_hash,
            "previous_event_hash": self.previous_event_hash,
            "event_hash": self.event_hash,
        }


def make_release_execution_id(
    intent_id: str,
    action: str,
    idempotency_key: str,
) -> str:
    _require_non_empty("intent_id", intent_id)
    _validate_choice("action", action, EXECUTION_ACTIONS)
    _require_non_empty("idempotency_key", idempotency_key)
    payload = {
        "intent_id": intent_id,
        "action": action,
        "idempotency_key": idempotency_key,
    }
    return f"release_exec_{_slug(intent_id)}_{_slug(action)}_{_stable_hash(payload)}"


def make_release_execution_result_id(execution_id: str) -> str:
    _require_non_empty("execution_id", execution_id)
    return f"release_result_{_slug(execution_id)}"


def make_release_execution_event_id(
    execution_id: str,
    event_type: str,
    timestamp: str,
) -> str:
    _require_non_empty("execution_id", execution_id)
    _validate_choice("event_type", event_type, EXECUTION_EVENT_TYPES)
    _require_non_empty("timestamp", timestamp)
    payload = {
        "execution_id": execution_id,
        "event_type": event_type,
        "timestamp": timestamp,
    }
    return f"release_execution_audit_{_slug(event_type)}_{_stable_hash(payload)}"


def canonical_execution_payload_hash(payload: JsonValue) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    stable_json = json.dumps(
        payload_copy,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()}"


def build_execution_audit_event(
    *,
    event_id: str,
    execution_id: str,
    intent_id: str,
    event_type: ExecutionAuditEventType,
    actor: str,
    timestamp: str,
    payload: JsonValue,
    previous_event_hash: str,
) -> ReleaseExecutionAuditEvent:
    payload_hash = canonical_execution_payload_hash(payload)
    event_payload: dict[str, JsonValue] = {
        "event_id": event_id,
        "execution_id": execution_id,
        "intent_id": intent_id,
        "event_type": event_type,
        "actor": actor,
        "timestamp": timestamp,
        "payload_hash": payload_hash,
        "previous_event_hash": previous_event_hash,
    }
    event_hash = canonical_execution_payload_hash(event_payload)
    return ReleaseExecutionAuditEvent(
        event_id=event_id,
        execution_id=execution_id,
        intent_id=intent_id,
        event_type=event_type,
        actor=actor,
        timestamp=timestamp,
        payload_hash=payload_hash,
        previous_event_hash=previous_event_hash,
        event_hash=event_hash,
    )


def validate_execution_audit_event_hash(
    event: ReleaseExecutionAuditEvent,
) -> bool:
    if not isinstance(event, ReleaseExecutionAuditEvent):
        raise TypeError("event must be ReleaseExecutionAuditEvent")
    event_payload = event.to_dict()
    event_hash = event_payload.pop("event_hash")
    expected_event_hash = canonical_execution_payload_hash(event_payload)
    if event_hash != expected_event_hash:
        raise ValueError(
            "event_hash does not match canonical execution audit event payload"
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


def _validate_target_flag_state(
    action: str,
    target_flag_state: dict[str, JsonValue],
) -> None:
    _copy_json_safe(target_flag_state, path="target_flag_state")
    if target_flag_state.get("flag_name") != FEATURE_FLAG_NAME:
        raise ValueError(f"target flag flag_name must be {FEATURE_FLAG_NAME}")
    if target_flag_state.get("scope") != FEATURE_FLAG_SCOPE:
        raise ValueError(f"target flag scope must be {FEATURE_FLAG_SCOPE}")
    enabled = target_flag_state.get("enabled")
    if type(enabled) is not bool:
        raise TypeError("target flag enabled must be bool")
    if action == "release" and enabled is not True:
        raise ValueError("release target flag must be enabled")
    if action == "rollback" and enabled is not False:
        raise ValueError("rollback target flag must be disabled")


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
    if allow_genesis and value == GENESIS_EXECUTION_EVENT_HASH:
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
    if normalized_key in FORBIDDEN_PAYLOAD_KEYS:
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
    "EXECUTION_ACTIONS",
    "EXECUTION_EVENT_TYPES",
    "EXECUTION_RESULT_STATUSES",
    "FEATURE_FLAG_NAME",
    "FEATURE_FLAG_SCOPE",
    "FORBIDDEN_PAYLOAD_KEYS",
    "GENESIS_EXECUTION_EVENT_HASH",
    "ExecutionAction",
    "ExecutionAuditEventType",
    "ExecutionResultStatus",
    "FeatureFlagState",
    "JsonValue",
    "ReleaseExecutionAuditEvent",
    "ReleaseExecutionRequest",
    "ReleaseExecutionResult",
    "build_execution_audit_event",
    "canonical_execution_payload_hash",
    "make_release_execution_event_id",
    "make_release_execution_id",
    "make_release_execution_result_id",
    "validate_execution_audit_event_hash",
    "validate_json_safe",
]
