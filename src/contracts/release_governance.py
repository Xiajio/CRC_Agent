from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
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

ReleaseTargetScope = Literal["shadow", "feature_flag_candidate"]
ReleaseIntentStatus = Literal[
    "draft",
    "pending_approval",
    "approved",
    "rejected",
    "cancelled",
]
ReleaseApproverRole = Literal[
    "release_manager",
    "clinical_safety_reviewer",
    "evidence_reviewer",
]
ReleaseApprovalDecision = Literal["approve", "reject", "request_changes"]
ReleaseRollbackPlanStatus = Literal["proposed", "accepted"]
ReleaseAuditEventType = Literal[
    "intent_created",
    "approval_recorded",
    "rollback_plan_recorded",
    "intent_cancelled",
    "governance_read",
]

TARGET_SCOPES: tuple[ReleaseTargetScope, ...] = (
    "shadow",
    "feature_flag_candidate",
)
INTENT_STATUSES: tuple[ReleaseIntentStatus, ...] = (
    "draft",
    "pending_approval",
    "approved",
    "rejected",
    "cancelled",
)
APPROVER_ROLES: tuple[ReleaseApproverRole, ...] = (
    "release_manager",
    "clinical_safety_reviewer",
    "evidence_reviewer",
)
APPROVAL_DECISIONS: tuple[ReleaseApprovalDecision, ...] = (
    "approve",
    "reject",
    "request_changes",
)
ROLLBACK_PLAN_STATUSES: tuple[ReleaseRollbackPlanStatus, ...] = (
    "proposed",
    "accepted",
)
AUDIT_EVENT_TYPES: tuple[ReleaseAuditEventType, ...] = (
    "intent_created",
    "approval_recorded",
    "rollback_plan_recorded",
    "intent_cancelled",
    "governance_read",
)
GENESIS_EVENT_HASH = "sha256:GENESIS"
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "cookie",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }
)


@dataclass(frozen=True)
class ReleaseIntent:
    intent_id: str
    source_release_report_id: str
    source_report_path: str
    harness_run_ids: list[str]
    literature_run_id: str | None
    version_chain: dict[str, JsonValue]
    release_decision_snapshot: str
    rollback_target: str
    requested_by: str
    requested_at: str
    target_scope: ReleaseTargetScope
    status: ReleaseIntentStatus
    blocking_summary: dict[str, JsonValue]

    def __post_init__(self) -> None:
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty(
            "source_release_report_id", self.source_release_report_id
        )
        _require_repo_relative_path("source_report_path", self.source_report_path)
        _require_string_list(
            "harness_run_ids", self.harness_run_ids, min_items=1
        )
        _require_optional_string("literature_run_id", self.literature_run_id)
        if type(self.version_chain) is not dict:
            raise TypeError("version_chain must be a dictionary")
        validate_json_safe(self.version_chain, path="version_chain")
        _require_non_empty(
            "release_decision_snapshot", self.release_decision_snapshot
        )
        _require_non_empty("rollback_target", self.rollback_target)
        _require_non_empty("requested_by", self.requested_by)
        _require_non_empty("requested_at", self.requested_at)
        _validate_choice("target_scope", self.target_scope, TARGET_SCOPES)
        _validate_choice("status", self.status, INTENT_STATUSES)
        if type(self.blocking_summary) is not dict:
            raise TypeError("blocking_summary must be a dictionary")
        validate_json_safe(self.blocking_summary, path="blocking_summary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "source_release_report_id": self.source_release_report_id,
            "source_report_path": self.source_report_path,
            "harness_run_ids": list(self.harness_run_ids),
            "literature_run_id": self.literature_run_id,
            "version_chain": _copy_json_safe(
                self.version_chain, path="version_chain"
            ),
            "release_decision_snapshot": self.release_decision_snapshot,
            "rollback_target": self.rollback_target,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "target_scope": self.target_scope,
            "status": self.status,
            "blocking_summary": _copy_json_safe(
                self.blocking_summary, path="blocking_summary"
            ),
        }


@dataclass(frozen=True)
class ReleaseApproval:
    approval_id: str
    intent_id: str
    approver_role: ReleaseApproverRole
    decision: ReleaseApprovalDecision
    reason: str
    signed_by: str
    signed_at: str
    required: bool

    def __post_init__(self) -> None:
        _require_non_empty("approval_id", self.approval_id)
        _require_non_empty("intent_id", self.intent_id)
        _validate_choice("approver_role", self.approver_role, APPROVER_ROLES)
        _validate_choice("decision", self.decision, APPROVAL_DECISIONS)
        _require_non_empty("reason", self.reason)
        _require_non_empty("signed_by", self.signed_by)
        _require_non_empty("signed_at", self.signed_at)
        if type(self.required) is not bool:
            raise TypeError("required must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "intent_id": self.intent_id,
            "approver_role": self.approver_role,
            "decision": self.decision,
            "reason": self.reason,
            "signed_by": self.signed_by,
            "signed_at": self.signed_at,
            "required": self.required,
        }


@dataclass(frozen=True)
class ReleaseRollbackPlan:
    rollback_plan_id: str
    intent_id: str
    rollback_target: str
    owner: str
    status: ReleaseRollbackPlanStatus
    verification_steps: list[str]
    created_at: str

    def __post_init__(self) -> None:
        _require_non_empty("rollback_plan_id", self.rollback_plan_id)
        _require_non_empty("intent_id", self.intent_id)
        _require_non_empty("rollback_target", self.rollback_target)
        _require_non_empty("owner", self.owner)
        _validate_choice("status", self.status, ROLLBACK_PLAN_STATUSES)
        _require_string_list(
            "verification_steps", self.verification_steps, min_items=2
        )
        _require_non_empty("created_at", self.created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_plan_id": self.rollback_plan_id,
            "intent_id": self.intent_id,
            "rollback_target": self.rollback_target,
            "owner": self.owner,
            "status": self.status,
            "verification_steps": list(self.verification_steps),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ReleaseAuditEvent:
    event_id: str
    intent_id: str
    event_type: ReleaseAuditEventType
    actor: str
    timestamp: str
    payload: JsonValue
    payload_hash: str
    previous_event_hash: str
    event_hash: str

    def __post_init__(self) -> None:
        _require_non_empty("event_id", self.event_id)
        _require_non_empty("intent_id", self.intent_id)
        _validate_choice("event_type", self.event_type, AUDIT_EVENT_TYPES)
        _require_non_empty("actor", self.actor)
        _require_non_empty("timestamp", self.timestamp)
        validate_json_safe(self.payload, path="payload")
        _reject_forbidden_payload_keys(self.payload)
        _require_hash("payload_hash", self.payload_hash)
        _require_hash(
            "previous_event_hash", self.previous_event_hash, allow_genesis=True
        )
        _require_hash("event_hash", self.event_hash)
        expected_payload_hash = canonical_payload_hash(self.payload)
        if self.payload_hash != expected_payload_hash:
            raise ValueError("payload_hash does not match canonical payload")
        validate_audit_event_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "intent_id": self.intent_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "payload": _copy_json_safe(self.payload, path="payload"),
            "payload_hash": self.payload_hash,
            "previous_event_hash": self.previous_event_hash,
            "event_hash": self.event_hash,
        }


def make_release_intent_id(source_release_report_id: str) -> str:
    _require_non_empty("source_release_report_id", source_release_report_id)
    payload = {"source_release_report_id": source_release_report_id}
    return f"release_intent_{_slug(source_release_report_id)}_{_stable_hash(payload)}"


def make_release_approval_id(
    intent_id: str,
    approver_role: str,
    signed_at: str,
) -> str:
    _require_non_empty("intent_id", intent_id)
    _validate_choice("approver_role", approver_role, APPROVER_ROLES)
    _require_non_empty("signed_at", signed_at)
    payload = {
        "intent_id": intent_id,
        "approver_role": approver_role,
        "signed_at": signed_at,
    }
    return (
        f"release_approval_{_slug(intent_id)}_{_slug(approver_role)}_"
        f"{_stable_hash(payload)}"
    )


def make_release_rollback_plan_id(intent_id: str, created_at: str) -> str:
    _require_non_empty("intent_id", intent_id)
    _require_non_empty("created_at", created_at)
    payload = {"intent_id": intent_id, "created_at": created_at}
    return f"rollback_plan_{_slug(intent_id)}_{_stable_hash(payload)}"


def make_release_audit_event_id(
    intent_id: str,
    event_type: str,
    timestamp: str,
) -> str:
    _require_non_empty("intent_id", intent_id)
    _validate_choice("event_type", event_type, AUDIT_EVENT_TYPES)
    _require_non_empty("timestamp", timestamp)
    payload = {
        "intent_id": intent_id,
        "event_type": event_type,
        "timestamp": timestamp,
    }
    return f"release_audit_{_slug(event_type)}_{_stable_hash(payload)}"


def canonical_payload_hash(payload: JsonValue) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    stable_json = json.dumps(
        payload_copy,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()}"


def build_audit_event(
    *,
    event_id: str,
    intent_id: str,
    event_type: ReleaseAuditEventType,
    actor: str,
    timestamp: str,
    payload: JsonValue,
    previous_event_hash: str,
) -> ReleaseAuditEvent:
    payload_copy = _copy_json_safe(payload, path="payload")
    payload_hash = canonical_payload_hash(payload_copy)
    event_without_hash: dict[str, JsonValue] = {
        "event_id": event_id,
        "intent_id": intent_id,
        "event_type": event_type,
        "actor": actor,
        "timestamp": timestamp,
        "payload": payload_copy,
        "payload_hash": payload_hash,
        "previous_event_hash": previous_event_hash,
    }
    event_hash = canonical_payload_hash(event_without_hash)
    return ReleaseAuditEvent(
        event_id=event_id,
        intent_id=intent_id,
        event_type=event_type,
        actor=actor,
        timestamp=timestamp,
        payload=payload_copy,
        payload_hash=payload_hash,
        previous_event_hash=previous_event_hash,
        event_hash=event_hash,
    )


def validate_audit_event_hash(event: ReleaseAuditEvent) -> bool:
    if not isinstance(event, ReleaseAuditEvent):
        raise TypeError("event must be ReleaseAuditEvent")
    event_payload = event.to_dict()
    event_hash = event_payload.pop("event_hash")
    expected_event_hash = canonical_payload_hash(event_payload)
    if event_hash != expected_event_hash:
        raise ValueError("event_hash does not match canonical audit event payload")
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


def _stable_hash(payload: dict[str, JsonValue]) -> str:
    validate_json_safe(payload)
    stable_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_json.encode("utf-8")).hexdigest()[:8]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return slug or "unknown"


def _require_non_empty(field_name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_optional_string(field_name: str, value: str | None) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")


def _require_string_list(
    field_name: str,
    value: object,
    *,
    min_items: int = 0,
) -> None:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list")
    if len(value) < min_items:
        if field_name == "verification_steps":
            raise ValueError(
                "verification_steps must contain at least two steps"
            )
        raise ValueError(f"{field_name} must contain at least {min_items} item")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")


def _require_repo_relative_path(field_name: str, value: str) -> None:
    _require_non_empty(field_name, value)
    windows_path = PureWindowsPath(value)
    if (
        PurePosixPath(value).is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError(f"{field_name} must be a repo-relative path")
    normalized_parts = value.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in normalized_parts):
        raise ValueError(f"{field_name} must be a repo-relative path")


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
    if allow_genesis and value == GENESIS_EVENT_HASH:
        return
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a sha256 hash")


def _reject_forbidden_payload_keys(value: JsonValue) -> None:
    if isinstance(value, list):
        for item in value:
            _reject_forbidden_payload_keys(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = key.strip().lower().replace("-", "_")
            if normalized_key in FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"payload contains forbidden key: {key}")
            _reject_forbidden_payload_keys(item)


__all__ = [
    "APPROVAL_DECISIONS",
    "APPROVER_ROLES",
    "AUDIT_EVENT_TYPES",
    "FORBIDDEN_PAYLOAD_KEYS",
    "GENESIS_EVENT_HASH",
    "INTENT_STATUSES",
    "ROLLBACK_PLAN_STATUSES",
    "TARGET_SCOPES",
    "JsonValue",
    "ReleaseApproval",
    "ReleaseApprovalDecision",
    "ReleaseApproverRole",
    "ReleaseAuditEvent",
    "ReleaseAuditEventType",
    "ReleaseIntent",
    "ReleaseIntentStatus",
    "ReleaseRollbackPlan",
    "ReleaseRollbackPlanStatus",
    "ReleaseTargetScope",
    "build_audit_event",
    "canonical_payload_hash",
    "make_release_approval_id",
    "make_release_audit_event_id",
    "make_release_intent_id",
    "make_release_rollback_plan_id",
    "validate_audit_event_hash",
    "validate_json_safe",
]
