from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
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

SignalType = Literal[
    "doctor_action_trace",
    "evidence_delta",
    "harness_failure",
    "cohort_feasibility_gap",
    "release_monitoring_alert",
]
TargetArea = Literal[
    "prompt",
    "rubric",
    "route",
    "template",
    "evidence_ingest",
    "test_case",
]
CandidatePatchType = TargetArea
CandidatePatchStatus = Literal[
    "candidate",
    "needs_harness",
    "needs_human_review",
    "rejected",
    "approved_for_release_intent",
]
LearningJobStatus = Literal[
    "draft",
    "shadow_only",
    "ready_for_harness",
    "harness_failed",
    "awaiting_human_review",
    "rejected",
    "approved_for_release_intent",
    "archived",
]
LearningJobType = Literal[
    "candidate_patch_generation",
    "candidate_evidence_ingest",
    "candidate_test_case_generation",
]

SIGNAL_TYPES: tuple[SignalType, ...] = (
    "doctor_action_trace",
    "evidence_delta",
    "harness_failure",
    "cohort_feasibility_gap",
    "release_monitoring_alert",
)
TARGET_AREAS: tuple[TargetArea, ...] = (
    "prompt",
    "rubric",
    "route",
    "template",
    "evidence_ingest",
    "test_case",
)
CANDIDATE_PATCH_TYPES: tuple[CandidatePatchType, ...] = TARGET_AREAS
CANDIDATE_PATCH_STATUSES: tuple[CandidatePatchStatus, ...] = (
    "candidate",
    "needs_harness",
    "needs_human_review",
    "rejected",
    "approved_for_release_intent",
)
LEARNING_JOB_STATUSES: tuple[LearningJobStatus, ...] = (
    "draft",
    "shadow_only",
    "ready_for_harness",
    "harness_failed",
    "awaiting_human_review",
    "rejected",
    "approved_for_release_intent",
    "archived",
)
LEARNING_JOB_TYPES: tuple[LearningJobType, ...] = (
    "candidate_patch_generation",
    "candidate_evidence_ingest",
    "candidate_test_case_generation",
)

FORBIDDEN_LEARNING_PAYLOAD_KEYS = frozenset(
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
        "doctor_note_text",
        "hidden_reasoning",
        "medical_record_number",
        "mrn",
        "password",
        "patient_id",
        "patient_ids",
        "patient_identifier",
        "patient_identifiers",
        "patient_name",
        "patient_names",
        "patient_record",
        "patient_records",
        "private_key",
        "raw_patient_identifier",
        "raw_patient_identifiers",
        "refresh_token",
        "secret",
        "session_id",
        "session_ids",
        "session_token",
        "token",
        "training_row",
        "training_rows",
    }
)
_SENSITIVE_KEY_TOKENS = frozenset(
    {
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
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
        "row",
        "rows",
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
        "row",
        "rows",
    }
)


@dataclass(frozen=True)
class LearningSignal:
    signal_id: str
    signal_type: SignalType
    source_ref: dict[str, JsonValue]
    reason_code: str
    target_area: TargetArea
    severity: str
    summary: str
    deidentified: bool
    created_at: str

    def __post_init__(self) -> None:
        _require_non_empty("signal_id", self.signal_id)
        _validate_choice("signal_type", self.signal_type, SIGNAL_TYPES)
        _validate_choice("target_area", self.target_area, TARGET_AREAS)
        _require_non_empty("reason_code", self.reason_code)
        _reject_forbidden_payload_values(self.reason_code, path="reason_code")
        _require_non_empty("severity", self.severity)
        _require_non_empty("summary", self.summary)
        _reject_forbidden_payload_values(self.summary, path="summary")
        if self.deidentified is not True:
            raise ValueError("deidentified must be true")
        _require_non_empty("created_at", self.created_at)
        object.__setattr__(
            self,
            "source_ref",
            _validated_safe_dict("source_ref", self.source_ref),
        )

    def to_dict(self) -> dict[str, Any]:
        source_ref = _validated_safe_dict("source_ref", self.source_ref)
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "source_ref": source_ref,
            "reason_code": self.reason_code,
            "target_area": self.target_area,
            "severity": self.severity,
            "summary": self.summary,
            "deidentified": self.deidentified,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class CandidatePatch:
    patch_id: str
    patch_type: CandidatePatchType
    target_ref: dict[str, JsonValue]
    change_summary: str
    proposed_diff: dict[str, JsonValue]
    source_signal_ids: list[str]
    status: CandidatePatchStatus
    applies_automatically: bool

    def __post_init__(self) -> None:
        _require_non_empty("patch_id", self.patch_id)
        _validate_choice("patch_type", self.patch_type, CANDIDATE_PATCH_TYPES)
        _require_non_empty("change_summary", self.change_summary)
        _reject_forbidden_payload_values(
            self.change_summary,
            path="change_summary",
        )
        _validate_choice("status", self.status, CANDIDATE_PATCH_STATUSES)
        if self.applies_automatically is not False:
            raise ValueError("applies_automatically must be false")
        target_ref = _validated_safe_dict("target_ref", self.target_ref)
        _reject_clinical_safety_policy_target(target_ref)
        object.__setattr__(self, "target_ref", target_ref)
        object.__setattr__(
            self,
            "proposed_diff",
            _validated_safe_dict("proposed_diff", self.proposed_diff),
        )
        object.__setattr__(
            self,
            "source_signal_ids",
            _require_string_list("source_signal_ids", self.source_signal_ids),
        )

    @property
    def candidate_patch_id(self) -> str:
        return self.patch_id

    def to_dict(self) -> dict[str, Any]:
        target_ref = _validated_safe_dict("target_ref", self.target_ref)
        _reject_clinical_safety_policy_target(target_ref)
        proposed_diff = _validated_safe_dict("proposed_diff", self.proposed_diff)
        return {
            "patch_id": self.patch_id,
            "patch_type": self.patch_type,
            "target_ref": target_ref,
            "change_summary": self.change_summary,
            "proposed_diff": proposed_diff,
            "source_signal_ids": list(self.source_signal_ids),
            "status": self.status,
            "applies_automatically": self.applies_automatically,
        }


@dataclass(frozen=True)
class HarnessRequirement:
    required: bool
    case_pack_version: str
    required_levels: list[str]
    hard_fail_policy: str

    def __post_init__(self) -> None:
        if self.required is not True:
            raise ValueError("required must be true")
        _require_non_empty("case_pack_version", self.case_pack_version)
        object.__setattr__(
            self,
            "required_levels",
            _require_string_list("required_levels", self.required_levels),
        )
        _require_non_empty("hard_fail_policy", self.hard_fail_policy)
        _reject_forbidden_payload_values(
            self.hard_fail_policy,
            path="hard_fail_policy",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "case_pack_version": self.case_pack_version,
            "required_levels": list(self.required_levels),
            "hard_fail_policy": self.hard_fail_policy,
        }


@dataclass(frozen=True)
class HumanReviewRequirement:
    required: bool
    roles: list[str]
    status: str

    def __post_init__(self) -> None:
        if self.required is not True:
            raise ValueError("required must be true")
        object.__setattr__(
            self,
            "roles",
            _require_string_list("roles", self.roles),
        )
        _require_non_empty("status", self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "roles": list(self.roles),
            "status": self.status,
        }


@dataclass(frozen=True)
class LearningJob:
    job_id: str
    job_type: LearningJobType
    status: LearningJobStatus
    created_at: str
    source_signal_ids: list[str]
    candidate_patch_ids: list[str]
    required_harness: HarnessRequirement | dict[str, Any]
    human_review: HumanReviewRequirement | dict[str, Any]
    release_governance_ref: dict[str, JsonValue]
    idempotency_key: str

    def __post_init__(self) -> None:
        _require_non_empty("job_id", self.job_id)
        _validate_choice("job_type", self.job_type, LEARNING_JOB_TYPES)
        _validate_choice("status", self.status, LEARNING_JOB_STATUSES)
        _require_non_empty("created_at", self.created_at)
        object.__setattr__(
            self,
            "source_signal_ids",
            _require_string_list("source_signal_ids", self.source_signal_ids),
        )
        object.__setattr__(
            self,
            "candidate_patch_ids",
            _require_string_list(
                "candidate_patch_ids",
                self.candidate_patch_ids,
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "required_harness",
            _coerce_harness_requirement(self.required_harness),
        )
        object.__setattr__(
            self,
            "human_review",
            _coerce_human_review_requirement(self.human_review),
        )
        object.__setattr__(
            self,
            "release_governance_ref",
            _validated_safe_dict(
                "release_governance_ref",
                self.release_governance_ref,
            ),
        )
        _require_non_empty("idempotency_key", self.idempotency_key)

    @property
    def learning_job_id(self) -> str:
        return self.job_id

    @property
    def harness_requirement(self) -> HarnessRequirement:
        return self.required_harness

    @property
    def human_review_requirement(self) -> HumanReviewRequirement:
        return self.human_review

    def to_dict(self) -> dict[str, Any]:
        release_governance_ref = _validated_safe_dict(
            "release_governance_ref",
            self.release_governance_ref,
        )
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "created_at": self.created_at,
            "source_signal_ids": list(self.source_signal_ids),
            "candidate_patch_ids": list(self.candidate_patch_ids),
            "required_harness": self.required_harness.to_dict(),
            "human_review": self.human_review.to_dict(),
            "release_governance_ref": release_governance_ref,
            "idempotency_key": self.idempotency_key,
        }


def canonical_learning_payload_hash(payload: JsonValue) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    _reject_forbidden_payload_values(payload_copy, path="payload")
    stable_json = json.dumps(payload_copy, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(stable_json.encode('utf-8')).hexdigest()}"


def make_learning_signal_id(source_ref: dict[str, JsonValue]) -> str:
    source_ref_copy = _validated_safe_dict("source_ref", source_ref)
    return f"learning_signal_{_stable_hash(source_ref_copy)}"


def make_candidate_patch_id(patch_type: str, seed: str) -> str:
    _validate_choice("patch_type", patch_type, CANDIDATE_PATCH_TYPES)
    _require_non_empty("seed", seed)
    _reject_forbidden_payload_values(seed, path="seed")
    stable_suffix = _stable_hash({"patch_type": patch_type, "seed": seed})
    return f"candidate_patch_{_slug(patch_type)}_{stable_suffix}"


def make_learning_job_id(
    source_signal_ids: list[str],
    idempotency_key: str,
) -> str:
    signal_ids = _require_string_list("source_signal_ids", source_signal_ids)
    _require_non_empty("idempotency_key", idempotency_key)
    stable_suffix = _stable_hash(
        {"source_signal_ids": signal_ids, "idempotency_key": idempotency_key}
    )
    return f"learning_job_{stable_suffix}"


def _coerce_harness_requirement(
    value: HarnessRequirement | dict[str, Any],
) -> HarnessRequirement:
    if isinstance(value, HarnessRequirement):
        return value
    if type(value) is dict:
        return HarnessRequirement(**value)
    raise TypeError("harness_requirement must be HarnessRequirement")


def _coerce_human_review_requirement(
    value: HumanReviewRequirement | dict[str, Any],
) -> HumanReviewRequirement:
    if isinstance(value, HumanReviewRequirement):
        return value
    if type(value) is dict:
        return HumanReviewRequirement(**value)
    raise TypeError("human_review_requirement must be HumanReviewRequirement")


def _validated_safe_dict(field_name: str, value: Any) -> dict[str, JsonValue]:
    if type(value) is not dict:
        raise TypeError(f"{field_name} must be a dict")
    value_copy = _copy_json_safe(value, path=field_name)
    _reject_forbidden_payload_keys(value_copy)
    _reject_forbidden_payload_values(value_copy, path=field_name)
    return value_copy


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
    if normalized_key in FORBIDDEN_LEARNING_PAYLOAD_KEYS:
        return True
    if "apikey" in compact_key:
        return True
    if {"api", "key"}.issubset(path_token_set):
        return True
    if {"private", "key"}.issubset(path_token_set):
        return True
    if {"hidden", "reasoning"}.issubset(path_token_set):
        return True
    if {"chain", "of", "thought"}.issubset(path_token_set):
        return True
    if {"training", "row"}.issubset(path_token_set):
        return True
    if {"training", "rows"}.issubset(path_token_set):
        return True
    if token_set & _SENSITIVE_KEY_TOKENS:
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


def _reject_forbidden_payload_values(
    value: JsonValue,
    *,
    path: str,
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
            raise ValueError(f"payload contains forbidden content at {path}: {marker}")


def _find_forbidden_payload_value_marker(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", value).strip().lower()
    patterns = (
        (
            r"\bauthorization\b\s*[:=]\s*bearer\s+[A-Za-z0-9._-]{3,}",
            "authorization bearer token",
        ),
        (r"\bbearer\b\s+[A-Za-z0-9._-]{3,}", "bearer token"),
        (r"\bapi[\s_-]*key\b\s*[:=]\s*[A-Za-z0-9._-]{3,}", "api key"),
        (r"\bapikey\b\s*[:=]\s*[A-Za-z0-9._-]{3,}", "apikey"),
        (r"\btoken\b\s*[:=]\s*\S+", "token"),
        (r"\bhidden[\s_-]*reasoning\b\s*[:=]\s*\S+", "hidden reasoning"),
        (r"\bchain[\s_-]*of[\s_-]*thought\b\s*[:=]\s*\S+", "chain of thought"),
        (
            r"\braw[\s_-]*patient[\s_-]*identifier(?:s)?\b\s*[:=]\s*\S+",
            "raw patient identifier",
        ),
        (r"\btraining[\s_-]*row(?:s)?\b\s*[:=]\s*\S+", "training rows"),
    )
    for pattern, marker in patterns:
        if re.search(pattern, normalized) is not None:
            return marker
    return None


def _reject_clinical_safety_policy_target(target_ref: dict[str, JsonValue]) -> None:
    for field_name in ("kind", "id"):
        value = target_ref.get(field_name)
        if isinstance(value, str) and "clinical_safety_policy" in value.lower():
            raise ValueError("target_ref must not reference clinical_safety_policy")


def _require_string_list(
    field_name: str,
    value: Any,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    for item in value:
        _require_non_empty(field_name, item)
    return list(value)


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


def _stable_hash(payload: dict[str, JsonValue]) -> str:
    payload_copy = _copy_json_safe(payload, path="payload")
    _reject_forbidden_payload_keys(payload_copy)
    _reject_forbidden_payload_values(payload_copy, path="payload")
    stable_json = json.dumps(payload_copy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_json.encode("utf-8")).hexdigest()[:12]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "unknown"


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
    "CANDIDATE_PATCH_STATUSES",
    "CANDIDATE_PATCH_TYPES",
    "FORBIDDEN_LEARNING_PAYLOAD_KEYS",
    "LEARNING_JOB_STATUSES",
    "LEARNING_JOB_TYPES",
    "SIGNAL_TYPES",
    "TARGET_AREAS",
    "CandidatePatch",
    "CandidatePatchStatus",
    "CandidatePatchType",
    "HarnessRequirement",
    "HumanReviewRequirement",
    "JsonValue",
    "LearningJob",
    "LearningJobStatus",
    "LearningJobType",
    "LearningSignal",
    "SignalType",
    "TargetArea",
    "canonical_learning_payload_hash",
    "make_candidate_patch_id",
    "make_learning_job_id",
    "make_learning_signal_id",
    "validate_json_safe",
]
