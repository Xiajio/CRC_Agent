from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


NonEmptyString = Annotated[str, Field(min_length=1)]


class ReleaseExecutionRequestPayload(BaseModel):
    intent_id: NonEmptyString
    requested_by: NonEmptyString
    idempotency_key: NonEmptyString
    reason: NonEmptyString
    expected_rollback_plan_id: NonEmptyString

    model_config = ConfigDict(extra="forbid")


class ReleaseExecutionResponse(BaseModel):
    governance: dict[str, Any]
    preflight: dict[str, Any]
    feature_flag_state: dict[str, Any] | None
    requests: list[dict[str, Any]]
    results: list[dict[str, Any]]
    audit_events: list[dict[str, Any]]
    integrity: dict[str, Any]
    runtime: dict[str, Any]

    model_config = ConfigDict(extra="forbid")
