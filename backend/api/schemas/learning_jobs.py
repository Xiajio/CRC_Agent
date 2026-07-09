from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


NonEmptyString = Annotated[str, Field(min_length=1)]


class LearningSignalPayload(BaseModel):
    signal_type: NonEmptyString
    source_ref: dict[str, Any]
    reason_code: NonEmptyString
    target_area: NonEmptyString
    severity: NonEmptyString
    summary: NonEmptyString
    deidentified: bool
    created_at: NonEmptyString

    model_config = ConfigDict(extra="forbid")

    @field_validator("deidentified")
    @classmethod
    def deidentified_must_be_true(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("deidentified must be true")
        return value


class CreateLearningJobRequest(BaseModel):
    signals: list[LearningSignalPayload] = Field(min_length=1)
    requested_by: NonEmptyString
    idempotency_key: NonEmptyString

    model_config = ConfigDict(extra="forbid")
