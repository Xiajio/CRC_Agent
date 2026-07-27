from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


NonEmptyString = Annotated[str, Field(min_length=1)]
AutoResearchIdentifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
AutoResearchActor = Annotated[str, Field(min_length=1, max_length=128)]


class CohortFeasibilityRequestPayload(BaseModel):
    request_id: NonEmptyString
    project_id: NonEmptyString
    question: NonEmptyString
    cohort_criteria: dict[str, Any]
    data_scope: dict[str, Any]
    version_refs: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class CreateAutoResearchRunRequest(BaseModel):
    request_id: AutoResearchIdentifier
    project_id: AutoResearchIdentifier
    question: Annotated[str, Field(min_length=3, max_length=4000)]
    requested_by: AutoResearchActor
    idempotency_key: AutoResearchIdentifier
    max_sources: int = Field(default=8, ge=1, le=20)
    max_hypotheses: int = Field(default=3, ge=1, le=5)
    max_iterations: int = Field(default=2, ge=1, le=3)
    deidentified: Literal[True]

    model_config = ConfigDict(extra="forbid")

    @field_validator("deidentified", mode="before")
    @classmethod
    def require_explicit_deidentified_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("deidentified must be the JSON boolean true")
        return value
