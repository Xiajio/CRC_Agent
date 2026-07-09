from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


NonEmptyString = Annotated[str, Field(min_length=1)]


class CohortFeasibilityRequestPayload(BaseModel):
    request_id: NonEmptyString
    project_id: NonEmptyString
    question: NonEmptyString
    cohort_criteria: dict[str, Any]
    data_scope: dict[str, Any]
    version_refs: dict[str, Any]

    model_config = ConfigDict(extra="forbid")
