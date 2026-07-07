from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


NonEmptyString = Annotated[str, Field(min_length=1)]
ReleaseClosureStatus = Literal[
    "accepted",
    "accepted_with_observations",
    "rolled_back",
]


class ReleaseClosureRequest(BaseModel):
    intent_id: NonEmptyString
    release_execution_id: NonEmptyString
    closure_status: ReleaseClosureStatus
    closed_by: NonEmptyString
    rationale: NonEmptyString
    idempotency_key: NonEmptyString

    model_config = ConfigDict(extra="forbid")
