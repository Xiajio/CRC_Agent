from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from src.contracts.diagram import (
    DiagramExports,
    DiagramRuntime,
    DiagramSpec,
    DiagramValidationResult,
)


DiagramIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
DiagramActor = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
DiagramPrompt = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=12_000),
]
DiagramType = Literal["flowchart", "system_diagram"]
DiagramDirection = Literal["LR", "RL", "TB", "BT"]


class CompileDiagramRequest(BaseModel):
    prompt: DiagramPrompt
    requested_by: DiagramActor
    idempotency_key: DiagramIdentifier
    diagram_type: DiagramType
    direction: DiagramDirection
    deidentified: Literal[True]

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("deidentified", mode="before")
    @classmethod
    def require_explicit_deidentified_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("deidentified must be the JSON boolean true")
        return value


class CompileDiagramResponse(BaseModel):
    experiment_id: DiagramIdentifier
    spec: DiagramSpec
    validation: DiagramValidationResult
    exports: DiagramExports
    runtime: DiagramRuntime

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "CompileDiagramRequest",
    "CompileDiagramResponse",
    "DiagramActor",
    "DiagramDirection",
    "DiagramIdentifier",
    "DiagramPrompt",
    "DiagramType",
]
