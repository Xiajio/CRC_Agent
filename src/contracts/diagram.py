from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from src.contracts.deidentification import validate_deidentified_text


DiagramType = Literal["flowchart", "system_diagram"]
DiagramDirection = Literal["LR", "RL", "TB", "BT"]
DiagramNodeType = Literal[
    "process",
    "decision",
    "input",
    "output",
    "data",
    "model",
    "note",
]
DiagramEdgeType = Literal[
    "data_flow",
    "control_flow",
    "feedback",
    "constraint",
    "association",
]
DiagramConstraintType = Literal["same_rank", "preserve_order"]

_UNSAFE_CONTROL_CHARACTER_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


def reject_unsafe_control_characters(value: str) -> str:
    if _UNSAFE_CONTROL_CHARACTER_RE.search(value):
        raise ValueError("text must not contain unsafe control characters")
    return value

ElementIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
    ),
]
RequestIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
EndpointReference = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=129,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)?$",
    ),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
    AfterValidator(reject_unsafe_control_characters),
]


class DiagramCompileRequest(BaseModel):
    prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=3, max_length=12_000),
        AfterValidator(reject_unsafe_control_characters),
    ]
    requested_by: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
        AfterValidator(reject_unsafe_control_characters),
    ]
    idempotency_key: RequestIdentifier
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

    @field_validator("prompt")
    @classmethod
    def reject_apparent_identifiers(cls, value: str) -> str:
        return validate_deidentified_text("prompt", value)


class DiagramMetadata(BaseModel):
    title: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
        AfterValidator(reject_unsafe_control_characters),
    ]
    purpose: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
        AfterValidator(reject_unsafe_control_characters),
    ] = "technical_presentation"
    diagram_type: DiagramType
    aspect_ratio: Literal["16:9", "4:3", "A4_portrait", "A4_landscape"] = "16:9"
    language: Literal["zh-CN", "en-US"] = "zh-CN"

    model_config = ConfigDict(extra="forbid")


class DiagramLayout(BaseModel):
    direction: DiagramDirection
    algorithm: Literal["layered"] = "layered"
    node_spacing: int = Field(default=60, ge=20, le=200)
    group_spacing: int = Field(default=100, ge=40, le=320)

    model_config = ConfigDict(extra="forbid")


class DiagramStyle(BaseModel):
    theme: Literal["academic_light", "clinical_light", "monochrome"] = "academic_light"
    corner_radius: int = Field(default=10, ge=0, le=32)
    edge_routing: Literal["orthogonal", "polyline", "spline"] = "orthogonal"

    model_config = ConfigDict(extra="forbid")


class DiagramGroup(BaseModel):
    id: ElementIdentifier
    label: ShortText

    model_config = ConfigDict(extra="forbid")


class DiagramNode(BaseModel):
    id: ElementIdentifier
    type: DiagramNodeType = "process"
    label: ShortText
    group: ElementIdentifier | None = None
    order: int | None = Field(default=None, ge=0, le=10_000)
    ports: list[ElementIdentifier] = Field(default_factory=list, max_length=16)
    allow_isolated: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("ports")
    @classmethod
    def validate_unique_ports(cls, ports: list[str]) -> list[str]:
        if len(set(ports)) != len(ports):
            raise ValueError("node ports must be unique")
        return ports


class DiagramEdge(BaseModel):
    id: ElementIdentifier
    source: EndpointReference
    target: EndpointReference
    type: DiagramEdgeType = "data_flow"
    label: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
        AfterValidator(reject_unsafe_control_characters),
    ] | None = None

    model_config = ConfigDict(extra="forbid")


class DiagramConstraint(BaseModel):
    type: DiagramConstraintType
    nodes: list[ElementIdentifier] = Field(min_length=2, max_length=50)

    model_config = ConfigDict(extra="forbid")

    @field_validator("nodes")
    @classmethod
    def validate_unique_nodes(cls, nodes: list[str]) -> list[str]:
        if len(set(nodes)) != len(nodes):
            raise ValueError("constraint nodes must be unique")
        return nodes


class DiagramSpec(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    metadata: DiagramMetadata
    layout: DiagramLayout
    style: DiagramStyle = Field(default_factory=DiagramStyle)
    groups: list[DiagramGroup] = Field(default_factory=list, max_length=20)
    nodes: list[DiagramNode] = Field(min_length=1, max_length=50)
    edges: list[DiagramEdge] = Field(default_factory=list, max_length=100)
    constraints: list[DiagramConstraint] = Field(default_factory=list, max_length=50)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_graph_semantics(self) -> "DiagramSpec":
        group_ids = [group.id for group in self.groups]
        node_ids = [node.id for node in self.nodes]
        edge_ids = [edge.id for edge in self.edges]
        _require_unique(group_ids, "group ids")
        _require_unique(node_ids, "node ids")
        _require_unique(edge_ids, "edge ids")

        group_id_set = set(group_ids)
        node_by_id = {node.id: node for node in self.nodes}
        if group_id_set.intersection(node_by_id):
            raise ValueError("group ids and node ids must use separate namespaces")

        for node in self.nodes:
            if node.group is not None and node.group not in group_id_set:
                raise ValueError(
                    f"node {node.id!r} references unknown group {node.group!r}"
                )

        connected_nodes: set[str] = set()
        seen_edges: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            source_node, source_port = split_endpoint(edge.source)
            target_node, target_port = split_endpoint(edge.target)
            _validate_endpoint(source_node, source_port, node_by_id, edge.id, "source")
            _validate_endpoint(target_node, target_port, node_by_id, edge.id, "target")
            if source_node == target_node:
                raise ValueError(f"edge {edge.id!r} must not be a self-loop")
            edge_key = (edge.source, edge.target, edge.type)
            if edge_key in seen_edges:
                raise ValueError(
                    "duplicate edge relation: "
                    f"{edge.source!r} -> {edge.target!r} ({edge.type})"
                )
            seen_edges.add(edge_key)
            connected_nodes.update((source_node, target_node))

        isolated = [
            node.id
            for node in self.nodes
            if node.id not in connected_nodes and not node.allow_isolated
        ]
        if isolated:
            raise ValueError(
                "isolated nodes must set allow_isolated=true: " + ", ".join(isolated)
            )

        for constraint in self.constraints:
            unknown = [node_id for node_id in constraint.nodes if node_id not in node_by_id]
            if unknown:
                raise ValueError(
                    f"constraint {constraint.type!r} references unknown nodes: "
                    + ", ".join(unknown)
                )
        return self


class DiagramValidationResult(BaseModel):
    valid: bool = True
    errors: list[str] = Field(default_factory=list, max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=100)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_consistent_status(self) -> "DiagramValidationResult":
        if self.valid == bool(self.errors):
            raise ValueError("valid must be true exactly when errors is empty")
        return self


class DiagramExports(BaseModel):
    mermaid: Annotated[str, StringConstraints(min_length=1, max_length=200_000)]
    dot: Annotated[str, StringConstraints(min_length=1, max_length=200_000)]

    model_config = ConfigDict(extra="forbid")


class DiagramRuntime(BaseModel):
    mode: Literal["shadow"] = "shadow"
    persisted: Literal[False] = False
    renderer: Literal["source_only"] = "source_only"
    applies_automatically: Literal[False] = False
    clinical_state_mutated: Literal[False] = False

    model_config = ConfigDict(extra="forbid")


class DiagramCompileResult(BaseModel):
    experiment_id: RequestIdentifier
    spec: DiagramSpec
    validation: DiagramValidationResult = Field(default_factory=DiagramValidationResult)
    exports: DiagramExports
    runtime: DiagramRuntime = Field(default_factory=DiagramRuntime)

    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


def split_endpoint(endpoint: str) -> tuple[str, str | None]:
    node_id, separator, port_id = endpoint.partition(".")
    return node_id, port_id if separator else None


def make_diagram_experiment_id(request: DiagramCompileRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"diagram_experiment_{digest}"


def _require_unique(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _validate_endpoint(
    node_id: str,
    port_id: str | None,
    node_by_id: dict[str, DiagramNode],
    edge_id: str,
    endpoint_kind: str,
) -> None:
    node = node_by_id.get(node_id)
    if node is None:
        raise ValueError(
            f"edge {edge_id!r} {endpoint_kind} references unknown node {node_id!r}"
        )
    if port_id is not None and port_id not in node.ports:
        raise ValueError(
            f"edge {edge_id!r} {endpoint_kind} references unknown port "
            f"{node_id}.{port_id}"
        )


__all__ = [
    "DiagramCompileRequest",
    "DiagramCompileResult",
    "DiagramConstraint",
    "DiagramDirection",
    "DiagramEdge",
    "DiagramEdgeType",
    "DiagramExports",
    "DiagramGroup",
    "DiagramLayout",
    "DiagramMetadata",
    "DiagramNode",
    "DiagramNodeType",
    "DiagramRuntime",
    "DiagramSpec",
    "DiagramStyle",
    "DiagramType",
    "DiagramValidationResult",
    "make_diagram_experiment_id",
    "reject_unsafe_control_characters",
    "split_endpoint",
]
