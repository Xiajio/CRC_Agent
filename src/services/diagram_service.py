from __future__ import annotations

import html
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.contracts.diagram import (
    DiagramCompileRequest,
    DiagramCompileResult,
    DiagramEdge,
    DiagramExports,
    DiagramNode,
    DiagramRuntime,
    DiagramSpec,
    DiagramValidationResult,
    make_diagram_experiment_id,
    split_endpoint,
)


class DiagramServiceUnavailableError(RuntimeError):
    """Raised when no diagram reasoning model is configured."""


class DiagramGenerationError(RuntimeError):
    """Raised when the model cannot produce a diagram candidate."""


class DiagramOutputValidationError(RuntimeError):
    """Raised when a model-produced diagram fails the strict contract."""


class DiagramReasoner(Protocol):
    def generate_spec(self, request: DiagramCompileRequest) -> DiagramSpec | dict[str, Any]: ...


class LLMDiagramReasoner:
    def __init__(self, model: Any) -> None:
        self.model = model

    def generate_spec(self, request: DiagramCompileRequest) -> DiagramSpec:
        system_prompt = (
            "You convert deidentified technical descriptions into a strict DiagramSpec. "
            "Preserve the stated topology and labels; never invent clinical facts, file paths, "
            "URLs, rendering directives, executable content, or hidden nodes. Every non-isolated "
            "node must participate in an edge, every endpoint port must be declared, and all IDs "
            "must be stable ASCII identifiers. Return only the structured object."
        )
        human_prompt = (
            f"Diagram type: {request.diagram_type}\n"
            f"Required direction: {request.direction}\n"
            "The input is asserted to be deidentified. Create the smallest complete graph that "
            "faithfully represents it.\n\n"
            f"Description:\n{request.prompt}"
        )
        try:
            response = self.model.with_structured_output(
                DiagramSpec,
                method="function_calling",
            ).invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            )
        except Exception as exc:
            raise DiagramGenerationError("diagram model generation failed") from exc
        return _coerce_spec(response)


class DiagramService:
    def __init__(
        self,
        model: Any | None = None,
        *,
        reasoner: DiagramReasoner | None = None,
    ) -> None:
        if reasoner is not None:
            self.reasoner = reasoner
        elif model is not None:
            self.reasoner = LLMDiagramReasoner(model)
        else:
            raise DiagramServiceUnavailableError(
                "a diagram reasoning model or reasoner must be configured"
            )

    def compile(self, request: DiagramCompileRequest) -> DiagramCompileResult:
        try:
            request = DiagramCompileRequest.model_validate(request)
        except ValidationError as exc:
            raise ValueError("diagram compile request is invalid") from exc

        try:
            raw_spec = self.reasoner.generate_spec(request)
        except (DiagramGenerationError, DiagramOutputValidationError):
            raise
        except Exception as exc:
            raise DiagramGenerationError("diagram model generation failed") from exc

        try:
            spec = _coerce_spec(raw_spec)
        except DiagramOutputValidationError:
            raise
        if spec.metadata.diagram_type != request.diagram_type:
            raise DiagramOutputValidationError(
                "diagram output type does not match the requested diagram_type"
            )
        if spec.layout.direction != request.direction:
            raise DiagramOutputValidationError(
                "diagram output direction does not match the requested direction"
            )

        warnings = _review_warnings(spec)
        return DiagramCompileResult(
            experiment_id=make_diagram_experiment_id(request),
            spec=spec,
            validation=DiagramValidationResult(warnings=warnings),
            exports=DiagramExports(
                mermaid=compile_mermaid(spec),
                dot=compile_dot(spec),
            ),
            runtime=DiagramRuntime(),
        )


def compile_mermaid(spec: DiagramSpec) -> str:
    lines = [f"flowchart {spec.layout.direction}"]
    grouped_node_ids: set[str] = set()
    nodes_by_group: dict[str, list[DiagramNode]] = {group.id: [] for group in spec.groups}
    for node in spec.nodes:
        if node.group is not None:
            nodes_by_group[node.group].append(node)
            grouped_node_ids.add(node.id)

    for group in spec.groups:
        lines.append(
            f'  subgraph {_mermaid_group_id(group.id)}["{_escape_mermaid(group.label)}"]'
        )
        for node in nodes_by_group[group.id]:
            lines.extend("    " + item for item in _mermaid_node_declarations(node))
        lines.append("  end")
    for node in spec.nodes:
        if node.id not in grouped_node_ids:
            lines.extend("  " + item for item in _mermaid_node_declarations(node))
    for edge in spec.edges:
        lines.append("  " + _mermaid_edge(edge))
    return "\n".join(lines) + "\n"


def compile_dot(spec: DiagramSpec) -> str:
    rankdir = spec.layout.direction
    splines = {
        "orthogonal": "ortho",
        "polyline": "polyline",
        "spline": "spline",
    }[spec.style.edge_routing]
    node_sep = spec.layout.node_spacing / 72
    rank_sep = spec.layout.group_spacing / 72
    lines = [
        "digraph Diagram {",
        (
            f'  graph [rankdir="{rankdir}", splines="{splines}", '
            f'nodesep="{node_sep:.3f}", ranksep="{rank_sep:.3f}"];'
        ),
        '  node [shape="box"];',
    ]
    grouped_node_ids: set[str] = set()
    nodes_by_group: dict[str, list[DiagramNode]] = {group.id: [] for group in spec.groups}
    for node in spec.nodes:
        if node.group is not None:
            nodes_by_group[node.group].append(node)
            grouped_node_ids.add(node.id)

    for group in spec.groups:
        lines.extend(
            [
                f'  subgraph "cluster_{_escape_dot(group.id)}" {{',
                f'    label="{_escape_dot(group.label)}";',
            ]
        )
        for node in nodes_by_group[group.id]:
            lines.append("    " + _dot_node(node))
        lines.append("  }")
    for node in spec.nodes:
        if node.id not in grouped_node_ids:
            lines.append("  " + _dot_node(node))
    for constraint in spec.constraints:
        quoted_nodes = "; ".join(f'"{node_id}"' for node_id in constraint.nodes)
        if constraint.type == "same_rank":
            lines.append(f"  {{ rank=same; {quoted_nodes}; }}")
        elif constraint.type == "preserve_order":
            for source, target in zip(constraint.nodes, constraint.nodes[1:]):
                lines.append(
                    f'  "{source}" -> "{target}" [style="invis", weight="100"];'
                )
    for edge in spec.edges:
        lines.append("  " + _dot_edge(edge))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _coerce_spec(value: Any) -> DiagramSpec:
    try:
        if isinstance(value, DiagramSpec):
            return DiagramSpec.model_validate(value.model_dump(mode="python"))
        if isinstance(value, dict):
            return DiagramSpec.model_validate(value)
    except ValidationError as exc:
        raise DiagramOutputValidationError(
            "diagram model output failed DiagramSpec validation"
        ) from exc
    raise DiagramOutputValidationError(
        "diagram model output must be a DiagramSpec or JSON object"
    )


def _review_warnings(spec: DiagramSpec) -> list[str]:
    warnings: list[str] = []
    if not spec.constraints and len(spec.nodes) > 4:
        warnings.append("diagram has more than four nodes but no explicit layout constraints")
    long_labels = [node.id for node in spec.nodes if len(node.label) > 80]
    if long_labels:
        warnings.append("long node labels may require visual review: " + ", ".join(long_labels))
    return warnings


def _escape_mermaid(value: str) -> str:
    normalized = " ".join(value.split())
    return html.escape(normalized, quote=True).replace("|", "&#124;")


def _escape_dot(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized.replace("\\", "\\\\").replace('"', '\\"')


def _escape_dot_record(value: str) -> str:
    escaped = _escape_dot(value)
    for character in ("{", "}", "|", "<", ">"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _mermaid_node(node: DiagramNode) -> str:
    node_id = _mermaid_node_id(node.id)
    label = _escape_mermaid(node.label)
    if node.type == "decision":
        return f'{node_id}{{"{label}"}}'
    if node.type in {"input", "output"}:
        return f'{node_id}(["{label}"])'
    if node.type == "data":
        return f'{node_id}[("{label}")]'
    if node.type == "model":
        return f'{node_id}[["{label}"]]'
    return f'{node_id}["{label}"]'


def _mermaid_node_declarations(node: DiagramNode) -> list[str]:
    declarations = [_mermaid_node(node)]
    node_id = _mermaid_node_id(node.id)
    for port_id in node.ports:
        rendered_port_id = _mermaid_port_id(node.id, port_id)
        declarations.append(f'{rendered_port_id}(("{_escape_mermaid(port_id)}"))')
        declarations.append(f"{node_id} --- {rendered_port_id}")
    return declarations


def _mermaid_edge(edge: DiagramEdge) -> str:
    source_id = _mermaid_endpoint_id(edge.source)
    target_id = _mermaid_endpoint_id(edge.target)
    arrow = {
        "data_flow": "-->",
        "control_flow": "-->",
        "feedback": "-.->",
        "constraint": "-.->",
        "association": "---",
    }[edge.type]
    if edge.label:
        return f"{source_id} {arrow}|{_escape_mermaid(edge.label)}| {target_id}"
    return f"{source_id} {arrow} {target_id}"


def _mermaid_node_id(node_id: str) -> str:
    return f"n_{_mermaid_id_component(node_id)}"


def _mermaid_group_id(group_id: str) -> str:
    return f"g_{_mermaid_id_component(group_id)}"


def _mermaid_port_id(node_id: str, port_id: str) -> str:
    # Length-prefix both components so endpoint pairs cannot collapse when
    # underscores appear on either side of the node/port boundary.
    rendered_node_id = _mermaid_id_component(node_id)
    rendered_port_id = _mermaid_id_component(port_id)
    return f"p_{len(node_id)}_{rendered_node_id}_{len(port_id)}_{rendered_port_id}"


def _mermaid_id_component(value: str) -> str:
    # ElementIdentifier permits '_' and '-'. Encode both so renderer-local IDs
    # contain only alphanumerics/underscores without introducing collisions.
    return value.replace("_", "_u").replace("-", "_h")


def _mermaid_endpoint_id(endpoint: str) -> str:
    node_id, port_id = split_endpoint(endpoint)
    if port_id is not None:
        return _mermaid_port_id(node_id, port_id)
    return _mermaid_node_id(node_id)


def _dot_node(node: DiagramNode) -> str:
    if node.ports:
        port_fields = "|".join(
            f"<{port_id}> {_escape_dot_record(port_id)}" for port_id in node.ports
        )
        label = f"{{{_escape_dot_record(node.label)}|{{{port_fields}}}}}"
        return f'"{node.id}" [label="{label}", shape="record"];'
    shape = {
        "process": "box",
        "decision": "diamond",
        "input": "oval",
        "output": "oval",
        "data": "cylinder",
        "model": "component",
        "note": "note",
    }[node.type]
    return f'"{node.id}" [label="{_escape_dot(node.label)}", shape="{shape}"];'


def _dot_edge(edge: DiagramEdge) -> str:
    attributes: list[str] = []
    if edge.label:
        attributes.append(f'label="{_escape_dot(edge.label)}"')
    if edge.type in {"feedback", "constraint"}:
        attributes.append('style="dashed"')
    if edge.type == "association":
        attributes.append('dir="none"')
    suffix = f" [{', '.join(attributes)}]" if attributes else ""
    return f"{_dot_endpoint(edge.source)} -> {_dot_endpoint(edge.target)}{suffix};"


def _dot_endpoint(endpoint: str) -> str:
    node_id, port_id = split_endpoint(endpoint)
    if port_id is not None:
        return f'"{node_id}":"{port_id}"'
    return f'"{node_id}"'


__all__ = [
    "DiagramGenerationError",
    "DiagramOutputValidationError",
    "DiagramReasoner",
    "DiagramService",
    "DiagramServiceUnavailableError",
    "LLMDiagramReasoner",
    "compile_dot",
    "compile_mermaid",
]
