from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.contracts.diagram import (
    DiagramCompileRequest,
    DiagramCompileResult,
    DiagramExports,
    DiagramRuntime,
    DiagramSpec,
    DiagramValidationResult,
    make_diagram_experiment_id,
)


def _request(**overrides: object) -> DiagramCompileRequest:
    payload = {
        "prompt": "采样后进入显微成像。",
        "requested_by": "admin_operator",
        "idempotency_key": "diagram-contract-001",
        "diagram_type": "flowchart",
        "direction": "LR",
        "deidentified": True,
    }
    payload.update(overrides)
    return DiagramCompileRequest(**payload)


def _spec_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "metadata": {
            "title": "检测流程",
            "purpose": "technical_presentation",
            "diagram_type": "flowchart",
            "aspect_ratio": "16:9",
            "language": "zh-CN",
        },
        "layout": {
            "direction": "LR",
            "algorithm": "layered",
            "node_spacing": 60,
            "group_spacing": 100,
        },
        "style": {
            "theme": "academic_light",
            "corner_radius": 10,
            "edge_routing": "orthogonal",
        },
        "groups": [{"id": "hardware", "label": "硬件采集"}],
        "nodes": [
            {
                "id": "sampling",
                "type": "process",
                "label": "气路采样",
                "group": "hardware",
                "order": 1,
                "ports": ["out"],
            },
            {
                "id": "imaging",
                "type": "process",
                "label": "显微成像",
                "group": "hardware",
                "order": 2,
                "ports": ["in"],
            },
        ],
        "edges": [
            {
                "id": "flow",
                "source": "sampling.out",
                "target": "imaging.in",
                "type": "data_flow",
                "label": "样本",
            }
        ],
        "constraints": [
            {"type": "preserve_order", "nodes": ["sampling", "imaging"]}
        ],
    }


def test_diagram_contract_round_trips_and_keeps_shadow_boundaries() -> None:
    request = _request()
    spec = DiagramSpec.model_validate(_spec_payload())
    result = DiagramCompileResult(
        experiment_id=make_diagram_experiment_id(request),
        spec=spec,
        exports=DiagramExports(mermaid="flowchart LR\n", dot="strict digraph Diagram {}\n"),
    )

    restored = DiagramCompileResult.model_validate(result.to_dict())
    payload = restored.to_dict()

    assert payload["spec"]["schema_version"] == "1.0"
    assert payload["validation"] == {"valid": True, "errors": [], "warnings": []}
    assert payload["runtime"] == {
        "mode": "shadow",
        "persisted": False,
        "renderer": "source_only",
        "applies_automatically": False,
        "clinical_state_mutated": False,
    }
    json.dumps(payload, ensure_ascii=False)


def test_diagram_experiment_id_is_deterministic_and_content_bound() -> None:
    request = _request()

    assert make_diagram_experiment_id(request) == make_diagram_experiment_id(_request())
    assert make_diagram_experiment_id(request) != make_diagram_experiment_id(
        _request(prompt="另一条合法流程描述。")
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("duplicate_node", "node ids must be unique"),
        ("unknown_node", "references unknown node"),
        ("unknown_port", "references unknown port"),
        ("isolated_node", "isolated nodes"),
        ("self_loop", "must not be a self-loop"),
        ("duplicate_relation", "duplicate edge relation"),
        ("unknown_constraint_node", "references unknown nodes"),
    ],
)
def test_diagram_spec_rejects_invalid_graph_semantics(case: str, message: str) -> None:
    payload = _spec_payload()
    nodes = payload["nodes"]
    edges = payload["edges"]
    constraints = payload["constraints"]
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    assert isinstance(constraints, list)

    if case == "duplicate_node":
        nodes.append(dict(nodes[0]))
    elif case == "unknown_node":
        edges[0]["source"] = "missing.out"
    elif case == "unknown_port":
        edges[0]["target"] = "imaging.missing"
    elif case == "isolated_node":
        nodes.append({"id": "orphan", "type": "note", "label": "孤立说明"})
    elif case == "self_loop":
        edges[0].update(source="sampling.out", target="sampling.out")
    elif case == "duplicate_relation":
        duplicate = dict(edges[0])
        duplicate["id"] = "flow_copy"
        edges.append(duplicate)
    elif case == "unknown_constraint_node":
        constraints[0]["nodes"] = ["sampling", "missing"]

    with pytest.raises(ValidationError, match=message):
        DiagramSpec.model_validate(payload)


def test_diagram_spec_allows_explicitly_marked_isolated_note() -> None:
    payload = _spec_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes.append(
        {
            "id": "legend",
            "type": "note",
            "label": "图例",
            "allow_isolated": True,
        }
    )

    spec = DiagramSpec.model_validate(payload)

    assert spec.nodes[-1].id == "legend"
    assert spec.nodes[-1].allow_isolated is True


def test_diagram_spec_enforces_node_limit() -> None:
    payload = _spec_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes.extend(
        {
            "id": f"note_{index}",
            "type": "note",
            "label": f"说明 {index}",
            "allow_isolated": True,
        }
        for index in range(49)
    )

    with pytest.raises(ValidationError, match="50"):
        DiagramSpec.model_validate(payload)


def test_diagram_contract_rejects_non_shadow_runtime_and_identifying_request() -> None:
    with pytest.raises(ValidationError):
        DiagramRuntime(persisted=True)
    with pytest.raises(ValidationError):
        _request(deidentified=False)
    with pytest.raises(ValidationError):
        _request(deidentified=1)


@pytest.mark.parametrize(
    "prompt",
    [
        "Patient ID: SYNTHETIC-MRN-123456 enters the analysis graph.",
        "患者姓名：测试患者甲进入分析流程。",
        "Contact synthetic-pii@example.invalid before compilation.",
        "联系电话：" + "1" + "3800000000",
    ],
)
def test_diagram_contract_rejects_apparent_identifiers(prompt: str) -> None:
    with pytest.raises(ValidationError, match="apparent patient identifiers"):
        _request(prompt=prompt)


def test_diagram_contract_rejects_control_characters_and_inconsistent_validation() -> None:
    with pytest.raises(ValidationError, match="control characters"):
        _request(prompt="采样\x00后进入分析")

    payload = _spec_payload()
    nodes = payload["nodes"]
    assert isinstance(nodes, list)
    nodes[0]["label"] = "采样\x1b"
    with pytest.raises(ValidationError, match="control characters"):
        DiagramSpec.model_validate(payload)

    with pytest.raises(ValidationError, match="valid must be true"):
        DiagramValidationResult(valid=True, errors=["broken"])
