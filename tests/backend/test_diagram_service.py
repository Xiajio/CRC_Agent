from __future__ import annotations

import pytest

from src.contracts.diagram import DiagramCompileRequest, DiagramSpec
from src.services.diagram_service import (
    DiagramGenerationError,
    DiagramOutputValidationError,
    DiagramService,
    DiagramServiceUnavailableError,
    compile_dot,
    compile_mermaid,
)


def _request(**overrides: object) -> DiagramCompileRequest:
    payload = {
        "prompt": "收集样本，然后进入模型分析。",
        "requested_by": "admin_operator",
        "idempotency_key": "diagram-service-001",
        "diagram_type": "flowchart",
        "direction": "LR",
        "deidentified": True,
    }
    payload.update(overrides)
    return DiagramCompileRequest(**payload)


def _spec_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "metadata": {
            "title": "样本分析",
            "diagram_type": "flowchart",
        },
        "layout": {"direction": "LR"},
        "nodes": [
            {
                "id": "collect",
                "type": "input",
                "label": '收集 "A" & <样本>',
                "ports": ["out"],
            },
            {
                "id": "analyze",
                "type": "model",
                "label": "模型分析",
                "ports": ["in"],
            },
        ],
        "edges": [
            {
                "id": "sample_flow",
                "source": "collect.out",
                "target": "analyze.in",
                "type": "data_flow",
                "label": "输入",
            }
        ],
    }
    payload.update(overrides)
    return payload


class StubReasoner:
    def __init__(self, result: DiagramSpec | dict[str, object]) -> None:
        self.result = result
        self.requests: list[DiagramCompileRequest] = []

    def generate_spec(self, request: DiagramCompileRequest):
        self.requests.append(request)
        return self.result


def test_diagram_service_uses_stub_reasoner_and_returns_source_only_result() -> None:
    reasoner = StubReasoner(_spec_payload())
    service = DiagramService(reasoner=reasoner)
    request = _request()

    result = service.compile(request)

    assert reasoner.requests == [request]
    assert result.spec.metadata.diagram_type == "flowchart"
    assert result.spec.layout.direction == "LR"
    assert result.runtime.persisted is False
    assert result.runtime.renderer == "source_only"
    assert result.runtime.clinical_state_mutated is False
    assert result.exports.mermaid.startswith("flowchart LR\n")
    assert result.exports.dot.startswith("digraph Diagram {\n")


def test_mermaid_and_dot_compilers_are_deterministic_and_escape_labels() -> None:
    spec = DiagramSpec.model_validate(_spec_payload())

    first_mermaid = compile_mermaid(spec)
    second_mermaid = compile_mermaid(spec)
    first_dot = compile_dot(spec)
    second_dot = compile_dot(spec)

    assert first_mermaid == second_mermaid
    assert first_dot == second_dot
    assert "&quot;A&quot; &amp; &lt;样本&gt;" in first_mermaid
    assert '收集 \\"A\\" & \\<样本\\>' in first_dot
    assert "p_7_collect_3_out -->|输入| p_7_analyze_2_in" in first_mermaid
    assert '"collect":"out" -> "analyze":"in" [label="输入"]' in first_dot


def test_mermaid_uses_renderer_local_ids_and_preserves_port_topology() -> None:
    payload = _spec_payload()
    nodes = payload["nodes"]
    edges = payload["edges"]
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    nodes[0]["id"] = "end"
    edges[0]["source"] = "end.out"
    edges[0]["label"] = "输入|管道"

    rendered = compile_mermaid(DiagramSpec.model_validate(payload))

    assert 'n_end(["收集 &quot;A&quot; &amp; &lt;样本&gt;"])' in rendered
    assert 'p_3_end_3_out(("out"))' in rendered
    assert "p_3_end_3_out -->|输入&#124;管道| p_7_analyze_2_in" in rendered
    assert "\n  end" not in rendered


def test_compilers_keep_parallel_relations_on_distinct_ports() -> None:
    payload = _spec_payload()
    nodes = payload["nodes"]
    edges = payload["edges"]
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    nodes[0]["ports"] = ["primary", "secondary"]
    nodes[1]["ports"] = ["left", "right"]
    edges[:] = [
        {
            "id": "primary_flow",
            "source": "collect.primary",
            "target": "analyze.left",
        },
        {
            "id": "secondary_flow",
            "source": "collect.secondary",
            "target": "analyze.right",
        },
    ]
    spec = DiagramSpec.model_validate(payload)

    mermaid = compile_mermaid(spec)
    dot = compile_dot(spec)

    assert "p_7_collect_7_primary --> p_7_analyze_4_left" in mermaid
    assert "p_7_collect_9_secondary --> p_7_analyze_5_right" in mermaid
    assert '"collect":"primary" -> "analyze":"left";' in dot
    assert '"collect":"secondary" -> "analyze":"right";' in dot


def test_mermaid_port_ids_do_not_collide_across_underscore_boundaries() -> None:
    payload = _spec_payload()
    payload["nodes"] = [
        {"id": "a_b", "type": "process", "label": "first", "ports": ["c"]},
        {"id": "a", "type": "process", "label": "second", "ports": ["b_c"]},
        {"id": "sink", "type": "output", "label": "sink", "ports": ["x", "y"]},
    ]
    payload["edges"] = [
        {"id": "first_flow", "source": "a_b.c", "target": "sink.x"},
        {"id": "second_flow", "source": "a.b_c", "target": "sink.y"},
    ]

    rendered = compile_mermaid(DiagramSpec.model_validate(payload))

    assert 'p_3_a_ub_1_c(("c"))' in rendered
    assert 'p_1_a_3_b_uc(("b_c"))' in rendered
    assert "p_3_a_ub_1_c --> p_4_sink_1_x" in rendered
    assert "p_1_a_3_b_uc --> p_4_sink_1_y" in rendered


def test_renderer_local_ids_safely_encode_allowed_hyphens() -> None:
    payload = _spec_payload()
    nodes = payload["nodes"]
    edges = payload["edges"]
    assert isinstance(nodes, list)
    assert isinstance(edges, list)
    payload["groups"] = [{"id": "input-group", "label": "Inputs"}]
    nodes[0]["id"] = "source-node"
    nodes[0]["group"] = "input-group"
    nodes[0]["ports"] = ["out-port"]
    edges[0]["source"] = "source-node.out-port"
    spec = DiagramSpec.model_validate(payload)

    mermaid = compile_mermaid(spec)
    dot = compile_dot(spec)

    assert 'subgraph g_input_hgroup["Inputs"]' in mermaid
    assert 'n_source_hnode(["' in mermaid
    assert 'p_11_source_hnode_8_out_hport(("out-port"))' in mermaid
    assert 'subgraph "cluster_input-group" {' in dot


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"metadata": {"title": "错误类型", "diagram_type": "system_diagram"}},
            "does not match the requested diagram_type",
        ),
        (
            {"layout": {"direction": "TB"}},
            "does not match the requested direction",
        ),
    ],
)
def test_diagram_service_rejects_output_that_changes_requested_shape(
    overrides: dict[str, object],
    message: str,
) -> None:
    service = DiagramService(reasoner=StubReasoner(_spec_payload(**overrides)))

    with pytest.raises(DiagramOutputValidationError, match=message):
        service.compile(_request())


def test_diagram_service_fails_closed_for_invalid_reasoner_output() -> None:
    invalid = _spec_payload()
    edges = invalid["edges"]
    assert isinstance(edges, list)
    edges[0]["target"] = "missing.in"
    service = DiagramService(reasoner=StubReasoner(invalid))

    with pytest.raises(DiagramOutputValidationError, match="DiagramSpec validation"):
        service.compile(_request())


def test_diagram_service_wraps_unexpected_reasoner_failure() -> None:
    class FailingReasoner:
        def generate_spec(self, request: DiagramCompileRequest):
            del request
            raise RuntimeError("provider exploded")

    service = DiagramService(reasoner=FailingReasoner())

    with pytest.raises(DiagramGenerationError, match="model generation failed"):
        service.compile(_request())


def test_diagram_service_requires_a_model_or_reasoner() -> None:
    with pytest.raises(DiagramServiceUnavailableError):
        DiagramService()
