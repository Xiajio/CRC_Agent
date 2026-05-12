from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.graph_builder import (
    NodeName,
    _instrument_node,
    _reset_node_timings_at_turn_start,
)
from src.state import CRCAgentState


def test_planner_instrumentation_does_not_add_second_timing() -> None:
    planner_record = {"node": "planner", "duration_ms": 1.0}

    def planner_like_node(_state: CRCAgentState) -> dict[str, Any]:
        return {"node_timings": [planner_record]}

    wrapped = _instrument_node(NodeName.PLANNER, planner_like_node)

    result = wrapped(CRCAgentState(messages=[]))

    assert result["node_timings"] == [planner_record]


def test_new_graph_turn_does_not_inherit_previous_node_timings() -> None:
    def start_node(_state: CRCAgentState) -> dict[str, Any]:
        return {}

    def assessment_node(_state: CRCAgentState) -> dict[str, Any]:
        return {}

    builder = StateGraph(CRCAgentState)
    builder.add_node("start", _reset_node_timings_at_turn_start(start_node))
    builder.add_node("assessment", _instrument_node(NodeName.ASSESSMENT, assessment_node))
    builder.set_entry_point("start")
    builder.add_edge("start", "assessment")
    builder.add_edge("assessment", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [],
            "node_timings": [
                {"node": "planner", "duration_ms": 11.0},
                {"node": "citation", "duration_ms": 22.0},
            ],
        }
    )

    assert [timing["node"] for timing in result["node_timings"]] == ["assessment"]
