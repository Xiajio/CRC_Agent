from __future__ import annotations

from dataclasses import dataclass

from src.nodes import knowledge_nodes, parallel_subagents
from src.policies.tool_targets import classify_pending_step_target
from src.prompts import planner_prompts
from src.state import PlanStep
from src.tools.database_tools import ATOMIC_DATABASE_TOOLS, list_database_tools


@dataclass
class FakeTool:
    name: str
    description: str = "fake tool"

    def invoke(self, payload):
        return {"tool": self.name, "payload": payload}


def _fake_rag_tools() -> list[FakeTool]:
    return [
        FakeTool("search_clinical_guidelines"),
        FakeTool("search_treatment_recommendations"),
        FakeTool("search_staging_criteria"),
        FakeTool("search_drug_information"),
        FakeTool("list_guideline_toc"),
        FakeTool("read_guideline_chapter"),
        FakeTool("web_search"),
    ]


def test_plan_tool_types_have_route_targets() -> None:
    for tool_type in PlanStep.get_valid_tool_types():
        target = classify_pending_step_target(tool_type)
        assert target in {
            "assessment",
            "case_database",
            "knowledge",
            "rad_agent",
            "path_agent",
            "tool_executor",
        }, tool_type


def test_calculator_is_not_advertised_without_a_concrete_tool() -> None:
    assert "calculator" not in PlanStep.get_valid_tool_types()
    assert "calculator" not in planner_prompts.PLANNER_SYSTEM_PROMPT


def test_tool_executor_is_not_advertised_as_a_planner_tool() -> None:
    assert "tool_executor" not in PlanStep.get_valid_tool_types()
    assert "tool_executor" not in planner_prompts.PLANNER_SYSTEM_PROMPT
    assert classify_pending_step_target("tool_executor") != "tool_executor"


def test_rag_planner_contract_matches_dispatch_aliases() -> None:
    valid_tool_types = PlanStep.get_valid_tool_types()

    assert {
        "search_clinical_guidelines",
        "search_treatment_recommendations",
        "search_staging_criteria",
        "search_drug_information",
        "search",
    }.issubset(valid_tool_types)

    prompt = planner_prompts.PLANNER_SYSTEM_PROMPT
    assert "search_clinical_guidelines" in prompt
    assert "search_treatment_recommendations" in prompt
    assert "search_staging_criteria" in prompt
    assert "search_drug_information" in prompt
    assert "search_treatment_recommendations（别名：search）" not in prompt

    selected = knowledge_nodes._select_plan_rag_tool("search", _fake_rag_tools())
    assert selected is not None
    assert selected.name == "search_clinical_guidelines"


def test_search_alias_is_an_explicit_guideline_rag_alias() -> None:
    assert "search" in knowledge_nodes._EXPLICIT_RAG_TOOL_TYPES
    selected = knowledge_nodes._select_plan_rag_tool("search", _fake_rag_tools())
    assert selected is not None
    assert selected.name == "search_clinical_guidelines"


def test_explicit_rag_plan_values_select_matching_tools() -> None:
    tools = _fake_rag_tools()

    expected = {
        "search_treatment_recommendations": "search_treatment_recommendations",
        "search_staging_criteria": "search_staging_criteria",
        "search_drug_information": "search_drug_information",
        "search_clinical_guidelines": "search_clinical_guidelines",
        "search": "search_clinical_guidelines",
    }

    for tool_type, tool_name in expected.items():
        selected = knowledge_nodes._select_plan_rag_tool(tool_type, tools)
        assert selected is not None
        assert selected.name == tool_name


def test_guideline_structure_plan_values_select_matching_tools() -> None:
    tools = _fake_rag_tools()

    expected = {
        "list_guideline_toc": "list_guideline_toc",
        "toc": "list_guideline_toc",
        "read_guideline_chapter": "read_guideline_chapter",
        "read": "read_guideline_chapter",
        "chapter": "read_guideline_chapter",
    }

    for tool_type, tool_name in expected.items():
        selected = knowledge_nodes._select_plan_structural_tool(tool_type, tools)
        assert selected is not None
        assert selected.name == tool_name


def test_rag_invocation_helper_passes_top_k_when_supported() -> None:
    tool = FakeTool("search_treatment_recommendations")

    result = knowledge_nodes._invoke_rag_search_tool(tool, "stage iii crc", top_k=6)

    assert result == {
        "tool": "search_treatment_recommendations",
        "payload": {"query": "stage iii crc", "top_k": 6},
    }


def test_parallel_case_database_step_uses_atomic_database_tools(monkeypatch) -> None:
    database_tool = FakeTool("get_patient_case_info")
    monkeypatch.setattr(
        parallel_subagents,
        "ATOMIC_DATABASE_TOOLS",
        [database_tool],
        raising=False,
    )

    step = PlanStep(
        id="step_1",
        description="Fetch the patient case.",
        tool_needed="case_database_query",
        status="pending",
        assignee="case_database",
    )

    selected = parallel_subagents._select_tools_for_step(step, tools=[])

    assert [tool.name for tool in selected] == ["get_patient_case_info"]


def test_parallel_case_database_merges_atomic_tools_with_non_database_matches(monkeypatch) -> None:
    database_tool = FakeTool("get_patient_case_info")
    pathology_tool = FakeTool("pathology_slide_classify")
    monkeypatch.setattr(
        parallel_subagents,
        "ATOMIC_DATABASE_TOOLS",
        [database_tool],
        raising=False,
    )

    step = PlanStep(
        id="step_1",
        description="Fetch the patient case.",
        tool_needed="case_database_query",
        status="pending",
        assignee="case_database",
    )

    selected = parallel_subagents._select_tools_for_step(step, tools=[pathology_tool])

    assert [tool.name for tool in selected] == ["pathology_slide_classify", "get_patient_case_info"]


def test_list_database_tools_matches_atomic_database_tools() -> None:
    assert set(list_database_tools()) == {
        getattr(tool, "name", "")
        for tool in ATOMIC_DATABASE_TOOLS
        if getattr(tool, "name", "")
    }


def test_basic_utility_tools_are_executor_only(monkeypatch) -> None:
    from src import tools as tool_registry
    from src.tools import database_tools

    monkeypatch.setattr(tool_registry, "list_clinical_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "get_enhanced_rag_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "get_all_web_search_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "get_clinical_web_search_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "get_database_tools", lambda: [])
    monkeypatch.setattr(database_tools, "get_database_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "get_tumor_screening_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "get_tumor_localization_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "list_radiomics_tools", lambda: [])
    monkeypatch.setattr(tool_registry, "get_pathology_clam_tools", lambda: [])

    def tool_names(tools) -> set[str]:
        return {getattr(tool, "name", "") for tool in tools}

    utility_tool_names = {"word_count", "echo"}

    assert utility_tool_names.issubset(tool_names(tool_registry.list_all_tools()))
    assert utility_tool_names.isdisjoint(tool_names(tool_registry.list_tools()))
    assert utility_tool_names.isdisjoint(
        tool_names(tool_registry.list_tools_with_web_search())
    )
