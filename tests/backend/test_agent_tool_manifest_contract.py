from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.state import PlanStep
from src.tools.database_tools import ATOMIC_DATABASE_TOOLS
from src.tools.manifest import (
    ALLOWED_CATEGORIES,
    ALLOWED_GRAPH_SCOPES,
    ALLOWED_REGISTRIES,
    ALLOWED_ROUTE_TARGETS,
    ALLOWED_STATES,
    ToolSpec,
    build_tool_manifest_response,
    iter_tool_specs,
    project_tool_spec,
)


EXPECTED_GRAPH_TOOL_NAMES = {
    "PatientHistoryParserTool",
    "PolypDetectionTool",
    "PathologyParserTool",
    "VolumeCTSegmentorTool",
    "RectalMRStagerTool",
    "MolecularGuidelineTool",
    "tumor_screening_tool",
    "quick_tumor_check",
    "get_tumor_screening_status",
    "perform_comprehensive_tumor_check",
    "tumor_localization_tool",
    "batch_tumor_localization",
    "get_localization_status",
    "unet_segmentation_tool",
    "radiomics_feature_extraction_tool",
    "lasso_feature_selection_tool",
    "comprehensive_radiomics_analysis",
    "pathology_slide_classify",
    "quick_pathology_check",
    "get_pathology_clam_status",
    "perform_comprehensive_pathology_analysis",
    "search_clinical_guidelines",
    "search_treatment_recommendations",
    "search_staging_criteria",
    "search_drug_information",
    "list_guideline_toc",
    "read_guideline_chapter",
}

EXPECTED_GRAPH_WEB_TOOL_NAMES = {
    "search_clinical_guidelines",
    "search_treatment_recommendations",
    "search_staging_criteria",
    "search_drug_information",
    "list_guideline_toc",
    "read_guideline_chapter",
    "web_search",
    "search_clinical_evidence",
    "search_drug_online",
    "search_guideline_updates",
}

EXPECTED_OPTIONAL_TOOL_NAMES = {
    "search_by_guideline_source",
    "hybrid_guideline_search",
    "search_latest_research",
}

EXPECTED_EXECUTOR_ONLY_UTILITY_TOOL_NAMES = {"echo", "word_count"}


def _manifest_names() -> list[str]:
    return [spec.name for spec in iter_tool_specs()]


def _registry_names(registry: str) -> set[str]:
    return {
        spec.name
        for spec in iter_tool_specs()
        if registry in spec.registries
    }


def test_tool_manifest_names_are_unique() -> None:
    names = _manifest_names()

    assert names
    assert len(names) == len(set(names))


def test_tool_manifest_values_use_allowed_sets() -> None:
    for spec in iter_tool_specs():
        assert spec.category in ALLOWED_CATEGORIES
        assert spec.graph_scope in ALLOWED_GRAPH_SCOPES
        assert spec.state in ALLOWED_STATES
        assert set(spec.registries).issubset(ALLOWED_REGISTRIES)
        assert set(spec.route_targets).issubset(ALLOWED_ROUTE_TARGETS)


def test_tool_manifest_covers_planner_tool_types() -> None:
    planner_aliases = {
        alias
        for spec in iter_tool_specs()
        for alias in (spec.planner_aliases or ())
    }
    valid_tool_types = PlanStep.get_valid_tool_types()

    missing = valid_tool_types - planner_aliases
    extra = planner_aliases - valid_tool_types

    assert missing == {"ask_user"}
    assert extra == set()


def test_database_node_registry_matches_atomic_database_tools_bidirectionally() -> None:
    database_tool_names = {
        getattr(tool, "name", "")
        for tool in ATOMIC_DATABASE_TOOLS
        if getattr(tool, "name", "")
    }
    manifest_database_names = {
        spec.name
        for spec in iter_tool_specs()
        if "database_node" in spec.registries
    }

    assert manifest_database_names == database_tool_names


def test_graph_registry_matches_expected_tool_surface() -> None:
    assert _registry_names("graph") == EXPECTED_GRAPH_TOOL_NAMES


def test_graph_web_registry_matches_expected_tool_surface() -> None:
    assert _registry_names("graph_web") == EXPECTED_GRAPH_WEB_TOOL_NAMES


def test_optional_registry_matches_expected_candidate_surface() -> None:
    assert _registry_names("optional") == EXPECTED_OPTIONAL_TOOL_NAMES


def test_executor_only_utility_tools_have_exact_registry_and_scope() -> None:
    utility_specs = {
        spec.name: spec
        for spec in iter_tool_specs()
        if spec.name in EXPECTED_EXECUTOR_ONLY_UTILITY_TOOL_NAMES
    }

    assert set(utility_specs) == EXPECTED_EXECUTOR_ONLY_UTILITY_TOOL_NAMES
    for spec in utility_specs.values():
        assert spec.registries == ("executor",)
        assert spec.graph_scope == "executor_only"


def test_formatter_is_intentionally_absent_from_manifest_names() -> None:
    # formatter is a plain CardFormatter object appended by list_all_tools,
    # not a LangChain tool-callable name.
    assert "formatter" not in _manifest_names()


def test_manifest_projection_contains_only_safe_public_fields() -> None:
    response = build_tool_manifest_response(web_search_enabled=True)
    allowed_tool_keys = {
        "name",
        "category",
        "registries",
        "route_targets",
        "graph_scope",
        "planner_aliases",
        "requires_web",
        "available",
        "state",
    }
    forbidden_keys = {
        "factory_ref",
        "module",
        "module_path",
        "file_path",
        "path",
        "api_key",
        "token",
        "model_path",
        "notes",
    }

    assert response["tools"]
    for tool in response["tools"]:
        assert set(tool) == allowed_tool_keys
        assert forbidden_keys.isdisjoint(tool)


def test_manifest_groups_are_category_count_summaries() -> None:
    response = build_tool_manifest_response(web_search_enabled=False)
    tools = response["tools"]
    groups = response["groups"]

    assert isinstance(groups, list)
    assert groups
    assert [group["category"] for group in groups] == sorted(
        group["category"] for group in groups
    )

    for group in groups:
        assert set(group) == {"category", "count", "available_count"}
        category_tools = [
            tool for tool in tools if tool["category"] == group["category"]
        ]

        assert group["category"] in ALLOWED_CATEGORIES
        assert group["count"] == len(category_tools)
        assert group["available_count"] == sum(
            1 for tool in category_tools if tool["available"]
        )


def test_web_required_tools_become_unavailable_when_web_search_is_disabled() -> None:
    enabled_response = build_tool_manifest_response(web_search_enabled=True)
    disabled_response = build_tool_manifest_response(web_search_enabled=False)
    enabled_tools = {tool["name"]: tool for tool in enabled_response["tools"]}
    disabled_tools = {tool["name"]: tool for tool in disabled_response["tools"]}

    web_required_names = {
        spec.name
        for spec in iter_tool_specs()
        if spec.requires_web
    }

    assert web_required_names
    for name in web_required_names:
        assert enabled_tools[name]["available"] is True
        assert disabled_tools[name]["available"] is False


def test_disabled_tools_are_not_available_even_when_web_search_is_enabled() -> None:
    disabled_spec = ToolSpec(
        name="synthetic_disabled_tool",
        category=next(iter(ALLOWED_CATEGORIES)),
        registries=(),
        route_targets=(),
        graph_scope=next(iter(ALLOWED_GRAPH_SCOPES)),
        planner_aliases=(),
        requires_web=False,
        state="disabled",
    )

    projected = project_tool_spec(disabled_spec, web_search_enabled=True)

    assert projected["available"] is False


def test_manifest_import_does_not_load_heavy_runtime_modules() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    code = """
import sys
import src.tools.manifest
forbidden = {
    'src.rag.retriever',
    'torch',
    'cv2',
    'openslide',
}
loaded = sorted(name for name in forbidden if name in sys.modules)
if loaded:
    raise SystemExit('loaded forbidden modules: ' + ', '.join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr or result.stdout
