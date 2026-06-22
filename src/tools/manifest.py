"""Static metadata catalog for agent tool surfaces.

This module intentionally does not import runtime tool factories. It is used by
admin/API surfaces that need safe metadata without loading models, retrievers,
database services, or network-backed tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


ToolCategory: TypeAlias = Literal[
    "clinical",
    "rag",
    "web",
    "database",
    "imaging",
    "pathology",
    "tumor",
    "utility",
    "formatting",
]
ToolRegistry: TypeAlias = Literal[
    "graph",
    "graph_web",
    "executor",
    "database_node",
    "optional",
]
RouteTarget: TypeAlias = Literal[
    "knowledge",
    "case_database",
    "rad_agent",
    "path_agent",
    "web_search",
    "tool_executor",
    "decision",
]
GraphScope: TypeAlias = Literal[
    "doctor",
    "patient",
    "both",
    "node_local",
    "executor_only",
]
ToolState: TypeAlias = Literal[
    "available",
    "candidate",
    "internal",
    "disabled",
]


ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {
        "clinical",
        "rag",
        "web",
        "database",
        "imaging",
        "pathology",
        "tumor",
        "utility",
        "formatting",
    }
)
ALLOWED_REGISTRIES: frozenset[str] = frozenset(
    {"graph", "graph_web", "executor", "database_node", "optional"}
)
ALLOWED_ROUTE_TARGETS: frozenset[str] = frozenset(
    {
        "knowledge",
        "case_database",
        "rad_agent",
        "path_agent",
        "web_search",
        "tool_executor",
        "decision",
    }
)
ALLOWED_GRAPH_SCOPES: frozenset[str] = frozenset(
    {"doctor", "patient", "both", "node_local", "executor_only"}
)
ALLOWED_STATES: frozenset[str] = frozenset(
    {"available", "candidate", "internal", "disabled"}
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: ToolCategory
    registries: tuple[ToolRegistry, ...]
    route_targets: tuple[RouteTarget, ...]
    graph_scope: GraphScope
    planner_aliases: tuple[str, ...] = ()
    requires_web: bool = False
    patient_safe: bool = False
    heavy_dependency: bool = False
    state: ToolState = "available"
    notes: str = ""


_GRAPH_EXECUTOR = ("graph", "executor")
_RAG_REGISTRIES = ("graph", "graph_web", "executor")
_WEB_REGISTRIES = ("graph_web", "executor")
_DATABASE_REGISTRIES = ("database_node", "executor")


_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "PatientHistoryParserTool",
        "clinical",
        _GRAPH_EXECUTOR,
        ("decision", "tool_executor"),
        "both",
        patient_safe=True,
    ),
    ToolSpec(
        "PolypDetectionTool",
        "clinical",
        _GRAPH_EXECUTOR,
        ("decision", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "PathologyParserTool",
        "clinical",
        _GRAPH_EXECUTOR,
        ("path_agent", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "VolumeCTSegmentorTool",
        "clinical",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
        planner_aliases=("ct_analysis",),
        heavy_dependency=True,
    ),
    ToolSpec(
        "RectalMRStagerTool",
        "clinical",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
        planner_aliases=("imaging_analysis", "radiology"),
        heavy_dependency=True,
    ),
    ToolSpec(
        "MolecularGuidelineTool",
        "clinical",
        _GRAPH_EXECUTOR,
        ("knowledge", "decision", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "search_clinical_guidelines",
        "rag",
        _RAG_REGISTRIES,
        ("knowledge", "decision"),
        "both",
        planner_aliases=("search_clinical_guidelines", "search"),
        patient_safe=True,
    ),
    ToolSpec(
        "search_treatment_recommendations",
        "rag",
        _RAG_REGISTRIES,
        ("knowledge", "decision"),
        "both",
        planner_aliases=("search_treatment_recommendations",),
        patient_safe=True,
    ),
    ToolSpec(
        "search_staging_criteria",
        "rag",
        _RAG_REGISTRIES,
        ("knowledge",),
        "both",
        planner_aliases=("search_staging_criteria",),
        patient_safe=True,
    ),
    ToolSpec(
        "search_drug_information",
        "rag",
        _RAG_REGISTRIES,
        ("knowledge",),
        "both",
        planner_aliases=("search_drug_information",),
        patient_safe=True,
    ),
    ToolSpec(
        "list_guideline_toc",
        "rag",
        _RAG_REGISTRIES,
        ("knowledge",),
        "both",
        planner_aliases=("list_guideline_toc", "toc"),
        patient_safe=True,
    ),
    ToolSpec(
        "read_guideline_chapter",
        "rag",
        _RAG_REGISTRIES,
        ("knowledge",),
        "both",
        planner_aliases=("read_guideline_chapter", "read", "chapter"),
        patient_safe=True,
    ),
    ToolSpec(
        "search_by_guideline_source",
        "rag",
        ("optional",),
        ("knowledge",),
        "node_local",
        state="candidate",
    ),
    ToolSpec(
        "hybrid_guideline_search",
        "rag",
        ("optional",),
        ("knowledge",),
        "node_local",
        state="candidate",
    ),
    ToolSpec(
        "web_search",
        "web",
        _WEB_REGISTRIES,
        ("web_search", "decision"),
        "both",
        planner_aliases=("web_search", "web"),
        requires_web=True,
        patient_safe=True,
    ),
    ToolSpec(
        "search_clinical_evidence",
        "web",
        _WEB_REGISTRIES,
        ("web_search", "knowledge", "decision"),
        "doctor",
        requires_web=True,
    ),
    ToolSpec(
        "search_drug_online",
        "web",
        _WEB_REGISTRIES,
        ("web_search", "knowledge"),
        "doctor",
        requires_web=True,
    ),
    ToolSpec(
        "search_guideline_updates",
        "web",
        _WEB_REGISTRIES,
        ("web_search", "knowledge"),
        "doctor",
        requires_web=True,
    ),
    ToolSpec(
        "search_latest_research",
        "web",
        ("executor", "optional"),
        ("web_search", "knowledge"),
        "executor_only",
        requires_web=True,
        state="candidate",
    ),
    ToolSpec(
        "get_patient_case_info",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database",),
        "executor_only",
    ),
    ToolSpec(
        "summarize_patient_existing_info",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database",),
        "executor_only",
    ),
    ToolSpec(
        "upsert_patient_info",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database",),
        "executor_only",
    ),
    ToolSpec(
        "get_patient_imaging",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database", "rad_agent"),
        "executor_only",
    ),
    ToolSpec(
        "get_patient_pathology_slides",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database", "path_agent"),
        "executor_only",
    ),
    ToolSpec(
        "get_database_statistics",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database",),
        "executor_only",
    ),
    ToolSpec(
        "search_cases",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database",),
        "executor_only",
        planner_aliases=("database_query", "case_database_query"),
    ),
    ToolSpec(
        "list_imaging_folders",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database", "rad_agent"),
        "executor_only",
    ),
    ToolSpec(
        "get_random_case",
        "database",
        _DATABASE_REGISTRIES,
        ("case_database",),
        "executor_only",
    ),
    ToolSpec(
        "perform_comprehensive_tumor_check",
        "tumor",
        ("graph", "executor", "database_node"),
        ("case_database", "decision", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "tumor_screening_tool",
        "tumor",
        _GRAPH_EXECUTOR,
        ("decision", "tool_executor"),
        "doctor",
        planner_aliases=("tumor_detection", "tumor_screening"),
        heavy_dependency=True,
    ),
    ToolSpec(
        "quick_tumor_check",
        "tumor",
        _GRAPH_EXECUTOR,
        ("decision", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "get_tumor_screening_status",
        "tumor",
        _GRAPH_EXECUTOR,
        ("decision", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "tumor_localization_tool",
        "tumor",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
        heavy_dependency=True,
    ),
    ToolSpec(
        "batch_tumor_localization",
        "tumor",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
        heavy_dependency=True,
    ),
    ToolSpec(
        "get_localization_status",
        "tumor",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "unet_segmentation_tool",
        "imaging",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
        heavy_dependency=True,
    ),
    ToolSpec(
        "radiomics_feature_extraction_tool",
        "imaging",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
        heavy_dependency=True,
    ),
    ToolSpec(
        "lasso_feature_selection_tool",
        "imaging",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "comprehensive_radiomics_analysis",
        "imaging",
        _GRAPH_EXECUTOR,
        ("rad_agent", "tool_executor"),
        "doctor",
        heavy_dependency=True,
    ),
    ToolSpec(
        "pathology_slide_classify",
        "pathology",
        _GRAPH_EXECUTOR,
        ("path_agent", "tool_executor"),
        "doctor",
        planner_aliases=("pathology_analysis", "pathology", "clam"),
        heavy_dependency=True,
    ),
    ToolSpec(
        "quick_pathology_check",
        "pathology",
        _GRAPH_EXECUTOR,
        ("path_agent", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "get_pathology_clam_status",
        "pathology",
        _GRAPH_EXECUTOR,
        ("path_agent", "tool_executor"),
        "doctor",
    ),
    ToolSpec(
        "perform_comprehensive_pathology_analysis",
        "pathology",
        _GRAPH_EXECUTOR,
        ("path_agent", "tool_executor"),
        "doctor",
        heavy_dependency=True,
    ),
    ToolSpec(
        "echo",
        "utility",
        ("executor",),
        ("tool_executor",),
        "executor_only",
        state="internal",
    ),
    ToolSpec(
        "word_count",
        "utility",
        ("executor",),
        ("tool_executor",),
        "executor_only",
        state="internal",
    ),
)


def iter_tool_specs() -> tuple[ToolSpec, ...]:
    """Return the static manifest entries in deterministic display order."""

    return _TOOL_SPECS


def project_tool_spec(
    spec: ToolSpec, *, web_search_enabled: bool
) -> dict[str, Any]:
    """Project an internal ToolSpec into the safe public response shape."""

    available = spec.state != "disabled" and (
        not spec.requires_web or web_search_enabled
    )
    return {
        "name": spec.name,
        "category": spec.category,
        "registries": list(spec.registries),
        "route_targets": list(spec.route_targets),
        "graph_scope": spec.graph_scope,
        "planner_aliases": list(spec.planner_aliases),
        "requires_web": spec.requires_web,
        "available": available,
        "state": spec.state,
    }


def build_tool_manifest_response(*, web_search_enabled: bool) -> dict[str, Any]:
    """Build the runtime manifest payload consumed by admin endpoints."""

    tools = [
        project_tool_spec(spec, web_search_enabled=web_search_enabled)
        for spec in _TOOL_SPECS
    ]
    groups = []
    for category in sorted(ALLOWED_CATEGORIES):
        category_tools = [
            tool for tool in tools if tool["category"] == category
        ]
        groups.append(
            {
                "category": category,
                "count": len(category_tools),
                "available_count": sum(
                    1 for tool in category_tools if tool["available"]
                ),
            }
        )
    return {
        "tools": tools,
        "groups": groups,
        "runtime": {
            "web_search_enabled": web_search_enabled,
            "auth": "admin",
            "source": "src.tools.manifest",
        },
    }


__all__ = [
    "ALLOWED_CATEGORIES",
    "ALLOWED_GRAPH_SCOPES",
    "ALLOWED_REGISTRIES",
    "ALLOWED_ROUTE_TARGETS",
    "ALLOWED_STATES",
    "ToolSpec",
    "build_tool_manifest_response",
    "iter_tool_specs",
    "project_tool_spec",
]
