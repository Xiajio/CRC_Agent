"""Tool registry facade.

The package keeps imports lazy so metadata-only modules such as
``src.tools.manifest`` can be imported without loading retrievers, models, or
other heavy runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Basic
    "echo": ("src.tools.basic_tools", "echo"),
    "word_count": ("src.tools.basic_tools", "word_count"),
    # Clinical
    "list_clinical_tools": ("src.tools.clinical_tools", "list_clinical_tools"),
    # RAG
    "get_guideline_tool": ("src.tools.rag_tools", "get_guideline_tool"),
    "get_all_rag_tools": ("src.tools.rag_tools", "get_all_rag_tools"),
    "get_enhanced_rag_tools": ("src.tools.rag_tools", "get_enhanced_rag_tools"),
    # Web search
    "get_web_search_tool": ("src.tools.web_search_tools", "get_web_search_tool"),
    "get_all_web_search_tools": (
        "src.tools.web_search_tools",
        "get_all_web_search_tools",
    ),
    "get_clinical_web_search_tools": (
        "src.tools.web_search_tools",
        "get_clinical_web_search_tools",
    ),
    "WebSearchTool": ("src.tools.web_search_tools", "WebSearchTool"),
    "ClinicalEvidenceSearchTool": (
        "src.tools.web_search_tools",
        "ClinicalEvidenceSearchTool",
    ),
    "DrugInfoSearchTool": ("src.tools.web_search_tools", "DrugInfoSearchTool"),
    "GuidelineUpdateSearchTool": (
        "src.tools.web_search_tools",
        "GuidelineUpdateSearchTool",
    ),
    "LatestResearchSearchTool": (
        "src.tools.web_search_tools",
        "LatestResearchSearchTool",
    ),
    # Database and formatting
    "get_database_tools": ("src.tools.database_tools", "get_database_tools"),
    "list_database_tools": ("src.tools.database_tools", "list_database_tools"),
    "CardFormatter": ("src.tools.card_formatter", "CardFormatter"),
    "formatter": ("src.tools.card_formatter", "formatter"),
    # Tumor screening
    "get_tumor_screening_tools": (
        "src.tools.tumor_screening_tools",
        "get_tumor_screening_tools",
    ),
    "list_tumor_screening_tools": (
        "src.tools.tumor_screening_tools",
        "list_tumor_screening_tools",
    ),
    "tumor_screening_tool": (
        "src.tools.tumor_screening_tools",
        "tumor_screening_tool",
    ),
    "quick_tumor_check": ("src.tools.tumor_screening_tools", "quick_tumor_check"),
    # Tumor localization
    "get_tumor_localization_tools": (
        "src.tools.tumor_localization_tools",
        "get_tumor_localization_tools",
    ),
    "list_tumor_localization_tools": (
        "src.tools.tumor_localization_tools",
        "list_tumor_localization_tools",
    ),
    "tumor_localization_tool": (
        "src.tools.tumor_localization_tools",
        "tumor_localization_tool",
    ),
    "batch_tumor_localization": (
        "src.tools.tumor_localization_tools",
        "batch_tumor_localization",
    ),
    # Radiomics
    "list_radiomics_tools": ("src.tools.radiomics_tools", "list_radiomics_tools"),
    "unet_segmentation_tool": (
        "src.tools.radiomics_tools",
        "unet_segmentation_tool",
    ),
    "radiomics_feature_extraction_tool": (
        "src.tools.radiomics_tools",
        "radiomics_feature_extraction_tool",
    ),
    "lasso_feature_selection_tool": (
        "src.tools.radiomics_tools",
        "lasso_feature_selection_tool",
    ),
    "comprehensive_radiomics_analysis": (
        "src.tools.radiomics_tools",
        "comprehensive_radiomics_analysis",
    ),
    # Pathology CLAM
    "get_pathology_clam_tools": (
        "src.tools.pathology_clam_tools",
        "get_pathology_clam_tools",
    ),
    "list_pathology_clam_tools": (
        "src.tools.pathology_clam_tools",
        "list_pathology_clam_tools",
    ),
    "pathology_slide_classify": (
        "src.tools.pathology_clam_tools",
        "pathology_slide_classify",
    ),
    "quick_pathology_check": (
        "src.tools.pathology_clam_tools",
        "quick_pathology_check",
    ),
    "get_pathology_clam_status": (
        "src.tools.pathology_clam_tools",
        "get_pathology_clam_status",
    ),
}


def _load_export(name: str) -> Any:
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        return _load_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def list_clinical_tools():
    return _load_export("list_clinical_tools")()


def get_guideline_tool():
    return _load_export("get_guideline_tool")()


def get_all_rag_tools():
    return _load_export("get_all_rag_tools")()


def get_enhanced_rag_tools():
    return _load_export("get_enhanced_rag_tools")()


def get_web_search_tool():
    return _load_export("get_web_search_tool")()


def get_all_web_search_tools():
    return _load_export("get_all_web_search_tools")()


def get_clinical_web_search_tools():
    return _load_export("get_clinical_web_search_tools")()


def get_database_tools():
    return _load_export("get_database_tools")()


def list_database_tools():
    return _load_export("list_database_tools")()


def get_tumor_screening_tools():
    return _load_export("get_tumor_screening_tools")()


def list_tumor_screening_tools():
    return _load_export("list_tumor_screening_tools")()


def get_tumor_localization_tools():
    return _load_export("get_tumor_localization_tools")()


def list_tumor_localization_tools():
    return _load_export("list_tumor_localization_tools")()


def list_radiomics_tools():
    return _load_export("list_radiomics_tools")()


def get_pathology_clam_tools():
    return _load_export("get_pathology_clam_tools")()


def list_pathology_clam_tools():
    return _load_export("list_pathology_clam_tools")()


def list_tools():
    """Return full graph tool registry including clinical and RAG utilities."""

    tools = list_clinical_tools()

    try:
        tools.extend(get_tumor_screening_tools())
        print("[Tools] Loaded tumor screening tools")
    except Exception as exc:
        print(f"[Warning] Tumor screening tools failed to initialize: {exc}")

    try:
        tools.extend(get_tumor_localization_tools())
        print("[Tools] Loaded tumor localization tools")
    except Exception as exc:
        print(f"[Warning] Tumor localization tools failed to initialize: {exc}")

    try:
        tools.extend(list_radiomics_tools())
        print("[Tools] Loaded radiomics tools")
    except Exception as exc:
        print(f"[Warning] Radiomics tools failed to initialize: {exc}")

    try:
        tools.extend(get_pathology_clam_tools())
        print("[Tools] Loaded pathology CLAM tools")
    except Exception as exc:
        print(f"[Warning] Pathology CLAM tools failed to initialize: {exc}")

    try:
        tools.extend(get_enhanced_rag_tools())
        print("[Tools] Loaded enhanced RAG tools")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize guideline retriever. "
            "Ensure OPENAI_API_KEY/OPENAI_API_BASE are set "
            "(or configure LLM_API_KEY/LLM_API_BASE as fallback) "
            "and run python -m src.rag.ingest (use --reset to remove old sources)."
        ) from exc
    return tools


def list_tools_with_web_search():
    """Return full graph tool registry including clinical, RAG, and web tools."""

    tools = list_tools()
    try:
        tools.extend(get_clinical_web_search_tools())
    except Exception as exc:
        print(f"[Warning] Web search tools failed to initialize: {exc}")
    return tools


def list_all_tools():
    """Return all executor tools, including utility and optional web/database tools."""

    tools = list_clinical_tools()
    tools.extend([_load_export("word_count"), _load_export("echo")])

    try:
        tools.extend(get_enhanced_rag_tools())
    except Exception as exc:
        print(f"[Warning] RAG tools failed to initialize: {exc}")

    try:
        tools.extend(get_all_web_search_tools())
    except Exception as exc:
        print(f"[Warning] Web search tools failed to initialize: {exc}")

    try:
        tools.extend(get_database_tools())
        tools.append(_load_export("formatter"))
        print("[Info] Loaded database tools")
    except Exception as exc:
        print(f"[Warning] Database tools failed to initialize: {exc}")

    try:
        tools.extend(get_tumor_screening_tools())
        print("[Info] Loaded tumor screening tools")
    except Exception as exc:
        print(f"[Warning] Tumor screening tools failed to initialize: {exc}")

    try:
        tools.extend(get_tumor_localization_tools())
        print("[Info] Loaded tumor localization tools")
    except Exception as exc:
        print(f"[Warning] Tumor localization tools failed to initialize: {exc}")

    try:
        tools.extend(list_radiomics_tools())
        print("[Info] Loaded radiomics tools")
    except Exception as exc:
        print(f"[Warning] Radiomics tools failed to initialize: {exc}")

    try:
        tools.extend(get_pathology_clam_tools())
        print("[Info] Loaded pathology CLAM tools")
    except Exception as exc:
        print(f"[Warning] Pathology CLAM tools failed to initialize: {exc}")

    return tools


__all__ = [
    "list_tools",
    "list_tools_with_web_search",
    "list_all_tools",
    *sorted(_LAZY_EXPORTS),
]
