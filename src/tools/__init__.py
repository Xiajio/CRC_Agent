from .clinical_tools import list_clinical_tools
from .rag_tools import get_guideline_tool, get_all_rag_tools, get_enhanced_rag_tools
from .web_search_tools import (
    get_web_search_tool,
    get_all_web_search_tools,
    get_clinical_web_search_tools,
    WebSearchTool,
    ClinicalEvidenceSearchTool,
    DrugInfoSearchTool,
    GuidelineUpdateSearchTool,
    LatestResearchSearchTool,
)
from .database_tools import (
    get_database_tools,
    list_database_tools,
)
from .card_formatter import (
    CardFormatter,
    formatter,
)
from .tumor_screening_tools import (
    get_tumor_screening_tools,
    list_tumor_screening_tools,
    tumor_screening_tool,
    quick_tumor_check,
)
from .tumor_localization_tools import (
    get_tumor_localization_tools,
    list_tumor_localization_tools,
    tumor_localization_tool,
    batch_tumor_localization,
)
from .radiomics_tools import (
    list_radiomics_tools,
    unet_segmentation_tool,
    radiomics_feature_extraction_tool,
    lasso_feature_selection_tool,
    comprehensive_radiomics_analysis,
)
from .pathology_clam_tools import (
    get_pathology_clam_tools,
    list_pathology_clam_tools,
    pathology_slide_classify,
    quick_pathology_check,
    get_pathology_clam_status,
)


def list_tools():
    """Return full tool registry including clinical and RAG utilities."""

    tools = list_clinical_tools()
    
    # 添加肿瘤筛选工具
    try:
        tools.extend(get_tumor_screening_tools())
        print(f"[Tools] 已加载肿瘤筛选工具")
    except Exception as exc:
        print(f"[Warning] 肿瘤筛选工具初始化失败: {exc}")
    
    # 添加肿瘤定位工具
    try:
        tools.extend(get_tumor_localization_tools())
        print(f"[Tools] 已加载肿瘤定位工具")
    except Exception as exc:
        print(f"[Warning] 肿瘤定位工具初始化失败: {exc}")
    
    # 添加影像组学工具
    try:
        tools.extend(list_radiomics_tools())
        print(f"[Tools] 已加载影像组学工具（U-Net + PyRadiomics + LASSO）")
    except Exception as exc:
        print(f"[Warning] 影像组学工具初始化失败: {exc}")
    
    # 添加病理 CLAM 工具
    try:
        tools.extend(get_pathology_clam_tools())
        print(f"[Tools] 已加载病理 CLAM 工具（全切片分类 + 热力图）")
    except Exception as exc:
        print(f"[Warning] 病理 CLAM 工具初始化失败: {exc}")
    
    # 使用增强版 RAG 工具集（包含 list_guideline_toc 和 read_guideline_chapter）
    try:
        tools.extend(get_enhanced_rag_tools())
        print(f"[Tools] 已加载增强版 RAG 工具集")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to initialize guideline retriever. "
            "Ensure OPENAI_API_KEY/OPENAI_API_BASE are set (or configure LLM_API_KEY/LLM_API_BASE as fallback) "
            "and run python -m src.rag.ingest (use --reset to remove old sources)."
        ) from exc
    return tools


def list_tools_with_web_search():
    """
    Return full tool registry including clinical, RAG, and web search tools.
    
    包含联网搜索能力，可以实时获取最新资料。
    """
    tools = list_tools()
    try:
        tools.extend(get_clinical_web_search_tools())
    except Exception as exc:
        print(f"[Warning] 联网搜索工具初始化失败: {exc}")
    return tools


def list_all_tools():
    """
    Return all available tools including:
    - Clinical tools
    - RAG tools (all variants)
    - Web search tools
    - Database tools
    - Tumor screening tools
    - Tumor localization tools
    - Radiomics tools
    - Pathology CLAM tools
    """
    tools = list_clinical_tools()
    
    try:
        tools.extend(get_enhanced_rag_tools())
    except Exception as exc:
        print(f"[Warning] RAG工具初始化失败: {exc}")
    
    try:
        tools.extend(get_all_web_search_tools())
    except Exception as exc:
        print(f"[Warning] 联网搜索工具初始化失败: {exc}")
    
    # 添加数据库工具
    try:
        from .database_tools import get_database_tools as get_db_tools
        tools.extend(get_db_tools())
        # 添加卡片格式化工具
        from .card_formatter import formatter
        tools.append(formatter)
        print(f"[Info] 已加载数据库工具")
    except Exception as exc:
        print(f"[Warning] 数据库工具初始化失败: {exc}")
    
    # 添加肿瘤筛选工具
    try:
        tools.extend(get_tumor_screening_tools())
        print(f"[Info] 已加载肿瘤筛选工具")
    except Exception as exc:
        print(f"[Warning] 肿瘤筛选工具初始化失败: {exc}")
    
    # 添加肿瘤定位工具
    try:
        tools.extend(get_tumor_localization_tools())
        print(f"[Info] 已加载肿瘤定位工具")
    except Exception as exc:
        print(f"[Warning] 肿瘤定位工具初始化失败: {exc}")
    
    # 添加影像组学工具
    try:
        tools.extend(list_radiomics_tools())
        print(f"[Info] 已加载影像组学工具（U-Net + PyRadiomics + LASSO）")
    except Exception as exc:
        print(f"[Warning] 影像组学工具初始化失败: {exc}")
    
    # 添加病理 CLAM 工具
    try:
        tools.extend(get_pathology_clam_tools())
        print(f"[Info] 已加载病理 CLAM 工具（全切片分类 + 热力图）")
    except Exception as exc:
        print(f"[Warning] 病理 CLAM 工具初始化失败: {exc}")
    
    return tools


__all__ = [
    # Basic
    "list_tools",
    "list_tools_with_web_search",
    "list_all_tools",
    "list_clinical_tools",
    # RAG
    "get_guideline_tool",
    "get_all_rag_tools",
    "get_enhanced_rag_tools",
    # Web Search
    "get_web_search_tool",
    "get_all_web_search_tools",
    "get_clinical_web_search_tools",
    "WebSearchTool",
    "ClinicalEvidenceSearchTool",
    "DrugInfoSearchTool",
    "GuidelineUpdateSearchTool",
    "LatestResearchSearchTool",
    # Database Tools
    "get_database_tools",
    "list_database_tools",
    # Card Formatter (replaced IntelligentCaseQueryTool)
    "CardFormatter",
    "formatter",
    # Tumor Screening Tools (新增)
    "get_tumor_screening_tools",
    "list_tumor_screening_tools",
    "tumor_screening_tool",
    "quick_tumor_check",
    # Tumor Localization Tools (新增)
    "get_tumor_localization_tools",
    "list_tumor_localization_tools",
    "tumor_localization_tool",
    "batch_tumor_localization",
    # Radiomics Tools (新增)
    "list_radiomics_tools",
    "unet_segmentation_tool",
    "radiomics_feature_extraction_tool",
    "lasso_feature_selection_tool",
    "comprehensive_radiomics_analysis",
    # Pathology CLAM Tools (新增)
    "get_pathology_clam_tools",
    "list_pathology_clam_tools",
    "pathology_slide_classify",
    "quick_pathology_check",
    "get_pathology_clam_status",
]
