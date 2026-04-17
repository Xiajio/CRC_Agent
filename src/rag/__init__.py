"""
RAG utilities for guideline ingestion and retrieval.

简化版 RAG 系统:
1. 简化文档解析 (基础pypdf + 多模态LLM视觉识别)
2. 文档级元数据提取
3. 向量检索 + Rerank

[优化] 全局单例支持:
- warmup_retriever(): 预热全局 Retriever，可在系统启动时调用
- get_global_retriever(): 获取全局 Retriever 实例
"""

from .parser import DocumentParser, ParsedDocument, create_parser
from .reranker import (
    BaseReranker,
    CrossEncoderReranker,
    CohereReranker,
    LLMReranker,
    MetadataReranker,
    create_reranker,
)
from .retriever import (
    get_retriever,
    get_hybrid_retriever,
    hybrid_search,
    search_with_metadata_filter,
    search_treatment_recommendations,
    search_staging_criteria,
    search_drug_information,
    search_by_guideline_source,
    get_collection_stats,
    format_retrieved_docs,
    HybridRetriever,
    SimpleRetriever,
    _global_manager,  # [优化] 导出全局管理器
)
from .ingest import ingest


# === [优化] 全局 Retriever 接口 ===

def warmup_retriever() -> None:
    """
    预热全局 Retriever
    
    可在系统启动时调用，提前初始化所有组件，避免首次请求延迟
    
    Example:
        >>> from src.rag import warmup_retriever
        >>> warmup_retriever()  # 在 main() 或 app startup 中调用
    """
    _global_manager.warmup()


def get_global_retriever() -> SimpleRetriever:
    """
    获取全局 Retriever 实例
    
    [优化] 使用全局单例，避免重复初始化
    
    Returns:
        全局 SimpleRetriever 实例
    """
    return _global_manager.get_retriever()


def is_retriever_initialized() -> bool:
    """检查全局 Retriever 是否已初始化"""
    return _global_manager.is_initialized()


__all__ = [
    # Parser
    "DocumentParser",
    "ParsedDocument", 
    "create_parser",
    # Reranker
    "BaseReranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "LLMReranker",
    "MetadataReranker",
    "create_reranker",
    # Retriever
    "get_retriever",
    "get_hybrid_retriever",
    "hybrid_search",
    "search_with_metadata_filter",
    "search_treatment_recommendations",
    "search_staging_criteria",
    "search_drug_information",
    "search_by_guideline_source",
    "get_collection_stats",
    "format_retrieved_docs",
    "HybridRetriever",
    "SimpleRetriever",
    # [优化] 全局 Retriever 接口
    "warmup_retriever",
    "get_global_retriever",
    "is_retriever_initialized",
    # Ingest
    "ingest",
]
