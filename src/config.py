from typing import Literal
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")


class LLMSettings(BaseSettings):
    """Runtime configuration for the LLM backend."""

    mode: Literal["API", "Local"] = Field("API", alias="LLM_MODE")
    api_base: str = Field("", alias="LLM_API_BASE", description="LLM API Base URL (必须从环境变量读取)")
    # NOTE: Only required when LLM_MODE=API. We intentionally do NOT require it at
    # settings-parse time so `main_local.py` can run without any API credentials.
    api_key: str = Field("", alias="LLM_API_KEY")
    model: str = Field("mimo-v2-flash", alias="LLM_MODEL")
    provider: str = Field("", alias="LLM_PROVIDER", description="Optional provider hint for capability resolution")
    temperature: float = Field(0.5, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(4096, alias="LLM_MAX_TOKENS", description="Maximum tokens for API mode (max_tokens) or Local mode (max_new_tokens)")
    streaming: bool = Field(False, alias="LLM_STREAMING", description="Enable streaming inference to show tokens in real-time")
    
    # Thinking/Reasoning mode settings
    thinking_enabled: bool = Field(
        False, 
        alias="LLM_THINKING_ENABLED", 
        description="Enable extended thinking/reasoning mode for supported models (DeepSeek-R1, Qwen-QwQ, etc.)"
    )
    thinking_budget: int = Field(
        8192, 
        alias="LLM_THINKING_BUDGET", 
        description="Maximum tokens for thinking/reasoning content"
    )
    show_thinking: bool = Field(
        False, 
        alias="LLM_SHOW_THINKING", 
        description="Display thinking process in output (requires streaming=true for real-time display)"
    )
    
    local_model_path: str = Field(
        "/path/to/your/local/model",
        alias="LOCAL_MODEL_PATH",
        description="Local HF model path when LLM_MODE=Local",
    )

    local_backend: Literal["Auto", "HF", "VLLM"] = Field(
        "Auto",
        alias="LLM_LOCAL_BACKEND",
        description="Local backend selection: Auto, HF, VLLM",
    )
    local_vllm_dtype: str = Field(
        "auto",
        alias="LLM_LOCAL_VLLM_DTYPE",
        description="vLLM dtype (auto, fp8, bfloat16, float16)",
    )
    local_vllm_tensor_parallel_size: int = Field(
        1,
        alias="LLM_LOCAL_VLLM_TP",
        description="vLLM tensor parallel size",
    )
    local_vllm_gpu_memory_utilization: float = Field(
        0.9,
        alias="LLM_LOCAL_VLLM_GPU_MEM_UTIL",
        description="vLLM GPU memory utilization",
    )
    local_vllm_max_model_len: int = Field(
        0,
        alias="LLM_LOCAL_VLLM_MAX_LEN",
        description="vLLM max model length (0 to use default)",
    )
    
    # Local model generation control
    local_concise_mode: bool = Field(
        True,
        alias="LLM_LOCAL_CONCISE_MODE",
        description="Enable concise output mode for local models to reduce verbose thinking (default: True)"
    )
    local_repetition_penalty: float = Field(
        1.15,
        alias="LLM_LOCAL_REPETITION_PENALTY",
        description="Repetition penalty for local model generation to reduce redundancy (default: 1.15, range: 1.0-1.5)"
    )

    truncate_max_tokens: int = Field(
        15000,
        alias="NODE_TRUNCATE_MAX_TOKENS",
        description="Max tokens for message history truncation in nodes (default: 15000)"
    )

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_prefix="", extra="ignore")


class DocumentConverterSettings(BaseSettings):
    """文档转换 Agent 配置 - 用于处理长文档的上下文窗口和分块策略"""

    # 文档转换使用的模型
    model: str = Field(
        "gpt-4o-mini",
        alias="DOC_CONVERTER_MODEL",
        description="LLM model for document conversion (default: gpt-4o-mini)"
    )

    # LLM 输出 token 限制（4096 足够生成结构化 JSON）
    max_tokens: int = Field(
        4096, 
        alias="DOC_CONVERTER_MAX_TOKENS",
        description="Maximum output tokens for document conversion LLM (default: 4096)"
    )
    
    # 输入文本长度限制（字符数）- 降低默认值避免超出 LLM 处理能力
    max_text_length: int = Field(
        50000,
        alias="DOC_CONVERTER_MAX_TEXT_LENGTH",
        description="Maximum input text length in characters (default: 50000)"
    )

    # PDF 最大处理页数 - 降低默认值
    max_pages: int = Field(
        20,
        alias="DOC_CONVERTER_MAX_PAGES",
        description="Maximum number of PDF pages to process (default: 20)"
    )
    
    # 视觉模式最大图片数量
    max_images: int = Field(
        10, 
        alias="DOC_CONVERTER_MAX_IMAGES",
        description="Maximum number of images in vision mode (default: 10)"
    )
    
    # PDF 转图片 DPI
    pdf_dpi: int = Field(
        100, 
        alias="DOC_CONVERTER_PDF_DPI",
        description="DPI for PDF to image conversion (default: 100)"
    )
    
    # 是否启用长文档分段处理
    enable_chunked_processing: bool = Field(
        True, 
        alias="DOC_CONVERTER_ENABLE_CHUNKED",
        description="Enable chunked processing for very long documents (default: True)"
    )
    
    # 分段处理每段最大长度
    chunk_size: int = Field(
        20000, 
        alias="DOC_CONVERTER_CHUNK_SIZE",
        description="Chunk size for long document processing (default: 20000)"
    )
    
    # 分段重叠长度
    chunk_overlap: int = Field(
        2000, 
        alias="DOC_CONVERTER_CHUNK_OVERLAP",
        description="Overlap between chunks for context continuity (default: 2000)"
    )

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_prefix="", extra="ignore")


class RAGSettings(BaseSettings):
    """RAG 分块和检索配置（简化版）"""

    # === 文档解析配置 ===
    # 解析策略：vision(多模态LLM识别，默认) / basic(pypdf文本提取，备用)
    parse_strategy: str = Field(
        "vision",
        alias="RAG_PARSE_STRATEGY",
        description="Document parsing strategy: vision (default, multimodal LLM) / basic (pypdf fallback)"
    )
    
    # 多模态视觉识别模型
    vision_model: str = Field(
        "gpt-4o-mini",
        alias="RAG_VISION_MODEL",
        description="Multimodal LLM for PDF page recognition (default: gpt-4o-mini)"
    )
    
    # PDF 最大处理页数
    max_pages: int = Field(
        1000,
        alias="RAG_MAX_PAGES",
        description="Maximum number of PDF pages to process (default: 1000, set to a large number for long documents)"
    )
    
    # === 分块配置 ===
    # 文本分块大小
    chunk_size: int = Field(
        2000, 
        alias="RAG_CHUNK_SIZE",
        description="Chunk size for RAG text splitting (default: 2000)"
    )
    
    # 分块重叠大小
    chunk_overlap: int = Field(
        200, 
        alias="RAG_CHUNK_OVERLAP",
        description="Chunk overlap for RAG text splitting (default: 200)"
    )
    
    # === 元数据提取配置 ===
    # 是否启用 LLM 元数据增强
    metadata_enhancement_enabled: bool = Field(
        True,
        alias="RAG_METADATA_ENHANCEMENT_ENABLED",
        description="Enable LLM-based metadata enhancement (default: True)"
    )
    
    # 文档级元数据模型
    doc_metadata_model: str = Field(
        "gpt-4o-mini",
        alias="RAG_DOC_METADATA_MODEL",
        description="LLM model for document-level metadata extraction (default: gpt-4o-mini)"
    )
    
    # 元数据增强最大 tokens
    metadata_max_tokens: int = Field(
        2048,
        alias="RAG_METADATA_MAX_TOKENS",
        description="Maximum tokens for metadata enhancement LLM response (default: 2048)"
    )
    
    # === 检索配置 ===
    # 检索返回的文档数量
    retrieval_k: int = Field(
        6,
        alias="RAG_RETRIEVAL_K",
        description="Number of documents to retrieve (default: 6)"
    )

    # Embedding 模型
    embedding_model: str = Field(
        "text-embedding-3-small",
        alias="RAG_EMBEDDING_MODEL",
        description="Embedding model for RAG (default: text-embedding-3-small)"
    )

    # 是否启用 BM25 混合检索
    enable_bm25: bool = Field(
        True,
        alias="RAG_ENABLE_BM25",
        description="Enable BM25 hybrid retrieval (default: True)"
    )

    # BM25 索引路径
    embedding_backend: Literal["api", "local"] = Field(
        "api",
        alias="RAG_EMBEDDING_BACKEND",
        description="Embedding backend for RAG: api or local (default: api)"
    )

    local_embedding_model: str = Field(
        "",
        alias="RAG_LOCAL_EMBEDDING_MODEL",
        description="Local embedding model path or Hugging Face model id when RAG_EMBEDDING_BACKEND=local"
    )

    local_embedding_device: str = Field(
        "auto",
        alias="RAG_LOCAL_EMBEDDING_DEVICE",
        description="Local embedding device: auto, cuda, cpu, or mps"
    )

    local_embedding_batch_size: int = Field(
        16,
        alias="RAG_LOCAL_EMBEDDING_BATCH_SIZE",
        description="Batch size for local embedding inference"
    )

    local_embedding_normalize: bool = Field(
        True,
        alias="RAG_LOCAL_EMBEDDING_NORMALIZE",
        description="Normalize local embedding vectors before storage and retrieval"
    )

    persist_dir: str = Field(
        "chroma_db",
        alias="RAG_PERSIST_DIR",
        description="Path to Chroma vector store storage (default: 'chroma_db')"
    )

    bm25_index_path: str = Field(
        "bm25_index",
        alias="RAG_BM25_INDEX_PATH",
        description="Path to BM25 index storage (default: 'bm25_index')"
    )

    # === Rerank 重排序配置 ===
    # 是否启用重排序
    enable_rerank: bool = Field(
        True,
        alias="RAG_ENABLE_RERANK",
        description="Enable reranking of retrieved documents (default: True)"
    )
    
    # Rerank 模型类型：cross_encoder / cohere / llm
    rerank_model_type: str = Field(
        "cross_encoder",
        alias="RAG_RERANK_MODEL_TYPE",
        description="Rerank model type: cross_encoder/cohere/llm (default: cross_encoder)"
    )
    
    # Cross-Encoder 重排序模型
    rerank_model: str = Field(
        "BAAI/bge-reranker-base",
        alias="RAG_RERANK_MODEL",
        description="Cross-encoder model for reranking (default: BAAI/bge-reranker-base)"
    )
    
    # Cohere API Key (用于 Cohere Reranker)
    cohere_api_key: str = Field(
        "",
        alias="RAG_COHERE_API_KEY",
        description="Cohere API key for Cohere reranker (optional)"
    )

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_prefix="", extra="ignore")


class WebSearchSettings(BaseSettings):
    """联网搜索配置 - 使用 gpt-4o-search-preview 进行实时资料搜集"""
    
    # 是否启用联网搜索
    enabled: bool = Field(
        True,
        alias="WEB_SEARCH_ENABLED",
        description="Enable web search capability (default: True)"
    )
    
    # 联网搜索模型
    model: str = Field(
        "gpt-4o-search-preview",
        alias="WEB_SEARCH_MODEL",
        description="Model for web search (default: gpt-4o-search-preview)"
    )
    
    # 搜索结果最大 tokens
    max_tokens: int = Field(
        4096,
        alias="WEB_SEARCH_MAX_TOKENS",
        description="Maximum tokens for search response (default: 4096)"
    )
    
    # 温度参数（低温度确保结果一致）
    temperature: float = Field(
        0.3,
        alias="WEB_SEARCH_TEMPERATURE",
        description="Temperature for search model (default: 0.3, lower is more consistent)"
    )

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_prefix="", extra="ignore")


class CheckpointSettings(BaseSettings):
    """Checkpoint backend configuration."""

    kind: Literal["memory", "sqlite", "postgres", "redis", "langsmith"] = Field(
        "memory", alias="CHECKPOINT_KIND"
    )
    url: str | None = Field(None, alias="CHECKPOINT_URL")

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_prefix="", extra="ignore")


class ObservabilitySettings(BaseSettings):
    """Tracing/monitoring configuration."""

    tracing: bool = Field(False, alias="LANGSMITH_TRACING")
    api_key: str | None = Field(None, alias="LANGSMITH_API_KEY")
    project: str | None = Field(None, alias="LANGSMITH_PROJECT")

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_prefix="", extra="ignore")


class Settings(BaseSettings):
    """Top-level settings bag to pass through the graph builder."""

    llm: LLMSettings = Field(default_factory=LLMSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    doc_converter: DocumentConverterSettings = Field(default_factory=DocumentConverterSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)

    model_config = SettingsConfigDict(env_file=_ENV_PATH, env_prefix="", extra="ignore")


def load_settings() -> Settings:
    """Helper to load settings once."""

    return Settings()
