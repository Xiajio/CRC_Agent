
from __future__ import annotations

import json
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .reranker import BaseReranker, create_reranker
from .bm25_index import BM25Index, create_bm25_index


# === Configuration ===
def _get_rag_settings() -> Dict[str, Any]:
    """获取 RAG 配置"""
    try:
        from ..config import load_settings
        settings = load_settings()
        return {
            "embedding_model": settings.rag.embedding_model,
            "embedding_backend": settings.rag.embedding_backend,
            "local_embedding_model": settings.rag.local_embedding_model,
            "local_embedding_device": settings.rag.local_embedding_device,
            "local_embedding_batch_size": settings.rag.local_embedding_batch_size,
            "local_embedding_normalize": settings.rag.local_embedding_normalize,
            "retrieval_k": settings.rag.retrieval_k,
            "enable_rerank": settings.rag.enable_rerank,
            "rerank_model_type": settings.rag.rerank_model_type,
            "rerank_model": settings.rag.rerank_model,
        }
    except Exception:
        # 回退到环境变量
        return {
            "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            "embedding_backend": os.getenv("RAG_EMBEDDING_BACKEND", "api"),
            "local_embedding_model": os.getenv("RAG_LOCAL_EMBEDDING_MODEL", ""),
            "local_embedding_device": os.getenv("RAG_LOCAL_EMBEDDING_DEVICE", "auto"),
            "local_embedding_batch_size": int(os.getenv("RAG_LOCAL_EMBEDDING_BATCH_SIZE", "16")),
            "local_embedding_normalize": os.getenv("RAG_LOCAL_EMBEDDING_NORMALIZE", "true").lower() == "true",
            "retrieval_k": int(os.getenv("RAG_RETRIEVAL_K", "6")),
            "enable_rerank": os.getenv("RAG_ENABLE_RERANK", "true").lower() == "true",
            "rerank_model_type": os.getenv("RAG_RERANK_MODEL_TYPE", "cross_encoder"),
            "rerank_model": os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base"),
        }


_rag_config = _get_rag_settings()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", _rag_config["embedding_model"])
DEFAULT_RETRIEVAL_K = _rag_config["retrieval_k"]

COLLECTION_NAME = "clinical_guidelines"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_persist_dir() -> Path:
    configured = (os.getenv("RAG_PERSIST_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    try:
        from ..config import load_settings

        settings = load_settings()
        configured = (settings.rag.persist_dir or "").strip()
        if configured:
            return Path(configured).expanduser()
    except Exception:
        pass
    return PROJECT_ROOT / "chroma_db"


PERSIST_DIR = _resolve_persist_dir()


def _get_embedding_api_settings() -> Tuple[str, str, str]:
    """读取独立向量 Embedding 配置（不依赖 LLM/OpenAI 主配置）"""
    api_base = (os.getenv("EMBEDDING_API_BASE") or "").strip()
    api_key = (os.getenv("EMBEDDING_API_KEY") or "").strip()
    model = (os.getenv("EMBEDDING_MODEL") or EMBEDDING_MODEL or "text-embedding-3-small").strip()

    if not api_base:
        raise RuntimeError(
            "缺少 EMBEDDING_API_BASE。请在 .env 中设置独立向量 API 地址。"
        )
    if not api_key:
        raise RuntimeError(
            "缺少 EMBEDDING_API_KEY。请在 .env 中设置独立向量 API 密钥。"
        )
    return api_base, api_key, model


def _get_embedding_backend_settings() -> Dict[str, Any]:
    rag_config = _get_rag_settings()
    return {
        "backend": str(
            os.getenv("RAG_EMBEDDING_BACKEND", rag_config.get("embedding_backend", "api"))
        ).strip().lower() or "api",
        "model": (
            os.getenv("RAG_LOCAL_EMBEDDING_MODEL")
            or rag_config.get("local_embedding_model")
            or os.getenv("EMBEDDING_MODEL")
            or rag_config.get("embedding_model")
            or EMBEDDING_MODEL
        ).strip(),
        "device": str(
            os.getenv("RAG_LOCAL_EMBEDDING_DEVICE", rag_config.get("local_embedding_device", "auto"))
        ).strip().lower() or "auto",
        "batch_size": int(
            os.getenv("RAG_LOCAL_EMBEDDING_BATCH_SIZE", str(rag_config.get("local_embedding_batch_size", 16)))
        ),
        "normalize": str(
            os.getenv(
                "RAG_LOCAL_EMBEDDING_NORMALIZE",
                str(rag_config.get("local_embedding_normalize", True)),
            )
        ).strip().lower() == "true",
    }


def is_local_embedding_backend_enabled() -> bool:
    return _get_embedding_backend_settings()["backend"] == "local"


def _normalize_query_text(query: Any) -> str:
    if isinstance(query, str):
        return query
    if query is None:
        return ""
    try:
        return json.dumps(query, ensure_ascii=False)
    except Exception:
        return str(query)


_FATAL_VECTOR_ERROR_PATTERNS = (
    "error loading hnsw index",
    "hnsw segment reader",
    "error constructing hnsw segment reader",
    "error creating hnsw segment reader",
    "backfill request to compactor",
)


def _is_fatal_vectorstore_error(error: Exception) -> bool:
    message = str(error or "").lower()
    return any(pattern in message for pattern in _FATAL_VECTOR_ERROR_PATTERNS)


def _normalize_filter_scalar(value: Any) -> str:
    return str(value or "").strip().lower()


def _metadata_value_matches(actual: Any, expected: Any) -> bool:
    actual_norm = _normalize_filter_scalar(actual)

    if isinstance(expected, dict):
        if "$eq" in expected:
            return actual_norm == _normalize_filter_scalar(expected["$eq"])
        if "$in" in expected:
            values = expected.get("$in") or []
            return actual_norm in {_normalize_filter_scalar(item) for item in values}
        if "$ne" in expected:
            return actual_norm != _normalize_filter_scalar(expected["$ne"])
        return True

    if isinstance(expected, list):
        return actual_norm in {_normalize_filter_scalar(item) for item in expected}

    return actual_norm == _normalize_filter_scalar(expected)


def _matches_metadata_filter(metadata: Dict[str, Any], metadata_filter: Optional[Dict[str, Any]]) -> bool:
    if not metadata_filter or not isinstance(metadata_filter, dict):
        return True

    if "$and" in metadata_filter:
        clauses = metadata_filter.get("$and") or []
        return all(_matches_metadata_filter(metadata, clause) for clause in clauses)

    if "$or" in metadata_filter:
        clauses = metadata_filter.get("$or") or []
        return any(_matches_metadata_filter(metadata, clause) for clause in clauses)

    for key, expected in metadata_filter.items():
        if key.startswith("$"):
            continue
        if not _metadata_value_matches((metadata or {}).get(key), expected):
            return False
    return True


class _DashScopeEmbeddings:
    def __init__(self, model: str, api_key: str, api_base: str):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [_normalize_query_text(t) for t in texts]
        if not any(t.strip() for t in safe_texts):
            raise RuntimeError("input can not be empty.")
        payloads: List[Dict[str, Any]] = [{"input": safe_texts}]
        if len(safe_texts) == 1:
            payloads.append({"input": {"contents": safe_texts[0]}})
        payloads.append({"input": {"contents": safe_texts}})

        last_error: Optional[str] = None
        for payload in payloads:
            resp = self._client.post(
                f"{self.api_base}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, **payload},
            )
            if resp.status_code >= 400:
                try:
                    err = resp.json()
                    last_error = err.get("error", {}).get("message") or resp.text
                except Exception:
                    last_error = resp.text
                continue
            data = resp.json().get("data")
            if not data:
                last_error = "嵌入响应缺少 data 字段"
                continue
            return [item.get("embedding") for item in data]
        raise RuntimeError(last_error or "嵌入请求失败")


class LocalHFEmbeddings:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 16,
        normalize: bool = True,
    ):
        if not (model_name or "").strip():
            raise RuntimeError("RAG_LOCAL_EMBEDDING_MODEL is required when RAG_EMBEDDING_BACKEND=local.")
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.normalize = normalize
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._pooling = "cls" if "bge" in model_name.lower() else "mean"

    def _resolve_device(self, torch_module: Any) -> str:
        requested = (self.device or "auto").strip().lower()
        if requested == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            mps_backend = getattr(torch_module.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("RAG_LOCAL_EMBEDDING_DEVICE=cuda but CUDA is unavailable in the current torch build.")
        if requested == "mps":
            mps_backend = getattr(torch_module.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeError("RAG_LOCAL_EMBEDDING_DEVICE=mps but MPS is unavailable.")
        return requested

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Local embedding backend requires torch.") from exc
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Local embedding backend requires transformers.") from exc

        resolved_device = self._resolve_device(torch)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(resolved_device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self.device = resolved_device

    def _mean_pool(self, token_embeddings: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        safe_texts = [_normalize_query_text(text) for text in texts]
        results: List[List[float]] = []

        with self._torch.inference_mode():
            for start in range(0, len(safe_texts), self.batch_size):
                batch = safe_texts[start:start + self.batch_size]
                inputs = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self._model(**inputs)
                if self._pooling == "cls":
                    pooled = outputs.last_hidden_state[:, 0]
                else:
                    pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                if self.normalize:
                    pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                results.extend(pooled.detach().cpu().tolist())

        return results

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)


def create_embeddings() -> Any:
    backend_settings = _get_embedding_backend_settings()
    if backend_settings["backend"] == "local":
        return LocalHFEmbeddings(
            model_name=backend_settings["model"],
            device=backend_settings["device"],
            batch_size=backend_settings["batch_size"],
            normalize=backend_settings["normalize"],
        )
    api_base, api_key, model = _get_embedding_api_settings()
    if "dashscope" in api_base.lower():
        return _DashScopeEmbeddings(model=model, api_key=api_key, api_base=api_base)
    return OpenAIEmbeddings(model=model, api_key=api_key, base_url=api_base)


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """获取或创建 Chroma 向量存储"""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = create_embeddings()
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


@lru_cache(maxsize=1)
def _get_reranker() -> Optional[BaseReranker]:
    """获取重排序器"""
    rag_config = _get_rag_settings()
    if not rag_config["enable_rerank"]:
        return None
    return create_reranker()


# === [优化] 全局单例管理 ===
class _GlobalRetrieverManager:
    """
    全局 Retriever 管理器（单例模式）
    
    [优化] SimpleRetriever 和 Reranker 在系统启动时初始化一次，全局复用
    避免每次检索都重新实例化，减少内存分配和初始化开销
    """
    _instance: Optional["_GlobalRetrieverManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._retriever: Optional["SimpleRetriever"] = None
        self._vectorstore: Optional[Chroma] = None
        self._reranker: Optional[BaseReranker] = None
        self._init_lock = threading.Lock()
        self._initialized = True
    
    def get_retriever(self, candidate_k: int = 20) -> "SimpleRetriever":
        """
        获取全局 SimpleRetriever 实例
        
        [线程安全] 使用双重检查锁定模式
        """
        if self._retriever is None:
            with self._init_lock:
                if self._retriever is None:
                    print("[GlobalRetriever] 🚀 首次初始化全局 Retriever...")
                    self._vectorstore = _get_vectorstore()
                    self._reranker = _get_reranker()
                    self._retriever = SimpleRetriever(
                        vectorstore=self._vectorstore,
                        reranker=self._reranker,
                        candidate_k=candidate_k,
                    )
                    return self._retriever
                    print("[GlobalRetriever] ✅ 全局 Retriever 初始化完成，后续将复用")
        return self._retriever
    
    def get_vectorstore(self) -> Chroma:
        """获取全局向量存储"""
        if self._vectorstore is None:
            self._vectorstore = _get_vectorstore()
        return self._vectorstore
    
    def get_reranker(self) -> Optional[BaseReranker]:
        """获取全局 Reranker"""
        if self._reranker is None:
            self._reranker = _get_reranker()
        return self._reranker
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._retriever is not None
    
    def warmup(self) -> None:
        """
        预热全局组件
        
        可在系统启动时调用，提前初始化所有组件
        """
        print("[GlobalRetriever] 🔥 开始预热全局组件...")
        _ = self.get_retriever()
        print("[GlobalRetriever] 🔥 预热完成")


# 全局管理器单例
_global_manager = _GlobalRetrieverManager()
_retrieval_metrics_local = threading.local()


def reset_retrieval_metrics() -> None:
    _retrieval_metrics_local.records = []


def _record_retrieval_metrics(metrics: Dict[str, Any]) -> None:
    records = list(getattr(_retrieval_metrics_local, "records", []))
    records.append(metrics)
    _retrieval_metrics_local.records = records


def consume_retrieval_metrics() -> List[Dict[str, Any]]:
    records = list(getattr(_retrieval_metrics_local, "records", []))
    _retrieval_metrics_local.records = []
    return records


class SimpleRetriever:
    """
    简化版检索器（支持混合检索）
    
    使用向量检索 + BM25 + 可选的 Rerank 重排序。
    
    检索流程:
    1. 并行执行向量检索和 BM25 检索
    2. 加权融合（默认 70% 向量 + 30% BM25）
    3. Rerank 重排序（可选）
    4. 返回 top-k 结果
    
    [优化] 支持全局单例模式，避免重复初始化
    [v5.2 新增] 完整混合检索支持，动态权重调整
    """
    
    def __init__(
        self,
        vectorstore: Chroma = None,
        reranker: BaseReranker = None,
        candidate_k: int = 20,
        bm25_index: BM25Index = None,
        alpha: float = 0.7,  # 向量检索权重（0-1）
        use_hybrid: bool = True,  # 是否启用混合检索
        _silent: bool = False,  # [优化] 静默模式，不输出初始化日志
    ):
        """
        初始化检索器
        
        Args:
            vectorstore: Chroma 向量存储
            reranker: 重排序器
            candidate_k: 候选文档数量
            bm25_index: BM25 索引实例
            alpha: 向量检索权重（0-1），剩余为 BM25 权重
            use_hybrid: 是否启用混合检索
            _silent: 静默模式，不输出初始化日志（用于全局单例）
        """
        self.vectorstore = vectorstore or _get_vectorstore()
        self.reranker = reranker or _get_reranker()
        self.candidate_k = candidate_k
        self.alpha = alpha
        self.use_hybrid = use_hybrid
        self._last_retrieval_metrics: Dict[str, Any] = {}
        self._vector_available = self.vectorstore is not None
        self._vector_recovery_attempted = False
        self._vector_disabled_reason = ""
        
        # 初始化 BM25 索引
        if bm25_index is None and use_hybrid:
            self.bm25_index = create_bm25_index()
        else:
            self.bm25_index = bm25_index
        
        # 检查 BM25 可用性
        self._bm25_available = self.bm25_index is not None and \
                            self.bm25_index.bm25 is not None
        
        if False and not _silent:
            print(f"[SimpleRetriever] 初始化完成")
            print(f"  - 向量检索: ✓")
            print(f"  - BM25 检索: {'✓' if self._bm25_available else '✗'}")
            if self._bm25_available:
                print(f"  - 混合权重: 向量={self.alpha:.0%}, BM25={1-self.alpha:.0%}")
            print(f"  - Rerank: {'✓' if self.reranker else '✗'}")

    @staticmethod
    def _attach_score_metadata(
        doc: Document,
        score: Optional[float] = None,
        **extra_scores: Any,
    ) -> Document:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        if score is not None:
            normalized_score = float(score)
            metadata["score"] = normalized_score
            metadata["relevance"] = normalized_score
        for key, value in extra_scores.items():
            if value is None:
                continue
            metadata[key] = float(value) if isinstance(value, (int, float)) else value
        return Document(page_content=doc.page_content, metadata=metadata)

    @staticmethod
    def _document_from_bm25_payload(
        doc_dict: Dict[str, Any],
        score: Optional[float] = None,
    ) -> Document:
        payload = dict(doc_dict or {})
        content = str(payload.pop("content", "") or "")
        metadata = payload
        if score is not None:
            metadata["score"] = float(score)
            metadata["relevance"] = float(score)
            metadata["bm25_score"] = float(score)
        metadata.setdefault("retrieval_method", "bm25")
        return Document(page_content=content, metadata=metadata)
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        use_rerank: bool = True,
        metadata_filter: Dict[str, Any] = None,
        use_hybrid: bool = None,  # 是否使用混合检索（None 使用配置默认值）
    ) -> List[Document]:
        """
        执行检索（支持混合检索）
        
        【v5.2 优化】
        - 完整混合检索支持：向量 + BM25 加权融合
        - 动态权重调整：数字/英文缩写查询提高 BM25 权重
        - 更清晰的日志输出
        
        Args:
            query: 查询文本
            k: 返回结果数量
            use_rerank: 是否使用重排序
            metadata_filter: 元数据过滤条件
            use_hybrid: 是否使用混合检索（None 使用配置默认值）
            
        Returns:
            检索到的文档列表
        """
        if k is None:
            k = DEFAULT_RETRIEVAL_K
        query = _normalize_query_text(query)
        total_started_at = time.perf_counter()
        
        # 决定是否使用混合检索
        hybrid_mode = use_hybrid if use_hybrid is not None else self.use_hybrid
        retrieval_metrics: Dict[str, Any] = {
            "query": query[:200],
            "k": k,
            "mode": "hybrid" if hybrid_mode and self._bm25_available else "vector",
            "vector_ms": 0.0,
            "bm25_ms": 0.0,
            "fusion_ms": 0.0,
            "rerank_ms": 0.0,
        }
        
        print(f"[Retrieve] 查询: {query[:50]}...")
        
        # 检测查询特征，动态调整权重
        query_features = self._analyze_query(query)
        
        if hybrid_mode and self._bm25_available:
            # 使用混合检索
            final_docs = self._hybrid_search(query, k, metadata_filter, query_features)
            retrieval_metrics.update(self._last_retrieval_metrics or {})
        else:
            # 仅使用向量检索
            search_k = self.candidate_k if (use_rerank and self.reranker) else k
            vector_started_at = time.perf_counter()
            vector_results = self._vector_search(query, search_k, metadata_filter)
            retrieval_metrics["vector_ms"] = round((time.perf_counter() - vector_started_at) * 1000, 2)
            print(f"  - 向量检索: {len(vector_results)} 个候选")
            final_docs = vector_results[:k]
        
        # Rerank 重排序（带分数阈值过滤）
        if use_rerank and self.reranker and final_docs:
            rerank_started_at = time.perf_counter()
            reranked = self.reranker.rerank(query, final_docs, k=k)
            retrieval_metrics["rerank_ms"] = round((time.perf_counter() - rerank_started_at) * 1000, 2)
            final_docs = [
                self._attach_score_metadata(
                    doc,
                    score=score,
                    rerank_score=score,
                    final_rank=rank,
                )
                for rank, (doc, score) in enumerate(reranked, start=1)
            ]
            
            # [优化] 显示过滤效果
            if hasattr(self, '_pre_rerank_count'):
                filtered_count = self._pre_rerank_count - len(final_docs)
                if filtered_count > 0:
                    print(f"  - Rerank后: {len(final_docs)} 个结果 (过滤了 {filtered_count} 个低相关性结果)")
                else:
                    print(f"  - Rerank后: {len(final_docs)} 个结果")
        
        retrieval_metrics["result_count"] = len(final_docs)
        retrieval_metrics["total_ms"] = round((time.perf_counter() - total_started_at) * 1000, 2)
        self._last_retrieval_metrics = retrieval_metrics
        _record_retrieval_metrics(retrieval_metrics)
        return final_docs

    @staticmethod
    def _score_vector_results(results: List[Tuple[Document, float]]) -> List[Document]:
        scored_docs = []
        for rank, (doc, distance) in enumerate(results, start=1):
            relevance = 1.0 / (1.0 + float(distance))
            scored_docs.append(
                SimpleRetriever._attach_score_metadata(
                    doc,
                    score=relevance,
                    vector_score=relevance,
                    vector_distance=distance,
                    final_rank=rank,
                    retrieval_method="vector",
                )
            )
        return scored_docs

    def _execute_vector_query(
        self,
        query: str,
        k: int,
        metadata_filter: Dict[str, Any] = None,
    ) -> List[Tuple[Document, float]]:
        if self.vectorstore is None:
            return []
        if metadata_filter:
            return self.vectorstore.similarity_search_with_score(query, k=k, filter=metadata_filter)
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def _refresh_vectorstore(self) -> bool:
        cache_clear = getattr(_get_vectorstore, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()
        try:
            self.vectorstore = _get_vectorstore()
            self._vector_available = self.vectorstore is not None
            return self._vector_available
        except Exception as refresh_error:
            print(f"[Vector] 刷新向量索引失败: {refresh_error}")
            return False

    def _disable_vector_search(self, reason: Any) -> None:
        self._vector_available = False
        self._vector_disabled_reason = str(reason or "")
        self.vectorstore = None
        print(
            "[Vector] 检测到致命索引错误，已禁用向量检索；当前进程将回退到 BM25。"
            f" reason={self._vector_disabled_reason}"
        )
    
    def _vector_search(
        self,
        query: str,
        k: int,
        metadata_filter: Dict[str, Any] = None,
    ) -> List[Document]:
        """向量检索"""
        query = _normalize_query_text(query)
        if not query.strip():
            return []
        if not self._vector_available or self.vectorstore is None:
            return []
        try:
            results = self._execute_vector_query(query, k, metadata_filter)
            return self._score_vector_results(results)
        except Exception as e:
            print(f"[Vector] 检索失败: {e} | query_type={type(query).__name__} | query_len={len(query)}")
            if not _is_fatal_vectorstore_error(e):
                return []

            if not self._vector_recovery_attempted:
                self._vector_recovery_attempted = True
                print("[Vector] 检测到 HNSW 索引异常，尝试刷新向量索引并重试一次")
                try:
                    if self._refresh_vectorstore():
                        retry_results = self._execute_vector_query(query, k, metadata_filter)
                        return self._score_vector_results(retry_results)
                except Exception as retry_error:
                    print(f"[Vector] 刷新后重试失败: {retry_error}")
                    self._disable_vector_search(retry_error)
                    return []

            self._disable_vector_search(e)
            return []
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询特征，用于动态调整权重
        
        医学场景特征：
        - 数字（分期、剂量、数量）
        - 英文缩写（药物名、检查项目）
        - 医学术语（TNM、MSI、MMR 等）
        
        Args:
            query: 查询文本
            
        Returns:
            {"has_numbers": bool, "has_abbreviations": bool, "adjusted_alpha": float}
        """
        # 检测数字
        has_numbers = bool(re.search(r'\d+', query))
        
        # 检测英文缩写（2-6 个大写字母）
        abbreviations = re.findall(r'\b[A-Z]{2,6}\b', query)
        has_abbreviations = len(abbreviations) > 0
        
        # 常见医学术语检查（这些术语通常需要精确匹配）
        medical_terms = ['TNM', 'MSI', 'MMR', 'CEA', 'CA199', 'KRAS', 'BRAF', 'NRAS',
                       'FOLFOX', 'FOLFIRI', 'XELOX', 'CAPOX', 'PD-1', 'PD-L1']
        has_medical_terms = any(term in query.upper() for term in medical_terms)
        
        # 动态调整权重：如果查询包含强特征，提高 BM25 权重
        base_alpha = self.alpha
        if has_numbers or has_abbreviations or has_medical_terms:
            # 降低向量权重，提高 BM25 权重
            # 例如：从 0.7 降到 0.5
            adjusted_alpha = base_alpha * 0.7
            adjustment_reason = []
            if has_numbers:
                adjustment_reason.append("数字")
            if has_abbreviations:
                adjustment_reason.append(f"缩写({','.join(abbreviations[:3])})")
            if has_medical_terms:
                adjustment_reason.append("医学术语")
            
            print(f"  [权重调整] 检测到 {', '.join(adjustment_reason)}，BM25 权重提升")
            print(f"  - 向量权重: {base_alpha:.0%} → {adjusted_alpha:.0%}")
            print(f"  - BM25 权重: {1-base_alpha:.0%} → {1-adjusted_alpha:.0%}")
        else:
            adjusted_alpha = base_alpha
        
        return {
            "has_numbers": has_numbers,
            "has_abbreviations": has_abbreviations,
            "has_medical_terms": has_medical_terms,
            "adjusted_alpha": adjusted_alpha
        }
    
    def _bm25_search(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        BM25 检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            [(文档字典, BM25分数)] 列表，按分数降序
        """
        if not self._bm25_available:
            return []
        
        try:
            results = self.bm25_index.search(query, k=k, score_threshold=0.0)
            return results
        except Exception as e:
            print(f"[BM25] 检索失败: {e}")
            return []
    
    def _filter_bm25_results_by_metadata(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        metadata_filter: Dict[str, Any] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not metadata_filter:
            return results

        filtered: List[Tuple[Dict[str, Any], float]] = []
        for doc_dict, score in results:
            if _matches_metadata_filter(doc_dict or {}, metadata_filter):
                filtered.append((doc_dict, score))
        return filtered

    def _hybrid_search(
        self,
        query: str,
        k: int,
        metadata_filter: Dict[str, Any] = None,
        query_features: Dict[str, Any] = None,
    ) -> List[Document]:
        """
        混合检索（向量 + BM25）
        
        策略：
        1. 并行执行向量检索和 BM25 检索
        2. 使用加权求和融合分数（默认 70% 向量 + 30% BM25）
        3. 返回 top-k 结果
        
        Args:
            query: 查询文本
            k: 返回结果数量
            metadata_filter: 元数据过滤条件（仅用于向量检索）
            query_features: 查询特征（用于动态权重）
            
        Returns:
            混合检索结果列表
        """
        # 动态权重
        hybrid_metrics_started_at = time.perf_counter()
        alpha = query_features["adjusted_alpha"] if query_features else self.alpha
        beta = 1.0 - alpha
        
        # 并行检索
        search_k = self.candidate_k if (self.reranker) else k * 2
        vector_started_at = time.perf_counter()
        vector_results = self._vector_search(query, search_k, metadata_filter)
        vector_ms = round((time.perf_counter() - vector_started_at) * 1000, 2)
        bm25_started_at = time.perf_counter()
        bm25_results = self._filter_bm25_results_by_metadata(
            self._bm25_search(query, search_k),
            metadata_filter,
        )
        bm25_ms = round((time.perf_counter() - bm25_started_at) * 1000, 2)
        
        print(f"  - 向量检索: {len(vector_results)} 个候选")
        print(f"  - BM25 检索: {len(bm25_results)} 个候选")
        
        self._last_retrieval_metrics = {
            "mode": "hybrid",
            "vector_ms": vector_ms,
            "bm25_ms": bm25_ms,
            "fusion_ms": 0.0,
            "total_ms": round((time.perf_counter() - hybrid_metrics_started_at) * 1000, 2),
        }
        if not vector_results and not bm25_results:
            return []
        
        if not vector_results:
            # 仅 BM25 有结果
            return [
                self._attach_score_metadata(
                    self._document_from_bm25_payload(doc_dict, score=bm25_score),
                    score=bm25_score,
                    bm25_score=bm25_score,
                    final_rank=rank,
                    retrieval_method="bm25",
                )
                for rank, (doc_dict, bm25_score) in enumerate(bm25_results[:k], start=1)
            ]
        
        if not bm25_results:
            # 仅向量有结果
            return vector_results[:k]
        
        # 构建文档 ID 到文档的映射（用于去重）
        def get_doc_id(doc: Document) -> str:
            """获取文档唯一标识"""
            return f"{doc.metadata.get('source', 'unknown')}:{doc.metadata.get('page', 0)}"
        
        # 构建向量分数映射（按排名衰减）
        vector_scores = {}
        vector_docs = {}
        for i, doc in enumerate(vector_results):
            # 使用排名衰减分数：第1名=1.0，第2名=0.95，...
            rank_score = max(0.0, 1.0 - i / len(vector_results))
            doc_id = get_doc_id(doc)
            vector_scores[doc_id] = rank_score
            # 存储 Document 对象
            vector_docs[doc_id] = doc
        
        # 构建 BM25 分数映射（按排名衰减）
        bm25_scores = {}
        bm25_docs = {}
        for i, (doc_dict, bm25_score) in enumerate(bm25_results):
            # 归一化 BM25 分数（简单方法：按排名衰减）
            rank_score = max(0.0, 1.0 - i / len(bm25_results))
            doc_id = f"{doc_dict.get('source', 'unknown')}:{doc_dict.get('page', 0)}"
            bm25_scores[doc_id] = rank_score
            bm25_docs[doc_id] = self._document_from_bm25_payload(doc_dict, score=bm25_score)
        
        # 加权融合
        fusion_started_at = time.perf_counter()
        combined_scores = {}
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        all_doc_ids.discard('__doc')  # 移除 Document 存储键
        
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)
            combined_score = alpha * v_score + beta * b_score
            combined_scores[doc_id] = {
                'score': combined_score,
                'vector_score': v_score,
                'bm25_score': b_score,
                'doc': vector_docs.get(doc_id) or bm25_docs.get(doc_id)
            }
        
        # 按混合分数排序
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # 提取前 k 个文档
        final_docs = []
        for rank, (doc_id, score_info) in enumerate(sorted_results[:k], start=1):
            doc = score_info['doc']
            if doc:
                final_docs.append(
                    self._attach_score_metadata(
                        doc,
                        score=score_info['score'],
                        vector_score=score_info['vector_score'],
                        bm25_score=score_info['bm25_score'],
                        final_rank=rank,
                        retrieval_method="hybrid",
                    )
                )
        
        # 记录融合效果
        print(f"  - 混合融合: {len(final_docs)} 个结果")
        if len(final_docs) > 0:
            print(f"  - 融合示例 (Top-3):")
            for i, (doc_id, score_info) in enumerate(sorted_results[:3]):
                v_s = score_info['vector_score']
                b_s = score_info['bm25_score']
                c_s = score_info['score']
                print(f"    {i+1}. [{doc_id[:50]}...] 向量={v_s:.2f}, BM25={b_s:.2f}, 融合={c_s:.2f}")
        
        # 保存 rerank 前的数量（用于显示过滤效果）
        self._pre_rerank_count = len(final_docs)
        self._last_retrieval_metrics = {
            "mode": "hybrid",
            "vector_ms": vector_ms,
            "bm25_ms": bm25_ms,
            "fusion_ms": round((time.perf_counter() - fusion_started_at) * 1000, 2),
            "result_count": len(final_docs),
            "total_ms": round((time.perf_counter() - hybrid_metrics_started_at) * 1000, 2),
        }
        
        return final_docs


# === 兼容旧接口 ===

# HybridRetriever 别名（保持向后兼容）
HybridRetriever = SimpleRetriever


# === Convenience Functions ===

def get_retriever(k: int = None) -> Any:
    """
    获取基础向量检索器（兼容旧接口）
    
    Args:
        k: 检索返回的文档数量
    
    Returns:
        LangChain Retriever
    """
    if k is None:
        k = DEFAULT_RETRIEVAL_K
    
    vectorstore = _get_vectorstore()
    print(f"[RAG Retriever] 使用 k={k} 检索文档")
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_hybrid_retriever(
    k: int = None,
    **kwargs,
) -> SimpleRetriever:
    """
    获取检索器（优化版 - 使用全局单例）
    
    [优化] 使用全局单例，避免重复初始化
    
    Args:
        k: 最终返回的文档数量
        
    Returns:
        全局 SimpleRetriever 实例
    """
    candidate_k = k if k is not None else kwargs.get("candidate_k", DEFAULT_RETRIEVAL_K)
    return _global_manager.get_retriever(candidate_k=candidate_k)


def hybrid_search(
    query: str,
    k: int = None,
    use_rerank: bool = True,
    metadata_filter: Dict[str, Any] = None,
) -> List[Document]:
    """
    执行检索（兼容旧接口）
    
    Args:
        query: 查询文本
        k: 返回结果数量
        use_rerank: 是否使用重排序
        metadata_filter: 元数据过滤条件
        
    Returns:
        检索到的文档列表
    """
    retriever = get_hybrid_retriever()
    return retriever.retrieve(
        query=query,
        k=k,
        use_rerank=use_rerank,
        metadata_filter=metadata_filter,
    )


def search_with_metadata_filter(
    query: str,
    k: int = None,
    content_type: Optional[str] = None,
    medical_specialty: Optional[str] = None,
    disease_focus: Optional[str] = None,
    guideline_source: Optional[str] = None,
    use_hybrid: bool = True,
) -> List[Document]:
    """
    使用元数据过滤的检索
    
    Args:
        query: 查询文本
        k: 返回文档数量
        content_type: 内容类型过滤
        medical_specialty: 医学专科过滤
        disease_focus: 疾病类型过滤
        guideline_source: 指南来源过滤
        use_hybrid: 是否使用检索器（已忽略，始终使用向量检索）
    
    Returns:
        匹配的文档列表
    """
    if k is None:
        k = DEFAULT_RETRIEVAL_K
    
    # 构建过滤条件
    where_filter = {}
    
    if content_type:
        where_filter["content_type"] = content_type  # 修正：与 ingest.py 保持一致
    if medical_specialty:
        where_filter["medical_specialty"] = medical_specialty
    if guideline_source:
        where_filter["guideline_source"] = guideline_source
    if disease_focus:
        normalized = str(disease_focus).strip().lower()
        disease_aliases = {
            "colon cancer": ["Colon Cancer", "结直肠癌"],
            "结肠癌": ["Colon Cancer", "结直肠癌"],
            "rectal cancer": ["Rectal Cancer", "结直肠癌"],
            "直肠癌": ["Rectal Cancer", "结直肠癌"],
            "colorectal cancer": ["结直肠癌", "Colon Cancer", "Rectal Cancer"],
            "结直肠癌": ["结直肠癌", "Colon Cancer", "Rectal Cancer"],
        }
        aliases = disease_aliases.get(normalized, [disease_focus])
        where_filter["primary_disease"] = {"$in": aliases}
    
    return hybrid_search(
        query=query,
        k=k,
        metadata_filter=where_filter if where_filter else None,
    )


def search_treatment_recommendations(
    query: str,
    k: int = None,
    disease: str = None,
) -> List[Document]:
    """
    专门检索治疗方案相关内容
    注意：由于当前向量库中的 doc_type 都是"指南"，不使用 content_type 过滤
    """
    # 不使用 content_type 过滤，因为向量库中没有细粒度分类
    # 如果需要，可以通过 query 内容和 rerank 筛选
    return search_with_metadata_filter(
        query=query,
        k=k,
        disease_focus=disease,
    )


def search_staging_criteria(
    query: str,
    k: int = None,
) -> List[Document]:
    """
    专门检索分期标准相关内容
    注意：由于当前向量库中的 doc_type 都是"指南"，不使用 content_type 过滤
    """
    # 不使用 content_type 过滤，因为向量库中没有细粒度分类
    # 如果需要，可以通过 query 内容和 rerank 筛选
    return search_with_metadata_filter(
        query=query,
        k=k,
    )


def search_drug_information(
    query: str,
    drug_name: Optional[str] = None,
    k: int = None,
) -> List[Document]:
    """
    检索药物相关信息
    注意：由于当前向量库中的 doc_type 都是"指南"，不使用 content_type 过滤
    """
    if k is None:
        k = DEFAULT_RETRIEVAL_K
    
    if drug_name:
        # 先尝试药物名称过滤
        vectorstore = _get_vectorstore()
        # ChromaDB不支持$contains，使用$eq进行精确匹配
        where_filter = {"drug_mentions": {"$eq": drug_name}}
        results = vectorstore.similarity_search(query, k=k, filter=where_filter)
        if results:
            return results
    
    # 不使用 content_type 过滤，因为向量库中没有细粒度分类
    return search_with_metadata_filter(
        query=query,
        k=k,
    )


def search_by_guideline_source(
    query: str,
    source: str,
    k: int = None,
) -> List[Document]:
    """
    按指南来源检索
    """
    return search_with_metadata_filter(
        query=query,
        k=k,
        guideline_source=source,
    )


def get_collection_stats() -> Dict[str, Any]:
    """
    获取向量库统计信息
    """
    vectorstore = _get_vectorstore()
    collection = vectorstore._collection
    
    try:
        count = collection.count()
        
        # 采样获取元数据分布
        sample = collection.get(limit=min(100, count))
        
        content_types = {}
        sources = {}
        
        for meta in sample.get("metadatas", []):
            ct = meta.get("content_type", "未分类")
            content_types[ct] = content_types.get(ct, 0) + 1
            
            src = meta.get("guideline_source", "未知")
            sources[src] = sources.get(src, 0) + 1
        
        return {
            "total_chunks": count,
            "sample_size": len(sample.get("metadatas", [])),
            "doc_type_distribution": content_types,
            "guideline_source_distribution": sources,
        }
    except Exception as e:
        return {"error": str(e)}


def format_retrieved_docs(docs: List[Document], include_metadata: bool = True) -> str:
    """
    格式化检索结果用于显示
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        
        header = f"[{i}] source={source}"
        
        if include_metadata:
            content_type = meta.get("content_type", "")
            summary = meta.get("doc_summary", "")

            if content_type:
                header += f" | 类型: {content_type}"
            if summary:
                header += f"\n📝 {summary}"
        
        formatted.append(f"{header}\n{doc.page_content}")
    
    return "\n\n".join(formatted)
