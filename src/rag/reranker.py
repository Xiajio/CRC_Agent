"""
Rerank 重排序模块

支持多种重排序策略:
1. Cross-Encoder (本地模型，如 BGE-Reranker)
2. Cohere Reranker (API 服务)
3. LLM-based Reranker (使用 GPT 等模型)
4. Metadata-based Reranker (基于元数据的规则重排序)

重排序在混合检索后应用，提升最终结果的相关性。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

# Cross-Encoder 依赖
try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

# Cohere 依赖
try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False


class BaseReranker(ABC):
    """重排序器基类"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            k: 返回的文档数量（None 表示返回全部）
            
        Returns:
            [(文档, 重排序分数)] 列表，按分数降序
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder 重排序器
    
    使用本地 Cross-Encoder 模型（如 BGE-Reranker）进行重排序。
    优点: 无需 API，速度快，效果好
    
    【v5.1 优化】添加分数阈值过滤，避免返回低相关性结果
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = None,
        max_length: int = 512,
        score_threshold: float = None,  # [优化] 分数阈值
    ):
        """
        初始化 Cross-Encoder 重排序器
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 (cpu/cuda)
            max_length: 最大序列长度
            score_threshold: 分数阈值，低于此分数的结果将被过滤（None 表示不过滤）
        """
        if not HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers 未安装。"
                "请运行: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.max_length = max_length
        self.score_threshold = score_threshold
        
        print(f"[CrossEncoder] 加载模型: {model_name}")
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        print(f"[CrossEncoder] 模型加载完成")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = None,
        score_threshold: float = None,  # [优化] 支持动态阈值
    ) -> List[Tuple[Document, float]]:
        """
        Cross-Encoder 重排序
        
        【v5.1 优化】
        - 添加分数阈值过滤，低于阈值的结果不返回
        - 使用动态阈值：取平均分数 - 0.5*标准差 作为下限
        """
        if not documents:
            return []
        
        # 构建输入对
        pairs = [(query, doc.page_content) for doc in documents]
        
        # 计算相关性分数
        scores = self.model.predict(pairs)
        
        # 组合结果
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # [优化] 分数阈值过滤
        threshold = score_threshold or self.score_threshold
        if threshold is None and len(scores) > 3:
            # 动态阈值：使用统计方法过滤低分结果
            import numpy as np
            scores_arr = np.array(scores)
            mean_score = scores_arr.mean()
            std_score = scores_arr.std()
            # 阈值 = 平均分 - 0.5*标准差（保留较相关的结果）
            threshold = mean_score - 0.5 * std_score
            # 确保阈值不会太高，至少保留 top-k 结果
            max_allowed_threshold = sorted(scores, reverse=True)[min(k or 6, len(scores)-1)]
            threshold = min(threshold, max_allowed_threshold)
        
        if threshold is not None:
            filtered_docs = [(doc, score) for doc, score in scored_docs if score >= threshold]
            # 确保至少返回一些结果（即使低于阈值）
            if len(filtered_docs) < 2 and scored_docs:
                filtered_docs = scored_docs[:max(2, k or 6)]
            scored_docs = filtered_docs
        
        if k:
            scored_docs = scored_docs[:k]
        
        return scored_docs


class CohereReranker(BaseReranker):
    """
    Cohere Reranker
    
    使用 Cohere API 进行重排序。
    优点: 效果好，支持多语言
    缺点: 需要 API Key，有成本
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "rerank-multilingual-v3.0",
    ):
        """
        初始化 Cohere Reranker
        
        Args:
            api_key: Cohere API Key
            model: Rerank 模型名称
        """
        if not HAS_COHERE:
            raise ImportError(
                "cohere 未安装。"
                "请运行: pip install cohere"
            )
        
        self.api_key = api_key or os.getenv("COHERE_API_KEY") or os.getenv("RAG_COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 Cohere API Key")
        
        self.model = model
        self.client = cohere.Client(self.api_key)
        print(f"[Cohere] Reranker 初始化完成, 模型: {model}")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """Cohere API 重排序"""
        if not documents:
            return []
        
        # 提取文档内容
        doc_texts = [doc.page_content for doc in documents]
        
        # 调用 Cohere API
        response = self.client.rerank(
            query=query,
            documents=doc_texts,
            model=self.model,
            top_n=k or len(documents),
        )
        
        # 解析结果
        scored_docs = []
        for result in response.results:
            doc = documents[result.index]
            score = result.relevance_score
            scored_docs.append((doc, score))
        
        return scored_docs


class LLMReranker(BaseReranker):
    """
    LLM-based Reranker
    
    使用 LLM 进行重排序（基于 prompt）。
    优点: 灵活，可以考虑复杂语义
    缺点: 成本高，速度慢
    """
    
    def __init__(
        self,
        llm = None,
        batch_size: int = 5,
    ):
        """
        初始化 LLM Reranker
        
        Args:
            llm: LangChain LLM 实例
            batch_size: 每批处理的文档数量
        """
        self.llm = llm
        self.batch_size = batch_size
        
        if not self.llm:
            self._init_default_llm()
    
    def _init_default_llm(self):
        """初始化默认 LLM - 使用 OPENAI_API_BASE/KEY"""
        try:
            import os
            from langchain_openai import ChatOpenAI
            from ..config import load_settings
            
            settings = load_settings()
            self.llm = ChatOpenAI(
                model=settings.rag.rerank_model,  # 使用 RAG rerank 模型配置
                api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE") or os.getenv("LLM_API_BASE"),
                temperature=0,  # 重排序需要稳定结果
                max_tokens=settings.rag.metadata_max_tokens,
            )
        except Exception as e:
            print(f"[LLMReranker] LLM 初始化失败: {e}")
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """LLM 重排序"""
        if not documents or not self.llm:
            return [(doc, 0.0) for doc in documents]
        
        prompt_template = """请评估以下文档与查询的相关性。

查询: {query}

文档:
{documents}

请为每个文档评分（0-10分），输出格式:
文档1: X分
文档2: X分
...

只输出评分，不要解释。"""
        
        scored_docs = []
        
        # 分批处理
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            # 构建文档列表
            doc_list = "\n\n".join([
                f"文档{j+1}:\n{doc.page_content[:500]}..."
                for j, doc in enumerate(batch)
            ])
            
            prompt = prompt_template.format(query=query, documents=doc_list)
            
            try:
                response = self.llm.invoke(prompt)
                scores = self._parse_scores(response.content, len(batch))
                
                for doc, score in zip(batch, scores):
                    scored_docs.append((doc, score))
            except Exception as e:
                print(f"[LLMReranker] 评分失败: {e}")
                for doc in batch:
                    scored_docs.append((doc, 0.0))
        
        # 排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if k:
            scored_docs = scored_docs[:k]
        
        return scored_docs
    
    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """解析 LLM 返回的评分"""
        import re
        
        scores = []
        pattern = r'文档\d+[：:]\s*(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, response)
        
        for match in matches[:expected_count]:
            try:
                score = float(match) / 10.0  # 归一化到 0-1
                scores.append(score)
            except:
                scores.append(0.0)
        
        # 填充缺失的分数
        while len(scores) < expected_count:
            scores.append(0.0)
        
        return scores


class MetadataReranker(BaseReranker):
    """
    基于元数据的规则重排序器
    
    根据文档的元数据（如证据级别、来源、内容类型等）进行加权重排序。
    优点: 快速，无需模型
    缺点: 规则固定，需要元数据支持
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
    ):
        """
        初始化元数据重排序器
        
        Args:
            weights: 各元数据字段的权重配置
        """
        self.weights = weights or {
            "evidence_level": 0.3,
            "guideline_source": 0.2,
            "content_type_match": 0.2,
            "metadata_enhanced": 0.3,
        }
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """基于元数据的重排序"""
        if not documents:
            return []
        
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_metadata_score(doc, query)
            scored_docs.append((doc, score))
        
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if k:
            scored_docs = scored_docs[:k]
        
        return scored_docs
    
    def _calculate_metadata_score(self, doc: Document, query: str) -> float:
        """计算基于元数据的分数"""
        score = 0.0
        meta = doc.metadata
        
        # 证据级别分数
        evidence_level = meta.get("evidence_level", "")
        evidence_scores = {"I": 1.0, "II": 0.8, "III": 0.6, "IV": 0.4, "专家意见": 0.3}
        score += evidence_scores.get(evidence_level, 0.0) * self.weights.get("evidence_level", 0.3)
        
        # 指南来源分数
        source = meta.get("guideline_source", "")
        source_scores = {"NCCN": 1.0, "CSCO": 0.9, "ESMO": 0.9}
        score += source_scores.get(source, 0.3) * self.weights.get("guideline_source", 0.2)
        
        # 元数据增强标记
        if meta.get("metadata_enhanced"):
            score += self.weights.get("metadata_enhanced", 0.3)
        
        # 摘要与查询的简单匹配
        summary = meta.get("summary", "").lower()
        query_terms = query.lower().split()
        match_count = sum(1 for term in query_terms if term in summary)
        if query_terms:
            score += (match_count / len(query_terms)) * 0.2
        
        return score


class HybridReranker(BaseReranker):
    """
    混合重排序器
    
    结合多种重排序策略的分数。
    """
    
    def __init__(
        self,
        rerankers: List[Tuple[BaseReranker, float]],  # [(reranker, weight)]
    ):
        """
        初始化混合重排序器
        
        Args:
            rerankers: [(重排序器, 权重)] 列表
        """
        self.rerankers = rerankers
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """混合重排序"""
        if not documents:
            return []
        
        # 收集所有重排序结果
        all_scores: Dict[int, float] = {i: 0.0 for i in range(len(documents))}
        
        for reranker, weight in self.rerankers:
            try:
                results = reranker.rerank(query, documents)
                
                # 归一化分数
                if results:
                    max_score = max(score for _, score in results)
                    min_score = min(score for _, score in results)
                    score_range = max_score - min_score if max_score != min_score else 1.0
                    
                    for doc, score in results:
                        doc_idx = documents.index(doc)
                        normalized_score = (score - min_score) / score_range if score_range else 0.5
                        all_scores[doc_idx] += normalized_score * weight
            except Exception as e:
                print(f"[HybridReranker] {type(reranker).__name__} 失败: {e}")
        
        # 组合结果
        scored_docs = [(documents[i], score) for i, score in all_scores.items()]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if k:
            scored_docs = scored_docs[:k]
        
        return scored_docs


def create_reranker(
    model_type: str = None,
    model_name: str = None,
    cohere_api_key: str = None,
) -> Optional[BaseReranker]:
    """
    创建重排序器
    
    优先从配置文件加载参数
    
    Args:
        model_type: 模型类型 (cross_encoder/cohere/llm/metadata)
        model_name: 模型名称
        cohere_api_key: Cohere API Key
        
    Returns:
        重排序器实例，如果不可用则返回 None
    """
    try:
        from ..config import load_settings
        settings = load_settings()
        rag_config = settings.rag
        
        if not rag_config.enable_rerank:
            print("[Reranker] 重排序已禁用")
            return None
        
        model_type = model_type or rag_config.rerank_model_type
        model_name = model_name or rag_config.rerank_model
        cohere_api_key = cohere_api_key or rag_config.cohere_api_key
        
    except Exception as e:
        print(f"[Reranker] 配置加载失败: {e}")
        model_type = model_type or "cross_encoder"
        model_name = model_name or "BAAI/bge-reranker-base"
    
    print(f"[Reranker] 创建重排序器: type={model_type}")
    
    try:
        if model_type == "cross_encoder":
            if not HAS_CROSS_ENCODER:
                print("[Reranker] CrossEncoder 不可用，回退到 metadata")
                return MetadataReranker()
            return CrossEncoderReranker(model_name=model_name)
        
        elif model_type == "cohere":
            if not HAS_COHERE or not cohere_api_key:
                print("[Reranker] Cohere 不可用，回退到 metadata")
                return MetadataReranker()
            return CohereReranker(api_key=cohere_api_key)
        
        elif model_type == "llm":
            return LLMReranker()
        
        elif model_type == "metadata":
            return MetadataReranker()
        
        else:
            print(f"[Reranker] 未知类型: {model_type}，使用 metadata")
            return MetadataReranker()
            
    except Exception as e:
        print(f"[Reranker] 创建失败: {e}，回退到 metadata")
        return MetadataReranker()

