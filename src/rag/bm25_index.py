"""
BM25 keyword index module.

Lightweight BM25 keyword retrieval for the keyword-matching part of hybrid search.

Features:
1. Based on rank_bm25, no external service required.
2. Supports Chinese and English tokenization with jieba precise mode.
3. Supports incremental updates and persistence.
4. Works alongside the Chroma vector store.
5. Adds persistence checks for version, compression, and data integrity.

Persistence strategy:
- 使用 pickle + gzip 压缩存储
- Stores version and checksum metadata.
- Supports automatic recovery and rebuild.

Can later be extended to Elasticsearch for larger datasets.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Index version. Increment when index structure changes.
INDEX_VERSION = "1.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_index_path(index_path: str) -> Path:
    candidate = Path(index_path).expanduser()
    if candidate.is_absolute():
        return candidate
# BM25 dependency

# BM25 渚濊禆
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[BM25] Warning: rank_bm25 not installed. Install with: pip install rank-bm25")

# Chinese tokenizer dependency
try:
    import jieba
    # Keep jieba quiet by default.
    jieba.setLogLevel(jieba.logging.INFO)
    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    print("[BM25] Warning: jieba not installed. Chinese tokenization may be suboptimal.")


@dataclass
class IndexMetadata:
    """Index metadata."""
    version: str
    created_at: float
    updated_at: float
    document_count: int
    total_tokens: int
    checksum: str  # Data integrity checksum


class BM25Index:
    """
    BM25 keyword index.
    
    Used as the keyword-matching half of hybrid retrieval with vector search.
    
    Persistence enhancements:
    - Uses gzip compression to reduce disk usage.
    - Checks the version and rebuilds when the index format changes.
    - Validates data integrity to prevent corrupted indexes.
    """
    
    # Medical terms for jieba tokenization
    MEDICAL_TERMS = {
        # Common chemotherapy regimens
        "folfox", "folfiri", "folfoxiri", "xelox", "capox",
        # Targeted therapies
        "bevacizumab", "cetuximab", "panitumumab", "regorafenib",
        "fruquintinib", "aflibercept", "trifluridine", "tipiracil",
        "encorafenib", "binimetinib", "trastuzumab", "tucatinib",
        # Immunotherapy
        "pembrolizumab", "nivolumab", "dostarlimab", "sintilimab",
        # Laboratory and biomarkers
        "cea", "ca199", "ca125", "afp", "msi", "mmr", "kras", "braf", "nras",
        # Staging terms
        "tnm", "ajcc", "uicc",
    }
    
    def __init__(
        self,
        index_path: str = "bm25_index",
        tokenizer: str = "auto",  # auto, jieba, simple
        use_compression: bool = True,  # Whether to use gzip compression
    ):
        """
        Initialize the BM25 index.
        
        Args:
            index_path: Index storage path
            tokenizer: Tokenizer type (auto/jieba/simple)
            use_compression: Whether to store index data with gzip compression
        """
        self.index_path = _resolve_index_path(index_path)
        self.tokenizer_type = tokenizer
        self.use_compression = use_compression
        
        # Index data
        self.documents: List[Dict[str, Any]] = []  # Original documents
        self.corpus: List[List[str]] = []  # Tokenized corpus
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: Optional[IndexMetadata] = None
        
        # Create storage directory
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Add medical vocabulary to jieba dictionary
        if HAS_JIEBA:
            for term in self.MEDICAL_TERMS:
                jieba.add_word(term)
        
        # 加载已有索引
        self._load_index()
        
        print(f"[BM25Index] Initialized with {len(self.documents)} docs")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: 杈撳叆鏂囨湰
            
        Returns:
            Tokenized result list
        """
        if not text:
            return []
        
        # Preprocess: lowercase and remove special characters
        text = text.lower()
        
        if self.tokenizer_type == "jieba" or (self.tokenizer_type == "auto" and HAS_JIEBA):
            # Use jieba tokenization for Chinese text.
            tokens = list(jieba.cut(text))
        else:
            # Simple tokenization: whitespace plus non-alphanumeric separators.
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text)
        
        # Filter out stop words and very short tokens
        stop_words = {'\u7684', '\u662f', '\u5728', '\u548c', '\u4e86', '\u6709', '\u4e2d', '\u7b49', '\u53ca', '\u4e0e',
                      'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from'}
        
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        
        return tokens
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "content",
        id_field: str = "id",
    ) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of documents; each document is a dict.
            text_field: Field name containing document text.
            id_field: Field name containing document ID.
        """
        if not HAS_BM25:
            print("[BM25] rank_bm25 is unavailable; skipping index build")
            return
        
        new_docs = []
        new_corpus = []
        
        for doc in documents:
            text = doc.get(text_field, "")
            tokens = self._tokenize(text)
            
            if tokens:
                new_docs.append(doc)
                new_corpus.append(tokens)
        
        # Append to existing data.
        self.documents.extend(new_docs)
        self.corpus.extend(new_corpus)
        
        # 重建 BM25 索引
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        
        print(f"[BM25] Added {len(new_docs)} docs; total={len(self.documents)}")
        
        # Save index.
        self._save_index()
    
    def search(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search documents.
        
        Args:
            query: Query text
            k: 返回结果数量
            score_threshold: Minimum score threshold
            
        Returns:
            [(文档, 分数)] 列表，按分数降序
        """
        if not HAS_BM25 or self.bm25 is None:
            return []
        
        # Tokenize query.
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # 计算 BM25 分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取 top-k 结果
        scored_docs = [(self.documents[i], scores[i]) for i in range(len(scores))]
        scored_docs = [(doc, score) for doc, score in scored_docs if score > score_threshold]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:k]
    
    def search_with_ids(
        self,
        query: str,
        k: int = 10,
        id_field: str = "id",
    ) -> List[Tuple[str, float]]:
        """
        Search and return document IDs and scores.
        
        Args:
            query: Query text
            k: 返回结果数量
            id_field: ID field name
            
        Returns:
            [(文档ID, 分数)] 列表
        """
        results = self.search(query, k)
        return [(doc.get(id_field, str(i)), score) for i, (doc, score) in enumerate(results)]
    
    def clear(self) -> None:
        """清空索引"""
        self.documents = []
        self.corpus = []
        self.bm25 = None
        self.metadata = None
        
        # Delete all persisted files.
        for suffix in [".pkl", ".pkl.gz", ".meta.json"]:
            index_file = self.index_path / f"bm25_data{suffix}"
            if index_file.exists():
                index_file.unlink()
        
        print("[BM25] Index cleared")
    
    def _compute_checksum(self) -> str:
        """Compute a lightweight checksum."""
        # Use document count and corpus length as a lightweight checksum.
        content = f"{len(self.documents)}:{sum(len(c) for c in self.corpus)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _save_index(self) -> None:
        """
        Save index to disk with metadata and atomic writes.
        
        Features:
        - Optional gzip compression
        - Stores version, timestamp, and checksum metadata
        - 原子写入（先写临时文件再重命名）
        """
        try:
            now = time.time()
            
            # Build metadata.
            self.metadata = IndexMetadata(
                version=INDEX_VERSION,
                created_at=self.metadata.created_at if self.metadata else now,
                updated_at=now,
                document_count=len(self.documents),
                total_tokens=sum(len(c) for c in self.corpus),
                checksum=self._compute_checksum(),
            )
            
            # Prepare data.
            data = {
                "documents": self.documents,
                "corpus": self.corpus,
                "metadata": {
                    "version": self.metadata.version,
                    "created_at": self.metadata.created_at,
                    "updated_at": self.metadata.updated_at,
                    "document_count": self.metadata.document_count,
                    "total_tokens": self.metadata.total_tokens,
                    "checksum": self.metadata.checksum,
                }
            }
            
            # Select target file name.
            if self.use_compression:
                index_file = self.index_path / "bm25_data.pkl.gz"
                temp_file = self.index_path / "bm25_data.pkl.gz.tmp"
            else:
                index_file = self.index_path / "bm25_data.pkl"
                temp_file = self.index_path / "bm25_data.pkl.tmp"
            
            # Atomic write: write temp file first.
            if self.use_compression:
                with gzip.open(temp_file, "wb", compresslevel=6) as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(temp_file, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Rename temp file into place.
            temp_file.replace(index_file)
            
            # Also save readable metadata JSON.
            meta_file = self.index_path / "bm25_data.meta.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(data["metadata"], f, indent=2, ensure_ascii=False)
            
            file_size = index_file.stat().st_size / 1024  # KB
            print(f"[BM25] Index saved to {index_file} ({file_size:.1f} KB, {self.metadata.document_count} docs)")
            
        except Exception as e:
            print(f"[BM25] Failed to save index: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_index(self) -> None:
        """
        Load index from disk with validation.
        
        Features:
        - Checks version and skips incompatible indexes.
        - Validates data integrity.
        - Supports compressed and uncompressed formats.
        - Clears the index on load failure so it can be rebuilt.
        """
        loaded = False
        
        # Prefer compressed format.
        for compressed in [True, False]:
            if compressed:
                index_file = self.index_path / "bm25_data.pkl.gz"
            else:
                index_file = self.index_path / "bm25_data.pkl"
            
            if not index_file.exists():
                continue
            
            try:
                # 读取数据
                if compressed:
                    with gzip.open(index_file, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(index_file, "rb") as f:
                        data = pickle.load(f)
                
                # Check version.
                meta = data.get("metadata", {})
                saved_version = meta.get("version", "0.0.0")
                
                if saved_version != INDEX_VERSION:
                    print(f"[BM25] Index version mismatch (saved={saved_version}, current={INDEX_VERSION}); rebuilding")
                    continue
                
                # 加载数据
                self.documents = data.get("documents", [])
                self.corpus = data.get("corpus", [])
                
                # Validate integrity.
                saved_checksum = meta.get("checksum", "")
                current_checksum = self._compute_checksum()
                
                if saved_checksum and saved_checksum != current_checksum:
                    print("[BM25] Checksum mismatch detected; rebuilding index")
                    self.documents = []
                    self.corpus = []
                    continue
                
                # Rebuild BM25 object.
                if self.corpus and HAS_BM25:
                    self.bm25 = BM25Okapi(self.corpus)
                
                # Restore metadata.
                self.metadata = IndexMetadata(
                    version=saved_version,
                    created_at=meta.get("created_at", 0),
                    updated_at=meta.get("updated_at", 0),
                    document_count=meta.get("document_count", 0),
                    total_tokens=meta.get("total_tokens", 0),
                    checksum=saved_checksum,
                )
                
                file_size = index_file.stat().st_size / 1024
                print(f"[BM25] Index loaded: {len(self.documents)} docs ({file_size:.1f} KB)")
                loaded = True
                break
                
            except Exception as e:
                print(f"[BM25] Failed to load {index_file}: {e}")
                continue
        
        if not loaded:
            print("[BM25] No valid index found; starting with empty index")
    
    def rebuild(self) -> None:
        """
        强制重建索引
        
        Use when:
        - Tokenizer changes
        - 索引损坏
        - BM25 parameters need recalculation
        """
        if not self.corpus:
            print("[BM25] No corpus available to rebuild")
            return
        
        print(f"[BM25] Rebuilding index from {len(self.documents)} docs...")
        
        # Retokenize with the current tokenizer.
        new_corpus = []
        for doc in self.documents:
            text = doc.get("content", "")
            tokens = self._tokenize(text)
            new_corpus.append(tokens)
        
        self.corpus = new_corpus
        
        # Rebuild BM25.
        if HAS_BM25 and self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        
        # Save.
        self._save_index()
        print("[BM25] Index rebuild complete")
    
    def export_to_jsonl(self, output_path: str) -> None:
        """
        Export index as JSONL for Elasticsearch migration.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"[BM25] Exported {len(self.documents)} docs to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.corpus:
            return {"document_count": 0, "avg_doc_length": 0}
        
        total_tokens = sum(len(doc) for doc in self.corpus)
        avg_length = total_tokens / len(self.corpus) if self.corpus else 0
        
        return {
            "document_count": len(self.documents),
            "total_tokens": total_tokens,
            "avg_doc_length": avg_length,
            "index_path": str(self.index_path),
        }


class HybridScorer:
    """
    Hybrid scorer.
    
    Combines vector-search scores and BM25 scores for hybrid ranking.
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize hybrid scorer.
        
        Args:
            alpha: Vector-search weight (0-1); remaining weight goes to BM25.
        """
        self.alpha = alpha
    
    def combine_scores(
        self,
        vector_results: List[Tuple[str, float]],  # [(doc_id, score)]
        bm25_results: List[Tuple[str, float]],
        normalize: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Combine vector-search and BM25 scores.
        
        Args:
            vector_results: Vector-search results [(doc_id, score)]
            bm25_results: BM25 search results [(doc_id, score)]
            normalize: Whether to normalize scores.
            
        Returns:
            合并后的结果 [(doc_id, combined_score)]，按分数降序
        """
        # Normalize scores.
        if normalize:
            vector_results = self._normalize_scores(vector_results)
            bm25_results = self._normalize_scores(bm25_results)
        
        # Convert to dictionaries.
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # Merge all document IDs.
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        # 计算混合分数
        combined = []
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)
            combined_score = self.alpha * v_score + (1 - self.alpha) * b_score
            combined.append((doc_id, combined_score))
        
        # Sort by score descending.
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined
    
    def _normalize_scores(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """归一化分数到 0-1 范围"""
        if not results:
            return []
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(doc_id, 1.0) for doc_id, _ in results]
        
        normalized = [
            (doc_id, (score - min_score) / (max_score - min_score))
            for doc_id, score in results
        ]
        
        return normalized


def create_bm25_index(index_path: str = None) -> BM25Index:
    """
    创建 BM25 索引实例
    
    Prefer loading settings from configuration.
    """
    try:
        from ..config import load_settings
        settings = load_settings()
        
        if not settings.rag.enable_bm25:
            print("[BM25] BM25 is disabled")
            return None
        
        path = index_path or settings.rag.bm25_index_path
        return BM25Index(index_path=path)
        
    except Exception as e:
        print(f"[BM25] Failed to load config: {e}")
        return BM25Index(index_path=index_path or "bm25_index")


