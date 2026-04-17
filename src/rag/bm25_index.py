"""
BM25 鍏抽敭璇嶇储寮曟ā鍧?

杞婚噺绾х殑 BM25 鍏抽敭璇嶆绱㈠疄鐜帮紝鐢ㄤ簬娣峰悎妫€绱腑鐨勫叧閿瘝鍖归厤閮ㄥ垎銆?

鐗规€?
1. 鍩轰簬 rank_bm25 瀹炵幇锛屾棤闇€澶栭儴鏈嶅姟
2. 鏀寔涓嫳鏂囧垎璇嶏紙jieba 绮剧‘妯″紡锛?
3. 鏀寔澧為噺鏇存柊鍜屾寔涔呭寲
4. 涓?Chroma 鍚戦噺搴撳崗鍚屽伐浣?
5. 澧炲己鐨勬寔涔呭寲锛氱増鏈鏌ャ€佸帇缂┿€佸畬鏁存€ф牎楠?

鎸佷箙鍖栫瓥鐣?
- 浣跨敤 pickle + gzip 鍘嬬缉瀛樺偍
- 鍖呭惈鐗堟湰鍙峰拰鏍￠獙鍜?
- 鏀寔鑷姩鎭㈠鍜岄噸寤?

鍚庣画鍙墿灞曞埌 ElasticSearch 浠ユ敮鎸佹洿澶ц妯℃暟鎹€?
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

# 绱㈠紩鐗堟湰鍙凤紙淇敼绱㈠紩缁撴瀯鏃堕渶瑕侀€掑锛?
INDEX_VERSION = "1.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_index_path(index_path: str) -> Path:
    candidate = Path(index_path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate

# BM25 渚濊禆
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[BM25] Warning: rank_bm25 not installed. Install with: pip install rank-bm25")

# 涓枃鍒嗚瘝渚濊禆
try:
    import jieba
    # 璁剧疆 jieba 涓洪潤榛樻ā寮?
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
    checksum: str  # 鏁版嵁瀹屾暣鎬ф牎楠?


class BM25Index:
    """
    BM25 鍏抽敭璇嶇储寮?
    
    鐢ㄤ簬娣峰悎妫€绱腑鐨勫叧閿瘝鍖归厤閮ㄥ垎锛屼笌鍚戦噺妫€绱簰琛ャ€?
    
    鎸佷箙鍖栧寮?
    - 浣跨敤 gzip 鍘嬬缉锛屽噺灏戠鐩樺崰鐢?
    - 鐗堟湰妫€鏌ワ紝绱㈠紩缁撴瀯鍙樻洿鏃惰嚜鍔ㄩ噸寤?
    - 瀹屾暣鎬ф牎楠岋紝闃叉鏁版嵁鎹熷潖
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
        use_compression: bool = True,  # 鏄惁浣跨敤鍘嬬缉
    ):
        """
        鍒濆鍖?BM25 绱㈠紩
        
        Args:
            index_path: 绱㈠紩瀛樺偍璺緞
            tokenizer: 鍒嗚瘝鍣ㄧ被鍨?(auto/jieba/simple)
            use_compression: 鏄惁浣跨敤 gzip 鍘嬬缉瀛樺偍
        """
        self.index_path = _resolve_index_path(index_path)
        self.tokenizer_type = tokenizer
        self.use_compression = use_compression
        
        # 绱㈠紩鏁版嵁
        self.documents: List[Dict[str, Any]] = []  # 鍘熷鏂囨。
        self.corpus: List[List[str]] = []  # 鍒嗚瘝鍚庣殑璇枡
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: Optional[IndexMetadata] = None
        
        # 鍒涘缓瀛樺偍鐩綍
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # 娣诲姞鍖荤枟璇嶆眹鍒?jieba 璇嶅吀
        if HAS_JIEBA:
            for term in self.MEDICAL_TERMS:
                jieba.add_word(term)
        
        # 鍔犺浇宸叉湁绱㈠紩
        self._load_index()
        
        print(f"[BM25Index] Initialized with {len(self.documents)} docs")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        鍒嗚瘝
        
        Args:
            text: 杈撳叆鏂囨湰
            
        Returns:
            鍒嗚瘝缁撴灉鍒楄〃
        """
        if not text:
            return []
        
        # 棰勫鐞嗭細杞皬鍐欙紝绉婚櫎鐗规畩瀛楃
        text = text.lower()
        
        if self.tokenizer_type == "jieba" or (self.tokenizer_type == "auto" and HAS_JIEBA):
            # 浣跨敤 jieba 鍒嗚瘝锛堜腑鏂囦紭鍖栵級
            tokens = list(jieba.cut(text))
        else:
            # 绠€鍗曞垎璇嶏細绌烘牸 + 闈炲瓧姣嶆暟瀛楀瓧绗﹀垎鍓?
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
        娣诲姞鏂囨。鍒扮储寮?
        
        Args:
            documents: 鏂囨。鍒楄〃锛屾瘡涓枃妗ｆ槸涓€涓瓧鍏?
            text_field: 鏂囨湰鍐呭瀛楁鍚?
            id_field: 鏂囨。ID瀛楁鍚?
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
        
        # 杩藉姞鍒扮幇鏈夋暟鎹?
        self.documents.extend(new_docs)
        self.corpus.extend(new_corpus)
        
        # 閲嶅缓 BM25 绱㈠紩
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        
        print(f"[BM25] Added {len(new_docs)} docs; total={len(self.documents)}")
        
        # 淇濆瓨绱㈠紩
        self._save_index()
    
    def search(
        self,
        query: str,
        k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        鎼滅储鏂囨。
        
        Args:
            query: 鏌ヨ鏂囨湰
            k: 杩斿洖缁撴灉鏁伴噺
            score_threshold: 鏈€浣庡垎鏁伴槇鍊?
            
        Returns:
            [(鏂囨。, 鍒嗘暟)] 鍒楄〃锛屾寜鍒嗘暟闄嶅簭
        """
        if not HAS_BM25 or self.bm25 is None:
            return []
        
        # 鍒嗚瘝
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # 璁＄畻 BM25 鍒嗘暟
        scores = self.bm25.get_scores(query_tokens)
        
        # 鑾峰彇 top-k 缁撴灉
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
        鎼滅储骞惰繑鍥炴枃妗D鍜屽垎鏁?
        
        Args:
            query: 鏌ヨ鏂囨湰
            k: 杩斿洖缁撴灉鏁伴噺
            id_field: ID瀛楁鍚?
            
        Returns:
            [(鏂囨。ID, 鍒嗘暟)] 鍒楄〃
        """
        results = self.search(query, k)
        return [(doc.get(id_field, str(i)), score) for i, (doc, score) in enumerate(results)]
    
    def clear(self) -> None:
        """娓呯┖绱㈠紩"""
        self.documents = []
        self.corpus = []
        self.bm25 = None
        self.metadata = None
        
        # 鍒犻櫎鎵€鏈夋寔涔呭寲鏂囦欢
        for suffix in [".pkl", ".pkl.gz", ".meta.json"]:
            index_file = self.index_path / f"bm25_data{suffix}"
            if index_file.exists():
                index_file.unlink()
        
        print("[BM25] Index cleared")
    
    def _compute_checksum(self) -> str:
        """Compute a lightweight checksum."""
        # 浣跨敤鏂囨。鏁伴噺鍜岃鏂欓暱搴﹁绠楃畝鍗曟牎楠屽拰
        content = f"{len(self.documents)}:{sum(len(c) for c in self.corpus)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _save_index(self) -> None:
        """
        淇濆瓨绱㈠紩鍒扮鐩橈紙澧炲己鐗堬級
        
        鐗规€?
        - 鍙€?gzip 鍘嬬缉
        - 淇濆瓨鍏冩暟鎹紙鐗堟湰銆佹椂闂淬€佹牎楠屽拰锛?
        - 鍘熷瓙鍐欏叆锛堝厛鍐欎复鏃舵枃浠跺啀閲嶅懡鍚嶏級
        """
        try:
            now = time.time()
            
            # 鏋勫缓鍏冩暟鎹?
            self.metadata = IndexMetadata(
                version=INDEX_VERSION,
                created_at=self.metadata.created_at if self.metadata else now,
                updated_at=now,
                document_count=len(self.documents),
                total_tokens=sum(len(c) for c in self.corpus),
                checksum=self._compute_checksum(),
            )
            
            # 鍑嗗鏁版嵁
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
            
            # 閫夋嫨鏂囦欢鍚?
            if self.use_compression:
                index_file = self.index_path / "bm25_data.pkl.gz"
                temp_file = self.index_path / "bm25_data.pkl.gz.tmp"
            else:
                index_file = self.index_path / "bm25_data.pkl"
                temp_file = self.index_path / "bm25_data.pkl.tmp"
            
            # 鍘熷瓙鍐欏叆锛氬厛鍐欎复鏃舵枃浠?
            if self.use_compression:
                with gzip.open(temp_file, "wb", compresslevel=6) as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(temp_file, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 閲嶅懡鍚嶄负姝ｅ紡鏂囦欢
            temp_file.replace(index_file)
            
            # 鍚屾椂淇濆瓨鍙鐨勫厓鏁版嵁 JSON
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
        浠庣鐩樺姞杞界储寮曪紙澧炲己鐗堬級
        
        鐗规€?
        - 鐗堟湰妫€鏌ワ紝涓嶅吋瀹瑰垯璺宠繃
        - 瀹屾暣鎬ф牎楠?
        - 鏀寔鍘嬬缉鍜岄潪鍘嬬缉鏍煎紡
        - 鍔犺浇澶辫触鏃舵竻绌虹储寮曪紙绛夊緟閲嶅缓锛?
        """
        loaded = False
        
        # 浼樺厛鍔犺浇鍘嬬缉鏍煎紡
        for compressed in [True, False]:
            if compressed:
                index_file = self.index_path / "bm25_data.pkl.gz"
            else:
                index_file = self.index_path / "bm25_data.pkl"
            
            if not index_file.exists():
                continue
            
            try:
                # 璇诲彇鏁版嵁
                if compressed:
                    with gzip.open(index_file, "rb") as f:
                        data = pickle.load(f)
                else:
                    with open(index_file, "rb") as f:
                        data = pickle.load(f)
                
                # 妫€鏌ョ増鏈?
                meta = data.get("metadata", {})
                saved_version = meta.get("version", "0.0.0")
                
                if saved_version != INDEX_VERSION:
                    print(f"[BM25] Index version mismatch (saved={saved_version}, current={INDEX_VERSION}); rebuilding")
                    continue
                
                # 鍔犺浇鏁版嵁
                self.documents = data.get("documents", [])
                self.corpus = data.get("corpus", [])
                
                # 鏍￠獙瀹屾暣鎬?
                saved_checksum = meta.get("checksum", "")
                current_checksum = self._compute_checksum()
                
                if saved_checksum and saved_checksum != current_checksum:
                    print("[BM25] Checksum mismatch detected; rebuilding index")
                    self.documents = []
                    self.corpus = []
                    continue
                
                # 閲嶅缓 BM25 瀵硅薄
                if self.corpus and HAS_BM25:
                    self.bm25 = BM25Okapi(self.corpus)
                
                # 鎭㈠鍏冩暟鎹?
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
        寮哄埗閲嶅缓绱㈠紩
        
        鍦ㄤ互涓嬫儏鍐典笅璋冪敤锛?
        - 鍒嗚瘝鍣ㄥ彉鏇?
        - 绱㈠紩鎹熷潖
        - 闇€瑕侀噸鏂拌绠?BM25 鍙傛暟
        """
        if not self.corpus:
            print("[BM25] No corpus available to rebuild")
            return
        
        print(f"[BM25] Rebuilding index from {len(self.documents)} docs...")
        
        # 閲嶆柊鍒嗚瘝锛堜娇鐢ㄥ綋鍓嶅垎璇嶅櫒锛?
        new_corpus = []
        for doc in self.documents:
            text = doc.get("content", "")
            tokens = self._tokenize(text)
            new_corpus.append(tokens)
        
        self.corpus = new_corpus
        
        # 閲嶅缓 BM25
        if HAS_BM25 and self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        
        # 淇濆瓨
        self._save_index()
        print("[BM25] Index rebuild complete")
    
    def export_to_jsonl(self, output_path: str) -> None:
        """
        瀵煎嚭绱㈠紩涓?JSONL 鏍煎紡锛堢敤浜庤縼绉诲埌 Elasticsearch锛?
        
        Args:
            output_path: 杈撳嚭鏂囦欢璺緞
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"[BM25] Exported {len(self.documents)} docs to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """鑾峰彇绱㈠紩缁熻淇℃伅"""
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
    娣峰悎璇勫垎鍣?
    
    缁撳悎鍚戦噺妫€绱㈠垎鏁板拰 BM25 鍒嗘暟杩涜娣峰悎鎺掑簭銆?
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        鍒濆鍖栨贩鍚堣瘎鍒嗗櫒
        
        Args:
            alpha: 鍚戦噺妫€绱㈡潈閲?(0-1)锛屽墿浣欎负 BM25 鏉冮噸
        """
        self.alpha = alpha
    
    def combine_scores(
        self,
        vector_results: List[Tuple[str, float]],  # [(doc_id, score)]
        bm25_results: List[Tuple[str, float]],
        normalize: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        鍚堝苟鍚戦噺妫€绱㈠拰 BM25 鐨勫垎鏁?
        
        Args:
            vector_results: 鍚戦噺妫€绱㈢粨鏋?[(doc_id, score)]
            bm25_results: BM25 妫€绱㈢粨鏋?[(doc_id, score)]
            normalize: 鏄惁褰掍竴鍖栧垎鏁?
            
        Returns:
            鍚堝苟鍚庣殑缁撴灉 [(doc_id, combined_score)]锛屾寜鍒嗘暟闄嶅簭
        """
        # 褰掍竴鍖栧垎鏁?
        if normalize:
            vector_results = self._normalize_scores(vector_results)
            bm25_results = self._normalize_scores(bm25_results)
        
        # 杞崲涓哄瓧鍏?
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # 鍚堝苟鎵€鏈夋枃妗D
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        # 璁＄畻娣峰悎鍒嗘暟
        combined = []
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)
            combined_score = self.alpha * v_score + (1 - self.alpha) * b_score
            combined.append((doc_id, combined_score))
        
        # 鎸夊垎鏁伴檷搴忔帓鍒?
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined
    
    def _normalize_scores(self, results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """褰掍竴鍖栧垎鏁板埌 0-1 鑼冨洿"""
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
    鍒涘缓 BM25 绱㈠紩瀹炰緥
    
    浼樺厛浠庨厤缃枃浠跺姞杞藉弬鏁?
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


