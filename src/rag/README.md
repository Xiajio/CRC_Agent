# RAG 模块

RAG（Retrieval-Augmented Generation）模块是 LangG 的核心知识检索引擎，负责从临床指南文档中高效检索相关信息，采用混合检索策略（向量检索 + BM25 关键词检索 + 多级重排序）。

## 文件结构

```
src/rag/
├── __init__.py      # 模块入口，全局检索器单例管理（warmup / get / is_initialized）
├── parser.py        # 文档解析器（混合策略：文本提取 + Vision OCR）
├── ingest.py        # 文档摄入管线（Chroma + BM25 + 元数据增强 + 上下文增强 + 假设性问题）
├── retriever.py     # 混合检索器（向量 + BM25 融合 + 重排序 + 全局单例）
├── bm25_index.py    # BM25 关键词索引（jieba 分词 + 持久化 + 混合评分器）
├── reranker.py      # 多策略重排序器（Cross-Encoder / Cohere / LLM / 元数据 / 混合）
└── evidence.py      # 证据规范化（序列化/反序列化/去重/溯源/引用格式化）
```

## 核心组件

### 1. 文档解析器（parser.py）

采用混合解析策略：PyMuPDF 文本提取为主，按需调用 Vision LLM 处理扫描页和表格。

- `DocumentParser` — 可配置解析器（vision_model、max_pages、pdf_dpi、batch_size、text_threshold、image_area_threshold）
- `ParsedDocument` — 输出数据结构（content、metadata、images、perf_stats）
- 页面分类规则：文本密度 <100 字符、图片面积 >40%、含表格分隔符 → 视觉识别

### 2. 文档摄入（ingest.py）

完整的指南 PDF 摄入管线：

1. 解析原始文档 → `DocumentParser`
2. 提取文档级元数据 → LLM
3. 递归字符分块 → `RecursiveCharacterTextSplitter`（默认 2000 字符，重叠 200）
4. 提取章节标题 → H2/H3 匹配
5. 上下文增强 → LLM 生成 30-50 字语义前缀
6. 假设性问题嵌入 → 每块生成 3 个问题（是什么/怎么做/为什么）
7. 向量化存储 → Chroma（批量 8）
8. BM25 索引重建

命令行：`python -m src.rag.ingest [--reset] [--skip-metadata] [--no-contextual] [--no-hypothetical] [--chunk-size N]`

### 3. 检索器（retriever.py）

混合检索器，两阶段设计：向量检索候选集（20 个）→ 重排序 → top-k 输出。

- `SimpleRetriever`（别名 `HybridRetriever`）
  - 向量检索（Chroma）+ BM25 关键词检索并行
  - 加权融合（默认 70% 向量 / 30% BM25），含数字/缩写/医学术语时动态增加 BM25 权重
  - HNSW 索引错误自动检测与恢复（尝试刷新一次 → 永久降级为纯 BM25）
  - 线程安全全局单例（`_GlobalRetrieverManager`）
- Embedding 后端：API（OpenAI / DashScope）+ Local（HuggingFace，支持 BGE CLS pooling）
- 便捷函数：`hybrid_search()`、`search_with_metadata_filter()`、`search_treatment_recommendations()`、`search_staging_criteria()`、`search_drug_information()`、`search_by_guideline_source()`
- 全局单例管理：`warmup_retriever()`、`get_global_retriever()`、`is_retriever_initialized()`

### 4. BM25 索引（bm25_index.py）

传统的基于关键词的检索，与向量检索互补。

- `BM25Index` — 持久化磁盘索引（gzip 压缩、版本校验、原子写入），jieba 中文分词 + 医学词典（化疗方案/靶向药物/免疫药物/检查项目/分期术语）
- `HybridScorer` — 向量分数 + BM25 分数加权融合（alpha 参数），min-max 归一化
- 工厂：`create_bm25_index()`

### 5. 重排序器（reranker.py）

策略模式，多种重排序策略可切换和组合。

- `CrossEncoderReranker` — 本地 BGE-Reranker，支持静态/动态分数阈值
- `CohereReranker` — Cohere API（rerank-multilingual-v3.0）
- `LLMReranker` — LLM 评分（0-10 分数），通过提示工程
- `MetadataReranker` — 规则化排序（证据级别 + 指南来源权威性 + 元数据标记）
- `HybridReranker` — 多策略加权融合，自动归一化
- 工厂：`create_reranker()`，自动回退

### 6. 证据规范化（evidence.py）

纯函数模块，规范化检索结果为结构化证据。

- `normalize_evidence(item)` — 标准化为 evidence_id、source、page、section、text、snippet、scores、provenance
- `build_evidence_from_document(doc)` — LangChain Document → 证据
- `serialize_retrieved_evidence()` / `parse_retrieved_evidence()` — `<retrieved_evidence>` XML 标签序列化/反序列化
- `evidence_to_references()` — 证据 → 前端引用列表
- `dedupe_evidence_by_id()` / `dedupe_evidence()` — 去重
- `build_rag_trace()` / `make_rag_trace()` — 溯源日志
- `TOOL_RETRIEVAL_PROFILES` — 工具→检索配置文件映射

## 检索流程

```
用户查询 → hybrid_search()
  → Chroma 向量检索 (Top-20 候选)
  → BM25 关键词检索 (并行)
  → HybridScorer 加权融合
  → Reranker 重排序 (可选)
  → 返回 Top-K 文档
```

## 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_PARSE_STRATEGY` | 解析策略：`vision` / `basic` | `vision` |
| `RAG_CHUNK_SIZE` | 分块大小 | `2000` |
| `RAG_CHUNK_OVERLAP` | 重叠大小 | `200` |
| `RAG_RETRIEVAL_K` | 返回数量 | `4` |
| `RAG_ENABLE_BM25` | 启用 BM25 | `true` |
| `RAG_ENABLE_RERANK` | 启用重排序 | `true` |
| `RAG_RERANK_MODEL_TYPE` | 重排序器类型 | `cross_encoder` |
| `RAG_RERANK_MODEL` | Cross-Encoder 模型 | `BAAI/bge-reranker-base` |
| `RAG_EMBEDDING_BACKEND` | Embedding 后端 | `api` |
| `RAG_EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-v4` |
| `RAG_PERSIST_DIR` | Chroma 持久化目录 | `./chroma_db` |
| `RAG_BM25_INDEX_PATH` | BM25 索引路径 | `./bm25_index` |
| `RAG_METADATA_ENHANCEMENT_ENABLED` | 元数据增强 | `true` |

## 依赖

| 包名 | 用途 | 备注 |
|------|------|------|
| langchain / langchain-chroma / langchain-openai | LangChain 集成 | 核心 |
| chromadb | 向量数据库 | 核心 |
| pymupdf | PDF 处理 | 核心 |
| rank-bm25 | BM25 实现 | 核心 |
| jieba | 中文分词 | 核心 |
| sentence-transformers | Cross-Encoder 重排序 | 可选（rerank 组） |
| cohere | Cohere API 重排序 | 可选（cohere 组） |
