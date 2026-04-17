# RAG 模块技术文档

## 概述

RAG（Retrieval-Augmented Generation，检索增强生成）模块是 LangG 项目的核心知识检索引擎，负责从临床指南文档中高效检索相关信息，为临床决策支持系统提供精准的医学知识支撑。该模块采用混合检索策略，结合向量检索与关键词检索的优势，并配备多级重排序机制，确保检索结果既具有语义相关性，又兼顾关键词匹配度。

本模块的设计充分考虑了医学领域的特殊需求，在文档解析阶段采用布局感知技术智能识别表格、流程图等特殊内容，在检索阶段支持按疾病类型、指南来源、内容类型等多维度进行精准过滤，在结果优化阶段通过重排序机制进一步提升相关性排序质量。整体架构遵循模块化设计原则，各组件之间低耦合高内聚，便于独立扩展和维护。

---

## 目录结构

```
src/rag/
├── __init__.py          # 模块入口，导出公共接口
├── parser.py            # 文档解析器（混合策略）
├── ingest.py            # 文档 ingestion 流程
├── retriever.py         # 检索器与检索接口
├── reranker.py          # 多策略重排序器
└── bm25_index.py        # BM25 关键词索引
```

---

## 核心组件

### 1. 文档解析器（parser.py）

文档解析器是 RAG 流程的入口环节，负责将 PDF、Markdown、纯文本等格式的临床指南文档转换为结构化的文本内容。该模块采用混合解析策略，根据页面特征自动选择最优解析方式，在保证解析质量的同时最大化处理效率。

#### 混合解析策略

解析器核心采用 PyMuPDF（fitz）作为底层引擎，通过布局感知算法智能判断每个页面的类型。对于纯文本页面，直接提取文本内容，处理速度极快且无需调用外部 API；对于包含图片、扫描内容或表格的页面，则调用多模态 LLM（GPT-4o-mini）进行视觉识别。这种按需调用视觉能力的设计既节约了成本，又确保了复杂页面的解析质量。

页面类型判断基于以下规则：文本密度低于 100 字符的页面被判定为扫描页，需要视觉识别；图片面积占比超过 40% 的页面可能包含流程图或示意图，需要视觉识别；包含表格分隔符（| 或 \t）且跨多行的内容被识别为表格，需要视觉识别；对于 TNM 分期等特殊表格，通过检测每行中 TNM 标记的数量来识别。

#### 类定义与接口

```python
class DocumentParser:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        vision_model: str = "gpt-4o-mini",
        max_pages: int = 50,
        pdf_dpi: int = 150,
        batch_size: int = 5,
        text_threshold: int = 50,
        image_area_threshold: float = 0.25,
    ):
        """初始化文档解析器"""
```

解析器支持批量处理视觉页面，每批次默认处理 5 页，通过减少网络请求次数来优化整体处理速度。PDF 转图片采用 150 DPI 分辨率，在保证 OCR 清晰度的同时控制图片体积。解析完成后返回 `ParsedDocument` 对象，包含合并后的 Markdown 内容、原始元数据、提取的图片资源以及性能统计信息。

#### 输出数据结构

```python
@dataclass
class ParsedDocument:
    content: str                           # 合并的 Markdown 文本
    metadata: Dict[str, Any] = field(default_factory=dict)  # 原始元数据
    images: List[Tuple[int, bytes]] = field(default_factory=list)  # 图片资源
    perf_stats: Dict[str, Any] = field(default_factory=dict)  # 性能统计
```

---

### 2. 文档摄入流程（ingest.py）

文档摄入模块负责将原始指南文档处理并存储到向量数据库中，是构建知识库的完整流水线。该模块整合了文档解析、分块处理、元数据增强、向量化存储等多个环节，支持通过配置灵活控制各增强功能的启用状态。

#### 处理流程概述

摄入流程包含以下主要步骤：首先解析原始文档，提取文本内容和页面结构；然后使用 LLM 提取文档级元数据，包括文档标题、类型、指南来源、主要疾病、主要主题等全局信息；接着对文档进行递归字符分块，相邻块之间保留重叠区域以确保上下文连续性；随后对每个 chunk 应用上下文增强和假设性问题嵌入等高级优化技术；最后将处理后的 chunks 向量化并存储到 Chroma 向量库中。

#### 上下文增强机制

上下文增强（Contextual Chunking）通过为每个文本块生成语义前缀来解决传统分块丢失上下文信息的问题。具体实现是为每个 chunk 调用 LLM，根据其内容生成一段 30-50 字的简短描述，说明该片段所属的章节主题、涉及的疾病类型和治疗阶段。将上下文前缀与原文合并后进行向量化，可以显著提升检索时对片段语义的理解准确性。

#### 假设性问题嵌入

假设性问题嵌入（HyDE 技术变体）通过预先生成用户可能提问的问题并嵌入到 chunk 中，提升了系统对口语化查询的召回率。对于每个 chunk，系统生成 3 个涵盖"是什么"、"怎么做"、"为什么"三个维度的问题，这些问题使用符合患者和非专科医生习惯的口语化表达。检索时，用户的自然语言问题能够与预生成的假设性问题产生更高的语义相似度。

#### 分块策略配置

```python
def ingest(
    chunk_size: int = None,
    chunk_overlap: int = None,
    skip_metadata: bool = False,
    enable_contextual_enhancement: bool = True,
    enable_hypothetical_questions: bool = True,
) -> None:
```

分块大小默认 2000 字符，相邻块之间保留 200 字符的重叠。分块时使用的分隔符优先级为：段落分隔符（\n\n）、换行符（\n）、句号（。）、英文句号（.）、空格。这种多级分隔符策略能够在保持语义完整性的同时控制每个 chunk 的大小。

#### 命令行使用

```bash
# 执行完整摄入流程
python -m src.rag.ingest

# 重置向量库后重新摄入
python -m src.rag.ingest --reset

# 跳过元数据增强（加速）
python -m src.rag.ingest --skip-metadata

# 自定义分块参数
python -m src.rag.ingest --chunk-size 1500 --chunk-overlap 150

# 禁用高级增强功能
python -m src.rag.ingest --no-contextual --no-hypothetical
```

---

### 3. 检索器（retriever.py）

检索器模块提供向量检索和混合检索能力，是 RAG 系统的查询入口。该模块封装了 Chroma 向量库的查询接口，支持基础检索、元数据过滤、专业领域检索等多种查询模式，并集成了可选的重排序流程。

#### 检索流程架构

检索流程采用两阶段设计：第一阶段通过向量相似度检索获取候选文档集合，默认检索 20 个候选以确保足够的召回率；第二阶段对候选文档应用重排序（如果启用），最终返回按相关性排序的 top-k 结果。这种设计在保证检索质量的同时，通过控制重排序的输入规模来优化响应速度。

#### SimpleRetriever 类

```python
class SimpleRetriever:
    def __init__(
        self,
        vectorstore: Chroma = None,
        reranker: BaseReranker = None,
        candidate_k: int = 20,
    ):
        """初始化检索器"""
```

检索器默认使用 LRU 缓存机制缓存向量存储和重排序器的实例，避免重复初始化带来的开销。在生产环境中可以通过传入自定义的 vectorstore 和 reranker 来实现多租户隔离或特殊配置需求。

#### 便捷检索函数

```python
def hybrid_search(
    query: str,
    k: int = None,
    use_rerank: bool = True,
    metadata_filter: Dict[str, Any] = None,
) -> List[Document]:
```

该函数是面向业务层的主要接口，支持传入查询文本、返回数量、重排序开关和元数据过滤条件。元数据过滤支持 Chroma 的 where 语法，可以按照 `content_type`、`guideline_source`、`primary_disease` 等字段进行精确或模糊过滤。

#### 专业领域检索

系统针对医学场景预置了多个专业检索函数：治疗方案检索（`search_treatment_recommendations`）专门查询化疗方案、靶向治疗、免疫治疗等内容；分期标准检索（`search_staging_criteria`）专门查询 TNM 分期、临床分期等诊断标准；药物信息检索（`search_drug_information`）支持按药物名称精确检索药品说明、适应症、不良反应等内容；指南来源检索（`search_by_guideline_source`）允许用户指定查询 NCCN、CSCO、ESMO 等特定来源的指南。

---

### 4. 重排序器（reranker.py）

重排序模块负责对初始检索结果进行二次排序，进一步提升结果的相关性质量。模块采用策略模式设计，支持多种重排序算法的灵活切换和组合。

#### 支持的重排序策略

Cross-Encoder 重排序器使用本地部署的 BGE-Reranker 模型，通过将查询和文档作为输入对进行联合编码，计算精确的相关性分数。这种方式无需调用外部 API，响应速度快，适合对延迟敏感的生产场景。模型默认使用 `BAAI/bge-reranker-base`，支持通过配置切换为其他 Cross-Encoder 模型。

Cohere 重排序器调用 Cohere 官方 API 执行重排序，采用多语言重排序模型 `rerank-multilingual-v3.0`，对中英文混合查询有良好的支持。该方式需要配置 Cohere API Key，存在调用成本，但重排序效果通常优于本地模型。

LLM 重排序器通过提示工程让通用大语言模型对文档相关性进行评分，输出格式为 0-10 的数值分数。这种方式灵活性最高，可以根据业务需求定制评分标准，但成本也最高，适合对排序质量要求极高且预算充足的场景。

元数据重排序器基于文档元数据进行规则化排序，无需调用任何模型，速度最快。评分因素包括证据级别（I 级证据 > II 级证据 > 专家意见）、指南来源权威性（NCCN > CSCO > ESMO）、元数据增强标记等。

#### 混合重排序器

```python
class HybridReranker(BaseReranker):
    def __init__(
        self,
        rerankers: List[Tuple[BaseReranker, float]],  # [(reranker, weight)]
    ):
```

混合重排序器支持将多种重排序策略的分数进行加权融合。例如，可以将 Cross-Encoder 分数（权重 0.6）与元数据分数（权重 0.4）组合，综合考虑语义相关性和来源权威性。各子排序器的分数会先进行归一化处理后再加权求和，确保不同量纲的分数可以公平比较。

#### 创建重排序器

```python
def create_reranker(
    model_type: str = None,
    model_name: str = None,
    cohere_api_key: str = None,
) -> Optional[BaseReranker]:
```

工厂函数优先从配置文件加载重排序参数，根据配置创建对应类型的重排序器。如果指定类型不可用（如缺少依赖），会自动回退到元数据重排序器作为兜底方案。

---

### 5. BM25 关键词索引（bm25_index.py）

BM25 关键词索引模块提供传统的基于关键词的检索能力，与向量检索形成互补。向量检索擅长语义匹配但可能遗漏精确关键词，BM25 检索正好弥补这一缺陷，两者组合使用可以显著提升整体召回质量。

#### BM25Index 类

```python
class BM25Index:
    def __init__(
        self,
        index_path: str = "bm25_index",
        tokenizer: str = "auto",  # auto, jieba, simple
        use_compression: bool = True,
    ):
```

索引基于 `rank_bm25` 库实现，支持中英文分词。中文分词使用 jieba 精确模式，系统预置了医学领域专业词典，包含常见化疗方案（FOLFIRI、FOLFOXIRI 等）、靶向药物（贝伐珠单抗、西妥昔单抗等）、免疫治疗药物（帕博利珠单抗、纳武利尤单抗等）、检查项目（CEA、MSI、KRAS 等）以及分期术语（TNM、AJCC 等）。

#### 持久化与恢复

索引数据支持 gzip 压缩存储，包含版本号和校验和确保数据完整性。加载时会检查版本兼容性，不匹配则自动重建。索引文件存储在指定目录下，启动时自动加载已有索引，无需每次重建。

#### HybridScorer 混合评分器

```python
class HybridScorer:
    def __init__(self, alpha: float = 0.7):
        """
        初始化混合评分器
        
        Args:
            alpha: 向量检索权重 (0-1)，剩余为 BM25 权重
        """
        self.alpha = alpha
```

混合评分器将向量检索分数和 BM25 分数按权重合并。默认配置 alpha=0.7 表示向量检索占 70% 权重，BM25 占 30% 权重。分数在合并前会进行 min-max 归一化处理，确保两种不同量纲的分数可以公平加权。

---

## 配置项说明

RAG 模块的运行参数通过 `src/config.py` 中的配置类统一管理，支持从环境变量或配置文件读取。主要配置项包括以下几个方面：

### 向量检索配置

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| embedding_model | RAG_EMBEDDING_MODEL | text-embedding-3-small | 向量 embedding 模型名称 |
| retrieval_k | RAG_RETRIEVAL_K | 6 | 检索返回的文档数量 |
| enable_bm25 | 无 | true | 是否启用 BM25 混合检索 |
| bm25_index_path | 无 | bm25_index | BM25 索引存储路径 |

### 重排序配置

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| enable_rerank | RAG_ENABLE_RERANK | true | 是否启用重排序 |
| rerank_model_type | RAG_RERANK_MODEL_TYPE | cross_encoder | 重排序器类型 |
| rerank_model | RAG_RERANK_MODEL | BAAI/bge-reranker-base | Cross-Encoder 模型名称 |
| cohere_api_key | COHERE_API_KEY | - | Cohere API Key |

### 文档处理配置

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| chunk_size | RAG_CHUNK_SIZE | 2000 | 分块大小（字符数） |
| chunk_overlap | RAG_CHUNK_OVERLAP | 200 | 分块重叠大小 |
| metadata_enhancement_enabled | RAG_METADATA_ENHANCEMENT_ENABLED | true | 是否启用元数据增强 |
| doc_metadata_model | RAG_DOC_METADATA_MODEL | gpt-4o-mini | 元数据提取使用的模型 |
| parse_strategy | RAG_PARSE_STRATEGY | vision | 文档解析策略 |
| max_pages | RAG_MAX_PAGES | 1000 | 最大处理页数 |
| metadata_max_tokens | RAG_METADATA_MAX_TOKENS | 2048 | 元数据提取最大 token 数 |

### 视觉识别配置

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| vision_model | RAG_VISION_MODEL | gpt-4o-mini | 视觉识别模型名称 |
| pdf_dpi | 无 | 150 | PDF 转图片 DPI |
| batch_size | 无 | 5 | 视觉识别批处理大小 |
| text_threshold | 无 | 50 | 文本页判定阈值 |
| image_area_threshold | 无 | 0.25 | 图片面积占比阈值 |

---

## 使用示例

### 基础检索流程

```python
from src.rag import hybrid_search, format_retrieved_docs

# 执行混合检索
query = "结肠癌术后辅助化疗方案"
docs = hybrid_search(
    query=query,
    k=5,
    use_rerank=True
)

# 格式化输出
formatted = format_retrieved_docs(docs, include_metadata=True)
print(formatted)
```

### 带元数据过滤的检索

```python
from src.rag import search_with_metadata_filter

# 仅查询 NCCN 指南中的治疗方案
docs = search_with_metadata_filter(
    query="直肠癌新辅助治疗",
    content_type="治疗方案",
    guideline_source="NCCN",
    k=10
)
```

### 药物信息检索

```python
from src.rag import search_drug_information

# 查询特定药物信息
docs = search_drug_information(
    query="西妥昔单抗的不良反应",
    drug_name="西妥昔单抗"
)
```

### 直接使用检索器类

```python
from src.rag.retriever import SimpleRetriever

# 创建检索器实例
retriever = SimpleRetriever(
    candidate_k=20,
    reranker=None  # 禁用重排序
)

# 执行检索
docs = retriever.retrieve(
    query="晚期非小细胞肺癌的一线治疗",
    k=5,
    use_rerank=False,
    metadata_filter={"primary_disease": "肺癌"}
)
```

### 执行文档摄入

```python
from src.rag.ingest import ingest

# 执行完整摄入流程（启用所有增强功能）
ingest(
    chunk_size=2000,
    chunk_overlap=200,
    skip_metadata=False,
    enable_contextual_enhancement=True,
    enable_hypothetical_questions=True
)
```

---

## 架构设计

### 数据流概述

```
┌─────────────────┐
│  原始文档 (PDF)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  文档解析器      │◄──── 混合策略：PyMuPDF + Vision LLM
│  (parser.py)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  元数据提取      │◄──── LLM 提取文档级元数据
│  (ingest.py)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  文本分块        │◄──── 递归字符分块 + 重叠
└────────┬────────┘
         │
         ├──────────────────────┐
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  上下文增强      │    │  假设性问题嵌入  │
└────────┬────────┘    └────────┬────────┘
         │                      │
         └──────────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  向量化存储      │◄──── Chroma 向量库
                │  (ingest.py)    │
                └────────┬────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  向量检索        │ │  BM25 检索      │ │  元数据过滤     │
│  (retriever.py) │ │  (bm25_index)   │ │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  重排序器        │◄──── 多策略可选
                    │  (reranker.py)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  最终结果返回    │
                    └─────────────────┘
```

### 设计决策

**混合检索策略**：单一检索方式难以同时满足语义匹配和精确关键词匹配的需求。向量检索在处理同义词、表述变体时表现优秀，但可能遗漏特定的医学术语；BM25 正好相反，精确匹配能力强但缺乏语义泛化能力。两者按权重混合（默认向量 70% + BM25 30%）可以在两个维度上取得平衡。

**按需视觉识别**：医学文档中包含大量表格、流程图和扫描页面，直接使用 pypdf 提取会丢失关键信息。全量调用视觉 API 成本高昂且速度慢。本系统通过布局感知算法智能判断页面类型，只对真正需要视觉识别的页面调用 API，实现了成本与效果的平衡。

**两阶段检索**：在大规模文档集合中，直接对全量文档应用重排序模型是不现实的。两阶段设计先通过向量检索快速筛选候选集（20 个），再对候选集应用精细化重排序，在保证效果的同时控制计算成本。

---

## 依赖说明

### 核心依赖

| 包名 | 用途 | 最低版本 |
|------|------|----------|
| langchain | LLM 应用框架 | 0.1.x |
| langchain-chroma | Chroma 集成 | 0.1.x |
| langchain-openai | OpenAI 集成 | 0.1.x |
| chromadb | 向量数据库 | 0.4.x |
| pymupdf | PDF 处理 | 1.23.x |
| rank-bm25 | BM25 实现 | 0.5.x |
| jieba | 中文分词 | 0.42.x |

### 可选依赖

| 包名 | 用途 | 触发条件 |
|------|------|----------|
| sentence-transformers | Cross-Encoder 重排序 | rerank_model_type=cross_encoder |
| cohere | Cohere 重排序 API | rerank_model_type=cohere |

---

## 常见问题

### 1. 视觉识别失败

如果日志中出现 "Vision 能力不可用" 警告，请检查以下配置：确保环境变量中设置了 `OPENAI_API_KEY` 或 `LLM_API_KEY`；确认已安装 `pymupdf` 包（`pip install pymupdf`）；检查网络连接是否能够访问 OpenAI API。

### 2. 向量检索无结果

可能原因包括向量库为空（需先执行 `ingest`）、查询文本与索引内容不匹配、或者 Chroma 数据库损坏。可以执行以下检查：调用 `get_collection_stats()` 查看向量库中的文档数量；检查 `chroma_db` 目录是否存在且包含数据文件；如果数据损坏，删除 `chroma_db` 目录后重新执行摄入。

### 3. 重排序不生效

确认配置中 `enable_rerank` 设置为 `true`；检查对应依赖是否已安装（Cross-Encoder 需要 `sentence-transformers`，Cohere 需要 `cohere`）；查看日志中是否有重排序器的初始化信息。

### 4. 检索结果不相关

尝试调整混合检索权重（alpha 参数）；检查文档摄入时是否启用了上下文增强；考虑增加检索候选数量（candidate_k）；对专业术语查询，BM25 权重可以适当提高。

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| 1.0.0 | 2024-12 | 初始版本发布，基础 RAG 功能 |
| 1.1.0 | 2024-12 | 增强版：上下文增强、假设性问题嵌入、布局感知解析 |
| 1.2.0 | 2024-12 | 优化版：BM25 持久化、重排序策略增强 |

---

