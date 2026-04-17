from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..services.llm_service import create_compatible_chat_openai

# 导入本地模块
from .bm25_index import create_bm25_index
from .parser import DocumentParser, ParsedDocument, create_parser
from .retriever import create_embeddings, is_local_embedding_backend_enabled

# Load environment variables
load_dotenv()


# === Configuration ===
def _get_rag_settings() -> Dict[str, Any]:
    """获取 RAG 配置"""
    try:
        from ..config import load_settings
        settings = load_settings()
        return {
            "chunk_size": settings.rag.chunk_size,
            "chunk_overlap": settings.rag.chunk_overlap,
            "embedding_model": settings.rag.embedding_model,
            "metadata_enhancement_enabled": settings.rag.metadata_enhancement_enabled,
            "doc_metadata_model": settings.rag.doc_metadata_model,
            "metadata_max_tokens": settings.rag.metadata_max_tokens,
            "parse_strategy": settings.rag.parse_strategy,
            "max_pages": settings.rag.max_pages,
        }
    except Exception:
        # 回退到环境变量
        return {
            "chunk_size": int(os.getenv("RAG_CHUNK_SIZE", "2000")),
            "chunk_overlap": int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            "metadata_enhancement_enabled": os.getenv("RAG_METADATA_ENHANCEMENT_ENABLED", "true").lower() == "true",
            "doc_metadata_model": os.getenv("RAG_DOC_METADATA_MODEL", "gpt-4o-mini"),
            "metadata_max_tokens": int(os.getenv("RAG_METADATA_MAX_TOKENS", "2048")),
            "parse_strategy": os.getenv("RAG_PARSE_STRATEGY", "vision"),
            "max_pages": int(os.getenv("RAG_MAX_PAGES", "1000")),
        }


def _get_llm_settings() -> Dict[str, Any]:
    """获取 LLM 配置"""
    try:
        from ..config import load_settings
        settings = load_settings()
        return {
            "api_base": settings.llm.api_base,
            "api_key": settings.llm.api_key,
            "model": settings.llm.model,
        }
    except Exception:
        # 回退到环境变量
        return {
            "api_base": os.getenv("OPENAI_API_BASE", ""),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        }


_rag_config = _get_rag_settings()
EMBEDDING_MODEL = _rag_config["embedding_model"]
DEFAULT_CHUNK_SIZE = _rag_config["chunk_size"]
DEFAULT_CHUNK_OVERLAP = _rag_config["chunk_overlap"]

COLLECTION_NAME = "clinical_guidelines"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDELINE_DIR = PROJECT_ROOT / "data" / "guidelines"


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


# === Environment Setup ===
def _ensure_embedding_env_for_embeddings() -> None:
    """确保独立 Embedding API 环境变量（不依赖 LLM/OpenAI 主配置）"""
    if is_local_embedding_backend_enabled():
        return
    if not (os.getenv("EMBEDDING_API_BASE") or "").strip():
        raise RuntimeError(
            "缺少 EMBEDDING_API_BASE（用于 RAG 向量化）。请在 .env 中设置独立向量 API 地址。"
        )
    if not (os.getenv("EMBEDDING_API_KEY") or "").strip():
        raise RuntimeError(
            "缺少 EMBEDDING_API_KEY（用于 RAG 向量化）。请在 .env 中设置独立向量 API 密钥。"
        )


def _get_metadata_llm(model: str = None) -> ChatOpenAI:
    """获取用于元数据增强的 LLM - 使用 OPENAI_API_BASE/KEY"""
    rag_config = _get_rag_settings()
    
    model = model or rag_config["doc_metadata_model"]
    
    return create_compatible_chat_openai(
        model=model,
        temperature=0.1,
        max_tokens=rag_config["metadata_max_tokens"],
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE") or os.getenv("LLM_API_BASE"),
    )


# === 文档级元数据提取 ===

DOC_METADATA_PROMPT = """你是一个专业的医学文献分析助手。请分析文档并提取关键元数据信息。

## 文档信息
文件名: {filename}
文本长度: {text_length} 字符

## 文档内容（摘要）
{content_sample}

## 任务
请分析上述文档，提取以下元数据（JSON格式）:

```json
{{
    "document_title": "文档标题",
    "document_type": "指南/病例报告/综述/研究论文/其他",
    "guideline_source": "NCCN/CSCO/ESMO/其他/无",
    "guideline_version": "版本信息（如有）",
    "publication_year": "发布年份（如有）",
    "primary_disease": "主要疾病类型",
    "disease_subtypes": ["疾病亚型列表"],
    "main_topics": ["3-5个主要主题"],
    "language": "中文/英文/双语",
    "document_summary": "文档整体内容概述（100字以内）"
}}
```

请直接输出 JSON，不要添加其他解释。"""


def _extract_doc_metadata(
    parsed_doc: ParsedDocument,
    filename: str,
    llm: ChatOpenAI,
) -> Dict[str, Any]:
    """
    提取文档级元数据
    
    文档级元数据是全局信息，应用到所有 chunks
    """
    try:
        content = parsed_doc.content
        
        # 取样本内容（开头、中间、结尾各取一部分，确保覆盖文档各部分）
        sample_parts = []
        content_length = len(content)
        
        if content_length > 3000:
            # 计算采样策略：开头1000，中间1000，结尾1000，总共3000字符
            sample_size = 1000
            sample_parts.append(content[:sample_size])
            mid_start = (content_length - sample_size) // 2
            sample_parts.append(content[mid_start:mid_start + sample_size])
            sample_parts.append(content[-sample_size:])
            content_sample = "\n...\n".join(sample_parts)
        elif content_length > 1500:
            # 中等长度文档：开头一半和结尾一半
            half = content_length // 2
            sample_parts.append(content[:half])
            sample_parts.append(content[-half:])
            content_sample = "\n...\n".join(sample_parts)
        else:
            # 短文档使用完整内容
            content_sample = content
        
        prompt = DOC_METADATA_PROMPT.format(
            filename=filename,
            text_length=content_length,
            content_sample=content_sample[:4000],
        )
        
        response = llm.invoke(prompt)
        result = _parse_json_response(response.content)
        
        if result:
            print(f"[DocMetadata] 文档级元数据提取成功: {filename}")
            return result
        
    except Exception as e:
        print(f"[DocMetadata] 文档级元数据提取失败: {e}")
    
    # 返回默认元数据
    return {
        "document_title": filename,
        "document_type": "其他",
        "guideline_source": "未知",
        "primary_disease": "",
        "main_topics": [],
        "document_summary": "",
    }


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 返回的 JSON"""
    try:
        content = response.strip()
        
        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        return json.loads(content)
    except Exception:
        return None


# === 上下文增强分块配置 ===

# 为Chunk生成上下文前缀的Prompt
CHUNK_CONTEXT_PROMPT = """你是一个医学指南内容分析助手。请为以下指南片段生成一个简短的前缀描述，帮助后续检索理解上下文。

## 要求：
1. 描述这段文本所属的章节主题
2. 标注涉及的疾病类型和治疗阶段
3. 使用中文，控制在30-50字以内
4. 直接输出描述，不要添加任何解释

## 原文内容：
{chunk_content}

## 上下文前缀描述："""


def _generate_chunk_context(
    content: str,
    doc_metadata: Dict[str, Any],
    llm: ChatOpenAI,
) -> str:
    """
    为Chunk生成上下文前缀
    
    Args:
        content: Chunk内容
        doc_metadata: 文档级元数据
        llm: LLM实例
        
    Returns:
        上下文前缀描述
    """
    try:
        # 截取前800字符供LLM理解上下文
        sample_content = content[:800]
        
        prompt = CHUNK_CONTEXT_PROMPT.format(chunk_content=sample_content)
        response = llm.invoke([HumanMessage(content=prompt)])
        
        prefix = response.content.strip()
        
        # 如果返回为空，使用文档级元数据作为后备
        if not prefix:
            primary_disease = doc_metadata.get("primary_disease", "")
            doc_type = doc_metadata.get("document_type", "")
            prefix = f"{primary_disease} {doc_type}"
        
        return prefix
        
    except Exception as e:
        print(f"[ChunkContext] 生成上下文前缀失败: {e}")
        return doc_metadata.get("primary_disease", "")


def _enrich_chunk(
    chunk: Document,
    context_prefix: str,
    doc_metadata: Dict[str, Any],
) -> Document:
    """
    为Chunk添加上下文前缀
    
    Args:
        chunk: 原始Chunk
        context_prefix: 上下文前缀
        doc_metadata: 文档级元数据
        
    Returns:
        增强后的Chunk
    """
    # 构建增强后的内容：[上下文前缀] 原文内容
    enhanced_content = f"[{context_prefix}] {chunk.page_content}"
    
    # 创建新的元数据
    enhanced_metadata = {
        **chunk.metadata,
        "contextual_prefix": context_prefix,
        "is_contextual_enhanced": True,
        "doc_title": doc_metadata.get("document_title", ""),
        "doc_type": doc_metadata.get("document_type", ""),
        "content_type": doc_metadata.get("document_type", ""),
        "guideline_source": doc_metadata.get("guideline_source", ""),
        "primary_disease": doc_metadata.get("primary_disease", ""),
        "doc_summary": doc_metadata.get("document_summary", ""),
    }
    if doc_metadata.get("main_topics"):
        enhanced_metadata["main_topics"] = json.dumps(doc_metadata["main_topics"], ensure_ascii=False)

    return Document(
        page_content=enhanced_content,
        metadata=enhanced_metadata,
    )


# === 假设性问题嵌入配置 ===

# 为Chunk生成假设性问题的Prompt
HYPOTHETICAL_QUESTIONS_PROMPT = """基于以下医学指南内容，生成3个用户可能会提问的问题。

## 要求：
1. 问题使用口语化表达，符合患者或非专科医生的提问习惯
2. 涵盖"是什么"、"怎么做"、"为什么"三个维度
3. 每个问题控制在20-35字以内
4. 直接输出问题列表，每行一个，不要编号

## 指南内容：
{chunk_content}

## 假设性问题："""


def _generate_hypothetical_questions(
    content: str,
    llm: ChatOpenAI,
    max_questions: int = 3,
) -> List[str]:
    """
    为Chunk生成假设性问题
    
    Args:
        content: Chunk内容
        llm: LLM实例
        max_questions: 最大问题数量
        
    Returns:
        假设性问题列表
    """
    try:
        sample_content = content[:1000]
        
        prompt = HYPOTHETICAL_QUESTIONS_PROMPT.format(chunk_content=sample_content)
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # 解析问题列表
        questions = [
            q.strip().lstrip("0123456789.。、・")
            for q in response.content.split('\n')
            if q.strip() and len(q.strip()) > 10
        ]
        
        return questions[:max_questions]
        
    except Exception as e:
        print(f"[Hypothetical] 生成假设性问题失败: {e}")
        return []


def _add_hypothetical_questions(
    chunk: Document,
    questions: List[str],
) -> Document:
    """
    将假设性问题添加到Chunk中
    
    Args:
        chunk: 原始Chunk
        questions: 假设性问题列表
        
    Returns:
        增强后的Chunk
    """
    if not questions:
        return chunk
    
    # 将问题添加到内容末尾
    questions_text = f"\n\n[相关问题: {'; '.join(questions)}]"
    enhanced_content = chunk.page_content + questions_text
    
    # 更新元数据
    enhanced_metadata = {
        **chunk.metadata,
        "hypothetical_questions": json.dumps(questions, ensure_ascii=False),
        "has_hypothetical_questions": True,
    }
    
    return Document(
        page_content=enhanced_content,
        metadata=enhanced_metadata,
    )

def _chunk_document(
    content: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    简单的递归字符分块

    Args:
        content: 文档内容
        chunk_size: 分块大小
        chunk_overlap: 分块重叠区域大小
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.create_documents([content])


# === Document Processing ===

def _generate_chunk_id(content: str, source: str, index: int) -> str:
    """生成 chunk 的唯一 ID"""
    hash_input = f"{source}:{index}:{content[:100]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def _build_chroma_batch(chunks: List[Document]) -> Tuple[List[Document], List[str]]:
    docs: List[Document] = []
    ids: List[str] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        chunk_id = str(
            metadata.get("chunk_id")
            or _generate_chunk_id(
                chunk.page_content,
                str(metadata.get("source", "unknown")),
                int(metadata.get("chunk_index", 0) or 0),
            )
        )
        metadata["chunk_id"] = chunk_id
        docs.append(Document(page_content=chunk.page_content, metadata=metadata))
        ids.append(chunk_id)
    return docs, ids


def _extract_page_numbers(content: str) -> Dict[int, int]:
    """
    从解析后的内容中提取页码映射
    
    解析器会在每页内容前添加 "## Page X" 或 "## 第 X 页" 标记
    返回: {位置偏移: 页码} 的映射
    """
    page_map = {}
    # 匹配 "## Page X" 或 "## 第 X 页" 模式
    patterns = [
        r'##\s*Page\s+(\d+)',      # ## Page 45
        r'##\s*第\s*\d+\s*页',      # ## 第 45 页
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, content):
            page_num = int(match.group(1))
            position = match.start()
            page_map[position] = page_num
    
    return page_map


def _estimate_chunk_page(chunk_content: str, page_map: Dict[int, int]) -> int:
    """
    估算 chunk 所属的原始页码
    
    策略：
    1. 首先在 chunk 内容中查找页码标记（如 ## Page 45）
    2. 如果没找到，尝试在开头查找 "Page X" 或 "第 X 页" 模式
    3. 如果仍然没找到，返回保守估计值 1
    """
    if not chunk_content:
        return 1
    
    # 方式1：在 chunk 内容中查找完整的页码标记（如 ## Page 45 或 ## 第 45 页）
    for pattern in [r'##\s*Page\s+(\d+)', r'##\s*第\s*(\d+)\s*页']:
        match = re.search(pattern, chunk_content)
        if match:
            return int(match.group(1))
    
    # 方式2：在 chunk 开头查找不完整的页码标记（跨分块的情况）
    # 例如 chunk 可能以 "...Page 45" 结尾，或者以 "第 45 页" 结尾
    prefix_patterns = [
        r'(?:^|[^#]\s*)Page\s+(\d+)',      # Page 45（非 # 后面）
        r'(?:^|[^#]\s*)第\s*(\d+)\s*页',    # 第 45 页（非 # 后面）
    ]
    
    for pattern in prefix_patterns:
        match = re.search(pattern, chunk_content[:300])  # 检查开头300字符
        if match:
            return int(match.group(1))
    
    # 如果完全没找到页码信息，返回 1
    # 这是一个保守估计，可能会导致部分引用的页码不准确
    return 1


def _process_document(
    file_path: Path,
    parser: DocumentParser,
    llm: ChatOpenAI,
    chunk_size: int,
    chunk_overlap: int,
    skip_metadata: bool = False,
    enable_contextual_enhancement: bool = True,
    enable_hypothetical_questions: bool = True,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    处理单个文档（增强版）
    
    增强流程:
    1. 解析文档
    2. 提取文档级元数据（可选）
    3. 分块（保留页码信息）
    4. 提取章节标题层级（新增）
    5. 上下文增强：为每个Chunk生成语义前缀（新增）
    6. 假设性问题嵌入：生成口语化问答对（新增）
    7. 为每个 chunk 添加元数据
    
    Args:
        file_path: 文件路径
        parser: 文档解析器
        llm: LLM实例
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        skip_metadata: 是否跳过元数据增强
        enable_contextual_enhancement: 是否启用上下文增强
        enable_hypothetical_questions: 是否启用假设性问题嵌入
        
    Returns:
        (chunks, doc_metadata)
    """
    print(f"\n[Processing] {file_path.name}")
    
    # 1. 解析文档
    parsed = parser.parse(file_path)
    if not parsed.content.strip():
        print(f"[Skip] 空内容: {file_path.name}")
        return [], {}
    
    # 适配新版数据结构
    stats = parsed.perf_stats
    if stats:
        layout_aware = stats.get("layout_aware", False)
        print(f"  - 解析统计: 文本页={stats.get('hybrid_text_pages', 0)}, 视觉页={stats.get('hybrid_vision_pages', 0)}")
        if layout_aware:
            print(f"  - 解析模式: 布局感知解析")
    else:
        parser_name = parsed.metadata.get('parser', 'default')
        print(f"  - 解析器: {parser_name}")
    
    print(f"  - 内容长度: {len(parsed.content)} 字符")
    
    # 2. 提取文档级元数据
    doc_metadata = {}
    if not skip_metadata and llm:
        doc_metadata = _extract_doc_metadata(parsed, file_path.name, llm)
    else:
        doc_metadata = {"document_title": file_path.name}
    
    # 3. 从内容中提取页码信息
    page_map = _extract_page_numbers(parsed.content)
    
    # 4. 分块
    chunks = _chunk_document(
        parsed.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # 统计增强功能启用情况
    enhancement_count = 0
    
    # 5. 为每个 chunk 添加增强处理
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        # 显示进度（每10个chunk显示一次）
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  - 增强处理进度: {i + 1}/{total_chunks} chunks...")
        
        chunk.metadata["source"] = file_path.name
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = _generate_chunk_id(
            chunk.page_content, file_path.name, i
        )
        # 估算页码
        chunk.metadata["page"] = _estimate_chunk_page(chunk.page_content, page_map)
        
        # 5.1 提取章节标题层级（新增）
        headers = _extract_section_headers(parsed.content, chunk.page_content)
        if headers["h2"] or headers["h3"]:
            chunk.metadata["section_h2"] = headers["h2"] or ""
            chunk.metadata["section_h3"] = headers["h3"] or ""
            chunk.metadata["has_section_header"] = True
        else:
            chunk.metadata["has_section_header"] = False
        
        # 5.2 上下文增强（新增）
        if enable_contextual_enhancement and llm:
            context_prefix = _generate_chunk_context(chunk.page_content, doc_metadata, llm)
            chunk = _enrich_chunk(chunk, context_prefix, doc_metadata)
            enhancement_count += 1
        else:
            # 添加文档级元数据
            chunk.metadata["doc_title"] = doc_metadata.get("document_title", "")
            chunk.metadata["content_type"] = doc_metadata.get("document_type", "")
            chunk.metadata["guideline_source"] = doc_metadata.get("guideline_source", "")
            chunk.metadata["primary_disease"] = doc_metadata.get("primary_disease", "")
            chunk.metadata["doc_summary"] = doc_metadata.get("document_summary", "")
            if doc_metadata.get("main_topics"):
                chunk.metadata["main_topics"] = json.dumps(doc_metadata["main_topics"], ensure_ascii=False)
        
        # 5.3 假设性问题嵌入（新增）
        if enable_hypothetical_questions and llm:
            questions = _generate_hypothetical_questions(chunk.page_content, llm)
            if questions:
                chunk = _add_hypothetical_questions(chunk, questions)
                enhancement_count += 1
    
    print(f"  - 分块数量: {total_chunks}")
    if enable_contextual_enhancement or enable_hypothetical_questions:
        print(f"  - 增强处理: 完成 ({enhancement_count} 个chunks已增强)")
    
    return chunks, doc_metadata


def _extract_section_headers(content: str, chunk_content: str) -> Dict[str, str]:
    """
    从Chunk内容中提取最近的章节标题层级
    
    用于实现"在特定章节内搜索"这类精细化查询
    
    Args:
        content: 完整文档内容
        chunk_content: Chunk内容
        
    Returns:
        {"h1": 一级标题, "h2": 二级标题, "h3": 三级标题}
    """
    headers = {"h1": None, "h2": None, "h3": None}
    
    # 查找Chunk在文档中的起始位置
    try:
        position = content.find(chunk_content[:100])
        if position == -1:
            return headers
        
        # 向上查找最近的标题
        prefix = content[:position]
        lines = prefix.split('\n')
        
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.startswith('### '):
                headers["h3"] = stripped[4:].strip()
            elif stripped.startswith('## '):
                headers["h2"] = stripped[2:].strip()
                break  # 找到h2即可停止，向上不再查找
            elif stripped.startswith('# '):
                headers["h1"] = stripped[1:].strip()
                break
            
    except Exception:
        pass
    
    return headers


# === Main Ingest Function ===

def ingest(
    chunk_size: int = None,
    chunk_overlap: int = None,
    skip_metadata: bool = False,
    enable_contextual_enhancement: bool = True,
    enable_hypothetical_questions: bool = True,
) -> None:
    """
    加载指南文件并存入 Chroma（增强版）
    
    增强功能:
    1. 布局感知解析：智能识别表格、流程图等特殊页面
    2. 上下文增强：为每个Chunk生成语义前缀
    3. 假设性问题嵌入：生成口语化问答对
    4. 章节标题提取：支持按章节过滤检索
    
    Args:
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        skip_metadata: 是否跳过元数据增强
        enable_contextual_enhancement: 是否启用上下文增强
        enable_hypothetical_questions: 是否启用假设性问题嵌入
    """
    rag_config = _get_rag_settings()
    
    if chunk_size is None:
        chunk_size = rag_config["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = rag_config["chunk_overlap"]

    _ensure_embedding_env_for_embeddings()
    
    # 统计增强功能
    enhancements = []
    if enable_contextual_enhancement:
        enhancements.append("上下文增强")
    if enable_hypothetical_questions:
        enhancements.append("假设性问题")
    
    print(f"\n{'='*60}")
    print(f"[RAG Ingest] 开始处理 (增强版)")
    print(f"  - 分块参数: chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"  - 元数据增强: {'禁用' if skip_metadata else '启用'}")
    print(f"  - 智能解析: 启用 (布局感知)")
    if enhancements:
        print(f"  - 增强功能: {', '.join(enhancements)}")
    print(f"{'='*60}\n")
    
    # 初始化组件
    parse_strategy = (rag_config.get("parse_strategy") or "vision").lower()
    enable_vision = parse_strategy not in {"text", "basic", "pdf_text", "plain"}
    parser = create_parser(max_pages=rag_config.get("max_pages", 1000), enable_vision=enable_vision)
    llm = _get_metadata_llm() if not skip_metadata else None
    
    # 收集所有文档
    GUIDELINE_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Document] = []
    all_doc_metadata: List[Dict[str, Any]] = []
    
    for file_path in sorted(GUIDELINE_DIR.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".pdf", ".md", ".txt"}:
            continue
        
        chunks, doc_meta = _process_document(
            file_path=file_path,
            parser=parser,
            llm=llm,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_metadata=skip_metadata,
            enable_contextual_enhancement=enable_contextual_enhancement,
            enable_hypothetical_questions=enable_hypothetical_questions,
        )
        
        all_chunks.extend(chunks)
        if doc_meta:
            all_doc_metadata.append(doc_meta)
    
    if not all_chunks:
        print(f"[Warning] 未找到任何文档: {GUIDELINE_DIR}")
        return
    
    valid_chunks = [c for c in all_chunks if (c.page_content or "").strip()]
    dropped = len(all_chunks) - len(valid_chunks)
    if dropped:
        print(f"[Embedding] 跳过 {dropped} 个空内容 chunks")
    all_chunks = valid_chunks

    # 向量化并存储到 Chroma
    print(f"\n[Embedding] 向量化 {len(all_chunks)} 个 chunks...")
    embeddings = create_embeddings()
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    
    # 分批向量化以显示进度
    batch_size = 8
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for batch_start in range(0, len(all_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(all_chunks))
        batch_num = batch_start // batch_size + 1
        batch_chunks = all_chunks[batch_start:batch_end]
        
        print(f"  - 向量化进度: 批次 {batch_num}/{total_batches} ({batch_start + 1}-{batch_end}/{len(all_chunks)})")
        
        batch_docs, batch_ids = _build_chroma_batch(batch_chunks)
        vectorstore.add_documents(batch_docs, ids=batch_ids)

    # Rebuild BM25 from the same chunk set so hybrid retrieval can use exact-match signals.
    bm25_index = create_bm25_index()
    if bm25_index is not None:
        print(f"\n[BM25] Rebuilding lexical index from {len(all_chunks)} chunks...")
        bm25_index.clear()
        bm25_documents = []
        for chunk in all_chunks:
            meta = dict(chunk.metadata or {})
            bm25_documents.append(
                {
                    "id": meta.get("chunk_id") or _generate_chunk_id(
                        chunk.page_content,
                        str(meta.get("source", "unknown")),
                        int(meta.get("chunk_index", 0) or 0),
                    ),
                    "content": chunk.page_content,
                    **meta,
                }
            )
        bm25_index.add_documents(bm25_documents)
    
    print("[RAG Ingest] Ingest completed.")
    print(f"  - Documents processed: {len(all_doc_metadata)}")
    print(f"  - Chunks indexed: {len(all_chunks)}")
    print(f"  - Vector store: {PERSIST_DIR}")
    print(f"  - Collection: {COLLECTION_NAME}")
    return
    
    # 统计信息
    print(f"\n{'='*60}")
    print(f"[RAG Ingest] ✅ 入库完成!")
    print(f"  - 文档数: {len(all_doc_metadata)}")
    print(f"  - 分块数: {len(all_chunks)}")
    print(f"  - 向量存储: {PERSIST_DIR}")
    print(f"  - Collection: {COLLECTION_NAME}")
    if enhancements:
        print(f"  - 增强功能: {', '.join(enhancements)}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest guideline documents (enhanced version with contextual chunking)."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the persisted Chroma collections before ingesting."
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip LLM metadata enhancement (faster, for testing)."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Chunk size for text splitting (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--no-contextual",
        action="store_true",
        help="Disable contextual chunking enhancement."
    )
    parser.add_argument(
        "--no-hypothetical",
        action="store_true",
        help="Disable hypothetical questions embedding."
    )
    parser.add_argument(
        "--parse-strategy",
        type=str,
        default=None,
        help="Parsing strategy: vision or text (default from env RAG_PARSE_STRATEGY)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages per document (default from env RAG_MAX_PAGES)"
    )
    args = parser.parse_args()

    if args.reset:
        _ensure_embedding_env_for_embeddings()
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        embeddings = create_embeddings()

        for name in ("langchain", COLLECTION_NAME):
            try:
                vs = Chroma(
                    persist_directory=str(PERSIST_DIR),
                    embedding_function=embeddings,
                    collection_name=name,
                )
                vs.delete_collection()
                print(f"[reset] deleted collection={name}")
            except Exception as exc:
                print(f"[reset] skip delete collection={name}: {exc}")
        try:
            bm25_index = create_bm25_index()
            if bm25_index is not None:
                bm25_index.clear()
                print("[reset] cleared BM25 index")
        except Exception as exc:
            print(f"[reset] skip clearing BM25 index: {exc}")
    
    if args.parse_strategy:
        os.environ["RAG_PARSE_STRATEGY"] = args.parse_strategy
    if args.max_pages is not None:
        os.environ["RAG_MAX_PAGES"] = str(args.max_pages)

    ingest(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        skip_metadata=args.skip_metadata,
        enable_contextual_enhancement=not args.no_contextual,
        enable_hypothetical_questions=not args.no_hypothetical,
    )


if __name__ == "__main__":
    main()
