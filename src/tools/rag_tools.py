from __future__ import annotations

"""
RAG Tools with Metadata-Enhanced Search

提供多种检索模式:
1. 基础检索 - 向量相似度搜索
2. 治疗方案检索 - 专门针对治疗建议
3. 分期标准检索 - 专门针对诊断分期
4. 药物信息检索 - 专门针对药物说明
5. 指南来源检索 - 按 NCCN/CSCO/ESMO 过滤
"""

from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.rag.evidence import build_evidence_from_document, serialize_evidence_block
from src.rag.retriever import (
    get_retriever,
    search_with_metadata_filter,
    search_treatment_recommendations,
    search_staging_criteria,
    search_drug_information,
    search_by_guideline_source,
    hybrid_search,
    format_retrieved_docs,
)


# === Constants ===
CONTENT_TYPES = ["指南建议", "诊断标准", "治疗方案", "药物信息", "检查项目", "统计数据", "病例描述"]
MAX_CONTENT_LENGTH = 1200
RAG_TOOL_PROFILES = {
    "search_clinical_guidelines": "general",
    "search_treatment_recommendations": "treatment",
    "search_staging_criteria": "staging",
    "search_drug_information": "drug",
    "search_by_guideline_source": "source_filter",
    "hybrid_guideline_search": "hybrid",
    "list_guideline_toc": "toc",
    "read_guideline_chapter": "chapter",
}


# === Input Schemas ===
class RetrieverInput(BaseModel):
    """基础检索输入"""
    query: str = Field(
        default="结直肠癌 CRC 治疗 指南 NCCN CSCO ESMO adjuvant dMMR MSI-H",
        description="检索查询。用简短中文关键词描述你要查的场景/分期/分子标志物/治疗线数。",
    )
    top_k: int = Field(default=6, ge=1, le=15, description="返回命中片段条数。默认 6。")
    disease_focus: Optional[str] = Field(
        default=None,
        description="疾病名称过滤，如：结肠癌、直肠癌、结直肠癌",
    )


class TreatmentSearchInput(BaseModel):
    """治疗方案检索输入"""
    query: str = Field(
        default="结直肠癌 治疗方案 化疗 靶向 免疫治疗",
        description="检索查询，描述治疗场景，如：III期结肠癌辅助化疗",
    )
    disease: Optional[str] = Field(
        default=None,
        description="疾病名称过滤，如：结直肠癌、结肠癌、直肠癌、肝转移",
    )
    top_k: int = Field(default=6, ge=1, le=15, description="返回命中片段条数")


class StagingSearchInput(BaseModel):
    """分期标准检索输入"""
    query: str = Field(
        default="结直肠癌 TNM 分期 标准 诊断",
        description="检索查询，描述分期问题，如：T3N1M0分期标准",
    )
    top_k: int = Field(default=6, ge=1, le=15, description="返回命中片段条数")


class DrugSearchInput(BaseModel):
    """药物信息检索输入"""
    query: str = Field(
        default="结直肠癌 药物 化疗 靶向 免疫",
        description="检索查询，描述药物相关问题",
    )
    drug_name: Optional[str] = Field(
        default=None,
        description="具体药物名称，如：奥沙利铂、贝伐珠单抗、西妥昔单抗",
    )
    top_k: int = Field(default=6, ge=1, le=15, description="返回命中片段条数")


class GuidelineSourceInput(BaseModel):
    """指南来源检索输入"""
    query: str = Field(
        default="结直肠癌 治疗 指南 推荐",
        description="检索查询",
    )
    source: str = Field(
        description="指南来源: NCCN (美国)、CSCO (中国)、ESMO (欧洲)",
        pattern="^(NCCN|CSCO|ESMO)$",
    )
    top_k: int = Field(default=6, ge=1, le=15, description="返回命中片段条数")


class HybridSearchInput(BaseModel):
    """混合检索输入"""
    query: str = Field(
        default="结直肠癌 治疗 指南 NCCN CSCO",
        description="检索查询",
    )
    content_type: Optional[str] = Field(
        default=None,
        description=f"内容类型过滤: {', '.join(CONTENT_TYPES)}",
    )
    disease_focus: Optional[str] = Field(
        default=None,
        description="疾病类型过滤",
    )
    guideline_source: Optional[str] = Field(
        default=None,
        description="指南来源过滤: NCCN、CSCO、ESMO",
    )
    top_k: int = Field(default=6, ge=1, le=15, description="返回命中片段条数")


class GuidelineTOCInput(BaseModel):
    """指南目录查询输入"""
    guideline_name: str = Field(
        description="指南名称，如 'NCCN_Colon', 'CSCO_Rectal', 'NCCN', 'CSCO', 'ESMO'"
    )


class GuidelineReaderInput(BaseModel):
    """指南章节阅读输入"""
    guideline_name: str = Field(
        description="指南名称，如 'NCCN_Colon', 'CSCO_Rectal'"
    )
    chapter_name: str = Field(
        description="从目录中获取的章节名称，如 '治疗原则', 'Stage III Treatment'"
    )


# === Formatting ===
def _format_docs(
    docs: List[Document],
    include_metadata: bool = True,
    *,
    tool_name: str = "search_clinical_guidelines",
    query: str | None = None,
    retrieval_profile: str = "general",
) -> str:
    """
    格式化检索结果，支持结构化引用锚点。
    
    Output Format:
    
    [REF_1] [[Source:CSCO_2024.pdf|Page:45]]
    Content: ...
    
    <retrieved_metadata>[...]</retrieved_metadata>
    
    实现"双通道"传输：
    1. 文本通道：LLM 看到 [[Source:File|Page:N]]，学会引用这个 ID
    2. 数据通道：代码通过正则提取 <retrieved_metadata> JSON，State 获得精确的页码和文件路径
    """
    if not docs:
        return (
            "No relevant guideline chunks found. "
            "(提示：请确认已运行 `python -m src.rag.ingest` 将 `data/guidelines/*.pdf` 入库到 `chroma_db/`)"
        )

    import json
    
    parts: List[str] = []
    metadata_list: List[dict] = []
    evidence_list: List[dict] = []

    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        content = (getattr(d, "page_content", "") or "").strip()
        
        # 提取关键元数据
        source_file = meta.get("source", "unknown")
        # 假设 ingested documents 有 'page' 字段，如果没有则默认为 1
        page_num = meta.get("page", 1)
        # 如果 page 是字符串，尝试提取数字
        if isinstance(page_num, str):
            try:
                # 尝试从字符串中提取数字
                import re
                page_match = re.search(r'\d+', str(page_num))
                if page_match:
                    page_num = int(page_match.group())
                else:
                    page_num = 1
            except:
                page_num = 1
        elif not isinstance(page_num, int):
            page_num = 1
            
        doc_id = f"REF_{i}"
        
        # [优化点]: 生成前端可解析的锚点字符串
        # 格式: [[Source:Filename|Page:N]]
        # 清理文件名，移除路径前缀，只保留文件名
        source_name = source_file.split("/")[-1] if "/" in source_file else source_file
        source_name = source_name.split("\\")[-1] if "\\" in source_name else source_name
        citation_anchor = f"[[Source:{source_name}|Page:{page_num}]]"
        
        # 构建 LLM 阅读的文本块
        header = f"[{doc_id}] {citation_anchor}"
        
        # 添加元数据增强描述
        if include_metadata and meta.get("metadata_enhanced"):
            extras = []
            if meta.get("content_type"):
                extras.append(f"Type: {meta['content_type']}")
            if meta.get("evidence_level") and meta.get("evidence_level") != "无":
                extras.append(f"Level: {meta['evidence_level']}")
            if extras:
                header += f" ({' | '.join(extras)})"
        
        # 如果有摘要，添加到 header 下方
        if include_metadata and meta.get("summary"):
            header += f"\n📝 摘要: {meta['summary']}"

        # 截断过长内容
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH] + "..."

        parts.append(f"{header}\n{content}")
        
        # 收集结构化元数据，供 UI 使用
        metadata_list.append({
            "ref_id": doc_id,
            "source": source_name,
            "page": page_num,
            "preview": content[:100] + "..." if len(content) > 100 else content,
            "score": meta.get("score", 0),
            "content_type": meta.get("content_type"),
            "evidence_level": meta.get("evidence_level"),
        })
        evidence_list.append(
            build_evidence_from_document(
                Document(page_content=content, metadata=meta),
                index=i,
                query=query,
                tool_name=tool_name,
                retrieval_profile=retrieval_profile,
            )
        )

    # [关键优化]: 将 JSON 数据附加在末尾，供 Node 解析，但对 LLM 来说只是附加信息
    text_output = "\n\n".join(parts)
    json_output = json.dumps(metadata_list, ensure_ascii=False)
    
    # 使用 XML 标签包裹 JSON，方便 regex 提取
    evidence_output = serialize_evidence_block(evidence_list)
    return f"{text_output}\n\n{evidence_output}\n\n<retrieved_metadata>{json_output}</retrieved_metadata>"


# === Tools ===
class ClinicalGuidelineSearchTool(BaseTool):
    """基础临床指南搜索工具"""
    name: str = "search_clinical_guidelines"
    description: str = (
        "Primary Knowledge Source. Use this for:\n"
        "1. Standard of Care queries (CSCO/NCCN guidelines).\n"
        "2. Checking general treatment principles.\n"
        "3. Looking up guideline-based recommendations.\n"
        "Returns text with [[Source:File|Page:N]] anchors.\n"
        "When NOT to use: For real-time web search or latest research updates."
    )
    args_schema: type[BaseModel] = RetrieverInput

    def _run(self, query: str, top_k: int = 6, disease_focus: Optional[str] = None) -> str:
        q = (query or "").strip()
        if not q:
            q = RetrieverInput().query

        filters = {}
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
            filters["primary_disease"] = {"$in": disease_aliases.get(normalized, [disease_focus])}

        from src.rag.retriever import hybrid_search
        docs = hybrid_search(query=q, k=top_k, use_rerank=True, metadata_filter=filters if filters else None)
        
        return _format_docs(
            docs or [],
            tool_name=self.name,
            query=q,
            retrieval_profile=RAG_TOOL_PROFILES[self.name],
        )

    async def _arun(self, query: str, top_k: int = 6, disease_focus: Optional[str] = None) -> str:
        return self._run(query=query, top_k=top_k, disease_focus=disease_focus)


class TreatmentSearchTool(BaseTool):
    """治疗方案专用搜索工具"""
    name: str = "search_treatment_recommendations"
    description: str = (
        "Specialized Logic. Use ONLY when:\n"
        "1. Searching for specific Chemotherapy/Targeted regimens (e.g., 'FOLFOX vs XELOX').\n"
        "2. Looking for line-specific therapy (1st-line, 2nd-line).\n"
        "3. Querying stage-specific treatment recommendations.\n"
        "4. Neoadjuvant/Adjuvant treatment strategies.\n"
        "Do NOT use for general definitions or drug information (use search_drug_information instead)."
    )
    args_schema: type[BaseModel] = TreatmentSearchInput

    def _run(self, query: str, disease: Optional[str] = None, top_k: int = 6) -> str:
        q = (query or "").strip()
        if not q:
            return "请提供查询内容"

        # 使用混合检索，支持元数据过滤
        from src.rag.retriever import hybrid_search
        
        # 构建元数据过滤条件
        filters = {}
        if disease:
            normalized = str(disease).strip().lower()
            disease_aliases = {
                "colon cancer": ["Colon Cancer", "结直肠癌"],
                "结肠癌": ["Colon Cancer", "结直肠癌"],
                "rectal cancer": ["Rectal Cancer", "结直肠癌"],
                "直肠癌": ["Rectal Cancer", "结直肠癌"],
                "colorectal cancer": ["结直肠癌", "Colon Cancer", "Rectal Cancer"],
                "结直肠癌": ["结直肠癌", "Colon Cancer", "Rectal Cancer"],
            }
            filters["primary_disease"] = {"$in": disease_aliases.get(normalized, [disease])}
        
        docs = hybrid_search(query=q, k=top_k, use_rerank=True, metadata_filter=filters if filters else None)
        return _format_docs(
            docs or [],
            tool_name=self.name,
            query=q,
            retrieval_profile=RAG_TOOL_PROFILES[self.name],
        )

    async def _arun(self, query: str, disease: Optional[str] = None, top_k: int = 6) -> str:
        return self._run(query=query, disease=disease, top_k=top_k)


class StagingSearchTool(BaseTool):
    """分期标准搜索工具"""
    name: str = "search_staging_criteria"
    description: str = (
        "Use for TNM definitions. Examples:\n"
        "- 'What is T3 stage criteria?'\n"
        "- 'Define N2a nodes.'\n"
        "- 'TNM staging classification.'\n"
        "Helpful for validating Staging logic and understanding staging criteria."
    )
    args_schema: type[BaseModel] = StagingSearchInput

    def _run(self, query: str, top_k: int = 6) -> str:
        q = (query or "").strip()
        if not q:
            return "请提供查询内容"

        docs = search_staging_criteria(query=q, k=top_k)
        return _format_docs(
            docs or [],
            tool_name=self.name,
            query=q,
            retrieval_profile=RAG_TOOL_PROFILES[self.name],
        )

    async def _arun(self, query: str, top_k: int = 6) -> str:
        return self._run(query=query, top_k=top_k)


class DrugInfoSearchTool(BaseTool):
    """药物信息搜索工具"""
    name: str = "search_drug_information"
    description: str = (
        "Lookup for Drug Fact Sheets. Use for:\n"
        "- Dosage, Contraindications, Side Effects.\n"
        "- Targeted therapy markers (e.g., 'Cetuximab RAS status').\n"
        "- Drug interaction information.\n"
        "When NOT to use: For treatment regimen selection (use search_treatment_recommendations instead)."
    )
    args_schema: type[BaseModel] = DrugSearchInput

    def _run(self, query: str, drug_name: Optional[str] = None, top_k: int = 6) -> str:
        q = (query or "").strip()
        if not q:
            return "请提供查询内容"

        docs = search_drug_information(query=q, drug_name=drug_name, k=top_k)
        return _format_docs(
            docs or [],
            tool_name=self.name,
            query=q,
            retrieval_profile=RAG_TOOL_PROFILES[self.name],
        )

    async def _arun(self, query: str, drug_name: Optional[str] = None, top_k: int = 6) -> str:
        return self._run(query=query, drug_name=drug_name, top_k=top_k)


class GuidelineSourceSearchTool(BaseTool):
    """按指南来源搜索工具"""
    name: str = "search_by_guideline_source"
    description: str = (
        "Use when you need a SPECIFIC guideline source:\n"
        "- NCCN (American guidelines)\n"
        "- CSCO (Chinese guidelines)\n"
        "- ESMO (European guidelines)\n"
        "Use this when the query explicitly asks for a specific region's guidelines.\n"
        "When NOT to use: For general queries (use search_clinical_guidelines instead)."
    )
    args_schema: type[BaseModel] = GuidelineSourceInput

    def _run(self, query: str, source: str, top_k: int = 6) -> str:
        q = (query or "").strip()
        if not q:
            return "请提供查询内容"
        
        s = (source or "").strip().upper()
        if s not in ["NCCN", "CSCO", "ESMO"]:
            return f"无效的指南来源: {source}，请使用 NCCN/CSCO/ESMO"

        docs = search_by_guideline_source(query=q, source=s, k=top_k)
        return _format_docs(
            docs or [],
            tool_name=self.name,
            query=q,
            retrieval_profile=RAG_TOOL_PROFILES[self.name],
        )

    async def _arun(self, query: str, source: str, top_k: int = 6) -> str:
        return self._run(query=query, source=source, top_k=top_k)


class HybridSearchTool(BaseTool):
    """混合检索工具"""
    name: str = "hybrid_guideline_search"
    description: str = (
        "Advanced hybrid search: Combines vector similarity and metadata filtering.\n"
        "Use when:\n"
        "1. You need precise filtering by content_type, disease_focus, or guideline_source.\n"
        "2. The query requires multiple metadata constraints.\n"
        "3. Standard search tools return too many irrelevant results.\n"
        "When NOT to use: For simple queries (use search_clinical_guidelines instead)."
    )
    args_schema: type[BaseModel] = HybridSearchInput

    def _run(
        self, 
        query: str, 
        content_type: Optional[str] = None,
        disease_focus: Optional[str] = None,
        guideline_source: Optional[str] = None,
        top_k: int = 6
    ) -> str:
        q = (query or "").strip()
        if not q:
            return "请提供查询内容"

        # 构建过滤条件
        filters = {}
        if content_type:
            filters["content_type"] = content_type
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
            filters["primary_disease"] = {"$in": disease_aliases.get(normalized, [disease_focus])}
        if guideline_source:
            filters["guideline_source"] = guideline_source.upper()

        docs = hybrid_search(query=q, k=top_k, metadata_filter=filters if filters else None)
        return _format_docs(
            docs or [],
            tool_name=self.name,
            query=q,
            retrieval_profile=RAG_TOOL_PROFILES[self.name],
        )

    async def _arun(
        self, 
        query: str, 
        content_type: Optional[str] = None,
        disease_focus: Optional[str] = None,
        guideline_source: Optional[str] = None,
        top_k: int = 6
    ) -> str:
        return self._run(
            query=query, 
            content_type=content_type,
            disease_focus=disease_focus,
            guideline_source=guideline_source,
            top_k=top_k
        )


class GuidelineStructureTool(BaseTool):
    """
    【主动检索】查看指南目录
    
    当 Agent 不确定应该搜索什么具体内容时，先使用此工具
    查看指南的整体结构和章节目录，帮助规划下一步检索策略。
    """
    name: str = "list_guideline_toc"
    description: str = (
        "【主动上下文构建】查看指南目录结构。\n"
        "Use when:\n"
        "1. 不确定具体搜索关键词时，先查看指南全貌\n"
        "2. 需要了解某个指南包含哪些章节\n"
        "3. 想要浏览指南的整体组织结构\n"
        "Returns: 指南的章节目录列表，帮助你决定下一步要深入查看哪个章节。"
    )
    args_schema: type[BaseModel] = GuidelineTOCInput

    def _run(self, guideline_name: str) -> str:
        """
        返回指南的目录结构
        
        实现策略：
        1. 优先：如果有预先提取的目录元数据，直接返回
        2. 兜底：通过元数据查询获取该指南所有章节标题
        """
        g_name = (guideline_name or "").strip().upper()
        if not g_name:
            return "请提供指南名称，如 'NCCN', 'CSCO', 'ESMO'"

        # 尝试通过元数据获取章节列表
        try:
            from src.rag.retriever import search_with_metadata_filter
            
            # 查询该指南的所有文档
            docs = search_with_metadata_filter(
                query=f"{guideline_name} 目录 章节 treatment staging",
                guideline_source=g_name,
                k=20  # 获取更多结果以覆盖不同章节
            )
            
            if not docs:
                return f"未找到 {guideline_name} 指南的内容。请确认指南名称正确，或运行 `python -m src.rag.ingest` 导入指南。"
            
            # 提取章节信息
            chapters = set()
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                # 尝试提取章节信息
                chapter = meta.get("chapter") or meta.get("section") or meta.get("content_type")
                if chapter:
                    chapters.add(chapter)
            
            if chapters:
                chapter_list = sorted(list(chapters))
                result = f"📚 {guideline_name} 指南目录 (共 {len(chapter_list)} 个章节):\n\n"
                for i, ch in enumerate(chapter_list, 1):
                    result += f"{i}. {ch}\n"
                result += "\n💡 提示：使用 read_guideline_chapter 工具深入阅读特定章节。"
                return result
            else:
                # 兜底：返回基础信息
                return (
                    f"📚 {guideline_name} 指南已入库 (共 {len(docs)} 个文档片段)\n\n"
                    "💡 由于元数据中未提取章节信息，建议直接使用关键词搜索：\n"
                    "- 治疗方案: search_treatment_recommendations\n"
                    "- 分期标准: search_staging_criteria\n"
                    "- 药物信息: search_drug_information"
                )
        except Exception as e:
            return f"查询指南目录时出错: {str(e)}"

    async def _arun(self, guideline_name: str) -> str:
        return self._run(guideline_name=guideline_name)


class GuidelineReaderTool(BaseTool):
    """
    【主动检索】读取特定章节完整内容
    
    在通过 list_guideline_toc 确定目标章节后，使用此工具
    读取该章节的完整内容（非分片检索）。
    """
    name: str = "read_guideline_chapter"
    description: str = (
        "【主动上下文构建】读取指南特定章节的完整内容（非切片）。\n"
        "Use when:\n"
        "1. 已通过 list_guideline_toc 确定了目标章节\n"
        "2. 需要完整阅读某个章节，而非片段检索\n"
        "3. 想要获取某个主题的系统性内容\n"
        "Returns: 该章节的完整文本内容。"
    )
    args_schema: type[BaseModel] = GuidelineReaderInput

    def _run(self, guideline_name: str, chapter_name: str) -> str:
        """
        读取指定指南的指定章节
        
        实现策略：
        1. 通过元数据过滤定位该章节的所有文档片段
        2. 按页码排序后合并返回
        """
        g_name = (guideline_name or "").strip()
        c_name = (chapter_name or "").strip()
        
        if not g_name or not c_name:
            return "请提供指南名称和章节名称"

        try:
            from src.rag.retriever import search_with_metadata_filter
            
            # 查询该章节的所有文档（使用章节名作为查询）
            docs = search_with_metadata_filter(
                query=f"{guideline_name} {chapter_name}",
                guideline_source=g_name,
                k=15  # 获取足够多的片段以覆盖整个章节
            )
            
            if not docs:
                return f"未找到 {guideline_name} 指南中关于 '{chapter_name}' 的内容。"
            
            # 过滤出真正属于该章节的文档
            chapter_docs = []
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                doc_chapter = meta.get("chapter") or meta.get("section") or meta.get("content_type") or ""
                # 模糊匹配章节名
                if c_name.lower() in doc_chapter.lower():
                    chapter_docs.append(doc)
            
            if not chapter_docs:
                # 兜底：如果没有精确匹配，返回所有相关文档
                chapter_docs = docs[:10]
            
            # 按页码排序
            def get_page(doc):
                meta = getattr(doc, "metadata", {}) or {}
                page = meta.get("page", 0)
                try:
                    return int(page) if page else 0
                except:
                    return 0
            
            chapter_docs.sort(key=get_page)
            
            # 合并内容
            result = f"📖 {guideline_name} - {chapter_name}\n{'='*60}\n\n"
            for i, doc in enumerate(chapter_docs, 1):
                meta = getattr(doc, "metadata", {}) or {}
                page = meta.get("page", "N/A")
                content = getattr(doc, "page_content", "").strip()
                result += f"[Page {page}]\n{content}\n\n"
            
            result += f"\n{'='*60}\n"
            result += f"📊 共读取 {len(chapter_docs)} 个片段，总计约 {sum(len(getattr(d, 'page_content', '')) for d in chapter_docs)} 字符。"
            
            return result
        except Exception as e:
            return f"读取章节时出错: {str(e)}"

    async def _arun(self, guideline_name: str, chapter_name: str) -> str:
        return self._run(guideline_name=guideline_name, chapter_name=chapter_name)


# === Tool Factory ===
def get_guideline_tool() -> BaseTool:
    """获取基础指南搜索工具（向后兼容）"""
    return ClinicalGuidelineSearchTool()


def get_all_rag_tools() -> List[BaseTool]:
    """获取所有 RAG 工具"""
    return [
        ClinicalGuidelineSearchTool(),
        TreatmentSearchTool(),
        StagingSearchTool(),
        DrugInfoSearchTool(),
        GuidelineSourceSearchTool(),
        HybridSearchTool(),
        GuidelineStructureTool(),  # [新增] 主动上下文构建
        GuidelineReaderTool(),      # [新增] 章节完整阅读
    ]


def get_enhanced_rag_tools() -> List[BaseTool]:
    """
    获取增强版 RAG 工具集（推荐使用）
    
    包含:
    - search_clinical_guidelines: 基础检索
    - search_treatment_recommendations: 治疗方案检索
    - search_staging_criteria: 分期标准检索
    - search_drug_information: 药物信息检索
    - list_guideline_toc: 【新增】主动浏览指南目录
    - read_guideline_chapter: 【新增】完整阅读指南章节
    """
    return [
        ClinicalGuidelineSearchTool(),
        TreatmentSearchTool(),
        StagingSearchTool(),
        DrugInfoSearchTool(),
        GuidelineStructureTool(),  # [新增] 主动上下文构建
        GuidelineReaderTool(),      # [新增] 章节完整阅读
    ]
