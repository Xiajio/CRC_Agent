

"""
Document Parser (Hybrid Optimized)

基于 PyMuPDF (fitz) 和 多模态 LLM 的智能文档解析器。

优化特性：
1. 混合解析策略 (Hybrid Strategy)：
   - 按页检测文本密度。
   - 文本页：直接提取文本 (0成本, 极速)。
   - 图片/扫描页：调用 Vision 模型进行 OCR 和图表描述。
2. 批量并发 (Batch Processing)： Vision 请求按批次发送，减少网络开销。
3. 结构化输出：保留页码信息，生成的 Markdown 结构更清晰。
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..services.llm_service import create_compatible_chat_openai

# 统一使用 PyMuPDF 作为核心引擎，因为它比 pypdf 更快且功能更强
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# 备用依赖
from pypdf import PdfReader


@dataclass
class ParsedDocument:
    """解析后的文档结构"""
    # 最终合并的 Markdown 文本 (用于 RAG 索引)
    content: str = ""
    # 原始元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 提取的图片资源 (page_num, image_bytes)
    images: List[Tuple[int, bytes]] = field(default_factory=list)
    # 解析性能统计
    perf_stats: Dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """
    智能混合文档解析器

    策略：
    1. 优先使用 PyMuPDF (fitz) 遍历页面。
    2. 使用布局感知判断页面类型（文本/图片/表格）。
    3. 文本页：直接提取文本 (0成本, 极速)。
    4. 图片页/扫描页/表格页：渲染为图片 -> 调用 Vision LLM。
    5. 最终按页码顺序合并所有内容。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        vision_model: str = "gpt-4o-mini",
        max_pages: int = 50,
        pdf_dpi: int = 150,  # 150 DPI 足够 OCR，且体积适中
        batch_size: int = 5,  # Vision 模式下每批处理的页数
        text_threshold: int = 50,  # 判定为文本页的最小字符数
        image_area_threshold: float = 0.25,  # 图片面积占比阈值
        enable_vision: Optional[bool] = None,
    ):
        # 优先从环境变量读取配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE") or os.getenv("LLM_API_BASE")
        self.vision_model = os.getenv("RAG_VISION_MODEL", vision_model)
        # 支持从环境变量读取 max_pages，如果没有则使用传入的参数或默认值
        self.max_pages = int(os.getenv("RAG_MAX_PAGES", str(max_pages)))
        self.pdf_dpi = pdf_dpi
        self.batch_size = batch_size
        self.text_threshold = text_threshold
        self.image_area_threshold = image_area_threshold

        strategy = (os.getenv("RAG_PARSE_STRATEGY") or "vision").lower()
        if enable_vision is None:
            enable_vision = strategy not in {"text", "basic", "pdf_text", "plain"}
        self.has_vision_capability = HAS_PYMUPDF and bool(self.api_key) and enable_vision

        # 打印初始化信息
        print(f"[DocumentParser] 初始化完成")
        if not HAS_PYMUPDF:
            print("  - ⚠️ 警告: 未检测到 PyMuPDF (fitz)，将降级使用 pypdf (仅文本)。")
        
        # 使用新变量构建状态描述
        modes = ["basic (text)"]
        if self.has_vision_capability:
            modes.append("vision (hybrid)")
            
        print(f"  - 启用模式: {', '.join(modes)}")
        print(f"  - 视觉模型: {self.vision_model if self.has_vision_capability else 'N/A'}")
    
    def _should_use_vision(self, page: fitz.Page) -> Tuple[bool, str]:
        """
        增强版页面类型判断：基于布局感知决定是否使用 Vision
        
        判断依据：
        1. 文本密度：字数过少 -> 需要看图
        2. 图片覆盖率：大尺寸图片（流程图等）-> 需要看图
        3. 表格特征：明确的表格结构 -> 需要看图
        
        Returns:
            (should_use_vision: bool, reason: str)
        """
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        # 提取文本并计算长度
        text = page.get_text()
        text_len = len(text.strip())
        
        # 规则1：字数极少（<100），扫描页可能提取不到文字
        if text_len < 100:
            return True, f"文本极少({text_len}字符)"
        
        # 规则2：检测大尺寸图片（可能是流程图、示意图）
        # 图片面积占比超过40%才需要看图
        images = page.get_images()
        if images:
            max_img_ratio = 0
            for img in images:
                img_area = self._calculate_image_area(img, page)
                if img_area > 0:
                    img_ratio = img_area / page_area
                    max_img_ratio = max(max_img_ratio, img_ratio)
            
            if max_img_ratio > 0.40:
                return True, f"图片面积占比高({max_img_ratio:.1%})"
        
        # 规则3：严格检测表格特征
        lines = text.split('\n')
        if self._looks_like_strict_table(lines):
            return True, "检测到表格（带分隔符）"
        
        # 规则4：TNM分期等表格关键词模式（后备）
        if self._has_tnm_pattern(text):
            return True, "检测到TNM分期表格"
        
        # 文本充足，直接提取
        return False, ""
    
    def _looks_like_strict_table(self, lines: List[str]) -> bool:
        """
        严格检测表格特征（必须有分隔符）
        
        真正的表格特征：
        1. 包含 | 分隔符 或 \t 制表符
        2. 多行（>3行）
        """
        # 过滤空行
        non_empty_lines = [line for line in lines if line.strip()]
        if len(non_empty_lines) < 4:
            return False
        
        # 检查是否包含表格分隔符
        has_separator = any('|' in line for line in lines)
        has_tab = any('\t' in line for line in lines)
        
        if has_separator or has_tab:
            # 分隔符必须出现在多行中
            lines_with_separator = sum(1 for line in lines if '|' in line or '\t' in line)
            if lines_with_separator >= 3:
                return True
        
        return False
    
    def _has_tnm_pattern(self, text: str) -> bool:
        """
        检测TNM分期表格特征（后备方案）

        用于识别没有分隔符但有明显表格结构的页面
        通过检测每行是否包含多个TNM标记来判断是否为表格行
        """
        lines = text.split('\n')
        table_row_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测该行中每种TNM模式的出现次数
            t_count = len(re.findall(r'T[0-9a-z]+', line, re.IGNORECASE))
            n_count = len(re.findall(r'N[0-9]', line, re.IGNORECASE))
            m_count = len(re.findall(r'M[0-9]', line, re.IGNORECASE))

            # 如果一行包含2个或以上的TNM标记，疑似表格行
            if t_count + n_count + m_count >= 2:
                table_row_count += 1

        # 至少3行疑似表格行才判定为表格
        return table_row_count >= 3
    
    def _calculate_image_area(self, img_ref: Tuple, page: fitz.Page) -> float:
        """计算图片在页面中的面积（像素单位）"""
        try:
            xref = img_ref[0]
            rects = page.get_image_rects(xref)
            if rects:
                return max(rect.width * rect.height for rect in rects)

            img_rect = img_ref[1] if len(img_ref) > 1 else None
            if isinstance(img_rect, fitz.Rect):
                return img_rect.width * img_rect.height

        except Exception as e:
            print(f"[Parser] 图片面积计算失败: {e}")

        # 返回-1标记计算失败，让后续的文本密度判断发挥作用
        return -1
    
    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """解析文件入口"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        start_time = time.time()

        # 读取文件内容
        content = path.read_bytes()
        filename = path.name
        suffix = path.suffix.lower()

        result = None

        if suffix == ".pdf":
            result = self._parse_pdf_hybrid(content, filename)
        elif suffix in [".txt", ".md", ".markdown"]:
            text = content.decode("utf-8", errors="ignore")
            result = ParsedDocument(content=text, metadata={"source": filename})
        else:
            # 默认为文本
            text = content.decode("utf-8", errors="ignore")
            result = ParsedDocument(content=text, metadata={"source": filename})

        # 记录总耗时
        result.perf_stats["total_time"] = round(time.time() - start_time, 2)
        return result

    def _parse_pdf_hybrid(self, content: bytes, filename: str) -> ParsedDocument:
        """
        PDF 混合解析核心逻辑（布局感知版）
        
        使用增强的页面类型判断：
        - 纯文本页：直接提取
        - 图片页/表格页/扫描页：调用 Vision OCR
        """
        # 降级模式：如果没有 PyMuPDF，使用 pypdf
        if not HAS_PYMUPDF:
            return self._parse_pdf_fallback(content, filename)

        doc = fitz.open(stream=content, filetype="pdf")
        total_pages = min(len(doc), self.max_pages)

        parsed_pages = {}  # {page_num: text_content}
        vision_queue = []  # [(page_num, img_bytes, reason), ...]
        skip_pages = set()  # 记录跳过的页面（已合并到前一页）

        print(f"[Parser] 开始解析 {filename} ({total_pages} 页)...")

        # 1. 扫描页面，使用布局感知分类处理
        for i in range(total_pages):
            page = doc[i]
            page_num = i + 1

            # 使用增强版判断逻辑
            should_use_vision, reason = self._should_use_vision(page)

            if should_use_vision:
                if self.has_vision_capability:
                    # 渲染为图片
                    pix = page.get_pixmap(dpi=self.pdf_dpi)
                    img_bytes = pix.tobytes("jpeg")
                    vision_queue.append((page_num, img_bytes, reason))
                else:
                    # 没有 Vision 能力，保留可用文本
                    text = page.get_text().strip()
                    if text:
                        parsed_pages[page_num] = f"## Page {page_num}\n{text}"
                    else:
                        parsed_pages[page_num] = f"## Page {page_num}\n[需要Vision识别但能力不可用]"
            else:
                # 文本充足 -> 直接使用
                text = page.get_text()
                parsed_pages[page_num] = f"## Page {page_num}\n{text}"

        # 2. 批量处理 Vision 队列
        if vision_queue:
            print(f"[Parser] 发现 {len(vision_queue)} 页特殊内容，调用 Vision 模型...")
            # 打印特殊页面类型统计
            reasons = [r for _, _, r in vision_queue]
            for reason in set(reasons):
                count = reasons.count(reason)
                print(f"  - {reason}: {count} 页")
            
            vision_results = self._process_vision_batch(vision_queue)
            parsed_pages.update(vision_results)

        # 3. 按页码顺序合并
        sorted_content = []
        for i in range(1, total_pages + 1):
            if i in parsed_pages:
                sorted_content.append(parsed_pages[i])

        final_markdown = "\n\n".join(sorted_content)

        return ParsedDocument(
            content=final_markdown,
            metadata={"source": filename, "pages": total_pages},
            perf_stats={
                "hybrid_text_pages": total_pages - len(vision_queue),
                "hybrid_vision_pages": len(vision_queue),
                "layout_aware": True,  # 标记使用布局感知解析
            }
        )

    def _process_vision_batch(self, queue: List[Tuple[int, bytes, str]]) -> Dict[int, str]:
        """
        批量调用 Vision API 处理图片页（带进度显示）
        
        Args:
            queue: [(page_num, img_bytes, reason), ...] - reason 包含识别原因
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        results = {}
        total_pages = len(queue)
        total_batches = (total_pages + self.batch_size - 1) // self.batch_size
        
        print(f"[Vision] 开始 OCR 识别 {total_pages} 页 ({total_batches} 批次)...")

        llm = create_compatible_chat_openai(
            model=self.vision_model,
            api_key=self.api_key,
            base_url=self.api_base,
            temperature=0,
            max_tokens=4096
        )

        # 优化的 Vision Prompt - 针对医学文档优化
        system_prompt = """You are a specialized medical document OCR assistant.
Convert the provided document image into structured Markdown text.

Rules:
1. Extract ALL text accurately, preserving medical terminology.
2. If there are tables (especially TNM staging tables, treatment protocols), format them as Markdown tables.
3. If there are flowcharts/diagrams, describe them in > blockquotes with [Flowchart] or [Diagram] tags.
4. Maintain the original structure and hierarchy.
5. Do not add conversational filler. Start directly with the content."""

        # 分批处理
        for i in range(0, len(queue), self.batch_size):
            batch = queue[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            # 显示进度
            page_nums = [p[0] for p in batch]
            print(f"[Vision] 处理批次 {batch_num}/{total_batches} (页码: {page_nums})...")

            # 构建多图消息
            content_parts = [{"type": "text", "text": "Please transcribe these document pages:"}]

            for page_num, img_bytes, reason in batch:
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "text",
                    "text": f"\n--- Page {page_num} ({reason}) ---\n"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}", "detail": "low"}
                })

            msg = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_parts)
            ]
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    resp = llm.invoke(msg)
                    batch_text = resp.content
                    results[batch[0][0]] = batch_text
                    break
                except Exception as e:
                    print(f"[Parser] Vision batch error (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
                    if attempt < max_retries:
                        time.sleep(1.5 * attempt)
                    else:
                        for page_num, img_bytes, reason in batch:
                            try:
                                page_text = self._recognize_single_page(llm, img_bytes, page_num, system_prompt)
                                if page_text:
                                    results[page_num] = f"## Page {page_num}\n{page_text}"
                                else:
                                    results[page_num] = f"## Page {page_num}\n[OCR Failed: {reason}]"
                            except Exception as e2:
                                print(f"[Parser] Vision single-page error: {type(e2).__name__}: {e2}")
                                results[page_num] = f"## Page {page_num}\n[OCR Failed: {reason}]"

        return results

    def _parse_pdf_fallback(self, content: bytes, filename: str) -> ParsedDocument:
        """pypdf 纯文本提取 (无 PyMuPDF 时的备选)"""
        try:
            reader = PdfReader(io.BytesIO(content))
            texts = []
            for i, page in enumerate(reader.pages):
                if i >= self.max_pages: break
                text = page.extract_text() or ""
                texts.append(f"## Page {i+1}\n{text}")

            return ParsedDocument(
                content="\n\n".join(texts),
                metadata={"source": filename, "parser": "pypdf_fallback"}
            )
        except Exception as e:
            return ParsedDocument(content="", metadata={"error": str(e)})

    def parse_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """
        从字节内容解析文档
        
        Args:
            content: 文件字节内容
            filename: 文件名
            
        Returns:
            ParsedDocument 对象
        """
        suffix = Path(filename).suffix.lower()
        
        if suffix == ".pdf":
            return self._parse_pdf_bytes(content, filename)
        elif suffix in {".md", ".markdown"}:
            text = content.decode("utf-8", errors="ignore")
            return ParsedDocument(
                raw_text=text,
                markdown=text,
                parse_info={"strategy": "direct", "format": "markdown"}
            )
        elif suffix in {".txt", ".text"}:
            text = content.decode("utf-8", errors="ignore")
            return ParsedDocument(
                raw_text=text,
                parse_info={"strategy": "direct", "format": "text"}
            )
        else:
            return ParsedDocument(
                parse_info={"error": f"Unsupported format: {suffix}"}
            )
    
    def _parse_pdf(self, file_path: Path) -> ParsedDocument:
        """解析 PDF 文件"""
        content = file_path.read_bytes()
        return self._parse_pdf_bytes(content, file_path.name)
    
    def _parse_pdf_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """解析 PDF 字节内容"""
        # 默认使用多模态识别
        if self.strategy == "vision" and self.available_parsers["vision"]:
            return self._parse_with_vision(content, filename)
        elif self.strategy == "basic":
            return self._parse_with_basic(content, filename)
        else:
            # 如果 vision 不可用，回退到 basic
            if not self.available_parsers["vision"]:
                print(f"[DocumentParser] 视觉模式不可用（缺少 PyMuPDF 或 API Key），回退到基础解析")
            return self._parse_with_basic(content, filename)
    
    def _parse_with_basic(self, content: bytes, filename: str) -> ParsedDocument:
        """基础文本提取 (pypdf) - 仅作为备用"""
        try:
            print(f"[Basic] 基础解析: {filename}")
            
            reader = PdfReader(io.BytesIO(content))
            texts = []
            markdown_parts = []
            
            page_count = min(len(reader.pages), self.max_pages)
            
            for i in range(page_count):
                page = reader.pages[i]
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text)
                    markdown_parts.append(f"## 第 {i + 1} 页\n\n{page_text}")
            
            raw_text = "\n\n".join(texts)
            markdown = "\n\n".join(markdown_parts)
            
            print(f"[Basic] 解析完成: {len(raw_text)} 字符, {page_count} 页")
            
            return ParsedDocument(
                raw_text=raw_text,
                markdown=markdown,
                parse_info={
                    "strategy": "basic",
                    "filename": filename,
                    "char_count": len(raw_text),
                    "page_count": page_count,
                }
            )
            
        except Exception as e:
            print(f"[Basic] 解析失败: {e}")
            return ParsedDocument(
                parse_info={"error": str(e), "filename": filename}
            )
    
    def _parse_with_vision(self, content: bytes, filename: str) -> ParsedDocument:
        """
        使用多模态 LLM 视觉识别
        
        完整识别每页的：
        - 文本内容
        - 图片（转为描述）
        - 表格（转为 Markdown 表格）
        - 流程图（转为结构化描述）
        """
        if not HAS_PYMUPDF:
            print("[Vision] PyMuPDF 不可用，回退到基础解析")
            return self._parse_with_basic(content, filename)
        
        if not self.api_key:
            print("[Vision] API Key 不可用，回退到基础解析")
            return self._parse_with_basic(content, filename)
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            print(f"[Vision] 多模态识别: {filename}")
            
            # PDF 转图片
            images = self._pdf_to_images(content)
            if not images:
                print("[Vision] PDF 转图片失败，回退到基础解析")
                return self._parse_with_basic(content, filename)
            
            print(f"[Vision] 共 {len(images)} 页需要识别")
            
            # 初始化 LLM (temperature=0 确保视觉识别稳定，max_tokens 从配置读取)
            try:
                from ..config import load_settings
                settings = load_settings()
                vision_max_tokens = settings.rag.metadata_max_tokens
            except Exception:
                vision_max_tokens = 4096
            
            llm = create_compatible_chat_openai(
                model=self.vision_model,
                api_key=self.api_key,
                base_url=self.api_base,
                temperature=0,  # 视觉识别需要稳定输出
                max_tokens=vision_max_tokens,
            )
            
            # 系统提示词
            system_prompt = """你是一个专业的医学文档OCR和内容提取助手。请识别图片中的所有内容，包括：

1. **文本内容**：识别所有文字，保持原有的段落结构
2. **表格**：转换为 Markdown 表格格式
3. **图片/图表**：描述图片内容，如流程图、示意图等
4. **流程图**：详细描述流程步骤和逻辑关系

输出要求：
- 使用 Markdown 格式组织内容
- 标题用 #, ##, ### 等标记
- 表格用 Markdown 表格语法
- 图片/流程图描述用 > 引用格式，标注 [图片描述] 或 [流程图]
- 保持原文语言（中文保持中文，英文保持英文）
- 确保医学术语准确

直接输出识别内容，不要添加额外解释。"""

            all_texts = []
            markdown_parts = []
            
            # 批量处理页面
            for batch_start in range(0, len(images), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(images))
                batch_images = images[batch_start:batch_end]
                
                print(f"[Vision] 处理第 {batch_start + 1}-{batch_end}/{len(images)} 页...")
                
                # 构建多图片请求
                content_parts = [
                    {"type": "text", "text": f"请识别以下 {len(batch_images)} 页文档内容（第 {batch_start + 1} 到 {batch_end} 页）："}
                ]
                
                for i, img_bytes in enumerate(batch_images):
                    base64_img = base64.b64encode(img_bytes).decode("utf-8")
                    is_jpeg = img_bytes[:2] == b'\xff\xd8'
                    mime_type = "image/jpeg" if is_jpeg else "image/png"
                    
                    page_num = batch_start + i + 1
                    content_parts.append({
                        "type": "text", 
                        "text": f"\n--- 第 {page_num} 页 ---"
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_img}",
                            "detail": "high"  # 使用高质量模式
                        }
                    })
                
                try:
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=content_parts)
                    ]
                    
                    response = llm.invoke(messages)
                    batch_text = response.content.strip()
                    
                    if batch_text:
                        all_texts.append(batch_text)
                        markdown_parts.append(batch_text)
                        
                except Exception as e:
                    print(f"[Vision] 批次 {batch_start + 1}-{batch_end} 识别失败: {e}")
                    # 单页重试
                    for i, img_bytes in enumerate(batch_images):
                        page_num = batch_start + i + 1
                        try:
                            page_text = self._recognize_single_page(llm, img_bytes, page_num, system_prompt)
                            if page_text:
                                all_texts.append(page_text)
                                markdown_parts.append(f"## 第 {page_num} 页\n\n{page_text}")
                        except Exception as e2:
                            print(f"[Vision] 第 {page_num} 页识别失败: {e2}")
            
            raw_text = "\n\n".join(all_texts)
            markdown = "\n\n".join(markdown_parts)
            
            print(f"[Vision] 识别完成: {len(raw_text)} 字符, {len(images)} 页")
            
            return ParsedDocument(
                raw_text=raw_text,
                markdown=markdown,
                parse_info={
                    "strategy": "vision",
                    "filename": filename,
                    "char_count": len(raw_text),
                    "page_count": len(images),
                    "model": self.vision_model,
                }
            )
            
        except Exception as e:
            print(f"[Vision] 识别失败: {e}")
            import traceback
            traceback.print_exc()
            return self._parse_with_basic(content, filename)
    
    def _recognize_single_page(
        self, 
        llm, 
        img_bytes: bytes, 
        page_num: int,
        system_prompt: str
    ) -> str:
        """单页识别（用于批量失败时的重试）"""
        from langchain_core.messages import HumanMessage, SystemMessage
        
        base64_img = base64.b64encode(img_bytes).decode("utf-8")
        is_jpeg = img_bytes[:2] == b'\xff\xd8'
        mime_type = "image/jpeg" if is_jpeg else "image/png"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": f"请识别第 {page_num} 页的内容："},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_img}",
                        "detail": "high"
                    }
                }
            ])
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
    
    def _pdf_to_images(self, content: bytes) -> List[bytes]:
        """将 PDF 转换为图片序列"""
        if not HAS_PYMUPDF:
            return []
        
        images = []
        try:
            pdf_doc = fitz.open(stream=content, filetype="pdf")
            page_count = min(len(pdf_doc), self.max_pages)
            
            print(f"[PDF转图片] 开始转换 {page_count} 页 (DPI={self.pdf_dpi})")
            
            for page_num in range(page_count):
                page = pdf_doc[page_num]
                zoom = self.pdf_dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                # 使用 JPEG 格式，质量 90，平衡清晰度和大小
                img_bytes = pix.tobytes("jpeg")
                images.append(img_bytes)
            
            pdf_doc.close()
            print(f"[PDF转图片] 成功转换 {page_count} 页")
            
        except Exception as e:
            print(f"[PDF转图片失败] {e}")
        
        return images
    
    def _parse_markdown(self, file_path: Path) -> ParsedDocument:
        """解析 Markdown 文件"""
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return ParsedDocument(
            raw_text=text,
            markdown=text,
            parse_info={
                "strategy": "direct",
                "format": "markdown",
                "filename": file_path.name,
            }
        )
    
    def _parse_text(self, file_path: Path) -> ParsedDocument:
        """解析纯文本文件"""
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        return ParsedDocument(
            raw_text=text,
            parse_info={
                "strategy": "direct",
                "format": "text",
                "filename": file_path.name,
            }
        )


def create_parser(**kwargs) -> DocumentParser:
    return DocumentParser(**kwargs)
