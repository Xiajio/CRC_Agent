"""
联网搜索服务 (Web Search Service)

使用 gpt-4o-search-preview 模型进行联网资料搜集。

特点：
1. 详细的查询输入，确保搜索准确
2. 严格的结果验证，避免幻觉
3. 明确标注信息来源
4. 无相关资料时明确返回"没有找到相关资料"

深度研究服务 (Deep Research Service) - Agentic Optimization
============================================================
引入"规划-执行-综合"的循环模式：

特点：
1. 结构化输出：使用 Pydantic 强制返回 List[SourceItem]
2. 查询拆解：对于复杂问题，生成 3-5 个子查询，并行搜索
3. 显式引用检查：在 Python 层过滤黑名单域名
4. 循证综合：将多个来源综合为最终报告，每句话标注引用来源
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator


# ================= 全局单例管理 =================

class _WebSearchServiceManager:
    """
    WebSearchService 全局单例管理器
    
    【优化】确保 WebSearchService 只初始化一次，避免重复实例化
    """
    _instance: Optional["_WebSearchServiceManager"] = None
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
        
        self._web_search_service: Optional["WebSearchService"] = None
        self._deep_research_service: Optional["DeepResearchService"] = None
        self._init_lock = threading.Lock()
        self._initialized = True
    
    def get_web_search_service(self) -> "WebSearchService":
        """
        获取全局 WebSearchService 单例
        
        [线程安全] 使用双重检查锁定模式
        """
        if self._web_search_service is None:
            with self._init_lock:
                if self._web_search_service is None:
                    print("[WebSearchService] 🚀 首次初始化全局单例...")
                    self._web_search_service = WebSearchService()
                    print("[WebSearchService] ✅ 全局单例初始化完成，后续将复用")
        return self._web_search_service
    
    def get_deep_research_service(self) -> "DeepResearchService":
        """
        获取全局 DeepResearchService 单例
        """
        if self._deep_research_service is None:
            with self._init_lock:
                if self._deep_research_service is None:
                    print("[DeepResearchService] 🚀 首次初始化全局单例...")
                    self._deep_research_service = DeepResearchService()
                    print("[DeepResearchService] ✅ 全局单例初始化完成，后续将复用")
        return self._deep_research_service
    
    def is_initialized(self) -> bool:
        """检查 WebSearchService 是否已初始化"""
        return self._web_search_service is not None


# 全局管理器单例
_global_ws_manager = _WebSearchServiceManager()


# ================= 1. 结构化定义 (机器可读) =================

class SourceItem(BaseModel):
    """结构化来源信息 - 便于后续节点引用"""
    title: str = Field(description="资料标题")
    url: str = Field(description="来源链接")
    snippet: str = Field(description="关键内容摘要 (含具体数据)")
    source_type: str = Field(
        description="来源类型: Guideline/Paper/News/Official/Database",
        pattern="^(Guideline|Paper|News|Official|Database)$"
    )
    year: str = Field(description="发布年份")
    credibility_score: int = Field(description="可信度评分 (1-10)", ge=1, le=10)

    @field_validator("url")
    @classmethod
    def url_must_be_valid(cls, v: str) -> str:
        """验证 URL 格式基本有效性"""
        url_pattern = r"^https?://[^\s]+$"
        if not re.match(url_pattern, v):
            raise ValueError(f"URL 格式无效: {v}")
        return v


class ResearchResult(BaseModel):
    """
    结构化的研究报告，便于后续节点引用
    
    设计目标：
    1. 可直接被后续图节点消费
    2. 支持来源追溯和验证
    3. 明确标注缺失信息
    """
    summary: str = Field(description="综合回答（每句话末尾标注引用来源）")
    sources: List[SourceItem] = Field(description="引用来源列表")
    missing_info: List[str] = Field(
        default_factory=list,
        description="仍未找到的信息"
    )
    sub_queries_used: List[str] = Field(
        default_factory=list,
        description="使用的子查询列表"
    )


class WebSearchService:
    """
    联网搜索服务
    
    使用 gpt-4o-search-preview 模型进行实时网络搜索，
    获取最新的医学资料、研究文献、临床指南更新等。
    """
    
    # 搜索系统提示词
    SEARCH_SYSTEM_PROMPT = """你是一个专业的医学资料搜索助手。你的任务是：

1. **搜索网络资料**：根据用户的查询，搜索最新、最权威的医学资料
2. **严格验证信息**：只返回你确实从网络搜索到的信息，不要编造任何内容
3. **标注来源**：所有信息必须标注来源（文献、指南、官网等）
4. **承认无知**：如果搜索不到相关资料，必须明确说明"没有找到相关资料"

## 输出格式要求

如果找到相关资料，按以下格式输出：
```
## 搜索结果

### 1. [标题]
- **来源**: [具体来源，如 NCCN官网、PubMed文献、医院官网等]
- **发布时间**: [如果能确定]
- **内容摘要**: [关键信息]

### 2. [标题]
...

## 来源汇总
- [来源1 URL 或文献引用]
- [来源2 URL 或文献引用]
```

如果没有找到相关资料，输出：
```
## 搜索结果

没有找到与"{查询内容}"相关的资料。

可能的原因：
1. 该问题过于专业或小众
2. 该问题涉及最新研究，尚未有公开资料
3. 查询关键词需要调整

建议：
- 尝试使用不同的关键词
- 查询更通用的相关主题
```

## 重要提示
- **绝对不要编造信息**：如果不确定，说"不确定"或"未找到"
- **不要猜测**：不要根据推理生成可能的答案
- **保持诚实**：宁可说"没有找到"也不要编造"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = None,  # 从配置读取
        temperature: float = None,  # 从配置读取
        max_tokens: int = None,  # 从配置读取
    ):
        """
        初始化联网搜索服务
        
        Args:
            api_key: API Key
            api_base: API Base URL
            model: 搜索模型（默认从配置读取）
            temperature: 温度参数（低温度确保结果一致）
            max_tokens: 最大输出 tokens
        """
        # 从配置加载默认值
        try:
            from ..config import load_settings
            settings = load_settings()
            ws_config = settings.web_search
            llm_config = settings.llm
            
            self.api_key = api_key or llm_config.api_key or os.getenv("LLM_API_KEY")
            self.api_base = api_base or llm_config.api_base or os.getenv("LLM_API_BASE")
            self.model = model or ws_config.model
            self.temperature = temperature if temperature is not None else ws_config.temperature
            self.max_tokens = max_tokens if max_tokens is not None else ws_config.max_tokens
        except Exception:
            # 回退到环境变量
            self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            self.api_base = api_base or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE")
            self.model = model or os.getenv("WEB_SEARCH_MODEL", "gpt-4o-search-preview")
            self.temperature = temperature if temperature is not None else float(os.getenv("WEB_SEARCH_TEMPERATURE", "0.1"))
            self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("WEB_SEARCH_MAX_TOKENS", "4096"))
        
        if not self.api_key:
            raise ValueError("需要提供 API Key (LLM_API_KEY 或 OPENAI_API_KEY)")
        
        self.llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # [优化] 只在非单例模式下打印初始化日志
        # 单例模式的初始化日志由管理器打印
        if not _global_ws_manager.is_initialized():
            print(f"[WebSearchService] 初始化完成，模型: {self.model}")
    
    def search(
        self,
        query: str,
        context: Optional[str] = None,
        search_type: str = "general",
    ) -> Dict[str, Any]:
        """
        执行联网搜索
        
        Args:
            query: 搜索查询
            context: 上下文信息（可选，如患者信息、具体场景）
            search_type: 搜索类型 (general/clinical/drug/guideline/research)
        
        Returns:
            搜索结果字典
        """
        # 构建详细的搜索请求
        search_prompt = self._build_search_prompt(query, context, search_type)
        
        try:
            messages = [
                SystemMessage(content=self.SEARCH_SYSTEM_PROMPT),
                HumanMessage(content=search_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result_text = response.content.strip()
            
            # 验证结果是否有效
            has_results = self._validate_results(result_text)
            
            return {
                "success": True,
                "has_results": has_results,
                "query": query,
                "search_type": search_type,
                "result": result_text,
                "model": self.model,
            }
            
        except Exception as e:
            print(f"[WebSearch] 搜索失败: {e}")
            return {
                "success": False,
                "has_results": False,
                "query": query,
                "error": str(e),
                "result": f"搜索出错: {str(e)}",
            }
    
    def search_clinical_evidence(
        self,
        topic: str,
        disease: Optional[str] = None,
        treatment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        搜索临床证据
        
        Args:
            topic: 搜索主题
            disease: 疾病名称
            treatment: 治疗方案
        
        Returns:
            搜索结果
        """
        # 构建详细的临床查询
        query_parts = [f"最新临床证据：{topic}"]
        if disease:
            query_parts.append(f"疾病：{disease}")
        if treatment:
            query_parts.append(f"治疗方案：{treatment}")
        
        context = "\n".join([
            "请搜索以下临床相关资料：",
            f"- 主题: {topic}",
            f"- 疾病: {disease or '不限'}",
            f"- 治疗: {treatment or '不限'}",
            "",
            "优先搜索：",
            "1. 最新临床指南（NCCN、ESMO、CSCO 等）",
            "2. 高质量临床研究（RCT、Meta分析）",
            "3. 权威医学数据库（PubMed、Cochrane）",
            "4. 药品说明书和官方资料",
        ])
        
        return self.search(
            query=" | ".join(query_parts),
            context=context,
            search_type="clinical",
        )
    
    def search_drug_info(
        self,
        drug_name: str,
        info_type: str = "all",
    ) -> Dict[str, Any]:
        """
        搜索药物信息
        
        Args:
            drug_name: 药物名称
            info_type: 信息类型 (all/dosage/interaction/adverse/indication)
        
        Returns:
            搜索结果
        """
        info_type_map = {
            "all": "完整药物信息",
            "dosage": "用法用量",
            "interaction": "药物相互作用",
            "adverse": "不良反应",
            "indication": "适应症",
        }
        
        info_desc = info_type_map.get(info_type, "完整药物信息")
        
        context = f"""请搜索药物 "{drug_name}" 的 {info_desc}。

优先搜索：
1. 药品说明书（官方版本）
2. FDA / NMPA 药品信息
3. 权威药物数据库
4. 临床指南中的用药建议

注意事项：
- 必须标注信息来源
- 剂量信息要精确
- 特殊人群用药需注明
- 如无可靠来源，请明确说明"""
        
        return self.search(
            query=f"药物 {drug_name} {info_desc}",
            context=context,
            search_type="drug",
        )
    
    def search_latest_guidelines(
        self,
        disease: str,
        guideline_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        搜索最新指南更新
        
        Args:
            disease: 疾病名称
            guideline_source: 指南来源 (NCCN/CSCO/ESMO/all)
        
        Returns:
            搜索结果
        """
        source_desc = guideline_source if guideline_source and guideline_source != "all" else "各大权威机构"
        
        context = f"""请搜索 "{disease}" 的最新临床指南更新。

搜索范围：
- 来源: {source_desc}
- 时间: 优先最近1-2年的更新

需要的信息：
1. 指南版本和发布日期
2. 主要更新内容
3. 与之前版本的关键差异
4. 官方发布链接

如果没有找到最新更新，请明确说明"未找到{disease}的最新指南更新"。"""
        
        return self.search(
            query=f"{disease} 最新临床指南 {source_desc} 更新",
            context=context,
            search_type="guideline",
        )
    
    def search_research(
        self,
        topic: str,
        study_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        搜索最新研究
        
        Args:
            topic: 研究主题
            study_type: 研究类型 (RCT/meta-analysis/cohort/all)
        
        Returns:
            搜索结果
        """
        study_desc = study_type if study_type and study_type != "all" else "各类临床研究"
        
        context = f"""请搜索关于 "{topic}" 的最新研究。

搜索优先级：
1. 最新发表的高质量研究（近1-3年）
2. 研究类型: {study_desc}
3. 发表在高影响因子期刊

需要的信息：
- 研究标题和作者
- 发表期刊和时间
- 主要研究结论
- 研究的临床意义

如果没有找到相关研究，请明确说明。"""
        
        return self.search(
            query=f"{topic} 最新研究 {study_desc}",
            context=context,
            search_type="research",
        )
    
    def _build_search_prompt(
        self,
        query: str,
        context: Optional[str],
        search_type: str,
    ) -> str:
        """构建详细的搜索提示"""
        search_type_hints = {
            "general": "通用医学信息搜索",
            "clinical": "临床证据和指南搜索",
            "drug": "药物信息搜索",
            "guideline": "临床指南搜索",
            "research": "最新研究搜索",
        }
        
        type_hint = search_type_hints.get(search_type, "通用搜索")
        
        prompt_parts = [
            f"## 搜索请求",
            f"",
            f"**搜索类型**: {type_hint}",
            f"**查询内容**: {query}",
        ]
        
        if context:
            prompt_parts.extend([
                f"",
                f"**详细上下文**:",
                context,
            ])
        
        prompt_parts.extend([
            f"",
            f"---",
            f"",
            f"请执行网络搜索，返回最相关、最权威的信息。",
            f"如果没有找到可靠信息，请明确说明\"没有找到相关资料\"。",
        ])
        
        return "\n".join(prompt_parts)
    
    def _validate_results(self, result_text: str) -> bool:
        """验证搜索结果是否有效"""
        # 检查是否明确表示没有找到结果
        no_result_patterns = [
            "没有找到",
            "未找到",
            "无法找到",
            "找不到",
            "no results",
            "not found",
        ]
        
        result_lower = result_text.lower()
        for pattern in no_result_patterns:
            if pattern in result_lower:
                return False
        
        # 检查是否有实质内容
        if len(result_text) < 100:
            return False
        
        return True


# ================= 2. 深度研究服务类 (Agentic Optimization) =================

class DeepResearchService(WebSearchService):
    """
    深度联网研究服务 - Agentic Optimization
    
    引入"规划-执行-综合"的循环模式：
    1. 规划层 (Planning): 将复杂问题拆解为多个子查询
    2. 执行层 (Execution): 并行执行多个搜索任务
    3. 综合层 (Synthesis): 将多个来源综合为结构化报告
    
    特点：
    - 结构化输出：使用 Pydantic 强制返回 SourceItem 列表
    - 显式引用：每一句话都标注来源索引
    - 安全边界：过滤低质/黑名单来源
    """
    
    # 来源黑名单 - 过滤低质量来源
    SOURCE_BLACKLIST = [
        "baidu.com",
        "zhidao.baidu.com",
        "baike.baidu.com",
        "zhihu.com",
        "www.zhihu.com",
        "iask.sina.com.cn",
        "kankan.sina.com.cn",
        "qq.com",
        "weibo.com",
        "sina.com.cn",
        "sohu.com",
        "douban.com",
        "blog.sina.com.cn",
        "blog.csdn.net",
        "cnblogs.com",
        "jianshu.com",
        "36kr.com",
        "geekpark.net",
    ]
    
    # 来源可信度评分参考
    CREDIBILITY_REFERENCE = {
        "nccn.org": (9, "Guideline"),
        "csco.org.cn": (9, "Guideline"),
        "esmo.org": (9, "Guideline"),
        "cancer.gov": (9, "Official"),
        "fda.gov": (9, "Official"),
        "nmpa.gov.cn": (9, "Official"),
        "pubmed.gov": (8, "Paper"),
        "nejm.org": (10, "Paper"),
        "thelancet.com": (10, "Paper"),
        "jamanetwork.com": (10, "Paper"),
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        llm_fast: Optional[ChatOpenAI] = None,
    ):
        """
        初始化深度研究服务
        
        Args:
            llm_fast: 用于查询拆解的快速 LLM 实例（可选，默认使用主 LLM）
        """
        super().__init__(api_key, api_base, model, temperature, max_tokens)
        
        # 快速 LLM 用于查询拆解（使用较高温度以获得创造性）
        if llm_fast is None:
            self.llm_fast = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.api_base,
                temperature=0.3,
                max_tokens=1024,
            )
        else:
            self.llm_fast = llm_fast
        
        print(f"[DeepResearchService] 初始化完成")
    
    def _decompose_query(self, complex_query: str) -> List[str]:
        """
        [规划层] 将复杂问题拆解为多个具体的搜索关键词
        
        例如："直肠癌术后治疗" -> 
        [
            "直肠癌 NCCN 指南 2024 术后治疗",
            "直肠癌术后辅助化疗 最新研究 2024",
            "直肠癌术后放化疗方案 CapOx FOLFOX 对比"
        ]
        
        Args:
            complex_query: 复杂查询
            
        Returns:
            拆解后的子查询列表（最多5个）
        """
        decomposition_prompt = f"""作为一个专业的医学研究员，将以下复杂临床查询拆解为 3-5 个具体的、互补的搜索引擎关键词。

目标：
1. 每个关键词应该能独立搜索到权威医学资料
2. 覆盖不同维度（指南、研究、药物、方案等）
3. 包含必要的限定词（年份、机构、方案名称等）

查询: {complex_query}

要求：
- 只返回关键词列表，每行一个
- 不要编号，不要前缀
- 不要超过 5 个
- 直接开始输出"""

        try:
            response = self.llm_fast.invoke([
                SystemMessage(content="你是一个专业的医学研究助手，擅长将复杂问题拆解为有效的搜索关键词。"),
                HumanMessage(content=decomposition_prompt)
            ])
            
            queries = []
            for line in response.content.strip().split("\n"):
                line = line.strip()
                # 移除可能的编号、前缀
                line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)
                if line and len(line) >= 5:
                    queries.append(line)
            
            # 限制数量并返回
            return queries[:5] if queries else [complex_query]
            
        except Exception as e:
            print(f"   [Query Decomposition] 失败，使用原始查询: {e}")
            return [complex_query]
    
    def _is_blacklisted_source(self, url: str) -> bool:
        """
        检查来源是否在黑名单中
        
        Args:
            url: 来源 URL
            
        Returns:
            True 如果在黑名单中
        """
        url_lower = url.lower()
        for domain in self.SOURCE_BLACKLIST:
            if domain in url_lower:
                return True
        return False
    
    def _calculate_credibility(self, url: str, source_type: str) -> int:
        """
        计算来源可信度评分
        
        Args:
            url: 来源 URL
            source_type: 来源类型
            
        Returns:
            可信度评分 (1-10)
        """
        # 先检查可信度参考表
        for domain, (score, ref_type) in self.CREDIBILITY_REFERENCE.items():
            if domain in url.lower():
                return score
        
        # 根据来源类型给基础分
        type_scores = {
            "Guideline": 9,
            "Paper": 8,
            "Official": 8,
            "Database": 7,
            "News": 5,
        }
        return type_scores.get(source_type, 5)
    
    def _extract_year(self, text: str) -> str:
        """
        从文本中提取年份
        
        Args:
            text: 包含年份的文本
            
        Returns:
            年份字符串
        """
        # 匹配 20XX 或 19XX 年份
        year_match = re.search(r"(19|20)\d{2}", text)
        if year_match:
            return year_match.group(0)
        return "未知"
    
    def _search_single_structured(self, query: str) -> List[SourceItem]:
        """
        [执行层] 单次搜索，强制返回结构化数据
        
        使用 with_structured_output 绑定 SourceItem 列表
        
        Args:
            query: 搜索查询
            
        Returns:
            SourceItem 列表
        """
        # 使用结构化输出 LLM
        structured_llm = self.llm.with_structured_output(List[SourceItem])
        
        system_prompt = """你是一个医学搜索引擎接口，负责从搜索结果中提取结构化信息。

请根据搜索查询，提取 3-5 个最权威的来源。

【来源优先级】
1. NCCN/CSCO/ESMO 指南原文 (source_type: "Guideline")
2. PubMed/The Lancet/NEJM/JAMA 核心期刊 (source_type: "Paper")
3. FDA/NMPA/官方机构发布 (source_type: "Official")
4. 权威数据库 (source_type: "Database")

【输出要求】
- title: 资料标题
- url: 完整的官方链接（必须是真实可访问的）
- snippet: 包含具体数据的关键内容摘要（至少50字）
- source_type: 从 Guideline/Paper/News/Official/Database 中选择
- year: 发布年份
- credibility_score: 根据来源权威性评分 (1-10)

【严禁】
- 不要引用百度百科、知乎、个人博客
- 不要编造 URL
- 不要使用通用搜索引擎结果页作为来源"""

        try:
            response = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"请搜索并提取以下查询的权威来源:\n\n查询: {query}")
            ])
            
            # 后处理：验证和过滤
            valid_sources = []
            for item in response:
                # URL 黑名单检查
                if self._is_blacklisted_source(item.url):
                    print(f"   [Filter] 过滤黑名单来源: {item.title[:30]}...")
                    continue
                
                # 重新计算可信度评分（确保一致性）
                item.credibility_score = self._calculate_credibility(item.url, item.source_type)
                
                # 确保年份字段有效
                if item.year == "未知" or not item.year:
                    item.year = self._extract_year(item.snippet)
                
                valid_sources.append(item)
            
            return valid_sources
            
        except Exception as e:
            print(f"   [Search Error] {query[:50]}...: {e}")
            return []
    
    def search_deep(
        self,
        query: str,
        max_workers: int = 3,
        min_sources: int = 3,
    ) -> ResearchResult:
        """
        [主入口] 深度搜索模式：拆解 -> 并行搜索 -> 综合
        
        Args:
            query: 原始查询
            max_workers: 最大并行线程数
            min_sources: 最少需要的有效来源数
            
        Returns:
            ResearchResult: 结构化的研究报告
        """
        print(f"🔍 [DeepResearch] 正在分析查询: {query}")
        
        # 1. 规划层：拆解查询
        sub_queries = self._decompose_query(query)
        print(f"   -> 拆解为 {len(sub_queries)} 个子任务:")
        for i, q in enumerate(sub_queries, 1):
            print(f"      {i}. {q}")
        
        # 2. 执行层：并行搜索
        print(f"   -> 并行执行搜索...")
        raw_results: List[SourceItem] = []
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(sub_queries))) as executor:
            futures = [
                executor.submit(self._search_single_structured, q) 
                for q in sub_queries
            ]
            for future in futures:
                result = future.result()
                if result:
                    raw_results.extend(result)
        
        print(f"   -> 原始结果: {len(raw_results)} 条")
        
        # 3. 去重与清洗 (Python 逻辑)
        unique_sources: Dict[str, SourceItem] = {}
        for item in raw_results:
            if item.url not in unique_sources:
                # 安全边界：黑名单检查
                if self._is_blacklisted_source(item.url):
                    print(f"   [Filter] 过滤黑名单来源: {item.title[:30]}...")
                    continue
                unique_sources[item.url] = item
        
        valid_sources = list(unique_sources.values())
        
        # 按可信度排序
        valid_sources.sort(key=lambda x: x.credibility_score, reverse=True)
        
        print(f"   -> 去重后: {len(valid_sources)} 个有效来源")
        
        # 如果来源不足，尝试使用原始查询再搜索一次
        if len(valid_sources) < min_sources:
            print(f"   -> 有效来源不足，补充搜索...")
            fallback_results = self._search_single_structured(query)
            for item in fallback_results:
                if item.url not in unique_sources and not self._is_blacklisted_source(item.url):
                    valid_sources.append(item)
                    unique_sources[item.url] = item
        
        # 4. 综合层：生成最终报告
        print(f"   -> 生成最终报告...")
        report = self._synthesize_report(query, valid_sources, sub_queries)
        
        print(f"   -> 完成！共 {len(report.sources)} 个有效来源")
        return report
    
    def _synthesize_report(
        self,
        query: str,
        sources: List[SourceItem],
        sub_queries: List[str],
    ) -> ResearchResult:
        """
        [综合层] 将多个来源综合为最终回答
        
        Args:
            query: 原始查询
            sources: 有效来源列表
            sub_queries: 使用的子查询列表
            
        Returns:
            ResearchResult: 结构化研究报告
        """
        if not sources:
            return ResearchResult(
                summary=f"未找到与 '{query}' 相关的权威医学资料。建议尝试调整查询关键词或查询更广泛的相关主题。",
                sources=[],
                missing_info=[f"'{query}' 的相关信息"],
                sub_queries_used=sub_queries,
            )
        
        # 将来源转换为文本 Context
        context_parts = []
        for i, s in enumerate(sources, 1):
            context_parts.append(
                f"[来源 {i}] {s.title}\n"
                f"   链接: {s.url}\n"
                f"   类型: {s.source_type} | 年份: {s.year} | 可信度: {s.credibility_score}/10\n"
                f"   摘要: {s.snippet}"
            )
        context_str = "\n\n".join(context_parts)
        
        synthesis_prompt = f"""基于以下检索到的权威医学资料，回答临床问题。

【原始问题】
{query}

【要求】
1. **循证写作**：每一句话都要在句末标注引用来源索引，如 [1], [2]
2. **数据引用**：尽量引用具体的临床数据、统计数据
3. **解决冲突**：如果不同来源有冲突（如不同指南推荐不同），请明确指出来源差异
4. **临床建议**：总结对临床决策的启示
5. **避免重复**：不要简单罗列信息，要综合分析

【输出格式】
直接输出综合报告，不要有额外的格式前缀。

【检索资料库】
{context_str}
"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="你是一个专业的医学证据综合助手，擅长将多个权威来源综合为循证回答。"),
                HumanMessage(content=synthesis_prompt)
            ])
            final_summary = response.content.strip()
        except Exception as e:
            print(f"   [Synthesis Error] {e}")
            # 回退：简单拼接
            final_summary = f"关于 '{query}'，找到以下资料：\n\n"
            for i, s in enumerate(sources, 1):
                final_summary += f"[{i}] {s.snippet}\n\n"
        
        # 分析缺失信息
        missing_info = self._analyze_missing_info(query, sources)
        
        return ResearchResult(
            summary=final_summary,
            sources=sources,
            missing_info=missing_info,
            sub_queries_used=sub_queries,
        )
    
    def _analyze_missing_info(
        self,
        query: str,
        sources: List[SourceItem],
    ) -> List[str]:
        """
        分析仍未找到的信息
        
        Args:
            query: 原始查询
            sources: 已找到的来源
            
        Returns:
            缺失信息列表
        """
        # 检查是否缺少特定类型的来源
        source_types = set(s.source_type for s in sources)
        missing = []
        
        if "Guideline" not in source_types:
            missing.append("权威临床指南（如 NCCN/CSCO/ESMO）")
        if "Paper" not in source_types:
            missing.append("核心期刊研究文献")
        
        # 检查年份覆盖
        years = [s.year for s in sources if s.year != "未知"]
        if years:
            max_year = max(int(y) for y in years if y.isdigit())
            if max_year < 2023:
                missing.append(f"近 {2025 - max_year} 年的最新资料")
        
        return missing
    
    def search_guideline_deep(self, disease: str) -> ResearchResult:
        """
        深度搜索临床指南
        
        Args:
            disease: 疾病名称
            
        Returns:
            ResearchResult: 结构化研究报告
        """
        return self.search_deep(f"{disease} 临床指南 NCCN CSCO ESMO 最新推荐")
    
    def search_treatment_deep(
        self,
        disease: str,
        treatment: Optional[str] = None,
    ) -> ResearchResult:
        """
        深度搜索治疗方案
        
        Args:
            disease: 疾病名称
            treatment: 治疗方案（可选）
            
        Returns:
            ResearchResult: 结构化研究报告
        """
        if treatment:
            query = f"{disease} {treatment} 治疗方案 临床指南 最新研究"
        else:
            query = f"{disease} 治疗方案 一线 二线 推荐"
        return self.search_deep(query)
    
    def search_drug_deep(
        self,
        drug_name: str,
        info_type: str = "all",
    ) -> ResearchResult:
        """
        深度搜索药物信息
        
        Args:
            drug_name: 药物名称
            info_type: 信息类型
            
        Returns:
            ResearchResult: 结构化研究报告
        """
        type_map = {
            "all": "药物信息 用法用量 不良反应 相互作用",
            "dosage": "药物用法用量 剂量 疗程",
            "interaction": "药物相互作用 配伍禁忌",
            "adverse": "药物不良反应 副作用 安全性",
            "indication": "药物适应症 适应人群",
        }
        context = type_map.get(info_type, type_map["all"])
        return self.search_deep(f"{drug_name} {context} FDA NMPA 官方说明书")


# ================= 3. 便捷函数 =================

def create_web_search_service(use_singleton: bool = True) -> WebSearchService:
    """
    创建联网搜索服务实例
    
    【优化】默认使用全局单例，避免重复初始化
    
    Args:
        use_singleton: 是否使用全局单例（默认 True）
    
    Returns:
        WebSearchService 实例
    """
    if use_singleton:
        return _global_ws_manager.get_web_search_service()
    
    # 非单例模式：创建新实例
    try:
        from ..config import load_settings
        settings = load_settings()
        
        return WebSearchService(
            api_key=settings.llm.api_key,
            api_base=settings.llm.api_base,
            model=settings.web_search.model,
            temperature=settings.web_search.temperature,
            max_tokens=settings.web_search.max_tokens,
        )
    except Exception as e:
        print(f"[WebSearchService] 配置加载失败，使用默认值: {e}")
        return WebSearchService()


def create_deep_research_service(use_singleton: bool = True) -> DeepResearchService:
    """
    创建深度研究服务实例
    
    【优化】默认使用全局单例，避免重复初始化
    
    Args:
        use_singleton: 是否使用全局单例（默认 True）
    
    Returns:
        DeepResearchService 实例
    """
    if use_singleton:
        return _global_ws_manager.get_deep_research_service()
    
    # 非单例模式：创建新实例
    try:
        from ..config import load_settings
        settings = load_settings()
        
        return DeepResearchService(
            api_key=settings.llm.api_key,
            api_base=settings.llm.api_base,
            model=settings.web_search.model,
            temperature=settings.web_search.temperature,
            max_tokens=settings.web_search.max_tokens,
        )
    except Exception as e:
        print(f"[DeepResearchService] 配置加载失败，使用默认值: {e}")
        return DeepResearchService()


def deep_research(
    query: str,
    max_workers: int = 3,
    min_sources: int = 3,
) -> ResearchResult:
    """
    快速深度研究
    
    Args:
        query: 研究查询
        max_workers: 最大并行线程数
        min_sources: 最少需要的有效来源数
        
    Returns:
        ResearchResult: 结构化研究报告
    """
    service = create_deep_research_service()
    return service.search_deep(
        query=query,
        max_workers=max_workers,
        min_sources=min_sources,
    )


def search_guideline_deep(disease: str) -> ResearchResult:
    """
    快速深度搜索临床指南
    
    Args:
        disease: 疾病名称
        
    Returns:
        ResearchResult: 结构化研究报告
    """
    service = create_deep_research_service()
    return service.search_guideline_deep(disease)


def search_treatment_deep(
    disease: str,
    treatment: Optional[str] = None,
) -> ResearchResult:
    """
    快速深度搜索治疗方案
    
    Args:
        disease: 疾病名称
        treatment: 治疗方案
        
    Returns:
        ResearchResult: 结构化研究报告
    """
    service = create_deep_research_service()
    return service.search_treatment_deep(disease, treatment)


# 便捷函数
def web_search(query: str, context: Optional[str] = None) -> str:
    """
    快速联网搜索
    
    Args:
        query: 搜索查询
        context: 上下文（可选）
    
    Returns:
        搜索结果文本
    """
    service = create_web_search_service()
    result = service.search(query, context)
    return result["result"]


def search_clinical_info(
    topic: str,
    disease: Optional[str] = None,
    treatment: Optional[str] = None,
) -> str:
    """
    搜索临床信息
    
    Args:
        topic: 主题
        disease: 疾病
        treatment: 治疗
    
    Returns:
        搜索结果文本
    """
    service = create_web_search_service()
    result = service.search_clinical_evidence(topic, disease, treatment)
    return result["result"]


def search_drug(drug_name: str, info_type: str = "all") -> str:
    """
    搜索药物信息
    
    Args:
        drug_name: 药物名称
        info_type: 信息类型
    
    Returns:
        搜索结果文本
    """
    service = create_web_search_service()
    result = service.search_drug_info(drug_name, info_type)
    return result["result"]
