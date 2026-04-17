"""
联网搜索工具 (Web Search Tools)

使用 gpt-4o-search-preview 模型进行实时网络资料搜集。

特点：
1. 详细的查询输入，确保搜索准确
2. 严格的结果验证，避免幻觉
3. 明确标注信息来源
4. 无相关资料时明确返回"没有找到相关资料"
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.services.web_search_service import (
    WebSearchService,
    create_web_search_service,
)


# === Input Schemas ===

class WebSearchInput(BaseModel):
    """通用联网搜索输入"""
    query: str = Field(
        description="搜索查询内容。请提供详细、具体的查询描述，包括：疾病名称、治疗方案、药物名称、具体问题等。",
        min_length=5,
    )
    context: Optional[str] = Field(
        default=None,
        description="查询的上下文信息，如患者情况、具体场景、需要关注的重点等。提供上下文可以获得更精准的搜索结果。",
    )


class ClinicalSearchInput(BaseModel):
    """临床证据搜索输入"""
    topic: str = Field(
        description="搜索的临床主题，如：III期结肠癌辅助化疗、MSI-H免疫治疗、KRAS突变靶向治疗等",
        min_length=3,
    )
    disease: Optional[str] = Field(
        default=None,
        description="疾病名称，如：结直肠癌、肝转移、直肠癌等",
    )
    treatment: Optional[str] = Field(
        default=None,
        description="治疗方案，如：FOLFOX化疗、贝伐珠单抗、PD-1抑制剂等",
    )


class DrugSearchInput(BaseModel):
    """药物信息搜索输入"""
    drug_name: str = Field(
        description="药物名称（通用名或商品名），如：奥沙利铂、贝伐珠单抗、pembrolizumab、Keytruda等",
        min_length=2,
    )
    info_type: str = Field(
        default="all",
        description="需要查询的信息类型：all(完整信息)、dosage(用法用量)、interaction(药物相互作用)、adverse(不良反应)、indication(适应症)",
    )


class GuidelineSearchInput(BaseModel):
    """指南更新搜索输入"""
    disease: str = Field(
        description="疾病名称，如：结直肠癌、肺癌、乳腺癌等",
        min_length=2,
    )
    guideline_source: Optional[str] = Field(
        default=None,
        description="指南来源：NCCN(美国)、CSCO(中国)、ESMO(欧洲)、all(所有)",
    )


class ResearchSearchInput(BaseModel):
    """最新研究搜索输入"""
    topic: str = Field(
        description="研究主题，如：结直肠癌免疫治疗、MSI检测方法、ctDNA监测等",
        min_length=3,
    )
    study_type: Optional[str] = Field(
        default=None,
        description="研究类型：RCT(随机对照试验)、meta-analysis(荟萃分析)、cohort(队列研究)、all(所有)",
    )


# === Tools ===

class WebSearchTool(BaseTool):
    """通用联网搜索工具"""
    name: str = "web_search"
    description: str = (
        "【联网搜索】使用 gpt-4o-search-preview 模型进行实时网络搜索。\n"
        "用于获取最新的医学资料、临床研究、指南更新等。\n"
        "参数：query（必填，详细的搜索描述），context（可选，上下文信息）。\n"
        "注意：如果搜索不到相关资料，会明确返回'没有找到相关资料'，绝不编造信息。"
    )
    args_schema: type[BaseModel] = WebSearchInput
    
    _service: Optional[WebSearchService] = None
    
    def _get_service(self) -> WebSearchService:
        if self._service is None:
            self._service = create_web_search_service()
        return self._service
    
    def _run(self, query: str, context: Optional[str] = None) -> str:
        q = (query or "").strip()
        if not q or len(q) < 5:
            return "请提供更详细的搜索查询（至少5个字符）"
        
        service = self._get_service()
        result = service.search(query=q, context=context)
        return result["result"]
    
    async def _arun(self, query: str, context: Optional[str] = None) -> str:
        return self._run(query=query, context=context)


class ClinicalEvidenceSearchTool(BaseTool):
    """临床证据搜索工具"""
    name: str = "search_clinical_evidence"
    description: str = (
        "【联网搜索临床证据】搜索最新的临床证据、治疗方案、指南建议。\n"
        "优先搜索：NCCN/ESMO/CSCO指南、RCT研究、Meta分析、PubMed文献。\n"
        "参数：topic（必填，临床主题），disease（可选，疾病名称），treatment（可选，治疗方案）。\n"
        "适用于：需要最新临床证据支持决策时使用。"
    )
    args_schema: type[BaseModel] = ClinicalSearchInput
    
    _service: Optional[WebSearchService] = None
    
    def _get_service(self) -> WebSearchService:
        if self._service is None:
            self._service = create_web_search_service()
        return self._service
    
    def _run(
        self,
        topic: str,
        disease: Optional[str] = None,
        treatment: Optional[str] = None,
    ) -> str:
        t = (topic or "").strip()
        if not t:
            return "请提供搜索的临床主题"
        
        service = self._get_service()
        result = service.search_clinical_evidence(
            topic=t,
            disease=disease,
            treatment=treatment,
        )
        return result["result"]
    
    async def _arun(
        self,
        topic: str,
        disease: Optional[str] = None,
        treatment: Optional[str] = None,
    ) -> str:
        return self._run(topic=topic, disease=disease, treatment=treatment)


class DrugInfoSearchTool(BaseTool):
    """药物信息联网搜索工具"""
    name: str = "search_drug_online"
    description: str = (
        "【联网搜索处方药物信息】搜索处方药物的详细信息，包括用法用量、不良反应、药物相互作用等。\n"
        "只适用于处方药物，如：化疗药物、靶向药物、免疫药物等。\n"
        "不适用于营养补充剂、维生素、非处方药。\n"
        "优先搜索：官方药品说明书、FDA/NMPA信息、权威药物数据库。\n"
        "参数：drug_name（必填，处方药物名称），info_type（可选：all/dosage/interaction/adverse/indication）。\n"
        "适用于：医生需要查询处方药物的临床用药信息时使用。"
    )
    args_schema: type[BaseModel] = DrugSearchInput
    
    _service: Optional[WebSearchService] = None
    
    def _get_service(self) -> WebSearchService:
        if self._service is None:
            self._service = create_web_search_service()
        return self._service
    
    def _run(self, drug_name: str, info_type: str = "all") -> str:
        d = (drug_name or "").strip()
        if not d:
            return "请提供药物名称"
        
        service = self._get_service()
        result = service.search_drug_info(drug_name=d, info_type=info_type)
        return result["result"]
    
    async def _arun(self, drug_name: str, info_type: str = "all") -> str:
        return self._run(drug_name=drug_name, info_type=info_type)


class GuidelineUpdateSearchTool(BaseTool):
    """指南更新搜索工具"""
    name: str = "search_guideline_updates"
    description: str = (
        "【联网搜索指南更新】搜索最新的临床指南更新和变化。\n"
        "搜索范围：NCCN、CSCO、ESMO等权威机构的指南更新。\n"
        "参数：disease（必填，疾病名称），guideline_source（可选：NCCN/CSCO/ESMO/all）。\n"
        "适用于：需要了解最新指南变化、版本更新时使用。"
    )
    args_schema: type[BaseModel] = GuidelineSearchInput
    
    _service: Optional[WebSearchService] = None
    
    def _get_service(self) -> WebSearchService:
        if self._service is None:
            self._service = create_web_search_service()
        return self._service
    
    def _run(self, disease: str, guideline_source: Optional[str] = None) -> str:
        d = (disease or "").strip()
        if not d:
            return "请提供疾病名称"
        
        service = self._get_service()
        result = service.search_latest_guidelines(
            disease=d,
            guideline_source=guideline_source,
        )
        return result["result"]
    
    async def _arun(self, disease: str, guideline_source: Optional[str] = None) -> str:
        return self._run(disease=disease, guideline_source=guideline_source)


class LatestResearchSearchTool(BaseTool):
    """最新研究搜索工具"""
    name: str = "search_latest_research"
    description: str = (
        "【联网搜索最新研究】搜索最新的临床研究和文献。\n"
        "搜索范围：PubMed、Cochrane、高影响因子期刊。\n"
        "参数：topic（必填，研究主题），study_type（可选：RCT/meta-analysis/cohort/all）。\n"
        "适用于：需要最新研究证据、了解前沿进展时使用。"
    )
    args_schema: type[BaseModel] = ResearchSearchInput
    
    _service: Optional[WebSearchService] = None
    
    def _get_service(self) -> WebSearchService:
        if self._service is None:
            self._service = create_web_search_service()
        return self._service
    
    def _run(self, topic: str, study_type: Optional[str] = None) -> str:
        t = (topic or "").strip()
        if not t:
            return "请提供研究主题"
        
        service = self._get_service()
        result = service.search_research(topic=t, study_type=study_type)
        return result["result"]
    
    async def _arun(self, topic: str, study_type: Optional[str] = None) -> str:
        return self._run(topic=topic, study_type=study_type)


# === Tool Factory ===

def get_web_search_tool() -> BaseTool:
    """获取通用联网搜索工具"""
    return WebSearchTool()


def get_all_web_search_tools() -> list[BaseTool]:
    """获取所有联网搜索工具"""
    return [
        WebSearchTool(),
        ClinicalEvidenceSearchTool(),
        DrugInfoSearchTool(),
        GuidelineUpdateSearchTool(),
        LatestResearchSearchTool(),
    ]


def get_clinical_web_search_tools() -> list[BaseTool]:
    """
    获取临床相关的联网搜索工具集
    
    包含：
    - web_search: 通用搜索
    - search_clinical_evidence: 临床证据搜索
    - search_drug_online: 药物信息搜索
    - search_guideline_updates: 指南更新搜索
    """
    return [
        WebSearchTool(),
        ClinicalEvidenceSearchTool(),
        DrugInfoSearchTool(),
        GuidelineUpdateSearchTool(),
    ]

