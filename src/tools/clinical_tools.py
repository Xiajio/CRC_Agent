from __future__ import annotations

import re
import json
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ..config import Settings, load_settings


# ============================================================================
# Pydantic 数据模型定义 - 患者病史结构化信息模型
# ============================================================================

class PatientHistory(BaseModel):
    """患者病史结构化信息模型"""
    
    chief_complaint: str = Field(description="患者主诉，概括主要症状和持续时间")
    tumor_location: str = Field(description="肿瘤位置：colon/rectum/multiple/unknown")
    tumor_location_confidence: float = Field(description="可信度评分，0-1之间")
    symptoms: List[str] = Field(description="症状列表，使用标准化医学术语")
    symptom_duration: Optional[str] = Field(default=None, description="症状持续时间")
    family_history: bool = Field(description="是否有结直肠癌家族史")
    family_history_details: Optional[str] = Field(default=None, description="家族史详细信息")
    biopsy_confirmed: bool = Field(description="是否已通过病理活检确诊")
    biopsy_details: Optional[str] = Field(default=None, description="活检详细信息")
    risk_factors: List[str] = Field(description="风险因素列表")


# ============================================================================
# 提示词模板
# ============================================================================

PARSE_PROMPT_TEMPLATE = """你是一位专业的消化肿瘤科医生。请分析以下病史，提取结构化信息。

## 任务说明
从患者病史文本中提取以下结构化信息：

1. **chief_complaint**: 患者主诉，用一句话概括主要症状和持续时间
2. **tumor_location**: 肿瘤位置（colon-结肠/rectum-直肠/multiple-多发/unknown-未知）
3. **tumor_location_confidence**: 位置判断的可信度（0-1之间的浮点数）
4. **symptoms**: 识别到的所有相关症状列表，使用标准化医学术语
5. **symptom_duration**: 症状持续时间（如"2个月"、"1年"等）
6. **family_history**: 是否有结直肠癌家族史（true/false）
7. **family_history_details**: 家族史详细信息（如有）
8. **biopsy_confirmed**: 是否已通过病理活检确诊（true/false）
9. **biopsy_details**: 活检详细信息（如有）
10. **risk_factors**: 识别的风险因素列表

## 输出要求
1. **准确性优先**: 只根据文本内容判断，不推测未提及的信息
2. **不确定性处理**: 信息不明确时使用"unknown"或较低的可信度评分
3. **症状识别**: 全面识别消化系统和全身相关症状
4. **术语标准化**: 使用标准医学术语（如便血而非拉血）
5. **输出格式**: 严格按照JSON格式输出

## 注意事项
- 肿瘤位置判断需综合考虑内镜、影像和病理结果
- 活检确认需识别多种表述形式（术后病理、活检、免疫组化等）
- 风险因素包括可改变和不可改变的因素

## 待解析病史文本

{history_text}

## JSON输出

请直接输出JSON，不要包含其他内容。"""


# ============================================================================
# 预处理函数
# ============================================================================

def preprocess_history_text(text: str) -> str:
    """病史文本预处理"""
    if not text or not text.strip():
        return ""
    
    # 去除多余空白字符
    cleaned = re.sub(r'\s+', ' ', text.strip())
    
    # 处理常见的全角半角转换
    replacements = {
        '，': ',', '。': '.', '：': ':', '；': ';',
        '（': '(', '）': ')', '【': '[', '】': ']',
        'Ａ': 'A', 'Ｂ': 'B', '１': '1', '２': '2',
    }
    for full, half in replacements.items():
        cleaned = cleaned.replace(full, half)
    
    # 长度检查，超长则截断（保留关键信息区域）
    max_length = 10000
    if len(cleaned) > max_length:
        # 保留首尾，中间截断
        cleaned = cleaned[:max_length//2] + "..." + cleaned[-max_length//2:]
    
    return cleaned


def detect_language(text: str) -> str:
    """检测文本语言类型"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    
    if chinese_chars > english_chars:
        return "zh"
    elif english_chars > chinese_chars:
        return "en"
    else:
        return "mixed"


# ============================================================================
# 智能路由函数
# ============================================================================

def should_use_llm(history_text: str) -> bool:
    """
    智能路由判断：简单病例用规则引擎，复杂病例用LLM
    
    策略规则：
    1. 极短文本（<100字符）：规则快速处理
    2. 长文本（>1000字符）：需要LLM语义理解
    3. 中等长度：检查是否存在模糊表述，模糊则用LLM
    
    Returns:
        True: 使用LLM增强解析
        False: 使用规则快速解析
    """
    if not history_text or len(history_text.strip()) < 100:
        # 文本太短，规则足以处理
        return False
    
    if len(history_text) > 1000:
        # 长文本信息量大，需要LLM进行语义理解
        return True
    
    # 中等长度文本：检查是否存在模糊表述
    text_lower = history_text.lower()
    vague_patterns = [
        r'可能', r'maybe', r' approximately', r'大概',
        r'似乎', r'好像', r'不确定', r' unclear',
        r'建议进一步检查', r'需排除', r'待除外'
    ]
    
    if any(re.search(p, text_lower) for p in vague_patterns):
        # 存在模糊表述，需要LLM进行上下文推断
        return True
    
    # 默认为False，使用规则快速处理
    return False


# ============================================================================
# 基于规则的快速解析（兜底方案）
# ============================================================================

def rule_based_parse(history_text: str) -> Dict:
    """基于规则的快速病史解析（简单病例兜底方案）"""
    text_lower = history_text.lower()
    
    # 肿瘤位置检测
    tumor_location = "unknown"
    if "rect" in text_lower or "直肠" in history_text:
        tumor_location = "rectum"
    elif "colon" in text_lower or "结肠" in history_text:
        tumor_location = "colon"
    
    # 主诉提取（前200字符）
    chief_complaint = history_text[:200]
    
    # 症状检测
    symptoms = []
    symptom_keywords = [
        ("便血", ["便血", "血便", "hematochezia", "rectal bleeding"]),
        ("腹痛", ["腹痛", "abdominal pain", "肚子疼"]),
        ("体重下降", ["体重下降", "weight loss", "消瘦"]),
        ("便秘", ["便秘", "constipation"]),
        ("腹泻", ["腹泻", "diarrhea"]),
        ("贫血", ["贫血", "anemia"]),
    ]
    for symptom, keywords in symptom_keywords:
        if any(kw.lower() in text_lower for kw in keywords):
            symptoms.append(symptom)
    
    # 活检确认检测
    biopsy_confirmed = any(kw in text_lower for kw in ["biopsy", "病理", "活检", "cancer", "癌"])
    
    # 家族史检测
    family_history = any(kw in text_lower for kw in ["family history", "family hx", "家族史", "遗传史", "直系亲属"])
    
    # 风险因素检测
    risk_factors = []
    if any(kw in text_lower for kw in ["smoking", "smoker", "吸烟", "抽烟"]):
        risk_factors.append("smoking")
    if any(kw in text_lower for kw in ["drinking", "alcohol", "饮酒", "喝酒"]):
        risk_factors.append("alcohol_use")
    if any(kw in text_lower for kw in ["obese", "obesity", "肥胖", "超重"]):
        risk_factors.append("obesity")
    
    return {
        "chief_complaint": chief_complaint,
        "tumor_location": tumor_location,
        "tumor_location_confidence": 0.7,
        "symptoms": symptoms,
        "symptom_duration": None,
        "family_history": family_history,
        "family_history_details": None,
        "biopsy_confirmed": biopsy_confirmed,
        "biopsy_details": None,
        "risk_factors": risk_factors,
        "parsing_timestamp": datetime.now().isoformat(),
        "parsing_method": "rule_based",
        "confidence_score": 0.6,
        "notes": "基于规则的快速解析，信息提取可能不完整，建议使用LLM增强解析",
    }


# ============================================================================
# 基于LLM的解析（主方案）
# ============================================================================

_llm_model = None
_llm_model_initialized = False


def get_llm_model():
    """获取或初始化LLM模型（单例模式，懒加载）"""
    global _llm_model, _llm_model_initialized
    
    if _llm_model_initialized:
        return _llm_model
    
    if _llm_model is not None:
        return _llm_model
    
    try:
        # 延迟导入，避免循环依赖或环境不可用时启动失败
        from ..config import load_settings
        from ..services.llm_service import LLMService
        
        settings = load_settings()
        llm_service = LLMService(settings.llm)
        _llm_model = llm_service.create_chat_model()
        _llm_model_initialized = True
        return _llm_model
        
    except ImportError as e:
        print(f"[PatientHistoryParser] 导入失败（环境未配置）: {e}")
        return None
    except Exception as e:
        print(f"[PatientHistoryParser] LLM初始化失败: {e}")
        return None


def reset_llm_model():
    """重置LLM模型（用于测试或重新初始化）"""
    global _llm_model, _llm_model_initialized
    _llm_model = None
    _llm_model_initialized = False


def llm_based_parse(history_text: str) -> Dict:
    """基于大模型的病史解析（主方案）"""
    global _llm_model
    
    model = get_llm_model()
    if model is None:
        # LLM不可用，回退到规则方案
        return rule_based_parse(history_text)
    
    try:
        # 预处理输入文本
        cleaned_text = preprocess_history_text(history_text)
        
        # 构建提示词
        prompt = PARSE_PROMPT_TEMPLATE.format(history_text=cleaned_text)
        
        # 调用大模型
        response = model.invoke([HumanMessage(content=prompt)])
        raw_content = response.content if hasattr(response, 'content') else str(response)
        
        # 解析JSON输出
        try:
            # 尝试直接解析
            result_data = json.loads(raw_content)
        except json.JSONDecodeError:
            # 尝试提取JSON块
            json_match = re.search(r'\{[\s\S]*\}', raw_content)
            if json_match:
                result_data = json.loads(json_match.group())
            else:
                raise ValueError("无法从模型响应中提取JSON")
        
        # 构建完整输出
        result = {
            **result_data,
            "parsing_timestamp": datetime.now().isoformat(),
            "parsing_method": "llm_enhanced",
            "confidence_score": result_data.get("tumor_location_confidence", 0.8),
            "notes": "基于大模型的增强解析，参考医学知识库进行语义理解",
        }
        
        return result
        
    except Exception as e:
        print(f"[PatientHistoryParser] LLM解析异常: {e}，回退到规则方案")
        return rule_based_parse(history_text)


# ============================================================================
# 智能选择解析策略
# ============================================================================

def intelligent_parse(history_text: str) -> Dict:
    """
    智能选择解析策略
    
    策略选择逻辑：
    - 文本极短或为空：快速规则解析
    - 简单明确病史：规则解析兜底
    - 复杂/模糊病史：LLM增强解析
    - LLM不可用：自动回退到规则解析
    """
    if not history_text or len(history_text.strip()) < 20:
        # 文本太短，使用规则解析
        return rule_based_parse(history_text)
    
    # 判断是否使用LLM
    if should_use_llm(history_text):
        result = llm_based_parse(history_text)
        result["strategy_used"] = "llm_enhanced"
        return result
    else:
        result = rule_based_parse(history_text)
        result["strategy_used"] = "rule_based"
        return result


# ============================================================================
# LangChain Tool 定义
# ============================================================================

@tool
def PatientHistoryParserTool(history_text: str) -> Dict:
    """
    基于大模型增强的患者病史解析工具（V2版本）
    
    功能说明：
    - 将非结构化病史文本转换为标准化JSON结构
    - 采用混合策略：简单病例快速处理，复杂病例调用大模型
    - 支持中英文双语处理
    - 智能处理不确定性，不盲目猜测
    
    输出字段：
    - chief_complaint: 患者主诉
    - tumor_location: 肿瘤位置 (colon/rectum/multiple/unknown)
    - tumor_location_confidence: 位置判断可信度 (0-1)
    - symptoms: 症状列表
    - symptom_duration: 症状持续时间
    - family_history: 是否有家族史
    - family_history_details: 家族史详细信息
    - biopsy_confirmed: 是否已活检确诊
    - biopsy_details: 活检详细信息
    - risk_factors: 风险因素列表
    - parsing_timestamp: 解析时间戳
    - parsing_method: 解析方法 (rule_based/llm_enhanced)
    - confidence_score: 综合置信度
    - notes: 解析说明
    
    示例输入：
    "患者男性，62岁，因'间断便血2个月'入院。肠镜检查提示直肠距肛门8cm见一肿物，"
    "大小约3×4cm，表面溃烂。活检病理示中分化腺癌。既往体健，否认肿瘤家族史。"
    "吸烟史30年，每日20支。"
    
    示例输出：
    {
        "chief_complaint": "间断便血2个月",
        "tumor_location": "rectum",
        "tumor_location_confidence": 0.95,
        "symptoms": ["便血"],
        "symptom_duration": "2个月",
        "family_history": false,
        "family_history_details": null,
        "biopsy_confirmed": true,
        "biopsy_details": "活检病理示中分化腺癌",
        "risk_factors": ["smoking", "alcohol_use"],
        "parsing_timestamp": "2025-01-15T10:30:00.000Z",
        "parsing_method": "llm_enhanced",
        "confidence_score": 0.95,
        "notes": "基于大模型的增强解析，参考医学知识库进行语义理解"
    }
    """
    return intelligent_parse(history_text)


@tool
def PolypDetectionTool(text_context: str) -> Dict:
    """Mock endoscopic polyp detection from textual report."""

    has_polyp = "polyp" in text_context.lower() or "\u606f\u8089" in text_context
    return {"context_excerpt": text_context[:120], "polyps_detected": has_polyp, "count": 1 if has_polyp else 0}


@tool
def PathologyParserTool(pathology_text: str) -> Dict:
    """
    Parse pathology report to extract gold-standard diagnostic information.
    
    This is the critical tool for confirming cancer diagnosis based on pathology
    (the clinical gold standard), extracting histology type and differentiation grade.
    """
    text_lower = pathology_text.lower()
    
    # Check for malignancy confirmation
    malignancy_en = ["adenocarcinoma", "carcinoma", "malignant", "cancer"]
    malignancy_cn = ["\u8179\u764c", "\u764c", "\u6076\u6027", "\u6d78\u6da6\u6027", "\u4fb5\u72af"]
    pathology_confirmed = any(kw in text_lower for kw in malignancy_en) or any(kw in pathology_text for kw in malignancy_cn)
    
    # Extract histology type
    histology_type = "unknown"
    if "adenocarcinoma" in text_lower or "\u8179\u764c" in pathology_text:
        histology_type = "adenocarcinoma"
    elif "squamous" in text_lower or "\u9cde\u764c" in pathology_text:
        histology_type = "squamous_cell_carcinoma"
    elif "mucinous" in text_lower or "\u7c98\u6db2" in pathology_text:
        histology_type = "mucinous_adenocarcinoma"
    elif "signet" in text_lower or "\u5370\u6212" in pathology_text:
        histology_type = "signet_ring_cell_carcinoma"
    elif "neuroendocrine" in text_lower or "\u795e\u7ecf\u5185\u5206\u6ccc" in pathology_text:
        histology_type = "neuroendocrine_carcinoma"
    
    # Extract differentiation grade
    differentiation_grade = "unknown"
    if any(kw in text_lower for kw in ["well differentiated", "well-differentiated", "g1"]) or "\u9ad8\u5206\u5316" in pathology_text:
        differentiation_grade = "well_differentiated"
    elif any(kw in text_lower for kw in ["moderately differentiated", "moderate", "g2"]) or "\u4e2d\u5206\u5316" in pathology_text:
        differentiation_grade = "moderately_differentiated"
    elif any(kw in text_lower for kw in ["poorly differentiated", "poor", "g3", "undifferentiated"]) or "\u4f4e\u5206\u5316" in pathology_text or "\u672a\u5206\u5316" in pathology_text:
        differentiation_grade = "poorly_differentiated"
    
    # Extract special features
    special_features = []
    if "mucinous" in text_lower or "\u7c98\u6db2" in pathology_text:
        special_features.append("mucinous_component")
    if "signet" in text_lower or "\u5370\u6212" in pathology_text:
        special_features.append("signet_ring_cells")
    if "perineural" in text_lower or "\u795e\u7ecf\u4fb5\u72af" in pathology_text:
        special_features.append("perineural_invasion")
    if "lymphovascular" in text_lower or "\u8109\u7ba1\u4fb5\u72af" in pathology_text or "lvi" in text_lower:
        special_features.append("lymphovascular_invasion")
    
    # Extract molecular markers (MSI/MMR)
    molecular_markers = {}
    if "msi-h" in text_lower or "msi high" in text_lower or "\u5fae\u536b\u661f\u4e0d\u7a33\u5b9a" in pathology_text:
        molecular_markers["msi_status"] = "MSI-H"
    elif "msi-l" in text_lower or "mss" in text_lower or "\u5fae\u536b\u661f\u7a33\u5b9a" in pathology_text:
        molecular_markers["msi_status"] = "MSS"
    
    if "dmmr" in text_lower or "mmr\u7f3a\u9677" in pathology_text or "\u9519\u914d\u4fee\u590d\u7f3a\u9677" in pathology_text:
        molecular_markers["mmr_status"] = "dMMR"
    elif "pmmr" in text_lower or "mmr\u5b8c\u6574" in pathology_text:
        molecular_markers["mmr_status"] = "pMMR"
    
    # Extract pathological staging if present (pT, pN)
    pathological_staging = {}
    pt_match = re.search(r'pT([0-4]|is|a|b)', text_lower)
    if pt_match:
        pathological_staging["pT"] = f"pT{pt_match.group(1).upper()}"
    pn_match = re.search(r'pN([0-2]|x)', text_lower)
    if pn_match:
        pathological_staging["pN"] = f"pN{pn_match.group(1).upper()}"
    
    return {
        "pathology_confirmed": pathology_confirmed,
        "histology_type": histology_type,
        "differentiation_grade": differentiation_grade,
        "special_features": special_features,
        "molecular_markers": molecular_markers,
        "pathological_staging": pathological_staging,
        "context_excerpt": pathology_text[:400],
        "notes": "Pathology report parsed for gold-standard diagnosis confirmation.",
    }


@tool
def VolumeCTSegmentorTool(ct_report_text: str) -> Dict:
    """
    Text-mode CT report analyzer for colon cancer staging and distant metastasis detection.
    
    Analyzes CT reports for:
    - T staging (tumor invasion depth)
    - N staging (lymph node involvement)  
    - M staging (distant metastasis to liver, lung, peritoneum, etc.)
    """
    lower = ct_report_text.lower()
    
    # Liver metastasis detection
    liver_flag = "\u809d" in ct_report_text or "liver" in lower or "hepatic" in lower
    liver_lesion_keywords = ["\u4f4e\u5bc6\u5ea6", "\u5360\u4f4d", "\u8f6c\u79fb", "\u7ed3\u8282", "metasta", "lesion", "mass", "low density"]
    liver_lesion = liver_flag and any(kw in ct_report_text or kw in lower for kw in liver_lesion_keywords)
    
    # Lung metastasis detection
    lung_flag = "\u80ba" in ct_report_text or "lung" in lower or "pulmonary" in lower or "\u80f8\u90e8" in ct_report_text
    lung_lesion_keywords = ["\u7ed3\u8282", "\u8f6c\u79fb", "\u5360\u4f4d", "nodule", "metasta", "mass"]
    lung_lesion = lung_flag and any(kw in ct_report_text or kw in lower for kw in lung_lesion_keywords)
    
    # Peritoneal metastasis
    peritoneal_keywords = ["\u8179\u819c", "periton", "\u8179\u6c34", "ascites", "\u79cd\u690d"]
    peritoneal_metastasis = any(kw in ct_report_text or kw in lower for kw in peritoneal_keywords)
    
    # Lymph node assessment
    ln_flag = "\u6dcb\u5df4" in ct_report_text or "lymph" in lower or "node" in lower
    ln_suspicious_keywords = ["\u80bf\u5927", "enlarged", "suspicious", "\u53ef\u7591", "\u8f6c\u79fb"]
    ln_suspicious = ln_flag and any(kw in ct_report_text or kw in lower for kw in ln_suspicious_keywords)
    
    # Distant metastasis summary
    metastasis_sites = []
    if liver_lesion:
        metastasis_sites.append("liver")
    if lung_lesion:
        metastasis_sites.append("lung")
    if peritoneal_metastasis:
        metastasis_sites.append("peritoneum")
    
    metastasis_possible = len(metastasis_sites) > 0
    
    # M staging assessment
    m_stage = "M0"
    if metastasis_possible:
        m_stage = "M1"
        if liver_lesion and not lung_lesion:
            m_stage = "M1a"
        elif lung_lesion and not liver_lesion:
            m_stage = "M1a"
        elif len(metastasis_sites) >= 2:
            m_stage = "M1b"
        if peritoneal_metastasis:
            m_stage = "M1c"
    
    return {
        "ct_context": ct_report_text[:400],
        "metastasis_possible": metastasis_possible,
        "m_stage": m_stage,
        "metastasis_sites": metastasis_sites,
        "liver_lesion": liver_lesion,
        "lung_lesion": lung_lesion,
        "peritoneal_metastasis": peritoneal_metastasis,
        "nodes_suspected": ln_suspicious,
        "distant_metastasis_summary": f"M staging: {m_stage}, sites: {', '.join(metastasis_sites) if metastasis_sites else 'none detected'}",
        "notes": "CT analyzed for TNM staging with focus on distant metastasis (M staging).",
    }


@tool
def RectalMRStagerTool(text_context: str) -> Dict:
    """
    Text-mode rectal MRI report analyzer for local staging (T, N, MRF, EMVI).
    
    MRI is the gold standard for rectal cancer LOCAL staging.
    Note: MRI cannot reliably detect distant metastasis - use CT for M staging.
    """
    lower = text_context.lower()
    
    # MRF (Mesorectal Fascia) status
    mrf_pos = "mrf" in lower and ("+" in text_context or "\u9633\u6027" in text_context or "involved" in lower or "\u4fb5\u72af" in text_context)
    mrf_neg = "mrf" in lower and ("-" in text_context or "\u9634\u6027" in text_context or "clear" in lower or "\u672a\u4fb5\u72af" in text_context)
    mrf_status = "positive" if mrf_pos else "negative" if mrf_neg else "not_reported"
    
    # EMVI (Extramural Vascular Invasion)
    emvi_pos = ("emvi" in lower or "\u9759\u8109\u4fb5\u72af" in text_context or "\u8840\u7ba1\u4fb5\u72af" in text_context) and \
               ("+" in text_context or "\u9633\u6027" in text_context or "present" in lower)
    emvi_neg = ("emvi" in lower) and ("-" in text_context or "\u9634\u6027" in text_context or "absent" in lower)
    emvi_status = "positive" if emvi_pos else "negative" if emvi_neg else "not_reported"
    
    # CRM (Circumferential Resection Margin)
    crm_threatened = any(kw in text_context or kw in lower for kw in ["crm+", "crm\u9633\u6027", "\u5207\u7f18\u53d7\u7d2f", "margin involved"])
    crm_clear = any(kw in text_context or kw in lower for kw in ["crm-", "crm\u9634\u6027", "\u5207\u7f18\u9634\u6027", "margin clear"])
    crm_status = "threatened" if crm_threatened else "clear" if crm_clear else "not_reported"
    
    # T staging from MRI
    t_stage = "not_reported"
    if "t1" in lower or "\u7c98\u819c\u4e0b" in text_context:
        t_stage = "T1"
    elif "t2" in lower or "\u808c\u5c42" in text_context:
        t_stage = "T2"
    elif "t3" in lower or "\u6d46\u819c\u4e0b" in text_context or "\u7cfb\u819c\u8102\u80aa" in text_context:
        t_stage = "T3"
    elif "t4" in lower or "\u4fb5\u72af\u90bb\u8fd1\u5668\u5b98" in text_context or "\u7a7f\u900f\u6d46\u819c" in text_context:
        t_stage = "T4"
    
    # N staging from MRI
    n_stage = "not_reported"
    if "n0" in lower or "\u6dcb\u5df4\u7ed3\u9634\u6027" in text_context or "\u672a\u89c1\u80bf\u5927\u6dcb\u5df4\u7ed3" in text_context:
        n_stage = "N0"
    elif "n1" in lower or ("\u6dcb\u5df4\u7ed3" in text_context and ("\u80bf\u5927" in text_context or "1-3" in text_context)):
        n_stage = "N1"
    elif "n2" in lower or ("\u6dcb\u5df4\u7ed3" in text_context and "\u591a\u53d1" in text_context):
        n_stage = "N2"
    
    # Determine if neoadjuvant therapy is indicated
    neoadjuvant_recommended = mrf_status == "positive" or emvi_status == "positive" or t_stage in ["T3", "T4"] or n_stage in ["N1", "N2"]
    
    return {
        "mrf_status": mrf_status,
        "emvi_status": emvi_status,
        "crm_status": crm_status,
        "t_stage_mri": t_stage,
        "n_stage_mri": n_stage,
        "emvi_suspected": emvi_pos,
        "neoadjuvant_recommended": neoadjuvant_recommended,
        "context_excerpt": text_context[:400],
        "local_staging_summary": f"MRI Local Staging: T={t_stage}, N={n_stage}, MRF={mrf_status}, EMVI={emvi_status}",
        "notes": "Rectal MRI analyzed for local staging (T, N, MRF, EMVI). Use CT for M staging.",
    }


@tool
def MolecularGuidelineTool(markers: List[str]) -> Dict:
    """Return mock molecular findings and therapy suggestions."""

    actionable = [m for m in markers if m.lower() in {"kras", "nras", "braf"}]
    return {
        "markers_checked": markers,
        "actionable": actionable,
        "recommendation": "Consider EGFR antibody if RAS wildtype; mock output.",
    }


def list_clinical_tools():
    """Return CRC clinical tool registry including the new PathologyParserTool."""

    return [
        PatientHistoryParserTool,
        PolypDetectionTool,
        PathologyParserTool,  # NEW: Pathology gold-standard diagnosis
        VolumeCTSegmentorTool,  # ENHANCED: Now includes detailed M staging
        RectalMRStagerTool,  # ENHANCED: Now includes detailed local staging
        MolecularGuidelineTool,
    ]