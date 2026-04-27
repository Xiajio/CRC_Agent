""" Assessment and Diagnosis Nodes
(Final Fix v3: Fast Pass 策略 - 从"跳过节点"改为"轻量化通过"，确保数据流水线完整)
(v4: 引入 PatientProfile (只读沙箱) 概念 + 主动追问机制)
(v5: JSON 解析增强 + 字段验证)
(v6: 修复 PatientProfile 重复定义问题，使用 state.py 中的定义)
"""

import json
import re
import os
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..state import CRCAgentState, PatientProfile
from ..prompts import (
    CASE_INTEGRITY_SYSTEM_PROMPT,
    ASSESSMENT_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
)
from ..services.patient_card_projector import project_patient_self_report_card
from .node_utils import (
    _select_tools,
    _execute_tool_calls,
    _unwrap_nested_json,
    _invoke_structured_with_recovery,
    _is_postop_context,
    _latest_user_text,
    auto_update_roadmap_from_state,
    _truncate_message_history,
    _build_pinned_context,
    _build_summary_memory,
    _build_profile_change_entry,
)


def _extract_tnm_tokens_from_text(text: str) -> Dict[str, str]:
    raw = str(text or "")
    compact = re.sub(r"\s+", "", raw)

    combined = re.search(
        r"([cpr]?T[0-4Xx](?:is|[A-Ca-c])?)([cpr]?N[0-3Xx](?:[A-Ca-c])?)([cpr]?M[01Xx](?:[A-Ca-c])?)",
        compact,
        re.IGNORECASE,
    )
    if combined:
        return {
            "cT": combined.group(1),
            "cN": combined.group(2),
            "cM": combined.group(3),
        }

    extracted: Dict[str, str] = {}
    patterns = {
        "cT": r"([cpr]?T\s*[0-4Xx](?:is|[A-Ca-c])?)",
        "cN": r"([cpr]?N\s*[0-3Xx](?:[A-Ca-c])?)",
        "cM": r"([cpr]?M\s*[01Xx](?:[A-Ca-c])?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            extracted[key] = re.sub(r"\s+", "", match.group(1))
    return extracted


def _infer_location_category(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return "Unknown"

    colon_markers = [
        "乙状结肠", "乙狀結腸", "结肠", "結腸", "盲肠", "升结肠", "横结肠",
        "降结肠", "sigmoid", "colon", "colonic",
    ]
    rectal_markers = ["直肠", "直腸", "rectum", "rectal"]

    if any(marker in lowered for marker in colon_markers):
        return "Colon"

    distance_match = re.search(r"距肛(?:门|門|缘|緣)?\s*(\d+(?:\.\d+)?)\s*cm", lowered)
    if distance_match:
        try:
            distance_cm = float(distance_match.group(1))
            return "Rectum" if distance_cm <= 15 else "Colon"
        except ValueError:
            pass

    if any(marker in lowered for marker in rectal_markers):
        return "Rectum"

    return "Unknown"


def _quick_case_integrity_from_text(full_text: str) -> "CaseIntegrity":
    text = str(full_text or "")
    lowered = text.lower()

    has_confirmed_diagnosis = any(
        marker in lowered
        for marker in [
            "adenocarcinoma",
            "pathology",
            "biopsy",
            "活检",
            "病理",
            "腺癌",
        ]
    )

    tnm_tokens = _extract_tnm_tokens_from_text(text)
    has_t = bool(tnm_tokens.get("cT"))
    has_n = bool(tnm_tokens.get("cN"))
    has_m = bool(tnm_tokens.get("cM")) or any(
        marker in lowered
        for marker in [
            "无远处转移",
            "未见远处转移",
            "未见明确转移",
            "no distant metastasis",
            "without distant metastasis",
        ]
    )
    if has_t and has_n and has_m:
        tnm_status = "Complete"
    elif has_t or has_n or has_m:
        tnm_status = "Partial"
    else:
        tnm_status = "Missing"

    tumor_location_category = _infer_location_category(text)
    mmr_status_availability = "Provided" if any(
        marker in lowered for marker in ["pmmr", "dmmr", "mss", "msi", "mmr", "错配修复"]
    ) else "Not_Provided"

    n_token = str(tnm_tokens.get("cN", ""))
    m_token = str(tnm_tokens.get("cM", ""))
    is_advanced_stage = bool(
        re.search(r"n[1-3]", n_token, re.IGNORECASE)
        or re.search(r"m1", m_token, re.IGNORECASE)
        or re.search(r"\biii期\b|\biv期\b", text, re.IGNORECASE)
    )
    symptom_hits = any(
        marker in lowered
        for marker in ["便血", "腹痛", "体重下降", "排便", "symptom", "constipation", "diarrhea"]
    )
    is_symptom_only = symptom_hits and not has_confirmed_diagnosis

    return CaseIntegrity(
        has_confirmed_diagnosis=has_confirmed_diagnosis,
        tumor_location_category=tumor_location_category,
        tnm_status=tnm_status,
        is_advanced_stage=is_advanced_stage,
        mmr_status_availability=mmr_status_availability,
        is_symptom_only=is_symptom_only,
        reasoning="Fast heuristic integrity path.",
    )


def _quick_extract_molecular_markers(text: str) -> Dict[str, Any]:
    lowered = str(text or "").lower()
    markers: Dict[str, Any] = {
        "RAS": "Unknown",
        "KRAS": "Unknown",
        "NRAS": "Unknown",
        "BRAF": "Unknown",
        "MSI-H": False,
        "MSS": False,
        "dMMR": False,
        "pMMR": False,
        "MMR": "Unknown",
        "MSI": "Unknown",
    }
    if "pmmr" in lowered or "mss" in lowered:
        markers["pMMR"] = True
        markers["MSS"] = True
        markers["MMR"] = "pMMR"
        markers["MSI"] = "MSS"
    if "dmmr" in lowered or "msi-h" in lowered:
        markers["dMMR"] = True
        markers["MSI-H"] = True
        markers["MMR"] = "dMMR"
        markers["MSI"] = "MSI-H"
    return markers


def _normalize_clinical_tnm_value(key: str, value: str) -> str:
    token = re.sub(r"\s+", "", str(value or ""))
    if not token:
        return ""
    expected_prefix = key[0].lower()
    lowered = token.lower()
    if lowered.startswith(expected_prefix):
        return f"c{token}" if not lowered.startswith("c") else token
    return f"c{token}"


def _normalize_clinical_tnm_tokens(tokens: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key in ("cT", "cN", "cM"):
        value = tokens.get(key)
        normalized[key] = _normalize_clinical_tnm_value(key, value) if value else ""
    return normalized


def _infer_stage_group_from_tnm(tokens: Dict[str, str]) -> str:
    cT = str(tokens.get("cT") or "").upper()
    cN = str(tokens.get("cN") or "").upper()
    cM = str(tokens.get("cM") or "").upper()

    if "M1" in cM:
        return "IV期"
    if "N1" in cN or "N2" in cN:
        return "III期"
    if any(marker in cT for marker in ["T3", "T4"]):
        return "II期"
    if any(marker in cT for marker in ["T1", "T2"]) and "M0" in cM:
        return "I期"
    return ""


def _format_tnm_display(tokens: Dict[str, str]) -> str:
    cT = str(tokens.get("cT") or "")
    cN = str(tokens.get("cN") or "")
    cM = str(tokens.get("cM") or "")
    return "".join([
        cT,
        cN[1:] if cN.lower().startswith("c") else cN,
        cM[1:] if cM.lower().startswith("c") else cM,
    ])


def _infer_tumor_subsite(text: str) -> str:
    lowered = str(text or "").lower()
    subsite_markers = [
        ("乙状结肠", ["乙状结肠", "sigmoid"]),
        ("降结肠", ["降结肠", "descending colon"]),
        ("横结肠", ["横结肠", "transverse colon"]),
        ("升结肠", ["升结肠", "ascending colon"]),
        ("盲肠", ["盲肠", "cecum", "caecum"]),
        ("直肠", ["直肠", "rectum", "rectal"]),
    ]
    for label, markers in subsite_markers:
        if any(marker in lowered for marker in markers):
            return label
    return ""


def _quick_extract_histology(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"(中分化腺癌|高分化腺癌|低分化腺癌|腺癌|adenocarcinoma)", raw, re.IGNORECASE)
    if match:
        value = match.group(1)
        if value.lower() == "adenocarcinoma":
            return "腺癌"
        return value
    return "未知"


def _quick_extract_diagnosis_from_text(full_text: str) -> "DiagnosisExtractionResult":
    text = str(full_text or "")
    lowered = text.lower()
    tnm_tokens = _normalize_clinical_tnm_tokens(_extract_tnm_tokens_from_text(text))

    if not tnm_tokens.get("cM"):
        if any(
            marker in lowered
            for marker in ["无远处转移", "未见远处转移", "未见明确转移", "no distant metastasis", "without distant metastasis"]
        ):
            tnm_tokens["cM"] = "cM0"

    location = _infer_location_category(text)
    subsite = _infer_tumor_subsite(text)
    markers = _quick_extract_molecular_markers(text)
    pathology_confirmed = any(
        marker in lowered for marker in ["活检", "病理", "biopsy", "pathology", "adenocarcinoma", "腺癌"]
    )
    histology = _quick_extract_histology(text)
    stage_group = _infer_stage_group_from_tnm(tnm_tokens)

    stage_summary_parts = []
    location_text = subsite or ("结肠" if location == "Colon" else "直肠" if location == "Rectum" else "")
    pathology_text = "".join([part for part in [location_text, histology if histology != "未知" else ""] if part])
    if pathology_text:
        stage_summary_parts.append(pathology_text)
    if histology != "未知":
        pass
    tnm_compact = _format_tnm_display(tnm_tokens)
    if tnm_compact:
        if stage_group:
            stage_summary_parts.append(f"{tnm_compact}（{stage_group}）")
        else:
            stage_summary_parts.append(tnm_compact)
    clinical_stage_summary = "；".join([part for part in stage_summary_parts if part])
    tnm_payload: Dict[str, Any] = {
        "clinical": dict(tnm_tokens),
        "stage_group": stage_group,
        **tnm_tokens,
    }

    return DiagnosisExtractionResult(
        pathology_confirmed=pathology_confirmed,
        tumor_location=location,
        tumor_subsite=subsite,
        histology_type=histology,
        molecular_markers=markers,
        rectal_mri_params={},
        tnm_staging=tnm_payload,
        clinical_stage_summary=clinical_stage_summary,
    )

def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _looks_like_non_gi_vague_symptom(text: str) -> bool:
    compact = _compact(text)
    patterns = (
        r"(?:浑身|全身|哪里都|整个人).{0,4}(?:不舒服|不适|难受)",
    )
    return any(re.search(pattern, compact) for pattern in patterns)


def _has_crc_assessment_anchor(text: str) -> bool:
    compact = _compact(text).lower()
    markers = (
        "结直肠癌",
        "直肠癌",
        "结肠癌",
        "病理",
        "活检",
        "腺癌",
        "肿瘤",
        "癌",
        "adenocarcinoma",
        "tnm",
        "分期",
        "治疗",
        "方案",
        "化疗",
        "放疗",
        "手术",
        "靶向",
        "免疫",
        "mmr",
        "msi",
        "cea",
    )
    return any(str(marker).lower() in compact for marker in markers) or bool(re.search(r"(?:c?t[0-4x]|c?n[0-3x]|c?m[01x])", compact))


def _should_use_non_crc_symptom_clarification(text: str, integrity: "CaseIntegrity") -> bool:
    return _looks_like_non_gi_vague_symptom(text) and not integrity.has_confirmed_diagnosis and not _has_crc_assessment_anchor(text)


def _create_non_crc_symptom_clarification_message() -> str:
    return (
        "这句话更像是一般身体不适描述，还不足以直接进入结直肠癌评估。\n"
        "如果主要是肠胃或腹部不适，我可以继续按门诊分诊追问；如果是浑身不舒服、全身难受这类非胃肠道症状，建议先到全科、内科或急诊线下就诊。\n"
        "您也可以直接补充更具体的情况，例如“腹痛3天”“有便血”“腹泻2周”或“我想咨询治疗方案”。"
    )


# 删除旧的 PatientProfile 定义，直接使用 state.py 中的版本
# 旧的 PatientProfile 定义已删除，避免与 state.py 中的定义冲突


# ==============================================================================
# 2. 宽松的 Schema (全部 String)
# ==============================================================================

class ClinicalAssessmentResult(BaseModel):
    """临床评估结果"""
    risk_level: str = Field(description="风险等级 (High/Moderate/Average)")
    red_flags: List[str] = Field(description="高危症状", default_factory=list)
    missing_critical_data: List[str] = Field(description="缺失数据", default_factory=list)
    assessment_summary: str = Field(description="病情总结")
    reasoning: str = Field(description="分析过程")
    
    @field_validator('risk_level')
    @classmethod
    def validate_risk_level(cls, v):
        """确保风险等级只接受有效值"""
        valid_levels = {"High", "Moderate", "Average"}
        if v not in valid_levels:
            # 自动修正常见错误格式
            v_normalized = v.strip()
            if v_normalized.lower() == "high":
                return "High"
            elif v_normalized.lower() == "moderate":
                return "Moderate"
            elif v_normalized.lower() == "average":
                return "Average"
            # 无法识别，返回默认 Average
            print(f"⚠️ [Validation] 无效的风险等级 '{v}'，自动修正为 'Average'")
            return "Average"
        return v
    
    @field_validator('red_flags')
    @classmethod
    def validate_red_flags(cls, v):
        """清理 red_flags 中的特殊字符"""
        cleaned = []
        for flag in v:
            # 移除可能导致 JSON 问题的特殊字符
            cleaned_flag = flag.replace('"', '\\"').replace('\\', '').strip()
            # 确保不包含中文标点和复杂字符
            if cleaned_flag and len(cleaned_flag) < 100:
                cleaned.append(cleaned_flag)
        return cleaned

class DiagnosisExtractionResult(BaseModel):
    """诊断结果"""
    # [修复] 添加默认值，避免 Pydantic 验证错误
    pathology_confirmed: bool = Field(False, description="是否病理确诊")
    tumor_location: Literal["Rectum", "Colon", "Unknown"] = Field("Unknown", description="肿瘤部位 (Rectum/Colon/Unknown)")
    tumor_subsite: str = Field("", description="原发灶亚部位")
    histology_type: str = Field("未知", description="病理类型")
    molecular_markers: Dict[str, Any] = Field(default_factory=dict, description="分子标志物")
    rectal_mri_params: Dict[str, Any] = Field(default_factory=dict, description="MRI参数")
    # [新增] TNM分期信息
    tnm_staging: Dict[str, Any] = Field(default_factory=dict, description="TNM分期")
    clinical_stage_summary: str = Field("", description="临床分期总结")

    @field_validator("tumor_location", mode="before")
    @classmethod
    def normalize_tumor_location(cls, v):
        """将各种位置表述规范化为 Rectum/Colon/Unknown。"""
        if v is None:
            return "Unknown"
        inferred = _infer_location_category(v)
        if inferred != "Unknown":
            return inferred
        inferred = _infer_location_category(v)
        if inferred != "Unknown":
            return inferred
        s = str(v).strip().lower()
        if s in {"unknown", "n/a", "na", ""}:
            return "Unknown"
        return "Unknown"
        if not s or s in {"unknown", "n/a", "na", "?"}:
            return "Unknown"
        return "Unknown"
        if not s:
            return "Unknown"
        if "rect" in s or "直" in s:
            return "Rectum"
        if "colon" in s or "结" in s:
            return "Colon"
        if s in {"unknown", "n/a", "na", "?"}:
            return "Unknown"
        # 非法值：医疗场景宁可 Unknown，不要猜
        return "Unknown"

    @property
    def derived_mmr_status(self) -> str:
        """基于分子标志物推导 MMR 状态（替代 _extract_mmr_status）。"""
        markers = self.molecular_markers or {}
        # 允许多种键名/值形态
        if markers.get("MSI-H") or markers.get("dMMR") or str(markers.get("MSI", "")).upper() == "H":
            return "dMMR"
        if markers.get("MSS") or markers.get("pMMR"):
            return "pMMR"
        # 有些数据源会直接给出 mmr_status
        mmr = markers.get("MMR") or markers.get("mmr_status")
        if isinstance(mmr, str):
            if "dmmr" in mmr.lower() or "msi-h" in mmr.lower():
                return "dMMR"
            if "pmmr" in mmr.lower() or "mss" in mmr.lower():
                return "pMMR"
        return "Unknown"

    @property
    def derived_kras_status(self) -> str:
        """基于分子标志物推导 RAS/KRAS 状态（替代 _extract_kras_status）。"""
        markers = self.molecular_markers or {}
        val = markers.get("RAS") or markers.get("KRAS") or markers.get("NRAS")
        if isinstance(val, bool):
            return "Unknown"
        if val is None:
            return "Unknown"
        s = str(val).strip()
        return s if s in {"WildType", "Mutant"} else "Unknown"

    @property
    def derived_braf_status(self) -> str:
        """基于分子标志物推导 BRAF 状态（替代 _extract_braf_status）。"""
        markers = self.molecular_markers or {}
        val = markers.get("BRAF")
        if isinstance(val, bool):
            return "Unknown"
        if val is None:
            return "Unknown"
        s = str(val).strip()
        return s if s in {"WildType", "Mutant"} else "Unknown"


# ==============================================================================
# 3.1 Semantic Guard: CaseIntegrity 语义完整性模型
# ==============================================================================

class CaseIntegrity(BaseModel):
    """
    [Semantic Guard] 语义层面的病例完整性检查结果
    替代原来的正则匹配逻辑，使用 LLM 进行语义判断
    """
    # 基础三要素检查
    has_confirmed_diagnosis: bool = Field(
        description="用户是否明确提及已确诊（含病理、活检或明确的'确诊为...'描述）"
    )
    tumor_location_category: Literal["Rectum", "Colon", "Unknown"] = Field(
        description="根据解剖位置归类。例如：'盲肠'->Colon, '乙状结肠'->Colon, '距肛5cm'->Rectum"
    )
    tnm_status: Literal["Complete", "Partial", "Missing"] = Field(
        description="TNM分期完整性。Complete=含T/N/M信息(包括'无转移'等描述); Partial=部分缺失; Missing=全无"
    )

    # 进阶：风险与数据缺失逻辑
    is_advanced_stage: bool = Field(
        description="基于TNM判断是否为进展期(III/IV期)。例如 N>=1 或 M>=1。如果'未见转移'则为False。"
    )

    # 解决死循环的关键：用户对缺失数据的态度
    mmr_status_availability: Literal["Provided", "Not_Provided", "User_Refused_Or_Unknown"] = Field(
        description="用户关于MMR/MSI状态的反馈。Provided=已提供; Not_Provided=未提及; User_Refused_Or_Unknown=用户明确说'没做'、'不知道'或拒绝提供。"
    )

    # 症状描述识别：判断用户是否仅描述症状而未确诊
    is_symptom_only: bool = Field(
        description="用户输入是否仅包含症状描述（如便血、腹痛）而未提及确诊信息。用于区分症状咨询和已确诊患者"
    )

    reasoning: str = Field(description="简短的判断依据")

    @field_validator("tumor_location_category", mode="before")
    @classmethod
    def normalize_location_category(cls, v):
        if v is None:
            return "Unknown"
        inferred = _infer_location_category(v)
        if inferred != "Unknown":
            return inferred
        s = str(v).strip().lower()
        if "rect" in s or "直" in s:
            return "Rectum"
        if "colon" in s or "结" in s:
            return "Colon"
        if s in {"unknown", "n/a", "na", ""}:
            return "Unknown"
        return "Unknown"

    @field_validator("tnm_status", mode="before")
    @classmethod
    def normalize_tnm_status(cls, v):
        if v is None:
            return "Missing"
        s = str(v).strip().lower()
        if s in {"complete", "completed", "full"}:
            return "Complete"
        if s in {"partial", "incomplete"}:
            return "Partial"
        if s in {"missing", "none"}:
            return "Missing"
        return "Missing"

    @field_validator("mmr_status_availability", mode="before")
    @classmethod
    def normalize_mmr_availability(cls, v):
        if v is None:
            return "Not_Provided"
        s = str(v).strip().lower()
        if s in {"provided", "yes", "y"}:
            return "Provided"
        if s in {"not_provided", "not provided", "no", "n"}:
            return "Not_Provided"
        if "refus" in s or "unknown" in s or "不知道" in s or "没做" in s:
            return "User_Refused_Or_Unknown"
        return "Not_Provided"


# ==============================================================================
# 3.1.1 信息缺失卡片格式化工具
# ==============================================================================

def _format_missing_info_card(title: str, content: str) -> str:
    """
    格式化信息缺失时的卡片展示

    Args:
        title: 卡片标题（如"诊疗方案详情"）
        content: 需要补充的具体内容

    Returns:
        格式化后的卡片文本
    """
    return (
        "⚠️ 注意：本次回复未生成结构化卡片，以下为文本详情：\n\n"
        f"📝 {title}\n"
        f"{content}"
    )


def _create_inquiry_message(category: str, detail: str) -> str:
    """
    创建追问消息（带卡片格式）

    Args:
        category: 缺失信息类别
        detail: 详细信息

    Returns:
        格式化的追问消息
    """
    return _format_missing_info_card(
        f"{category}缺失提醒",
        detail
    )


def _prepend_crc_entry_explanation(
    state: CRCAgentState,
    inquiry_message: str,
) -> tuple[str, bool]:
    findings = state.findings or {}
    encounter_track = state.encounter_track or findings.get("encounter_track")
    explanation_shown = bool(state.entry_explanation_shown or findings.get("entry_explanation_shown", False))
    clinical_entry_reason = state.clinical_entry_reason or findings.get("clinical_entry_reason")

    if encounter_track != "crc_clinical" or explanation_shown:
        return inquiry_message, explanation_shown

    if clinical_entry_reason not in {"system_known_crc_signal", "user_provided_abnormal_test"}:
        return inquiry_message, explanation_shown

    explanation = (
        "我先不按普通消化门诊分诊继续问症状，因为系统已识别到已有病理、肠镜或影像异常线索，"
        "这已经进入结直肠肿瘤临床评估范围。接下来我会先补齐病理、影像和分期信息。"
    )
    return f"{explanation}\n\n{inquiry_message}", True


# ==============================================================================
# 3.2 Semantic Guard: check_case_integrity 语义完整性检查器
# ==============================================================================

def check_case_integrity(
    user_text: str,
    model,
    pinned_context: str = "",
    summary_memory: str = ""
) -> CaseIntegrity:
    """
    [Semantic Guard v7] 使用 LLM 进行语义完整性判断
    完全替代 _is_complete_case_info 的正则逻辑

    Args:
        user_text: 用户的输入文本
        model: 用于语义判断的 LLM 实例

    Returns:
        CaseIntegrity: 包含所有语义判断结果的对象
    """

    # 纯语义 + Structured Output：失败时返回保守默认值（不再用 regex 兜底，医疗场景宁可 Unknown）
    if os.getenv("ASSESSMENT_FAST_RULES_ONLY", "true").strip().lower() not in {"0", "false", "no"}:
        return _quick_case_integrity_from_text("\n".join([summary_memory or "", pinned_context or "", user_text or ""]))

    system_prompt = CASE_INTEGRITY_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_text}")
    ])
    def _fallback_integrity_from_text(full_text: str) -> CaseIntegrity:
        """
        当结构化输出失败时，使用轻量规则兜底，避免误判“病理缺失”。
        """
        text = (full_text or "").lower()

        has_confirmed_diagnosis = any(
            kw in text for kw in [
                "病理", "活检", "术后病理", "腺癌", "癌", "确诊", "adenocarcinoma"
            ]
        )

        tnm_tokens = _extract_tnm_tokens_from_text(full_text)
        has_t = bool(tnm_tokens.get("cT"))
        has_n = bool(tnm_tokens.get("cN"))
        has_m = bool(tnm_tokens.get("cM")) or any(
            kw in text for kw in ["无远处转移", "未见远处转移", "no distant metastasis"]
        )

        if has_t and has_n and has_m:
            tnm_status = "Complete"
        elif has_t or has_n or has_m:
            tnm_status = "Partial"
        else:
            tnm_status = "Missing"

        if any(kw in text for kw in ["直肠", "距肛", "rectum", "rectal"]):
            tumor_location_category = "Rectum"
        elif any(kw in text for kw in ["结肠", "乙状", "盲肠", "升结肠", "横结肠", "降结肠", "colon", "sigmoid"]):
            tumor_location_category = "Colon"
        else:
            tumor_location_category = "Unknown"

        tumor_location_category = _infer_location_category(full_text)

        mmr_status_availability = "Provided" if any(
            kw in text for kw in ["pmmr", "dmmr", "mss", "msi", "mlh1", "msh2", "msh6", "pms2"]
        ) else "Not_Provided"

        n_token = str(tnm_tokens.get("cN", ""))
        m_token = str(tnm_tokens.get("cM", ""))
        is_advanced_stage = bool(
            re.search(r"n[1-3]", n_token, re.IGNORECASE)
            or re.search(r"m1", m_token, re.IGNORECASE)
        )
        symptom_hits = any(kw in text for kw in ["便血", "腹痛", "腹胀", "便秘", "腹泻", "体重下降", "乏力", "symptom"])
        is_symptom_only = symptom_hits and not has_confirmed_diagnosis

        return CaseIntegrity(
            has_confirmed_diagnosis=has_confirmed_diagnosis,
            tumor_location_category=tumor_location_category,
            tnm_status=tnm_status,
            is_advanced_stage=is_advanced_stage,
            mmr_status_availability=mmr_status_availability,
            is_symptom_only=is_symptom_only,
            reasoning="Fallback: structured output parse failed; used lightweight semantic rules."
        )

    payload = {
        "user_text": user_text,
        "summary_memory": summary_memory,
        "pinned_context": pinned_context
    }
    return _invoke_structured_with_recovery(
        prompt=prompt,
        model=model,
        schema=CaseIntegrity,
        payload=payload,
        log_prefix="[Semantic Guard]",
        fallback_factory=lambda _payload, _err: _fallback_integrity_from_text(
            "\n".join([summary_memory or "", pinned_context or "", user_text or ""])
        ),
    )


def _semantic_extract_diagnosis(
    user_text: str,
    model,
    pinned_context: str = "",
    summary_memory: str = ""
) -> DiagnosisExtractionResult:
    """
    [纯语义提取] Fast Pass/Diagnosis 共用的诊断信息抽取器。
    - 只依赖 LLM Structured Output，不做 regex 兜底
    - 失败时返回保守默认值（Unknown/空），由追问机制或后续节点处理
    """
    if os.getenv("ASSESSMENT_FAST_RULES_ONLY", "true").strip().lower() not in {"0", "false", "no"}:
        return _quick_extract_diagnosis_from_text("\n".join([summary_memory or "", pinned_context or "", user_text or ""]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", DIAGNOSIS_SYSTEM_PROMPT),
        ("user", "{user_text}")
    ])
    payload = {
        "user_text": user_text,
        "summary_memory": summary_memory,
        "pinned_context": pinned_context
    }
    return _invoke_structured_with_recovery(
        prompt=prompt,
        model=model,
        schema=DiagnosisExtractionResult,
        payload=payload,
        log_prefix="[Diagnosis Extract]",
        fallback_factory=lambda _payload, _err: DiagnosisExtractionResult(),
    )


# ==============================================================================
# 4. Assessment Node (Fast Pass 修复版 + PatientProfile + Active Inquiry)
# ==============================================================================

def node_assessment(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    
    # 从统一的 prompts 模块导入 System Prompt
    system_prompt = ASSESSMENT_SYSTEM_PROMPT
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Patient Profile: {patient_profile}"),
        ])
        | model.with_structured_output(ClinicalAssessmentResult)
    )

    def _run(state: CRCAgentState):
        # 使用 _latest_user_text 获取用户的原始输入，而不是可能包含RAG内容的最后一条消息
        user_text = _latest_user_text(state)
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)
        intent = (state.findings or {}).get("user_intent", "assessment")
        current_findings = state.findings or {}
        encounter_track = state.encounter_track or current_findings.get("encounter_track")
        is_crc_entry_context = encounter_track == "crc_clinical"

        # [新增] 检查是否有从数据库查询来的临床数据
        db_clinical_data = current_findings.get("data_source") == "database_query"
        
        # [新增] 检查是否有影像分析报告
        has_radiology_report = "radiology_report" in current_findings

        if db_clinical_data:
            if show_thinking:
                print(f"[Assessment] Detected database query data, using clinical info for patient ID {current_findings.get('db_query_patient_id')}...")

            # 使用数据库中的临床信息构建评估
            # 这些信息来自 get_patient_case_info 工具
            db_tumor_location = current_findings.get("tumor_location", "unknown")
            db_pathology_confirmed = current_findings.get("pathology_confirmed", False)
            db_tnm_staging = current_findings.get("tnm_staging", {})
            db_histology_type = current_findings.get("histology_type", "腺癌")
            db_molecular_markers = current_findings.get("molecular_markers", {})
            db_mmr_status = current_findings.get("mmr_status", None)
            db_clinical_summary = current_findings.get("clinical_stage_summary", "")
            # [新增] 提取基本信息
            db_age = current_findings.get("age")
            db_gender = current_findings.get("gender")
            db_ecog = current_findings.get("ecog_score")

            # 构建患者信息的完整描述，供 LLM 分析
            db_patient_info_text = f"""
患者ID: {current_findings.get('db_query_patient_id')}
性别/年龄: {db_gender}/{db_age}
ECOG评分: {db_ecog}
肿瘤位置: {db_tumor_location}
病理类型: {db_histology_type}
TNM分期: {db_tnm_staging}
MMR状态: {db_mmr_status or '未知'}
分子标志物: {db_molecular_markers}
临床分期: {db_clinical_summary}
"""

            if show_thinking:
                print(f"[Assessment] Database patient info: location={db_tumor_location}, "
                      f"pathology={db_pathology_confirmed}, TNM={db_tnm_staging}")

            # 构建 findings_delta，包含从数据库提取的信息
            findings_delta = {
                "pathology_confirmed": db_pathology_confirmed,
                "biopsy_confirmed": db_pathology_confirmed,
                "tumor_location": db_tumor_location,
                "histology_type": db_histology_type,
                "molecular_markers": db_molecular_markers,
                "tnm_staging": db_tnm_staging,
                "clinical_stage_summary": db_clinical_summary,
                "mmr_status": db_mmr_status,
                "data_source": "database_query",
                "db_query_patient_id": current_findings.get('db_query_patient_id'),
                "age": db_age,
                "gender": db_gender,
                "ecog_score": db_ecog,
            }

            # [Fast Pass] 如果数据库信息完整，直接构建 PatientProfile 并返回
            is_info_complete = (
                db_pathology_confirmed and
                db_tumor_location != "unknown" and
                db_tnm_staging and
                any(db_tnm_staging.values())
            )

            # [修复] 检查用户意图是否匹配，只有治疗决策相关意图才触发 Fast Pass
            is_intent_appropriate = intent in [
                "treatment_decision",
                "clinical_assessment",
                "search_treatment_recommendations",
                "multi_task",
            ]

            if is_info_complete and is_intent_appropriate:
                if show_thinking:
                    print(f"[Assessment] Database info complete, executing Fast Pass mode...")

                # 创建 PatientProfile (已在文件顶部导入)
                new_profile = PatientProfile(
                    tumor_type=db_tumor_location.capitalize() if db_tumor_location != "unknown" else "Unknown",
                    pathology_confirmed=True,
                    mmr_status=db_mmr_status or "Unknown",
                    tnm_staging=db_tnm_staging,
                    is_locked=True,
                    age=db_age,
                    gender=db_gender,
                    ecog_score=db_ecog
                )

                # 检查是否为进展期
                n_stage = db_tnm_staging.get("cN", "")
                m_stage = db_tnm_staging.get("cM", "")
                is_advanced = ("N1" in n_stage or "N2" in n_stage or "M1" in m_stage)

                # [新增] 如果有影像报告，整合到返回结果中
                assessment_text = f"数据库查询患者信息：{db_clinical_summary or db_histology_type}"
                if has_radiology_report:
                    radiology_report = current_findings.get("radiology_report", {})
                    imaging_summary = radiology_report.get("ai_interpretation", "")
                    assessment_text += f"\n\n【影像学AI分析】\n{imaging_summary[:500]}..."  # 截取前500字符
                
                profile_entry = _build_profile_change_entry(
                    state.patient_profile,
                    new_profile,
                    source="assessment_fast_pass_db"
                )
                updates = {
                    "patient_profile": new_profile,
                    "findings": {
                        **findings_delta,
                        "risk_level": "High" if is_advanced else "Moderate",
                        "assessment_draft": assessment_text,
                        "fast_pass_mode": True,
                        "is_advanced_stage": is_advanced,
                    },
                    "assessment_draft": json.dumps({
                        "risk": "High" if is_advanced else "Moderate",
                        "summary": db_clinical_summary or f"{db_tumor_location} {db_histology_type}",
                        "reasoning": f"从数据库查询获取的完整患者信息 (ID: {current_findings.get('db_query_patient_id')})"
                    }, ensure_ascii=False),
                    "missing_critical_data": [],
                    "clinical_stage": "Assessment_Completed",
                    "roadmap": auto_update_roadmap_from_state(state)
                }
                if profile_entry:
                    updates["patient_profile_timeline"] = [profile_entry]
                return updates

            # [部分信息] 如果信息不完整但有一些数据，合并到常规处理
            if show_thinking:
                print(f"[Assessment] Database info partially complete, continuing regular assessment...")

            # 将数据库信息合并到 user_text 中，供后续 LLM 分析使用
            user_text = f"【从数据库查询的患者信息】\n{db_patient_info_text}\n【用户问题】\n{user_text}"

        # [Semantic Guard v6] 使用 LLM 进行语义完整性判断
        # 替代原来的 _is_complete_case_info 正则匹配逻辑
        integrity: CaseIntegrity = check_case_integrity(
            user_text,
            model,
            pinned_context=pinned_context,
            summary_memory=summary_memory
        )

        if show_thinking:
            # 截断过长的reasoning，避免输出大段RAG内容
            reasoning_preview = integrity.reasoning[:100] + "..." if len(integrity.reasoning) > 100 else integrity.reasoning
            print(f"[Semantic Guard] Location: {integrity.tumor_location_category}, "
                  f"MMR Status: {integrity.mmr_status_availability}, "
                  f"Reasoning: {reasoning_preview}")

        # ================================================================
        # 策略分支 0: 症状描述反问（最高优先级）
        # 如果用户仅描述症状而未确诊，应该先反问确认是否为结直肠癌相关
        # ================================================================

        # 检查是否仅包含症状描述（未确诊）
        if integrity.is_symptom_only and not is_crc_entry_context:
            if show_thinking:
                print(f"💬 [Assessment] 检测到症状描述，需要反问确认诊断")

            # 使用友好的反问消息
            inquiry_message = """您好！我了解到您描述了【{}】等症状。

请问：**您是否已经确诊为结直肠癌（直肠癌或结肠癌）？**

- 如果已确诊，请提供诊断信息（如病理报告、影像学报告等）
- 如果未确诊，建议您先到消化科就诊，进行肠镜等相关检查

---

📌 **注**：我是一个专门服务结直肠癌患者的决策支持系统，需要明确的诊断信息才能提供准确的分期和治疗方案建议。""".format(user_text[:50])

            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Confirmed Diagnosis"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "symptom_inquiry",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump()
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state)
            }

        # ================================================================
        # 策略分支 1: 基础三要素缺失检查
        # 使用语义判断替代正则匹配
        # ================================================================

        # 检查是否缺少病理确诊
        if is_crc_entry_context and _should_use_non_crc_symptom_clarification(user_text, integrity):
            inquiry_message = _create_non_crc_symptom_clarification_message()
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Symptom Clarification"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "symptom_inquiry",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump()
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state)
            }

        if not integrity.has_confirmed_diagnosis:
            if show_thinking:
                print(f"🛑 [Assessment] 缺少病理确诊信息")

            # 使用卡片格式展示
            inquiry_message = _create_inquiry_message(
                "病理确诊信息",
                "请提供病理确诊信息（如活检或术后病理报告），以便进行准确的分期和治疗方案制定。"
            )
            inquiry_message, entry_explanation_shown = _prepend_crc_entry_explanation(state, inquiry_message)

            updates = {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Pathology Report"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "pathology_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "entry_explanation_shown": entry_explanation_shown,
                    "semantic_integrity": integrity.model_dump()
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state)
            }
            if entry_explanation_shown:
                updates["entry_explanation_shown"] = True
            return updates

        # 检查 TNM 分期是否完整（Complete 才允许进入后续决策/快速通道）
        if integrity.tnm_status != "Complete":
            if show_thinking:
                print(f"🛑 [Assessment] TNM 分期不完整: {integrity.tnm_status}")

            # 使用卡片格式展示
            inquiry_message = _create_inquiry_message(
                "TNM分期信息",
                "请补充 TNM 分期信息（尤其是 cT、cN、cM）。可以直接口述影像学报告结论（如“cT3b cN1 cM0/未见远处转移”），无需提供原始报告全文。"
            )
            inquiry_message, entry_explanation_shown = _prepend_crc_entry_explanation(state, inquiry_message)

            updates = {
                "clinical_stage": "Pending",
                "missing_critical_data": ["TNM Staging"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "tnm_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "entry_explanation_shown": entry_explanation_shown,
                    "semantic_integrity": integrity.model_dump()
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state)
            }
            if entry_explanation_shown:
                updates["entry_explanation_shown"] = True
            return updates

        # 检查肿瘤位置是否明确
        if integrity.tumor_location_category == "Unknown":
            if show_thinking:
                print(f"🛑 [Assessment] 肿瘤位置不明确")

            # 使用卡片格式展示
            inquiry_message = _create_inquiry_message(
                "肿瘤位置信息",
                "请明确肿瘤位置（直肠还是结肠），以及距肛缘距离（如有），以便选择合适的分期标准和治疗方案。"
            )
            inquiry_message, entry_explanation_shown = _prepend_crc_entry_explanation(state, inquiry_message)

            updates = {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Tumor Location"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "location_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "entry_explanation_shown": entry_explanation_shown,
                    "semantic_integrity": integrity.model_dump()
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state)
            }
            if entry_explanation_shown:
                updates["entry_explanation_shown"] = True
            return updates

        # ================================================================
        # 策略分支 2: 主动追问 (Active Inquiry)
        # 当检测到进展期肿瘤但未提供 MMR/MSI 状态，且用户没有拒绝时触发
        # ================================================================
        needs_inquiry = (
            integrity.is_advanced_stage
            and integrity.mmr_status_availability == "Not_Provided"
        )

        if needs_inquiry:
            if show_thinking:
                print(f"🛑 [Assessment] 发现进展期肿瘤缺失 MMR/MSI 状态，触发主动追问。")

            # 使用卡片格式展示
            inquiry_message = _format_missing_info_card(
                "MMR/MSI检测信息缺失",
                "检测到患者为进展期结直肠癌。为了制定精准的化疗或免疫治疗方案，**请提供 MSI（微卫星不稳定性）或 MMR（错配修复蛋白）的检测结果**。\n\n"
                "如您暂无此检测结果，建议尽快进行免疫组化(IHC)或PCR检测，因为 MMR/MSI 状态可能影响后续的治疗选择（如是否适用免疫治疗）。"
            )
            inquiry_message, entry_explanation_shown = _prepend_crc_entry_explanation(state, inquiry_message)

            updates = {
                "clinical_stage": "Inquiry_Pending",
                "missing_critical_data": ["MMR/MSI Status"],
                "findings": {
                    "risk_level": "High",
                    "inquiry_type": "mmr_status_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "entry_explanation_shown": entry_explanation_shown,
                    "semantic_integrity": integrity.model_dump()
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state)
            }
            if entry_explanation_shown:
                updates["entry_explanation_shown"] = True
            return updates

        # ================================================================
        # 策略分支 3: 阻断死循环
        # 如果用户明确说"没做"或"不知道"，我们就不追问了
        # 标记为 Unknown 让后续 Decision 节点处理
        # ================================================================
        mmr_data_for_profile = "Unknown"
        if integrity.mmr_status_availability == "Provided":
            # 这里可以让后续的 extraction 节点去提取具体是 pMMR 还是 dMMR
            mmr_data_for_profile = "To_Be_Extracted"
        elif integrity.mmr_status_availability == "User_Refused_Or_Unknown":
            # 用户明确拒绝或不知道，不追问，但需要后续处理
            mmr_data_for_profile = "Unknown"

        # ================================================================
        # 策略分支 4: Fast Pass 模式 (构建只读 PatientProfile)
        # 当基础信息完整且无关键缺失项时触发
        # ================================================================

        # 判断是否满足 Fast Pass 条件：基础三要素完整 + 不需要追问 + 意图匹配
        # [修复] 添加意图检查，避免在不相关的问题时触发 Fast Pass
        is_intent_appropriate = intent in [
            "treatment_decision",
            "clinical_assessment",
            "search_treatment_recommendations",
            "multi_task",
        ]
        is_fast_pass_eligible = (
            integrity.has_confirmed_diagnosis
            and integrity.tumor_location_category != "Unknown"
            and integrity.tnm_status == "Complete"
            and not needs_inquiry
            and is_intent_appropriate  # [新增] 只有治疗决策相关意图才触发 Fast Pass
        )

        if is_fast_pass_eligible:
            if show_thinking:
                print(f"🩺 [Assessment] Fast Pass 模式：轻量化提取并构建 PatientProfile...")
            
            # 纯语义提取（Structured Output）
            diag_res = _semantic_extract_diagnosis(
                user_text,
                model,
                pinned_context=pinned_context,
                summary_memory=summary_memory
            )
            
            # [新增] 从用户文本中提取基本信息（年龄、性别、主诉）
            import re
            age = None
            gender = None
            chief_complaint = user_text[:100] if user_text else ""  # 取前100字符作为主诉
            
            # 提取年龄（支持 "58岁", "58岁", "58 years old" 等格式）
            age_matches = re.findall(r'(\d+)\s*岁', user_text)
            if age_matches:
                age = int(age_matches[0])
            else:
                # 英文格式
                age_matches = re.findall(r'(\d+)\s*(years?|y\.o\.?)', user_text, re.IGNORECASE)
                if age_matches:
                    age = int(age_matches[0])
            
            # 提取性别
            if "女" in user_text or "女" in user_text or "female" in user_text.lower():
                gender = "女"
            elif "男" in user_text or "男" in user_text or "male" in user_text.lower():
                gender = "男"
            
            # 提取主诉（从患者信息开头到第一个句号或逗号）
            if "主诉" in user_text:
                match = re.search(r'主诉[：:]\s*([^，。;\n]+)', user_text)
                if match:
                    chief_complaint = match.group(1).strip()
            
            if show_thinking:
                print(f"  [Fast Pass] 提取的基本信息: age={age}, gender={gender}, chief_complaint={chief_complaint[:50]}...")
            
            # [核心修改 v4] 构建不可变的 PatientProfile
            molecular_markers = diag_res.molecular_markers or {}
            
            new_profile = PatientProfile(
                tumor_type=diag_res.tumor_location,
                pathology_confirmed=diag_res.pathology_confirmed,
                tnm_staging=diag_res.tnm_staging or {},
                mmr_status=diag_res.derived_mmr_status,
                age=age,
                gender=gender,
                chief_complaint=chief_complaint,
                is_locked=True  # 锁定档案，防止后续节点篡改
            )
            
            # 构建 findings_delta，确保 Decision 节点能拿到完整数据
            # [Semantic Guard] 使用语义检查结果更新 tumor_location
            semantic_location = integrity.tumor_location_category.lower()

            findings_delta = {
                "risk_level": "High",  # 完整病例默认高风险
                "red_flags": [],       # 简化处理
                "assessment_draft": f"Fast Pass 提取：{diag_res.clinical_stage_summary}",
                # 【核心】复制诊断数据到 findings，确保数据完整性
                "pathology_confirmed": diag_res.pathology_confirmed,
                "biopsy_confirmed": diag_res.pathology_confirmed,
                # [Semantic Guard] 使用语义分类结果，优先使用 LLM 判断的位置
                "tumor_location": semantic_location if semantic_location != "unknown" else diag_res.tumor_location.lower(),
                "tumor_subsite": diag_res.tumor_subsite,
                "histology_type": diag_res.histology_type,
                "molecular_markers": molecular_markers,
                "tnm_staging": diag_res.tnm_staging or {},
                "clinical_stage_staging": diag_res.clinical_stage_summary,
                "clinical_stage_group": (diag_res.tnm_staging or {}).get("stage_group", ""),
                # 由 Schema 派生的关键分子状态（供后续节点/路由使用）
                "mmr_status": diag_res.derived_mmr_status,
                "kras_status": diag_res.derived_kras_status,
                "braf_status": diag_res.derived_braf_status,
                # [Semantic Guard] 标记是否为进展期
                "is_advanced_stage": integrity.is_advanced_stage,
                # [Semantic Guard] 保存语义检查结果，供后续节点参考
                "semantic_integrity": integrity.model_dump(),
                # [标记] 此标志告诉后续节点"这是 Fast Pass 模式，评估已简化但数据完整"
                "fast_pass_mode": True,
            }
            
            profile_entry = _build_profile_change_entry(
                state.patient_profile,
                new_profile,
                source="assessment_fast_pass_semantic"
            )
            updates = {
                "patient_profile": new_profile,  # [核心] 存入只读 PatientProfile
                "findings": findings_delta,
                "assessment_draft": json.dumps({
                    "risk": "High",
                    "summary": diag_res.clinical_stage_summary or "完整病例",
                    "reasoning": f"Fast Pass 模式：Semantic Guard 语义检查通过，PatientProfile 已锁定。Location={integrity.tumor_location_category}, Advanced={integrity.is_advanced_stage}"
                }, ensure_ascii=False),
                "missing_critical_data": [],  # 完整病例无缺失数据
                "clinical_stage": "Assessment_Completed",
                "error": None,
                "roadmap": auto_update_roadmap_from_state(state)
            }
            if profile_entry:
                updates["patient_profile_timeline"] = [profile_entry]

            if show_thinking:
                print(f"🩺 [Assessment] Fast Pass 完成：TNM={diag_res.tnm_staging}, location={integrity.tumor_location_category}")
                print(f"🔒 [PatientProfile] 已锁定，tumor_type={new_profile.tumor_type}, mmr_status={new_profile.mmr_status}")

            return updates

        # ================================================================
        # 策略分支 5: 常规流程 (非完整病例，走完整的慢速分析流程)
        # ================================================================
        try:
            # [关键修复] 激进截断消息历史，防止 token 超限
            truncated_messages = _truncate_message_history(
                state.messages,
                max_tokens=int(os.getenv("NODE_TRUNCATE_MAX_TOKENS", "15000")),
                keep_last_n=10,
                max_chars_per_message=2000
            )
            pinned_context = _build_pinned_context(state)
            summary_memory = _build_summary_memory(state)
            res = chain.invoke({
                "messages": truncated_messages,
                "patient_profile": state.patient_profile,
                "summary_memory": summary_memory,
                "pinned_context": pinned_context
            })
            
            # --- 脏数据清洗 ---
            risk_clean = "Average"
            risk_raw = str(res.risk_level).lower()
            if "high" in risk_raw or "高" in risk_raw: risk_clean = "High"
            elif "mod" in risk_raw or "中" in risk_raw: risk_clean = "Moderate"
            
            # --- 智能诊断信息提取 ---
            # [Semantic Guard] 在常规流程中也保存语义检查结果
            findings_delta = {
                "risk_level": risk_clean,
                "red_flags": res.red_flags,
                "assessment_draft": res.assessment_summary,
                # [Semantic Guard] 保存语义完整性检查结果
                "semantic_integrity": integrity.model_dump(),
                # [Semantic Guard] 标记是否为进展期
                "is_advanced_stage": getattr(integrity, 'is_advanced_stage', False),
            }

            updates = {
                "findings": findings_delta,
                "assessment_draft": json.dumps({
                    "risk": risk_clean,
                    "summary": res.assessment_summary,
                    "reason": res.reasoning
                }, ensure_ascii=False),
                "missing_critical_data": res.missing_critical_data,
                "clinical_stage": "Assessment",
                "error": None
            }
            
            if show_thinking:
                print(f"📋 [Assessment] Risk: {risk_clean} | Missing: {len(res.missing_critical_data)} items")
            
            # [新增] 自动更新路线图
            updates["roadmap"] = auto_update_roadmap_from_state(state)
            
            return updates

        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ [Assessment Fail] {e}")
            
            # [原] 兜底错误处理
            error_updates = {
                "clinical_stage": "Assessment",
                "error": f"评估解析失败（未做正则降级，以免误判）：{error_msg}",
                "findings": {"risk_level": "High", "assessment_draft": "评估解析失败，请人工复核。"},
                "missing_critical_data": ["System_Error_Manual_Check"]
            }
            temp_state = state.model_copy(update=error_updates)
            error_updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
            return error_updates

    return _run


def _prepare_scene_assessment_state(state: CRCAgentState, scene: str) -> CRCAgentState:
    current_findings = dict(state.findings or {})
    encounter_track = state.encounter_track or current_findings.get("encounter_track")

    if scene == "doctor":
        if encounter_track == "crc_clinical":
            return state

        next_findings = dict(current_findings)
        next_findings["encounter_track"] = "crc_clinical"
        return state.model_copy(
            update={
                "encounter_track": "crc_clinical",
                "findings": next_findings,
            }
        )

    if scene == "patient":
        if encounter_track is not None:
            return state

        next_findings = dict(current_findings)
        next_findings["encounter_track"] = "patient_assessment"
        return state.model_copy(
            update={
                "encounter_track": "patient_assessment",
                "findings": next_findings,
            }
        )

    return state


def _scene_assessment_entry(
    scene: str,
    model,
    tools: List[BaseTool],
    streaming: bool = False,
    show_thinking: bool = True,
) -> Runnable:
    assessment = node_assessment(model=model, tools=tools, streaming=streaming, show_thinking=show_thinking)

    def _run(state: CRCAgentState):
        prepared_state = _prepare_scene_assessment_state(state, scene)
        result = assessment(prepared_state)
        if scene != "patient" or not isinstance(result, dict):
            return result

        projected_card = project_patient_self_report_card(prepared_state.model_copy(update=result))
        if projected_card is None:
            return result

        next_result = dict(result)
        next_result["patient_card"] = projected_card
        return next_result

    return _run


def node_doctor_assessment(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    return _scene_assessment_entry("doctor", model, tools, streaming=streaming, show_thinking=show_thinking)


def node_patient_assessment(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    return _scene_assessment_entry("patient", model, tools, streaming=streaming, show_thinking=show_thinking)


# ==============================================================================
# 5. Diagnosis Node (Enhanced: Extract complete staging info)
# ==============================================================================

def node_diagnosis(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    
    # 从统一的 prompts 模块导入 System Prompt
    system_prompt = DIAGNOSIS_SYSTEM_PROMPT
    
    # 分离 Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Patient Profile: {patient_profile}"),
    ])
    
    def _run(state: CRCAgentState):
        try:
            # [关键修复] 激进截断消息历史，防止 token 超限
            truncated_messages = _truncate_message_history(
                state.messages,
                max_tokens=int(os.getenv("NODE_TRUNCATE_MAX_TOKENS", "15000")),
                keep_last_n=10,
                max_chars_per_message=2000
            )
            pinned_context = _build_pinned_context(state)
            summary_memory = _build_summary_memory(state)
            payload = {
                "messages": truncated_messages,
                "patient_profile": state.patient_profile,
                "summary_memory": summary_memory,
                "pinned_context": pinned_context
            }
            res = _invoke_structured_with_recovery(
                prompt=prompt_template,
                model=model,
                schema=DiagnosisExtractionResult,
                payload=payload,
                log_prefix="[Diagnosis]",
                fallback_factory=lambda _payload, _err: DiagnosisExtractionResult(),
            )

            # --- 脏数据清洗 ---
            loc_clean = res.tumor_location.lower()

            # --- 构造完整的 findings_delta ---
            findings_delta = {
                "pathology_confirmed": res.pathology_confirmed,
                "biopsy_confirmed": res.pathology_confirmed,
                "tumor_location": loc_clean,
                "histology_type": res.histology_type,
                "molecular_markers": res.molecular_markers,
                "mri_assessment": res.rectal_mri_params,
                # [新增] 保存 TNM 分期信息，供路由决策使用
                "tnm_staging": getattr(res, 'tnm_staging', None),
                "clinical_stage_summary": getattr(res, 'clinical_stage_summary', None),
                # 由 Schema 派生的关键分子状态（供后续节点/路由使用）
                "mmr_status": res.derived_mmr_status,
                "kras_status": res.derived_kras_status,
                "braf_status": res.derived_braf_status,
            }

            updates = {
                "findings": findings_delta,
                "clinical_stage": "Diagnosis",
                "error": None,
                "pathology_confirmed": res.pathology_confirmed,
                "histology_type": res.histology_type
            }

            if not res.pathology_confirmed:
                user_txt = _latest_user_text(state)
                if _is_postop_context(user_txt):
                    updates["pathology_confirmed"] = True
                    updates["findings"]["pathology_confirmed"] = True
                else:
                    inquiry_message = _create_inquiry_message(
                        "病理确诊信息",
                        "请提供病理确诊信息（如活检或术后病理报告），以便进行准确的分期和治疗方案制定。"
                    )
                    inquiry_message, entry_explanation_shown = _prepend_crc_entry_explanation(state, inquiry_message)
                    updates["clinical_stage"] = "Diagnosis_Pending"
                    updates["missing_critical_data"] = ["Pathology Report"]
                    updates["findings"]["fast_pass_mode"] = False
                    updates["findings"]["inquiry_type"] = "pathology_required"
                    updates["findings"]["inquiry_message"] = inquiry_message
                    updates["findings"]["active_inquiry"] = True
                    updates["findings"]["entry_explanation_shown"] = entry_explanation_shown
                    updates["messages"] = [AIMessage(content=inquiry_message)]
                    if entry_explanation_shown:
                        updates["entry_explanation_shown"] = True
                    print("🛑 [Diagnosis] 无病理确诊，暂停。")
            else:
                print(f"✅ [Diagnosis] 确诊: {res.histology_type} @ {loc_clean}")

            # [新增] 自动更新路线图
            temp_state = state.model_copy(update=updates)
            updates["roadmap"] = auto_update_roadmap_from_state(temp_state)

            return updates

        except Exception as e:
            print(f"⚠️ [Diagnosis Fail] {e}")
            # 失败不降级到 regex，直接交给追问/重试机制
            error_updates = {"clinical_stage": "Diagnosis", "error": f"无法提取诊断信息，请重试: {e}"}
            # 即使出错也更新路线图
            temp_state = state.model_copy(update=error_updates)
            error_updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
            return error_updates

    return _run


# ==============================================================================
# 6. Staging Router (Fast Pass 修复版)
# ==============================================================================

def node_staging_router(state: CRCAgentState):
    # Phase 1 transitional router: narrow-scope exception. Do not expand this into a second policy layer.
    """
    [核心修复 v3] 分期路由：根据肿瘤位置选择分期节点
    
    Fast Pass 策略：
    - 如果有 fast_pass_mode 标志（完整病例），直接去 decision，跳过 staging
    - 否则正常路由到对应的 staging 节点
    
    路由逻辑：
    1. Fast Pass 模式 -> decision
    2. 未确诊 -> decision
    3. 未知位置 -> decision（无法确定则去 decision）
    4. 直肠 -> rectal_staging
    5. 结肠 -> colon_staging
    """
    f = state.findings or {}

    if state.clinical_stage == "Diagnosis_Pending" or "Pathology Report" in (state.missing_critical_data or []):
        print("[Staging Router] Diagnosis_Pending：等待病理报告，结束本轮。")
        return "end_turn"
    
    # [新增] Fast Pass 模式：完整病例直接去 decision，跳过 staging 校验
    # 虽然 Staging 节点有快速校验功能，但为了极致性能，完整病例直接去 decision
    if f.get("fast_pass_mode"):
        print("[Staging Router] Fast Pass 模式：跳过 staging，直接进入 decision")
        return "decision"
    
    # 检查是否需要跳过 Staging
    if not f.get("pathology_confirmed", False): 
        return "decision"
    
    loc = f.get("tumor_location", "unknown")
    if loc == "unknown":
        # 无法确定位置，尝试从文本推断
        user_text = _latest_user_text(state)
        if "直肠" in user_text or "直肠癌" in user_text:
            return "rectal_staging"
        elif "结肠" in user_text or "结肠癌" in user_text:
            return "colon_staging"
        else:
            return "decision"
    
    # 正常路由到对应的 Staging 节点
    if loc == "rectum": 
        return "rectal_staging"
    if loc == "colon": 
        return "colon_staging"
    return "decision"
