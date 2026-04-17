from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from ..state import CRCAgentState
from .node_utils import _build_pinned_context, _build_summary_memory, _is_postop_context, _latest_user_text, auto_update_roadmap_from_state


CRC_DIAGNOSIS_MARKERS = (
    "结直肠癌",
    "直肠癌",
    "结肠癌",
    "病理",
    "活检",
    "腺癌",
    "肿瘤",
    "癌",
    "adenocarcinoma",
)
GI_SYMPTOM_MARKERS = (
    "腹痛",
    "肚子痛",
    "隐痛",
    "便血",
    "黑便",
    "腹泻",
    "便秘",
    "排便变化",
    "大便变细",
    "消瘦",
    "发热",
    "呕吐",
    "不舒服",
    "不适",
    "难受",
)
GI_VAGUE_MARKERS = (
    r"(?:肠胃|胃肠|胃|腹部|肚子).{0,4}(?:不舒服|不适|难受)",
)
NON_GI_VAGUE_MARKERS = (
    r"(?:浑身|全身|哪里都|整个人).{0,4}(?:不舒服|不适|难受)",
)
COLON_MARKERS = ("结肠", "乙状结肠", "盲肠", "升结肠", "横结肠", "降结肠", "colon", "sigmoid")
RECTUM_MARKERS = ("直肠", "距肛", "rectum", "rectal")
NO_METASTASIS_MARKERS = ("无远处转移", "未见远处转移", "no distant metastasis", "without distant metastasis")


class ClinicalAssessmentResult(BaseModel):
    risk_level: str = Field(default="Average")
    red_flags: List[str] = Field(default_factory=list)
    missing_critical_data: List[str] = Field(default_factory=list)
    assessment_summary: str = Field(default="")
    reasoning: str = Field(default="")

    @field_validator("risk_level")
    @classmethod
    def _normalize_risk_level(cls, value: str) -> str:
        text = str(value or "Average").strip().lower()
        if text == "high":
            return "High"
        if text == "moderate":
            return "Moderate"
        return "Average"


class DiagnosisExtractionResult(BaseModel):
    pathology_confirmed: bool = False
    tumor_location: Literal["Rectum", "Colon", "Unknown"] = "Unknown"
    tumor_subsite: str = ""
    histology_type: str = "Unknown"
    molecular_markers: Dict[str, Any] = Field(default_factory=dict)
    rectal_mri_params: Dict[str, Any] = Field(default_factory=dict)
    tnm_staging: Dict[str, Any] = Field(default_factory=dict)
    clinical_stage_summary: str = ""

    @property
    def derived_mmr_status(self) -> str:
        markers = self.molecular_markers or {}
        if markers.get("dMMR") or markers.get("MSI-H") or str(markers.get("MMR", "")).lower() == "dmmr":
            return "dMMR"
        if markers.get("pMMR") or markers.get("MSS") or str(markers.get("MMR", "")).lower() == "pmmr":
            return "pMMR"
        return "Unknown"

    @property
    def derived_kras_status(self) -> str:
        value = (self.molecular_markers or {}).get("KRAS") or (self.molecular_markers or {}).get("RAS")
        return str(value) if value else "Unknown"

    @property
    def derived_braf_status(self) -> str:
        value = (self.molecular_markers or {}).get("BRAF")
        return str(value) if value else "Unknown"


class CaseIntegrity(BaseModel):
    has_confirmed_diagnosis: bool = False
    tumor_location_category: Literal["Rectum", "Colon", "Unknown"] = "Unknown"
    tnm_status: Literal["Complete", "Partial", "Missing"] = "Missing"
    is_advanced_stage: bool = False
    mmr_status_availability: Literal["Provided", "Not_Provided"] = "Not_Provided"
    is_symptom_only: bool = False
    reasoning: str = ""


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _format_missing_info_card(title: str, detail: str) -> str:
    return f"⚠️ 注意：本次回复未生成结构化卡片，以下为文本详情:\n\n📋 {title}\n{detail}"


def _create_inquiry_message(category: str, detail: str) -> str:
    return _format_missing_info_card(f"{category}", detail)


def _prepend_crc_entry_explanation(state: CRCAgentState, inquiry_message: str) -> tuple[str, bool]:
    findings = state.findings or {}
    encounter_track = state.encounter_track or findings.get("encounter_track")
    explanation_shown = bool(state.entry_explanation_shown or findings.get("entry_explanation_shown", False))
    clinical_entry_reason = state.clinical_entry_reason or findings.get("clinical_entry_reason")
    if encounter_track != "crc_clinical" or explanation_shown:
        return inquiry_message, explanation_shown
    if clinical_entry_reason not in {"system_known_crc_signal", "user_provided_abnormal_test"}:
        return inquiry_message, explanation_shown
    explanation = "系统检测到您提到了与结直肠癌评估直接相关的异常检查或明确线索，因此先进入临床评估流程。"
    return f"{explanation}\n\n{inquiry_message}", True


def _extract_tnm_tokens_from_text(text: str) -> Dict[str, str]:
    raw = str(text or "")
    compact = _compact(raw)
    combined = re.search(
        r"([cpr]?T[0-4Xx](?:is|[A-Ca-c])?)([cpr]?N[0-3Xx](?:[A-Ca-c])?)([cpr]?M[01Xx](?:[A-Ca-c])?)",
        compact,
        re.IGNORECASE,
    )
    if combined:
        return {"cT": combined.group(1), "cN": combined.group(2), "cM": combined.group(3)}

    extracted: Dict[str, str] = {}
    for key, pattern in {
        "cT": r"([cpr]?T\s*[0-4Xx](?:is|[A-Ca-c])?)",
        "cN": r"([cpr]?N\s*[0-3Xx](?:[A-Ca-c])?)",
        "cM": r"([cpr]?M\s*[01Xx](?:[A-Ca-c])?)",
    }.items():
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            extracted[key] = re.sub(r"\s+", "", match.group(1))
    return extracted


def _infer_location_category(text: str) -> Literal["Rectum", "Colon", "Unknown"]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return "Unknown"
    if any(marker.lower() in lowered for marker in COLON_MARKERS):
        return "Colon"
    if any(marker.lower() in lowered for marker in RECTUM_MARKERS):
        return "Rectum"
    distance_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", lowered)
    if distance_match and "距肛" in lowered:
        try:
            return "Rectum" if float(distance_match.group(1)) <= 15 else "Colon"
        except ValueError:
            pass
    return "Unknown"


def _infer_stage_group_from_tnm(tokens: Dict[str, str]) -> str:
    cT = str(tokens.get("cT") or "").upper()
    cN = str(tokens.get("cN") or "").upper()
    cM = str(tokens.get("cM") or "").upper()
    if "M1" in cM:
        return "IV期"
    if "N1" in cN or "N2" in cN:
        return "III期"
    if any(marker in cT for marker in ("T3", "T4")):
        return "II期"
    if any(marker in cT for marker in ("T1", "T2")) and "M0" in cM:
        return "I期"
    return ""


def _quick_extract_histology(text: str) -> str:
    lowered = str(text or "").lower()
    if "adenocarcinoma" in lowered or "腺癌" in text:
        return "腺癌"
    if "病理" in text or "活检" in text:
        return "已提示病理信息"
    return "Unknown"


def _quick_extract_molecular_markers(text: str) -> Dict[str, Any]:
    lowered = str(text or "").lower()
    markers: Dict[str, Any] = {}
    if "pmmr" in lowered or "mss" in lowered:
        markers.update({"MMR": "pMMR", "pMMR": True, "MSS": True})
    if "dmmr" in lowered or "msi-h" in lowered:
        markers.update({"MMR": "dMMR", "dMMR": True, "MSI-H": True})
    if "kras" in lowered:
        markers["KRAS"] = "mentioned"
    if "braf" in lowered:
        markers["BRAF"] = "mentioned"
    return markers


def _format_tnm_display(tokens: Dict[str, str]) -> str:
    cT = str(tokens.get("cT") or "")
    cN = str(tokens.get("cN") or "")
    cM = str(tokens.get("cM") or "")
    return "".join([cT, cN[1:] if cN.lower().startswith("c") else cN, cM[1:] if cM.lower().startswith("c") else cM])


def _quick_extract_diagnosis_from_text(full_text: str) -> DiagnosisExtractionResult:
    text = str(full_text or "")
    lowered = text.lower()
    tnm_tokens = _extract_tnm_tokens_from_text(text)
    if not tnm_tokens.get("cM") and any(marker in text for marker in NO_METASTASIS_MARKERS):
        tnm_tokens["cM"] = "cM0"
    location = _infer_location_category(text)
    histology = _quick_extract_histology(text)
    stage_group = _infer_stage_group_from_tnm(tnm_tokens)
    tnm_display = _format_tnm_display(tnm_tokens)
    parts = []
    if location != "Unknown":
        parts.append("直肠" if location == "Rectum" else "结肠")
    if histology != "Unknown":
        parts.append(histology)
    if tnm_display:
        parts.append(f"{tnm_display}（{stage_group}）" if stage_group else tnm_display)
    summary = "，".join(parts)
    return DiagnosisExtractionResult(
        pathology_confirmed=any(marker.lower() in lowered or marker in text for marker in CRC_DIAGNOSIS_MARKERS),
        tumor_location=location,
        tumor_subsite="",
        histology_type=histology,
        molecular_markers=_quick_extract_molecular_markers(text),
        rectal_mri_params={},
        tnm_staging={"clinical": dict(tnm_tokens), "stage_group": stage_group, **tnm_tokens},
        clinical_stage_summary=summary,
    )


def _looks_like_non_gi_vague_symptom(text: str) -> bool:
    compact = _compact(text)
    return any(re.search(pattern, compact) for pattern in NON_GI_VAGUE_MARKERS)


def _looks_like_gi_vague_symptom(text: str) -> bool:
    compact = _compact(text)
    return any(re.search(pattern, compact) for pattern in GI_VAGUE_MARKERS)


def _has_crc_assessment_anchor(text: str) -> bool:
    compact = _compact(text).lower()
    return any(marker.lower() in compact for marker in CRC_DIAGNOSIS_MARKERS + ("tnm", "分期", "治疗", "方案", "化疗", "放疗", "手术", "靶向", "免疫", "mmr", "msi", "cea")) or bool(re.search(r"(?:c?t[0-4x]|c?n[0-3x]|c?m[01x])", compact))


def _should_use_non_crc_symptom_clarification(text: str, integrity: CaseIntegrity) -> bool:
    return _looks_like_non_gi_vague_symptom(text) and not integrity.has_confirmed_diagnosis and not _has_crc_assessment_anchor(text)


def _create_non_crc_symptom_clarification_message() -> str:
    return (
        "这句话更像是一般身体不适描述，还不足以直接进入结直肠癌评估。\n"
        "如果主要是肠胃或腹部不适，我可以继续按门诊分诊追问；如果是浑身不舒服、全身难受这类非胃肠道症状，建议先到全科、内科或急诊线下就诊。\n"
        "您也可以直接补充更具体的情况，例如“腹痛3天”“有便血”“腹泻2周”或“我想咨询治疗方案”。"
    )


def check_case_integrity(user_text: str, model=None, pinned_context: str = "", summary_memory: str = "") -> CaseIntegrity:
    del model
    full_text = "\n".join([summary_memory or "", pinned_context or "", user_text or ""])
    lowered = full_text.lower()
    tnm_tokens = _extract_tnm_tokens_from_text(full_text)
    has_t = bool(tnm_tokens.get("cT"))
    has_n = bool(tnm_tokens.get("cN"))
    has_m = bool(tnm_tokens.get("cM")) or any(marker in full_text for marker in NO_METASTASIS_MARKERS)
    if has_t and has_n and has_m:
        tnm_status = "Complete"
    elif has_t or has_n or has_m:
        tnm_status = "Partial"
    else:
        tnm_status = "Missing"
    n_token = str(tnm_tokens.get("cN") or "")
    m_token = str(tnm_tokens.get("cM") or "")
    is_advanced = bool(re.search(r"n[1-3]", n_token, re.IGNORECASE) or re.search(r"m1", m_token, re.IGNORECASE) or any(marker in full_text for marker in ("III期", "IV期", "Ⅲ期", "Ⅳ期")))
    symptom_only = any(marker in full_text for marker in GI_SYMPTOM_MARKERS) and not any(marker.lower() in lowered or marker in full_text for marker in CRC_DIAGNOSIS_MARKERS)
    return CaseIntegrity(
        has_confirmed_diagnosis=any(marker.lower() in lowered or marker in full_text for marker in CRC_DIAGNOSIS_MARKERS),
        tumor_location_category=_infer_location_category(full_text),
        tnm_status=tnm_status,
        is_advanced_stage=is_advanced,
        mmr_status_availability="Provided" if any(marker in lowered for marker in ("pmmr", "dmmr", "mss", "msi", "mmr")) else "Not_Provided",
        is_symptom_only=symptom_only,
        reasoning="heuristic_case_integrity",
    )


def node_assessment(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    del model, tools, streaming

    def _run(state: CRCAgentState) -> Dict[str, Any]:
        user_text = _latest_user_text(state)
        integrity = check_case_integrity(
            user_text,
            pinned_context=_build_pinned_context(state),
            summary_memory=_build_summary_memory(state),
        )
        findings = state.findings or {}
        encounter_track = state.encounter_track or findings.get("encounter_track")

        if integrity.is_symptom_only and encounter_track != "crc_clinical":
            inquiry_message = (
                "为了先判断您目前更像哪类胃肠道问题，我需要先补充症状信息。\n"
                "请直接说明主要不适是什么，以及大概持续多久，例如“腹痛3天”“有便血”“腹泻2周”。"
            )
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Symptom Details"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "symptom_inquiry",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        if encounter_track == "crc_clinical" and _should_use_non_crc_symptom_clarification(user_text, integrity):
            inquiry_message = _create_non_crc_symptom_clarification_message()
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Symptom Clarification"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "symptom_inquiry",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        if not integrity.has_confirmed_diagnosis:
            inquiry_message = _create_inquiry_message(
                "病理确诊信息缺失提醒",
                "请提供病理确诊信息（如活检或术后病理报告），以便进行准确的分期和治疗方案制定。",
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
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }
            if entry_explanation_shown:
                updates["entry_explanation_shown"] = True
            return updates

        if integrity.tnm_status != "Complete":
            inquiry_message = _create_inquiry_message("TNM 分期信息缺失提醒", "请补充 cT/cN/cM 或明确的影像分期信息。")
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["TNM Staging"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "tnm_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        if integrity.tumor_location_category == "Unknown":
            inquiry_message = _create_inquiry_message("肿瘤部位信息缺失提醒", "请说明病灶位于直肠还是结肠。")
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Tumor Location"],
                "findings": {
                    "risk_level": "Average",
                    "inquiry_type": "location_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        if integrity.is_advanced_stage and integrity.mmr_status_availability == "Not_Provided":
            inquiry_message = _create_inquiry_message("MMR/MSI 状态缺失提醒", "如果已经做过 IHC 或 MSI 检测，请直接告诉我结果；如果暂时没有，也可以先继续后续评估。")
            return {
                "clinical_stage": "Inquiry_Pending",
                "missing_critical_data": ["MMR/MSI Status"],
                "findings": {
                    "risk_level": "High",
                    "inquiry_type": "mmr_status_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        risk_level = "High" if integrity.is_advanced_stage else "Moderate"
        summary = f"已具备基础临床评估信息，可进入后续诊断与决策流程（部位：{integrity.tumor_location_category}，TNM：{integrity.tnm_status}）。"
        return {
            "findings": {
                **findings,
                "risk_level": risk_level,
                "assessment_draft": summary,
                "semantic_integrity": integrity.model_dump(),
                "is_advanced_stage": integrity.is_advanced_stage,
            },
            "assessment_draft": json.dumps({"risk": risk_level, "summary": summary, "reasoning": integrity.reasoning}, ensure_ascii=False),
            "missing_critical_data": [],
            "clinical_stage": "Assessment_Completed",
            "error": None,
            "roadmap": auto_update_roadmap_from_state(state),
        }

    return _run


def node_diagnosis(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    del model, tools, streaming, show_thinking

    def _run(state: CRCAgentState) -> Dict[str, Any]:
        user_text = _latest_user_text(state)
        result = _quick_extract_diagnosis_from_text("\n".join([_build_summary_memory(state), _build_pinned_context(state), user_text]))
        updates: Dict[str, Any] = {
            "findings": {
                **(state.findings or {}),
                "pathology_confirmed": result.pathology_confirmed,
                "biopsy_confirmed": result.pathology_confirmed,
                "tumor_location": result.tumor_location.lower(),
                "tumor_subsite": result.tumor_subsite,
                "histology_type": result.histology_type,
                "molecular_markers": result.molecular_markers,
                "tnm_staging": result.tnm_staging,
                "clinical_stage_summary": result.clinical_stage_summary,
                "mmr_status": result.derived_mmr_status,
                "kras_status": result.derived_kras_status,
                "braf_status": result.derived_braf_status,
            },
            "clinical_stage": "Diagnosis",
            "error": None,
            "pathology_confirmed": result.pathology_confirmed,
            "histology_type": result.histology_type,
        }
        if not result.pathology_confirmed and not _is_postop_context(user_text):
            inquiry_message = _create_inquiry_message(
                "病理确诊信息缺失提醒",
                "请提供病理确诊信息（如活检或术后病理报告），以便进行准确的分期和治疗方案制定。",
            )
            inquiry_message, entry_explanation_shown = _prepend_crc_entry_explanation(state, inquiry_message)
            updates["clinical_stage"] = "Diagnosis_Pending"
            updates["missing_critical_data"] = ["Pathology Report"]
            updates["findings"]["inquiry_type"] = "pathology_required"
            updates["findings"]["inquiry_message"] = inquiry_message
            updates["findings"]["active_inquiry"] = True
            updates["findings"]["entry_explanation_shown"] = entry_explanation_shown
            updates["messages"] = [AIMessage(content=inquiry_message)]
            if entry_explanation_shown:
                updates["entry_explanation_shown"] = True
        temp_state = state.model_copy(update=updates)
        updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
        return updates

    return _run


def node_staging_router(state: CRCAgentState):
    findings = state.findings or {}
    if state.clinical_stage == "Diagnosis_Pending" or "Pathology Report" in (state.missing_critical_data or []):
        return "end_turn"
    if not findings.get("pathology_confirmed", False):
        return "decision"
    tumor_location = findings.get("tumor_location", "unknown")
    if tumor_location == "rectum":
        return "rectal_staging"
    if tumor_location == "colon":
        return "colon_staging"
    return "decision"


__all__ = [
    "ClinicalAssessmentResult",
    "DiagnosisExtractionResult",
    "CaseIntegrity",
    "check_case_integrity",
    "node_assessment",
    "node_diagnosis",
    "node_staging_router",
    "_create_inquiry_message",
    "_prepend_crc_entry_explanation",
    "_extract_tnm_tokens_from_text",
    "_infer_location_category",
]
