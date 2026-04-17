from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Mapping

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from ..state import CRCAgentState, PatientProfile
from .node_utils import (
    _build_pinned_context,
    _build_profile_change_entry,
    _build_summary_memory,
    _is_postop_context,
    _latest_user_text,
    auto_update_roadmap_from_state,
)

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
FAST_PASS_INTENTS = {"treatment_decision", "clinical_assessment", "search_treatment_recommendations", "multi_task"}


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


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, "", "Unknown"):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    match = re.search(r"[+-]?\d+", text)
    return int(match.group(0)) if match else None


def _format_missing_info_card(title: str, detail: str) -> str:
    return f"⚠️ 注意：本次回复未生成结构化卡片，以下为文本详情:\n\n📋 {title}\n{detail}"


def _create_inquiry_message(category: str, detail: str) -> str:
    return _format_missing_info_card(category, detail)


def _prepend_crc_entry_explanation(state: CRCAgentState, inquiry_message: str) -> tuple[str, bool]:
    findings = state.findings or {}
    encounter_track = state.encounter_track or findings.get("encounter_track")
    explanation_shown = bool(state.entry_explanation_shown or findings.get("entry_explanation_shown", False))
    clinical_entry_reason = state.clinical_entry_reason or findings.get("clinical_entry_reason")
    if encounter_track != "crc_clinical" or explanation_shown:
        return inquiry_message, explanation_shown
    if clinical_entry_reason not in {"system_known_crc_signal", "user_provided_abnormal_test"}:
        return inquiry_message, explanation_shown
    explanation = "系统检测到您提到了与结直肠癌评估直接相关的异常检查或明确信号，因此先进入临床评估流程。"
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


def _flatten_tnm_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {"cT": "", "cN": "", "cM": ""}
    clinical = value.get("clinical") if isinstance(value.get("clinical"), Mapping) else {}
    flattened = {
        "cT": str(value.get("cT") or clinical.get("cT") or "").strip(),
        "cN": str(value.get("cN") or clinical.get("cN") or "").strip(),
        "cM": str(value.get("cM") or clinical.get("cM") or "").strip(),
    }
    return flattened


def _tnm_status_from_tokens(tokens: Mapping[str, str]) -> Literal["Complete", "Partial", "Missing"]:
    has_t = bool(tokens.get("cT"))
    has_n = bool(tokens.get("cN"))
    has_m = bool(tokens.get("cM"))
    if has_t and has_n and has_m:
        return "Complete"
    if has_t or has_n or has_m:
        return "Partial"
    return "Missing"


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


def _normalize_location_token(value: Any) -> Literal["Rectum", "Colon", "Unknown"]:
    text = str(value or "").strip().lower()
    if not text:
        return "Unknown"
    if text in {"rectum", "rectal"} or any(marker in text for marker in ["直肠", "距肛"]):
        return "Rectum"
    if text in {"colon", "sigmoid"} or any(marker in text for marker in ["结肠", "乙状结肠", "盲肠", "升结肠", "横结肠", "降结肠"]):
        return "Colon"
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


def _is_advanced_stage_from_tnm(tokens: Mapping[str, str], stage_summary: str = "") -> bool:
    cN = str(tokens.get("cN") or "").upper()
    cM = str(tokens.get("cM") or "").upper()
    summary = str(stage_summary or "").upper()
    return any(
        (
            "N1" in cN,
            "N2" in cN,
            "M1" in cM,
            "III" in summary,
            "IV" in summary,
            "Ⅲ" in summary,
            "Ⅳ" in summary,
        )
    )


def _quick_extract_histology(text: str) -> str:
    lowered = str(text or "").lower()
    raw = str(text or "")
    if "adenocarcinoma" in lowered or "腺癌" in raw:
        return "腺癌"
    if "病理" in raw or "活检" in raw:
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
    return "".join(filter(None, [tokens.get("cT", ""), tokens.get("cN", ""), tokens.get("cM", "")]))


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


def _normalize_mmr_status(mmr_status: Any, markers: Any) -> str:
    text = str(mmr_status or "").strip()
    if text and text.lower() not in {"unknown", "none", "null"}:
        return text
    marker_map = markers if isinstance(markers, Mapping) else {}
    if marker_map.get("dMMR") or marker_map.get("MSI-H") or str(marker_map.get("MMR", "")).lower() == "dmmr":
        return "dMMR"
    if marker_map.get("pMMR") or marker_map.get("MSS") or str(marker_map.get("MMR", "")).lower() == "pmmr":
        return "pMMR"
    return "Unknown"


def _build_database_case_summary(findings: Mapping[str, Any], tnm_tokens: Mapping[str, str]) -> str:
    location = _normalize_location_token(findings.get("tumor_location"))
    location_text = "直肠" if location == "Rectum" else "结肠" if location == "Colon" else "肿瘤部位待补充"
    histology = str(findings.get("histology_type") or "病理类型待补充").strip()
    stage_summary = str(findings.get("clinical_stage_summary") or findings.get("clinical_stage_staging") or "").strip()
    tnm_display = _format_tnm_display(dict(tnm_tokens))
    if stage_summary:
        return stage_summary
    parts = [location_text, histology]
    if tnm_display:
        parts.append(tnm_display)
    return " ".join(part for part in parts if part)


def _build_database_fast_pass(state: CRCAgentState, integrity: CaseIntegrity) -> Dict[str, Any] | None:
    findings = state.findings or {}
    if findings.get("data_source") != "database_query":
        return None
    if str(findings.get("user_intent") or "") not in FAST_PASS_INTENTS:
        return None

    pathology_confirmed = bool(findings.get("pathology_confirmed") or findings.get("biopsy_confirmed"))
    tnm_tokens = _flatten_tnm_dict(findings.get("tnm_staging"))
    location = _normalize_location_token(findings.get("tumor_location"))
    if not (pathology_confirmed and location != "Unknown" and all(tnm_tokens.values())):
        return None

    mmr_status = _normalize_mmr_status(findings.get("mmr_status"), findings.get("molecular_markers"))
    tumor_type = "Rectum" if location == "Rectum" else "Colon"
    new_profile = PatientProfile(
        tumor_type=tumor_type,
        pathology_confirmed=True,
        tnm_staging=dict(tnm_tokens),
        mmr_status=mmr_status,
        is_locked=True,
        age=_coerce_optional_int(findings.get("age")),
        gender=str(findings.get("gender") or "").strip() or None,
        ecog_score=_coerce_optional_int(findings.get("ecog_score")),
    )
    is_advanced = _is_advanced_stage_from_tnm(tnm_tokens, str(findings.get("clinical_stage_summary") or "")) or integrity.is_advanced_stage
    risk_level = "High" if is_advanced else "Moderate"
    case_summary = _build_database_case_summary(findings, tnm_tokens)
    patient_id = str(findings.get("db_query_patient_id") or "").strip()
    assessment_text = f"已读取数据库病例{f'（ID: {patient_id}）' if patient_id else ''}的完整核心临床信息，可直接进入后续治疗决策：{case_summary}。"
    updated_findings = {
        **findings,
        "risk_level": risk_level,
        "assessment_draft": assessment_text,
        "fast_pass_mode": True,
        "is_advanced_stage": is_advanced,
        "pathology_confirmed": True,
        "biopsy_confirmed": True,
        "tumor_location": location.lower(),
        "tnm_staging": {**(findings.get("tnm_staging") or {}), **tnm_tokens, "clinical": dict(tnm_tokens)},
        "mmr_status": mmr_status,
        "patient_profile_locked": True,
        "semantic_integrity": integrity.model_dump(),
    }
    profile_entry = _build_profile_change_entry(state.patient_profile, new_profile, source="assessment_fast_pass_db")
    updates: Dict[str, Any] = {
        "patient_profile": new_profile,
        "findings": updated_findings,
        "assessment_draft": json.dumps(
            {
                "risk": risk_level,
                "summary": case_summary,
                "reasoning": f"数据库病例核心信息完整，直接恢复 fast-pass 决策链路{f'（ID: {patient_id}）' if patient_id else ''}。",
            },
            ensure_ascii=False,
        ),
        "missing_critical_data": [],
        "clinical_stage": "Assessment_Completed",
        "error": None,
        "pathology_confirmed": True,
        "histology_type": str(findings.get("histology_type") or "Unknown"),
        "roadmap": auto_update_roadmap_from_state(state),
    }
    if profile_entry:
        updates["patient_profile_timeline"] = [profile_entry]
    return updates


def check_case_integrity(user_text: str, model=None, pinned_context: str = "", summary_memory: str = "") -> CaseIntegrity:
    del model
    full_text = "\n".join([summary_memory or "", pinned_context or "", user_text or ""])
    lowered = full_text.lower()
    tnm_tokens = _extract_tnm_tokens_from_text(full_text)
    if not tnm_tokens.get("cM") and any(marker in full_text for marker in NO_METASTASIS_MARKERS):
        tnm_tokens["cM"] = "cM0"
    tnm_status = _tnm_status_from_tokens(tnm_tokens)
    is_advanced = _is_advanced_stage_from_tnm(tnm_tokens, full_text)
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
    del model, tools, streaming, show_thinking

    def _run(state: CRCAgentState) -> Dict[str, Any]:
        user_text = _latest_user_text(state)
        integrity = check_case_integrity(
            user_text,
            pinned_context=_build_pinned_context(state),
            summary_memory=_build_summary_memory(state),
        )
        findings = state.findings or {}
        encounter_track = state.encounter_track or findings.get("encounter_track")

        database_fast_pass = _build_database_fast_pass(state, integrity)
        if database_fast_pass is not None:
            return database_fast_pass

        if integrity.is_symptom_only and encounter_track != "crc_clinical":
            inquiry_message = (
                "为了先判断您目前更像哪类胃肠道问题，我需要先补充症状信息。\n"
                "请直接说明主要不适是什么，以及大概持续多久，例如“腹痛3天”“有便血”“腹泻2周”。"
            )
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Symptom Details"],
                "findings": {
                    **findings,
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
                    **findings,
                    "risk_level": "Average",
                    "inquiry_type": "symptom_inquiry",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        effective_pathology_confirmed = bool(findings.get("pathology_confirmed") or findings.get("biopsy_confirmed") or integrity.has_confirmed_diagnosis)
        effective_tnm_tokens = _flatten_tnm_dict(findings.get("tnm_staging"))
        if not any(effective_tnm_tokens.values()):
            effective_tnm_tokens = _extract_tnm_tokens_from_text("\n".join([_build_summary_memory(state), _build_pinned_context(state), user_text]))
            if not effective_tnm_tokens.get("cM") and any(marker in user_text for marker in NO_METASTASIS_MARKERS):
                effective_tnm_tokens["cM"] = "cM0"
        effective_tnm_status = _tnm_status_from_tokens(effective_tnm_tokens)
        effective_location = _normalize_location_token(findings.get("tumor_location"))
        if effective_location == "Unknown":
            effective_location = integrity.tumor_location_category
        effective_mmr_status = _normalize_mmr_status(findings.get("mmr_status"), findings.get("molecular_markers"))
        effective_mmr_availability = "Provided" if effective_mmr_status != "Unknown" or integrity.mmr_status_availability == "Provided" else "Not_Provided"
        effective_is_advanced = _is_advanced_stage_from_tnm(effective_tnm_tokens, str(findings.get("clinical_stage_summary") or "")) or integrity.is_advanced_stage

        if not effective_pathology_confirmed:
            inquiry_message = _create_inquiry_message(
                "病理确诊信息缺失提醒",
                "请提供病理确诊信息（如活检或术后病理报告），以便进行准确的分期和治疗方案制定。",
            )
            inquiry_message, entry_explanation_shown = _prepend_crc_entry_explanation(state, inquiry_message)
            updates = {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Pathology Report"],
                "findings": {
                    **findings,
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

        if effective_tnm_status != "Complete":
            inquiry_message = _create_inquiry_message("TNM 分期信息缺失提醒", "请补充 cT/cN/cM 或明确的影像分期信息。")
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["TNM Staging"],
                "findings": {
                    **findings,
                    "risk_level": "Average",
                    "inquiry_type": "tnm_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        if effective_location == "Unknown":
            inquiry_message = _create_inquiry_message("肿瘤部位信息缺失提醒", "请说明病灶位于直肠还是结肠。")
            return {
                "clinical_stage": "Pending",
                "missing_critical_data": ["Tumor Location"],
                "findings": {
                    **findings,
                    "risk_level": "Average",
                    "inquiry_type": "location_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        if effective_is_advanced and effective_mmr_availability == "Not_Provided":
            inquiry_message = _create_inquiry_message("MMR/MSI 状态缺失提醒", "如果已经做过 IHC 或 MSI 检测，请直接告诉我结果；如果暂时没有，也可以先继续后续评估。")
            return {
                "clinical_stage": "Inquiry_Pending",
                "missing_critical_data": ["MMR/MSI Status"],
                "findings": {
                    **findings,
                    "risk_level": "High",
                    "inquiry_type": "mmr_status_required",
                    "inquiry_message": inquiry_message,
                    "active_inquiry": True,
                    "semantic_integrity": integrity.model_dump(),
                },
                "messages": [AIMessage(content=inquiry_message)],
                "roadmap": auto_update_roadmap_from_state(state),
            }

        risk_level = "High" if effective_is_advanced else "Moderate"
        summary = f"已具备基础临床评估信息，可进入后续诊断与决策流程（部位：{effective_location}，TNM：{effective_tnm_status}）。"
        return {
            "findings": {
                **findings,
                "risk_level": risk_level,
                "assessment_draft": summary,
                "semantic_integrity": integrity.model_dump(),
                "is_advanced_stage": effective_is_advanced,
                "pathology_confirmed": effective_pathology_confirmed,
                "biopsy_confirmed": bool(findings.get("biopsy_confirmed") or effective_pathology_confirmed),
                "tumor_location": effective_location.lower(),
                "tnm_staging": {**(findings.get("tnm_staging") or {}), **effective_tnm_tokens, "clinical": dict(effective_tnm_tokens)},
                "mmr_status": effective_mmr_status,
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
                "clinical_stage_staging": result.clinical_stage_summary,
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
    if findings.get("fast_pass_mode"):
        return "decision"
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
]
