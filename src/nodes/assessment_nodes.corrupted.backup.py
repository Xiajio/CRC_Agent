""" Assessment and Diagnosis Nodes
(Final Fix v3: Fast Pass 绛栫暐 - 浠?璺宠繃鑺傜偣"鏀逛负"杞婚噺鍖栭€氳繃"锛岀‘淇濇暟鎹祦姘寸嚎瀹屾暣)
(v4: 寮曞叆 PatientProfile (鍙娌欑) 姒傚康 + 涓诲姩杩介棶鏈哄埗)
(v5: JSON 瑙ｆ瀽澧炲己 + 瀛楁楠岃瘉)
(v6: 淇 PatientProfile 閲嶅瀹氫箟闂锛屼娇鐢?state.py 涓殑瀹氫箟)
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
        "??",
        "????",
        "???",
        "???",
        "???",
        "??",
        "sigmoid",
        "colon",
        "colonic",
    ]
    rectal_markers = ["??", "rectum", "rectal"]

    if any(marker in lowered for marker in colon_markers):
        return "Colon"

    distance_match = re.search(r"璺濊倹(?:闂▅闁€|缂榺绶??\s*(\d+(?:\.\d+)?)\s*cm", lowered)
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
            "娲绘",
            "鐥呯悊",
            "鑵虹檶",
        ]
    )

    tnm_tokens = _extract_tnm_tokens_from_text(text)
    has_t = bool(tnm_tokens.get("cT"))
    has_n = bool(tnm_tokens.get("cN"))
    has_m = bool(tnm_tokens.get("cM")) or any(
        marker in lowered
        for marker in [
            "?????",
            "??????",
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
        marker in lowered for marker in ["pmmr", "dmmr", "mss", "msi", "mmr", "閿欓厤淇"]
    ) else "Not_Provided"

    n_token = str(tnm_tokens.get("cN", ""))
    m_token = str(tnm_tokens.get("cM", ""))
    is_advanced_stage = bool(
        re.search(r"n[1-3]", n_token, re.IGNORECASE)
        or re.search(r"m1", m_token, re.IGNORECASE)
        or re.search(r"\biii鏈焅b|\biv鏈焅b", text, re.IGNORECASE)
    )
    symptom_hits = any(
        marker in lowered
        for marker in ["渚胯", "鑵圭棝", "浣撻噸涓嬮檷", "鎺掍究", "symptom", "constipation", "diarrhea"]
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
        return "IV?"
    if "N1" in cN or "N2" in cN:
        return "III?"
    if any(marker in cT for marker in ["T3", "T4"]):
        return "II?"
    if any(marker in cT for marker in ["T1", "T2"]) and "M0" in cM:
        return "I?"
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
        ("????", ["????", "sigmoid"]),
        ("???", ["???", "descending colon"]),
        ("???", ["???", "transverse colon"]),
        ("???", ["???", "ascending colon"]),
        ("??", ["??", "cecum", "caecum"]),
        ("??", ["??", "rectum", "rectal"]),
    ]
    for label, markers in subsite_markers:
        if any(marker in lowered for marker in markers):
            return label
    return ""


def _quick_extract_histology(text: str) -> str:
    raw = str(text or "")
    match = re.search(r"(涓垎鍖栬吅鐧寍楂樺垎鍖栬吅鐧寍浣庡垎鍖栬吅鐧寍鑵虹檶|adenocarcinoma)", raw, re.IGNORECASE)
    if match:
        value = match.group(1)
        if value.lower() == "adenocarcinoma":
            return "鑵虹檶"
        return value
    return "鏈煡"


def _quick_extract_diagnosis_from_text(full_text: str) -> "DiagnosisExtractionResult":
    text = str(full_text or "")
    lowered = text.lower()
    tnm_tokens = _normalize_clinical_tnm_tokens(_extract_tnm_tokens_from_text(text))

    if not tnm_tokens.get("cM"):
        if any(
            marker in lowered
            for marker in ["?????", "??????", "no distant metastasis", "without distant metastasis"]
        ):
            tnm_tokens["cM"] = "cM0"
            tnm_tokens["cM"] = "cM0"

    location = _infer_location_category(text)
    subsite = _infer_tumor_subsite(text)
    markers = _quick_extract_molecular_markers(text)
    pathology_confirmed = any(
        marker in lowered for marker in ["娲绘", "鐥呯悊", "biopsy", "pathology", "adenocarcinoma", "鑵虹檶"]
    )
    histology = _quick_extract_histology(text)
    stage_group = _infer_stage_group_from_tnm(tnm_tokens)

    stage_summary_parts = []
    location_text = subsite or ("缁撹偁" if location == "Colon" else "鐩磋偁" if location == "Rectum" else "")
    pathology_text = "".join([part for part in [location_text, histology if histology != "鏈煡" else ""] if part])
    if pathology_text:
        stage_summary_parts.append(pathology_text)
    if histology != "鏈煡":
        pass
    tnm_compact = _format_tnm_display(tnm_tokens)
    if tnm_compact:
    if tnm_compact:
        if stage_group:
            stage_summary_parts.append(f"{tnm_compact}?{stage_group}?")
        else:
            stage_summary_parts.append(tnm_compact)
    clinical_stage_summary = "?".join([part for part in stage_summary_parts if part])
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

# 鍒犻櫎鏃х殑 PatientProfile 瀹氫箟锛岀洿鎺ヤ娇鐢?state.py 涓殑鐗堟湰
# 鏃х殑 PatientProfile 瀹氫箟宸插垹闄わ紝閬垮厤涓?state.py 涓殑瀹氫箟鍐茬獊


# ==============================================================================
# 2. 瀹芥澗鐨?Schema (鍏ㄩ儴 String)
# ==============================================================================

class ClinicalAssessmentResult(BaseModel):
    """涓村簥璇勪及缁撴灉"""
    risk_level: str = Field(description="椋庨櫓绛夌骇 (High/Moderate/Average)")
    red_flags: List[str] = Field(description="楂樺嵄鐥囩姸", default_factory=list)
    missing_critical_data: List[str] = Field(description="缂哄け鏁版嵁", default_factory=list)
    assessment_summary: str = Field(description="鐥呮儏鎬荤粨")
    reasoning: str = Field(description="鍒嗘瀽杩囩▼")
    
    @field_validator('risk_level')
    @classmethod
    def validate_risk_level(cls, v):
        """Normalize risk level labels."""
        valid_levels = {"High", "Moderate", "Average"}
        if v not in valid_levels:
            v_normalized = v.strip()
            if v_normalized.lower() == "high":
                return "High"
            elif v_normalized.lower() == "moderate":
                return "Moderate"
            elif v_normalized.lower() == "average":
                return "Average"
            print(f"[Validation] Unexpected risk level {v!r}, defaulting to Average")
            return "Average"
        return v
    
    @field_validator("red_flags")
    @classmethod
    def validate_red_flags(cls, v):
        """Clean red flag strings for downstream serialization."""
        cleaned = []
        for flag in v:
            cleaned_flag = flag.replace(""", "\"").replace("\", "").strip()
            if cleaned_flag and len(cleaned_flag) < 100:
                cleaned.append(cleaned_flag)
        return cleaned

class DiagnosisExtractionResult(BaseModel):
    """璇婃柇缁撴灉"""
    # [淇] 娣诲姞榛樿鍊硷紝閬垮厤 Pydantic 楠岃瘉閿欒
    pathology_confirmed: bool = Field(False, description="鏄惁鐥呯悊纭瘖")
    tumor_location: Literal["Rectum", "Colon", "Unknown"] = Field("Unknown", description="鑲跨槫閮ㄤ綅 (Rectum/Colon/Unknown)")
    tumor_subsite: str = Field("", description="鍘熷彂鐏朵簹閮ㄤ綅")
    histology_type: str = Field("鏈煡", description="鐥呯悊绫诲瀷")
    molecular_markers: Dict[str, Any] = Field(default_factory=dict, description="鍒嗗瓙鏍囧織鐗?)
    rectal_mri_params: Dict[str, Any] = Field(default_factory=dict, description="MRI鍙傛暟")
    # [鏂板] TNM鍒嗘湡淇℃伅
    tnm_staging: Dict[str, Any] = Field(default_factory=dict, description="TNM鍒嗘湡")
    clinical_stage_summary: str = Field("", description="涓村簥鍒嗘湡鎬荤粨")

    @field_validator("tumor_location", mode="before")
    @classmethod
    def normalize_tumor_location(cls, v):
        """灏嗗悇绉嶄綅缃〃杩拌鑼冨寲涓?Rectum/Colon/Unknown銆?""
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
        if "rect" in s or "鐩? in s:
            return "Rectum"
        if "colon" in s or "缁? in s:
            return "Colon"
        if s in {"unknown", "n/a", "na", "?"}:
            return "Unknown"
        # 闈炴硶鍊硷細鍖荤枟鍦烘櫙瀹佸彲 Unknown锛屼笉瑕佺寽
        return "Unknown"

    @property
    def derived_mmr_status(self) -> str:
        """鍩轰簬鍒嗗瓙鏍囧織鐗╂帹瀵?MMR 鐘舵€侊紙鏇夸唬 _extract_mmr_status锛夈€?""
        markers = self.molecular_markers or {}
        # 鍏佽澶氱閿悕/鍊煎舰鎬?        if markers.get("MSI-H") or markers.get("dMMR") or str(markers.get("MSI", "")).upper() == "H":
            return "dMMR"
        if markers.get("MSS") or markers.get("pMMR"):
            return "pMMR"
        # 鏈変簺鏁版嵁婧愪細鐩存帴缁欏嚭 mmr_status
        mmr = markers.get("MMR") or markers.get("mmr_status")
        if isinstance(mmr, str):
            if "dmmr" in mmr.lower() or "msi-h" in mmr.lower():
                return "dMMR"
            if "pmmr" in mmr.lower() or "mss" in mmr.lower():
                return "pMMR"
        return "Unknown"

    @property
    def derived_kras_status(self) -> str:
        """鍩轰簬鍒嗗瓙鏍囧織鐗╂帹瀵?RAS/KRAS 鐘舵€侊紙鏇夸唬 _extract_kras_status锛夈€?""
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
        """鍩轰簬鍒嗗瓙鏍囧織鐗╂帹瀵?BRAF 鐘舵€侊紙鏇夸唬 _extract_braf_status锛夈€?""
        markers = self.molecular_markers or {}
        val = markers.get("BRAF")
        if isinstance(val, bool):
            return "Unknown"
        if val is None:
            return "Unknown"
        s = str(val).strip()
        return s if s in {"WildType", "Mutant"} else "Unknown"


# ==============================================================================
# 3.1 Semantic Guard: CaseIntegrity 璇箟瀹屾暣鎬фā鍨?# ==============================================================================

class CaseIntegrity(BaseModel):
    """
    [Semantic Guard] 璇箟灞傞潰鐨勭梾渚嬪畬鏁存€ф鏌ョ粨鏋?    鏇夸唬鍘熸潵鐨勬鍒欏尮閰嶉€昏緫锛屼娇鐢?LLM 杩涜璇箟鍒ゆ柇
    """
    # 鍩虹涓夎绱犳鏌?    has_confirmed_diagnosis: bool = Field(
        description="鐢ㄦ埛鏄惁鏄庣‘鎻愬強宸茬‘璇婏紙鍚梾鐞嗐€佹椿妫€鎴栨槑纭殑'纭瘖涓?..'鎻忚堪锛?
    )
    tumor_location_category: Literal["Rectum", "Colon", "Unknown"] = Field(
        description="鏍规嵁瑙ｅ墫浣嶇疆褰掔被銆備緥濡傦細'鐩茶偁'->Colon, '涔欑姸缁撹偁'->Colon, '璺濊倹5cm'->Rectum"
    )
    tnm_status: Literal["Complete", "Partial", "Missing"] = Field(
        description="TNM鍒嗘湡瀹屾暣鎬с€侰omplete=鍚玊/N/M淇℃伅(鍖呮嫭'鏃犺浆绉?绛夋弿杩?; Partial=閮ㄥ垎缂哄け; Missing=鍏ㄦ棤"
    )

    # 杩涢樁锛氶闄╀笌鏁版嵁缂哄け閫昏緫
    is_advanced_stage: bool = Field(
        description="鍩轰簬TNM鍒ゆ柇鏄惁涓鸿繘灞曟湡(III/IV鏈?銆備緥濡?N>=1 鎴?M>=1銆傚鏋?鏈杞Щ'鍒欎负False銆?
    )

    # 瑙ｅ喅姝诲惊鐜殑鍏抽敭锛氱敤鎴峰缂哄け鏁版嵁鐨勬€佸害
    mmr_status_availability: Literal["Provided", "Not_Provided", "User_Refused_Or_Unknown"] = Field(
        description="鐢ㄦ埛鍏充簬MMR/MSI鐘舵€佺殑鍙嶉銆侾rovided=宸叉彁渚? Not_Provided=鏈彁鍙? User_Refused_Or_Unknown=鐢ㄦ埛鏄庣‘璇?娌″仛'銆?涓嶇煡閬?鎴栨嫆缁濇彁渚涖€?
    )

    # 鐥囩姸鎻忚堪璇嗗埆锛氬垽鏂敤鎴锋槸鍚︿粎鎻忚堪鐥囩姸鑰屾湭纭瘖
    is_symptom_only: bool = Field(
        description="鐢ㄦ埛杈撳叆鏄惁浠呭寘鍚棁鐘舵弿杩帮紙濡備究琛€銆佽吂鐥涳級鑰屾湭鎻愬強纭瘖淇℃伅銆傜敤浜庡尯鍒嗙棁鐘跺挩璇㈠拰宸茬‘璇婃偅鑰?
    )

    reasoning: str = Field(description="绠€鐭殑鍒ゆ柇渚濇嵁")

    @field_validator("tumor_location_category", mode="before")
    @classmethod
    def normalize_location_category(cls, v):
        if v is None:
            return "Unknown"
        inferred = _infer_location_category(v)
        if inferred != "Unknown":
            return inferred
        s = str(v).strip().lower()
        if "rect" in s or "鐩? in s:
            return "Rectum"
        if "colon" in s or "缁? in s:
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
        if "refus" in s or "unknown" in s or "涓嶇煡閬? in s or "娌″仛" in s:
            return "User_Refused_Or_Unknown"
        return "Not_Provided"


# ==============================================================================
# 3.1.1 淇℃伅缂哄け鍗＄墖鏍煎紡鍖栧伐鍏?# ==============================================================================

def _format_missing_info_card(title: str, content: str) -> str:
    """
    鏍煎紡鍖栦俊鎭己澶辨椂鐨勫崱鐗囧睍绀?
    Args:
        title: 鍗＄墖鏍囬锛堝"璇婄枟鏂规璇︽儏"锛?        content: 闇€瑕佽ˉ鍏呯殑鍏蜂綋鍐呭

    Returns:
        鏍煎紡鍖栧悗鐨勫崱鐗囨枃鏈?    """
    return (
        "鈿狅笍 娉ㄦ剰锛氭湰娆″洖澶嶆湭鐢熸垚缁撴瀯鍖栧崱鐗囷紝浠ヤ笅涓烘枃鏈鎯咃細\n\n"
        f"馃摑 {title}\n"
        f"{content}"
    )


def _create_inquiry_message(category: str, detail: str) -> str:
    """
    鍒涘缓杩介棶娑堟伅锛堝甫鍗＄墖鏍煎紡锛?
    Args:
        category: 缂哄け淇℃伅绫诲埆
        detail: 璇︾粏淇℃伅

    Returns:
        鏍煎紡鍖栫殑杩介棶娑堟伅
    """
    return _format_missing_info_card(
        f"{category}缂哄け鎻愰啋",
        detail
    )



def _looks_like_non_gi_vague_symptom(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return False
    return bool(
        re.search(r"(?:浑身|全身|哪里都|整个人).{0,4}(?:不舒服|不适|难受)", compact)
    )


def _has_crc_assessment_anchor(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or "")).lower()
    if not compact:
        return False

    if any(
        keyword in compact
        for keyword in (
            "结直肠癌",
            "直肠癌",
            "结肠癌",
            "肿瘤",
            "癌",
            "病理",
            "活检",
            "分期",
            "tnm",
            "术后",
            "治疗",
            "方案",
            "化疗",
            "放疗",
            "手术",
            "靶向",
            "免疫",
            "cea",
            "mmr",
            "msi",
        )
    ):
        return True

    return bool(re.search(r"(?:c?t[0-4x]|c?n[0-3x]|c?m[01x])", compact))


def _should_use_non_crc_symptom_clarification(text: str, integrity: "CaseIntegrity") -> bool:
    return (
        _looks_like_non_gi_vague_symptom(text)
        and not integrity.has_confirmed_diagnosis
        and not _has_crc_assessment_anchor(text)
    )


def _create_non_crc_symptom_clarification_message() -> str:
    return (
        "这句话更像是一般身体不适描述，还不足以直接进入结直肠癌评估。\n"
        "如果主要是肠胃或腹部不适，我可以继续按门诊分诊追问；如果是浑身不舒服、全身难受这类非胃肠道症状，建议先到全科、内科或急诊线下就诊。\n"
        "您也可以直接补充更具体的情况，例如“腹痛3天”“有便血”“腹泻2周”或“我想咨询治疗方案”。"
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
        "鎴戝厛涓嶆寜鏅€氭秷鍖栭棬璇婂垎璇婄户缁棶鐥囩姸锛屽洜涓虹郴缁熷凡璇嗗埆鍒板凡鏈夌梾鐞嗐€佽偁闀滄垨褰卞儚寮傚父绾跨储锛?
        "杩欏凡缁忚繘鍏ョ粨鐩磋偁鑲跨槫涓村簥璇勪及鑼冨洿銆傛帴涓嬫潵鎴戜細鍏堣ˉ榻愮梾鐞嗐€佸奖鍍忓拰鍒嗘湡淇℃伅銆?
    )
    return f"{explanation}\n\n{inquiry_message}", True


# ==============================================================================
# 3.2 Semantic Guard: check_case_integrity 璇箟瀹屾暣鎬ф鏌ュ櫒
# ==============================================================================

def check_case_integrity(
    user_text: str,
    model,
    pinned_context: str = "",
    summary_memory: str = ""
) -> CaseIntegrity:
    """
    [Semantic Guard v7] 浣跨敤 LLM 杩涜璇箟瀹屾暣鎬у垽鏂?    瀹屽叏鏇夸唬 _is_complete_case_info 鐨勬鍒欓€昏緫

    Args:
        user_text: 鐢ㄦ埛鐨勮緭鍏ユ枃鏈?        model: 鐢ㄤ簬璇箟鍒ゆ柇鐨?LLM 瀹炰緥

    Returns:
        CaseIntegrity: 鍖呭惈鎵€鏈夎涔夊垽鏂粨鏋滅殑瀵硅薄
    """

    # 绾涔?+ Structured Output锛氬け璐ユ椂杩斿洖淇濆畧榛樿鍊硷紙涓嶅啀鐢?regex 鍏滃簳锛屽尰鐤楀満鏅畞鍙?Unknown锛?    if os.getenv("ASSESSMENT_FAST_RULES_ONLY", "true").strip().lower() not in {"0", "false", "no"}:
        return _quick_case_integrity_from_text("\n".join([summary_memory or "", pinned_context or "", user_text or ""]))

    system_prompt = CASE_INTEGRITY_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_text}")
    ])
    def _fallback_integrity_from_text(full_text: str) -> CaseIntegrity:
        """
        褰撶粨鏋勫寲杈撳嚭澶辫触鏃讹紝浣跨敤杞婚噺瑙勫垯鍏滃簳锛岄伩鍏嶈鍒も€滅梾鐞嗙己澶扁€濄€?        """
        text = (full_text or "").lower()

        has_confirmed_diagnosis = any(
            kw in text for kw in [
                "??", "??", "????", "??", "?", "adenocarcinoma"
            ]
        )

        tnm_tokens = _extract_tnm_tokens_from_text(full_text)
        has_t = bool(tnm_tokens.get("cT"))
        has_n = bool(tnm_tokens.get("cN"))
        has_m = bool(tnm_tokens.get("cM")) or any(
            kw in text for kw in ["?????", "??????", "no distant metastasis", "without distant metastasis"]
        )

        if has_t and has_n and has_m:
            tnm_status = "Complete"
        elif has_t or has_n or has_m:
            tnm_status = "Partial"
        else:
            tnm_status = "Missing"

        if any(kw in text for kw in ["??", "??", "rectum", "rectal"]):
            tumor_location_category = "Rectum"
        elif any(kw in text for kw in ["??", "????", "??", "???", "???", "???", "colon", "sigmoid"]):
            tumor_location_category = "Colon"
        else:
            tumor_location_category = "Unknown"
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
        symptom_hits = any(kw in text for kw in ["渚胯", "鑵圭棝", "鑵硅儉", "渚跨", "鑵规郴", "浣撻噸涓嬮檷", "涔忓姏", "symptom"])
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
    [绾涔夋彁鍙朷 Fast Pass/Diagnosis 鍏辩敤鐨勮瘖鏂俊鎭娊鍙栧櫒銆?    - 鍙緷璧?LLM Structured Output锛屼笉鍋?regex 鍏滃簳
    - 澶辫触鏃惰繑鍥炰繚瀹堥粯璁ゅ€硷紙Unknown/绌猴級锛岀敱杩介棶鏈哄埗鎴栧悗缁妭鐐瑰鐞?    """
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
# 4. Assessment Node (Fast Pass 淇鐗?+ PatientProfile + Active Inquiry)
# ==============================================================================

def node_assessment(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    
    # 浠庣粺涓€鐨?prompts 妯″潡瀵煎叆 System Prompt
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
        # 浣跨敤 _latest_user_text 鑾峰彇鐢ㄦ埛鐨勫師濮嬭緭鍏ワ紝鑰屼笉鏄彲鑳藉寘鍚玆AG鍐呭鐨勬渶鍚庝竴鏉℃秷鎭?        user_text = _latest_user_text(state)
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)
        intent = (state.findings or {}).get("user_intent", "assessment")
        current_findings = state.findings or {}
        encounter_track = state.encounter_track or current_findings.get("encounter_track")
        is_crc_entry_context = encounter_track == "crc_clinical"

        # [鏂板] 妫€鏌ユ槸鍚︽湁浠庢暟鎹簱鏌ヨ鏉ョ殑涓村簥鏁版嵁
        db_clinical_data = current_findings.get("data_source") == "database_query"
        
        # [鏂板] 妫€鏌ユ槸鍚︽湁褰卞儚鍒嗘瀽鎶ュ憡
        has_radiology_report = "radiology_report" in current_findings

        if db_clinical_data:
            if show_thinking:
                print(f"[Assessment] Detected database query data, using clinical info for patient ID {current_findings.get('db_query_patient_id')}...")

            # 浣跨敤鏁版嵁搴撲腑鐨勪复搴婁俊鎭瀯寤鸿瘎浼?            # 杩欎簺淇℃伅鏉ヨ嚜 get_patient_case_info 宸ュ叿
            db_tumor_location = current_findings.get("tumor_location", "unknown")
            db_pathology_confirmed = current_findings.get("pathology_confirmed", False)
            db_tnm_staging = current_findings.get("tnm_staging", {})
            db_histology_type = current_findings.get("histology_type", "鑵虹檶")
            db_molecular_markers = current_findings.get("molecular_markers", {})
            db_mmr_status = current_findings.get("mmr_status", None)
            db_clinical_summary = current_findings.get("clinical_stage_summary", "")
            # [鏂板] 鎻愬彇鍩烘湰淇℃伅
            db_age = current_findings.get("age")
            db_gender = current_findings.get("gender")
            db_ecog = current_findings.get("ecog_score")

            # 鏋勫缓鎮ｈ€呬俊鎭殑瀹屾暣鎻忚堪锛屼緵 LLM 鍒嗘瀽
            db_patient_info_text = f"""
鎮ｈ€匢D: {current_findings.get('db_query_patient_id')}
鎬у埆/骞撮緞: {db_gender}/{db_age}
ECOG璇勫垎: {db_ecog}
鑲跨槫浣嶇疆: {db_tumor_location}
鐥呯悊绫诲瀷: {db_histology_type}
TNM鍒嗘湡: {db_tnm_staging}
MMR鐘舵€? {db_mmr_status or '鏈煡'}
鍒嗗瓙鏍囧織鐗? {db_molecular_markers}
涓村簥鍒嗘湡: {db_clinical_summary}
"""

            if show_thinking:
                print(f"[Assessment] Database patient info: location={db_tumor_location}, "
                      f"pathology={db_pathology_confirmed}, TNM={db_tnm_staging}")

            # 鏋勫缓 findings_delta锛屽寘鍚粠鏁版嵁搴撴彁鍙栫殑淇℃伅
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

            # [Fast Pass] 濡傛灉鏁版嵁搴撲俊鎭畬鏁达紝鐩存帴鏋勫缓 PatientProfile 骞惰繑鍥?            is_info_complete = (
                db_pathology_confirmed and
                db_tumor_location != "unknown" and
                db_tnm_staging and
                any(db_tnm_staging.values())
            )

            # [淇] 妫€鏌ョ敤鎴锋剰鍥炬槸鍚﹀尮閰嶏紝鍙湁娌荤枟鍐崇瓥鐩稿叧鎰忓浘鎵嶈Е鍙?Fast Pass
            is_intent_appropriate = intent in [
                "treatment_decision",
                "clinical_assessment",
                "search_treatment_recommendations",
                "multi_task",
            ]

            if is_info_complete and is_intent_appropriate:
                if show_thinking:
                    print(f"[Assessment] Database info complete, executing Fast Pass mode...")

                # 鍒涘缓 PatientProfile (宸插湪鏂囦欢椤堕儴瀵煎叆)
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

                # 妫€鏌ユ槸鍚︿负杩涘睍鏈?                n_stage = db_tnm_staging.get("cN", "")
                m_stage = db_tnm_staging.get("cM", "")
                is_advanced = ("N1" in n_stage or "N2" in n_stage or "M1" in m_stage)

                # [鏂板] 濡傛灉鏈夊奖鍍忔姤鍛婏紝鏁村悎鍒拌繑鍥炵粨鏋滀腑
                assessment_text = f"鏁版嵁搴撴煡璇㈡偅鑰呬俊鎭細{db_clinical_summary or db_histology_type}"
                if has_radiology_report:
                    radiology_report = current_findings.get("radiology_report", {})
                    imaging_summary = radiology_report.get("ai_interpretation", "")
                    assessment_text += f"\n\n銆愬奖鍍忓AI鍒嗘瀽銆慭n{imaging_summary[:500]}..."  # 鎴彇鍓?00瀛楃
                
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
                        "reasoning": f"浠庢暟鎹簱鏌ヨ鑾峰彇鐨勫畬鏁存偅鑰呬俊鎭?(ID: {current_findings.get('db_query_patient_id')})"
                    }, ensure_ascii=False),
                    "missing_critical_data": [],
                    "clinical_stage": "Assessment_Completed",
                    "roadmap": auto_update_roadmap_from_state(state)
                }
                if profile_entry:
                    updates["patient_profile_timeline"] = [profile_entry]
                return updates

            # [閮ㄥ垎淇℃伅] 濡傛灉淇℃伅涓嶅畬鏁翠絾鏈変竴浜涙暟鎹紝鍚堝苟鍒板父瑙勫鐞?            if show_thinking:
                print(f"[Assessment] Database info partially complete, continuing regular assessment...")

            # 灏嗘暟鎹簱淇℃伅鍚堝苟鍒?user_text 涓紝渚涘悗缁?LLM 鍒嗘瀽浣跨敤
            user_text = f"銆愪粠鏁版嵁搴撴煡璇㈢殑鎮ｈ€呬俊鎭€慭n{db_patient_info_text}\n銆愮敤鎴烽棶棰樸€慭n{user_text}"

        # [Semantic Guard v6] 浣跨敤 LLM 杩涜璇箟瀹屾暣鎬у垽鏂?        # 鏇夸唬鍘熸潵鐨?_is_complete_case_info 姝ｅ垯鍖归厤閫昏緫
        integrity: CaseIntegrity = check_case_integrity(
            user_text,
            model,
            pinned_context=pinned_context,
            summary_memory=summary_memory
        )

        if show_thinking:
            # 鎴柇杩囬暱鐨剅easoning锛岄伩鍏嶈緭鍑哄ぇ娈礡AG鍐呭
            reasoning_preview = integrity.reasoning[:100] + "..." if len(integrity.reasoning) > 100 else integrity.reasoning
            print(f"[Semantic Guard] Location: {integrity.tumor_location_category}, "
                  f"MMR Status: {integrity.mmr_status_availability}, "
                  f"Reasoning: {reasoning_preview}")

        # ================================================================
        # 绛栫暐鍒嗘敮 0: 鐥囩姸鎻忚堪鍙嶉棶锛堟渶楂樹紭鍏堢骇锛?        # 濡傛灉鐢ㄦ埛浠呮弿杩扮棁鐘惰€屾湭纭瘖锛屽簲璇ュ厛鍙嶉棶纭鏄惁涓虹粨鐩磋偁鐧岀浉鍏?        # ================================================================

        # 妫€鏌ユ槸鍚︿粎鍖呭惈鐥囩姸鎻忚堪锛堟湭纭瘖锛?        if integrity.is_symptom_only and not is_crc_entry_context:
            if show_thinking:
                print(f"馃挰 [Assessment] 妫€娴嬪埌鐥囩姸鎻忚堪锛岄渶瑕佸弽闂‘璁よ瘖鏂?)

            # 浣跨敤鍙嬪ソ鐨勫弽闂秷鎭?            inquiry_message = """鎮ㄥソ锛佹垜浜嗚В鍒版偍鎻忚堪浜嗐€恵}銆戠瓑鐥囩姸銆?
璇烽棶锛?*鎮ㄦ槸鍚﹀凡缁忕‘璇婁负缁撶洿鑲犵檶锛堢洿鑲犵檶鎴栫粨鑲犵檶锛夛紵**

- 濡傛灉宸茬‘璇婏紝璇锋彁渚涜瘖鏂俊鎭紙濡傜梾鐞嗘姤鍛娿€佸奖鍍忓鎶ュ憡绛夛級
- 濡傛灉鏈‘璇婏紝寤鸿鎮ㄥ厛鍒版秷鍖栫灏辫瘖锛岃繘琛岃偁闀滅瓑鐩稿叧妫€鏌?
---

馃搶 **娉?*锛氭垜鏄竴涓笓闂ㄦ湇鍔＄粨鐩磋偁鐧屾偅鑰呯殑鍐崇瓥鏀寔绯荤粺锛岄渶瑕佹槑纭殑璇婃柇淇℃伅鎵嶈兘鎻愪緵鍑嗙‘鐨勫垎鏈熷拰娌荤枟鏂规寤鸿銆?"".format(user_text[:50])

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
        # 绛栫暐鍒嗘敮 1: 鍩虹涓夎绱犵己澶辨鏌?        # 浣跨敤璇箟鍒ゆ柇鏇夸唬姝ｅ垯鍖归厤
        # ================================================================

        # 妫€鏌ユ槸鍚︾己灏戠梾鐞嗙‘璇?        if _should_use_non_crc_symptom_clarification(user_text, integrity):
            if show_thinking:
                print("[Assessment] Non-GI vague symptom detected, using neutral clarification instead of pathology reminder...")

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
                print(f"馃洃 [Assessment] 缂哄皯鐥呯悊纭瘖淇℃伅")

            # 浣跨敤鍗＄墖鏍煎紡灞曠ず
            inquiry_message = _create_inquiry_message(
                "鐥呯悊纭瘖淇℃伅",
                "璇锋彁渚涚梾鐞嗙‘璇婁俊鎭紙濡傛椿妫€鎴栨湳鍚庣梾鐞嗘姤鍛婏級锛屼互渚胯繘琛屽噯纭殑鍒嗘湡鍜屾不鐤楁柟妗堝埗瀹氥€?
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

        # 妫€鏌?TNM 鍒嗘湡鏄惁瀹屾暣锛圕omplete 鎵嶅厑璁歌繘鍏ュ悗缁喅绛?蹇€熼€氶亾锛?        if integrity.tnm_status != "Complete":
            if show_thinking:
                print(f"馃洃 [Assessment] TNM 鍒嗘湡涓嶅畬鏁? {integrity.tnm_status}")

            # 浣跨敤鍗＄墖鏍煎紡灞曠ず
            inquiry_message = _create_inquiry_message(
                "TNM鍒嗘湡淇℃伅",
                "璇疯ˉ鍏?TNM 鍒嗘湡淇℃伅锛堝挨鍏舵槸 cT銆乧N銆乧M锛夈€傚彲浠ョ洿鎺ュ彛杩板奖鍍忓鎶ュ憡缁撹锛堝鈥渃T3b cN1 cM0/鏈杩滃杞Щ鈥濓級锛屾棤闇€鎻愪緵鍘熷鎶ュ憡鍏ㄦ枃銆?
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

        # 妫€鏌ヨ偪鐦や綅缃槸鍚︽槑纭?        if integrity.tumor_location_category == "Unknown":
            if show_thinking:
                print(f"馃洃 [Assessment] 鑲跨槫浣嶇疆涓嶆槑纭?)

            # 浣跨敤鍗＄墖鏍煎紡灞曠ず
            inquiry_message = _create_inquiry_message(
                "鑲跨槫浣嶇疆淇℃伅",
                "璇锋槑纭偪鐦や綅缃紙鐩磋偁杩樻槸缁撹偁锛夛紝浠ュ強璺濊倹缂樿窛绂伙紙濡傛湁锛夛紝浠ヤ究閫夋嫨鍚堥€傜殑鍒嗘湡鏍囧噯鍜屾不鐤楁柟妗堛€?
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
        # 绛栫暐鍒嗘敮 2: 涓诲姩杩介棶 (Active Inquiry)
        # 褰撴娴嬪埌杩涘睍鏈熻偪鐦や絾鏈彁渚?MMR/MSI 鐘舵€侊紝涓旂敤鎴锋病鏈夋嫆缁濇椂瑙﹀彂
        # ================================================================
        needs_inquiry = (
            integrity.is_advanced_stage
            and integrity.mmr_status_availability == "Not_Provided"
        )

        if needs_inquiry:
            if show_thinking:
                print(f"馃洃 [Assessment] 鍙戠幇杩涘睍鏈熻偪鐦ょ己澶?MMR/MSI 鐘舵€侊紝瑙﹀彂涓诲姩杩介棶銆?)

            # 浣跨敤鍗＄墖鏍煎紡灞曠ず
            inquiry_message = _format_missing_info_card(
                "MMR/MSI ??????",
                "?????? IHC ? MSI ????????????????????????????????????????"
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
        # 绛栫暐鍒嗘敮 3: 闃绘柇姝诲惊鐜?        # 濡傛灉鐢ㄦ埛鏄庣‘璇?娌″仛"鎴?涓嶇煡閬?锛屾垜浠氨涓嶈拷闂簡
        # 鏍囪涓?Unknown 璁╁悗缁?Decision 鑺傜偣澶勭悊
        # ================================================================
        mmr_data_for_profile = "Unknown"
        if integrity.mmr_status_availability == "Provided":
            # 杩欓噷鍙互璁╁悗缁殑 extraction 鑺傜偣鍘绘彁鍙栧叿浣撴槸 pMMR 杩樻槸 dMMR
            mmr_data_for_profile = "To_Be_Extracted"
        elif integrity.mmr_status_availability == "User_Refused_Or_Unknown":
            # 鐢ㄦ埛鏄庣‘鎷掔粷鎴栦笉鐭ラ亾锛屼笉杩介棶锛屼絾闇€瑕佸悗缁鐞?            mmr_data_for_profile = "Unknown"

        # ================================================================
        # 绛栫暐鍒嗘敮 4: Fast Pass 妯″紡 (鏋勫缓鍙 PatientProfile)
        # 褰撳熀纭€淇℃伅瀹屾暣涓旀棤鍏抽敭缂哄け椤规椂瑙﹀彂
        # ================================================================

        # 鍒ゆ柇鏄惁婊¤冻 Fast Pass 鏉′欢锛氬熀纭€涓夎绱犲畬鏁?+ 涓嶉渶瑕佽拷闂?+ 鎰忓浘鍖归厤
        # [淇] 娣诲姞鎰忓浘妫€鏌ワ紝閬垮厤鍦ㄤ笉鐩稿叧鐨勯棶棰樻椂瑙﹀彂 Fast Pass
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
            and is_intent_appropriate  # [鏂板] 鍙湁娌荤枟鍐崇瓥鐩稿叧鎰忓浘鎵嶈Е鍙?Fast Pass
        )

        if is_fast_pass_eligible:
            if show_thinking:
                print(f"馃┖ [Assessment] Fast Pass 妯″紡锛氳交閲忓寲鎻愬彇骞舵瀯寤?PatientProfile...")
            
            # 绾涔夋彁鍙栵紙Structured Output锛?            diag_res = _semantic_extract_diagnosis(
                user_text,
                model,
                pinned_context=pinned_context,
                summary_memory=summary_memory
            )
            
            # [鏂板] 浠庣敤鎴锋枃鏈腑鎻愬彇鍩烘湰淇℃伅锛堝勾榫勩€佹€у埆銆佷富璇夛級
            import re
            age = None
            gender = None
            chief_complaint = user_text[:100] if user_text else ""  # 鍙栧墠100瀛楃浣滀负涓昏瘔
            
            # ????????58????58 years old?
            age_matches = re.findall(r"(\d+)\s*?", user_text)
            if age_matches:
                age = int(age_matches[0])
            else:
                age_matches = re.findall(r"(\d+)\s*(?:years?|y\.o\.?)", user_text, re.IGNORECASE)
                if age_matches:
                    age = int(age_matches[0])

            # ????
            lowered_user_text = user_text.lower()
            if "?" in user_text or "female" in lowered_user_text:
                gender = "?"
            elif "?" in user_text or "male" in lowered_user_text:
                gender = "?"
            
            # 鎻愬彇涓昏瘔锛堜粠鎮ｈ€呬俊鎭紑澶村埌绗竴涓彞鍙锋垨閫楀彿锛?            if "涓昏瘔" in user_text:
                match = re.search(r'涓昏瘔[锛?]\s*([^锛屻€?\n]+)', user_text)
                if match:
                    chief_complaint = match.group(1).strip()
            
            if show_thinking:
                print(f"  [Fast Pass] 鎻愬彇鐨勫熀鏈俊鎭? age={age}, gender={gender}, chief_complaint={chief_complaint[:50]}...")
            
            # [鏍稿績淇敼 v4] 鏋勫缓涓嶅彲鍙樼殑 PatientProfile
            molecular_markers = diag_res.molecular_markers or {}
            
            new_profile = PatientProfile(
                tumor_type=diag_res.tumor_location,
                pathology_confirmed=diag_res.pathology_confirmed,
                tnm_staging=diag_res.tnm_staging or {},
                mmr_status=diag_res.derived_mmr_status,
                age=age,
                gender=gender,
                chief_complaint=chief_complaint,
                is_locked=True  # 閿佸畾妗ｆ锛岄槻姝㈠悗缁妭鐐圭鏀?            )
            
            # 鏋勫缓 findings_delta锛岀‘淇?Decision 鑺傜偣鑳芥嬁鍒板畬鏁存暟鎹?            # [Semantic Guard] 浣跨敤璇箟妫€鏌ョ粨鏋滄洿鏂?tumor_location
            semantic_location = integrity.tumor_location_category.lower()

            findings_delta = {
                "risk_level": "High",  # 瀹屾暣鐥呬緥榛樿楂橀闄?                "red_flags": [],       # 绠€鍖栧鐞?                "assessment_draft": f"Fast Pass 鎻愬彇锛歿diag_res.clinical_stage_summary}",
                # 銆愭牳蹇冦€戝鍒惰瘖鏂暟鎹埌 findings锛岀‘淇濇暟鎹畬鏁存€?                "pathology_confirmed": diag_res.pathology_confirmed,
                "biopsy_confirmed": diag_res.pathology_confirmed,
                # [Semantic Guard] 浣跨敤璇箟鍒嗙被缁撴灉锛屼紭鍏堜娇鐢?LLM 鍒ゆ柇鐨勪綅缃?                "tumor_location": semantic_location if semantic_location != "unknown" else diag_res.tumor_location.lower(),
                "tumor_subsite": diag_res.tumor_subsite,
                "histology_type": diag_res.histology_type,
                "molecular_markers": molecular_markers,
                "tnm_staging": diag_res.tnm_staging or {},
                "clinical_stage_staging": diag_res.clinical_stage_summary,
                "clinical_stage_group": (diag_res.tnm_staging or {}).get("stage_group", ""),
                # 鐢?Schema 娲剧敓鐨勫叧閿垎瀛愮姸鎬侊紙渚涘悗缁妭鐐?璺敱浣跨敤锛?                "mmr_status": diag_res.derived_mmr_status,
                "kras_status": diag_res.derived_kras_status,
                "braf_status": diag_res.derived_braf_status,
                # [Semantic Guard] 鏍囪鏄惁涓鸿繘灞曟湡
                "is_advanced_stage": integrity.is_advanced_stage,
                # [Semantic Guard] 淇濆瓨璇箟妫€鏌ョ粨鏋滐紝渚涘悗缁妭鐐瑰弬鑰?                "semantic_integrity": integrity.model_dump(),
                # [鏍囪] 姝ゆ爣蹇楀憡璇夊悗缁妭鐐?杩欐槸 Fast Pass 妯″紡锛岃瘎浼板凡绠€鍖栦絾鏁版嵁瀹屾暣"
                "fast_pass_mode": True,
            }
            
            profile_entry = _build_profile_change_entry(
                state.patient_profile,
                new_profile,
                source="assessment_fast_pass_semantic"
            )
            updates = {
                "patient_profile": new_profile,  # [鏍稿績] 瀛樺叆鍙 PatientProfile
                "findings": findings_delta,
                "assessment_draft": json.dumps({
                    "risk": "High",
                    "summary": diag_res.clinical_stage_summary or "瀹屾暣鐥呬緥",
                    "reasoning": f"Fast Pass 妯″紡锛歋emantic Guard 璇箟妫€鏌ラ€氳繃锛孭atientProfile 宸查攣瀹氥€侺ocation={integrity.tumor_location_category}, Advanced={integrity.is_advanced_stage}"
                }, ensure_ascii=False),
                "missing_critical_data": [],  # 瀹屾暣鐥呬緥鏃犵己澶辨暟鎹?                "clinical_stage": "Assessment_Completed",
                "error": None,
                "roadmap": auto_update_roadmap_from_state(state)
            }
            if profile_entry:
                updates["patient_profile_timeline"] = [profile_entry]

            if show_thinking:
                print(f"馃┖ [Assessment] Fast Pass 瀹屾垚锛歍NM={diag_res.tnm_staging}, location={integrity.tumor_location_category}")
                print(f"馃敀 [PatientProfile] 宸查攣瀹氾紝tumor_type={new_profile.tumor_type}, mmr_status={new_profile.mmr_status}")

            return updates

        # ================================================================
        # 绛栫暐鍒嗘敮 5: 甯歌娴佺▼ (闈炲畬鏁寸梾渚嬶紝璧板畬鏁寸殑鎱㈤€熷垎鏋愭祦绋?
        # ================================================================
        try:
            # [鍏抽敭淇] 婵€杩涙埅鏂秷鎭巻鍙诧紝闃叉 token 瓒呴檺
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
            
            # --- 鑴忔暟鎹竻娲?---
            risk_clean = "Average"
            risk_raw = str(res.risk_level).lower()
            if "high" in risk_raw or "楂? in risk_raw: risk_clean = "High"
            elif "mod" in risk_raw or "涓? in risk_raw: risk_clean = "Moderate"
            
            # --- 鏅鸿兘璇婃柇淇℃伅鎻愬彇 ---
            # [Semantic Guard] 鍦ㄥ父瑙勬祦绋嬩腑涔熶繚瀛樿涔夋鏌ョ粨鏋?            findings_delta = {
                "risk_level": risk_clean,
                "red_flags": res.red_flags,
                "assessment_draft": res.assessment_summary,
                # [Semantic Guard] 淇濆瓨璇箟瀹屾暣鎬ф鏌ョ粨鏋?                "semantic_integrity": integrity.model_dump(),
                # [Semantic Guard] 鏍囪鏄惁涓鸿繘灞曟湡
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
                print(f"馃搵 [Assessment] Risk: {risk_clean} | Missing: {len(res.missing_critical_data)} items")
            
            # [鏂板] 鑷姩鏇存柊璺嚎鍥?            updates["roadmap"] = auto_update_roadmap_from_state(state)
            
            return updates

        except Exception as e:
            error_msg = str(e)
            print(f"鈿狅笍 [Assessment Fail] {e}")
            
            # [鍘焆 鍏滃簳閿欒澶勭悊
            error_updates = {
                "clinical_stage": "Assessment",
                "error": f"璇勪及瑙ｆ瀽澶辫触锛堟湭鍋氭鍒欓檷绾э紝浠ュ厤璇垽锛夛細{error_msg}",
                "findings": {"risk_level": "High", "assessment_draft": "璇勪及瑙ｆ瀽澶辫触锛岃浜哄伐澶嶆牳銆?},
                "missing_critical_data": ["System_Error_Manual_Check"]
            }
            temp_state = state.model_copy(update=error_updates)
            error_updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
            return error_updates

    return _run


# ==============================================================================
# 5. Diagnosis Node (Enhanced: Extract complete staging info)
# ==============================================================================

def node_diagnosis(model, tools: List[BaseTool], streaming: bool = False, show_thinking: bool = True) -> Runnable:
    
    # 浠庣粺涓€鐨?prompts 妯″潡瀵煎叆 System Prompt
    system_prompt = DIAGNOSIS_SYSTEM_PROMPT
    
    # 鍒嗙 Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Patient Profile: {patient_profile}"),
    ])
    
    def _run(state: CRCAgentState):
        try:
            # [鍏抽敭淇] 婵€杩涙埅鏂秷鎭巻鍙诧紝闃叉 token 瓒呴檺
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

            # --- 鑴忔暟鎹竻娲?---
            loc_clean = res.tumor_location.lower()

            # --- 鏋勯€犲畬鏁寸殑 findings_delta ---
            findings_delta = {
                "pathology_confirmed": res.pathology_confirmed,
                "biopsy_confirmed": res.pathology_confirmed,
                "tumor_location": loc_clean,
                "histology_type": res.histology_type,
                "molecular_markers": res.molecular_markers,
                "mri_assessment": res.rectal_mri_params,
                # [鏂板] 淇濆瓨 TNM 鍒嗘湡淇℃伅锛屼緵璺敱鍐崇瓥浣跨敤
                "tnm_staging": getattr(res, 'tnm_staging', None),
                "clinical_stage_summary": getattr(res, 'clinical_stage_summary', None),
                # 鐢?Schema 娲剧敓鐨勫叧閿垎瀛愮姸鎬侊紙渚涘悗缁妭鐐?璺敱浣跨敤锛?                "mmr_status": res.derived_mmr_status,
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
                        "鐥呯悊纭瘖淇℃伅",
                        "璇锋彁渚涚梾鐞嗙‘璇婁俊鎭紙濡傛椿妫€鎴栨湳鍚庣梾鐞嗘姤鍛婏級锛屼互渚胯繘琛屽噯纭殑鍒嗘湡鍜屾不鐤楁柟妗堝埗瀹氥€?
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
                    print("馃洃 [Diagnosis] 鏃犵梾鐞嗙‘璇婏紝鏆傚仠銆?)
            else:
                print(f"鉁?[Diagnosis] 纭瘖: {res.histology_type} @ {loc_clean}")

            # [鏂板] 鑷姩鏇存柊璺嚎鍥?            temp_state = state.model_copy(update=updates)
            updates["roadmap"] = auto_update_roadmap_from_state(temp_state)

            return updates

        except Exception as e:
            print(f"鈿狅笍 [Diagnosis Fail] {e}")
            # 澶辫触涓嶉檷绾у埌 regex锛岀洿鎺ヤ氦缁欒拷闂?閲嶈瘯鏈哄埗
            error_updates = {"clinical_stage": "Diagnosis", "error": f"鏃犳硶鎻愬彇璇婃柇淇℃伅锛岃閲嶈瘯: {e}"}
            # 鍗充娇鍑洪敊涔熸洿鏂拌矾绾垮浘
            temp_state = state.model_copy(update=error_updates)
            error_updates["roadmap"] = auto_update_roadmap_from_state(temp_state)
            return error_updates

    return _run


# ==============================================================================
# 6. Staging Router (Fast Pass 淇鐗?
# ==============================================================================

def node_staging_router(state: CRCAgentState):
    # Phase 1 transitional router: narrow-scope exception. Do not expand this into a second policy layer.
    """
    [鏍稿績淇 v3] 鍒嗘湡璺敱锛氭牴鎹偪鐦や綅缃€夋嫨鍒嗘湡鑺傜偣
    
    Fast Pass 绛栫暐锛?    - 濡傛灉鏈?fast_pass_mode 鏍囧織锛堝畬鏁寸梾渚嬶級锛岀洿鎺ュ幓 decision锛岃烦杩?staging
    - 鍚﹀垯姝ｅ父璺敱鍒板搴旂殑 staging 鑺傜偣
    
    璺敱閫昏緫锛?    1. Fast Pass 妯″紡 -> decision
    2. 鏈‘璇?-> decision
    3. 鏈煡浣嶇疆 -> decision锛堟棤娉曠‘瀹氬垯鍘?decision锛?    4. 鐩磋偁 -> rectal_staging
    5. 缁撹偁 -> colon_staging
    """
    f = state.findings or {}

    if state.clinical_stage == "Diagnosis_Pending" or "Pathology Report" in (state.missing_critical_data or []):
        print("[Staging Router] Diagnosis_Pending锛氱瓑寰呯梾鐞嗘姤鍛婏紝缁撴潫鏈疆銆?)
        return "end_turn"
    
    # [鏂板] Fast Pass 妯″紡锛氬畬鏁寸梾渚嬬洿鎺ュ幓 decision锛岃烦杩?staging 鏍￠獙
    # 铏界劧 Staging 鑺傜偣鏈夊揩閫熸牎楠屽姛鑳斤紝浣嗕负浜嗘瀬鑷存€ц兘锛屽畬鏁寸梾渚嬬洿鎺ュ幓 decision
    if f.get("fast_pass_mode"):
        print("[Staging Router] Fast Pass 妯″紡锛氳烦杩?staging锛岀洿鎺ヨ繘鍏?decision")
        return "decision"
    
    # 妫€鏌ユ槸鍚﹂渶瑕佽烦杩?Staging
    if not f.get("pathology_confirmed", False): 
        return "decision"
    
    loc = f.get("tumor_location", "unknown")
    if loc == "unknown":
        # 鏃犳硶纭畾浣嶇疆锛屽皾璇曚粠鏂囨湰鎺ㄦ柇
        user_text = _latest_user_text(state)
        if "鐩磋偁" in user_text or "鐩磋偁鐧? in user_text:
            return "rectal_staging"
        elif "缁撹偁" in user_text or "缁撹偁鐧? in user_text:
            return "colon_staging"
        else:
            return "decision"
    
    # 姝ｅ父璺敱鍒板搴旂殑 Staging 鑺傜偣
    if loc == "rectum": 
        return "rectal_staging"
    if loc == "colon": 
        return "colon_staging"
    return "decision"

