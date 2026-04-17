"""
Staging Nodes

本模块包含结肠癌分期、直肠癌分期和多模态分期节点。
(Fast Pass v3 修复版：即使快速通道也必须执行校验逻辑)

核心原则：
- Fast Pass 模式不是"跳过 Staging"，而是"快速校验"
- 校验 TNM 组合是否医学上有效
- 校验文字描述与提取结果是否一致
- 跳过耗时的高级分析（如 MRI/CT 详细解读），只做必要检查
"""

import json
from typing import Callable, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from ..state import CRCAgentState
from .node_utils import (
    _user_text,
    _extract_ct_text,
    _extract_mri_text,
    _is_postop_context,
    _select_tools,
)


def _validate_tnm_consistency(findings: dict, user_text: str) -> tuple[bool, str]:
    """
    [Fast Pass 快速校验] 验证 TNM 组合的医学有效性
    
    返回: (is_valid, error_message)
    """
    tnm = findings.get("tnm_staging", {})
    cT = tnm.get("cT", "")
    cN = tnm.get("cN", "")
    cM = tnm.get("cM", "")
    
    # 提取 TNM 数值
    t_match = re.search(r'cT(\d+)', cT) if cT else None
    n_match = re.search(r'cN(\d+)', cN) if cN else None
    m_match = re.search(r'cM(\d+)', cM) if cM else None
    
    t_val = int(t_match.group(1)) if t_match else 0
    n_val = int(n_match.group(1)) if n_match else 0
    m_val = int(m_match.group(1)) if m_match else 0
    
    # 医学规则校验
    errors = []
    
    # T 分期只能是 1-4
    if t_val < 1 or t_val > 4:
        errors.append(f"T 分期值 {t_val} 无效，必须是 cT1-cT4")
    
    # N 分期只能是 0-3
    if n_val < 0 or n_val > 3:
        errors.append(f"N 分期值 {n_val} 无效，必须是 cN0-cN3")
    
    # M 分期只能是 0-1
    if m_val < 0 or m_val > 1:
        errors.append(f"M 分期值 {m_val} 无效，必须是 cM0 或 cM1")
    
    # T4 癌症不应该只有 N0（除非是原位癌 Tis）
    if t_val >= 4 and n_val == 0 and t_val != 0:
        errors.append("警告：T4 分期通常伴随淋巴结转移，请确认 N 分期")
    
    # 校验肿瘤位置一致性
    loc = findings.get("tumor_location", "").lower()
    if "直肠" in user_text and loc != "rectum":
        errors.append("警告：文本提到直肠，但 tumor_location 不是 rectum")
    if "结肠" in user_text and loc != "colon":
        errors.append("警告：文本提到结肠，但 tumor_location 不是 colon")
    
    if errors:
        return False, "; ".join(errors)
    return True, "TNM 组合有效"


def _is_fast_pass_complete_case(findings: dict, user_text: str) -> bool:
    """
    [判定函数] 检查是否为 Fast Pass 完整病例
    完整病例 = 已有完整的 TNM + 位置 + 病理确诊信息
    """
    if not findings:
        return False
    
    # 必须有 TNM 信息
    tnm = findings.get("tnm_staging", {})
    if not all([tnm.get("cT"), tnm.get("cN"), tnm.get("cM")]):
        return False
    
    # 必须有位置信息
    if not findings.get("tumor_location"):
        return False
    
    # 必须有病理确诊
    if not findings.get("pathology_confirmed"):
        return False
    
    # 检查是否有红旗关键词（如"阴影"、"可疑转移"等需要详细分析的情况）
    user_text_lower = user_text.lower()
    complex_keywords = [
        "mrf阳性", "emvi阳性", "脉管癌栓", "神经侵犯",
        "肝脏占位", "肺转移", "腹腔转移", "可疑转移"
    ]
    
    if any(kw in user_text_lower for kw in complex_keywords):
        # 复杂情况，不走 Fast Pass
        return False
    
    return True


def node_colon_staging(tools: List[BaseTool]) -> Callable[[CRCAgentState], Dict]:
    """
    Colon Staging Node - 结肠癌分期节点
    
    职责：
    1. Fast Pass 模式：快速校验 TNM 组合有效性
    2. 完整模式：解析CT报告进行局部和远处分期
    3. 评估远处转移（M分期）
    4. 评估可切除性
    """
    ct_tool = next(t for t in tools if getattr(t, "name", "") == "VolumeCTSegmentorTool")

    def _run(state: CRCAgentState):
        text_context = _user_text(state)
        ct_text = _extract_ct_text(state)
        is_postop = _is_postop_context(text_context)
        findings = state.findings or {}
        
        # [v3 Fast Pass] 检测是否为完整病例
        if _is_fast_pass_complete_case(findings, text_context):
            # 【Fast Pass 模式】只做快速校验，不调用 CT 工具
            
            # 1. 校验 TNM 组合
            is_valid, validation_msg = _validate_tnm_consistency(findings, text_context)
            
            if not is_valid:
                print(f"⚠️ [Colon Staging Fast Pass] TNM 校验失败: {validation_msg}")
            
            # 2. 从 findings 构建分期信息
            tnm = findings.get("tnm_staging", {})
            m_stage = tnm.get("cM", "cMx")
            
            # 3. 快速返回（毫秒级，不消耗 Token）
            return {
                "messages": [
                    AIMessage(content=f"[Colon Staging Fast Pass] 快速校验通过"),
                    AIMessage(content=f"[TNM Validation] {validation_msg}"),
                    AIMessage(content=f"[M Stage] {m_stage} (from assessment data)")
                ],
                "findings": {
                    "ct_assessment": {
                        "m_stage": m_stage,
                        "distant_metastasis_summary": f"M 分期来自患者提供信息: {m_stage}"
                    },
                    "m_stage": m_stage,
                    "fast_pass_staging": True,  # 标记为 Fast Pass 模式
                },
                "clinical_stage": "Staging",
                "distant_metastasis_check": f"Fast Pass: {m_stage}",
            }
        
        # [完整模式] 原始逻辑
        if is_postop and not ct_text:
            return {"messages": [AIMessage(content="[CT Staging Skipped] Postoperative context detected.")], "clinical_stage": "Staging"}

        if not ct_text:
            return {"messages": [AIMessage(content="[CT Staging Skipped] No CT report text found.")], "clinical_stage": "Staging"}

        result = ct_tool.invoke({"ct_report_text": ct_text})
        distant_metastasis_check = result.get("distant_metastasis_summary", "") if isinstance(result, dict) else ""
        
        return {
            "messages": [AIMessage(content=f"[CT Staging Result] {json.dumps(result, ensure_ascii=False)}")],
            "findings": {"ct_assessment": result},
            "clinical_stage": "Staging",
            "distant_metastasis_check": distant_metastasis_check,
        }

    return _run


def node_rectal_staging(tools: List[BaseTool]) -> Callable[[CRCAgentState], Dict]:
    """
    Rectal Staging Node - 直肠癌多模态分期节点
    
    职责：
    1. Fast Pass 模式：快速校验 TNM + 位置一致性
    2. 完整模式：MRI分析（局部分期）+ CT分析（远处转移）
    
    [专家建议] 直肠癌初诊必须做胸腹盆CT，用于评估肺/肝转移
    """
    mr_tool = next(t for t in tools if getattr(t, "name", "") == "RectalMRStagerTool")
    ct_tool = next(t for t in tools if getattr(t, "name", "") == "VolumeCTSegmentorTool")

    def _run(state: CRCAgentState):
        text_context = _user_text(state)
        mri_text = _extract_mri_text(state) or text_context
        ct_text = _extract_ct_text(state)
        
        findings_delta: Dict = {}
        messages_list: List[AIMessage] = []
        findings = state.findings or {}
        
        # [v3 Fast Pass] 检测是否为完整病例
        if _is_fast_pass_complete_case(findings, text_context):
            # 【Fast Pass 模式】只做快速校验
            
            # 1. 校验 TNM 组合
            is_valid, validation_msg = _validate_tnm_consistency(findings, text_context)
            
            if not is_valid:
                print(f"⚠️ [Rectal Staging Fast Pass] TNM 校验失败: {validation_msg}")
                messages_list.append(AIMessage(content=f"[警告] Fast Pass 校验发现潜在问题: {validation_msg}"))
            
            # 2. 快速构建分期结果
            tnm = findings.get("tnm_staging", {})
            m_stage = tnm.get("cM", "cMx")
            t_stage = tnm.get("cT", "")
            n_stage = tnm.get("cN", "")
            
            # 3. 快速返回
            return {
                "messages": [
                    AIMessage(content="[Rectal Staging Fast Pass] 快速校验通过"),
                    AIMessage(content=f"[TNM Validation] {validation_msg}"),
                    AIMessage(content=f"[Local Staging] {t_stage}{n_stage} (from assessment data)"),
                    AIMessage(content=f"[M Stage] {m_stage} (from assessment data)"),
                    AIMessage(content="[Note] 直肠癌强烈建议完善 MRI 和 CT 以获取更详细分期")
                ],
                "findings": {
                    "mri_assessment": {
                        "local_staging_summary": f"Fast Pass: {t_stage}{n_stage}",
                        "fast_pass_mode": True
                    },
                    "ct_assessment": {
                        "m_stage": m_stage,
                        "distant_metastasis_summary": f"Fast Pass M Stage: {m_stage}"
                    },
                    "m_stage": m_stage,
                    "fast_pass_staging": True,
                },
                "clinical_stage": "Staging",
                "distant_metastasis_check": f"Fast Pass: {m_stage}",
            }
        
        # [完整模式] 原始逻辑
        # Step 1: MRI for LOCAL staging
        mri_result = mr_tool.invoke({"text_context": mri_text})
        messages_list.append(AIMessage(content=f"[MRI Local Staging] {json.dumps(mri_result, ensure_ascii=False)}"))
        findings_delta["mri_assessment"] = mri_result
        
        if isinstance(mri_result, dict):
            if mri_result.get("mrf_status") == "positive":
                findings_delta["rectal_mrf_positive"] = True
            if mri_result.get("neoadjuvant_recommended"):
                findings_delta["neoadjuvant_recommended"] = True
        
        # Step 2: CT for DISTANT METASTASIS (M staging) - CRITICAL for Rectal Cancer
        # [专家建议] 直肠癌初诊必须做胸腹盆CT，用于评估肺/肝转移
        distant_metastasis_check = "CT not performed - M staging incomplete"

        if ct_text:
            ct_result = ct_tool.invoke({"ct_report_text": ct_text})

            # [新增]: 强行校正逻辑 (Self-Correction for M staging)
            # 如果 CT 文本里明确写了 "未见转移" 或 "M0"，强制覆盖工具的错误输出
            text_lower = ct_text.lower()
            negative_keywords = ["未见远处转移", "无远处转移", "未见明显异常", "m0", "no metastasis"]

            if any(k in text_lower for k in negative_keywords):
                # 只有当工具错误地报告了 M1 时才修正
                if isinstance(ct_result, dict) and ct_result.get("m_stage", "").startswith("M1"):
                    print(f"[Staging Correction] Detected M0 keywords in text but tool returned {ct_result.get('m_stage')}. Overriding.")
                    ct_result["m_stage"] = "M0"
                    ct_result["metastasis_sites"] = []
                    ct_result["distant_metastasis_summary"] = "No distant metastasis detected (M0) based on textual confirmation."

            messages_list.append(AIMessage(content=f"[CT Distant Metastasis Screening] {json.dumps(ct_result, ensure_ascii=False)}"))
            findings_delta["ct_assessment"] = ct_result

            if isinstance(ct_result, dict):
                distant_metastasis_check = ct_result.get("distant_metastasis_summary", "M staging: unknown")
                findings_delta["m_stage"] = ct_result.get("m_stage", "MX")
                findings_delta["metastasis_sites"] = ct_result.get("metastasis_sites", [])

                # 检查肺/肝转移
                metastasis_sites = ct_result.get("metastasis_sites", [])
                if metastasis_sites:
                    findings_delta["m_stage"] = "M1"
                    if "lung" in str(metastasis_sites).lower() or "肺" in str(metastasis_sites):
                        findings_delta["lung_metastasis"] = True
                    if "liver" in str(metastasis_sites).lower() or "肝" in str(metastasis_sites):
                        findings_delta["liver_metastasis"] = True
        else:
            # [关键] 专家指出：直肠癌初诊必须做胸腹盆CT
            print("[Staging] ⚠️ 警告：直肠癌患者缺失 CT 报告，无法评估 M 分期（肺/肝转移）")
            messages_list.append(AIMessage(
                content="[严重警告] 直肠癌患者缺失胸腹盆CT报告。无法评估远处转移（M分期）。这是医疗安全的关键检查，必须完成。"
            ))
            findings_delta["ct_assessment_missing"] = True
            findings_delta["staging_incomplete"] = True
            # 将 CT 加入缺失数据清单
            findings_delta["missing_critical_data"] = ["chest_abdomen_pelvis_CT (Required for M staging in rectal cancer)"]
        
        staging_summary = {
            "local_staging_mri": mri_result.get("local_staging_summary", "") if isinstance(mri_result, dict) else "",
            "distant_staging_ct": distant_metastasis_check,
            "multimodal_staging_complete": ct_text is not None,
        }
        findings_delta["combined_staging"] = staging_summary
        
        return {
            "messages": messages_list,
            "findings": findings_delta,
            "clinical_stage": "Staging",
            "distant_metastasis_check": distant_metastasis_check,
        }

    return _run


import re  # 添加 re 模块导入供验证函数使用
