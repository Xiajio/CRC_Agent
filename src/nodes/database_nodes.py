"""
Database Query Nodes
(Migrated from intent_nodes.py - Pure LLM Tool Calling for Case Database)
"""

import json
import os
import re
import traceback
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage

from ..state import CRCAgentState
from ..prompts import DATABASE_QUERY_SYSTEM_PROMPT
from .node_utils import (
    _latest_user_text,
    _truncate_message_history,
    _build_pinned_context,
    _build_summary_memory,
)
from ..tools.database_tools import ATOMIC_DATABASE_TOOLS
from ..tools.card_formatter import CardFormatter
from .planner import get_current_pending_step, mark_step_completed


# === UI-friendly summarization helpers ===
def _format_case_summary_markdown(case_data: dict) -> str:
    """把数据库病例信息整理成更友好的 Markdown 摘要（用于“总结/目前情况”类请求）。"""
    if not isinstance(case_data, dict) or case_data.get("error"):
        return "未找到可用于总结的病例信息。"

    pid = case_data.get("patient_id", "N/A")
    gender = case_data.get("gender", "N/A")
    age = case_data.get("age", "N/A")
    ecog = case_data.get("ecog_score", "N/A")

    location = case_data.get("tumor_location", "N/A")
    histology = case_data.get("histology_type", "N/A")
    mmr = case_data.get("mmr_status", "N/A")
    cea = case_data.get("cea_level", "N/A")

    ct = case_data.get("ct_stage", "")
    cn = case_data.get("cn_stage", "")
    cm = case_data.get("cm_stage", "") or "0"
    tnm = case_data.get("tnm_stage") or f"cT{ct}N{cn}M{cm}"

    clinical_stage = case_data.get("clinical_stage", "N/A")

    lines = []
    lines.append(f"### 患者概况（病例 {pid}）")
    lines.append(f"- **性别/年龄**：{gender} / {age} 岁")
    lines.append(f"- **体能状态（ECOG）**：{ecog}")
    lines.append("")
    lines.append("### 肿瘤与分期")
    lines.append(f"- **部位**：{location}")
    lines.append(f"- **病理分化**：{histology}")
    lines.append(f"- **TNM**：{tnm}")
    lines.append(f"- **临床分期（数据库字段）**：{clinical_stage}")
    lines.append("")
    lines.append("### 实验室/分子特征")
    lines.append(f"- **CEA**：{cea}")
    lines.append(f"- **MMR/MSI**：{mmr}")
    lines.append("")
    lines.append("### 小结")
    lines.append("- 以上为数据库已记录信息的汇总。若需要进一步“临床解读/治疗建议”，通常还需要结合影像学评估、病理细节（如脉管/神经侵犯）、合并症与实验室结果等。")

    return "\n".join(lines).strip()


def _format_case_brief(case_data: dict) -> str:
    """Build a single-line patient brief for tool detail display."""
    if not isinstance(case_data, dict) or case_data.get("error"):
        return "未找到病例信息。"

    cm_stage = case_data.get("cm_stage", "") or "0"
    return (
        f"病例 {case_data.get('patient_id', 'N/A')}: "
        f"{case_data.get('gender', 'N/A')}/{case_data.get('age', 'N/A')}岁, "
        f"{case_data.get('tumor_location', 'N/A')} {case_data.get('histology_type', 'N/A')}分化癌, "
        f"分期 cT{case_data.get('ct_stage', '')}N{case_data.get('cn_stage', '')}M{cm_stage}"
    )


def _format_database_tool_message(tool_name: str, result: object) -> str:
    """Convert database tool results into concise tool detail text."""
    if not isinstance(result, dict):
        return str(result)

    if result.get("error"):
        return result.get("error", "工具执行失败")

    if tool_name == "get_patient_case_info":
        return f"{_format_case_brief(result)}\n\n已生成结构化卡片，可在侧边栏按需展开查看。"

    if tool_name == "get_patient_imaging":
        patient_id = result.get("folder_name", "N/A")
        total_images = result.get("total_images", 0)
        return f"影像样本：患者 {patient_id}，共 {total_images} 张影像\n\n已生成结构化卡片，可在侧边栏按需展开查看。"

    if tool_name == "get_patient_pathology_slides":
        patient_id = result.get("folder_name", "N/A")
        total_images = result.get("total_images", 0)
        return f"病理切片样本：患者 {patient_id}，共 {total_images} 张切片预览\n\n已生成结构化卡片，可在侧边栏按需展开查看。"

    if tool_name == "summarize_patient_existing_info":
        return result.get("summary", "查询完成。")

    return json.dumps(result, ensure_ascii=False)


def _build_patient_profile_payload(case_data: dict) -> dict | None:
    """Build a lightweight patient_profile payload for frontend card reconstruction."""
    if not isinstance(case_data, dict) or case_data.get("error"):
        return None

    cm_stage = case_data.get("cm_stage", "") or "0"
    return {
        "tumor_type": case_data.get("tumor_location") or "Unknown",
        "pathology_confirmed": True,
        "tnm_staging": {
            "cT": f"cT{case_data.get('ct_stage', '')}",
            "cN": f"cN{case_data.get('cn_stage', '')}",
            "cM": f"cM{cm_stage}",
        },
        "mmr_status": case_data.get("mmr_status") or "Unknown",
        "age": case_data.get("age"),
        "gender": case_data.get("gender"),
        "ecog_score": case_data.get("ecog_score"),
        "is_locked": False,
    }


def _extract_json_payload(text: str | None) -> dict | None:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("{") and candidate.endswith("}"):
        try:
            return json.loads(candidate)
        except Exception:
            return None
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(candidate[start:end + 1])
        except Exception:
            return None
    return None


def _safe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_database_search_filters(raw_filters: dict | None) -> dict | None:
    if not isinstance(raw_filters, dict) or not raw_filters:
        return None

    normalized: dict[str, object] = {}
    list_fields = {"tumor_location", "ct_stage", "cn_stage", "histology_type", "mmr_status"}
    numeric_fields = {"patient_id", "age_min", "age_max", "cea_max"}

    for key, value in raw_filters.items():
        if value in (None, "", []):
            continue
        if key in list_fields:
            if isinstance(value, list):
                normalized[key] = value
            else:
                normalized[key] = [value]
            continue
        if key in numeric_fields:
            parsed = _safe_int(value)
            if parsed is not None:
                normalized[key] = parsed
            continue
        normalized[key] = value

    return normalized or None


def _build_database_workbench_context(
    *,
    mode: str,
    query_text: str | None,
    filters: dict | None = None,
    selected_patient_id: int | None = None,
) -> dict[str, object]:
    return {
        "visible": True,
        "mode": mode,
        "query_text": query_text,
        "filters": _normalize_database_search_filters(filters),
        "selected_patient_id": selected_patient_id,
    }


def _merge_database_workbench_findings(
    findings: dict[str, object] | None,
    context: dict[str, object] | None,
) -> dict[str, object] | None:
    if not context:
        return findings if findings else None

    merged = dict(findings or {})
    merged["database_workbench"] = context
    return merged



# 添加辅助函数：标准化肿瘤位置
def _is_imaging_analysis_support_request(state: CRCAgentState, user_text: str | None) -> bool:
    """Whether this database turn is only preparing context for imaging analysis."""
    findings = state.findings or {}
    if findings.get("user_intent") == "imaging_analysis":
        return True

    sub_tasks = findings.get("sub_tasks") or []
    if findings.get("multi_task_mode") and "imaging_analysis" in sub_tasks:
        return True

    text = user_text or ""
    if not text:
        return False

    report_keywords = ["影像分析报告", "查看影像分析报告", "读取影像分析报告"]
    if any(keyword in text for keyword in report_keywords):
        return False

    analysis_keywords = [
        "影像分析",
        "影像组学",
        "肿瘤检测",
        "肿瘤筛查",
        "肿瘤定位",
        "影像分割",
        "特征提取",
        "radiomics",
        "分析CT",
        "分析MRI",
        "分析影像",
    ]
    if any(keyword in text for keyword in analysis_keywords):
        return True

    current_step = get_current_pending_step(state)
    if current_step and current_step.tool_needed.lower() in {"case_database_query", "database_query"}:
        for step in state.current_plan or []:
            tool_needed = step.tool_needed.lower()
            if step.status == "pending" and step.id != current_step.id and any(
                keyword in tool_needed for keyword in ["imaging_analysis", "tumor_detection", "radiology", "ct_analysis"]
            ):
                return True

    return False


def _build_case_findings_update(case_data: dict) -> dict[str, object]:
    """Extract structured clinical context from a database case record."""
    inferred_from_tnm = _infer_tnm_from_text(case_data.get("tnm_stage"))
    inferred_from_clinical = _infer_tnm_from_text(case_data.get("clinical_stage"))
    inferred_t = inferred_from_tnm.get("T") or inferred_from_clinical.get("T")
    inferred_n = inferred_from_tnm.get("N") or inferred_from_clinical.get("N")
    inferred_m = inferred_from_tnm.get("M") or inferred_from_clinical.get("M")
    cT = _normalize_tnm_component("cT", case_data.get("ct_stage")) or _normalize_tnm_component("cT", inferred_t)
    cN = _normalize_tnm_component("cN", case_data.get("cn_stage")) or _normalize_tnm_component("cN", inferred_n)
    cM = _normalize_tnm_component("cM", case_data.get("cm_stage")) or _normalize_tnm_component("cM", inferred_m)

    patient_id = case_data.get("patient_id")
    return {
        "pathology_confirmed": True,
        "biopsy_confirmed": True,
        "tumor_location": _normalize_tumor_location(case_data.get("tumor_location", "")),
        "histology_type": case_data.get("histology_type", "腺癌"),
        "tnm_staging": {
            "cT": cT,
            "cN": cN,
            "cM": cM,
        },
        "molecular_markers": _extract_molecular_markers(case_data),
        "clinical_stage_summary": case_data.get("clinical_stage", ""),
        "patient_info": {
            "patient_id": patient_id,
            "age": case_data.get("age"),
            "gender": case_data.get("gender"),
            "cea_level": case_data.get("cea_level"),
            "ecog_score": case_data.get("ecog_score"),
        },
        "data_source": "database_query",
        "db_query_patient_id": str(patient_id),
        "current_patient_id": str(patient_id).zfill(3) if patient_id is not None else None,
    }


def _normalize_tumor_location(location: str) -> str:
    """标准化肿瘤位置为英文格式"""
    if not location:
        return "unknown"

    location_str = str(location)
    location_lower = location_str.lower()

    # 精确匹配（完整词汇）
    rectum_full = ["直肠", "直肠癌", "rectum", "rectal"]
    colon_full = ["结肠", "结肠癌", "colon", "乙状结肠", "升结肠", "横结肠", "降结肠", "肝曲", "脾曲", "回盲部"]

    for kw in rectum_full:
        if kw in location_str:
            return "rectum"

    for kw in colon_full:
        if kw in location_str:
            return "colon"

    # 模糊匹配（单个字符或简短词汇）
    # 直肠相关
    rectum_chars = ["直", " rectum", " rectal"]
    for kw in rectum_chars:
        if kw in location_str or kw in location_lower:
            return "rectum"

    # 结肠相关（单个字符）
    colon_chars = ["结", "升", "横", "降", "乙", "肝", "脾", "盲"]
    for kw in colon_chars:
        if kw in location_str:
            return "colon"

    return "unknown"


def _normalize_tnm_component(prefix: str, value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    if s.lower().startswith(prefix.lower()):
        return f"{prefix}{s[len(prefix):]}"
    match = re.match(r'^(?:[cCpPyY]?)([TNM])(.+)$', s, re.IGNORECASE)
    if match and match.group(1).lower() == prefix[1].lower():
        return f"{prefix}{match.group(2)}"
    return f"{prefix}{s}"


def _infer_tnm_from_text(text: object) -> dict:
    if not text:
        return {}
    s = str(text)
    t_match = re.search(r'\bT([0-4][a-c]?)\b', s, re.IGNORECASE)
    n_match = re.search(r'\bN([0-3][a-c]?)\b', s, re.IGNORECASE)
    m_match = re.search(r'\bM([01])\b', s, re.IGNORECASE)
    return {
        "T": t_match.group(1) if t_match else "",
        "N": n_match.group(1) if n_match else "",
        "M": m_match.group(1) if m_match else "",
    }


def _extract_molecular_markers(case_data: dict) -> dict:
    """从病例数据中提取分子标志物信息"""
    markers = {}

    # MMR/MSI 状态
    mmr_status = case_data.get('mmr_status', '')
    if isinstance(mmr_status, str):
        # 数据库返回的格式可能是 "pMMR (MSS)" 或 "dMMR (MSI-H)"
        if 'pMMR' in mmr_status or 'MSS' in mmr_status:
            markers["MSS"] = True
        elif 'dMMR' in mmr_status or 'MSI-H' in mmr_status:
            markers["MSI-H"] = True
    elif isinstance(mmr_status, int):
        # 原始数据是数字
        if mmr_status == 1:
            markers["MSS"] = True
        elif mmr_status == 2:
            markers["MSI-H"] = True

    # RAS 状态
    ras_status = case_data.get('ras_status', '')
    if 'wild' in str(ras_status).lower() or '野生' in str(ras_status):
        markers["RAS"] = "WildType"
    elif 'mutant' in str(ras_status).lower() or '突变' in str(ras_status):
        markers["RAS"] = "Mutant"

    # BRAF 状态
    braf_status = case_data.get('braf_status', '')
    if 'wild' in str(braf_status).lower() or '野生' in str(braf_status):
        markers["BRAF"] = "WildType"
    elif 'mutant' in str(braf_status).lower() or '突变' in str(braf_status):
        markers["BRAF"] = "Mutant"

    return markers


def node_case_database(model, tools=None, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    """
    病例数据库查询节点 (Refactored: Pure LLM Tool Calling)

    职责：
    1. 接收用户请求
    2. 使用 LLM 语义理解决定调用哪个原子工具
    3. 执行工具并生成 UI 卡片

    不再依赖正则表达式，让 LLM 自己决定调用哪个工具。
    """
    # 绑定工具到模型
    model_with_tools = model.bind_tools(ATOMIC_DATABASE_TOOLS)

    # 复用 CardFormatter 的卡片格式化逻辑
    helper = CardFormatter()

    def _run(state: CRCAgentState):
        user_text = _latest_user_text(state)

        if show_thinking:
            print(f"[Case Database] 接收指令: {user_text}")
        
        # [修复] 辅助函数：在返回前标记计划步骤为完成
        def _finalize_return(return_dict: dict) -> dict:
            """统一处理返回前的计划状态更新"""
            pending_step = get_current_pending_step(state)
            if pending_step:
                updated_plan = mark_step_completed(state, pending_step.id)
                return_dict["current_plan"] = updated_plan
                if show_thinking:
                    print(f"[Case Database] 标记步骤完成: [{pending_step.id}] {pending_step.description}")
            return return_dict

        # 从统一的 prompts 模块导入 System Prompt
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)
        sys_prompt = SystemMessage(
            content=DATABASE_QUERY_SYSTEM_PROMPT.format(
                summary_memory=summary_memory,
                pinned_context=pinned_context
            )
        )

        payload = _extract_json_payload(user_text)
        if payload and "patient_id" in payload:
            patient_id = _safe_int(payload.get("patient_id"))
            normalized_id = str(patient_id).zfill(3) if patient_id is not None else None
            findings_updates = dict(state.findings or {})
            if normalized_id:
                findings_updates["current_patient_id"] = normalized_id
            findings_updates["patient_record"] = payload
            findings_updates = _merge_database_workbench_findings(
                findings_updates,
                _build_database_workbench_context(
                    mode="edit",
                    query_text=user_text,
                    filters={"patient_id": patient_id} if patient_id is not None else None,
                    selected_patient_id=patient_id,
                ),
            )
            message = "已识别为病例编辑请求。请在数据库编辑表单中确认字段后再保存写回。"
            if normalized_id:
                message = f"{message}\n\n当前目标患者：{normalized_id}"
            return _finalize_return({
                "messages": [AIMessage(content=message)],
                "clinical_stage": "CaseDatabase",
                "findings": findings_updates,
                "current_patient_id": normalized_id,
            })
        if payload and ("受试者编号" in payload or "patient_id" in payload):
            tool_map = {tool.name: tool for tool in ATOMIC_DATABASE_TOOLS}
            upsert_tool = tool_map.get("upsert_patient_info")
            if upsert_tool:
                try:
                    tool_result = upsert_tool.invoke({"json_data": json.dumps(payload, ensure_ascii=False)})
                except Exception as e:
                    tool_result = {"error": f"写入失败: {str(e)}"}
            else:
                tool_result = {"error": "未找到写入工具"}

            subject_id = payload.get("受试者编号") or payload.get("patient_id")
            normalized_id = str(int(subject_id)).zfill(3) if subject_id is not None else None
            findings_updates = dict(state.findings or {})
            if normalized_id:
                findings_updates["current_patient_id"] = normalized_id
            findings_updates["patient_record"] = payload
            message = tool_result.get("message") if isinstance(tool_result, dict) else str(tool_result)
            if normalized_id:
                message = f"{message}\n\n已记录患者ID：{normalized_id}"
            return _finalize_return({
                "messages": [AIMessage(content=message)],
                "clinical_stage": "CaseDatabase",
                "findings": findings_updates,
                "current_patient_id": normalized_id,
            })

        try:
            # ============================================================
            # [Deterministic Tool Calling Patch]
            # Streamlit 前端场景中，LLM 偶发“不调用工具只泛化回答”，会导致：
            # - 未写入 ToolMessage（缺少可总结的病例数据）
            # - 未更新 current_patient_id（Intent Router 认为“没有活跃患者”）
            # 因此对“明确数据库查询/影像查询”做代码级别强制工具调用。
            # ============================================================

            def _extract_patient_id_from_text(text: str | None) -> str | None:
                """从文本中提取患者编号，避免把年龄等数字误判为ID。"""
                if not text:
                    return None

                # 1) 明确的“ID/编号/病例/患者”等上下文
                explicit_patterns = [
                    r"(?:患者|病人|病例|病历|编号|ID|id|case)\s*[:：#]?\s*(\d{1,3})",
                    r"(\d{1,3})\s*号\s*(?:患者|病人|病例|病历)",
                    r"(?:第|No\.?)\s*(\d{1,3})\s*号",
                ]
                for pattern in explicit_patterns:
                    m = re.search(pattern, text)
                    if m:
                        return m.group(1)

                # 2) 兜底：仅当出现“患者/病例”等关键词且数字不被“岁/年”修饰时才采用
                if any(kw in text for kw in ["患者", "病人", "病例", "病历"]):
                    m = re.search(r"(\d{1,3})(?!\s*岁)(?!\s*年)(?!\s*year)", text)
                    if m:
                        # 过滤典型年龄表达（如“58岁”“60 年龄”）
                        before = text[max(0, m.start() - 3):m.start()]
                        after = text[m.end():m.end() + 3]
                        if "岁" not in after and "年" not in after and "龄" not in before:
                            return m.group(1)

                fallback_match = re.search(r"\b(\d{1,3})\b", text)
                if fallback_match:
                    return fallback_match.group(1)

                return None

            def _infer_patient_id_from_history() -> str | None:
                """
                Streamlit 场景里 current_patient_id 可能未持久化/被 None 覆盖，
                但用户上一轮往往明确提过“93号/093”等。这里从历史 HumanMessage 回溯推断。
                """
                for msg in reversed(state.messages or []):
                    if isinstance(msg, HumanMessage):
                        inferred = _extract_patient_id_from_text(msg.content or "")
                        if inferred:
                            return inferred
                return None

            # 提取 patient_id（优先：本轮文本；其次：state.current_patient_id；再次：历史回溯）
            extracted_patient_id = _extract_patient_id_from_text(user_text or "")
            active_patient_id = (
                extracted_patient_id
                or (str(state.current_patient_id) if state.current_patient_id else None)
                or _infer_patient_id_from_history()
            )
            imaging_analysis_support_request = _is_imaging_analysis_support_request(state, user_text)

            # 关键词检测：是否明显是数据库查询 / 影像查询
            info_keywords = ["查询", "患者", "病例", "病历", "信息", "情况", "总结", "基本信息"]
            existing_info_keywords = [
                "已有信息",
                "已有资料",
                "现有信息",
                "现有资料",
                "目前有哪些数据",
                "目前有什么数据",
                "有哪些数据",
                "有什么数据",
                "都有哪些记录",
                "已有哪些记录",
            ]
            imaging_keywords = ["影像", "CT", "MRI", "图片", "片子", "影像信息", "影像分析报告", "影像组学", "影像分析"]
            pathology_keywords = ["病理", "切片", "WSI", "svs", "病理切片", "病理图像"]
            view_keywords = ["查看", "浏览", "列出", "查询", "显示"]
            is_obvious_db_query = any(kw in (user_text or "") for kw in info_keywords)
            is_existing_info_request = any(kw in (user_text or "") for kw in existing_info_keywords)
            needs_imaging = any(kw in (user_text or "") for kw in imaging_keywords) and not imaging_analysis_support_request
            needs_pathology = any(kw in (user_text or "") for kw in pathology_keywords)
            needs_pathology_view = needs_pathology and any(kw in (user_text or "") for kw in view_keywords)
            is_summary_request = any(kw in (user_text or "") for kw in ["总结", "概况", "目前情况", "病患情况", "情况"])

            # 如果用户明确提到“影像”，也视为数据库查询（需要患者上下文）
            if needs_imaging or needs_pathology or is_existing_info_request:
                is_obvious_db_query = True

            # 1. 首先检查是否需要强制调用肿瘤检测工具（代码级别检测）
            TUMOR_KEYWORDS = ["肿瘤检测", "癌症筛查", "病灶识别", "影像诊断", "CT检测", "检测肿瘤", "筛查肿瘤", "分析影像"]
            needs_tumor_check = any(kw in user_text for kw in TUMOR_KEYWORDS) and not imaging_analysis_support_request

            # 提取 patient_id（如果有）
            current_patient_id = active_patient_id

            response = None
            final_messages = []
            # 初始化卡片变量，避免未定义错误
            patient_card = None
            imaging_card = None
            summary_text = "查询完成。"
            case_result = None  # 避免未赋值引用（deterministic path 需要）

            if imaging_analysis_support_request and current_patient_id:
                case_tool = next((t for t in ATOMIC_DATABASE_TOOLS if getattr(t, "name", "") == "get_patient_case_info"), None)
                findings_updates = {}
                patient_profile_payload = None

                if case_tool:
                    case_result = case_tool.invoke({"patient_id": int(current_patient_id)})
                    if isinstance(case_result, dict) and "error" not in case_result:
                        card_data = helper.format_patient_card(case_result["patient_id"])
                        if isinstance(card_data, dict) and card_data.get("type") == "patient_card":
                            patient_card = card_data
                        patient_profile_payload = _build_patient_profile_payload(case_result)
                        findings_updates.update(_build_case_findings_update(case_result))

                return _finalize_return({
                    "messages": [],
                    "clinical_stage": "CaseDatabase",
                    "patient_profile": patient_profile_payload,
                    "patient_card": patient_card,
                    "imaging_card": None,
                    "pathology_slide_card": None,
                    "error": None,
                    "findings": _merge_database_workbench_findings(
                        findings_updates if findings_updates else None,
                        _build_database_workbench_context(
                            mode="detail",
                            query_text=user_text,
                            filters={"patient_id": _safe_int(current_patient_id)},
                            selected_patient_id=_safe_int(current_patient_id),
                        ),
                    ),
                    "current_patient_id": str(current_patient_id).zfill(3) if current_patient_id else None,
                })

            # 如果检测到肿瘤检测相关关键词，直接构造工具调用
            if needs_tumor_check and current_patient_id:
                patient_id_int = int(current_patient_id)

                tumor_tool = next((t for t in ATOMIC_DATABASE_TOOLS if hasattr(t, 'name') and t.name == "perform_comprehensive_tumor_check"), None)

                if tumor_tool:
                    try:
                        # 直接调用工具 - 肿瘤检测工具期望字符串类型的 patient_id
                        result = tumor_tool.invoke({"patient_id": current_patient_id})
                    except Exception as e:
                        result = {"error": str(e)}

                    # 生成 ToolMessage
                    tool_msg = ToolMessage(
                        tool_call_id="tumor_check_call",
                        content=str(result),
                        name="perform_comprehensive_tumor_check"
                    )
                    final_messages = [tool_msg]

                    # 初始化变量
                    tumor_detection_card = None
                    imaging_card = None

                    # 使用 CardFormatter 格式化完整的肿瘤检测结果（包含图片 base64）
                    if isinstance(result, dict) and not result.get("error"):
                        tumor_card = helper.format_comprehensive_tumor_detection(result, include_images=True)

                        # 生成自然语言总结
                        has_tumor = result.get("has_tumor", False)
                        images_with_tumor = result.get("images_with_tumor", 0)
                        total_images = result.get("total_images", 0)
                        detection_rate = result.get("tumor_detection_rate", "0%")
                        max_conf = result.get("max_confidence", 0)

                        status_str = "发现疑似肿瘤区域" if has_tumor else "未发现明显肿瘤"

                        summary_text = (
                            f"【肿瘤检测完成】\n"
                            f"患者ID: {result.get('patient_id')}\n"
                            f"检测结果: {status_str}\n"
                            f"检测率: {detection_rate} ({images_with_tumor}/{total_images} 张影像)\n"
                            f"最高置信度: {max_conf:.1%}"
                        )

                        # 生成肿瘤检测卡片（用于前端展示）
                        if isinstance(tumor_card, dict) and tumor_card.get("type") == "tumor_detection_card":
                            tumor_detection_card = tumor_card
                        else:
                            # 降级方案：生成简单的影像卡片
                            imaging_card = {
                                "type": "imaging_sample",
                                "data": {
                                    "folder_name": result.get("patient_id"),
                                    "images": [{
                                        "image_path": img.get("image_path"),
                                        "image_name": img.get("image_name", "detection.png")
                                    } for img in result.get("sample_images_with_tumor", [])],
                                    "total_images": len(result.get("sample_images_with_tumor", []))
                                },
                                "text_summary": summary_text
                            }
                    else:
                        summary_text = f"检测失败: {result.get('error', '未知错误')}"

                    # 返回结果 - 通过 additional_kwargs 传递肿瘤检测卡片
                    if tumor_detection_card:
                        ai_msg = AIMessage(content=summary_text)
                        ai_msg.additional_kwargs["tumor_detection_card"] = tumor_detection_card
                        final_messages.append(ai_msg)

                    return _finalize_return({
                        "messages": final_messages,
                        "clinical_stage": "CaseDatabase",
                        "patient_card": None,
                        "imaging_card": imaging_card,
                        # [关键] 写入强类型字段，供 Intent Router/后续轮次识别“该患者”
                        "current_patient_id": str(current_patient_id).zfill(3) if current_patient_id else None,
                        "radiomics_report_card": None,
                        "error": None,
                    })

            # [修复] 尝试解析用户意图是否为“添加/更新”操作，但格式不完全符合JSON
            # 例如： "添加3号病人信息，24岁"
            # 这种自然语言指令会被 LLM 识别为工具调用，但如果是 Deterministic 逻辑，需要在这里处理
            # 暂时不做复杂的 NLP，依赖 LLM 的 tool calling 能力
        
            # [增强] 检查是否是"添加/更新"操作的自然语言描述
            is_add_update_intent = any(kw in user_text for kw in ["添加", "更新", "修改", "录入", "补充"])
            if is_add_update_intent:
                # 如果是添加操作，跳过确定性逻辑，直接让 LLM 决定调用 upsert_patient_info
                # 这样可以处理 "添加3号病人信息，24岁" 这种非结构化输入
                target_patient_id = _safe_int(current_patient_id)
                findings_updates = {}
                if current_patient_id:
                    findings_updates["current_patient_id"] = str(current_patient_id).zfill(3)
                return _finalize_return({
                    "messages": [AIMessage(content="已进入病例编辑模式。请在数据库编辑表单中确认并点击保存后写回数据库。")],
                    "clinical_stage": "CaseDatabase",
                    "findings": _merge_database_workbench_findings(
                        findings_updates,
                        _build_database_workbench_context(
                            mode="edit" if target_patient_id is not None else "search",
                            query_text=user_text,
                            filters={"patient_id": target_patient_id} if target_patient_id is not None else None,
                            selected_patient_id=target_patient_id,
                        ),
                    ),
                    "current_patient_id": str(current_patient_id).zfill(3) if current_patient_id else None,
                })
            elif is_obvious_db_query and current_patient_id:
                # 尽量同时拉取“病例信息”和“影像信息”（当用户提到影像时）
                final_messages = []
                patient_card = None
                imaging_card = None
                pathology_slide_card = None
                tumor_detection_card = None
                radiomics_report_card = None  # 初始化影像组学报告卡片
                summary_text = "查询完成。"
                existing_info_result = None

                # 1) “已有信息/有哪些数据”优先走专用汇总工具；否则走病例信息工具
                if is_existing_info_request:
                    existing_info_tool = next(
                        (t for t in ATOMIC_DATABASE_TOOLS if getattr(t, "name", "") == "summarize_patient_existing_info"),
                        None,
                    )
                    if existing_info_tool:
                        existing_info_result = existing_info_tool.invoke({"patient_id": str(current_patient_id)})
                        final_messages.append(
                            ToolMessage(
                                tool_call_id="forced_summarize_patient_existing_info",
                                content=_format_database_tool_message("summarize_patient_existing_info", existing_info_result),
                                name="summarize_patient_existing_info",
                            )
                        )
                        if isinstance(existing_info_result, dict) and "error" not in existing_info_result:
                            case_result = existing_info_result.get("case_info")
                            summary_text = existing_info_result.get("summary", summary_text)
                            if isinstance(case_result, dict) and "error" not in case_result:
                                card_data = helper.format_patient_card(case_result["patient_id"])
                                if isinstance(card_data, dict) and card_data.get("type") == "patient_card":
                                    patient_card = card_data

                if case_result is None:
                    case_tool = next((t for t in ATOMIC_DATABASE_TOOLS if getattr(t, "name", "") == "get_patient_case_info"), None)
                    if case_tool:
                        case_result = case_tool.invoke({"patient_id": int(current_patient_id)})
                        final_messages.append(
                            ToolMessage(
                                tool_call_id="forced_get_patient_case_info",
                                content=_format_database_tool_message("get_patient_case_info", case_result),
                                name="get_patient_case_info",
                            )
                        )
                        if isinstance(case_result, dict) and "error" not in case_result:
                            card_data = helper.format_patient_card(case_result["patient_id"])
                            if isinstance(card_data, dict) and card_data.get("type") == "patient_card":
                                patient_card = card_data
                                # 默认一行提示即可；若是“总结/目前情况”请求，稍后会用更完整摘要覆盖
                                summary_text = card_data.get("text_summary", summary_text)

                # 2) 影像信息（仅当用户提到影像时）
                if needs_imaging:
                    img_tool = next((t for t in ATOMIC_DATABASE_TOOLS if getattr(t, "name", "") == "get_patient_imaging"), None)
                    if img_tool:
                        img_result = img_tool.invoke({"patient_id": str(current_patient_id)})
                        final_messages.append(
                            ToolMessage(
                                tool_call_id="forced_get_patient_imaging",
                                content=_format_database_tool_message("get_patient_imaging", img_result),
                                name="get_patient_imaging",
                            )
                        )
                        if isinstance(img_result, dict) and "error" not in img_result:
                            imaging_card = {
                                "type": "imaging_sample",
                                "data": img_result,
                                "text_summary": f"影像样本：患者 {str(current_patient_id).zfill(3)}，共 {img_result.get('total_images', 0)} 张影像",
                            }
                            # 如果本轮主要是影像查询，覆盖 summary
                            if needs_imaging:
                                summary_text = imaging_card["text_summary"]
                        else:
                            error_msg = img_result.get("error", "未知错误") if isinstance(img_result, dict) else "查询失败"
                            imaging_card = {
                                "type": "imaging_sample",
                                "data": {},
                                "text_summary": f"📷 影像样本：未找到患者 {str(current_patient_id).zfill(3)} 的影像文件",
                                "error": error_msg
                            }
                            summary_text += f"\n\n⚠️ 影像信息：{error_msg}"

                            # [新增] 检查是否存在影像组学分析报告
                            radiomics_report_card = None
                            try:
                                from pathlib import Path
                                # 检查影像组学分析报告路径
                                radiomics_report_path = None
                                # 数据库根目录
                                db_root = Path("data/Case Database/Radiographic Imaging")
                                patient_folder = db_root / str(current_patient_id).zfill(3)
                                radiomics_folder = patient_folder / "radiomics_analysis"
                                report_file = radiomics_folder / "comprehensive_analysis_report.json"

                                if report_file.exists():
                                    radiomics_report_path = str(report_file)

                                    # 读取报告文件
                                    with open(report_file, 'r', encoding='utf-8') as f:
                                        report_data = json.load(f)

                                    # 生成影像组学报告卡片
                                    radiomics_report_card = helper.format_radiomics_report_card(report_data, str(current_patient_id).zfill(3))
                            except Exception as e:
                                # 静默失败，不影响主流程
                                pass

                # 2.2) 病理切片信息（仅当用户提到病理切片且是查看类请求）
                if needs_pathology_view:
                    path_tool = next((t for t in ATOMIC_DATABASE_TOOLS if getattr(t, "name", "") == "get_patient_pathology_slides"), None)
                    if path_tool:
                        path_result = path_tool.invoke({"patient_id": str(current_patient_id)})
                        final_messages.append(
                            ToolMessage(
                                tool_call_id="forced_get_patient_pathology_slides",
                                content=_format_database_tool_message("get_patient_pathology_slides", path_result),
                                name="get_patient_pathology_slides",
                            )
                        )
                        if isinstance(path_result, dict) and "error" not in path_result:
                            pathology_slide_card = helper.format_pathology_slide_card(path_result)
                            summary_text = pathology_slide_card.get("text_summary", summary_text)
                        else:
                            error_msg = path_result.get("error", "未知错误") if isinstance(path_result, dict) else "查询失败"
                            pathology_slide_card = {
                                "type": "pathology_slide_card",
                                "data": {},
                                "text_summary": f"📋 病理切片样本：未找到患者 {str(current_patient_id).zfill(3)} 的病理切片文件",
                                "error": error_msg
                            }
                            summary_text += f"\n\n⚠️ 病理切片：{error_msg}"

                # 2.5) 如果用户明确要“总结/目前情况”，用更完整的摘要替换一行复读
                if is_summary_request and isinstance(case_result, dict) and "error" not in case_result:
                    summary_text = _format_case_summary_markdown(case_result)

                # 3) 生成最终 AIMessage（携带卡片）
                ai_msg = AIMessage(content=summary_text)
                if patient_card:
                    ai_msg.additional_kwargs["patient_card"] = patient_card
                if imaging_card:
                    ai_msg.additional_kwargs["imaging_card"] = imaging_card
                if pathology_slide_card:
                    ai_msg.additional_kwargs["pathology_slide_card"] = pathology_slide_card
                if tumor_detection_card:
                    ai_msg.additional_kwargs["tumor_detection_card"] = tumor_detection_card
                if radiomics_report_card:
                    ai_msg.additional_kwargs["radiomics_report_card"] = radiomics_report_card
                    # 更新 summary_text 以反映影像组学报告的存在
                    if "影像分析报告" in user_text:
                        summary_text = f"## 影像分析报告\n\n{summary_text}\n\n✅ **已生成影像组学分析卡片，包含完整分析结果（YOLO检测、U-Net分割、PyRadiomics特征提取、LASSO特征筛选）**"
                        ai_msg.content = summary_text
                final_messages.append(ai_msg)

                # 4) 提取并更新 findings（沿用原逻辑）
                findings_updates = {}
                if isinstance(case_result, dict) and "error" not in case_result:
                    inferred_from_tnm = _infer_tnm_from_text(case_result.get("tnm_stage"))
                    inferred_from_clinical = _infer_tnm_from_text(case_result.get("clinical_stage"))
                    inferred_t = inferred_from_tnm.get("T") or inferred_from_clinical.get("T")
                    inferred_n = inferred_from_tnm.get("N") or inferred_from_clinical.get("N")
                    inferred_m = inferred_from_tnm.get("M") or inferred_from_clinical.get("M")
                    cT = _normalize_tnm_component("cT", case_result.get("ct_stage")) or _normalize_tnm_component("cT", inferred_t)
                    cN = _normalize_tnm_component("cN", case_result.get("cn_stage")) or _normalize_tnm_component("cN", inferred_n)
                    cM = _normalize_tnm_component("cM", case_result.get("cm_stage")) or _normalize_tnm_component("cM", inferred_m)
                    extracted_clinical_data = _build_case_findings_update(case_result)
                    findings_updates.update(extracted_clinical_data)
                elif is_obvious_db_query and not current_patient_id:
                    # 明确是数据库总结/查询，但无法确定患者ID：直接追问，避免掉回 LLM 卡住
                    final_messages.append(AIMessage(content="请提供患者编号（例如 93 或 093），我才能总结该患者目前情况。"))

                return _finalize_return({
                    "messages": final_messages,
                    "clinical_stage": "CaseDatabase",
                    "patient_profile": _build_patient_profile_payload(case_result),
                    "patient_card": patient_card,
                    "imaging_card": imaging_card,
                    "pathology_slide_card": pathology_slide_card,
                    "error": None,
                    "findings": findings_updates if findings_updates else None,
                    # [关键] 写入强类型字段，供 Intent Router/后续轮次识别“该患者”
                    "radiomics_report_card": radiomics_report_card if "radiomics_report_card" in locals() else None,
                    "current_patient_id": str(current_patient_id).zfill(3) if current_patient_id else None,
                })

            # 如果没有检测到肿瘤检测关键词，让 LLM 决定
            # [关键修复] 激进截断消息历史，防止 token 超限
            truncated_messages = _truncate_message_history(
                state.messages,
                max_tokens=int(os.getenv("NODE_TRUNCATE_MAX_TOKENS", "15000")),
                keep_last_n=10,
                max_chars_per_message=2000
            )
            response = model_with_tools.invoke([sys_prompt] + truncated_messages)
            final_messages = [response] if response.tool_calls else []

            patient_card = None
            imaging_card = None
            pathology_slide_card = None
            tumor_detection_card = None
            radiomics_report_card = None  # 初始化影像组学报告卡片
            summary_text = "查询完成。"
            patient_profile_payload = None
            database_workbench_context = None

            # 2. 处理工具调用
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    name = tool_call["name"]
                    args = tool_call["args"]

                    # 查找并执行工具
                    selected_tool = next((t for t in ATOMIC_DATABASE_TOOLS if hasattr(t, 'name') and t.name == name), None)
                    if selected_tool:
                        try:
                            result = selected_tool.invoke(args)
                        except Exception as e:
                            result = f"Error: {str(e)}"

                        # 生成 ToolMessage
                        final_messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=_format_database_tool_message(name, result),
                            name=name
                        ))

                        # [新增] 提取临床关键信息并更新 findings
                        extracted_clinical_data = {}

                        # UI 卡片适配逻辑
                        if name == "get_patient_imaging":
                            if isinstance(result, dict) and "error" not in result:
                                imaging_card = {
                                    "type": "imaging_sample",
                                    "data": result,
                                    "text_summary": f"影像样本：患者 {args.get('patient_id')}，共 {result.get('total_images', 0)} 张影像"
                                }
                                summary_text = imaging_card["text_summary"]
                        elif name == "get_patient_pathology_slides":
                            if isinstance(result, dict) and "error" not in result:
                                pathology_slide_card = helper.format_pathology_slide_card(result)
                                summary_text = pathology_slide_card.get("text_summary", summary_text)

                        elif name == "get_database_statistics":
                            if isinstance(result, dict) and "error" not in result:
                                # 格式化数据库统计信息为友好的 Markdown
                                lines = []
                                lines.append("## 📊 数据库统计信息\n")
                                lines.append(f"### 总病例数")
                                lines.append(f"- **{result.get('total_cases', 0)}** 例\n")

                                # 性别分布
                                gender_dist = result.get('gender_distribution', {})
                                if gender_dist:
                                    lines.append("### 性别分布")
                                    lines.append(f"- 男性: {gender_dist.get('male', 0)} 人")
                                    lines.append(f"- 女性: {gender_dist.get('female', 0)} 人\n")

                                # 年龄统计
                                age_stats = result.get('age_statistics', {})
                                if age_stats:
                                    lines.append("### 年龄统计")
                                    lines.append(f"- 最小年龄: {age_stats.get('min', 'N/A')} 岁")
                                    lines.append(f"- 最大年龄: {age_stats.get('max', 'N/A')} 岁")
                                    lines.append(f"- 平均年龄: {age_stats.get('mean', 'N/A')} 岁\n")

                                # 肿瘤部位分布
                                location_dist = result.get('tumor_location_distribution', {})
                                if location_dist:
                                    lines.append("### 肿瘤部位分布")
                                    for location, count in location_dist.items():
                                        lines.append(f"- {location}: {count} 例")
                                    lines.append("")

                                # cT分期分布
                                ct_dist = result.get('ct_stage_distribution', {})
                                if ct_dist:
                                    lines.append("### cT分期分布")
                                    for stage, count in sorted(ct_dist.items()):
                                        lines.append(f"- cT{stage}: {count} 例")
                                    lines.append("")

                                # MMR状态分布
                                mmr_dist = result.get('mmr_status_distribution', {})
                                if mmr_dist:
                                    lines.append("### MMR状态分布")
                                    for status, count in mmr_dist.items():
                                        status_label = "pMMR/MSS" if status == 0 else "dMMR/MSI-H"
                                        lines.append(f"- {status_label}: {count} 例")
                                    lines.append("")

                                # CEA统计
                                cea_stats = result.get('cea_statistics', {})
                                if cea_stats:
                                    lines.append("### CEA水平统计")
                                    lines.append(f"- 最小值: {cea_stats.get('min', 'N/A')} ng/mL")
                                    lines.append(f"- 最大值: {cea_stats.get('max', 'N/A')} ng/mL")
                                    lines.append(f"- 平均值: {cea_stats.get('mean', 'N/A')} ng/mL\n")

                                summary_text = "\n".join(lines)
                                database_workbench_context = _build_database_workbench_context(
                                    mode="stats",
                                    query_text=user_text,
                                )

                        elif name == "get_patient_case_info":
                            if isinstance(result, dict) and "error" not in result:
                                # 复用 helper 的格式化逻辑
                                card_data = helper.format_patient_card(result['patient_id'])
                                if isinstance(card_data, dict) and card_data.get("type") == "patient_card":
                                    patient_card = card_data
                                    summary_text = card_data.get("text_summary", "查询完成")
                                patient_profile_payload = _build_patient_profile_payload(result)

                                # [核心修复] 提取关键临床信息到 findings
                                # 这些信息将被后续的 assessment 节点使用
                                extracted_clinical_data = {
                                    # 基础诊断信息
                                    "pathology_confirmed": True,  # 数据库中的病例都是已确诊的
                                    "biopsy_confirmed": True,

                                    # 肿瘤位置信息
                                    "tumor_location": _normalize_tumor_location(result.get('tumor_location', '')),

                                    # 病理类型
                                    "histology_type": result.get('histology_type', '腺癌'),

                                    # TNM 分期信息
                                    "tnm_staging": {
                                        "cT": f"cT{result.get('ct_stage', '')}",
                                        "cN": f"cN{result.get('cn_stage', '')}",
                                        "cM": f"cM{result.get('cm_stage', '')}"
                                    },

                                    # 分子标志物
                                    "molecular_markers": _extract_molecular_markers(result),

                                    # 临床分期总结
                                    "clinical_stage_summary": result.get('clinical_stage', ''),

                                    # [新增] 患者基本信息
                                    "patient_info": {
                                        "patient_id": result.get('patient_id'),
                                        "age": result.get('age'),
                                        "gender": result.get('gender'),
                                        "cea_level": result.get('cea_level'),
                                        "ecog_score": result.get('ecog_score'),
                                    },

                                    # [新增] 标记数据来源为数据库查询
                                    "data_source": "database_query",
                                    "db_query_patient_id": str(result.get('patient_id')),
                                }

                                # 如果有 MMR 状态，也保存
                                if result.get('mmr_status'):
                                    extracted_clinical_data["mmr_status"] = result.get('mmr_status')
                                    # 更新分子标志物
                                    if result.get('mmr_status') in ['pMMR', 'MSS']:
                                        extracted_clinical_data["molecular_markers"]["MSS"] = True
                                    elif result.get('mmr_status') in ['dMMR', 'MSI-H']:
                                        extracted_clinical_data["molecular_markers"]["MSI-H"] = True

                                # 更新患者ID（用于后续对话引用）
                                if result.get('patient_id'):
                                    # 记录当前活跃的患者ID，供 Router 参考
                                    current_patient_id = str(result.get('patient_id')).zfill(3)
                                    database_workbench_context = _build_database_workbench_context(
                                        mode="detail",
                                        query_text=user_text,
                                        filters={"patient_id": _safe_int(result.get("patient_id"))},
                                        selected_patient_id=_safe_int(result.get("patient_id")),
                                    )

                                if show_thinking:
                                    print(f"[Case Database] Extracted clinical data: location={extracted_clinical_data.get('tumor_location')}, "
                                          f"TNM={extracted_clinical_data.get('tnm_staging')}")

                        elif name == "summarize_patient_existing_info":
                            if isinstance(result, dict) and "error" not in result:
                                summary_text = result.get("summary", summary_text)
                                case_info = result.get("case_info")
                                if isinstance(case_info, dict) and "error" not in case_info:
                                    patient_profile_payload = _build_patient_profile_payload(case_info)
                                    card_data = helper.format_patient_card(case_info["patient_id"])
                                    if isinstance(card_data, dict) and card_data.get("type") == "patient_card":
                                        patient_card = card_data
                                    extracted_clinical_data = {
                                        "pathology_confirmed": True,
                                        "biopsy_confirmed": True,
                                        "tumor_location": _normalize_tumor_location(case_info.get("tumor_location", "")),
                                        "histology_type": case_info.get("histology_type", "腺癌"),
                                        "tnm_staging": {
                                            "cT": f"cT{case_info.get('ct_stage', '')}",
                                            "cN": f"cN{case_info.get('cn_stage', '')}",
                                            "cM": f"cM{case_info.get('cm_stage', '') or '0'}",
                                        },
                                        "molecular_markers": _extract_molecular_markers(case_info),
                                        "clinical_stage_summary": case_info.get("clinical_stage", ""),
                                        "patient_info": {
                                            "patient_id": case_info.get("patient_id"),
                                            "age": case_info.get("age"),
                                            "gender": case_info.get("gender"),
                                            "cea_level": case_info.get("cea_level"),
                                            "ecog_score": case_info.get("ecog_score"),
                                        },
                                        "data_source": "database_query",
                                        "db_query_patient_id": str(case_info.get("patient_id")),
                                    }
                                    current_patient_id = str(case_info.get("patient_id")).zfill(3)

                        elif name == "search_cases":
                            database_workbench_context = _build_database_workbench_context(
                                mode="search",
                                query_text=user_text,
                                filters=args if isinstance(args, dict) else None,
                            )
                            if isinstance(result, list) and len(result) > 0:
                                # 格式化搜索结果为友好的列表
                                lines = []
                                lines.append(f"## 🔍 病例搜索结果")
                                lines.append(f"\n找到 **{len(result)}** 例符合条件的患者：\n")

                                for idx, case in enumerate(result, 1):
                                    if isinstance(case, dict) and "error" not in case:
                                        lines.append(f"### {idx}. 患者 {case.get('patient_id', 'N/A')}")
                                        lines.append(f"- **性别/年龄**: {case.get('gender', 'N/A')} / {case.get('age', 'N/A')} 岁")
                                        lines.append(f"- **肿瘤部位**: {case.get('tumor_location', 'N/A')}")
                                        lines.append(f"- **病理类型**: {case.get('histology_type', 'N/A')}")
                                        lines.append(f"- **TNM分期**: cT{case.get('ct_stage', '')}N{case.get('cn_stage', '')}M{case.get('cm_stage', '') or '0'}")
                                        lines.append(f"- **临床分期**: {case.get('clinical_stage', 'N/A')}")
                                        lines.append(f"- **CEA水平**: {case.get('cea_level', 'N/A')} ng/mL")
                                        lines.append(f"- **MMR状态**: {case.get('mmr_status', 'N/A')}\n")

                                summary_text = "\n".join(lines)
                            else:
                                summary_text = "未找到符合条件的病例。"

                        elif name == "perform_comprehensive_tumor_check":
                            if isinstance(result, dict) and not result.get("error"):
                                # 使用 CardFormatter 格式化完整的肿瘤检测结果（包含图片 base64）
                                tumor_card = helper.format_comprehensive_tumor_detection(result, include_images=True)

                                # 生成自然语言总结
                                has_tumor = result.get("has_tumor", False)
                                images_with_tumor = result.get("images_with_tumor", 0)
                                total_images = result.get("total_images", 0)
                                detection_rate = result.get("tumor_detection_rate", "0%")
                                max_conf = result.get("max_confidence", 0)

                                status_str = "发现疑似肿瘤区域" if has_tumor else "未发现明显肿瘤"

                                summary_text = (
                                    f"【肿瘤检测完成】\n"
                                    f"患者ID: {result.get('patient_id')}\n"
                                    f"检测结果: {status_str}\n"
                                    f"检测率: {detection_rate} ({images_with_tumor}/{total_images} 张影像)\n"
                                    f"最高置信度: {max_conf:.1%}"
                                )

                                # 生成肿瘤检测卡片（用于前端展示）
                                if isinstance(tumor_card, dict) and tumor_card.get("type") == "tumor_detection_card":
                                    # 使用 tumor_detection_card 传递给前端
                                    tumor_detection_card = tumor_card
                                    imaging_card = None  # 不再使用旧的 imaging_card
                                else:
                                    # 降级方案：生成简单的影像卡片
                                    tumor_detection_card = None
                                    imaging_card = {
                                        "type": "imaging_sample",
                                        "data": {
                                            "folder_name": result.get("patient_id"),
                                            "images": [{
                                                "image_path": img.get("image_path"),
                                                "image_name": img.get("image_name", "detection.png")
                                            } for img in result.get("sample_images_with_tumor", [])],
                                            "total_images": len(result.get("sample_images_with_tumor", []))
                                        },
                                        "text_summary": summary_text
                                    }
                            else:
                                summary_text = f"检测失败: {result.get('error', '未知错误')}"

                    else:
                        # 工具未找到的错误处理
                        error_msg = f"未知工具: {name}"
                        final_messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=error_msg,
                            name=name
                        ))

            # 3. 生成总结消息和卡片
            if patient_card or imaging_card or pathology_slide_card or tumor_detection_card:
                ai_msg = AIMessage(content=summary_text)
                if patient_card:
                    ai_msg.additional_kwargs["patient_card"] = patient_card
                if imaging_card:
                    ai_msg.additional_kwargs["imaging_card"] = imaging_card
                if pathology_slide_card:
                    ai_msg.additional_kwargs["pathology_slide_card"] = pathology_slide_card
                if tumor_detection_card:
                    ai_msg.additional_kwargs["tumor_detection_card"] = tumor_detection_card
                if radiomics_report_card:
                    ai_msg.additional_kwargs["radiomics_report_card"] = radiomics_report_card
                final_messages.append(ai_msg)
            elif not response.tool_calls:
                # LLM 没有调用任何工具（可能是闲聊或无法识别）
                if response.content:
                    final_messages.append(response)
                else:
                    final_messages.append(AIMessage(content="无法识别查询意图，请尝试使用ID查询或查看数据库统计。"))

            # [修复] 初始化 extracted_clinical_data，确保在所有代码路径下都有定义
            # 即使没有工具调用，这个变量也需要存在
            if 'extracted_clinical_data' not in locals():
                extracted_clinical_data = {}

            # [核心修复] 构建返回的 findings，携带提取的临床信息
            # 这样后续的 assessment 节点就能访问数据库查询的结果
            findings_updates = {}

            # 如果有提取的临床数据，添加到 findings 中
            if extracted_clinical_data:
                findings_updates.update(extracted_clinical_data)

            # 如果有患者ID，更新 current_patient_id
            if 'current_patient_id' in locals() and current_patient_id:
                findings_updates['current_patient_id'] = current_patient_id

            fallback_patient_match = re.search(r"(\d{1,3})", user_text or "")
            fallback_patient_id = current_patient_id or (fallback_patient_match.group(1) if fallback_patient_match else None)

            if database_workbench_context is None and fallback_patient_id:
                database_workbench_context = _build_database_workbench_context(
                    mode="detail",
                    query_text=user_text,
                    filters={"patient_id": _safe_int(fallback_patient_id)},
                    selected_patient_id=_safe_int(fallback_patient_id),
                )

            return _finalize_return({
                "messages": final_messages,
                "clinical_stage": "CaseDatabase",
                "patient_profile": patient_profile_payload,
                "patient_card": patient_card,
                "imaging_card": imaging_card,
                "pathology_slide_card": pathology_slide_card,
                "error": None,
                # [核心] 将提取的临床信息传递到 findings
                "findings": _merge_database_workbench_findings(
                    findings_updates if findings_updates else None,
                    database_workbench_context,
                ),
                # [关键] 写入强类型字段，供 Intent Router/后续轮次识别“该患者”
                "current_patient_id": str(current_patient_id).zfill(3) if current_patient_id else None,
            })


        except Exception as e:
            error_msg = f"查询执行异常: {str(e)}"
            print(f"[Case Database Error] {error_msg}")
            traceback.print_exc()
            return _finalize_return({
                "messages": [AIMessage(content=error_msg)],
                "clinical_stage": "CaseDatabase",
                "error": str(e),
            })

    return _run
