"""
Planner Node - 自主规划节点

负责将复杂任务拆解为可执行的原子步骤，实现主动上下文构建 (Active Context)。
替代传统的硬编码正则表达式路由，让 LLM 自己决定下一步要做什么。
"""

import re
import time
from typing import List, Dict, Any, Callable
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from ..state import CRCAgentState, PlanStep
from ..prompts import (
    PLANNER_SYSTEM_PROMPT,
    SELF_CORRECTION_PROMPT_TEMPLATE,
    PLANNING_USER_PROMPT_TEMPLATE,
    MULTI_TASK_USER_PROMPT_TEMPLATE,
)


def _latest_human_text(state: CRCAgentState) -> str:
    for msg in reversed(state.messages or []):
        if getattr(msg, "type", "") == "human":
            return getattr(msg, "content", "") or ""
    return ""


def _extract_inline_tnm(text: str) -> Dict[str, str]:
    raw = str(text or "")
    compact = re.sub(r"\s+", "", raw)

    combined = re.search(
        r"([cpr]?T[0-4Xx](?:is|[A-Ca-c])?)([cpr]?N[0-3Xx](?:[A-Ca-c])?)([cpr]?M[01Xx])",
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
    for prefix in ("T", "N", "M"):
        match = re.search(rf"([cpr]?{prefix}[0-4Xx](?:is|[A-Ca-c])?)", raw, re.IGNORECASE)
        if match:
            extracted[f"c{prefix}"] = match.group(1)
    return extracted


def _has_complete_tnm(tnm: Dict[str, Any]) -> bool:
    if not isinstance(tnm, dict):
        return False

    def _value(*keys: str) -> str:
        for key in keys:
            value = tnm.get(key)
            if value:
                return str(value).strip()
        return ""

    return bool(_value("T", "cT") and _value("N", "cN") and _value("M", "cM"))


def _has_inline_case_minimum_context(text: str) -> Dict[str, bool]:
    lowered = str(text or "").lower()
    pathology_confirmed = any(
        keyword in lowered
        for keyword in ["病理", "活检", "腺癌", "adenocarcinoma", "中分化", "低分化", "高分化"]
    )
    tnm = _extract_inline_tnm(text)
    return {
        "pathology_confirmed": pathology_confirmed,
        "tnm_complete": _has_complete_tnm(tnm),
    }


def _get_profile_summary(state: CRCAgentState) -> str:
    """
    生成患者档案摘要
    
    优先级：
    1. 如果有数据库查询数据（findings.data_source == 'database_query'），优先使用
    2. 否则使用 patient_profile
    """
    findings = state.findings or {}
    profile = state.patient_profile
    
    # [优先] 检查是否有数据库查询数据
    if findings.get("data_source") == "database_query":
        summary_parts = []
        
        # 从 findings 中提取数据库查询信息
        patient_info = findings.get("patient_info", {})
        tnm_staging = findings.get("tnm_staging", {})
        
        # 患者 ID
        patient_id = findings.get("db_query_patient_id") or findings.get("current_patient_id")
        if patient_id:
            summary_parts.append(f"🆔 患者ID: {patient_id}")
        
        # 基本信息
        if patient_info:
            age = patient_info.get("age")
            gender = patient_info.get("gender")
            if age and gender:
                summary_parts.append(f"基本信息: {gender}/{age}岁")
        
        # 肿瘤部位
        tumor_location = findings.get("tumor_location")
        histology = findings.get("histology_type")
        if tumor_location or histology:
            tumor_desc = f"肿瘤: {tumor_location or ''} {histology or ''}".strip()
            summary_parts.append(tumor_desc)
        
        # 病理确认
        if findings.get("pathology_confirmed"):
            summary_parts.append("✓ 病理已确认")
        
        # TNM 分期
        if isinstance(tnm_staging, dict) and tnm_staging:
            tnm_str = " ".join([f"{k}:{v}" for k, v in tnm_staging.items() if v])
            summary_parts.append(f"TNM分期: {tnm_str}")
        
        # 分子标志物
        molecular = findings.get("molecular_markers", {})
        if molecular:
            if molecular.get("MSS"):
                summary_parts.append("MMR状态: pMMR/MSS")
            elif molecular.get("MSI-H"):
                summary_parts.append("MMR状态: dMMR/MSI-H")
        
        # 数据来源标记
        summary_parts.append("📊 数据来源: 数据库查询")
        
        if not summary_parts:
            return "❌ 数据库查询数据解析失败"
        
        return "\n".join(summary_parts)
    
    # [兜底] 使用 patient_profile
    if not profile:
        return "❌ 患者档案未建立"
    
    summary_parts = []
    
    # 基础信息
    if profile.tumor_type and profile.tumor_type != "Unknown":
        summary_parts.append(f"肿瘤类型: {profile.tumor_type}")
    
    if hasattr(profile, 'primary_site') and profile.primary_site:
        summary_parts.append(f"原发部位: {profile.primary_site}")
    
    # 病理确认
    if profile.pathology_confirmed:
        summary_parts.append("✓ 病理已确认")
    else:
        summary_parts.append("⚠️  病理未确认")
    
    # TNM 分期
    tnm = profile.tnm_staging or {}
    if isinstance(tnm, dict) and tnm:
        tnm_str = " ".join([f"{k}:{v}" for k, v in tnm.items() if v])
        summary_parts.append(f"TNM分期: {tnm_str}")
    else:
        summary_parts.append("⚠️  TNM分期未确定")
    
    # 分子标志物
    if profile.mmr_status and profile.mmr_status != "Unknown":
        summary_parts.append(f"MMR状态: {profile.mmr_status}")
    
    # Profile 锁定状态
    if profile.is_locked:
        summary_parts.append("🔒 档案已锁定（信息完整）")
    else:
        summary_parts.append("🔓 档案开放（可能需要补充信息）")
    
    if not summary_parts:
        return "❌ 患者档案为空"
    
    return "\n".join(summary_parts)


def _get_user_intent_summary(state: CRCAgentState) -> str:
    """提取用户意图摘要（支持多任务）"""
    findings = state.findings or {}
    intent = findings.get("user_intent", "unknown")
    
    # 获取最近的用户消息
    user_msg = ""
    if state.messages:
        for msg in reversed(state.messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                user_msg = msg.content
                break
    
    # [新增] 检查是否为多任务模式
    is_multi_task = findings.get("multi_task_mode", False)
    sub_tasks = findings.get("sub_tasks", [])
    
    if is_multi_task and sub_tasks:
        summary = f"意图类型: multi_task (多任务模式)\n"
        summary += f"包含子任务 ({len(sub_tasks)}个):\n"
        for i, task in enumerate(sub_tasks, 1):
            summary += f"  {i}. {task}\n"
        summary += f"用户问题: {user_msg}"
        return summary
    else:
        return f"意图类型: {intent}\n用户问题: {user_msg}"


def _safe_format(template: str, **kwargs) -> str:
    """
    Safe formatter to avoid KeyError on literal braces and format specifier errors.

    This function uses a regex-based approach to replace format keys without
    interpreting braces in JSON examples or other literal contexts.
    """
    import re

    result = template
    for key, value in kwargs.items():
        # Replace {key} with the actual value
        # Use word boundaries to avoid partial matches
        pattern = r'\{' + re.escape(key) + r'\}'
        result = re.sub(pattern, str(value), result)

    return result


def _detect_missing_context(state: CRCAgentState) -> Dict[str, str]:
    """
    识别当前意图下必需但缺失的关键信息，
    返回 {缺失字段名: 建议使用的工具} 映射，用于触发 Active Context Step。
    
    这是"主动上下文规划"的核心：在 LLM 规划前先诊断上下文是否足够。
    """
    findings = state.findings or {}
    intent = findings.get("user_intent", "")
    profile = state.patient_profile
    latest_user_text = _latest_human_text(state)
    inline_ctx = _has_inline_case_minimum_context(latest_user_text)

    missing: Dict[str, str] = {}

    # 1. 治疗决策：需要基本分期与病理确认
    if intent == "treatment_decision":
        if not ((profile and profile.pathology_confirmed) or inline_ctx["pathology_confirmed"]):
            missing["pathology_confirmed"] = "case_database_query"
        tnm = (profile and profile.tnm_staging) or findings.get("tnm_staging") or {}
        if not (_has_complete_tnm(tnm) or inline_ctx["tnm_complete"]):
            missing["tnm_staging"] = "case_database_query"

    # 2. imaging_query：至少要有 patient_id 或检查号
    if intent == "imaging_query":
        patient_id = findings.get("db_query_patient_id") or findings.get("current_patient_id")
        if not patient_id:
            missing["patient_id"] = "case_database_query"

    # 3. multi_task：需要结构化子任务列表
    if findings.get("multi_task_mode", False):
        sub_tasks = findings.get("sub_tasks") or []
        if not sub_tasks:
            missing["sub_tasks"] = "ask_user"  # 让 LLM/用户澄清子任务列表

    # 4. 临床评估：需要病理确认和基本分期
    if intent == "clinical_assessment":
        if not ((profile and profile.pathology_confirmed) or inline_ctx["pathology_confirmed"]):
            missing["pathology_confirmed"] = "case_database_query"
        tnm = (profile and profile.tnm_staging) or findings.get("tnm_staging") or {}
        if not (_has_complete_tnm(tnm) or inline_ctx["tnm_complete"]):
            missing["tnm_staging"] = "case_database_query"

    # 5. 影像分析：需要患者ID和影像检查信息
    if intent == "imaging_analysis":
        patient_id = findings.get("db_query_patient_id") or findings.get("current_patient_id")
        if not patient_id:
            missing["patient_id"] = "case_database_query"
        imaging_id = findings.get("imaging_id") or findings.get("current_imaging_id")
        if not imaging_id:
            missing["imaging_id"] = "case_database_query"

    # 6. 病理分析：需要患者ID和病理检查信息
    if intent == "pathology_analysis":
        patient_id = findings.get("db_query_patient_id") or findings.get("current_patient_id")
        if not patient_id:
            missing["patient_id"] = "case_database_query"
        pathology_id = findings.get("pathology_id") or findings.get("current_pathology_id")
        if not pathology_id:
            missing["pathology_id"] = "case_database_query"

    return missing


def _should_skip_planning(state: CRCAgentState) -> tuple[bool, str]:
    """
    判断是否应该跳过规划（三层判断架构）
    
    Layer 0: 快捷跳过（闲聊 / 简单查询 / Fast Pass）
    Layer 1: 上下文诊断（是否先补全 context）
    Layer 2: 真正 LLM 规划 + 自我纠错
    
    返回 (should_skip, reason) 元组：
    - should_skip=True: 跳过规划，直接返回
    - should_skip=False: 需要进入规划流程（包括上下文填充或 LLM 规划）
    """
    current_plan = state.current_plan or []
    findings = state.findings or {}
    current_intent = findings.get("user_intent", "")

    # =================================================================
    # Layer 0: 快捷跳过 - 闲聊/简单查询/寒暄场景
    # =================================================================
    if current_intent in ["general_chat", "greeting", "thanks", "off_topic_redirect",
                          "case_database_query", "imaging_query", "knowledge_query"]:
        return True, f"简单查询/寒暄场景 ({current_intent})，无需规划"

    # =================================================================
    # Layer 2: 检查失败步骤 → 进入自我纠错，不跳过
    # =================================================================
    has_failed = any(s.status == 'failed' for s in current_plan)
    if has_failed:
        return False, "检测到失败步骤，需要重新规划（自我纠错）"

    # =================================================================
    # Layer 2: 检查意图变化 → 若非自然过渡则需要重新规划
    # =================================================================
    previous_intent = getattr(state, '_previous_intent', None) or findings.get("_previous_intent", "")
    if previous_intent and previous_intent != current_intent:
        natural_transitions = [
            ("clinical_assessment", "treatment_decision"),
            ("knowledge_query", "treatment_decision"),
        ]
        is_natural = (previous_intent, current_intent) in natural_transitions

        if not is_natural:
            return False, f"用户意图变化：{previous_intent} -> {current_intent}，需要重新规划"

    # =================================================================
    # Layer 0: 检查是否有未完成步骤 → 保持专注，跳过
    # =================================================================
    if current_plan and any(s.status == 'pending' for s in current_plan):
        return True, "当前有未完成的计划步骤"

    # =================================================================
    # Layer 0: Fast Pass - profile 已锁且是治疗决策 → 跳过
    # =================================================================
    if current_intent == "treatment_decision" and state.patient_profile and state.patient_profile.is_locked:
        return True, "检测到 Fast Pass 条件"
    
    # =================================================================
    # Layer 0: 治疗决策意图且无数据库查询需求 → 跳过（交给下游 SubAgent）
    # =================================================================
    if current_intent == "treatment_decision":
        user_msg = ""
        if state.messages:
            for msg in reversed(state.messages):
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_msg = msg.content.lower()
                    break
        
        needs_db_query = any(kw in user_msg for kw in ["患者", "病人", "号", "id", "查询", "数据库"])
        if not needs_db_query:
            return True, "治疗决策意图，由 SubAgent 批量检索"

    # =================================================================
    # Layer 1: 上下文缺失检测 → 不跳过，而是进入上下文填充阶段
    # =================================================================
    missing_ctx = _detect_missing_context(state)
    if missing_ctx:
        return False, f"上下文缺失字段: {', '.join(missing_ctx.keys())}"

    # =================================================================
    # Layer 2: 默认进入 LLM 规划
    # =================================================================
    return False, ""


def _user_explicitly_requests_case_lookup(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False

    lowered = raw.lower()
    explicit_phrases = [
        "查询病例",
        "检索病例",
        "搜索病例",
        "调取病例",
        "读取病例",
        "查询数据库",
        "检索数据库",
        "搜索数据库",
        "调取数据库",
        "从数据库",
        "病例库",
        "病历库",
        "case database",
        "query database",
        "search database",
        "look up case",
    ]
    if any(phrase in lowered for phrase in explicit_phrases):
        return True

    patterns = [
        r"(查询|检索|搜索|调取|读取).{0,12}(病例|病历|数据库|病例库|病历库|编号)",
        r"(根据|按|用).{0,12}(编号|病历号|病例号|patient\s*id).{0,12}(查询|检索|搜索|调取|读取)",
        r"\b(query|search|lookup|look up|fetch|load|retrieve)\b.{0,20}\b(case|record|database|patient id|case id)\b",
    ]
    return any(re.search(pattern, raw, re.IGNORECASE) for pattern in patterns)


def _should_skip_planning_v2(state: CRCAgentState) -> tuple[bool, str]:
    current_plan = state.current_plan or []
    findings = state.findings or {}
    current_intent = findings.get("user_intent", "")

    if current_intent in [
        "general_chat",
        "greeting",
        "thanks",
        "off_topic_redirect",
        "case_database_query",
        "imaging_query",
        "knowledge_query",
    ]:
        return True, f"skip planning for intent={current_intent}"

    if any(step.status == "failed" for step in current_plan):
        return False, "failed plan steps require replanning"

    previous_intent = getattr(state, "_previous_intent", None) or findings.get("_previous_intent", "")
    if previous_intent and previous_intent != current_intent:
        natural_transitions = {
            ("clinical_assessment", "treatment_decision"),
            ("knowledge_query", "treatment_decision"),
        }
        if (previous_intent, current_intent) not in natural_transitions:
            return False, f"intent changed: {previous_intent} -> {current_intent}"

    if current_plan and any(step.status == "pending" for step in current_plan):
        return True, "existing pending plan can continue"

    if current_intent == "treatment_decision" and state.patient_profile and state.patient_profile.is_locked:
        return True, "fast pass with locked profile"

    if current_intent == "treatment_decision":
        user_msg = _latest_human_text(state)
        if not _user_explicitly_requests_case_lookup(user_msg):
            return True, "treatment decision does not require planning"

    missing_ctx = _detect_missing_context(state)
    if missing_ctx:
        return False, f"missing context: {', '.join(missing_ctx.keys())}"

    return False, ""


# === Planner Node ===

def node_planner(
    model: BaseChatModel,
    streaming: bool = False,
    show_thinking: bool = False,
) -> Callable[[CRCAgentState], Dict[str, Any]]:
    """
    规划节点工厂函数（主动上下文规划版）
    
    核心流程：
    1. Layer 0: 快速跳过检查（闲聊 / Fast Pass）
    2. Layer 1: 上下文诊断 - 若有缺失，生成 Active Context Steps
    3. Layer 2: LLM 规划 + 自我纠错 + 后处理
    
    Args:
        model: LLM 模型
        streaming: 是否启用流式输出
        show_thinking: 是否显示思考过程
    
    Returns:
        规划节点函数
    """
    
    def _planner_node(state: CRCAgentState) -> Dict[str, Any]:
        started_at = time.perf_counter()

        def _with_timing(payload: Dict[str, Any] | None) -> Dict[str, Any]:
            result = dict(payload or {})
            timings = list(result.get("node_timings") or [])
            timings.append({
                "node": "planner",
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            })
            result["node_timings"] = timings
            return result
        """
        规划节点主逻辑（主动上下文规划版）
        
        职责：
        1. 快速跳过检查（闲聊 / Fast Pass）
        2. 上下文诊断：检测缺失信息，生成 Active Context Steps
        3. LLM 规划：生成计划步骤
        4. 后处理：意图过滤、工具修正、并行组清洗
        5. 更新 scratchpad 记录思考过程
        """
        # =================================================================
        # Layer 0: 快速跳过检查（静默）
        # =================================================================
        should_skip, skip_reason = _should_skip_planning_v2(state)
        if should_skip:
            return _with_timing({})
        
        # 以下是正常规划流程，需要输出日志
        print("\n" + "="*60)
        print("[Planner] 规划节点启动")
        if skip_reason:
            print(f"  原因: {skip_reason}")
        print("="*60)
        
        # =================================================================
        # 熔断保护：检查迭代次数
        # =================================================================
        iteration = state.plan_iteration_count or 0
        if iteration >= 5:
            print(f"[Planner] ⚠️ 规划迭代次数达到上限 ({iteration})，停止重新规划")
            scratchpad_entry = f"\n[熔断] 规划迭代超过 {iteration} 次，停止重新规划\n"
            return _with_timing({
                "scratchpad": (state.scratchpad or "") + scratchpad_entry,
                "current_plan": [],
            })
        
        # =================================================================
        # Layer 1: 上下文诊断 - 优先生成 Active Context Steps
        # =================================================================
        missing_ctx = _detect_missing_context(state)
        if missing_ctx:
            print(f"\n[Planner] 🔍 上下文诊断：发现缺失字段 {list(missing_ctx.keys())}")
            
            # 用规则生成若干 PlanStep（不调用 LLM）
            ctx_steps: List[PlanStep] = []
            for field_name, tool_name in missing_ctx.items():
                step = PlanStep(
                    id=f"ctx_{field_name}",
                    description=f"补全关键上下文字段: {field_name}",
                    tool_needed=tool_name,
                    status="pending",
                    parallel_group="ctx_build",
                    reasoning=f"该字段为当前意图必需，来自 Active Context 诊断"
                )
                ctx_steps.append(step)

            # 将 context 步骤插入到现有计划前面
            current_plan = state.current_plan or []
            new_plan = ctx_steps + current_plan

            scratchpad_entry = f"\n[上下文构建] 发现缺失字段: {list(missing_ctx.keys())}，插入 {len(ctx_steps)} 个上下文填充步骤\n"
            print(f"[Planner] {scratchpad_entry.strip()}")
            
            return _with_timing({
                "current_plan": new_plan,
                "scratchpad": (state.scratchpad or "") + scratchpad_entry,
                "plan_iteration_count": iteration + 1,
            })
        
        # =================================================================
        # Layer 2: LLM 规划流程
        # =================================================================
        
        # 2.1 检查是否有失败的步骤（自我纠错模式）
        current_plan = state.current_plan or []
        failed_steps = [s for s in current_plan if s.status == 'failed']
        is_self_correction = len(failed_steps) > 0
        
        # 2.2 准备上下文
        profile_summary = _get_profile_summary(state)
        intent_summary = _get_user_intent_summary(state)
        
        print(f"\n[Planner] 患者档案状态:\n{profile_summary}")
        print(f"\n[Planner] 用户意图:\n{intent_summary}")
        
        if is_self_correction:
            print(f"\n[Planner] 🔄 自我纠错模式：检测到 {len(failed_steps)} 个失败步骤")
            for step in failed_steps:
                print(f"  ❌ [{step.id}] {step.description}")
                print(f"     错误: {step.error_message}")
                print(f"     重试次数: {step.retry_count}")
        
        # 2.3 构建 Planner Prompt（使用辅助函数）
        system_msg, user_msg = _build_planner_messages(
            state=state,
            profile_summary=profile_summary,
            intent_summary=intent_summary,
            failed_steps=failed_steps,
            is_self_correction=is_self_correction
        )
        
        # 2.4 调用 LLM 生成计划
        try:
            from pydantic import BaseModel as PydanticBaseModel, Field, AliasChoices
            
            class PlanOutput(PydanticBaseModel):
                """规划输出"""
                plan_steps: List[PlanStep] = Field(
                    ...,
                    validation_alias=AliasChoices('plan_steps', 'plan', 'PlanStep'),
                    description="计划步骤列表"
                )
            
            planner = model.with_structured_output(PlanOutput)
            result = planner.invoke([system_msg, user_msg])
            raw_plan = result.plan_steps if result else []
            
            if not raw_plan:
                print("[Planner] 无需规划，返回空计划")
                return _with_timing({})
            
            # 2.5 规划后处理（使用辅助函数）
            new_plan = _post_process_plan(
                raw_plan=raw_plan,
                state=state,
                show_thinking=show_thinking
            )
            
            # 检查过滤后是否为空
            if not new_plan:
                print("[Planner] 治疗决策意图，过滤后无需执行计划（知识检索由 SubAgent 负责）")
                return _with_timing({})
            
            # 2.6 输出计划
            print(f"\n[Planner] 生成计划 ({len(new_plan)} 步骤):")
            for step in new_plan:
                print(f"  • [{step.id}] {step.description}")
                assignee = getattr(step, "assignee", None)
                group = getattr(step, "parallel_group", None)
                assignee_info = f" | 执行者: {assignee}" if assignee else ""
                group_info = f" | 并行组: {group}" if group else ""
                print(f"    工具: {step.tool_needed}{assignee_info}{group_info} | 推理: {step.reasoning}")
            
            # 2.7 更新 scratchpad
            if is_self_correction:
                scratchpad_entry = f"\n[自我纠错 - 迭代 {iteration + 1}]\n"
                scratchpad_entry += f"失败步骤: {[s.id for s in failed_steps]}\n"
                scratchpad_entry += f"新计划: {len(new_plan)} 个步骤\n"
            else:
                scratchpad_entry = f"\n[规划 - {intent_summary.split(':')[0]}]\n"
                scratchpad_entry += f"档案状态: {profile_summary.split(chr(10))[0]}\n"
                scratchpad_entry += f"生成计划: {len(new_plan)} 个步骤\n"
            
            new_scratchpad = (state.scratchpad or "") + scratchpad_entry
            
            # 2.8 返回更新
            print("\n[Planner] 规划完成，更新状态")
            print("="*60 + "\n")
            
            # 保存当前意图，用于下次检测意图变化
            current_intent = (state.findings or {}).get("user_intent", "")
            findings_update = {"_previous_intent": current_intent} if current_intent else {}
            
            updates = {
                "current_plan": new_plan,
                "scratchpad": new_scratchpad,
                "plan_iteration_count": iteration + 1,
            }
            if findings_update:
                updates["findings"] = findings_update
            return _with_timing(updates)
            
        except Exception as e:
            print(f"[Planner] 错误: {str(e)}")
            return _with_timing({
                "scratchpad": (state.scratchpad or "") + f"\n[规划错误] {str(e)}\n",
                "plan_iteration_count": iteration + 1,
            })
    
    return _planner_node


# === Helper Functions ===

def get_current_pending_step(state: CRCAgentState) -> PlanStep | None:
    """
    获取当前待执行的计划步骤
    
    Returns:
        第一个状态为 pending 的步骤，如果没有则返回 None
    """
    plan = state.current_plan or []
    for step in plan:
        if step.status == "pending":
            return step
    return None


def mark_step_completed(state: CRCAgentState, step_id: str) -> List[PlanStep]:
    """标记某个步骤为已完成"""
    plan = state.current_plan or []
    updated_plan = []
    for step in plan:
        if step.id == step_id:
            step.status = "completed"
        updated_plan.append(step)
    return updated_plan


def mark_step_in_progress(state: CRCAgentState, step_id: str) -> List[PlanStep]:
    """标记某个步骤为进行中"""
    plan = state.current_plan or []
    updated_plan = []
    for step in plan:
        if step.id == step_id:
            step.status = "in_progress"
        updated_plan.append(step)
    return updated_plan


def mark_step_failed(state: CRCAgentState, step_id: str, error_message: str) -> List[PlanStep]:
    """
    标记某个步骤为失败
    
    Args:
        state: 当前状态
        step_id: 失败步骤的 ID
        error_message: 错误信息
    
    Returns:
        更新后的计划列表
    """
    plan = state.current_plan or []
    updated_plan = []
    for step in plan:
        if step.id == step_id:
            step.status = "failed"
            step.error_message = error_message
            step.retry_count += 1
        updated_plan.append(step)
    return updated_plan


def has_too_many_retries(state: CRCAgentState) -> bool:
    """
    检查是否有步骤重试次数过多
    
    Returns:
        如果有步骤重试超过 3 次，返回 True
    """
    plan = state.current_plan or []
    for step in plan:
        if step.retry_count >= 3:
            return True
    return False


def _build_planner_messages(state: CRCAgentState,
                            profile_summary: str,
                            intent_summary: str,
                            failed_steps: List[PlanStep],
                            is_self_correction: bool) -> tuple[SystemMessage, HumanMessage]:
    """
    构建 Planner 的 SystemMessage 和 UserMessage
    根据是否为自我纠错模式选择不同的 Prompt 模板
    """
    system_msg = SystemMessage(content=_safe_format(
        PLANNER_SYSTEM_PROMPT,
        profile_summary=profile_summary
    ))

    if is_self_correction:
        error_context_lines = []
        for step in failed_steps:
            error_context_lines.extend([
                f"步骤 [{step.id}]: {step.description}",
                f"工具类型: {step.tool_needed}",
                f"错误信息: {step.error_message}",
                f"已重试次数: {step.retry_count}",
                "",
            ])
        error_context = "\n".join(error_context_lines)

        user_msg_content = _safe_format(
            SELF_CORRECTION_PROMPT_TEMPLATE,
            intent_summary=intent_summary,
            error_context=error_context
        )
    else:
        findings = state.findings or {}
        is_multi_task = findings.get("multi_task_mode", False)
        if is_multi_task:
            user_msg_content = _safe_format(
                MULTI_TASK_USER_PROMPT_TEMPLATE,
                intent_summary=intent_summary,
                missing_critical_data=state.missing_critical_data or '无'
            )
        else:
            user_msg_content = _safe_format(
                PLANNING_USER_PROMPT_TEMPLATE,
                intent_summary=intent_summary,
                missing_critical_data=state.missing_critical_data or '无'
            )

    return system_msg, HumanMessage(content=user_msg_content)


def _filter_by_intent(raw_plan: List[PlanStep], 
                       current_intent: str, 
                       show_thinking: bool = False) -> List[PlanStep]:
    """
    按意图过滤不合适的步骤
    例如：treatment_decision 不要知识检索，imaging_query 不要 imaging_analysis
    """
    if not raw_plan:
        return raw_plan
    
    filtered_plan = []
    
    # 1. 治疗决策意图：过滤掉知识检索步骤
    if current_intent == "treatment_decision":
        knowledge_tool_keywords = [
            "search", "treatment", "guideline", "toc", "read", "chapter",
            "knowledge", "retrieval", "指南", "检索", "查阅"
        ]
        for step in raw_plan:
            tool_lower = step.tool_needed.lower()
            desc_lower = step.description.lower()
            is_knowledge_step = any(kw in tool_lower or kw in desc_lower for kw in knowledge_tool_keywords)
            
            if is_knowledge_step:
                if show_thinking:
                    print(f"[Planner] ⚠️ 过滤掉知识检索步骤: [{step.id}] {step.description}")
                    print(f"           原因: 治疗决策意图的知识检索由 Decision 节点的 SubAgent 统一执行")
            else:
                filtered_plan.append(step)
        return filtered_plan
    
    # 2. 影像查询意图：修正 imaging_analysis 为 case_database_query
    if current_intent == "imaging_query":
        for step in raw_plan:
            if step.tool_needed.lower() == "imaging_analysis":
                step.tool_needed = "case_database_query"
                if show_thinking:
                    print(f"[Planner] ⚠️ 修正工具类型：用户意图为 imaging_query，将 imaging_analysis 修正为 case_database_query")
            filtered_plan.append(step)
        return filtered_plan
    
    return raw_plan


def _fill_missing_tool_types(plan: List[PlanStep], 
                               show_thinking: bool = False) -> List[PlanStep]:
    """
    基于 assignee + description 自动填充缺失的工具类型
    """
    if not plan:
        return plan
    
    for step in plan:
        if not (step.tool_needed or "").strip():
            assignee = (getattr(step, "assignee", "") or "").lower()
            desc = (step.description or "").lower()
            
            if assignee == "case_database":
                step.tool_needed = "case_database_query"
            elif assignee == "web_search":
                step.tool_needed = "web_search"
            elif assignee == "rad_agent":
                step.tool_needed = "imaging_analysis"
            elif assignee == "path_agent":
                step.tool_needed = "pathology_analysis"
            elif any(k in desc for k in ["评估", "诊断", "临床", "assessment", "diagnosis", "clinical"]):
                step.tool_needed = "ask_user"
            elif any(k in desc for k in ["病例", "病人", "患者", "数据库", "case", "database", "query", "patient", "id"]):
                step.tool_needed = "case_database_query"
            elif any(k in desc for k in ["影像", "ct", "mri", "radiology", "tumor", "imaging"]):
                step.tool_needed = "imaging_analysis"
            elif any(k in desc for k in ["病理", "切片", "pathology", "clam"]):
                step.tool_needed = "pathology_analysis"
            else:
                step.tool_needed = "search_treatment_recommendations"
            
            if show_thinking:
                print(f"[Planner] ⚠️ 工具类型为空，自动推断为 '{step.tool_needed}'")
    
    return plan


def _auto_correct_tool(step: PlanStep, 
                       current_intent: str, 
                       show_thinking: bool = False) -> PlanStep:
    """
    对非法工具类型做自动修正
    """
    tool_lower = step.tool_needed.lower()
    original_tool = step.tool_needed

    if "imaging" in tool_lower or "tumor" in tool_lower or "radiology" in tool_lower or "ct" in tool_lower or "mri" in tool_lower or "影像" in original_tool:
        if current_intent == "imaging_query":
            step.tool_needed = "case_database_query"
            if show_thinking:
                print(f"[Planner] ⚠️ 工具类型自动修正: '{original_tool}' -> 'case_database_query' (imaging_query 意图)")
        else:
            step.tool_needed = "imaging_analysis"
            if show_thinking:
                print(f"[Planner] ⚠️ 工具类型自动修正: '{original_tool}' -> 'imaging_analysis'")
    elif "pathology" in tool_lower or "clam" in tool_lower or "切片" in original_tool or "病理" in original_tool:
        step.tool_needed = "pathology_analysis"
        if show_thinking:
            print(f"[Planner] ⚠️ 工具类型自动修正: '{original_tool}' -> 'pathology_analysis'")
    elif "treatment" in tool_lower or "decision" in tool_lower or "clinical" in tool_lower or "reasoning" in tool_lower:
        desc_lower = (step.description or "").lower()
        if any(k in desc_lower for k in ["评估", "诊断", "临床", "assessment", "diagnosis", "clinical"]):
            step.tool_needed = "ask_user"
            if show_thinking:
                print(f"[Planner] ⚠️ 工具类型自动修正: '{original_tool}' -> 'ask_user'")
        else:
            step.tool_needed = "search_treatment_recommendations"
            if show_thinking:
                print(f"[Planner] ⚠️ 工具类型自动修正: '{original_tool}' -> 'search_treatment_recommendations'")
    elif "database" in tool_lower or "query" in tool_lower or "case" in tool_lower:
        step.tool_needed = "case_database_query"
        if show_thinking:
            print(f"[Planner] ⚠️ 工具类型自动修正: '{original_tool}' -> 'case_database_query'")
    else:
        step.tool_needed = "search_treatment_recommendations"
        if show_thinking:
            print(f"[Planner] ⚠️ 工具类型未识别: '{original_tool}'，默认使用 'search_treatment_recommendations'")

    return step


def _normalize_parallel_groups(plan: List[PlanStep]) -> List[PlanStep]:
    """
    清洗并行组：只保留 size >= 2 的并行组
    """
    if not plan:
        return plan
    
    parallel_group_counts: Dict[str, int] = {}
    for step in plan:
        group = getattr(step, "parallel_group", None)
        if group:
            parallel_group_counts[group] = parallel_group_counts.get(group, 0) + 1

    if parallel_group_counts:
        for step in plan:
            group = getattr(step, "parallel_group", None)
            if group and parallel_group_counts.get(group, 0) < 2:
                step.parallel_group = None

    return plan


def _post_process_plan(raw_plan: List[PlanStep],
                       state: CRCAgentState,
                       show_thinking: bool = False) -> List[PlanStep]:
    """
    规划后处理：对 PlanStep 做"意图 + 工具匹配修正"策略化处理
    
    步骤：
    1. 按意图过滤不合适的步骤
    2. 自动填充缺失工具类型
    3. 对非法工具类型做自动修正
    4. 并行组清洗
    """
    findings = state.findings or {}
    current_intent = findings.get("user_intent", "")
    
    # 1. 按意图过滤
    plan_after_intent_filter = _filter_by_intent(raw_plan, current_intent, show_thinking)
    
    # 2. 自动填充空工具类型
    plan_after_fill_tools = _fill_missing_tool_types(plan_after_intent_filter, show_thinking)
    
    # 3. 对非法工具做自动修正
    final_plan: List[PlanStep] = []
    for step in plan_after_fill_tools:
        if not step.is_valid_tool():
            step = _auto_correct_tool(step, current_intent, show_thinking)
        final_plan.append(step)
    
    # 4. 并行组清洗
    final_plan = _normalize_parallel_groups(final_plan)
    
    return final_plan
