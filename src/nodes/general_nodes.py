"""
General Chat and Response Synthesis Nodes
(Refactored from intent_nodes.py - Separated for better code organization)

本文件包含：
- node_general_chat: 闲聊/引导节点 + 响应综合节点

职责：
- 处理正常闲聊（你好/谢谢）
- 处理偏题输入（off_topic_redirect），温柔引导回医疗主线
- 处理模糊输入，礼貌询问澄清
- 计划完成后，综合所有步骤的结果，生成最终响应
"""

import json
import re
from typing import Optional, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, ToolMessage

from ..state import CRCAgentState
from ..prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
)
from .node_utils import (
    _latest_user_text,
    _invoke_with_streaming,
    _ensure_message,
    _build_pinned_context,
    _build_summary_memory,
)

# 导入 general prompts
from ..prompts.general_prompts import (
    REDIRECT_USER_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT_TEMPLATE,
    INFO_ONLY_SYSTEM_PROMPT,
    INFO_ONLY_USER_PROMPT_TEMPLATE,
    PLAN_FOLLOWUP_SYSTEM_PROMPT,
    PLAN_FOLLOWUP_USER_PROMPT_TEMPLATE,
    SIMPLE_FACT_SYSTEM_PROMPT,
    SIMPLE_FACT_USER_PROMPT_TEMPLATE,
)


# =============================================================================
# 辅助函数：提取最近对话历史
# =============================================================================

_FAST_GENERAL_CHAT_REPLIES = {
    "greeting": "您好，我在。可以继续查询患者、解读报告，或生成诊疗建议。",
    "thanks": "不客气。需要我继续帮您查询患者、解读报告，或整理方案吗？",
    "goodbye": "好的，需要时随时叫我。",
}

_GREETING_INPUTS = {
    "你好",
    "你好啊",
    "您好",
    "嗨",
    "哈喽",
    "hello",
    "hi",
    "hey",
    "在吗",
    "在嘛",
    "有人吗",
}

_THANKS_INPUTS = {
    "谢谢",
    "谢谢你",
    "感谢",
    "谢了",
    "辛苦了",
    "thanks",
    "thankyou",
    "thankyou!",
    "thankyou.",
}

_GOODBYE_INPUTS = {
    "再见",
    "拜拜",
    "bye",
    "goodbye",
}


def _normalize_smalltalk(user_input: str) -> str:
    normalized = (user_input or "").strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[?？!！,，。．、~～…]+$", "", normalized)
    return normalized


def _fast_general_chat_reply(user_input: str) -> Optional[str]:
    normalized = _normalize_smalltalk(user_input)
    if not normalized:
        return None
    if normalized in _GREETING_INPUTS:
        return _FAST_GENERAL_CHAT_REPLIES["greeting"]
    if normalized in _THANKS_INPUTS:
        return _FAST_GENERAL_CHAT_REPLIES["thanks"]
    if normalized in _GOODBYE_INPUTS:
        return _FAST_GENERAL_CHAT_REPLIES["goodbye"]
    return None

def _is_simple_fact_question(user_input: str) -> bool:
    """
    判断用户问题是否为简单事实性问题
    
    简单事实性问题的特征：
    - 查询具体的数值、名称、位置等单一信息
    - 如：年龄、性别、病变部位、分期、肿瘤大小等
    - 不涉及"怎么"、"如何"、"建议"、"怎么办"等需要提供方案的词汇
    
    复杂问题的特征：
    - 包含"怎么"、"如何"、"建议"、"怎么办"、"治疗"、"方案"等词汇
    - 需要综合分析和提供专业建议
    """
    # 复杂问题的关键词
    complex_keywords = [
        "怎么", "如何", "建议", "怎么办", "治疗", "方案", 
        "解读", "分析", "评估", "下一步", "应该", "需要",
        "预后", "风险", "策略", "管理", "处理"
    ]
    
    # 检查是否包含复杂关键词
    for keyword in complex_keywords:
        if keyword in user_input:
            return False
    
    # 简单事实性问题的关键词（这些词通常表示只是查询一个具体信息）
    simple_question_patterns = [
        "是什么", "在哪里", "多大", "是什么", "是哪里",
        "几岁", "多高", "多重", "哪些", "有没有"
    ]
    
    # 检查是否是简单问题模式
    for pattern in simple_question_patterns:
        if pattern in user_input:
            return True
    
    # 如果问题很短（通常简单问题比较简短），也倾向于认为是简单问题
    # 但需要排除明显的问候语
    greetings = ["你好", "谢谢", "早上好", "晚上好", "再见"]
    if len(user_input) <= 20 and user_input not in greetings:
        return True
    
    # 默认情况下，如果不是复杂问题，也不是明确简单问题，倾向于认为是简单问题
    # 这样可以避免过度生成建议
    return True


def _get_recent_conversation_history(state: CRCAgentState, max_turns: int = 3) -> str:
    """
    提取最近的对话历史，用于偏题引导的上下文理解。
    使用语义重要性分类替代硬截断。
    
    Args:
        state: Agent 状态
        max_turns: 最多返回的对话轮数（一轮 = 用户消息 + AI 回复）
    
    Returns:
        格式化的对话历史字符串，用于填充到 Prompt 中
    """
    from langchain_core.messages import HumanMessage
    
    try:
        from .memory_nodes import classify_message_importance, truncate_by_importance
    except ImportError:
        classify_message_importance = lambda msg: "medium"
        def truncate_by_importance(content: str, importance: str) -> str:
            if importance == "critical":
                return content
            if importance == "high":
                return content[:500] + "..." if len(content) > 500 else content
            if importance == "medium":
                return content[:300] + "..." if len(content) > 300 else content
            return content[:100] + "..." if len(content) > 100 else content
    
    if not state.messages:
        return "（无历史对话）"
    
    history_parts = []
    turn_count = 0
    messages = list(state.messages)
    
    for i in range(len(messages) - 2, -1, -1):
        if turn_count >= max_turns:
            break
        
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            importance = classify_message_importance(msg)
            content = truncate_by_importance(msg.content, importance)
            history_parts.insert(0, f"用户: {content}")
            turn_count += 1
        elif isinstance(msg, AIMessage) and msg.content:
            importance = classify_message_importance(msg)
            content = truncate_by_importance(msg.content, importance)
            history_parts.insert(0, f"助手: {content}")
    
    if not history_parts:
        return "（无历史对话）"
    
    return "\n".join(history_parts)


# =============================================================================
# General Chat Node - 闲聊/引导节点 + 响应综合节点
# =============================================================================

def node_general_chat(model, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    """
    闲聊/引导节点 + 响应综合节点
    
    职责：
    - 处理正常闲聊（你好/谢谢）
    - 处理偏题输入（off_topic_redirect），温柔引导回医疗主线
    - 处理模糊输入，礼貌询问澄清
    - [新增] 计划完成后，综合所有步骤的结果，生成最终响应
    - [增强] 基于对话历史、长期记忆和患者档案进行上下文对话
    """
    
    # 基础 Prompt（处理正常闲聊）
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_CHAT_SYSTEM_PROMPT),
        ("human", "{user_input}"),
    ])
    
    # 增强 Prompt（处理偏题时，提供更多上下文）
    redirect_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_CHAT_SYSTEM_PROMPT),
        ("human", REDIRECT_USER_PROMPT_TEMPLATE),
    ])
    
    # 综合响应 Prompt（计划完成后，综合所有步骤的结果）
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_SYSTEM_PROMPT),
        ("human", SYNTHESIS_USER_PROMPT_TEMPLATE),
    ])

    # 方案追问解释 Prompt（不触发新诊断）
    plan_followup_prompt = ChatPromptTemplate.from_messages([
        ("system", PLAN_FOLLOWUP_SYSTEM_PROMPT),
        ("human", PLAN_FOLLOWUP_USER_PROMPT_TEMPLATE),
    ])
    
    base_chain = base_prompt | model
    redirect_chain = redirect_prompt | model
    synthesis_chain = synthesis_prompt | model
    
    def _run(state: CRCAgentState):
        user_text = _latest_user_text(state)
        intent = (state.findings or {}).get("user_intent", "general_chat")
        plan_followup = (state.findings or {}).get("plan_followup", False)
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)
        recent_conversation = _get_recent_conversation_history(state, max_turns=5)
        fast_reply = _fast_general_chat_reply(user_text)
        if fast_reply is not None:
            return {
                "messages": [AIMessage(content=fast_reply)],
                "clinical_stage": "General",
                "error": None,
            }
        
        # 检查是否是计划完成后的综合响应场景
        plan = state.current_plan or []
        has_completed_plan = plan and all(s.status == 'completed' for s in plan)
        
        if plan_followup and state.decision_json:
            response = _invoke_with_streaming(
                plan_followup_prompt | model,
                {
                    "user_question": user_text,
                    "decision_json": json.dumps(state.decision_json, ensure_ascii=False),
                    "summary_memory": summary_memory,
                    "pinned_context": pinned_context,
                    "recent_conversation": recent_conversation,
                },
                streaming,
                show_thinking,
            )
            msg = _ensure_message(response)
            return {
                "messages": [msg],
                "clinical_stage": "General",
                "error": None,
            }

        if has_completed_plan:
            if show_thinking:
                print(f"📊 [General Chat] 综合响应模式: 整合 {len(plan)} 个步骤的结果")
            
            # 判断是否为纯信息展示任务
            # 注意：knowledge_query 应该生成自然的回答，不能只用 info_only_prompt
            # 只有数据库查询和影像信息查询才使用纯信息展示模式
            info_only_intents = {
                "case_database_query",    # 数据库查询
                "imaging_query",          # 影像信息查询
                # 移除 "knowledge_query" - 知识查询需要生成自然回答
            }
            # 安全获取 sub_tasks，确保非空再判断
            current_sub_tasks = (state.findings or {}).get("sub_tasks") or []
            is_info_only_task = (
                intent in info_only_intents or
                # 确保 sub_tasks 非空再判断，且是纯信息类任务
                (bool(current_sub_tasks) and all(t in info_only_intents for t in current_sub_tasks))
            )
            
            # 判断是否为简单事实性问题
            is_simple_fact = _is_simple_fact_question(user_text)
            
            # 从消息历史中提取所有消息结果（步骤结果）
            # 同时收集 AIMessage 和 ToolMessage，确保工具调用的结果也能展示
            collected_info_parts = []
            for i, msg in enumerate(state.messages):
                # 收集 AIMessage（排除意图分类等系统消息）
                if isinstance(msg, AIMessage) and msg.content:
                    if "意图分类" not in msg.content and "Intent" not in msg.content:
                        collected_info_parts.append(f"### 信息片段 {i+1}\n{msg.content}\n")
                # 收集 ToolMessage（工具调用结果）
                elif isinstance(msg, ToolMessage) and msg.content:
                    tool_name = getattr(msg, "name", "Unknown Tool")
                    collected_info_parts.append(f"### 工具调用结果 - {tool_name}\n{msg.content}\n")
            
            collected_info = "\n".join(collected_info_parts) if collected_info_parts else "（暂无收集到的信息）"
            
            # 根据任务类型选择不同的 prompt
            if is_info_only_task:
                # 纯信息展示模式：只整理和展示数据，不提供建议
                info_only_prompt = ChatPromptTemplate.from_messages([
                    ("system", INFO_ONLY_SYSTEM_PROMPT),
                    ("human", INFO_ONLY_USER_PROMPT_TEMPLATE),
                ])
                if show_thinking:
                    print(f"📋 [General Chat] 纯信息展示模式")
                response = _invoke_with_streaming(
                    info_only_prompt | model,
                    {
                        "user_question": user_text,
                        "collected_info": collected_info,
                        "summary_memory": summary_memory,
                        "pinned_context": pinned_context,
                        "recent_conversation": recent_conversation,
                    },
                    streaming,
                    show_thinking
                )
            elif is_simple_fact:
                # 简单事实性回答：只回答问题，不提供额外建议
                simple_fact_prompt = ChatPromptTemplate.from_messages([
                    ("system", SIMPLE_FACT_SYSTEM_PROMPT),
                    ("human", SIMPLE_FACT_USER_PROMPT_TEMPLATE),
                ])
                if show_thinking:
                    print(f"❓ [General Chat] 简单事实性回答模式")
                response = _invoke_with_streaming(
                    simple_fact_prompt | model,
                    {
                        "user_question": user_text,
                        "collected_info": collected_info,
                        "summary_memory": summary_memory,
                        "pinned_context": pinned_context,
                        "recent_conversation": recent_conversation,
                    },
                    streaming,
                    show_thinking
                )
            else:
                # 原有的综合响应模式：整合信息并提供建议
                if show_thinking:
                    print(f"📊 [General Chat] 综合响应模式：整合信息并提供建议")
                response = _invoke_with_streaming(
                    synthesis_chain,
                    {
                        "user_question": user_text,
                        "collected_info": collected_info,
                        "summary_memory": summary_memory,
                        "pinned_context": pinned_context,
                        "recent_conversation": recent_conversation,
                    },
                    streaming,
                    show_thinking
                )
            
            msg = _ensure_message(response)
            return {
                "messages": [msg],
                "clinical_stage": "General",
                "error": None,
            }
        
        # 原有逻辑：根据意图选择不同的处理链
        if intent == "off_topic_redirect":
            # 提取最近对话历史（用于引导回正轨）
            recent_history = _get_recent_conversation_history(state, max_turns=3)
            if show_thinking:
                print(f"🔄 [General Chat] Off-topic redirect mode for: '{user_text[:30]}...'")
            response = _invoke_with_streaming(
                redirect_chain,
                {
                    "user_input": user_text,
                    "recent_conversation": recent_history,
                    "summary_memory": summary_memory,
                    "pinned_context": pinned_context,
                },
                streaming,
                show_thinking
            )
        else:
            response = _invoke_with_streaming(
                base_chain,
                {
                    "user_input": user_text,
                    "summary_memory": summary_memory,
                    "pinned_context": pinned_context,
                    "recent_conversation": recent_conversation,
                },
                streaming,
                show_thinking
            )
        
        msg = _ensure_message(response)
        return {
            "messages": [msg],
            "clinical_stage": "General",
            "error": None,
        }
    
    return _run
