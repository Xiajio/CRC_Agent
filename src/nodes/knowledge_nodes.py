"""
Knowledge Retrieval Nodes
(Final Fix: Robust Schema, Prompt Escaping, and Nested JSON Handling)
"""

import json
from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from ..state import CRCAgentState, PatientProfile
from ..rag.evidence import make_rag_trace
from ..prompts import (
    SEARCH_PLANNER_SYSTEM_PROMPT,
    SUFFICIENCY_EVALUATOR_SYSTEM_PROMPT,
    KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT,
    GENERAL_KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT,
)
from .node_utils import (
    _clean_and_validate_json,
    _extract_first_json_object,
    _extract_text_content,
    _latest_user_text,
    _user_text,
    _invoke_with_streaming,
    _ensure_message,
    _extract_structured_evidence,
    _unwrap_nested_json,  # Important: Use the unwrapper!
    _create_rag_digest,
    _truncate_message_history,
    _build_pinned_context,
    _build_summary_memory,
)
from .sub_agent import create_web_researcher, SubAgentContext, run_isolated_web_search
from .knowledge_utils import _get_patient_state_description

# ==============================================================================
# 1. Search Planning Schema
# ==============================================================================

class SearchPlan(BaseModel):
    """Search Strategy Plan"""
    needs_search: bool = Field(description="Whether external search is needed")
    search_query: str = Field(description="Optimized search query string")
    selected_tool: str = Field(description="Tool name (e.g. search_drug_online)")
    tool_arguments: dict = Field(description="Arguments for the tool", default_factory=dict)
    reasoning: str = Field(description="Why this tool was chosen")

class KnowledgeSufficiencyEval(BaseModel):
    """Local Knowledge Evaluation"""
    is_sufficient: bool = Field(description="Is local context sufficient?")
    missing_info: str = Field(description="What is missing (if any)")


def _build_synthesis_payload(
    *,
    question: str,
    context: str,
    summary_memory: str,
    pinned_context: str,
    requires_context: bool,
    patient_profile: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a payload that satisfies all synthesis prompt variables."""
    payload: Dict[str, Any] = {
        "context": context or "",
        "question": question,
        "summary_memory": summary_memory,
        "pinned_context": pinned_context,
        "requires_context": requires_context,
    }
    if patient_profile is not None:
        payload["patient_profile"] = patient_profile
    return payload


def _parse_search_plan_from_raw_response(raw_response: Any) -> SearchPlan:
    """Recover SearchPlan from raw model text when strict schema parsing fails."""
    content = _extract_text_content(raw_response)
    parsed = _clean_and_validate_json(content)
    if parsed is None:
        parsed = _extract_first_json_object(content)
    if parsed is None:
        raise ValueError("search plan parse failed")

    parsed = _unwrap_nested_json(
        parsed,
        ["needs_search", "search_query", "selected_tool", "tool_arguments", "reasoning", "search_layer"],
    )
    return SearchPlan(**parsed)


def _parse_sufficiency_eval_from_raw_response(raw_response: Any) -> KnowledgeSufficiencyEval:
    """Recover KnowledgeSufficiencyEval from raw model text when strict schema parsing fails."""
    content = _extract_text_content(raw_response)
    parsed = _clean_and_validate_json(content)
    if parsed is None:
        parsed = _extract_first_json_object(content)
    if parsed is None:
        raise ValueError("sufficiency eval parse failed")

    parsed = _unwrap_nested_json(parsed, ["is_sufficient", "missing_info"])
    return KnowledgeSufficiencyEval(**parsed)

# ==============================================================================
# 2. Helper Chains (Prompt Escaping Fixes)
# ==============================================================================

def _create_search_planner(model):
    """Chain to plan web search with hierarchical strategy"""
    # 从统一的 prompts 模块导入 System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SEARCH_PLANNER_SYSTEM_PROMPT),
        ("human", "{user_input}")
    ])
    structured_chain = prompt | model.with_structured_output(SearchPlan).bind(temperature=0)
    raw_chain = prompt | model.bind(temperature=0)

    def _invoke(ctx: Dict[str, Any]) -> SearchPlan:
        try:
            return structured_chain.invoke(ctx)
        except Exception:
            raw_response = raw_chain.invoke(ctx)
            return _parse_search_plan_from_raw_response(raw_response)

    return RunnableLambda(_invoke)

def _create_sufficiency_evaluator(model):
    """Chain to evaluate if local RAG is enough"""
    # 从统一的 prompts 模块导入 System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUFFICIENCY_EVALUATOR_SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    structured_chain = prompt | model.with_structured_output(KnowledgeSufficiencyEval).bind(temperature=0)
    raw_chain = prompt | model.bind(temperature=0)

    def _invoke(ctx: Dict[str, Any]) -> KnowledgeSufficiencyEval:
        try:
            return structured_chain.invoke(ctx)
        except Exception:
            raw_response = raw_chain.invoke(ctx)
            return _parse_sufficiency_eval_from_raw_response(raw_response)

    return RunnableLambda(_invoke)


# ==============================================================================
# 3. Node: Knowledge Retrieval
# ==============================================================================

def node_knowledge_retrieval(
    model, 
    tools: List[BaseTool], 
    streaming: bool = False, 
    show_thinking: bool = True,
    use_sub_agent: bool = True  # [新增] 是否使用子智能体隔离进行联网搜索
) -> Runnable:
    """
    Intelligent Knowledge Retrieval Node
    
    支持两种模式：
    1. use_sub_agent=True（默认）：联网搜索在隔离的子智能体中进行，不污染主上下文
    2. use_sub_agent=False：直接调用工具，兼容旧版行为
    """
    local_rag_tool = next((t for t in tools if hasattr(t, 'name') and getattr(t, "name", "") == "search_clinical_guidelines"), None)
    web_tool_map = {t.name: t for t in tools if hasattr(t, 'name') and getattr(t, "name", "") != "search_clinical_guidelines"}
    
    # 收集所有联网搜索工具（用于子智能体模式）
    web_tools_list = list(web_tool_map.values())
    
    # Pre-load chains
    search_planner = _create_search_planner(model)
    sufficiency_evaluator = _create_sufficiency_evaluator(model)
    
    # 从统一的 prompts 模块导入 Synthesis Prompt
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    synthesis_chain = synthesis_prompt | model

    general_synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    general_synthesis_chain = general_synthesis_prompt | model

    async def _run(state: CRCAgentState):
        # =====================================================================
        # [新增] 计划驱动模式（Plan-Driven Mode）
        # 如果有当前计划步骤，优先执行计划，而不是走原有的自动检索逻辑
        # =====================================================================
        from ..nodes.planner import get_current_pending_step, mark_step_completed, mark_step_failed
        
        user_query = _latest_user_text(state)
        pinned_context = _build_pinned_context(state)
        summary_memory = _build_summary_memory(state)
        requires_context = (state.findings or {}).get("requires_context")
        if requires_context is None:
            use_patient_context = _should_use_patient_context(user_query)
        else:
            use_patient_context = bool(requires_context)
        current_step = get_current_pending_step(state)

        # Fast path for short, general nutrition/wellness questions.
        # These do not benefit from the full retrieval-planning pipeline.
        if (
            not current_step
            and not use_patient_context
            and user_query
            and len(user_query.strip()) <= 32
            and _is_non_standard_care_question(user_query)
            and not _contains_clinical_markers(user_query)
        ):
            if show_thinking:
                print("\n[Knowledge] ⚡ 通用知识快路径（跳过检索规划）")

            response = _invoke_with_streaming(
                general_synthesis_chain,
                _build_synthesis_payload(
                    question=user_query,
                    context="",
                    summary_memory=summary_memory,
                    pinned_context=pinned_context,
                    requires_context=False,
                ),
                streaming,
                show_thinking,
            )
            return {
                "messages": [_ensure_message(response)],
                "clinical_stage": "Knowledge",
                "error": None,
            }
        
        if current_step:
            # 进入计划驱动模式
            print(f"\n[Knowledge] 📋 计划驱动模式")
            print(f"  步骤: [{current_step.id}] {current_step.description}")
            print(f"  工具: {current_step.tool_needed}")
            
            tool_type = current_step.tool_needed.lower()
            
            try:
                # 根据 tool_needed 路由到对应的工具
                result_content = None
                error_detected = False
                
                # 1. TOC (Table of Contents) - 目录查询
                if "toc" in tool_type or "目录" in tool_type:
                    toc_tool = next((t for t in tools if hasattr(t, 'name') and t.name == "list_guideline_toc"), None)
                    if toc_tool:
                        # 从描述中提取指南名称
                        guideline_name = _extract_guideline_name(current_step.description)
                        result = toc_tool.invoke({"guideline_name": guideline_name})
                        result_content = str(result)
                        
                        # 检查是否有错误
                        if "未找到" in result_content or "错误" in result_content or "请确认" in result_content:
                            error_detected = True
                    else:
                        result_content = "错误：list_guideline_toc 工具不可用"
                        error_detected = True
                
                # 2. CHAPTER - 章节阅读
                elif "chapter" in tool_type or "章节" in tool_type:
                    chapter_tool = next((t for t in tools if hasattr(t, 'name') and t.name == "read_guideline_chapter"), None)
                    if chapter_tool:
                        # 从描述中提取指南名称和章节名称
                        guideline_name, chapter_name = _extract_guideline_and_chapter(current_step.description)
                        result = chapter_tool.invoke({
                            "guideline_name": guideline_name,
                            "chapter_name": chapter_name
                        })
                        result_content = str(result)
                        
                        # 检查是否有错误
                        if "未找到" in result_content or "错误" in result_content:
                            error_detected = True
                    else:
                        result_content = "错误：read_guideline_chapter 工具不可用"
                        error_detected = True
                
                # 3. SEARCH - 检索
                elif "search" in tool_type:
                    # 通用知识问题优先使用 web_search，避免临床指南污染
                    if not use_patient_context:
                        web_tool = web_tool_map.get("web_search")
                        if web_tool:
                            search_query = _normalize_web_query(
                                _extract_search_query(current_step.description, state, use_patient_context=False)
                            )
                            result = web_tool.invoke({"query": search_query})
                            result_content = str(result)
                            if len(result_content) < 30:
                                error_detected = True
                        else:
                            result_content = "错误：web_search 工具不可用"
                            error_detected = True
                    else:
                        # 使用现有的 RAG 工具
                        if local_rag_tool:
                            # 从描述中提取查询关键词
                            search_query = _extract_search_query(current_step.description, state, use_patient_context=True)
                            result = local_rag_tool.invoke({"query": search_query, "top_k": 6})
                            result_content = str(result)
                            
                            # 检查是否有检索结果
                            if "No relevant" in result_content or len(result_content) < 50:
                                error_detected = True
                        else:
                            result_content = "错误：search_clinical_guidelines 工具不可用"
                            error_detected = True
                
                # 4. WEB_SEARCH - 在线搜索
                elif "web" in tool_type or "在线" in tool_type:
                    web_tool = web_tool_map.get("web_search")
                    if web_tool:
                        search_query = _normalize_web_query(
                            _extract_search_query(current_step.description, state, use_patient_context=use_patient_context)
                        )
                        result = web_tool.invoke({"query": search_query})
                        result_content = str(result)
                        
                        if len(result_content) < 30:
                            error_detected = True
                    else:
                        result_content = "错误：web_search 工具不可用"
                        error_detected = True
                
                else:
                    # 兜底：尝试智能映射无效工具类型到有效工具
                    tool_lower = tool_type.lower()
                    mapped_tool = None
                    
                    # 智能映射：将常见的无效工具名映射到有效工具
                    if "treatment" in tool_lower or "decision" in tool_lower or "clinical" in tool_lower:
                        # 治疗决策相关 -> 使用 search_treatment_recommendations
                        mapped_tool = "search_treatment_recommendations"
                        print(f"  🔄 自动映射: '{tool_type}' -> '{mapped_tool}'")
                        if not use_patient_context:
                            web_tool = web_tool_map.get("web_search")
                            if web_tool:
                                search_query = _normalize_web_query(
                                    _extract_search_query(current_step.description, state, use_patient_context=False)
                                )
                                result = web_tool.invoke({"query": search_query})
                                result_content = str(result)
                                if len(result_content) < 30:
                                    error_detected = True
                            else:
                                result_content = "错误：web_search 工具不可用"
                                error_detected = True
                        else:
                            if local_rag_tool:
                                search_query = _extract_search_query(current_step.description, state, use_patient_context=True)
                                result = local_rag_tool.invoke({"query": search_query, "top_k": 6})
                                result_content = str(result)
                                
                                if "No relevant" in result_content or len(result_content) < 50:
                                    error_detected = True
                            else:
                                result_content = "错误：search_clinical_guidelines 工具不可用"
                                error_detected = True
                    else:
                        # 无法映射，标记失败
                        new_plan = mark_step_failed(
                            state,
                            current_step.id,
                            f"不支持的工具类型: {current_step.tool_needed}。有效的工具类型包括: list_guideline_toc, read_guideline_chapter, search_treatment_recommendations, database_query, web_search, ask_user"
                        )
                        return {
                            "current_plan": new_plan,
                            "scratchpad": state.scratchpad + f"\n[步骤失败] {current_step.id}: 不支持的工具类型\n"
                        }
                
                # 检查执行结果
                if error_detected or not result_content:
                    # 标记为失败
                    error_msg = f"工具执行失败: {result_content[:200] if result_content else '无输出'}"
                    new_plan = mark_step_failed(state, current_step.id, error_msg)
                    
                    print(f"  ❌ 步骤失败: {error_msg}")
                    
                    return {
                        "current_plan": new_plan,
                        "scratchpad": state.scratchpad + f"\n[步骤失败] {current_step.id}: {error_msg}\n",
                        "messages": [AIMessage(content=f"⚠️ 执行失败，正在重新规划...\n错误: {error_msg}")]
                    }
                
                # 成功：标记为完成
                new_plan = mark_step_completed(state, current_step.id)

                print(f"  ✅ 步骤完成")

                # 提取引用
                final_content, refs, evidence = _extract_structured_evidence(result_content)
                if not use_patient_context:
                    final_content = _filter_context_for_general_knowledge(user_query, final_content)

                # [修复] 使用 synthesis_chain 综合生成答案，而不是直接返回检索内容
                # 这样可以生成自然的、易读的回答，而不是原始的检索片段
                # 创建适合计划驱动模式的 synthesis prompt（包含 context 占位符）
                plan_driven_synthesis_prompt = ChatPromptTemplate.from_messages([
                    ("system", KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT),
                    ("human", "【用户问题】\n{question}\n\n【上下文】\n{context}")
                ])
                plan_driven_general_prompt = ChatPromptTemplate.from_messages([
                    ("system", GENERAL_KNOWLEDGE_SYNTHESIS_SYSTEM_PROMPT),
                    ("human", "【用户问题】\n{question}\n\n【上下文】\n{context}")
                ])
                plan_driven_synthesis_chain = (
                    plan_driven_synthesis_prompt | model
                    if use_patient_context
                    else plan_driven_general_prompt | model
                )

                try:
                    synthesis_payload = {
                        "context": final_content,
                        "question": _extract_search_query(current_step.description, state, use_patient_context=use_patient_context),
                        "summary_memory": summary_memory,
                        "pinned_context": pinned_context,
                        "requires_context": use_patient_context,
                    }
                    if use_patient_context:
                        synthesis_payload["patient_profile"] = (
                            str(state.patient_profile.model_dump())
                            if state.patient_profile and hasattr(state.patient_profile, "model_dump")
                            else (str(state.patient_profile) if state.patient_profile else "")
                        )
                    response = _invoke_with_streaming(
                        plan_driven_synthesis_chain,
                        synthesis_payload,
                        streaming,
                        show_thinking
                    )
                    final_msg = _ensure_message(response)
                except Exception as e:
                    # 如果综合生成失败，回退到原始内容
                    print(f"  ⚠️ 综合生成失败，使用原始内容: {e}")
                    final_msg = AIMessage(content=final_content)

                updates = {
                    "current_plan": new_plan,
                    "scratchpad": state.scratchpad + f"\n[步骤完成] {current_step.id}\n",
                    "messages": [final_msg],
                    "clinical_stage": "Knowledge"
                }

                if refs:
                    updates["retrieved_references"] = refs
                if evidence:
                    updates["retrieved_evidence"] = evidence
                    updates["rag_trace"] = [
                        make_rag_trace(
                            tool_name=str(current_step.tool_needed or "knowledge"),
                            query=_extract_search_query(
                                current_step.description,
                                state,
                                use_patient_context=use_patient_context,
                            ),
                            retrieval_profile="knowledge",
                            evidence=evidence,
                        )
                    ]

                return updates
                
            except Exception as e:
                # 捕获异常，标记为失败
                error_msg = f"异常: {str(e)}"
                new_plan = mark_step_failed(state, current_step.id, error_msg)
                
                print(f"  ❌ 异常: {e}")
                
                return {
                    "current_plan": new_plan,
                    "scratchpad": state.scratchpad + f"\n[异常] {current_step.id}: {e}\n",
                    "messages": [AIMessage(content=f"⚠️ 执行出错，正在重新规划...\n错误: {error_msg}")]
                }
        
        # =====================================================================
        # 原有逻辑：如果没有计划步骤，走自动检索模式
        # =====================================================================
        print(f"\n[Knowledge] 🔍 自动检索模式（无计划驱动）")
        
        user_query = _latest_user_text(state)
        
        # 1. Local RAG (Layer 0 - Baseline)
        local_context = ""
        retrieved_refs = []
        local_refs = []
        retrieved_evidence = []
        rag_trace = []
        if local_rag_tool and use_patient_context:
            try:
                # Simple query for RAG
                raw_res = local_rag_tool.invoke({"query": user_query, "top_k": 4})
                local_context, refs, evidence = _extract_structured_evidence(str(raw_res))
                local_refs = refs
                retrieved_refs.extend(refs)
                retrieved_evidence.extend(evidence)
                if evidence:
                    rag_trace.append(
                        make_rag_trace(
                            tool_name=str(getattr(local_rag_tool, "name", "search_clinical_guidelines")),
                            query=user_query,
                            retrieval_profile="general",
                            evidence=evidence,
                        )
                    )
                if show_thinking:
                    print(f"📚 [Knowledge] Local RAG retrieved {len(refs)} references")
            except Exception as e:
                print(f"[Knowledge] Local RAG Error: {e}")

        # 2. Evaluate Sufficiency
        needs_web_search = False
        search_reason = "Local info missing"
        
        if not local_context or len(local_context) < 50:
            needs_web_search = True
        else:
            try:
                # [Fix]: Unwrap potential nested JSON
                eval_raw = sufficiency_evaluator.invoke({
                    "question": user_query,
                    "context": local_context[:3000],
                    "summary_memory": summary_memory,
                    "pinned_context": pinned_context
                })
                
                if not eval_raw.is_sufficient:
                    needs_web_search = True
                    search_reason = eval_raw.missing_info
                    print(f"[Knowledge] 本地知识库不足: {search_reason}")
                else:
                    print(f"[Knowledge] 本地知识库已足够。")
            except Exception as e:
                print(f"[Knowledge] 评估出错: {e}, 将回退到联网搜索。")
                needs_web_search = True
        
        # 2b. Relevance check: avoid drifting to unrelated local context
        if local_context and not _is_context_relevant(user_query, local_context):
            if show_thinking:
                print("[Knowledge] 本地知识库内容与问题不匹配，忽略本地内容并改为联网搜索。")
            local_context = ""
            if local_refs:
                retrieved_refs = [r for r in retrieved_refs if r not in local_refs]
                local_refs = []
            needs_web_search = True
            if not search_reason or search_reason == "Local info missing":
                search_reason = "本地内容与问题不匹配"

        # 3. Hierarchical Web Search Strategy (🎯 分层检索)
        web_context = ""
        web_layers_executed = []
        sub_agent_used = False
        
        if needs_web_search:
            # ============================================================
            # 模式 A：子智能体隔离（推荐，上下文完全隔离）
            # ============================================================
            if use_sub_agent and web_tools_list:
                if show_thinking:
                    print(f"🔒 [Knowledge] 使用子智能体隔离模式进行联网搜索...")
                
                try:
                    patient_ctx = None
                    if use_patient_context and state.patient_profile:
                        patient_ctx = {
                            "profile": state.patient_profile.model_dump() if hasattr(state.patient_profile, "model_dump") else str(state.patient_profile),
                            "current_state": _get_patient_state_description(state), # [Context Injection]
                        }
                    elif state.patient_profile:
                        # 即使 use_patient_context=False，如果存在活跃患者，也注入状态
                        patient_ctx = {
                            "current_state": _get_patient_state_description(state)
                        }
                        # 即使 use_patient_context=False，如果存在活跃患者，也注入状态
                        patient_ctx = {
                            "current_state": _get_patient_state_description(state)
                        }
                    
                    # 执行隔离的联网搜索
                    # 子智能体会自主进行多次搜索、筛选、阅读
                    # 所有中间过程都在沙箱中，主智能体只收到最终报告
                    # [优化] 知识问答只需1次迭代，快速响应
                    sub_result = await run_isolated_web_search(
                        model=model,
                        query=user_query,
                        web_tools=web_tools_list,
                        local_context=local_context,
                        patient_context=patient_ctx,
                        show_thinking=show_thinking,
                        max_iterations=1  # 知识问答只需1次迭代，提升响应速度
                    )
                    
                    if sub_result.success:
                        # 蒸馏后的报告（子智能体的完整历史已被销毁）
                        web_context = sub_result.report
                        web_layers_executed.append("SubAgent (isolated)")
                        sub_agent_used = True
                        
                        # 合并引用
                        if sub_result.references:
                            retrieved_refs.extend(sub_result.references)
                        
                        if show_thinking:
                            print(f"✅ [SubAgent] 搜索完成，报告长度: {len(web_context)} 字符")
                            print(f"   子智能体消耗: ~{sub_result.raw_token_count} tokens, {sub_result.iterations} 次迭代")
                    else:
                        if show_thinking:
                            print(f"⚠️ [SubAgent] 搜索失败，回退到直接模式: {sub_result.error}")
                
                except Exception as e:
                    if show_thinking:
                        print(f"⚠️ [SubAgent] 初始化失败，回退到直接模式: {e}")
            
            # ============================================================
            # 模式 B：直接调用（兼容旧版，但会污染上下文）
            # ============================================================
            if not sub_agent_used:
                if show_thinking:
                    print(f"📋 [Knowledge] 使用直接调用模式进行联网搜索...")
                
                web_context_parts = []
                
                # Generate search plan (this now includes layer selection)
                try:
                    plan = search_planner.invoke({
                        "user_input": user_query,
                        "summary_memory": summary_memory,
                        "pinned_context": pinned_context
                    })
                    search_layer = getattr(plan, 'search_layer', 'unknown')
                    
                    if show_thinking:
                        print(f"🎯 [Knowledge] 检索层级 {search_layer}: {plan.reasoning}")
                        print(f"   工具: {plan.selected_tool} | 查询词: {plan.search_query}")
                    
                    # Execute the planned search
                    selected_tool = web_tool_map.get(plan.selected_tool)
                    if not selected_tool:
                        selected_tool = web_tool_map.get("web_search")
                        plan.tool_arguments = {"query": plan.search_query}

                    if selected_tool:
                        try:
                            # Fix: Map search query to correct tool parameter
                            args = _map_tool_arguments(plan.selected_tool, plan.search_query, plan.tool_arguments)
                            if plan.selected_tool == "web_search" and "query" in args:
                                args["query"] = _normalize_web_query(args["query"])
                            
                            tool_res = selected_tool.invoke(args)
                            layer_context = str(tool_res)
                            web_context_parts.append(f"【Layer {search_layer} - {plan.selected_tool}】\n{layer_context}")
                            web_layers_executed.append(f"Layer {search_layer}")
                            
                            # 🎯 [NEW] Hierarchical Search: Execute complementary searches based on layer
                            # If it's a non-standard care question, automatically run all 3 layers
                            if use_patient_context and _is_non_standard_care_question(user_query):
                                complementary_searches = _get_complementary_searches(
                                    plan,
                                    search_layer,
                                    web_tool_map,
                                    user_query,
                                    state.patient_profile if use_patient_context else None
                                )
                                for comp_search in complementary_searches:
                                    web_context_parts.append(comp_search["context"])
                                    web_layers_executed.append(comp_search["layer"])
                                    if show_thinking:
                                        print(f"   🔄 [Auto-run] {comp_search['layer']}: {comp_search['tool']}")
                            
                        except Exception as te:
                            web_context_parts.append(f"Tool error: {te}")
                except Exception as e:
                    print(f"[Knowledge] Planning Error: {e}")
                    # Fallback to simple web search
                    fallback_tool = web_tool_map.get("web_search")
                    if fallback_tool:
                        web_context_parts.append(
                            str(fallback_tool.invoke({"query": _normalize_web_query(user_query)}))
                        )
                        web_layers_executed.append("Fallback")
                
                # Combine web contexts
                web_context = "\n\n".join(web_context_parts)
        
        # 4. Synthesize Answer with Evidence Grading
        if not use_patient_context:
            local_context = _filter_context_for_general_knowledge(user_query, local_context)
            web_context = _filter_context_for_general_knowledge(user_query, web_context)
        final_context = f"【Local Knowledge】\n{local_context}\n\n【Hierarchical Web Search】\n{web_context}"
        
        if show_thinking and web_layers_executed:
            print(f"📊 [Knowledge] 已执行的证据层级: {', '.join(web_layers_executed)}")
        
        try:
            synthesis_payload = _build_synthesis_payload(
                question=user_query,
                context=final_context,
                summary_memory=summary_memory,
                pinned_context=pinned_context,
                requires_context=use_patient_context,
            )
            if use_patient_context:
                synthesis_payload["patient_profile"] = (
                    str(state.patient_profile.model_dump())
                    if state.patient_profile and hasattr(state.patient_profile, "model_dump")
                    else (str(state.patient_profile) if state.patient_profile else "")
                )
            response = _invoke_with_streaming(
                synthesis_chain if use_patient_context else general_synthesis_chain,
                synthesis_payload,
                streaming,
                show_thinking
            )
            final_msg = _ensure_message(response)
            
            updates = {
                "messages": [final_msg],
                "clinical_stage": "Knowledge",
                "error": None
            }
            if retrieved_refs:
                updates["retrieved_references"] = retrieved_refs
            if retrieved_evidence:
                updates["retrieved_evidence"] = retrieved_evidence
            if rag_trace:
                updates["rag_trace"] = rag_trace
            
            return updates

        except Exception as e:
            # [优化] 错误时不输出完整的原始上下文，只输出摘要
            error_digest = _create_rag_digest(
                rag_context=final_context,
                references=retrieved_refs,
                queries=[user_query],
                max_digest_chars=500
            )
            return {
                "messages": [AIMessage(content=f"知识检索已完成，但综合生成时出错。\n{error_digest}\n\n错误: {str(e)[:200]}")],
                "clinical_stage": "Knowledge",
                "error": str(e)
            }

    return _run


def _is_non_standard_care_question(query: str) -> bool:
    """判断是否为非标准治疗问题，需要多层检索"""
    keywords_non_standard = [
        # 饮食/营养
        "vitamin", "supplement", "diet", "nutrition", "食物", "维生素", "补品", "营养",
        # 替代疗法
        "traditional chinese", "tc", "herbal", "acupuncture", "moxibustion",
        "中医", "中药", "针灸", "艾灸", "草药",
        # 偏方/民间疗法
        "folk", "remedy", "偏方", "民间", "食疗",
        # 其他补剂
        "curcumin", "turmeric", "gingko", "ginseng", "omega-3"
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords_non_standard)


def _should_use_patient_context(query: str) -> bool:
    """判断是否需要结合患者个体情况进行回答"""
    q = (query or "").lower()

    # 明确指向患者或本人
    patient_markers = ["患者", "病人", "本人", "我", "他", "她", "ta", "tumor", "癌", "化疗", "放疗", "手术"]
    action_markers = [
        "可以", "能否", "是否", "适合", "需要", "建议", "注意", "禁忌", "忌口",
        "服用", "吃", "饮食", "营养", "补充", "剂量", "用量", "相互作用"
    ]

    if any(m in q for m in patient_markers) and any(m in q for m in action_markers):
        return True

    # 明显是个体化饮食/用药场景
    personalized_patterns = [
        "患者可以", "患者能否", "患者是否", "饮食注意", "饮食有什么注意", "能不能吃", "能否吃",
        "需要注意什么", "禁忌", "忌口"
    ]
    if any(p in q for p in personalized_patterns):
        return True

    return False


def _contains_clinical_markers(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    markers = [
        "结直肠癌", "结肠癌", "直肠癌", "癌", "肿瘤",
        "化疗", "放疗", "手术", "靶向", "免疫",
        "nccn", "csco", "esmo", "asco", "folfox", "capeox",
        "奥沙利铂", "伊立替康", "5-fu", "贝伐珠单抗", "西妥昔单抗"
    ]
    return any(m.lower() in t for m in markers)


def _filter_context_for_general_knowledge(query: str, context: str) -> str:
    """通用知识问答时过滤明显的临床/患者上下文"""
    if not context:
        return context
    if _contains_clinical_markers(context) and not _contains_clinical_markers(query):
        return ""
    return context


def _is_context_relevant(query: str, context: str) -> bool:
    """粗略判断本地上下文是否与问题相关，避免被不相关内容污染"""
    import re

    if not context or len(context) < 20:
        return False

    query_lower = query.strip().lower()
    context_lower = context.lower()

    # 特殊规则：维生素类问题需匹配具体字母/编号
    vitamin_tokens = re.findall(r"(?:维生素|vitamin)\s*([a-z0-9]+)", query_lower, flags=re.I)
    if vitamin_tokens:
        for token in vitamin_tokens:
            pattern = rf"(?:维生素|vitamin)\s*{re.escape(token)}"
            if not re.search(pattern, context_lower, flags=re.I):
                return False

    # 关键词重叠（中英文）
    keywords = set(re.findall(r"[a-z0-9]{2,}", query_lower))
    keywords.update(re.findall(r"[\u4e00-\u9fff]{2,}", query))

    stopwords = {
        "什么", "怎么", "如何", "是否", "可以", "能否", "作用", "用途", "有用", "相关", "需要",
        "what", "how", "use", "uses", "about", "need", "should"
    }
    keywords = {k for k in keywords if k not in stopwords}

    if not keywords:
        return True

    return any(k.lower() in context_lower for k in keywords)


def _get_complementary_searches(
    primary_plan: SearchPlan,
    primary_layer: str,
    web_tool_map: dict,
    original_query: str,
    patient_profile: Optional[PatientProfile]
) -> List[dict]:
    """
    获取补充搜索，确保三层证据完整性
    
    Returns:
        List of {"context": str, "layer": str, "tool": str}
    """
    complementary = []
    
    # Layer 1 (Authority): 如果主搜索不是权威层，补充指南搜索
    if primary_layer != 1:
        guideline_tool = web_tool_map.get("search_guideline_updates")
        if guideline_tool:
            try:
                # 提取关键主题词
                topic = _extract_topic(original_query)
                disease_query = f"{topic} NCCN guideline recommendation standard of care"
                res = guideline_tool.invoke({"disease": disease_query})
                complementary.append({
                    "context": f"【Layer 1 - Authority (Complementary)】\n{str(res)}",
                    "layer": "Layer 1 (Guideline)",
                    "tool": "search_guideline_updates"
                })
            except Exception as e:
                complementary.append({
                    "context": f"【Layer 1 - Authority (Complementary)】\nSearch failed: {e}",
                    "layer": "Layer 1 (Guideline)",
                    "tool": "search_guideline_updates"
                })
    
    # Layer 2 (Evidence): 如果主搜索不是证据层，补充临床试验搜索
    if primary_layer != 2:
        evidence_tool = web_tool_map.get("search_clinical_evidence")
        if evidence_tool:
            try:
                topic = _extract_topic(original_query)
                topic_query = f"{topic} clinical trial meta-analysis efficacy safety"
                res = evidence_tool.invoke({"topic": topic_query})
                complementary.append({
                    "context": f"【Layer 2 - Evidence (Complementary)】\n{str(res)}",
                    "layer": "Layer 2 (Evidence)",
                    "tool": "search_clinical_evidence"
                })
            except Exception as e:
                complementary.append({
                    "context": f"【Layer 2 - Evidence (Complementary)】\nSearch failed: {e}",
                    "layer": "Layer 2 (Evidence)",
                    "tool": "search_clinical_evidence"
                })
    
    # Layer 3 (Safety): 如果主搜索不是安全层，检查药物相互作用
    if primary_layer != 3:
        drug_tool = web_tool_map.get("search_drug_online")
        if drug_tool:
            try:
                topic = _extract_topic(original_query)
                # 获取患者当前的化疗药物（如果有）
                chemo_drugs = _extract_chemo_drugs(patient_profile)
                if chemo_drugs:
                    drug_query = f"{topic} interaction with {', '.join(chemo_drugs)}"
                else:
                    drug_query = f"{topic} interaction oxaliplatin 5-FU capecitabine"
                res = drug_tool.invoke({"drug_name": drug_query})
                complementary.append({
                    "context": f"【Layer 3 - Safety (Complementary)】\n{str(res)}",
                    "layer": "Layer 3 (Safety)",
                    "tool": "search_drug_online"
                })
            except Exception as e:
                complementary.append({
                    "context": f"【Layer 3 - Safety (Complementary)】\nSearch failed: {e}",
                    "layer": "Layer 3 (Safety)",
                    "tool": "search_drug_online"
                })
    
    return complementary


def _map_tool_arguments(tool_name: str, search_query: str, tool_arguments: Optional[dict]) -> dict:
    """
    根据工具名称映射正确的参数
    
    解决问题：不同工具需要不同的参数名（query, drug_name, topic, disease等）
    """
    # web_search 强制规范 query 长度（即使已有 tool_arguments）
    if tool_name == "web_search":
        query_value = None
        if tool_arguments and isinstance(tool_arguments, dict):
            query_value = tool_arguments.get("query")
        return {"query": _normalize_web_query(query_value or search_query)}

    # 如果已有有效的 tool_arguments，优先使用
    if tool_arguments and isinstance(tool_arguments, dict) and len(tool_arguments) > 0:
        # 检查是否包含有效参数（非空值）
        non_empty_args = {k: v for k, v in tool_arguments.items() if v}
        if non_empty_args:
            return tool_arguments
    
    # 根据工具名称映射参数
    tool_param_mapping = {
        "web_search": {"query": _normalize_web_query(search_query)},
        "search_drug_online": {"drug_name": search_query, "info_type": "all"},
        "search_clinical_evidence": {"topic": search_query},
        "search_guideline_updates": {"disease": search_query},
        "search_latest_research": {"topic": search_query},
    }
    
    # 返回映射的参数，如果找不到则默认使用 query
    return tool_param_mapping.get(tool_name, {"query": search_query})


def _extract_topic(query: str) -> str:
    """从查询中提取核心主题词"""
    # 简单的提取逻辑，实际可以使用更复杂的NLP
    words = query.split()
    # 过滤停用词
    stop_words = {"can", "is", "are", "do", "does", "what", "how", "why", "when", "where", "the", "a", "an", "能", "可以", "什么", "怎么", "为什么"}
    topic_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
    return " ".join(topic_words[:3]) if topic_words else query[:50]


def _normalize_web_query(query: Optional[str]) -> str:
    """保证 web_search 查询长度，避免工具校验失败"""
    q = (query or "").strip()
    if not q:
        return "医学 知识 作用 机制"
    if len(q) < 5:
        return f"{q} 作用 功效 机制"
    return q


def _extract_chemo_drugs(profile: Optional[PatientProfile]) -> List[str]:
    """从患者档案中提取化疗药物"""
    if not profile:
        return []
    chemo_keywords = ["oxaliplatin", "5-fu", "capecitabine", "folfox", "xelox", "folfoxiri", "bevacizumab", "cetuximab"]
    found = []
    
    # [修复] PatientProfile 是 Pydantic 模型，从 tnm_staging 属性获取
    # 检查 decision_json 和 findings 从 state 中获取，这里主要从其他来源检查
    # 由于 PatientProfile 没有 decision_json 和 findings 字段，这个函数需要调整逻辑
    
    # 从 profile 的 tnm_staging 中检查是否有治疗相关信息
    tnm_staging = profile.tnm_staging or {}
    staging_str = str(tnm_staging)
    
    for drug in chemo_keywords:
        if drug.lower() in staging_str.lower():
            if drug not in found:
                found.append(drug)
    
    return list(set(found))  # 去重


# ==============================================================================
# 5. 计划驱动模式的辅助函数
# ==============================================================================

def _extract_guideline_name(description: str) -> str:
    """从步骤描述中提取指南名称"""
    # 常见指南关键词
    guidelines = ["NCCN", "CSCO", "ESMO", "ASCO"]
    
    for g in guidelines:
        if g.upper() in description.upper():
            return g
    
    # 兜底：返回 NCCN
    return "NCCN"


def _extract_guideline_and_chapter(description: str) -> tuple[str, str]:
    """从步骤描述中提取指南名称和章节名称"""
    # 提取指南名称
    guideline_name = _extract_guideline_name(description)
    
    # 提取章节名称（简单的启发式方法）
    # 例如："读取 NCCN 指南的 'Stage III Treatment' 章节"
    import re
    
    # 尝试提取引号中的内容
    quoted = re.findall(r'["""\'](.*?)["""\']', description)
    if quoted:
        return guideline_name, quoted[0]
    
    # 尝试提取"的"之后、"章节"之前的内容
    match = re.search(r'的\s*([^\s]+)\s*章节', description)
    if match:
        return guideline_name, match.group(1)
    
    # 兜底：使用整个描述作为章节名
    return guideline_name, description.split("章节")[0].strip()


def _extract_search_query(description: str, state: CRCAgentState, use_patient_context: bool = True) -> str:
    """从步骤描述和状态中提取搜索查询"""
    # 1. 尝试从描述中提取
    import re
    
    # 如果描述中包含"检索"、"查询"等关键词，提取后面的内容
    patterns = [
        r'检索["""\']?(.*?)["""\']?(?:的|相关)',
        r'查询["""\']?(.*?)["""\']?(?:的|相关)',
        r'搜索["""\']?(.*?)["""\']?(?:的|相关)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            candidate = match.group(1).strip()
            # 过滤“查询到的”这种结构误提取的无意义片段
            if len(candidate) < 2 or candidate in {"到", "到的"}:
                continue
            return candidate
    
    # 2. 如果不允许使用患者上下文，退回到描述/最新用户问题
    if not use_patient_context:
        cleaned = description
        for word in ["检索", "查询", "搜索", "查看", "获取"]:
            cleaned = cleaned.replace(word, "")
        cleaned = cleaned.strip()
        return cleaned if cleaned else _latest_user_text(state)

    # 3. 从患者档案构建查询
    findings = state.findings or {}
    profile = state.patient_profile
    
    query_parts = []
    
    # 肿瘤部位
    tumor_location = findings.get("tumor_location") or (profile and getattr(profile, 'primary_site', None))
    if tumor_location:
        query_parts.append(tumor_location)
    
    # TNM 分期
    tnm = findings.get("tnm_staging") or (profile and profile.tnm_staging)
    if tnm and isinstance(tnm, dict):
        tnm_str = " ".join([v for v in tnm.values() if v])
        if tnm_str:
            query_parts.append(tnm_str)
    
    # 添加"治疗"关键词
    query_parts.extend(["治疗", "指南", "推荐"])
    
    # 4. 如果还是没有，使用描述本身
    if not query_parts:
        # 移除常见的动词
        cleaned = description
        for word in ["检索", "查询", "搜索", "查看", "获取"]:
            cleaned = cleaned.replace(word, "")
        return cleaned.strip()
    
    return " ".join(query_parts)

# ==============================================================================
# 4. Web Search Node (Standalone)
# ==============================================================================

def node_web_search_agent(tools: List[BaseTool]) -> Runnable:
    """
    Simple wrapper for web search if needed directly
    """
    web_tool = next((t for t in tools if hasattr(t, 'name') and t.name == "web_search"), None)
    
    def _run(state: CRCAgentState):
        query = _latest_user_text(state)
        if not web_tool:
            return {"messages": [AIMessage("No search tool available.")]}
        res = web_tool.invoke({"query": _normalize_web_query(query)})
        return {"messages": [AIMessage(f"Search Result:\n{res}")], "clinical_stage": "WebSearch"}
        
    return _run
