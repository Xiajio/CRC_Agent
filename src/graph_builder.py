import asyncio
import inspect
import time
from enum import Enum
from typing import List, Dict, Any, Callable

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from .checkpoint import get_checkpointer
from .config import Settings
from .nodes.clinical_nodes import (
    node_intent_classifier,
    node_doctor_assessment,
    node_patient_assessment,
    node_knowledge_retrieval,
    node_colon_staging,
    node_decision,
    node_diagnosis,
    node_rectal_staging,
    node_staging_router,
    node_critic,
    route_by_critic_v2,
    node_finalize,
    node_general_chat,
)
from .nodes.clinical_entry_nodes import (
    node_clinical_entry_resolver,
    node_outpatient_triage,
)
from .nodes.database_nodes import (
    node_case_database,
)
from .nodes.router import (
    dynamic_router,
    route_after_assessment,
    route_after_clinical_entry,
    route_after_intent,
)
from .policies.tool_targets import classify_pending_step_target
from .nodes.planner import node_planner
# [New] Parallel sub-agent node
from .nodes.parallel_subagents import node_parallel_subagents
# [New] Radiology agent node
from .nodes.radiology_nodes import node_rad_agent
from .nodes.pathology_nodes import node_pathology_agent
# [New] Web search and tool execution nodes
from .nodes.knowledge_nodes import node_web_search_agent
from .nodes.tools_executor import node_tool_executor
from .nodes.citation_nodes import node_citation_agent
from .nodes.evaluation_nodes import node_llm_judge, route_after_evaluator
from .nodes.memory_nodes import node_memory_manager
from .nodes.chat_main_node import node_chat_main
from .observability import init_observability
# [Optimization] Global Retriever warmup
from .rag import warmup_retriever
from .services.llm_service import LLMService
from .state import CRCAgentState, ensure_agent_state
from .tools import list_tools, list_tools_with_web_search


# === 1. 定义节点名称常量 (Type Safety) ===
class NodeName(str, Enum):
    """Graph Node Constants."""
    INTENT = "intent_router"
    PLANNER = "planner"  # [新] 规划节点
    KNOWLEDGE = "knowledge"
    CASE_DATABASE = "case_database"  # Case database query
    RAD_AGENT = "rad_agent"  # Radiology agent
    PATH_AGENT = "path_agent"  # Pathology agent
    WEB_SEARCH = "web_search"  # Web search agent
    TOOL_EXECUTOR = "tool_executor"  # Generic tool executor
    PARALLEL_SUBAGENTS = "parallel_subagents"  # [新] 并子智能体
    ASSESSMENT = "assessment"
    DIAGNOSIS = "diagnosis"
    STAGING_COLON = "colon_staging"
    STAGING_RECTAL = "rectal_staging"
    DECISION = "decision"
    CRITIC = "critic"
    CITATION = "citation"
    EVALUATOR = "evaluator"
    FINALIZE = "finalize"
    MEMORY = "memory_manager"  # Memory manager node (GC & Summary)
    CHAT_MAIN = "chat_main"  # Main chat node
    GENERAL_CHAT = "general_chat"
    CLINICAL_ENTRY_RESOLVER = "clinical_entry_resolver"
    OUTPATIENT_TRIAGE = "outpatient_triage"


_TIMED_NODE_NAMES = {
    NodeName.PLANNER,
    NodeName.ASSESSMENT,
    NodeName.DECISION,
    NodeName.CITATION,
    NodeName.EVALUATOR,
}


def _append_node_timing(output: Any, node_name: str, started_at: float) -> Dict[str, Any]:
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    normalized_name = node_name.value if isinstance(node_name, NodeName) else str(node_name)
    record = {
        "node": normalized_name,
        "duration_ms": duration_ms,
    }
    if isinstance(output, dict):
        merged = dict(output)
        existing = list(merged.get("node_timings") or [])
        existing.append(record)
        merged["node_timings"] = existing
        return merged
    return {"node_timings": [record]}


def _instrument_node(node_name: NodeName, node_fn: Callable[..., Any]) -> Callable[..., Any]:
    if node_name not in _TIMED_NODE_NAMES:
        return node_fn

    if inspect.iscoroutinefunction(node_fn):
        async def _async_wrapped(state: CRCAgentState):
            started_at = time.perf_counter()
            output = await node_fn(state)
            return _append_node_timing(output, node_name, started_at)

        return _async_wrapped

    def _sync_wrapped(state: CRCAgentState):
        started_at = time.perf_counter()
        output = node_fn(state)
        return _append_node_timing(output, node_name, started_at)

    return _sync_wrapped


# === 2. 路由逻辑 (Routing Logic) ===

def _plan_driven_router(state: CRCAgentState) -> str:
    """Route planner-driven tasks after intent classification."""
    from .nodes.router import dynamic_router

    plan = state.current_plan or []
    findings = state.findings or {}
    user_intent = findings.get("user_intent", "")

    # =================================================================
    # [Fix v2] Intent guard with highest priority.
    # Even if Planner produced a stale or incorrect plan,
    # off_topic_redirect must route to general chat.
    # This fallback prevents off-topic turns from entering the clinical flow.
    # =================================================================
    if user_intent == "off_topic_redirect":
        print("[Graph Router] off_topic_redirect detected; force route -> general_chat")
        return "general_chat"

    if not plan:
        # No plan, end the current turn.
        print("[Graph Router] no plan, end")
        return "end"

    is_multi_task = findings.get("multi_task_mode", False)
    
    # === 1. Check failed steps ===
    failed_steps = [s for s in plan if s.status == 'failed']
    if failed_steps:
        # Failed steps return to Planner for self-correction.
        failed_count = len(failed_steps)
        max_retry = max(s.retry_count for s in failed_steps)
        
        if max_retry >= 2:
            # Too many retries; check whether other tasks remain.
            pending_steps = [s for s in plan if s.status == 'pending']
            if pending_steps:
                print("[Graph Router] failed step retried too many times; continue pending steps")
            else:
                print("[Graph Router] failed step retried too many times; no pending steps, end")
                return "end"
        
        print(f"[Graph Router] detected {failed_count} failed steps; return to Planner for self-correction")
        return "planner"

    # === 1.5 Check parallel task group ===
    parallel_pending = next((s for s in plan if s.status == 'pending' and s.parallel_group), None)
    if parallel_pending:
        print(f"[Graph Router] detected parallel task group {parallel_pending.parallel_group}")
        return "parallel_subagents"
    
    # === 2. Check pending steps and dispatch directly, bypassing Planner ===
    pending_step = next((s for s in plan if s.status == 'pending'), None)
    if pending_step:
        print(f"[Graph Router] dispatch pending step without planner")
        print(f"  step: [{pending_step.id}] {pending_step.description}")
        print(f"  tool: {pending_step.tool_needed}")
        
        # Route directly by tool_needed.
        tool_type = pending_step.tool_needed.lower()
        target = classify_pending_step_target(
            pending_step.tool_needed,
            getattr(pending_step, "assignee", "") or "",
        )

        if target == "tool_executor":
            return target

        if any(kw in tool_type for kw in ["toc", "目录", "chapter", "章节", "search"]):
            return "knowledge"
        elif any(kw in tool_type for kw in ["database", "db", "case"]):
            return "case_database"
        elif any(kw in tool_type for kw in ["imaging", "影像", "tumor", "肿瘤", "ct", "mri"]):
            return "rad_agent"
        elif any(kw in tool_type for kw in ["pathology", "病理", "clam", "切片"]):
            return "path_agent"
        elif any(kw in tool_type for kw in ["ask_user", "询问"]):
            # [Fix v2] Only non-chat intents should continue to assessment.
            # Prevent casual/off-topic ask_user plans from entering clinical assessment.
            if user_intent in ["general_chat", "off_topic_redirect", "greeting", "thanks"]:
                print(f"[Graph Router] ask_user detected chat intent ({user_intent}); redirect -> general_chat")
                return "general_chat"
            return "assessment"
        else:
            # 兜底：默认去 knowledge 节点
            return "knowledge"
    
    # === 3. Plan complete, route by intent ===
    completed_count = sum(1 for s in plan if s.status == 'completed')
    print(f"[Graph Router] plan complete: {completed_count} completed steps")

    # 定义有需要进入临床评估流程的意图集合
    CLINICAL_INTENTS = {"treatment_decision", "clinical_assessment"}

    user_intent = findings.get("user_intent", "")
    sub_tasks = findings.get("sub_tasks", [])

    # Use set intersection for clearer extensible logic.
    needs_clinical_decision = (
        user_intent in CLINICAL_INTENTS or
        bool(set(sub_tasks) & CLINICAL_INTENTS)
    )

    if needs_clinical_decision:
        print(f"[Graph Router] clinical decision task detected (intent={user_intent}); route -> assessment")
        return "assessment"
    else:
        print("[Graph Router] non-clinical data/query task; route -> chat_main")
        return "chat_main"


def route_after_knowledge(state: CRCAgentState) -> str:
    """
    Route after Knowledge node execution.
    
    Use _plan_driven_router directly instead of returning to Planner.
    """
    return _plan_driven_router(state)


def route_after_case_database(state: CRCAgentState) -> str:
    """
    Route after Case Database node execution.
    
    Use _plan_driven_router directly instead of returning to Planner.
    """
    return _plan_driven_router(state)


def route_after_rad_agent(state: CRCAgentState) -> str:
    """
    Route after radiology agent execution.
    
    Use _plan_driven_router, with special handling for imaging-only tasks.
    
    Special rules:
    - If user intent is pure imaging analysis, end after the plan completes.
    - If multi-task mode includes treatment decision, continue into Assessment.
    """
    plan = state.current_plan or []
    findings = state.findings or {}
    
    # Check whether failed or pending steps remain.
    failed_steps = [s for s in plan if s.status == 'failed']
    pending_step = next((s for s in plan if s.status == 'pending'), None)
    
    # Failed steps return to Planner for self-correction.
    if failed_steps:
        print("[Graph Router] radiology agent found failed steps; return to Planner")
        return "planner"
    
    # Pending steps route directly through _plan_driven_router.
    if pending_step:
        return _plan_driven_router(state)
    
    # Plan complete, decide by intent.
    # 定义有需要进入临床评估流程的意图集合
    CLINICAL_INTENTS = {"treatment_decision", "clinical_assessment"}

    user_intent = findings.get("user_intent", "")
    sub_tasks = findings.get("sub_tasks", [])

    # Use set intersection for clearer extensible logic.
    needs_clinical_decision = (
        user_intent in CLINICAL_INTENTS or
        bool(set(sub_tasks) & CLINICAL_INTENTS)
    )

    if needs_clinical_decision:
        print(f"[Graph Router] radiology complete; clinical decision task detected (intent={user_intent}); route -> assessment")
        return "assessment"

    print(f"[Graph Router] 影像分析完成，用户意图为 {user_intent}，直接结束（不进入治疗流程）")
    return "end"


def route_after_path_agent(state: CRCAgentState) -> str:
    """
    Route after pathology agent execution.
    """
    plan = state.current_plan or []
    findings = state.findings or {}

    failed_steps = [s for s in plan if s.status == 'failed']
    pending_step = next((s for s in plan if s.status == 'pending'), None)

    if failed_steps:
        print("[Graph Router] pathology agent found failed steps; return to Planner")
        return "planner"

    if pending_step:
        return _plan_driven_router(state)

    CLINICAL_INTENTS = {"treatment_decision", "clinical_assessment"}
    user_intent = findings.get("user_intent", "")
    sub_tasks = findings.get("sub_tasks", [])

    needs_clinical_decision = (
        user_intent in CLINICAL_INTENTS or
        bool(set(sub_tasks) & CLINICAL_INTENTS)
    )

    if needs_clinical_decision:
        print(f"[Graph Router] pathology complete; clinical decision task detected (intent={user_intent}); route -> assessment")
        return "assessment"

    print(f"[Graph Router] 病理分析完成，用户意图为 {user_intent}，直接结束（不进入治疗流程）")
    return "end"


 
    




def route_after_patient_intent(state: CRCAgentState) -> str:
    target = route_after_intent(state)
    if target == "case_database":
        return "knowledge"
    if target in {"planner", "clinical_entry_resolver", "general_chat", "knowledge"}:
        return target
    return "general_chat"


def route_after_doctor_intent(state: CRCAgentState) -> str:
    target = route_after_intent(state)
    if target == "clinical_entry_resolver":
        return "assessment"
    if target == "chat_main":
        return "general_chat"
    return target


def route_after_doctor_clinical_entry(state: CRCAgentState) -> str:
    del state
    return "assessment"


def route_after_doctor_planner(state: CRCAgentState) -> str:
    target = dynamic_router(state)
    if target == "clinical_entry_resolver":
        return "assessment"
    if target == "chat_main":
        return "general_chat"
    return target


def route_after_doctor_followup(state: CRCAgentState) -> str:
    target = _plan_driven_router(state)
    if target == "chat_main":
        return "general_chat"
    return target


def route_after_doctor_post_assessment(state: CRCAgentState) -> str:
    target = route_after_assessment(state)
    if target == "chat_main":
        return "general_chat"
    return target


def route_after_doctor_rad_agent(state: CRCAgentState) -> str:
    target = route_after_rad_agent(state)
    if target == "chat_main":
        return "general_chat"
    return target


def route_after_doctor_path_agent(state: CRCAgentState) -> str:
    target = route_after_path_agent(state)
    if target == "chat_main":
        return "general_chat"
    return target


def route_after_patient_planner(state: CRCAgentState) -> str:
    target = dynamic_router(state)
    remapped_targets = {
        "planner": "planner",
        "chat_main": "chat_main",
        "general_chat": "general_chat",
        "knowledge": "knowledge",
        "case_database": "knowledge",
        "assessment": "assessment",
        "decision": "chat_main",
        "rad_agent": "knowledge",
        "path_agent": "knowledge",
        "web_search": "knowledge",
        "tool_executor": "knowledge",
        "parallel_subagents": "knowledge",
        "end_turn": "end",
        "end": "end",
    }
    return remapped_targets.get(target, "chat_main")


def route_after_patient_assessment(state: CRCAgentState) -> str:
    target = route_after_assessment(state)
    if target in {"chat_main", "general_chat"}:
        return target
    if target == "end_turn":
        return "end"
    return "chat_main"
# === 3. 工具加载辅助函数 (Modularity) ===
def _load_agent_tools(settings: Settings) -> List[BaseTool]:
    """
    Loads tools based on configuration with clear logging.
    Separates tool loading logic from graph construction.
    """
    if settings.web_search.enabled:
        tools = list_tools_with_web_search()
        web_count = sum(1 for t in tools if hasattr(t, 'name') and 'search' in t.name.lower())
        print(f"[Graph] Web Search: ENABLED (Model: {settings.web_search.model})")
        print(f"[Graph] Tools Loaded: {len(tools)} total ({web_count} web search tools)")
    else:
        tools = list_tools()
        print("[Graph] Web Search: DISABLED")
        print(f"[Graph] Tools Loaded: {len(tools)} total (Basic Set)")
        print("  - Hint: Set WEB_SEARCH_ENABLED=true to enable online capabilities.")
    return tools


# === 4. Build graph ===

def build_doctor_graph(settings: Settings) -> Runnable:

    # 1. Initialization
    init_observability(settings.observability)
    
    # Warm up global Retriever once at startup for reuse.
    warmup_retriever()
    
    tools = _load_agent_tools(settings)
    llm = LLMService(settings.llm).create_chat_model()

    common_config = {
        "model": llm,
        "streaming": settings.llm.streaming,
        "show_thinking": settings.llm.show_thinking
    }

    # 2. Graph Builder
    builder = StateGraph(CRCAgentState)

    # Clinical entry track nodes are registered alongside the existing graph.
    # They stay lightweight and only classify the encounter track.
    builder.add_node(NodeName.CLINICAL_ENTRY_RESOLVER, node_clinical_entry_resolver(**common_config))
    builder.add_node(NodeName.OUTPATIENT_TRIAGE, node_outpatient_triage(**common_config))

    # --- Nodes Registration ---
    builder.add_node(NodeName.INTENT, node_intent_classifier(**common_config))
    builder.add_node(NodeName.PLANNER, node_planner(**common_config))  # Planner node
    builder.add_node(NodeName.KNOWLEDGE, node_knowledge_retrieval(tools=tools, **common_config, use_sub_agent=False))
    builder.add_node(NodeName.CASE_DATABASE, node_case_database(tools=tools, **common_config))
    builder.add_node(NodeName.RAD_AGENT, node_rad_agent(tools=tools, **common_config))  # Radiology agent
    builder.add_node(NodeName.PATH_AGENT, node_pathology_agent(tools=tools, **common_config))  # Pathology agent
    builder.add_node(NodeName.WEB_SEARCH, node_web_search_agent(tools=tools))  # Web search agent
    builder.add_node(NodeName.TOOL_EXECUTOR, node_tool_executor)  # Generic tool executor
    builder.add_node(NodeName.PARALLEL_SUBAGENTS, node_parallel_subagents(tools=tools, **common_config))
    builder.add_node(NodeName.MEMORY, node_memory_manager(**common_config))  # [新] 内存管理

    # 核心临床节点
    builder.add_node(NodeName.ASSESSMENT, _instrument_node(NodeName.ASSESSMENT, node_doctor_assessment(tools=tools, **common_config)))
    builder.add_node(NodeName.DIAGNOSIS, node_diagnosis(tools=tools, **common_config))

    # Staging nodes
    builder.add_node(NodeName.STAGING_COLON, node_colon_staging(tools=tools))
    builder.add_node(NodeName.STAGING_RECTAL, node_rectal_staging(tools=tools))

    # Decision loop nodes
    builder.add_node(NodeName.DECISION, _instrument_node(NodeName.DECISION, node_decision(tools=tools, **common_config, use_sub_agent=False)))
    builder.add_node(NodeName.CRITIC, node_critic(**common_config))
    builder.add_node(NodeName.CITATION, _instrument_node(NodeName.CITATION, node_citation_agent(**common_config)))
    builder.add_node(NodeName.EVALUATOR, _instrument_node(NodeName.EVALUATOR, node_llm_judge(**common_config)))
    builder.add_node(NodeName.FINALIZE, node_finalize(**common_config))
    builder.add_node(NodeName.GENERAL_CHAT, node_general_chat(**common_config))

    # --- Edges Definition ---

    # Entry Point
    builder.set_entry_point(NodeName.INTENT)

    # INTENT -> direct route or Planner -> dynamic_router
    builder.add_conditional_edges(
        NodeName.INTENT,
        route_after_doctor_intent,
        {
            "planner": NodeName.PLANNER,
            "general_chat": NodeName.GENERAL_CHAT,
            "knowledge": NodeName.KNOWLEDGE,
            "case_database": NodeName.CASE_DATABASE,
            "assessment": NodeName.ASSESSMENT,
        },
    )

    builder.add_conditional_edges(
        NodeName.CLINICAL_ENTRY_RESOLVER,
        route_after_doctor_clinical_entry,
        {
            "assessment": NodeName.ASSESSMENT,
        },
    )

    builder.add_edge(NodeName.OUTPATIENT_TRIAGE, END)
    
    # Conditional routing after Planner.
    builder.add_conditional_edges(
        NodeName.PLANNER,
        route_after_doctor_planner,
        {
            "end_turn": END,  # Missing data question interrupts the turn.
            "general_chat": NodeName.GENERAL_CHAT,
            "knowledge": NodeName.KNOWLEDGE,
            "case_database": NodeName.CASE_DATABASE,  # Case database query
            "rad_agent": NodeName.RAD_AGENT,  # Radiology route
            "path_agent": NodeName.PATH_AGENT,  # Pathology route
            "assessment": NodeName.ASSESSMENT,  # Default assessment plan
            "decision": NodeName.DECISION,  # Fast Pass direct decision
            "web_search": NodeName.WEB_SEARCH,  # [新] 网络搜索
            "tool_executor": NodeName.TOOL_EXECUTOR,  # Tool executor
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,  # Parallel sub-agents
        },
    )
    
    # [Optimization v4.2] Smart routing after Knowledge.
    # 使用 _plan_driven_router 直接分发，无绕回 Planner
    builder.add_conditional_edges(
        NodeName.KNOWLEDGE,
        route_after_doctor_followup,
        {
            "planner": NodeName.PLANNER,       # Return to Planner only on failure.
            "knowledge": NodeName.KNOWLEDGE,   # Direct route to another knowledge step.
            "case_database": NodeName.CASE_DATABASE,  # Direct route to case_database.
            "rad_agent": NodeName.RAD_AGENT,   # Direct route to radiology analysis.
            "path_agent": NodeName.PATH_AGENT, # Direct route to pathology analysis.
            "assessment": NodeName.ASSESSMENT, # Plan complete; enter clinical decision flow.
            "general_chat": NodeName.GENERAL_CHAT,
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,
            "end": END,
        },
    )
    
    # [Optimization v4.2] Smart routing after Case Database.
    builder.add_conditional_edges(
        NodeName.CASE_DATABASE,
        route_after_doctor_followup,
        {
            "planner": NodeName.PLANNER,       # Return to Planner only on failure.
            "knowledge": NodeName.KNOWLEDGE,   # Direct route to knowledge.
            "case_database": NodeName.CASE_DATABASE,  # Direct route to another case_database step.
            "rad_agent": NodeName.RAD_AGENT,   # Direct route to radiology analysis.
            "path_agent": NodeName.PATH_AGENT, # Direct route to pathology analysis.
            "assessment": NodeName.ASSESSMENT, # Plan complete; enter clinical decision flow.
            "general_chat": NodeName.GENERAL_CHAT,
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,
            "end": END,
        },
    )

    # Clinical Flow: Assessment -> Diagnosis OR End (Inquiry)
    builder.add_conditional_edges(
        NodeName.ASSESSMENT,
        route_after_doctor_post_assessment,
        {
            "decision": NodeName.DECISION,  # Fast Pass route.
            "diagnosis": NodeName.DIAGNOSIS,
            "general_chat": NodeName.GENERAL_CHAT,
            "end_turn": END,  # Active inquiry interrupts the turn.
        },
    )

    # Diagnosis -> Staging
    builder.add_conditional_edges(
        NodeName.DIAGNOSIS,
        node_staging_router,
        {
            "colon_staging": NodeName.STAGING_COLON,
            "rectal_staging": NodeName.STAGING_RECTAL,
            "decision": NodeName.DECISION,
            "end_turn": END,
        },
    )

    # Staging -> Decision
    builder.add_edge(NodeName.STAGING_COLON, NodeName.DECISION)
    builder.add_edge(NodeName.STAGING_RECTAL, NodeName.DECISION)

    # Decision -> Critic
    builder.add_edge(NodeName.DECISION, NodeName.CRITIC)

    # Critic -> Finalize OR Loop
    builder.add_conditional_edges(
        NodeName.CRITIC,
        route_by_critic_v2,
        {
            "decision": NodeName.DECISION,
            "finalize": NodeName.CITATION,
        },
    )

    # Citation -> Evaluator
    builder.add_edge(NodeName.CITATION, NodeName.EVALUATOR)

    # Evaluator -> Decision OR Finalize
    builder.add_conditional_edges(
        NodeName.EVALUATOR,
        route_after_evaluator,
        {
            "decision": NodeName.DECISION,
            "finalize": NodeName.FINALIZE,
        },
    )

    # Finalize -> End
    builder.add_edge(NodeName.FINALIZE, END)
    builder.add_edge(NodeName.GENERAL_CHAT, END)
    # Removed direct KNOWLEDGE -> END edge; route via route_after_knowledge.
    # Removed direct CASE_DATABASE -> END edge; route via route_after_case_database.
    
    # [Optimization v4.2] Smart routing after RadAgent.
    # 使用 _plan_driven_router 直接分发，无绕回 Planner
    builder.add_conditional_edges(
        NodeName.RAD_AGENT,
        route_after_doctor_rad_agent,
        {
            "planner": NodeName.PLANNER,       # Return to Planner only on failure.
            "knowledge": NodeName.KNOWLEDGE,   # Direct route to knowledge.
            "case_database": NodeName.CASE_DATABASE,  # Direct route to case_database.
            "rad_agent": NodeName.RAD_AGENT,   # Direct route to the next radiology step.
            "assessment": NodeName.ASSESSMENT, # Includes treatment decision; enter clinical flow.
            "general_chat": NodeName.GENERAL_CHAT,
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,
            "end": END,
        },
    )

    # [New] Smart routing after PathAgent.
    builder.add_conditional_edges(
        NodeName.PATH_AGENT,
        route_after_doctor_path_agent,
        {
            "planner": NodeName.PLANNER,       # Return to Planner only on failure.
            "knowledge": NodeName.KNOWLEDGE,   # Direct route to knowledge.
            "case_database": NodeName.CASE_DATABASE,  # Direct route to case_database.
            "path_agent": NodeName.PATH_AGENT, # Direct route to the next pathology step.
            "assessment": NodeName.ASSESSMENT, # Includes treatment decision; enter clinical flow.
            "general_chat": NodeName.GENERAL_CHAT,
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,
            "end": END,
        },
    )

    # Parallel Subagents -> Router
    builder.add_conditional_edges(
        NodeName.PARALLEL_SUBAGENTS,
        route_after_doctor_followup,
        {
            "planner": NodeName.PLANNER,
            "knowledge": NodeName.KNOWLEDGE,
            "case_database": NodeName.CASE_DATABASE,
            "rad_agent": NodeName.RAD_AGENT,
            "path_agent": NodeName.PATH_AGENT,
            "assessment": NodeName.ASSESSMENT,
            "general_chat": NodeName.GENERAL_CHAT,
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,
            "end": END,
        },
    )
    
    # Memory -> End
    builder.add_edge(NodeName.MEMORY, END)

    # 3. Compilation
    graph = builder.compile(checkpointer=get_checkpointer(settings.checkpoint))

    return graph




def build_patient_graph(settings: Settings) -> Runnable:
    init_observability(settings.observability)
    warmup_retriever()

    tools = _load_agent_tools(settings)
    llm = LLMService(settings.llm).create_chat_model()

    common_config = {
        "model": llm,
        "streaming": settings.llm.streaming,
        "show_thinking": settings.llm.show_thinking,
    }

    builder = StateGraph(CRCAgentState)
    builder.add_node(NodeName.INTENT, node_intent_classifier(**common_config))
    builder.add_node(NodeName.PLANNER, node_planner(**common_config))
    builder.add_node(NodeName.CLINICAL_ENTRY_RESOLVER, node_clinical_entry_resolver(**common_config))
    builder.add_node(NodeName.OUTPATIENT_TRIAGE, node_outpatient_triage(**common_config))
    builder.add_node(NodeName.KNOWLEDGE, node_knowledge_retrieval(tools=tools, **common_config, use_sub_agent=False))
    builder.add_node(NodeName.ASSESSMENT, _instrument_node(NodeName.ASSESSMENT, node_patient_assessment(tools=tools, **common_config)))
    builder.add_node(NodeName.CHAT_MAIN, node_chat_main(**common_config))
    builder.add_node(NodeName.GENERAL_CHAT, node_general_chat(**common_config))

    builder.set_entry_point(NodeName.INTENT)
    builder.add_conditional_edges(
        NodeName.INTENT,
        route_after_patient_intent,
        {
            "planner": NodeName.PLANNER,
            "clinical_entry_resolver": NodeName.CLINICAL_ENTRY_RESOLVER,
            "general_chat": NodeName.GENERAL_CHAT,
            "knowledge": NodeName.KNOWLEDGE,
        },
    )
    builder.add_conditional_edges(
        NodeName.CLINICAL_ENTRY_RESOLVER,
        route_after_clinical_entry,
        {
            "assessment": NodeName.ASSESSMENT,
            "outpatient_triage": NodeName.OUTPATIENT_TRIAGE,
        },
    )
    builder.add_edge(NodeName.OUTPATIENT_TRIAGE, END)
    builder.add_conditional_edges(
        NodeName.PLANNER,
        route_after_patient_planner,
        {
            "planner": NodeName.PLANNER,
            "chat_main": NodeName.CHAT_MAIN,
            "general_chat": NodeName.GENERAL_CHAT,
            "knowledge": NodeName.KNOWLEDGE,
            "assessment": NodeName.ASSESSMENT,
            "end": END,
        },
    )
    builder.add_edge(NodeName.KNOWLEDGE, END)
    builder.add_conditional_edges(
        NodeName.ASSESSMENT,
        route_after_patient_assessment,
        {
            "chat_main": NodeName.CHAT_MAIN,
            "general_chat": NodeName.GENERAL_CHAT,
            "end": END,
        },
    )
    builder.add_edge(NodeName.CHAT_MAIN, END)
    builder.add_edge(NodeName.GENERAL_CHAT, END)

    return builder.compile(checkpointer=get_checkpointer(settings.checkpoint))


def build_graph(settings: Settings) -> Runnable:
    return build_doctor_graph(settings)
def simple_run(message: str):
    """Helper for quick end-to-end testing."""
    settings = Settings()
    graph = build_graph(settings)

    print(f"\nRunning simple test with query: '{message}'")

    async def _run():
        return await graph.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "patient_profile": {},
                "findings": {}
            },
            config={"configurable": {"thread_id": "simple-run-test"}},
        )

    result = asyncio.run(_run())
    return ensure_agent_state(result)

