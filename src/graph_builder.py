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
# [鏂板] 骞惰瀛愭櫤鑳戒綋鑺傜偣
from .nodes.parallel_subagents import node_parallel_subagents
# [鏂板] 寮曞叆褰卞儚鍖荤敓鏅鸿兘浣撹妭鐐?
from .nodes.radiology_nodes import node_rad_agent
from .nodes.pathology_nodes import node_pathology_agent
# [鏂板] 寮曞叆缃戠粶鎼滅储鍜屽伐鍏锋墽琛岃妭鐐?
from .nodes.knowledge_nodes import node_web_search_agent
from .nodes.tools_executor import node_tool_executor
from .nodes.citation_nodes import node_citation_agent
from .nodes.evaluation_nodes import node_llm_judge, route_after_evaluator
from .nodes.memory_nodes import node_memory_manager
from .nodes.chat_main_node import node_chat_main
from .observability import init_observability
# [浼樺寲] 寮曞叆鍏ㄥ眬 Retriever 棰勭儹
from .rag import warmup_retriever
from .services.llm_service import LLMService
from .state import CRCAgentState, ensure_agent_state
from .tools import list_tools, list_tools_with_web_search


# === 1. 瀹氫箟鑺傜偣鍚嶇О甯搁噺 (Type Safety) ===
class NodeName(str, Enum):
    """Graph Node Constants."""
    INTENT = "intent_router"
    PLANNER = "planner"  # [鏂板] 瑙勫垝鑺傜偣
    KNOWLEDGE = "knowledge"
    CASE_DATABASE = "case_database"  # 鏂板锛氱梾渚嬫暟鎹簱鏌ヨ
    RAD_AGENT = "rad_agent"  # [鏂板] 褰卞儚鍖荤敓鏅鸿兘浣?
    PATH_AGENT = "path_agent"  # [鏂板] 鐥呯悊鍖荤敓鏅鸿兘浣?
    WEB_SEARCH = "web_search"  # [鏂板] 缃戠粶鎼滅储鏅鸿兘浣?
    TOOL_EXECUTOR = "tool_executor"  # [鏂板] 閫氱敤宸ュ叿鎵ц鍣?
    PARALLEL_SUBAGENTS = "parallel_subagents"  # [鏂板] 骞惰瀛愭櫤鑳戒綋
    ASSESSMENT = "assessment"
    DIAGNOSIS = "diagnosis"
    STAGING_COLON = "colon_staging"
    STAGING_RECTAL = "rectal_staging"
    DECISION = "decision"
    CRITIC = "critic"
    CITATION = "citation"
    EVALUATOR = "evaluator"
    FINALIZE = "finalize"
    MEMORY = "memory_manager"  # [鏂板] 鍐呭瓨绠＄悊鑺傜偣 (GC & Summary)
    CHAT_MAIN = "chat_main"  # [鏂板] 鑱婂ぉ涓昏妭鐐?
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


# === 2. 璺敱閫昏緫 (Routing Logic) ===

def _plan_driven_router(state: CRCAgentState) -> str:
    """Route planner-driven tasks after intent classification."""
    from .nodes.router import dynamic_router

    plan = state.current_plan or []
    findings = state.findings or {}
    user_intent = findings.get("user_intent", "")

    # =================================================================
    # [淇 v2] 鎰忓浘鐔旀柇鏈哄埗锛堟渶楂樹紭鍏堢骇锛?
    # 鍗充娇 Planner 閿欒鍦扮敓鎴愪簡璁″垝锛屾垨鑰呮湁閬楃暀璁″垝锛?
    # 鍙鎰忓浘鏄?off_topic_redirect锛屽繀椤诲己鍒跺幓闂茶亰鑺傜偣銆?
    # 杩欐槸鍏滃簳闃插尽锛岄槻姝?Planner 婕忕綉涔嬮奔杩涘叆涓村簥娴佺▼銆?
    # =================================================================
    if user_intent == "off_topic_redirect":
        print("[Graph Router] 馃洝锔?鎰忓浘鐔旀柇锛氭娴嬪埌鍋忛鎰忓浘(off_topic_redirect)锛屽己鍒惰矾鐢?-> general_chat")
        return "general_chat"

    if not plan:
        # 娌℃湁璁″垝锛岀洿鎺ョ粨鏉?
        print("[Graph Router] no plan, end")
        return "end"

    is_multi_task = findings.get("multi_task_mode", False)
    
    # === 1. 妫€鏌ュけ璐ユ楠?===
    failed_steps = [s for s in plan if s.status == 'failed']
    if failed_steps:
        # 澶辫触闇€瑕佸洖鍒?Planner 杩涜鑷垜绾犻敊
        failed_count = len(failed_steps)
        max_retry = max(s.retry_count for s in failed_steps)
        
        if max_retry >= 2:
            # 閲嶈瘯娆℃暟杩囧锛屾鏌ユ槸鍚︽湁鍏朵粬浠诲姟
            pending_steps = [s for s in plan if s.status == 'pending']
            if pending_steps:
                print(f"[Graph Router] 澶辫触姝ラ閲嶈瘯杩囧锛屼絾杩樻湁寰呮墽琛屾楠わ紝缁х画")
            else:
                print(f"[Graph Router] 澶辫触姝ラ閲嶈瘯杩囧锛屾棤鍏朵粬寰呮墽琛屾楠わ紝缁撴潫")
                return "end"
        
        print(f"[Graph Router] 妫€娴嬪埌 {failed_count} 涓け璐ユ楠わ紝鍥炲埌 Planner 鑷垜绾犻敊")
        return "planner"

    # === 1.5 妫€鏌ュ苟琛屼换鍔＄粍 ===
    parallel_pending = next((s for s in plan if s.status == 'pending' and s.parallel_group), None)
    if parallel_pending:
        print(f"[Graph Router] 鈿?妫€娴嬪埌骞惰浠诲姟缁? {parallel_pending.parallel_group}")
        return "parallel_subagents"
    
    # === 2. 妫€鏌ュ緟鎵ц姝ラ - 鐩存帴璺敱锛堢粫杩?Planner锛?==
    pending_step = next((s for s in plan if s.status == 'pending'), None)
    if pending_step:
        print(f"[Graph Router] dispatch pending step without planner")
        print(f"  姝ラ: [{pending_step.id}] {pending_step.description}")
        print(f"  宸ュ叿: {pending_step.tool_needed}")
        
        # 鏍规嵁 tool_needed 鐩存帴璺敱鍒板搴旇妭鐐?
        tool_type = pending_step.tool_needed.lower()
        target = classify_pending_step_target(
            pending_step.tool_needed,
            getattr(pending_step, "assignee", "") or "",
        )

        if target == "tool_executor":
            return target

        if any(kw in tool_type for kw in ["toc", "鐩綍", "chapter", "绔犺妭", "search"]):
            return "knowledge"
        elif any(kw in tool_type for kw in ["database", "db", "case"]):
            return "case_database"
        elif any(kw in tool_type for kw in ["imaging", "褰卞儚", "tumor", "鑲跨槫", "ct", "mri"]):
            return "rad_agent"
        elif any(kw in tool_type for kw in ["pathology", "鐥呯悊", "clam", "鍒囩墖"]):
            return "path_agent"
        elif any(kw in tool_type for kw in ["ask_user", "璇㈤棶"]):
            # [淇 v2] 鍙岄噸淇濋櫓锛氬彧鏈夐潪闂茶亰鎰忓浘鎵嶅幓 assessment
            # 闃叉闂茶亰/鍋忛鎰忓浘鐨勯敊璇鍒掕璺敱鍒颁复搴婅瘎浼?
            if user_intent in ["general_chat", "off_topic_redirect", "greeting", "thanks"]:
                print(f"[Graph Router] 鈿狅笍 ask_user宸ュ叿妫€娴嬪埌闂茶亰鎰忓浘({user_intent})锛岄噸瀹氬悜 -> general_chat")
                return "general_chat"
            return "assessment"
        else:
            # 鍏滃簳锛氶粯璁ゅ幓 knowledge 鑺傜偣
            return "knowledge"
    
    # === 3. 璁″垝瀹屾垚 - 鏍规嵁鎰忓浘璺敱 ===
    completed_count = sum(1 for s in plan if s.status == 'completed')
    print(f"[Graph Router] plan complete: {completed_count} completed steps")

    # 瀹氫箟鎵€鏈夐渶瑕佽繘鍏ヤ复搴婅瘎浼版祦绋嬬殑鎰忓浘闆嗗悎
    CLINICAL_INTENTS = {"treatment_decision", "clinical_assessment"}

    user_intent = findings.get("user_intent", "")
    sub_tasks = findings.get("sub_tasks", [])

    # 浣跨敤闆嗗悎浜ら泦鍒ゆ柇锛岄€昏緫鏇存竻鏅颁笖鏄撴墿灞?
    needs_clinical_decision = (
        user_intent in CLINICAL_INTENTS or
        bool(set(sub_tasks) & CLINICAL_INTENTS)
    )

    if needs_clinical_decision:
        print(f"[Graph Router] 妫€娴嬪埌涓村簥鍐崇瓥浠诲姟 (intent={user_intent})锛岃繘鍏ヤ复搴婅瘎浼版祦绋?..")
        return "assessment"
    else:
        print(f"[Graph Router] 绾煡璇?鏁版嵁鏌ヨ锛岀敓鎴愮患鍚堝搷搴?..")
        return "chat_main"


def route_after_knowledge(state: CRCAgentState) -> str:
    """
    Knowledge 鑺傜偣鎵ц鍚庣殑璺敱閫昏緫锛堜紭鍖栫増锛?
    
    [浼樺寲] 鐩存帴浣跨敤 _plan_driven_router 杩涜璺敱锛屾棤闇€鍥炲埌 Planner
    """
    return _plan_driven_router(state)


def route_after_case_database(state: CRCAgentState) -> str:
    """
    Case Database 鑺傜偣鎵ц鍚庣殑璺敱閫昏緫锛堜紭鍖栫増锛?
    
    [浼樺寲] 鐩存帴浣跨敤 _plan_driven_router 杩涜璺敱锛屾棤闇€鍥炲埌 Planner
    """
    return _plan_driven_router(state)


def route_after_rad_agent(state: CRCAgentState) -> str:
    """
    褰卞儚鍒嗘瀽鑺傜偣 (rad_agent) 鎵ц鍚庣殑璺敱閫昏緫锛堜紭鍖栫増锛?
    
    [浼樺寲] 浣跨敤 _plan_driven_router 缁熶竴璺敱锛岀壒娈婂鐞嗗奖鍍忓垎鏋愬満鏅?
    
    鐗规畩瑙勫垯锛?
    - 濡傛灉鐢ㄦ埛鎰忓浘鏄函褰卞儚鍒嗘瀽 (imaging_analysis)锛岃鍒掑畬鎴愬悗鐩存帴缁撴潫
    - 濡傛灉鏄浠诲姟妯″紡涓斿寘鍚不鐤楀喅绛栦换鍔★紝鎵嶇户缁繘鍏?Assessment
    """
    plan = state.current_plan or []
    findings = state.findings or {}
    
    # 妫€鏌ユ槸鍚︽湁澶辫触鎴栧緟鎵ц姝ラ
    failed_steps = [s for s in plan if s.status == 'failed']
    pending_step = next((s for s in plan if s.status == 'pending'), None)
    
    # 濡傛灉鏈夊け璐ユ楠わ紝鍥炲埌 Planner 杩涜鑷垜绾犻敊
    if failed_steps:
        print(f"[Graph Router] 褰卞儚鍒嗘瀽鍚庢娴嬪埌澶辫触姝ラ锛屽洖鍒?Planner 鑷垜绾犻敊")
        return "planner"
    
    # 濡傛灉鏈夊緟鎵ц姝ラ锛岀洿鎺ヨ矾鐢憋紙浣跨敤浼樺寲鐨?_plan_driven_router锛?
    if pending_step:
        return _plan_driven_router(state)
    
    # 璁″垝瀹屾垚锛屾牴鎹剰鍥惧喅瀹?
    # 瀹氫箟鎵€鏈夐渶瑕佽繘鍏ヤ复搴婅瘎浼版祦绋嬬殑鎰忓浘闆嗗悎
    CLINICAL_INTENTS = {"treatment_decision", "clinical_assessment"}

    user_intent = findings.get("user_intent", "")
    sub_tasks = findings.get("sub_tasks", [])

    # 浣跨敤闆嗗悎浜ら泦鍒ゆ柇锛岄€昏緫鏇存竻鏅颁笖鏄撴墿灞?
    needs_clinical_decision = (
        user_intent in CLINICAL_INTENTS or
        bool(set(sub_tasks) & CLINICAL_INTENTS)
    )

    if needs_clinical_decision:
        print(f"[Graph Router] 褰卞儚鍒嗘瀽瀹屾垚锛屾娴嬪埌涓村簥鍐崇瓥浠诲姟 (intent={user_intent})锛岃繘鍏ヤ复搴婅瘎浼版祦绋?..")
        return "assessment"

    print(f"[Graph Router] 褰卞儚鍒嗘瀽瀹屾垚锛岀敤鎴锋剰鍥句负 {user_intent}锛岀洿鎺ョ粨鏉燂紙涓嶈繘鍏ユ不鐤楁祦绋嬶級")
    return "end"


def route_after_path_agent(state: CRCAgentState) -> str:
    """
    鐥呯悊鍒嗘瀽鑺傜偣 (path_agent) 鎵ц鍚庣殑璺敱閫昏緫锛堜紭鍖栫増锛?
    """
    plan = state.current_plan or []
    findings = state.findings or {}

    failed_steps = [s for s in plan if s.status == 'failed']
    pending_step = next((s for s in plan if s.status == 'pending'), None)

    if failed_steps:
        print("[Graph Router] 鐥呯悊鍒嗘瀽鍚庢娴嬪埌澶辫触姝ラ锛屽洖鍒?Planner 鑷垜绾犻敊")
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
        print(f"[Graph Router] 鐥呯悊鍒嗘瀽瀹屾垚锛屾娴嬪埌涓村簥鍐崇瓥浠诲姟 (intent={user_intent})锛岃繘鍏ヤ复搴婅瘎浼版祦绋?..")
        return "assessment"

    print(f"[Graph Router] 鐥呯悊鍒嗘瀽瀹屾垚锛岀敤鎴锋剰鍥句负 {user_intent}锛岀洿鎺ョ粨鏉燂紙涓嶈繘鍏ユ不鐤楁祦绋嬶級")
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
# === 3. 宸ュ叿鍔犺浇杈呭姪鍑芥暟 (Modularity) ===
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
        print(f"[Graph] Web Search: 鉂?DISABLED")
        print(f"[Graph] Tools Loaded: {len(tools)} total (Basic Set)")
        print("  - Hint: Set WEB_SEARCH_ENABLED=true to enable online capabilities.")
    return tools


# === 4. Graph 鏋勫缓 ===

def build_doctor_graph(settings: Settings) -> Runnable:

    # 1. Initialization
    init_observability(settings.observability)
    
    # [浼樺寲] 棰勭儹鍏ㄥ眬 Retriever锛堝湪绯荤粺鍚姩鏃跺垵濮嬪寲涓€娆★紝鍚庣画澶嶇敤锛?
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
    builder.add_node(NodeName.PLANNER, node_planner(**common_config))  # [鏂板] 瑙勫垝鑺傜偣锛坢odel 宸插湪 common_config 涓級
    builder.add_node(NodeName.KNOWLEDGE, node_knowledge_retrieval(tools=tools, **common_config, use_sub_agent=False))
    builder.add_node(NodeName.CASE_DATABASE, node_case_database(tools=tools, **common_config))  # 鏂板
    builder.add_node(NodeName.RAD_AGENT, node_rad_agent(tools=tools, **common_config))  # [鏂板] 褰卞儚鍖荤敓鏅鸿兘浣?
    builder.add_node(NodeName.PATH_AGENT, node_pathology_agent(tools=tools, **common_config))  # [鏂板] 鐥呯悊鍖荤敓鏅鸿兘浣?
    builder.add_node(NodeName.WEB_SEARCH, node_web_search_agent(tools=tools))  # [鏂板] 缃戠粶鎼滅储鏅鸿兘浣?
    builder.add_node(NodeName.TOOL_EXECUTOR, node_tool_executor)  # [鏂板] 閫氱敤宸ュ叿鎵ц鍣?
    builder.add_node(NodeName.PARALLEL_SUBAGENTS, node_parallel_subagents(tools=tools, **common_config))
    builder.add_node(NodeName.MEMORY, node_memory_manager(**common_config))  # [鏂板] 鍐呭瓨绠＄悊

    # 鏍稿績涓村簥鑺傜偣
    builder.add_node(NodeName.ASSESSMENT, _instrument_node(NodeName.ASSESSMENT, node_doctor_assessment(tools=tools, **common_config)))
    builder.add_node(NodeName.DIAGNOSIS, node_diagnosis(tools=tools, **common_config))

    # Staging 鑺傜偣
    builder.add_node(NodeName.STAGING_COLON, node_colon_staging(tools=tools))
    builder.add_node(NodeName.STAGING_RECTAL, node_rectal_staging(tools=tools))

    # 鍐崇瓥闂幆鑺傜偣
    builder.add_node(NodeName.DECISION, _instrument_node(NodeName.DECISION, node_decision(tools=tools, **common_config, use_sub_agent=False)))
    builder.add_node(NodeName.CRITIC, node_critic(**common_config))
    builder.add_node(NodeName.CITATION, _instrument_node(NodeName.CITATION, node_citation_agent(**common_config)))
    builder.add_node(NodeName.EVALUATOR, _instrument_node(NodeName.EVALUATOR, node_llm_judge(**common_config)))
    builder.add_node(NodeName.FINALIZE, node_finalize(**common_config))
    builder.add_node(NodeName.GENERAL_CHAT, node_general_chat(**common_config))

    # --- Edges Definition ---

    # Entry Point
    builder.set_entry_point(NodeName.INTENT)

    # INTENT -> (鍗曡烦鐩磋揪 | Planner) -> dynamic_router
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
    
    # Planner 涔嬪悗鐨勫姩鎬佽矾鐢?
    builder.add_conditional_edges(
        NodeName.PLANNER,
        route_after_doctor_planner,
        {
            "end_turn": END,  # P0: 缂烘暟鎹拷闂紝涓柇 -> End
            "general_chat": NodeName.GENERAL_CHAT,
            "knowledge": NodeName.KNOWLEDGE,
            "case_database": NodeName.CASE_DATABASE,  # 鏂板锛氱梾渚嬫暟鎹簱鏌ヨ
            "rad_agent": NodeName.RAD_AGENT,  # [鏂板] 褰卞儚鍒嗘瀽璺敱
            "path_agent": NodeName.PATH_AGENT,  # [鏂板] 鐥呯悊鍒嗘瀽璺敱
            "assessment": NodeName.ASSESSMENT,  # 榛樿鍘昏瘎浼?瑙勫垝
            "decision": NodeName.DECISION,  # Fast Pass 鐩存帴鍘诲喅绛?
            "web_search": NodeName.WEB_SEARCH,  # [鏂板] 缃戠粶鎼滅储
            "tool_executor": NodeName.TOOL_EXECUTOR,  # [鏂板] 宸ュ叿鎵ц鍣?
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,  # [鏂板] 骞惰瀛愭櫤鑳戒綋
        },
    )
    
    # [浼樺寲 v4.2] Knowledge 鑺傜偣鎵ц鍚庣殑鏅鸿兘璺敱
    # 浣跨敤 _plan_driven_router 鐩存帴鍒嗗彂锛屾棤闇€缁曞洖 Planner
    builder.add_conditional_edges(
        NodeName.KNOWLEDGE,
        route_after_doctor_followup,
        {
            "planner": NodeName.PLANNER,       # 浠呭湪澶辫触鏃跺洖鍒?Planner 鑷垜绾犻敊
            "knowledge": NodeName.KNOWLEDGE,   # [浼樺寲] 鐩存帴璺敱鍒颁笅涓€涓?knowledge 姝ラ
            "case_database": NodeName.CASE_DATABASE,  # [浼樺寲] 鐩存帴璺敱鍒?case_database
            "rad_agent": NodeName.RAD_AGENT,   # [浼樺寲] 鐩存帴璺敱鍒板奖鍍忓垎鏋?
            "path_agent": NodeName.PATH_AGENT, # [浼樺寲] 鐩存帴璺敱鍒扮梾鐞嗗垎鏋?
            "assessment": NodeName.ASSESSMENT, # 璁″垝瀹屾垚锛岃繘鍏ヤ复搴婂喅绛栨祦绋?
            "general_chat": NodeName.GENERAL_CHAT,
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,
            "end": END,
        },
    )
    
    # [浼樺寲 v4.2] Case Database 鑺傜偣鎵ц鍚庣殑鏅鸿兘璺敱
    builder.add_conditional_edges(
        NodeName.CASE_DATABASE,
        route_after_doctor_followup,
        {
            "planner": NodeName.PLANNER,       # 浠呭湪澶辫触鏃跺洖鍒?Planner 鑷垜绾犻敊
            "knowledge": NodeName.KNOWLEDGE,   # [浼樺寲] 鐩存帴璺敱鍒?knowledge
            "case_database": NodeName.CASE_DATABASE,  # [浼樺寲] 鐩存帴璺敱鍒颁笅涓€涓?case_database 姝ラ
            "rad_agent": NodeName.RAD_AGENT,   # [浼樺寲] 鐩存帴璺敱鍒板奖鍍忓垎鏋?
            "path_agent": NodeName.PATH_AGENT, # [浼樺寲] 鐩存帴璺敱鍒扮梾鐞嗗垎鏋?
            "assessment": NodeName.ASSESSMENT, # 璁″垝瀹屾垚锛岃繘鍏ヤ复搴婂喅绛栨祦绋?
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
            "decision": NodeName.DECISION,  # Fast Pass 璺崇骇
            "diagnosis": NodeName.DIAGNOSIS,
            "general_chat": NodeName.GENERAL_CHAT,
            "end_turn": END,  # 涓诲姩杩介棶涓柇
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
    # [宸插垹闄 builder.add_edge(NodeName.KNOWLEDGE, END)  # 鏀圭敤 route_after_knowledge
    # [宸插垹闄 builder.add_edge(NodeName.CASE_DATABASE, END)  # 鏀圭敤 route_after_case_database
    
    # [浼樺寲 v4.2] RadAgent 鑺傜偣鎵ц鍚庣殑鏅鸿兘璺敱
    # 浣跨敤 _plan_driven_router 鐩存帴鍒嗗彂锛屾棤闇€缁曞洖 Planner
    builder.add_conditional_edges(
        NodeName.RAD_AGENT,
        route_after_doctor_rad_agent,
        {
            "planner": NodeName.PLANNER,       # 浠呭湪澶辫触鏃跺洖鍒?Planner 鑷垜绾犻敊
            "knowledge": NodeName.KNOWLEDGE,   # [浼樺寲] 鐩存帴璺敱鍒?knowledge
            "case_database": NodeName.CASE_DATABASE,  # [浼樺寲] 鐩存帴璺敱鍒?case_database
            "rad_agent": NodeName.RAD_AGENT,   # [浼樺寲] 鐩存帴璺敱鍒颁笅涓€涓奖鍍忓垎鏋愭楠?
            "assessment": NodeName.ASSESSMENT, # 鍖呭惈娌荤枟鍐崇瓥浠诲姟锛岃繘鍏ヤ复搴婃祦绋?
            "general_chat": NodeName.GENERAL_CHAT,
            "parallel_subagents": NodeName.PARALLEL_SUBAGENTS,
            "end": END,
        },
    )

    # [鏂板] PathAgent 鑺傜偣鎵ц鍚庣殑鏅鸿兘璺敱
    builder.add_conditional_edges(
        NodeName.PATH_AGENT,
        route_after_doctor_path_agent,
        {
            "planner": NodeName.PLANNER,       # 浠呭湪澶辫触鏃跺洖鍒?Planner 鑷垜绾犻敊
            "knowledge": NodeName.KNOWLEDGE,   # [浼樺寲] 鐩存帴璺敱鍒?knowledge
            "case_database": NodeName.CASE_DATABASE,  # [浼樺寲] 鐩存帴璺敱鍒?case_database
            "path_agent": NodeName.PATH_AGENT, # [浼樺寲] 鐩存帴璺敱鍒颁笅涓€涓梾鐞嗗垎鏋愭楠?
            "assessment": NodeName.ASSESSMENT, # 鍖呭惈娌荤枟鍐崇瓥浠诲姟锛岃繘鍏ヤ复搴婃祦绋?
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

