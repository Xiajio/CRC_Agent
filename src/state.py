import json
from typing import Annotated, Any, Dict, List, Optional, Union
import re
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# === 1. 强类型模型定义 ===


def _coerce_optional_int(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "n/a", "na", "unknown"}:
            return None
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        match = re.search(r"[+-]?\d+", text)
        if match:
            return int(match.group(0))
    return value

class PatientProfile(BaseModel):
    """
    患者画像（强类型，不可变）
    
    用于全链路数据一致性，确保 Type Safety。
    frozen=True 确保不可变性，强制通过更新操作而非直接修改。
    """
    tumor_type: str = "Unknown"
    pathology_confirmed: bool = False
    tnm_staging: Dict[str, Any] = Field(default_factory=dict)
    mmr_status: str = "Unknown"
    is_locked: bool = False
    chief_complaint: Optional[str] = None  # [新增] 主诉/患者问题
    age: Optional[int] = None  # [新增] 年龄
    gender: Optional[str] = None  # [新增] 性别
    ecog_score: Optional[Union[str, int]] = None # [新增] ECOG评分

    @field_validator("age", mode="before")
    @classmethod
    def _coerce_age(cls, value: Any) -> Any:
        return _coerce_optional_int(value)
    
    model_config = ConfigDict(frozen=True)


class SourceItem(BaseModel):
    """结构化引用源（用于 UI Source Cards）"""
    id: str
    page: Optional[int] = None
    content: str
    source: str


class RoadmapItem(BaseModel):
    """诊疗路线图单项"""
    step_id: str
    step_name: str
    status: str = "pending"  # pending | in_progress | completed | blocked
    details: str = ""
    icon: str = ""


class DecisionLogItem(BaseModel):
    """决策记录项"""
    turn: Optional[int] = None
    decision: str


class PendingQuestionItem(BaseModel):
    """待解决问题项"""
    question: str


class StructuredSummary(BaseModel):
    """
    [新增] 结构化摘要记忆
    用于替代纯文本 summary_memory，支持更精准的上下文管理
    
    v2.0 优化：
    - 新增 anchor_events 关键事件锚点
    - 新增 immutable_info 不变信息层
    - 新增 dynamic_info 动态信息层
    """
    profile_snapshot: Optional[PatientProfile] = Field(None, description="患者画像快照")
    
    # 记录对话过程中做出的关键决策
    decision_log: List[Union[str, DecisionLogItem, Dict[str, Any]]] = Field(default_factory=list, description="关键决策记录")
    
    # 记录当前尚未解决或需要后续跟进的问题
    pending_questions: List[Union[str, PendingQuestionItem, Dict[str, Any]]] = Field(default_factory=list, description="待解决的问题列表")
    
    # 记录最后一次更新摘要时的对话轮数
    last_update_turn: int = Field(0, description="最后更新时的对话轮数")
    
    # 保留自然语言摘要，用于兼容旧逻辑或提供给 LLM 阅读
    text_summary: str = Field("", description="自然语言摘要（兼容旧逻辑）")
    
    # [新增] 关键事件锚点 - 永不丢弃的重要事件
    anchor_events: List[Dict[str, Any]] = Field(default_factory=list, description="关键事件锚点列表")
    
    # [新增] 不变信息层 - 永不压缩的核心信息
    immutable_info: Dict[str, Any] = Field(default_factory=dict, description="不变信息：确诊时间、癌种、分期、过敏史等")
    
    # [新增] 动态信息层 - 需要定期更新的信息
    dynamic_info: Dict[str, Any] = Field(default_factory=dict, description="动态信息：当前治疗方案、最近关注点等")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_summary_payload(cls, data: Any):
        if data is None:
            return data
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        profile_snapshot = payload.get("profile_snapshot")
        if isinstance(profile_snapshot, str):
            text = profile_snapshot.strip()
            if not text:
                payload["profile_snapshot"] = None
            else:
                try:
                    payload["profile_snapshot"] = json.loads(text)
                except Exception:
                    payload["profile_snapshot"] = None
        return payload


class PlanStep(BaseModel):
    """
    原子计划步骤 - 用于自主规划 (DAG 增强版)
    
    Agent 将复杂任务拆解为一系列可执行的原子步骤，
    每个步骤包含描述、所需工具、状态和推理过程。
    
    增强字段（对齐 MiroMindOS 的 DAG 协议）：
    - parent_step_id: 父步骤ID，表达依赖关系
    - branch_id: 分支ID，支持并行分支和备选方案
    - state_hash: 状态哈希，支持可回滚
    """
    id: str = Field(description="步骤唯一标识，如 'step_1'")
    description: str = Field(description="步骤描述，例如：'查阅NCCN直肠癌指南关于T3期的治疗建议'")
    tool_needed: str = Field(
        description=(
            "预计使用的工具类型。只能使用以下类型之一："
            "list_guideline_toc, toc, read_guideline_chapter, read, chapter, "
            "search_treatment_recommendations, search, database_query, "
            "case_database_query, web_search, web, ask_user, calculator, "
            "imaging_analysis, tumor_detection, radiology, tumor_screening, ct_analysis, "
            "pathology_analysis, pathology, clam"
        )
    )
    status: str = Field(default="pending", description="pending | in_progress | completed | failed")
    reasoning: str = Field(default="", description="为什么需要这一步的推理过程")
    error_message: str = Field(default="", description="如果执行失败，记录错误信息")
    retry_count: int = Field(default=0, description="重试次数，防止无限循环")
    assignee: Optional[str] = Field(
        default=None,
        description="并行子任务指派目标（如 knowledge/web_search/case_database/rad_agent/path_agent）"
    )
    parallel_group: Optional[str] = Field(
        default=None,
        description="并行执行分组标识（同组步骤可并发执行）"
    )
    
    # === DAG 增强字段（对齐 MiroMindOS）===
    parent_step_id: Optional[str] = Field(
        default=None,
        description="父步骤ID，用来表达依赖关系"
    )
    branch_id: Optional[str] = Field(
        default=None,
        description="分支ID（同一备选方案/子路径）"
    )
    state_hash: Optional[str] = Field(
        default=None,
        description="此步执行前/后的状态哈希，用于回滚"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新时间")
    
    @classmethod
    def get_valid_tool_types(cls) -> set:
        """返回所有有效的工具类型"""
        return {
            "list_guideline_toc", "toc",
            "read_guideline_chapter", "read", "chapter",
            "search_treatment_recommendations", "search",
            "database_query", "case_database_query",
            "web_search", "web",
            "ask_user", "calculator",
            # 影像分析工具
            "imaging_analysis", "tumor_detection", "radiology",
            "tumor_screening", "ct_analysis",
            "pathology_analysis", "pathology", "clam"
        }
    
    def is_valid_tool(self) -> bool:
        """检查工具类型是否有效"""
        tool = self.tool_needed.lower()
        valid_tools = self.get_valid_tool_types()
        
        # 完全匹配
        if tool in valid_tools:
            return True
        
        # 部分匹配（用于处理组合工具名）
        for valid_tool in valid_tools:
            if valid_tool in tool:
                return True
        
        return False


# === 证据链追溯模型（对齐 MiroMindOS）===

class Claim(BaseModel):
    """决策结论结构"""
    claim_id: str = Field(description="结论的稳定ID")
    text: str = Field(description="结论内容")
    importance: str = Field(default="MEDIUM", description="HIGH / MEDIUM / LOW")
    claim_type: str = Field(default="fact", description="fact / recommendation / risk / assumption")


class EvidenceLink(BaseModel):
    """证据链接结构 - 连接结论与来源"""
    claim_id: str = Field(description="对应的结论ID")
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="与 retrieved_references 里的 source_id 对应"
    )
    confidence: float = Field(default=1.0, description="置信度 0-1")
    reasoning_chain: List[str] = Field(
        default_factory=list,
        description="推理要点（由模型生成）"
    )


class DecisionWithEvidence(BaseModel):
    """带证据的决策输出结构"""
    claims: List[Claim] = Field(default_factory=list, description="结论列表")
    evidence_links: List[EvidenceLink] = Field(default_factory=list, description="证据链接列表")
    overall_confidence: str = Field(default="MEDIUM", description="HIGH / MEDIUM / LOW")
    human_review_required: bool = Field(default=False, description="是否需要人工审核")


class RetrievedReference(BaseModel):
    """增强的引用来源结构（带 ID 的一等公民）"""
    source_id: str = Field(description="来源唯一ID")
    title: str = Field(default="", description="来源标题")
    url: str = Field(default="", description="来源URL")
    page: Optional[int] = Field(default=None, description="页码")
    evidence_id: Optional[str] = None
    section: Optional[str] = None
    snippet: str = Field(description="内容片段")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow, description="检索时间")
    relevance: float = Field(default=1.0, description="相关性评分")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_reference(cls, data: Any):
        """
        兼容历史引用结构：
        - source_id <- source_id | ref_id | source:page
        - snippet   <- snippet | preview | content
        - title     <- title | source
        - relevance <- relevance | score
        """
        if data is None:
            return data
        if isinstance(data, cls):
            return data
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        if not isinstance(data, dict):
            return data

        d = dict(data)

        source = str(d.get("source") or "").strip()
        page_val = d.get("page")
        page: Optional[int] = None
        if isinstance(page_val, int):
            page = page_val
        elif isinstance(page_val, str):
            m = re.search(r"\d+", page_val)
            if m:
                try:
                    page = int(m.group())
                except Exception:
                    page = None

        source_id = d.get("source_id") or d.get("ref_id")
        if not source_id:
            if source and page is not None:
                source_id = f"{source}:{page}"
            elif source:
                source_id = source
            else:
                source_id = "unknown"

        snippet = d.get("snippet") or d.get("preview") or d.get("content") or ""
        if not isinstance(snippet, str):
            snippet = str(snippet)
        if not snippet.strip():
            if source and page is not None:
                snippet = f"Source={source}, Page={page}"
            elif source:
                snippet = f"Source={source}"
            else:
                snippet = "Reference snippet unavailable."

        title = d.get("title") or source or ""
        url = d.get("url") or ""

        relevance = d.get("relevance")
        if relevance is None:
            relevance = d.get("score", 1.0)
        try:
            relevance = float(relevance)
        except Exception:
            relevance = 1.0

        normalized = {
            "source_id": str(source_id),
            "title": str(title),
            "url": str(url),
            "page": page,
            "evidence_id": d.get("evidence_id"),
            "section": d.get("section"),
            "snippet": snippet,
            "relevance": relevance,
        }
        if d.get("retrieved_at") is not None:
            normalized["retrieved_at"] = d.get("retrieved_at")
        return normalized


# === 2. Reducers ===

def merge_dicts(left: Dict[str, Any] | None, right: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    浅合并 Reducer。

    用于 findings 等动态字典字段，支持增量更新。

    ⚠️ 注意事项：此实现为浅合并（Shallow Merge）。
    - 如果 findings 中存在嵌套字典（如 findings["tnm"] = {"T": "cT3", "N": "cN1"}），
      更新时若只传递部分字段（如 {"tnm": {"M": "cM0"}}），会覆盖掉原来的整个嵌套字典。
    - 业务逻辑保证每次更新都是全量字段或不涉及深层嵌套更新时方可安全使用。
    - 如需深合并，请使用第三方库如 deepmerge 或自定义递归实现。
    """
    merged: Dict[str, Any] = {}
    if left:
        merged.update(left)
    if right:
        merged.update(right)
    return merged


def replace_list(left: List[str] | None, right: List[str] | None) -> List[str]:
    """
    列表替换 Reducer。
    
    用于 missing_critical_data：每次 Assessment 重新评估当前缺失项，
    不应累积历史缺失项。
    """
    if right is not None:
        return right
    return left or []


def append_list(left: List[Dict[str, Any]] | None, right: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """
    列表追加 Reducer。
    
    用于时间轴/审计日志，保留历史追加新条目。
    """
    if left is None:
        left = []
    if right:
        return left + right
    return left


def merge_evidence_by_id(
    left: List[Dict[str, Any]] | None,
    right: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    anonymous = 0
    for item in (left or []) + (right or []):
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id:
            anonymous += 1
            evidence_id = f"anon:{anonymous}"
            item = {**item, "evidence_id": evidence_id}
        merged[evidence_id] = item
    return list(merged.values())


def update_profile(left: Optional[PatientProfile], right: Optional[PatientProfile]) -> Optional[PatientProfile]:
    """
    [核心 Reducer] Profile 更新逻辑
    
    - Profile 是不可变的 (frozen)，通过新对象替换旧对象。
    - 只有当 right 非空时才替换，确保原子性更新。
    """
    if right is not None:
        return right
    return left


def update_roadmap(left: List[Dict[str, Any]] | None, right: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """
    Roadmap 智能合并 Reducer。
    
    - 按 step_id 更新现有步骤，追加新步骤。
    - 保持顺序（左边优先，新增放后面）。
    """
    if not left:
        return right or []
    if not right:
        return left
    
    # 左边优先构建索引
    left_map: Dict[str, Dict[str, Any]] = {item["step_id"]: item for item in left}
    
    # 右边覆盖或新增
    for item in right:
        left_map[item["step_id"]] = item
    
    # 按 step_id 排序返回（保持左边顺序）
    return list(left_map.values())


# === 3. State Definition ===

class CRCAgentState(BaseModel):
    """
    CRC Agent State (v3.0 - Strong Typing)
    
    核心特点：
    - 强类型 patient_profile 确保全链路数据一致性
    - 支持缺失数据追问循环
    - 支持 Decision-Critic 自我纠错回路
    - 诊疗路线图可视化
    """

    # --- 1. Conversation & Context ---
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    
    # [核心修改] 强类型 Profile（替代 Dict[str, Any]）
    # 使用 update_profile reducer 确保原子性更新
    patient_profile: Annotated[Optional[PatientProfile], update_profile] = None
    
    # findings 用于存储非结构化的临时发现（如 "用户提到腹痛"）
    # 兼容旧代码：某些字段可能仍从 findings 读取
    findings: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    
    clinical_stage: str = "Assessment"
    error: Optional[str] = None

    # --- 2. Structured Findings (Core Memory) ---
    
    # 缺失数据清单（Router 使用，支持追问循环）
    # 使用 replace_list 而非 append，因为每次 Assessment 重新评估当前缺失项
    missing_critical_data: Annotated[List[str], replace_list] = Field(default_factory=list)

    # --- 3. Artifacts (Outputs) ---
    assessment_draft: Optional[str] = None
    decision_json: Optional[Dict[str, Any]] = None
    medical_card: Optional[Dict[str, Any]] = None
    final_output: Optional[str] = None
    
    retrieved_evidence: Annotated[List[Dict[str, Any]], merge_evidence_by_id] = Field(default_factory=list)
    retrieved_references: List[RetrievedReference] = Field(default_factory=list)
    subagent_reports: Annotated[List[Dict[str, Any]], append_list] = Field(default_factory=list)
    node_timings: Annotated[List[Dict[str, Any]], append_list] = Field(default_factory=list)
    stage_timings: Annotated[List[Dict[str, Any]], append_list] = Field(default_factory=list)
    retrieval_timings: Annotated[List[Dict[str, Any]], append_list] = Field(default_factory=list)
    rag_trace: Annotated[List[Dict[str, Any]], append_list] = Field(default_factory=list)
    
    # [增强] 结构化引用源（使用 RetrievedReference 类型，支持 source_id 追溯）
    # 并行子智能体汇总报告
    
    # [新增] 带证据的决策结构（对齐 MiroMindOS）
    decision_with_evidence: Optional[DecisionWithEvidence] = Field(
        default=None,
        description="带证据链的决策输出结构"
    )

    # --- 4. Logic Gates (Flags) ---
    pathology_confirmed: bool = False
    histology_type: Optional[str] = None
    distant_metastasis_check: Optional[str] = None
    
    # Critic（审查者）状态字段，支持自我纠错回路
    critic_verdict: Optional[str] = None  # "APPROVED" | "REJECTED" | None
    critic_feedback: Optional[str] = None  # 具体修改建议
    critic_review_signal: Optional[Dict[str, Any]] = None

    # 迭代控制（防止 Decision-Critic 无限循环）
    iteration_count: int = 0
    rejection_count: int = 0
    feedback_history: List[str] = Field(default_factory=list)
    iteration_limit_reached: bool = False
    requires_human_review: bool = False
    # 引用覆盖与评估结果
    # Stable structured reports consumed by the policy normalization layer.
    citation_report: Optional[Dict[str, Any]] = None
    evaluation_report: Optional[Dict[str, Any]] = None
    evaluator_review_signal: Optional[Dict[str, Any]] = None
    evaluation_retry_count: int = 0
    
    # [新增] Safety Guard 拦截记录
    safety_violation: Optional[str] = None  # 记录被 SafetyGuard 拦截的原因

    # --- 5. Clinical Roadmap (诊疗路线图) ---
    # 显式展示诊疗进度，缓解患者焦虑
    # 注意：虽然上面定义了 RoadmapItem 模型，但这里使用 Dict[str, Any] 以便与 update_roadmap reducer 兼容
    # 实际使用时，List 中的每个 dict 应为 RoadmapItem.model_dump() 的结果
    roadmap: Annotated[List[Dict[str, Any]], update_roadmap] = Field(default_factory=list)

    # --- 6. Current Active Patient Context ---
    # 用于追踪当前对话中关注的患者ID，让 Router 知道"这个患者"指的是谁
    # 使用 str 类型以保留 "093" 这种前导零格式，与 Router 代码和 API 调用中的字符串格式保持一致
    current_patient_id: Optional[str] = None

    # --- 6.1 Encounter Track / Triage Context ---
    # 用于区分门诊分诊轨道与 CRC 临床追问轨道，避免把症状问诊和病例补全混在一起
    encounter_track: Optional[str] = None
    clinical_entry_reason: Optional[str] = None
    entry_explanation_shown: bool = False
    known_crc_signals: Dict[str, Any] = Field(default_factory=dict)
    triage_risk_level: Optional[str] = None
    triage_disposition: Optional[str] = None
    triage_suggested_tests: List[str] = Field(default_factory=list)
    triage_summary: Optional[str] = None
    symptom_snapshot: Dict[str, Any] = Field(default_factory=dict)

    # --- 7. 自主规划上下文 (Autonomous Planning Context) ===
    # [新增] 动态计划列表：Agent 的待办事项，支持主动上下文构建
    current_plan: List[PlanStep] = Field(
        default_factory=list,
        description="当前执行的计划步骤列表，Agent 通过此字段实现自主规划"
    )
    
    # [新增] DAG 执行图（对齐 MiroMindOS）
    # step_id -> 节点信息（含子节点ID列表、状态哈希等）
    execution_graph: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="DAG 执行图：step_id -> 节点信息"
    )
    
    # [新增] 步骤执行历史（时间序列记录）
    step_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="时间序列的执行记录，用于回溯和调试"
    )
    
    # [新增] 推理草稿纸：用于上下文蒸馏和思考过程记录
    scratchpad: str = Field(
        default="",
        description="Agent 的思考草稿本，记录中间推理过程，用于上下文压缩"
    )
    
    # [新增] 规划迭代计数：防止自我纠错无限循环
    plan_iteration_count: int = Field(
        default=0,
        description="规划迭代次数，用于熔断保护"
    )
    
    # --- 8. Long-term Memory / Audit ---
    # 摘要记忆（长期上下文）
    summary_memory: Optional[str] = None
    # [新增] 结构化摘要对象
    structured_summary: Optional[StructuredSummary] = None
    
    # 摘要记忆覆盖到的 messages 下标（用于增量总结）
    summary_memory_cursor: int = 0
    # 患者档案变更时间轴（审计日志）
    patient_profile_timeline: Annotated[List[Dict[str, Any]], append_list] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def ensure_agent_state(result: "CRCAgentState | dict") -> "CRCAgentState":
    """
    规范化 LangGraph 输出为 CRCAgentState。
    
    处理 LangGraph 返回的字典格式，确保类型安全。
    """
    if isinstance(result, CRCAgentState):
        return result
    return CRCAgentState.model_validate(result)


# === DAG 辅助函数 ===

def update_step_status(
    state: "CRCAgentState",
    step_id: str,
    new_status: str,
    meta: Dict[str, Any] = None
) -> "CRCAgentState":
    """
    更新步骤状态并维护 DAG 执行图和历史记录
    
    Args:
        state: 当前状态
        step_id: 步骤ID
        new_status: 新状态 (pending/in_progress/completed/failed)
        meta: 额外的元数据
    
    Returns:
        更新后的状态副本
    """
    from datetime import datetime

    plan = state.current_plan or []
    step = next((s for s in plan if s.id == step_id), None)
    
    if not step:
        return state

    updated_at = datetime.utcnow()
    updated_step = step.model_copy(update={
        "status": new_status,
        "updated_at": updated_at,
    })
    new_plan = [updated_step if s.id == step_id else s for s in plan]
    
    new_graph = dict(state.execution_graph or {})
    node = new_graph.setdefault(step_id, {
        "id": step_id,
        "children": [],
        "parent": step.parent_step_id,
        "branch_id": step.branch_id,
    })
    node["status"] = new_status
    node["meta"] = meta or {}
    
    new_history = list(state.step_history or [])
    new_history.append({
        "step_id": step_id,
        "status": new_status,
        "timestamp": updated_at.isoformat(),
        "meta": meta or {},
    })
    
    return state.model_copy(update={
        "current_plan": new_plan,
        "execution_graph": new_graph,
        "step_history": new_history,
    })


def export_execution_dag(state: "CRCAgentState") -> Dict[str, List[Dict]]:
    """
    导出执行 DAG 用于可视化
    
    Returns:
        {"nodes": [...], "edges": [...]} 格式，可用于前端渲染
    """
    nodes = []
    edges = []
    
    for step in (state.current_plan or []):
        nodes.append({
            "id": step.id,
            "label": step.description[:50] + "..." if len(step.description) > 50 else step.description,
            "status": step.status,
            "branch_id": step.branch_id,
            "parent_step_id": step.parent_step_id,
            "tool_needed": step.tool_needed,
        })
        if step.parent_step_id:
            edges.append({
                "from": step.parent_step_id,
                "to": step.id
            })
    
    return {"nodes": nodes, "edges": edges}


def find_step_by_id(state: "CRCAgentState", step_id: str) -> Optional[PlanStep]:
    """根据 ID 查找计划步骤"""
    plan = state.current_plan or []
    return next((s for s in plan if s.id == step_id), None)


def get_branch_steps(state: "CRCAgentState", branch_id: str) -> List[PlanStep]:
    """获取指定分支的所有步骤"""
    plan = state.current_plan or []
    return [s for s in plan if s.branch_id == branch_id]


def get_failed_branches(state: "CRCAgentState") -> List[str]:
    """获取所有失败分支的 branch_id"""
    plan = state.current_plan or []
    failed_branches = set()
    for step in plan:
        if step.status == "failed" and step.branch_id:
            failed_branches.add(step.branch_id)
    return list(failed_branches)


# === 证据链追溯函数 ===

def trace_evidence_chain(
    state: "CRCAgentState",
    claim_id: str
) -> Dict[str, Any]:
    """
    追溯单个结论的完整证据链
    
    Args:
        state: 当前状态
        claim_id: 要追溯的结论ID
    
    Returns:
        包含结论、置信度、推理链和证据来源的字典
    """
    decision = state.decision_with_evidence
    
    if not decision:
        return {"claim_id": claim_id, "error": "No decision found"}
    
    link = next(
        (l for l in decision.evidence_links if l.claim_id == claim_id),
        None
    )
    if not link:
        return {"claim_id": claim_id, "error": "No evidence link found"}
    
    ref_index = {
        ref.source_id: ref
        for ref in (state.retrieved_references or [])
    }
    
    evidence = []
    for source_id in link.evidence_sources:
        ref = ref_index.get(source_id)
        if not ref:
            continue
        evidence.append({
            "source_id": ref.source_id,
            "title": ref.title,
            "url": ref.url,
            "page": ref.page,
            "snippet": ref.snippet,
            "retrieved_at": ref.retrieved_at.isoformat() if ref.retrieved_at else None,
            "relevance": ref.relevance,
        })
    
    claim = next(
        (c for c in decision.claims if c.claim_id == claim_id),
        None
    )
    
    return {
        "claim": {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "importance": claim.importance,
            "claim_type": claim.claim_type,
        } if claim else {"claim_id": claim_id},
        "confidence": link.confidence,
        "reasoning_chain": link.reasoning_chain,
        "evidence": evidence,
    }


def get_all_claims_with_evidence(state: "CRCAgentState") -> List[Dict[str, Any]]:
    """获取所有带证据的结论"""
    decision = state.decision_with_evidence
    
    if not decision:
        return []
    
    return [
        trace_evidence_chain(state, claim.claim_id)
        for claim in decision.claims
    ]


def check_claim_coverage(state: "CRCAgentState") -> Dict[str, Any]:
    """
    检查所有结论的证据覆盖情况
    
    Returns:
        包含覆盖率统计和建议的字典
    """
    decision = state.decision_with_evidence
    
    if not decision:
        return {
            "total_claims": 0,
            "covered_claims": 0,
            "coverage_rate": 0.0,
            "uncovered_claims": [],
            "high_importance_uncovered": [],
        }
    
    uncovered = []
    high_importance_uncovered = []
    
    for claim in decision.claims:
        link = next(
            (l for l in decision.evidence_links if l.claim_id == claim.claim_id),
            None
        )
        if not link or not link.evidence_sources:
            uncovered.append(claim.claim_id)
            if claim.importance == "HIGH":
                high_importance_uncovered.append(claim.claim_id)
    
    total = len(decision.claims)
    covered = total - len(uncovered)
    coverage_rate = covered / total if total > 0 else 0.0
    
    return {
        "total_claims": total,
        "covered_claims": covered,
        "coverage_rate": coverage_rate,
        "uncovered_claims": uncovered,
        "high_importance_uncovered": high_importance_uncovered,
        "needs_human_review": (
            coverage_rate < 0.8 or
            len(high_importance_uncovered) > 0
        ),
    }
