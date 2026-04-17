"""
Memory Management Node v2.0
增强版上下文记忆管理策略

核心优化：
1. Token 预算替代条数限制
2. 语义重要性分类替代硬截断
3. 增量摘要（每轮检测关键信息变化）
4. 关键事件锚点机制
5. 摘要分层（不变信息 vs 动态信息）
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import re
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ..state import CRCAgentState, StructuredSummary, PatientProfile
from .node_utils import _format_messages_for_summary, _invoke_structured_with_recovery

# ==============================================================================
# 1. Configuration
# ==============================================================================

TOKEN_BUDGET_MAX = 6000
TOKEN_ENCODING_NAME = "cl100k_base"

GC_MAX_MESSAGES = 20
SUMMARY_TRIGGER_THRESHOLD = 6

ANCHOR_EVENT_PATTERNS = [
    (r"分期[变改]|T[0-4]|N[0-3]|M[01]|Stage\s*[IVX]", "分期变化"),
    (r"治疗方案|化疗方案|手术|放疗|靶向|免疫", "治疗方案调整"),
    (r"拒绝|不同意的?|不接受的?", "患者拒绝治疗"),
    (r"副作用|不良反应|并发症|过敏|皮疹|恶心", "不良事件"),
    (r"家属|家人|配偶|子女|父母", "家属意见"),
    (r"确诊|诊断|病理|活检", "确诊信息"),
    (r"基因突变|MSS|MSI|dMMR|pMMR|KRAS|NRAS|BRAF", "基因状态"),
]

SEMANTIC_RETENTION_RULES = {
    "检查结果": {"priority": "high", "max_truncate": None, "description": "完整保留检查结果"},
    "诊断信息": {"priority": "critical", "max_truncate": None, "description": "完整保留诊断信息"},
    "分期信息": {"priority": "critical", "max_truncate": None, "description": "完整保留分期信息"},
    "治疗方案": {"priority": "high", "max_truncate": None, "description": "完整保留治疗方案"},
    "基因检测": {"priority": "critical", "max_truncate": None, "description": "完整保留基因检测结果"},
    "过敏史": {"priority": "critical", "max_truncate": None, "description": "完整保留过敏史"},
    "基础病": {"priority": "high", "max_truncate": None, "description": "完整保留基础疾病"},
    "闲聊/寒暄": {"priority": "low", "max_truncate": 50, "description": "可截断或丢弃"},
    "重复确认": {"priority": "low", "max_truncate": 0, "description": "可丢弃"},
    "一般问答": {"priority": "medium", "max_truncate": 200, "description": "保留关键信息"},
}

IMMEDIATE_SUMMARY_KEYWORDS = [
    "分期变了", "分期变化", "分期调整", "改成", "改为", "拒绝", "不做了", "不做了",
    "副作用", "不良反应", "并发症", "过敏了", "不耐受", "耐药",
    "基因检测", "基因突变", "病理结果", "确诊了", "确认了",
    "手术", "化疗", "放疗", "靶向", "免疫", "治疗方案",
    "cea", "ct", "mri", "检查结果", "复查",
    "转移", "复发", "进展", "恶化",
    "家属", "家人", "配偶", "子女", "父母", "同意",
    "放弃", "终止", "更改", "调整",
]

# ==============================================================================
# 2. Token Counting Utilities
# ==============================================================================

def get_token_encoder():
    """获取 token 编码器（延迟加载）"""
    try:
        import tiktoken
        return tiktoken.get_encoding(TOKEN_ENCODING_NAME)
    except ImportError:
        return None

def estimate_token_count(text: str) -> int:
    """估算文本的 token 数量"""
    if not text:
        return 0
    encoder = get_token_encoder()
    if encoder:
        return len(encoder.encode(text))
    return len(text) // 4

def count_messages_tokens(messages: List[BaseMessage]) -> int:
    """计算消息列表的总 token 数"""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        total += estimate_token_count(content)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                total += estimate_token_count(json.dumps(tc))
    return total

# ==============================================================================
# 3. Semantic Importance Classification
# ==============================================================================

def classify_message_importance(msg: BaseMessage) -> str:
    """
    根据消息内容分类语义重要性
    
    Returns:
        'critical': 关键信息，绝不能截断
        'high': 重要信息，优先保留
        'medium': 一般信息，可适当截断
        'low': 低价值信息，可丢弃或大幅截断
    """
    if isinstance(msg, ToolMessage):
        return "high"
    
    content = getattr(msg, "content", "") or ""
    content_lower = content.lower()
    
    critical_patterns = ["确诊", "分期", "t4", "t3", "t2", "t1", "n1", "n2", "m1", 
                         "基因", "kras", "braf", "msi", "dmmr", "pmmr", "过敏", 
                         "拒绝", "不同意", "不做了", "不耐受"]
    high_patterns = ["治疗", "化疗", "手术", "放疗", "靶向", "免疫", "方案", 
                     "副作用", "检查", "结果", "cea", "ct", "mri", "肠镜"]
    low_patterns = ["你好", "谢谢", "在吗", "hi", "hello", "好的", "明白", "嗯"]
    
    for pattern in critical_patterns:
        if pattern in content_lower:
            return "critical"
    
    for pattern in high_patterns:
        if pattern in content_lower:
            return "high"
    
    for pattern in low_patterns:
        if pattern in content_lower:
            return "low"
    
    return "medium"

def truncate_by_importance(content: str, importance: str) -> str:
    """根据重要性级别决定截断策略"""
    if importance == "critical":
        return content
    if importance == "high":
        return content[:500] + "..." if len(content) > 500 else content
    if importance == "medium":
        return content[:300] + "..." if len(content) > 300 else content
    return content[:100] + "..." if len(content) > 100 else content

# ==============================================================================
# 4. Event Anchoring
# ==============================================================================

def detect_key_events(content: str) -> List[Dict[str, Any]]:
    """检测关键事件并返回锚点"""
    events = []
    content_lower = content.lower()
    
    for pattern, event_type in ANCHOR_EVENT_PATTERNS:
        if re.search(pattern, content_lower):
            events.append({
                "type": event_type,
                "content": content[:200],
                "detected_at": "message"
            })
            break
    
    return events

def should_trigger_immediate_summary(content: str) -> bool:
    """判断是否需要立即触发摘要更新"""
    content_lower = content.lower()
    for keyword in IMMEDIATE_SUMMARY_KEYWORDS:
        if keyword in content_lower:
            return True
    return False

# ==============================================================================
# 5. Token-Based Context Compression
# ==============================================================================

def compress_context_by_token(
    messages: List[BaseMessage], 
    max_tokens: int = TOKEN_BUDGET_MAX,
    preserve_system: bool = True,
    preserve_tool_calls: bool = True
) -> List[BaseMessage]:
    """
    基于 Token 预算的上下文压缩
    
    优先级：
    1. SystemMessage - 始终保留
    2. 关键事件锚点相关消息
    3. 工具调用链（AIMessage + ToolMessages）
    4. 近期对话（按语义重要性）
    5. 摘要记忆
    
    Args:
        messages: 消息列表
        max_tokens: 最大 token 预算
        preserve_system: 是否保留系统消息
        preserve_tool_calls: 是否保留工具调用链
    
    Returns:
        压缩后的消息列表
    """
    if not messages:
        return []
    
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
    
    current_tokens = count_messages_tokens(system_messages) if preserve_system else 0
    
    if current_tokens >= max_tokens:
        return system_messages[:1] if preserve_system else []
    
    available_tokens = max_tokens - current_tokens
    
    kept_messages = []
    
    if preserve_tool_calls:
        tool_blocks, remaining_msgs = extract_tool_call_blocks(non_system_messages)
        for block in tool_blocks:
            block_tokens = count_messages_tokens(block)
            if current_tokens + block_tokens <= max_tokens:
                kept_messages.extend(block)
                current_tokens += block_tokens
                available_tokens = max_tokens - current_tokens
            else:
                break
    else:
        remaining_msgs = non_system_messages
    
    remaining_msgs.reverse()
    for msg in remaining_msgs:
        importance = classify_message_importance(msg)
        content = getattr(msg, "content", "") or ""
        
        truncated_content = truncate_by_importance(content, importance)
        msg_tokens = estimate_token_count(truncated_content)
        
        if importance == "critical":
            kept_messages.append(msg)
            current_tokens += msg_tokens
        elif current_tokens + msg_tokens <= max_tokens:
            kept_messages.append(msg)
            current_tokens += msg_tokens
        elif importance == "high" and available_tokens > 100:
            small_trunc = truncate_by_importance(content, "medium")
            if current_tokens + estimate_token_count(small_trunc) <= max_tokens:
                kept_messages.append(msg)
                current_tokens += estimate_token_count(small_trunc)
    
    kept_messages.reverse()
    
    if preserve_system:
        return system_messages + kept_messages
    return kept_messages

def extract_tool_call_blocks(messages: List[BaseMessage]) -> Tuple[List[List[BaseMessage]], List[BaseMessage]]:
    """提取工具调用块和非工具消息"""
    tool_blocks = []
    non_tool_msgs = []
    
    i = 0
    while i < len(messages):
        msg = messages[i]
        
        if isinstance(msg, ToolMessage):
            block = [msg]
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                block.append(messages[j])
                j += 1
            
            if i > 0 and isinstance(messages[i-1], AIMessage) and messages[i-1].tool_calls:
                block.insert(0, messages[i-1])
                i = i - 1
            
            tool_blocks.append(block)
            i = j
        else:
            non_tool_msgs.append(msg)
            i += 1
    
    return tool_blocks, non_tool_msgs


def _merge_anchor_events(
    existing_events: Optional[List[Dict[str, Any]]],
    new_events: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merge anchor events while preserving order and avoiding duplicates."""
    merged: List[Dict[str, Any]] = []
    for event in list(existing_events or []) + list(new_events or []):
        if event not in merged:
            merged.append(event)
    return merged


def _rebase_summary_cursor_after_gc(
    messages: List[BaseMessage],
    cursor: int,
    removed_ids: set[str],
) -> int:
    """Rebase the summary cursor after actual message removals."""
    if cursor <= 0 or not removed_ids:
        return max(0, cursor)

    removed_before_cursor = sum(
        1
        for msg in messages[:cursor]
        if getattr(msg, "id", None) in removed_ids
    )
    return max(0, cursor - removed_before_cursor)

# ==============================================================================
# 6. Incremental Summary
# ==============================================================================

INCREMENTAL_SUMMARY_PROMPT = """你是一个医疗对话增量摘要器。

任务：判断新增对话是否包含关键信息变化，如果有，则更新对应的摘要字段。

关键变化类型：
1. 分期变化 - 如 "分期从II期变成III期"、"cT3N1" 等
2. 治疗方案调整 - 如 "改用XELOX方案"、"不手术了" 等
3. 患者意愿变化 - 如 "拒绝化疗"、"想手术" 等
4. 不良事件 - 如 "出现皮疹"、"不能耐受" 等
5. 检查结果更新 - 如 "CEA升高"、"CT显示" 等
6. 基因状态 - 如 "KRAS突变"、"MSI-H" 等

现有摘要：
{existing_summary}

新增对话：
{new_dialogue}

请输出 JSON 格式的增量更新（如果无关键变化，输出空对象 {{}}）：
{{
    "field_changed": "dynamic_info" 或 "immutable_info" 或 "anchor_events",
    "field_name": "治疗方案" / "分期" / "患者意愿" 等,
    "old_value": "旧值",
    "new_value": "新值",
    "reason": "变化原因"
}}

如果无关键变化，直接输出 {{}}。
"""

def incremental_summary(
    existing_summary: StructuredSummary,
    new_dialogue: str,
    model
) -> Optional[Dict[str, Any]]:
    """增量摘要 - 判断是否有关键变化并更新"""
    
    if not new_dialogue or not should_trigger_immediate_summary(new_dialogue):
        return None
    
    try:
        prompt = ChatPromptTemplate.from_template(INCREMENTAL_SUMMARY_PROMPT)
        chain = prompt | model.bind(temperature=0)
        
        existing_text = existing_summary.text_summary or ""
        result = chain.invoke({
            "existing_summary": existing_text,
            "new_dialogue": new_dialogue
        })
        
        content = getattr(result, "content", str(result)).strip()
        if content == "{}" or not content:
            return None
        
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError:
            return None
            
    except Exception as e:
        print(f"⚠️ [Incremental Summary] Error: {e}")
        return None

# ==============================================================================
# 7. Structured Summary with Layers
# ==============================================================================

LAYERED_SUMMARY_PROMPT = """你是一个专业的医疗对话记忆管理器。

请将新的对话内容整合到现有的【分层结构化摘要】中。

【分层结构说明】
1. 不变信息层 (immutable_info)：确诊时间、癌种、分期、过敏史、基础病等核心信息（永不压缩）
2. 动态信息层 (dynamic_info)：当前治疗方案、最近检查结果、患者情绪状态等（定期更新）
3. 关键事件锚点 (anchor_events)：分期变化、治疗方案调整、患者拒绝等重大事件（永不丢弃）
4. 文本摘要 (text_summary)：简洁的自然语言摘要

现有摘要：
{existing_summary}

新增对话：
{new_dialogue}

请输出更新后的分层结构化摘要（JSON格式）。
"""

def update_layered_summary(
    existing_summary: StructuredSummary,
    new_dialogue: str,
    model
) -> StructuredSummary:
    """更新分层结构化摘要"""
    
    prompt = ChatPromptTemplate.from_template(LAYERED_SUMMARY_PROMPT)
    existing_json = (
        existing_summary.model_dump_json()
        if hasattr(existing_summary, "model_dump_json")
        else json.dumps(existing_summary, ensure_ascii=False)
    )
    try:
        result = _invoke_structured_with_recovery(
            prompt=prompt,
            model=model,
            schema=StructuredSummary,
            payload={
                "existing_summary": existing_json,
                "new_dialogue": new_dialogue
            },
            log_prefix="[Layered Summary]",
            fallback_factory=lambda _payload, _err: existing_summary,
        )
        if result and hasattr(result, "text_summary"):
            return result
    except Exception as e:
        print(f"⚠️ [Layered Summary] 更新失败，保留现有摘要: {e}")
    return existing_summary

# ==============================================================================
# 8. Node Implementation
# ==============================================================================

def node_memory_manager(model, streaming: bool = False, show_thinking: bool = True) -> Runnable:
    """
    增强版内存管理节点 v2.0
    
    核心功能：
    1. Token 预算压缩
    2. 语义重要性分类
    3. 增量摘要（每轮检测关键变化）
    4. 关键事件锚点
    5. 分层摘要
    """
    
    def _run(state: CRCAgentState):
        updates = {}
        messages = state.messages
        current_len = len(messages)
        removed_ids: set[str] = set()
        
        total_tokens = count_messages_tokens(messages)
        
        if total_tokens > TOKEN_BUDGET_MAX:
            if show_thinking:
                print(f"🧹 [Memory V2] Token budget exceeded ({total_tokens} > {TOKEN_BUDGET_MAX}), compressing...")
            
            compressed = compress_context_by_token(
                messages, 
                max_tokens=TOKEN_BUDGET_MAX,
                preserve_system=True,
                preserve_tool_calls=True
            )
            
            to_remove = []
            kept_ids = {msg.id for msg in compressed if msg.id}
            for msg in messages:
                if msg.id and msg.id not in kept_ids:
                    to_remove.append(RemoveMessage(id=msg.id))
            
            if to_remove:
                if show_thinking:
                    print(f"🧹 [Memory V2] Pruning {len(to_remove)} messages...")
                updates["messages"] = to_remove
                removed_ids = {msg.id for msg in to_remove if msg.id}
        
        structured_summary = state.structured_summary
        if not structured_summary:
            structured_summary = StructuredSummary()
        
        cursor = max(0, state.summary_memory_cursor or 0)
        rebased_cursor = _rebase_summary_cursor_after_gc(messages, cursor, removed_ids)
        new_messages = messages[cursor:] if cursor < len(messages) else []
        
        if new_messages:
            new_dialogue = _format_messages_for_summary(new_messages, max_chars=1000)
            detected_events = []
            
            for msg in new_messages:
                content = getattr(msg, "content", "") or ""
                events = detect_key_events(content)
                if not events:
                    continue
                detected_events.extend(events)
                if show_thinking:
                    print(f"🎯 [Memory V2] Detected key events: {[e.get('type') for e in events]}")
            if detected_events:
                structured_summary.anchor_events = _merge_anchor_events(
                    structured_summary.anchor_events,
                    detected_events,
                )
            
            incremental_update = incremental_summary(
                structured_summary,
                new_dialogue,
                model
            )
            
            if incremental_update and show_thinking:
                print(f"🧠 [Memory V2] Incremental update: {incremental_update.get('field_name', 'unknown')}")
            
            current_turn = len([m for m in messages if isinstance(m, HumanMessage)])
            
            if len(new_messages) >= SUMMARY_TRIGGER_THRESHOLD or incremental_update:
                if show_thinking:
                    print(f"🧠 [Memory V2] Updating layered summary...")
                
                new_summary = update_layered_summary(
                    structured_summary,
                    new_dialogue,
                    model
                )

                if structured_summary.anchor_events:
                    new_summary.anchor_events = _merge_anchor_events(
                        new_summary.anchor_events,
                        structured_summary.anchor_events,
                    )
                
                if incremental_update:
                    if incremental_update.get("field_changed") == "anchor_events":
                        events = _merge_anchor_events(structured_summary.anchor_events, [{
                            "type": incremental_update.get("field_name"),
                            "old_value": incremental_update.get("old_value"),
                            "new_value": incremental_update.get("new_value"),
                            "reason": incremental_update.get("reason"),
                        }])
                        new_summary.anchor_events = events
                    
                    elif incremental_update.get("field_changed") == "immutable_info":
                        immutable = dict(structured_summary.immutable_info or {})
                        immutable[incremental_update.get("field_name")] = incremental_update.get("new_value")
                        new_summary.immutable_info = immutable
                    
                    elif incremental_update.get("field_changed") == "dynamic_info":
                        dynamic = dict(structured_summary.dynamic_info or {})
                        dynamic[incremental_update.get("field_name")] = incremental_update.get("new_value")
                        new_summary.dynamic_info = dynamic
                
                updates["structured_summary"] = new_summary
                updates["summary_memory"] = (
                    (new_summary.text_summary or "").strip()
                    or (structured_summary.text_summary or "").strip()
                    or (state.summary_memory or "").strip()
                )
                updates["summary_memory_cursor"] = current_len - len(removed_ids)
            elif removed_ids and rebased_cursor != cursor:
                updates["summary_memory_cursor"] = rebased_cursor
        elif removed_ids and rebased_cursor != cursor:
            updates["summary_memory_cursor"] = rebased_cursor
        
        return updates
    
    return _run
