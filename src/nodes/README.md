# Clinical Nodes 模块完整文档

本目录包含用于结直肠癌智能诊疗系统的所有节点模块，支持从患者信息收集、临床评估、分期诊断到治疗决策的完整流程。

## 文件结构

```
src/nodes/
├── __init__.py                  # 统一导出所有公共接口 (v5.2)
├── node_utils.py                # 共享工具函数和辅助类
├── intent_nodes.py              # 意图分类和路由节点（语义路由、偏题防御、拼写纠错）
├── knowledge_nodes.py           # 知识检索和联网搜索节点（计划驱动 + 子智能体隔离）
├── assessment_nodes.py          # 风险评估和诊断节点（Semantic Guard + PatientProfile）
├── staging_nodes.py             # 结肠癌和直肠癌分期节点
├── radiology_nodes.py           # 影像医生智能体节点（YOLO检测、U-Net分割、影像组学）
├── pathology_nodes.py           # 病理医生智能体节点（CLAM病理分析）
├── decision_nodes.py            # 决策和审核节点（Decision-Critic 循环 + SubAgent批量检索）
├── decision_evaluation_nodes.py # 决策质量评估节点（LLM-Judge）
├── citation_nodes.py            # 引用校验节点
├── database_nodes.py            # 病例数据库查询节点
├── router.py                    # 路由函数（计划驱动路由 + 优先级路由）
├── planner.py                   # 自主规划节点（自我纠错、多任务处理、工具类型映射）
├── sub_agent.py                 # 子智能体沙箱框架（上下文隔离、结果蒸馏、批量检索）
├── parallel_subagents.py        # 并行子智能体执行节点
├── general_nodes.py             # 闲聊/响应综合节点
├── policy.py                    # 策略决策节点
├── error_handler.py             # 错误处理节点
├── tools_executor.py            # 通用工具执行节点
├── SELF_CORRECTION.md           # 自我纠错架构文档
└── README.md                    # Nodes 模块文档
```

## 架构概述

系统采用模块化设计，每个节点负责特定功能：

1. **意图理解层**：`intent_nodes.py` → `planner.py`
2. **信息收集层**：`database_nodes.py` → `radiology_nodes.py` → `pathology_nodes.py`
3. **临床评估层**：`assessment_nodes.py` → `staging_nodes.py`
4. **决策支持层**：`knowledge_nodes.py` → `decision_nodes.py` → `evaluation_nodes.py`
5. **输出层**：`general_nodes.py` → `citation_nodes.py`

---

## 节点详细说明

### 1. node_utils.py - 共享工具和辅助函数

**包含内容：**

| 类别 | 函数/类 | 描述 |
|------|---------|------|
| 终端输出 | `ThinkingColors` | 思考过程颜色配置类 |
| 终端输出 | `ThinkingResult` | 思考结果容器类 |
| JSON处理 | `_clean_json_string()` | 清理JSON字符串中的`<think>`标签和Markdown |
| JSON处理 | `_clean_and_validate_json()` | 清理并验证JSON格式 |
| JSON处理 | `_unwrap_nested_json()` | 解析嵌套JSON结构 |
| 引用处理 | `_extract_and_update_references()` | 从RAG工具输出中提取元数据 |
| 文本处理 | `_calculate_text_similarity()` | Jaccard相似度计算 |
| 思考解析 | `_parse_thinking_tags()` | 解析`<think>`标签 |
| 思考解析 | `_extract_thinking_from_chunk()` | 从流式chunk中提取思考内容 |
| 流式处理 | `_invoke_with_streaming()` | 流式LLM调用 |
| 消息处理 | `_ensure_message()` | 确保返回有效的LangChain消息对象 |
| 工具执行 | `_execute_tool_calls()` | 执行工具调用 |
| 工具执行 | `_execute_tool_calls_robust()` | 鲁棒工具执行（带重试） |
| 用户提取 | `_user_text()` | 提取用户文本内容 |
| 用户提取 | `_latest_user_text()` | 提取最新用户消息 |
| 改进度 | `_calculate_improvement()` | 计算改进度 |
| 重复检测 | `_is_repeated_rejection()` | 检测重复拒绝 |
| 兜底方案 | `_generate_fallback_plan()` | 生成兜底方案 |
| 兜底方案 | `_build_fallback_search_query()` | 构建兜底搜索查询 |
| 上下文判断 | `_is_postop_context()` | 判断是否为术后上下文 |
| 决策判断 | `_needs_full_decision()` | 判断是否需要完整决策 |
| 工具选择 | `_select_tools()` | 选择合适的工具 |
| 上下文管理 | `_estimate_tokens()` | 估算token数量 |
| 上下文管理 | `_truncate_message_history()` | 截断消息历史 |
| 上下文管理 | `_compress_rag_context()` | 压缩RAG上下文 |
| 上下文管理 | `_create_rag_digest()` | 创建RAG摘要 |

**API规格：**

```python
def _estimate_tokens(text: str) -> int:
    """估算文本的token数量"""
    return len(text) // 4  # 粗略估算

def _truncate_message_history(
    messages: List[BaseMessage], 
    max_tokens: int = 12000
) -> List[BaseMessage]:
    """截断消息历史以适应上下文窗口"""
    # 实现细节...
    pass

def _execute_tool_calls_robust(
    tool_calls: List[Dict], 
    tools_map: Dict[str, BaseTool],
    max_retries: int = 2
) -> List[ToolMessage]:
    """鲁棒执行工具调用，支持重试"""
    # 实现细节...
    pass
```

**使用示例：**

```python
from .node_utils import (
    _latest_user_text,
    _execute_tool_calls,
    _unwrap_nested_json,
    _estimate_tokens
)

# 基本用法
user_text = _latest_user_text(state)
tokens = _estimate_tokens(user_text)

# JSON处理
json_str = '{"data": {"key": "value"}}'
cleaned = _clean_json_string(json_str)
parsed = _unwrap_nested_json(cleaned)
```

**配置选项：**

```python
# 上下文窗口配置
MAX_MESSAGE_TOKENS = 12000
TRUNCATION_STRATEGY = "middle"  # "middle" | "earliest" | "latest"

# Token估算配置
TOKENS_PER_CHINESE_CHAR = 0.25
TOKENS_PER_ENGLISH_WORD = 0.75
```

**错误处理：**

- JSON解析失败时返回`None`，调用方应提供兜底处理
- Token估算使用粗略算法，实际数量可能存在±10%误差
- 工具执行失败时会记录错误日志并返回错误消息

**性能考虑：**

- `_estimate_tokens()`使用字符数估算，比完整分词快10-100倍
- `_truncate_message_history()`使用二分查找优化截断位置
- `_compress_rag_context()`在长上下文中自动压缩冗余信息

---

### 2. intent_nodes.py - 意图分类与路由节点 (v5.2)

**核心功能：**

| 函数 | 描述 |
|------|------|
| `node_intent_classifier()` | 意图分类节点（语义LLM路由） |
| `route_by_intent()` | 基于意图的路由函数 |
| `_get_recent_conversation_history()` | 提取最近对话历史 |
| `_format_messages_for_summary()` | 将消息格式化为摘要 |
| `_maybe_update_summary_memory()` | 生成/更新摘要记忆 |

**v5.2 最新更新：**

- **语义化路由**：完全移除正则表达式，使用LLM进行语义理解
- **偏题防御**：`off_topic_redirect`意图，处理非医疗话题和乱码输入
- **拼写纠错**：`correction_suggestion`字段，自动纠正用户输入错误
- **多任务支持**：`multi_task`意图，支持单个命令中的多个任务
- **对话历史理解**：提取最近3轮对话历史，增强上下文感知
- **综合响应模式**：计划完成后，`general_chat`节点负责综合所有步骤结果

**意图分类类型：**

| 意图 | 描述 | 示例 |
|------|------|------|
| `imaging_analysis` | 影像分析 | "分析CT影像" |
| `pathology_analysis` | 病理分析 | "分析病理切片" |
| `imaging_query` | 影像信息查询 | "查看影像列表" |
| `case_database_query` | 查库 | "查一下093号" |
| `clinical_assessment` | 临床评估 | "我便血3天" |
| `treatment_decision` | 治疗方案 | "该怎么治?" |
| `knowledge_query` | 纯医学知识问答 | "什么是T3期?" |
| `general_chat` | 纯闲聊 | "你好/谢谢" |
| `off_topic_redirect` | 偏题/乱码/非医疗话题 | "今天天气如何" |
| `multi_task` | 多任务复合命令 | "查询病例并给出治疗方案" |

**IntentDecision Schema：**

```python
class IntentDecision(BaseModel):
    """用户意图分类结果（支持多任务识别）"""
    category: Literal[
        "imaging_analysis", "pathology_analysis", "imaging_query",
        "case_database_query", "clinical_assessment", "treatment_decision",
        "knowledge_query", "general_chat", "off_topic_redirect", "multi_task"
    ]
    sub_tasks: Optional[List[Literal[...]]] = None
    correction_suggestion: Optional[str] = None
    reasoning: str = ""
```

**路由规则：**

```python
ROUTING_RULES = {
    "imaging_analysis": "rad_agent",
    "pathology_analysis": "path_agent",
    "imaging_query": "case_database",
    "case_database_query": "case_database",
    "treatment_decision": "decision",
    "knowledge_query": "knowledge",
    "general_chat": "general_chat",
    "off_topic_redirect": "general_chat",
    "multi_task": "assessment",  # 多任务由规划器节点处理
    "clinical_assessment": "assessment",
}
```

**使用示例：**

```python
from intent_nodes import node_intent_classifier, route_by_intent

# 创建意图分类节点
classifier = node_intent_classifier(
    model=llm,
    show_thinking=True
)

# 在状态机上使用
result = classifier.invoke(state)
intent = result.findings["user_intent"]

# 路由到对应节点
next_node = route_by_intent(state)
```

**配置选项：**

```python
# 意图分类配置
INTENT_CLASSIFIER_TEMPERATURE = 0.0  # 使用确定性的低温度
SUMMARY_TRIGGER_TURNS = 8  # 多少轮对话后触发摘要
KEEP_LAST_TURNS = 4  # 保留最近几轮

# 偏题防御配置
OFF_TOPIC_KEYWORDS = [...]  # 医疗领域关键词列表
MAX_HISTORY_TURNS = 3  # 参考对话历史轮数
```

**错误处理：**

- 意图分类失败时默认使用`clinical_assessment`
- 多任务解析失败时降级为单任务处理
- LLM返回格式异常时使用正则兜底提取

**性能考虑：**

- 意图分类使用结构化输出，避免JSON解析开销
- 摘要记忆机制减少长对话的上下文膨胀
- 批量处理多个意图时共享LLM调用

---

### 3. planner.py - 自主规划与自我纠错节点 (v5.2)

**核心功能：**

| 函数 | 描述 |
|------|------|
| `node_planner()` | 自主规划节点 |
| `get_current_pending_step()` | 获取当前待执行步骤 |
| `mark_step_completed()` | 标记步骤完成 |
| `mark_step_failed()` | 标记步骤失败 |
| `has_too_many_retries()` | 检查重试次数 |
| `_get_profile_summary()` | 生成患者档案摘要 |
| `_get_user_intent_summary()` | 提取用户意图摘要 |

**v5.2 核心特性：**

- **自主规划**：基于患者档案和用户意图生成可执行计划
- **自我纠错**：检测失败步骤，分析错误原因，调整策略重新规划
- **多任务处理**：支持单个命令中的多个子任务，使用不同ID前缀（task1_step1, task2_step1）
- **工具类型自动映射**：智能映射无效工具名到有效工具
- **熔断保护**：步骤重试上限3次，规划迭代上限5次
- **治疗决策优化**：治疗决策意图不再生成知识检索步骤，由Decision节点的SubAgent统一执行批量检索

**PlanStep Schema：**

```python
class PlanStep(BaseModel):
    """计划步骤"""
    id: str  # 例如 "task1_step1"
    tool_needed: str  # 工具名称
    description: str  # 步骤描述
    status: Literal["pending", "in_progress", "completed", "failed"]
    priority: int = 0
    retry_count: int = 0
    assignee: Optional[str] = None
    parallel_group: Optional[str] = None  # 并行组标识
```

**支持的工具类型：**

| 工具类型 | 别名 | 描述 |
|---------|------|------|
| `list_guideline_toc` | `toc` | 查看指南目录 |
| `read_guideline_chapter` | `chapter` | 阅读指南章节 |
| `search_treatment_recommendations` | `search` | 检索治疗推荐 |
| `case_database_query` | `database_query` | 查询病例数据库 |
| `web_search` | `web` | 在线搜索 |
| `ask_user` | - | 询问用户补充信息 |
| `imaging_analysis` | - | 影像分析（自动映射） |
| `pathology_analysis` | - | 病理分析（自动映射） |

**使用示例：**

```python
from planner import (
    node_planner,
    get_current_pending_step,
    mark_step_completed,
    mark_step_failed
)

# 创建规划节点
planner = node_planner(
    model=llm,
    show_thinking=True
)

# 生成计划
plan = planner.invoke(state)

# 获取当前步骤
current_step = get_current_pending_step(state)
if current_step:
    print(f"待执行: {current_step.description}")

# 标记完成
updated_plan = mark_step_completed(state, current_step.id)

# 检查重试
if has_too_many_retries(state, current_step):
    print("步骤重试次数过多，触发熔断")
```

**配置选项：**

```python
# 规划配置
MAX_PLAN_ITERATIONS = 5  # 最大规划迭代次数
MAX_STEP_RETRIES = 3  # 单步骤最大重试次数
MAX_TOOL_NAME_VARIANTS = 5  # 工具名变体最大数量

# 熔断配置
CIRCUIT_BREAKER_THRESHOLD = 3  # 熔断阈值
FALLBACK_PLAN_ENABLED = True  # 是否启用兜底方案
```

**错误处理：**

- 规划失败时生成兜底方案
- 步骤重试超限时触发Planner自我纠错
- 工具调用失败时尝试自动映射到有效工具

**性能考虑：**

- 静默跳过优化：快速跳过检查时不输出日志
- 使用_profile_summary避免重复解析患者信息
- 并行任务优化：自动识别可并行的步骤组

---

### 4. knowledge_nodes.py - 知识检索节点 (v5.2)

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_knowledge_retrieval()` | 智能知识检索节点 |
| `node_web_search_agent()` | 联网搜索代理节点 |
| `SearchPlan` | 搜索策略计划结构 |
| `KnowledgeSufficiencyEval` | 知识充分性评估结构 |
| `_create_search_planner()` | 创建搜索规划器 |
| `_create_sufficiency_evaluator()` | 创建充分性评估器 |
| `_should_use_patient_context()` | 判断是否需要患者上下文 |

**v5.2 计划驱动模式：**

- **计划优先执行**：检测`current_plan`中的待执行步骤，根据`tool_needed`智能路由
- **工具执行验证**：自动检测工具执行结果（检查"未找到"等错误关键词）
- **失败步骤标记**：执行失败时自动标记步骤为`failed`，触发Planner自我纠错
- **步骤完成标记**：成功时自动标记步骤为`completed`，更新`current_plan`

**v5.1 子智能体隔离模式：**

- **上下文完全隔离**：联网搜索在子智能体沙箱中进行，不污染主智能体上下文
- **结果蒸馏**：主智能体只看到最终的`<report>`内容，所有中间过程自动销毁
- **自动销毁历史**：任务完成后自动销毁沙箱历史，释放内存
- **Token优化**：大幅降低token消耗，提高响应效率

**v2.2.0 分层检索策略（自动模式）：**

| 层级 | 名称 | 描述 |
|------|------|------|
| Layer 1 | Authority | 权威层：指南搜索 |
| Layer 2 | Evidence | 证据层：临床试验/Meta分析搜索 |
| Layer 3 | Safety | 安全层：药物相互作用检查 |

**SearchPlan Schema：**

```python
class SearchPlan(BaseModel):
    """搜索策略计划"""
    needs_search: bool  # 是否需要外部搜索
    search_query: str  # 优化的搜索查询字符串
    selected_tool: str  # 工具名称
    tool_arguments: dict = {}  # 工具参数
    reasoning: str  # 选择原因
```

**使用示例：**

```python
from knowledge_nodes import node_knowledge_retrieval

# 创建知识检索节点
knowledge_retriever = node_knowledge_retrieval(
    model=llm,
    tools=available_tools,
    streaming=True,
    show_thinking=True,
    use_sub_agent=True  # 使用子智能体隔离
)

# 执行检索
result = knowledge_retriever.invoke(state)
```

**配置选项：**

```python
# 检索配置
USE_SUB_AGENT = True  # 是否使用子智能体隔离
MAX_ITERATIONS = 4  # 最大迭代次数
TOP_K_RESULTS = 12  # 返回结果数量

# 分层检索配置
ENABLE_HIERARCHICAL_SEARCH = True
HIERARCHY_LAYERS = ["Authority", "Evidence", "Safety"]
```

**错误处理：**

- 搜索失败时尝试备用工具
- 工具执行结果包含"未找到"时标记步骤失败
- 子智能体执行失败时返回错误报告而非阻断流程

**性能考虑：**

- 子智能体隔离避免主上下文膨胀
- 批量检索减少重复查询
- 分层策略优先检索高优先级结果

---

### 5. assessment_nodes.py - 评估和诊断节点 (v5.0)

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_assessment()` | 风险评估节点 |
| `node_diagnosis()` | 诊断节点 |
| `node_staging_router()` | 分期路由函数 |
| `CaseIntegrity` | 语义完整性检查模型 |
| `ClinicalAssessmentResult` | 临床评估结果结构 |
| `DiagnosisExtractionResult` | 诊断结果结构 |
| `check_case_integrity()` | 语义完整性检查函数 |
| `_semantic_extract_diagnosis()` | 纯语义诊断提取函数 |

**v5.0 Semantic Guard 重大升级：**

- **CaseIntegrity模型**：完全替代`_is_complete_case_info`正则逻辑，使用LLM语义判断
- **语义分类**：`tumor_location_category`（Rectum/Colon/Unknown）、`tnm_status`（Complete/Partial/Missing）
- **进展期判断**：`is_advanced_stage`基于TNM自动判断
- **MMR态度检测**：`mmr_status_availability`（Provided/Not_Provided/User_Refused_Or_Unknown）

**v3.0 Fast Pass策略：**

- **架构原则变更**：从"跳过节点"改为"轻量化通过"，确保数据流水线完整
- **Fast Pass模式**：完整病例使用轻量化格式化LLM调用，只提取JSON不做复杂分析

**v3.0 Fast Pass策略：**

- **架构原则变更**：从"跳过节点"改为"轻量化通过"，确保数据流水线完整
- **Fast Pass模式**：完整病例使用轻量化格式化LLM调用，只提取JSON不做复杂分析

**ClinicalAssessmentResult 详细API规格：**

```python
class ClinicalAssessmentResult(BaseModel):
    """
    临床评估结果结构
    
    Version Compatibility:
        v5.0+: 使用field_validator自动修正常见格式错误
        v4.0: 新增red_flags字段验证器
        v3.0: 初始Fast Pass版本
    
    Attributes:
        risk_level (str): 风险等级 (High/Moderate/Average)
        red_flags (List[str]): 高危症状列表
        missing_critical_data (List[str]): 缺失数据列表
        assessment_summary (str): 病情总结
        reasoning (str): 分析过程
    
    Examples:
        >>> result = ClinicalAssessmentResult(
        ...     risk_level="High",
        ...     red_flags=["便血", "体重下降"],
        ...     missing_critical_data=["MMR状态"],
        ...     assessment_summary="患者为高风险直肠癌",
        ...     reasoning="基于TNM分期和症状判断"
        ... )
    """
    risk_level: str = Field(
        description="风险等级 (High/Moderate/Average)",
        examples=["High", "Moderate", "Average"]
    )
    red_flags: List[str] = Field(
        description="高危症状",
        default_factory=list,
        examples=[["便血", "体重下降", "腹痛"]]
    )
    missing_critical_data: List[str] = Field(
        description="缺失数据",
        default_factory=list,
        examples=[["MMR状态", "MRI报告"]]
    )
    assessment_summary: str = Field(
        description="病情总结",
        examples=["患者为高风险直肠癌，建议尽快治疗"]
    )
    reasoning: str = Field(
        description="分析过程",
        examples=["基于TNM分期和症状判断"]
    )
    
    @field_validator('risk_level')
    @classmethod
    def validate_risk_level(cls, v):
        """确保风险等级只接受有效值"""
        valid_levels = {"High", "Moderate", "Average"}
        if v not in valid_levels:
            # 自动修正常见错误格式
            v_normalized = v.strip()
            if v_normalized.lower() == "high":
                return "High"
            elif v_normalized.lower() == "moderate":
                return "Moderate"
            elif v_normalized.lower() == "average":
                return "Average"
            # 无法识别，返回默认 Average
            print(f"⚠️ [Validation] 无效的风险等级 '{v}'，自动修正为 'Average'")
            return "Average"
        return v
    
    @field_validator('red_flags')
    @classmethod
    def validate_red_flags(cls, v):
        """清理 red_flags 中的特殊字符"""
        cleaned = []
        for flag in v:
            # 移除可能导致 JSON 问题的特殊字符
            cleaned_flag = flag.replace('"', '\\"').replace('\\', '').strip()
            # 确保不包含中文标点和复杂字符
            if cleaned_flag and len(cleaned_flag) < 100:
                cleaned.append(cleaned_flag)
        return cleaned

class DiagnosisExtractionResult(BaseModel):
    """
    诊断结果结构
    
    Version Compatibility:
        v5.0+: 添加tnm_staging字段
        v4.0: 新增derived_mmr_status, derived_kras_status, derived_braf_status属性
        v3.0: 初始版本
    
    Attributes:
        pathology_confirmed (bool): 是否病理确诊
        tumor_location (Literal["Rectum", "Colon", "Unknown"]): 肿瘤部位
        histology_type (str): 病理类型
        molecular_markers (Dict[str, Any]): 分子标志物
        rectal_mri_params (Dict[str, Any]): MRI参数
        tnm_staging (Dict[str, Any]): TNM分期
        clinical_stage_summary (str): 临床分期总结
    
    Examples:
        >>> result = DiagnosisExtractionResult(
        ...     pathology_confirmed=True,
        ...     tumor_location="Rectum",
        ...     histology_type="腺癌",
        ...     molecular_markers={"MSI-H": True, "KRAS": "WildType"},
        ...     tnm_staging={"cT": "T3", "cN": "N1", "cM": "M0"},
        ...     clinical_stage_summary="直肠癌 cT3N1M0 III期"
        ... )
    """
    pathology_confirmed: bool = Field(
        False,
        description="是否病理确诊",
        examples=[True, False]
    )
    tumor_location: Literal["Rectum", "Colon", "Unknown"] = Field(
        "Unknown",
        description="肿瘤部位 (Rectum/Colon/Unknown)",
        examples=["Rectum", "Colon", "Unknown"]
    )
    histology_type: str = Field(
        "未知",
        description="病理类型",
        examples=["腺癌", "鳞癌", "黏液腺癌"]
    )
    molecular_markers: Dict[str, Any] = Field(
        default_factory=dict,
        description="分子标志物",
        examples=[{"MSI-H": True, "KRAS": "WildType", "BRAF": "WildType"}]
    )
    rectal_mri_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="MRI参数",
        examples=[{"distance_from_anorectum": "5cm", "extramural_invasion": True}]
    )
    tnm_staging: Dict[str, Any] = Field(
        default_factory=dict,
        description="TNM分期",
        examples=[{"cT": "T3", "cN": "N1", "cM": "M0", "clinical_stage": "III"}]
    )
    clinical_stage_summary: str = Field(
        "",
        description="临床分期总结",
        examples=["直肠癌 cT3N1M0 III期"]
    )

    @field_validator("tumor_location", mode="before")
    @classmethod
    def normalize_tumor_location(cls, v):
        """将各种位置表述规范化为 Rectum/Colon/Unknown。"""
        if v is None:
            return "Unknown"
        s = str(v).strip().lower()
        if not s:
            return "Unknown"
        if "rect" in s or "直" in s:
            return "Rectum"
        if "colon" in s or "结" in s:
            return "Colon"
        if s in {"unknown", "n/a", "na", "?"}:
            return "Unknown"
        # 非法值：医疗场景宁可 Unknown，不要猜
        return "Unknown"

    @property
    def derived_mmr_status(self) -> str:
        """
        基于分子标志物推导 MMR 状态
        
        Returns:
            str: dMMR/pMMR/Unknown
        
        Examples:
            >>> result.molecular_markers = {"MSI-H": True}
            >>> result.derived_mmr_status
            'dMMR'
        """
        markers = self.molecular_markers or {}
        # 允许多种键名/值形态
        if markers.get("MSI-H") or markers.get("dMMR") or str(markers.get("MSI", "")).upper() == "H":
            return "dMMR"
        if markers.get("MSS") or markers.get("pMMR"):
            return "pMMR"
        # 有些数据源会直接给出 mmr_status
        mmr = markers.get("MMR") or markers.get("mmr_status")
        if isinstance(mmr, str):
            if "dmmr" in mmr.lower() or "msi-h" in mmr.lower():
                return "dMMR"
            if "pmmr" in mmr.lower() or "mss" in mmr.lower():
                return "pMMR"
        return "Unknown"

    @property
    def derived_kras_status(self) -> str:
        """基于分子标志物推导 RAS/KRAS 状态"""
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
        """基于分子标志物推导 BRAF 状态"""
        markers = self.molecular_markers or {}
        val = markers.get("BRAF")
        if isinstance(val, bool):
            return "Unknown"
        if val is None:
            return "Unknown"
        s = str(val).strip()
        return s if s in {"WildType", "Mutant"} else "Unknown"

**使用示例：**

```python
from assessment_nodes import (
    node_assessment,
    node_diagnosis,
    check_case_integrity
)

# 创建评估节点
assessor = node_assessment(
    model=llm,
    tools=available_tools,
    streaming=True,
    show_thinking=True
)

# 执行评估
result = assessor.invoke(state)

# 检查病例完整性
integrity = check_case_integrity(state)
if integrity.is_complete:
    print("完整病例，使用Fast Pass模式")
```

**配置选项：**

```python
# Fast Pass配置
FAST_PASS_ENABLED = True
FAST_PASS_MIN_FIELDS = ["pathology_confirmed", "tumor_location", "tnm_staging"]

# MMR状态检测
AUTO_DETECT_MMR = True
MMR_PROMPT_ON_MISSING = True
```

**错误处理：**

- JSON解析失败时使用Fail-Open机制
- 字段验证器自动修正常见格式错误
- 缺失关键数据时生成追问请求

**性能考虑：**

- Fast Pass模式跳过复杂分析，毫秒级返回
- 字段验证器批量处理，避免重复LLM调用
- PatientProfile复用减少信息解析开销

---

### 6. staging_nodes.py - 分期节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_colon_staging()` | 结肠癌分期节点 |
| `node_rectal_staging()` | 直肠癌多模态分期节点 |
| `_validate_tnm_consistency()` | TNM组合医学有效性校验 |
| `_is_fast_pass_complete_case()` | Fast Pass完整病例判定 |

**v2.5.0 Fast Pass快速校验：**

- **核心原则**：Fast Pass不是"跳过Staging"，而是"快速校验"
- **TNM有效性校验**：验证T(1-4)、N(0-3)、M(0-1)组合的医学合理性
- **位置一致性校验**：检查文本描述与`tumor_location`字段是否一致
- **复杂情况检测**：识别MRF阳性、EMVI阳性、转移灶等复杂情况

**结肠癌分期功能：**

- 解析CT报告，评估局部和远处分期（M分期）
- Fast Pass模式：快速校验TNM组合有效性，不调用CT工具

**直肠癌分期功能：**

- MRI分析：局部分期（T、N、MRF、EMVI）
- CT分析：远处转移筛查（M分期）
- 智能校正：当CT文本明确显示M0但工具错误返回M1时自动修正

**使用示例：**

```python
from staging_nodes import (
    node_colon_staging,
    node_rectal_staging,
    _validate_tnm_consistency,
    _is_fast_pass_complete_case
)

# 创建结肠癌分期节点
colon_stager = node_colon_staging(tools=available_tools)

# 创建直肠癌分期节点
rectal_stager = node_rectal_staging(tools=available_tools)

# 快速校验TNM
findings = state.findings
is_valid, msg = _validate_tnm_consistency(findings, user_text)
if not is_valid:
    print(f"警告: {msg}")
```

**配置选项：**

```python
# Fast Pass配置
FAST_PASS_STAGING = True
VALIDATE_TNM_CONSISTENCY = True

# 复杂情况检测
COMPLEX_CASE_KEYWORDS = [
    "mrf阳性", "emvi阳性", "脉管癌栓", "神经侵犯",
    "肝脏占位", "肺转移", "腹腔转移"
]
```

**错误处理：**

- TNM值无效时输出警告但继续处理
- 位置不一致时自动标记需要校验
- 缺失CT报告时输出严重警告

**性能考虑：**

- Fast Pass模式毫秒级返回
- 跳过MRI/CT工具调用节省时间
- 智能校正避免重复工具调用

---

### 7. radiology_nodes.py - 影像医生智能体 (v2.0)

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_rad_agent()` | 影像医生智能体节点 |
| `_detect_analysis_mode()` | 检测分析模式 |
| `_run_detection_analysis()` | YOLO肿瘤检测分析 |
| `_run_segmentation_analysis()` | U-Net肿瘤分割分析 |
| `_run_radiomics_analysis()` | PyRadiomics特征提取分析 |
| `_run_comprehensive_analysis()` | 完整影像组学分析 |
| `_finalize_return()` | 计划步骤完成标记 |

**v2.0 支持的分析模式：**

| 模式 | 工具 | 描述 |
|------|------|------|
| `detection` | perform_comprehensive_tumor_check | YOLO快速筛查，返回肿瘤检出率、置信度 |
| `segmentation` | unet_segmentation_tool | U-Net精确分割，返回面积、边界框 |
| `radiomics` | radiomics_feature_extraction_tool | PyRadiomics特征提取（1500维） |
| `comprehensive` | comprehensive_radiomics_analysis | 完整分析：YOLO→U-Net→PyRadiomics→LASSO |

**AnalysisMode类型：**

```python
AnalysisMode = Literal["detection", "segmentation", "radiomics", "comprehensive"]

def _detect_analysis_mode(user_text: str) -> AnalysisMode:
    """根据用户输入检测分析模式"""
    comprehensive_keywords = ["完整分析", "影像组学", "radiomics"]
    if any(kw in user_text.lower() for kw in comprehensive_keywords):
        return "comprehensive"
    # ... 其他模式检测
```

**使用示例：**

```python
from radiology_nodes import node_rad_agent

# 创建影像医生智能体
rad_agent = node_rad_agent(
    tools=available_tools,
    model=llm,
    streaming=True,
    show_thinking=True
)

# 执行影像分析
result = rad_agent.invoke(state)
report = result.findings.get("radiology_report")
```

**配置选项：**

```python
# 分析模式配置
DEFAULT_MODE = "detection"
SUPPORTED_MODES = ["detection", "segmentation", "radiomics", "comprehensive"]

# 工具配置
YOLO_MODEL = "yolov8"
UNET_MODEL = "unet"
PYRADIOMICS_VERSION = "3.0"
```

**错误处理：**

- 工具调用失败时生成错误报告
- 重复分析检测：已有报告时跳过工具调用
- 患者ID提取失败时使用state中的默认ID

**性能考虑：**

- 报告复用机制避免重复分析
- 计划步骤自动管理减少状态同步开销
- 模式检测使用关键词匹配而非LLM调用

---

### 8. pathology_nodes.py - 病理医生智能体

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_path_agent()` | 病理医生智能体节点 |
| `_detect_pathology_mode()` | 检测病理分析模式 |
| `_extract_patient_id()` | 从文本提取患者ID |
| `_extract_slide_path()` | 提取切片文件路径 |
| `_build_status_text()` | 构建状态文本 |
| `_build_pathology_card_base()` | 构造病理卡片基础字段 |

**PathologyMode类型：**

```python
PathologyMode = Literal["full", "quick", "status"]

def _detect_pathology_mode(user_text: str) -> PathologyMode:
    """根据用户输入检测病理分析模式"""
    status_keywords = ["状态", "依赖", "版本", "status"]
    if any(kw in user_text for kw in status_keywords):
        return "status"
    quick_keywords = ["快速", "筛查", "初筛"]
    if any(kw in user_text for kw in quick_keywords):
        return "quick"
    return "full"
```

**支持的功能：**

- 病理切片分类（完整分析）
- 快速病理筛查
- 工具状态查询
- 综合病理分析（按患者ID自动查找切片）

**使用示例：**

```python
from pathology_nodes import node_path_agent

# 创建病理医生智能体
path_agent = node_path_agent(
    tools=available_tools,
    model=llm,
    streaming=True,
    show_thinking=True
)

# 执行病理分析
result = path_agent.invoke(state)
```

**配置选项：**

```python
# CLAM工具配置
CLAM_MODEL_PATH = "path/to/model"
CLAM_TOOL_DIR = "path/to/clam"

# 分析模式配置
DEFAULT_MODE = "full"
SUPPORTED_MODES = ["full", "quick", "status"]
```

**错误处理：**

- 工具依赖缺失时返回状态报告
- 切片路径无效时尝试自动查找
- GPU不可用时自动切换CPU模式

**性能考虑：**

- 快速模式跳过热力图生成
- 状态查询直接返回缓存信息
- 批量切片分析支持并行处理

---

### 9. decision_nodes.py - 决策和审核节点 (v5.2)

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_decision()` | 决策节点 |
| `node_critic()` | 审核节点 |
| `route_by_critic_v2()` | 基于迭代次数和审核的路由 |
| `node_finalize()` | 最终化节点 |
| `TreatmentAction` | 治疗措施结构 |
| `ClinicalDecisionSchema` | 决策方案结构 |
| `CriticEvaluationSchema` | 审核结果结构 |
| `TreatmentSearchQueries` | 语义搜索关键词生成器 |

**v5.2 核心特性 - SubAgent批量检索：**

- **统一检索入口**：所有指南检索由Decision节点调用`run_isolated_rag_search`统一执行
- **批量模式**：合并多个查询为单次批量检索，减少重复查询
- **提高效率**：使用top_k=12一次性获取足够的结果，减少迭代次数
- **Token优化**：显著降低token消耗

**v4.0 语义查询生成：**

- **LLM语义查询**：完全移除硬编码if-else拼接逻辑，使用LLM生成3-5条精准检索关键词
- **多查询合并**：执行多条查询，合并检索结果并去重引用
- **变体兼容**：`TreatmentSearchQueries`使用`model_validator`兼容模型常见输出变体

**ClinicalDecisionSchema 详细API规格：**

```python
class ClinicalDecisionSchema(BaseModel):
    """
    临床决策方案结构
    
    Attributes:
        summary (str): 患者病情摘要及诊断分期结论，应包含关键病情信息
        treatment_plan (List[TreatmentAction]): 按时间顺序排列的治疗步骤列表
        follow_up (List[str]): 随访计划建议列表
    
    Version Compatibility:
        v5.2+: 使用List[TreatmentAction]替代Dict，稳定性提升
        v4.0+: 新增field_validator自动清理特殊字符
    
    Examples:
        >>> schema = ClinicalDecisionSchema(
        ...     summary="患者为直肠癌T3N1M0，dMMR",
        ...     treatment_plan=[
        ...         TreatmentAction(
        ...             title="1. 新辅助治疗",
        ...             content="CAPOX方案化疗3周期 Reason: 降低术后复发风险"
        ...         )
        ...     ],
        ...     follow_up=["术后3个月复查CT", "术后6个月肠镜检查"]
        ... )
    """
    summary: str = Field(
        description="患者病情摘要及诊断分期结论",
        min_length=10,
        max_length=500,
        examples=["患者为直肠癌T3N1M0，dMMR，建议新辅助治疗后手术"]
    )
    treatment_plan: List[TreatmentAction] = Field(
        description="按时间顺序排列的治疗步骤列表",
        min_length=1,
        description="❌ 严禁使用字典结构，必须是列表"
    )
    follow_up: List[str] = Field(
        description="随访计划",
        min_length=1,
        max_length=10,
        examples=["术后3个月复查CT", "术后6个月肠镜检查"]
    )

class TreatmentAction(BaseModel):
    """
    单项治疗措施结构
    
    Attributes:
        title (str): 步骤标题，必须包含序号
        content (str): 详细方案，包含药物名称、周期、剂量、Reasoning
    
    Version Compatibility:
        v5.2+: title强制要求包含序号
        v4.0+: content强制要求包含Reasoning
    
    Examples:
        >>> action = TreatmentAction(
        ...     title="1. 新辅助治疗",
        ...     content="CAPOX方案化疗3周期 Reason: 降低术后复发风险"
        ... )
    """
    title: str = Field(
        description="步骤标题（必须包含序号，例如 '1. 新辅助治疗'）",
        pattern=r"^\d+\.",  # 必须是数字+句点开头
        examples=["1. 新辅助治疗", "2. 手术", "3. 辅助化疗"]
    )
    content: str = Field(
        description="详细方案（必须包含药物名称、周期、剂量、以及该步骤的临床获益理由（Reasoning））",
        min_length=20,
        examples=["CAPOX方案化疗3周期 Reason: 降低术后复发风险"]
    )

class CriticEvaluationSchema(BaseModel):
    """审核结果结构"""
    model_config = ConfigDict(
        extra="ignore",  # 忽略额外字段
        str_strip_whitespace=True,  # 自动去除首尾空格
    )
    
    verdict: Literal["APPROVED", "REJECTED", "APPROVED_WITH_WARNINGS"] = Field(
        description="审核结论",
        examples=["APPROVED", "REJECTED", "APPROVED_WITH_WARNINGS"]
    )
    feedback: str = Field(description="审核意见")
    
    @field_validator('feedback', mode='before')
    @classmethod
    def clean_feedback(cls, v):
        """清理 feedback 中可能破坏 JSON 的字符"""
        if not isinstance(v, str):
            return str(v)
        v = v.strip()
        return v

class TreatmentSearchQueries(BaseModel):
    """
    智能生成的搜索关键词列表
    
    Version Compatibility:
        v5.2+: 新增model_validator兼容多种输出变体
        v4.0: 初始版本
    
    Attributes:
        queries (List[str]): 3-5个精准的医学搜索关键词
    
    Examples:
        >>> queries = TreatmentSearchQueries(queries=[
        ...     "直肠癌 T3N1 新辅助治疗 指南",
        ...     "dMMR 结直肠癌 免疫治疗 循证医学"
        ... ])
    """
    model_config = ConfigDict(extra="ignore")

    queries: List[str] = Field(
        description="3-5个精准的医学搜索关键词。例如：['直肠癌 T3N1 新辅助治疗 指南', 'dMMR 结直肠癌 免疫治疗 循证医学']",
        max_length=5,
        min_length=1
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_variants(cls, data: Any):
        """
        兼容模型常见输出变体，避免因为字段名/格式轻微偏差导致整体失败：
        - {"query": "..."} -> {"queries": ["..."]}
        - {"keywords": [...]} / {"queries": "..."} -> 标准化为 List[str]
        - str: 允许包含尾随解释文本，尽量提取首个 JSON 对象
        """
        if data is None:
            return {"queries": []}

        # 如果是字符串，尽量提取 JSON（处理 trailing characters / 代码块）
        if isinstance(data, str):
            import re
            s = data.strip()
            s = s.replace("```json", "").replace("```", "").strip()
            # ... 更多解析逻辑
            return {"queries": [s]}

        if isinstance(data, list):
            return {"queries": data}

        if isinstance(data, dict):
            if "queries" not in data:
                if "query" in data:
                    data["queries"] = [data.get("query")]
                elif "keywords" in data:
                    data["queries"] = data.get("keywords")
                else:
                    data["queries"] = []
            return data

        return {"queries": []}

**使用示例：**

```python
from decision_nodes import (
    node_decision,
    node_critic,
    node_finalize,
    route_by_critic_v2
)

# 创建决策节点
decision_maker = node_decision(
    model=llm,
    tools=available_tools,
    streaming=True,
    show_thinking=True
)

# 创建审核节点
critic = node_critic(
    model=llm,
    streaming=True,
    show_thinking=True
)

# 创建最终化节点
finalizer = node_finalize(model=llm)

# 执行决策流程
decision = decision_maker.invoke(state)
critique = critic.invoke(state)
next_node = route_by_critic_v2(state)
```

**配置选项：**

```python
# 决策配置
MAX_DECISION_ITERATIONS = 4  # 最大决策迭代次数
BATCH_SEARCH_MODE = True  # 批量检索模式
TOP_K_RESULTS = 12  # 检索结果数量

# 审核配置
CRITIC_TEMPERATURE = 0.1  # 审核使用低温度
ENABLE_STRUCTURED_OUTPUT = True
```

**错误处理：**

- JSON解析失败时使用Fail-Open机制
- 迭代熔断：3次失败后生成兜底方案
- 工具连续失败2次后建议切换其他工具

**性能考虑：**

- 批量检索减少重复查询
- 语义查询生成避免无效检索
- 自动路线图更新减少手动操作

---

### 10. decision_evaluation_nodes.py - 决策质量评估节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_llm_judge()` | LLM-Judge评估节点 |
| `EvaluationReport` | 评估报告结构 |
| `_should_reject()` | 判断是否需要拒绝 |

**EvaluationReport：**

```python
class EvaluationReport(BaseModel):
    """LLM-Judge评估结果"""
    factual_accuracy: int  # 1-5分
    citation_accuracy: int  # 1-5分
    completeness: int  # 1-5分
    safety: int  # 1-5分
    verdict: Literal["PASS", "FAIL"]  # 最终判断
    feedback: str  # 简要说明
```

**评估维度：**

| 维度 | 描述 | 最低通过分数 |
|------|------|-------------|
| 事实正确性 | 医学事实的准确性 | 3 |
| 引用准确性 | 引用的文献来源是否正确 | 3 |
| 完整性 | 是否遗漏重要信息 | 3 |
| 安全性 | 治疗建议的安全性 | 3 |

**使用示例：**

```python
from evaluation_nodes import node_llm_judge, route_after_evaluator

# 创建评估节点
judge = node_llm_judge(
    model=llm,
    streaming=True,
    show_thinking=True
)

# 执行评估
result = judge.invoke(state)

# 路由决策
next_node = route_after_evaluator(state)
if next_node == "decision":
    # 需要重新决策
    print("评估未通过，需要重新决策")
```

**配置选项：**

```python
# 评估配置
MIN_PASS_SCORE = 3  # 最低通过分数
MAX_RETRY_COUNT = 2  # 最大重试次数
EVALUATION_TEMPERATURE = 0.0  # 使用确定性输出
```

**错误处理：**

- 评估失败时默认通过（Fail-Open）
- 所有分数低于3分时触发重试
- 重试超限时强制最终化

**性能考虑：**

- 评估节点在决策后执行，避免不必要的评估
- 批量评估多个维度共享LLM调用
- 简单规则判断减少LLM调用频率

---

### 11. citation_nodes.py - 引用校验节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_citation_agent()` | 引用校验节点 |
| `CitationReport` | 引用报告结构 |

**CitationReport：**

```python
class CitationReport(BaseModel):
    """引用校验结果"""
    coverage_score: int  # 0-100 引用覆盖评分
    missing_claims: List[str]  # 缺少引用的关键结论
    needs_more_sources: bool  # 是否需要补充引用来源
    notes: str  # 简要说明
```

**使用示例：**

```python
from citation_nodes import node_citation_agent

# 创建引用校验节点
citation_checker = node_citation_agent(
    model=llm,
    streaming=True,
    show_thinking=True
)

# 执行校验
result = citation_checker.invoke(state)
coverage = result.citation_report.coverage_score
```

**配置选项：**

```python
# 引用配置
MIN_COVERAGE_SCORE = 70  # 最低覆盖分数
CITATION_TEMPERATURE = 0.0
```

**错误处理：**

- 校验失败时返回零分覆盖
- 缺少引用时标记需要补充

**性能考虑：**

- 与评估节点共享LLM调用
- 引用覆盖评分使用确定性输出

---

### 12. database_nodes.py - 病例数据库查询节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_case_database()` | 病例数据库查询节点 |
| `_format_case_summary_markdown()` | 病例信息Markdown摘要格式化 |
| `_normalize_tumor_location()` | 肿瘤位置标准化处理 |
| `_extract_molecular_markers()` | 从病例数据中提取分子标志物 |

**功能特性：**

- 支持纯LLM工具调用进行病例数据库查询
- 病例信息友好的Markdown摘要展示
- 肿瘤位置智能标准化（中英文转换）
- 分子标志物自动提取和分析

**使用示例：**

```python
from database_nodes import (
    node_case_database,
    _format_case_summary_markdown,
    _normalize_tumor_location
)

# 创建数据库查询节点
db_query = node_case_database(
    model=llm,
    tools=available_tools,
    streaming=True,
    show_thinking=True
)

# 执行查询
result = db_query.invoke(state)

# 格式化摘要
summary = _format_case_summary_markdown(case_data)

# 标准化位置
normalized = _normalize_tumor_location("升结肠癌")
```

**配置选项：**

```python
# 数据库配置
SUPPORTED_LOCATIONS = ["rectum", "colon", "unknown"]
MARKER_EXTRACTION_ENABLED = True
SUMMARY_MAX_LENGTH = 1000
```

**错误处理：**

- 查询失败时返回错误信息
- 位置无法标准化时使用"unknown"
- 分子标志物格式异常时返回空字典

**性能考虑：**

- 使用LLM自动选择查询工具
- 摘要格式化减少手动处理
- 批量提取分子标志物

---

### 13. sub_agent.py - 子智能体沙箱框架 (v5.2)

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `SubAgentContext` | 子智能体隔离上下文管理器 |
| `SubAgentResult` | 子智能体执行结果 |
| `create_rag_researcher()` | RAG检索子智能体工厂函数 |
| `create_web_researcher()` | 网络研究子智能体工厂函数 |
| `run_isolated_rag_search()` | 同步RAG检索执行辅助函数 |
| `run_isolated_web_search()` | 同步网络搜索执行辅助函数 |

**v5.2 核心特性：**

- **上下文隔离**：子智能体在独立的messages列表中工作，不污染主智能体上下文
- **结果蒸馏**：只返回`<report>`标签包裹的精华内容，丢弃中间过程
- **自动销毁**：任务完成后自动销毁沙箱历史，释放内存
- **Token优化**：大幅降低token消耗，提高响应效率
- **重试与熔断**：内置最大迭代次数限制，防止无限循环
- **引用提取**：自动从工具结果中提取结构化引用

**SubAgentContext 详细API规格：**

```python
class SubAgentContext:
    """
    子智能体隔离上下文管理器
    
    核心特性：
    1. 独立的 messages 列表（不污染主智能体）
    2. 任务结束后自动销毁历史
    3. 强制结果蒸馏（只返回 <report> 内容）
    4. 内置重试和熔断机制
    
    Version Compatibility:
        v5.2: 新增自动工具映射，Token优化
        v5.1: 优化迭代逻辑，减少无效迭代
        v5.0: 初始隔离框架版本
    
    Attributes:
        model (BaseChatModel): LLM 模型实例
        system_prompt (str): 子智能体的系统提示（定义其角色和行为）
        task_description (str): 主智能体委派的具体任务
        max_iterations (int): 最大迭代次数（防止无限循环），默认4
        show_thinking (bool): 是否显示子智能体的思考过程（调试用）
        _sandbox_messages (List[BaseMessage]): 独立的消息历史（隔离核心）
        _collected_references (List[Dict[str, Any]]): 收集的引用
    
    Examples:
        >>> agent = SubAgentContext(
        ...     model=model,
        ...     system_prompt="你是一个专业的医学文献检索员...",
        ...     task_description="查找 T3N1 直肠癌的新辅助治疗方案"
        ... )
        >>> result = agent.execute_with_tools(rag_tools)
        >>> print(result.report)  # 蒸馏后的结果
    """
    
    # 蒸馏提取的标签
    REPORT_TAG_PATTERN = re.compile(r'<report>(.*?)</report>', re.DOTALL | re.IGNORECASE)
    SUMMARY_TAG_PATTERN = re.compile(r'<summary>(.*?)</summary>', re.DOTALL | re.IGNORECASE)
    
    def __init__(
        self,
        model,
        system_prompt: str,
        task_description: str,
        max_iterations: int = 4,
        show_thinking: bool = True
    ):
        """
        初始化子智能体上下文
        
        Args:
            model: LLM 模型实例
            system_prompt: 子智能体的系统提示（定义其角色和行为）
            task_description: 主智能体委派的具体任务
            max_iterations: 最大迭代次数（防止无限循环）【优化：默认 4 次】
            show_thinking: 是否显示子智能体的思考过程（调试用）
        
        Returns:
            SubAgentContext: 初始化后的子智能体上下文
        
        Raises:
            ValueError: 如果 model 为 None 或 max_iterations < 1
        """
        self.model = model
        self.system_prompt = system_prompt
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.show_thinking = show_thinking
        
        # 独立的消息历史（这是隔离的核心）
        self._sandbox_messages: List[BaseMessage] = []
        
        # 收集的引用
        self._collected_references: List[Dict[str, Any]] = []
        
        # Token 计数（用于监控）
        self._token_count = 0
        self._iteration_count = 0
        
        # [优化] 工具调用统计，用于失败后切换工具
        self._tool_call_count: Dict[str, int] = {}
        self._tool_failure_count: Dict[str, int] = {}
    
    def execute_with_tools(
        self,
        tools: List[BaseTool],
        context_data: Dict[str, Any] = None
    ) -> SubAgentResult:
        """
        在隔离环境中执行任务
        
        Args:
            tools: 子智能体可用的工具列表
            context_data: 额外的上下文数据（如患者信息）
        
        Returns:
            SubAgentResult: 蒸馏后的结果，包含：
                - success: 是否成功
                - report: 提取的 <report> 标签内容
                - references: 收集的结构化引用
                - raw_token_count: 子智能体消耗的 token 数
                - iterations: 执行轮数
                - error: 错误信息（如果失败）
        
        Version Compatibility:
            v5.2: 新增 context_data 参数支持
            v5.1: 优化错误处理和工具切换逻辑
        """
        try:
            # 1. 初始化沙箱消息
            self._init_sandbox(context_data)
            
            # 2. 执行循环（工具调用 -> 结果 -> 继续/结束）
            final_response = self._run_agent_loop(tools)
            
            # 3. 蒸馏结果（提取 <report> 标签内容）
            report = self._distill_report(final_response)
            
            if self.show_thinking:
                print(f"📦 [SubAgent] 任务完成，蒸馏报告长度: {len(report)} 字符")
                print(f"   迭代次数: {self._iteration_count}, 沙箱消息数: {len(self._sandbox_messages)}")
            
            return SubAgentResult(
                success=True,
                report=report,
                references=self._collected_references,
                raw_token_count=self._token_count,
                iterations=self._iteration_count
            )
            
        except Exception as e:
            if self.show_thinking:
                print(f"❌ [SubAgent] 执行失败: {e}")
            return SubAgentResult(
                success=False,
                report=f"子智能体执行失败: {str(e)}",
                error=str(e),
                iterations=self._iteration_count
            )
        finally:
            # 4. 销毁沙箱历史（这是隔离的关键）
            self._destroy_sandbox()
    
    def _destroy_sandbox(self):
        """销毁沙箱历史（隔离的关键）"""
        self._sandbox_messages = []
        self._collected_references = []
        self._token_count = 0
        self._iteration_count = 0

class SubAgentResult:
    """
    子智能体执行结果（蒸馏后的精华）
    
    Version Compatibility:
        v5.0+: 初始版本
    
    Attributes:
        success (bool): 是否成功执行
        report (str): 提取的 <report> 内容或最终摘要
        references (List[Dict[str, Any]]): 结构化引用列表
        raw_token_count (int): 子智能体消耗的 token 数
        iterations (int): 执行了多少轮
        error (Optional[str]): 错误信息
    
    Examples:
        >>> result = SubAgentResult(
        ...     success=True,
        ...     report="根据指南推荐...",
        ...     references=[{"title": "指南", "url": "..."}],
        ...     raw_token_count=1500,
        ...     iterations=3
        ... )
    """
    
    def __init__(
        self,
        success: bool,
        report: str,
        references: List[Dict[str, Any]] = None,
        raw_token_count: int = 0,
        iterations: int = 0,
        error: str = None
    ):
        self.success = success
        self.report = report
        self.references = references or []
        self.raw_token_count = raw_token_count
        self.iterations = iterations
        self.error = error

**工厂函数：**

```python
def create_rag_researcher(
    model,
    task: str,
    max_iterations: int = 4,
    show_thinking: bool = True
) -> SubAgentContext:
    """创建RAG检索子智能体"""
    system_prompt = """你是一个专业的医学文献检索员。
请使用可用的RAG工具检索相关医学文献，并将结果整理为 <report></report>。"""
    return SubAgentContext(
        model=model,
        system_prompt=system_prompt,
        task_description=task,
        max_iterations=max_iterations,
        show_thinking=show_thinking
    )

def create_web_researcher(
    model,
    task: str,
    max_iterations: int = 4,
    show_thinking: bool = True
) -> SubAgentContext:
    """创建网络研究子智能体"""
    system_prompt = """你是一个专业的医学网络研究员。
请使用网络搜索工具查找最新医学信息，并将结果整理为 <report></report>。"""
    # 实现细节...
    pass
```

**使用示例：**

```python
from sub_agent import (
    SubAgentContext,
    create_rag_researcher,
    run_isolated_rag_search
)

# 方式1：使用工厂函数
agent = create_rag_researcher(
    model=llm,
    task="查找直肠癌T3N1的新辅助治疗方案",
    max_iterations=4,
    show_thinking=True
)
result = agent.execute_with_tools(rag_tools, context_data)
print(result.report)  # 蒸馏后的结果
print(result.references)  # 提取的引用

# 方式2：使用同步辅助函数
result = run_isolated_rag_search(
    model=llm,
    task="查找结肠癌辅助化疗方案",
    tools=rag_tools,
    show_thinking=True
)

# 方式3：自定义子智能体
agent = SubAgentContext(
    model=llm,
    system_prompt="你是一个专业的医学助手...",
    task_description="执行特定任务",
    max_iterations=3,
    show_thinking=True
)
result = agent.execute_with_tools(tools)
```

**配置选项：**

```python
# 子智能体配置
DEFAULT_MAX_ITERATIONS = 4
REPORT_TAG_PATTERN = r'<report>(.*?)</report>'
SUMMARY_TAG_PATTERN = r'<summary>(.*?)</summary>'

# 批量检索配置
BATCH_SEARCH_ENABLED = True
BATCH_QUERY_LIMIT = 5
```

**错误处理：**

- 执行失败时返回错误报告而非抛出异常
- 工具调用失败时尝试备用工具
- 最大迭代次数到达时返回已收集的结果

**性能考虑：**

- 沙箱消息独立管理，不影响主上下文
- 结果蒸馏只返回精华内容
- 自动销毁机制释放内存
- 批量模式减少迭代次数

---

### 14. parallel_subagents.py - 并行子智能体执行节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_parallel_subagents()` | 并行执行节点 |
| `_infer_assignee()` | 推断任务执行者 |
| `_select_tools_for_step()` | 为步骤选择工具 |
| `_run_subagent_for_step()` | 为单个步骤运行子智能体 |

**功能特性：**

- 读取Planner输出的parallel_group + assignee
- 同组并行执行多个子任务
- 汇总子智能体报告与引用

**使用示例：**

```python
from parallel_subagents import node_parallel_subagents

# 创建并行执行节点
parallel_executor = node_parallel_subagents(
    model=llm,
    tools=available_tools,
    show_thinking=True,
    max_workers=4  # 最大并行数
)

# 执行并行任务
result = parallel_executor.invoke(state)
# result.subagent_reports  # 汇总的报告
# result.retrieved_references  # 汇总的引用
```

**配置选项：**

```python
# 并行配置
MAX_WORKERS = 4  # 最大并行数
TIMEOUT_PER_SUBAGENT = 60  # 单个子智能体超时时间（秒）
```

**错误处理：**

- 单个子智能体失败不影响其他任务
- 超时任务自动取消
- 部分失败时返回部分结果

**性能考虑：**

- 使用ThreadPoolExecutor实现并行
- 批量汇总减少后续处理开销
- 超时控制避免长时间阻塞

---

### 15. general_nodes.py - 闲聊/响应综合节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_general_chat()` | 闲聊/引导节点 + 响应综合节点 |
| `_get_recent_conversation_history()` | 提取最近对话历史 |
| `_format_messages_for_summary()` | 将消息格式化为摘要 |

**功能特性：**

- 处理正常闲聊（你好/谢谢）
- 处理偏题输入（off_topic_redirect），温柔引导回医疗主线
- 处理模糊输入，礼貌询问澄清
- 计划完成后，综合所有步骤的结果，生成最终响应

**使用示例：**

```python
from general_nodes import node_general_chat

# 创建闲聊节点
chat_node = node_general_chat(
    model=llm,
    streaming=True,
    show_thinking=True
)

# 执行闲聊/综合
result = chat_node.invoke(state)
```

**配置选项：**

```python
# 对话历史配置
MAX_HISTORY_TURNS = 3  # 参考对话历史轮数
SUMMARY_TRUNCATE_LENGTH = 200  # 消息截断长度

# 偏题引导配置
OFF_TOPIC_PROMPT_TEMPLATE = REDIRECT_USER_PROMPT_TEMPLATE
SYNTHESIS_PROMPT = SYNTHESIS_SYSTEM_PROMPT
```

**错误处理：**

- 综合失败时返回原始结果
- 偏题引导失败时返回通用响应

**性能考虑：**

- 对话历史截断避免上下文膨胀
- 批量综合减少LLM调用

---

### 16. router.py - 路由函数

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `RouteRule` | 路由规则结构 |
| `_is_profile_locked()` | 检查档案是否锁定 |
| `_check_missing_critical_info()` | 检查关键信息是否缺失 |
| `ROUTING_RULES` | 路由规则列表 |
| `route_by_critic_v2()` | 基于迭代次数和审核的路由 |
| `route_after_rad_agent()` | 影像分析后路由 |
| `route_after_evaluator()` | 评估后路由 |

**路由优先级：**

| 优先级 | 规则名称 | 条件 | 目标节点 |
|--------|----------|------|----------|
| 90 | critical_info_missing | 关键信息缺失 | assessment |
| 80 | needs_planning | 需要规划 | assessment |
| 55 | imaging_analysis | 影像分析请求 | rad_agent |
| 55 | pathology_analysis | 病理分析请求 | path_agent |
| 50 | general_chat | 闲聊请求 | general_chat |
| 50 | knowledge_query | 知识查询 | knowledge |
| 50 | case_database_query | 数据库查询 | case_database |
| 50 | treatment_decision | 治疗决策 | decision |

**使用示例：**

```python
from router import (
    ROUTING_RULES,
    route_by_intent,
    route_after_evaluator
)

# 获取下一个节点
next_node = route_by_intent(state)

# 评估后路由
next_node = route_after_evaluator(state)
```

**配置选项：**

```python
# 路由配置
DEFAULT_TARGET = "assessment"  # 默认路由目标
PRIORITY_THRESHOLDS = [90, 80, 55, 50]  # 优先级阈值
```

---

### 17. policy.py - 策略决策节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `build_policy_node()` | 创建策略节点 |

**使用示例：**

```python
from policy import build_policy_node

# 创建策略节点
policy = build_policy_node(model=llm, tools=available_tools)
result = policy.invoke(state)
```

---

### 18. error_handler.py - 错误处理节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `handle_error()` | 错误处理函数 |

**使用示例：**

```python
from error_handler import handle_error

# 错误恢复
result = handle_error(state)
# 返回恢复后的消息
```

---

### 19. tools_executor.py - 通用工具执行节点

**包含内容：**

| 函数/类 | 描述 |
|---------|------|
| `node_tool_executor()` | 通用工具执行器 |
| `_format_tool_output()` | 格式化工具输出 |

**功能特性：**

- 监听并执行工具调用
- 统一格式化输出（UI卡片）
- 将结果写回State

**使用示例：**

```python
from tools_executor import node_tool_executor

# 创建工具执行器
executor = node_tool_executor
result = executor.invoke(state)
```

**配置选项：**

```python
# 工具执行配置
ENABLE_OUTPUT_FORMATTING = True
CARD_FORMATTERS = {
    "tumor_detection_card": formatter.format_tumor_screening_result,
    "radiology_report_card": formatter.format_radiomics_report,
    # ...
}
```

---

## 版本兼容性

| 节点模块 | 最低Python版本 | 最低LangChain版本 | 依赖包 |
|----------|---------------|-------------------|--------|
| node_utils.py | 3.10 | 0.2.x | pydantic |
| intent_nodes.py | 3.10 | 0.2.x | pydantic |
| planner.py | 3.10 | 0.2.x | - |
| knowledge_nodes.py | 3.10 | 0.2.x | - |
| assessment_nodes.py | 3.10 | 0.2.x | pydantic |
| staging_nodes.py | 3.10 | 0.2.x | - |
| radiology_nodes.py | 3.10 | 0.2.x | - |
| pathology_nodes.py | 3.10 | 0.2.x | - |
| decision_nodes.py | 3.10 | 0.2.x | pydantic |
| sub_agent.py | 3.10 | 0.2.x | - |
| parallel_subagents.py | 3.10 | 0.2.x | - |

---

## 配置示例

```python
# 节点配置
NODE_CONFIG = {
    "intent_classifier": {
        "temperature": 0.0,
        "summary_trigger_turns": 8,
        "max_history_turns": 3,
    },
    "planner": {
        "max_iterations": 5,
        "max_step_retries": 3,
        "circuit_breaker_threshold": 3,
    },
    "decision": {
        "max_iterations": 4,
        "batch_search_mode": True,
        "top_k_results": 12,
    },
    "sub_agent": {
        "max_iterations": 4,
        "batch_search_enabled": True,
        "report_pattern": r'<report>(.*?)</report>',
    },
    "radiology": {
        "default_mode": "detection",
        "supported_modes": ["detection", "segmentation", "radiomics", "comprehensive"],
    },
    "pathology": {
        "default_mode": "full",
        "supported_modes": ["full", "quick", "status"],
    },
}

# 环境变量配置
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

---

## 错误处理指南

### 常见错误类型

| 错误类型 | 处理策略 | 示例 |
|----------|----------|------|
| JSON解析失败 | Fail-Open，返回默认结构 | decision_nodes.py |
| LLM调用失败 | 重试机制，熔断保护 | sub_agent.py |
| 工具调用失败 | 备用工具，错误报告 | tools_executor.py |
| 路由失败 | 默认路由，兜底方案 | router.py |
| 评估失败 | 默认通过，继续流程 | evaluation_nodes.py |

### 错误恢复模式

```python
# 1. Fail-Open 模式
try:
    result = llm.invoke(...)
except Exception:
    return default_result  # 使用默认值继续

# 2. 重试机制
for attempt in range(max_retries):
    try:
        result = llm.invoke(...)
        break
    except Exception:
        if attempt == max_retries - 1:
            raise
        continue

# 3. 熔断保护
if failure_count >= CIRCUIT_BREAKER_THRESHOLD:
    return fallback_plan
```

---

## 性能优化指南

### 上下文管理

```python
# 1. Token估算与截断
tokens = _estimate_tokens(message_history)
if tokens > MAX_TOKENS:
    truncated = _truncate_message_history(message_history)

# 2. 摘要记忆
summary = _maybe_update_summary_memory(state, model)

# 3. 压缩RAG上下文
compressed = _compress_rag_context(retrieved_docs)
```

### 并行处理

```python
# 使用并行子智能体执行
from parallel_subagents import node_parallel_subagents

parallel_node = node_parallel_subagents(
    model=llm,
    tools=tools,
    max_workers=4
)
```

### 缓存策略

```python
# 复用已有结果
if findings.get("radiology_report"):
    return findings  # 跳过重复分析
```

---

## 测试指南

### 单元测试示例

```python
import pytest
from intent_nodes import node_intent_classifier

def test_intent_classification():
    state = create_test_state("直肠癌怎么治?")
    classifier = node_intent_classifier(model=mock_llm)
    result = classifier.invoke(state)
    assert result.findings["user_intent"] == "treatment_decision"

def test_tnm_validation():
    findings = {"tnm_staging": {"cT": "cT3", "cN": "cN1", "cM": "cM0"}}
    is_valid, _ = _validate_tnm_consistency(findings, "直肠癌T3N1M0")
    assert is_valid == True
```

### 集成测试示例

```python
def test_full_pipeline():
    # 模拟完整流程
    state = initial_state
    state = intent_classifier.invoke(state)
    state = planner.invoke(state)
    state = database_query.invoke(state)
    # ...
    assert state.final_response is not None
```

---

## 贡献指南

### 添加新节点

1. 在`src/nodes/`下创建新文件
2. 实现节点函数，遵循以下签名：

```python
def node_your_node(
    model,
    tools: List[BaseTool] = None,
    streaming: bool = False,
    show_thinking: bool = True
) -> Runnable:
    """节点描述"""
    # 实现...
    pass
```

3. 在`__init__.py`中导出

### 节点命名规范

- 文件名：`{功能}_nodes.py`
- 函数名：`node_{功能}()`
- 类名：`{功能}Schema` / `{功能}Result`

---

## 许可证

本模块遵循项目主许可证。
