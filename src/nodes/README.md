# Agent 节点模块

本目录包含 LangG 系统的所有 LangGraph Agent 节点实现，覆盖意图分类、规划、知识检索、临床评估、诊断分期、治疗决策、质量审查到记忆管理的完整流程。

## 文件结构

```
src/nodes/
├── __init__.py              # 包入口
├── node_utils.py            # 共享工具库（流式处理/JSON解析/工具执行/RAG负载提取）
├── intent_nodes.py          # 意图分类节点（10种意图 + 多任务 + 拼写纠正）
├── planner.py               # 自主规划节点（PlanStep DAG + 自修正 + 上下文诊断）
├── knowledge_nodes.py       # 知识检索节点（计划驱动 + 层次化搜索 + 子代理隔离）
├── knowledge_utils.py       # 知识检索辅助（患者状态描述注入）
├── database_nodes.py        # 病例数据库查询节点（LLM工具调用编排）
├── radiology_nodes.py       # 影像分析节点（YOLO → U-Net → PyRadiomics → LASSO）
├── pathology_nodes.py       # 病理分析节点（CLAM全切片分类）
├── assessment_nodes.py      # 临床评估 + 诊断提取节点（Fast Pass + 语义守卫）
├── staging_nodes.py         # TNM分期节点（结肠/直肠，Fast Pass快速校验）
├── decision_nodes.py        # 治疗决策 + 批判审查节点（模板/检索/子代理三模式）
├── citation_nodes.py        # 引用校验节点（覆盖率评分 + 缺失声明检测）
├── evaluation_nodes.py      # LLM-Judge质量评估节点（四维评分）
├── general_nodes.py         # 通用对话 + 回复合成节点
├── clinical_entry_nodes.py  # 临床入口路由（门诊分诊 vs CRC评估）
├── triage_nodes.py          # 门诊分诊问答节点
├── chat_main_node.py        # 患者访谈对话节点
├── memory_nodes.py          # 记忆管理节点（分层摘要 + 令牌预算压缩）
├── router.py                # 策略驱动路由桥接
├── policy.py                # 通用策略节点
├── parallel_subagents.py    # 并行子代理执行节点
├── sub_agent.py             # 子代理上下文隔离框架（沙箱执行）
├── tools_executor.py        # 通用工具执行节点
├── error_handler.py         # 错误恢复节点
└── clinical_nodes.py        # 聚合导出模块（向后兼容）
```

## 架构分层

```
意图理解层:  intent_nodes → planner
信息收集层:  database_nodes / radiology_nodes / pathology_nodes / knowledge_nodes / web_search
临床评估层:  clinical_entry_nodes → triage_nodes / assessment_nodes → diagnosis → staging_nodes
决策支持层:  decision_nodes → critic → citation → evaluator
输出层:      general_nodes / chat_main_node
基础设施层:  memory_nodes / tools_executor / error_handler / sub_agent / parallel_subagents
```

## 节点详细说明

### intent_nodes.py — 意图分类

将用户输入分类为 10 种意图之一：`imaging_analysis`、`pathology_analysis`、`imaging_query`、`case_database_query`、`clinical_assessment`、`treatment_decision`、`knowledge_query`、`general_chat`、`off_topic_redirect`、`multi_task`。

- `node_intent_classifier(model, ...)` — 图节点，LLM 结构化输出分类意图，支持快速通道（空输入/问候/感谢直接返回 `general_chat`）
- `route_by_intent(state)` — 路由函数，将意图映射到下游节点
- `IntentDecision` — Pydantic 模型，包含 category、sub_tasks、requires_context、correction_suggestion、reasoning

### planner.py — 自主规划

将用户意图分解为可执行的 `PlanStep` DAG，支持自修正和上下文诊断。

- `node_planner(model, ...)` — 图节点，三层跳过检查（快速跳过 → 缺失上下文 → LLM规划），最多5次迭代
- `get_current_pending_step(state)` / `mark_step_completed()` / `mark_step_failed()` — 计划步骤管理
- `_detect_missing_context(state)` — 核心逻辑：根据意图类型识别缺失的必要上下文
- `PlanStep` — 支持 id、tool_needed、status、assignee、parallel_group、parent_step_id、branch_id、state_hash

### knowledge_nodes.py — 知识检索

双模式知识检索：计划驱动模式（执行 Planner 生成的步骤）和自动检索模式（本地 RAG → 充分性评估 → 网络搜索）。

- `node_knowledge_retrieval(model, tools, ...)` — 图节点，支持层次化搜索（Authority/Evidence/Safety/Fallback），非标准治疗问题使用多层搜索
- `node_web_search_agent(tools)` — 独立网络搜索节点
- `_create_search_planner(model)` / `_create_sufficiency_evaluator(model)` — 搜索策略规划与充分性评估
- `SearchPlan` / `KnowledgeSufficiencyEval` — Pydantic 模型

### database_nodes.py — 病例数据库查询

纯 LLM 工具调用节点，提供患者信息、影像、病理切片、统计等多维度查询。

- `node_case_database(model, tools, ...)` — 图节点，确定型路径匹配常见查询模式，兜底 LLM 工具调用
- `_format_case_summary_markdown(case_data)` / `_format_case_brief(case_data)` — 病例摘要格式化
- `_normalize_case_database_patient_id(value)` — 患者ID补零规范化

### radiology_nodes.py — 影像分析

编排影像 AI 工具链，支持四种分析模式。

- `node_rad_agent(tools, model, ...)` — 图节点，根据用户文本检测分析模式并路由
- `_run_detection_analysis()` / `_run_segmentation_analysis()` / `_run_radiomics_analysis()` / `_run_comprehensive_analysis()` — 四种模式执行器
- 已有报告时自动复用，避免重复调用 AI 工具

### pathology_nodes.py — 病理分析

调度 CLAM 病理全切片分析工具链。

- `node_pathology_agent(tools, model, ...)` — 图节点，支持 full/quick/status 三种模式
- 通过患者 ID 自动查找切片文件，支持多切片综合分析
- `_build_pathology_card_base()` — 构造病理卡片基础字段

### assessment_nodes.py — 临床评估与诊断

系统中最复杂的节点之一，负责病例完整性检查、临床评估和诊断提取。

- `check_case_integrity(user_text, model, ...)` — 图节点（语义守卫），LLM 结构化输出判断病例完整性，支持快速规则兜底
- `node_assessment(model, tools, ...)` — 图节点，多分支策略（纯症状/缺失信息追问/MMR追问/死循环防护/Fast Pass/完整评估）
- `node_doctor_assessment()` / `node_patient_assessment()` — 场景特化入口
- `node_diagnosis(model, tools, ...)` — 诊断提取，结构化输出
- `node_staging_router(state)` — 分期路由：rectal_staging / colon_staging / decision
- `ClinicalAssessmentResult` / `DiagnosisExtractionResult` / `CaseIntegrity` — Pydantic 模型

### staging_nodes.py — TNM 分期

- `node_colon_staging(tools)` — 结肠癌分期：Fast Pass 快速 TNM 校验 + CT M 分期评估
- `node_rectal_staging(tools)` — 直肠癌分期：Fast Pass + MRI 局部分期 + CT 远处分期 + M 分期自校正
- `_validate_tnm_consistency(findings, user_text)` — TNM 组合医学有效性校验

### decision_nodes.py — 治疗决策与审查

治疗决策的核心引擎，支持三种检索模式。

- `node_decision(model, tools, ...)` — 图节点，三模式：模板快速通道（无RAG）/ 子代理隔离RAG检索 / 直接RAG检索
- `node_critic(model, ...)` — 批判审查节点，APPROVED / REJECTED / APPROVED_WITH_WARNINGS
- `TreatmentAction` / `ClinicalDecisionSchema` / `CriticEvaluationSchema` — Pydantic 模型
- `_build_template_decision()` / `_build_template_decision_v2()` — 基于分期的模板化治疗方案

### citation_nodes.py — 引用校验

- `node_citation_agent(model, ...)` — 图节点，验证治疗决策的引用覆盖率，支持快速启发式绕过
- `CitationReport` — coverage_score (0-100)、missing_claims、needs_more_sources

### evaluation_nodes.py — 质量评估

- `node_llm_judge(model, ...)` — 图节点，四维评分（factual_accuracy / citation_accuracy / completeness / safety，各1-5分），支持快速通道
- `route_after_evaluator(state)` — 路由：decision（重试）或 finalize
- `EvaluationReport` — PASS/FAIL 判定 + 各维度分数 + 反馈

### general_nodes.py — 通用对话

- `node_general_chat(model, ...)` — 图节点，多模式：快速回复（问候/感谢/再见）、计划跟进、计划完成综合、纯信息展示、简单事实问答、偏题重定向

### clinical_entry_nodes.py — 临床入口路由

重新导出 `triage_nodes.py` 的节点用于临床入口路径判断。

- `node_clinical_entry_resolver` / `node_outpatient_triage` / `route_after_clinical_entry` / `route_after_outpatient_triage`

### triage_nodes.py — 门诊分诊

交互式门诊分诊，采集症状焦点、持续时间、便血、排便习惯改变、体重下降、发热等信息。

- `node_outpatient_triage(model, ...)` — 图节点，顺序问答，检测停滞并提供切换提示
- `node_clinical_entry_resolver(model, ...)` — 入口节点，判断门诊分诊 vs CRC 评估
- `_triage_from_symptoms(text)` — 规则分诊：risk_level + disposition + suggested_tests
- `TRIAGE_QUESTION_SCHEMAS` / `TRIAGE_QUESTION_MAP` — 分诊问题定义

### chat_main_node.py — 患者访谈

交互式患者对话，结构化字段采集（性别、年龄、ECOG、组织学、肿瘤位置、T/N分期、CEA、MMR），每次采集自动写入数据库。

- `node_chat_main(model, tools, ...)` — 图节点，完整访谈循环
- `CHAT_MAIN_TOOLS` / `CHAT_MAIN_SYSTEM_PROMPT` — 工具列表与系统提示

### memory_nodes.py — 记忆管理

分层结构化摘要（immutable_info / dynamic_info / anchor_events），令牌预算压缩，关键事件锚定。

- `node_memory_manager(model, ...)` — 图节点，处理令牌预算压缩、增量摘要更新、分层摘要维护
- `compress_context_by_token(messages, max_tokens, ...)` — 保留系统消息和工具调用链的智能压缩
- `incremental_summary()` / `update_layered_summary()` — 摘要更新

### router.py — 策略路由

- `route_after_intent(state)` / `dynamic_router(state)` / `route_after_assessment(state)` / `route_after_clinical_entry(state)` — 路由函数，委托给 policies 模块

### sub_agent.py — 子代理隔离框架

- `SubAgentContext` — 沙箱执行环境，独立消息历史、结果蒸馏（<report>标签）、自动销毁
- `SubAgentResult` — 包含 success、report、references、token_count、iterations、error
- `create_rag_researcher(model, task, ...)` / `create_web_researcher(model, task, ...)` — 工厂函数
- `run_isolated_rag_search()` / `run_isolated_web_search()` — 同步辅助函数

### parallel_subagents.py — 并行子代理

- `node_parallel_subagents(model, tools, ...)` — 图节点，asyncio.gather 并行执行同组步骤，汇总报告和引用

### tools_executor.py — 工具执行

- `node_tool_executor(state)` — 图节点，执行 AIMessage 中的 tool_calls，格式化输出
- `_format_tool_output(tool_name, output, args)` — 集中式格式化适配器

### error_handler.py — 错误恢复

- `handle_error(state)` — 图节点，记录错误并生成优雅恢复消息

### policy.py — 策略节点

- `build_policy_node(model, tools)` — 图节点工厂，创建通用 LLM 工具调用策略节点

### node_utils.py — 共享工具库

约 40+ 个导出函数，涵盖以下类别：
- JSON 处理：`_clean_json_string`、`_clean_and_validate_json`、`_extract_first_json_object`、`_unwrap_nested_json`
- 流式处理：`_invoke_with_streaming`、`_parse_thinking_tags`、`_split_inline_thinking`、`_extract_thinking_from_chunk`
- 工具执行：`_execute_tool_calls`、`_execute_tool_calls_robust`、`_select_tools`
- 上下文管理：`_truncate_message_history`、`_build_pinned_context`、`_build_summary_memory`
- 状态提取：`_latest_user_text`、`_extract_text_content`、`_extract_rag_payload`
- 结构化恢复：`_invoke_structured_with_recovery`（直接 → 原始JSON → 文本解析 → 兜底工厂）
- 临床文本提取：`_extract_ct_text`、`_extract_mri_text`、`_extract_pathology_text`

### knowledge_utils.py — 知识检索辅助

- `_get_patient_state_description(state)` — 构建患者临床状态的结构化摘要，注入知识查询

### clinical_nodes.py — 聚合导出

向后兼容的聚合模块，从各子模块重新导出所有公共接口（50+ 符号）。
