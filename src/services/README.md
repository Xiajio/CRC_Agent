# Services 模块

本目录包含 LangG 系统的核心服务层，为 Agent 节点提供 LLM 调用、文档转换、网络搜索、病例数据访问和患者卡片投射等基础能力。

## 文件结构

```
src/services/
├── __init__.py                 # 模块入口
├── llm_service.py              # LLM 服务（模型创建 + Thinking 模式 + 提供者适配）
├── web_search_service.py       # 网络搜索服务 + Deep Research 服务
├── document_converter.py       # 文档→医疗卡片转换（含 PII 脱敏）
├── case_excel_service.py       # 病例 Excel 读写服务
├── virtual_database_service.py # 虚拟病例数据库服务
├── patient_card_projector.py   # 患者卡片多源投射（含冲突检测）
├── provider_capabilities.py    # LLM 提供者能力检测
└── local_hf_chat.py            # 本地 HuggingFace / vLLM 对话模型
```

## 组件说明

### llm_service.py — LLM 服务

统一的 LLM 模型创建和调用抽象。

- `LLMService` — 根据 `LLMSettings` 创建模型：API 模式（`create_compatible_chat_openai`）或 Local 模式（`LocalHFChatModel` / `LocalVLLMChatModel`）
- `ThinkingChatOpenAI(ChatOpenAI)` — 添加 Thinking/Reasoning 模式支持（DeepSeek-R1 / Qwen-QwQ），注入 `extra_body` 参数（enable_thinking、thinking_budget），兼容流式、JSON 模式
- `create_compatible_chat_openai()` — 工厂函数，处理字符串响应、系统消息注入（不支持的提供者自动合并到首条用户消息）、Thinking 模式集成

### web_search_service.py — 网络搜索与 Deep Research

- `WebSearchService` — 基于 `gpt-4o-search-preview` 的实时网络搜索，含结果验证（"未找到"检测 + 最小长度检查）
  - 专项搜索：`search_clinical_evidence()`、`search_drug_info()`、`search_latest_guidelines()`、`search_research()`
- `DeepResearchService(WebSearchService)` — Agentic "计划-执行-综合" 循环
  - `_decompose_query()` — 将复杂查询分解为 3-5 子查询
  - `search_deep()` — 分解 → 并行搜索（ThreadPoolExecutor）→ 去重 → LLM 综合
  - `ResearchResult` — summary（含内联引用）+ sources + missing_info + sub_queries_used
  - 来源可信度评分（NCCN=9、PubMed=8、NEJM=10 等）
  - 来源黑名单过滤（百度知道、知乎、博客等）
- 全局单例管理：`_WebSearchServiceManager`（线程安全，双检锁）

### document_converter.py — 文档转换器

将 PDF/Word/图片/文本医疗文档转换为结构化 `MedicalVisualizationCard` JSON。

- `DocumentConverter` — 混合解析（PyMuPDF 文本 + Vision LLM OCR），长文档分段处理（ThreadPoolExecutor 并行），LLM 智能合并
- `MedicalVisualizationCard` / `CardData` / `PatientSummary` / `DiagnosisBlock` / `StagingBlock` / `KeyFinding` / `TreatmentStep` — Pydantic 输出模式
- OCR 错误自动纠正（validator：TI → T1）
- PII 脱敏：`_scrub_pii()` 移除手机号、身份证、电话号
- 转换失败时生成完整兜底卡片
- 接地验证：诊断和分期字段检查 `evidence_quote`
- 便捷函数：`create_converter()`、`convert_uploaded_file()`

### case_excel_service.py — 病例 Excel 服务

- `upsert_case_record(excel_path, data)` — 标准化并 upsert 病例记录到 Excel（openpyxl），自动创建文件
- `load_case_records(excel_path)` — 读取全部行，类型强制转换
- `find_case_record(excel_path, patient_id)` — 单记录查询
- `normalize_case_payload(data)` — 输入数据校验和标准化
- `PREFERRED_CASE_HEADERS` / `CASE_FIELD_ALIASES` — 规范字段名 + 多别名映射
- 类型强制：`_coerce_int`、`_coerce_float`、`_coerce_bool`、`_coerce_gender`、`_coerce_ecog`、`_coerce_mmr_status`、`_normalize_t_stage`、`_normalize_n_stage`

### virtual_database_service.py — 虚拟病例数据库

- `VirtualCaseDatabase` — 从 `classification.xlsx` 加载的内存数据库
  - `get_all_cases()`、`get_case_by_id()`、`get_random_case()`、`get_statistics()`
- `query_cases()` — 多条件过滤（部位/分期/组织学/MMR/年龄/CEA）
- `get_imaging_by_patient_id()` / `get_pathology_slides_by_patient_id()` — 影像和病理资源查询
- 单例：`get_case_database()`

### patient_card_projector.py — 患者卡片投射

多源数据合并，统一患者卡片视图。

- `project_patient_card()` — 从四个来源合并字段（patient_profile、findings、symptom_snapshot、medical_card），按优先级仲裁，追踪每字段的分辨状态（confirmed / conflict / pending），计算 completion_ratio
- `project_patient_self_report_card(state)` — 便捷包装，从代理状态提取子字典

### provider_capabilities.py — 提供者能力检测

- `ProviderCapabilities` — 冻结数据类（provider、supports_system_messages、structured_output_strategy、thinking_transport、supports_thinking）
- 预定义配置：`openai`、`openai_compatible`、`minimax`（无系统消息、raw-first 结构化输出）、`deepseek`、`qwen`
- `resolve_provider_name()` — 基于模型名和 base URL 的启发式检测
- `resolve_provider_capabilities()` — 返回匹配的能力配置

### local_hf_chat.py — 本地模型支持

- `LocalHFChatModel(BaseChatModel)` — 包装本地 HuggingFace 因果 LM 为 LangChain 聊天模型
  - 支持 `tokenizer.apply_chat_template`、流式（TextIteratorStreamer）、Flash Attention 2 自动兜底、简洁模式
- `LocalVLLMChatModel(BaseChatModel)` — vLLM 后端（FP8、张量并行、GPU 内存利用），流式生成
- `LocalHFChatModelWithTools(BaseChatModel)` — 扩展工具调用能力：从输出中解析 JSON 工具调用（平衡括号 + 代码块提取），校验工具名和必需参数

## 与 Agent 节点的集成

- **LLM 服务** — 为所有节点提供模型实例（结构化输出、流式输出、Thinking 模式）
- **文档转换器** — 为上传管线提供文档→结构化医疗卡片转换
- **网络搜索** — 为 `knowledge_nodes` 和 `decision_nodes` 提供实时搜索与 Deep Research
- **病例 Excel** — 为 `database_nodes` 和 `chat_main_node` 提供病例数据读写
- **虚拟数据库** — 为数据库工具提供内存查询
- **患者卡片投射** — 为 `assessment_nodes` 和前端快照提供多源合并的患者卡片
- **提供者能力** — 为 LLM 服务提供自动适配（系统消息注入、Thinking 传输方式）
