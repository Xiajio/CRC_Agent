# LangG —— 结直肠癌智能临床决策支持系统

基于 **LangGraph** 的多智能体临床决策支持系统，面向**结直肠癌（CRC）**的智能诊疗全流程。提供患者分诊与医生临床决策辅助两大场景，集成 RAG 知识检索、医学影像 AI 分析和历史病例数据库。

## 核心功能

| 场景 | 功能 |
|------|------|
| **患者端** | 智能分诊问答（门诊分诊）、症状采集、病历资料上传、身份登记、自报告卡片生成 |
| **医生端** | 意图分类 → 规划 → 知识检索 / 影像分析 / 病理分析 / 病例查询 → 临床评估 → 诊断 → TNM 分期 → 治疗决策 → 批判审查 → 证据引用 → 质量评估 → 记忆管理 |
| **知识检索** | 混合 RAG 引擎：Chroma 向量检索 + BM25 关键词检索（jieba 分词） + Cross-Encoder / Cohere / LLM 重排序 |
| **影像 AI** | YOLOv8 肿瘤检测、U-Net 肿瘤分割、PyRadiomics 影像组学特征提取 |
| **病理 AI** | CLAM 病理切片分类（WSI 全切片图像）、热力图生成 |
| **病例库** | 历史病例 Excel 数据库 + SQLite 患者登记处（事件溯源），支持结构化筛选与自然语言查询 |
| **网络安全搜索** | Deep Research 服务：查询分解 → 并行搜索 → 去重 → LLM 综合，带来源可信度评分 |

## 技术架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Frontend (React 18 + TypeScript + Vite + TailwindCSS)             │
│  pages/workspace-page  ·  features/chat/cards/doctor/database/   │
│  features/patient-identity/registry/roadmap/execution-plan/      │
│  app/store (stream-reducer)  ·  app/api (SSE + latency trace)    │
│  features/workspace/ (6 hooks: sessions/streaming/latency/cards/  │
│                       uploads/nav)                                │
└────────────────────────┬─────────────────────────────────────────┘
                         │ SSE streaming (18 种事件类型)
┌────────────────────────▼─────────────────────────────────────────┐
│  Backend BFF (FastAPI + Uvicorn)                                  │
│  routes/ sessions · chat · database · patient-registry ·         │
│          uploads · assets   (21 个 REST 端点)                     │
│  services/ graph_service · graph_factory · session_store ·       │
│            upload_service · patient_registry_service ·           │
│            patient_commands · patient_context_resolver ·         │
│            database_service · database_intent_service ·           │
│            context_maintenance · payload_builder ·                │
│            chat_latency_trace · asset_service · settings          │
│  adapters/ state_snapshot · event_normalizer · card_extractor ·  │
│            card_payload_sanitizer · reference_normalizer ·        │
│            message_content                                        │
│  schemas/ events (18 种) · responses · database · patient_registry │
└────────────────────────┬─────────────────────────────────────────┘
                         │ SceneGraphRouter: patient_graph / doctor_graph
┌────────────────────────▼─────────────────────────────────────────┐
│  Agent Core (LangGraph)                                           │
│  graph_builder.py → build_patient_graph() / build_doctor_graph() │
│  state.py (CRCAgentState: 57 个字段, PlanStep DAG, 证据链追溯)   │
│  nodes/ (27 个 Python 模块, 23 个 NodeName)  ·  policies/  ·  prompts/ │
│  rag/ (Chroma + BM25 + 混合检索 + 重排序)  ·  tools/            │
│  services/ (LLM服务/网络搜索/文档转换/病例Excel/患者卡片投射)    │
└──────────────────────────────────────────────────────────────────┘
```

### 医生端 Agent 工作流（23 个节点，医生图使用其中 20 个）

```
INTENT → PLANNER → KNOWLEDGE / CASE_DATABASE / RAD_AGENT / PATH_AGENT
       / WEB_SEARCH / TOOL_EXECUTOR / PARALLEL_SUBAGENTS
       → ASSESSMENT → DIAGNOSIS → STAGING (colon/rectal)
       → DECISION → CRITIC → CITATION → EVALUATOR → FINALIZE
       ⇄ CHAT_MAIN / GENERAL_CHAT / CLINICAL_ENTRY_RESOLVER / OUTPATIENT_TRIAGE / MEMORY
```

路由由 Planner 生成的 `PlanStep` DAG 动态驱动，支持并行子代理执行、失败自修正循环（Decision ↔ Critic / Decision ↔ Evaluator）。

### 患者端 Agent 工作流（9 个节点）

```
INTENT → PLANNER → CLINICAL_ENTRY_RESOLVER → OUTPATIENT_TRIAGE / ASSESSMENT
       → KNOWLEDGE / CHAT_MAIN → GENERAL_CHAT
```

门诊分诊支持交互式追问卡片（`triage_question_card`），生成风险等级 + 处置建议 + 建议检查。

## 目录结构

```
LangG/
├── src/                          # Python 核心：LangGraph Agent
│   ├── graph_builder.py          # 双图构建（doctor / patient）+ 路由函数 + 节点计时
│   ├── state.py                  # CRCAgentState（57 字段）+ PlanStep + Reducer（含 merge_node_timings）
│   ├── config.py                 # 分层配置（LLM / RAG / 文档转换 / 网络搜索 / 检查点 / 可观测性）
│   ├── checkpoint.py             # 检查点工厂（Memory / SQLite / Postgres / Redis）
│   ├── observability.py          # LangSmith 追踪集成
│   ├── nodes/                    # Agent 节点实现（27 个 Python 文件）
│   │   ├── intent_nodes.py       # 意图分类（10 种意图 + 多任务）
│   │   ├── planner.py            # 任务分解 → PlanStep DAG（含自修正、上下文诊断）
│   │   ├── knowledge_nodes.py    # 知识检索（层次化搜索：权威/证据/安全/兜底）
│   │   ├── database_nodes.py     # 病例数据库查询（含工具调用编排）
│   │   ├── radiology_nodes.py    # 影像分析（YOLO → U-Net → PyRadiomics）
│   │   ├── pathology_nodes.py    # 病理分析（CLAM 全切片分类）
│   │   ├── assessment_nodes.py   # 临床评估 + 诊断提取（语义守卫 + 快速通道）
│   │   ├── staging_nodes.py      # TNM 分期（结肠/直肠）
│   │   ├── decision_nodes.py     # 治疗决策（模板快速通道 / RAG 检索）+ 批判审查 + 最终输出
│   │   ├── citation_nodes.py     # 引用验证（覆盖率评分 + 缺失声明检测）
│   │   ├── evaluation_nodes.py   # LLM-Judge 质量评估（四维评分）
│   │   ├── general_nodes.py      # 通用对话 + 回复合成
│   │   ├── clinical_entry_nodes.py # 临床入口路由
│   │   ├── triage_nodes.py       # 门诊分诊问答 + 临床入口解析
│   │   ├── chat_main_node.py     # 患者访谈对话（结构化字段采集）
│   │   ├── memory_nodes.py       # 记忆管理（分层摘要 + 令牌预算压缩）
│   │   ├── patient_identity.py   # 患者身份解析（registry_patient_id / case_database_patient_id）
│   │   ├── router.py             # 策略驱动路由桥接
│   │   ├── policy.py             # 通用策略节点
│   │   ├── parallel_subagents.py # 并行子代理执行
│   │   ├── sub_agent.py          # 子代理上下文隔离框架
│   │   ├── tools_executor.py     # 通用工具执行节点
│   │   ├── error_handler.py      # 错误恢复
│   │   ├── node_utils.py         # 共享工具库（流式/JSON/工具执行/RAG 负载）
│   │   └── knowledge_utils.py    # 患者状态描述注入
│   ├── policies/                 # 路由与审查策略（8 个文件）
│   │   ├── routing_policy.py     # 路由决策（after_intent / dynamic / after_assessment）
│   │   ├── review_policy.py      # 审查决策（after_critic / after_evaluator）
│   │   ├── turn_facts.py         # TurnFacts 提取与路由标志推导
│   │   ├── tool_targets.py       # 步骤→节点目标映射
│   │   ├── diagnostics.py        # 策略一致性诊断
│   │   ├── constants.py          # 策略常量
│   │   └── types.py              # 策略类型定义
│   ├── prompts/                  # LLM 提示词模板
│   │   ├── intent_prompts.py     # 意图分类
│   │   ├── planner_prompts.py    # 计划生成 + 自修正
│   │   ├── knowledge_prompts.py  # 检索规划 + 充分性评估 + 知识综合
│   │   ├── assessment_prompts.py # 病例完整性守卫 + 评估 + 诊断
│   │   ├── decision_prompts.py   # 治疗决策 + 批判审查 + 查询生成
│   │   ├── evaluation_prompts.py # 引用检查 + LLM-Judge
│   │   ├── database_prompts.py   # 数据库查询
│   │   └── general_prompts.py    # 通用对话 / 综合 / 信息展示
│   ├── rag/                      # RAG 引擎
│   │   ├── parser.py             # 文档解析（混合：文本提取 + Vision OCR）
│   │   ├── ingest.py             # 指南摄取管线（Chroma + BM25）
│   │   ├── retriever.py          # 混合检索器（向量 + BM25 融合 + 重排序 + 全局单例）
│   │   ├── bm25_index.py         # BM25 关键词索引（jieba 中文分词 + 持久化）
│   │   ├── reranker.py           # 重排序器（Cross-Encoder / Cohere / LLM）
│   │   └── evidence.py           # 证据规范化（序列化/去重/溯源）
│   ├── tools/                    # LangChain 工具集
│   │   ├── rag_tools.py          # 指南检索工具
│   │   ├── clinical_tools.py     # 临床数据提取
│   │   ├── database_tools.py     # 病例库 CRUD + 搜索
│   │   ├── pathology_clam_tools.py # CLAM 病理分类工具
│   │   ├── radiomics_tools.py    # 影像组学工具（U-Net + PyRadiomics）
│   │   ├── tumor_screening_tools.py  # YOLOv8 肿瘤检测
│   │   ├── tumor_localization_tools.py # U-Net 肿瘤分割
│   │   ├── web_search_tools.py   # 网络搜索工具
│   │   ├── card_formatter.py     # 卡片格式化（11 种卡片类型）
│   │   ├── basic_tools.py        # 基础工具
│   │   └── tool/                 # 第三方 AI 模型文件
│   │       ├── Tumor_Detection/  # YOLOv8 肿瘤检测脚本
│   │       ├── Tumor_Localization/ # U-Net 肿瘤分割脚本
│   │       └── Pathological_Slide_Classification/CLAM_Tool/  # CLAM 病理分类
│   ├── services/                 # 核心服务
│   │   ├── llm_service.py        # LLM 服务（含 Thinking 模式 + 提供者兼容适配）
│   │   ├── web_search_service.py # 网络搜索 + Deep Research 服务
│   │   ├── document_converter.py # 文档→医疗卡片转换
│   │   ├── case_excel_service.py # 病例 Excel 读写
│   │   ├── virtual_database_service.py # 虚拟病例数据库
│   │   ├── patient_card_projector.py  # 患者卡片多源投射
│   │   ├── provider_capabilities.py   # LLM 提供者能力检测
│   │   └── local_hf_chat.py     # 本地 HF/vLLM 对话模型
│   └── __init__.py
├── backend/                      # FastAPI BFF 层
│   ├── app.py                    # 应用工厂 + 生命周期引导 + CORS/认证中间件
│   └── api/
│       ├── routes/               # REST API 路由（21 个端点）
│       │   ├── sessions.py       # 会话 CRUD + 患者身份绑定（6 端点）
│       │   ├── chat.py           # SSE 流式对话
│       │   ├── database.py       # 病例数据库查询/搜索/更新（5 端点）
│       │   ├── patient_registry.py # 患者登记处 CRUD（7 端点）
│       │   ├── uploads.py        # 文件上传
│       │   └── assets.py         # 资产文件服务
│       ├── services/             # 后端服务（18 个文件）
│       │   ├── graph_service.py  # 图编排（SSE 流式 + 会话锁 + 心跳）
│       │   ├── graph_factory.py  # 图工厂（real / fixture 模式）
│       │   ├── session_store.py  # 线程安全内存会话存储
│       │   ├── upload_service.py # 上传管线（分类/去重/卡片提取/注册处写入）
│       │   ├── patient_registry_service.py # SQLite 事件溯源注册处
│       │   ├── patient_commands.py   # 患者命令服务（事件溯源强制执行）
│       │   ├── patient_context_resolver.py # 患者上下文解析（缓存/失效）
│       │   ├── database_service.py   # 数据库搜索/过滤/统计
│       │   ├── database_intent_service.py # 自然语言→结构化过滤器
│       │   ├── context_maintenance.py # 后台上下文摘要
│       │   ├── payload_builder.py    # 图输入负载构建
│       │   ├── chat_latency_trace.py # 两阶段延迟追踪
│       │   ├── asset_service.py  # 资产文件加载
│       │   ├── settings.py       # 运行时配置
│       │   ├── fixture_graph_runner.py    # 固定数据回放器
│       │   └── upload_fixture_cards.py    # 固定上传卡片加载
│       ├── schemas/              # Pydantic 数据模型
│       │   ├── events.py         # SSE 事件类型（18 种）
│       │   ├── responses.py      # REST 响应模型
│       │   ├── database.py       # 数据库查询/响应模型
│       │   └── patient_registry.py # 注册处请求/响应模型
│       └── adapters/             # 前端适配层（6 个生产模块 + 9 个测试）
│           ├── state_snapshot.py # 代理状态 → 前端快照
│           ├── event_normalizer.py # 图输出 → SSE 事件
│           ├── card_extractor.py # 卡片提取（11 种类型）+ 去重
│           ├── card_payload_sanitizer.py # 卡片负载清理 + 图像预览
│           ├── reference_normalizer.py   # 引用规范化
│           └── message_content.py # 消息内容清理
├── frontend/                     # React SPA 前端（103 个源文件）
│   └── src/
│       ├── main.tsx              # 入口
│       ├── app/
│       │   ├── router.tsx        # 路由
│       │   ├── providers.tsx     # ApiClient 上下文
│       │   ├── api/              # API 客户端（SSE 流 + 延迟追踪）
│       │   └── store/            # 状态管理（stream-reducer）
│       ├── pages/
│       │   ├── workspace-page.tsx # 主工作区（患者+医生双场景编排）
│       │   └── database-page.tsx  # 数据库控制台
│       ├── features/
│       │   ├── chat/             # 对话面板（流式渲染/内联卡片/思维链/延迟显示）
│       │   ├── cards/            # 卡片渲染系统（11 种卡片 + 分诊交互卡）
│       │   ├── doctor/           # 医生场景布局 + 事件流 + 数据库视图
│       │   ├── database/         # 数据库工作台（搜索/过滤/表格/编辑/自然语言查询）
│       │   ├── patient-identity/ # 患者身份面板
│       │   ├── patient-profile/  # 患者档案面板
│       │   ├── patient-registry/ # 患者登记处（浏览器/搜索/预览/告警/记录）
│       │   ├── roadmap/          # 临床路线图面板
│       │   ├── execution-plan/   # 执行计划面板 + 参考文献列表
│       │   ├── uploads/          # 文件上传面板
│       │   └── workspace/        # 工作区钩子（6 个 hooks + 5 个对应测试）
│       ├── components/
│       │   ├── ui/               # UI 组件库（Button/Card/Input/Textarea/MessageBubble/TopNav/AppShell/PanelGrid）
│       │   └── layout/           # 布局组件（ClinicalTopNav）
│       └── styles/               # 全局样式（globals.css）
├── tests/                        # 测试
│   ├── frontend/                 # 前端集成测试（workspace-scenes / conversation-panel 等）
│   ├── backend/                  # 后端测试（chat_latency_trace 等）
│   ├── e2e/                      # Playwright E2E 测试
│   │   └── acceptance/           # 验收测试规格
│   └── fixtures/                 # 测试固定数据
├── scripts/                      # 启动与管理脚本
│   ├── start_real.ps1            # 一键启动（后端 + 前端）
│   ├── start_backend_real.ps1    # 后端启动
│   ├── start_frontend.ps1        # 前端启动
│   ├── start_backend_fixture.ps1 # 固定数据模式后端
│   ├── start_backend_acceptance_fixture.ps1 # 验收测试后端
│   ├── prepare_acceptance_case_db.py  # 验收测试数据库准备
│   ├── run_e2e_full_acceptance.ps1   # E2E 验收测试
│   └── capture_graph_fixtures.py # 图固定数据捕获
└── pyproject.toml                # Python 项目配置
```

## 快速开始

### 环境要求

- **Python** >= 3.10
- **Node.js** >= 18
- **PowerShell**（Windows 启动脚本）或手动启动各服务

### 1. 安装 Python 依赖

```bash
# 核心依赖（LangGraph + FastAPI + RAG）
pip install -e .

# 完整功能（含 PDF 解析/OCR/重排序/检查点持久化）：
pip install -e ".[full]"

# 按需安装：
pip install -e ".[vision]"      # PDF/Word 文档解析
pip install -e ".[ocr]"         # OCR（扫描 PDF）
pip install -e ".[rerank]"      # Cross-Encoder 重排序（含 torch）
pip install -e ".[cohere]"      # Cohere API 重排序
pip install -e ".[mineru]"      # Magic-PDF 解析
pip install -e ".[checkpoint]"  # 持久化检查点（SQLite / Postgres / Redis）
```

### 2. 安装前端依赖

```bash
cd frontend
npm install
```

### 3. 配置环境变量

编辑项目根目录的 `.env` 文件：

```bash
# LLM（默认使用 MiniMax API）
LLM_MODE=API
LLM_API_BASE=https://api.minimaxi.com/v1
LLM_API_KEY=<your-api-key>
LLM_MODEL=MiniMax-M2.7-highspeed
LLM_TEMPERATURE=0.5
LLM_STREAMING=true
LLM_THINKING_ENABLED=false      # 启用思考链

# Embedding（默认使用阿里云 DashScope）
EMBEDDING_BACKEND=api
EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=<your-api-key>
EMBEDDING_MODEL=text-embedding-v4

# RAG
RAG_RETRIEVAL_K=4
RAG_CHUNK_SIZE=2000
RAG_ENABLE_RERANK=true
RAG_ENABLE_BM25=true

# 网络搜索
WEB_SEARCH_ENABLED=true

# 本地 LLM（可选）
LLM_MODE=Local
LLM_LOCAL_MODEL_PATH=/path/to/model
LLM_LOCAL_BACKEND=HF            # HF / VLLM
```

### 4. 启动服务

**一键启动**（Windows PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_real.ps1 -WarmupRag
```

**手动启动**：

```bash
# 终端 1：启动后端（端口 8000）
AUTH_MODE=bearer API_BEARER_TOKEN=local-dev-token uvicorn backend.app:app --host 127.0.0.1 --port 8000

# 终端 2：启动前端（端口 4173）
cd frontend && VITE_API_BEARER_TOKEN=local-dev-token npm run dev:e2e
```

启动后访问 `http://localhost:4173`。

## 配置说明

### LLM 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_MODE` | 运行模式：`API` / `Local` | `API` |
| `LLM_API_BASE` | API 端点 | `https://api.minimaxi.com/v1` |
| `LLM_API_KEY` | API 密钥 | — |
| `LLM_MODEL` | 模型名称 | `MiniMax-M2.7-highspeed` |
| `LLM_PROVIDER` | 提供者（auto 检测） | — |
| `LLM_TEMPERATURE` | 生成温度 | `0.5` |
| `LLM_MAX_TOKENS` | 最大输出令牌数 | `4096` |
| `LLM_STREAMING` | 启用流式输出 | `True` |
| `LLM_THINKING_ENABLED` | 启用思考模式 | `False` |
| `LLM_THINKING_BUDGET` | 思考预算（tokens） | `8192` |
| `LLM_LOCAL_MODEL_PATH` | 本地模型路径 | — |
| `LLM_LOCAL_BACKEND` | 本地后端：`HF` / `VLLM` | `HF` |

### RAG 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_PARSE_STRATEGY` | 文档解析：`vision` / `basic` | `vision` |
| `RAG_CHUNK_SIZE` | 文档切块大小 | `2000` |
| `RAG_CHUNK_OVERLAP` | 切块重叠大小 | `200` |
| `RAG_RETRIEVAL_K` | 检索返回数量 | `4` |
| `RAG_ENABLE_BM25` | 启用 BM25 关键词检索 | `true` |
| `RAG_ENABLE_RERANK` | 启用重排序 | `true` |
| `RAG_RERANK_MODEL_TYPE` | 重排序模型类型 | `cross_encoder` |
| `RAG_RERANK_MODEL` | 重排序模型名 | `BAAI/bge-reranker-base` |
| `RAG_EMBEDDING_BACKEND` | Embedding 后端：`api` / `local` | `api` |
| `RAG_EMBEDDING_MODEL` | Embedding 模型 | `text-embedding-v4` |
| `RAG_PERSIST_DIR` | Chroma 持久化目录 | `./chroma_db` |
| `RAG_BM25_INDEX_PATH` | BM25 索引路径 | `./bm25_index` |
| `RAG_METADATA_ENHANCEMENT` | 启用元数据增强 | `true` |

### 文档转换配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DOC_CONVERTER_MODEL` | 文档转换模型 | `gpt-4o` |
| `DOC_CONVERTER_MAX_PAGES` | 最大处理页数 | `50` |
| `DOC_CONVERTER_PDF_DPI` | PDF 渲染 DPI | `200` |
| `DOC_CONVERTER_ENABLE_CHUNKED` | 启用长文档分段处理 | `true` |

### 网络搜索配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `WEB_SEARCH_ENABLED` | 启用网络搜索 | `true` |
| `WEB_SEARCH_MODEL` | 搜索模型 | `gpt-4o-search-preview` |

### 检查点与持久化

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHECKPOINT_KIND` | 检查点后端：`memory` / `sqlite` / `postgres` / `redis` | `memory` |
| `CHECKPOINT_URL` | 检查点存储 URL | — |

### 可观测性

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LANGSMITH_TRACING` | 启用 LangSmith 追踪 | `false` |
| `LANGSMITH_API_KEY` | LangSmith API 密钥 | — |
| `LANGSMITH_PROJECT` | LangSmith 项目名 | — |

### 运行时配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AUTH_MODE` | 认证模式：`none` / `bearer` | `bearer` |
| `API_BEARER_TOKEN` | Bearer 令牌；`AUTH_MODE=bearer` 时必填 | — |
| `API_ADMIN_BEARER_TOKEN` | 管理端点 Bearer 令牌；未设置时回退到 `API_BEARER_TOKEN` | — |
| `VITE_API_BEARER_TOKEN` | 前端请求后端时使用的 Bearer 令牌 | — |
| `FRONTEND_ORIGINS` | CORS 允许源 | `http://localhost:5173` |
| `GRAPH_RUNNER_MODE` | 图运行模式：`real` / `fixture` | `real` |
| `RAG_WARMUP` | 启动时预热 RAG | `true` |
| `CHAT_PERF_LOG` | 对话性能日志 | `0` |
| `CHAT_LATENCY_TRACE` | 延迟追踪（Phase1 详细事件） | `0` |
| `UPLOAD_CONVERTER_MODE` | 上传转换模式：`real` / `fixture` | `real` |

本地浏览器 UI 不应暴露单独的 admin token。若本地 UI 需要删除患者或 upsert 病例，使用单 token 模式：只设置 `API_BEARER_TOKEN`，不要设置 `API_ADMIN_BEARER_TOKEN`。

## 测试

```bash
# 后端单元/集成测试（pytest）
pytest tests/backend/

# 前端单元测试（vitest）
cd frontend && npm test

# E2E 测试（playwright）
cd frontend && npm run test:e2e

# E2E 验收测试
cd frontend && npm run test:e2e:acceptance

# 全量 E2E 验收（PowerShell）
powershell -ExecutionPolicy Bypass -File .\scripts\run_e2e_full_acceptance.ps1
```

### Python 依赖审计

```bash
pip install pip-audit
pip-audit
```

在 CI 或本地安全检查中，应在安装项目 Python 依赖后运行 `pip-audit`。如果使用冻结依赖文件，可改为 `pip-audit -r requirements.txt`。

## 运行模式

系统支持两种图执行模式：

| 模式 | 说明 |
|------|------|
| **real**（默认） | 完整 LangGraph Agent 推理，连接真实 LLM |
| **fixture** | 使用预录 Graph tick 回放，用于测试和演示 |

```bash
# Real 模式
AUTH_MODE=bearer API_BEARER_TOKEN=local-dev-token uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Fixture 模式
AUTH_MODE=bearer API_BEARER_TOKEN=local-dev-token GRAPH_RUNNER_MODE=fixture GRAPH_FIXTURE_CASE=database_case uvicorn backend.app:app --host 127.0.0.1 --port 8000

# 可用固定数据用例: database_case, decision_case, safety_case, knowledge_case,
#                   offtopic_date_case, offtopic_date_after_plan_case, upload_followup_case
```

上传转换也支持 fixture 模式（`UPLOAD_CONVERTER_MODE=fixture`）。

## 场景说明

系统支持两种用户场景，通过顶部导航切换：

| 场景 | 用途 | 图类型 |
|------|------|--------|
| **patient** | 患者自报告：门诊分诊、症状采集、身份登记、资料上传 | `patient_graph`（9 节点） |
| **doctor** | 医生临床决策：诊断分期、治疗方案、影像/病理 AI、证据引用、质量审查 | `doctor_graph`（20 节点） |

医生场景支持绑定患者登记处中的患者，自动注入患者摘要和告警信息到图上下文中。

## 关键设计

### 事件溯源患者登记处

患者登记处使用事件溯源模式（SQLite 后端），所有患者数据变更以不可变事件记录。多源数据冲突按优先级仲裁（医生审编 > 病理 > 影像 > 患者自述 > 未知），物化快照投影供图和前端消费。

### Planner 驱动的自适应路由

Agent 核心由 Planner 生成的 `PlanStep` DAG 动态驱动：支持原子步骤、并行组、分支、状态哈希；失败自修正循环（最多 5 次 Planner 迭代，步骤最多 3 次重试）；快速通道优化（模板决策、快速 TNM 验证、简单事实问答）。

### 证据链可追溯

所有治疗决策通过声明（Claim）→ 证据链接（EvidenceLink）→ 检索来源（RetrievedReference）全链路可追溯，支持覆盖率检查和人工审查建议。

### 子代理上下文隔离

知识检索和网页搜索在沙箱子代理中执行（`SubAgentContext`），隔离消息历史、防止上下文污染、支持自动工具故障转移。

### Split Patient Identity

患者身份拆分为 `registry_patient_id`（患者登记处 ID）和 `case_database_patient_id`（历史病例库 ID），通过 `patient_identity.py` 节点统一解析，前端卡片通过 `CardPatientContext` 双字段传递给后端提示请求。

### 每轮节点计时

`node_timings` 使用 `merge_node_timings` reducer 实现每轮重置 + 同节点去重，INTENT 节点作为每轮入口自动清空上一轮计时数据。

## 相关文档

- [Agent 节点文档](src/nodes/README.md)
- [RAG 模块文档](src/rag/README.md)
- [工具模块文档](src/tools/README.md)
- [服务层文档](src/services/README.md)
