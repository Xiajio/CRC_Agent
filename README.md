# LangG — 结直肠癌智能临床决策支持系统

基于 **LangGraph** 的多智能体临床决策支持系统，面向**结直肠癌（CRC）**的智能诊疗全流程。提供患者分诊与医生临床决策辅助两大场景，集成 RAG 知识检索、医学影像 AI 分析和历史病例数据库。

## 核心功能

| 场景 | 功能 |
|------|------|
| **患者端** | 智能分诊问答、症状采集、病历资料上传、身份登记 |
| **医生端** | 临床评估、诊断分期（TNM）、治疗方案决策、影像/病理 AI 分析、指南证据检索与引用、质量评审 |
| **知识检索** | 混合 RAG 引擎（Chroma 向量检索 + BM25 关键词检索 + Cross-Encoder 重排序），面向中文临床指南 |
| **影像 AI** | YOLOv8 肿瘤检测、U-Net 肿瘤分割、CLAM 病理切片分类、PyRadiomics 影像组学 |
| **病例库** | 历史病例 Excel 数据库，支持结构化筛选与自然语言查询 |

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (React 18 + TypeScript + Vite + TailwindCSS)  │
│  pages/workspace-page  ·  components/  ·  features/     │
└──────────────────────┬──────────────────────────────────┘
                       │ SSE streaming
┌──────────────────────▼──────────────────────────────────┐
│  Backend BFF (FastAPI + Uvicorn)                        │
│  routes/sessions · chat · database · patient-registry   │
│  services/graph_service · session_store · upload        │
└──────────────────────┬──────────────────────────────────┘
                       │ LangGraph invocation
┌──────────────────────▼──────────────────────────────────┐
│  Agent Core (LangGraph)                                 │
│  graph_builder → nodes/ (intent · planner · knowledge   │
│  · radiology · pathology · assessment · staging ·       │
│  decision · critic · citation · evaluator · memory)     │
│  + RAG engine (rag/)  +  Tools (tools/)                 │
└─────────────────────────────────────────────────────────┘
```

**医生端 Agent 工作流**：Intent → Planner → Knowledge/Radiology/Pathology/WebSearch → Assessment → Diagnosis → Staging → Decision → Critic → Citation → Evaluator → Finalize

**患者端 Agent 工作流**：Intent → Planner → Clinical Entry/Triage → Knowledge → Assessment → Chat

## 目录结构

```
LangG/
├── src/                    # Python 核心：LangGraph Agent
│   ├── graph_builder.py    # Graph 构建（doctor / patient）
│   ├── state.py            # CRCAgentState 状态定义
│   ├── config.py           # 配置模型（Pydantic Settings）
│   ├── nodes/              # Agent 节点实现（20+ 文件）
│   ├── policies/           # 路由与评审策略
│   ├── prompts/            # LLM 提示词模板
│   ├── rag/                # RAG 引擎（解析/摄取/检索/重排）
│   ├── tools/              # LangChain 工具集
│   └── services/           # LLM / 搜索 / 文档转换等服务
├── backend/                # FastAPI BFF 层
│   └── api/
│       ├── routes/         # REST API 路由
│       └── services/       # 图服务 / 会话 / 上传 / 注册
├── frontend/               # React SPA 前端
│   └── src/
│       ├── app/            # 路由、状态、API 客户端
│       ├── pages/          # WorkspacePage / DatabasePage
│       ├── features/       # 聊天 / 卡片 / 数据库 / 上传
│       └── components/     # UI 组件与布局
├── data/                   # 病例影像数据 / 临床指南 PDF
├── docs/                   # 架构文档与迁移方案
├── tests/                  # 测试（backend / frontend / fixtures）
├── scripts/                # 启动与管理脚本
├── runtime/                # 运行时数据（SQLite / 上传文件）
├── chroma_db/              # Chroma 向量库
├── bm25_index/             # BM25 关键词索引
├── pyproject.toml          # Python 项目配置
└── .env                    # 环境变量
```

## 快速开始

### 环境要求

- **Python** >= 3.10
- **Node.js** >= 18
- **PowerShell**（Windows 启动脚本）或手动启动各服务

### 1. 安装 Python 依赖

```bash
pip install -e .

# 完整功能（含 PDF 解析、OCR、重排序、持久化）：
pip install -e ".[full]"
```

### 2. 安装前端依赖

```bash
cd frontend
npm install
```

### 3. 配置环境变量

编辑项目根目录的 `.env` 文件，至少配置以下变量：

```bash
# LLM（默认使用 MiniMax API）
LLM_MODE=API
LLM_API_BASE=https://api.minimaxi.com/v1
LLM_API_KEY=<your-api-key>
LLM_MODEL=MiniMax-M2.7-highspeed

# Embedding（默认使用阿里云 DashScope）
EMBEDDING_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=<your-api-key>
EMBEDDING_MODEL=text-embedding-v4
```

### 4. 启动服务

**一键启动**（Windows PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_real.ps1 -WarmupRag
```

**手动启动**：

```bash
# 终端 1：启动后端（端口 8000）
uvicorn backend.app:app --host 0.0.0.0 --port 8000

# 终端 2：启动前端（端口 4173）
cd frontend && npm run build && npm run preview
```

启动后访问 `http://localhost:4173`。

## 配置说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_MODE` | LLM 运行模式：`API` / `Local` | `API` |
| `LLM_MODEL` | 模型名称 | `MiniMax-M2.7-highspeed` |
| `LLM_TEMPERATURE` | 生成温度 | `0.5` |
| `LLM_STREAMING` | 启用流式输出 | `True` |
| `RAG_RETRIEVAL_K` | 检索返回数量 | `4` |
| `RAG_CHUNK_SIZE` | 文档切块大小 | `2000` |
| `RAG_ENABLE_RERANK` | 启用重排序 | `true` |
| `WEB_SEARCH_ENABLED` | 启用网络搜索 | `true` |
| `CHECKPOINT_KIND` | 状态持久化：`memory` / `sqlite` / `postgres` / `redis` | `memory` |
| `LANGSMITH_TRACING` | 启用 LangSmith 追踪 | `false` |

完整配置项参见 `.env` 文件及 `src/config.py`。

## 测试

```bash
# 后端测试（pytest）
pytest tests/backend/

# 前端单元测试（vitest）
cd frontend && npm test

# E2E 测试（playwright）
cd frontend && npm run test:e2e

# E2E 验收测试
cd frontend && npm run test:e2e:acceptance
```

## 运行模式

系统支持两种图执行模式，通过环境变量控制：

| 模式 | 说明 |
|------|------|
| **real**（默认） | 完整 LangGraph Agent 推理，连接真实 LLM |
| **fixture** | 使用预录 Graph tick 回放，用于测试和演示 |

```bash
# Fixture 模式
uvicorn backend.app:app --host 0.0.0.0 --port 8000
# 后端自动检测 fixture 模式（通过 GRAPH_RUNNER_MODE 环境变量）
```

## 相关文档

- [架构图谱](docs/current-architecture-map.md) — 数据流、API 路由、Graph 装配
- [Agent 节点文档](src/nodes/README.md) — 各节点功能说明
- [RAG 模块文档](src/rag/README.md) — 文档解析、摄取、检索、重排
- [工具模块文档](src/tools/README.md) — 临床工具、RAG 工具、影像工具
- [服务层文档](src/services/README.md) — LLM 服务、搜索服务、文档转换
