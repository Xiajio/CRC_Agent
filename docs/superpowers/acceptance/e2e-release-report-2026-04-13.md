# E2E 全量验收执行报告（2026-04-13）

## 1. 背景与目标

本次工作基于 [2026-04-11-e2e-full-acceptance.md](/D:/亿铸智能体/LangG_New/docs/superpowers/plans/2026-04-11-e2e-full-acceptance.md) 中定义的目标执行，目标是建立一套可重复、可审计的全量验收机制，用于在“前后端真实联调 + 受控 fixture 模型与数据”的前提下完成发布前质量门禁。

本次验收的核心目标如下：

- 打通真实前端、真实后端与受控 fixture 图回放的完整链路。
- 让病例查询、治疗方案、安全告警、知识问答、跑题重定向、数据库控制台、上传与卡片动作等关键用户路径具备稳定自动化覆盖。
- 让验收结果可复现，避免依赖实时模型、外部检索或网络状态导致 golden fixture 不可信。
- 输出可交付的 evidence、runbook、checklist 和 release report 模板，用于后续 8 小时全量验收执行。

## 2. 环境信息

- 日期：`2026-04-13`
- 执行者：`Codex`
- 仓库根目录：`D:\亿铸智能体\LangG_New`
- 分支：`N/A（当前 workspace 不是 git repo）`
- Commit：`N/A（当前 workspace 不是 git repo）`
- Backend Python：`Python 3.10.19`
- Node：`v25.8.2`
- npm：`11.11.1`
- 前端全量验收命令：`D:\anaconda3\envs\LangG\npm.cmd --prefix D:\亿铸智能体\LangG_New\frontend run test:e2e:acceptance`
- 一键验收命令：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1`
- 自动化证据目录：`output/acceptance`

## 3. 主要修改内容

### 3.1 验收基础设施

- 在 [payload_builder.py](/D:/亿铸智能体/LangG_New/backend/api/services/payload_builder.py) 中完成每轮 `fixture_case` / `fixture_tick_delay_ms` / `current_patient_id` 的 allowlist 透传，保证 fixture 选择能力只用于测试，不开放任意 `context` 回灌。
- 新增受控 acceptance 数据库物料化能力：
  - [acceptance_case_db.py](/D:/亿铸智能体/LangG_New/backend/api/services/testing/acceptance_case_db.py)
  - [prepare_acceptance_case_db.py](/D:/亿铸智能体/LangG_New/scripts/prepare_acceptance_case_db.py)
  - [seed.json](/D:/亿铸智能体/LangG_New/tests/fixtures/acceptance_case_db/seed.json)
- 新增受控上传转换能力：
  - [upload_fixture_cards.py](/D:/亿铸智能体/LangG_New/backend/api/services/upload_fixture_cards.py)
  - [upload_service.py](/D:/亿铸智能体/LangG_New/backend/api/services/upload_service.py)
  - [acceptance-note.json](/D:/亿铸智能体/LangG_New/tests/fixtures/upload_cards/acceptance-note.json)
  - [acceptance-summary.json](/D:/亿铸智能体/LangG_New/tests/fixtures/upload_cards/acceptance-summary.json)

### 3.2 Graph Fixture 可信化

- 在 [capture_graph_fixtures.py](/D:/亿铸智能体/LangG_New/scripts/capture_graph_fixtures.py) 中将 Task 2 的高风险 case 改为“混合生成模式”：
  - `database_case` 保持 live capture
  - `decision_case`
  - `safety_case`
  - `knowledge_case`
  - `offtopic_date_case`
  - `offtopic_date_after_plan_case`
  - `upload_followup_case`
    以上改为脚本内生成的 deterministic template fixtures
- 新模板 fixture 不仅写入 `assessment_draft`，也显式写入公开 assistant `messages` 与必要的 `current_patient_id`，从而保证：
  - UI 显示结果正确
  - `normalize_tick` 事件正确
  - `GET /api/sessions/.../messages` 历史与 UI 一致
- 更新 [tests/fixtures/graph_ticks/README.md](/D:/亿铸智能体/LangG_New/tests/fixtures/graph_ticks/README.md)，明确 fixture 的 live/template 生成策略。

### 3.3 Playwright Acceptance Harness

- 完成 acceptance harness 与 support helpers：
  - [playwright.acceptance.config.ts](/D:/亿铸智能体/LangG_New/frontend/playwright.acceptance.config.ts)
  - [acceptance-fixtures.ts](/D:/亿铸智能体/LangG_New/tests/e2e/support/acceptance-fixtures.ts)
  - [backend-session.ts](/D:/亿铸智能体/LangG_New/tests/e2e/support/backend-session.ts)
  - [workspace-driver.ts](/D:/亿铸智能体/LangG_New/tests/e2e/support/workspace-driver.ts)
  - [upload-driver.ts](/D:/亿铸智能体/LangG_New/tests/e2e/support/upload-driver.ts)
  - [database-driver.ts](/D:/亿铸智能体/LangG_New/tests/e2e/support/database-driver.ts)
  - [evidence.ts](/D:/亿铸智能体/LangG_New/tests/e2e/support/evidence.ts)
- 完成并激活三组 E2E：
  - [workspace-core.spec.ts](/D:/亿铸智能体/LangG_New/tests/e2e/acceptance/workspace-core.spec.ts)
  - [database-console.spec.ts](/D:/亿铸智能体/LangG_New/tests/e2e/acceptance/database-console.spec.ts)
  - [uploads-and-cards.spec.ts](/D:/亿铸智能体/LangG_New/tests/e2e/acceptance/uploads-and-cards.spec.ts)

### 3.4 用例解锁与稳定性修复

- 原先因 Task 2 不可信而 `fixme` 的 acceptance 用例已全部恢复为真实断言执行：
  - `workspace-core.spec.ts`
    - decision flow
    - safety flow
    - knowledge query
    - off-topic redirect after plan
  - `uploads-and-cards.spec.ts`
    - post-upload follow-up
    - card prompt actions
- 在 [database-driver.ts](/D:/亿铸智能体/LangG_New/tests/e2e/support/database-driver.ts) 中补齐等待逻辑：
  - structured filter 后等待 `POST /api/database/cases/search`
  - save 后等待 `POST /api/database/cases/upsert`
  从而消除 full pack 并行运行时的假失败。

### 3.5 文档与执行入口

- 完成一键 runner 与验收文档：
  - [run_e2e_full_acceptance.ps1](/D:/亿铸智能体/LangG_New/scripts/run_e2e_full_acceptance.ps1)
  - [e2e-full-acceptance-runbook.md](/D:/亿铸智能体/LangG_New/docs/superpowers/acceptance/e2e-full-acceptance-runbook.md)
  - [e2e-manual-review-checklist.md](/D:/亿铸智能体/LangG_New/docs/superpowers/acceptance/e2e-manual-review-checklist.md)
  - [e2e-release-report-template.md](/D:/亿铸智能体/LangG_New/docs/superpowers/acceptance/e2e-release-report-template.md)

## 4. 最终验证结果

### 4.1 后端回归

执行命令：

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_payload_builder.py tests/backend/test_capture_graph_fixtures.py tests/backend/test_acceptance_case_db.py tests/backend/test_uploads_routes.py -v
```

结果：

- `34 passed`
- 遗留 warning：`src/services/document_converter.py` 中 Pydantic V1 风格 `@validator` 废弃警告

### 4.2 前端与 E2E 回归

执行命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1
```

结果：

- Playwright 全量 pack：`14 passed`
- 细分结果：
  - `workspace-core.spec.ts`：`6 passed`
  - `database-console.spec.ts`：`4 passed`
  - `uploads-and-cards.spec.ts`：`4 passed`

### 4.3 Evidence

自动化证据已写入：

- [output/acceptance](/D:/亿铸智能体/LangG_New/output/acceptance)

目录中包含：

- workspace core 相关截图和 JSON evidence
- uploads and cards 相关截图和 JSON evidence
- database console 相关截图和 JSON evidence

## 5. 目标达成情况

相对于最初目标，本次工作已经达成：

- 已建立可重复执行的全量验收链路。
- 已完成真实前后端联调 + 受控 fixture/数据的稳定化。
- 已覆盖工作区、数据库、上传、卡片动作三大核心产品面。
- 已解决 Task 2 的 fixture 信任问题，且不再依赖外部网络与模型状态。
- 已消除此前的 conditional pass / skipped 状态，当前自动化验收无遗留阻断项。

未纳入“自动化已完成”结论的部分：

- [e2e-manual-review-checklist.md](/D:/亿铸智能体/LangG_New/docs/superpowers/acceptance/e2e-manual-review-checklist.md) 尚未人工签字
- 最终发布报告模板尚未由人工补充签署字段

## 6. 风险与遗留问题

当前剩余问题不属于阻断发布的自动化缺陷，但仍建议后续跟进：

- `src/services/document_converter.py` 存在 Pydantic V1 `@validator` 废弃警告，当前不影响本轮验收通过，但后续升级到更严格版本时需要处理。
- 当前 workspace 不是 git repo，因此计划中的 git checkpoint 未执行；实现与验证已完成，但无法提供 commit/branch 元数据。
- 医疗文案、视觉质量、卡片语义与图像可用性的最终签署仍需人工验收。

## 7. 结论

### 自动化结论

- 结论：`PASS`
- 说明：后端 acceptance-support 回归 `34 passed`，Playwright 全量 E2E `14 passed`，`Conditional-pass exceptions: none`

### 发布建议

- 建议：`PASS WITH CONDITIONS`
- 条件：在正式发布前，由产品/医学/测试完成以下人工签字：
  - 医疗文案
  - 视觉质量
  - 卡片语义
  - 图像/病理预览可用性
  - Trust & Safety 呈现

### 发布说明

本轮工作的关键成果不是单点修复，而是把整套 acceptance 体系从“部分通过 + 依赖手工判断 + fixture 不可信”提升到了“受控数据、稳定 fixture、可重复全量执行、自动化全绿”的状态，可作为后续版本发布门禁的基线。
