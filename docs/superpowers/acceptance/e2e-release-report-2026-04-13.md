# E2E 全量验收执行报告（2026-04-13，当前仓库补正）

## 1. 当前状态

本文件保留 2026-04-13 全量验收报告的交付入口，但已按当前仓库 `D:\YiZhu_Agnet\LangG` 修正 handoff 口径。原报告中的 `D:\亿铸智能体\LangG_New` 路径和“当前 workspace 不是 git repo”结论不再适用于本仓库。

当前仓库可执行的验收交付主线是：

- 真实病例人工复核 fixture：`tests\fixtures\graph_ticks\real_case_human_review.json`
- 浏览器验收脚本：`scripts\run_real_case_browser_acceptance.cjs`
- 当前 runbook：`docs\superpowers\acceptance\e2e-full-acceptance-runbook.md`
- 当前报告：`docs\superpowers\acceptance\real-case-human-review-acceptance-report-2026-05-03.md`
- 证据目录：`output\browser-acceptance\real_case_human_review\`

## 2. 环境信息

- 日期：`2026-05-03`
- 仓库根目录：`D:\YiZhu_Agnet\LangG`
- 分支：`main`
- 当前 HEAD（文档补齐时）：`ed02c0d`
- Backend Python：`D:\anaconda3\envs\LangG\python.exe`
- npm：`D:\anaconda3\envs\LangG\npm.cmd`
- Node：`D:\anaconda3\envs\LangG\node.exe` 或 `node` on `PATH`
- 前端构建命令：`D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build`
- 真实病例浏览器验收命令：`D:\anaconda3\envs\LangG\node.exe scripts/run_real_case_browser_acceptance.cjs`

## 3. 与旧报告的差异

旧报告描述的是一套历史全量 E2E pack：

- 后端 acceptance-support：`34 passed`
- 前端 Playwright 全量 pack：`14 passed`
- 旧证据目录：`output\acceptance\`
- 旧一键入口：`scripts\run_e2e_full_acceptance.ps1`

当前仓库中，`tests\e2e\acceptance` 目录不存在，因此旧 Playwright full-pack 命令不能作为当前 handoff 的唯一依据。`scripts\run_e2e_full_acceptance.ps1` 仍保留为历史入口，只有在旧 E2E 套件恢复后才应作为全量自动化门禁运行。

## 4. 当前真实病例验收范围

本次交付聚焦真实病例人工复核场景：

- 病例：62 岁男性，活检确认 pMMR 低位直肠腺癌，MRI `cT3N1M0`，无远处转移，ECOG 1
- 预期结论：治疗建议保留，但必须进入人工肿瘤专科复核
- 安全标记：`HUMAN_REVIEW_REQUIRED`
- 引用状态：无直接 guideline references 绑定，必须明确披露
- UI 证据：警告、建议保留说明、无直接引用说明、执行计划、roadmap、blocked review step、clinical event stream

## 5. 自动化证据

真实病例浏览器验收已在沙箱外执行通过。当前环境中，前端重新构建命令被环境权限阻断（沙箱内 esbuild `spawn EPERM`，沙箱外 `Access is denied`），本次浏览器验收使用已有 `frontend\dist\index.html`。

本次实际结果：

- `ok: true`
- `fixtureCase: real_case_human_review`
- `planRows: 3`
- `roadmapSteps: 4`
- `blockedRoadmapSteps: 1`
- `eventChips: 17`
- `roadmapEventChips: 4`
- `warningCount: 5`
- `failedResponses: []`

本次产生以下证据：

- `output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.json`
- `output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.png`
- `output\browser-acceptance\real_case_human_review\real-case-backend.out.log`
- `output\browser-acceptance\real_case_human_review\real-case-backend.err.log`

JSON 结果至少应包含：

- `ok: true`
- `fixtureCase: real_case_human_review`
- `planRows >= 1`
- `roadmapSteps >= 1`
- `blockedRoadmapSteps >= 1`
- `warningCount >= 1`

## 6. 人工复核签署项

当前自动化只能证明浏览器链路、fixture 回放和安全呈现可见；以下仍需要人工签署：

- 医疗文案：病例分期、TNT/新辅助治疗表述、MDT 复核建议是否合理
- 视觉质量：警告、计划、roadmap、事件流是否清晰可读
- 引用可信度：无直接引用披露是否醒目，是否避免将建议包装成已引用结论
- Trust & Safety：`HUMAN_REVIEW_REQUIRED` 是否足够明确，是否避免自动放行

## 7. 结论口径

- 当前可交付结论：`PASS WITH HUMAN REVIEW REQUIRED`
- 自动化范围：真实病例 fixture + fixture backend + built frontend + headless browser UI 验收
- 发布限制：没有人工医学/安全签署前，不得把该病例建议作为无需复核的最终治疗方案

本文件的当前作用是把历史全量验收材料与当前仓库真实病例 handoff 对齐。完整填写请使用 `real-case-human-review-acceptance-report-2026-05-03.md`。
