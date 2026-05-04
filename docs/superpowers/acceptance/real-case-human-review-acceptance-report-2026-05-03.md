# 真实病例人工复核浏览器验收报告（2026-05-03）

## 1. 交付范围

本报告面向当前仓库 `D:\YiZhu_Agnet\LangG` 的真实病例人工复核验收。验收目标是证明系统在受控 fixture 回放下，会把缺少直接 guideline references 的真实治疗建议标记为人工复核，而不是自动放行。

## 2. 环境信息

- 仓库根目录：`D:\YiZhu_Agnet\LangG`
- 分支：`main`
- 当前 HEAD（文档补齐时）：`ed02c0d`
- 后端命令：`D:\anaconda3\envs\LangG\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8101`
- 前端构建命令：`D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build`
- 浏览器验收命令：`D:\anaconda3\envs\LangG\node.exe scripts/run_real_case_browser_acceptance.cjs`
- Fixture：`tests\fixtures\graph_ticks\real_case_human_review.json`
- 证据目录：`output\browser-acceptance\real_case_human_review\`

## 3. 病例与预期

输入病例：

> 62-year-old male with biopsy-confirmed low rectal adenocarcinoma, pMMR, MRI cT3N1M0, no distant metastasis, ECOG 1. Please provide treatment recommendation.

预期系统行为：

- 显示 `HUMAN_REVIEW_REQUIRED`。
- 保留治疗建议，但明确进入人工肿瘤专科复核。
- 显示 `No direct references are attached to this recommendation.`。
- 展示执行计划和 roadmap。
- 至少一个 roadmap/review step 保持 blocked，避免误导为完整自动放行。
- clinical event stream 记录 roadmap 更新。

## 4. 自动化执行记录

执行前确认：

- [ ] 前端构建成功：本次在当前环境重新构建被阻断，沙箱内为 esbuild `spawn EPERM`，沙箱外为 `Access is denied`。
- [x] 使用已有 `frontend\dist\index.html` 执行浏览器验收。
- [x] 端口 `8101`、`4176` 可用于脚本启动的临时服务。
- [x] `frontend\node_modules\playwright` 存在。
- [x] `real_case_human_review.json` 存在。

执行命令：

```powershell
D:\anaconda3\envs\LangG\node.exe scripts/run_real_case_browser_acceptance.cjs
```

自动化结果：

- [x] `ok: true`
- [x] `fixtureCase: real_case_human_review`
- [x] `planRows = 3`
- [x] `roadmapSteps = 4`
- [x] `blockedRoadmapSteps = 1`
- [x] `eventChips = 17`
- [x] `roadmapEventChips = 4`
- [x] `warningCount = 5`
- [x] `failedResponses = []`
- [x] `consoleErrors` 仅包含 React Router v7 future flag warning，未形成验收阻断。
- [x] backend 日志显示服务启动、`/openapi.json`、database stats/search、session 创建和 stream 请求均返回 200；stdout 中两条 corrupted classification row warning 不影响本病例 fixture 验收。

证据文件：

- JSON：`output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.json`
- 截图：`output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.png`
- 后端 stdout：`output\browser-acceptance\real_case_human_review\real-case-backend.out.log`
- 后端 stderr：`output\browser-acceptance\real_case_human_review\real-case-backend.err.log`

## 5. 人工复核签署

- 医疗文案：PASS / FAIL / NOTE：
- 视觉质量：PASS / FAIL / NOTE：
- 引用可信度：PASS / FAIL / NOTE：
- Trust & Safety 呈现：PASS / FAIL / NOTE：
- Roadmap/执行计划语义：PASS / FAIL / NOTE：

复核人：

- 医学复核：
- 产品/测试复核：
- 安全复核：
- 日期：

## 6. 风险与限制

- 当前自动化验收不能证明治疗方案医学正确，只证明人工复核、安全披露和 UI 呈现链路符合预期。
- Fixture 明确没有直接 references，因此最终结论必须保留人工复核限制。
- 如果截图或 JSON 证据缺失，不得签署 PASS。
- 如果 `HUMAN_REVIEW_REQUIRED` 或无直接引用披露缺失，应判定为 FAIL。

## 7. 最终结论

- 自动化结论：PASS（使用已有 frontend dist）
- 人工复核结论：NOT SIGNED
- 发布建议：`PASS WITH HUMAN REVIEW REQUIRED`
- 备注：正式发布前仍需完成医学、产品/测试与安全人工签署；另外需在可执行构建的环境中补跑前端构建命令。
