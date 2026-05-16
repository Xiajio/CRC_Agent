# 医生端多模态会诊驾驶舱设计

日期：2026-05-16
状态：已批准设计，等待实施计划
范围：轻量闭环首版

## 背景

医生端当前已有 `会诊` 和 `患者数据库` 两个生产 tab，`多模态` 和 `报表` 在导航数据中存在但处于 disabled 状态。后端与前端已经具备多模态相关基础能力：

- 后端已有 radiology/pathology 节点和工具链，可通过医生 graph 的自然语言请求触发影像、肿瘤检测、病理、影像组学分析。
- SSE 与 `stream-reducer` 已支持 `card.upsert`、`message.done`、`critic.verdict`、`references.append`、`status.node`、`done` 等事件。
- 前端卡片系统已能渲染 `imaging_card`、`tumor_detection_card`、`pathology_card`、`pathology_slide_card`、`radiomics_report_card`。
- 医生端已能绑定 patient registry 患者、读取患者摘要、上传资料、records、alerts、eventLog、critic。

因此首版多模态页不需要新增独立后端任务接口。目标是把已有能力组织成医生端可用的“多模态会诊驾驶舱”，让医生在同一页查看患者、多模态证据、复核信号，并通过快捷 prompt 继续使用现有 doctor graph。

## 用户选择

已确认的产品方向：

- 布局方向：A. 会诊驾驶舱。
- 首版范围：A. 轻量闭环。
- 设计边界：新增医生端 `多模态` tab，复用现有 state/cards/chat prompt；不新增后端多模态任务 API。

## 目标

医生在医生端打开 `多模态` tab 后，可以完成以下工作：

1. 确认当前绑定患者和历史病例样本上下文。
2. 查看已存在的影像、肿瘤检测、病理、影像组学卡片。
3. 看到上传资料、registry alerts、critic、人审提示和最近事件。
4. 通过快捷按钮向现有医生会诊 graph 发送分析或总结请求。
5. 将多模态发现自然带回会诊流程，而不切换到外部工具或新任务系统。

## 非目标

首版明确不做以下内容：

- 不新增后端分析任务队列、进度 API、WebSocket 或轮询接口。
- 不重写 LangGraph 节点调度、工具选择策略或现有 SSE 协议。
- 不做 DICOM、WSI、热力图叠加、缩放标注等专业阅片器。
- 不重写上传服务、资产服务或 patient registry 写入链路。
- 不新增第二套医疗卡片渲染系统。
- 不生成最终会诊报告文件；报表导出属于后续独立功能。

## 信息架构

页面嵌入现有医生端 shell，作为第三个生产 tab：

- `会诊`：保持现有医生问答和卡片流。
- `患者数据库`：保持历史病例库与 patient registry 浏览。
- `多模态`：新增轻量会诊驾驶舱。

`多模态` 页面采用三栏密度：

1. 左栏：患者上下文。
   - Registry patient ID。
   - Case sample ID。
   - 年龄、性别、肿瘤部位、TNM、MMR/MSI。
   - 上传资料与 registry records 摘要。
   - registry alerts。
2. 中栏：多模态证据板。
   - 影像组：`imaging_card`、`tumor_detection_card`、`tumor_screening_result`。
   - 病理组：`pathology_card`、`pathology_slide_card`。
   - 组学组：`radiomics_report_card`。
   - 每组使用现有卡片 renderer，不重新实现卡片内部内容。
3. 右栏：临床动作与复核信号。
   - 快捷操作按钮。
   - critic 人审提示。
   - 最近事件流。
   - 缺失数据提示。

在较窄视口下，三栏按现有 dashboard 响应式规则折叠为上下堆叠，优先顺序为患者上下文、快捷操作、多模态证据、事件与复核。

## 组件设计

### `DoctorSceneShell`

职责只保留为医生端 shell 和 tab 分发：

- 将 `DoctorTab` 扩展为 `"consultation" | "database" | "multimodal"`。
- 将 `多模态` 从 disabled nav item 变为生产 tab。
- 当 `activeDoctorTab === "multimodal"` 时渲染 `DoctorMultimodalView`。
- 不在 shell 内实现多模态卡片筛选、prompt 构造或复核汇总逻辑。

### `DoctorMultimodalView`

新增页面容器，接收来自 `DoctorSceneShell` 的现有 props：

- `registryPatientId`
- `caseDatabasePatientId`
- `patientRegistry`
- `cards`
- `critic`
- `eventLog`
- `isStreaming`
- `disabled`
- `onCardPromptRequest`
- `patientContext`

它负责组织页面布局，不直接读取全局 store，也不直接调用 API。

### `doctor-multimodal-utils.ts`

新增纯函数模块：

- `groupMultimodalCards(cards)`：把 cards 分成 imaging、pathology、radiomics 三组。
- `summarizeMultimodalAvailability(groups, context)`：计算每组是否有卡片、是否缺少 case sample、是否缺少绑定患者。
- `buildMultimodalPrompt(action, context)`：构造中文快捷 prompt。
- `buildMultimodalPromptContext(patientContext)`：从已有 patient context 生成 graph context。

这些函数必须可单测，避免把条件逻辑散落在 JSX 中。

### 复用组件

继续复用：

- `ClinicalCardsPanel` 和 `renderCardContent`：负责卡片内容渲染。
- `ClinicalPatientSummary`、`ClinicalUploads`、`ClinicalEventStream` 可在后续实施中视代码可访问性决定是否提取为共享组件。若提取，只移动代码，不改变行为。
- 现有 Button/Card/UI token。

## 快捷操作设计

多模态页提供四个首版动作：

1. `分析当前患者影像`
   - 需要 `case_database_patient_id`。
   - prompt：`请分析当前患者的影像资料，优先检查肿瘤检测、分割和影像组学结果，并生成面向会诊的摘要。`
2. `分析当前患者病理`
   - 需要 `case_database_patient_id`。
   - prompt：`请分析当前患者的病理资料，优先检查切片分类、肿瘤概率、模型置信度和需要人工复核的内容。`
3. `生成多模态摘要`
   - 允许只有 registry patient ID。
   - prompt：`请基于当前患者上下文汇总影像、病理、影像组学和既有医疗卡片，输出多模态会诊摘要，并明确缺失资料。`
4. `带入会诊对话`
   - 允许有任一患者上下文。
   - prompt：`请把当前多模态发现带入会诊流程，结合已有患者摘要、医疗卡片、复核提示和证据来源，继续临床评估。`

所有动作调用现有 `onCardPromptRequest(prompt, context)`。`context` 只包含已经存在的 `registry_patient_id` 和 `case_database_patient_id`，不新增 schema 字段。

## 数据流

数据流保持现有单向模式：

1. `WorkspacePage` 维护 doctor session state。
2. `WorkspacePage` 将 `doctor.state.cards`、`eventLog`、`critic`、patient context 和 `patientRegistry` 传给 `DoctorSceneShell`。
3. `DoctorSceneShell` 根据 active tab 将 props 传给 `DoctorMultimodalView`。
4. `DoctorMultimodalView` 使用 utility 函数分组 cards 和构造 prompt。
5. 用户点击快捷操作后，调用 `onCardPromptRequest(prompt, context)`。
6. 现有 doctor graph 运行，后端通过 SSE 返回 message/card/status/critic。
7. `stream-reducer` 更新 state。
8. 多模态页从更新后的 props 重新渲染。

这个流程不绕过现有 session lock、latency probe、context maintenance 或 SSE reducer。

## 空状态与错误状态

页面必须清晰处理以下状态：

- 未绑定患者：
  - 显示“请先从患者库或历史病例带入患者”。
  - 影像和病理分析按钮禁用。
  - 多模态摘要按钮禁用。
- 有 registry patient ID、没有 case sample：
  - 患者摘要可显示。
  - 生成摘要和带入会诊可用。
  - 影像和病理分析按钮禁用，并显示需要病例样本编号。
- 有 case sample、没有任何多模态卡片：
  - 每组显示行动导向空状态。
  - 快捷分析按钮可用。
- registry 加载失败：
  - 显示 `patientRegistry.error`。
  - 不阻塞现有 cards 展示。
- doctor graph 正在 streaming：
  - 快捷操作按钮禁用。
  - 页面保留现有内容，不清空卡片。
- critic 要求人审：
  - 在复核区显示 `HUMAN_REVIEW_REQUIRED` 和整理后的 feedback。
- alerts 存在：
  - 在左栏或复核区列出 alerts，不自动覆盖患者摘要。

## UI 约束

该页面是医生工作台，不是营销页：

- 信息密度应高于患者端，但保持分组清晰。
- 使用现有临床工作台视觉系统、卡片半径、表格/列表风格和颜色 token。
- 不引入装饰性 hero、宣传文案、渐变大背景或额外插画。
- 不嵌套卡片；页面区块用现有 dashboard column 与 panel stack。
- 按钮文案是实际动作，不写说明型长文。
- 卡片内部已有人工复核提示、置信度展示和原始数据 disclosure，首版不重复实现。

## 测试策略

### Unit tests

新增 `frontend/src/features/doctor/doctor-multimodal-utils.test.ts`：

- 多模态卡片按类型正确分到 imaging/pathology/radiomics。
- 未知 card type 不进入多模态组。
- prompt 构造使用稳定中文文案。
- patient context 只传递 `registry_patient_id` 和 `case_database_patient_id`。
- 缺少 case sample 时影像/病理动作被标记为不可用。

### Component tests

新增 `frontend/src/features/doctor/doctor-multimodal-view.test.tsx`：

- 未绑定患者时显示空状态并禁用动作。
- 有 case sample 和多模态 cards 时显示三组证据。
- 点击快捷按钮调用 `onCardPromptRequest`，带正确 prompt 和 context。
- 有 registry alerts 时显示预警。
- 有 critic 人审要求时显示复核提示。
- streaming/disabled 时动作禁用。

更新 `frontend/src/features/doctor/doctor-scene-shell.test.tsx`：

- 顶部导航显示并启用 `多模态`。
- 点击 `多模态` 后渲染 `DoctorMultimodalView`。
- 切回 `会诊` 和 `患者数据库` 不改变既有行为。

如现有 `use-doctor-view-state` 测试覆盖 tab state，则扩展它；否则在 shell 测试中覆盖。

### Regression tests

运行：

- `npm --prefix frontend run test -- --run frontend/src/features/doctor/doctor-multimodal-utils.test.ts frontend/src/features/doctor/doctor-multimodal-view.test.tsx frontend/src/features/doctor/doctor-scene-shell.test.tsx`
- `npm --prefix frontend run test -- --run`
- `npm --prefix frontend run build`

如本地环境无法运行 npm 命令，实施者必须记录失败原因和未验证风险。

## 实施边界

首版实施应集中在 `frontend/src/features/doctor/` 及必要测试文件。只有在复用现有 summary/uploads/event stream 组件必须提取时，才触及 `DoctorSceneShell` 中已有内部组件。任何后端改动、SSE schema 改动、上传服务改动、RAG 或 graph 调度改动都不属于本设计。

## 验收标准

设计验收标准：

- 医生端顶部导航有可点击 `多模态` tab。
- 在绑定患者且已有多模态卡片时，多模态页能显示患者上下文、卡片分组、复核信号和快捷动作。
- 点击快捷动作走现有 `onCardPromptRequest`，不会调用不存在的新 API。
- 未绑定患者、缺少 case sample、无卡片、registry error、streaming 等状态都有明确 UI。
- 会诊页和患者数据库页现有行为不回退。
- 测试覆盖主要 utility、view 和 shell 分发行为。

## 后续扩展

后续可独立设计：

- 后端多模态任务 API 与进度跟踪。
- 图像/切片专业查看器。
- 多模态结果导出到报告页。
- 人工复核队列和签署流。
