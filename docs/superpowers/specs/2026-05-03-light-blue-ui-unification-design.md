# 浅蓝白 UI Shell 与基础组件统一设计

日期：2026-05-03

## 背景

当前前端样式主要集中在 `frontend/src/styles/tokens.css` 和 `frontend/src/styles/globals.css`。`globals.css` 已超过 2300 行，Tailwind 配置基本没有承载设计系统。代码中同时存在 `workspace-*` / `database-*` 和 `clinical-*` / `patient-*` 两套视觉语言。

Workspace/Database 体系大量使用玫瑰/浆果色硬编码，例如 `#8e4a55`、`#91515a` 和 `rgba(165, 73, 83, ...)`。Clinical 体系则以 navy header、蓝色主操作和白色卡片为主。两套体系在颜色、间距、圆角、字号、消息气泡和按钮输入框形态上都不一致，导致 `/database`、医生端 database tab、患者端/医生端工作台看起来不像同一产品。

本设计采用“UI Shell + 基础组件同步收敛”方案：统一浅蓝白视觉 token，抽取共享 `AppShell`、`TopNav`、`PanelGrid`、`Card`、`Button`、`Input`、`Textarea`、`MessageBubble`，并让 `/database`、医生 database tab、患者/医生工作台共享同一套布局和基础组件。

## 目标

1. 将主视觉统一为浅蓝白临床工作台风格，移除 workspace/database 的玫瑰色主视觉。
2. 保留现有三栏信息架构，不做明显 redesign。
3. 允许轻度布局整理：统一 header 高度、panel 宽度节奏、card 密度、按钮输入框和消息区形态。
4. 结构性合并通用 UI 组件，让 workspace、database、clinical 页面共享基础组件边界。
5. 扩展并贯彻设计 token，减少硬编码颜色、间距、圆角和字号。
6. 保留业务 hooks、API client、SSE 流、患者库、上传、卡片渲染等业务契约。
7. 保留关键 `data-testid`，降低测试迁移成本。

## 非目标

1. 不重写后端 API 或前端业务数据流。
2. 不重做产品信息架构，不把三栏工作台改成全新的页面模式。
3. 不引入新的 UI 框架或图标库作为本次重构前置条件。
4. 不一次性删除所有旧 CSS；旧 class 可以作为迁移兼容层存在。
5. 不把本次工作扩大为完整组件库建设、Storybook 或设计系统站点。

## 推荐方案

采用方案 2：UI Shell + 基础组件同步收敛。

新增 `frontend/src/components/ui/`，提供共享 UI primitives。页面层继续负责业务编排，但不再直接拼大量视觉 class。`/database`、医生端 database tab、患者端/医生端工作台统一使用浅蓝白 token 和共享 shell/grid/card/button/input/message 组件。

这能真正解决“两套视觉系统共存”，同时保持页面结构和业务行为稳定。主要成本是测试回归范围较大，尤其是 `workspace-page.test.tsx`、`doctor-scene-shell.test.tsx`、`conversation-panel.test.tsx` 和 database 相关测试。

## 架构设计

### UI 层

新增目录：

- `frontend/src/components/ui/app-shell.tsx`
- `frontend/src/components/ui/top-nav.tsx`
- `frontend/src/components/ui/panel-grid.tsx`
- `frontend/src/components/ui/card.tsx`
- `frontend/src/components/ui/button.tsx`
- `frontend/src/components/ui/input.tsx`
- `frontend/src/components/ui/textarea.tsx`
- `frontend/src/components/ui/message-bubble.tsx`
- `frontend/src/components/ui/index.ts`

这些组件只负责语义、布局、样式变体和可访问性，不读取业务状态，不调用 API，不理解患者库、会话、卡片 payload 或 SSE。

### 页面层

`DatabasePage`、`WorkspacePage`、`DoctorSceneShell`、`DoctorDatabaseView` 继续负责 hooks、业务 props、tab 状态、session 状态和事件处理。页面层通过 `AppShell`、`TopNav`、`PanelGrid`、`Card` 等组合 UI，不再直接承担视觉系统细节。

### 兼容层

现有 `ClinicalTopNav` 和 `WorkspaceLayout` 不立即删除。第一轮实现中可以让它们包装新 `TopNav` 和 `PanelGrid`，保留旧 props 和关键测试标识。等页面迁移完成后，再清理无用 wrapper 和旧 CSS。

## 视觉 Token

`tokens.css` 扩展为浅蓝白语义 token。旧 `--clinical-*` token 先 alias 到新 token，避免一次性打断现有样式。

### 色彩

```css
--color-canvas: #f4f8fc;
--color-surface: #ffffff;
--color-surface-soft: #f8fbff;
--color-primary: #1466d8;
--color-primary-hover: #0f58bf;
--color-primary-soft: #eaf4ff;
--color-navy: #061f3d;
--color-text: #182434;
--color-text-muted: #66758a;
--color-border: #dbe7f3;
--color-border-strong: #bfd3ea;
--color-success: #24a66a;
--color-warning: #f06423;
--color-danger: #cc2f47;
```

玫瑰/浆果色不再作为主视觉。Warning/Danger/Success 只用于状态语义，不用于页面主色调。

### 间距

采用 4px 基准，保持中等密度。

```css
--space-1: 4px;
--space-2: 8px;
--space-3: 12px;
--space-4: 16px;
--space-5: 20px;
--space-6: 24px;
--space-8: 32px;
```

默认规则：

- Page padding：desktop 12px，mobile 8px。
- Panel gap：12px 或 16px。
- Card padding：默认 16px，紧凑卡片 12px，无内边距卡片用于自带 header/body 的复合卡。

### 圆角

收敛为 5 档。

```css
--radius-xs: 4px;
--radius-sm: 6px;
--radius-md: 8px;
--radius-lg: 10px;
--radius-pill: 999px;
```

默认规则：

- Card：8px。
- Button/Input/Textarea/MessageBubble：8px。
- 紧凑 icon button：6px 或 8px。
- Avatar、badge、pill：999px。

### 字号

收敛为 7 级。

```css
--font-xs: 12px;
--font-sm: 13px;
--font-md: 14px;
--font-base: 16px;
--font-lg: 18px;
--font-xl: 20px;
--font-2xl: 24px;
```

默认规则：

- 工作台正文：14px。
- 元信息、badge、辅助说明：12px 或 13px。
- Panel 标题：16px。
- 顶栏品牌/主标题：18px 或 20px。

### 状态

- Hover：浅蓝背景或蓝色边框增强。
- Focus：`0 0 0 3px rgba(20, 102, 216, 0.16)`。
- Disabled：`opacity: 0.55` + `cursor: not-allowed`。
- Selected：`--color-primary-soft` 背景 + `--color-primary` 或 `--color-border-strong` 边框。
- Error：保留 `--color-danger`，但错误容器背景使用浅红而非高饱和色块。

## 组件设计

### AppShell

职责：

- 提供页面背景、顶栏区域和主内容区域。
- 统一浅蓝白画布。
- 支持 `className` 和测试标识透传。

约束：

- 不读取 scene、session 或路由业务状态。
- 不实现业务 tab 切换。

### TopNav

职责：

- 替代 `ClinicalTopNav` 和 database 的 `workspace-global-header` 视觉形态。
- 支持品牌、nav items、active key、actions、status pill、profile switch。
- 统一 navy/blue 顶栏或浅色顶栏策略。本次默认保留 navy 顶栏，但所有辅助控件和页面主体统一浅蓝白。

迁移：

- `ClinicalTopNav` 先包装 `TopNav`。
- `/database` header 改用 `TopNav`，不再使用玫瑰色 `workspace-brand` / `workspace-stage-badge`。

### PanelGrid

职责：

- 替代 `WorkspaceLayout` 内部 grid 和 clinical dashboard grid。
- 支持三栏、双栏、center-only。
- 支持 left/right open 状态。
- 保留 `data-testid="workspace-layout-grid"`、`left-rail`、`center-workspace`、`right-inspector` 等测试标识，或提供显式 prop 传入。

默认布局：

- 三栏仍保持 left / center / right。
- 中等密度 gap 12px。
- mobile 720px 下堆叠。
- 1450px、1150px 断点可继续沿用，但使用统一变量和组件 props 表达。

### Card

职责：

- 替代 `workspace-card`、`workspace-banner`、`clinical-card`。
- 支持 header slot、footer slot、padding、tone、selected/loading/empty 状态。

建议 props：

- `padding="none" | "sm" | "md"`
- `tone="surface" | "soft" | "warning" | "danger"`
- `selected?: boolean`

默认视觉：

- 白底，浅蓝灰边框，8px 圆角，轻量阴影。
- 不使用玻璃拟态作为默认卡片效果，避免 database 和 clinical 密度不一致。

### Button

职责：

- 替代 `workspace-primary-button`、`workspace-secondary-button`、`workspace-button`、`clinical-reset-button`。

建议 props：

- `variant="primary" | "secondary" | "ghost" | "danger"`
- `size="sm" | "md"`

默认视觉：

- Primary：蓝底白字。
- Secondary：白底蓝/深色文字，浅蓝边框。
- Ghost：透明或浅蓝 hover。
- Danger：仅用于危险操作，不影响主色调。

### Input / Textarea

职责：

- 统一 `database-input`、`database-select`、`workspace-composer-input` 的基础视觉。
- 表单字段业务和校验仍在业务组件中处理。

默认视觉：

- 白底或 `--color-surface-soft`。
- 8px 圆角。
- 蓝色 focus ring。
- 中等密度高度和 padding。

### MessageBubble

职责：

- 替代当前 `workspace-message-bubble clinical-message-bubble` 双 class 混用。
- 统一消息流视觉。

默认形态：

- 全宽或近全宽 clinical 工作台消息，而不是 workspace 的 85% 对话气泡。
- 用户消息白底，AI 消息浅蓝底。
- 左侧 avatar 区作为组件结构，不再依赖 `::before` / `::after` 伪元素画头像。
- Header、thinking disclosure、inline cards 保持现有内容能力。

## 页面迁移

### `/database`

优先迁移，因为玫瑰色硬编码最集中。

迁移结果：

- 使用 `AppShell + TopNav + PanelGrid`。
- 过滤器、自然语言查询、结果表、详情、编辑表单外壳使用 `Card/Button/Input`。
- 保留三栏结构和现有 database hook。
- 移除玫瑰色 button、pill、table selected、distribution bar 等主视觉。

### 医生端 database tab

迁移结果：

- `DoctorDatabaseView` 继续组织 historical case base / patient registry 业务。
- source switch 使用 `Button` 或轻量 segmented control 样式。
- 内部复用 `/database` 同一套 panels 和 `PanelGrid`。
- 外层仍位于 `clinical-app-shell` 语义内，但视觉来自统一 shell/token。

### 患者端/医生端工作台

迁移结果：

- `WorkspacePage` 继续负责 scene/session/SSE/upload/reset/latency/cards 编排。
- `DoctorSceneShell` 继续负责医生端业务布局。
- 外围布局、卡片、按钮、输入框、消息气泡逐步替换为新 UI 组件。
- 患者端和医生端保持同一产品视觉，只通过内容和 nav active state 区分。

## CSS 迁移策略

1. 第一阶段扩展 token，并把旧 `--clinical-*` alias 到新 token。
2. 将 `workspace-*` / `database-*` 中玫瑰色硬编码替换为浅蓝白 token。
3. 新组件样式可先放入 `globals.css` 的清晰分区，使用 `.ui-*` 前缀。
4. 业务组件迁移到新 React 组件后，逐步删除不用的旧视觉规则。
5. 最终 `workspace-*` / `clinical-*` 只保留必要的业务语义或兼容 wrapper，不再承载独立视觉系统。

## 实施拆分

### 阶段 1：Token 基线

修改：

- `frontend/src/styles/tokens.css`
- `frontend/src/styles/globals.css`

内容：

- 新增浅蓝白 token。
- alias 旧 clinical token。
- 替换 workspace/database 玫瑰硬编码。

验收：

- `/database` 视觉主色变为蓝白。
- clinical 页面不明显退化。
- 不改 React 结构。

### 阶段 2：基础 UI 组件

新增：

- `Card`
- `Button`
- `Input`
- `Textarea`
- `MessageBubble`

内容：

- 建立 props 和样式边界。
- 添加 focused tests。

验收：

- 新组件测试通过。
- 不要求页面全面迁移。

### 阶段 3：Shell/Grid 组件

新增：

- `AppShell`
- `TopNav`
- `PanelGrid`

修改：

- `ClinicalTopNav` 包装 `TopNav`。
- `WorkspaceLayout` 包装 `PanelGrid`。

验收：

- 旧页面测试标识保留。
- `clinical-top-nav.test.tsx` 和 workspace layout 相关测试通过。

### 阶段 4：页面迁移

顺序：

1. `/database`
2. `DoctorDatabaseView`
3. `ConversationPanel`
4. 患者/医生工作台外围 cards 和 panels

验收：

- 每迁移一个区域，运行对应 focused tests。
- 三栏结构和业务行为保持。

### 阶段 5：CSS 清理

内容：

- 删除未使用的玫瑰色规则。
- 删除无用 `workspace-*` / `clinical-*` 视觉规则。
- 保留必要兼容类和测试标识。

验收：

- `globals.css` 不再维护两套主视觉。
- grep 不再出现 workspace/database 主视觉玫瑰色硬编码。

## 测试策略

优先保持并运行：

- `frontend/src/pages/workspace-page.test.tsx`
- `frontend/src/features/doctor/doctor-scene-shell.test.tsx`
- `frontend/src/features/chat/conversation-panel.test.tsx`
- `frontend/src/features/database/database-results-table.test.tsx`
- `frontend/src/features/database/use-database-workbench.test.tsx`
- `frontend/src/components/layout/clinical-top-nav.test.tsx`

建议新增：

- `frontend/src/components/ui/button.test.tsx`
- `frontend/src/components/ui/card.test.tsx`
- `frontend/src/components/ui/panel-grid.test.tsx`
- `frontend/src/components/ui/message-bubble.test.tsx`

测试重点：

1. Button variant、disabled 和 type 行为。
2. Card header、padding、tone 和 selected 状态。
3. PanelGrid 三栏、双栏、center-only、left/right collapsed 状态。
4. MessageBubble user/ai tone、avatar/header/content。
5. TopNav disabled item、active item、profile switch 和 status pill。
6. 页面测试中保留关键 `data-testid`。

## 浏览器回归

实现阶段至少检查：

- `/database`
- `/` 患者场景
- `/` 医生场景
- 医生端 database tab

检查点：

1. 页面主视觉为浅蓝白，没有玫瑰色主操作。
2. 三栏信息架构保留。
3. Header 高度和视觉语言一致。
4. Card padding、panel gap、按钮高度、输入框高度一致。
5. 消息区不溢出，用户/AI 消息形态统一。
6. 720px mobile 下布局可用，无文字重叠。
7. focus、hover、disabled、selected 状态清晰。

## 风险与缓解

风险：结构性 React 重构可能破坏既有测试。

缓解：先通过 wrapper 保留旧组件 API 和 `data-testid`，迁移页面时逐步替换。

风险：一次性删除旧 CSS 容易引入视觉回归。

缓解：旧 class 先映射到新 token，清理放最后。

风险：MessageBubble 改结构可能影响 inline cards、thinking disclosure 或滚动行为。

缓解：优先为 `ConversationPanel` 添加/保留 focused tests，再迁移消息组件。

风险：TopNav 合并可能影响患者/医生 scene 切换和医生 database tab。

缓解：`ClinicalTopNav` 先作为兼容包装层，确认测试和浏览器流程稳定后再直接迁移调用方。

风险：浅蓝白 token 改动导致医疗状态色弱化。

缓解：success/warning/danger 保留为状态 token，只限制它们不成为页面主视觉。

## 验收标准

1. `/database`、医生 database tab、患者端、医生端工作台共享浅蓝白主视觉。
2. Workspace/Database 玫瑰色硬编码不再作为主视觉出现。
3. `Card`、`Button`、`Input`、`Textarea`、`MessageBubble`、`PanelGrid`、`TopNav` 有明确组件边界。
4. 三栏信息架构保留，布局只做轻度整理。
5. 字号、间距、圆角主要通过 token 表达，不继续新增碎片值。
6. 现有业务 hooks、API client、SSE、上传、患者库、卡片渲染契约不变。
7. Focused frontend tests 通过。
8. `npm --prefix frontend run build` 通过。
9. 浏览器回归中 desktop 和 mobile 无明显视觉断裂或内容重叠。

## 决策

采用“UI Shell + 基础组件同步收敛”的结构性重构路线。先统一 token 和旧样式主视觉，再抽基础 UI 组件和 shell/grid，最后逐页迁移并清理旧 CSS。目标风格为浅蓝白、中等密度、临床工作台感，保留三栏信息架构和业务行为。
