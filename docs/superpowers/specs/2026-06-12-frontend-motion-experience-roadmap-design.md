# 前端统一体感系统分阶段路线图设计

## 背景

当前前端并不是从零开始做动效。项目已经有一套克制且可测试的 motion 基础：

- `frontend/src/styles/tokens.css` 定义 CSS motion 变量。
- `frontend/src/components/motion/motion-tokens.ts` 将同一套 token 映射给 GSAP。
- `frontend/src/components/motion/motion-system.test.ts` 校验 CSS token 与 TypeScript adapter 一致，并限制宽泛 transition 与高成本动画属性。
- CSS 已经承担 hover、focus、button、card、highlight pulse 等微交互。
- GSAP 已经通过 `useGsapContext` 被收窄到合适范围：shell 入场、视图切换、SVG 解剖图等，并具备 reduced-motion 门控、作用域隔离和自动 cleanup。

因此本次设计不引入第二套全功能 React 动画库。已确认方向是：**GSAP-only + 现有 CSS-first 微交互体系**。

目标是让全前端共享一套体感语言，同时保留各场景视觉身份：

- 医生端继续是近黑色 clinical cockpit。
- 患者端继续是浅色 assistant 体验。
- 数据库和多模态视图继续保持任务型、可扫描、高密度。

这份 spec 定义一个完整分阶段路线图，优先提升交互顺滑度，再逐步增强流程连续性、医学可视化动效和性能护栏。

## 目标

- 建立全前端统一的 motion 体验系统，覆盖反馈节奏、缓动、入场、状态切换和 reduced-motion 行为。
- 保持 CSS-first 微交互：hover、focus、active、disabled、selected、loading、基础 card 状态继续由 CSS 驱动。
- 只在 CSS 不擅长的场景扩展 GSAP：stagger reveal、SVG path 动画、共享 indicator 移动、panel transition、复杂时序。
- 第一阶段就让用户明显感到 button、tab、panel、card、message 的反馈更顺。
- 保留医生端、患者端、数据库、多模态各自的视觉语言，不把所有页面强行统一成同一种皮肤。
- 保持动效体系可测试、可约束、可维护，符合医疗工具的稳定性要求。

## 非目标

- 不新增 `motion`、`framer-motion` 或其他全功能动画库。
- 不引入 `@gsap/react` 替换现有 hook 体系，除非后续有单独设计决策。
- 不把医生端、患者端、数据库、多模态重做成同一套视觉主题。
- 不修改后端协议、临床数据模型、路由、reducer 或业务流程。
- 不做装饰性整页加载编排、弹跳动画、展示级动效或拖慢任务完成的动画。
- 不全局追加 `backdrop-filter`、glass、glow、blur、SVG filter 等视觉效果。
- 默认不动画 `width`、`height`、`top`、`margin`、`box-shadow`、`stroke-width`、`filter blur` 等布局或绘制成本高的属性。
- 不为了体感统一而大规模重命名现有主题 token、`data-theme` 架构或业务 class 命名。

## 范围

本路线图覆盖 React 前端的主要用户界面：

- 共享 UI primitive：button、card、input、textarea、top nav、segmented control、tab、panel shell。
- 对话体验：message bubble、empty state、composer、streaming 状态、latency/status 文案。
- 医生端 cockpit：三栏会诊、roadmap、execution plan、event stream、anatomy panel。
- 患者端 assistant：assistant home、quick actions、composer、message flow、profile/upload tab。
- 数据库工作台：filter、natural query bar、result table、pagination、detail/edit panel。
- 多模态视图：影像、病理、实验室 panel，分组分析 card，action 状态。
- SVG 医学可视化：结直肠解剖图、whole-body overview、未来 roadmap path。

范围是“全前端统一体感系统”，不是“一次性全前端视觉重做”。路线图必须分阶段推进，Phase 1 应该能作为低风险 PR 独立落地。

## 架构原则

### 1. Token 是唯一来源

`tokens.css` 继续作为 CSS motion 值来源，`motion-tokens.ts` 继续作为 GSAP 适配层。新增 duration、easing、distance、scale、opacity、stagger 等值时，必须同时进入 CSS 与 TS，并由 `motion-system.test.ts` 锁定。

建议 token 方向：

- `feedback`：120-180ms，用于 button、hover、focus ring、chip、小型反馈。
- `highlight`：220-280ms，用于 selected state、pulse、医学区域高亮。
- `transition`：240-320ms，用于 tab、panel、view fade、message 状态变化。
- `enter`：300-380ms，用于 panel 或数据到达后的 list 入场。
- `staggerStep`：16-28ms，用于列表逐项延迟，并且必须限制总时长。
- `fluidEase`：CSS 使用 `cubic-bezier(0.16, 1, 0.3, 1)`，用于更高级的 ease-out。
- `gsapEaseOut`：默认继续使用 `power3.out`；只有实现验证后，才考虑换成 `expo.out` 或注册 `CustomEase`。

token 扩展要克制。目标是稳定词汇，不是建立复杂动画主题对象。

### 2. CSS-first 处理微交互

以下场景继续由 CSS 管理：

- Button hover、active、disabled、loading。
- Input 和 textarea 的 focus-visible。
- Card hover、selected。
- Message bubble 的轻量状态变化。
- Anatomy region 的 hover 和基础 selected 外观。
- Status pill、chip、compact badge。

CSS transition 必须使用 token 化 duration 与 easing。继续禁止 `transition: all`。

### 3. GSAP 只处理高级动效

GSAP 负责需要时序、DOM 测量、SVG 控制或共享元素移动的场景：

- `useStaggerReveal`：受限列表、card、message 批量入场。
- `useTabIndicatorMotion` 或窄范围 `useFlipLayout`：tab indicator / panel indicator 移动。
- SVG `stroke-dasharray` / `stroke-dashoffset` 路径动画。
- 多个 anatomy path 同时激活时的高亮时序。
- 超出简单 CSS transition 的 route / panel reveal。

新增 GSAP hook 默认必须走现有 `useGsapContext`，继承 reduced-motion、scope、cleanup 和测试约束。

### 4. 统一体感，保留场景视觉

Motion 系统统一，材质和视觉主题不被压平：

- 医生端可以保留暗色 surface、冷蓝 focus、轻量医学 glow。
- 患者端可以保留浅色 surface、清晰边界、直接 assistant 感。
- 数据库和多模态可以保留高密度、稳定、可扫描的任务型布局。

同一类交互应有相同节奏和物理感，但视觉材料继续通过现有 theme token 分场景表达。

### 5. 动效不能隐藏内容

内容必须先可见，再增强。Reveal 动画只能增强已渲染内容，不能依赖某个 class 或 effect 成功执行后内容才出现。这样可以保护 hidden tab、headless test、reduced motion 和低性能设备。

### 6. 可访问性是硬约束

Reduced motion 不是可选项。键盘 focus 必须可见。动画不能成为表达状态的唯一方式。loading、completed、warning、blocked、error 等状态必须有文本或语义视觉反馈。

## CSS 与 GSAP 分工

| 交互 | 归属 | 说明 |
| --- | --- | --- |
| Button hover / press / disabled | CSS | 只动 transform、opacity、color、background 等低风险属性。 |
| Focus-visible | CSS | reduced motion 下仍然清晰可见。 |
| Card hover / selected | CSS | 使用 token-backed 材质变化，保持克制。 |
| Message append reveal | CSS 或 GSAP | 单条消息可 CSS；批量或列表入场用 GSAP。 |
| Tab active 颜色 | CSS | 文本、背景、状态色仍归 CSS。 |
| Tab indicator 移动 | GSAP Flip 或 transform 测量 | 能滑动就不要 fade out / fade in。 |
| Panel route / view transition | 现有 `useViewTransition` / GSAP | 必须 clear props，并尊重 reduced motion。 |
| Roadmap status dot | CSS 基础状态，GSAP 时序增强 | 不对每个 backend tick 做持续动画。 |
| Roadmap 连接线 | 后续 GSAP SVG path | 用 dashoffset，不动画布局伪元素。 |
| Anatomy hover / selected | CSS 基础，GSAP pulse sequence | 避免动画 `strokeWidth`，优先 transform、opacity、halo。 |
| Progress bar | CSS | shimmer 必须 reduced-motion-safe，且默认不持续吸引注意。 |
| Streaming 文本节奏 | 性能层 | 后续考虑 rAF 合帧，不做逐字符装饰动效。 |

## Phase 1：交互反馈统一

### 目标

通过统一 timing、easing 和反馈行为，让全前端立刻感觉更顺、更一致，但不新增依赖、不改业务逻辑。

### 工作内容

- 只在必要时扩展 motion token：`feedback`、`transition`、`enter`、`staggerStep`。
- 更新 CSS 与 TS token mirror 测试。
- 审计 button、tab、input、card、message、composer、panel 中分散的 duration / easing。
- 将常见 transition 统一为 token-backed 写法。
- 清理已知 `transition: all` 债务，尤其是 TSX inline style 中的 broad transition，例如 registry / recent patients 这类局部 hover 实现。
- 在结构允许的地方，引入统一 tab indicator 动效，而不是靠 active block 的突变或重挂载。
- message 和 panel 入场优先复用现有 `useShellReveal` / `useViewTransition`，或使用简单 CSS transition。
- 更新或新增测试，证明：
  - CSS 与 TS motion token 一致。
  - CSS 与 TSX inline style 中都没有新增 `transition: all`。
  - GSAP 动画源不包含高成本动画属性。
  - reduced motion 继续通过 `useGsapContext` 生效。

### 验收标准

- 常见交互 transition 不再使用临时 duration / easing。
- Button、tab、panel、message reveal、card feedback 在医生端、患者端、数据库、多模态中节奏一致。
- 没有新增动画依赖。
- 各场景视觉身份不被改变。
- 前端测试和 build 继续通过。

## Phase 2：扩展 GSAP hook 族

### 目标

增加少量可复用 GSAP hook，用于高级体感，同时延续现有 scoped、reduced-motion-safe 架构。

### 拟新增 hook

`useStaggerReveal`

- 用途：受限列表、新插入 card/message group 入场。
- 输入：容器 ref、item selector、deps、可选 max items、可选 stagger step。
- 行为：只用 `fromTo` 控制 opacity 与 transform；duration/ease 来自 `motionTokens`。
- 护栏：限制总时长，避免长列表慢慢“演完”。

`useTabIndicatorMotion`

- 用途：active tab indicator 平滑移动，而不是消失再出现。
- 输入：tab 容器 ref、active key、active tab selector、indicator selector。
- 行为：优先 transform-based movement；必要时使用 GSAP Flip。
- 护栏：indicator 不拦截点击，reduced motion 下直接到位。

`useSvgPathReveal`

- 用途：roadmap line 或医学 path 高亮绘制。
- 输入：path ref 或 selector、active state、duration token。
- 行为：根据 path length 设置 dasharray/dashoffset，再动画 dashoffset。
- 护栏：reduced motion 下有静态可见 fallback。

`useMotionTimeline`（可选，后置）

- 用途：为单一 bounded component 组合小型时序。
- 限制：不能发展成整页 choreography API。只有多个组件重复同一 timeline 模式时才引入。

### 测试契约

`motion-system.test.ts` 应扩展对新 hook 源码的扫描：

- 新 hook 必须使用 `useGsapContext`。
- 新 hook 必须引用 `motionTokens`。
- 源码不得包含 `boxShadow`、`strokeWidth`、`filter`、`width`、`height`、`top`、`margin` 或类似布局动画模式。
- hook 必须通过现有 context pathway 完成 cleanup。

### 验收标准

- 至少一个 list / card / message reveal 使用 `useStaggerReveal`。
- 至少一个 tab 或 segmented indicator 使用共享 indicator motion。
- 新 hook 均被契约测试覆盖。
- Reduced-motion 用户获得静态或瞬时状态变化。
- 如果使用 GSAP Flip，插件注册必须集中在 motion hook 或 motion system 入口，不允许在多个组件内重复注册。
- 不引入 `@gsap/react`；现阶段继续复用项目已有 `useGsapContext`。

## Phase 3：流程与医学可视化增强

### 目标

在 motion 基础稳定后，用新 hook 提升 workflow 和医学视觉连续性，但不增加临床任务噪音。

### Roadmap panel

- 保持 roadmap 状态语义不变。
- 在有实际收益时，用 SVG path 替换或补充静态 vertical pseudo-element line。
- active step 变化时，使用 `stroke-dashoffset` 表达进度沿路径推进。
- completed、in_progress、waiting、blocked、skipped 等状态必须继续有文本可见。
- reduced motion 下直接渲染静态 active path segment。

### Execution plan

- 新生成 plan rows 可以使用轻量 stagger reveal。
- 不对每一次 backend 状态 tick 做 row 动画。
- blocked / error 需要通过颜色、标签、图标或状态文本可见，不能只靠动效。

### Anatomy SVG

- 保留现有可访问 path button。
- hover / selected 基础态继续由 CSS 管理。
- active region 变化时可以加入 GSAP 时序：
  - 轻微 scale / opacity pulse；
  - selected region 背后的静态或短时 halo path；
  - selected path 的短时 dash reveal。
- 不动画 `strokeWidth`、SVG filter 或持续发光 blur。

### Database 与 multimodal

- 动效只用于表达状态变化：
  - filter drawer / panel transition；
  - result refresh reveal；
  - progress / loading 状态 polish。
- 表格必须稳定，不做影响扫描的 row motion。
- 多模态分析结果批量出现时，可以使用 bounded stagger。

### Patient assistant

- 保持浅色 assistant 的直接、快速。
- quick action press、tab switch、empty-to-chat transition 使用统一 motion token。
- 不把医生端暗色 glow 或 dashboard 编排带到患者端。

### 验收标准

- Roadmap 和 anatomy 动效提升连续性，但不改变临床语义。
- Database 和 multimodal 动效不降低扫描效率。
- 患者端和医生端共享节奏，但视觉身份仍然不同。
- 视觉回归截图中没有重叠、内容跳动、隐藏文本。

## Phase 4：性能护栏

### 目标

保护增强后的体感系统，避免高频渲染、无意义 reflow 和昂贵 compositing 破坏流畅度。

### SSE 渲染

当前 reducer 会在 `message.delta` 到达时立即 append 文本。后续性能阶段应在 SSE callback / page 层评估 rAF 合帧，让 React state update 对齐显示帧。它是性能护栏，不是逐字符动画功能。

要求：

- 只对 active streaming message 的文本 delta 做合帧。
- 保持事件顺序和最终 `message.done` 内容正确。
- `card.upsert`、error、safety alert、completion state 不应被无理由延迟超过一帧。
- 增加高频 delta 与最终内容一致性的测试。

### Layout containment

三栏 cockpit 可以评估 containment，但不应默认使用 `contain: strict`，因为它包含 size containment，容易破坏自适应高度。优先评估：

```css
contain: layout paint style;
```

只在 side rail 或稳定 panel 上应用，并验证高度、overflow、sticky 行为和响应式堆叠。

### Compositing budget

- `will-change` 只在动画期间使用，结束后清理。
- 不扩大永久 `backdrop-filter` 使用范围。
- 不使用持续 glow、blur 或 filter 动画。
- 优先使用 transform / opacity。

### 验收标准

- 高频 streaming 不导致 side rail 可见 jank。
- 新 GSAP hook 会清理临时 animation props。
- Reduced-motion 和低性能设备获得稳定 UI。
- 浏览器验证覆盖 desktop、tablet、mobile。

## 组件影响

### Shared UI

- `Button`、`Card`、`Input`、`Textarea`、`TopNav` 和 tab-like controls 使用共享 motion token。
- 组件 API 不应为了装饰性动效增加 props。
- feature component 不得自定义局部 duration / easing。

### Conversation panel

- Message 出现应轻量、快速。
- Streaming 文本不做逐字符动画。
- Composer send button 和 textarea focus 使用全局反馈节奏。
- Latency / status label 可以轻量 transition，但不能延迟显示。

### Doctor cockpit

- 现有近黑色视觉 polish 保留。
- Roadmap、execution plan、anatomy 是后续 GSAP 增强的最高价值区域。
- 中间对话频繁更新时，三栏布局必须保持稳定。

### Patient assistant

- 保持浅色 assistant 的清爽直接。
- Motion 服务于 quick action、tab switch、empty-to-chat transition。
- 避免引入医生端 glow、暗色 cockpit 质感或 dashboard 编排。

### Database

- Motion 用于表达状态变化，不装饰表格。
- Filtering、query execution、result refresh、detail panel transition 应快速且稳定。
- 表格 row 不应在用户扫描时产生不可预期移动。

### Multimodal

- 新分析 card 批量可用时，可以 bounded stagger 入场。
- action button、disabled、loading 状态共享反馈 token。
- 影像、病理、实验室 panel 优先保证检查清晰度。

## 可访问性与 reduced motion

- GSAP 默认通过 `useGsapContext` 接入，因为它已经咨询 `usePrefersReducedMotion`。
- CSS keyframes 必须有明确 `@media (prefers-reduced-motion: reduce)` 兜底。
- active、selected、completed、blocked、warning、error 状态在无动画时仍必须可见。
- 键盘 focus 不能被 motion-only affordance 替代。
- 移动端不得因为动效产生文字重叠、内容遮挡或布局抖动。

## 测试与验证

### 单元与契约测试

- 每新增一个 motion hook，都扩展 `frontend/src/components/motion/motion-system.test.ts`。
- 保持 `tokens.css` 与 `motion-tokens.ts` 的 mirror 断言。
- 保持禁止 broad transition 和高成本 GSAP 属性的断言。
- 增加 TSX 源码扫描，禁止 inline style 中出现 `transition: "all ..."` 或等价 broad transition。
- 如果 motion 改动影响 class contract 或渲染结构，补组件测试。

### 浏览器验证

需要通过 Playwright 或 Browser 检查：

- 医生端 cockpit desktop 和 mobile。
- 患者端 assistant empty 与 active chat。
- 数据库 filter、result refresh、detail panel。
- 多模态 populated cards。
- Anatomy active-region selection。
- Roadmap waiting、in_progress、completed、blocked 状态。

### 构建与回归

每个实现阶段至少需要：

```powershell
npm --prefix frontend run test -- --run
npm --prefix frontend run build
```

Phase 3 之后应补 focused visual e2e 或人工截图检查。

## 风险与护栏

### 风险：新增 Motion 形成第三套范式

不新增 `motion` / `framer-motion`。当前项目已经有 CSS token、GSAP adapter 和契约测试。第二套完整动画库会削弱既有架构边界。

### 风险：医疗工具被过度编排

动效应帮助用户理解状态变化，不应变成整页表演、拖慢日常任务或抢走临床内容注意力。

### 风险：SVG 效果变成绘制负担

避免动态 SVG filter blur / drop-shadow。优先使用静态 halo、opacity、transform、dashoffset。

### 风险：Containment 破坏布局

不要广泛使用 `contain: strict`。任何 containment 都必须跨 desktop、tablet、mobile 验证。

### 风险：Stagger 拖慢长列表

Stagger 必须有总时长上限。数据库结果、长 event list 不应逐行慢速入场。

### 风险：Reveal 隐藏内容

默认内容必须先可见，即使动画没有运行也不能出现空白。

## 验收标准

- 全前端使用一套 motion token 词汇描述 timing、easing、distance、scale、stagger。
- CSS 继续负责常见微交互。
- GSAP 继续负责高级时序、SVG path 和共享 indicator motion。
- 不新增全功能动画依赖。
- 新 GSAP hook 使用 `useGsapContext`，并由契约测试覆盖。
- `frontend/package.json` 不新增 `motion`、`framer-motion`、`@gsap/react` 或第二套动画框架依赖。
- CSS 与 TSX 中不存在未解释的 `transition: all`。
- 新增 GSAP 动画卸载后不残留影响布局或可见性的 inline style。
- CSS 和 GSAP 路径都尊重 reduced motion。
- Button、tab、panel、card、composer、message 在医生端、患者端、数据库、多模态中体感一致。
- 医生端和患者端视觉身份仍然清晰不同。
- Roadmap 和 anatomy 增强不改变临床语义。
- Streaming 文本不做逐字符动画。
- 每个实现阶段前端测试、build 和浏览器验证通过。

## 已确认决策

- Motion 架构：**GSAP-only + 现有 CSS-first 微交互**。
- 依赖策略：**不引入 Motion / Framer Motion**。
- React 集成策略：**不引入 `@gsap/react`，继续沿用现有 `useGsapContext`**。
- 第一阶段重点：**交互反馈更顺**。
- 动效强度：**高级感更强，但仍受医疗工具边界约束**。
- 范围：**全前端统一体感系统，保留场景视觉**。
- 路线：**完整分阶段 roadmap，不做一次性大改**。

## 待实施前确认的细节

- Tab indicator 第一版使用 CSS pseudo-element + measured transform，还是直接使用 GSAP Flip。
- 是否需要新增 `staggerStep` / `enterYSmall` token，还是先复用现有 `enterY` 与 duration。
- Message append 第一阶段先用 CSS class，还是等 Phase 2 的 `useStaggerReveal` / `useMessageReveal`。
- GSAP Flip 的注册位置放在具体 hook 文件，还是集中到 motion system 入口。
- Registry / recent patients 中的 inline broad transition 是否作为 Phase 1 前置清理任务。
- 医学 SVG path reveal 是否坚持原生 `strokeDashoffset`，不引入 DrawSVG 等额外插件。

## 后续实施计划提示

用户审阅通过后，应将本设计转换为 implementation plan。计划应按 phase 拆分，并保证 Phase 1 足够窄，可以作为一个 PR 落地。

建议第一轮实施任务：

1. 扩展或整理 motion token 与 mirror 测试。
2. 审计并统一 CSS 微交互 duration / easing。
3. 新增窄范围 `useStaggerReveal` hook 与测试。
4. 仅在结构低风险的位置新增 tab indicator motion。
5. 浏览器验证医生端、患者端、数据库、多模态关键页面。

后续计划再单独覆盖 SVG roadmap、anatomy 增强和 SSE / rendering 性能护栏。
