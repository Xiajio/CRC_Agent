# Apple-Inspired Clinical Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Apple-inspired clinical workspace polish so the doctor consultation page reads as a premium product stage with a lightweight command layer and subdued context rails.

**Architecture:** Keep existing React component boundaries and apply the premium pass through focused design-system changes. `ClinicalEmptyState` becomes the SVG icon host, `ClinicalTopNav` keeps using the shared `TopNav`, and `globals.css` owns the material, layout, rail, stage, composer, and responsive treatments.

**Tech Stack:** React 18, TypeScript, Vite, Vitest, Testing Library, CSS, Playwright/Browser validation.

---

## File Map

- Modify `frontend/src/components/layout/clinical-empty-state.tsx`
  - Add code-native SVG icon rendering for each empty-state icon variant.
  - Keep the existing public props and `data-testid` hooks.
- Modify `frontend/src/components/layout/clinical-empty-state.test.tsx`
  - Add assertions that the icon is SVG-based and keeps the variant class.
- Modify `frontend/src/components/layout/clinical-top-nav.test.tsx`
  - Add tests for command-layer structure if new wrapper classes are added.
- Modify `frontend/src/styles/consultation-cockpit-layout.test.ts`
  - Lock CSS hooks for glass command layer, product stage, subdued rails, SVG icon cleanup, and mobile wrapping.
- Modify `frontend/src/styles/globals.css`
  - Add clinical material tokens.
  - Refine top navigation into a compact translucent command layer.
  - Make the center conversation stage visually dominant.
  - Subdue side rails and inspector panels.
  - Remove pseudo-icon drawing rules from empty states.
  - Refine mobile and tablet behavior.
- Validate through Browser/Chrome Extension or Playwright screenshots.

## Task 1: CSS Contract Red Tests

**Files:**
- Modify: `frontend/src/styles/consultation-cockpit-layout.test.ts`

- [x] **Step 1: Add failing assertions for the Apple-inspired visual system**

Add expectations inside `defines the polished clinical visual system hooks`:

```ts
expect(css).toContain("--clinical-glass-ink");
expect(blockFor(".clinical-top-nav")).toContain("backdrop-filter");
expect(blockFor(".clinical-nav-tabs")).toContain("border-radius: 999px");
expect(blockFor(".clinical-conversation-card")).toContain("var(--clinical-stage-shadow)");
expect(blockFor(".clinical-left-column")).toContain("opacity");
expect(blockFor(".clinical-right-column")).toContain("opacity");
expect(blockFor(".clinical-empty-state-icon svg")).toContain("stroke-width");
expect(blockFor(".clinical-empty-state-icon::before")).toBe("");
```

Add a new test for mobile command-layer wrapping:

```ts
it("keeps the Apple-inspired command layer compact on mobile", () => {
  const media700 = mediaBlockFor("max-width: 700px");

  expect(blockFor(".clinical-top-nav", media700)).toContain("grid-template-columns: 1fr");
  expect(blockFor(".clinical-nav-tabs", media700)).toContain("grid-template-columns");
  expect(blockFor(".clinical-nav-tab", media700)).toContain("min-height");
  expect(blockFor(".clinical-user-area", media700)).toContain("justify-content: flex-start");
});
```

- [x] **Step 2: Run the CSS contract test and confirm red**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/styles/consultation-cockpit-layout.test.ts
```

Expected: FAIL because the new tokens, `backdrop-filter`, SVG icon CSS, and mobile 700px hooks do not exist yet.

## Task 2: SVG Empty-State Icon System

**Files:**
- Modify: `frontend/src/components/layout/clinical-empty-state.tsx`
- Modify: `frontend/src/components/layout/clinical-empty-state.test.tsx`
- Modify: `frontend/src/styles/globals.css`

- [x] **Step 1: Add failing component expectations**

Add this assertion to the first `ClinicalEmptyState` test:

```ts
const icon = screen.getByTestId("clinical-empty-state-icon");
expect(icon.querySelector("svg")).toBeInTheDocument();
expect(icon.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
```

- [x] **Step 2: Run the component test and confirm red**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/components/layout/clinical-empty-state.test.tsx
```

Expected: FAIL because the icon span does not contain an SVG yet.

- [x] **Step 3: Implement SVG icon rendering**

In `clinical-empty-state.tsx`, add a local icon path map and render an inline SVG:

```tsx
const emptyStateIconPaths: Record<ClinicalEmptyStateIcon, ReactNode> = {
  cards: (
    <>
      <rect x="6" y="7" width="12" height="10" rx="2.5" />
      <path d="M9 11h6" />
    </>
  ),
  chat: (
    <>
      <path d="M5 8.5A4.5 4.5 0 0 1 9.5 4h5A4.5 4.5 0 0 1 19 8.5v2A4.5 4.5 0 0 1 14.5 15H11l-4 3v-3.7A4.5 4.5 0 0 1 5 10.5z" />
    </>
  ),
  events: (
    <>
      <circle cx="7" cy="7" r="2" />
      <circle cx="17" cy="8" r="2" />
      <circle cx="10" cy="17" r="2" />
      <path d="M8.8 7.5 15.2 8M8 8.8l1.3 6.2" />
    </>
  ),
  plan: (
    <>
      <path d="M7 7h10M7 12h10M7 17h6" />
    </>
  ),
  references: (
    <>
      <path d="M8 5h8a2 2 0 0 1 2 2v12H8a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" />
      <path d="M9 9h6M9 13h5" />
    </>
  ),
  roadmap: (
    <>
      <circle cx="6.5" cy="7" r="2" />
      <circle cx="17.5" cy="17" r="2" />
      <path d="M8.5 7h4a3 3 0 0 1 0 6h-1a3 3 0 0 0 0 6h4" />
    </>
  ),
  summary: (
    <>
      <circle cx="12" cy="7.5" r="3" />
      <path d="M6.5 19c1.2-3.4 3-5 5.5-5s4.3 1.6 5.5 5" />
    </>
  ),
  uploads: (
    <>
      <path d="M12 17V6" />
      <path d="m8.5 9.5 3.5-3.5 3.5 3.5" />
      <path d="M6 18.5h12" />
    </>
  ),
};
```

Render:

```tsx
<span className={`clinical-empty-state-icon clinical-empty-state-icon-${icon}`} data-testid="clinical-empty-state-icon" aria-hidden="true">
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    {emptyStateIconPaths[icon]}
  </svg>
</span>
```

- [x] **Step 4: Remove pseudo-icon drawing CSS**

In `globals.css`, delete the drawing rules for:

```css
.clinical-empty-state-icon::before
.clinical-empty-state-icon-chat::before
.clinical-empty-state-icon-events::before
.clinical-empty-state-icon-roadmap::before
.clinical-empty-state-icon-plan::before
.clinical-empty-state-icon-references::before
.clinical-empty-state-icon-uploads::before
.clinical-empty-state-icon-summary::before
```

Add:

```css
.clinical-empty-state-icon svg {
  width: 17px;
  height: 17px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.85;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.clinical-empty-state-icon-cards svg,
.clinical-empty-state-icon-summary svg {
  width: 18px;
  height: 18px;
}
```

- [x] **Step 5: Run component and CSS tests**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/components/layout/clinical-empty-state.test.tsx src/styles/consultation-cockpit-layout.test.ts
```

Expected: empty-state component tests pass; CSS test still fails until material and layout CSS is implemented.

## Task 3: Apple Product Stage CSS Pass

**Files:**
- Modify: `frontend/src/styles/globals.css`
- Modify: `frontend/src/styles/consultation-cockpit-layout.test.ts`

- [x] **Step 1: Add material tokens near the clinical app styles**

Add tokens before `.clinical-app-shell`:

```css
:root {
  --clinical-bg: #f4f7fb;
  --clinical-ink: #071b2f;
  --clinical-ink-soft: rgba(7, 27, 47, 0.72);
  --clinical-glass-ink: rgba(7, 27, 47, 0.84);
  --clinical-glass-border: rgba(191, 211, 235, 0.22);
  --clinical-surface: #ffffff;
  --clinical-border: #d9e5f1;
  --clinical-border-soft: rgba(158, 179, 204, 0.28);
  --clinical-stage-shadow: 0 22px 60px rgba(18, 43, 76, 0.12);
  --clinical-panel-shadow: 0 10px 28px rgba(18, 43, 76, 0.07);
  --clinical-control-shadow: 0 8px 24px rgba(4, 22, 43, 0.16);
}
```

- [x] **Step 2: Refine top navigation into the command layer**

Update `.clinical-top-nav` and related classes:

```css
.clinical-top-nav {
  min-height: 56px;
  grid-template-columns: minmax(220px, 320px) minmax(360px, 1fr) auto auto;
  gap: 12px;
  padding: 7px 24px;
  border-bottom: 1px solid var(--clinical-glass-border);
  background: linear-gradient(90deg, rgba(5, 20, 36, 0.92), var(--clinical-glass-ink));
  backdrop-filter: blur(18px) saturate(1.35);
  box-shadow: var(--clinical-control-shadow);
}

.clinical-nav-tabs {
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 40px;
  border: 1px solid rgba(217, 229, 241, 0.16);
  border-radius: 999px;
  padding: 3px;
  background: rgba(255, 255, 255, 0.07);
}

.clinical-nav-tab {
  min-width: 124px;
  height: 32px;
  border-radius: 999px;
  padding: 0 16px;
  font-size: 0.84rem;
  font-weight: 650;
}

.clinical-nav-tab-active {
  background: rgba(255, 255, 255, 0.16);
  box-shadow: inset 0 -2px 0 rgba(68, 201, 255, 0.9), 0 8px 22px rgba(0, 0, 0, 0.14);
}
```

- [x] **Step 3: Refine stage, rail, and panel weight**

Update these classes:

```css
.clinical-app-shell {
  background:
    radial-gradient(circle at 50% -16%, rgba(99, 179, 237, 0.18), transparent 36%),
    var(--clinical-bg);
}

.clinical-card,
.clinical-conversation-card {
  border-color: var(--clinical-border);
  box-shadow: var(--clinical-panel-shadow);
}

.clinical-conversation-card {
  min-height: clamp(470px, 60vh, 680px);
  border-color: rgba(129, 160, 193, 0.34);
  box-shadow: var(--clinical-stage-shadow);
}

.clinical-left-column,
.clinical-right-column {
  opacity: 0.92;
}

.clinical-right-column .clinical-card,
.clinical-left-column .clinical-card {
  box-shadow: 0 8px 22px rgba(18, 43, 76, 0.055);
}
```

- [x] **Step 4: Refine composer and empty-state premium treatment**

Use a quiet input tray and lower-contrast empty state:

```css
.clinical-composer-region {
  background: linear-gradient(180deg, rgba(251, 253, 255, 0.72), #ffffff);
}

.clinical-conversation-card .clinical-composer-textarea {
  min-height: 62px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.94);
}

.clinical-empty-state {
  color: #5d6b7c;
}

.clinical-empty-state-icon {
  background: rgba(237, 245, 255, 0.78);
  color: #0a5aa5;
}
```

- [x] **Step 5: Add mobile command-layer refinements**

Inside the existing mobile media area, or add a `@media (max-width: 700px)` block:

```css
@media (max-width: 700px) {
  .clinical-top-nav {
    grid-template-columns: 1fr;
    gap: 10px;
    padding: 12px 16px 14px;
  }

  .clinical-nav-tabs {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    width: 100%;
    height: auto;
  }

  .clinical-nav-tab {
    min-width: 0;
    min-height: 36px;
    padding: 0 8px;
  }

  .clinical-user-area {
    justify-content: flex-start;
  }
}
```

- [x] **Step 6: Run focused CSS test**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/styles/consultation-cockpit-layout.test.ts
```

Expected: PASS.

## Task 4: Visual QA, Build, And Commit

**Files:**
- No new production files beyond Tasks 1-3.
- Possible screenshot artifacts saved outside repo under `C:\Users\msi\AppData\Local\Temp`.

- [x] **Step 1: Run full frontend tests**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run
```

Expected: `39 passed`, `219 passed` or updated counts with zero failures.

- [x] **Step 2: Run build**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: `tsc --noEmit` and `vite build` complete with exit code 0.

- [x] **Step 3: Start fixture backend and frontend**

Run in background sessions:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\start_backend_fixture.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\start_frontend.ps1
```

Expected:

- Backend listens on `http://127.0.0.1:8000`.
- Frontend listens on `http://127.0.0.1:4173/`.

- [x] **Step 4: Browser validation**

Use Browser plugin first. If Codex in-app browser navigation times out again, use Chrome Extension backend and record the fallback.

Checks:

- Page URL is `http://127.0.0.1:4173/`.
- Title is `LangG Clinical Workspace`.
- DOM contains `会诊`, `医疗卡片待生成`, `执行计划待生成`, and `SSE 连接正常`.
- No framework error overlay.
- Console contains no relevant app errors. React Router v7 future warnings are acceptable.
- Click `多模态`, verify multimodal content appears; click `会诊`, verify consultation content returns.

- [x] **Step 5: Capture screenshots**

Use Playwright with persisted doctor scene:

```powershell
D:\anaconda3\envs\LangG\node.exe -e "const { chromium } = require('playwright'); (async () => { const targets = [{ name: 'desktop', viewport: { width: 1440, height: 900 }, path: 'C:/Users/msi/AppData/Local/Temp/langg-apple-stage-desktop.png' }, { name: 'mobile', viewport: { width: 390, height: 844 }, path: 'C:/Users/msi/AppData/Local/Temp/langg-apple-stage-mobile.png' }]; const browser = await chromium.launch(); for (const target of targets) { const context = await browser.newContext({ viewport: target.viewport, deviceScaleFactor: 1 }); await context.addInitScript(() => localStorage.setItem('langg.workspace.active-scene', 'doctor')); const page = await context.newPage(); await page.goto('http://127.0.0.1:4173/', { waitUntil: 'domcontentloaded' }); await page.getByText('医疗卡片待生成').waitFor({ timeout: 15000 }); await page.screenshot({ path: target.path, fullPage: false }); await context.close(); } await browser.close(); })().catch((error) => { console.error(error); process.exit(1); });"
```

Expected: desktop and mobile screenshots show the command layer, stage, rails, and SVG empty icons without overlap or clipping.

- [x] **Step 6: Diff and whitespace check**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors and only intended files changed.

- [x] **Step 7: Commit**

Run:

```powershell
git add frontend/src/components/layout/clinical-empty-state.tsx frontend/src/components/layout/clinical-empty-state.test.tsx frontend/src/styles/consultation-cockpit-layout.test.ts frontend/src/styles/globals.css
git commit -m "feat: apply apple-inspired clinical workspace polish"
```

Expected: commit succeeds on `clinical-visual-polish`.

## Self-Review

- Spec coverage: command layer, product stage, context rails, empty states, SVG icons, responsive behavior, browser validation, full tests, and build are covered.
- Placeholder scan: no TBD/TODO/fill-in instructions remain.
- Type consistency: `ClinicalEmptyStateIcon` remains the shared union type; no new exported API is required.
- Scope check: this is a single front-end visual-system pass with no backend or route changes.
