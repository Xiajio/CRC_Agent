# Clinical Visual Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Polish the clinical workspace visual system with a reusable empty state, refined top navigation, and quieter card/composer styling.

**Architecture:** Add one shared presentation component under `components/layout`, wire it into existing empty-state call sites, and keep business data flow unchanged. Use CSS contract tests to pin key visual-system selectors while existing component tests protect rendering behavior.

**Tech Stack:** React 18, TypeScript, Vitest, Testing Library, global CSS.

---

### Task 1: Empty State Component

**Files:**
- Create: `frontend/src/components/layout/clinical-empty-state.tsx`
- Create: `frontend/src/components/layout/clinical-empty-state.test.tsx`

- [ ] **Step 1: Write failing tests**

Add tests that expect `ClinicalEmptyState` to render a title, message, icon tone class, compact class, and optional action button.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/components/layout/clinical-empty-state.test.tsx
```

Expected: fail because the component does not exist.

- [ ] **Step 3: Implement the component**

Create a small typed component with `title`, `message`, `icon`, `compact`, `actionLabel`, and `onAction`.

- [ ] **Step 4: Verify test passes**

Run the same test command and expect pass.

### Task 2: Adopt Empty States

**Files:**
- Modify: `frontend/src/features/cards/clinical-cards-panel.tsx`
- Modify: `frontend/src/features/chat/conversation-panel.tsx`
- Modify: `frontend/src/features/doctor/doctor-scene-shell.tsx`
- Modify: `frontend/src/features/roadmap/roadmap-panel.tsx`
- Modify: `frontend/src/features/execution-plan/execution-plan-panel.tsx`
- Test: existing component tests for these modules

- [ ] **Step 1: Add failing assertions**

Update existing tests to expect `data-testid="clinical-empty-state"` where cards, roadmap, execution plan, and conversation empty states render.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/components/layout/clinical-empty-state.test.tsx src/features/cards/clinical-cards-panel.test.tsx src/features/roadmap/roadmap-panel.test.tsx src/features/execution-plan/execution-plan-panel.test.tsx src/features/chat/conversation-panel.test.tsx
```

Expected: fail until call sites use the shared component.

- [ ] **Step 3: Wire the component into empty branches**

Replace plain `<p className="clinical-empty-note">` and medical-card empty wrappers with `ClinicalEmptyState`, preserving existing visible message text.

- [ ] **Step 4: Verify focused tests pass**

Run the same focused test command and expect pass.

### Task 3: CSS Visual Polish

**Files:**
- Modify: `frontend/src/styles/globals.css`
- Modify: `frontend/src/styles/consultation-cockpit-layout.test.ts`

- [ ] **Step 1: Add failing CSS contract assertions**

Assert that CSS includes `.clinical-empty-state`, `.clinical-empty-state-icon`, `.clinical-empty-state-compact`, refined top-nav active state, and softer clinical card shadow token.

- [ ] **Step 2: Run CSS test to verify it fails**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/styles/consultation-cockpit-layout.test.ts
```

Expected: fail until CSS is added.

- [ ] **Step 3: Implement CSS polish**

Update clinical card, panel header, top nav, status pill, reset button, empty-state, and composer styles.

- [ ] **Step 4: Verify CSS test passes**

Run the CSS test command and expect pass.

### Task 4: Verification

**Files:**
- Read: `frontend/package.json`

- [ ] **Step 1: Run focused tests**

Run the combined focused test command from Task 2 plus the CSS contract test.

- [ ] **Step 2: Run production build**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

- [ ] **Step 3: Browser QA**

Start backend fixture and frontend, verify the consultation screen in Browser/Chrome Extension, then capture desktop and mobile screenshots. The target flow is app loads -> doctor consultation cockpit renders polished empty states/top nav -> scene tabs remain interactive.

- [ ] **Step 4: Commit**

Commit the implementation on `clinical-visual-polish`.
