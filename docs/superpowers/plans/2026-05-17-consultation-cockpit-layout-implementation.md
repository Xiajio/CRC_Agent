# Consultation Cockpit Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the doctor consultation view read as a consultation cockpit, with the conversation area as the primary working surface on desktop and narrow screens.

**Architecture:** Keep React component boundaries intact and enforce the layout contract at the stylesheet level. Add one focused CSS contract test that reads `globals.css`, then update only the clinical dashboard rules needed for desktop proportion, responsive ordering, and event-stream density.

**Tech Stack:** React 18, Vite, TypeScript, Vitest, Tailwind-compatible global CSS.

---

### Task 1: Add Consultation Cockpit CSS Contract

**Files:**
- Create: `frontend/src/styles/consultation-cockpit-layout.test.ts`
- Read: `frontend/src/styles/globals.css`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/styles/consultation-cockpit-layout.test.ts` with a CSS contract that asserts the consultation center uses `minmax(640px, 1fr)`, the left/right support rails are compact clamps, the medium desktop right rail is `340px`, stacked layout starts with `"center"`, and the event stream uses compact row rules.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/styles/consultation-cockpit-layout.test.ts
```

Expected: FAIL because current `globals.css` still uses a 500px right rail, 420px medium right rail, left-first stacked order, and a taller event stream.

### Task 2: Implement CSS Layout Pass

**Files:**
- Modify: `frontend/src/styles/globals.css`
- Test: `frontend/src/styles/consultation-cockpit-layout.test.ts`

- [ ] **Step 1: Update desktop cockpit grid**

Change `.clinical-dashboard` to use:

```css
grid-template-columns: clamp(260px, 18vw, 280px) minmax(640px, 1fr) clamp(340px, 24vw, 370px);
gap: 10px;
padding: 10px;
```

- [ ] **Step 2: Make consultation the dominant working surface**

Change `.clinical-conversation-card` to:

```css
min-height: clamp(430px, 56vh, 620px);
```

- [ ] **Step 3: Compact the event stream**

Change event stream rules to use `min-height: 0`, a four-column compact row on desktop, smaller chip headers, smaller chip body padding, and `max-height: 3.2em`.

- [ ] **Step 4: Adjust breakpoints**

At `max-width: 1450px`, use `280px minmax(580px, 1fr) 340px`. At `max-width: 1150px`, stack `"center"` before `"left"`, `"right"`, and `"event"`. At `max-width: 720px`, keep the conversation minimum at `420px` and collapse event chips to one column.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test -- --run src/styles/consultation-cockpit-layout.test.ts src/features/doctor/doctor-scene-shell.test.tsx
```

Expected: PASS, or PASS tests with a known cache-permission failure that must be rerun with an approved cache-writing path.

### Task 3: Build And Browser QA

**Files:**
- Read: `frontend/package.json`
- Validate rendered route: doctor consultation screen at `http://127.0.0.1:4173/`

- [ ] **Step 1: Run production build**

Run:

```powershell
$env:PATH='D:\anaconda3\envs\LangG;' + $env:PATH; cmd /c D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

Expected: exit 0.

- [ ] **Step 2: Start backend fixture and frontend preview**

Use the repo scripts in this worktree and verify backend on `127.0.0.1:8000`, frontend on `127.0.0.1:4173`.

- [ ] **Step 3: Verify with Browser / Chrome Extension**

The flow under test is: app loads -> doctor consultation cockpit renders -> primary visible controls respond without runtime errors.

Check page identity, nonblank DOM, no framework overlay, console health, and one interaction such as changing a top tab or using an existing consultation control.

- [ ] **Step 4: Capture visual evidence**

Use Browser screenshot first. If Chrome Extension screenshot times out, record that blocker and use the project Playwright path as fallback for screenshots while keeping Browser DOM/console verification as the primary browser check.

- [ ] **Step 5: Review diff and commit**

Run:

```powershell
git diff -- frontend/src/styles/globals.css frontend/src/styles/consultation-cockpit-layout.test.ts
git status --short
git add frontend/src/styles/globals.css frontend/src/styles/consultation-cockpit-layout.test.ts docs/superpowers/plans/2026-05-17-consultation-cockpit-layout-implementation.md
git commit -m "feat: emphasize consultation cockpit layout"
```

Expected: commit contains only the cockpit CSS pass, contract test, and implementation plan.
