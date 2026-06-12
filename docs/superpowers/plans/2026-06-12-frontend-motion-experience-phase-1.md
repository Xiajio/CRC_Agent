# Frontend Motion Experience Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first PR-sized slice of the frontend motion roadmap: token-backed interaction feedback, no new animation runtime, no broad transitions, and removal of the known recent-patients inline motion debt.

**Architecture:** Keep the existing CSS-first + GSAP-only architecture. CSS continues to own hover/focus/button/card/table feedback, while GSAP remains scoped through `useGsapContext`. This plan changes tokens, contract tests, CSS transitions, and one local TSX inline-style debt point; it does not add new hooks, dependencies, business logic, route changes, or visual theme rewrites.

**Tech Stack:** React 18, TypeScript, Vite, Vitest, Testing Library, Tailwind CSS variables, existing GSAP 3.12.7.

---

## Scope Check

The design spec is a multi-phase roadmap. This plan intentionally implements **Phase 1: Interaction Feedback Foundation** only. Later phases should get separate plans:

- Phase 2: `useStaggerReveal`, `useTabIndicatorMotion`, and GSAP hook expansion.
- Phase 3: roadmap SVG path, anatomy SVG enhancement, workflow visualization.
- Phase 4: SSE rAF batching and layout containment.

This Phase 1 plan produces testable software on its own: token-backed feedback timing, stricter contracts, and removal of the known `transition: all` debt.

## File Map

- Modify: `frontend/src/components/motion/motion-system.test.ts`
  - Adds dependency guard, exact easing expectation, and CSS ad hoc timing guard.
- Modify: `frontend/src/styles/style-architecture-contract.test.ts`
  - Adds TSX inline broad-transition scan.
- Modify: `frontend/src/styles/tokens.css`
  - Updates CSS ease token to the approved fluid curve.
- Modify: `frontend/src/components/motion/motion-tokens.ts`
  - Mirrors the CSS ease token in TypeScript.
- Modify: `frontend/src/styles/globals.css`
  - Replaces hardcoded `160ms ease` / `0.2s ease` transitions with motion tokens.
  - Adds token-backed recent-patients classes.
- Modify: `frontend/src/features/patient-registry/recent-patients-panel.tsx`
  - Removes inline `transition: "all 0.2s ease"` and other local visual material styles from the preview button.
- Create: `frontend/src/features/patient-registry/recent-patients-panel.test.tsx`
  - Locks preview button behavior and class contract.

## Task 1: Add Motion Architecture Red Tests

**Files:**
- Modify: `frontend/src/components/motion/motion-system.test.ts`
- Modify: `frontend/src/styles/style-architecture-contract.test.ts`

- [ ] **Step 1: Add dependency and ad hoc CSS timing guards**

In `frontend/src/components/motion/motion-system.test.ts`, add this constant after the existing source constants:

```ts
const packageJson = JSON.parse(
  readFileSync(resolve(process.cwd(), "package.json"), "utf8"),
) as {
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
};
```

Add these helper functions after `blockFor`:

```ts
function dependencyVersion(name: string) {
  return packageJson.dependencies?.[name] ?? packageJson.devDependencies?.[name] ?? null;
}

function matchingLines(source: string, regex: RegExp, fileLabel: string) {
  const lineRegex = new RegExp(regex.source, regex.flags.replace(/g/g, ""));
  return source
    .split(/\r?\n/)
    .map((line, index) => ({ line, number: index + 1 }))
    .filter(({ line }) => lineRegex.test(line))
    .map(({ line, number }) => `${fileLabel}:${number}: ${line.trim()}`);
}
```

Add these tests inside `describe("motion design system", () => { ... })`:

```ts
it("keeps GSAP as the only JavaScript animation runtime", () => {
  expect(dependencyVersion("gsap")).toBe("3.12.7");
  expect(dependencyVersion("motion")).toBeNull();
  expect(dependencyVersion("framer-motion")).toBeNull();
  expect(dependencyVersion("@gsap/react")).toBeNull();
});

it("keeps globals.css transition timing token-backed", () => {
  expect(
    matchingLines(
      globalsCss,
      /\b(?:0\.2s|160ms|240ms)\s+ease\b/,
      "src/styles/globals.css",
    ),
  ).toEqual([]);
});
```

Also add this exact easing expectation to the existing `"keeps CSS motion tokens mirrored..."` test after the `cssToken("motion-ease-out")` mirror assertion:

```ts
expect(cssToken("motion-ease-out")).toBe("cubic-bezier(0.16, 1, 0.3, 1)");
```

- [ ] **Step 2: Add TSX inline broad-transition scan**

In `frontend/src/styles/style-architecture-contract.test.ts`, add this helper after `inlineStyleReferences`:

```ts
function inlineBroadTransitionReferences(source: string, fileLabel: string) {
  return matchingLines(source, /\btransition\s*:\s*["'`]all\b/i, fileLabel);
}
```

Add this test inside `describe("style architecture contract", () => { ... })`, before the card renderer inline-style test:

```ts
it("keeps broad inline transitions out of TSX", () => {
  const offenders = nonTestTsxFiles().flatMap((file) => inlineBroadTransitionReferences(read(file), file));

  expect(offenders).toEqual([]);
});
```

- [ ] **Step 3: Run focused tests and confirm red**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/motion/motion-system.test.ts src/styles/style-architecture-contract.test.ts
```

Expected: FAIL.

Expected failures:

- `motion-ease-out` is still `cubic-bezier(0.22, 1, 0.36, 1)`.
- `globals.css` still contains `160ms ease` and `0.2s ease`.
- `recent-patients-panel.tsx` still contains `transition: "all 0.2s ease"`.

- [ ] **Step 4: Commit red tests**

Run:

```powershell
git add frontend/src/components/motion/motion-system.test.ts frontend/src/styles/style-architecture-contract.test.ts
git commit -m "test: lock frontend motion architecture"
```

Expected: commit succeeds with only test files staged.

## Task 2: Update Motion Ease Token

**Files:**
- Modify: `frontend/src/styles/tokens.css`
- Modify: `frontend/src/components/motion/motion-tokens.ts`
- Test: `frontend/src/components/motion/motion-system.test.ts`

- [ ] **Step 1: Update CSS motion easing**

In `frontend/src/styles/tokens.css`, replace:

```css
--motion-ease-out: cubic-bezier(0.22, 1, 0.36, 1);
```

with:

```css
--motion-ease-out: cubic-bezier(0.16, 1, 0.3, 1);
```

- [ ] **Step 2: Mirror the easing in TypeScript**

In `frontend/src/components/motion/motion-tokens.ts`, replace:

```ts
easeOut: "cubic-bezier(0.22, 1, 0.36, 1)",
```

with:

```ts
easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
```

- [ ] **Step 3: Run the motion token test**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/motion/motion-system.test.ts
```

Expected: still FAIL only because `globals.css` has hardcoded `160ms ease` / `0.2s ease`. The token mirror and exact easing assertions should pass.

- [ ] **Step 4: Commit token update**

Run:

```powershell
git add frontend/src/styles/tokens.css frontend/src/components/motion/motion-tokens.ts
git commit -m "style: update frontend motion ease token"
```

Expected: commit succeeds.

## Task 3: Tokenize Existing CSS Transitions

**Files:**
- Modify: `frontend/src/styles/globals.css`
- Test: `frontend/src/components/motion/motion-system.test.ts`

- [ ] **Step 1: Tokenize base card transitions**

In `frontend/src/styles/globals.css`, replace the `.ui-card` transition block with:

```css
.ui-card {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-surface);
  box-shadow: var(--shadow-card);
  transition:
    border-color var(--motion-duration-feedback) var(--motion-ease-out),
    box-shadow var(--motion-duration-feedback) var(--motion-ease-out);
}
```

- [ ] **Step 2: Tokenize shared button transitions**

In `.ui-button`, replace:

```css
transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
```

with:

```css
transition:
  background-color var(--motion-duration-feedback) var(--motion-ease-out),
  border-color var(--motion-duration-feedback) var(--motion-ease-out),
  color var(--motion-duration-feedback) var(--motion-ease-out),
  box-shadow var(--motion-duration-feedback) var(--motion-ease-out);
```

- [ ] **Step 3: Tokenize clinical panel card transitions**

In the shared block for `.ui-card-clinical-panel, .clinical-card, .clinical-conversation-card`, replace the transition block with:

```css
transition:
  border-color var(--motion-duration-feedback) var(--motion-ease-out),
  box-shadow var(--motion-duration-feedback) var(--motion-ease-out);
```

- [ ] **Step 4: Tokenize medical card transitions**

In `.clinical-medical-card`, replace the transition block with:

```css
transition:
  border-color var(--motion-duration-feedback) var(--motion-ease-out),
  box-shadow var(--motion-duration-feedback) var(--motion-ease-out);
```

- [ ] **Step 5: Tokenize event console toggle transitions**

In `.clinical-event-console-toggle`, replace:

```css
transition: border-color 160ms ease, color 160ms ease;
```

with:

```css
transition:
  border-color var(--motion-duration-feedback) var(--motion-ease-out),
  color var(--motion-duration-feedback) var(--motion-ease-out);
```

- [ ] **Step 6: Run the motion test**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/motion/motion-system.test.ts
```

Expected: PASS for `motion-system.test.ts`.

- [ ] **Step 7: Commit CSS transition tokenization**

Run:

```powershell
git add frontend/src/styles/globals.css
git commit -m "style: tokenize frontend transition timing"
```

Expected: commit succeeds.

## Task 4: Remove Recent Patients Inline Motion Debt

**Files:**
- Create: `frontend/src/features/patient-registry/recent-patients-panel.test.tsx`
- Modify: `frontend/src/features/patient-registry/recent-patients-panel.tsx`
- Modify: `frontend/src/styles/globals.css`
- Test: `frontend/src/features/patient-registry/recent-patients-panel.test.tsx`
- Test: `frontend/src/styles/style-architecture-contract.test.ts`

- [ ] **Step 1: Add a focused component test**

Create `frontend/src/features/patient-registry/recent-patients-panel.test.tsx`:

```tsx
import "@testing-library/jest-dom/vitest";

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { PatientRegistryItem } from "../../app/api/types";
import { RecentPatientsPanel } from "./recent-patients-panel";

const items: PatientRegistryItem[] = [
  {
    patient_id: 101,
    status: "active",
    updated_at: "2026-06-12T09:00:00Z",
    tumor_location: "直肠",
    clinical_stage: "cT3N1M0",
    mmr_status: "pMMR",
  },
  {
    patient_id: 202,
    status: "active",
    updated_at: "2026-06-12T10:00:00Z",
  },
];

function renderPanel(overrides: Partial<React.ComponentProps<typeof RecentPatientsPanel>> = {}) {
  const onPreviewPatient = vi.fn();

  render(
    <RecentPatientsPanel
      title="Recent patients"
      emptyMessage="No recent patients"
      items={items}
      previewedPatientId={101}
      isLoading={false}
      isLoadingPreview={false}
      error={null}
      onPreviewPatient={onPreviewPatient}
      {...overrides}
    />,
  );

  return { onPreviewPatient };
}

describe("RecentPatientsPanel", () => {
  it("renders recent patients with token-backed preview button classes", () => {
    renderPanel();

    const previewed = screen.getByRole("button", { name: "preview patient 101" });
    const other = screen.getByRole("button", { name: "preview patient 202" });

    expect(previewed).toHaveClass("recent-patient-button", "recent-patient-button-active");
    expect(previewed).toHaveAttribute("aria-pressed", "true");
    expect(previewed).not.toHaveAttribute("style");
    expect(other).toHaveClass("recent-patient-button");
    expect(other).not.toHaveClass("recent-patient-button-active");
    expect(screen.getByText("直肠 / cT3N1M0 / pMMR")).toBeInTheDocument();
    expect(screen.getByText("暂无摘要")).toBeInTheDocument();
  });

  it("calls onPreviewPatient when a patient is selected", () => {
    const { onPreviewPatient } = renderPanel();

    fireEvent.click(screen.getByRole("button", { name: "preview patient 202" }));

    expect(onPreviewPatient).toHaveBeenCalledWith(202);
  });

  it("renders loading and empty states without patient buttons", () => {
    const { rerender } = render(
      <RecentPatientsPanel
        title="Recent patients"
        emptyMessage="No recent patients"
        items={[]}
        previewedPatientId={null}
        isLoading
        isLoadingPreview={false}
        error={null}
        onPreviewPatient={vi.fn()}
      />,
    );

    expect(screen.getByText("正在加载最近患者...")).toHaveClass("recent-patients-loading");

    rerender(
      <RecentPatientsPanel
        title="Recent patients"
        emptyMessage="No recent patients"
        items={[]}
        previewedPatientId={null}
        isLoading={false}
        isLoadingPreview={false}
        error={null}
        onPreviewPatient={vi.fn()}
      />,
    );

    expect(screen.getByText("No recent patients")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /preview patient/i })).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the new component test and confirm red**

Run:

```powershell
npm --prefix frontend run test -- --run src/features/patient-registry/recent-patients-panel.test.tsx
```

Expected: FAIL because the component still uses inline styles and does not emit `recent-patient-button` classes.

- [ ] **Step 3: Refactor the recent patients component**

In `frontend/src/features/patient-registry/recent-patients-panel.tsx`, replace the visual inline styles with classes. The rendered structure should match this shape:

```tsx
return (
  <Card as="section" variant="clinical-panel" data-testid="recent-patients-panel">
    <h2 className="recent-patients-heading">{title}</h2>
    {error ? <p className="clinical-copy clinical-copy-alert">{error}</p> : null}
    {isLoading ? <p className="clinical-copy recent-patients-loading">正在加载最近患者...</p> : null}
    {!isLoading && items.length === 0 ? <p className="clinical-copy">{emptyMessage}</p> : null}
    {items.length > 0 ? (
      <div className="recent-patients-scroll">
        <ul className="clinical-list recent-patients-list">
          {items.map((item) => {
            const isPreviewed = previewedPatientId === item.patient_id;
            return (
              <li key={item.patient_id}>
                <button
                  type="button"
                  className={`clinical-list-item recent-patient-button ${
                    isPreviewed ? "clinical-step-current recent-patient-button-active" : ""
                  }`}
                  onClick={() => onPreviewPatient(item.patient_id)}
                  disabled={isLoadingPreview}
                  aria-label={`preview patient ${item.patient_id}`}
                  aria-pressed={isPreviewed}
                >
                  <div>
                    <strong className="recent-patient-title">
                      {`患者 #${item.patient_id}`}
                    </strong>
                    <p className="clinical-copy clinical-copy-tight recent-patient-summary">
                      {patientSummary(item)}
                    </p>
                  </div>
                  <span className="clinical-meta-text recent-patient-status">
                    {isPreviewed ? "正在预览" : "预览患者"}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    ) : null}
  </Card>
);
```

Keep the existing `patientSummary` function, props, and `Card` usage. Do not change data flow or preview behavior.

- [ ] **Step 4: Add token-backed CSS classes**

In `frontend/src/styles/globals.css`, place these classes near the existing patient registry or clinical list styles:

```css
.recent-patients-heading {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: 0;
  color: var(--color-text);
  font-size: var(--font-base);
  font-weight: 700;
}

.recent-patients-loading {
  color: var(--color-primary);
}

.recent-patients-scroll {
  max-height: calc(100vh - 200px);
  overflow-y: auto;
  padding-right: var(--space-2);
  margin-right: calc(var(--space-2) * -1);
}

.recent-patients-list {
  gap: var(--space-2);
  margin-top: var(--space-2);
}

.recent-patient-button {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  border: 1px solid color-mix(in srgb, var(--color-primary) 12%, transparent);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
  background: color-mix(in srgb, var(--color-surface) 82%, transparent);
  color: var(--color-text);
  text-align: left;
  font: inherit;
  cursor: pointer;
  transition:
    border-color var(--motion-duration-feedback) var(--motion-ease-out),
    background-color var(--motion-duration-feedback) var(--motion-ease-out),
    color var(--motion-duration-feedback) var(--motion-ease-out),
    transform var(--motion-duration-feedback) var(--motion-ease-out);
}

.recent-patient-button:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--color-primary) 28%, transparent);
  background: var(--color-primary-soft);
  transform: translateY(-1px);
}

.recent-patient-button:focus-visible {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px var(--focus-ring);
}

.recent-patient-button:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.recent-patient-button-active {
  border-color: color-mix(in srgb, var(--color-primary) 28%, transparent);
  background: var(--color-primary-soft);
}

.recent-patient-title {
  color: var(--color-text);
  font-size: var(--font-base);
}

.recent-patient-button-active .recent-patient-title {
  color: var(--color-primary);
}

.recent-patient-summary {
  margin-top: var(--space-1);
  font-size: var(--font-sm);
}

.recent-patient-status {
  font-size: var(--font-sm);
}
```

- [ ] **Step 5: Run focused component and style tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/features/patient-registry/recent-patients-panel.test.tsx src/styles/style-architecture-contract.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit recent patients refactor**

Run:

```powershell
git add frontend/src/features/patient-registry/recent-patients-panel.tsx frontend/src/features/patient-registry/recent-patients-panel.test.tsx frontend/src/styles/globals.css
git commit -m "style: remove broad recent patients transition"
```

Expected: commit succeeds.

## Task 5: Verify Phase 1 Contract

**Files:**
- Validate only

- [ ] **Step 1: Run focused motion and style contracts**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/motion/motion-system.test.ts src/styles/style-architecture-contract.test.ts src/features/patient-registry/recent-patients-panel.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run the existing style layout contract set**

Run:

```powershell
npm --prefix frontend run lint:styles:layout
```

Expected: PASS.

- [ ] **Step 3: Run the full frontend test suite**

Run:

```powershell
npm --prefix frontend run test -- --run
```

Expected: PASS.

- [ ] **Step 4: Run the production build**

Run:

```powershell
npm --prefix frontend run build
```

Expected: PASS with `tsc --noEmit` and Vite build output.

- [ ] **Step 5: Check dependency policy**

Run:

```powershell
Select-String -Path frontend/package.json -Pattern '"motion"|"framer-motion"|"@gsap/react"'
```

Expected: no output.

- [ ] **Step 6: Check broad transition policy**

Run:

```powershell
rg -n 'transition:\s*all|transition:\s*["'']all|0\.2s ease|160ms ease|240ms ease' frontend/src
```

Expected: no output except references inside test files if the regex appears as part of a guard assertion.

- [ ] **Step 7: Commit verification notes only if needed**

If no files changed during verification, do not commit. If snapshot or generated metadata changes unexpectedly, inspect it and do not commit unless it is an intentional test artifact.

## Task 6: Browser Smoke Validation

**Files:**
- Validate only

- [ ] **Step 1: Start the frontend dev server**

Run:

```powershell
npm --prefix frontend run dev -- --host 127.0.0.1 --port 4173
```

Expected: Vite serves at `http://127.0.0.1:4173/`. If port `4173` is in use, retry with `--port 4174` and use that URL for browser validation.

- [ ] **Step 2: Inspect key surfaces**

Use Browser or Playwright to inspect:

- Doctor cockpit desktop at `1440x900`.
- Patient assistant desktop at `1440x900`.
- Database view desktop at `1440x900`.
- Mobile layout at `390x844`.

Expected:

- Buttons, cards, tabs, and recent patient preview buttons animate with the same timing rhythm.
- Doctor dark cockpit does not regain large hover shadows.
- Patient light assistant remains light and direct.
- Recent patients preview buttons have no inline `style` attribute and no layout jump.
- No text overlaps at mobile width.

- [ ] **Step 3: Stop the dev server**

Stop the Vite process started in Step 1. Do not leave a dev server running after validation.

## Final Acceptance

Implementation is complete when:

- `motion-system.test.ts` enforces GSAP-only runtime, exact CSS ease, token mirror, no ad hoc CSS timing, and no paint-heavy GSAP props.
- `style-architecture-contract.test.ts` scans TSX for inline `transition: "all ..."` and passes.
- `tokens.css` and `motion-tokens.ts` both use `cubic-bezier(0.16, 1, 0.3, 1)` for CSS ease-out.
- `globals.css` no longer contains `160ms ease`, `240ms ease`, or `0.2s ease` transition declarations.
- `recent-patients-panel.tsx` no longer contains inline visual material styles or `transition: "all 0.2s ease"`.
- `package.json` does not add `motion`, `framer-motion`, or `@gsap/react`.
- Focused tests, full frontend tests, build, and browser smoke validation pass.
