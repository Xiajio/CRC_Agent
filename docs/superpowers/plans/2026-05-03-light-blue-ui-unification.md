# Light Blue UI Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `/database`, doctor database, patient workspace, and doctor workspace around a shared light-blue UI shell and reusable base components.

**Architecture:** Add a small `frontend/src/components/ui/` layer for primitives, then wrap existing layout components before migrating pages. Keep business hooks and API contracts unchanged while replacing rose workspace/database styling with light-blue tokens and shared React components.

**Tech Stack:** React 18, TypeScript, Vite, Vitest, Testing Library, CSS custom properties in `tokens.css` and `globals.css`.

---

## File Structure

Create:

- `frontend/src/components/ui/app-shell.tsx` - page shell for canvas, top nav slot, and body.
- `frontend/src/components/ui/top-nav.tsx` - reusable top navigation with brand, tabs, actions, status, and profile switch.
- `frontend/src/components/ui/panel-grid.tsx` - reusable left/center/right layout with existing test ids.
- `frontend/src/components/ui/card.tsx` - shared card surface.
- `frontend/src/components/ui/button.tsx` - shared button variants.
- `frontend/src/components/ui/input.tsx` - shared input/select field primitives.
- `frontend/src/components/ui/textarea.tsx` - shared textarea primitive.
- `frontend/src/components/ui/message-bubble.tsx` - shared user/AI message shell.
- `frontend/src/components/ui/index.ts` - UI exports.
- `frontend/src/components/ui/ui-components.test.tsx` - focused primitive tests.
- `frontend/src/components/ui/panel-grid.test.tsx` - layout behavior tests.
- `frontend/src/features/database/database-page.test.tsx` - page-level smoke test for the light-blue database shell.

Modify:

- `frontend/src/styles/tokens.css` - add light-blue semantic tokens and alias clinical tokens.
- `frontend/src/styles/globals.css` - add `.ui-*` styles, use tokens, and remove rose main visual values.
- `frontend/src/components/layout/clinical-top-nav.tsx` - wrap `TopNav` without changing the public API.
- `frontend/src/components/layout/workspace-layout.tsx` - wrap `AppShell` and `PanelGrid` without changing the public API.
- `frontend/src/pages/database-page.tsx` - use `TopNav` and shared buttons/cards.
- `frontend/src/features/database/*.tsx` - migrate panels to shared `Card`, `Button`, `Input`, and `Select`.
- `frontend/src/features/doctor/doctor-database-view.tsx` - use shared buttons for source switch.
- `frontend/src/features/chat/conversation-panel.tsx` - use `Card`, `Button`, `Textarea`, and `MessageBubble`.
- `frontend/src/pages/workspace-page.tsx` - keep business orchestration and preserve shell classes while relying on token-backed CSS.
- `frontend/src/features/doctor/doctor-scene-shell.tsx` - keep business orchestration, consume unified shell styles.

Do not modify:

- `frontend/src/app/api/*`
- `frontend/src/app/store/*`
- backend files
- business hooks such as `useDatabaseWorkbench`, `useSceneSessions`, and registry hooks except if tests reveal import-only adjustments.

---

## Task 1: Token Baseline And Rose Color Removal

**Files:**

- Modify: `frontend/src/styles/tokens.css`
- Modify: `frontend/src/styles/globals.css`
- Test by command only: CSS token grep and focused frontend tests

- [ ] **Step 1: Record current rose-color usage**

Run:

```powershell
rg -n "#8e4a55|#91515a|rgba\(165, 73, 83|rgba\(142, 74, 85" frontend\src\styles
```

Expected: Finds workspace/database rose visual rules in `globals.css`. Keep this output for comparison.

- [ ] **Step 2: Replace `tokens.css` with light-blue semantic tokens**

Update `frontend/src/styles/tokens.css` so it contains these variables and keeps old clinical names as aliases:

```css
:root {
  color-scheme: light;
  --color-canvas: #f4f8fc;
  --color-surface: #ffffff;
  --color-surface-soft: #f8fbff;
  --color-primary: #1466d8;
  --color-primary-hover: #0f58bf;
  --color-primary-soft: #eaf4ff;
  --color-navy: #061f3d;
  --color-navy-soft: #082b52;
  --color-text: #182434;
  --color-text-muted: #66758a;
  --color-border: #dbe7f3;
  --color-border-strong: #bfd3ea;
  --color-success: #24a66a;
  --color-warning: #f06423;
  --color-danger: #cc2f47;
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --radius-xs: 4px;
  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --radius-pill: 999px;
  --font-xs: 12px;
  --font-sm: 13px;
  --font-md: 14px;
  --font-base: 16px;
  --font-lg: 18px;
  --font-xl: 20px;
  --font-2xl: 24px;
  --bg-canvas: var(--color-canvas);
  --bg-panel: var(--color-surface);
  --bg-accent: linear-gradient(135deg, #eef6ff 0%, #ffffff 56%, #f7fbff 100%);
  --text-primary: var(--color-text);
  --text-secondary: var(--color-text-muted);
  --border-subtle: var(--color-border);
  --shadow-panel: 0 12px 30px rgba(23, 43, 77, 0.08);
  --radius-panel: var(--radius-md);
  --clinical-navy: var(--color-navy);
  --clinical-navy-2: var(--color-navy-soft);
  --clinical-primary: var(--color-primary);
  --clinical-primary-soft: var(--color-primary-soft);
  --clinical-success: var(--color-success);
  --clinical-warning: var(--color-warning);
  --clinical-danger: var(--color-danger);
  --clinical-muted: var(--color-text-muted);
  --font-body: "Noto Sans SC", "IBM Plex Sans", "Segoe UI", sans-serif;
}
```

- [ ] **Step 3: Add `.ui-*` token-backed styles to `globals.css`**

Insert a new section after the global `* { box-sizing: border-box; }` rule:

```css
.ui-app-shell {
  min-height: 100vh;
  background: var(--color-canvas);
  color: var(--color-text);
}

.ui-app-body {
  padding: var(--space-3);
}

.ui-card {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  box-shadow: 0 1px 2px rgba(23, 43, 77, 0.04);
}

.ui-card-padding-none { padding: 0; }
.ui-card-padding-sm { padding: var(--space-3); }
.ui-card-padding-md { padding: var(--space-4); }
.ui-card-soft { background: var(--color-surface-soft); }
.ui-card-selected { border-color: var(--color-border-strong); background: var(--color-primary-soft); }
.ui-card-warning { border-color: rgba(240, 100, 35, 0.28); background: #fff7ed; }
.ui-card-danger { border-color: rgba(204, 47, 71, 0.28); background: #fff1f4; }

.ui-button {
  min-height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  border-radius: var(--radius-md);
  border: 1px solid transparent;
  padding: 0 var(--space-4);
  font: inherit;
  font-size: var(--font-md);
  font-weight: 700;
  cursor: pointer;
  transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease, box-shadow 0.2s ease;
}

.ui-button-primary { background: var(--color-primary); color: #ffffff; }
.ui-button-primary:hover:not(:disabled) { background: var(--color-primary-hover); }
.ui-button-secondary { background: var(--color-surface); border-color: var(--color-border-strong); color: var(--color-navy); }
.ui-button-secondary:hover:not(:disabled) { background: var(--color-primary-soft); border-color: var(--color-primary); color: var(--color-primary); }
.ui-button-ghost { background: transparent; color: var(--color-navy); }
.ui-button-ghost:hover:not(:disabled) { background: var(--color-primary-soft); color: var(--color-primary); }
.ui-button-danger { background: var(--color-danger); color: #ffffff; }
.ui-button-sm { min-height: 30px; padding: 0 var(--space-3); font-size: var(--font-sm); }
.ui-button-md { min-height: 34px; padding: 0 var(--space-4); }
.ui-button:disabled { opacity: 0.55; cursor: not-allowed; }

.ui-field {
  display: grid;
  gap: var(--space-2);
}

.ui-field-label {
  color: var(--color-text-muted);
  font-size: var(--font-sm);
  font-weight: 700;
}

.ui-input,
.ui-textarea,
.ui-select {
  width: 100%;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface-soft);
  color: var(--color-text);
  font: inherit;
  font-size: var(--font-md);
}

.ui-input,
.ui-select {
  min-height: 38px;
  padding: 0 var(--space-3);
}

.ui-textarea {
  min-height: 72px;
  padding: var(--space-3);
  resize: vertical;
}

.ui-input:focus,
.ui-select:focus,
.ui-textarea:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(20, 102, 216, 0.16);
}
```

- [ ] **Step 4: Replace rose workspace/database colors with token colors**

In `frontend/src/styles/globals.css`, replace the main visual values:

```text
#8e4a55 -> var(--color-primary)
#91515a -> var(--color-primary)
#7d3e46 -> var(--color-primary-hover)
#82515a -> var(--color-navy)
rgba(165, 73, 83, 0.08) -> rgba(20, 102, 216, 0.08)
rgba(165, 73, 83, 0.1) -> rgba(20, 102, 216, 0.1)
rgba(165, 73, 83, 0.12) -> rgba(20, 102, 216, 0.12)
rgba(165, 73, 83, 0.14) -> rgba(20, 102, 216, 0.14)
rgba(165, 73, 83, 0.15) -> rgba(20, 102, 216, 0.15)
rgba(165, 73, 83, 0.16) -> rgba(20, 102, 216, 0.16)
rgba(165, 73, 83, 0.18) -> rgba(20, 102, 216, 0.18)
rgba(165, 73, 83, 0.2) -> rgba(20, 102, 216, 0.2)
rgba(165, 73, 83, 0.24) -> rgba(20, 102, 216, 0.24)
rgba(165, 73, 83, 0.25) -> rgba(20, 102, 216, 0.25)
rgba(165, 73, 83, 0.28) -> rgba(20, 102, 216, 0.28)
rgba(165, 73, 83, 0.4) -> rgba(20, 102, 216, 0.4)
rgba(142, 74, 85, 0.25) -> rgba(20, 102, 216, 0.25)
rgba(142, 74, 85, 0.35) -> rgba(20, 102, 216, 0.35)
linear-gradient(135deg, #8e4a55 0%, #a35d68 100%) -> linear-gradient(135deg, var(--color-primary) 0%, #3b82f6 100%)
linear-gradient(135deg, #fff0ee 0%, #ffe4e1 100%) -> linear-gradient(135deg, #f8fbff 0%, var(--color-primary-soft) 100%)
```

Do not replace `--color-danger` or warning/status colors.

- [ ] **Step 5: Verify rose main visual values are gone**

Run:

```powershell
rg -n "#8e4a55|#91515a|rgba\(165, 73, 83|rgba\(142, 74, 85" frontend\src\styles
```

Expected: No results. If results remain in comments or legacy code that is not main visual, document why; otherwise replace them.

- [ ] **Step 6: Run baseline tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/layout/clinical-top-nav.test.tsx src/features/database/database-results-table.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit token baseline**

Run:

```powershell
git add frontend/src/styles/tokens.css frontend/src/styles/globals.css
git commit -m "style: unify workspace colors with light blue tokens"
```

---

## Task 2: Shared Card, Button, Input, Textarea, And MessageBubble

**Files:**

- Create: `frontend/src/components/ui/card.tsx`
- Create: `frontend/src/components/ui/button.tsx`
- Create: `frontend/src/components/ui/input.tsx`
- Create: `frontend/src/components/ui/textarea.tsx`
- Create: `frontend/src/components/ui/message-bubble.tsx`
- Create: `frontend/src/components/ui/index.ts`
- Create: `frontend/src/components/ui/ui-components.test.tsx`
- Modify: `frontend/src/styles/globals.css`

- [ ] **Step 1: Write failing primitive tests**

Create `frontend/src/components/ui/ui-components.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent } from "@testing-library/react";

import { Button, Card, Input, MessageBubble, Textarea } from ".";

describe("shared UI primitives", () => {
  it("renders card tone, padding, and selected state", () => {
    render(
      <Card tone="soft" padding="sm" selected data-testid="card">
        Content
      </Card>,
    );

    const card = screen.getByTestId("card");
    expect(card).toHaveClass("ui-card", "ui-card-soft", "ui-card-padding-sm", "ui-card-selected");
    expect(card).toHaveTextContent("Content");
  });

  it("renders button variants and forwards click handlers", () => {
    const onClick = vi.fn();
    render(
      <Button variant="primary" size="sm" onClick={onClick}>
        Run
      </Button>,
    );

    const button = screen.getByRole("button", { name: "Run" });
    expect(button).toHaveClass("ui-button", "ui-button-primary", "ui-button-sm");
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("renders input and textarea with shared classes", () => {
    render(
      <>
        <Input label="Patient ID" value="33" onChange={() => undefined} />
        <Textarea aria-label="Prompt" value="hello" onChange={() => undefined} />
      </>,
    );

    expect(screen.getByLabelText("Patient ID")).toHaveClass("ui-input");
    expect(screen.getByLabelText("Prompt")).toHaveClass("ui-textarea");
  });

  it("renders user and assistant message bubbles", () => {
    render(
      <ol>
        <MessageBubble author="user" label="User">
          Hello
        </MessageBubble>
        <MessageBubble author="assistant" label="Assistant">
          Hi
        </MessageBubble>
      </ol>,
    );

    expect(screen.getByText("User")).toBeInTheDocument();
    expect(screen.getByText("Assistant")).toBeInTheDocument();
    expect(screen.getByText("Hello").closest("li")).toHaveClass("ui-message-bubble-user");
    expect(screen.getByText("Hi").closest("li")).toHaveClass("ui-message-bubble-assistant");
  });
});
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/ui/ui-components.test.tsx
```

Expected: FAIL because `frontend/src/components/ui` does not exist.

- [ ] **Step 3: Implement `Card`**

Create `frontend/src/components/ui/card.tsx`:

```tsx
import type { HTMLAttributes, ReactNode } from "react";

type CardPadding = "none" | "sm" | "md";
type CardTone = "surface" | "soft" | "warning" | "danger";

export interface CardProps extends HTMLAttributes<HTMLElement> {
  as?: "article" | "div" | "section";
  children: ReactNode;
  footer?: ReactNode;
  header?: ReactNode;
  padding?: CardPadding;
  selected?: boolean;
  tone?: CardTone;
}

function classNames(values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function Card({
  as: Component = "div",
  children,
  className,
  footer,
  header,
  padding = "md",
  selected = false,
  tone = "surface",
  ...props
}: CardProps) {
  return (
    <Component
      className={classNames([
        "ui-card",
        `ui-card-padding-${padding}`,
        tone !== "surface" ? `ui-card-${tone}` : null,
        selected ? "ui-card-selected" : null,
        className,
      ])}
      {...props}
    >
      {header ? <div className="ui-card-header">{header}</div> : null}
      <div className="ui-card-body">{children}</div>
      {footer ? <div className="ui-card-footer">{footer}</div> : null}
    </Component>
  );
}
```

- [ ] **Step 4: Implement `Button`**

Create `frontend/src/components/ui/button.tsx`:

```tsx
import type { ButtonHTMLAttributes, ReactNode } from "react";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  size?: ButtonSize;
  variant?: ButtonVariant;
}

function classNames(values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function Button({
  children,
  className,
  size = "md",
  type = "button",
  variant = "secondary",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={classNames(["ui-button", `ui-button-${variant}`, `ui-button-${size}`, className])}
      {...props}
    >
      {children}
    </button>
  );
}
```

- [ ] **Step 5: Implement `Input` and `Textarea`**

Create `frontend/src/components/ui/input.tsx`:

```tsx
import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";

type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  label?: ReactNode;
};

type SelectProps = SelectHTMLAttributes<HTMLSelectElement> & {
  label?: ReactNode;
};

export function Input({ className, id, label, ...props }: InputProps) {
  const input = <input id={id} className={["ui-input", className].filter(Boolean).join(" ")} {...props} />;

  if (!label) {
    return input;
  }

  return (
    <label className="ui-field">
      <span className="ui-field-label">{label}</span>
      {input}
    </label>
  );
}

export function Select({ children, className, id, label, ...props }: SelectProps) {
  const select = (
    <select id={id} className={["ui-select", className].filter(Boolean).join(" ")} {...props}>
      {children}
    </select>
  );

  if (!label) {
    return select;
  }

  return (
    <label className="ui-field">
      <span className="ui-field-label">{label}</span>
      {select}
    </label>
  );
}
```

Create `frontend/src/components/ui/textarea.tsx`:

```tsx
import type { TextareaHTMLAttributes } from "react";

export function Textarea({ className, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={["ui-textarea", className].filter(Boolean).join(" ")} {...props} />;
}
```

- [ ] **Step 6: Implement `MessageBubble`**

Create `frontend/src/components/ui/message-bubble.tsx`:

```tsx
import type { HTMLAttributes, ReactNode } from "react";

export interface MessageBubbleProps extends HTMLAttributes<HTMLLIElement> {
  author: "assistant" | "user";
  children: ReactNode;
  label: ReactNode;
}

export function MessageBubble({ author, children, className, label, ...props }: MessageBubbleProps) {
  return (
    <li
      className={[
        "ui-message-bubble",
        author === "user" ? "ui-message-bubble-user" : "ui-message-bubble-assistant",
        className,
      ].filter(Boolean).join(" ")}
      {...props}
    >
      <span className="ui-message-avatar" aria-hidden="true">
        {author === "user" ? "U" : "AI"}
      </span>
      <div className="ui-message-content">
        <div className="ui-message-header">
          <strong>{label}</strong>
        </div>
        {children}
      </div>
    </li>
  );
}
```

- [ ] **Step 7: Export UI primitives**

Create `frontend/src/components/ui/index.ts`:

```ts
export { Button, type ButtonProps, type ButtonSize, type ButtonVariant } from "./button";
export { Card, type CardProps } from "./card";
export { Input, Select } from "./input";
export { MessageBubble, type MessageBubbleProps } from "./message-bubble";
export { Textarea } from "./textarea";
```

- [ ] **Step 8: Add message styles**

Append to the `.ui-*` section in `frontend/src/styles/globals.css`:

```css
.ui-card-header {
  min-height: 48px;
  display: flex;
  align-items: center;
  gap: var(--space-3);
  border-bottom: 1px solid var(--color-border);
  padding: 0 var(--space-4);
}

.ui-card-body { min-width: 0; }
.ui-card-footer { border-top: 1px solid var(--color-border); padding: var(--space-3) var(--space-4); }

.ui-message-bubble {
  position: relative;
  width: 100%;
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr);
  gap: var(--space-3);
  border-radius: var(--radius-md);
  padding: var(--space-3);
  color: var(--color-text);
}

.ui-message-bubble-user {
  border: 1px solid var(--color-border);
  background: var(--color-surface);
}

.ui-message-bubble-assistant {
  border: 1px solid #dcecff;
  background: var(--color-primary-soft);
}

.ui-message-avatar {
  width: 34px;
  height: 34px;
  display: inline-grid;
  place-items: center;
  border-radius: var(--radius-pill);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  color: var(--color-primary);
  font-size: var(--font-xs);
  font-weight: 800;
}

.ui-message-header {
  margin-bottom: var(--space-1);
  color: var(--color-primary);
  font-size: var(--font-md);
}
```

- [ ] **Step 9: Run primitive tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/ui/ui-components.test.tsx
```

Expected: PASS.

- [ ] **Step 10: Commit primitives**

Run:

```powershell
git add frontend/src/components/ui frontend/src/styles/globals.css
git commit -m "feat: add shared light blue UI primitives"
```

---

## Task 3: AppShell, TopNav, PanelGrid, And Compatibility Wrappers

**Files:**

- Create: `frontend/src/components/ui/app-shell.tsx`
- Create: `frontend/src/components/ui/top-nav.tsx`
- Create: `frontend/src/components/ui/panel-grid.tsx`
- Create: `frontend/src/components/ui/panel-grid.test.tsx`
- Modify: `frontend/src/components/ui/index.ts`
- Modify: `frontend/src/components/layout/clinical-top-nav.tsx`
- Modify: `frontend/src/components/layout/workspace-layout.tsx`
- Modify: `frontend/src/styles/globals.css`

- [ ] **Step 1: Write failing shell/grid tests**

Create `frontend/src/components/ui/panel-grid.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AppShell, PanelGrid, TopNav, type TopNavItem } from ".";

const items: TopNavItem[] = [
  { key: "consultation", label: "Consultation" },
  { key: "database", label: "Database" },
  { key: "reports", label: "Reports", disabled: true },
];

describe("shell and panel grid", () => {
  it("renders the app shell with top nav and body", () => {
    render(
      <AppShell topNav={<TopNav brandLabel="Clinical" navLabel="Primary" items={items} activeKey="database" onSelect={() => undefined} statusLabel="Ready" profileLabel="Doctor" profileAriaLabel="Switch scene" />}>
        Body
      </AppShell>,
    );

    expect(screen.getByText("Clinical")).toBeInTheDocument();
    expect(screen.getByText("Body")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Database" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("button", { name: "Reports" })).toBeDisabled();
  });

  it("preserves workspace grid test ids and collapsed panel state", () => {
    render(
      <PanelGrid
        left={<div>Left</div>}
        center={<div>Center</div>}
        right={<div>Right</div>}
        leftOpen={false}
        rightOpen
      />,
    );

    expect(screen.getByTestId("workspace-layout-grid")).toHaveAttribute("data-layout-mode", "no-left");
    expect(screen.getByTestId("left-rail")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByTestId("right-inspector")).toHaveAttribute("data-panel-state", "open");
  });
});
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/ui/panel-grid.test.tsx
```

Expected: FAIL because `AppShell`, `TopNav`, and `PanelGrid` are not exported yet.

- [ ] **Step 3: Implement `AppShell`**

Create `frontend/src/components/ui/app-shell.tsx`:

```tsx
import type { HTMLAttributes, ReactNode } from "react";

export interface AppShellProps extends HTMLAttributes<HTMLElement> {
  children: ReactNode;
  topNav?: ReactNode;
}

export function AppShell({ children, className, topNav, ...props }: AppShellProps) {
  return (
    <main className={["ui-app-shell", className].filter(Boolean).join(" ")} {...props}>
      {topNav}
      <div className="ui-app-body">{children}</div>
    </main>
  );
}
```

- [ ] **Step 4: Implement `TopNav`**

Create `frontend/src/components/ui/top-nav.tsx`:

```tsx
import type { ReactNode } from "react";

export type TopNavItem = {
  key: string;
  label: string;
  disabled?: boolean;
};

export interface TopNavProps {
  actions?: ReactNode;
  actionsLabel?: string;
  activeKey: string;
  brandLabel: string;
  className?: string;
  items: TopNavItem[];
  navLabel: string;
  onProfileClick?: () => void;
  onSelect: (key: string) => void;
  profileAriaLabel: string;
  profileLabel: string;
  statusLabel: string;
  statusTone?: "connected" | "safe";
}

export function TopNav({
  actions,
  actionsLabel = "场景操作",
  activeKey,
  brandLabel,
  className,
  items,
  navLabel,
  onProfileClick,
  onSelect,
  profileAriaLabel,
  profileLabel,
  statusLabel,
  statusTone = "connected",
}: TopNavProps) {
  return (
    <header className={["ui-top-nav", className].filter(Boolean).join(" ")} data-testid="workspace-toolbar">
      <div className="ui-top-nav-brand">
        <span className="ui-top-nav-logo" aria-hidden="true" />
        <span>{brandLabel}</span>
      </div>
      <nav className="ui-top-nav-tabs" aria-label={navLabel}>
        {items.map((item) => {
          const isActive = item.key === activeKey;
          const isDisabled = Boolean(item.disabled);

          return (
            <button
              key={item.key}
              type="button"
              className={["ui-top-nav-tab", isActive ? "ui-top-nav-tab-active" : null].filter(Boolean).join(" ")}
              aria-current={isActive ? "page" : undefined}
              aria-disabled={isDisabled ? "true" : undefined}
              aria-pressed={isActive}
              disabled={isDisabled}
              onClick={() => {
                if (!isDisabled) {
                  onSelect(item.key);
                }
              }}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
      {actions ? (
        <div className="ui-top-nav-actions" aria-label={actionsLabel}>
          {actions}
        </div>
      ) : null}
      <div className="ui-top-nav-user-area">
        <span className={["ui-top-nav-status", statusTone === "safe" ? "ui-top-nav-status-safe" : null].filter(Boolean).join(" ")}>
          <span aria-hidden="true" />
          {statusLabel}
        </span>
        <button type="button" className="ui-profile-switch clinical-profile-switch" aria-label={profileAriaLabel} onClick={onProfileClick}>
          <span className="ui-profile-avatar" aria-hidden="true" />
          <span>{profileLabel}</span>
          <span aria-hidden="true">v</span>
        </button>
      </div>
    </header>
  );
}
```

- [ ] **Step 5: Implement `PanelGrid`**

Create `frontend/src/components/ui/panel-grid.tsx`:

```tsx
import type { ReactNode } from "react";

export interface PanelGridProps {
  center: ReactNode;
  centerTestId?: string;
  left: ReactNode;
  leftOpen?: boolean;
  leftTestId?: string;
  right: ReactNode;
  rightOpen?: boolean;
  rightTestId?: string;
}

export function PanelGrid({
  center,
  centerTestId = "center-workspace",
  left,
  leftOpen = true,
  leftTestId = "left-rail",
  right,
  rightOpen = true,
  rightTestId = "right-inspector",
}: PanelGridProps) {
  const layoutMode = leftOpen
    ? rightOpen
      ? "full"
      : "no-right"
    : rightOpen
      ? "no-left"
      : "center-only";

  return (
    <div className={`ui-panel-grid ui-panel-grid-${layoutMode}`} data-testid="workspace-layout-grid" data-layout-mode={layoutMode}>
      <aside className="ui-panel ui-panel-left" data-testid={leftTestId} data-panel-state={leftOpen ? "open" : "closed"} aria-hidden={leftOpen ? undefined : "true"}>
        {left}
      </aside>
      <section className="ui-panel ui-panel-center" data-testid={centerTestId}>
        {center}
      </section>
      <aside className="ui-panel ui-panel-right" data-testid={rightTestId} data-panel-state={rightOpen ? "open" : "closed"} aria-hidden={rightOpen ? undefined : "true"}>
        {right}
      </aside>
    </div>
  );
}
```

- [ ] **Step 6: Export shell/grid components**

Modify `frontend/src/components/ui/index.ts`:

```ts
export { AppShell, type AppShellProps } from "./app-shell";
export { Button, type ButtonProps, type ButtonSize, type ButtonVariant } from "./button";
export { Card, type CardProps } from "./card";
export { Input, Select } from "./input";
export { MessageBubble, type MessageBubbleProps } from "./message-bubble";
export { PanelGrid, type PanelGridProps } from "./panel-grid";
export { Textarea } from "./textarea";
export { TopNav, type TopNavItem, type TopNavProps } from "./top-nav";
```

- [ ] **Step 7: Add shell/grid styles**

Append to `.ui-*` section in `frontend/src/styles/globals.css`:

```css
.ui-top-nav {
  position: sticky;
  top: 0;
  z-index: 20;
  min-height: 62px;
  display: grid;
  grid-template-columns: minmax(250px, 360px) minmax(440px, 1fr) auto auto;
  align-items: center;
  gap: var(--space-4);
  padding: 0 var(--space-6);
  background: linear-gradient(90deg, var(--color-navy) 0%, #08213f 44%, #061b33 100%);
  color: #f7fbff;
  box-shadow: 0 8px 18px rgba(4, 22, 43, 0.14);
}

.ui-top-nav-brand,
.ui-top-nav-tabs,
.ui-top-nav-actions,
.ui-top-nav-user-area,
.ui-profile-switch {
  display: flex;
  align-items: center;
}

.ui-top-nav-brand { gap: var(--space-3); font-size: var(--font-base); font-weight: 800; }
.ui-top-nav-tabs { min-width: 0; height: 100%; justify-content: center; gap: 0; }
.ui-top-nav-tab {
  height: 100%;
  min-height: 62px;
  border: 0;
  border-bottom: 3px solid transparent;
  background: transparent;
  color: rgba(247, 251, 255, 0.72);
  padding: 0 var(--space-6);
  font: inherit;
  font-size: var(--font-md);
  font-weight: 700;
  cursor: pointer;
}
.ui-top-nav-tab-active { border-bottom-color: #6fb5ff; color: #ffffff; }
.ui-top-nav-tab:disabled { opacity: 0.45; cursor: not-allowed; }
.ui-top-nav-user-area { gap: var(--space-4); justify-content: flex-end; }
.ui-top-nav-status { min-height: 28px; display: inline-flex; align-items: center; gap: var(--space-2); border-radius: var(--radius-pill); background: rgba(234, 244, 255, 0.14); padding: 0 var(--space-3); font-size: var(--font-sm); font-weight: 700; }
.ui-profile-switch { gap: var(--space-2); border: 0; background: transparent; color: #ffffff; font: inherit; font-weight: 700; cursor: pointer; }
.ui-profile-avatar { width: 34px; height: 34px; border-radius: var(--radius-pill); background: #ffffff; }

.ui-panel-grid {
  display: grid;
  grid-template-areas: "left center right";
  grid-template-columns: 280px minmax(0, 1fr) 320px;
  gap: var(--space-3);
}
.ui-panel-grid-no-left { grid-template-areas: "center right"; grid-template-columns: minmax(0, 1fr) 320px; }
.ui-panel-grid-no-right { grid-template-areas: "left center"; grid-template-columns: 280px minmax(0, 1fr); }
.ui-panel-grid-center-only { grid-template-areas: "center"; grid-template-columns: minmax(0, 1fr); }
.ui-panel { min-width: 0; }
.ui-panel-left { grid-area: left; }
.ui-panel-center { grid-area: center; }
.ui-panel-right { grid-area: right; }
.ui-panel[aria-hidden="true"] { display: none; }
```

- [ ] **Step 8: Wrap `ClinicalTopNav` with `TopNav`**

Modify `frontend/src/components/layout/clinical-top-nav.tsx` to keep exports and render `TopNav`:

```tsx
import { TopNav, type TopNavItem } from "../ui";

export type ClinicalNavItem = TopNavItem;

export function ClinicalNodeLogo() {
  return <span className="ui-top-nav-logo clinical-logo-mark" aria-hidden="true" />;
}

export function ClinicalUserIcon() {
  return <span className="ui-profile-avatar" aria-hidden="true" />;
}

type ClinicalTopNavProps = {
  brandLabel: string;
  navLabel: string;
  items: ClinicalNavItem[];
  activeKey: string;
  onSelect: (key: string) => void;
  actions?: React.ReactNode;
  actionsLabel?: string;
  statusLabel: string;
  statusTone: "connected" | "safe";
  profileLabel: string;
  profileAriaLabel: string;
  onProfileClick?: () => void;
  className?: string;
};

export function ClinicalTopNav(props: ClinicalTopNavProps) {
  return <TopNav {...props} className={["clinical-top-nav", props.className].filter(Boolean).join(" ")} />;
}
```

- [ ] **Step 9: Wrap `WorkspaceLayout` with `AppShell` and `PanelGrid`**

Modify `frontend/src/components/layout/workspace-layout.tsx`:

```tsx
import type { ReactNode } from "react";

import { AppShell, PanelGrid } from "../ui";

export interface WorkspaceLayoutProps {
  leftRail: ReactNode;
  centerWorkspace: ReactNode;
  rightInspector: ReactNode;
  toolbar?: ReactNode;
  leftRailOpen?: boolean;
  rightInspectorOpen?: boolean;
}

export function WorkspaceLayout({
  leftRail,
  centerWorkspace,
  rightInspector,
  toolbar,
  leftRailOpen = true,
  rightInspectorOpen = true,
}: WorkspaceLayoutProps) {
  return (
    <AppShell className="workspace-shell" topNav={toolbar ? <div className="workspace-toolbar">{toolbar}</div> : undefined}>
      <PanelGrid
        left={leftRail}
        center={centerWorkspace}
        right={rightInspector}
        leftOpen={leftRailOpen}
        rightOpen={rightInspectorOpen}
      />
    </AppShell>
  );
}
```

- [ ] **Step 10: Run shell/grid tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/ui/panel-grid.test.tsx src/components/layout/clinical-top-nav.test.tsx
```

Expected: PASS. If `clinical-top-nav.test.tsx` expects exact legacy class names, keep `clinical-profile-switch` on the profile button as shown.

- [ ] **Step 11: Commit shell/grid**

Run:

```powershell
git add frontend/src/components/ui frontend/src/components/layout/clinical-top-nav.tsx frontend/src/components/layout/workspace-layout.tsx frontend/src/styles/globals.css
git commit -m "feat: add shared app shell and panel grid"
```

---

## Task 4: Migrate Database Page And Database Panels

**Files:**

- Create: `frontend/src/features/database/database-page.test.tsx`
- Modify: `frontend/src/pages/database-page.tsx`
- Modify: `frontend/src/features/database/database-natural-query-bar.tsx`
- Modify: `frontend/src/features/database/database-filters-panel.tsx`
- Modify: `frontend/src/features/database/database-detail-panel.tsx`
- Modify: `frontend/src/features/database/database-edit-form.tsx`
- Modify: `frontend/src/features/database/database-workbench-panel.tsx`
- Modify: `frontend/src/features/database/database-results-table.tsx`
- Modify: `frontend/src/styles/globals.css`

- [ ] **Step 1: Write database shell smoke test**

Create `frontend/src/features/database/database-page.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AppProviders } from "../../app/providers";
import { buildApiClientStub } from "../../test/test-utils";
import { DatabasePage } from "../../pages/database-page";

describe("DatabasePage light blue shell", () => {
  it("renders the database page with shared toolbar and panel grid", async () => {
    render(
      <AppProviders apiClient={buildApiClientStub()}>
        <DatabasePage />
      </AppProviders>,
    );

    expect(await screen.findByText(/虚拟数据库控制台|数据库控制台/)).toBeInTheDocument();
    expect(screen.getByTestId("workspace-layout-grid")).toHaveAttribute("data-layout-mode", "full");
    expect(screen.getByTestId("left-rail")).toBeInTheDocument();
    expect(screen.getByTestId("center-workspace")).toBeInTheDocument();
    expect(screen.getByTestId("right-inspector")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify current behavior**

Run:

```powershell
npm --prefix frontend run test -- --run src/features/database/database-page.test.tsx
```

Expected: PASS or FAIL only on the title text. If it fails on title text, update the regex to the actual visible title and rerun. Do not change component behavior for this test yet.

- [ ] **Step 3: Migrate `DatabaseNaturalQueryBar`**

Modify `frontend/src/features/database/database-natural-query-bar.tsx`:

```tsx
import { Button, Card, Input } from "../../components/ui";

interface DatabaseNaturalQueryBarProps {
  value: string;
  warnings: string[];
  unsupportedTerms: string[];
  isParsing: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export function DatabaseNaturalQueryBar({
  value,
  warnings,
  unsupportedTerms,
  isParsing,
  onChange,
  onSubmit,
}: DatabaseNaturalQueryBarProps) {
  return (
    <Card>
      <div className="database-section-heading">
        <h2>{"自然语言查询"}</h2>
        <p className="workspace-copy workspace-copy-tight">
          {"让大模型先解析检索意图，再同步到下方结构化筛选。"}
        </p>
      </div>
      <Input
        label="自然语言查询"
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="例如：帮我找出 30-40 岁、有肝转移的患者"
      />
      <div className="database-action-row">
        <Button variant="secondary" onClick={onSubmit} disabled={isParsing || !value.trim()}>
          {isParsing ? "解析中..." : "解析查询"}
        </Button>
      </div>
      {warnings.length > 0 ? (
        <div className="database-feedback-list" role="status">
          {warnings.map((warning) => (
            <p key={warning} className="workspace-copy workspace-copy-tight">
              {warning}
            </p>
          ))}
        </div>
      ) : null}
      {unsupportedTerms.length > 0 ? (
        <p className="workspace-copy workspace-copy-tight">
          {`未支持的条件：${unsupportedTerms.join("、")}`}
        </p>
      ) : null}
    </Card>
  );
}
```

- [ ] **Step 4: Migrate `DatabaseFiltersPanel` controls**

In `frontend/src/features/database/database-filters-panel.tsx`, import:

```tsx
import { Button, Card, Input, Select } from "../../components/ui";
```

Replace the outer `<div className="workspace-card">` with `<Card>`, closing with `</Card>`. Replace `input` fields with `Input`, `select` fields with `Select`, and action buttons with:

```tsx
<Button variant="secondary" onClick={onApply} disabled={isSearching}>
  {isSearching ? "检索中..." : "应用筛选"}
</Button>
<Button variant="ghost" onClick={onReset} disabled={isSearching}>
  {"重置"}
</Button>
```

For selects, preserve existing `aria-label` values. Example:

```tsx
<Select
  label="家族史"
  aria-label="家族史筛选"
  value={triStateValue(filters.family_history)}
  onChange={(event) =>
    onFiltersChange({
      ...filters,
      family_history: readTriState(event.target.value),
    })
  }
>
  {TRI_STATE_OPTIONS.map((option) => (
    <option key={option.value || "empty"} value={option.value}>
      {option.label}
    </option>
  ))}
</Select>
```

- [ ] **Step 5: Migrate remaining database card shells**

For `database-detail-panel.tsx`, `database-edit-form.tsx`, and `database-workbench-panel.tsx`:

1. Import `Card`, `Button`, `Input`, and `Select` from `../../components/ui` in every database panel file that renders a card shell, button, text input, or select.
2. Replace `workspace-card` and `workspace-banner` wrappers with `Card`.
3. Replace form controls with shared `Input`/`Select`.
4. Replace `workspace-button`, `workspace-primary-button`, and `workspace-secondary-button` with `Button`.

Use this mapping:

```text
workspace-card -> <Card>
workspace-banner -> <Card tone="soft">
workspace-banner-error -> <Card tone="danger">
workspace-primary-button -> <Button variant="primary">
workspace-secondary-button -> <Button variant="secondary">
workspace-button -> <Button variant="secondary">
database-input -> <Input>
database-select -> <Select>
```

- [ ] **Step 6: Keep table semantics and only token-back styles**

In `database-results-table.tsx`, keep table markup. Only replace button classes:

```tsx
className="database-table-button ui-button ui-button-ghost ui-button-sm"
```

Do not replace table DOM with cards.

- [ ] **Step 7: Run database tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/features/database/database-page.test.tsx src/features/database/database-results-table.test.tsx src/features/database/use-database-workbench.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit database migration**

Run:

```powershell
git add frontend/src/pages/database-page.tsx frontend/src/features/database frontend/src/styles/globals.css
git commit -m "refactor: migrate database UI to shared components"
```

---

## Task 5: Migrate Doctor Database Source Switch

**Files:**

- Modify: `frontend/src/features/doctor/doctor-database-view.tsx`
- Test: `frontend/src/features/doctor/doctor-scene-shell.test.tsx`

- [ ] **Step 1: Update doctor database source switch**

In `frontend/src/features/doctor/doctor-database-view.tsx`, import:

```tsx
import { Button } from "../../components/ui";
```

Replace `DatabaseSourceToolbar` with:

```tsx
function DatabaseSourceToolbar({
  activeSource,
  onSourceChange,
}: {
  activeSource: DoctorDatabaseSource;
  onSourceChange: (value: DoctorDatabaseSource) => void;
}) {
  return (
    <div className="ui-segmented-control" aria-label="Database source">
      <Button
        variant={activeSource === "historical_case_base" ? "primary" : "secondary"}
        size="sm"
        onClick={() => onSourceChange("historical_case_base")}
        aria-label="historical case base"
      >
        历史病例
      </Button>
      <Button
        variant={activeSource === "patient_registry" ? "primary" : "secondary"}
        size="sm"
        onClick={() => onSourceChange("patient_registry")}
        aria-label="patient registry"
      >
        患者库
      </Button>
    </div>
  );
}
```

- [ ] **Step 2: Add segmented-control styles**

Append to `frontend/src/styles/globals.css`:

```css
.ui-segmented-control {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1);
  border: 1px solid rgba(219, 231, 243, 0.6);
  border-radius: var(--radius-lg);
  background: rgba(255, 255, 255, 0.12);
}
```

- [ ] **Step 3: Run doctor shell tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/features/doctor/doctor-scene-shell.test.tsx
```

Expected: PASS.

- [ ] **Step 4: Commit doctor database switch**

Run:

```powershell
git add frontend/src/features/doctor/doctor-database-view.tsx frontend/src/styles/globals.css
git commit -m "refactor: unify doctor database source controls"
```

---

## Task 6: Migrate ConversationPanel To MessageBubble And Shared Inputs

**Files:**

- Modify: `frontend/src/features/chat/conversation-panel.tsx`
- Modify: `frontend/src/features/chat/conversation-panel.test.tsx`
- Modify: `frontend/src/styles/globals.css`

- [ ] **Step 1: Add focused assertions for shared message bubble**

In `frontend/src/features/chat/conversation-panel.test.tsx`, add this test inside the existing suite:

```tsx
it("renders messages through shared light-blue message bubbles", () => {
  render(
    <ConversationPanel
      messages={[
        { cursor: "1", type: "user", content: "hello" },
        { cursor: "2", type: "ai", content: "hi" },
      ]}
      draft=""
      statusNode={null}
      isStreaming={false}
      isLoadingHistory={false}
      canLoadHistory={false}
      disabled={false}
      errorMessage={null}
      onLoadHistory={() => undefined}
      onDraftChange={() => undefined}
      onSubmit={() => undefined}
    />,
  );

  expect(screen.getByText("hello").closest("li")).toHaveClass("ui-message-bubble-user");
  expect(screen.getByText("hi").closest("li")).toHaveClass("ui-message-bubble-assistant");
  expect(screen.getByRole("textbox")).toHaveClass("ui-textarea");
});
```

If the local `FrontendMessage` type requires additional fields, use the smallest valid message object already used elsewhere in this test file.

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
npm --prefix frontend run test -- --run src/features/chat/conversation-panel.test.tsx
```

Expected: FAIL because `ConversationPanel` still renders old workspace/clinical message classes and textarea class.

- [ ] **Step 3: Migrate imports**

In `frontend/src/features/chat/conversation-panel.tsx`, add:

```tsx
import { Button, Card, MessageBubble, Textarea } from "../../components/ui";
```

- [ ] **Step 4: Replace outer conversation shell**

Replace:

```tsx
<section className="workspace-card clinical-conversation-card" data-testid="conversation-panel">
```

with:

```tsx
<Card as="section" padding="none" className="clinical-conversation-card" data-testid="conversation-panel">
```

and replace the matching closing `</section>` with `</Card>`.

- [ ] **Step 5: Replace load-history button**

Replace the load history `<button>` with:

```tsx
<Button variant="secondary" size="sm" disabled={isLoadingHistory} onClick={onLoadHistory}>
  {isLoadingHistory ? "正在加载历史..." : "加载更早消息"}
</Button>
```

- [ ] **Step 6: Replace message `li` with `MessageBubble`**

Replace:

```tsx
<li
  key={message.cursor}
  className={`workspace-message-bubble clinical-message-bubble ${isUser ? "bubble-user" : "bubble-ai"}`}
>
  <div className="bubble-header clinical-bubble-header">
    <strong>{messageLabel(message)}</strong>
  </div>
```

with:

```tsx
<MessageBubble key={message.cursor} author={isUser ? "user" : "assistant"} label={messageLabel(message)}>
```

Remove the old header closing `</div>`. Replace the matching `</li>` with `</MessageBubble>`.

- [ ] **Step 7: Replace composer textarea**

Replace:

```tsx
<textarea
  className="workspace-composer-input"
```

with:

```tsx
<Textarea
  className="clinical-composer-textarea"
```

Keep all existing `placeholder`, `value`, `disabled`, `onChange`, and `onKeyDown` logic unchanged.

- [ ] **Step 8: Replace send button class**

Replace `className="workspace-composer-send"` with:

```tsx
className="workspace-composer-send ui-button ui-button-primary"
```

Keep the SVG and `aria-label`.

- [ ] **Step 9: Run conversation tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/features/chat/conversation-panel.test.tsx
```

Expected: PASS.

- [ ] **Step 10: Commit conversation migration**

Run:

```powershell
git add frontend/src/features/chat/conversation-panel.tsx frontend/src/features/chat/conversation-panel.test.tsx frontend/src/styles/globals.css
git commit -m "refactor: migrate conversation panel to shared message UI"
```

---

## Task 7: Migrate Workspace Outer Shells And Clinical Cards

**Files:**

- Modify: `frontend/src/pages/workspace-page.tsx`
- Modify: `frontend/src/features/doctor/doctor-scene-shell.tsx`
- Modify: `frontend/src/features/cards/clinical-cards-panel.tsx`
- Modify: `frontend/src/features/cards/patient-background-panel.tsx`
- Modify: `frontend/src/features/execution-plan/execution-plan-panel.tsx`
- Modify: `frontend/src/features/roadmap/roadmap-panel.tsx`
- Test: `frontend/src/pages/workspace-page.test.tsx`
- Test: `frontend/src/features/doctor/doctor-scene-shell.test.tsx`

- [ ] **Step 1: Migrate reusable clinical panel shells to `Card`**

For each panel component listed in this task, import:

```tsx
import { Card } from "../../components/ui";
```

Use the correct relative path. For files under `frontend/src/features/cards`, `../../components/ui` is correct. For files under `frontend/src/features/execution-plan` and `frontend/src/features/roadmap`, use `../../components/ui`.

Replace outer panel containers:

```text
<section className="clinical-card ..."> -> <Card as="section" padding="none" className="...">
<section className="workspace-card ..."> -> <Card as="section" padding="none" className="...">
</section> -> </Card>
```

Keep all current child markup, `data-testid`, refs, and event handlers.

- [ ] **Step 2: Keep patient and doctor app shells visually aligned**

In `frontend/src/pages/workspace-page.tsx` and `frontend/src/features/doctor/doctor-scene-shell.tsx`, do not change business props. Keep the existing outer shell class combinations:

```tsx
<main className="clinical-app-shell clinical-app-shell-patient">
```

The `.clinical-app-shell` styles are token-backed by earlier tasks, so this task does not introduce a second page structure here.

- [ ] **Step 3: Run workspace and doctor tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/pages/workspace-page.test.tsx src/features/doctor/doctor-scene-shell.test.tsx src/features/cards/clinical-cards-panel.test.tsx src/features/cards/patient-background-panel.test.tsx src/features/execution-plan/execution-plan-panel.test.tsx src/features/roadmap/roadmap-panel.test.tsx
```

Expected: PASS. If a test expects `.clinical-card`, preserve that class in `className` while still using `Card`.

- [ ] **Step 4: Commit workspace shell migration**

Run:

```powershell
git add frontend/src/pages/workspace-page.tsx frontend/src/features/doctor/doctor-scene-shell.tsx frontend/src/features/cards frontend/src/features/execution-plan frontend/src/features/roadmap
git commit -m "refactor: align workspace panels with shared card shell"
```

---

## Task 8: CSS Cleanup, Full Verification, And Browser Review

**Files:**

- Modify: `frontend/src/styles/globals.css`
- Modify: migrated files only when Step 2 finds remaining TSX references to legacy visual selectors

- [ ] **Step 1: Find unused legacy visual selectors**

Run:

```powershell
rg -n "workspace-message-bubble|clinical-message-bubble|bubble-user|bubble-ai|workspace-primary-button|workspace-secondary-button|workspace-button|database-input|database-select" frontend\src
```

Expected: Some legacy selectors may remain in CSS or compatibility paths. Any remaining TSX usage should be intentional and documented in the next step.

- [ ] **Step 2: Remove dead TSX usage before CSS deletion**

If the command in Step 1 finds TSX usage of these selectors, migrate that usage to `Card`, `Button`, `Input`, `Select`, `Textarea`, or `MessageBubble`. Do not delete CSS while TSX still depends on it.

Use this final mapping:

```text
workspace-primary-button -> Button variant="primary"
workspace-secondary-button -> Button variant="secondary"
workspace-button -> Button variant="secondary"
database-input -> Input
database-select -> Select
workspace-composer-input -> Textarea
workspace-message-bubble clinical-message-bubble -> MessageBubble
workspace-card clinical-card -> Card while preserving semantic className if tests need it
```

- [ ] **Step 3: Delete legacy visual CSS blocks that have no TSX usage**

From `frontend/src/styles/globals.css`, delete blocks for selectors with no remaining usage. Keep selectors used by tests or compatibility wrappers. Do not delete table layout, clinical grid, roadmap, execution plan, and patient registry styles unless the corresponding components no longer reference them.

- [ ] **Step 4: Verify rose color removal**

Run:

```powershell
rg -n "#8e4a55|#91515a|#a35d68|rgba\(165, 73, 83|rgba\(142, 74, 85" frontend\src
```

Expected: No results.

- [ ] **Step 5: Run focused frontend tests**

Run:

```powershell
npm --prefix frontend run test -- --run src/components/ui/ui-components.test.tsx src/components/ui/panel-grid.test.tsx src/components/layout/clinical-top-nav.test.tsx src/features/database/database-page.test.tsx src/features/database/database-results-table.test.tsx src/features/database/use-database-workbench.test.tsx src/features/chat/conversation-panel.test.tsx src/features/doctor/doctor-scene-shell.test.tsx src/pages/workspace-page.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Run full frontend test suite**

Run:

```powershell
npm --prefix frontend run test -- --run
```

Expected: PASS.

- [ ] **Step 7: Run frontend build**

Run:

```powershell
npm --prefix frontend run build
```

Expected: PASS.

- [ ] **Step 8: Browser review**

Start dev server:

```powershell
npm --prefix frontend run dev -- --host 127.0.0.1
```

Open and verify:

```text
http://127.0.0.1:5173/database
http://127.0.0.1:5173/
```

Manual checks:

1. `/database` is light-blue/white, not rose.
2. Patient scene keeps three columns at desktop sizes.
3. Doctor scene keeps three columns at desktop sizes.
4. Doctor database tab uses the shared segmented source switch.
5. Conversation messages render with shared full-width user/assistant bubbles.
6. Buttons and inputs share height, radius, and focus ring.
7. At 720px width, panels stack without text overlap.

- [ ] **Step 9: Commit cleanup and verification fixes**

Run:

```powershell
git add frontend/src
git commit -m "chore: clean up legacy visual styles"
```

---

## Final Acceptance

Before reporting completion:

- [ ] `rg -n "#8e4a55|#91515a|#a35d68|rgba\(165, 73, 83|rgba\(142, 74, 85" frontend\src` returns no results.
- [ ] `npm --prefix frontend run test -- --run` passes.
- [ ] `npm --prefix frontend run build` passes.
- [ ] Browser review covers `/database`, patient scene, doctor scene, and doctor database tab.
- [ ] `git status --short` shows only intentional changes or is clean.

If any test or build command fails, stop and debug that failure before continuing to the next task.
