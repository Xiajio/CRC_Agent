import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles/globals.css"), "utf8");
const tokensCss = readFileSync(resolve(process.cwd(), "src/styles/tokens.css"), "utf8");

function blockFor(selector: string, source = css) {
  return blocksFor(selector, source)[0] ?? "";
}

function blocksFor(selector: string, source = css) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return [...source.matchAll(new RegExp(`${escaped}\\s*\\{(?<body>[\\s\\S]*?)\\n\\s*\\}`, "gm"))].map(
    (match) => match.groups?.body ?? "",
  );
}

function mediaBlockFor(query: string) {
  const marker = `@media (${query})`;
  const start = css.indexOf(marker);
  if (start === -1) return "";

  let depth = 0;
  const openBrace = css.indexOf("{", start);
  for (let i = openBrace; i < css.length; i += 1) {
    if (css[i] === "{") depth += 1;
    if (css[i] === "}") depth -= 1;
    if (depth === 0) return css.slice(openBrace + 1, i);
  }

  return "";
}

describe("consultation cockpit layout CSS", () => {
  it("keeps clinical page shell spacing around toolbar and body content", () => {
    const shell = blockFor(".clinical-page-shell");
    const media1100 = mediaBlockFor("max-width: 1100px");

    expect(shell).toContain("min-height: 100vh");
    expect(shell).toContain("display: grid");
    expect(shell).toContain("gap: var(--space-3)");
    expect(shell).toContain("padding: var(--space-4)");
    expect(shell).toContain("background: var(--color-canvas)");
    expect(shell).toContain("color: var(--color-text)");
    expect(blockFor(".clinical-page-shell", media1100)).toContain("padding: var(--space-3)");
    expect(blockFor(".clinical-app-shell-database .clinical-page-shell")).toContain("padding-top: 12px");
  });

  it("gives the base desktop cockpit a dominant consultation center and compact support rails", () => {
    const dashboard = blockFor(".clinical-dashboard");

    expect(dashboard).toContain("var(--dashboard-left-width) minmax(0, 1fr) var(--dashboard-right-width)");
    expect(dashboard).toContain("gap: var(--space-3)");
    expect(dashboard).toContain('"left center right"');
    expect(dashboard).toContain('"event event right"');

    expect(blockFor(".clinical-patient-dashboard")).toContain(
      "var(--dashboard-left-width) minmax(0, 1fr) var(--dashboard-right-width)",
    );
    expect(blockFor(".clinical-multimodal-dashboard")).toContain(
      "var(--dashboard-left-width) minmax(0, 1fr) var(--dashboard-right-width)",
    );
    expect(blockFor(".clinical-patient-dashboard-no-right")).toContain(
      "var(--dashboard-left-width) minmax(0, 1fr)",
    );
  });

  it("keeps the cockpit usable at medium desktop widths without a wide right rail", () => {
    const media1450 = mediaBlockFor("max-width: 1450px");
    const dashboard = blockFor(".clinical-dashboard", media1450);

    expect(dashboard).toContain("var(--dashboard-left-width) minmax(0, 1fr) var(--dashboard-right-width)");
    expect(dashboard).not.toContain("420px");
    expect(dashboard).not.toContain("500px");
  });

  it("places consultation before supporting context once the layout stacks", () => {
    const media1150 = mediaBlockFor("max-width: 1150px");
    const dashboard = blockFor(".clinical-dashboard", media1150);

    expect(dashboard).toContain('"center"');
    expect(dashboard.indexOf('"center"')).toBeLessThan(dashboard.indexOf('"left"'));
    expect(dashboard.indexOf('"left"')).toBeLessThan(dashboard.indexOf('"right"'));
    expect(dashboard.indexOf('"right"')).toBeLessThan(dashboard.indexOf('"event"'));
  });

  it("makes the conversation and event stream proportions match a cockpit workflow", () => {
    expect(blocksFor(".clinical-conversation-card")).toContainEqual(
      expect.stringContaining("clamp(470px, 60vh, 680px)"),
    );
    expect(blockFor(".clinical-message-list")).toContain("max-width: 72ch");
    expect(blockFor(".clinical-message-list")).toContain("margin: 0 auto");
    expect(blockFor(".clinical-event-stream")).toContain("min-height: 0");
    expect(blockFor(".clinical-event-row")).toContain("repeat(2, minmax(0, 1fr))");
    expect(blockFor(".clinical-event-chip p")).not.toContain("max-height");
    expect(blockFor(".clinical-event-chip p")).not.toContain("overflow: hidden");
  });

  it("defines the polished clinical visual system hooks", () => {
    expect(tokensCss).toContain("--color-primary: #0071e3");
    expect(tokensCss).toContain("--clinical-primary: var(--color-primary)");
    expect(tokensCss).toContain("--clinical-apple-bg: var(--color-canvas)");
    expect(tokensCss).toContain(
      "--shadow-card: 0 0 0 0.5px rgba(0, 0, 0, 0.04), 0 4px 12px rgba(0, 0, 0, 0.06)",
    );
    expect(tokensCss).toContain(
      "--shadow-pop: 0 0 0 0.5px rgba(0, 0, 0, 0.05), 0 12px 32px rgba(0, 0, 0, 0.08)",
    );
    expect(tokensCss).not.toContain("#1466d8");
    expect(css.match(/^:root\s*\{/gm) ?? []).toHaveLength(0);
    expect(css).not.toContain("#1466d8");
    expect(blockFor(".ui-top-nav")).not.toContain("linear-gradient");
    expect(blockFor(".clinical-app-shell")).toContain("var(--clinical-apple-bg)");
    expect(blockFor(".clinical-app-shell")).not.toContain("radial-gradient");
    expect(blockFor(".clinical-top-nav")).toContain("backdrop-filter");
    expect(blockFor(".clinical-top-nav")).toContain("var(--clinical-command-surface)");
    expect(blockFor(".clinical-top-nav")).toContain("color: var(--clinical-command-ink)");
    expect(blockFor(".clinical-nav-tabs")).toContain("border-radius: var(--radius-pill)");
    expect(blockFor(".clinical-nav-tab-active")).toContain("var(--clinical-surface)");
    expect(blockFor(".clinical-nav-tab-active")).toContain("var(--clinical-accent-blue)");
    expect(blockFor(".clinical-nav-tab-active")).toContain("var(--clinical-button-border-shadow)");
    expect(blockFor(".clinical-nav-tab-active")).not.toContain("0 1px 4px");
    expect(blockFor(".clinical-reset-button")).toContain("var(--clinical-button-border-shadow)");
    expect(blockFor(".clinical-sse-pill")).toContain("var(--clinical-button-border-shadow)");
    expect(blockFor(".clinical-conversation-card .clinical-composer-send:hover:not(:disabled)")).toContain(
      "transform: none",
    );
    expect(blockFor(".clinical-logo-mark circle")).toContain("var(--clinical-accent-blue)");
    expect(blocksFor(".clinical-conversation-card")).toContainEqual(
      expect.stringContaining("var(--shadow-pop)"),
    );
    expect(blocksFor(".clinical-left-column").join("\n")).not.toContain("opacity");
    expect(blocksFor(".clinical-right-column").join("\n")).not.toContain("opacity");
    expect(blockFor(".clinical-empty-state")).toContain("grid-template-columns");
    expect(blockFor(".clinical-empty-state-icon")).toContain("border-radius: 8px");
    expect(blockFor(".clinical-empty-state-icon svg")).toContain("stroke-width");
    expect(blockFor(".clinical-empty-state-icon::before")).toBe("");
    expect(blockFor(".clinical-empty-state-compact")).toContain("padding");
    expect(blockFor(".clinical-conversation-card .clinical-composer-textarea")).toContain("box-shadow");
  });

  it("uses restrained bubbles, medical cards, and database table polish", () => {
    expect(blockFor(".ui-message-bubble-assistant")).toContain("background: var(--color-surface)");
    expect(blockFor(".ui-message-bubble-assistant::before")).toContain("var(--color-primary)");
    expect(blockFor(".ui-message-bubble-user")).toContain("background: var(--color-surface-muted)");
    expect(blockFor(".clinical-medical-card")).toContain("min-height: 180px");
    expect(blockFor(".clinical-medical-card")).toContain("border-radius: var(--radius-lg)");
    expect(blockFor(".clinical-human-review")).toContain("width: 10px");
    expect(blockFor(".clinical-human-review")).not.toContain("font-size");
    expect(blockFor(".database-table th")).toContain("position: sticky");
    expect(blockFor(".database-table tbody tr:nth-child(even)")).toContain("var(--color-surface-muted)");
    expect(blockFor(".database-table tbody tr:hover")).toContain("var(--color-primary-soft)");
    expect(blockFor(".database-table td:nth-child(1)")).toContain("text-align: right");
    expect(blockFor(".database-table td:nth-child(1)")).toContain("tabular-nums");
  });

  it("keeps the Apple-inspired command layer compact on mobile", () => {
    const media720 = mediaBlockFor("max-width: 720px");

    expect(blockFor(".clinical-top-nav", media720)).toContain("grid-template-columns: 1fr");
    expect(blockFor(".clinical-nav-tabs", media720)).toContain("grid-template-columns");
    expect(blockFor(".clinical-nav-tabs", media720)).toContain("repeat(3, minmax(0, 1fr))");
    expect(blockFor(".clinical-top-nav-patient .clinical-nav-tabs", media720)).toContain(
      "repeat(2, minmax(0, 1fr))",
    );
    expect(blockFor(".clinical-nav-tab", media720)).toContain("min-height");
    expect(blockFor(".clinical-user-area", media720)).toContain("justify-content: flex-start");
  });
});
