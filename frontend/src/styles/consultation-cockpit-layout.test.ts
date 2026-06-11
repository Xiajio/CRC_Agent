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
  return [...source.matchAll(new RegExp(`(?:^|\\n)\\s*${escaped}\\s*\\{(?<body>[\\s\\S]*?)\\n\\s*\\}`, "gm"))].map(
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
    expect(dashboard).toContain('"event event event"');

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
    expect(blockFor(".clinical-event-row")).toContain("repeat(auto-fill, minmax(280px, 1fr))");
    expect(blockFor(".clinical-event-chip p")).not.toContain("max-height");
    expect(blockFor(".clinical-event-chip p")).not.toContain("overflow: hidden");
  });

  it("ships a collapsible console-style event stream", () => {
    expect(blockFor(".clinical-event-console-meta")).toContain("font-family: var(--font-mono)");
    expect(blockFor(".clinical-event-console-meta")).toContain("margin-left: auto");
    expect(blockFor(".clinical-event-console-toggle")).toContain("cursor: pointer");
    expect(blockFor('[data-theme="doctor-cockpit"] .clinical-event-stream')).toContain(
      "background: var(--color-canvas)",
    );
  });

  it("defines the dual scene themes on the document root", () => {
    expect(tokensCss).toContain(':root[data-theme="doctor-cockpit"]');
    expect(tokensCss).toContain(':root[data-theme="patient-care"]');

    const cockpit = blockFor(':root[data-theme="doctor-cockpit"]', tokensCss);
    expect(cockpit).toContain("color-scheme: dark");
    expect(cockpit).toContain("--color-canvas: #090a0f");
    expect(cockpit).toContain("--color-surface: #12141c");
    expect(cockpit).toContain("--color-surface-muted: #1a1d26");
    expect(cockpit).toContain("--color-primary: #4d8dff");
    expect(cockpit).toContain("--color-primary-soft: rgba(77, 141, 255, 0.14)");
    expect(cockpit).toContain("--color-text: #eaf2ff");
    expect(cockpit).toContain("--color-text-muted: rgba(255, 255, 255, 0.56)");
    expect(cockpit).toContain("--color-border: rgba(255, 255, 255, 0.08)");
    expect(cockpit).toContain("--shadow-card-resting: none");
    expect(cockpit).toContain("--clinical-glass-border: rgba(255, 255, 255, 0.06)");
    expect(cockpit).toContain("--dashboard-left-width: 280px");

    const care = blockFor(':root[data-theme="patient-care"]', tokensCss);
    expect(care).toContain("color-scheme: light");
    expect(care).toContain("--color-canvas: #ffffff");
    expect(care).toContain("--color-surface-muted: #f8f9fa");
    expect(care).toContain("--color-primary: #087f6f");
    expect(care).toContain("--color-text: #111827");
    expect(care).toContain("--color-text-muted: #4b5563");
    expect(care).toContain("--color-border: #e5e7eb");
    expect(care).toContain("--shadow-card: 0 1px 3px rgba(0, 0, 0, 0.05)");
    expect(care).toContain("--shadow-card-resting: none");
    expect(care).toContain("--dashboard-left-width: 260px");
  });

  it("adds a non-interactive aurora layer behind doctor shells", () => {
    expect(blockFor(".clinical-app-shell-doctor")).toContain("isolation: isolate");
    expect(blockFor(".clinical-app-shell-doctor")).toContain("overflow-x: clip");
    expect(blockFor(".clinical-app-shell-doctor::before")).toContain("radial-gradient");
    expect(blockFor(".clinical-app-shell-doctor::before")).toContain("filter: blur(80px)");
    expect(blockFor(".clinical-app-shell-doctor::before")).toContain("pointer-events: none");
    expect(blockFor(".clinical-app-shell-doctor::before")).toContain("z-index: 0");
    expect(blockFor(".clinical-app-shell-doctor > *")).toContain("z-index: 1");
    expect(blockFor(".clinical-app-shell-doctor .clinical-top-nav")).toContain("z-index: 20");

    expect(css.indexOf(".clinical-app-shell-doctor > *")).toBeLessThan(
      css.indexOf(".clinical-app-shell-doctor .clinical-top-nav"),
    );
    expect(css.indexOf(".clinical-app-shell-doctor .clinical-top-nav")).toBeLessThan(
      css.indexOf(".clinical-app-shell-database .clinical-page-shell"),
    );
  });

  it("removes hard borders from doctor cockpit major cards", () => {
    const panel = blockFor('[data-theme="doctor-cockpit"] .ui-card-clinical-panel');
    const card = blockFor('[data-theme="doctor-cockpit"] .clinical-card');
    const conversation = blockFor('[data-theme="doctor-cockpit"] .clinical-conversation-card');
    const hover = blockFor('[data-theme="doctor-cockpit"] .clinical-conversation-card:hover');
    const softPanel = blockFor('[data-theme="doctor-cockpit"] .ui-card-clinical-panel.ui-card-soft');
    const selectedPanel = blockFor('[data-theme="doctor-cockpit"] .ui-card-clinical-panel.ui-card-selected');
    const internalBorders = blockFor(
      '[data-theme="doctor-cockpit"] .clinical-multimodal-card .clinical-panel-header',
    );

    expect(panel).toContain("border-color: transparent");
    expect(panel).toContain("background: var(--color-surface)");
    expect(panel).toContain("box-shadow: none");
    expect(card).toContain("border-color: transparent");
    expect(card).toContain("background: var(--color-surface)");
    expect(card).toContain("box-shadow: none");
    expect(conversation).toContain("background: #151821");
    expect(conversation).toContain("border-color: transparent");
    expect(conversation).toContain("box-shadow: none");
    expect(css).toContain(
      [
        '[data-theme="doctor-cockpit"] .ui-card-clinical-panel:hover,',
        '[data-theme="doctor-cockpit"] .clinical-card:hover,',
        '[data-theme="doctor-cockpit"] .clinical-conversation-card:hover',
      ].join("\n"),
    );
    expect(hover).toContain("border-color: transparent");
    expect(hover).toContain("box-shadow: none");
    expect(softPanel).toContain("background: var(--color-surface-muted)");
    expect(selectedPanel).toContain("border-color: transparent");
    expect(selectedPanel).toContain("background: var(--color-primary-soft)");
    expect(css).toContain(
      [
        '[data-theme="doctor-cockpit"] .clinical-panel-header,',
        '[data-theme="doctor-cockpit"] .clinical-card-section-bordered,',
        '[data-theme="doctor-cockpit"] .clinical-multimodal-card .clinical-panel-header',
      ].join("\n"),
    );
    expect(internalBorders).toContain("border-color: var(--color-border-soft)");
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
    expect(blockFor(".clinical-company-logo")).toContain("height: 44px");
    expect(blockFor(".clinical-company-logo")).toContain("object-fit: contain");
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

  it("sharpens the patient assistant layout and light theme surfaces", () => {
    const media1450 = mediaBlockFor("max-width: 1450px");

    expect(blockFor(".clinical-patient-dashboard.clinical-patient-dashboard-assistant")).toContain(
      "minmax(0, min(860px, 100%))",
    );
    expect(blockFor(".clinical-patient-dashboard.clinical-patient-dashboard-profile")).toContain(
      "minmax(0, min(860px, 100%))",
    );
    expect(blockFor(".clinical-patient-dashboard.clinical-patient-dashboard-upload")).toContain(
      "minmax(0, min(860px, 100%))",
    );
    expect(blockFor(".clinical-patient-dashboard-assistant", media1450)).toContain(
      "minmax(0, min(860px, 100%))",
    );
    expect(blockFor(".clinical-patient-left-column-collapsed")).toContain("display: none");
    expect(blockFor(".patient-assistant-home")).toContain("text-align: center");
    expect(blockFor('[data-theme="patient-care"] .clinical-conversation-card')).toContain("box-shadow: none");
    expect(blockFor('[data-theme="patient-care"] .clinical-conversation-card')).toContain(
      "border: 1px solid var(--color-border)",
    );
    expect(blockFor('[data-theme="patient-care"] .clinical-conversation-card .clinical-composer-textarea')).toContain(
      "color: var(--color-text)",
    );
  });

  it("keeps the Apple-inspired command layer compact on mobile", () => {
    const media720 = mediaBlockFor("max-width: 720px");

    expect(blockFor(".clinical-top-nav", media720)).toContain("grid-template-columns: 1fr");
    expect(blockFor(".clinical-nav-tabs", media720)).toContain("grid-template-columns");
    expect(blockFor(".clinical-nav-tabs", media720)).toContain("repeat(auto-fit, minmax(108px, 1fr))");
    expect(blockFor(".clinical-top-nav-doctor .clinical-nav-tabs", media720)).toContain(
      "repeat(3, minmax(0, 1fr))",
    );
    expect(blockFor(".clinical-top-nav-patient .clinical-nav-tabs", media720)).toContain(
      "repeat(3, minmax(0, 1fr))",
    );
    expect(blockFor(".clinical-nav-tab", media720)).toContain("min-height");
    expect(blockFor(".clinical-user-area", media720)).toContain("justify-content: flex-start");
  });
});
