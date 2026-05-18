import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles/globals.css"), "utf8");

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
  it("gives the base desktop cockpit a dominant consultation center and compact support rails", () => {
    const dashboard = blockFor(".clinical-dashboard");

    expect(dashboard).toContain("minmax(640px, 1fr)");
    expect(dashboard).toContain("clamp(260px, 18vw, 280px)");
    expect(dashboard).toContain("clamp(340px, 24vw, 370px)");
    expect(dashboard).not.toContain("500px");
    expect(dashboard).toContain('"left center right"');
    expect(dashboard).toContain('"event event right"');
  });

  it("keeps the cockpit usable at medium desktop widths without a wide right rail", () => {
    const media1450 = mediaBlockFor("max-width: 1450px");
    const dashboard = blockFor(".clinical-dashboard", media1450);

    expect(dashboard).toContain("280px minmax(580px, 1fr) 340px");
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
    expect(blockFor(".clinical-event-stream")).toContain("min-height: 0");
    expect(blockFor(".clinical-event-row")).toContain("repeat(4, minmax(0, 1fr))");
    expect(blockFor(".clinical-event-chip p")).toContain("max-height: 3.2em");
  });

  it("defines the polished clinical visual system hooks", () => {
    expect(css).toContain("--clinical-glass-ink");
    expect(css).toContain("--clinical-panel-shadow");
    expect(blockFor(".clinical-top-nav")).toContain("backdrop-filter");
    expect(blockFor(".clinical-nav-tabs")).toContain("border-radius: 999px");
    expect(blockFor(".clinical-nav-tab-active")).toContain("box-shadow");
    expect(blocksFor(".clinical-conversation-card")).toContainEqual(
      expect.stringContaining("var(--clinical-stage-shadow)"),
    );
    expect(blocksFor(".clinical-left-column")).toContainEqual(expect.stringContaining("opacity"));
    expect(blocksFor(".clinical-right-column")).toContainEqual(expect.stringContaining("opacity"));
    expect(blockFor(".clinical-empty-state")).toContain("grid-template-columns");
    expect(blockFor(".clinical-empty-state-icon")).toContain("border-radius: 8px");
    expect(blockFor(".clinical-empty-state-icon svg")).toContain("stroke-width");
    expect(blockFor(".clinical-empty-state-icon::before")).toBe("");
    expect(blockFor(".clinical-empty-state-compact")).toContain("padding");
    expect(blockFor(".clinical-conversation-card .clinical-composer-textarea")).toContain("box-shadow");
  });

  it("keeps the Apple-inspired command layer compact on mobile", () => {
    const media700 = mediaBlockFor("max-width: 700px");

    expect(blockFor(".clinical-top-nav", media700)).toContain("grid-template-columns: 1fr");
    expect(blockFor(".clinical-nav-tabs", media700)).toContain("grid-template-columns");
    expect(blockFor(".clinical-nav-tab", media700)).toContain("min-height");
    expect(blockFor(".clinical-user-area", media700)).toContain("justify-content: flex-start");
  });
});
