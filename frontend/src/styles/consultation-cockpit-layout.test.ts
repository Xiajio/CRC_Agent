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
      expect.stringContaining("clamp(430px, 56vh, 620px)"),
    );
    expect(blockFor(".clinical-event-stream")).toContain("min-height: 0");
    expect(blockFor(".clinical-event-row")).toContain("repeat(4, minmax(0, 1fr))");
    expect(blockFor(".clinical-event-chip p")).toContain("max-height: 3.2em");
  });
});
