import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const css = readFileSync(resolve(process.cwd(), "src/styles/globals.css"), "utf8");

function selectorIndex(selector: string) {
  const index = css.indexOf(selector);
  expect(index, `${selector} should exist`).toBeGreaterThanOrEqual(0);
  return index;
}

function blockFor(selector: string) {
  const index = selectorIndex(selector);
  const blockStart = css.indexOf("{", index);
  const blockEnd = css.indexOf("}", blockStart);

  expect(blockStart, `${selector} should open a declaration block`).toBeGreaterThan(index);
  expect(blockEnd, `${selector} should close a declaration block`).toBeGreaterThan(blockStart);

  return css.slice(blockStart + 1, blockEnd);
}

describe("card material boundary CSS", () => {
  it("keeps clinical panel Card state selectors after the material boundary", () => {
    const material = selectorIndex(".ui-card-clinical-panel,");
    const soft = selectorIndex(".ui-card-clinical-panel.ui-card-soft");
    const warning = selectorIndex(".ui-card-clinical-panel.ui-card-warning");
    const danger = selectorIndex(".ui-card-clinical-panel.ui-card-danger");
    const selected = selectorIndex(".ui-card-clinical-panel.ui-card-selected");
    const hover = selectorIndex(".ui-card-clinical-panel:hover,");

    expect(soft).toBeGreaterThan(material);
    expect(warning).toBeGreaterThan(material);
    expect(danger).toBeGreaterThan(material);
    expect(selected).toBeGreaterThan(soft);
    expect(selected).toBeGreaterThan(warning);
    expect(selected).toBeGreaterThan(danger);
    expect(hover).toBeGreaterThan(selected);

    expect(blockFor(".ui-card-clinical-panel.ui-card-soft")).toContain(
      "background: var(--color-surface-muted);",
    );
    expect(blockFor(".ui-card-clinical-panel.ui-card-warning")).toContain(
      "border-color: var(--color-warning-border);",
    );
    expect(blockFor(".ui-card-clinical-panel.ui-card-warning")).toContain(
      "background: var(--color-warning-soft);",
    );
    expect(blockFor(".ui-card-clinical-panel.ui-card-danger")).toContain(
      "border-color: var(--color-danger-border);",
    );
    expect(blockFor(".ui-card-clinical-panel.ui-card-danger")).toContain(
      "background: var(--color-danger-soft);",
    );
    expect(blockFor(".ui-card-clinical-panel.ui-card-selected")).toContain(
      "border-color: var(--color-border-strong);",
    );
    expect(blockFor(".ui-card-clinical-panel.ui-card-selected")).toContain(
      "background: var(--color-primary-soft);",
    );

    const hoverBlock = blockFor(".ui-card-clinical-panel:hover,");
    expect(hoverBlock).toContain("border-color: var(--color-border-strong);");
    expect(hoverBlock).toContain("box-shadow: var(--shadow-pop);");
  });

  it("keeps bare clinical panel headings compact without changing panel headers", () => {
    const bareHeading = ".ui-card-clinical-panel > .ui-card-body > h2";
    const panelHeaderHeading = ".clinical-panel-header h2";

    expect(selectorIndex(bareHeading)).toBeGreaterThan(selectorIndex(".ui-card-clinical-panel,"));
    expect(selectorIndex(panelHeaderHeading)).toBeGreaterThan(selectorIndex(bareHeading));
    expect(css).not.toContain(".ui-card-clinical-panel h2 {");

    const block = blockFor(bareHeading);
    expect(block).toContain("margin: 0;");
    expect(block).toContain("font-size: var(--font-base);");
    expect(block).toContain("letter-spacing: 0.02em;");
  });
});
