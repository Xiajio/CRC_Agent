import { existsSync, readFileSync, readdirSync } from "node:fs";
import { extname, join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

const srcRoot = resolve(process.cwd(), "src");
const globalsCssPath = resolve(process.cwd(), "src/styles/globals.css");
const tokensCssPath = resolve(process.cwd(), "src/styles/tokens.css");

function collectFiles(root: string, extensions: string[]) {
  const files: string[] = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectFiles(path, extensions));
      continue;
    }
    if (extensions.includes(extname(entry.name))) {
      files.push(path);
    }
  }
  return files;
}

function read(path: string) {
  return readFileSync(path, "utf8");
}

function nonTestTsxFiles() {
  return collectFiles(srcRoot, [".tsx"]).filter((file) => !file.endsWith(".test.tsx"));
}

function workspaceCssSelectors(source: string) {
  return [...source.matchAll(/([^{}]+)\{/g)]
    .flatMap((match) => {
      const number = lineNumberAt(source, match.index ?? 0);
      return match[1].split(",").map((selector) => `${number}: ${selector.trim()}`);
    })
    .filter((selector) => selector.includes(".workspace-"));
}

function lineNumberAt(source: string, index: number) {
  return source.slice(0, index).split(/\r?\n/).length;
}

function isAllowedWorkspaceReference(line: string) {
  return (
    /^\s*import\b/.test(line) ||
    /\bfrom\s*["'][^"']*(?:\/|\\)workspace(?:\/|\\)/.test(line) ||
    /(?:\/|\\)features(?:\/|\\)workspace(?:\/|\\)/.test(line) ||
    /data-testid|aria-describedby|getByTestId|queryByTestId|findByTestId/.test(line)
  );
}

function workspaceVisualClassReferences(source: string) {
  return source
    .split(/\r?\n/)
    .map((line, index) => ({ line, number: index + 1 }))
    .filter(({ line }) => /workspace-[a-z0-9-]+/.test(line))
    .filter(({ line }) => !isAllowedWorkspaceReference(line))
    .filter(({ line }) => /["'`][^"'`]*workspace-[a-z0-9-]+[^"'`]*["'`]|closest\(["']\.workspace-/.test(line));
}

function customPropertyValue(name: string, source: string) {
  const match = source.match(new RegExp(`${name}:\\s*([^;]+);`, "m"));
  return match?.[1].trim() ?? "";
}

describe("style architecture contract", () => {
  it("keeps workspace visual classes out of CSS selectors", () => {
    const globalsCss = read(globalsCssPath);
    const selectors = workspaceCssSelectors(globalsCss);

    expect(selectors).toEqual([]);
    expect(globalsCss).not.toContain(".clinical-conversation-card .workspace-composer-send");
  });

  it("keeps workspace visual classes out of TSX class names", () => {
    const offenders = nonTestTsxFiles().flatMap((file) =>
      workspaceVisualClassReferences(read(file)).map(({ line, number }) => `${file}:${number}: ${line.trim()}`),
    );

    expect(offenders).toEqual([]);
  });

  it("keeps globals.css on the visual token scale", () => {
    const globalsCss = read(globalsCssPath);

    expect(globalsCss.match(/#[0-9a-fA-F]{6}/g) ?? []).toEqual([]);
    expect(globalsCss.match(/font-size\s*:\s*0\.\d+rem/g) ?? []).toEqual([]);
    expect(globalsCss.match(/font-weight\s*:\s*(650|750|760|800)\b/g) ?? []).toEqual([]);
    expect(globalsCss.match(/border-radius\s*:\s*(14|16|18|22)px/g) ?? []).toEqual([]);
    expect(globalsCss.match(/padding(?:-[a-z]+)?\s*:[^;\n]*(9px|10px|14px|18px|22px)/g) ?? []).toEqual([]);
  });

  it("keeps structural gradients out of tokens", () => {
    const tokensCss = read(tokensCssPath);
    const panelHeaderSurface = customPropertyValue("--panel-header-surface", tokensCss);
    const bgAccent = customPropertyValue("--bg-accent", tokensCss);

    expect(panelHeaderSurface).not.toBe("");
    expect(bgAccent).not.toBe("");
    expect(panelHeaderSurface).not.toMatch(/gradient\(/);
    expect(bgAccent).not.toMatch(/gradient\(/);
  });

  it("keeps card renderer visuals out of inline styles", () => {
    const cardRendererFiles = [
      resolve(process.cwd(), "src/features/cards/card-renderers.tsx"),
      resolve(process.cwd(), "src/features/cards/card-renderers-extended.tsx"),
    ];

    for (const file of cardRendererFiles) {
      expect(existsSync(file), file).toBe(true);
      const source = read(file);

      expect(source, file).not.toContain("style={{");
      expect(source.match(/#[0-9a-fA-F]{6}/g) ?? [], file).toEqual([]);
      expect(source.match(/fontSize\s*:/g) ?? [], file).toEqual([]);
      expect(workspaceVisualClassReferences(source), file).toEqual([]);
    }
  });
});
