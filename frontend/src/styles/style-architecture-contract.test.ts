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

function isAllowedWorkspacePathLine(line: string) {
  return (
    /^\s*import\b/.test(line) ||
    /\bfrom\s*["'][^"']*(?:\/|\\)workspace(?:\/|\\)/.test(line) ||
    /(?:\/|\\)features(?:\/|\\)workspace(?:\/|\\)/.test(line)
  );
}

function stripAllowedWorkspaceSubstrings(line: string) {
  return line
    .replace(/\bdata-testid\s*=\s*(?:"[^"]*"|'[^']*'|\{[^}]*\})/g, "")
    .replace(/\baria-describedby\s*=\s*(?:"[^"]*"|'[^']*'|\{[^}]*\})/g, "")
    .replace(/\b(?:getByTestId|queryByTestId|findByTestId)\(\s*(?:"[^"]*"|'[^']*'|`[^`]*`)[^)]*\)/g, "");
}

function hasWorkspaceVisualReference(line: string) {
  return (
    /["'`][^"'`]*workspace-[a-z0-9-]+[^"'`]*["'`]/.test(line) ||
    /["'`]workspace-["'`]\s*\+/.test(line) ||
    /`[^`]*workspace-\$\{/.test(line) ||
    /\[\s*["'`]workspace["'`]\s*,[^\]]+\]\.join\(\s*["'`]-["'`]\s*\)/.test(line) ||
    /closest\(["']\.workspace-/.test(line)
  );
}

function workspaceVisualClassReferences(source: string) {
  return source
    .split(/\r?\n/)
    .map((line, index) => ({ line, number: index + 1 }))
    .filter(({ line }) => !isAllowedWorkspacePathLine(line))
    .map(({ line, number }) => ({ line: stripAllowedWorkspaceSubstrings(line), number }))
    .filter(({ line }) => hasWorkspaceVisualReference(line));
}

function customPropertyValue(name: string, source: string) {
  const match = source.match(new RegExp(`${name}\\s*:\\s*([^;]+);`, "m"));
  return match?.[1].trim() ?? "";
}

function matchingLines(source: string, regex: RegExp, fileLabel: string) {
  const lineRegex = new RegExp(regex.source, regex.flags.replace(/g/g, ""));
  return source
    .split(/\r?\n/)
    .map((line, index) => ({ line, number: index + 1 }))
    .filter(({ line }) => lineRegex.test(line))
    .map(({ line, number }) => `${fileLabel}:${number}: ${line.trim()}`);
}

function inlineStyleReferences(source: string, fileLabel: string) {
  return [...source.matchAll(/\bstyle\s*=\s*(?:\{|\r?\n)/g)].map((match) => {
    const index = match.index ?? 0;
    const number = lineNumberAt(source, index);
    const lineStart = source.lastIndexOf("\n", index) + 1;
    const lineEnd = source.indexOf("\n", index);
    const line = source.slice(lineStart, lineEnd === -1 ? source.length : lineEnd).trim();
    return `${fileLabel}:${number}: ${line}`;
  });
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

    expect(matchingLines(globalsCss, /#[0-9a-fA-F]{6}/, globalsCssPath)).toEqual([]);
    expect(matchingLines(globalsCss, /font-size\s*:\s*0\.\d+rem/, globalsCssPath)).toEqual([]);
    expect(matchingLines(globalsCss, /font-weight\s*:\s*(650|750|760|800)\b/, globalsCssPath)).toEqual([]);
    expect(matchingLines(globalsCss, /border-radius\s*:\s*(14|16|18|22)px/, globalsCssPath)).toEqual([]);
    expect(
      matchingLines(globalsCss, /padding(?:-[a-z]+)?\s*:[^;\n]*(9px|10px|14px|18px|22px)/, globalsCssPath),
    ).toEqual([]);
  });

  it("keeps structural gradients out of tokens", () => {
    const tokensCss = read(tokensCssPath);
    const panelHeaderSurface = customPropertyValue("--panel-header-surface", tokensCss);
    const bgAccent = customPropertyValue("--bg-accent", tokensCss);

    expect(panelHeaderSurface).not.toBe("");
    expect(bgAccent).not.toBe("");
    expect(panelHeaderSurface).not.toMatch(/gradient\(/i);
    expect(bgAccent).not.toMatch(/gradient\(/i);
  });

  it("keeps card renderer visuals out of inline styles", () => {
    const cardRendererFiles = [
      resolve(process.cwd(), "src/features/cards/card-renderers.tsx"),
      resolve(process.cwd(), "src/features/cards/card-renderers-extended.tsx"),
    ];

    for (const file of cardRendererFiles) {
      expect(existsSync(file), file).toBe(true);
      const source = read(file);

      expect(inlineStyleReferences(source, file)).toEqual([]);
      expect(source.match(/#[0-9a-fA-F]{6}/g) ?? [], file).toEqual([]);
      expect(source.match(/fontSize\s*:/g) ?? [], file).toEqual([]);
      expect(workspaceVisualClassReferences(source), file).toEqual([]);
    }
  });
});
