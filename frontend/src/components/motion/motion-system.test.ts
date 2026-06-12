import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { motionTokens } from "./motion-tokens";

const globalsCss = readFileSync(resolve(process.cwd(), "src/styles/globals.css"), "utf8");
const tokensCss = readFileSync(resolve(process.cwd(), "src/styles/tokens.css"), "utf8");
const motionTokenSource = readFileSync(resolve(process.cwd(), "src/components/motion/motion-tokens.ts"), "utf8");
const gsapContextSource = readFileSync(resolve(process.cwd(), "src/components/motion/use-gsap-context.ts"), "utf8");
const shellRevealSource = readFileSync(resolve(process.cwd(), "src/components/motion/use-shell-reveal.ts"), "utf8");
const highlightFlashSource = readFileSync(resolve(process.cwd(), "src/components/motion/use-highlight-flash.ts"), "utf8");
const highlightPulseSource = readFileSync(resolve(process.cwd(), "src/components/motion/use-highlight-pulse.ts"), "utf8");
const viewTransitionSource = readFileSync(resolve(process.cwd(), "src/components/motion/use-view-transition.ts"), "utf8");
const anatomyMapSource = readFileSync(resolve(process.cwd(), "src/features/anatomy/colorectal-anatomy-map.tsx"), "utf8");
const wholeBodyOverviewSource = readFileSync(
  resolve(process.cwd(), "src/features/anatomy/whole-body-anatomy-overview.tsx"),
  "utf8",
);
const packageJson = JSON.parse(
  readFileSync(resolve(process.cwd(), "package.json"), "utf8"),
) as {
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
};

function cssToken(name: string) {
  const match = tokensCss.match(new RegExp(`--${name}:\\s*([^;]+);`));
  return match?.[1].trim() ?? "";
}

function blockFor(selector: string, source = globalsCss) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = source.match(new RegExp(`${escaped}\\s*\\{(?<body>[\\s\\S]*?)\\n\\s*\\}`, "m"));
  return match?.groups?.body ?? "";
}

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

function stripStaticSvgStrokeWidth(source: string) {
  return source.replace(/\s+strokeWidth=\{\d+(?:\.\d+)?\}/g, "");
}

describe("motion design system", () => {
  it("keeps CSS motion tokens mirrored by the TypeScript GSAP adapter", () => {
    expect(cssToken("motion-duration-feedback")).toBe(motionTokens.css.durationFeedback);
    expect(cssToken("motion-duration-highlight")).toBe(motionTokens.css.durationHighlight);
    expect(cssToken("motion-duration-transition")).toBe(motionTokens.css.durationTransition);
    expect(cssToken("motion-duration-enter")).toBe(motionTokens.css.durationEnter);
    expect(cssToken("motion-ease-out")).toBe(motionTokens.css.easeOut);
    expect(cssToken("motion-ease-out")).toBe("cubic-bezier(0.16, 1, 0.3, 1)");
    expect(cssToken("motion-gsap-ease-out")).toBe(motionTokens.css.gsapEaseOut);
    expect(cssToken("motion-enter-y")).toBe(motionTokens.css.enterY);
    expect(cssToken("motion-highlight-scale")).toBe(motionTokens.css.highlightScale);
    expect(cssToken("motion-highlight-ring-opacity")).toBe(motionTokens.css.highlightRingOpacity);

    expect(motionTokens.duration.feedback).toBe(0.16);
    expect(motionTokens.duration.highlight).toBe(0.24);
    expect(motionTokens.duration.transition).toBe(0.24);
    expect(motionTokens.duration.enter).toBe(0.32);
    expect(motionTokens.ease.out).toBe("power3.out");
  });

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
        /\b(?:\d+(?:\.\d+)?ms|\d*\.\d+s|\d+s)\s+ease\b/,
        "src/styles/globals.css",
      ),
    ).toEqual([]);
  });

  it("removes broad transitions and keeps highlight feedback on transform and opacity", () => {
    expect(globalsCss).not.toMatch(/transition:\s*all\b/);
    expect(blockFor(".motion-highlight-pulse::after")).toContain("opacity: 0");
    expect(blockFor(".motion-highlight-pulse::after")).toContain("transform: scale(1)");
    expect(globalsCss).toContain("@keyframes motion-highlight-pulse");
    expect(globalsCss).toContain("opacity: var(--motion-highlight-ring-opacity)");
    expect(globalsCss).toContain("transform: scale(var(--motion-highlight-scale))");
    expect(blockFor(".anatomy-map-region")).toContain("opacity var(--motion-duration-feedback)");
    expect(blockFor(".anatomy-map-region")).toContain("transform var(--motion-duration-feedback)");
    expect(blockFor(".anatomy-map-region")).not.toContain("stroke-width 160ms");
    expect(blockFor(".anatomy-highlight-visuals")).toContain("grid-template-columns");
    expect(blockFor(".whole-body-anatomy-region")).toContain("opacity var(--motion-duration-feedback)");
    expect(blockFor(".whole-body-anatomy-region")).toContain("transform var(--motion-duration-feedback)");
    expect(blockFor(".whole-body-anatomy-region")).not.toContain("stroke-width 160ms");
    expect(blockFor('.whole-body-anatomy-region[data-active="true"]')).toContain(
      "transform: scale(var(--motion-highlight-scale))",
    );
  });

  it("keeps GSAP scoped, reduced-motion gated, and free of paint-heavy animation props", () => {
    const animatedSources = [
      motionTokenSource,
      gsapContextSource,
      shellRevealSource,
      highlightFlashSource,
      highlightPulseSource,
      viewTransitionSource,
      anatomyMapSource,
      stripStaticSvgStrokeWidth(wholeBodyOverviewSource),
    ].join("\n");

    expect(gsapContextSource).toContain("usePrefersReducedMotion");
    expect(gsapContextSource).toContain("gsap.context");
    expect(gsapContextSource).toContain("context.revert()");
    expect(shellRevealSource).toContain("useGsapContext");
    expect(viewTransitionSource).toContain("useGsapContext");
    expect(anatomyMapSource).toContain("useGsapContext");
    expect(wholeBodyOverviewSource).toContain("useGsapContext");
    expect(highlightFlashSource).toContain("useHighlightPulse");
    expect(highlightPulseSource).toContain("usePrefersReducedMotion");

    expect(animatedSources).not.toContain("box" + "Shadow");
    expect(animatedSources).not.toContain("stroke" + "Width");
    expect(animatedSources).not.toContain("auto" + "Alpha");
    expect(animatedSources).not.toContain("power" + "2.out");
    expect(animatedSources).not.toMatch(/\bwidth:\s*["']?\+=?/);
    expect(animatedSources).not.toMatch(/\bheight:\s*["']?\+=?/);
    expect(animatedSources).not.toMatch(/\btop:\s*["']?\+=?/);
    expect(animatedSources).not.toMatch(/\bmargin\w*:\s*["']?\+=?/);
  });
});
