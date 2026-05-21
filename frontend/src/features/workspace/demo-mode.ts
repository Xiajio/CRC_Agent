import type { Scene } from "../../app/api/types";

const DEMO_TICK_DELAY_MS = 450;

export function isReplayDemoMode(): boolean {
  const env = import.meta.env as Record<string, string | boolean | undefined>;
  if (env.VITE_DEMO_MODE === "replay") {
    return true;
  }

  if (typeof window === "undefined") {
    return false;
  }

  try {
    return window.localStorage.getItem("demoMode") === "replay";
  } catch {
    return false;
  }
}

export function resolveReplayFixtureCase(scene: Scene, prompt: string): string {
  if (scene === "doctor") {
    return "demo_doctor_decision";
  }

  const normalized = prompt.trim();
  if (normalized.includes("超过1个月") || normalized.includes("1个月以上")) {
    return "demo_patient_triage_final";
  }

  return "demo_patient_triage_question";
}

export function buildReplayDemoContext(
  scene: Scene,
  prompt: string,
  context?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  if (!isReplayDemoMode()) {
    return context;
  }

  return {
    ...(context ?? {}),
    fixture_case: resolveReplayFixtureCase(scene, prompt),
    fixture_tick_delay_ms: DEMO_TICK_DELAY_MS,
  };
}
