import { afterEach, describe, expect, it } from "vitest";

import { buildReplayDemoContext, resolveReplayFixtureCase } from "./demo-mode";

afterEach(() => {
  window.localStorage.clear();
});

describe("resolveReplayFixtureCase", () => {
  it("routes patient first symptom prompt to the triage question fixture", () => {
    expect(resolveReplayFixtureCase("patient", "最近两个月大便带血")).toBe("demo_patient_triage_question");
  });

  it("routes patient triage answer prompt to the final triage fixture", () => {
    expect(resolveReplayFixtureCase("patient", "持续时间超过1个月")).toBe("demo_patient_triage_final");
  });

  it("routes doctor prompts to the decision fixture", () => {
    expect(resolveReplayFixtureCase("doctor", "生成治疗建议")).toBe("demo_doctor_decision");
  });
});

describe("buildReplayDemoContext", () => {
  it("does not change context when replay demo mode is off", () => {
    expect(buildReplayDemoContext("patient", "最近两个月大便带血", { registry_patient_id: 1 })).toEqual({
      registry_patient_id: 1,
    });
  });

  it("adds a fixture case when local demo mode is replay", () => {
    window.localStorage.setItem("demoMode", "replay");

    expect(buildReplayDemoContext("patient", "最近两个月大便带血", { registry_patient_id: 1 })).toEqual({
      registry_patient_id: 1,
      fixture_case: "demo_patient_triage_question",
      fixture_tick_delay_ms: 450,
    });
  });
});
