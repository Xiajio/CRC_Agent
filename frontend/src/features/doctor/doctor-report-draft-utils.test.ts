import type { FrontendMessage } from "../../app/api/types";
import {
  buildDoctorReportDraftPrompt,
  buildDoctorReportPromptContext,
  DOCTOR_REPORT_DRAFT_ACTIONS,
  latestReportDraftFromMessages,
} from "./doctor-report-draft-utils";

function message(overrides: Partial<FrontendMessage>): FrontendMessage {
  return {
    cursor: "1",
    type: "ai",
    content: "",
    assetRefs: [],
    ...overrides,
  };
}

describe("doctor report draft utilities", () => {
  it("defines the lightweight report draft actions in doctor-facing order", () => {
    expect(DOCTOR_REPORT_DRAFT_ACTIONS.map((action) => action.title)).toEqual([
      "病例摘要草稿",
      "会诊报告草稿",
      "交接记录草稿",
    ]);
  });

  it("builds stable Markdown-oriented prompts for report draft actions", () => {
    const prompts = DOCTOR_REPORT_DRAFT_ACTIONS.map((action) => buildDoctorReportDraftPrompt(action));

    expect(prompts[0]).toContain("病例摘要草稿");
    expect(prompts[1]).toContain("会诊报告草稿");
    expect(prompts[2]).toContain("交接记录草稿");
    for (const prompt of prompts) {
      expect(prompt).toContain("Markdown");
      expect(prompt).toContain("资料来源");
      expect(prompt).toContain("缺失资料");
      expect(prompt).toContain("人工复核");
      expect(prompt).toContain("即使资料缺失，也必须输出病例/报告草稿模板");
      expect(prompt).toContain("缺失资料/待核实");
      expect(prompt).toContain("不要只输出缺失提醒");
    }
  });

  it("normalizes report prompt context from registry and case identifiers", () => {
    expect(
      buildDoctorReportPromptContext({
        registry_patient_id: "7",
        case_database_patient_id: 93,
      }),
    ).toEqual({
      registry_patient_id: 7,
      case_database_patient_id: "093",
    });
  });

  it("omits invalid report prompt context values", () => {
    expect(
      buildDoctorReportPromptContext({
        registry_patient_id: "bad-id",
        case_database_patient_id: "",
      }),
    ).toEqual({});
  });

  it("selects the latest non-empty assistant text as the report draft", () => {
    const latest = latestReportDraftFromMessages([
      message({ cursor: "u1", type: "user", content: "生成报告" }),
      message({ cursor: "a1", type: "ai", content: "旧草稿" }),
      message({ cursor: "a2", type: "ai", content: "   " }),
      message({ cursor: "a3", type: "ai", content: "最新报告草稿" }),
    ]);

    expect(latest).toEqual({
      cursor: "a3",
      text: "最新报告草稿",
    });
  });

  it("returns null when there is no assistant draft text", () => {
    expect(
      latestReportDraftFromMessages([
        message({ cursor: "u1", type: "user", content: "生成报告" }),
        message({ cursor: "a1", type: "ai", content: { structured: true } }),
      ]),
    ).toBeNull();
  });
});
