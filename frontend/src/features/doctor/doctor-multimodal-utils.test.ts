import { describe, expect, it } from "vitest";

import type { JsonObject } from "../../app/api/types";
import type { CardPatientContext } from "../cards/card-renderers-extended";
import {
  buildMultimodalActionState,
  buildMultimodalPrompt,
  buildMultimodalPromptContext,
  groupMultimodalCards,
  MULTIMODAL_ACTIONS,
} from "./doctor-multimodal-utils";

describe("doctor multimodal utilities", () => {
  it("groups multimodal cards into imaging, pathology, and radiomics buckets", () => {
    const cards = [
      { cardType: "imaging_card", payload: { id: 1 } as JsonObject },
      { cardType: "tumor_detection_card", payload: { id: 2 } as JsonObject },
      { cardType: "tumor_screening_result", payload: { id: 7 } as JsonObject },
      { cardType: "pathology_card", payload: { id: 3 } as JsonObject },
      { cardType: "pathology_slide_card", payload: { id: 4 } as JsonObject },
      { cardType: "radiomics_report_card", payload: { id: 5 } as JsonObject },
      { cardType: "decision_card", payload: { id: 6 } as JsonObject },
    ];

    expect(groupMultimodalCards(cards)).toEqual([
      {
        key: "imaging",
        title: "影像",
        summary: "整理影像样本与肿瘤检测卡片。",
        cards: [
          { cardType: "imaging_card", payload: { id: 1 } },
          { cardType: "tumor_detection_card", payload: { id: 2 } },
          { cardType: "tumor_screening_result", payload: { id: 7 } },
        ],
      },
      {
        key: "pathology",
        title: "病理",
        summary: "整理病理报告与病理切片卡片。",
        cards: [
          { cardType: "pathology_card", payload: { id: 3 } },
          { cardType: "pathology_slide_card", payload: { id: 4 } },
        ],
      },
      {
        key: "radiomics",
        title: "影像组学",
        summary: "整理影像组学分析卡片。",
        cards: [{ cardType: "radiomics_report_card", payload: { id: 5 } }],
      },
    ]);
  });

  it("ignores non-multimodal card types", () => {
    const cards = [
      { cardType: "decision_card", payload: { id: 1 } as JsonObject },
      { cardType: "patient_card", payload: { id: 2 } as JsonObject },
    ];

    expect(groupMultimodalCards(cards)).toEqual([]);
  });

  it("builds a normalized multimodal prompt context", () => {
    const patientContext: CardPatientContext = {
      registry_patient_id: "7",
      case_database_patient_id: 7,
    };

    expect(buildMultimodalPromptContext(patientContext)).toEqual({
      registry_patient_id: 7,
      case_database_patient_id: "007",
    });
  });

  it("omits invalid or empty patient context values", () => {
    const patientContext: CardPatientContext = {
      registry_patient_id: "not-a-number",
      case_database_patient_id: "",
    };

    expect(buildMultimodalPromptContext(patientContext)).toEqual({});
  });

  it("requires case sample context for imaging and pathology actions", () => {
    const registryOnly = buildMultimodalPromptContext({
      registry_patient_id: 12,
    });
    const caseSample = buildMultimodalPromptContext({
      case_database_patient_id: 12,
    });

    expect(buildMultimodalActionState(MULTIMODAL_ACTIONS[0], registryOnly).disabled).toBe(true);
    expect(buildMultimodalActionState(MULTIMODAL_ACTIONS[1], caseSample).disabled).toBe(false);
  });

  it("requires registry patient context for summary and handoff actions", () => {
    const registryOnly = buildMultimodalPromptContext({
      registry_patient_id: 12,
    });
    const caseSample = buildMultimodalPromptContext({
      case_database_patient_id: 12,
    });

    expect(buildMultimodalActionState(MULTIMODAL_ACTIONS[2], registryOnly).disabled).toBe(false);
    expect(buildMultimodalActionState(MULTIMODAL_ACTIONS[3], registryOnly).disabled).toBe(false);
    expect(buildMultimodalActionState(MULTIMODAL_ACTIONS[2], caseSample).disabled).toBe(true);
    expect(buildMultimodalActionState(MULTIMODAL_ACTIONS[3], caseSample).disabled).toBe(true);
  });

  it("disables every action when there is no patient context", () => {
    for (const action of MULTIMODAL_ACTIONS) {
      expect(buildMultimodalActionState(action, {}).disabled).toBe(true);
    }
  });

  it("returns stable Chinese prompts for each action", () => {
    expect(MULTIMODAL_ACTIONS.map((action) => buildMultimodalPrompt(action))).toEqual([
      "请结合病例样本整理影像要点，并给出临床可执行的结论。",
      "请结合病例样本整理病理要点，并给出临床可执行的结论。",
      "请结合登记患者信息生成病例摘要。",
      "请结合登记患者信息生成交接说明。",
    ]);
  });
});
