import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PatientBackgroundPanel } from "./patient-background-panel";

describe("PatientBackgroundPanel", () => {
  it("renders a compact empty state for an empty patient card scaffold", () => {
    render(
      <PatientBackgroundPanel
        cards={{
          patient_card: {
            type: "patient_card",
            patient_id: "current",
            card_meta: { source_mode: "patient_self_report" },
            data: {
              patient_info: { gender: null, age: null, ecog: null, cea: null },
              history_block: { chief_complaint: null, symptom_duration: null, risk_factors: null },
              diagnosis_block: { confirmed: null, primary_site: null, mmr_status: null },
              staging_block: { clinical_stage: null, ct_stage: null, cn_stage: null, cm_stage: null },
            },
          },
        }}
      />,
    );

    expect(screen.getByText("当前患者")).toBeInTheDocument();
    expect(screen.getByText("暂未采集到可用背景信息。")).toBeInTheDocument();
    expect(screen.queryByText("查看原始数据")).not.toBeInTheDocument();
  });

  it("renders only confirmed background fields and uses field_meta display text", () => {
    render(
      <PatientBackgroundPanel
        cards={{
          patient_card: {
            type: "patient_card",
            patient_id: "P-1024",
            card_meta: { source_mode: "patient_self_report" },
            field_meta: {
              patient_info: {
                age: { display: "58岁" },
              },
            },
            data: {
              patient_info: { gender: "女性", age: null, ecog: 1, cea: null },
              history_block: { chief_complaint: "腹痛", symptom_duration: "4小时", risk_factors: ["吸烟史"] },
              diagnosis_block: { confirmed: null, primary_site: null, mmr_status: null },
              staging_block: { clinical_stage: null, ct_stage: null, cn_stage: null, cm_stage: null },
            },
          },
        }}
      />,
    );

    expect(screen.getByText("患者 #P-1024")).toBeInTheDocument();
    expect(screen.getByText("女性")).toBeInTheDocument();
    expect(screen.getByText("58岁")).toBeInTheDocument();
    expect(screen.getByText("腹痛")).toBeInTheDocument();
    expect(screen.getByText("吸烟史")).toBeInTheDocument();
    expect(screen.getByText("待补充")).toBeInTheDocument();
  });
});
