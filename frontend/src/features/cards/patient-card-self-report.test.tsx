import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { JsonObject } from "../../app/api/types";
import { renderCardContent } from "./card-renderers";

function renderPatientCard(payload: Record<string, unknown>, onPromptRequest = vi.fn()) {
  render(
    <div>
      {renderCardContent({
        cardType: "patient_card",
        payload: payload as JsonObject,
        onPromptRequest,
      })}
    </div>,
  );
}

describe("patient card self-report rendering", () => {
  it("renders the full skeleton for patient self-report cards and prefers field_meta display text", () => {
    renderPatientCard({
      type: "patient_card",
      card_meta: {
        source_mode: "patient_self_report",
      },
      patient_id: "SR-1001",
      field_meta: {
        patient_info: {
          gender: {
            display: "待确认",
          },
          age: {
            display: "待确认（来源不一致）",
          },
          ecog: {
            display: "ECOG待确认",
          },
          cea: {
            display: "待确认",
          },
        },
        diagnosis_block: {
          confirmed: {
            display: "待确认",
          },
          primary_site: {
            display: "待确认（来源不一致）",
          },
          mmr_status: {
            display: "待确认",
          },
        },
        staging_block: {
          clinical_stage: {
            display: "待确认",
          },
          ct_stage: {
            display: "待确认",
          },
          cn_stage: {
            display: "待确认",
          },
          cm_stage: {
            display: "待确认",
          },
        },
        history_block: {
          chief_complaint: {
            display: "待确认",
          },
          symptom_duration: {
            display: "待确认（来源不一致）",
          },
          family_history: {
            display: "待确认",
          },
          family_history_details: {
            display: "待确认",
          },
          biopsy_confirmed: {
            display: "待确认",
          },
          biopsy_details: {
            display: "待确认（来源不一致）",
          },
          risk_factors: {
            display: "待确认（风险因素）",
          },
        },
      },
      data: {
        patient_info: {
          gender: null,
          age: null,
          ecog: null,
          cea: null,
        },
        diagnosis_block: {
          confirmed: null,
          primary_site: null,
          mmr_status: null,
        },
        staging_block: {
          clinical_stage: null,
          ct_stage: null,
          cn_stage: null,
          cm_stage: null,
        },
        history_block: {
          chief_complaint: null,
          symptom_duration: null,
          family_history: null,
          family_history_details: null,
          biopsy_confirmed: null,
          biopsy_details: null,
          risk_factors: null,
        },
      },
    });

    expect(screen.getByText("患者 #SR-1001")).toBeInTheDocument();
    expect(screen.getAllByText("待确认").length).toBeGreaterThan(0);
    expect(screen.getAllByText("待确认（来源不一致）").length).toBeGreaterThan(0);
    expect(screen.getByText("ECOG待确认")).toBeInTheDocument();
    expect(screen.getByText("待确认（风险因素）")).toBeInTheDocument();
    expect(screen.getByText("诊断信息")).toBeInTheDocument();
    expect(screen.getByText("基础病史")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "生成治疗方案" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "查询影像资料" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "撰写病程记录" })).not.toBeInTheDocument();
  });

  it("falls back to dotted field_meta keys for compatibility", () => {
    renderPatientCard({
      type: "patient_card",
      card_meta: {
        source_mode: "patient_self_report",
      },
      patient_id: "SR-1002",
      field_meta: {
        "patient_info.gender": {
          display: "待确认（点号兼容）",
        },
        "history_block.risk_factors": {
          display: "待确认（点号兼容）",
        },
      },
      data: {
        patient_info: {
          gender: null,
        },
        history_block: {
          risk_factors: null,
        },
      },
    });

    expect(screen.getByText("患者 #SR-1002")).toBeInTheDocument();
    expect(screen.getAllByText("待确认（点号兼容）").length).toBeGreaterThan(0);
    expect(screen.getByText("危险因素")).toBeInTheDocument();
  });

  it("keeps legacy patient cards sparse and preserves doctor quick actions", () => {
    renderPatientCard({
      type: "patient_card",
      patient_id: "DB-2002",
      data: {
        patient_info: {
          gender: "female",
          age: 56,
          ecog: 1,
          cea: 12.3,
        },
        diagnosis_block: {
          confirmed: "结直肠癌",
          primary_site: "直肠",
          mmr_status: "pMMR",
        },
      },
    });

    expect(screen.getByText("患者 #DB-2002")).toBeInTheDocument();
    expect(screen.getByText("female")).toBeInTheDocument();
    expect(screen.getByText("56岁")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("12.3")).toBeInTheDocument();
    expect(screen.queryByText("基础病史")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "生成治疗方案" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查询影像资料" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "撰写病程记录" })).toBeInTheDocument();
  });
});
