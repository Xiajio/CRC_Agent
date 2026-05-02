import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { DatabaseFilters, DatabaseStatsResponse } from "../../frontend/src/app/api/types";
import { renderCardContent } from "../../frontend/src/features/cards/card-renderers";
import { DatabaseEditForm } from "../../frontend/src/features/database/database-edit-form";
import { DatabaseFiltersPanel } from "../../frontend/src/features/database/database-filters-panel";
import { DatabaseWorkbenchPanel } from "../../frontend/src/features/database/database-workbench-panel";
import {
  createDefaultFilters,
  createDefaultSearchRequest,
  normalizeIntentFilters,
  normalizeRecordForUpsert,
  topDistributionEntry,
} from "../../frontend/src/features/database/use-database-workbench";

const EMPTY_STATS = {
  total_cases: 0,
} as DatabaseStatsResponse;

describe("database schema expansion", () => {
  it("extends default filters and normalizes new filter and record fields", () => {
    expect(createDefaultFilters()).toMatchObject({
      family_history: null,
      biopsy_confirmed: null,
      ecog_min: null,
      ecog_max: null,
    });

    expect(
      normalizeIntentFilters({
        family_history: true,
        biopsy_confirmed: false,
        ecog_min: 1,
        ecog_max: 3,
      }),
    ).toMatchObject({
      family_history: true,
      biopsy_confirmed: false,
      ecog_min: 1,
      ecog_max: 3,
    });

    expect(
      normalizeRecordForUpsert({
        patient_id: "7",
        ecog_score: "2",
        family_history: "\u662f",
        biopsy_confirmed: "\u5426",
        risk_factors: "\u5438\u70df, \u996e\u9152",
      }),
    ).toMatchObject({
      patient_id: 7,
      ecog_score: 2,
      family_history: true,
      biopsy_confirmed: false,
      risk_factors: ["\u5438\u70df", "\u996e\u9152"],
    });
  });

  it("updates edit form tri-state booleans and risk factor arrays", () => {
    const onFieldChange = vi.fn();

    render(
      <DatabaseEditForm
        record={{
          patient_id: 11,
          family_history: null,
          biopsy_confirmed: null,
          risk_factors: ["\u5438\u70df"],
        }}
        isSaving={false}
        onFieldChange={onFieldChange}
        onSave={() => undefined}
      />,
    );

    fireEvent.change(screen.getByLabelText("\u5bb6\u65cf\u53f2"), { target: { value: "true" } });
    fireEvent.change(screen.getByLabelText("\u75c5\u7406\u6d3b\u68c0\u786e\u8ba4"), { target: { value: "false" } });
    fireEvent.change(screen.getByLabelText("\u5371\u9669\u56e0\u7d20"), { target: { value: "\u5438\u70df, \u996e\u9152" } });

    expect(onFieldChange).toHaveBeenCalledWith("family_history", true);
    expect(onFieldChange).toHaveBeenCalledWith("biopsy_confirmed", false);
    expect(onFieldChange).toHaveBeenCalledWith("risk_factors", ["\u5438\u70df", "\u996e\u9152"]);
  });

  it("emits the new structured filters from the filters panel", () => {
    const onFiltersChange = vi.fn();
    const filters: DatabaseFilters = {
      ...createDefaultFilters(),
      tumor_location: [],
      ct_stage: [],
      cn_stage: [],
      histology_type: [],
      mmr_status: [],
    };

    render(
      <DatabaseFiltersPanel
        filters={filters}
        isSearching={false}
        onFiltersChange={onFiltersChange}
        onApply={() => undefined}
        onReset={() => undefined}
      />,
    );

    fireEvent.change(screen.getByLabelText("\u5bb6\u65cf\u53f2\u7b5b\u9009"), { target: { value: "true" } });
    fireEvent.change(screen.getByLabelText("\u6d3b\u68c0\u786e\u8ba4\u7b5b\u9009"), { target: { value: "false" } });
    fireEvent.change(screen.getByLabelText("ECOG \u4e0b\u9650"), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText("ECOG \u4e0a\u9650"), { target: { value: "3" } });

    expect(onFiltersChange).toHaveBeenCalledWith(expect.objectContaining({ family_history: true }));
    expect(onFiltersChange).toHaveBeenCalledWith(expect.objectContaining({ biopsy_confirmed: false }));
    expect(onFiltersChange).toHaveBeenCalledWith(expect.objectContaining({ ecog_min: 1 }));
    expect(onFiltersChange).toHaveBeenCalledWith(expect.objectContaining({ ecog_max: 3 }));
  });

  it("renders patient history block content on the patient card", () => {
    render(
      <div>
        {renderCardContent({
          cardType: "patient_card",
          payload: {
            type: "patient_card",
            patient_id: 18,
            data: {
              patient_info: {
                gender: "\u5973",
                age: 52,
                ecog: 1,
                cea: 4.3,
              },
              diagnosis_block: {
                confirmed: "\u76f4\u80a0\u817a\u764c",
                primary_site: "\u76f4\u80a0",
                mmr_status: "pMMR (MSS)",
              },
              staging_block: {
                clinical_stage: "III\u671f",
                ct_stage: "3",
                cn_stage: "1",
                cm_stage: "M0",
              },
              history_block: {
                chief_complaint: "\u8179\u75db",
                symptom_duration: "3\u5929",
                family_history: true,
                family_history_details: "\u7236\u4eb2\u7ed3\u76f4\u80a0\u764c",
                biopsy_confirmed: true,
                biopsy_details: "\u80a0\u955c\u6d3b\u68c0\u5df2\u63d0\u793a\u817a\u764c",
                risk_factors: ["\u5438\u70df", "\u80a5\u80d6"],
              },
            },
          },
        })}
      </div>,
    );

    expect(screen.getByText("\u57fa\u7840\u75c5\u53f2")).toBeInTheDocument();
    expect(screen.getByText("\u8179\u75db")).toBeInTheDocument();
    expect(screen.getByText("\u7236\u4eb2\u7ed3\u76f4\u80a0\u764c")).toBeInTheDocument();
    expect(screen.getByText("\u5438\u70df\u3001\u80a5\u80d6")).toBeInTheDocument();
  });

  it("keeps database workbench stable when stats are missing distribution fields", () => {
    expect(topDistributionEntry(null)).toBe("-");

    render(
      <DatabaseWorkbenchPanel
        mode="stats"
        naturalQuery=""
        stats={EMPTY_STATS}
        searchRequest={createDefaultSearchRequest()}
        searchResponse={null}
        selectedPatientId={null}
        isParsing={false}
        isSearching={false}
        isBootstrapping={false}
        warnings={[]}
        unsupportedTerms={[]}
        error={null}
        onNaturalQueryChange={() => undefined}
        onNaturalQuerySubmit={() => undefined}
        onSelectPatient={() => undefined}
        onSortChange={() => undefined}
        onPageChange={() => undefined}
      />,
    );

    expect(screen.getByText("\u603b\u75c5\u4f8b\u6570")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
    expect(screen.getByText("\u6682\u65e0\u53ef\u7528\u90e8\u4f4d\u7edf\u8ba1\u3002")).toBeInTheDocument();
  });
});