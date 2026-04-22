import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makePatientRegistryDetail } from "../../test/test-utils";
import { RegistryBrowserPane } from "./registry-browser-pane";

function renderPane(overrides: Partial<Parameters<typeof RegistryBrowserPane>[0]> = {}) {
  const props: Parameters<typeof RegistryBrowserPane>[0] = {
    searchState: {
      patientId: "",
      tumorLocation: "",
      mmrStatus: "",
      clinicalStage: "",
      limit: 20,
    },
    searchResults: [
      {
        patient_id: 33,
        status: "draft",
        created_by_session_id: "sess-patient",
        updated_at: "2026-04-16T00:00:00Z",
        tumor_location: "rectum",
        mmr_status: "dMMR",
        clinical_stage: "cT3N1M0",
      },
    ],
    previewPatientId: 33,
    previewDetail: makePatientRegistryDetail({ patient_id: 33 }),
    previewRecords: [],
    previewAlerts: [],
    currentPatientId: null,
    isSearching: false,
    isLoadingPreview: true,
    isBindingCurrentPatient: false,
    isDeletingPatient: false,
    isClearingRegistry: false,
    error: null,
    onSearchFieldChange: vi.fn(),
    onSearchSubmit: vi.fn(),
    onPreviewPatient: vi.fn(),
    onSetCurrentPatient: vi.fn(),
    onDeletePatient: vi.fn(),
    onClearRegistry: vi.fn(),
    ...overrides,
  };

  return render(<RegistryBrowserPane {...props} />);
}

describe("RegistryBrowserPane", () => {
  it("disables preview, bind, and delete actions while preview loading is active", () => {
    renderPane();

    expect(screen.getByRole("button", { name: "previewing 33" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "set current patient 33" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "delete patient 33" })).toBeDisabled();
  });
});
