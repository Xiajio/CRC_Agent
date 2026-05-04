import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DoctorDatabaseView } from "./doctor-database-view";

function renderView(overrides: Partial<Parameters<typeof DoctorDatabaseView>[0]> = {}) {
  const props: Parameters<typeof DoctorDatabaseView>[0] = {
    activeSource: "patient_registry",
    onSourceChange: vi.fn(),
    currentPatientId: null,
    databaseWorkbench: {} as never,
    registryBrowser: {
      recentPatients: [],
      searchState: {
        patientId: "",
        tumorLocation: "",
        mmrStatus: "",
        clinicalStage: "",
        limit: 20,
      },
      searchResults: [],
      previewPatientId: null,
      previewDetail: null,
      previewRecords: [],
      previewAlerts: [],
      isLoadingRecent: false,
      isSearching: false,
      isLoadingPreview: false,
      isDeletingPatient: false,
      isClearingRegistry: false,
      error: null,
      setSearchField: vi.fn(),
      runSearch: vi.fn(),
      previewPatient: vi.fn(),
      deletePatient: vi.fn(),
      clearRegistry: vi.fn(),
    } as never,
    isBindingCurrentPatient: false,
    onSetCurrentPatient: vi.fn(),
    ...overrides,
  };

  return render(<DoctorDatabaseView {...props} />);
}

describe("DoctorDatabaseView", () => {
  it("renders database source switch buttons with Chinese visible labels", () => {
    const onSourceChange = vi.fn();

    renderView({ onSourceChange });

    const historicalCaseBase = screen.getByRole("button", { name: "historical case base" });
    const patientRegistry = screen.getByRole("button", { name: "patient registry" });
    expect(historicalCaseBase).toHaveTextContent("历史病例");
    expect(patientRegistry).toHaveTextContent("患者库");

    fireEvent.click(historicalCaseBase);
    fireEvent.click(patientRegistry);

    expect(onSourceChange).toHaveBeenNthCalledWith(1, "historical_case_base");
    expect(onSourceChange).toHaveBeenNthCalledWith(2, "patient_registry");
  });
});
