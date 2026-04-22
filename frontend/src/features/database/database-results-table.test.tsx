import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DatabasePagination, DatabaseSort } from "../../app/api/types";
import { DatabaseResultsTable } from "./database-results-table";

function renderTable(overrides: Partial<Parameters<typeof DatabaseResultsTable>[0]> = {}) {
  const props: Parameters<typeof DatabaseResultsTable>[0] = {
    items: [
      {
        patient_id: 33,
        age: 52,
        gender: "female",
        ecog_score: 1,
        tumor_location: "rectum",
        clinical_stage: "cT3N1M0",
        mmr_status: "dMMR",
      },
    ],
    total: 40,
    pagination: { page: 1, page_size: 20 } satisfies DatabasePagination,
    sort: { field: "patient_id", direction: "asc" } satisfies DatabaseSort,
    selectedPatientId: null,
    isSearching: false,
    isLoadingDetail: false,
    onSelectPatient: vi.fn(),
    onSortChange: vi.fn(),
    onPageChange: vi.fn(),
    ...overrides,
  };

  return render(<DatabaseResultsTable {...props} />);
}

describe("DatabaseResultsTable", () => {
  afterEach(() => {
    cleanup();
  });

  it("disables row selection while detail loading is active", () => {
    renderTable({ isLoadingDetail: true });

    expect(screen.getByRole("button", { name: "查看 33" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "上一页" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "下一页" })).toBeEnabled();
  });

  it("keeps row selection enabled when detail is idle", () => {
    const onSelectPatient = vi.fn();
    renderTable({ onSelectPatient });

    fireEvent.click(screen.getByRole("button", { name: "查看 33" }));

    expect(onSelectPatient).toHaveBeenCalledWith(33);
  });
});
