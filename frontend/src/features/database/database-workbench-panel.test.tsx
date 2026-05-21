import { render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import { DatabaseWorkbenchPanel } from "./database-workbench-panel";
import { createDefaultFilters } from "./use-database-workbench";

function renderDatabaseWorkbenchPanel(
  overrides: Partial<ComponentProps<typeof DatabaseWorkbenchPanel>> = {},
) {
  return render(
    <DatabaseWorkbenchPanel
      mode="detail"
      naturalQuery=""
      stats={null}
      searchRequest={{
        filters: createDefaultFilters(),
        pagination: { page: 1, page_size: 10 },
        sort: { field: "patient_id", direction: "asc" },
      }}
      searchResponse={null}
      selectedPatientId={93}
      isParsing={false}
      isSearching={false}
      isLoadingDetail={false}
      isBootstrapping={false}
      warnings={[]}
      unsupportedTerms={[]}
      error={null}
      onNaturalQueryChange={vi.fn()}
      onNaturalQuerySubmit={vi.fn()}
      onSelectPatient={vi.fn()}
      onSetCurrentCaseDatabasePatient={vi.fn()}
      onSortChange={vi.fn()}
      onPageChange={vi.fn()}
      {...overrides}
    />,
  );
}

describe("DatabaseWorkbenchPanel", () => {
  it("exposes stable selectors for demo automation", () => {
    renderDatabaseWorkbenchPanel();

    expect(screen.getByTestId("database-workbench")).toBeInTheDocument();
    expect(screen.getByTestId("database-query-input")).toBe(screen.getByRole("textbox"));
    expect(screen.getByTestId("database-case-093-bring-in")).toBeInTheDocument();
  });
});
