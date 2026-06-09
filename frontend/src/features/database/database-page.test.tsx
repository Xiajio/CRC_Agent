import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { AppProviders } from "../../app/providers";
import { DatabasePage } from "../../pages/database-page";
import { buildApiClientStub } from "../../test/test-utils";

function renderDatabasePage() {
  return render(
    <AppProviders apiClient={buildApiClientStub()}>
      <DatabasePage />
    </AppProviders>,
  );
}

describe("DatabasePage", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders the database workspace in a full three-panel layout", async () => {
    renderDatabasePage();

    expect(await screen.findByText("亿铸科技 -- 虚拟数据库控制台")).toBeVisible();
    await waitFor(() => {
      expect(screen.queryByText("正在同步数据库工作台...")).not.toBeInTheDocument();
    });

    expect(screen.getByTestId("panel-grid")).toHaveAttribute("data-layout-mode", "full");
    expect(screen.getByTestId("left-rail")).toBeVisible();
    expect(screen.getByTestId("center-workspace")).toBeVisible();
    expect(screen.getByTestId("right-inspector")).toBeVisible();
  });
});
