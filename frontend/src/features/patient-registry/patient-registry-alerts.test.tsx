import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PatientRegistryAlertsPanel } from "./patient-registry-alerts";

describe("PatientRegistryAlertsPanel", () => {
  it("renders loading, empty, and record alert labels with Chinese copy", () => {
    const { rerender } = render(<PatientRegistryAlertsPanel alerts={[]} isLoading />);

    expect(screen.getByRole("heading", { name: /患者库预警/ })).toBeInTheDocument();
    expect(screen.getByText("正在加载预警信息...")).toBeInTheDocument();

    rerender(<PatientRegistryAlertsPanel alerts={[]} isLoading={false} />);

    expect(screen.getByText("暂无预警信息。")).toBeInTheDocument();

    rerender(
      <PatientRegistryAlertsPanel
        alerts={[
          {
            kind: "conflict",
            message: "字段冲突",
            patient_id: 33,
            record_id: 7,
          },
        ]}
        isLoading={false}
      />,
    );

    expect(screen.getByText("conflict / 记录 #7")).toBeInTheDocument();
    expect(screen.getByText("字段冲突")).toBeInTheDocument();
  });
});
