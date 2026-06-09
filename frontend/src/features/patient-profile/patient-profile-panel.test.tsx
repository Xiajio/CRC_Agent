import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { PatientProfilePanel } from "./patient-profile-panel";

describe("PatientProfilePanel", () => {
  it("shows split patient identity fields ahead of legacy current_patient_id", () => {
    render(
      <PatientProfilePanel
        patientProfile={{
          current_patient_id: "legacy-current",
          case_database_patient_id: "093",
          registry_patient_id: 7,
          age: 52,
        }}
      />,
    );

    const labels = screen.getAllByRole("term").map((term) => term.textContent);

    expect(labels.slice(0, 2)).toEqual(["病例库样本ID", "登记患者ID"]);
    expect(screen.getByText("病例库样本ID")).toBeInTheDocument();
    expect(screen.getByText("登记患者ID")).toBeInTheDocument();
    expect(screen.queryByText("当前患者ID")).not.toBeInTheDocument();
    expect(screen.queryByText("legacy-current")).not.toBeInTheDocument();
  });

  it("keeps current_patient_id visible only as a compatibility identity", () => {
    render(<PatientProfilePanel patientProfile={{ current_patient_id: "legacy-current" }} />);

    const row = screen.getByText("兼容患者ID").closest(".clinical-detail-row");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByText("legacy-current")).toBeInTheDocument();
  });
});
