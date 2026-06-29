import { render, screen } from "@testing-library/react";

import { PatientCareCards } from "./patient-care-cards";

test("renders three patient care card groups", () => {
  render(
    <PatientCareCards
      cards={{
        focusMetrics: ["留意便血或黑便是否加重"],
        periodicChecks: ["尽快预约消化专科门诊"],
        dailyActions: ["记录便血颜色、次数和伴随症状"],
      }}
      isLoading={false}
    />,
  );

  expect(screen.getByText("最近需要留意的信号")).toBeInTheDocument();
  expect(screen.getByText("可安排的检查事项")).toBeInTheDocument();
  expect(screen.getByText("居家记录与行动")).toBeInTheDocument();
  expect(screen.getByText("尽快预约消化专科门诊")).toBeInTheDocument();
});

test("renders default empty text for missing groups", () => {
  render(
    <PatientCareCards
      cards={{ focusMetrics: [], periodicChecks: [], dailyActions: [] }}
      isLoading={false}
    />,
  );

  expect(screen.getAllByText("暂无可展示内容")).toHaveLength(3);
});
