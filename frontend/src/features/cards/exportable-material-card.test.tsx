import { render, screen } from "@testing-library/react";

import { renderCardContent } from "./card-renderers";

test("renders exportable material markdown download card", () => {
  render(
    <>
      {renderCardContent({
        cardType: "exportable_material_card",
        payload: {
          title: "门诊分诊报告",
          markdown: "# 门诊分诊报告\n\n请携带检查结果复诊。",
          suggested_filename: "triage-report",
          material_type: "triage_report",
        },
      })}
    </>,
  );

  expect(screen.getByText("门诊分诊报告")).toBeInTheDocument();
  expect(screen.getByText("下载 Markdown")).toBeInTheDocument();
  expect(screen.getAllByText(/请携带检查结果复诊/).length).toBeGreaterThan(0);
});
