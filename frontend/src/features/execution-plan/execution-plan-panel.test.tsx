import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ExecutionPlanPanel } from "./execution-plan-panel";

describe("ExecutionPlanPanel", () => {
  it("renders empty plan and reference states", () => {
    render(<ExecutionPlanPanel plan={[]} references={[]} />);

    expect(screen.getByText("暂无执行计划。")).toBeInTheDocument();
    expect(screen.getByText("暂无参考资料。")).toBeInTheDocument();
    expect(screen.queryByText("NCCN 指南片段")).not.toBeInTheDocument();
  });

  it("renders provided plan steps and references", () => {
    render(
      <ExecutionPlanPanel
        plan={[{ title: "retrieve guidelines", status: "completed" }]}
        references={[{ title: "NCCN 指南片段", subtitle: "证据摘要", tag: "NCCN" }]}
      />,
    );

    expect(screen.getByText("检索指南")).toBeInTheDocument();
    expect(screen.getByText("已完成")).toBeInTheDocument();
    expect(screen.getByText("NCCN 指南片段")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查看全部参考资料（1）" })).toBeInTheDocument();
  });
});
