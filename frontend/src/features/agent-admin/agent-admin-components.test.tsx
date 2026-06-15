import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  AgentAdminDisabledAction,
  AgentAdminMetricStrip,
  AgentAdminStatusChip,
} from "./agent-admin-components";

describe("AgentAdmin shared components", () => {
  it("renders metric labels, values, and aligned tone classes", () => {
    render(
      <AgentAdminMetricStrip
        metrics={[
          { label: "活跃会话", value: "1/1", tone: "red" },
          { label: "运行状态", value: "ready", tone: "success" },
        ]}
      />,
    );

    expect(screen.getByText("活跃会话")).toBeInTheDocument();
    expect(screen.getByText("1/1")).toBeInTheDocument();
    expect(screen.getByText("活跃会话").closest("article")).toHaveClass("agent-admin-metric-red");
    expect(screen.getByText("运行状态").closest("article")).toHaveClass("agent-admin-metric-success");
  });

  it("renders status chip text", () => {
    render(<AgentAdminStatusChip tone="warning">待配置</AgentAdminStatusChip>);

    expect(screen.getByText("待配置")).toBeInTheDocument();
  });

  it("keeps disabled actions focusable while exposing the reason", () => {
    const onClick = vi.fn();

    render(<AgentAdminDisabledAction label="运行学习任务" reason="一期只读" onClick={onClick} />);

    const action = screen.getByRole("button", { name: /运行学习任务/ });
    expect(action).toHaveAttribute("aria-disabled", "true");
    expect(action).not.toBeDisabled();
    expect(screen.getByText("一期只读")).toBeInTheDocument();

    fireEvent.click(action);
    expect(onClick).not.toHaveBeenCalled();
  });
});
