import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RoadmapPanel } from "./roadmap-panel";

describe("RoadmapPanel", () => {
  it("renders an empty roadmap state when no steps exist", () => {
    render(<RoadmapPanel roadmap={[]} stage={null} />);

    expect(screen.getByText("暂无工作流路线。")).toBeInTheDocument();
    expect(screen.queryByText("意图识别")).not.toBeInTheDocument();
  });

  it("renders provided roadmap steps", () => {
    render(
      <RoadmapPanel
        roadmap={[{ id: "intent", title: "intent", status: "completed" }]}
        stage={null}
      />,
    );

    expect(screen.getByText("意图识别")).toBeInTheDocument();
    expect(screen.getByText("已完成")).toBeInTheDocument();
  });
});
