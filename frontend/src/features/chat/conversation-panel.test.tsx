import { render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import { ConversationPanel } from "./conversation-panel";

function renderConversationPanel(overrides: Partial<ComponentProps<typeof ConversationPanel>> = {}) {
  return render(
    <ConversationPanel
      messages={[]}
      draft=""
      statusNode="planner"
      isStreaming={false}
      isLoadingHistory={false}
      canLoadHistory={false}
      disabled={false}
      errorMessage={null}
      onLoadHistory={vi.fn()}
      onDraftChange={vi.fn()}
      onSubmit={vi.fn()}
      {...overrides}
    />,
  );
}

describe("ConversationPanel latency status", () => {
  it("renders no latency UI when the prop is absent", () => {
    renderConversationPanel();

    expect(screen.getByTestId("status-node")).toHaveTextContent("planner");
    expect(screen.queryByTestId("latency-status")).not.toBeInTheDocument();
    expect(screen.queryByText("本轮正在生成...")).not.toBeInTheDocument();
    expect(screen.queryByText(/界面完成/)).not.toBeInTheDocument();
  });

  it("renders the streaming latency label", () => {
    renderConversationPanel({
      latencyStatus: {
        kind: "streaming",
      },
    });

    expect(screen.getByTestId("latency-status")).toHaveTextContent("本轮正在生成...");
  });

  it("renders the completed latency label with seconds", () => {
    renderConversationPanel({
      latencyStatus: {
        kind: "completed",
        uiCompleteMs: 1234,
      },
    });

    expect(screen.getByTestId("latency-status")).toHaveTextContent("界面完成 1.23 秒");
  });
});
