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
    expect(screen.queryByText("\u672c\u8f6e\u8017\u65f6\u8ba1\u65f6\u4e2d...")).not.toBeInTheDocument();
    expect(screen.queryByText(/\u672c\u8f6e\u754c\u9762\u5b8c\u6210/)).not.toBeInTheDocument();
  });

  it("renders the streaming latency label", () => {
    renderConversationPanel({
      latencyStatus: {
        kind: "streaming",
      },
    });

    expect(screen.getByTestId("latency-status")).toHaveTextContent("\u672c\u8f6e\u8017\u65f6\u8ba1\u65f6\u4e2d...");
  });

  it("renders the completed latency label with seconds", () => {
    renderConversationPanel({
      latencyStatus: {
        kind: "completed",
        uiCompleteMs: 1234,
      },
    });

    expect(screen.getByTestId("latency-status")).toHaveTextContent("\u672c\u8f6e\u754c\u9762\u5b8c\u6210 1.23s");
  });
});
