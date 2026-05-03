import { fireEvent, render, screen } from "@testing-library/react";
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
  it("renders messages and composer with shared UI classes", () => {
    renderConversationPanel({
      messages: [
        { cursor: "1", type: "user", content: "hello", assetRefs: [] },
        { cursor: "2", type: "ai", content: "hi", assetRefs: [] },
      ],
    });

    expect(screen.getByText("hello").closest("li")).toHaveClass("ui-message-bubble-user");
    expect(screen.getByText("hi").closest("li")).toHaveClass("ui-message-bubble-assistant");
    expect(screen.getByRole("textbox")).toHaveClass("ui-textarea");
  });

  it("submits trimmed drafts on Enter and keeps Shift+Enter inside the textarea", () => {
    const onSubmit = vi.fn();

    renderConversationPanel({
      draft: "  treatment plan  ",
      onSubmit,
    });

    const textbox = screen.getByRole("textbox");
    fireEvent.keyDown(textbox, { key: "Enter", shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();

    fireEvent.keyDown(textbox, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("does not submit from Enter when the draft is blank or the composer is disabled", () => {
    const onSubmit = vi.fn();

    const { rerender } = renderConversationPanel({
      draft: "   ",
      onSubmit,
    });

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    expect(onSubmit).not.toHaveBeenCalled();

    rerender(
      <ConversationPanel
        messages={[]}
        draft="ready"
        statusNode="planner"
        isStreaming={false}
        isLoadingHistory={false}
        canLoadHistory={false}
        disabled={false}
        draftDisabled
        errorMessage={null}
        onLoadHistory={vi.fn()}
        onDraftChange={vi.fn()}
        onSubmit={onSubmit}
      />,
    );

    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" });
    expect(onSubmit).not.toHaveBeenCalled();
  });

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
