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

  it("filters diagnosis status chatter from persisted assistant messages", () => {
    renderConversationPanel({
      messages: [
        {
          cursor: "2",
          type: "ai",
          content: [
            "[Decision] template-fast patient summary",
            "\u274c **\u8bca\u65ad\u6d41\u7a0b\u5ba1\u6838\u672a\u901a\u8fc7** (Critic: REJECTED)",
            "\ud83d\udccb \u6cbb\u7597\u65b9\u6848\u5df2\u751f\u6210: \u4e34\u5e8a\u51b3\u7b56\u6458\u8981",
            "Visible structured diagnosis",
          ].join("\n"),
          assetRefs: [],
        },
      ],
    });

    expect(screen.getByText("Visible structured diagnosis")).toBeInTheDocument();
    expect(screen.queryByText(/\[Decision\]/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Critic: REJECTED/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\u6cbb\u7597\u65b9\u6848/u)).not.toBeInTheDocument();
  });

  it("renders clinical recommendation markdown as structured report content", () => {
    renderConversationPanel({
      messages: [
        {
          cursor: "2",
          type: "ai",
          content: [
            "# 🏥 临床治疗建议",
            "",
            "**摘要**: 患者为结肠癌，当前临床分期支持 cT4bN1cM0。",
            "",
            "### 手术方案",
            "推荐结肠癌根治术。",
            "",
            "### 质控提示",
            "- 引用依据不足：覆盖评分 65。",
          ].join("\n"),
          assetRefs: [],
        },
      ],
    });

    expect(screen.getByRole("heading", { name: /临床治疗建议/ })).toBeInTheDocument();
    expect(screen.getByText("摘要")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "手术方案" })).toBeInTheDocument();
    expect(screen.getByText("引用依据不足：覆盖评分 65。")).toBeInTheDocument();
    expect(screen.queryByText(/### 手术方案/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\*\*摘要\*\*/)).not.toBeInTheDocument();
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
