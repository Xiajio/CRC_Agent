import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { FrontendMessage } from "../../frontend/src/app/api/types";
import { ConversationPanel } from "../../frontend/src/features/chat/conversation-panel";

function renderConversation(messages: FrontendMessage[]) {
  return render(
    <ConversationPanel
      messages={messages}
      draft=""
      statusNode={null}
      isStreaming={false}
      isLoadingHistory={false}
      canLoadHistory={false}
      disabled={false}
      errorMessage={null}
      onLoadHistory={vi.fn()}
      onDraftChange={vi.fn()}
      onSubmit={vi.fn()}
    />,
  );
}

describe("conversation panel thinking disclosure", () => {
  it("does not infer thinking blocks from message content alone", () => {
    renderConversation([
      {
        cursor: "m1",
        type: "ai",
        content:
          "\u6839\u636e\u7cfb\u7edf\u63d0\u793a\uff0c\u6211\u5e94\u8be5\u5148\u5206\u6790\u3002\n\n\u6700\u7ec8\u7b54\u590d\uff1a\u60a8\u597d",
        assetRefs: [],
      },
    ]);

    expect(screen.queryByText("\u601d\u8003\u8fc7\u7a0b")).not.toBeInTheDocument();
  });

  it("renders the thinking disclosure only when explicit thinking is provided", () => {
    renderConversation([
      {
        cursor: "m1",
        type: "ai",
        content: "\u60a8\u597d",
        thinking: "\u6839\u636e\u7cfb\u7edf\u63d0\u793a\uff0c\u6211\u5e94\u8be5\u5148\u5206\u6790\u3002",
        assetRefs: [],
      },
    ]);

    expect(screen.getByText("\u601d\u8003\u8fc7\u7a0b")).toBeInTheDocument();
    expect(screen.getByText("\u6839\u636e\u7cfb\u7edf\u63d0\u793a\uff0c\u6211\u5e94\u8be5\u5148\u5206\u6790\u3002")).toBeInTheDocument();
  });
});