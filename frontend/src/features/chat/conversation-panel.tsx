import type { FrontendMessage } from "../../app/api/types";
import { cardTitle, renderCardContent, type CardPromptHandler } from "../cards/card-renderers-extended";

type ConversationPanelProps = {
  messages: FrontendMessage[];
  draft: string;
  statusNode: string | null;
  isStreaming: boolean;
  isLoadingHistory: boolean;
  canLoadHistory: boolean;
  disabled: boolean;
  draftDisabled?: boolean;
  errorMessage: string | null;
  onLoadHistory: () => void;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onCardPromptRequest?: CardPromptHandler;
  activeTriageQuestionId?: string | null;
};

const INTERNAL_LINE_PATTERNS = [
  /^\s*\[Router\].*$/gm,
  /^\s*\[Intent\].*$/gm,
  /^\s*\[Planner\].*$/gm,
  /^\s*审核[:：].*$/gm,
  /^\s*\*\*知识检索完成\*\*.*$/gm,
];

function executionStatusLabel(statusNode: string | null, isStreaming: boolean): string {
  if (statusNode === "memory_manager") {
    return "整理上下文";
  }
  if (statusNode) {
    return statusNode;
  }
  return isStreaming ? "生成中..." : "就绪";
}

function messageLabel(message: FrontendMessage): string {
  return message.type === "ai" ? "助手" : "用户";
}

function normalizeMessageText(content: unknown): { text: string } {
  let text = "";
  if (typeof content === "string") {
    text = content;
  } else if (content === null || content === undefined) {
    text = "";
  } else {
    text = JSON.stringify(content);
  }

  text = text.replace(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>\s*/gi, "");
  for (const pattern of INTERNAL_LINE_PATTERNS) {
    text = text.replace(pattern, "");
  }

  const trimmed = text.trim();
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
    try {
      JSON.parse(trimmed);
      return { text: "" };
    } catch {
      // Keep non-JSON content as-is.
    }
  }

  return { text: trimmed };
}

function stripGenericInlineCardHints(text: string): string {
  return text
    .replace(/点击卡片查看详情。?/g, "")
    .replace(/已为您生成(?:以下)?卡片。?/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function shouldHideInlineMessageText(text: string, message: FrontendMessage): boolean {
  if (!message.inlineCards || message.inlineCards.length === 0) {
    return false;
  }
  return !text;
}

function renderMessageContent(text: string, thinkText?: string) {
  return (
    <>
      {thinkText ? (
        <details className="workspace-card-disclosure" style={{ marginBottom: "12px" }}>
          <summary style={{ fontSize: "0.85rem", color: "#8e4a55" }}>思考过程</summary>
          <div
            style={{
              padding: "10px 14px",
              fontSize: "0.8rem",
              whiteSpace: "pre-wrap",
              color: "var(--text-secondary)",
            }}
          >
            {thinkText}
          </div>
        </details>
      ) : null}
      {text ? (
        <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.65 }}>
          {text}
        </div>
      ) : null}
    </>
  );
}

export function ConversationPanel({
  messages,
  draft,
  statusNode,
  isStreaming,
  isLoadingHistory,
  canLoadHistory,
  disabled,
  draftDisabled,
  errorMessage,
  onLoadHistory,
  onDraftChange,
  onSubmit,
  onCardPromptRequest,
  activeTriageQuestionId,
}: ConversationPanelProps) {
  const executionLabel = executionStatusLabel(statusNode, isStreaming);
  const textareaDisabled = draftDisabled ?? disabled;

  return (
    <div
      className="workspace-card"
      data-testid="conversation-panel"
      style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}
    >
      <h2 style={{ display: "flex", alignItems: "center", gap: "8px", margin: 0, paddingBottom: "12px", borderBottom: "1px solid rgba(165, 73, 83, 0.1)" }}>
        <span style={{ fontSize: "1.2rem" }}>💬</span> 智能对话
      </h2>

      {errorMessage ? <p className="workspace-copy workspace-copy-alert" style={{ marginTop: "12px" }}>{errorMessage}</p> : null}

      <div style={{ flex: 1, overflowY: "auto", paddingRight: "4px", marginTop: "16px" }}>
        {canLoadHistory ? (
          <div style={{ textAlign: "center", marginBottom: "16px" }}>
            <button
              type="button"
              className="workspace-secondary-button"
              style={{ marginTop: 0 }}
              disabled={isLoadingHistory}
              onClick={onLoadHistory}
            >
              {isLoadingHistory ? "加载历史中..." : "加载更早消息"}
            </button>
          </div>
        ) : null}

        {messages.length > 0 ? (
          <ol className="workspace-message-list" style={{ paddingTop: 0 }}>
            {messages.map((message) => {
              const isUser = message.type !== "ai";
              const { text: normalizedText } = normalizeMessageText(message.content);
              const thinkText = (message.thinking ?? "").trim();
              const displayText = message.inlineCards?.length
                ? stripGenericInlineCardHints(normalizedText)
                : normalizedText;
              const hideText = shouldHideInlineMessageText(displayText, message) || (!displayText && !thinkText);

              if (hideText && (!message.inlineCards || message.inlineCards.length === 0)) {
                return null;
              }

              return (
                <li
                  key={message.cursor}
                  className={`workspace-message-bubble ${isUser ? "bubble-user" : "bubble-ai"}`}
                >
                  <div className="bubble-header">
                    <strong>{messageLabel(message)}</strong>
                  </div>
                  {!hideText || thinkText ? (
                    <div className="bubble-content">
                      {renderMessageContent(hideText ? "" : displayText, thinkText || undefined)}
                    </div>
                  ) : null}
                  {message.inlineCards?.length ? (
                    <div className="workspace-inline-card-stack">
                      {message.inlineCards.map((card, index) => (
                        <div
                          key={`${message.cursor}-${card.cardType}-${index}`}
                          className="workspace-inline-card-wrapper"
                        >
                          <div className="workspace-inline-card">
                            <strong className="workspace-inline-card-title">
                              {cardTitle(card.cardType, card.payload)}
                            </strong>
                            {renderCardContent({
                              cardType: card.cardType,
                              payload: card.payload,
                              onPromptRequest: onCardPromptRequest,
                              isInteractive:
                                card.cardType !== "triage_question_card"
                                || (typeof card.payload.question_id === "string"
                                  && card.payload.question_id === activeTriageQuestionId),
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </li>
              );
            })}
          </ol>
        ) : (
          <p className="workspace-copy" style={{ marginTop: 0 }}>
            还没有对话内容。
          </p>
        )}
      </div>

      <div
        style={{
          marginTop: "16px",
          paddingTop: "12px",
          borderTop: "1px solid rgba(165, 73, 83, 0.1)",
        }}
      >
        <div className="workspace-status-row" style={{ marginTop: 0, marginBottom: "12px" }}>
          <span className="workspace-meta" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "1.1rem" }}>⚙️</span> 执行节点
          </span>
          <strong className="workspace-status-node" data-testid="status-node" style={{ color: "#8e4a55", background: "rgba(165, 73, 83, 0.08)", padding: "4px 10px", borderRadius: "12px", fontSize: "0.85rem" }}>
            {executionLabel}
          </strong>
        </div>

        <div className="workspace-composer" style={{ marginTop: 0 }}>
          <div style={{ position: "relative" }}>
            <textarea
              className="workspace-composer-input"
              placeholder="输入你的问题，例如分诊症状、治疗方案或数据库查询"
              value={draft}
              disabled={textareaDisabled}
              onChange={(event) => onDraftChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  if (!textareaDisabled && draft.trim()) {
                    onSubmit();
                  }
                }
              }}
              style={{
                boxShadow: "inset 0 2px 6px rgba(133, 70, 78, 0.04)",
                border: "1px solid rgba(165, 73, 83, 0.2)",
                paddingRight: "50px"
              }}
            />
            <button
              type="button"
              className="workspace-composer-send"
              disabled={textareaDisabled || !draft.trim()}
              onClick={onSubmit}
              aria-label="发送消息"
              style={{
                position: "absolute",
                right: "12px",
                bottom: "16px",
                width: "36px",
                height: "36px",
                padding: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                borderRadius: "10px",
                background: (textareaDisabled || !draft.trim()) ? "rgba(165, 73, 83, 0.15)" : "linear-gradient(135deg, #8e4a55 0%, #a35d68 100%)",
                color: (textareaDisabled || !draft.trim()) ? "rgba(142, 74, 85, 0.5)" : "#ffffff",
                border: "none",
                cursor: (textareaDisabled || !draft.trim()) ? "not-allowed" : "pointer",
                transition: "all 0.2s ease",
                boxShadow: (textareaDisabled || !draft.trim()) ? "none" : "0 2px 8px rgba(142, 74, 85, 0.25)",
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
