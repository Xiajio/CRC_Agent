import type { FrontendMessage } from "../../app/api/types";
import type { ReactNode } from "react";
import { Button, Card, MessageBubble, Textarea } from "../../components/ui";
import { cardTitle, renderCardContent, type CardPromptHandler } from "../cards/card-renderers-extended";

export type ConversationLatencyStatus =
  | {
      kind: "streaming";
    }
  | {
      kind: "completed";
      uiCompleteMs: number;
    };

type ConversationPanelProps = {
  messages: FrontendMessage[];
  draft: string;
  statusNode: string | null;
  isStreaming: boolean;
  latencyStatus?: ConversationLatencyStatus | null;
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
  /^\s*\[Decision\].*$/gm,
  /^\s*(?:\u2705|\u274c)\s*\*\*\u8bca\u65ad\u6d41\u7a0b\u5ba1\u6838.*$/gm,
  /^\s*\ud83d\udccb\s*\u6cbb\u7597\u65b9\u6848\u5df2\u751f\u6210:.*$/gm,
];

function executionStatusLabel(statusNode: string | null, isStreaming: boolean): string {
  if (statusNode === "memory_manager") {
    return "记忆管理";
  }
  if (statusNode) {
    return statusNode;
  }
  return isStreaming ? "生成中..." : "空闲";
}

function latencyStatusLabel(latencyStatus?: ConversationLatencyStatus): string | null {
  if (!latencyStatus) {
    return null;
  }

  if (latencyStatus.kind === "streaming") {
    return "本轮正在生成...";
  }

  return `界面完成 ${(latencyStatus.uiCompleteMs / 1000).toFixed(2)} 秒`;
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
      // Keep non-JSON content as user-facing text.
    }
  }

  return { text: stripLegacyClinicalReportNoise(trimmed) };
}

function shouldHideInlineMessageText(text: string, message: FrontendMessage): boolean {
  if (!message.inlineCards || message.inlineCards.length === 0) {
    return false;
  }
  return !text;
}

function stripLegacyClinicalReportNoise(text: string): string {
  if (!text.includes("临床治疗建议")) {
    return text;
  }

  return text
    .replace(/^>\s*\[!WARNING\][\s\S]*?(?=^#\s*.*临床治疗建议)/m, "")
    .replace(/^\s*\{[\s\S]*?"verdict"[\s\S]*?"feedback"[\s\S]*?\}\s*(?=^#\s*.*临床治疗建议)/m, "")
    .replace(/\n>\s*\*审核意见:[\s\S]*?(?=\n={10,}|\n#|\n###|$)/g, "")
    .replace(/\n={10,}[\s\S]*$/m, "")
    .trim();
}

function plainInlineMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .trim();
}

function isClinicalReportText(text: string): boolean {
  return /(^|\n)\s*#\s*.*临床治疗建议/.test(text);
}

function flushList(nodes: ReactNode[], listItems: string[], key: string) {
  if (listItems.length === 0) {
    return;
  }
  nodes.push(
    <ul key={key} className="clinical-report-list">
      {listItems.map((item, index) => (
        <li key={`${key}-${index}`}>{plainInlineMarkdown(item)}</li>
      ))}
    </ul>,
  );
  listItems.length = 0;
}

function renderClinicalReport(text: string): ReactNode {
  const nodes: ReactNode[] = [];
  const listItems: string[] = [];

  for (const [index, rawLine] of text.split("\n").entries()) {
    const line = rawLine.trim();
    if (!line) {
      flushList(nodes, listItems, `list-${index}`);
      continue;
    }

    const heading = line.match(/^(#{1,6})\s*(.+)$/);
    if (heading) {
      flushList(nodes, listItems, `list-${index}`);
      const level = heading[1].length;
      const title = plainInlineMarkdown(heading[2]);
      if (level <= 1) {
        nodes.push(<h3 key={`heading-${index}`} className="clinical-report-title">{title}</h3>);
      } else {
        nodes.push(<h4 key={`heading-${index}`} className="clinical-report-section-title">{title}</h4>);
      }
      continue;
    }

    if (line.startsWith("- ")) {
      listItems.push(line.slice(2).trim());
      continue;
    }

    flushList(nodes, listItems, `list-${index}`);
    const labeled = line.match(/^\*\*(.+?)\*\*:?\s*(.*)$/);
    if (labeled) {
      nodes.push(
        <p key={`paragraph-${index}`} className="clinical-report-summary">
          <strong>{plainInlineMarkdown(labeled[1])}</strong>
          {labeled[2] ? <span>{plainInlineMarkdown(labeled[2])}</span> : null}
        </p>,
      );
      continue;
    }

    nodes.push(
      <p key={`paragraph-${index}`} className="clinical-report-paragraph">
        {plainInlineMarkdown(line)}
      </p>,
    );
  }

  flushList(nodes, listItems, "list-final");

  return <div className="clinical-report-content">{nodes}</div>;
}

function renderMessageText(text: string) {
  if (isClinicalReportText(text)) {
    return renderClinicalReport(text);
  }
  return <div className="clinical-message-text">{text}</div>;
}

function renderMessageContent(text: string, thinkText?: string) {
  return (
    <>
      {thinkText ? (
        <details className="workspace-card-disclosure clinical-thinking-disclosure">
          <summary>推理过程</summary>
          <div>{thinkText}</div>
        </details>
      ) : null}
      {text ? renderMessageText(text) : null}
    </>
  );
}

export function ConversationPanel({
  messages,
  draft,
  statusNode,
  isStreaming,
  latencyStatus,
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
  const latencyLabel = latencyStatusLabel(latencyStatus ?? undefined);
  const textareaDisabled = draftDisabled ?? disabled;

  return (
    <Card as="section" padding="none" className="clinical-conversation-card" data-testid="conversation-panel">
      <div className="clinical-panel-header clinical-conversation-header">
        <span className="clinical-panel-icon clinical-chat-icon" aria-hidden="true" />
        <h2>对话</h2>
      </div>

      {errorMessage ? <p className="workspace-copy workspace-copy-alert clinical-error-copy">{errorMessage}</p> : null}

      <div className="clinical-conversation-scroll">
        {canLoadHistory ? (
          <div className="clinical-history-row">
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={isLoadingHistory}
              onClick={onLoadHistory}
            >
              {isLoadingHistory ? "正在加载历史..." : "加载更早消息"}
            </Button>
          </div>
        ) : null}

        {messages.length > 0 ? (
          <ol className="workspace-message-list clinical-message-list">
            {messages.map((message) => {
              const isUser = message.type !== "ai";
              const { text: normalizedText } = normalizeMessageText(message.content);
              const thinkText = (message.thinking ?? "").trim();
              const hideText = shouldHideInlineMessageText(normalizedText, message) || (!normalizedText && !thinkText);

              if (hideText && (!message.inlineCards || message.inlineCards.length === 0)) {
                return null;
              }

              return (
                <MessageBubble
                  key={message.cursor}
                  author={isUser ? "user" : "assistant"}
                  label={messageLabel(message)}
                >
                  {!hideText || thinkText ? (
                    <div className="bubble-content">
                      {renderMessageContent(hideText ? "" : normalizedText, thinkText || undefined)}
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
                </MessageBubble>
              );
            })}
          </ol>
        ) : (
          <p className="workspace-copy clinical-empty-conversation">暂无对话。</p>
        )}
      </div>

      <div className="clinical-composer-region">
        <div className="workspace-status-row clinical-status-row">
          <span className="workspace-meta clinical-runtime-label">
            <span className="clinical-status-pulse" aria-hidden="true" /> 运行状态
          </span>
          <div className="clinical-runtime-pills">
            <strong className="workspace-status-node" data-testid="status-node">
              {executionLabel}
            </strong>
            {latencyLabel ? (
              <strong className="workspace-status-node" data-testid="latency-status">
                {latencyLabel}
              </strong>
            ) : null}
          </div>
        </div>

        <div className="workspace-composer clinical-composer">
          <div className="clinical-composer-box">
            <Textarea
              className="clinical-composer-textarea"
              placeholder="询问评估、治疗方案、引用依据或相似病例"
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
            />
            <button
              type="button"
              className="workspace-composer-send ui-button ui-button-primary"
              disabled={textareaDisabled || !draft.trim()}
              onClick={onSubmit}
              aria-label="发送消息"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}
