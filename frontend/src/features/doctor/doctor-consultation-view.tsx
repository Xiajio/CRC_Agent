import type { FrontendMessage } from "../../app/api/types";
import type { CardPromptHandler } from "../cards/card-renderers-extended";
import { ConversationPanel, type ConversationLatencyStatus } from "../chat/conversation-panel";

export type DoctorLatencyStatus = ConversationLatencyStatus;

export type DoctorConsultationViewProps = {
  messages: FrontendMessage[];
  draft: string;
  statusNode: string | null;
  isStreaming: boolean;
  isLoadingHistory: boolean;
  canLoadHistory: boolean;
  disabled: boolean;
  errorMessage: string | null;
  latencyStatus?: DoctorLatencyStatus | null;
  onLoadHistory: () => void;
  onDraftChange: (value: string) => void;
  onSubmit: () => void;
  onCardPromptRequest?: CardPromptHandler;
};

export function DoctorConsultationView({
  messages,
  draft,
  statusNode,
  isStreaming,
  isLoadingHistory,
  canLoadHistory,
  disabled,
  errorMessage,
  latencyStatus,
  onLoadHistory,
  onDraftChange,
  onSubmit,
  onCardPromptRequest,
}: DoctorConsultationViewProps) {
  return (
    <ConversationPanel
      messages={messages}
      draft={draft}
      statusNode={statusNode}
      isStreaming={isStreaming}
      isLoadingHistory={isLoadingHistory}
      canLoadHistory={canLoadHistory}
      disabled={disabled}
      errorMessage={errorMessage}
      latencyStatus={latencyStatus}
      onLoadHistory={onLoadHistory}
      onDraftChange={onDraftChange}
      onSubmit={onSubmit}
      onCardPromptRequest={onCardPromptRequest}
    />
  );
}
