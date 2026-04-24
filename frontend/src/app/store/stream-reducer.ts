import type {
  CardUpsertEvent,
  FrontendMessage,
  InlineCard,
  JsonObject,
  MessageHistoryResponse,
  RecoverySnapshot,
  ContextMaintenanceState,
  SafetyAlertState,
  SessionMessage,
  SessionResponse,
  SessionState,
  StreamEvent,
} from "../api/types";

const INLINE_CARD_TYPES = new Set([
  "patient_card",
  "imaging_card",
  "tumor_detection_card",
  "radiomics_report_card",
  "triage_card",
  "triage_question_card",
  "decision_card",
]);
const INLINE_CARD_PRIORITY: Record<string, number> = {
  patient_card: 1,
  imaging_card: 2,
  tumor_detection_card: 3,
  radiomics_report_card: 4,
  triage_card: 5,
  triage_question_card: 5,
  decision_card: 6,
};

type FrontendInlineCard = NonNullable<FrontendMessage["inlineCards"]>[number];

function normalizeWorkflowNode(value: unknown): string | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }

  const normalized = value.trim().toLowerCase().replace(/[\s_]+/g, "-");
  if (normalized.includes("intent")) {
    return "intent";
  }
  if (normalized.includes("planner") || normalized.includes("planning")) {
    return "planner";
  }
  if (normalized.includes("tool-router") || normalized.includes("toolrouter")) {
    return "tool-router";
  }
  if (normalized.includes("assessment")) {
    return "assessment";
  }
  if (normalized.includes("decision")) {
    return "decision";
  }
  if (normalized.includes("citation")) {
    return "citation";
  }
  if (normalized.includes("evaluator") || normalized.includes("evaluation")) {
    return "evaluator";
  }
  if (normalized.includes("final")) {
    return "finalize";
  }

  return normalized;
}

function roadmapStepKey(step: JsonObject): string | null {
  return (
    normalizeWorkflowNode(step.id) ??
    normalizeWorkflowNode(step.step_id) ??
    normalizeWorkflowNode(step.title) ??
    normalizeWorkflowNode(step.step_name)
  );
}

function advanceRoadmapFromNode(roadmap: JsonObject[], node: string): JsonObject[] {
  if (roadmap.length === 0) {
    return roadmap;
  }

  const activeKey = normalizeWorkflowNode(node);
  if (!activeKey) {
    return roadmap;
  }

  const activeIndex = roadmap.findIndex((step) => roadmapStepKey(step) === activeKey);
  if (activeIndex < 0) {
    return roadmap;
  }

  return roadmap.map((step, index) => {
    if (index < activeIndex) {
      return { ...step, status: "completed" };
    }
    if (index === activeIndex) {
      return { ...step, status: "in_progress" };
    }
    return step.status ? step : { ...step, status: "waiting" };
  });
}

function markActivePlanStepBlocked(plan: JsonObject[], message: string): JsonObject[] {
  if (plan.length === 0) {
    return plan;
  }

  const activeIndex = plan.findIndex((step) => step.status === "in_progress");
  const pendingIndex = plan.findIndex((step) => step.status === "pending" || step.status === "waiting");
  const targetIndex = activeIndex >= 0 ? activeIndex : pendingIndex;
  if (targetIndex < 0) {
    return plan;
  }

  return plan.map((step, index) => (
    index === targetIndex
      ? { ...step, status: "blocked", error_message: message }
      : step
  ));
}

function normalizeInlineCard(card: InlineCard | { cardType: string; payload: JsonObject }): FrontendInlineCard {
  if ("cardType" in card) {
    return {
      cardType: card.cardType,
      payload: card.payload,
    };
  }

  return {
    cardType: card.card_type,
    payload: card.payload,
  };
}

function pruneInlineCards(cards: FrontendInlineCard[]): FrontendInlineCard[] {
  if (cards.length === 0) {
    return cards;
  }

  const highestPriority = Math.max(
    ...cards.map((card) => INLINE_CARD_PRIORITY[card.cardType] ?? 0),
  );

  if (highestPriority <= 0) {
    return cards;
  }

  return cards.filter((card) => (INLINE_CARD_PRIORITY[card.cardType] ?? 0) === highestPriority);
}

function mergeInlineCards(
  existing: FrontendInlineCard[] | undefined,
  incoming: FrontendInlineCard[] | undefined,
): FrontendInlineCard[] | undefined {
  const merged = [...(existing ?? [])];
  const seen = new Set(
    merged.map((card) => JSON.stringify([card.cardType, card.payload])),
  );

  for (const card of incoming ?? []) {
    const key = JSON.stringify([card.cardType, card.payload]);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(card);
  }

  const pruned = pruneInlineCards(merged);
  return pruned.length > 0 ? pruned : undefined;
}

function eligibleInlineCardFromEvent(event: CardUpsertEvent): FrontendInlineCard | null {
  if (!INLINE_CARD_TYPES.has(event.card_type)) {
    return null;
  }

  return {
    cardType: event.card_type,
    payload: event.payload,
  };
}

function attachInlineCardsToLatestAssistantMessage(
  messages: FrontendMessage[],
  cursor: string | null,
  cards: FrontendInlineCard[],
): FrontendMessage[] {
  if (!cursor || cards.length === 0) {
    return messages;
  }

  return messages.map((message) => {
    if (message.cursor !== cursor || message.type !== "ai") {
      return message;
    }

    return {
      ...message,
      inlineCards: mergeInlineCards(message.inlineCards, cards),
    };
  });
}

function normalizeMessage(message: SessionMessage): FrontendMessage {
  return {
    cursor: message.cursor,
    type: message.type,
    content: message.content,
    id: message.id ?? undefined,
    name: message.name ?? undefined,
    toolCallId: message.tool_call_id ?? undefined,
    status: message.status ?? undefined,
    assetRefs: message.asset_refs ?? [],
    inlineCards: message.inline_cards?.map(normalizeInlineCard),
  };
}

function messagesFromSnapshot(snapshot: RecoverySnapshot): FrontendMessage[] {
  return snapshot.messages.map(normalizeMessage);
}

function cardsToRecord(cards: CardUpsertEvent[]): Record<string, Record<string, unknown>> {
  return cards.reduce<Record<string, Record<string, unknown>>>((acc, card) => {
    acc[card.card_type] = card.payload;
    return acc;
  }, {});
}

function messageKey(message: FrontendMessage): string {
  return message.cursor || message.id || JSON.stringify([message.type, message.content]);
}

function findMessageCursorById(messages: FrontendMessage[], messageId: string): string | null {
  const match = messages.find((message) => message.id === messageId);
  return match?.cursor ?? null;
}

function nextMessageCursor(messages: FrontendMessage[]): string {
  const numericCursors = messages
    .map((message) => Number.parseInt(message.cursor, 10))
    .filter((value) => Number.isFinite(value));

  if (numericCursors.length === 0) {
    return String(messages.length);
  }

  return String(Math.max(...numericCursors) + 1);
}

export function createInitialSessionState(): SessionState {
  return {
    sessionId: null,
    threadId: null,
    snapshotVersion: 0,
    runtime: null,
    messages: [],
    messagesTotal: 0,
    messagesNextBeforeCursor: null,
    cards: {},
    roadmap: [],
    findings: {},
    patientProfile: null,
    patientIdentity: null,
    stage: null,
    references: [],
    plan: [],
    critic: null,
    safetyAlert: null,
    assessmentDraft: null,
    currentPatientId: null,
    uploadedAssets: {},
    contextMaintenance: null,
    contextState: null,
    statusNode: null,
    lastError: null,
    activeRunId: null,
    pendingInlineCards: [],
    latestAssistantMessageCursor: null,
    streamingMessageCursors: {},
  };
}

export function hydrateSessionState(state: SessionState, response: SessionResponse): SessionState {
  const snapshot = response.snapshot;
  return {
    ...state,
    sessionId: response.session_id,
    threadId: response.thread_id,
    snapshotVersion: response.snapshot_version,
    runtime: response.runtime,
    messages: messagesFromSnapshot(snapshot),
    messagesTotal: snapshot.messages_total,
    messagesNextBeforeCursor: snapshot.messages_next_before_cursor,
    cards: cardsToRecord(snapshot.cards),
    roadmap: snapshot.roadmap,
    findings: snapshot.findings,
    patientProfile: snapshot.patient_profile,
    patientIdentity: snapshot.patient_identity ?? null,
    stage: snapshot.stage,
    references: snapshot.references,
    plan: snapshot.plan,
    critic: snapshot.critic,
    safetyAlert: (snapshot.safety_alert as SafetyAlertState | null) ?? null,
    assessmentDraft: snapshot.assessment_draft,
    currentPatientId: snapshot.current_patient_id,
    uploadedAssets: snapshot.uploaded_assets,
    contextMaintenance: (snapshot.context_maintenance as ContextMaintenanceState | null) ?? null,
    contextState: snapshot.context_state ?? null,
    statusNode: null,
    lastError: null,
    activeRunId: null,
    pendingInlineCards: [],
    latestAssistantMessageCursor: null,
    streamingMessageCursors: {},
  };
}

export function mergeMessageHistory(
  state: SessionState,
  history: MessageHistoryResponse,
  mode: "prepend" | "append",
): SessionState {
  const incoming = history.messages.map(normalizeMessage);
  const combined = mode === "prepend" ? [...incoming, ...state.messages] : [...state.messages, ...incoming];
  const deduped: FrontendMessage[] = [];
  const seen = new Set<string>();

  for (const message of combined) {
    const key = messageKey(message);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(message);
  }

  return {
    ...state,
    threadId: history.thread_id,
    snapshotVersion: history.snapshot_version,
    messages: deduped,
    messagesTotal: history.messages_total,
    messagesNextBeforeCursor: history.next_before_cursor,
    pendingInlineCards: [],
    latestAssistantMessageCursor: null,
    streamingMessageCursors: {},
  };
}

export function reduceStreamEvent(state: SessionState, event: StreamEvent): SessionState {
  switch (event.type) {
    case "status.node":
      return {
        ...state,
        statusNode: event.node,
        roadmap: advanceRoadmapFromNode(state.roadmap, event.node),
      };
    case "message.delta":
      {
        const existingCursor =
          state.streamingMessageCursors[event.message_id] ??
          findMessageCursorById(state.messages, event.message_id);

        if (existingCursor) {
          return {
            ...state,
            messages: state.messages.map((message) => {
              if (message.cursor !== existingCursor || message.type !== "ai") {
                return message;
              }

              return {
                ...message,
                content: `${typeof message.content === "string" ? message.content : ""}${event.delta}`,
                id: event.message_id,
                node: event.node ?? message.node,
              };
            }),
            latestAssistantMessageCursor: existingCursor,
            streamingMessageCursors: {
              ...state.streamingMessageCursors,
              [event.message_id]: existingCursor,
            },
          };
        }

        const cursor = nextMessageCursor(state.messages);
        return {
          ...state,
          messages: [
            ...state.messages,
            {
              cursor,
              type: "ai",
              content: event.delta,
              id: event.message_id,
              assetRefs: [],
              node: event.node ?? undefined,
              inlineCards: state.pendingInlineCards.length > 0 ? state.pendingInlineCards : undefined,
            },
          ],
          messagesTotal: Math.max(state.messagesTotal, state.messages.length + 1),
          pendingInlineCards: [],
          latestAssistantMessageCursor: cursor,
          streamingMessageCursors: {
            ...state.streamingMessageCursors,
            [event.message_id]: cursor,
          },
        };
      }
    case "message.done":
      {
        const eventInlineCards = (event.inline_cards ?? []).map(normalizeInlineCard);
        const existingCursor =
          (event.message_id ? state.streamingMessageCursors[event.message_id] : null) ??
          (event.message_id ? findMessageCursorById(state.messages, event.message_id) : null);

        if (existingCursor) {
          const nextStreamingMessageCursors = { ...state.streamingMessageCursors };
          if (event.message_id) {
            delete nextStreamingMessageCursors[event.message_id];
          }

          return {
            ...state,
            messages: state.messages.map((message) => {
              if (message.cursor !== existingCursor || message.type !== "ai") {
                return message;
              }

              const pendingAndEventCards = mergeInlineCards(state.pendingInlineCards, eventInlineCards);
              return {
                ...message,
                content: event.content,
                thinking: event.thinking ?? undefined,
                id: event.message_id ?? message.id,
                node: event.node ?? message.node,
                inlineCards: mergeInlineCards(message.inlineCards, pendingAndEventCards),
              };
            }),
            messagesTotal: Math.max(state.messagesTotal, state.messages.length),
            pendingInlineCards: [],
            latestAssistantMessageCursor: event.role === "assistant" ? existingCursor : null,
            streamingMessageCursors: nextStreamingMessageCursors,
          };
        }

        const cursor = nextMessageCursor(state.messages);
        const inlineCards = mergeInlineCards(state.pendingInlineCards, eventInlineCards);

        return {
          ...state,
          messages: [
            ...state.messages,
            {
              cursor,
              type: event.role === "assistant" ? "ai" : event.role,
              content: event.content,
              thinking: event.thinking ?? undefined,
              id: event.message_id ?? undefined,
              assetRefs: [],
              node: event.node ?? undefined,
              inlineCards,
            },
          ],
          messagesTotal: Math.max(state.messagesTotal, state.messages.length + 1),
          pendingInlineCards: [],
          latestAssistantMessageCursor: event.role === "assistant" ? cursor : null,
        };
      }
    case "card.upsert":
      {
        const inlineCard = eligibleInlineCardFromEvent(event);
        const nextMessages = inlineCard
          ? attachInlineCardsToLatestAssistantMessage(
              state.messages,
              state.latestAssistantMessageCursor,
              [inlineCard],
            )
          : state.messages;

        const attachedToLatestMessage = inlineCard
          ? nextMessages.some(
              (message) =>
                message.cursor === state.latestAssistantMessageCursor &&
                (message.inlineCards?.some((card) => card.cardType === inlineCard.cardType) ?? false),
            )
          : false;

        return {
          ...state,
          messages: nextMessages,
          cards: {
            ...state.cards,
            [event.card_type]: event.payload,
          },
          pendingInlineCards:
            inlineCard && !attachedToLatestMessage
              ? mergeInlineCards(state.pendingInlineCards, [inlineCard]) ?? []
              : state.pendingInlineCards,
        };
      }
    case "roadmap.update":
      return {
        ...state,
        roadmap: event.roadmap,
      };
    case "findings.patch":
      return {
        ...state,
        findings: {
          ...state.findings,
          ...event.patch,
        },
      };
    case "patient_profile.update":
      return {
        ...state,
        patientProfile: event.profile,
      };
    case "stage.update":
      return {
        ...state,
        stage: event.stage,
      };
    case "references.append": {
      const next = [...state.references];
      const seen = new Set(
        next.map((item) => (typeof item.id === "string" ? item.id : JSON.stringify(item))),
      );
      for (const item of event.items) {
        const key = typeof item.id === "string" ? item.id : JSON.stringify(item);
        if (!seen.has(key)) {
          seen.add(key);
          next.push(item);
        }
      }
      return {
        ...state,
        references: next,
      };
    }
    case "safety.alert":
      return {
        ...state,
        safetyAlert: {
          message: event.message,
          blocking: true,
        },
      };
    case "critic.verdict":
      return {
        ...state,
        critic: {
          verdict: event.verdict,
          feedback: event.feedback ?? null,
          iteration_count: event.iteration_count ?? null,
        },
      };
    case "plan.update":
      return {
        ...state,
        plan: event.plan,
      };
    case "error":
      return {
        ...state,
        plan: markActivePlanStepBlocked(state.plan, event.message),
        lastError: {
          code: event.code,
          message: event.message,
          recoverable: event.recoverable,
        },
      };
    case "context.maintenance":
      return {
        ...state,
        contextMaintenance: {
          status: event.status,
          message: event.message,
        },
      };
    case "done":
      return {
        ...state,
        threadId: event.thread_id,
        activeRunId: null,
        snapshotVersion: event.snapshot_version,
        statusNode: null,
        pendingInlineCards: [],
      };
    default:
      return state;
  }
}
