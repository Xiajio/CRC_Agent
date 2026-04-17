import type { MessageHistoryResponse, SessionResponse, SessionState, StreamEvent } from "../api/types";
import {
  createInitialSessionState,
  hydrateSessionState,
  mergeMessageHistory,
  reduceStreamEvent,
} from "./stream-reducer";

export interface SessionStore {
  getState(): SessionState;
  initialize(response: SessionResponse): void;
  applyEvent(event: StreamEvent): void;
  prependHistory(history: MessageHistoryResponse): void;
  appendHistory(history: MessageHistoryResponse): void;
  reset(): void;
}

export function createSessionStore(initialState: SessionState = createInitialSessionState()): SessionStore {
  let state = initialState;

  return {
    getState() {
      return state;
    },

    initialize(response) {
      state = hydrateSessionState(createInitialSessionState(), response);
    },

    applyEvent(event) {
      state = reduceStreamEvent(state, event);
    },

    prependHistory(history) {
      state = mergeMessageHistory(state, history, "prepend");
    },

    appendHistory(history) {
      state = mergeMessageHistory(state, history, "append");
    },

    reset() {
      state = createInitialSessionState();
    },
  };
}
