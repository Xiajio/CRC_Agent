export interface UiState {
  selectedCardType: string | null;
  composerDraft: string;
  isStreaming: boolean;
  lastError: string | null;
}

export interface UiStore {
  getState(): UiState;
  patch(next: Partial<UiState>): void;
  reset(): void;
}

export function createInitialUiState(): UiState {
  return {
    selectedCardType: null,
    composerDraft: "",
    isStreaming: false,
    lastError: null,
  };
}

export function createUiStore(initialState: UiState = createInitialUiState()): UiStore {
  let state = initialState;

  return {
    getState() {
      return state;
    },
    patch(next) {
      state = {
        ...state,
        ...next,
      };
    },
    reset() {
      state = createInitialUiState();
    },
  };
}
