import { fireEvent, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { SessionState } from "../app/api/types";
import { createInitialSessionState } from "../app/store/stream-reducer";
import { buildApiClientStub, renderWorkspaceWithSceneSessions } from "../test/test-utils";

let mockSceneSessions: any;
let lastClinicalCardsProps: any;

vi.mock("../features/workspace/use-scene-sessions", () => ({
  useSceneSessions: () => mockSceneSessions,
}));

vi.mock("../features/patient-registry/use-patient-registry", () => ({
  usePatientRegistry: () => ({ bindPatient: vi.fn() }),
}));

vi.mock("../features/patient-registry/use-registry-browser", () => ({
  useRegistryBrowser: () => ({}),
}));

vi.mock("../features/database/use-database-workbench", () => ({
  useDatabaseWorkbench: () => ({}),
}));

vi.mock("../components/layout/workspace-layout", () => ({
  WorkspaceLayout: ({
    toolbar,
    leftRail,
    centerWorkspace,
    rightInspector,
  }: {
    toolbar: ReactNode;
    leftRail: ReactNode;
    centerWorkspace: ReactNode;
    rightInspector: ReactNode;
  }) => (
    <main data-testid="workspace-layout">
      <div data-testid="workspace-toolbar">{toolbar}</div>
      <div data-testid="workspace-left-rail">{leftRail}</div>
      <div data-testid="workspace-center">{centerWorkspace}</div>
      <div data-testid="workspace-right">{rightInspector}</div>
    </main>
  ),
}));

vi.mock("../features/uploads/uploads-panel", () => ({
  UploadsPanel: () => <div data-testid="uploads-panel" />,
}));

vi.mock("../features/cards/clinical-cards-panel", () => ({
  ClinicalCardsPanel: (props: any) => {
    lastClinicalCardsProps = props;
    return <div data-testid="clinical-cards-panel" />;
  },
}));

vi.mock("../features/doctor/doctor-scene-shell", () => ({
  DoctorSceneShell: () => <div data-testid="doctor-scene-shell" />,
}));

vi.mock("../features/execution-plan/execution-plan-panel", () => ({
  ExecutionPlanPanel: () => <div data-testid="execution-plan-panel" />,
}));

vi.mock("../features/roadmap/roadmap-panel", () => ({
  RoadmapPanel: () => <div data-testid="roadmap-panel" />,
}));

vi.mock("../features/chat/conversation-panel", () => ({
  ConversationPanel: ({
    draft,
    onDraftChange,
    onSubmit,
    onCardPromptRequest,
  }: {
    draft: string;
    onDraftChange: (value: string) => void;
    onSubmit: () => void;
    onCardPromptRequest?: (prompt: string, context?: Record<string, unknown>) => void;
  }) => (
    <section data-testid="mock-conversation-panel">
      <output data-testid="composer-draft">{draft}</output>
      <button type="button" onClick={() => onDraftChange("typed composer")}>
        set composer draft
      </button>
      <button type="button" onClick={() => onDraftChange("draft for card")}>
        set card draft
      </button>
      <button type="button" onClick={() => onSubmit()}>
        submit composer draft
      </button>
      <button
        type="button"
        onClick={() =>
          onCardPromptRequest?.("There has been fever for 3 days.", {
            triage_interaction: {
              question_id: "triage-q-fever-1",
              field_key: "fever",
              selection_mode: "single",
              selected_option_ids: ["fever"],
              other_text: null,
            },
          })
        }
      >
        submit triage answer
      </button>
    </section>
  ),
}));

function makeSessionState(overrides: Partial<SessionState> = {}): SessionState {
  return {
    ...createInitialSessionState(),
    ...overrides,
  };
}

function makeSceneSessions(overrides: Partial<typeof mockSceneSessions> = {}) {
  return {
    activeScene: "patient",
    setActiveScene: vi.fn(),
    bootstrapStatus: "ready",
    bootstrapError: null,
    patient: {
      state: makeSessionState({
        sessionId: "patient-session",
        currentPatientId: 101,
        cards: {
          triage_card: { type: "triage_card", title: "Triage" },
          triage_question_card: {
            type: "triage_question_card",
            question_id: "triage-q-fever-1",
          },
        },
        findings: {
          encounter_track: "outpatient_triage",
          active_inquiry: true,
          inquiry_type: "outpatient_triage",
        },
      }),
      setState: vi.fn(),
    },
    doctor: {
      state: makeSessionState({
        sessionId: "doctor-session",
      }),
      setState: vi.fn(),
    },
    applyResponseToScene: vi.fn(),
    ...overrides,
  };
}

describe("WorkspacePage patient triage submission wiring", () => {
  beforeEach(() => {
    lastClinicalCardsProps = null;
    mockSceneSessions = makeSceneSessions();
  });

  it("keeps normal composer submissions text-only and clears the draft", async () => {
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    fireEvent.click(await screen.findByRole("button", { name: /set composer draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("typed composer");

    fireEvent.click(screen.getByRole("button", { name: /submit composer draft/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(streamTurn).toHaveBeenCalledWith(
      "patient-session",
      {
        message: {
          role: "user",
          content: "typed composer",
        },
      },
      expect.any(Function),
      expect.any(AbortSignal),
    );
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("");
  });

  it("submits triage card prompts with context and keeps them out of the inspector", async () => {
    const streamTurn = vi.fn(async () => undefined);
    const apiClient = buildApiClientStub({ streamTurn });

    renderWorkspaceWithSceneSessions(apiClient);

    await screen.findByTestId("clinical-cards-panel");
    expect(lastClinicalCardsProps?.cards).toEqual({});

    fireEvent.click(screen.getByRole("button", { name: /set card draft/i }));
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("draft for card");

    fireEvent.click(screen.getByRole("button", { name: /submit triage answer/i }));

    await waitFor(() => expect(streamTurn).toHaveBeenCalledTimes(1));
    expect(streamTurn).toHaveBeenCalledWith(
      "patient-session",
      {
        message: {
          role: "user",
          content: "There has been fever for 3 days.",
        },
        context: {
          triage_interaction: {
            question_id: "triage-q-fever-1",
            field_key: "fever",
            selection_mode: "single",
            selected_option_ids: ["fever"],
            other_text: null,
          },
        },
      },
      expect.any(Function),
      expect.any(AbortSignal),
    );
    expect(screen.getByTestId("composer-draft")).toHaveTextContent("draft for card");
  });
});
