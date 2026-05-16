import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ClinicalEventLogEntry, JsonObject } from "../../app/api/types";
import type { CardPatientContext } from "../cards/card-renderers-extended";
import { buildMultimodalPromptContext, groupMultimodalCards, MULTIMODAL_ACTIONS } from "./doctor-multimodal-utils";

const clinicalCardsPanelMock = vi.fn(({ title, cards, emptyMessage }: { title?: string; cards: Record<string, JsonObject>; emptyMessage?: string }) => (
  <section data-testid={`panel-${title ?? "untitled"}`} data-empty-message={emptyMessage ?? ""} data-cards={Object.keys(cards).join(",")} />
));

vi.mock("../cards/clinical-cards-panel", () => ({
  ClinicalCardsPanel: (props: { title?: string; cards: Record<string, JsonObject>; emptyMessage?: string }) => clinicalCardsPanelMock(props),
}));

import { DoctorMultimodalView } from "./doctor-multimodal-view";

beforeEach(() => {
  clinicalCardsPanelMock.mockClear();
});

function renderView(overrides: Partial<Parameters<typeof DoctorMultimodalView>[0]> = {}) {
  const patientContext: CardPatientContext = overrides.patientContext ?? null;
  const props: Parameters<typeof DoctorMultimodalView>[0] = {
    registryPatientId: null,
    caseDatabasePatientId: null,
    patientRegistry: {
      boundPatientDetail: null,
      boundPatientRecords: [],
      boundPatientAlerts: [],
      isLoadingBoundPatient: false,
    } as never,
    cards: {},
    critic: null,
    eventLog: [],
    isStreaming: false,
    disabled: false,
    patientContext,
    onCardPromptRequest: vi.fn(),
    ...overrides,
  };

  return render(<DoctorMultimodalView {...props} />);
}

function findActionButtons() {
  return screen.getAllByRole("button", { name: /.+/ });
}

describe("DoctorMultimodalView", () => {
  it("renders empty states and disables actions when there is no patient context", () => {
    renderView();

    expect(screen.getByTestId("doctor-multimodal-view")).toBeInTheDocument();
    expect(screen.getByText("未绑定注册患者")).toBeInTheDocument();
    expect(screen.getByText("未绑定病例样本")).toBeInTheDocument();
    expect(screen.getByText("暂无患者资料")).toBeInTheDocument();
    expect(screen.getByText("暂无告警")).toBeInTheDocument();
    expect(screen.getByText("暂无影像卡片")).toBeInTheDocument();
    expect(screen.getByText("暂无病理卡片")).toBeInTheDocument();
    expect(screen.getByText("暂无放射组学卡片")).toBeInTheDocument();

    for (const button of findActionButtons()) {
      expect(button).toBeDisabled();
    }
  });

  it("passes grouped multimodal cards to ClinicalCardsPanel and excludes non multimodal cards", () => {
    const grouped = groupMultimodalCards([
      { cardType: "imaging_card", payload: { summary: "Imaging summary" } as JsonObject },
      { cardType: "pathology_card", payload: { summary: "Pathology summary" } as JsonObject },
      { cardType: "decision_card", payload: { summary: "Not multimodal" } as JsonObject },
    ]);

    renderView({
      cards: {
        imaging_card: { summary: "Imaging summary" },
        pathology_card: { summary: "Pathology summary" },
        decision_card: { summary: "Not multimodal" },
      },
    });

    expect(clinicalCardsPanelMock).toHaveBeenCalledTimes(grouped.length);
    expect(clinicalCardsPanelMock.mock.calls.map(([props]) => props.title)).toEqual(grouped.map((group) => group.title));
    const flattenedCards = clinicalCardsPanelMock.mock.calls.flatMap(([props]) => Object.keys(props.cards));
    expect(flattenedCards).toEqual(["imaging_card", "pathology_card"]);
    expect(screen.queryByText("Not multimodal")).not.toBeInTheDocument();
  });

  it("enables summary and handoff actions with registry patient context but keeps imaging and pathology disabled without case id", () => {
    const onCardPromptRequest = vi.fn();

    renderView({
      registryPatientId: 42,
      patientContext: { registry_patient_id: 42 },
      onCardPromptRequest,
    });

    const buttons = findActionButtons();
    expect(buttons[0]).toBeDisabled();
    expect(buttons[1]).toBeDisabled();
    expect(buttons[2]).toBeEnabled();
    expect(buttons[3]).toBeEnabled();

    fireEvent.click(buttons[2]);
    fireEvent.click(buttons[3]);

    expect(onCardPromptRequest).toHaveBeenNthCalledWith(1, MULTIMODAL_ACTIONS[2].prompt, buildMultimodalPromptContext({ registry_patient_id: 42 }));
    expect(onCardPromptRequest).toHaveBeenNthCalledWith(2, MULTIMODAL_ACTIONS[3].prompt, buildMultimodalPromptContext({ registry_patient_id: 42 }));
  });

  it("merges explicit case database ids into the prompt context when patientContext only has registry data", () => {
    const onCardPromptRequest = vi.fn();

    renderView({
      registryPatientId: 42,
      caseDatabasePatientId: "007",
      patientContext: { registry_patient_id: 42 },
      onCardPromptRequest,
    });

    const buttons = findActionButtons();
    expect(buttons[0]).toBeEnabled();
    fireEvent.click(buttons[0]);

    expect(onCardPromptRequest).toHaveBeenCalledWith(
      MULTIMODAL_ACTIONS[0].prompt,
      buildMultimodalPromptContext({ registry_patient_id: 42, case_database_patient_id: "007" }),
    );
  });

  it("uses explicit registry ids even when patientContext is empty", () => {
    const onCardPromptRequest = vi.fn();

    renderView({
      registryPatientId: 42,
      patientContext: {},
      onCardPromptRequest,
    });

    const buttons = findActionButtons();
    expect(buttons[2]).toBeEnabled();
    expect(buttons[3]).toBeEnabled();

    fireEvent.click(buttons[2]);

    expect(onCardPromptRequest).toHaveBeenCalledWith(
      MULTIMODAL_ACTIONS[2].prompt,
      buildMultimodalPromptContext({ registry_patient_id: 42 }),
    );
  });

  it("enables imaging actions when the case database patient id is present", () => {
    const onCardPromptRequest = vi.fn();

    renderView({
      registryPatientId: 42,
      caseDatabasePatientId: "007",
      patientContext: { registry_patient_id: 42, case_database_patient_id: "007" },
      onCardPromptRequest,
    });

    const buttons = findActionButtons();
    expect(buttons[0]).toBeEnabled();
    expect(buttons[1]).toBeEnabled();

    fireEvent.click(buttons[0]);

    expect(onCardPromptRequest).toHaveBeenCalledWith(MULTIMODAL_ACTIONS[0].prompt, buildMultimodalPromptContext({ registry_patient_id: 42, case_database_patient_id: "007" }));
  });

  it("renders critic review output without leaking raw thinking blocks", () => {
    const critic: JsonObject = {
      requires_human_review: true,
      verdict: "REJECTED",
      feedback: "<think>internal note</think>{\"feedback\":\"Need a clearer differential diagnosis.\"}",
    };
    const eventLog: ClinicalEventLogEntry[] = [
      {
        id: "event-1",
        kind: "critic",
        title: "Critic event",
        detail: "<think>hidden</think>Critic detail that should be compacted",
        tone: "warning",
        requiresHumanReview: true,
      },
    ];

    renderView({
      critic,
      eventLog,
    });

    expect(screen.getAllByText("HUMAN_REVIEW_REQUIRED")).toHaveLength(2);
    expect(screen.getByText("REJECTED")).toBeInTheDocument();
    expect(screen.getByText("Need a clearer differential diagnosis.")).toBeInTheDocument();
    expect(screen.getByText("Critic detail that should be compacted")).toBeInTheDocument();
    expect(screen.queryByText(/<think>/)).not.toBeInTheDocument();
  });

  it("disables actions when the dashboard is disabled or streaming", () => {
    const onCardPromptRequest = vi.fn();

    const { rerender } = render(
      <DoctorMultimodalView
        registryPatientId={42}
        caseDatabasePatientId="007"
        patientRegistry={{
          boundPatientDetail: null,
          boundPatientRecords: [],
          boundPatientAlerts: [],
          isLoadingBoundPatient: false,
        } as never}
        cards={{}}
        critic={null}
        eventLog={[]}
        isStreaming={false}
        disabled={false}
        patientContext={{ registry_patient_id: 42, case_database_patient_id: "007" }}
        onCardPromptRequest={onCardPromptRequest}
      />,
    );

    expect(findActionButtons()[0]).toBeEnabled();

    rerender(
      <DoctorMultimodalView
        registryPatientId={42}
        caseDatabasePatientId="007"
        patientRegistry={{
          boundPatientDetail: null,
          boundPatientRecords: [],
          boundPatientAlerts: [],
          isLoadingBoundPatient: false,
        } as never}
        cards={{}}
        critic={null}
        eventLog={[]}
        isStreaming={false}
        disabled={true}
        patientContext={{ registry_patient_id: 42, case_database_patient_id: "007" }}
        onCardPromptRequest={onCardPromptRequest}
      />,
    );

    expect(findActionButtons()[0]).toBeDisabled();

    rerender(
      <DoctorMultimodalView
        registryPatientId={42}
        caseDatabasePatientId="007"
        patientRegistry={{
          boundPatientDetail: null,
          boundPatientRecords: [],
          boundPatientAlerts: [],
          isLoadingBoundPatient: false,
        } as never}
        cards={{}}
        critic={null}
        eventLog={[]}
        isStreaming={true}
        disabled={false}
        patientContext={{ registry_patient_id: 42, case_database_patient_id: "007" }}
        onCardPromptRequest={onCardPromptRequest}
      />,
    );

    expect(findActionButtons()[0]).toBeDisabled();
  });
});
