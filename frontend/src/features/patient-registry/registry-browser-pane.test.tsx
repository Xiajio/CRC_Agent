import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { makePatientRegistryDetail } from "../../test/test-utils";
import { RegistryBrowserPane } from "./registry-browser-pane";

function renderPane(overrides: Partial<Parameters<typeof RegistryBrowserPane>[0]> = {}) {
  const props: Parameters<typeof RegistryBrowserPane>[0] = {
    searchState: {
      patientId: "",
      tumorLocation: "",
      mmrStatus: "",
      clinicalStage: "",
      limit: 20,
    },
    searchResults: [
      {
        patient_id: 33,
        status: "draft",
        created_by_session_id: "sess-patient",
        updated_at: "2026-04-16T00:00:00Z",
        tumor_location: "rectum",
        mmr_status: "dMMR",
        clinical_stage: "cT3N1M0",
      },
    ],
    previewPatientId: 33,
    previewDetail: makePatientRegistryDetail({ patient_id: 33 }),
    previewRecords: [],
    previewAlerts: [],
    currentPatientId: null,
    isSearching: false,
    isLoadingPreview: true,
    isBindingCurrentPatient: false,
    isDeletingPatient: false,
    isClearingRegistry: false,
    error: null,
    onSearchFieldChange: vi.fn(),
    onSearchSubmit: vi.fn(),
    onPreviewPatient: vi.fn(),
    onSetCurrentPatient: vi.fn(),
    onDeletePatient: vi.fn(),
    onClearRegistry: vi.fn(),
    ...overrides,
  };

  return render(<RegistryBrowserPane {...props} />);
}

describe("RegistryBrowserPane", () => {
  it("disables preview, bind, and delete actions while preview loading is active", () => {
    renderPane();

    expect(screen.getByRole("button", { name: "previewing 33" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "set current patient 33" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "delete patient 33" })).toBeDisabled();
  });

  it("renders the registry search form and preview actions with Chinese labels", () => {
    renderPane({
      previewPatientId: 44,
      previewDetail: makePatientRegistryDetail({ patient_id: 44 }),
      isLoadingPreview: false,
    });

    const browser = screen.getByTestId("registry-browser-pane");
    expect(within(browser).getByRole("heading", { name: /患者库检索/ })).toBeInTheDocument();
    expect(within(browser).getByText("患者 ID")).toBeInTheDocument();
    expect(within(browser).getByText("肿瘤部位")).toBeInTheDocument();
    expect(within(browser).getByText("MMR状态")).toBeInTheDocument();
    expect(within(browser).getByText("临床分期")).toBeInTheDocument();
    expect(within(browser).getByRole("button", { name: "检索患者库" })).toBeInTheDocument();
    expect(within(browser).getByRole("button", { name: "clear registry" })).toHaveTextContent("清空患者库");
    expect(within(browser).getByRole("button", { name: "preview patient 33" })).toHaveTextContent("预览 #33");
    expect(within(browser).getByText("患者 #33")).toBeInTheDocument();

    const preview = screen.getByTestId("registry-preview-panel");
    expect(within(preview).getByRole("heading", { name: /患者库预览: 患者 #44/ })).toBeInTheDocument();
    expect(within(preview).getByText("主诉: rectal bleeding")).toBeInTheDocument();
    expect(within(preview).getByText("年龄: 52")).toBeInTheDocument();
    expect(within(preview).getByText("性别: female")).toBeInTheDocument();
    expect(within(preview).getByText("肿瘤部位: rectum")).toBeInTheDocument();
    expect(within(preview).getByText("临床分期: cT3N1M0")).toBeInTheDocument();
    expect(within(preview).getByText("MMR状态: dMMR")).toBeInTheDocument();
    expect(within(preview).getByRole("button", { name: "set current patient 44" })).toHaveTextContent("设为当前患者");
    expect(within(preview).getByRole("button", { name: "delete patient 44" })).toHaveTextContent("删除患者");
  });

  it("confirms with Chinese copy before clearing or deleting registry patients", () => {
    const onClearRegistry = vi.fn();
    const onDeletePatient = vi.fn();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);

    renderPane({
      previewPatientId: 44,
      previewDetail: makePatientRegistryDetail({ patient_id: 44 }),
      isLoadingPreview: false,
      onClearRegistry,
      onDeletePatient,
    });

    fireEvent.click(screen.getByRole("button", { name: "clear registry" }));
    expect(confirmSpy).toHaveBeenCalledWith("确定要清空患者库中的所有患者吗？此操作仅用于开发环境清理。");
    expect(onClearRegistry).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "delete patient 44" }));
    expect(confirmSpy).toHaveBeenCalledWith("确定要从患者库中删除患者 #44 吗？");
    expect(onDeletePatient).toHaveBeenCalledWith(44);

    confirmSpy.mockRestore();
  });

  it("explains why clear and delete actions are disabled for the current patient", () => {
    renderPane({
      currentPatientId: 33,
      isLoadingPreview: false,
    });

    expect(screen.getByText("清空患者库前请先重置当前的医生会话场景。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "clear registry" })).toHaveTextContent("清空患者库");
    expect(screen.getByRole("button", { name: "clear registry" })).toBeDisabled();
    expect(screen.getByText("删除此患者记录前，请先重置或更改当前患者。")).toBeInTheDocument();
  });
});
