import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiClientError } from "../../app/api/client";
import type { RecoverySnapshot } from "../../app/api/types";
import { AppProviders } from "../../app/providers";
import { PatientIdentityPanel } from "./patient-identity-panel";
import { buildApiClientStub, makeSessionResponse } from "../../test/test-utils";
import { render } from "@testing-library/react";

function renderPanel(
  apiClient = buildApiClientStub(),
  patientIdentity?: RecoverySnapshot["patient_identity"],
) {
  return render(
    <AppProviders apiClient={apiClient}>
      <PatientIdentityPanel sessionId="patient-session" patientIdentity={patientIdentity ?? null} />
    </AppProviders>,
  );
}

describe("PatientIdentityPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the empty state CTA", () => {
    renderPanel();

    expect(screen.getByText("目前还没有填写患者信息。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "填写患者信息" })).toBeInTheDocument();
  });

  it("treats an unlocked empty snapshot as empty", () => {
    renderPanel(buildApiClientStub(), {
      patient_name: null,
      patient_number: null,
      identity_locked: false,
    });

    expect(screen.getByText("目前还没有填写患者信息。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "填写患者信息" })).toBeInTheDocument();
    expect(screen.queryByText("如需修改，请在医生端数据库中处理")).not.toBeInTheDocument();
  });

  it("enters editing when the CTA is clicked", () => {
    renderPanel();

    fireEvent.click(screen.getByRole("button", { name: "填写患者信息" }));

    expect(screen.getByLabelText("患者名称")).toBeInTheDocument();
    expect(screen.getByLabelText("患者编号")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "取消" })).toBeInTheDocument();
  });

  it("submits successfully and shows the returned snapshot", async () => {
    const saveSessionPatientIdentity = vi.fn(async () =>
      makeSessionResponse({
        session_id: "patient-session",
        snapshot: {
          patient_identity: {
            patient_name: "王小明",
            patient_number: "P-2001",
            identity_locked: true,
          },
        } as any,
      }),
    );
    const apiClient = buildApiClientStub({
      saveSessionPatientIdentity,
    } as any);

    renderPanel(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "填写患者信息" }));
    fireEvent.change(screen.getByLabelText("患者名称"), { target: { value: "王小明" } });
    fireEvent.change(screen.getByLabelText("患者编号"), { target: { value: "P-2001" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));

    await waitFor(() =>
      expect(saveSessionPatientIdentity).toHaveBeenCalledWith("patient-session", "王小明", "P-2001"),
    );
    expect(screen.getByText("患者名称：王小明")).toBeInTheDocument();
    expect(screen.getByText("患者编号：P-2001")).toBeInTheDocument();
    expect(screen.getByText("如需修改，请在医生端数据库中处理")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "填写患者信息" })).not.toBeInTheDocument();
  });

  it("keeps editing and shows the duplicate number message", async () => {
    const saveSessionPatientIdentity = vi.fn(async () => {
      throw new ApiClientError(409, "Conflict", { detail: { code: "PATIENT_NUMBER_ALREADY_EXISTS" } });
    });
    const apiClient = buildApiClientStub({
      saveSessionPatientIdentity,
    } as any);

    renderPanel(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "填写患者信息" }));
    fireEvent.change(screen.getByLabelText("患者名称"), { target: { value: "王小明" } });
    fireEvent.change(screen.getByLabelText("患者编号"), { target: { value: "P-2001" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));

    await waitFor(() => expect(screen.getByText("患者编号已存在，请更换")).toBeInTheDocument());
    expect(screen.getByLabelText("患者名称")).toHaveValue("王小明");
    expect(screen.getByLabelText("患者编号")).toHaveValue("P-2001");
  });

  it("maps the locked identity error to the locked copy", async () => {
    const saveSessionPatientIdentity = vi.fn(async () => {
      throw new ApiClientError(409, "Conflict", { detail: { code: "PATIENT_IDENTITY_LOCKED" } });
    });
    const apiClient = buildApiClientStub({
      saveSessionPatientIdentity,
    } as any);

    renderPanel(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "填写患者信息" }));
    fireEvent.change(screen.getByLabelText("患者名称"), { target: { value: "王小明" } });
    fireEvent.change(screen.getByLabelText("患者编号"), { target: { value: "P-2001" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));

    await waitFor(() => expect(screen.getByText("如需修改，请在医生端数据库中处理")).toBeInTheDocument());
    expect(screen.getByLabelText("患者名称")).toHaveValue("王小明");
    expect(screen.getByLabelText("患者编号")).toHaveValue("P-2001");
  });

  it("falls back to the generic save failure copy on 422", async () => {
    const saveSessionPatientIdentity = vi.fn(async () => {
      throw new ApiClientError(422, "Validation failed", { detail: [{ msg: "field required" }] });
    });
    const apiClient = buildApiClientStub({
      saveSessionPatientIdentity,
    } as any);

    renderPanel(apiClient);

    fireEvent.click(screen.getByRole("button", { name: "填写患者信息" }));
    fireEvent.change(screen.getByLabelText("患者名称"), { target: { value: "王小明" } });
    fireEvent.change(screen.getByLabelText("患者编号"), { target: { value: "P-2001" } });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));

    await waitFor(() => expect(screen.getByText("保存失败，请稍后重试")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "保存" })).toBeInTheDocument();
  });

  it("shows the saved state without edit affordances", () => {
    renderPanel(
      buildApiClientStub(),
      {
        patient_name: "王小明",
        patient_number: "P-2001",
        identity_locked: true,
      },
    );

    expect(screen.getByText("患者名称：王小明")).toBeInTheDocument();
    expect(screen.getByText("患者编号：P-2001")).toBeInTheDocument();
    expect(screen.getByText("如需修改，请在医生端数据库中处理")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "填写患者信息" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "保存" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "取消" })).not.toBeInTheDocument();
  });
});
