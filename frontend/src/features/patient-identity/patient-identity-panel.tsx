import { useEffect, useState } from "react";

import { ApiClientError } from "../../app/api/client";
import type { PatientIdentitySnapshot } from "../../app/api/types";
import { useApiClient } from "../../app/providers";

type PanelMode = "empty" | "editing" | "saving" | "error" | "saved";

type PanelState = {
  mode: PanelMode;
  patientName: string;
  patientNumber: string;
  errorMessage: string | null;
};

export interface PatientIdentityPanelProps {
  sessionId: string | null;
  patientIdentity?: PatientIdentitySnapshot | null;
  onSaved?: (patientIdentity: PatientIdentitySnapshot) => void;
}

const GENERIC_SAVE_ERROR = "保存失败，请稍后重试";
const REQUIRED_FIELD_ERROR = "患者名称和患者编号均为必填项";
const LOCKED_MESSAGE = "如需修改，请在医生端数据库中处理";
const DUPLICATE_MESSAGE = "患者编号已存在，请更换";

function createStateFromIdentity(patientIdentity?: PatientIdentitySnapshot | null): PanelState {
  const hasSavedIdentity =
    patientIdentity != null
    && (
      patientIdentity.identity_locked
      || patientIdentity.patient_name !== null
      || patientIdentity.patient_number !== null
    );

  if (hasSavedIdentity && patientIdentity) {
    return {
      mode: "saved",
      patientName: patientIdentity.patient_name ?? "",
      patientNumber: patientIdentity.patient_number ?? "",
      errorMessage: null,
    };
  }

  return {
    mode: "empty",
    patientName: "",
    patientNumber: "",
    errorMessage: null,
  };
}

function getErrorCode(detail: unknown): string | null {
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    return null;
  }

  if (detail && typeof detail === "object") {
    const record = detail as Record<string, unknown>;
    if (typeof record.code === "string") {
      return record.code;
    }
    if ("detail" in record) {
      return getErrorCode(record.detail);
    }
  }

  return null;
}

function mapSaveError(error: unknown): string {
  if (error instanceof ApiClientError) {
    const code = getErrorCode(error.detail);
    if (code === "PATIENT_NUMBER_ALREADY_EXISTS") {
      return DUPLICATE_MESSAGE;
    }
    if (code === "PATIENT_IDENTITY_LOCKED") {
      return LOCKED_MESSAGE;
    }
    if (code === "NOT_PATIENT_SESSION" || code === "PATIENT_IDENTITY_NOT_FOUND") {
      return GENERIC_SAVE_ERROR;
    }
    if (error.status === 422) {
      return GENERIC_SAVE_ERROR;
    }
  }

  return GENERIC_SAVE_ERROR;
}

export function PatientIdentityPanel({
  sessionId,
  patientIdentity,
  onSaved,
}: PatientIdentityPanelProps) {
  const apiClient = useApiClient();
  const [panelState, setPanelState] = useState<PanelState>(() => createStateFromIdentity(patientIdentity));

  useEffect(() => {
    setPanelState((current) => {
      if (current.mode === "editing" || current.mode === "saving") {
        return current;
      }

      return createStateFromIdentity(patientIdentity);
    });
  }, [patientIdentity?.patient_name, patientIdentity?.patient_number, patientIdentity?.identity_locked]);

  async function handleSave() {
    const patientName = panelState.patientName.trim();
    const patientNumber = panelState.patientNumber.trim();

    if (!patientName || !patientNumber) {
      setPanelState((current) => ({
        ...current,
        mode: "error",
        errorMessage: REQUIRED_FIELD_ERROR,
      }));
      return;
    }

    if (!sessionId) {
      setPanelState((current) => ({
        ...current,
        mode: "error",
        errorMessage: GENERIC_SAVE_ERROR,
      }));
      return;
    }

    setPanelState((current) => ({
      ...current,
      mode: "saving",
      errorMessage: null,
    }));

    try {
      const response = await apiClient.saveSessionPatientIdentity(sessionId, patientName, patientNumber);
      const nextIdentity = response.snapshot.patient_identity ?? {
        patient_name: patientName,
        patient_number: patientNumber,
        identity_locked: true,
      };

      setPanelState({
        mode: "saved",
        patientName: nextIdentity.patient_name ?? "",
        patientNumber: nextIdentity.patient_number ?? "",
        errorMessage: null,
      });
      onSaved?.(nextIdentity);
    } catch (error) {
      setPanelState((current) => ({
        ...current,
        mode: "error",
        errorMessage: mapSaveError(error),
      }));
    }
  }

  function handleStartEditing() {
    setPanelState((current) => ({
      ...current,
      mode: "editing",
      errorMessage: null,
    }));
  }

  function handleCancel() {
    setPanelState(createStateFromIdentity(patientIdentity));
  }

  function handleFieldChange(field: "patientName" | "patientNumber", value: string) {
    setPanelState((current) => ({
      ...current,
      mode: "editing",
      [field]: value,
      errorMessage: null,
    }));
  }

  const isSaving = panelState.mode === "saving";
  const showForm = panelState.mode === "editing" || panelState.mode === "saving" || panelState.mode === "error";
  const showError = panelState.mode === "error" ? panelState.errorMessage : null;

  return (
    <section
      className="workspace-card"
      style={{ position: "sticky", top: "16px", display: "flex", flexDirection: "column", gap: "16px" }}
      data-testid="patient-identity-panel"
      aria-label="患者信息"
    >
      <h2 style={{ margin: 0, fontSize: "1.125rem", fontWeight: 600 }}>患者信息</h2>

      {panelState.mode === "empty" ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px", alignItems: "flex-start" }}>
          <p className="workspace-copy" style={{ margin: 0 }}>目前还没有填写患者信息。</p>
          <button type="button" className="workspace-primary-button" onClick={handleStartEditing}>
            填写患者信息
          </button>
        </div>
      ) : null}

      {panelState.mode === "saved" ? (
        <div className="workspace-copy">
          <p>患者名称：{panelState.patientName || "—"}</p>
          <p>患者编号：{panelState.patientNumber || "—"}</p>
          <p>{LOCKED_MESSAGE}</p>
        </div>
      ) : null}

      {showForm ? (
        <form
          style={{ display: "flex", flexDirection: "column", gap: "16px" }}
          onSubmit={(event) => {
            event.preventDefault();
            void handleSave();
          }}
        >
          <label className="workspace-field" style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <span>患者名称</span>
            <input
              aria-label="患者名称"
              value={panelState.patientName}
              disabled={isSaving}
              onChange={(event) => handleFieldChange("patientName", event.target.value)}
              style={{
                padding: "8px 12px",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
                fontSize: "0.875rem",
                outline: "none",
                width: "100%",
                boxSizing: "border-box",
              }}
            />
          </label>
          <label className="workspace-field" style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <span>患者编号</span>
            <input
              aria-label="患者编号"
              value={panelState.patientNumber}
              disabled={isSaving}
              onChange={(event) => handleFieldChange("patientNumber", event.target.value)}
              style={{
                padding: "8px 12px",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
                fontSize: "0.875rem",
                outline: "none",
                width: "100%",
                boxSizing: "border-box",
              }}
            />
          </label>
          {showError ? (
            <p className="workspace-copy workspace-copy-alert" role="alert">
              {showError}
            </p>
          ) : null}
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            <button
              type="submit"
              className="workspace-primary-button"
              disabled={isSaving || !sessionId}
            >
              {isSaving ? "保存中..." : "保存"}
            </button>
            <button
              type="button"
              className="workspace-secondary-button"
              disabled={isSaving}
              onClick={handleCancel}
            >
              取消
            </button>
          </div>
        </form>
      ) : null}
    </section>
  );
}
