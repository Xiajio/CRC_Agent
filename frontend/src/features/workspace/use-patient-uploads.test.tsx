import { act, renderHook, waitFor } from "@testing-library/react";
import type { Dispatch, SetStateAction } from "react";
import { describe, expect, it, vi } from "vitest";

import { ApiClientError } from "../../app/api/client";
import type { SessionState } from "../../app/api/types";
import { createInitialSessionState } from "../../app/store/stream-reducer";
import { buildApiClientStub, makeSessionResponse } from "../../test/test-utils";
import { readUploadMaxBytes, uploadTooLargeMessage } from "../workspace/workspace-flow-utils";
import { usePatientUploads } from "./use-patient-uploads";

type Deferred<T> = {
  promise: Promise<T>;
  resolve(value: T): void;
};

const MISSING_PATIENT_SESSION_ERROR = "\u60a3\u8005\u4f1a\u8bdd\u8fd8\u672a\u51c6\u5907\u597d\u4e0a\u4f20\u3002";
const UPLOADING_PREFIX = "\u6b63\u5728\u4e0a\u4f20 ";
const SUCCESS_PREFIX = "\u5df2\u4e0a\u4f20 ";

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((ready) => {
    resolve = ready;
  });
  return { promise, resolve };
}

type PatientController = {
  state: SessionState;
  setState: Dispatch<SetStateAction<SessionState>>;
};

function makePatientController(initialState: SessionState): PatientController {
  let state = initialState;
  return {
    get state() {
      return state;
    },
    setState(update) {
      state = typeof update === "function" ? update(state) : update;
    },
  };
}

describe("usePatientUploads", () => {
  it("captures the current patient session id and refreshes it after a successful upload", async () => {
    const refreshed = makeSessionResponse({
      session_id: "patient-session",
      scene: "patient",
      snapshot: {
        patient_identity: {
          patient_name: "Zhang San",
          patient_number: "P-1001",
          identity_locked: true,
        },
        uploaded_assets: {},
      },
    });
    const apiClient = buildApiClientStub({
      uploadFile: vi.fn(async () => ({
        asset_id: "1",
        filename: "report.pdf",
        content_type: "application/pdf",
        size: 12,
        sha256: "sha",
        reused: false,
        derived: { record_id: 1 },
      })),
      getSession: vi.fn(async () => refreshed),
    });
    const applyPatientResponse = vi.fn();

    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: "patient-session",
    });
    const { result } = renderHook(
      () =>
        usePatientUploads({
          apiClient,
          patientSessionId: "patient-session",
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
    );

    const file = new File(["report"], "report.pdf", { type: "application/pdf" });

    await act(async () => {
      await result.current.uploadFile(file);
    });

    expect(apiClient.uploadFile).toHaveBeenCalledWith("patient-session", file);
    expect(apiClient.getSession).toHaveBeenCalledWith("patient-session");
    expect(applyPatientResponse).toHaveBeenCalledWith(refreshed);
    expect(patient.state.uploadedAssets).toEqual({
      "1": {
        filename: "report.pdf",
        derived: { record_id: 1 },
      },
    });
    expect(patient.state.patientIdentity).toEqual({
      patient_name: "Zhang San",
      patient_number: "P-1001",
      identity_locked: true,
    });
    expect(result.current.uploadStatus).toBe(`${SUCCESS_PREFIX}report.pdf`);
    expect(result.current.isUploading).toBe(false);
    expect(result.current.errorMessage).toBeNull();
  });

  it("blocks uploads over the max size and keeps upload disabled state off", async () => {
    const apiClient = buildApiClientStub();
    const applyPatientResponse = vi.fn();
    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: "patient-session",
    });
    const { result } = renderHook(
      () =>
        usePatientUploads({
          apiClient,
          patientSessionId: "patient-session",
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
    );

    const oversized = new File(["report"], "too-large.pdf", { type: "application/pdf" });
    Object.defineProperty(oversized, "size", {
      value: readUploadMaxBytes() + 1,
    });

    await act(async () => {
      await result.current.uploadFile(oversized);
    });

    expect(apiClient.uploadFile).not.toHaveBeenCalled();
    expect(result.current.errorMessage).toBe(uploadTooLargeMessage(readUploadMaxBytes()));
    expect(result.current.uploadStatus).toBeNull();
    expect(result.current.isUploading).toBe(false);
  });

  it("maps backend 413 upload errors to the friendly size message", async () => {
    const apiClient = buildApiClientStub({
      uploadFile: vi.fn(async () => {
        throw new ApiClientError(
          413,
          "UPLOAD_TOO_LARGE: maximum size is 26214400 bytes",
          { detail: "UPLOAD_TOO_LARGE: maximum size is 26214400 bytes" },
        );
      }),
    });
    const applyPatientResponse = vi.fn();
    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: "patient-session",
    });
    const { result } = renderHook(
      () =>
        usePatientUploads({
          apiClient,
          patientSessionId: "patient-session",
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
    );
    const file = new File(["report"], "report.pdf", { type: "application/pdf" });

    await act(async () => {
      await result.current.uploadFile(file);
    });

    expect(result.current.errorMessage).toBe(uploadTooLargeMessage(readUploadMaxBytes()));
    expect(result.current.uploadStatus).toBeNull();
    expect(result.current.isUploading).toBe(false);
    expect(applyPatientResponse).not.toHaveBeenCalled();
  });

  it("returns a clear missing-session message and skips API calls when patient session is absent", async () => {
    const apiClient = buildApiClientStub();
    const applyPatientResponse = vi.fn();
    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: null,
    });
    const { result } = renderHook(
      () =>
        usePatientUploads({
          apiClient,
          patientSessionId: null,
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
    );
    const file = new File(["report"], "report.pdf", { type: "application/pdf" });

    await act(async () => {
      await result.current.uploadFile(file);
    });

    expect(result.current.errorMessage).toBe(MISSING_PATIENT_SESSION_ERROR);
    expect(apiClient.uploadFile).not.toHaveBeenCalled();
    expect(apiClient.getSession).not.toHaveBeenCalled();
    expect(applyPatientResponse).not.toHaveBeenCalled();
  });

  it("clears only status with clearUploadStatus and still allows in-flight upload completion", async () => {
    const refreshedSession = makeSessionResponse({
      session_id: "patient-session",
      scene: "patient",
      snapshot: {
        patient_identity: null,
        uploaded_assets: {
          "1": {
            filename: "report.pdf",
            derived: { record_id: 1 },
          },
        },
      },
    });
    const upload = createDeferred<{
      asset_id: string;
      filename: string;
      content_type: string;
      size: number;
      sha256: string;
      reused: boolean;
      derived: { record_id: number };
    }>();
    const getSession = createDeferred<ReturnType<typeof makeSessionResponse>>();
    const apiClient = buildApiClientStub({
      uploadFile: vi.fn(() => upload.promise),
      getSession: vi.fn(() => getSession.promise),
    });
    const applyPatientResponse = vi.fn();
    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: "patient-session",
    });

    const { result } = renderHook(
      () =>
        usePatientUploads({
          apiClient,
          patientSessionId: "patient-session",
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
    );

    const file = new File(["report"], "report.pdf", { type: "application/pdf" });
    act(() => {
      void result.current.uploadFile(file);
    });

    await waitFor(() => expect(result.current.uploadStatus).toBe(`${UPLOADING_PREFIX}${file.name}...`));
    act(() => {
      result.current.clearUploadStatus();
    });
    expect(result.current.uploadStatus).toBeNull();

    upload.resolve({
      asset_id: "1",
      filename: "report.pdf",
      content_type: "application/pdf",
      size: 8,
      sha256: "sha",
      reused: false,
      derived: { record_id: 1 },
    });
    getSession.resolve(refreshedSession);

    await waitFor(() => expect(result.current.isUploading).toBe(false));
    expect(applyPatientResponse).toHaveBeenCalledWith(refreshedSession);
    expect(patient.state.uploadedAssets).toEqual({
      "1": {
        filename: "report.pdf",
        derived: { record_id: 1 },
      },
    });
    expect(result.current.uploadStatus).toBe(`${SUCCESS_PREFIX}report.pdf`);
  });

  it("clearError only clears the error message", async () => {
    const apiClient = buildApiClientStub();
    const applyPatientResponse = vi.fn();
    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: "patient-session",
    });
    const { result } = renderHook(
      () =>
        usePatientUploads({
          apiClient,
          patientSessionId: "patient-session",
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
    );

    const oversized = new File(["report"], "too-large.pdf", { type: "application/pdf" });
    Object.defineProperty(oversized, "size", {
      value: readUploadMaxBytes() + 1,
    });
    await act(async () => {
      await result.current.uploadFile(oversized);
    });
    expect(result.current.errorMessage).toBe(uploadTooLargeMessage(readUploadMaxBytes()));

    act(() => {
      result.current.clearError();
    });
    expect(result.current.errorMessage).toBeNull();
    expect(result.current.uploadStatus).toBeNull();
  });

  it("uses upload sequence checks so resetUploadState() invalidates in-flight completion", async () => {
    const refreshed = makeSessionResponse({
      session_id: "patient-session",
      scene: "patient",
      snapshot: {
        patient_identity: {
          patient_name: "Zhang San",
          patient_number: "P-1001",
          identity_locked: true,
        },
      },
    });
    const upload = createDeferred<{
      asset_id: string;
      filename: string;
      content_type: string;
      size: number;
      sha256: string;
      reused: boolean;
      derived: { record_id: number };
    }>();
    const getSession = createDeferred<ReturnType<typeof makeSessionResponse>>();
    const apiClient = buildApiClientStub({
      uploadFile: vi.fn(() => upload.promise),
      getSession: vi.fn(() => getSession.promise),
    });
    const applyPatientResponse = vi.fn();

    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: "patient-session",
      uploadedAssets: { old: { filename: "old.pdf", derived: {} } },
    });

    const { result } = renderHook(
      () =>
        usePatientUploads({
          apiClient,
          patientSessionId: "patient-session",
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
    );

    const file = new File(["report"], "report.pdf", { type: "application/pdf" });
    act(() => {
      void result.current.uploadFile(file);
    });

    await waitFor(() => expect(result.current.isUploading).toBe(true));

    act(() => {
      result.current.resetUploadState();
    });
    expect(result.current.isUploading).toBe(false);
    expect(result.current.uploadStatus).toBeNull();
    expect(result.current.errorMessage).toBeNull();

    upload.resolve({
      asset_id: "1",
      filename: "report.pdf",
      content_type: "application/pdf",
      size: 6,
      sha256: "sha",
      reused: false,
      derived: { record_id: 1 },
    });
    getSession.resolve(refreshed);

    await waitFor(() => expect(result.current.isUploading).toBe(false));
    expect(patient.state.uploadedAssets).toEqual({ old: { filename: "old.pdf", derived: {} } });
    expect(applyPatientResponse).not.toHaveBeenCalled();
    expect(apiClient.getSession).not.toHaveBeenCalled();
  });

  it("invalidates in-flight upload completion when patient session id changes", async () => {
    const upload = createDeferred<{
      asset_id: string;
      filename: string;
      content_type: string;
      size: number;
      sha256: string;
      reused: boolean;
      derived: { record_id: number };
    }>();
    const getSession = createDeferred<ReturnType<typeof makeSessionResponse>>();
    const apiClient = buildApiClientStub({
      uploadFile: vi.fn(() => upload.promise),
      getSession: vi.fn(() => getSession.promise),
    });
    const applyPatientResponse = vi.fn();
    const patient = makePatientController({
      ...createInitialSessionState(),
      sessionId: "old-session",
      uploadedAssets: {
        existing: {
          filename: "existing.pdf",
          derived: { record_id: 9 },
        },
      },
    });

    const { result, rerender } = renderHook(
      ({ patientSessionId }) =>
        usePatientUploads({
          apiClient,
          patientSessionId,
          setPatientState: patient.setState,
          applyPatientResponse,
        }),
      { initialProps: { patientSessionId: "old-session" } },
    );

    const file = new File(["report"], "report.pdf", { type: "application/pdf" });
    act(() => {
      void result.current.uploadFile(file);
    });

    await waitFor(() => expect(result.current.isUploading).toBe(true));

    act(() => {
      rerender({ patientSessionId: "new-session" });
    });
    expect(result.current.isUploading).toBe(false);
    expect(result.current.uploadStatus).toBeNull();
    expect(result.current.errorMessage).toBeNull();

    upload.resolve({
      asset_id: "1",
      filename: "report.pdf",
      content_type: "application/pdf",
      size: 8,
      sha256: "sha",
      reused: false,
      derived: { record_id: 1 },
    });
    getSession.resolve(
      makeSessionResponse({
        session_id: "new-session",
        scene: "patient",
        snapshot: {
          patient_identity: {
            patient_name: "Liu",
            patient_number: "P-2002",
            identity_locked: true,
          },
        },
      }),
    );

    await waitFor(() => expect(result.current.isUploading).toBe(false));
    expect(patient.state.uploadedAssets).toEqual({
      existing: {
        filename: "existing.pdf",
        derived: { record_id: 9 },
      },
    });
    expect(applyPatientResponse).not.toHaveBeenCalled();
    expect(apiClient.getSession).not.toHaveBeenCalled();
  });
});

