import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";

import type { ApiClient } from "../../app/api/client";
import type { SessionResponse, SessionState, UploadResponse } from "../../app/api/types";
import {
  readUploadMaxBytes,
  readWorkspaceErrorMessage,
  uploadTooLargeMessage,
} from "./workspace-flow-utils";

export type UsePatientUploadsOptions = {
  apiClient: ApiClient;
  patientSessionId: string | null;
  setPatientState: Dispatch<SetStateAction<SessionState>>;
  applyPatientResponse: (response: SessionResponse) => void;
};

export type PatientUploadsController = {
  isUploading: boolean;
  uploadStatus: string | null;
  errorMessage: string | null;
  uploadFile(file: File): Promise<void>;
  clearUploadStatus(): void;
  clearError(): void;
  resetUploadState(): void;
};

const MISSING_PATIENT_SESSION_ERROR = "\u60a3\u8005\u4f1a\u8bdd\u8fd8\u672a\u51c6\u5907\u597d\u4e0a\u4f20\u3002";

export function usePatientUploads({
  apiClient,
  patientSessionId,
  setPatientState,
  applyPatientResponse,
}: UsePatientUploadsOptions): PatientUploadsController {
  const uploadSequenceRef = useRef(0);
  const previousPatientSessionIdRef = useRef(patientSessionId);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (previousPatientSessionIdRef.current !== patientSessionId) {
      uploadSequenceRef.current += 1;
      setIsUploading(false);
      setUploadStatus(null);
      setErrorMessage(null);
    }

    previousPatientSessionIdRef.current = patientSessionId;
  }, [patientSessionId]);

  const resetUploadState = () => {
    uploadSequenceRef.current += 1;
    setUploadStatus(null);
    setErrorMessage(null);
    setIsUploading(false);
  };

  const clearUploadStatus = () => {
    setUploadStatus(null);
  };

  const clearError = () => {
    setErrorMessage(null);
  };

  const mergeUploadedAsset = (response: UploadResponse) => {
    setPatientState((current) => ({
      ...current,
      uploadedAssets: {
        ...current.uploadedAssets,
        [String(response.asset_id)]: {
          filename: response.filename,
          derived: response.derived,
        },
      },
    }));
  };

  const uploadFile = async (file: File) => {
    const uploadSessionId = patientSessionId;

    if (!uploadSessionId) {
      setErrorMessage(MISSING_PATIENT_SESSION_ERROR);
      setUploadStatus(null);
      return;
    }

    const uploadMaxBytes = readUploadMaxBytes();
    if (file.size > uploadMaxBytes) {
      setErrorMessage(uploadTooLargeMessage(uploadMaxBytes));
      setUploadStatus(null);
      return;
    }

    const sequence = ++uploadSequenceRef.current;
    setIsUploading(true);
    setErrorMessage(null);
    setUploadStatus(`\u6b63\u5728\u4e0a\u4f20 ${file.name}...`);

    try {
      const uploadResponse = await apiClient.uploadFile(uploadSessionId, file);
      if (sequence !== uploadSequenceRef.current) {
        return;
      }

      // Keep optimistic assets visible when a refresh fails or is superseded, matching existing behavior.
      mergeUploadedAsset(uploadResponse);
      const refreshed = await apiClient.getSession(uploadSessionId);
      if (sequence !== uploadSequenceRef.current) {
        return;
      }

      applyPatientResponse(refreshed);
      setPatientState((current) => ({
        ...current,
        patientIdentity: refreshed.snapshot.patient_identity ?? null,
      }));
      setUploadStatus(`\u5df2\u4e0a\u4f20 ${uploadResponse.filename}`);
    } catch (error) {
      if (sequence !== uploadSequenceRef.current) {
        return;
      }

      setErrorMessage(readWorkspaceErrorMessage(error));
      setUploadStatus(null);
    } finally {
      if (sequence === uploadSequenceRef.current) {
        setIsUploading(false);
      }
    }
  };

  return {
    isUploading,
    uploadStatus,
    errorMessage,
    uploadFile,
    clearUploadStatus,
    clearError,
    resetUploadState,
  };
}
