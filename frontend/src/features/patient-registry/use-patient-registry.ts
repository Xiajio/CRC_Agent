import { useEffect, useRef, useState } from "react";

import { ApiClientError } from "../../app/api/client";
import type {
  PatientRegistryAlert,
  PatientRegistryDetail,
  PatientRegistryRecord,
  SessionResponse,
} from "../../app/api/types";
import { useApiClient } from "../../app/providers";

function readErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) {
    return error.message;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Patient registry request failed.";
}

export function usePatientRegistry(options: { enabled: boolean; registryPatientId: number | null }) {
  const { enabled, registryPatientId } = options;
  const apiClient = useApiClient();
  const [boundPatientDetail, setBoundPatientDetail] = useState<PatientRegistryDetail | null>(null);
  const [boundPatientRecords, setBoundPatientRecords] = useState<PatientRegistryRecord[]>([]);
  const [boundPatientAlerts, setBoundPatientAlerts] = useState<PatientRegistryAlert[]>([]);
  const [isLoadingBoundPatient, setIsLoadingBoundPatient] = useState(false);
  const [isBindingPatient, setIsBindingPatient] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadRequestIdRef = useRef(0);
  const missingRegistryIdRef = useRef<number | null>(null);

  async function loadBoundPatient(patientId: number) {
    const requestId = loadRequestIdRef.current + 1;
    loadRequestIdRef.current = requestId;
    setIsLoadingBoundPatient(true);
    setError(null);
    try {
      const [detailResponse, recordsResponse, alertsResponse] = await Promise.all([
        apiClient.getPatientRegistryDetail(patientId),
        apiClient.getPatientRecords(patientId),
        apiClient.getPatientRegistryAlerts(patientId),
      ]);
      if (loadRequestIdRef.current !== requestId) {
        return;
      }
      missingRegistryIdRef.current = null;
      setBoundPatientDetail(detailResponse);
      setBoundPatientRecords(recordsResponse.items);
      setBoundPatientAlerts(alertsResponse.items);
    } catch (error) {
      if (loadRequestIdRef.current !== requestId) {
        return;
      }
      if (error instanceof ApiClientError && error.status === 404) {
        missingRegistryIdRef.current = patientId;
      }
      setError(readErrorMessage(error));
    } finally {
      if (loadRequestIdRef.current === requestId) {
        setIsLoadingBoundPatient(false);
      }
    }
  }

  async function bindPatient(sessionId: string, patientId: number): Promise<SessionResponse> {
    setIsBindingPatient(true);
    setError(null);
    try {
      const response = await apiClient.bindPatient(sessionId, patientId);
      await loadBoundPatient(patientId);
      return response;
    } finally {
      setIsBindingPatient(false);
    }
  }

  useEffect(() => {
    if (!enabled) {
      loadRequestIdRef.current += 1;
      setBoundPatientDetail(null);
      setBoundPatientRecords([]);
      setBoundPatientAlerts([]);
      setIsLoadingBoundPatient(false);
      return;
    }
    if (registryPatientId === null) {
      loadRequestIdRef.current += 1;
      missingRegistryIdRef.current = null;
      setBoundPatientDetail(null);
      setBoundPatientRecords([]);
      setBoundPatientAlerts([]);
      setIsLoadingBoundPatient(false);
      return;
    }
    if (missingRegistryIdRef.current === registryPatientId) {
      return;
    }
    setBoundPatientDetail(null);
    setBoundPatientRecords([]);
    setBoundPatientAlerts([]);
    void loadBoundPatient(registryPatientId);
  }, [enabled, registryPatientId]);

  return {
    boundPatientDetail,
    boundPatientRecords,
    boundPatientAlerts,
    isLoadingBoundPatient,
    isBindingPatient,
    error,
    bindPatient,
  };
}
