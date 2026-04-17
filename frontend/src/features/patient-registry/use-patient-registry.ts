import { useEffect, useState } from "react";

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

export function usePatientRegistry(options: { enabled: boolean; currentPatientId: number | null }) {
  const { enabled, currentPatientId } = options;
  const apiClient = useApiClient();
  const [boundPatientDetail, setBoundPatientDetail] = useState<PatientRegistryDetail | null>(null);
  const [boundPatientRecords, setBoundPatientRecords] = useState<PatientRegistryRecord[]>([]);
  const [boundPatientAlerts, setBoundPatientAlerts] = useState<PatientRegistryAlert[]>([]);
  const [isLoadingBoundPatient, setIsLoadingBoundPatient] = useState(false);
  const [isBindingPatient, setIsBindingPatient] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadBoundPatient(patientId: number) {
    setIsLoadingBoundPatient(true);
    try {
      const [detailResponse, recordsResponse, alertsResponse] = await Promise.all([
        apiClient.getPatientRegistryDetail(patientId),
        apiClient.getPatientRecords(patientId),
        apiClient.getPatientRegistryAlerts(patientId),
      ]);
      setBoundPatientDetail(detailResponse);
      setBoundPatientRecords(recordsResponse.items);
      setBoundPatientAlerts(alertsResponse.items);
    } catch (error) {
      setError(readErrorMessage(error));
    } finally {
      setIsLoadingBoundPatient(false);
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
      setBoundPatientDetail(null);
      setBoundPatientRecords([]);
      setBoundPatientAlerts([]);
      return;
    }
    if (currentPatientId === null) {
      setBoundPatientDetail(null);
      setBoundPatientRecords([]);
      setBoundPatientAlerts([]);
      return;
    }
    void loadBoundPatient(currentPatientId);
  }, [enabled, currentPatientId]);

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
