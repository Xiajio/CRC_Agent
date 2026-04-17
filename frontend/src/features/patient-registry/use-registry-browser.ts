import { useEffect, useState } from "react";

import { ApiClientError } from "../../app/api/client";
import type {
  PatientRegistryAlert,
  PatientRegistryDetail,
  PatientRegistryItem,
  PatientRegistryRecord,
  PatientRegistrySearchRequest,
} from "../../app/api/types";
import { useApiClient } from "../../app/providers";

type RegistryBrowserSearchState = {
  patientId: string;
  tumorLocation: string;
  mmrStatus: string;
  clinicalStage: string;
  limit: number;
};

const DEFAULT_SEARCH_STATE: RegistryBrowserSearchState = {
  patientId: "",
  tumorLocation: "",
  mmrStatus: "",
  clinicalStage: "",
  limit: 20,
};

function readErrorMessage(error: unknown): string {
  if (error instanceof ApiClientError) {
    return error.message;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Patient registry request failed.";
}

function buildSearchRequest(state: RegistryBrowserSearchState): PatientRegistrySearchRequest {
  const patientId = Number(state.patientId);
  return {
    patient_id: Number.isFinite(patientId) ? patientId : null,
    tumor_location: state.tumorLocation.trim() || null,
    mmr_status: state.mmrStatus.trim() || null,
    clinical_stage: state.clinicalStage.trim() || null,
    limit: state.limit,
  };
}

export function useRegistryBrowser(options: { enabled: boolean }) {
  const { enabled } = options;
  const apiClient = useApiClient();
  const [recentPatients, setRecentPatients] = useState<PatientRegistryItem[]>([]);
  const [searchState, setSearchState] = useState<RegistryBrowserSearchState>(DEFAULT_SEARCH_STATE);
  const [searchResults, setSearchResults] = useState<PatientRegistryItem[]>([]);
  const [previewPatientId, setPreviewPatientId] = useState<number | null>(null);
  const [previewDetail, setPreviewDetail] = useState<PatientRegistryDetail | null>(null);
  const [previewRecords, setPreviewRecords] = useState<PatientRegistryRecord[]>([]);
  const [previewAlerts, setPreviewAlerts] = useState<PatientRegistryAlert[]>([]);
  const [isLoadingRecent, setIsLoadingRecent] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);
  const [isDeletingPatient, setIsDeletingPatient] = useState(false);
  const [isClearingRegistry, setIsClearingRegistry] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshRecent(limit = 20) {
    if (!enabled) {
      return;
    }

    setIsLoadingRecent(true);
    try {
      const response = await apiClient.getRecentPatients(limit);
      setRecentPatients(response.items);
    } catch (nextError) {
      setError(readErrorMessage(nextError));
    } finally {
      setIsLoadingRecent(false);
    }
  }

  async function runSearch() {
    if (!enabled) {
      return;
    }

    setIsSearching(true);
    setError(null);
    try {
      const response = await apiClient.searchPatientRegistry(buildSearchRequest(searchState));
      setSearchResults(response.items);
    } catch (nextError) {
      setError(readErrorMessage(nextError));
    } finally {
      setIsSearching(false);
    }
  }

  async function previewPatient(patientId: number) {
    setIsLoadingPreview(true);
    setError(null);
    try {
      const [detailResponse, recordsResponse, alertsResponse] = await Promise.all([
        apiClient.getPatientRegistryDetail(patientId),
        apiClient.getPatientRecords(patientId),
        apiClient.getPatientRegistryAlerts(patientId),
      ]);
      setPreviewPatientId(patientId);
      setPreviewDetail(detailResponse);
      setPreviewRecords(recordsResponse.items);
      setPreviewAlerts(alertsResponse.items);
    } catch (nextError) {
      setError(readErrorMessage(nextError));
    } finally {
      setIsLoadingPreview(false);
    }
  }

  async function deletePatient(patientId: number) {
    if (!enabled) {
      return false;
    }

    setIsDeletingPatient(true);
    setError(null);
    try {
      await apiClient.deletePatientRegistryPatient(patientId);
      setRecentPatients((current) => current.filter((item) => item.patient_id !== patientId));
      setSearchResults((current) => current.filter((item) => item.patient_id !== patientId));
      setPreviewPatientId((current) => (current === patientId ? null : current));
      if (previewPatientId === patientId) {
        setPreviewDetail(null);
        setPreviewRecords([]);
        setPreviewAlerts([]);
      }
      await refreshRecent();
      return true;
    } catch (nextError) {
      setError(readErrorMessage(nextError));
      return false;
    } finally {
      setIsDeletingPatient(false);
    }
  }

  async function clearRegistry() {
    if (!enabled) {
      return false;
    }

    setIsClearingRegistry(true);
    setError(null);
    try {
      await apiClient.clearPatientRegistry();
      setRecentPatients([]);
      setSearchResults([]);
      setPreviewPatientId(null);
      setPreviewDetail(null);
      setPreviewRecords([]);
      setPreviewAlerts([]);
      return true;
    } catch (nextError) {
      setError(readErrorMessage(nextError));
      return false;
    } finally {
      setIsClearingRegistry(false);
    }
  }

  function setSearchField(field: keyof RegistryBrowserSearchState, value: string | number) {
    setSearchState((current) => ({
      ...current,
      [field]: value,
    }));
  }

  useEffect(() => {
    if (!enabled) {
      return;
    }
    void refreshRecent();
  }, [enabled]);

  return {
    recentPatients,
    searchState,
    searchResults,
    previewPatientId,
    previewDetail,
    previewRecords,
    previewAlerts,
    isLoadingRecent,
    isSearching,
    isLoadingPreview,
    isDeletingPatient,
    isClearingRegistry,
    error,
    setSearchField,
    refreshRecent,
    runSearch,
    previewPatient,
    deletePatient,
    clearRegistry,
  };
}
