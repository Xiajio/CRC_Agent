import "@testing-library/jest-dom/vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../../app/providers";
import type {
  PatientRegistryAlertsResponse,
  PatientRegistryDetail,
  PatientRegistryListResponse,
  PatientRegistryDeleteResponse,
  PatientRegistryRecordsResponse,
} from "../../app/api/types";
import {
  buildApiClientStub,
  makePatientRegistryAlertsResponse,
  makePatientRegistryDetail,
  makePatientRegistryListResponse,
  makePatientRegistryRecordsResponse,
} from "../../test/test-utils";
import { useRegistryBrowser } from "./use-registry-browser";

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;

  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

function renderBrowser(apiClient = buildApiClientStub()) {
  return renderHook(() => useRegistryBrowser({ enabled: true }), {
    wrapper: ({ children }) => <AppProviders apiClient={apiClient}>{children}</AppProviders>,
  });
}

function makePatientResponses(patientId: number) {
  return {
    detail: makePatientRegistryDetail({ patient_id: patientId }),
    records: makePatientRegistryRecordsResponse({
      items: [
        {
          record_id: patientId,
          patient_id: patientId,
          asset_id: patientId + 100,
          record_type: "report",
          summary_text: `summary ${patientId}`,
          source: "registry",
          created_at: "2026-04-16T00:00:00Z",
        },
      ],
    }),
    alerts: makePatientRegistryAlertsResponse({
      items: [
        {
          kind: "info",
          message: `alert ${patientId}`,
          patient_id: patientId,
        },
      ],
    }),
  };
}

describe("useRegistryBrowser", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("keeps the newest preview response when earlier requests resolve later", async () => {
    const first = {
      detail: createDeferred<PatientRegistryDetail>(),
      records: createDeferred<PatientRegistryRecordsResponse>(),
      alerts: createDeferred<PatientRegistryAlertsResponse>(),
    };
    const second = {
      detail: createDeferred<PatientRegistryDetail>(),
      records: createDeferred<PatientRegistryRecordsResponse>(),
      alerts: createDeferred<PatientRegistryAlertsResponse>(),
    };
    const apiClient = buildApiClientStub({
      getPatientRegistryDetail: vi.fn((patientId: number) => (patientId === 1 ? first.detail.promise : second.detail.promise)),
      getPatientRecords: vi.fn((patientId: number) => (patientId === 1 ? first.records.promise : second.records.promise)),
      getPatientRegistryAlerts: vi.fn((patientId: number) => (patientId === 1 ? first.alerts.promise : second.alerts.promise)),
    });
    const { result } = renderBrowser(apiClient);

    act(() => {
      void result.current.previewPatient(1);
      void result.current.previewPatient(2);
    });

    await act(async () => {
      second.detail.resolve(makePatientResponses(2).detail);
      second.records.resolve(makePatientResponses(2).records);
      second.alerts.resolve(makePatientResponses(2).alerts);
      await Promise.all([second.detail.promise, second.records.promise, second.alerts.promise]);
    });

    await waitFor(() => expect(result.current.previewPatientId).toBe(2));
    expect(result.current.previewDetail?.patient_id).toBe(2);
    expect(result.current.previewRecords).toHaveLength(1);
    expect(result.current.previewAlerts).toHaveLength(1);

    await act(async () => {
      first.detail.resolve(makePatientResponses(1).detail);
      first.records.resolve(makePatientResponses(1).records);
      first.alerts.resolve(makePatientResponses(1).alerts);
      await Promise.all([first.detail.promise, first.records.promise, first.alerts.promise]);
    });

    expect(result.current.previewPatientId).toBe(2);
    expect(result.current.previewDetail?.patient_id).toBe(2);
    expect(result.current.previewRecords[0]?.patient_id).toBe(2);
    expect(result.current.previewAlerts[0]?.patient_id).toBe(2);
    await waitFor(() => expect(result.current.isLoadingPreview).toBe(false));
  });

  it("clears the deleted preview only when it is still current", async () => {
    const preview = {
      detail: createDeferred<PatientRegistryDetail>(),
      records: createDeferred<PatientRegistryRecordsResponse>(),
      alerts: createDeferred<PatientRegistryAlertsResponse>(),
    };
    const replacement = {
      detail: createDeferred<PatientRegistryDetail>(),
      records: createDeferred<PatientRegistryRecordsResponse>(),
      alerts: createDeferred<PatientRegistryAlertsResponse>(),
    };
    const deleteDeferred = createDeferred<PatientRegistryDeleteResponse>();
    const getRecentPatients = vi.fn(async () => makePatientRegistryListResponse({ items: [] }));
    const apiClient = buildApiClientStub({
      getRecentPatients,
      getPatientRegistryDetail: vi.fn((patientId: number) => (patientId === 1 ? preview.detail.promise : replacement.detail.promise)),
      getPatientRecords: vi.fn((patientId: number) => (patientId === 1 ? preview.records.promise : replacement.records.promise)),
      getPatientRegistryAlerts: vi.fn((patientId: number) => (patientId === 1 ? preview.alerts.promise : replacement.alerts.promise)),
      deletePatientRegistryPatient: vi.fn(() => deleteDeferred.promise),
    });
    const { result } = renderBrowser(apiClient);

    act(() => {
      void result.current.previewPatient(1);
    });

    await act(async () => {
      preview.detail.resolve(makePatientResponses(1).detail);
      preview.records.resolve(makePatientResponses(1).records);
      preview.alerts.resolve(makePatientResponses(1).alerts);
      await Promise.all([preview.detail.promise, preview.records.promise, preview.alerts.promise]);
    });

    act(() => {
      void result.current.deletePatient(1);
    });

    await act(async () => {
      void result.current.previewPatient(2);
      replacement.detail.resolve(makePatientResponses(2).detail);
      replacement.records.resolve(makePatientResponses(2).records);
      replacement.alerts.resolve(makePatientResponses(2).alerts);
      await Promise.all([replacement.detail.promise, replacement.records.promise, replacement.alerts.promise]);
    });

    await act(async () => {
      deleteDeferred.resolve({
        patient_id: 1,
        deleted_records: 1,
        deleted_assets: 1,
        deleted_asset_paths: [],
        record_ids: [],
      });
      await deleteDeferred.promise;
    });

    await waitFor(() => expect(result.current.isDeletingPatient).toBe(false));
    expect(result.current.previewPatientId).toBe(2);
    expect(result.current.previewDetail?.patient_id).toBe(2);
    expect(result.current.previewRecords[0]?.patient_id).toBe(2);
    expect(result.current.previewAlerts[0]?.patient_id).toBe(2);
    expect(getRecentPatients).toHaveBeenCalled();
  });

  it("omits patient_id from search requests when the patientId field is blank", async () => {
    const searchPatientRegistry = vi.fn(async () => makePatientRegistryListResponse());
    const apiClient = buildApiClientStub({ searchPatientRegistry });
    const { result } = renderBrowser(apiClient);

    act(() => {
      result.current.setSearchField("patientId", "");
      void result.current.runSearch();
    });

    await waitFor(() => expect(searchPatientRegistry).toHaveBeenCalledTimes(1));
    expect(searchPatientRegistry).toHaveBeenCalledWith({
      patient_id: null,
      tumor_location: null,
      mmr_status: null,
      clinical_stage: null,
      limit: 20,
    });
  });
});
