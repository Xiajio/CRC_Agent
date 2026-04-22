import "@testing-library/jest-dom/vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../../app/providers";
import type {
  PatientRegistryAlertsResponse,
  PatientRegistryDetail,
  PatientRegistryRecordsResponse,
} from "../../app/api/types";
import {
  buildApiClientStub,
  makePatientRegistryAlertsResponse,
  makePatientRegistryDetail,
  makePatientRegistryRecordsResponse,
} from "../../test/test-utils";
import { usePatientRegistry } from "./use-patient-registry";

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;

  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
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

function renderPatientRegistry(
  apiClient = buildApiClientStub(),
  initialProps = { enabled: true, currentPatientId: 1 as number | null },
) {
  return renderHook((props: { enabled: boolean; currentPatientId: number | null }) => usePatientRegistry(props), {
    initialProps,
    wrapper: ({ children }) => <AppProviders apiClient={apiClient}>{children}</AppProviders>,
  });
}

describe("usePatientRegistry", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("clears bound patient detail immediately when currentPatientId changes", async () => {
    const first = makePatientResponses(1);
    const second = {
      detail: createDeferred<PatientRegistryDetail>(),
      records: createDeferred<PatientRegistryRecordsResponse>(),
      alerts: createDeferred<PatientRegistryAlertsResponse>(),
    };
    const apiClient = buildApiClientStub({
      getPatientRegistryDetail: vi.fn((patientId: number) =>
        patientId === 1 ? Promise.resolve(first.detail) : second.detail.promise,
      ),
      getPatientRecords: vi.fn((patientId: number) =>
        patientId === 1 ? Promise.resolve(first.records) : second.records.promise,
      ),
      getPatientRegistryAlerts: vi.fn((patientId: number) =>
        patientId === 1 ? Promise.resolve(first.alerts) : second.alerts.promise,
      ),
    });
    const { result, rerender } = renderPatientRegistry(apiClient);

    await waitFor(() => expect(result.current.boundPatientDetail?.patient_id).toBe(1));
    expect(result.current.boundPatientRecords[0]?.patient_id).toBe(1);
    expect(result.current.boundPatientAlerts[0]?.patient_id).toBe(1);

    rerender({ enabled: true, currentPatientId: 2 });

    expect(result.current.boundPatientDetail).toBeNull();
    expect(result.current.boundPatientRecords).toEqual([]);
    expect(result.current.boundPatientAlerts).toEqual([]);

    await act(async () => {
      second.detail.resolve(makePatientResponses(2).detail);
      second.records.resolve(makePatientResponses(2).records);
      second.alerts.resolve(makePatientResponses(2).alerts);
      await Promise.all([second.detail.promise, second.records.promise, second.alerts.promise]);
    });

    await waitFor(() => expect(result.current.boundPatientDetail?.patient_id).toBe(2));
  });

  it("ignores stale bound patient responses that resolve after a newer patient load", async () => {
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
    const { result, rerender } = renderPatientRegistry(apiClient);

    rerender({ enabled: true, currentPatientId: 2 });

    await act(async () => {
      second.detail.resolve(makePatientResponses(2).detail);
      second.records.resolve(makePatientResponses(2).records);
      second.alerts.resolve(makePatientResponses(2).alerts);
      await Promise.all([second.detail.promise, second.records.promise, second.alerts.promise]);
    });

    await waitFor(() => expect(result.current.boundPatientDetail?.patient_id).toBe(2));

    await act(async () => {
      first.detail.resolve(makePatientResponses(1).detail);
      first.records.resolve(makePatientResponses(1).records);
      first.alerts.resolve(makePatientResponses(1).alerts);
      await Promise.all([first.detail.promise, first.records.promise, first.alerts.promise]);
    });

    expect(result.current.boundPatientDetail?.patient_id).toBe(2);
    expect(result.current.boundPatientRecords[0]?.patient_id).toBe(2);
    expect(result.current.boundPatientAlerts[0]?.patient_id).toBe(2);
  });
});
