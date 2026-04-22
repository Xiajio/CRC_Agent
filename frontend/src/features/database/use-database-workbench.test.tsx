import "@testing-library/jest-dom/vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../../app/providers";
import type { DatabaseCaseDetailResponse } from "../../app/api/types";
import { buildApiClientStub } from "../../test/test-utils";
import { useDatabaseWorkbench } from "./use-database-workbench";

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;

  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

function makeDetailResponse(patientId: number): DatabaseCaseDetailResponse {
  return {
    patient_id: String(patientId),
    case_record: {
      patient_id: patientId,
      clinical_stage: `cT${patientId}N0M0`,
    },
    available_data: {
      case_info: true,
      imaging: false,
      pathology_slides: false,
    },
    cards: {},
  };
}

function renderWorkbench(apiClient = buildApiClientStub()) {
  return renderHook(() => useDatabaseWorkbench({ autoBootstrap: false }), {
    wrapper: ({ children }) => <AppProviders apiClient={apiClient}>{children}</AppProviders>,
  });
}

describe("useDatabaseWorkbench", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("keeps the newest detail response when earlier requests resolve later", async () => {
    const first = createDeferred<DatabaseCaseDetailResponse>();
    const second = createDeferred<DatabaseCaseDetailResponse>();
    const getDatabaseCaseDetail = vi.fn((patientId: number) =>
      patientId === 1 ? first.promise : second.promise,
    );
    const apiClient = buildApiClientStub({ getDatabaseCaseDetail });
    const { result } = renderWorkbench(apiClient);

    act(() => {
      void result.current.loadCaseDetail(1);
      void result.current.loadCaseDetail(2);
    });

    await act(async () => {
      second.resolve(makeDetailResponse(2));
      await second.promise;
    });

    await waitFor(() => expect(result.current.selectedPatientId).toBe(2));
    expect(result.current.detail?.patient_id).toBe("2");
    expect(result.current.editRecord?.patient_id).toBe(2);

    await act(async () => {
      first.resolve(makeDetailResponse(1));
      await first.promise;
    });

    expect(result.current.selectedPatientId).toBe(2);
    expect(result.current.detail?.patient_id).toBe("2");
    expect(result.current.editRecord?.patient_id).toBe(2);
    await waitFor(() => expect(result.current.isLoadingDetail).toBe(false));
  });
});
