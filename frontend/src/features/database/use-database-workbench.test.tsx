import "@testing-library/jest-dom/vitest";
import { renderHook, waitFor, act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { AppProviders } from "../../app/providers";
import type { DatabaseCaseDetailResponse } from "../../app/api/types";
import { buildApiClientStub } from "../../test/test-utils";
import {
  findMissingRequiredFields,
  normalizeRecordForUpsert,
  useDatabaseWorkbench,
} from "./use-database-workbench";

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
    patient_id: patientId,
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
    expect(result.current.detail?.patient_id).toBe(2);
    expect(result.current.editRecord?.patient_id).toBe(2);

    await act(async () => {
      first.resolve(makeDetailResponse(1));
      await first.promise;
    });

    expect(result.current.selectedPatientId).toBe(2);
    expect(result.current.detail?.patient_id).toBe(2);
    expect(result.current.editRecord?.patient_id).toBe(2);
    await waitFor(() => expect(result.current.isLoadingDetail).toBe(false));
  });
});

describe("normalizeRecordForUpsert", () => {
  it("forwards null for numeric fields when input cannot be parsed", () => {
    const normalized = normalizeRecordForUpsert({
      patient_id: 7,
      age: "",
      cea_level: "abc",
      ecog_score: null,
    });

    expect(normalized.age).toBeNull();
    expect(normalized.cea_level).toBeNull();
    expect(normalized.ecog_score).toBeNull();
  });

  it("clears boolean fields when input is empty or null", () => {
    const normalized = normalizeRecordForUpsert({
      patient_id: 7,
      family_history: "",
      biopsy_confirmed: null,
    });

    expect(normalized.family_history).toBeNull();
    expect(normalized.biopsy_confirmed).toBeNull();
  });

  it("clears risk_factors to empty array when input is empty", () => {
    const normalized = normalizeRecordForUpsert({
      patient_id: 7,
      risk_factors: "",
    });

    expect(normalized.risk_factors).toEqual([]);
  });
});

describe("findMissingRequiredFields", () => {
  it("reports every missing required field", () => {
    expect(
      findMissingRequiredFields({
        patient_id: 7,
        gender: null,
        age: null,
        histology_type: "",
      }),
    ).toEqual(expect.arrayContaining(["gender", "age", "histology_type"]));
  });

  it("accepts numeric zero and non-empty strings", () => {
    expect(
      findMissingRequiredFields({
        patient_id: 7,
        gender: 1,
        age: 0,
        histology_type: "adenocarcinoma",
        tumor_location: "rectum",
        ct_stage: "3",
        cn_stage: "1",
        clinical_stage: "III",
        cea_level: 0,
        mmr_status: 1,
      }),
    ).toEqual([]);
  });
});

describe("useDatabaseWorkbench.saveRecord", () => {
  function makeDetail(patientId: number): DatabaseCaseDetailResponse {
    return {
      patient_id: patientId,
      case_record: {
        patient_id: patientId,
        gender: 1,
        age: 55,
        histology_type: "adenocarcinoma",
        tumor_location: "rectum",
        ct_stage: "3",
        cn_stage: "1",
        clinical_stage: "III",
        cea_level: 5.5,
        mmr_status: 1,
      },
      available_data: { case_info: true, imaging: false, pathology_slides: false },
      cards: {},
    };
  }

  it("calls upsert with mode=full when all required fields are present", async () => {
    const detail = makeDetail(7);
    const getDatabaseCaseDetail = vi.fn(async () => detail);
    const upsertDatabaseCase = vi.fn(async () => detail);
    const apiClient = buildApiClientStub({ getDatabaseCaseDetail, upsertDatabaseCase });
    const { result } = renderWorkbench(apiClient);

    await act(async () => {
      await result.current.loadCaseDetail(7);
    });
    await waitFor(() => expect(result.current.detail?.patient_id).toBe(7));

    await act(async () => {
      await result.current.saveRecord();
    });

    expect(upsertDatabaseCase).toHaveBeenCalledTimes(1);
    const payload = upsertDatabaseCase.mock.calls[0][0];
    expect(payload.mode).toBe("full");
    expect(payload.record).toMatchObject({ patient_id: 7, age: 55, cea_level: 5.5 });
  });

  it("blocks the upsert request and surfaces an error when required fields are missing", async () => {
    const detail = makeDetail(7);
    const getDatabaseCaseDetail = vi.fn(async () => detail);
    const upsertDatabaseCase = vi.fn(async () => detail);
    const apiClient = buildApiClientStub({ getDatabaseCaseDetail, upsertDatabaseCase });
    const { result } = renderWorkbench(apiClient);

    await act(async () => {
      await result.current.loadCaseDetail(7);
    });
    await waitFor(() => expect(result.current.detail?.patient_id).toBe(7));

    act(() => {
      result.current.setEditField("age", "");
    });
    act(() => {
      result.current.setEditField("cea_level", "");
    });

    await act(async () => {
      await result.current.saveRecord();
    });

    expect(upsertDatabaseCase).not.toHaveBeenCalled();
    expect(result.current.pageError ?? "").toContain("age");
    expect(result.current.pageError ?? "").toContain("cea_level");
  });
});
