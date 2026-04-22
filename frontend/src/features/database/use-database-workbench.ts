import { useEffect, useRef, useState } from "react";

import { ApiClientError } from "../../app/api/client";
import { useApiClient } from "../../app/providers";
import type {
  DatabaseCaseDetailResponse,
  DatabaseFilters,
  DatabaseNumericStatistics,
  DatabaseSearchRequest,
  DatabaseSearchResponse,
  DatabaseSortField,
  DatabaseStatsResponse,
  JsonObject,
} from "../../app/api/types";

export interface UseDatabaseWorkbenchOptions {
  /**
   * Legacy compatibility shim.
   * The historical-only workbench ignores this value, but the doctor page still
   * passes it while the wider scene refactor is in progress.
   */
  scope?: DatabaseWorkbenchScope;
  autoBootstrap?: boolean;
  bootstrapKey?: string;
  bootstrapRequest?: DatabaseSearchRequest | null;
  initialNaturalQuery?: string;
  initialSelectedPatientId?: number | null;
}

export type DatabaseWorkbenchScope = "historical_case_base" | "patient_registry";

const NUMERIC_RECORD_FIELDS = new Set([
  "patient_id",
  "age",
  "ecog",
  "ecog_score",
  "cea",
  "cea_level",
]);
const BOOLEAN_RECORD_FIELDS = new Set(["family_history", "biopsy_confirmed"]);

export function createDefaultFilters(): DatabaseFilters {
  return {
    patient_id: null,
    tumor_location: [],
    ct_stage: [],
    cn_stage: [],
    histology_type: [],
    mmr_status: [],
    age_min: null,
    age_max: null,
    cea_max: null,
    family_history: null,
    biopsy_confirmed: null,
    ecog_min: null,
    ecog_max: null,
  };
}

export function createDefaultSearchRequest(): DatabaseSearchRequest {
  return {
    filters: createDefaultFilters(),
    pagination: {
      page: 1,
      page_size: 20,
    },
    sort: {
      field: "patient_id",
      direction: "asc",
    },
  };
}

export function buildSearchRequestFromFilters(filters?: Partial<DatabaseFilters> | null): DatabaseSearchRequest {
  return {
    ...createDefaultSearchRequest(),
    filters: normalizeIntentFilters(filters ?? {}),
  };
}

function cloneSearchRequest(request: DatabaseSearchRequest): DatabaseSearchRequest {
  return {
    ...request,
    filters: {
      ...createDefaultFilters(),
      ...request.filters,
      tumor_location: [...(request.filters.tumor_location ?? [])],
      ct_stage: [...(request.filters.ct_stage ?? [])],
      cn_stage: [...(request.filters.cn_stage ?? [])],
      histology_type: [...(request.filters.histology_type ?? [])],
      mmr_status: [...(request.filters.mmr_status ?? [])],
    },
    pagination: { ...request.pagination },
    sort: { ...request.sort },
  };
}

export function readDatabaseWorkbenchError(error: unknown): string {
  if (error instanceof ApiClientError) {
    return error.message;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Database workbench request failed.";
}

export function numericSummary(stats?: DatabaseNumericStatistics | null): string {
  if (!stats) {
    return "-";
  }

  const empty = [stats.min, stats.max, stats.mean].every((value) => value === null || value === undefined);
  if (empty) {
    return "-";
  }

  const min = stats.min ?? "-";
  const max = stats.max ?? "-";
  const mean = typeof stats.mean === "number" ? stats.mean.toFixed(1) : "-";
  return `${min} / ${max} / ${mean}`;
}

function normalizeOptionalNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function normalizeOptionalBoolean(value: unknown): boolean | null {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    if (value === 1) {
      return true;
    }
    if (value === 0) {
      return false;
    }
    return null;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["true", "1", "yes", "y", "是"].includes(normalized)) {
      return true;
    }
    if (["false", "0", "no", "n", "否"].includes(normalized)) {
      return false;
    }
  }
  return null;
}

function normalizeRiskFactors(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value !== "string") {
    return [];
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return [];
  }

  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch {
    // Fall through to delimiter split.
  }

  return trimmed
    .split(/[,;\n]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function normalizeIntentFilters(filters: Partial<DatabaseFilters>): DatabaseFilters {
  return {
    ...createDefaultFilters(),
    ...filters,
    tumor_location: Array.isArray(filters.tumor_location) ? filters.tumor_location : [],
    ct_stage: Array.isArray(filters.ct_stage) ? filters.ct_stage : [],
    cn_stage: Array.isArray(filters.cn_stage) ? filters.cn_stage : [],
    histology_type: Array.isArray(filters.histology_type) ? filters.histology_type : [],
    mmr_status: Array.isArray(filters.mmr_status) ? filters.mmr_status : [],
    patient_id: normalizeOptionalNumber(filters.patient_id),
    age_min: normalizeOptionalNumber(filters.age_min),
    age_max: normalizeOptionalNumber(filters.age_max),
    cea_max: normalizeOptionalNumber(filters.cea_max),
    family_history: normalizeOptionalBoolean(filters.family_history),
    biopsy_confirmed: normalizeOptionalBoolean(filters.biopsy_confirmed),
    ecog_min: normalizeOptionalNumber(filters.ecog_min),
    ecog_max: normalizeOptionalNumber(filters.ecog_max),
  };
}

function normalizeRecordValue(field: string, value: unknown): unknown {
  if (field === "risk_factors") {
    return normalizeRiskFactors(value);
  }

  if (BOOLEAN_RECORD_FIELDS.has(field)) {
    return normalizeOptionalBoolean(value);
  }

  if (NUMERIC_RECORD_FIELDS.has(field)) {
    const normalized = normalizeOptionalNumber(value);
    return normalized ?? value;
  }

  if (typeof value !== "string") {
    return value;
  }

  return value.trim() ? value : "";
}

export function normalizeRecordForUpsert(record: Record<string, unknown>): JsonObject {
  const normalized: Record<string, unknown> = {};

  for (const [field, value] of Object.entries(record)) {
    normalized[field] = normalizeRecordValue(field, value);
  }

  return normalized as JsonObject;
}

export function readPatientId(detail: DatabaseCaseDetailResponse | null): number | null {
  if (!detail) {
    return null;
  }

  const raw = detail.case_record?.patient_id ?? detail.patient_id;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

export function topDistributionEntry(entries?: Record<string, number> | null): string {
  const normalizedEntries = entries ?? {};
  const [first] = Object.entries(normalizedEntries).sort((left, right) => right[1] - left[1]);
  if (!first) {
    return "-";
  }

  return `${first[0]} (${first[1]})`;
}

function containsPatient(response: DatabaseSearchResponse, patientId: number | null): boolean {
  return patientId !== null && response.items.some((item) => Number(item.patient_id) === patientId);
}

export function useDatabaseWorkbench(options: UseDatabaseWorkbenchOptions = {}) {
  const {
    autoBootstrap = true,
    bootstrapKey = "default",
    bootstrapRequest = null,
    initialNaturalQuery = "",
    initialSelectedPatientId = null,
  } = options;

  const apiClient = useApiClient();
  const [stats, setStats] = useState<DatabaseStatsResponse | null>(null);
  const [searchRequest, setSearchRequest] = useState<DatabaseSearchRequest>(() =>
    cloneSearchRequest(bootstrapRequest ?? createDefaultSearchRequest()),
  );
  const [searchResponse, setSearchResponse] = useState<DatabaseSearchResponse | null>(null);
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(initialSelectedPatientId);
  const [detail, setDetail] = useState<DatabaseCaseDetailResponse | null>(null);
  const [editRecord, setEditRecord] = useState<Record<string, unknown> | null>(null);
  const [naturalQuery, setNaturalQuery] = useState(initialNaturalQuery);
  const [intentWarnings, setIntentWarnings] = useState<string[]>([]);
  const [unsupportedTerms, setUnsupportedTerms] = useState<string[]>([]);
  const [pageError, setPageError] = useState<string | null>(null);
  const [isBootstrapping, setIsBootstrapping] = useState(autoBootstrap);
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [isParsing, setIsParsing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const bootstrapRequestRef = useRef<DatabaseSearchRequest | null>(bootstrapRequest);
  const detailRequestIdRef = useRef(0);

  useEffect(() => {
    bootstrapRequestRef.current = bootstrapRequest;
  }, [bootstrapRequest]);

  async function runSearch(nextRequest: DatabaseSearchRequest) {
    setIsSearching(true);
    setPageError(null);

    try {
      const resolvedRequest = cloneSearchRequest(nextRequest);
      const response = await apiClient.searchDatabaseCases(resolvedRequest);
      setSearchRequest(resolvedRequest);
      setSearchResponse(response);

      if (selectedPatientId !== null && !containsPatient(response, selectedPatientId)) {
        setSelectedPatientId(null);
        setDetail(null);
        setEditRecord(null);
      }

      return response;
    } catch (error) {
      setPageError(readDatabaseWorkbenchError(error));
      return null;
    } finally {
      setIsSearching(false);
    }
  }

  async function loadCaseDetail(patientId: number) {
    const requestId = detailRequestIdRef.current + 1;
    detailRequestIdRef.current = requestId;
    setIsLoadingDetail(true);
    setPageError(null);

    try {
      const response = await apiClient.getDatabaseCaseDetail(patientId);
      if (detailRequestIdRef.current !== requestId) {
        return null;
      }
      setSelectedPatientId(patientId);
      setDetail(response);
      setEditRecord(response.case_record ? { ...response.case_record } : null);
      return response;
    } catch (error) {
      if (detailRequestIdRef.current !== requestId) {
        return null;
      }
      setPageError(readDatabaseWorkbenchError(error));
      return null;
    } finally {
      if (detailRequestIdRef.current === requestId) {
        setIsLoadingDetail(false);
      }
    }
  }

  useEffect(() => {
    if (!autoBootstrap) {
      setIsBootstrapping(false);
      return;
    }

    let cancelled = false;

    const bootstrap = async () => {
      const initialRequest = cloneSearchRequest(bootstrapRequestRef.current ?? createDefaultSearchRequest());
      const initialPatientId = initialSelectedPatientId ?? null;

      setIsBootstrapping(true);
      setPageError(null);
      setIntentWarnings([]);
      setUnsupportedTerms([]);
      setNaturalQuery(initialNaturalQuery);
      setSelectedPatientId(initialPatientId);
      setDetail(null);
      setEditRecord(null);

      try {
        const [statsResponse, casesResponse] = await Promise.all([
          apiClient.getDatabaseStats(),
          apiClient.searchDatabaseCases(initialRequest),
        ]);

        if (cancelled) {
          return;
        }

        setStats(statsResponse);
        setSearchRequest(initialRequest);
        setSearchResponse(casesResponse);

        if (initialPatientId !== null) {
          const detailResponse = await apiClient.getDatabaseCaseDetail(initialPatientId);
          if (cancelled) {
            return;
          }
          setDetail(detailResponse);
          setEditRecord(detailResponse.case_record ? { ...detailResponse.case_record } : null);
        }
      } catch (error) {
        if (!cancelled) {
          setPageError(readDatabaseWorkbenchError(error));
        }
      } finally {
        if (!cancelled) {
          setIsBootstrapping(false);
        }
      }
    };

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, [apiClient, autoBootstrap, bootstrapKey, initialNaturalQuery, initialSelectedPatientId]);

  async function handleNaturalQuerySubmit() {
    if (!naturalQuery.trim()) {
      return;
    }

    setIsParsing(true);
    setPageError(null);

    try {
      const response = await apiClient.parseDatabaseQueryIntent(naturalQuery.trim());
      const nextRequest: DatabaseSearchRequest = {
        ...searchRequest,
        filters: normalizeIntentFilters(response.filters),
        pagination: {
          ...searchRequest.pagination,
          page: 1,
        },
      };

      setIntentWarnings(response.warnings);
      setUnsupportedTerms(response.unsupported_terms);
      await runSearch(nextRequest);
    } catch (error) {
      setPageError(readDatabaseWorkbenchError(error));
    } finally {
      setIsParsing(false);
    }
  }

  async function saveRecord() {
    if (!editRecord) {
      return;
    }

    const patientId = readPatientId(detail);
    if (patientId === null) {
      setPageError("Patient ID is required to save a record.");
      return;
    }

    setIsSaving(true);
    setPageError(null);

    try {
      const response = await apiClient.upsertDatabaseCase({
        record: normalizeRecordForUpsert({
          ...(detail?.case_record ?? {}),
          ...editRecord,
          patient_id: patientId,
        }),
      });

      setDetail(response);
      setSelectedPatientId(patientId);
      setEditRecord(response.case_record ? { ...response.case_record } : null);

      const [statsResponse, casesResponse] = await Promise.all([
        apiClient.getDatabaseStats(),
        apiClient.searchDatabaseCases(searchRequest),
      ]);

      setStats(statsResponse);
      setSearchResponse(casesResponse);

      if (!containsPatient(casesResponse, patientId)) {
        setSelectedPatientId(null);
        setDetail(null);
        setEditRecord(null);
      }
    } catch (error) {
      setPageError(readDatabaseWorkbenchError(error));
    } finally {
      setIsSaving(false);
    }
  }

  function setFilters(nextFilters: DatabaseFilters) {
    setSearchRequest((current) => ({
      ...current,
      filters: nextFilters,
    }));
  }

  function resetWorkbench() {
    setIntentWarnings([]);
    setUnsupportedTerms([]);
    setNaturalQuery("");
    setSelectedPatientId(null);
    setDetail(null);
    setEditRecord(null);
    void runSearch(createDefaultSearchRequest());
  }

  function setEditField(field: string, value: unknown) {
    setEditRecord((current) => ({
      ...(current ?? {}),
      [field]: value,
    }));
  }

  function handleSortChange(field: DatabaseSortField) {
    const direction =
      searchRequest.sort.field === field && searchRequest.sort.direction === "asc" ? "desc" : "asc";

    void runSearch({
      ...searchRequest,
      sort: {
        field,
        direction,
      },
    });
  }

  function handlePageChange(page: number) {
    if (page === searchRequest.pagination.page) {
      return;
    }

    void runSearch({
      ...searchRequest,
      pagination: {
        ...searchRequest.pagination,
        page,
      },
    });
  }

  return {
    stats,
    searchRequest,
    searchResponse,
    selectedPatientId,
    detail,
    editRecord,
    naturalQuery,
    intentWarnings,
    unsupportedTerms,
    pageError,
    isBootstrapping,
    isSearching,
    isLoadingDetail,
    isParsing,
    isSaving,
    setNaturalQuery,
    setFilters,
    runSearch,
    loadCaseDetail,
    handleNaturalQuerySubmit,
    saveRecord,
    resetWorkbench,
    setEditField,
    handleSortChange,
    handlePageChange,
  };
}
