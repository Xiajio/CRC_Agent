import { useEffect, useRef, useState, type ReactNode } from "react";

import { ApiClientError, type ApiClient } from "../../app/api/client";
import type {
  AdminCohortFeasibilityRequest,
  AdminCohortFeasibilityResponse,
  AdminCreateAutoResearchRunRequest,
  AdminCreateLearningJobRequest,
  AdminCancelReleaseIntentRequest,
  AdminCreateReleaseIntentRequest,
  AdminExecuteReleaseRequest,
  AdminAcknowledgeReleaseMonitoringAlertRequest,
  AdminRecordReleaseClosureRequest,
  AdminRecordReleaseMonitoringCheckRequest,
  AdminRecordReleaseApprovalRequest,
  AdminRecordReleaseRollbackPlanRequest,
  AdminReleaseClosureResponse,
  AdminReleaseDashboardResponse,
  AdminReleaseExecutionResponse,
  AdminReleaseGovernanceResponse,
  AdminReleaseMonitoringResponse,
  AdminRulesResponse,
  AdminLearningJobsResponse,
  AdminAutoResearchRunResponse,
  AdminAutoResearchRunsResponse,
  AdminToolManifestResponse,
} from "../../app/api/types";
import type { Scene, SessionState } from "../../app/api/types";
import { ClinicalTopNav } from "../../components/layout/clinical-top-nav";
import { classNames } from "../../components/ui";
import {
  ADMIN_NAV_ITEMS,
  AGENT_ADMIN_TASKS,
  type AgentAdminTaskId,
} from "./agent-admin-model";
import { AgentAdminTaskPages } from "./agent-admin-pages";

function activeRuntime(patient: SessionState, doctor: SessionState): string {
  return doctor.runtime?.runner_mode ?? patient.runtime?.runner_mode ?? "unknown";
}

type AgentAdminViewProps = {
  activeScene: Scene;
  patient: SessionState;
  doctor: SessionState;
  surfaceSwitcher: ReactNode;
  apiClient?: Partial<
    Pick<
      ApiClient,
      | "getAdminTools"
      | "getAdminRules"
      | "getAdminLearningJobs"
      | "createAdminLearningJob"
      | "evaluateAdminCohortFeasibility"
      | "getAdminAutoResearchRuns"
      | "createAdminAutoResearchRun"
      | "getAdminAutoResearchRun"
      | "getAdminReleaseDashboard"
      | "getAdminReleaseGovernance"
      | "getAdminReleaseExecution"
      | "getAdminReleaseMonitoring"
      | "getAdminReleaseClosure"
      | "createAdminReleaseIntent"
      | "recordAdminReleaseApproval"
      | "recordAdminReleaseRollbackPlan"
      | "cancelAdminReleaseIntent"
      | "executeAdminRelease"
      | "executeAdminReleaseRollback"
      | "recordAdminReleaseMonitoringCheck"
      | "acknowledgeAdminReleaseMonitoringAlert"
      | "recordAdminReleaseClosure"
    >
  >;
};

export type AgentAdminToolsResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminToolManifestResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminRulesResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminRulesResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminLearningJobsResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminLearningJobsResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminCohortFeasibilityResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminCohortFeasibilityResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminAutoResearchRunsResource =
  | { status: "idle" }
  | { status: "loading" }
  | {
      status: "success";
      data: AdminAutoResearchRunsResponse;
      refreshing?: boolean;
      refreshError?: { status?: number; message: string };
    }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminAutoResearchRunResource =
  | { status: "idle" }
  | { status: "loading"; runId: string }
  | {
      status: "success";
      data: AdminAutoResearchRunResponse;
      refreshing?: boolean;
      refreshError?: { status?: number; message: string };
    }
  | { status: "error"; runId: string; error: { status?: number; message: string } };

export type AgentAdminReleaseDashboardResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminReleaseDashboardResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminReleaseGovernanceResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminReleaseGovernanceResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminReleaseExecutionResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminReleaseExecutionResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminReleaseMonitoringResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminReleaseMonitoringResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminReleaseClosureResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminReleaseClosureResponse }
  | { status: "error"; error: { status?: number; message: string } };

export type AgentAdminReleaseGovernanceActionState =
  | { status: "idle" }
  | { status: "running"; label: string }
  | { status: "error"; message: string };

export type AgentAdminReleaseExecutionActionState =
  | { status: "idle" }
  | { status: "running"; label: string }
  | { status: "error"; message: string };

export type AgentAdminReleaseMonitoringActionState =
  | { status: "idle" }
  | { status: "running"; label: string }
  | { status: "error"; message: string };

export type AgentAdminReleaseClosureActionState =
  | { status: "idle" }
  | { status: "running"; label: string }
  | { status: "error"; message: string };

export type AgentAdminResearchActionState =
  | { status: "idle" }
  | { status: "running"; label: string }
  | { status: "success"; message: string }
  | { status: "warning"; message: string }
  | { status: "error"; message: string };

export type AgentAdminAutoResearchActionState = AgentAdminResearchActionState;

export type AgentAdminReleaseGovernanceActions = {
  createIntent: (request: AdminCreateReleaseIntentRequest) => Promise<void>;
  recordApproval: (intentId: string, request: AdminRecordReleaseApprovalRequest) => Promise<void>;
  recordRollbackPlan: (intentId: string, request: AdminRecordReleaseRollbackPlanRequest) => Promise<void>;
  cancelIntent: (intentId: string, request: AdminCancelReleaseIntentRequest) => Promise<void>;
};

export type AgentAdminReleaseExecutionActions = {
  executeRelease: (request: AdminExecuteReleaseRequest) => Promise<void>;
  executeRollback: (request: AdminExecuteReleaseRequest) => Promise<void>;
};

export type AgentAdminReleaseMonitoringActions = {
  recordCheck: (request: AdminRecordReleaseMonitoringCheckRequest) => Promise<void>;
  acknowledgeAlert: (alertId: string, request: AdminAcknowledgeReleaseMonitoringAlertRequest) => Promise<void>;
};

export type AgentAdminReleaseClosureActions = {
  recordReleaseClosure: (request: AdminRecordReleaseClosureRequest) => Promise<void>;
};

export type AgentAdminResearchActions = {
  refresh: () => Promise<void>;
  createLearningJob: (request: AdminCreateLearningJobRequest) => Promise<void>;
  evaluateCohortFeasibility: (request: AdminCohortFeasibilityRequest) => Promise<void>;
};

export type AgentAdminAutoResearchActions = {
  refreshRuns: () => Promise<void>;
  selectRun: (runId: string) => Promise<void>;
  refreshRun: () => Promise<void>;
  createRun: (request: AdminCreateAutoResearchRunRequest) => Promise<void>;
};

type ResearchRefreshOutcome =
  | { status: "success" }
  | { status: "error"; message: string }
  | { status: "stale" };

function apiErrorDetails(error: unknown, fallbackMessage: string): { status?: number; message: string } {
  if (error instanceof ApiClientError) {
    return { status: error.status, message: error.message };
  }

  return {
    message: error instanceof Error ? error.message : fallbackMessage,
  };
}

function researchValidationDetail(value: unknown): string | null {
  const detail =
    value && typeof value === "object" && !Array.isArray(value) && "detail" in value
      ? (value as { detail: unknown }).detail
      : value;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (!Array.isArray(detail)) {
    return null;
  }

  const messages = detail.flatMap((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      return typeof entry === "string" && entry.trim() ? [entry] : [];
    }

    const item = entry as { loc?: unknown; msg?: unknown };
    const message = typeof item.msg === "string" ? item.msg.trim() : "";
    if (!message) {
      return [];
    }
    const location = Array.isArray(item.loc)
      ? item.loc
          .filter((part) => part !== "body")
          .map(String)
          .join(".")
      : "";
    return [location ? `${location}: ${message}` : message];
  });

  return messages.length > 0 ? messages.join("; ") : null;
}

function researchApiErrorMessage(error: unknown, fallbackMessage: string): string {
  if (error instanceof ApiClientError) {
    const validationMessage = researchValidationDetail(error.detail);
    if (validationMessage) {
      return validationMessage;
    }
    return error.message && !error.message.includes("[object Object]")
      ? error.message
      : fallbackMessage;
  }
  return error instanceof Error && error.message ? error.message : fallbackMessage;
}

export function AgentAdminView({
  activeScene,
  patient,
  doctor,
  surfaceSwitcher,
  apiClient,
}: AgentAdminViewProps) {
  const [activeTaskId, setActiveTaskId] = useState<AgentAdminTaskId>("overview");
  const [toolsResource, setToolsResource] = useState<AgentAdminToolsResource>({ status: "idle" });
  const [rulesResource, setRulesResource] = useState<AgentAdminRulesResource>({ status: "idle" });
  const [learningJobsResource, setLearningJobsResource] = useState<AgentAdminLearningJobsResource>({ status: "idle" });
  const [cohortFeasibilityResource, setCohortFeasibilityResource] = useState<AgentAdminCohortFeasibilityResource>({ status: "idle" });
  const [autoResearchRunsResource, setAutoResearchRunsResource] = useState<AgentAdminAutoResearchRunsResource>({ status: "idle" });
  const [autoResearchRunResource, setAutoResearchRunResource] = useState<AgentAdminAutoResearchRunResource>({ status: "idle" });
  const [selectedAutoResearchRunId, setSelectedAutoResearchRunId] = useState<string | null>(null);
  const [releaseDashboardResource, setReleaseDashboardResource] = useState<AgentAdminReleaseDashboardResource>({ status: "idle" });
  const [releaseGovernanceResource, setReleaseGovernanceResource] = useState<AgentAdminReleaseGovernanceResource>({ status: "idle" });
  const [releaseExecutionResource, setReleaseExecutionResource] = useState<AgentAdminReleaseExecutionResource>({ status: "idle" });
  const [releaseMonitoringResource, setReleaseMonitoringResource] = useState<AgentAdminReleaseMonitoringResource>({ status: "idle" });
  const [releaseClosureResource, setReleaseClosureResource] = useState<AgentAdminReleaseClosureResource>({ status: "idle" });
  const [releaseGovernanceActionState, setReleaseGovernanceActionState] = useState<AgentAdminReleaseGovernanceActionState>({ status: "idle" });
  const [releaseExecutionActionState, setReleaseExecutionActionState] = useState<AgentAdminReleaseExecutionActionState>({ status: "idle" });
  const [releaseMonitoringActionState, setReleaseMonitoringActionState] = useState<AgentAdminReleaseMonitoringActionState>({ status: "idle" });
  const [releaseClosureActionState, setReleaseClosureActionState] = useState<AgentAdminReleaseClosureActionState>({ status: "idle" });
  const [researchActionState, setResearchActionState] = useState<AgentAdminResearchActionState>({ status: "idle" });
  const [autoResearchActionState, setAutoResearchActionState] = useState<AgentAdminAutoResearchActionState>({ status: "idle" });
  const learningJobsRequestSeq = useRef(0);
  const cohortFeasibilityRequestSeq = useRef(0);
  const autoResearchRunsRequestSeq = useRef(0);
  const autoResearchRunRequestSeq = useRef(0);
  const selectedAutoResearchRunIdRef = useRef<string | null>(null);
  const autoResearchActionRequestSeq = useRef(0);
  const releaseDashboardRequestSeq = useRef(0);
  const researchActionRequestSeq = useRef(0);
  const releaseMonitoringRequestSeq = useRef(0);
  const releaseClosureRequestSeq = useRef(0);
  const watchedState = activeScene === "doctor" ? doctor : patient;
  const watchedSceneLabel = activeScene === "doctor" ? "医生会话" : "患者会话";

  function selectAutoResearchRunId(runId: string | null) {
    selectedAutoResearchRunIdRef.current = runId;
    setSelectedAutoResearchRunId(runId);
  }

  useEffect(() => () => {
    learningJobsRequestSeq.current += 1;
    cohortFeasibilityRequestSeq.current += 1;
    autoResearchRunsRequestSeq.current += 1;
    autoResearchRunRequestSeq.current += 1;
    autoResearchActionRequestSeq.current += 1;
    releaseDashboardRequestSeq.current += 1;
    researchActionRequestSeq.current += 1;
  }, []);

  async function refreshLearningJobsResource(options: { setLoading?: boolean } = {}): Promise<ResearchRefreshOutcome> {
    const requestSeq = learningJobsRequestSeq.current + 1;
    learningJobsRequestSeq.current = requestSeq;

    if (!apiClient || typeof apiClient.getAdminLearningJobs !== "function") {
      const message = "LearningJob read API is unavailable";
      setLearningJobsResource({ status: "error", error: { message } });
      return { status: "error", message };
    }
    if (options.setLoading) {
      setLearningJobsResource({ status: "loading" });
    }

    try {
      const data = await apiClient.getAdminLearningJobs();
      if (learningJobsRequestSeq.current !== requestSeq) {
        return { status: "stale" };
      }
      setLearningJobsResource({ status: "success", data });
      return { status: "success" };
    } catch (error) {
      if (learningJobsRequestSeq.current !== requestSeq) {
        return { status: "stale" };
      }
      const details = apiErrorDetails(error, "Unknown admin learning jobs error");
      setLearningJobsResource({ status: "error", error: details });
      return { status: "error", message: details.message };
    }
  }

  async function refreshAutoResearchRunsResource(options: { setLoading?: boolean } = {}): Promise<ResearchRefreshOutcome> {
    const requestSeq = autoResearchRunsRequestSeq.current + 1;
    autoResearchRunsRequestSeq.current = requestSeq;

    if (!apiClient || typeof apiClient.getAdminAutoResearchRuns !== "function") {
      const message = "Auto-research runs API is unavailable";
      setAutoResearchRunsResource((current) => (
        current.status === "success"
          ? { ...current, refreshing: false, refreshError: { message } }
          : { status: "error", error: { message } }
      ));
      return { status: "error", message };
    }
    if (options.setLoading) {
      setAutoResearchRunsResource((current) => (
        current.status === "success"
          ? { ...current, refreshing: true, refreshError: undefined }
          : { status: "loading" }
      ));
    }

    try {
      const data = await apiClient.getAdminAutoResearchRuns();
      if (autoResearchRunsRequestSeq.current !== requestSeq) {
        return { status: "stale" };
      }
      setAutoResearchRunsResource({ status: "success", data });

      const currentSelectedId = selectedAutoResearchRunIdRef.current;
      const nextSelectedRun =
        data.runs.find((run) => run.run_id === currentSelectedId)
        ?? data.runs[0]
        ?? null;
      if (nextSelectedRun === null) {
        autoResearchRunRequestSeq.current += 1;
        selectAutoResearchRunId(null);
        setAutoResearchRunResource({ status: "idle" });
      } else if (nextSelectedRun.run_id !== currentSelectedId) {
        autoResearchRunRequestSeq.current += 1;
        selectAutoResearchRunId(nextSelectedRun.run_id);
        setAutoResearchRunResource({
          status: "success",
          data: {
            run: nextSelectedRun,
            integrity: data.integrity,
            runtime: data.runtime,
          },
        });
      } else {
        setAutoResearchRunResource((current) => {
          if (
            current.status === "success"
            || current.status === "loading"
            || current.status === "error"
          ) {
            return current;
          }
          return {
            status: "success",
            data: {
              run: nextSelectedRun,
              integrity: data.integrity,
              runtime: data.runtime,
            },
          };
        });
      }
      return { status: "success" };
    } catch (error) {
      if (autoResearchRunsRequestSeq.current !== requestSeq) {
        return { status: "stale" };
      }
      const details = apiErrorDetails(error, "Unknown auto-research runs error");
      setAutoResearchRunsResource((current) => (
        current.status === "success"
          ? { ...current, refreshing: false, refreshError: details }
          : { status: "error", error: details }
      ));
      return { status: "error", message: details.message };
    }
  }

  async function refreshAutoResearchRunResource(
    runId: string,
    options: { preserveData?: boolean } = {},
  ): Promise<ResearchRefreshOutcome> {
    const requestSeq = autoResearchRunRequestSeq.current + 1;
    autoResearchRunRequestSeq.current = requestSeq;

    if (!apiClient || typeof apiClient.getAdminAutoResearchRun !== "function") {
      const message = "Auto-research Run detail API is unavailable";
      setAutoResearchRunResource((current) => (
        options.preserveData
        && current.status === "success"
        && current.data.run.run_id === runId
          ? { ...current, refreshing: false, refreshError: { message } }
          : { status: "error", runId, error: { message } }
      ));
      return { status: "error", message };
    }

    setAutoResearchRunResource((current) => (
      options.preserveData
      && current.status === "success"
      && current.data.run.run_id === runId
        ? { ...current, refreshing: true, refreshError: undefined }
        : { status: "loading", runId }
    ));

    try {
      const data = await apiClient.getAdminAutoResearchRun(runId);
      if (
        autoResearchRunRequestSeq.current !== requestSeq
        || selectedAutoResearchRunIdRef.current !== runId
      ) {
        return { status: "stale" };
      }
      setAutoResearchRunResource({ status: "success", data });
      return { status: "success" };
    } catch (error) {
      if (
        autoResearchRunRequestSeq.current !== requestSeq
        || selectedAutoResearchRunIdRef.current !== runId
      ) {
        return { status: "stale" };
      }
      const details = apiErrorDetails(error, "Unknown auto-research Run detail error");
      setAutoResearchRunResource((current) => (
        options.preserveData
        && current.status === "success"
        && current.data.run.run_id === runId
          ? { ...current, refreshing: false, refreshError: details }
          : { status: "error", runId, error: details }
      ));
      return { status: "error", message: details.message };
    }
  }

  async function refreshReleaseDashboardResource(options: { setLoading?: boolean } = {}): Promise<ResearchRefreshOutcome> {
    const requestSeq = releaseDashboardRequestSeq.current + 1;
    releaseDashboardRequestSeq.current = requestSeq;

    if (!apiClient || typeof apiClient.getAdminReleaseDashboard !== "function") {
      const message = "Release dashboard read API is unavailable";
      setReleaseDashboardResource({ status: "error", error: { message } });
      return { status: "error", message };
    }
    if (options.setLoading) {
      setReleaseDashboardResource({ status: "loading" });
    }

    try {
      const data = await apiClient.getAdminReleaseDashboard();
      if (releaseDashboardRequestSeq.current !== requestSeq) {
        return { status: "stale" };
      }
      setReleaseDashboardResource({ status: "success", data });
      return { status: "success" };
    } catch (error) {
      if (releaseDashboardRequestSeq.current !== requestSeq) {
        return { status: "stale" };
      }
      const details = apiErrorDetails(error, "Unknown admin release dashboard error");
      setReleaseDashboardResource({ status: "error", error: details });
      return { status: "error", message: details.message };
    }
  }

  async function refreshReleaseMonitoringResource(options: { setLoading?: boolean } = {}) {
    if (!apiClient || typeof apiClient.getAdminReleaseMonitoring !== "function") {
      return;
    }

    const requestSeq = releaseMonitoringRequestSeq.current + 1;
    releaseMonitoringRequestSeq.current = requestSeq;
    if (options.setLoading) {
      setReleaseMonitoringResource({ status: "loading" });
    }

    try {
      const data = await apiClient.getAdminReleaseMonitoring();
      if (releaseMonitoringRequestSeq.current !== requestSeq) {
        return;
      }
      setReleaseMonitoringResource({ status: "success", data });
    } catch (error) {
      if (releaseMonitoringRequestSeq.current !== requestSeq) {
        return;
      }
      setReleaseMonitoringResource({
        status: "error",
        error: apiErrorDetails(error, "Unknown admin release monitoring error"),
      });
    }
  }

  async function refreshReleaseClosureResource(options: { setLoading?: boolean } = {}) {
    if (!apiClient || typeof apiClient.getAdminReleaseClosure !== "function") {
      return;
    }

    const requestSeq = releaseClosureRequestSeq.current + 1;
    releaseClosureRequestSeq.current = requestSeq;
    if (options.setLoading) {
      setReleaseClosureResource({ status: "loading" });
    }

    try {
      const data = await apiClient.getAdminReleaseClosure();
      if (releaseClosureRequestSeq.current !== requestSeq) {
        return;
      }
      setReleaseClosureResource({ status: "success", data });
    } catch (error) {
      if (releaseClosureRequestSeq.current !== requestSeq) {
        return;
      }
      setReleaseClosureResource({
        status: "error",
        error: apiErrorDetails(error, "Unknown admin release closure error"),
      });
    }
  }

  useEffect(() => {
    if (activeTaskId !== "tools") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminTools !== "function") {
      setToolsResource({ status: "idle" });
      return;
    }

    let cancelled = false;
    setToolsResource({ status: "loading" });

    void apiClient.getAdminTools().then(
      (data) => {
        if (!cancelled) {
          setToolsResource({ status: "success", data });
        }
      },
      (error) => {
        if (cancelled) {
          return;
        }

        if (error instanceof ApiClientError) {
          setToolsResource({
            status: "error",
            error: { status: error.status, message: error.message },
          });
          return;
        }

        setToolsResource({
          status: "error",
          error: {
            message: error instanceof Error ? error.message : "Unknown admin tools error",
          },
        });
      },
    );

    return () => {
      cancelled = true;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "rules") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminRules !== "function") {
      setRulesResource({ status: "idle" });
      return;
    }

    let cancelled = false;
    setRulesResource({ status: "loading" });

    void apiClient.getAdminRules().then(
      (data) => {
        if (!cancelled) {
          setRulesResource({ status: "success", data });
        }
      },
      (error) => {
        if (cancelled) {
          return;
        }

        setRulesResource({
          status: "error",
          error: apiErrorDetails(error, "Unknown admin rules error"),
        });
      },
    );

    return () => {
      cancelled = true;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "learning") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminLearningJobs !== "function") {
      setLearningJobsResource({ status: "idle" });
      return;
    }

    void refreshLearningJobsResource({ setLoading: true });

    return () => {
      learningJobsRequestSeq.current += 1;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "learning") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminAutoResearchRuns !== "function") {
      setAutoResearchRunsResource({ status: "idle" });
      return;
    }

    setAutoResearchRunResource((current) => {
      if (current.status === "loading") {
        return { status: "idle" };
      }
      if (current.status === "success" && current.refreshing) {
        return { ...current, refreshing: false };
      }
      return current;
    });
    void refreshAutoResearchRunsResource({ setLoading: true });

    return () => {
      autoResearchRunsRequestSeq.current += 1;
      autoResearchRunRequestSeq.current += 1;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "release" && activeTaskId !== "learning") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminReleaseDashboard !== "function") {
      setReleaseDashboardResource({ status: "idle" });
      return;
    }

    void refreshReleaseDashboardResource({ setLoading: true });

    return () => {
      releaseDashboardRequestSeq.current += 1;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "release") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminReleaseMonitoring !== "function") {
      setReleaseMonitoringResource({ status: "idle" });
      return;
    }

    void refreshReleaseMonitoringResource({ setLoading: true });

    return () => {
      releaseMonitoringRequestSeq.current += 1;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "release") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminReleaseClosure !== "function") {
      setReleaseClosureResource({ status: "idle" });
      return;
    }

    void refreshReleaseClosureResource({ setLoading: true });

    return () => {
      releaseClosureRequestSeq.current += 1;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "release") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminReleaseGovernance !== "function") {
      setReleaseGovernanceResource({ status: "idle" });
      return;
    }

    let cancelled = false;
    setReleaseGovernanceResource({ status: "loading" });

    void apiClient.getAdminReleaseGovernance().then(
      (data) => {
        if (!cancelled) {
          setReleaseGovernanceResource({ status: "success", data });
        }
      },
      (error) => {
        if (cancelled) {
          return;
        }

        setReleaseGovernanceResource({
          status: "error",
          error: apiErrorDetails(error, "Unknown admin release governance error"),
        });
      },
    );

    return () => {
      cancelled = true;
    };
  }, [activeTaskId, apiClient]);

  useEffect(() => {
    if (activeTaskId !== "release") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminReleaseExecution !== "function") {
      setReleaseExecutionResource({ status: "idle" });
      return;
    }

    let cancelled = false;
    setReleaseExecutionResource({ status: "loading" });

    void apiClient.getAdminReleaseExecution().then(
      (data) => {
        if (!cancelled) {
          setReleaseExecutionResource({ status: "success", data });
        }
      },
      (error) => {
        if (cancelled) {
          return;
        }

        setReleaseExecutionResource({
          status: "error",
          error: apiErrorDetails(error, "Unknown admin release execution error"),
        });
      },
    );

    return () => {
      cancelled = true;
    };
  }, [activeTaskId, apiClient]);

  const releaseGovernanceActions: AgentAdminReleaseGovernanceActions = {
    async createIntent(request) {
      if (!apiClient || typeof apiClient.createAdminReleaseIntent !== "function") {
        setReleaseGovernanceActionState({ status: "error", message: "Release governance create API is unavailable" });
        return;
      }

      setReleaseGovernanceActionState({ status: "running", label: "Create intent" });
      try {
        const data = await apiClient.createAdminReleaseIntent(request);
        setReleaseGovernanceResource({ status: "success", data });
        setReleaseGovernanceActionState({ status: "idle" });
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseGovernanceActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release intent create error").message,
        });
      }
    },

    async recordApproval(intentId, request) {
      if (!apiClient || typeof apiClient.recordAdminReleaseApproval !== "function") {
        setReleaseGovernanceActionState({ status: "error", message: "Release governance approval API is unavailable" });
        return;
      }

      setReleaseGovernanceActionState({ status: "running", label: "Record approval" });
      try {
        const data = await apiClient.recordAdminReleaseApproval(intentId, request);
        setReleaseGovernanceResource({ status: "success", data });
        setReleaseGovernanceActionState({ status: "idle" });
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseGovernanceActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release approval error").message,
        });
      }
    },

    async recordRollbackPlan(intentId, request) {
      if (!apiClient || typeof apiClient.recordAdminReleaseRollbackPlan !== "function") {
        setReleaseGovernanceActionState({ status: "error", message: "Release governance rollback-plan API is unavailable" });
        return;
      }

      setReleaseGovernanceActionState({ status: "running", label: "Record rollback plan" });
      try {
        const data = await apiClient.recordAdminReleaseRollbackPlan(intentId, request);
        setReleaseGovernanceResource({ status: "success", data });
        setReleaseGovernanceActionState({ status: "idle" });
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseGovernanceActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release rollback-plan error").message,
        });
      }
    },

    async cancelIntent(intentId, request) {
      if (!apiClient || typeof apiClient.cancelAdminReleaseIntent !== "function") {
        setReleaseGovernanceActionState({ status: "error", message: "Release governance cancel API is unavailable" });
        return;
      }

      setReleaseGovernanceActionState({ status: "running", label: "Cancel intent" });
      try {
        const data = await apiClient.cancelAdminReleaseIntent(intentId, request);
        setReleaseGovernanceResource({ status: "success", data });
        setReleaseGovernanceActionState({ status: "idle" });
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseGovernanceActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release cancel error").message,
        });
      }
    },
  };

  const releaseExecutionActions: AgentAdminReleaseExecutionActions = {
    async executeRelease(request) {
      if (!apiClient || typeof apiClient.executeAdminRelease !== "function") {
        setReleaseExecutionActionState({ status: "error", message: "Release execution API is unavailable" });
        return;
      }

      setReleaseExecutionActionState({ status: "running", label: "Execute release" });
      try {
        const data = await apiClient.executeAdminRelease(request);
        setReleaseExecutionResource({ status: "success", data });
        setReleaseExecutionActionState({ status: "idle" });
        await refreshReleaseMonitoringResource();
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseExecutionActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release execution error").message,
        });
      }
    },

    async executeRollback(request) {
      if (!apiClient || typeof apiClient.executeAdminReleaseRollback !== "function") {
        setReleaseExecutionActionState({ status: "error", message: "Release rollback API is unavailable" });
        return;
      }

      setReleaseExecutionActionState({ status: "running", label: "Execute rollback" });
      try {
        const data = await apiClient.executeAdminReleaseRollback(request);
        setReleaseExecutionResource({ status: "success", data });
        setReleaseExecutionActionState({ status: "idle" });
        await refreshReleaseMonitoringResource();
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseExecutionActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release rollback error").message,
        });
      }
    },
  };

  const releaseMonitoringActions: AgentAdminReleaseMonitoringActions = {
    async recordCheck(request) {
      if (!apiClient || typeof apiClient.recordAdminReleaseMonitoringCheck !== "function") {
        setReleaseMonitoringActionState({ status: "error", message: "Release monitoring check API is unavailable" });
        return;
      }

      setReleaseMonitoringActionState({ status: "running", label: "Record monitoring check" });
      try {
        const data = await apiClient.recordAdminReleaseMonitoringCheck(request);
        setReleaseMonitoringResource({ status: "success", data });
        setReleaseMonitoringActionState({ status: "idle" });
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseMonitoringActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release monitoring check error").message,
        });
      }
    },

    async acknowledgeAlert(alertId, request) {
      if (!apiClient || typeof apiClient.acknowledgeAdminReleaseMonitoringAlert !== "function") {
        setReleaseMonitoringActionState({ status: "error", message: "Release monitoring acknowledgement API is unavailable" });
        return;
      }

      setReleaseMonitoringActionState({ status: "running", label: "Acknowledge monitoring alert" });
      try {
        const data = await apiClient.acknowledgeAdminReleaseMonitoringAlert(alertId, request);
        setReleaseMonitoringResource({ status: "success", data });
        setReleaseMonitoringActionState({ status: "idle" });
        await refreshReleaseClosureResource();
      } catch (error) {
        setReleaseMonitoringActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release monitoring acknowledgement error").message,
        });
      }
    },
  };

  const releaseClosureActions: AgentAdminReleaseClosureActions = {
    async recordReleaseClosure(request) {
      if (!apiClient || typeof apiClient.recordAdminReleaseClosure !== "function") {
        setReleaseClosureActionState({ status: "error", message: "Release closure API is unavailable" });
        return;
      }

      setReleaseClosureActionState({ status: "running", label: "Record closure" });
      try {
        const data = await apiClient.recordAdminReleaseClosure(request);
        setReleaseClosureResource({ status: "success", data });
        setReleaseClosureActionState({ status: "idle" });
      } catch (error) {
        setReleaseClosureActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release closure error").message,
        });
      }
    },
  };

  const researchActions: AgentAdminResearchActions = {
    async refresh() {
      const actionSeq = researchActionRequestSeq.current + 1;
      researchActionRequestSeq.current = actionSeq;
      setResearchActionState({ status: "running", label: "刷新影子科研数据" });

      const outcomes = await Promise.all([
        refreshLearningJobsResource({ setLoading: true }),
        refreshReleaseDashboardResource({ setLoading: true }),
        refreshAutoResearchRunsResource({ setLoading: true }),
      ]);
      if (researchActionRequestSeq.current !== actionSeq) {
        return;
      }
      if (outcomes.some((outcome) => outcome.status === "stale")) {
        setResearchActionState({ status: "idle" });
        return;
      }

      const failureMessages = outcomes.flatMap((outcome) =>
        outcome.status === "error" ? [outcome.message] : [],
      );
      if (failureMessages.length > 0) {
        setResearchActionState({
          status: "error",
          message: `影子科研数据刷新未完成：${failureMessages.join("；")}`,
        });
        return;
      }
      setResearchActionState({ status: "success", message: "影子科研数据已刷新。" });
    },

    async createLearningJob(request) {
      const actionSeq = researchActionRequestSeq.current + 1;
      researchActionRequestSeq.current = actionSeq;
      if (!apiClient || typeof apiClient.createAdminLearningJob !== "function") {
        setResearchActionState({ status: "error", message: "LearningJob create API is unavailable" });
        return;
      }

      setResearchActionState({ status: "running", label: "创建影子 LearningJob" });
      try {
        const data = await apiClient.createAdminLearningJob(request);
        const refreshOutcome = await refreshLearningJobsResource({ setLoading: true });
        if (researchActionRequestSeq.current !== actionSeq) {
          return;
        }

        const createdMessage = `已创建影子 LearningJob ${data.job.job_id}，候选仍为 inert。`;
        if (refreshOutcome.status === "error") {
          setResearchActionState({
            status: "success",
            message: `${createdMessage} 列表刷新失败：${refreshOutcome.message}`,
          });
          return;
        }
        if (refreshOutcome.status === "stale") {
          setResearchActionState({
            status: "success",
            message: `${createdMessage} 列表将于下次打开时刷新。`,
          });
          return;
        }
        setResearchActionState({ status: "success", message: createdMessage });
      } catch (error) {
        if (researchActionRequestSeq.current !== actionSeq) {
          return;
        }
        setResearchActionState({
          status: "error",
          message: researchApiErrorMessage(error, "Unknown LearningJob create error"),
        });
      }
    },

    async evaluateCohortFeasibility(request) {
      const actionSeq = researchActionRequestSeq.current + 1;
      researchActionRequestSeq.current = actionSeq;
      if (!apiClient || typeof apiClient.evaluateAdminCohortFeasibility !== "function") {
        const message = "Cohort feasibility API is unavailable";
        setCohortFeasibilityResource({ status: "error", error: { message } });
        setResearchActionState({ status: "error", message });
        return;
      }

      const requestSeq = cohortFeasibilityRequestSeq.current + 1;
      cohortFeasibilityRequestSeq.current = requestSeq;
      setCohortFeasibilityResource({ status: "loading" });
      setResearchActionState({ status: "running", label: "运行影子队列可行性评估" });

      try {
        const data = await apiClient.evaluateAdminCohortFeasibility(request);
        if (cohortFeasibilityRequestSeq.current !== requestSeq) {
          return;
        }
        setCohortFeasibilityResource({ status: "success", data });
        if (researchActionRequestSeq.current === actionSeq) {
          setResearchActionState({
            status: "success",
            message: `影子队列可行性评估 ${data.result_id} 已完成，状态 ${data.status}。`,
          });
        }
      } catch (error) {
        if (cohortFeasibilityRequestSeq.current !== requestSeq) {
          return;
        }
        const message = researchApiErrorMessage(error, "Unknown cohort feasibility error");
        const status = error instanceof ApiClientError ? error.status : undefined;
        setCohortFeasibilityResource({ status: "error", error: { status, message } });
        if (researchActionRequestSeq.current === actionSeq) {
          setResearchActionState({ status: "error", message });
        }
      }
    },
  };

  const autoResearchActions: AgentAdminAutoResearchActions = {
    async refreshRuns() {
      const actionSeq = autoResearchActionRequestSeq.current + 1;
      autoResearchActionRequestSeq.current = actionSeq;
      setAutoResearchActionState({ status: "running", label: "刷新自动科研 Runs" });
      const outcome = await refreshAutoResearchRunsResource({ setLoading: true });
      if (autoResearchActionRequestSeq.current !== actionSeq) {
        return;
      }
      if (outcome.status === "stale") {
        setAutoResearchActionState({ status: "idle" });
      } else if (outcome.status === "error") {
        setAutoResearchActionState({ status: "error", message: outcome.message });
      } else {
        setAutoResearchActionState({ status: "success", message: "自动科研 Runs 已刷新。" });
      }
    },

    async selectRun(runId) {
      selectAutoResearchRunId(runId);
      await refreshAutoResearchRunResource(runId);
    },

    async refreshRun() {
      const runId = selectedAutoResearchRunIdRef.current;
      if (!runId) {
        return;
      }
      await refreshAutoResearchRunResource(runId, { preserveData: true });
    },

    async createRun(request) {
      const actionSeq = autoResearchActionRequestSeq.current + 1;
      autoResearchActionRequestSeq.current = actionSeq;
      if (!apiClient || typeof apiClient.createAdminAutoResearchRun !== "function") {
        setAutoResearchActionState({ status: "error", message: "Auto-research create API is unavailable" });
        return;
      }

      setAutoResearchActionState({ status: "running", label: "运行自动科研闭环" });
      try {
        const data = await apiClient.createAdminAutoResearchRun(request);
        if (autoResearchActionRequestSeq.current !== actionSeq) {
          return;
        }
        // The POST response is authoritative for the newly written Run. Invalidate any
        // list request that started before it so an older snapshot cannot overwrite it.
        autoResearchRunsRequestSeq.current += 1;
        setAutoResearchRunsResource((current) => {
          const existing = current.status === "success" ? current.data.runs : [];
          const runs = [data.run, ...existing.filter((run) => run.run_id !== data.run.run_id)];
          return {
            status: "success",
            data: {
              runs,
              integrity: data.integrity,
              runtime: data.runtime,
            },
          };
        });
        autoResearchRunRequestSeq.current += 1;
        selectAutoResearchRunId(data.run.run_id);
        setAutoResearchRunResource({
          status: "success",
          data: {
            run: data.run,
            integrity: data.integrity,
            runtime: data.runtime,
          },
        });

        const runVerb = data.reused ? "已复用" : "已记录";
        const failedStage = data.run.stages.find((stage) => stage.status === "failed");
        const failedStageSummary = failedStage
          ? `失败阶段 ${failedStage.name}：${failedStage.error ?? failedStage.summary}`
          : "请检查阶段时间线。";
        if (data.run.status === "completed_shadow") {
          setAutoResearchActionState({
            status: "success",
            message: `${runVerb}自动科研 Run ${data.run.run_id}；闭环执行完成，仍需人工复核。`,
          });
        } else if (data.run.status === "partial_shadow") {
          setAutoResearchActionState({
            status: "warning",
            message: `${runVerb}部分完成的 Run ${data.run.run_id}；${failedStageSummary}`,
          });
        } else {
          setAutoResearchActionState({
            status: "error",
            message: `${runVerb}执行失败的 Run ${data.run.run_id}；${failedStageSummary} 可检查输入后使用新的幂等键重试。`,
          });
        }
        if (typeof apiClient.getAdminAutoResearchRuns === "function") {
          // Reconcile immediately with the append-only ledger. Existing successful
          // data remains visible while this authoritative refresh is in flight.
          void refreshAutoResearchRunsResource();
        }
      } catch (error) {
        if (autoResearchActionRequestSeq.current !== actionSeq) {
          return;
        }
        setAutoResearchActionState({
          status: "error",
          message: researchApiErrorMessage(error, "Unknown auto-research create error"),
        });
      }
    },
  };

  function navigateTask(taskId: AgentAdminTaskId) {
    if (taskId === "tools" && activeTaskId !== "tools" && apiClient && typeof apiClient.getAdminTools === "function") {
      setToolsResource({ status: "loading" });
    }
    if (taskId === "rules" && activeTaskId !== "rules" && apiClient && typeof apiClient.getAdminRules === "function") {
      setRulesResource({ status: "loading" });
    }
    if (taskId === "learning" && activeTaskId !== "learning" && apiClient && typeof apiClient.getAdminLearningJobs === "function") {
      setLearningJobsResource({ status: "loading" });
    }
    if (taskId === "learning" && activeTaskId !== "learning" && apiClient && typeof apiClient.getAdminAutoResearchRuns === "function") {
      setAutoResearchRunsResource((current) => (
        current.status === "success" ? { ...current, refreshing: true } : { status: "loading" }
      ));
    }
    if (
      (taskId === "release" || taskId === "learning")
      && activeTaskId !== taskId
      && apiClient
      && typeof apiClient.getAdminReleaseDashboard === "function"
    ) {
      setReleaseDashboardResource({ status: "loading" });
    }
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseGovernance === "function") {
      setReleaseGovernanceResource({ status: "loading" });
    }
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseExecution === "function") {
      setReleaseExecutionResource({ status: "loading" });
    }
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseMonitoring === "function") {
      setReleaseMonitoringResource({ status: "loading" });
    }
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseClosure === "function") {
      setReleaseClosureResource({ status: "loading" });
    }
    setActiveTaskId(taskId);
  }

  return (
    <main className="clinical-app-shell agent-admin-shell" data-testid="agent-admin-console">
      <ClinicalTopNav
        brandLabel="智能体后台"
        brandLogoVariant="light"
        navLabel="智能体后台导航"
        items={ADMIN_NAV_ITEMS}
        activeKey={activeTaskId}
        onSelect={(key) => {
          if (AGENT_ADMIN_TASKS.some((task) => task.id === key)) {
            navigateTask(key as AgentAdminTaskId);
          }
        }}
        statusLabel="受控后台"
        statusTone="safe"
        profileLabel="后台"
        profileAriaLabel="切换工作台"
        profileControl={surfaceSwitcher}
        className="agent-admin-top-nav"
      />

      <section className="agent-admin-context-strip" aria-label="后台上下文">
        <div>
          <span>当前观察</span>
          <strong>{watchedSceneLabel}</strong>
        </div>
        <div>
          <span>患者 Session</span>
          <strong>{patient.sessionId ?? "未创建"}</strong>
        </div>
        <div>
          <span>医生 Session</span>
          <strong>{doctor.sessionId ?? "未创建"}</strong>
        </div>
        <div>
          <span>Runtime</span>
          <strong>{activeRuntime(patient, doctor)}</strong>
        </div>
        <div>
          <span>Active Run</span>
          <strong>{watchedState.activeRunId ?? "idle"}</strong>
        </div>
      </section>

      <div className="agent-admin-console-layout">
        <nav className="agent-admin-subtask-rail" aria-label="后台子任务">
          <div className="agent-admin-rail-header">
            <span>后台子任务</span>
            <small>shadow / review-gated</small>
          </div>
          {AGENT_ADMIN_TASKS.map((task) => {
            const Icon = task.icon;
            const isActive = task.id === activeTaskId;
            return (
              <button
                key={task.id}
                type="button"
                className={classNames([
                  "agent-admin-task-button",
                  isActive && "agent-admin-task-button-active",
                ])}
                aria-current={isActive ? "page" : undefined}
                onClick={() => navigateTask(task.id)}
              >
                <span className="agent-admin-task-icon" aria-hidden="true">
                  <Icon size={17} strokeWidth={2} />
                </span>
                <span className="agent-admin-task-copy">
                  <span className="agent-admin-task-title-row">
                    <strong>{task.label}</strong>
                    <em>{task.status}</em>
                  </span>
                  <small>{task.responsibility}</small>
                </span>
              </button>
            );
          })}
        </nav>

        <AgentAdminTaskPages
          activeTaskId={activeTaskId}
          activeScene={activeScene}
          patient={patient}
          doctor={doctor}
          onNavigateTask={navigateTask}
          toolsResource={toolsResource}
          rulesResource={rulesResource}
          learningJobsResource={learningJobsResource}
          cohortFeasibilityResource={cohortFeasibilityResource}
          autoResearchRunsResource={autoResearchRunsResource}
          autoResearchRunResource={autoResearchRunResource}
          selectedAutoResearchRunId={selectedAutoResearchRunId}
          autoResearchActionState={autoResearchActionState}
          autoResearchActions={autoResearchActions}
          researchActionState={researchActionState}
          researchActions={researchActions}
          releaseDashboardResource={releaseDashboardResource}
          releaseGovernanceResource={releaseGovernanceResource}
          releaseExecutionResource={releaseExecutionResource}
          releaseMonitoringResource={releaseMonitoringResource}
          releaseClosureResource={releaseClosureResource}
          releaseGovernanceActionState={releaseGovernanceActionState}
          releaseExecutionActionState={releaseExecutionActionState}
          releaseMonitoringActionState={releaseMonitoringActionState}
          releaseClosureActionState={releaseClosureActionState}
          releaseGovernanceActions={releaseGovernanceActions}
          releaseExecutionActions={releaseExecutionActions}
          releaseMonitoringActions={releaseMonitoringActions}
          releaseClosureActions={releaseClosureActions}
        />
      </div>
    </main>
  );
}
