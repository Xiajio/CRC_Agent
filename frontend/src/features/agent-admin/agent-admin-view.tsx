import { useEffect, useRef, useState, type ReactNode } from "react";

import { ApiClientError, type ApiClient } from "../../app/api/client";
import type {
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

function apiErrorDetails(error: unknown, fallbackMessage: string): { status?: number; message: string } {
  if (error instanceof ApiClientError) {
    return { status: error.status, message: error.message };
  }

  return {
    message: error instanceof Error ? error.message : fallbackMessage,
  };
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
  const [releaseDashboardResource, setReleaseDashboardResource] = useState<AgentAdminReleaseDashboardResource>({ status: "idle" });
  const [releaseGovernanceResource, setReleaseGovernanceResource] = useState<AgentAdminReleaseGovernanceResource>({ status: "idle" });
  const [releaseExecutionResource, setReleaseExecutionResource] = useState<AgentAdminReleaseExecutionResource>({ status: "idle" });
  const [releaseMonitoringResource, setReleaseMonitoringResource] = useState<AgentAdminReleaseMonitoringResource>({ status: "idle" });
  const [releaseClosureResource, setReleaseClosureResource] = useState<AgentAdminReleaseClosureResource>({ status: "idle" });
  const [releaseGovernanceActionState, setReleaseGovernanceActionState] = useState<AgentAdminReleaseGovernanceActionState>({ status: "idle" });
  const [releaseExecutionActionState, setReleaseExecutionActionState] = useState<AgentAdminReleaseExecutionActionState>({ status: "idle" });
  const [releaseMonitoringActionState, setReleaseMonitoringActionState] = useState<AgentAdminReleaseMonitoringActionState>({ status: "idle" });
  const [releaseClosureActionState, setReleaseClosureActionState] = useState<AgentAdminReleaseClosureActionState>({ status: "idle" });
  const releaseMonitoringRequestSeq = useRef(0);
  const releaseClosureRequestSeq = useRef(0);
  const watchedState = activeScene === "doctor" ? doctor : patient;
  const watchedSceneLabel = activeScene === "doctor" ? "医生会话" : "患者会话";

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
    if (activeTaskId !== "release") {
      return;
    }

    if (!apiClient || typeof apiClient.getAdminReleaseDashboard !== "function") {
      setReleaseDashboardResource({ status: "idle" });
      return;
    }

    let cancelled = false;
    setReleaseDashboardResource({ status: "loading" });

    void apiClient.getAdminReleaseDashboard().then(
      (data) => {
        if (!cancelled) {
          setReleaseDashboardResource({ status: "success", data });
        }
      },
      (error) => {
        if (cancelled) {
          return;
        }

        if (error instanceof ApiClientError) {
          setReleaseDashboardResource({
            status: "error",
            error: { status: error.status, message: error.message },
          });
          return;
        }

        setReleaseDashboardResource({
          status: "error",
          error: {
            message: error instanceof Error ? error.message : "Unknown admin release dashboard error",
          },
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

  function navigateTask(taskId: AgentAdminTaskId) {
    if (taskId === "tools" && activeTaskId !== "tools" && apiClient && typeof apiClient.getAdminTools === "function") {
      setToolsResource({ status: "loading" });
    }
    if (taskId === "rules" && activeTaskId !== "rules" && apiClient && typeof apiClient.getAdminRules === "function") {
      setRulesResource({ status: "loading" });
    }
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseDashboard === "function") {
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
        statusLabel="只读观测"
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
            <small>Phase 1 read-only</small>
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
