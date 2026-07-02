import { useEffect, useState, type ReactNode } from "react";

import { ApiClientError, type ApiClient } from "../../app/api/client";
import type {
  AdminCancelReleaseIntentRequest,
  AdminCreateReleaseIntentRequest,
  AdminRecordReleaseApprovalRequest,
  AdminRecordReleaseRollbackPlanRequest,
  AdminReleaseDashboardResponse,
  AdminReleaseGovernanceResponse,
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
      | "getAdminReleaseDashboard"
      | "getAdminReleaseGovernance"
      | "createAdminReleaseIntent"
      | "recordAdminReleaseApproval"
      | "recordAdminReleaseRollbackPlan"
      | "cancelAdminReleaseIntent"
    >
  >;
};

export type AgentAdminToolsResource =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: AdminToolManifestResponse }
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

export type AgentAdminReleaseGovernanceActionState =
  | { status: "idle" }
  | { status: "running"; label: string }
  | { status: "error"; message: string };

export type AgentAdminReleaseGovernanceActions = {
  createIntent: (request: AdminCreateReleaseIntentRequest) => Promise<void>;
  recordApproval: (intentId: string, request: AdminRecordReleaseApprovalRequest) => Promise<void>;
  recordRollbackPlan: (intentId: string, request: AdminRecordReleaseRollbackPlanRequest) => Promise<void>;
  cancelIntent: (intentId: string, request: AdminCancelReleaseIntentRequest) => Promise<void>;
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
  const [releaseDashboardResource, setReleaseDashboardResource] = useState<AgentAdminReleaseDashboardResource>({ status: "idle" });
  const [releaseGovernanceResource, setReleaseGovernanceResource] = useState<AgentAdminReleaseGovernanceResource>({ status: "idle" });
  const [releaseGovernanceActionState, setReleaseGovernanceActionState] = useState<AgentAdminReleaseGovernanceActionState>({ status: "idle" });
  const watchedState = activeScene === "doctor" ? doctor : patient;
  const watchedSceneLabel = activeScene === "doctor" ? "医生会话" : "患者会话";

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
      } catch (error) {
        setReleaseGovernanceActionState({
          status: "error",
          message: apiErrorDetails(error, "Unknown release cancel error").message,
        });
      }
    },
  };

  function navigateTask(taskId: AgentAdminTaskId) {
    if (taskId === "tools" && activeTaskId !== "tools" && apiClient && typeof apiClient.getAdminTools === "function") {
      setToolsResource({ status: "loading" });
    }
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseDashboard === "function") {
      setReleaseDashboardResource({ status: "loading" });
    }
    if (taskId === "release" && activeTaskId !== "release" && apiClient && typeof apiClient.getAdminReleaseGovernance === "function") {
      setReleaseGovernanceResource({ status: "loading" });
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
          releaseDashboardResource={releaseDashboardResource}
          releaseGovernanceResource={releaseGovernanceResource}
          releaseGovernanceActionState={releaseGovernanceActionState}
          releaseGovernanceActions={releaseGovernanceActions}
        />
      </div>
    </main>
  );
}
