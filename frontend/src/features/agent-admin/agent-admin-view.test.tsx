import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type {
  AdminReleaseDashboardResponse,
  AdminReleaseGovernanceResponse,
  AdminToolManifestResponse,
} from "../../app/api/types";
import type { SessionState } from "../../app/api/types";
import { createInitialSessionState } from "../../app/store/stream-reducer";
import {
  ADMIN_NAV_ITEMS,
  AGENT_ADMIN_TASKS,
  buildEvidenceRows,
  buildMemoryAutomationSummary,
  buildMemoryFactRows,
  buildLearningReadiness,
  buildMemoryLayerRows,
  buildMemoryLifecycleRows,
  buildMemoryRows,
  buildMemoryVisualizationRows,
  buildPermissionRows,
  buildRuleCatalogGroups,
  buildRuleCatalogRows,
  buildRuleGroupRows,
  buildSessionSummary,
  buildToolInventoryRows,
  buildToolGroupRows,
  buildToolReachabilityRows,
  buildTraceRows,
  RULE_CATALOG,
  TOOL_INVENTORY,
} from "./agent-admin-model";
import { AgentAdminView } from "./agent-admin-view";

function makeState(overrides: Partial<SessionState> = {}): SessionState {
  return {
    ...createInitialSessionState(),
    ...overrides,
  };
}

function makeAdminToolsManifest(): AdminToolManifestResponse {
  return {
    tools: [
      {
        name: "search_clinical_guidelines",
        category: "rag",
        registries: ["graph", "graph_web", "executor"],
        route_targets: ["knowledge"],
        graph_scope: "both",
        planner_aliases: ["search_clinical_guidelines", "search"],
        requires_web: false,
        available: true,
        state: "available",
      },
      {
        name: "search_latest_research",
        category: "web",
        registries: ["executor", "optional"],
        route_targets: ["knowledge", "web_search"],
        graph_scope: "executor_only",
        planner_aliases: ["search_latest_research"],
        requires_web: true,
        available: false,
        state: "candidate",
      },
    ],
    groups: [
      { category: "rag", count: 1, available_count: 1 },
      { category: "web", count: 1, available_count: 0 },
    ],
    runtime: {
      web_search_enabled: false,
      auth: "admin",
      source: "src.tools.manifest",
    },
  };
}

function makeAdminReleaseDashboard(): AdminReleaseDashboardResponse {
  return {
    version_chain: {
      agent_policy_version: "agent_policy_20260629_0",
      clinical_safety_policy_version: "crc_safety_policy_v0",
      evidence_index_version: "rag_crc_guideline_20260620",
      judge_rubric_version: "crc_rubric_v0",
    },
    release_decision: "feature_flag_or_pass",
    rollback_target: "agent_policy_20260624_0",
    human_signoff: {
      required: true,
      status: "missing",
      reason: "Step 11 is read-only; sign-off must be recorded by a later audited write path.",
    },
    summary: {
      hard_fail_count: 0,
      p0_cases_total: 5,
      p0_cases_passed: 5,
      literature_claims: 3,
      literature_isolation_violations: 0,
      clinical_rag_ingest_enabled: false,
    },
    runs: [
      {
        run_id: "harness_20260629_001",
        kind: "p0_crc_harness",
        status: "pass",
        source_path: "reports/harness/harness_20260629_001.json",
        hard_fail_count: 0,
      },
      {
        run_id: "literature_harness_20260630_001",
        kind: "literature_shadow_harness",
        status: "shadow_only",
        source_path: "reports/literature/literature_harness_20260630_001.json",
        hard_fail_count: 0,
      },
    ],
    blocking_gates: [
      {
        id: "no_literature_clinical_rag",
        label: "Unreviewed literature stays out of clinical RAG",
        state: "locked",
        reason: "Clinical RAG ingest is disabled in Step 11.",
      },
    ],
    disabled_actions: [
      {
        id: "record_human_signoff",
        label: "Record human sign-off",
        reason: "Requires a later audited write-path design.",
      },
      {
        id: "publish_feature_flag",
        label: "Publish feature flag release",
        reason: "Step 11 observes readiness only.",
      },
      {
        id: "rollback_release",
        label: "Rollback release",
        reason: "Rollback execution is outside this read-only slice.",
      },
    ],
    runtime: {
      auth: "admin",
      source: "reports/static_release_artifacts",
      mode: "read_only",
    },
  };
}

function makeAdminReleaseGovernance(
  overrides: Partial<AdminReleaseGovernanceResponse> = {},
): AdminReleaseGovernanceResponse {
  const activeIntent = {
    intent_id: "release_intent_test",
    source_release_report_id: "release_safety_20260629_001",
    source_report_path: "reports/release_safety/release_safety_20260629_001.json",
    harness_run_ids: ["harness_20260629_001"],
    literature_run_id: "literature_harness_20260630_001",
    version_chain: {
      agent_policy_version: "agent_policy_20260629_0",
      clinical_safety_policy_version: "crc_safety_policy_v0",
    },
    release_decision_snapshot: "feature_flag_or_pass",
    rollback_target: "agent_policy_20260624_0",
    requested_by: "admin_operator",
    requested_at: "2026-07-02T00:00:00+08:00",
    target_scope: "shadow" as const,
    status: "pending_approval" as const,
    derived_status: "pending_approval" as const,
    blocking_summary: { hard_fail_count: 0 },
  };
  return {
    dashboard_snapshot: {
      release_decision: "feature_flag_or_pass",
      rollback_target: "agent_policy_20260624_0",
      hard_fail_count: 0,
      literature_status: "shadow_only",
    },
    intents: [activeIntent],
    active_intent: activeIntent,
    approvals: [],
    required_approvals: [
      { role: "release_manager", status: "missing", latest_decision: null },
      { role: "clinical_safety_reviewer", status: "missing", latest_decision: null },
    ],
    rollback_plan: null,
    audit_events: [
      {
        event_id: "release_audit_test",
        intent_id: "release_intent_test",
        event_type: "intent_created",
        actor: "admin_operator",
        timestamp: "2026-07-02T00:00:00+08:00",
        payload_hash: "sha256:payload",
        previous_event_hash: "sha256:GENESIS",
        event_hash: "sha256:event",
      },
    ],
    integrity: { status: "verified", warnings: [] },
    disabled_execution_actions: [
      {
        id: "execute_release",
        label: "Execute release",
        disabled: true,
        reason: "Step 12 records governance only.",
      },
      {
        id: "execute_rollback",
        label: "Execute rollback",
        disabled: true,
        reason: "Rollback execution requires a later execution-path design.",
      },
    ],
    runtime: {
      auth: "admin",
      source: "reports/release_governance",
      mode: "audit_only",
    },
    ...overrides,
  };
}

function clickToolsTask() {
  const button = screen
    .getAllByRole("button")
    .find((candidate) => candidate.textContent?.includes("tool surfaces"));
  expect(button).toBeDefined();
  fireEvent.click(button!);
}

function clickReleaseTask() {
  const button = screen
    .getAllByRole("button")
    .find((candidate) => candidate.textContent?.includes("version chain / harness runs"));
  expect(button).toBeDefined();
  fireEvent.click(button!);
}

describe("AgentAdminView", () => {
  it("renders shared admin primitives with stable labels", () => {
    render(
      <AgentAdminView
        activeScene="patient"
        patient={makeState({ sessionId: "patient-shared", snapshotVersion: 1 })}
        doctor={makeState({ sessionId: "doctor-shared", snapshotVersion: 2 })}
        surfaceSwitcher={<button type="button">{"\u540e\u53f0\u5207\u6362\u83dc\u5355"}</button>}
      />,
    );

    expect(screen.getByLabelText("\u540e\u53f0\u4e0a\u4e0b\u6587")).toBeInTheDocument();
    expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("\u8fd0\u884c\u5065\u5eb7\u65f6\u95f4\u7ebf");
    expect(screen.getByText("\u98ce\u9669\u961f\u5217")).toBeInTheDocument();
    expect(screen.getByText("\u6700\u8fd1\u53d8\u5316")).toBeInTheDocument();
  });

  it("renders the overview as a health command center", () => {
    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-overview", snapshotVersion: 4 })}
        doctor={makeState({
          sessionId: "doctor-overview",
          snapshotVersion: 8,
          statusNode: "planner",
          activeRunId: "run-overview",
          references: [{ title: "overview evidence", source: "RAG", confidence: 0.91 }],
        })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveAttribute("data-task-id", "overview");
    expect(page).toHaveTextContent("运行健康时间线");
    expect(page).toHaveTextContent("当前图状态");
    expect(page).toHaveTextContent("风险队列");
    expect(page).toHaveTextContent("最近变化");
    expect(page).toHaveTextContent("doctor-overview");
    expect(page).toHaveTextContent("run-overview");
  });

  it("renders the operations console with the detailed subtask rail", () => {
    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({
          sessionId: "patient-session-1",
          snapshotVersion: 4,
          contextMaintenance: { status: "completed", message: "已完成上下文维护" },
        })}
        doctor={makeState({
          sessionId: "doctor-session-9",
          snapshotVersion: 8,
          statusNode: "planner",
          activeRunId: "run-77",
          plan: [{ id: "retrieve-guidelines", title: "检索指南", status: "in_progress" }],
          references: [{ title: "NCCN guideline", source: "RAG", confidence: 0.92 }],
        })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    expect(screen.getByRole("banner")).toHaveTextContent("智能体后台");
    expect(screen.getByRole("img", { name: "亿铸科技公司标识" })).toHaveAttribute(
      "src",
      expect.stringContaining("yizhu-company-logo-light"),
    );
    expect(screen.getByRole("button", { name: "后台切换菜单" })).toBeInTheDocument();
    expect(screen.getAllByText("patient-session-1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("doctor-session-9").length).toBeGreaterThan(0);
    expect(screen.getAllByText("run-77").length).toBeGreaterThan(0);

    const rail = screen.getByRole("navigation", { name: "后台子任务" });
    expect(within(rail).getAllByRole("button")).toHaveLength(10);
    for (const label of ["总览", "会话", "记忆", "规则", "工具", "学习", "Trace", "证据", "Release", "设置只读"]) {
      expect(within(rail).getByText(label)).toBeInTheDocument();
    }

    expect(within(rail).getByText("summary memory / 永久事实")).toBeInTheDocument();
    expect(within(rail).getByText("daily paper readiness")).toBeInTheDocument();
  });

  it.each([
    { label: /总览/, taskId: "overview", hidden: "权限矩阵", visible: "运行健康时间线" },
    { label: /会话/, taskId: "sessions", hidden: "运行健康时间线", visible: "差异对比" },
    { label: /记忆/, taskId: "memory", hidden: "差异对比", visible: "自动化维护流水线" },
    { label: /规则/, taskId: "rules", hidden: "记忆事实", visible: "规则目录" },
    { label: /工具/, taskId: "tools", hidden: "规则目录", visible: "可达性矩阵" },
    { label: /学习/, taskId: "learning", hidden: "可达性矩阵", visible: "学习流水线" },
    { label: /Trace/, taskId: "trace", hidden: "学习流水线", visible: "event table" },
    { label: /证据/, taskId: "evidence", hidden: "event table", visible: "RAG pipeline" },
    { label: /Release/, taskId: "release", hidden: "RAG pipeline", visible: "Release Dashboard" },
    { label: /设置只读/, taskId: "read-only", hidden: "RAG pipeline", visible: "权限矩阵" },
  ])("shows only the selected $taskId page", ({ label, taskId, hidden, visible }) => {
    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-switch" })}
        doctor={makeState({ sessionId: "doctor-switch" })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    const rail = screen.getByRole("navigation", { name: "后台子任务" });
    fireEvent.click(within(rail).getByRole("button", { name: label }));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveAttribute("data-task-id", taskId);
    expect(page).toHaveTextContent(visible);
    expect(page).not.toHaveTextContent(hidden);
  });

  it("renders sessions as a patient doctor comparison workspace", () => {
    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-compare", snapshotVersion: 5 })}
        doctor={makeState({ sessionId: "doctor-compare", snapshotVersion: 9, activeRunId: "doctor-run" })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /会话/ }));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveAttribute("data-task-id", "sessions");
    expect(page).toHaveTextContent("患者 Session");
    expect(page).toHaveTextContent("医生 Session");
    expect(page).toHaveTextContent("差异对比");
    expect(page).toHaveTextContent("recent session events");
    expect(page).toHaveTextContent("doctor-run");
  });

  it("renders memory as an automation lifecycle workbench", () => {
    render(
      <AgentAdminView
        activeScene="patient"
        patient={makeState({
          sessionId: "patient-memory",
          snapshotVersion: 5,
          contextState: {
            summary_memory: "患者已经学习过检查方案",
            summary_memory_cursor: 3,
            structured_summary: {
              immutable_info: ["长期诊断信息"],
              dynamic_info: ["最近症状变化"],
              anchor_events: ["首次确诊"],
            },
          },
          contextMaintenance: { status: "completed", message: "已完成上下文维护" },
        })}
        doctor={makeState({ sessionId: "doctor-memory" })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /记忆/ }));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("这是会话上下文记忆，不是模型权重训练");
    expect(page).toHaveTextContent("记忆健康");
    expect(page).toHaveTextContent("摘要状态");
    expect(page).toHaveTextContent("待刷新");
    expect(page).toHaveTextContent("记忆分层导航");
    expect(page).toHaveTextContent("自动化维护流水线");
    expect(page).toHaveTextContent("收集");
    expect(page).toHaveTextContent("摘要");
    expect(page).toHaveTextContent("结构化");
    expect(page).toHaveTextContent("同步");
    expect(page).toHaveTextContent("过期检查");
    expect(page).toHaveTextContent("来源与维护审计");
    expect(page).toHaveTextContent("当前记忆可视化");
    expect(page).toHaveTextContent("长期诊断信息");
    expect(page).toHaveTextContent("压缩会话上下文");
    expect(page).toHaveTextContent("稳定事实");
    expect(
      within(page).queryByRole("button", {
        name: /delete|edit|merge|pin|approve|run maintenance|删除|编辑|合并|置顶|批准|运行维护/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("renders memory empty, unstructured, and failed states explicitly", () => {
    render(
      <AgentAdminView
        activeScene="patient"
        patient={makeState({
          sessionId: "patient-memory-empty",
          snapshotVersion: 2,
          contextState: {
            structured_summary: {
              dynamic_info: [null],
            },
          },
          contextMaintenance: { status: "failed", message: "维护失败", error: "summary timeout" },
        })}
        doctor={makeState({ sessionId: "doctor-memory-empty" })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /记忆/ }));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("待生成");
    expect(page).toHaveTextContent("结构化字段不可读");
    expect(page).toHaveTextContent("summary timeout");
    expect(page).toHaveTextContent("failed");
    expect(page).toHaveTextContent("暂无摘要记忆");
    expect(page).toHaveTextContent("维护日志为空");
  });

  it("renders rules as a catalog tree and inspector", () => {
    render(
      <AgentAdminView
        activeScene="patient"
        patient={makeState({ sessionId: "patient-rules" })}
        doctor={makeState({ sessionId: "doctor-rules" })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /规则/ }));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("规则分组");
    expect(page).toHaveTextContent("routing.intent.knowledge_query");
    expect(page).toHaveTextContent("owner module");
    expect(page).toHaveTextContent("editable: false");
  });

  it("renders tools as an inventory table and reachability map", () => {
    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-tools" })}
        doctor={makeState({ sessionId: "doctor-tools" })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /工具/ }));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("工具清单");
    expect(page).toHaveTextContent("search_latest_research");
    expect(page).toHaveTextContent("可达性矩阵");
    expect(page).toHaveTextContent("WEB_SEARCH_ENABLED");
  });

  it("renders tool rows from the admin runtime manifest", async () => {
    const manifest = makeAdminToolsManifest();
    const apiClient = { getAdminTools: vi.fn(async () => manifest) };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-tools-api" })}
        doctor={makeState({ sessionId: "doctor-tools-api" })}
        surfaceSwitcher={<button type="button">鍚庡彴鍒囨崲鑿滃崟</button>}
        apiClient={apiClient}
      />,
    );

    clickToolsTask();

    await waitFor(() => expect(apiClient.getAdminTools).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("runtime manifest");
    expect(page).toHaveTextContent("search_clinical_guidelines");
    expect(page).toHaveTextContent("search_latest_research");
    expect(page).toHaveTextContent("WEB_SEARCH_ENABLED");
    expect(page).not.toHaveTextContent("query_case_database");
    expect(page).not.toHaveTextContent("get_patient_registry");
  });

  it("keeps the loaded runtime manifest when the active tools task is clicked again", async () => {
    const manifest = makeAdminToolsManifest();
    const apiClient = { getAdminTools: vi.fn(async () => manifest) };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-tools-repeat" })}
        doctor={makeState({ sessionId: "doctor-tools-repeat" })}
        surfaceSwitcher={<button type="button">鍚庡彴鍒囨崲鑿滃崟</button>}
        apiClient={apiClient}
      />,
    );

    clickToolsTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("search_clinical_guidelines"));

    clickToolsTask();

    const page = screen.getByTestId("agent-admin-task-page");
    expect(apiClient.getAdminTools).toHaveBeenCalledTimes(1);
    expect(page).toHaveTextContent("search_clinical_guidelines");
    expect(page).not.toHaveTextContent("reading runtime manifest");
  });

  it("does not render fallback inventory while the runtime manifest is loading", async () => {
    const apiClient = {
      getAdminTools: vi.fn(() => new Promise<AdminToolManifestResponse>(() => {})),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-tools-loading" })}
        doctor={makeState({ sessionId: "doctor-tools-loading" })}
        surfaceSwitcher={<button type="button">鍚庡彴鍒囨崲鑿滃崟</button>}
        apiClient={apiClient}
      />,
    );

    clickToolsTask();

    await waitFor(() => expect(apiClient.getAdminTools).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("reading runtime manifest");
    expect(page).not.toHaveTextContent("fallback inventory");
    expect(page).not.toHaveTextContent("query_case_database");
    expect(page).not.toHaveTextContent("get_patient_registry");
  });

  it("renders explicit empty states when the admin runtime manifest has no tools", async () => {
    const manifest: AdminToolManifestResponse = {
      tools: [],
      groups: [],
      runtime: {
        web_search_enabled: false,
        auth: "admin",
        source: "src.tools.manifest",
      },
    };
    const apiClient = { getAdminTools: vi.fn(async () => manifest) };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-tools-empty" })}
        doctor={makeState({ sessionId: "doctor-tools-empty" })}
        surfaceSwitcher={<button type="button">鍚庡彴鍒囨崲鑿滃崟</button>}
        apiClient={apiClient}
      />,
    );

    clickToolsTask();

    await waitFor(() => expect(apiClient.getAdminTools).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("runtime manifest");
    expect(page).toHaveTextContent("runtime manifest returned no tools");
    expect(page).toHaveTextContent("runtime manifest returned no groups");
    expect(page).not.toHaveTextContent("reading runtime manifest");
    expect(page).not.toHaveTextContent("fallback inventory");
  });

  it("falls back to the static inventory when the admin runtime manifest fails", async () => {
    const apiClient = { getAdminTools: vi.fn(async () => {
      throw new Error("Forbidden");
    }) };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-tools-fallback" })}
        doctor={makeState({ sessionId: "doctor-tools-fallback" })}
        surfaceSwitcher={<button type="button">鍚庡彴鍒囨崲鑿滃崟</button>}
        apiClient={apiClient}
      />,
    );

    clickToolsTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("runtime manifest unavailable"));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("fallback inventory");
    expect(page).toHaveTextContent("Forbidden");
    expect(page).toHaveTextContent("search_latest_research");
  });

  it("renders the release dashboard task from the admin rail", async () => {
    const releaseDashboard = makeAdminReleaseDashboard();
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => releaseDashboard),
      getAdminReleaseGovernance: vi.fn(async () => makeAdminReleaseGovernance()),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release" })}
        doctor={makeState({ sessionId: "doctor-release" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(apiClient.getAdminReleaseDashboard).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveAttribute("data-task-id", "release");
    expect(page).toHaveTextContent("Release Dashboard");
    expect(page).toHaveTextContent("agent_policy_20260629_0");
    expect(page).toHaveTextContent("crc_safety_policy_v0");
    expect(page).toHaveTextContent("harness_20260629_001");
    expect(page).toHaveTextContent("literature_harness_20260630_001");
    expect(page).toHaveTextContent("feature_flag_or_pass");
    expect(page).toHaveTextContent("agent_policy_20260624_0");
    expect(page).toHaveTextContent("Step 11 observes readiness only");
    expect(page).toHaveTextContent("Clinical RAG ingest is disabled in Step 11");
  });

  it("renders release governance state and disabled execution controls", async () => {
    const releaseGovernance = makeAdminReleaseGovernance();
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => makeAdminReleaseDashboard()),
      getAdminReleaseGovernance: vi.fn(async () => releaseGovernance),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-governance" })}
        doctor={makeState({ sessionId: "doctor-governance" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(apiClient.getAdminReleaseGovernance).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("Release governance");
    expect(page).toHaveTextContent("release_intent_test");
    expect(page).toHaveTextContent("release_manager");
    expect(page).toHaveTextContent("clinical_safety_reviewer");
    expect(page).toHaveTextContent("release_audit_test");
    expect(page).toHaveTextContent("verified");
    for (const label of ["Execute release", "Execute rollback"]) {
      const action = within(page).getByText(label).closest("button");
      expect(action).toBeDisabled();
    }
    expect(page).toHaveTextContent("Record approval");
    expect(page).toHaveTextContent("Record rollback plan");
    expect(page).toHaveTextContent("Cancel intent");
  });

  it("creates a release governance intent and renders the updated read model", async () => {
    const emptyGovernance = makeAdminReleaseGovernance({
      intents: [],
      active_intent: null,
      audit_events: [],
    });
    const createdGovernance = makeAdminReleaseGovernance();
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => makeAdminReleaseDashboard()),
      getAdminReleaseGovernance: vi.fn(async () => emptyGovernance),
      createAdminReleaseIntent: vi.fn(async () => createdGovernance),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-create-intent" })}
        doctor={makeState({ sessionId: "doctor-create-intent" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("Create intent"));

    fireEvent.change(screen.getByLabelText("Requested by"), {
      target: { value: "release_admin" },
    });
    fireEvent.change(screen.getByLabelText("Intent reason"), {
      target: { value: "Prepare audited governance." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create intent" }));

    await waitFor(() => expect(apiClient.createAdminReleaseIntent).toHaveBeenCalledTimes(1));

    expect(apiClient.createAdminReleaseIntent).toHaveBeenCalledWith({
      requested_by: "release_admin",
      target_scope: "shadow",
      status: "pending_approval",
      reason: "Prepare audited governance.",
    });
    expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("release_intent_test");
  });

  it("does not fetch release dashboard until the release task is selected", () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => makeAdminReleaseDashboard()),
      getAdminReleaseGovernance: vi.fn(async () => makeAdminReleaseGovernance()),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-idle" })}
        doctor={makeState({ sessionId: "doctor-release-idle" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    expect(apiClient.getAdminReleaseDashboard).not.toHaveBeenCalled();
  });

  it("shows release dashboard loading state without fallback data", async () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(() => new Promise<AdminReleaseDashboardResponse>(() => {})),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-loading" })}
        doctor={makeState({ sessionId: "doctor-release-loading" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(apiClient.getAdminReleaseDashboard).toHaveBeenCalledTimes(1));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveTextContent("reading release dashboard");
    expect(page).not.toHaveTextContent("agent_policy_20260629_0");
  });

  it("shows release loading before top nav release fetch can reuse stale dashboard data", async () => {
    let requestCount = 0;
    let pageTextWhenTopNavFetch = "";
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(() => {
        requestCount += 1;
        if (requestCount === 2) {
          pageTextWhenTopNavFetch = screen.getByTestId("agent-admin-task-page").textContent ?? "";
          return new Promise<AdminReleaseDashboardResponse>(() => {});
        }
        return Promise.resolve(makeAdminReleaseDashboard());
      }),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-top-nav" })}
        doctor={makeState({ sessionId: "doctor-release-top-nav" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("agent_policy_20260629_0"));

    fireEvent.click(screen.getByRole("button", { name: "巡检总览" }));
    expect(screen.getByTestId("agent-admin-task-page")).toHaveAttribute("data-task-id", "overview");

    fireEvent.click(screen.getByRole("button", { name: "Release" }));

    await waitFor(() => expect(apiClient.getAdminReleaseDashboard).toHaveBeenCalledTimes(2));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(pageTextWhenTopNavFetch).toContain("reading release dashboard");
    expect(pageTextWhenTopNavFetch).not.toContain("agent_policy_20260629_0");
    expect(page).toHaveTextContent("reading release dashboard");
    expect(page).not.toHaveTextContent("agent_policy_20260629_0");
  });

  it("shows release dashboard error state without breaking the admin shell", async () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => {
        throw new Error("Forbidden");
      }),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-error" })}
        doctor={makeState({ sessionId: "doctor-release-error" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("release dashboard unavailable"));
    expect(screen.getByTestId("agent-admin-console")).toBeInTheDocument();
  });

  it("renders release mutation controls as disabled read-only actions", async () => {
    const apiClient = {
      getAdminTools: vi.fn(async () => makeAdminToolsManifest()),
      getAdminReleaseDashboard: vi.fn(async () => makeAdminReleaseDashboard()),
    };

    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-release-disabled" })}
        doctor={makeState({ sessionId: "doctor-release-disabled" })}
        surfaceSwitcher={<button type="button">admin surface switcher</button>}
        apiClient={apiClient}
      />,
    );

    clickReleaseTask();

    await waitFor(() => expect(screen.getByTestId("agent-admin-task-page")).toHaveTextContent("Record human sign-off"));

    const page = screen.getByTestId("agent-admin-task-page");
    for (const label of ["Record human sign-off", "Publish feature flag release", "Rollback release"]) {
      const action = within(page).getByText(label).closest("button");
      expect(action).toBeDisabled();
    }
  });

  it.each([
    { button: /学习/, taskId: "learning", required: ["发现论文", "人工审核", "写入知识库", "Run now", "一期不执行每日任务"] },
    { button: /Trace/, taskId: "trace", required: ["trace.start", "status.node", "latency panel", "event table"] },
    { button: /证据/, taskId: "evidence", required: ["证据池", "retrieval profile", "citation coverage", "RAG pipeline"] },
    { button: /设置只读/, taskId: "read-only", required: ["权限矩阵", "编辑规则", "disabled", "不创建第三种 graph scene"] },
  ])("renders $taskId as a distinct workbench", ({ button, taskId, required }) => {
    render(
      <AgentAdminView
        activeScene="doctor"
        patient={makeState({ sessionId: "patient-final" })}
        doctor={makeState({
          sessionId: "doctor-final",
          activeRunId: "run-final",
          statusNode: "tool_executor",
          references: [{ title: "final evidence", source: "RAG", confidence: 0.77 }],
        })}
        surfaceSwitcher={<button type="button">后台切换菜单</button>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: button }));

    const page = screen.getByTestId("agent-admin-task-page");
    expect(page).toHaveAttribute("data-task-id", taskId);
    for (const text of required) {
      expect(page).toHaveTextContent(text);
    }
  });

  it("builds the complete task and manifest model", () => {
    expect(AGENT_ADMIN_TASKS.map((task) => task.id)).toEqual([
      "overview",
      "sessions",
      "memory",
      "rules",
      "tools",
      "learning",
      "trace",
      "evidence",
      "release",
      "read-only",
    ]);
    expect(ADMIN_NAV_ITEMS.map((item) => item.key)).toEqual(["overview", "trace", "learning", "release", "read-only"]);
    expect(RULE_CATALOG.some((rule) => rule.id === "routing.intent.knowledge_query")).toBe(true);
    expect(TOOL_INVENTORY.some((tool) => tool.name === "search_latest_research")).toBe(true);
    expect(buildRuleGroupRows().some((row) => row.name === "路由规则")).toBe(true);
    expect(buildRuleCatalogRows()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "routing.intent.knowledge_query",
          editable: false,
          ownerModule: "routing",
        }),
      ]),
    );
    expect(buildRuleCatalogGroups()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "路由规则",
          rules: expect.arrayContaining([expect.objectContaining({ id: "routing.intent.knowledge_query" })]),
        }),
      ]),
    );
    expect(buildToolInventoryRows()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "search_latest_research",
          dependency: "WEB_SEARCH_ENABLED controls Web search reachability",
        }),
      ]),
    );
    expect(buildToolGroupRows()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "Web search", count: expect.any(Number), status: "WEB_SEARCH_ENABLED 控制" }),
      ]),
    );
  });

  it("derives session, memory, evidence, learning, and permission rows", () => {
    const state = makeState({
      sessionId: "session-a",
      snapshotVersion: 3,
      activeRunId: "run-a",
      statusNode: "planner",
      references: [{ title: "指南来源", source: "RAG", confidence: 0.88 }],
      contextState: {
        summary_memory: "患者关注治疗方案",
        structured_summary: {
          immutable_info: ["乳腺癌病史"],
          dynamic_info: ["正在等待检查"],
          anchor_events: ["首次问诊"],
        },
      },
      contextMaintenance: { status: "completed", message: "已整理" },
    });

    expect(buildSessionSummary("患者", state)).toMatchObject({
      label: "患者",
      sessionId: "session-a",
      snapshot: "v3",
      status: "running",
      activeRunId: "run-a",
      currentNode: "planner",
    });
    expect(buildMemoryRows("患者", state).map((row) => row.type)).toContain("永久事实");
    expect(buildMemoryFactRows("患者", state)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "患者", type: "永久事实", value: "乳腺癌病史" }),
      ]),
    );
    expect(buildEvidenceRows(state)[0]).toMatchObject({ title: "指南来源", confidence: "88%" });
    expect(buildLearningReadiness().map((row) => row.label)).toContain("调度器配置");
    expect(buildTraceRows(state)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "trace.start", detail: "run-a" }),
        expect.objectContaining({ name: "status.node", detail: "planner" }),
      ]),
    );
    expect(buildPermissionRows().find((row) => row.label === "编辑规则")?.state).toBe("disabled");
  });
});

describe("AgentAdminModel", () => {
  it("exposes admin nav, grouped manifests, and permission rows", () => {
    expect(ADMIN_NAV_ITEMS.map((item) => item.key)).toEqual(["overview", "trace", "learning", "release", "read-only"]);
    expect(buildRuleGroupRows().map((row) => row.name)).toContain("路由规则");
    expect(buildToolGroupRows()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "Web search", count: expect.any(Number), status: "WEB_SEARCH_ENABLED 控制" }),
      ]),
    );
    expect(buildToolReachabilityRows()).toEqual(buildToolGroupRows());
    expect(buildPermissionRows().map((row) => row.label)).toEqual(
      expect.arrayContaining(["编辑规则", "启停工具", "运行学习任务"]),
    );
  });

  it("renders memory fact rows from dictionary and object summary shapes", () => {
    const state = makeState({
      contextState: {
        structured_summary: {
          immutable_info: {
            diagnosis: "乳腺癌病史",
            age: 52,
          },
          dynamic_info: {
            symptom: { text: "疼痛减轻" },
            score: 3,
          },
          anchor_events: [
            { title: "首次确诊", date: "2026-01-01" },
            { code: "fallback-event", payload: { stage: "follow-up" } },
          ],
        },
      },
    });

    expect(buildMemoryRows("患者", state)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "永久事实", value: "2 条永久事实" }),
        expect.objectContaining({ type: "动态事实", value: "2 条动态事实" }),
        expect.objectContaining({ type: "锚点事件", value: "2 个锚点事件" }),
      ]),
    );
    expect(buildMemoryFactRows("患者", state)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "永久事实", value: "diagnosis: 乳腺癌病史" }),
        expect.objectContaining({ type: "永久事实", value: "age: 52" }),
        expect.objectContaining({ type: "动态事实", value: "symptom: 疼痛减轻" }),
        expect.objectContaining({ type: "动态事实", value: "score: 3" }),
        expect.objectContaining({ type: "锚点事件", value: "首次确诊" }),
        expect.objectContaining({
          type: "锚点事件",
          value: '{"code":"fallback-event","payload":{"stage":"follow-up"}}',
        }),
      ]),
    );
  });

  it("derives memory automation summary from patient and doctor context", () => {
    const patient = makeState({
      sessionId: "patient-memory-auto",
      snapshotVersion: 5,
      contextState: {
        summary_memory: "患者关注检查方案",
        summary_memory_cursor: 3,
        structured_summary: {
          immutable_info: { diagnosis: "乳腺癌病史", age: 52 },
          dynamic_info: ["正在等待检查"],
          anchor_events: [{ title: "首次确诊" }],
        },
      },
      contextMaintenance: { status: "failed", message: "维护失败", error: "summary timeout" },
    });
    const doctor = makeState({
      sessionId: "doctor-memory-auto",
      snapshotVersion: 2,
      contextState: {
        summary_memory: "医生侧保留治疗目标",
        summary_memory_cursor: 2,
        structured_summary: {
          immutable_info: ["治疗目标明确"],
          dynamic_info: { preference: "需要比较方案" },
          anchor_events: [],
        },
      },
      contextMaintenance: { status: "running", message: "正在刷新医生侧上下文" },
    });

    expect(buildMemoryAutomationSummary(patient, doctor)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: "摘要状态", value: "待刷新", tone: "warning" }),
        expect.objectContaining({ label: "永久事实", value: "3", detail: "患者 2 / 医生 1" }),
        expect.objectContaining({ label: "动态事实", value: "2", detail: "患者 1 / 医生 1" }),
        expect.objectContaining({ label: "锚点事件", value: "1", detail: "患者 1 / 医生 0" }),
        expect.objectContaining({ label: "维护状态", value: "failed", tone: "warning" }),
      ]),
    );
  });

  it("derives five memory layer rows with counts and visible states", () => {
    const patient = makeState({
      sessionId: "patient-memory-layers",
      snapshotVersion: 4,
      contextState: {
        summary_memory: "患者关注检查方案",
        summary_memory_cursor: 4,
        structured_summary: {
          immutable_info: ["长期诊断信息"],
          dynamic_info: [null],
        },
      },
      contextMaintenance: { status: "failed", message: "维护失败", error: "summary timeout" },
    });
    const doctor = makeState({ sessionId: "doctor-memory-layers" });

    const rows = buildMemoryLayerRows(patient, doctor);

    expect(rows.map((row) => row.label)).toEqual(["摘要记忆", "永久事实", "动态事实", "锚点事件", "维护日志"]);
    expect(rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          label: "摘要记忆",
          sourceKey: "summary_memory",
          patientCount: 1,
          doctorCount: 0,
          state: "active",
        }),
        expect.objectContaining({
          label: "动态事实",
          sourceKey: "dynamic_info",
          patientCount: 0,
          doctorCount: 0,
          state: "unstructured",
        }),
        expect.objectContaining({
          label: "维护日志",
          sourceKey: "contextMaintenance",
          patientCount: 1,
          doctorCount: 0,
          state: "failed",
        }),
      ]),
    );
  });

  it("derives memory lifecycle rows for collection through stale checks", () => {
    const patient = makeState({
      sessionId: "patient-memory-flow",
      snapshotVersion: 4,
      messages: [{ cursor: "1", type: "human", content: "我想了解检查方案", assetRefs: [] }],
      contextState: {
        summary_memory: "患者关注检查方案",
        summary_memory_cursor: 3,
        structured_summary: {
          immutable_info: { diagnosis: "乳腺癌病史" },
          dynamic_info: ["正在等待检查"],
          anchor_events: [{ title: "首次确诊" }],
        },
      },
      contextMaintenance: { status: "failed", message: "维护失败", error: "summary timeout" },
    });
    const doctor = makeState();

    const rows = buildMemoryLifecycleRows(patient, doctor);

    expect(rows.map((row) => row.stage)).toEqual(["收集", "摘要", "结构化", "同步", "过期检查"]);
    expect(rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ stage: "收集", state: "active", patient: "1 messages", doctor: "会话未创建" }),
        expect.objectContaining({ stage: "同步", state: "stale", patient: "stale" }),
        expect.objectContaining({ stage: "过期检查", state: "failed", patient: "failed", doctor: "idle" }),
      ]),
    );
  });

  it("derives memory visualization rows for active empty unstructured and failed memory", () => {
    const patient = makeState({
      sessionId: "patient-memory-visualization",
      snapshotVersion: 4,
      contextState: {
        summary_memory: "患者关注检查方案",
        summary_memory_cursor: 4,
        structured_summary: {
          immutable_info: { diagnosis: "乳腺癌病史" },
          dynamic_info: [null],
          anchor_events: [],
        },
      },
      contextMaintenance: { status: "failed", message: "维护失败", error: "summary timeout" },
    });
    const doctor = makeState({ sessionId: "doctor-memory-visualization" });

    expect(buildMemoryVisualizationRows(patient, doctor)).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          content: "患者关注检查方案",
          type: "摘要记忆",
          source: "患者",
          retentionReason: "压缩会话上下文",
          state: "active",
        }),
        expect.objectContaining({
          content: "diagnosis: 乳腺癌病史",
          type: "永久事实",
          source: "患者",
          retentionReason: "稳定事实",
          state: "active",
        }),
        expect.objectContaining({
          content: "结构化字段不可读",
          type: "动态事实",
          source: "患者",
          state: "unstructured",
        }),
        expect.objectContaining({
          content: "summary timeout",
          type: "维护日志",
          source: "患者",
          retentionReason: "自动维护状态",
          state: "failed",
        }),
        expect.objectContaining({
          content: "暂无摘要记忆",
          type: "摘要记忆",
          source: "医生",
          state: "empty",
        }),
      ]),
    );
  });

  it("renders read-only permissions from the permission model", () => {
    const permissionRows = buildPermissionRows().filter((row) =>
      ["\u7f16\u8f91\u89c4\u5219", "\u542f\u505c\u5de5\u5177", "\u8fd0\u884c\u5b66\u4e60\u4efb\u52a1"].includes(row.label),
    );
    expect(permissionRows).toHaveLength(3);

    render(
      <AgentAdminView
        activeScene="patient"
        patient={makeState({ sessionId: "patient-session-3" })}
        doctor={makeState({ sessionId: "doctor-session-3" })}
        surfaceSwitcher={<button type="button">鍚庡彴鍒囨崲鑿滃崟</button>}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /lock/ }));

    const taskPage = screen.getByTestId("agent-admin-task-page");
    expect(taskPage).toHaveAttribute("data-task-id", "read-only");
    expect(taskPage).toHaveTextContent("只读边界");
    expect(taskPage).toHaveTextContent("admin token / feature flags / read-only boundary");
    expect(taskPage).toHaveTextContent("Graph scene 保持为 patient");
  });
});
