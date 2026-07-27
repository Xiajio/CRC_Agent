import { describe, expect, it, vi } from "vitest";

import { createApiClient } from "./client";
import type {
  AdminAutoResearchRun,
  AdminAutoResearchRunResponse,
  AdminAutoResearchRunsResponse,
  AdminCohortFeasibilityRequest,
  AdminCreateAutoResearchRunRequest,
  AdminCreateAutoResearchRunResponse,
  AdminCreateLearningJobRequest,
  AdminReleaseClosureResponse,
  AdminReleaseMonitoringResponse,
} from "./types";

function jsonResponse(payload: unknown): Response {
  return {
    ok: true,
    json: vi.fn(async () => payload),
  } as unknown as Response;
}

function releaseClosureResponse(): AdminReleaseClosureResponse {
  return {
    status: "ready_to_close",
    latest_release: {
      intent_id: "intent-1",
      release_execution_id: "release-exec-1",
      released_at: "2026-07-03T09:00:00+08:00",
      rollback_execution_id: null,
      rolled_back_at: null,
    },
    closure_gate: {
      allowed: true,
      status: "ready_to_close",
      reasons: [],
      checks: [
        {
          name: "required_checks_complete",
          status: "pass",
          reason: "All required post-release checks are recorded.",
        },
      ],
    },
    latest_closure: null,
    latest_evidence_package: null,
    closures: [],
    evidence_packages: [],
    integrity: { status: "verified", warnings: [] },
    runtime: {
      auth: "admin",
      source: "reports/release_closure",
      mode: "post_release_closure",
    },
  };
}

function releaseMonitoringResponse(): AdminReleaseMonitoringResponse {
  return {
    status: "monitoring",
    latest_release: {
      intent_id: "intent-1",
      execution_id: "release-exec-1",
      released_at: "2026-07-03T09:00:00+08:00",
      flag_enabled: true,
      rollback_plan_id: "rollback-1",
    },
    required_checks: [
      {
        check_type: "p0_harness_replay",
        status: "missing",
        latest_check_id: null,
        reason: "Required post-release check has not been recorded.",
      },
    ],
    checks: [],
    alerts: [],
    rollback_trigger_candidate: null,
    acknowledgements: [],
    integrity: { status: "verified", warnings: [] },
    runtime: {
      auth: "admin",
      source: "reports/release_monitoring",
      mode: "post_release_monitoring",
    },
  };
}

function autoResearchRun(): AdminAutoResearchRun {
  return {
    run_id: "auto_research_run_001",
    request_hash: `sha256:${"a".repeat(64)}`,
    request: {
      request_id: "research_request_001",
      project_id: "research_crc_001",
      question: "Which source-grounded CRC hypotheses merit controlled follow-up?",
      requested_by: "admin_operator",
      idempotency_key: "auto-research-001",
      max_sources: 8,
      max_hypotheses: 3,
      max_iterations: 2,
      deidentified: true,
    },
    status: "completed_shadow",
    created_at: "2026-07-19T02:00:00+00:00",
    completed_at: "2026-07-19T02:01:00+00:00",
    stages: [
      {
        name: "literature_search",
        status: "completed",
        started_at: "2026-07-19T02:00:00+00:00",
        completed_at: "2026-07-19T02:00:10+00:00",
        summary: "Retrieved one verified PubMed abstract.",
        error: null,
      },
    ],
    sources: [
      {
        source_id: "research_source_001",
        title: "CRC evidence source",
        url: "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        abstract: "A source abstract used only for a shadow research fixture.",
        journal: "Fixture Journal",
        publication_year: "2026",
        source_type: "pubmed",
        query: "colorectal cancer",
        retrieved_at: "2026-07-19T02:00:09+00:00",
        pmid: "12345678",
      },
    ],
    hypotheses: [
      {
        hypothesis_id: "research_hypothesis_001",
        statement: "A source-grounded candidate hypothesis.",
        rationale: "The verified abstract supports controlled follow-up.",
        testable_prediction: "A predefined aggregate analysis can falsify the candidate.",
        supporting_source_ids: ["research_source_001"],
        counterevidence_source_ids: [],
        iteration: 1,
        review: {
          verdict: "advance",
          evidence_support_score: 0.8,
          novelty_score: 0.6,
          testability_score: 0.9,
          safety_risk: "shadow-only interpretation required",
          critique: "Requires independent human review.",
          revision_instructions: "",
        },
      },
    ],
    study_plans: [
      {
        plan_id: "research_plan_001",
        hypothesis_id: "research_hypothesis_001",
        study_type: "aggregate retrospective analysis",
        objective: "Falsify the candidate using approved aggregate data.",
        required_data: ["approved aggregate projection"],
        analysis_steps: ["Pre-register the aggregate analysis."],
        success_criteria: ["Meet the predefined effect threshold."],
        safety_constraints: ["Do not return patient-level rows."],
        execution_status: "not_executed",
      },
    ],
    report_markdown: "# Shadow report\n\nPending human review.",
    iteration_count: 1,
    provenance: {
      pipeline_version: "shadow_auto_research_v1",
      retriever: "pubmed",
      reasoner: "fixture",
    },
    human_review_status: "needs_human_review",
    mode: "shadow_only",
    applies_automatically: false,
    clinical_default_path_mutated: false,
    patient_level_rows_returned: false,
  };
}

describe("createApiClient", () => {
  // @ts-expect-error Review cleanup: admin auth must continue to flow through headers, not a separate option.
  createApiClient({ adminToken: "admin-token" });

  it("gets admin release closure", async () => {
    const payload = releaseClosureResponse();
    const fetch = vi.fn().mockResolvedValue(jsonResponse(payload));
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl: fetch,
      headers: { Authorization: "Bearer admin-token" },
    });

    await expect(client.getAdminReleaseClosure()).resolves.toEqual(payload);

    expect(fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/release-closure",
      { headers: { Authorization: "Bearer admin-token" } },
    );
  });

  it("records admin release closure with JSON request body", async () => {
    const payload = releaseClosureResponse();
    const fetch = vi.fn().mockResolvedValue(jsonResponse(payload));
    let latestInit: RequestInit | undefined;
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      latestInit = init;
      return jsonResponse(payload);
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer admin-token" },
    });
    const request = {
      intent_id: "intent-1",
      release_execution_id: "release-exec-1",
      closure_status: "accepted",
      closed_by: "release_manager",
      rationale: "Required checks passed.",
      idempotency_key: "close-1",
    } as const;

    await expect(client.recordAdminReleaseClosure(request)).resolves.toEqual(payload);

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/release-closure/closures",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(request),
      }),
    );
    const headers = latestInit?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer admin-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("loads admin tools with configured Authorization headers", async () => {
    const payload = {
      tools: [
        {
          name: "search_clinical_guidelines",
          category: "rag",
          registries: ["graph", "graph_web"],
          route_targets: ["knowledge"],
          graph_scope: "both",
          planner_aliases: ["search_clinical_guidelines", "search"],
          requires_web: false,
          available: true,
          state: "available",
        },
      ],
      groups: [{ category: "rag", count: 1, available_count: 1 }],
      runtime: {
        web_search_enabled: true,
        auth: "admin",
        source: "src.tools.manifest",
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getAdminTools()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/tools",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });

  it("loads admin learning jobs with configured Authorization headers", async () => {
    const payload = {
      jobs: [],
      candidates: [],
      integrity: { status: "verified", warnings: [] },
      disabled_actions: [],
      actions: {
        apply: { enabled: false, reason: "shadow_learning_jobs_only" },
        train: { enabled: false, reason: "shadow_learning_jobs_only" },
      },
      runtime: {
        auth: "admin",
        source: "reports/learning_jobs",
        mode: "shadow_learning_jobs",
      },
    };
    const fetchImpl = vi.fn(async () => jsonResponse(payload));
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer admin-token" },
    });

    await expect(client.getAdminLearningJobs()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/learning-jobs",
      { headers: { Authorization: "Bearer admin-token" } },
    );
  });

  it("creates an admin learning job with admin headers and a JSON body", async () => {
    const payload = {
      job: { job_id: "learning_job_001", status: "shadow_only" },
      signals: [],
      candidates: [],
    };
    let latestInit: RequestInit | undefined;
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      latestInit = init;
      return jsonResponse(payload);
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer admin-token" },
    });
    const request: AdminCreateLearningJobRequest = {
      signals: [
        {
          signal_type: "doctor_action_trace",
          source_ref: {
            kind: "doctor_action_trace",
            id: "doctor_action_trace_crc_shadow_001",
          },
          reason_code: "unsafe_disposition",
          target_area: "prompt",
          severity: "high",
          summary: "Aggregate deidentified shadow signal.",
          deidentified: true,
          created_at: "2026-07-09T10:00:00+08:00",
        },
      ],
      requested_by: "admin_user",
      idempotency_key: "learning-job-001",
    };

    await expect(client.createAdminLearningJob(request)).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/learning-jobs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(request),
      }),
    );
    const headers = latestInit?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer admin-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("evaluates cohort feasibility with admin headers and a JSON body", async () => {
    const payload = {
      result_id: "cohort_feasibility_001",
      request_id: "cohort_request_crc_001",
      status: "needs_review",
      estimated_count: 1,
      patient_level_rows_returned: false,
    };
    let latestInit: RequestInit | undefined;
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      latestInit = init;
      return jsonResponse(payload);
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer admin-token" },
    });
    const request: AdminCohortFeasibilityRequest = {
      request_id: "cohort_request_crc_001",
      project_id: "research_crc_001",
      question: "Is there enough aggregate CRC data for feasibility review?",
      cohort_criteria: {
        condition: "colorectal_cancer_or_crc_triage_risk",
        required_features: ["rectal_bleeding"],
      },
      data_scope: {
        source: "patient_record_projection",
        patient_level_export_requested: false,
        deidentified_only: true,
      },
      version_refs: {
        projection_version: "patient_record_projection_v0",
        clinical_safety_policy_version: "crc_safety_policy_v0",
      },
    };

    await expect(client.evaluateAdminCohortFeasibility(request)).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/research/cohort-feasibility",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(request),
      }),
    );
    const headers = latestInit?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer admin-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("loads admin auto-research runs with configured Authorization headers", async () => {
    const payload: AdminAutoResearchRunsResponse = {
      runs: [autoResearchRun()],
      integrity: { status: "verified", warnings: [] },
      runtime: {
        auth: "admin",
        source: "reports/auto_research",
        mode: "shadow_auto_research",
      },
    };
    const fetchImpl = vi.fn(async () => jsonResponse(payload));
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer admin-token" },
    });

    await expect(client.getAdminAutoResearchRuns()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/research/runs",
      { headers: { Authorization: "Bearer admin-token" } },
    );
  });

  it("creates an admin auto-research run with admin headers and a JSON body", async () => {
    const payload: AdminCreateAutoResearchRunResponse = {
      run: autoResearchRun(),
      reused: false,
      integrity: { status: "verified", warnings: [] },
      runtime: {
        auth: "admin",
        source: "reports/auto_research",
        mode: "shadow_auto_research",
      },
    };
    let latestInit: RequestInit | undefined;
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      latestInit = init;
      return jsonResponse(payload);
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer admin-token" },
    });
    const request: AdminCreateAutoResearchRunRequest = {
      request_id: "research_request_001",
      project_id: "research_crc_001",
      question: "Which source-grounded CRC hypotheses merit controlled follow-up?",
      requested_by: "admin_operator",
      idempotency_key: "auto-research-001",
      max_sources: 8,
      max_hypotheses: 3,
      max_iterations: 2,
      deidentified: true,
    };

    await expect(client.createAdminAutoResearchRun(request)).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/research/runs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(request),
      }),
    );
    const headers = latestInit?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer admin-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("loads one admin auto-research run with an encoded run id", async () => {
    const payload: AdminAutoResearchRunResponse = {
      run: autoResearchRun(),
      integrity: { status: "verified", warnings: [] },
      runtime: {
        auth: "admin",
        source: "reports/auto_research",
        mode: "shadow_auto_research",
      },
    };
    const fetchImpl = vi.fn(async () => jsonResponse(payload));
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer admin-token" },
    });

    await expect(client.getAdminAutoResearchRun("auto research/run 001")).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/research/runs/auto%20research%2Frun%20001",
      { headers: { Authorization: "Bearer admin-token" } },
    );
  });

  it("loads admin release dashboard with configured Authorization headers", async () => {
    const payload = {
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
        reason: "Step 11 is read-only",
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
      ],
      blocking_gates: [],
      disabled_actions: [],
      runtime: {
        auth: "admin",
        source: "reports/static_release_artifacts",
        mode: "read_only",
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getAdminReleaseDashboard()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/release-dashboard",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });

  it("loads admin release governance with configured Authorization headers", async () => {
    const payload = {
      dashboard_snapshot: {
        release_decision: "feature_flag_or_pass",
        rollback_target: "agent_policy_20260624_0",
        hard_fail_count: 0,
        literature_status: "shadow_only",
      },
      intents: [],
      active_intent: null,
      approvals: [],
      required_approvals: [
        {
          role: "release_manager",
          status: "missing",
          latest_decision: null,
        },
      ],
      rollback_plan: null,
      audit_events: [],
      integrity: { status: "verified", warnings: [] },
      disabled_execution_actions: [
        {
          id: "execute_release",
          label: "Execute release",
          disabled: true,
          reason: "Step 12 records governance only.",
        },
      ],
      runtime: {
        auth: "admin",
        source: "reports/release_governance",
        mode: "audit_only",
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getAdminReleaseGovernance()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/release-governance",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });

  it("records admin release governance actions with JSON request bodies", async () => {
    const payload = {
      dashboard_snapshot: { release_decision: "feature_flag_or_pass" },
      intents: [],
      active_intent: null,
      approvals: [],
      required_approvals: [],
      rollback_plan: null,
      audit_events: [],
      integrity: { status: "verified", warnings: [] },
      disabled_execution_actions: [],
      runtime: {
        auth: "admin",
        source: "reports/release_governance",
        mode: "audit_only",
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const calls: Array<[RequestInfo | URL, RequestInit | undefined]> = [];
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push([input, init]);
      return response;
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });
    const intentPayload = {
      requested_by: "admin_operator",
      target_scope: "shadow" as const,
      status: "pending_approval" as const,
      reason: "Prepare audited governance.",
    };
    const approvalPayload = {
      approver_role: "release_manager" as const,
      decision: "approve" as const,
      reason: "Release dashboard gates are clear.",
      signed_by: "release_admin",
    };
    const rollbackPayload = {
      owner: "release_manager",
      status: "accepted" as const,
      verification_steps: [
        "Confirm the active release report id.",
        "Run P0 harness before rollback execution.",
      ],
    };
    const cancelPayload = {
      actor: "release_manager",
      reason: "Release window closed.",
    };

    await expect(client.createAdminReleaseIntent(intentPayload)).resolves.toEqual(payload);
    await expect(client.recordAdminReleaseApproval("intent-1", approvalPayload)).resolves.toEqual(payload);
    await expect(client.recordAdminReleaseRollbackPlan("intent-1", rollbackPayload)).resolves.toEqual(payload);
    await expect(client.cancelAdminReleaseIntent("intent-1", cancelPayload)).resolves.toEqual(payload);

    expect(calls).toHaveLength(4);
    expect(calls[0]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-governance/intents",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(intentPayload),
      },
    ]);
    expect(calls[1]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-governance/intents/intent-1/approvals",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(approvalPayload),
      },
    ]);
    expect(calls[2]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-governance/intents/intent-1/rollback-plan",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(rollbackPayload),
      },
    ]);
    expect(calls[3]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-governance/intents/intent-1/cancel",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(cancelPayload),
      },
    ]);
    for (const [, init] of calls) {
      const headers = init?.headers as Headers;
      expect(headers.get("Authorization")).toBe("Bearer dev-token");
      expect(headers.get("Content-Type")).toBe("application/json");
    }
  });

  it("loads admin release execution with configured Authorization headers", async () => {
    const payload = {
      governance: {
        active_intent_id: "intent-1",
        derived_status: "approved",
        required_approvals_complete: true,
        rollback_plan_id: "rollback-1",
      },
      preflight: {
        release: { allowed: true, reasons: [] },
        rollback: { allowed: false, reasons: ["no successful release execution exists"] },
      },
      feature_flag_state: null,
      requests: [],
      results: [],
      audit_events: [],
      integrity: { status: "verified", warnings: [] },
      runtime: {
        auth: "admin",
        source: "reports/release_execution",
        mode: "controlled_local_execution",
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getAdminReleaseExecution()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/release-execution",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });

  it("executes admin release and rollback with JSON request bodies", async () => {
    const payload = {
      governance: {
        active_intent_id: "intent-1",
        derived_status: "approved",
        required_approvals_complete: true,
        rollback_plan_id: "rollback-1",
      },
      preflight: {
        release: { allowed: false, reasons: ["feature flag is already enabled"] },
        rollback: { allowed: true, reasons: [] },
      },
      feature_flag_state: {
        flag_name: "doctor_review_cockpit_v0",
        enabled: true,
        scope: "feature_flag_candidate",
        source_intent_id: "intent-1",
        source_execution_id: "release_exec_1",
        rollback_target: "agent_policy_20260624_0",
        updated_by: "release_manager",
        updated_at: "2026-07-03T09:00:00+08:00",
      },
      requests: [],
      results: [],
      audit_events: [],
      integrity: { status: "verified", warnings: [] },
      runtime: {
        auth: "admin",
        source: "reports/release_execution",
        mode: "controlled_local_execution",
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const calls: Array<[RequestInfo | URL, RequestInit | undefined]> = [];
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push([input, init]);
      return response;
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });
    const releaseRequest = {
      intent_id: "intent-1",
      requested_by: "release_manager",
      idempotency_key: "release-1",
      reason: "Approved release.",
      expected_rollback_plan_id: "rollback-1",
    };
    const rollbackRequest = {
      intent_id: "intent-1",
      requested_by: "release_manager",
      idempotency_key: "rollback-1",
      reason: "Rollback to the accepted target.",
      expected_rollback_plan_id: "rollback-1",
    };

    await expect(client.executeAdminRelease(releaseRequest)).resolves.toEqual(payload);
    await expect(client.executeAdminReleaseRollback(rollbackRequest)).resolves.toEqual(payload);

    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-execution/release",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(releaseRequest),
      },
    ]);
    expect(calls[1]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-execution/rollback",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(rollbackRequest),
      },
    ]);
    for (const [, init] of calls) {
      const headers = init?.headers as Headers;
      expect(headers.get("Authorization")).toBe("Bearer dev-token");
      expect(headers.get("Content-Type")).toBe("application/json");
    }
  });

  it("loads admin release monitoring with configured Authorization headers", async () => {
    const payload = releaseMonitoringResponse();
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getAdminReleaseMonitoring()).resolves.toEqual(payload);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/admin/release-monitoring",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });

  it("records admin release monitoring checks and acknowledges encoded alerts with JSON request bodies", async () => {
    const payload = releaseMonitoringResponse();
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const calls: Array<[RequestInfo | URL, RequestInit | undefined]> = [];
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      calls.push([input, init]);
      return response;
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });
    const checkRequest = {
      intent_id: "intent-1",
      execution_id: "release-exec-1",
      check_type: "p0_harness_replay" as const,
      status: "pass" as const,
      observed_by: "release_operator",
      summary: "P0 replay passed after release.",
      evidence_refs: ["reports/harness/harness_20260703_001.json"],
      metrics: { passed_cases: 5 },
      idempotency_key: "monitoring-check-1",
    };
    const acknowledgementRequest = {
      acknowledged_by: "release_operator",
      disposition: "investigating" as const,
      reason: "Reviewing the alert.",
    };

    await expect(client.recordAdminReleaseMonitoringCheck(checkRequest)).resolves.toEqual(payload);
    await expect(
      client.acknowledgeAdminReleaseMonitoringAlert("alert/with space", acknowledgementRequest),
    ).resolves.toEqual(payload);

    expect(calls).toHaveLength(2);
    expect(calls[0]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-monitoring/checks",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(checkRequest),
      },
    ]);
    expect(calls[1]).toEqual([
      "http://127.0.0.1:8000/api/admin/release-monitoring/alerts/alert%2Fwith%20space/acknowledge",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(acknowledgementRequest),
      },
    ]);
    for (const [, init] of calls) {
      const headers = init?.headers as Headers;
      expect(headers.get("Authorization")).toBe("Bearer dev-token");
      expect(headers.get("Content-Type")).toBe("application/json");
    }
  });

  it("downloads session assets through fetch with configured Authorization headers", async () => {
    const body = new Blob(["asset-content"], { type: "text/plain" });
    const response = {
      ok: true,
      blob: vi.fn(async () => body),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    const blob = await client.downloadSessionAsset("sess-1", "asset-1");

    expect(blob).toBe(body);
    expect(response.blob).toHaveBeenCalledTimes(1);
    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/sessions/sess-1/assets/asset-1",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });

  it("saves crc triage assessments through the patient session endpoint", async () => {
    const payload = {
      patient_id: 101,
      patient_version: 2,
      projection_version: 2,
      event_ids: ["event-1"],
      record_id: 9,
      reused: false,
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    let latestInit: RequestInit | undefined;
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      latestInit = init;
      return response;
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });
    const assessment = {
      record_type: "crc_triage_assessment" as const,
      chief_complaint: "bleeding",
      symptom_group: "crc_triage",
      risk_level: "medium",
      disposition: "urgent_gi_clinic",
      red_flags: ["rectal_bleeding"],
      known_crc_signals: { rectal_bleeding: true },
      suggested_tests: ["colonoscopy"],
      missing_information: [],
      qa_summary: [],
      patient_summary: "summary",
      next_step: "urgent_gi_clinic",
      source_session_id: "sess-1",
      source_subflow: "crc_triage" as const,
    };

    await expect(client.saveCrcTriageAssessment("sess-1", { assessment })).resolves.toEqual(payload);

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/sessions/sess-1/crc-triage/assessments",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify({ assessment }),
      },
    );
    const headers = latestInit?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer dev-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });

  it("loads patient records through the current patient session endpoint", async () => {
    const payload = {
      items: [
        {
          record_id: 1,
          patient_id: 101,
          asset_id: 9,
          record_type: "crc_triage_assessment",
          document_type: "crc_triage_assessment",
          ingest_decision: "record_only",
          snapshot_contributed: false,
          conflict_detected: false,
          summary_text: "建议尽快消化专科评估。",
          source: "patient_generated",
          created_at: "2026-06-25T08:00:00Z",
        },
      ],
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getSessionPatientRecords("sess-1")).resolves.toEqual(payload);

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/sessions/sess-1/patient-records",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });

  it("loads patient care cards through the current patient session endpoint", async () => {
    const payload = {
      focusMetrics: ["留意便血或黑便是否加重"],
      periodicChecks: ["尽快预约消化专科门诊"],
      dailyActions: ["记录便血颜色、次数和伴随症状"],
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });

    await expect(client.getSessionCareCards("sess-1")).resolves.toEqual(payload);

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/sessions/sess-1/care-cards",
      { headers: { Authorization: "Bearer dev-token" } },
    );
  });
  it("loads doctor review through the session endpoint", async () => {
    const payload = {
      feature_flag: "doctor_review_cockpit_v0",
      patient_id: 101,
      session_id: "sess-doctor",
      timeline: [],
      assertions: [],
      draft: {
        draft_id: "draft-101",
        sections: [],
      },
      available_actions: ["accept"],
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    const fetchImpl = vi.fn(async () => response);
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
    });

    await expect(client.getDoctorReview("sess-doctor")).resolves.toEqual(payload);

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/sessions/sess-doctor/doctor-review",
      { headers: undefined },
    );
  });

  it("records doctor action traces with a JSON request body", async () => {
    const payload = {
      patient_id: 101,
      patient_version: 3,
      projection_version: 4,
      event_ids: ["event-1"],
      trace: {
        trace_id: "trace-1",
        patient_id: 101,
        session_id: "sess-doctor",
        timestamp: "2026-06-29T04:00:00Z",
        action_type: "accept",
        target_object: null,
        target_refs: {
          assertion_id: "assertion-1",
        },
        before_after: null,
        reason_code: "unsupported_claim",
        reviewer_role: "physician_reviewer",
        deidentified: true,
      },
    };
    const response = {
      ok: true,
      json: vi.fn(async () => payload),
    } as unknown as Response;
    let latestInit: RequestInit | undefined;
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      latestInit = init;
      return response;
    });
    const client = createApiClient({
      baseUrl: "http://127.0.0.1:8000",
      fetchImpl,
      headers: { Authorization: "Bearer dev-token" },
    });
    const request = {
      action_type: "accept" as const,
      target_refs: {
        assertion_id: "assertion-1",
      },
      reason_code: "unsupported_claim" as const,
    };

    await expect(client.recordDoctorActionTrace("sess-doctor", request)).resolves.toEqual(payload);

    expect(fetchImpl).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/sessions/sess-doctor/doctor-review/action-traces",
      {
        method: "POST",
        headers: expect.any(Headers),
        body: JSON.stringify(request),
      },
    );
    const headers = latestInit?.headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer dev-token");
    expect(headers.get("Content-Type")).toBe("application/json");
  });
});
