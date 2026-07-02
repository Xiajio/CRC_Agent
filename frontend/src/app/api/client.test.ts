import { describe, expect, it, vi } from "vitest";

import { createApiClient } from "./client";

describe("createApiClient", () => {
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
