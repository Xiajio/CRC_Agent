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
