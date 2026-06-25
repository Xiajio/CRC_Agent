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
});
