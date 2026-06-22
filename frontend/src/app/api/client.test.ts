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
});
