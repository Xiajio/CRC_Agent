import { describe, expect, it } from "vitest";

import type { SessionState } from "../../app/api/types";
import { createInitialSessionState } from "../../app/store/stream-reducer";
import { buildTraceRows } from "./agent-admin-model";

describe("AgentAdmin model", () => {
  it("does not invent demo latency values when trace events have no duration", () => {
    const state: SessionState = {
      ...createInitialSessionState(),
      runTrace: {
        traceId: "trace-1",
        runId: "run-1",
        scene: "doctor",
        status: "active",
        graphPath: ["intent", "planner"],
        steps: [
          { name: "intent", at: "t1", attrs: {} },
          { name: "planner", at: "t2", attrs: { status: "active" } },
        ],
        summary: null,
        startedAt: "t0",
        finishedAt: null,
      },
    };

    const rows = buildTraceRows(state);

    expect(rows.map((row) => row.latency)).toEqual([null, null]);
    expect(JSON.stringify(rows)).not.toContain("42ms");
    expect(JSON.stringify(rows)).not.toContain("80ms");
  });
});
