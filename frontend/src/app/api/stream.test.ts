import { describe, expect, it, vi } from "vitest";

import type { StreamEvent } from "./types";
import { consumeSseResponse, streamJsonEvents } from "./stream";

function makeSseResponse(blocks: string[]): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      for (const block of blocks) {
        controller.enqueue(encoder.encode(block));
      }
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("consumeSseResponse", () => {
  it("triggers the trace tap before onEvent in order and captures arrival timestamps", async () => {
    const perfSpy = vi.spyOn(performance, "now");
    perfSpy.mockReturnValueOnce(11).mockReturnValueOnce(22);

    const callSequence: string[] = [];
    const traceTap = vi.fn((event: StreamEvent, receivedAt: number) => {
      callSequence.push(`tap:${event.type}:${receivedAt}`);
    });
    const onEvent = vi.fn((event: StreamEvent) => {
      callSequence.push(`event:${event.type}`);
    });
    const response = makeSseResponse([
      `data: {"type":"message.delta","message_id":"m1","delta":"a"}\n\n`,
      `data: {"type":"message.done","role":"assistant","content":"ok","message_id":"m1"}\n\n`,
    ]);

    await consumeSseResponse(response, onEvent, traceTap);

    expect(traceTap).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ type: "message.delta" }),
      11,
    );
    expect(traceTap).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ type: "message.done" }),
      22,
    );
    expect(onEvent).toHaveBeenNthCalledWith(1, expect.objectContaining({ type: "message.delta" }));
    expect(onEvent).toHaveBeenNthCalledWith(2, expect.objectContaining({ type: "message.done" }));
    expect(callSequence).toEqual([
      "tap:message.delta:11",
      "event:message.delta",
      "tap:message.done:22",
      "event:message.done",
    ]);
  });

  it("completes final-only turns with message.done and no delta", async () => {
    const traceTap = vi.fn();
    const onEvent = vi.fn();
    const response = makeSseResponse([
      `data: {"type":"message.done","role":"assistant","content":"answer","message_id":"m1"}\n\n`,
    ]);

    await consumeSseResponse(response, onEvent, traceTap);

    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "message.done",
        content: "answer",
      }),
    );
    expect(traceTap).toHaveBeenCalledTimes(1);
  });

  it("surfaces backend error details for failed stream responses", async () => {
    const response = new Response(JSON.stringify({ detail: "backend fixture unavailable" }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });

    await expect(consumeSseResponse(response, vi.fn())).rejects.toThrow("backend fixture unavailable");
  });

  it("falls back to plain-text error bodies and status-only responses", async () => {
    const plainTextResponse = new Response("maintenance in progress", {
      status: 503,
      headers: { "Content-Type": "text/plain" },
    });
    const emptyResponse = new Response("", {
      status: 502,
      headers: { "Content-Type": "application/json" },
    });

    await expect(consumeSseResponse(plainTextResponse, vi.fn())).rejects.toThrow("maintenance in progress");
    await expect(consumeSseResponse(emptyResponse, vi.fn())).rejects.toThrow(
      "Stream request failed with status 502",
    );
  });
});

describe("streamJsonEvents", () => {
  it("keeps existing call sites working when no trace tap is provided", async () => {
    const response = makeSseResponse([
      `data: {"type":"message.done","role":"assistant","content":"answer","message_id":"m1"}\n\n`,
    ]);
    const fetchImpl = vi.fn(async () => response);
    const onEvent = vi.fn();

    await streamJsonEvents({
      fetchImpl,
      url: "/stream",
      body: { message: { role: "user", content: "hello" } },
      onEvent,
    });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ type: "message.done" }));
  });
});
