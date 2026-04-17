import type { StreamEvent } from "./types";

export type FetchLike = typeof fetch;

export interface StreamJsonOptions<TBody> {
  fetchImpl?: FetchLike;
  url: string;
  body: TBody;
  headers?: HeadersInit;
  signal?: AbortSignal;
  onEvent: (event: StreamEvent) => void;
}

function parseEventBlock(block: string): StreamEvent | null {
  const lines = block.replace(/\r/g, "").split("\n");
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  return JSON.parse(dataLines.join("\n")) as StreamEvent;
}

export async function consumeSseResponse(
  response: Response,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  if (!response.ok) {
    throw new Error(`Stream request failed with status ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Stream response body is not readable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseEventBlock(block);
      if (event) {
        onEvent(event);
      }
      boundary = buffer.indexOf("\n\n");
    }

    if (done) {
      break;
    }
  }

  const trailingEvent = parseEventBlock(buffer);
  if (trailingEvent) {
    onEvent(trailingEvent);
  }
}

export async function streamJsonEvents<TBody>({
  fetchImpl = fetch,
  url,
  body,
  headers,
  signal,
  onEvent,
}: StreamJsonOptions<TBody>): Promise<void> {
  const response = await fetchImpl(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(headers ?? {}),
    },
    body: JSON.stringify(body),
    signal,
  });

  await consumeSseResponse(response, onEvent);
}
