let fallbackCounter = 0;

export function generateTraceId(): string {
  const crypto = globalThis.crypto;
  if (crypto?.randomUUID) {
    return crypto.randomUUID();
  }

  if (crypto?.getRandomValues) {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);

    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
    return [
      hex.slice(0, 4).join(""),
      hex.slice(4, 6).join(""),
      hex.slice(6, 8).join(""),
      hex.slice(8, 10).join(""),
      hex.slice(10, 16).join(""),
    ].join("-");
  }

  fallbackCounter += 1;
  const now = Date.now().toString(36);
  const perfNow =
    typeof performance !== "undefined" && Number.isFinite(performance.now())
      ? performance.now().toString(36).replace(".", "")
      : "";
  const random = Math.random().toString(36).slice(2, 10);

  return `trace-${now}-${perfNow}-${fallbackCounter.toString(36)}-${random}`;
}
