import { describe, expect, it } from "vitest";

import { buildBearerHeaders } from "./providers";

describe("buildBearerHeaders", () => {
  it("builds an Authorization header from the configured frontend bearer token", () => {
    expect(buildBearerHeaders("  dev-token  ")).toEqual({
      Authorization: "Bearer dev-token",
    });
  });

  it("does not create headers when no token is configured", () => {
    expect(buildBearerHeaders(undefined)).toBeUndefined();
    expect(buildBearerHeaders("   ")).toBeUndefined();
  });
});
