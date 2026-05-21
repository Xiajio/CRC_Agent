import { describe, expect, it } from "vitest";

import { formatCriticFeedback } from "./critic-feedback";

describe("formatCriticFeedback", () => {
  it("uses localized default feedback when no critic text is provided", () => {
    expect(formatCriticFeedback(null)).toBe("评审未通过该建议。");
    expect(formatCriticFeedback("")).toBe("评审未通过该建议。");
  });
});
