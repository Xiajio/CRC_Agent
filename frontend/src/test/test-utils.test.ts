import { describe, expect, it } from "vitest";

import { makeSessionResponse } from "./test-utils";

describe("frontend test-utils makeSessionResponse", () => {
  it("defaults registry identity without binding legacy current_patient_id", () => {
    const response = makeSessionResponse({ scene: "patient", patient_id: 101 });

    expect(response.snapshot.registry_patient_id).toBe(101);
    expect(response.snapshot.case_database_patient_id).toBeNull();
    expect(response.snapshot.current_patient_id).toBeNull();
  });

  it("preserves an explicitly provided legacy current_patient_id", () => {
    const response = makeSessionResponse({
      scene: "patient",
      patient_id: 101,
      snapshot: {
        current_patient_id: "legacy-current",
      },
    });

    expect(response.snapshot.registry_patient_id).toBe(101);
    expect(response.snapshot.current_patient_id).toBe("legacy-current");
  });
});
