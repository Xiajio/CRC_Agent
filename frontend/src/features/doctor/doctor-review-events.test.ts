import type { DoctorReasonCode } from "../../app/api/types";
import { buildAcceptTrace, buildEditTrace, buildMarkUnsafeTrace } from "./doctor-review-events";

describe("doctor review event builders", () => {
  it("builds accept trace payloads for draft risk summaries", () => {
    expect(
      buildAcceptTrace({
        draftId: "draft-123",
        assertionId: "assertion-456",
      }),
    ).toEqual({
      action_type: "accept",
      target_object: "draft.risk_summary",
      target_refs: {
        draft_id: "draft-123",
        assertion_id: "assertion-456",
      },
      reason_code: "workflow_mismatch",
      reviewer_role: "physician_reviewer",
    });
  });

  it("omits assertion refs from accept trace payloads when absent", () => {
    expect(buildAcceptTrace({ draftId: "draft-123" })).toEqual({
      action_type: "accept",
      target_object: "draft.risk_summary",
      target_refs: {
        draft_id: "draft-123",
      },
      reason_code: "workflow_mismatch",
      reviewer_role: "physician_reviewer",
    });
  });

  it("builds edit trace payloads with before and after text", () => {
    const reasonCode: DoctorReasonCode = "fact_wrong";

    expect(
      buildEditTrace({
        targetObject: "draft.risk_summary",
        draftId: "draft-123",
        before: "Before text",
        after: "After text",
        reasonCode,
      }),
    ).toEqual({
      action_type: "edit",
      target_object: "draft.risk_summary",
      target_refs: {
        draft_id: "draft-123",
      },
      before_after: {
        before: "Before text",
        after: "After text",
      },
      reason_code: "fact_wrong",
      reviewer_role: "physician_reviewer",
    });
  });

  it("builds mark unsafe trace payloads for assertions", () => {
    expect(buildMarkUnsafeTrace({ assertionId: "assertion-456" })).toEqual({
      action_type: "mark_unsafe",
      target_object: "assertion",
      target_refs: {
        assertion_id: "assertion-456",
      },
      reason_code: "unsafe_disposition",
      reviewer_role: "physician_reviewer",
    });
  });
});
