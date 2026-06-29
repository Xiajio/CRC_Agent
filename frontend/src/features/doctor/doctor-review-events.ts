import type { DoctorActionTraceRequest, DoctorReasonCode } from "../../app/api/types";

const REVIEWER_ROLE = "physician_reviewer";

export type BuildAcceptTraceInput = {
  draftId: string;
  assertionId?: string;
};

export type BuildEditTraceInput = {
  targetObject: string;
  draftId: string;
  before: string;
  after: string;
  reasonCode: DoctorReasonCode;
};

export type BuildMarkUnsafeTraceInput = {
  assertionId: string;
};

export function buildAcceptTrace({ draftId, assertionId }: BuildAcceptTraceInput): DoctorActionTraceRequest {
  return {
    action_type: "accept",
    target_object: "draft.risk_summary",
    target_refs: {
      draft_id: draftId,
      ...(assertionId ? { assertion_id: assertionId } : {}),
    },
    reason_code: "workflow_mismatch",
    reviewer_role: REVIEWER_ROLE,
  };
}

export function buildEditTrace({
  targetObject,
  draftId,
  before,
  after,
  reasonCode,
}: BuildEditTraceInput): DoctorActionTraceRequest {
  return {
    action_type: "edit",
    target_object: targetObject,
    target_refs: {
      draft_id: draftId,
    },
    before_after: {
      before,
      after,
    },
    reason_code: reasonCode,
    reviewer_role: REVIEWER_ROLE,
  };
}

export function buildMarkUnsafeTrace({ assertionId }: BuildMarkUnsafeTraceInput): DoctorActionTraceRequest {
  return {
    action_type: "mark_unsafe",
    target_object: "assertion",
    target_refs: {
      assertion_id: assertionId,
    },
    reason_code: "unsafe_disposition",
    reviewer_role: REVIEWER_ROLE,
  };
}
