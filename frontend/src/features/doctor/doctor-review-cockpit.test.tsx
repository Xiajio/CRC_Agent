import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  buildApiClientStub,
  makeDoctorActionTraceResponse,
  makeDoctorReviewResponse,
  renderWithProviders,
} from "../../test/test-utils";
import { DoctorReviewCockpit } from "./doctor-review-cockpit";

describe("DoctorReviewCockpit", () => {
  it("fetches and renders review details when enabled, then records accept traces", async () => {
    const getDoctorReview = vi.fn(async () =>
      makeDoctorReviewResponse({
        session_id: "sess-review",
        timeline: [
          {
            item_id: "timeline-1",
            kind: "draft_generated",
            title: "Risk summary drafted",
            created_at: "2026-06-29T10:00:00Z",
            assertion_refs: ["assertion-1"],
          },
        ],
        assertions: [
          {
            assertion_id: "assertion-1",
            patient_id: "101",
            session_id: "sess-review",
            source: "clinical_review",
            normalized_fact: {
              name: "tumor_location",
              value: "rectum",
            },
            evidence_refs: [],
            confidence: "high",
            reviewed_status: "unreviewed",
          },
        ],
        draft: {
          draft_id: "draft-1",
          sections: [
            {
              section_id: "risk_summary",
              text: "Model generated risk summary.",
              verification_status: "model_generated_unverified",
              provenance: [
                {
                  kind: "assertion",
                  assertion_id: "assertion-1",
                },
              ],
            },
          ],
        },
      }),
    );
    const recordDoctorActionTrace = vi.fn(async () => makeDoctorActionTraceResponse());
    const apiClient = buildApiClientStub({
      getDoctorReview,
      recordDoctorActionTrace,
    });

    renderWithProviders(
      <DoctorReviewCockpit sessionId="sess-review" enabled={true} />,
      apiClient,
    );

    expect(await screen.findByText("Risk summary drafted")).toBeInTheDocument();
    expect(screen.getByText("draft_generated")).toBeInTheDocument();
    expect(screen.getByText("tumor_location")).toBeInTheDocument();
    expect(screen.getByText("unreviewed")).toBeInTheDocument();
    expect(screen.getByText("Model generated risk summary.")).toBeInTheDocument();
    expect(screen.getByText("model_generated_unverified")).toBeInTheDocument();
    expect(screen.getByText("assertion-1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "accept risk summary" }));

    await waitFor(() => {
      expect(recordDoctorActionTrace).toHaveBeenCalledTimes(1);
    });
    expect(recordDoctorActionTrace).toHaveBeenCalledWith("sess-review", {
      action_type: "accept",
      target_object: "draft.risk_summary",
      target_refs: {
        draft_id: "draft-1",
        assertion_id: "assertion-1",
      },
      reason_code: "workflow_mismatch",
      reviewer_role: "physician_reviewer",
    });
  });

  it("does not fetch or render when disabled", () => {
    const getDoctorReview = vi.fn();
    const apiClient = buildApiClientStub({ getDoctorReview });

    const { container } = renderWithProviders(
      <DoctorReviewCockpit sessionId="sess-review" enabled={false} />,
      apiClient,
    );

    expect(container).toBeEmptyDOMElement();
    expect(getDoctorReview).not.toHaveBeenCalled();
  });
});
