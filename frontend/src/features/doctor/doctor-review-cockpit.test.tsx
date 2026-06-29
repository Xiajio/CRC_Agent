import { act, fireEvent, screen, waitFor } from "@testing-library/react";
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
            source: "patient_upload",
            normalized_fact: {
              name: "tumor_location",
              value: "rectum",
            },
            evidence_refs: [],
            confidence: "high",
            reviewed_status: "needs_evidence",
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
    expect(screen.getByText("needs_evidence")).toBeInTheDocument();
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

  it("hides stale review actions while a new session review is pending", async () => {
    let resolveSessionB!: (value: ReturnType<typeof makeDoctorReviewResponse>) => void;
    const sessionBReview = new Promise<ReturnType<typeof makeDoctorReviewResponse>>((resolve) => {
      resolveSessionB = resolve;
    });
    const getDoctorReview = vi.fn((sessionId: string) => {
      if (sessionId === "sess-a") {
        return Promise.resolve(
          makeDoctorReviewResponse({
            session_id: "sess-a",
            timeline: [
              {
                item_id: "timeline-a",
                kind: "draft_generated",
                title: "Session A risk summary",
                created_at: "2026-06-29T10:00:00Z",
                assertion_refs: ["assertion-a"],
              },
            ],
            assertions: [
              {
                assertion_id: "assertion-a",
                patient_id: "101",
                source: "model_draft",
                normalized_fact: { name: "session_a_fact" },
                evidence_refs: [],
                confidence: "high",
                reviewed_status: "unreviewed",
              },
            ],
            draft: {
              draft_id: "draft-a",
              sections: [
                {
                  section_id: "risk_summary",
                  text: "Session A draft.",
                  verification_status: "traceable",
                  provenance: [],
                },
              ],
            },
          }),
        );
      }

      return sessionBReview;
    });
    const recordDoctorActionTrace = vi.fn(async () => makeDoctorActionTraceResponse());
    const apiClient = buildApiClientStub({
      getDoctorReview,
      recordDoctorActionTrace,
    });

    const { rerender } = renderWithProviders(
      <DoctorReviewCockpit sessionId="sess-a" enabled={true} />,
      apiClient,
    );

    expect(await screen.findByText("Session A risk summary")).toBeInTheDocument();

    rerender(<DoctorReviewCockpit sessionId="sess-b" enabled={true} />);

    expect(screen.queryByText("Session A risk summary")).not.toBeInTheDocument();
    const pendingAcceptButton = screen.queryByRole("button", { name: "accept risk summary" });
    if (pendingAcceptButton) {
      fireEvent.click(pendingAcceptButton);
    }
    expect(recordDoctorActionTrace).not.toHaveBeenCalledWith("sess-a", expect.anything());

    await act(async () => {
      resolveSessionB(
        makeDoctorReviewResponse({
          session_id: "sess-b",
          timeline: [
            {
              item_id: "timeline-b",
              kind: "draft_generated",
              title: "Session B risk summary",
              created_at: "2026-06-29T10:01:00Z",
              assertion_refs: ["assertion-b"],
            },
          ],
          assertions: [
            {
              assertion_id: "assertion-b",
              patient_id: "101",
              source: "doctor_note",
              normalized_fact: { name: "session_b_fact" },
              evidence_refs: [],
              confidence: "high",
              reviewed_status: "unreviewed",
            },
          ],
          draft: {
            draft_id: "draft-b",
            sections: [
              {
                section_id: "risk_summary",
                text: "Session B draft.",
                verification_status: "traceable",
                provenance: [],
              },
            ],
          },
        }),
      );
      await sessionBReview;
    });

    expect(await screen.findByText("Session B risk summary")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "accept risk summary" }));

    await waitFor(() => {
      expect(recordDoctorActionTrace).toHaveBeenCalledTimes(1);
    });
    expect(recordDoctorActionTrace).toHaveBeenCalledWith(
      "sess-b",
      expect.objectContaining({
        target_refs: {
          draft_id: "draft-b",
          assertion_id: "assertion-b",
        },
      }),
    );
  });
});
