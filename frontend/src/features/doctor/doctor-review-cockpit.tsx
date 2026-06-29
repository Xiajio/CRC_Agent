import { useEffect, useState } from "react";

import type { DoctorReviewResponse, JsonObject } from "../../app/api/types";
import { useApiClient } from "../../app/providers";
import { buildAcceptTrace, buildMarkUnsafeTrace } from "./doctor-review-events";

export type DoctorReviewCockpitProps = {
  sessionId: string | null;
  enabled: boolean;
};

function factName(fact: JsonObject): string {
  if (typeof fact.name === "string" && fact.name.trim()) {
    return fact.name;
  }
  const [firstKey] = Object.keys(fact);
  return firstKey ?? "unknown_fact";
}

export function DoctorReviewCockpit({ sessionId, enabled }: DoctorReviewCockpitProps) {
  const apiClient = useApiClient();
  const [review, setReview] = useState<DoctorReviewResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!enabled || !sessionId) {
      return;
    }

    let isCurrent = true;
    setIsLoading(true);
    setErrorMessage(null);
    apiClient
      .getDoctorReview(sessionId)
      .then((nextReview) => {
        if (isCurrent) {
          setReview(nextReview);
        }
      })
      .catch((error: unknown) => {
        if (isCurrent) {
          setErrorMessage(error instanceof Error ? error.message : "Unable to load doctor review.");
        }
      })
      .finally(() => {
        if (isCurrent) {
          setIsLoading(false);
        }
      });

    return () => {
      isCurrent = false;
    };
  }, [apiClient, enabled, sessionId]);

  if (!enabled || !sessionId) {
    return null;
  }

  if (isLoading && review === null) {
    return <section aria-label="doctor review cockpit">Loading review.</section>;
  }

  if (errorMessage) {
    return <section aria-label="doctor review cockpit">Review error: {errorMessage}</section>;
  }

  if (!review) {
    return null;
  }

  const firstAssertionId = review.assertions[0]?.assertion_id;

  function handleAcceptRiskSummary() {
    if (!review) {
      return;
    }
    void apiClient.recordDoctorActionTrace(
      review.session_id,
      buildAcceptTrace({
        draftId: review.draft.draft_id,
        assertionId: firstAssertionId,
      }),
    );
  }

  function handleMarkFirstAssertionUnsafe() {
    if (!review || !firstAssertionId) {
      return;
    }
    void apiClient.recordDoctorActionTrace(
      review.session_id,
      buildMarkUnsafeTrace({ assertionId: firstAssertionId }),
    );
  }

  return (
    <section className="doctor-review-cockpit" aria-label="doctor review cockpit">
      <div>
        <button type="button" aria-label="accept risk summary" onClick={handleAcceptRiskSummary}>
          Accept
        </button>
        <button
          type="button"
          aria-label="mark first assertion unsafe"
          onClick={handleMarkFirstAssertionUnsafe}
        >
          Mark unsafe
        </button>
      </div>

      <section aria-label="review timeline">
        {review.timeline.map((item) => (
          <article key={item.item_id}>
            <h2>{item.title}</h2>
            <p>{item.kind}</p>
          </article>
        ))}
      </section>

      <section aria-label="review assertions">
        {review.assertions.map((assertion) => (
          <article key={assertion.assertion_id}>
            <h2>{factName(assertion.normalized_fact)}</h2>
            <p>{assertion.reviewed_status}</p>
          </article>
        ))}
      </section>

      <section aria-label="review draft">
        {review.draft.sections.map((section) => (
          <article key={section.section_id}>
            <p>{section.text}</p>
            <p>{section.verification_status}</p>
            {section.provenance.map((provenance) => (
              <p key={`${provenance.kind}-${provenance.assertion_id ?? provenance.record_id ?? provenance.id}`}>
                {provenance.assertion_id ?? provenance.record_id ?? provenance.id ?? provenance.kind}
              </p>
            ))}
          </article>
        ))}
      </section>
    </section>
  );
}
