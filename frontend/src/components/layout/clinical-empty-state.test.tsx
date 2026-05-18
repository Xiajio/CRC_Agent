import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ClinicalEmptyState } from "./clinical-empty-state";

describe("ClinicalEmptyState", () => {
  it("renders a polished empty state with title, message, and icon tone", () => {
    render(
      <ClinicalEmptyState
        icon="cards"
        title="No cards ready"
        message="Clinical evidence cards will appear after the next assistant response."
      />,
    );

    expect(screen.getByTestId("clinical-empty-state")).toHaveClass("clinical-empty-state");
    const icon = screen.getByTestId("clinical-empty-state-icon");
    expect(icon).toHaveClass("clinical-empty-state-icon-cards");
    expect(icon.querySelector("svg")).toBeInTheDocument();
    expect(icon.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByText("No cards ready")).toBeInTheDocument();
    expect(screen.getByText("Clinical evidence cards will appear after the next assistant response.")).toBeInTheDocument();
  });

  it("supports compact density and an optional action", () => {
    const onAction = vi.fn();

    render(
      <ClinicalEmptyState
        compact
        icon="events"
        title="No events"
        message="Events will appear as the workflow runs."
        actionLabel="Refresh"
        onAction={onAction}
      />,
    );

    expect(screen.getByTestId("clinical-empty-state")).toHaveClass("clinical-empty-state-compact");
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(onAction).toHaveBeenCalledTimes(1);
  });
});
