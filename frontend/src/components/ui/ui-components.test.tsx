import "@testing-library/jest-dom/vitest";

import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it } from "vitest";

import { Button, Card, Input, MessageBubble, Select, Textarea } from ".";
import type { CardVariant } from ".";
import { classNames } from "./class-names";

describe("shared UI primitives", () => {
  it("joins truthy class names", () => {
    expect(classNames(["one", false, null, undefined, "two"])).toBe("one two");
  });

  it("renders card semantic element with state, padding, tone, and custom classes", () => {
    const cardRef = createRef<HTMLElement>();
    const { rerender } = render(
      <Card
        data-testid="default-card"
        ref={cardRef}
        header={<h2>Default heading</h2>}
        footer={<button type="button">Footer action</button>}
      >
        Default card
      </Card>,
    );

    const defaultCard = screen.getByTestId("default-card");
    expect(cardRef.current).toBe(defaultCard);
    expect(defaultCard.tagName).toBe("DIV");
    expect(defaultCard).toHaveClass("ui-card", "ui-card-padding-md");
    expect(defaultCard).not.toHaveClass("ui-card-clinical-panel");
    expect(defaultCard).not.toHaveClass("ui-card-surface");
    expect(screen.getByText("Default heading").parentElement).toHaveClass("ui-card-header");
    expect(screen.getByText("Default card")).toHaveClass("ui-card-body");
    expect(screen.getByText("Footer action").parentElement).toHaveClass("ui-card-footer");

    rerender(
      <Card as="section" padding="sm" selected tone="warning" className="custom-card">
        Card content
      </Card>,
    );

    const card = screen.getByText("Card content").closest(".ui-card");
    if (!(card instanceof HTMLElement)) {
      throw new Error("Expected Card content to render inside a ui-card element");
    }
    expect(card.tagName).toBe("SECTION");
    expect(card).toHaveClass(
      "ui-card",
      "ui-card-padding-sm",
      "ui-card-warning",
      "ui-card-selected",
      "custom-card",
    );
  });

  it("renders clinical panel card variant without feature material classes", () => {
    const variant: CardVariant = "clinical-panel";

    render(
      <Card
        variant={variant}
        padding="none"
        selected
        tone="danger"
        className="custom-clinical-panel"
      >
        Clinical panel content
      </Card>,
    );

    const card = screen.getByText("Clinical panel content").closest(".ui-card");
    if (!(card instanceof HTMLElement)) {
      throw new Error("Expected clinical panel content to render inside a ui-card element");
    }

    expect(card).toHaveClass(
      "ui-card",
      "ui-card-clinical-panel",
      "ui-card-padding-none",
      "ui-card-danger",
      "ui-card-selected",
      "custom-clinical-panel",
    );
    expect(card).not.toHaveClass("clinical-card", "workspace-card");
    expect(card).not.toHaveAttribute("variant");
  });

  it("renders button children with default and requested API classes", () => {
    const { rerender } = render(
      <Button aria-label="Default action" data-testid="default-button">
        Default button
      </Button>,
    );

    const defaultButton = screen.getByTestId("default-button");
    expect(defaultButton).toHaveAttribute("type", "button");
    expect(defaultButton).toHaveAttribute("aria-label", "Default action");
    expect(defaultButton).toHaveClass(
      "ui-button",
      "ui-button-secondary",
      "ui-button-md",
    );

    rerender(
      <Button variant="primary" size="sm" disabled className="custom-button">
        Primary button
      </Button>,
    );

    const button = screen.getByRole("button", { name: "Primary button" });
    expect(button).toHaveClass(
      "ui-button",
      "ui-button-primary",
      "ui-button-sm",
      "custom-button",
    );
    expect(button).toBeDisabled();
  });

  it("preserves input, select, and textarea accessibility props and classes", () => {
    render(
      <>
        <Input aria-label="Patient name" className="custom-input" />
        <Input id="patient-age" label="Patient age" className="custom-labeled-input" />
        <Select aria-label="Priority" className="custom-select">
          <option>Routine</option>
        </Select>
        <Select id="department" label="Department" className="custom-labeled-select">
          <option>Oncology</option>
        </Select>
        <Textarea aria-label="Notes" className="custom-textarea" />
      </>,
    );

    expect(screen.getByLabelText("Patient name")).toHaveClass("ui-input", "custom-input");
    expect(screen.getByText("Patient age")).toHaveClass("ui-field-label");
    expect(screen.getByLabelText("Patient age")).toHaveClass("ui-input", "custom-labeled-input");
    expect(screen.getByLabelText("Patient age").closest("label")).toHaveClass("ui-field");
    expect(screen.getByLabelText("Priority")).toHaveClass("ui-select", "custom-select");
    expect(screen.getByRole("option", { name: "Routine" })).toBeInTheDocument();
    expect(screen.getByText("Department")).toHaveClass("ui-field-label");
    expect(screen.getByLabelText("Department")).toHaveClass(
      "ui-select",
      "custom-labeled-select",
    );
    expect(screen.getByLabelText("Department").closest("label")).toHaveClass("ui-field");
    expect(screen.getByRole("option", { name: "Oncology" })).toBeInTheDocument();
    expect(screen.getByLabelText("Notes")).toHaveClass("ui-textarea", "custom-textarea");
  });

  it("renders message bubbles by author with label and body children", () => {
    const { rerender } = render(
      <MessageBubble author="user" label="You" className="custom-message">
        User message
      </MessageBubble>,
    );

    const userBubble = screen.getByText("User message").closest("li");
    expect(screen.getByText("U")).toHaveClass("ui-message-avatar");
    expect(screen.getByText("U")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByText("You").closest(".ui-message-header")).toHaveClass(
      "ui-message-header",
    );
    expect(screen.getByText("User message").closest(".ui-message-content")).toHaveClass(
      "ui-message-content",
    );
    expect(userBubble).toHaveClass(
      "ui-message-bubble",
      "ui-message-user",
      "ui-message-bubble-user",
      "custom-message",
    );

    rerender(
      <MessageBubble author="assistant" label="Assistant">
        Assistant message
      </MessageBubble>,
    );

    expect(screen.getByText("AI")).toHaveClass("ui-message-avatar");
    expect(screen.getByText("AI")).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByText("Assistant").closest(".ui-message-header")).toHaveClass(
      "ui-message-header",
    );
    expect(screen.getByText("Assistant message").closest(".ui-message-content")).toHaveClass(
      "ui-message-content",
    );
    expect(screen.getByText("Assistant message").closest("li")).toHaveClass(
      "ui-message-bubble",
      "ui-message-assistant",
      "ui-message-bubble-assistant",
    );
  });
});
