import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { JsonObject } from "../../app/api/types";
import type { CardPromptHandler } from "./card-renderers";
import { renderCardContent } from "./card-renderers-extended";

function renderTriageQuestionCard(
  payload: Record<string, unknown>,
  onPromptRequest?: CardPromptHandler,
  isInteractive = true,
) {
  return render(
    <div>
      {renderCardContent({
        cardType: "triage_question_card",
        payload: payload as JsonObject,
        onPromptRequest,
        isInteractive,
      })}
    </div>,
  );
}

describe("triage question card rendering", () => {
  it("submits a single-select answer immediately with structured context", () => {
    const onPromptRequest = vi.fn();

    renderTriageQuestionCard(
      {
        type: "triage_question_card",
        question_id: "triage-q-fever-1",
        field_key: "fever",
        prompt: "Have you had a fever?",
        selection_mode: "single",
        options: [
          { id: "yes", label: "Yes", submit_text: "Yes, I have had a fever." },
          { id: "no", label: "No", submit_text: "No, I have not had a fever." },
        ],
        allow_other: false,
        submit_label: "Submit answer",
      },
      onPromptRequest,
    );

    expect(screen.getByText("Have you had a fever?")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Yes" }));

    expect(onPromptRequest).toHaveBeenCalledWith("Yes, I have had a fever.", {
      triage_interaction: {
        question_id: "triage-q-fever-1",
        field_key: "fever",
        selection_mode: "single",
        selected_option_ids: ["yes"],
        other_text: null,
      },
    });
    expect(screen.getByRole("button", { name: "Yes" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "No" })).toBeDisabled();
  });

  it("supports multi-select, inline Other input, and explicit submit", () => {
    const onPromptRequest = vi.fn();

    renderTriageQuestionCard(
      {
        type: "triage_question_card",
        question_id: "triage-q-bowel-1",
        field_key: "bowel_change",
        prompt: "Which symptoms are present?",
        selection_mode: "multiple",
        options: [
          { id: "pain", label: "Pain", submit_text: "I have abdominal pain." },
          { id: "nausea", label: "Nausea", submit_text: "I have nausea." },
        ],
        allow_other: true,
        other_label: "Other",
        other_placeholder: "Describe other symptoms",
        submit_label: "Submit answer",
      },
      onPromptRequest,
    );

    fireEvent.click(screen.getByRole("button", { name: "Pain" }));
    fireEvent.click(screen.getByRole("button", { name: "Other" }));

    expect(screen.getByPlaceholderText("Describe other symptoms")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Describe other symptoms"), {
      target: { value: "abdominal bloating" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Submit answer" }));

    expect(onPromptRequest).toHaveBeenCalledWith("I have abdominal pain.; abdominal bloating", {
      triage_interaction: {
        question_id: "triage-q-bowel-1",
        field_key: "bowel_change",
        selection_mode: "multiple",
        selected_option_ids: ["pain", "other"],
        other_text: "abdominal bloating",
      },
    });
    expect(screen.getByRole("button", { name: "Pain" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Other" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Submit answer" })).not.toBeInTheDocument();
    expect(screen.getByText("\u5df2\u63d0\u4ea4\u3002")).toBeInTheDocument();
  });

  it("renders stale triage question cards as read-only", () => {
    const onPromptRequest = vi.fn();

    renderTriageQuestionCard(
      {
        type: "triage_question_card",
        question_id: "triage-q-fever-1",
        field_key: "fever",
        prompt: "Have you had a fever?",
        selection_mode: "single",
        options: [
          { id: "yes", label: "Yes", submit_text: "Yes, I have had a fever." },
          { id: "no", label: "No", submit_text: "No, I have not had a fever." },
        ],
        allow_other: false,
      },
      onPromptRequest,
      false,
    );

    expect(screen.getByText("当前追问已失效。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Yes" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "No" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Yes" }));
    expect(onPromptRequest).not.toHaveBeenCalled();
  });

  it("merges submitted and inactive states into one Chinese status line", () => {
    const onPromptRequest = vi.fn();

    const { rerender } = renderTriageQuestionCard(
      {
        type: "triage_question_card",
        question_id: "triage-q-fever-1",
        field_key: "fever",
        prompt: "Have you had a fever?",
        selection_mode: "single",
        options: [
          { id: "yes", label: "Yes", submit_text: "Yes, I have had a fever." },
          { id: "no", label: "No", submit_text: "No, I have not had a fever." },
        ],
        allow_other: false,
      },
      onPromptRequest,
      true,
    );

    fireEvent.click(screen.getByRole("button", { name: "Yes" }));
    expect(screen.getByText("已提交。")).toBeInTheDocument();

    rerender(
      <div>
        {renderCardContent({
          cardType: "triage_question_card",
          payload: {
            type: "triage_question_card",
            question_id: "triage-q-fever-1",
            field_key: "fever",
            prompt: "Have you had a fever?",
            selection_mode: "single",
            options: [
              { id: "yes", label: "Yes", submit_text: "Yes, I have had a fever." },
              { id: "no", label: "No", submit_text: "No, I have not had a fever." },
            ],
            allow_other: false,
          } as JsonObject,
          onPromptRequest,
          isInteractive: false,
        })}
      </div>,
    );

    expect(screen.getByText("已提交，当前已进入下一题。")).toBeInTheDocument();
    expect(screen.queryByText("已提交。")).not.toBeInTheDocument();
    expect(screen.queryByText("当前追问已失效。")).not.toBeInTheDocument();
  });

  it("keeps the existing triage card renderer intact", () => {
    render(
      <div>
        {renderCardContent({
          cardType: "triage_card",
          payload: {
            type: "triage_card",
            risk_level: "medium",
            disposition: "urgent_gi_clinic",
            chief_symptoms: "rectal bleeding",
          },
        })}
      </div>,
    );

    expect(screen.getByText("rectal bleeding")).toBeInTheDocument();
    expect(screen.getByText("\u4e2d\u98ce\u9669")).toBeInTheDocument();
    expect(screen.getByText("\u5c3d\u5feb\u6d88\u5316\u95e8\u8bca")).toBeInTheDocument();
  });
});
