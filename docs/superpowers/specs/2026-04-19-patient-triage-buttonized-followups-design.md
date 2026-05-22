# Patient Triage Buttonized Follow-ups Design

**Date:** 2026-04-19  
**Status:** Approved for planning  
**Goal:** Add button-first follow-up interactions to the patient-side outpatient triage flow while preserving normal free-text chat, keeping the existing chat transport, and allowing a hybrid of schema-driven and LLM-generated options.

## 1. Context

The current workspace already supports:

- a dedicated patient scene and doctor scene
- a patient-side outpatient triage flow
- inline cards attached to assistant messages
- a standard chat transport that sends one `user` message per turn

The current patient-side triage experience is still text-first:

- the patient answers in the main chat composer
- the backend extracts structured triage signals from free text
- the triage flow asks fixed follow-up questions for a small number of fields

This works, but it has clear UX limitations:

- patients must infer the answer format from free-text prompts
- multi-select symptom answers are awkward to enter consistently
- the current flow is optimized for parsing, not for guided completion
- the platform already has button-capable UI primitives, but they are not used for patient triage follow-ups

The target product behavior is narrower than a general chat UI overhaul:

- only the **patient** scene gets this interaction
- only **outpatient triage follow-up questions** get buttons
- normal patient chat and all doctor-side chat stay unchanged
- buttons should be preferred, but free-text entry must remain available as a fallback

## 2. Design Principles

- Keep the feature scoped to patient-side outpatient triage follow-ups.
- Preserve the existing main chat request path instead of adding a separate submission API.
- Prefer structured option generation for common triage questions, with controlled LLM flexibility for special follow-ups.
- Keep normal conversation behavior unchanged outside the triage-follow-up state.
- Treat buttons as an enhanced answer surface, not as a replacement for chat.
- Make multi-select a first-class interaction for symptom collection.
- Ensure every button interaction can degrade safely back to free text.
- Reuse the existing inline-card rendering pipeline where possible.

## 3. Confirmed Decisions

The following choices are already resolved:

- Buttons appear only in the patient triage questioning flow.
- Normal chat remains text-first and should not show these controls.
- Buttons are preferred, but manual text entry remains available.
- `Other` should expand an input inside the same assistant follow-up block.
- Some questions must support multi-select.
- Multi-select answers submit only after the user clicks a dedicated submit button.
- Common triage prompts should use backend-defined schemas.
- Special prompts may use LLM-generated options within a constrained payload format.
- Buttons should appear below the assistant follow-up bubble, not in the right-side inspector.

## 4. Scope

### In scope

- Add a new inline card type for interactive triage follow-up answers
- Show button controls only when the patient scene is in active outpatient triage inquiry mode
- Support both single-select and multi-select questions
- Support `Other` with inline free-text expansion
- Keep the normal chat composer available as a fallback
- Generate common-option sets from stable schemas
- Allow constrained LLM generation for exceptional follow-up prompts
- Send button answers through the existing chat streaming request path
- Preserve compatibility with current triage text extraction

### Out of scope

- Buttonizing the entire patient chat experience
- Adding buttonized follow-ups to doctor scene flows
- Replacing the current chat transport with a dedicated forms API
- Rewriting the whole triage state machine around structured submissions only
- Building a general-purpose UI DSL for arbitrary LLM-generated widgets

## 5. Current-State Findings

### 5.1 Patient input is currently text-only

The patient conversation panel uses the shared chat composer with a `textarea` and send button. There is no patient-side follow-up control surface specific to triage.

### 5.2 Triage follow-ups are fixed and parser-oriented

The outpatient triage flow currently tracks a fixed set of fields and advances by parsing patient free text:

- `duration`
- `bleeding`
- `bowel_change`
- `weight_loss`
- `fever`

This is good enough for first-pass triage, but it does not offer a guided interaction model.

### 5.3 The frontend already has reusable button-card building blocks

The card-rendering layer already supports prompt buttons for other cards. That means the product does not need a fresh interaction framework to support triage buttons. The right move is to extend the existing inline card path rather than invent a second rendering channel.

### 5.4 The frontend already receives triage state in session snapshots and stream patches

The current state snapshot and stream reducer already preserve triage-related findings. This allows the UI to determine whether it is in triage inquiry mode without relying on assistant wording.

## 6. Target User Experience

## 6.1 When buttons should appear

Buttons should appear only when all of the following are true:

- active scene is `patient`
- `findings.encounter_track == "outpatient_triage"`
- `findings.active_inquiry == true`
- `findings.inquiry_type == "outpatient_triage"`
- the assistant turn includes a `triage_question_card`

If any of those conditions fail, the user should see the current normal chat experience.

### 6.2 Single-select behavior

- The assistant asks the follow-up question in normal text.
- An inline button block appears below the same assistant message.
- Clicking a normal single-select option immediately sends the answer.
- Clicking `Other` expands a text field inside the same block.
- The inline `Other` field is submitted from the same block rather than forcing the user back into the global composer.

### 6.3 Multi-select behavior

- The assistant asks the follow-up question in normal text.
- The user can select multiple options from the inline button block.
- If `Other` is selected, an inline text field appears in the same block.
- The user clicks `Submit answer` to send the combined response.
- The user can still ignore the button block and answer by typing manually into the regular composer.

### 6.4 Manual fallback behavior

The main patient chat composer remains enabled. A patient can always answer by typing instead of using buttons. The backend should continue to accept and parse those answers exactly as it does now.

## 7. Proposed Architecture

### 7.1 Add a new inline card: `triage_question_card`

The backend should emit a new inline card type during active outpatient triage follow-ups:

- `triage_question_card`

This card is rendered inline below the assistant follow-up message, just like existing inline cards attached to messages. It should not be rendered as a standalone right-rail card.

### 7.2 Keep the existing chat transport

Button-driven answers should still be sent through the existing streaming chat request. The frontend should continue sending:

- one `user` message
- through the normal `streamTurn` request

This avoids a new backend endpoint and keeps the feature aligned with current message history, snapshot recovery, and streaming state handling.

### 7.3 Use a hybrid option-generation strategy

The backend should support two generation modes:

1. **Schema mode**
- Used for common follow-up fields such as duration, bleeding, bowel change, weight loss, and fever
- Option labels and canonical submission text come from backend-owned templates
- This keeps behavior stable and testable

2. **Flexible mode**
- Used only when a follow-up does not fit a known schema cleanly
- LLM generates a constrained structured option payload
- Backend validates and normalizes the result before sending it to the frontend

The frontend should not care which mode produced the card. Both modes must converge to the same payload shape.

## 8. Card Payload Design

### 8.1 Proposed payload

```ts
type TriageQuestionCard = {
  type: "triage_question_card";
  version: 1;
  question_id: string;
  field_key?: string | null;
  title?: string | null;
  prompt: string;
  help_text?: string | null;
  selection_mode: "single" | "multiple";
  options: Array<{
    id: string;
    label: string;
    submit_text: string;
    exclusive?: boolean;
    requires_free_text?: boolean;
  }>;
  allow_other: boolean;
  other_label?: string | null;
  other_placeholder?: string | null;
  submit_label?: string | null;
};
```

### 8.2 Semantics

- `question_id`: stable identifier for the current triage prompt instance
- `field_key`: known schema field when available
- `selection_mode`: controls single-select vs multi-select interaction
- `submit_text`: canonical natural-language fragment used to construct the outbound user answer
- `exclusive`: marks options like `none` or `not sure` that should clear conflicting selections
- `requires_free_text`: marks options that should expand the inline text field
- `allow_other`: enables the generic `Other` path even if no normal option requires extra text

## 9. Submission Model

### 9.1 Outbound message content

The frontend should convert button selections into a normal natural-language user message so the current triage extractor remains usable.

Examples:

- single-select: `There has been blood in the stool, and it is bright red.`
- multi-select: `There has been fever and vomiting; other details: abdominal bloating became obvious last night.`

### 9.2 Optional structured context

In addition to `message.content`, the frontend should send structured metadata in `context` when the answer comes from a button interaction:

```ts
{
  message: {
    role: "user",
    content: "There has been fever and vomiting; other details: abdominal bloating became obvious last night."
  },
  context: {
    triage_interaction: {
      question_id: "triage-q-123",
      field_key: "fever",
      selection_mode: "multiple",
      selected_option_ids: ["fever", "vomiting", "other"],
      other_text: "Abdominal bloating became obvious last night."
    }
  }
}
```

Phase 1 rules:

- backend still primarily trusts `message.content`
- structured context is used for logging, validation, and future migration
- manual free-text answers do not need to send `triage_interaction`

## 10. Backend Design

### 10.1 Main node changes

The main implementation should extend `node_outpatient_triage()` to emit `triage_question_card` whenever:

- `active_inquiry == true`
- the current follow-up can be represented as a buttonized question

This should not change the node's core triage responsibilities:

- update symptom snapshot
- infer risk/disposition
- decide next pending field
- emit assistant follow-up text

The new work is additive:

- derive question model
- derive options
- attach `triage_question_card`

### 10.2 Schema registry

The backend should introduce a small triage question schema registry for common fields. Each schema entry should define:

- supported `field_key`
- default `selection_mode`
- option ids
- visible labels
- canonical `submit_text`
- whether `Other` is supported
- whether any option is exclusive

### 10.3 Flexible LLM generation

For non-schema follow-ups, the backend may ask the model for a structured option list. That output must be validated before use.

Validation rules should include:

- allowed `selection_mode` values only
- minimum and maximum option counts
- non-empty `label`
- non-empty `submit_text`
- no duplicate option ids
- no malformed `requires_free_text` / `exclusive` combinations

If validation fails, the backend must fall back to a plain text follow-up without the interactive card.

### 10.4 Recovery and snapshot behavior

Session snapshot recovery must preserve enough triage state to re-render the correct follow-up question after refresh or reconnect. The new card type should participate in the same card extraction and snapshot rebuild path as other inline cards.

## 11. Frontend Design

### 11.1 Rendering location

`triage_question_card` should render only as an inline card attached to the assistant follow-up message in the conversation panel.

### 11.2 Component behavior

The card renderer should manage local interaction state:

- selected option ids
- inline `Other` text
- submitting / disabled state

This state should remain local to the card component, keyed by `question_id`, instead of expanding global session state.

### 11.3 Message lifecycle behavior

- Once the answer is submitted, the card should disable itself for that message instance.
- When a new assistant follow-up with a new `question_id` arrives, the new card gets fresh local state.
- Old cards should not remain interactive once the conversation has advanced.

### 11.4 Normal chat isolation

If no `triage_question_card` exists, the conversation panel should behave exactly as it does today. This prevents leakage of triage-specific UI into normal patient chat.

## 12. Failure Handling

### 12.1 Backend generation failures

If schema lookup or LLM option generation fails:

- keep the normal assistant follow-up text
- omit `triage_question_card`
- allow the patient to answer via normal free text

### 12.2 Frontend interaction failures

If sending a button-generated answer fails:

- preserve the selected options and inline `Other` text
- keep the interaction available for retry
- do not discard the user's local selections immediately

### 12.3 Stale question submissions

If the frontend submits a stale `question_id` after the backend has already advanced:

- backend should avoid hard-failing the turn
- safest behavior is to treat `message.content` as a normal user answer
- structured metadata may be used to detect staleness for logging or future refinement

## 13. Testing Strategy

### 13.1 Backend tests

- schema-backed card generation for each common triage field
- validation and fallback behavior for malformed flexible card payloads
- compatibility of generated natural-language submissions with current text extraction
- snapshot recovery includes the active triage interactive state

### 13.2 Frontend component tests

- single-select immediate submission
- multi-select accumulation and explicit submit
- `Other` expansion inside the same assistant card
- disabled state after submission
- old card becomes non-interactive after newer prompt arrives

### 13.3 End-to-end tests

- patient symptom input triggers buttonized follow-up
- button answer advances the triage flow
- manual typed answer still advances the triage flow
- non-triage patient chat shows no buttonized follow-up UI
- doctor scene shows no buttonized follow-up UI

## 14. Rollout Plan

Phase 1 should target the current outpatient triage fields only:

- `duration`
- `bleeding`
- `bowel_change`
- `weight_loss`
- `fever`

Flexible LLM-generated follow-ups should be added only after the schema path is stable. The first release should optimize for reliability, not maximum UI freedom.

## 15. Recommended Implementation Direction

Proceed with a minimal, schema-first implementation that:

- adds `triage_question_card`
- supports single-select and multi-select
- supports inline `Other`
- preserves normal chat fallback
- uses existing chat transport and inline card rendering paths

This is the lowest-risk path that satisfies the product goal without overbuilding a generalized LLM widget system.
