# Chat UI Completion Latency Design

## Goal

Measure how long a chat turn takes from the user's submit action to the moment the assistant's final answer is fully rendered in the chat UI.

Primary example:

- User input: `你有什么用`
- Target metric: total UI completion time for the assistant's full reply

This spec defines the measurement contract, the frontend probe design, the UI surface for the metric, and the validation workflow.

## Problem Statement

The current chat flow streams assistant output over SSE and incrementally merges events into frontend session state, but it does not explicitly record how long a turn takes to finish from the user's perspective.

The backend already emits:

- `message.delta` for incremental text
- `message.done` for finalized assistant content
- `done` for overall stream completion and snapshot finalization

The missing piece is a user-facing latency definition that matches what the user actually experiences in the interface.

## Success Criteria

- For a pure text prompt such as `你有什么用`, the frontend records one stable latency value for the turn.
- The measured value represents UI completion, not backend-only completion.
- Aborted, superseded, or failed turns do not produce false successful latency records.
- The metric can be repeated across multiple runs for comparison.

## Non-Goals

- Backend tracing or root-cause attribution for model, network, or graph-node latency
- Historical analytics storage across sessions
- Dashboarding or observability export
- A backend protocol change for timestamps in the first version

## Alternatives Considered

### Option A: Frontend-only UI completion probe

Record the start time on submit, watch stream events on the frontend, and stop timing only after the finalized assistant message has been rendered to the chat panel.

Pros:

- Matches user experience most closely
- Requires no backend API change
- Lowest implementation risk

Cons:

- Does not decompose backend versus rendering cost

### Option B: Frontend plus backend timestamps

Add server timestamps to SSE events and correlate them with browser timing.

Pros:

- Makes bottlenecks easier to attribute

Cons:

- Requires protocol and schema changes
- Adds complexity before the core metric is proven useful

### Option C: Browser automation benchmark only

Use Playwright to repeatedly submit a fixed prompt and measure completion externally.

Pros:

- Good for regression testing

Cons:

- Not useful as an in-product metric by itself

## Recommended Approach

Adopt Option A first.

The primary metric is:

`UI completion time = assistant final render committed time - user submit time`

This is the correct first implementation because it measures what the user actually waits for, it aligns with the existing frontend architecture, and it avoids backend protocol churn.

## Current-Code References

The first implementation must align with these existing files:

- `frontend/src/pages/workspace-page.tsx`
- `frontend/src/app/store/stream-reducer.ts`
- `frontend/src/features/chat/conversation-panel.tsx`
- `frontend/src/features/doctor/doctor-scene-shell.tsx`
- `frontend/src/app/api/client.ts`
- `frontend/src/app/api/stream.ts`
- `frontend/src/app/api/types.ts`
- `backend/api/schemas/events.py`

## Measurement Definition

### Start Boundary

The timer starts inside `WorkspacePage.submitMessage()`, before any state updates related to the turn are made.

Specifically, the start timestamp must be captured before:

- `setIsStreaming(true)`
- `appendOptimisticUserMessage(...)`
- `apiClient.streamTurn(...)`

This ensures the metric includes optimistic UI work as part of the user-visible turn cost.

This start boundary applies to both submit paths that converge into `submitMessage()`:

- text composer submit via `submitPrompt()`
- card-driven prompt submit via `onCardPromptRequest`

### Logical Completion Boundary

The turn reaches logical completion when the frontend receives `message.done` for the assistant message and merges the final content into frontend state.

### UI Completion Boundary

The turn reaches UI completion only after:

1. A `message.done` event has been observed for the current turn.
2. The target assistant message for the current turn exists in the relevant scene's `messages` array.
3. React has committed the state update that carries that finalized message.
4. One `requestAnimationFrame` has elapsed after that commit.

This intentionally excludes backend `done`, because that event includes stream-level completion and snapshot bookkeeping that can happen after the user can already read the answer.

## Probe Design

The latency probe should live in the page layer, not inside the shared session reducer.

Recommended runtime shape:

```ts
type TurnLatencyProbe = {
  sequence: number;
  scene: "patient" | "doctor";
  prompt: string;
  status: "streaming" | "message_done" | "ui_complete" | "aborted" | "error";
  startedAt: number;
  messageDoneAt: number | null;
  renderCommittedAt: number | null;
  assistantMessageId: string | null;
  assistantCursor: string | null;
  finalContentText: string | null;
  uiCompleteMs: number | null;
  errorMessage: string | null;
};
```

### Sequence Contract

`probe.sequence` must be bound to the existing `streamSequenceRef` in `WorkspacePage`.

This is required so the probe can distinguish:

- the active turn
- a superseded turn that was replaced by a later submit

Before finalizing UI completion, the page logic must confirm:

- `activeProbeRef.current?.sequence === streamSequenceRef.current`

If the values no longer match, the probe must be marked `aborted` or superseded and must not emit a successful measurement.

### Ownership

Recommended ownership:

- `activeProbeRef`: mutable data for the currently running turn
- `recentCompletedProbe`: React state for the latest completed measurement shown in the UI

`recentCompletedProbe` must be state, not ref, so the UI re-renders when a measurement completes.

### Lifecycle

`recentCompletedProbe` should:

- be cleared when a new `submitMessage()` begins
- be cleared when the active scene is reset
- retain its `scene` field so display logic can decide whether to show it in the current panel

## Event Handling Contract

### On Submit

When the user submits a prompt:

- create a new probe
- set `startedAt = performance.now()`
- set `status = "streaming"`
- set `scene` from the active scene at submit time
- bind `sequence` to the new `streamSequenceRef` value
- clear `recentCompletedProbe`

### On `message.delta`

No completion metric is recorded yet. The event is still useful as an optional future TTFT metric, but TTFT is not the primary metric in this spec.

### On `message.done`

Record:

- `messageDoneAt = performance.now()`
- `assistantMessageId` when provided
- `finalContentText` for debugging and inspection
- `status = "message_done"`

Do not finalize the measurement here.

### `message.done` Matching Rules

`MessageDoneEvent.message_id` is optional in the current protocol, so the page logic must use a two-stage matching rule:

1. Prefer matching by `message_id` when present.
2. If `message_id` is null, fall back to the latest assistant cursor tracked for the turn.

The fallback cursor should align with reducer behavior and existing page state, not with backend `done`.

The design must not rely on backend `done` to recover assistant-message identity because:

- `done` is not a message-finalization signal
- reducer cleanup for `done` does not resolve the missing `message_id` problem

### On UI Commit

Use a page-level effect to detect that the finalized assistant message now exists in the relevant scene state for the current probe and that the current probe is still the active sequence. Then schedule a single `requestAnimationFrame` and record:

- `renderCommittedAt = performance.now()`
- `uiCompleteMs = renderCommittedAt - startedAt`
- `status = "ui_complete"`

### On Scene Switch

If the active scene changes while the probe is still incomplete:

- compare `activeScene` with `activeProbeRef.current.scene`
- if they differ and the probe is not already `ui_complete`, mark the probe `aborted`
- clear `activeProbeRef`

The completed measurement may remain in `recentCompletedProbe`, but panel display should only show it when the probe scene matches the panel scene.

### On Error

Backend business errors may arrive as SSE `error` events and update `state.lastError` without causing `streamTurn()` to reject.

Therefore page-level logic must also observe the active scene state's `lastError`. If a probe is still incomplete when `lastError` becomes non-null:

- mark the probe `error`
- copy `errorMessage`
- do not surface a successful latency value

### On Abort or Reject

If the request is aborted, superseded, or `streamTurn()` rejects before UI completion:

- mark the probe `aborted` or `error`
- do not surface a successful latency value

## Render Detection Strategy

The first implementation should use one consistent completion rule for both pure-text and card-assisted turns:

- a `message.done` event has been observed for the active probe
- the target assistant message for that probe exists in the relevant scene's `messages` array
- the active probe sequence still matches `streamSequenceRef.current`
- one `requestAnimationFrame` runs after the commit that exposed that target message

For pure text turns such as `你有什么用`, exact text equality with `finalContentText` may be used as a debugging sanity check, but it is not the required completion predicate.

This avoids overstating the value of string equality in the current reducer, where `message.done` already overwrites the stored message content with the finalized payload.

## UI Surface

Display the current measurement in the chat panel status area rather than in the global shell or inside individual message bubbles.

Recommended states:

- idle: hidden
- streaming: `本轮耗时计时中...`
- completed: `本轮界面完成 3.54s`

Optional supplemental text:

- short prompt preview, such as `问题：你有什么用`

The prompt preview must be truncated to at most 24 visible characters, then suffixed with an ellipsis if needed.

### Component Contract

`ConversationPanel` is reused by both patient and doctor flows, so latency display must be passed in as a prop rather than hard-coded in one page only.

Recommended direction:

- add a new prop such as `latencyStatus`
- derive that prop from page-level probe state
- pass it to `ConversationPanel` in patient flow
- pass it through `DoctorSceneShell` in doctor flow so both scenes can render the same status model

## Validation Plan

### Manual Validation

Use a fixed pure text prompt:

- `你有什么用`

Validation steps:

1. Open one scene, preferably `patient`.
2. Reset the scene so the test starts from a clean session.
3. Submit `你有什么用`.
4. Confirm the panel enters a measuring state while streaming.
5. Confirm a single completed latency value appears after the final answer is visibly rendered.

### Repeatability Check

Repeat the same prompt 5 to 10 times under the same conditions, resetting the scene between runs, and compare the resulting `uiCompleteMs` values.

### Edge Cases

Verify that successful latency is not reported when:

- the request is aborted
- the scene is switched mid-stream
- the backend emits an `error` event
- a second request supersedes the first
- a card-driven prompt path triggers the turn instead of composer submit

## Future Extensions

If the UI completion metric proves useful, the next layer of instrumentation can add:

- TTFT
- `messageDoneMs`
- render tail time from `message.done` to visual commit
- backend timestamps for decomposition

These are explicitly deferred until the UI completion metric is working and trusted.

## Acceptance Criteria

The design is complete when implementation can produce the following behavior for `你有什么用`:

- one turn starts timing at `submitMessage()` before optimistic updates
- one successful turn records exactly one `uiCompleteMs`
- the metric corresponds to the finalized answer being visible in the UI
- scene switches, errors, and superseded turns do not emit misleading success measurements
- both patient and doctor chat panels can display the same latency status model
