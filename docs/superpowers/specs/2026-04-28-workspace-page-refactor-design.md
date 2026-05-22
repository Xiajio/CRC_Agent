# WorkspacePage Refactor Design

Date: 2026-04-28

## Goal

Refactor `frontend/src/pages/workspace-page.tsx` for demo and UI acceptance work by moving business flow logic into focused workspace hooks while preserving the current UI behavior.

This is a maintainability refactor. It must not change backend APIs, SSE reducer semantics, patient/doctor scene behavior, upload behavior, or the existing `window.__chatLatency` debug bridge.

## Current Problem

`WorkspacePage` currently coordinates too many responsibilities:

- patient and doctor scene switching
- session state and persisted session ids through `useSceneSessions`
- prompt drafts and optimistic message insertion
- streaming turn submission, abort, supersede, and reducer updates
- message history loading
- reset behavior
- upload validation, upload status, upload API calls, and post-upload refresh
- card merging and visible-card derivation
- triage question visibility
- latency probe state and debug trace bridge
- final doctor and patient view prop wiring

Some responsibilities have already been extracted into `useSceneSessions`, `useDatabaseWorkbench`, `usePatientRegistry`, and `stream-reducer.ts`, but the page still contains enough workflow logic to make future UI changes risky.

## Confirmed Direction

Use the boundary:

```text
WorkspacePage = page orchestration layer
workspace hooks = business flow and state derivation
stream-reducer.ts = single SSE state transition entry point
window.__chatLatency = temporary page-level debug bridge
```

`WorkspacePage` should still compose the patient and doctor views. It should not directly understand streaming internals, upload internals, card merge rules, triage question derivation, or latency state-machine details.

## Non-Goals

- Do not introduce `PatientWorkspaceScene` or `DoctorWorkspaceScene` container components in this refactor.
- Do not rewrite `stream-reducer.ts`.
- Do not change the backend API or SSE event schema.
- Do not move `window.__chatLatency` into a hook yet.
- Do not add session persistence or production auth changes.
- Do not change visual design except where prop names or wiring require mechanical updates.

## Architecture

Add focused hooks under `frontend/src/features/workspace/`.

### `WorkspacePage`

Responsibilities:

- get base dependencies: `apiClient`, scene sessions, patient registry, registry browser, and database workbench
- call workspace hooks
- select active patient or doctor view
- wire props into existing UI components
- keep `window.__chatLatency` debug bridge wiring for now

It should no longer own the streaming turn lifecycle, upload lifecycle, card derivation, or probe state transitions.

### `useWorkspaceStreamingTurn`

Scene-scoped hook. `WorkspacePage` calls it once for patient and once for doctor.

Responsibilities:

- manage one scene's turn lifecycle
- submit prompts
- optimistically append the user message
- stream events from `apiClient.streamTurn`
- update session state only through `reduceStreamEvent`
- load paginated message history
- reset the scene session
- abort active work for reset, scene switch, or supersede

It must not own card visibility, layout props, patient upload behavior, or `window.__chatLatency`.

Suggested shape:

```ts
type WorkspaceStreamingTurnController = {
  isStreaming: boolean;
  isLoadingHistory: boolean;
  errorMessage: string | null;
  submitPrompt(prompt: string, context?: Record<string, unknown>): Promise<void>;
  loadMessageHistory(): Promise<void>;
  resetScene(): Promise<void>;
  abortActiveTurn(reason: "scene_switch" | "reset" | "superseded"): void;
};
```

Doctor-only workflow priming should be injected as a callback such as `primeInitialState`. The streaming hook should not hard-code doctor behavior.

### `useTurnLatencyProbe`

Pure probe state hook.

Responsibilities:

- begin a turn
- mark message done
- mark UI complete
- mark aborted, superseded, or error
- clear a scene's latency state
- calculate `ConversationLatencyStatus` per scene

It must not call APIs, mutate session state, consume SSE directly, or write `window.__chatLatency`.

Suggested shape:

```ts
type TurnLatencyProbeController = {
  activeProbe: TurnLatencyProbe | null;
  recentCompletedProbes: Record<Scene, TurnLatencyProbe | null>;
  beginTurn(input: BeginTurnInput): TurnLatencyProbe;
  markMessageDone(input: MessageDoneInput): void;
  markUiComplete(input: UiCompleteInput): void;
  markAborted(reason: "scene_switch" | "reset" | "superseded"): void;
  markError(scene: Scene, message: string): void;
  clearScene(scene: Scene): void;
  latencyStatusForScene(scene: Scene): ConversationLatencyStatus | null;
};
```

The page remains responsible for exposing the trace store through `window.__chatLatency`.

### `usePatientUploads`

Patient-session upload hook.

Responsibilities:

- validate upload size before API calls
- maintain `isUploading`, `uploadStatus`, and upload-specific error
- call `apiClient.uploadFile`
- update `uploadedAssets` optimistically from the upload response
- refresh the patient session through explicit callbacks
- translate upload errors, including backend 413 responses, into user-facing copy

It must not call `reduceStreamEvent` and must not reach into streaming hook internals.

Suggested shape:

```ts
type PatientUploadsController = {
  isUploading: boolean;
  uploadStatus: string | null;
  errorMessage: string | null;
  uploadFile(file: File): Promise<void>;
  clearUploadStatus(): void;
};
```

### `useWorkspaceCards`

Pure derivation hook.

Responsibilities:

- merge top-level session cards with inline message cards
- derive patient visible cards
- derive doctor visible cards
- determine active patient triage question id
- keep patient-side triage question card out of the background panel when appropriate

It must not fetch, reset, upload, or mutate session state.

Suggested shape:

```ts
type WorkspaceCardsModel = {
  patientVisibleCards: Record<string, JsonObject>;
  doctorVisibleCards: Record<string, JsonObject>;
  activePatientTriageQuestionId: string | null;
};
```

### `usePatientWorkspaceNav`

Lightweight patient workspace tab hook.

Responsibilities:

- hold the active patient workspace tab
- expose production nav items
- accept only supported tab selections
- reset the tab when requested

Navigation constants may live in a separate `patient-workspace-nav.ts` file if that keeps the hook smaller.

## Data Flow

### Prompt Submit

```text
WorkspacePage
 -> activeTurn.submitPrompt(prompt, context)
 -> useWorkspaceStreamingTurn
 -> optimistic user message
 -> apiClient.streamTurn(...)
 -> SSE event
 -> reduceStreamEvent(...)
 -> session state
 -> useWorkspaceCards derived cards
 -> UI props
```

### Upload

```text
UploadsPanel
 -> usePatientUploads.uploadFile(file)
 -> validate size
 -> apiClient.uploadFile(...)
 -> update uploadedAssets
 -> apiClient.getSession(patientSessionId)
 -> applyResponseToScene("patient", response)
 -> useWorkspaceCards derives updated cards
```

### Scene Switch

```text
WorkspacePage.handleSceneSwitch(nextScene)
 -> activeTurn.abortActiveTurn("scene_switch")
 -> useWorkspaceStreamingTurn marks the injected latency probe aborted
 -> clear scene error as needed
 -> setActiveScene(nextScene)
```

### Reset

```text
Reset button
 -> activeTurn.resetScene()
 -> abort current scene turn
 -> useWorkspaceStreamingTurn clears or aborts the injected latency probe for the scene
 -> apiClient.resetSession(sessionId)
 -> applyResponseToScene(activeScene, response)
 -> WorkspacePage clears draft and patient upload status for the active scene
```

## Boundary Rules

1. `stream-reducer.ts` remains the only place that translates SSE events into session state.
2. Hooks communicate through explicit parameters and callbacks rather than importing each other's internals.
3. `window.__chatLatency` remains page-level in this refactor.
4. New hooks do not render UI components.
5. Do not create a large `useWorkspaceController`; each hook must have one clear responsibility.
6. `useWorkspaceStreamingTurn` is scene-scoped and is called separately for patient and doctor.
7. `useWorkspaceStreamingTurn` may call methods on the injected latency probe controller, but it must not own probe state or debug-surface wiring.

## Testing Strategy

Keep page-level tests for orchestration and add focused hook tests for moved behavior.

### Keep in `workspace-page.test.tsx`

- patient and doctor scene switching
- patient profile/upload tab placement
- reset button wiring for the active scene
- `window.__chatLatency` bridge availability
- doctor scene prop wiring for patient registry, database workbench, and registry browser
- top-level ConversationPanel, UploadsPanel, and DoctorSceneShell wiring

These tests should verify page composition instead of every internal workflow step.

### Add `use-workspace-streaming-turn.test.tsx`

Cover:

- optimistic user message insertion
- SSE event forwarding through `reduceStreamEvent`
- `message.done` latency probe notification
- stream error to reducer error event
- abort and supersede behavior
- history loading through `getMessages` and `mergeMessageHistory`
- reset behavior and response application
- injected doctor workflow priming callback

### Add `use-turn-latency-probe.test.tsx`

Cover:

- turn start
- message done
- UI complete latency calculation
- aborted, superseded, reset, and error paths
- independent patient and doctor completed latency values
- stale sequence protection

### Add `use-patient-uploads.test.tsx`

Cover:

- oversized upload rejection before API call
- successful upload then session refresh
- backend 413 to friendly size message
- missing patient session error
- upload disabled/status behavior
- upload failure without session refresh

### Add `use-workspace-cards.test.ts`

Cover:

- state card and inline card merging
- patient visible card filtering
- active triage question id derivation
- doctor cards without patient triage visibility rules
- no side effects

Existing tests for `stream-reducer.ts`, `useSceneSessions`, and `chat-latency-trace` stay in place.

## Rollout Plan

1. Move pure helper types and functions out of `WorkspacePage` without changing behavior.
2. Extract `useTurnLatencyProbe` and add focused tests.
3. Extract `useWorkspaceCards` and add focused tests.
4. Extract `usePatientUploads` and add focused tests.
5. Extract `useWorkspaceStreamingTurn` in two smaller steps:
   - submit and SSE consumption
   - history, reset, abort, and scene switch integration
6. Rename page-level variables and props after logic is moved.
7. Trim `workspace-page.test.tsx` to page orchestration coverage.

Each step should leave `WorkspacePage` runnable and should pass focused tests before proceeding.

## Verification

Run focused frontend tests after each extraction:

```text
npm test -- --run src/pages/workspace-page.test.tsx src/features/workspace/*.test.tsx src/app/store/stream-reducer.test.ts src/app/api/chat-latency-trace.test.ts
```

Run the frontend build before calling the implementation complete:

```text
npm run build
```

If local environment constraints prevent the full build, run the focused Vitest set and document the build blocker.

## Risks

- The largest risk is moving stream and latency behavior at the same time. Mitigation: extract `useTurnLatencyProbe` before `useWorkspaceStreamingTurn`.
- Page tests may overfit to internals. Mitigation: migrate behavior assertions into hook tests and keep page tests focused on wiring.
- Debug trace behavior can regress if the trace store ownership changes. Mitigation: keep `window.__chatLatency` in `WorkspacePage` for this refactor.
- The streaming hook can become a second large page. Mitigation: make it scene-scoped and inject doctor-only behavior.

## Acceptance Criteria

- `WorkspacePage` reads as an orchestration component rather than a workflow implementation.
- Streaming, upload, cards, latency, and nav logic live in separate focused hooks or helper modules.
- `stream-reducer.ts` remains the only SSE state transition entry point.
- Existing patient and doctor UI behavior is preserved.
- `window.__chatLatency` still works from the page layer.
- Focused hook tests cover the behavior moved out of the page.
- No backend API changes are required.
