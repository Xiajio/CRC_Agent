# React Imaging Card Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore visible imaging previews in the current React + FastAPI platform by preserving safe `imaging_card` preview media through the backend adapter layer and verifying the result in the frontend workspace.

**Architecture:** Keep card production in `src/` unchanged for phase 1. Introduce a shared adapter-level payload sanitizer in `backend/api/adapters/` so both SSE events and session snapshots preserve a bounded subset of imaging previews. Reuse the existing React imaging card renderer in `frontend/src/features/cards/card-renderers.tsx` and tighten it only where needed for empty states and preview switching.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, LangGraph, pytest, httpx, React `18.3.1`, Vite `5.4.14`, TypeScript `5.6.3`, Vitest `2.1.8`, React Testing Library `16.1.0`

---

**Repository note:** `D:\亿铸智能体\LangG_New` is currently not a git repository. Replace each checkpoint with `git add` / `git commit` once the repo is initialized. Until then, use the listed files as manual checkpoint boundaries.

## Planned File Structure

**Backend**

- Create: `backend/api/adapters/card_payload_sanitizer.py`
- Modify: `backend/api/adapters/event_normalizer.py`
- Modify: `backend/api/adapters/state_snapshot.py`

**Frontend**

- Modify: `frontend/src/features/cards/card-renderers.tsx`

**Tests**

- Modify: `tests/backend/test_event_normalizer.py`
- Modify: `tests/backend/test_state_snapshot.py`
- Modify: `tests/backend/test_chat_stream_route.py`
- Modify: `tests/frontend/clinical-cards-panel.test.tsx`
- Modify: `tests/frontend/workspace-page.test.tsx`

### Task 1: Preserve Imaging Preview Media in Backend Adapters

**Files:**
- Create: `backend/api/adapters/card_payload_sanitizer.py`
- Modify: `backend/api/adapters/event_normalizer.py`
- Modify: `backend/api/adapters/state_snapshot.py`
- Test: `tests/backend/test_event_normalizer.py`
- Test: `tests/backend/test_state_snapshot.py`

- [ ] **Step 1: Write the failing backend tests for imaging preview preservation**

Add focused cases to `tests/backend/test_event_normalizer.py` and `tests/backend/test_state_snapshot.py` that prove:

- `imaging_card.data.images[*].image_base64` survives sanitization
- previews are capped to a fixed limit such as `8`
- unrelated `blob` / `artifact` / non-whitelisted binary fields are still removed

Suggested test shape:

```python
def test_normalize_tick_preserves_whitelisted_imaging_previews():
    imaging_card = {
        "type": "imaging_card",
        "data": {
            "folder_name": "093",
            "total_images": 176,
            "images": [
                {"image_name": "I1.png", "image_base64": "AAA", "blob": "drop-me"},
                {"image_name": "I2.png", "image_base64": "BBB"},
            ],
        },
    }

    events = normalize_tick("case_database", {"imaging_card": imaging_card}, messages=[])

    card = next(event for event in events if event.type == "card.upsert")
    assert card.card_type == "imaging_card"
    assert card.payload["data"]["images"][0]["image_base64"] == "AAA"
    assert "blob" not in json.dumps(card.payload)
```

- [ ] **Step 2: Run the backend adapter tests to verify failure**

Run:

```powershell
pytest tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py -v
```

Expected: FAIL because the current generic `_strip_binary()` removes `image_base64` from imaging-card payloads.

- [ ] **Step 3: Implement a shared card payload sanitizer**

Create `backend/api/adapters/card_payload_sanitizer.py` with a shared API that both adapters can call.

Suggested structure:

```python
IMAGING_PREVIEW_LIMIT = 8

def strip_binary(value: Any) -> Any:
    ...

def sanitize_card_payload(card_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if card_type == "imaging_card":
        return sanitize_imaging_card_payload(payload)
    return strip_binary(dict(payload))

def sanitize_imaging_card_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    ...
```

Implementation rules:

- preserve only safe preview fields from `imaging_card.data.images`
- cap previews to a fixed count
- reuse the helper in both `event_normalizer.py` and `state_snapshot.py`
- remove duplicated local `_strip_binary()` logic where practical

- [ ] **Step 4: Run the backend adapter tests again**

Run:

```powershell
pytest tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py -v
```

Expected: PASS with imaging preview media preserved only for the whitelisted imaging-card path.

- [ ] **Step 5: Checkpoint the backend adapter changes**

Files to verify:

- `backend/api/adapters/card_payload_sanitizer.py`
- `backend/api/adapters/event_normalizer.py`
- `backend/api/adapters/state_snapshot.py`
- `tests/backend/test_event_normalizer.py`
- `tests/backend/test_state_snapshot.py`

### Task 2: Prove the Streaming Path Still Delivers Preview Images

**Files:**
- Modify: `tests/backend/test_chat_stream_route.py`
- Modify: `tests/backend/test_event_normalizer.py` only if shared helpers require fixture assertions there

- [ ] **Step 1: Write the failing SSE regression test**

Extend `tests/backend/test_chat_stream_route.py` so the fixture-backed stream is required to emit an `imaging_card` event that still contains preview media.

Suggested assertion shape:

```python
card_events = [
    event for event in events
    if event["event"] == "card.upsert"
    and event["data"]["card_type"] == "imaging_card"
]

assert card_events
images = card_events[0]["data"]["payload"]["data"]["images"]
assert images
assert "image_base64" in images[0]
```

- [ ] **Step 2: Run the SSE regression test to verify failure**

Run:

```powershell
pytest tests/backend/test_chat_stream_route.py -v
```

Expected: FAIL before the adapter sanitizer is wired correctly through the streaming path.

- [ ] **Step 3: Finish any backend integration adjustments**

If the new sanitizer is not yet used uniformly, update the event path until the streamed `card.upsert` event carries the preserved imaging previews without reintroducing large arbitrary blobs.

Keep the streaming contract unchanged:

- event name remains `card.upsert`
- `card_type` remains `imaging_card`
- payload shape remains compatible with the React card renderer

- [ ] **Step 4: Run the backend stream tests**

Run:

```powershell
pytest tests/backend/test_chat_stream_route.py tests/backend/test_graph_service.py -v
```

Expected: PASS with no regression to heartbeat, error, or concurrent-run behavior.

- [ ] **Step 5: Checkpoint the backend stream coverage**

Files to verify:

- `tests/backend/test_chat_stream_route.py`
- any additional backend files touched while wiring the shared sanitizer through the stream path

### Task 3: Harden the React Imaging Card Renderer for Visible Preview Output

**Files:**
- Modify: `frontend/src/features/cards/card-renderers.tsx`
- Test: `tests/frontend/clinical-cards-panel.test.tsx`

- [ ] **Step 1: Write the failing frontend renderer tests**

Extend `tests/frontend/clinical-cards-panel.test.tsx` to cover:

- main preview + thumbnail strip rendering when previews are present
- switching the selected image when a thumbnail is clicked
- a clear empty state when the imaging card exists but contains no usable previews

Suggested interaction case:

```tsx
fireEvent.click(screen.getByRole("button", { name: /I1000001.png/i }))
expect(screen.getByRole("img", { name: "I1000001.png" })).toBeInTheDocument()
```

- [ ] **Step 2: Run the frontend card tests to verify failure**

Run:

```powershell
cd frontend
npm run test -- --run ../tests/frontend/clinical-cards-panel.test.tsx
```

Expected: FAIL if the renderer does not yet expose a stable empty state or thumbnail selection contract.

- [ ] **Step 3: Tighten the imaging renderer without changing the card protocol**

Update `frontend/src/features/cards/card-renderers.tsx` so that:

- it prefers preview images from `data.images`
- it supports both `image_base64` and, if present later, `image_url`
- it renders a stable empty-state message when previews are unavailable
- it keeps summary metadata visible even when preview media is missing

Suggested helper shape:

```tsx
function previewImageSrc(image: JsonObject): string | null {
  const base64 = asString(image.image_base64)
  if (base64) return `data:image/png;base64,${base64}`
  return asString(image.image_url)
}
```

- [ ] **Step 4: Run the frontend card tests again**

Run:

```powershell
cd frontend
npm run test -- --run ../tests/frontend/clinical-cards-panel.test.tsx
```

Expected: PASS with visible previews and stable fallback behavior.

- [ ] **Step 5: Checkpoint the imaging renderer changes**

Files to verify:

- `frontend/src/features/cards/card-renderers.tsx`
- `tests/frontend/clinical-cards-panel.test.tsx`

### Task 4: Verify Workspace-Level Snapshot Recovery in the New Platform

**Files:**
- Modify: `tests/frontend/workspace-page.test.tsx`

- [ ] **Step 1: Write the failing workspace-page tests**

Add page-level cases proving that an imaging card with preview media:

- renders from the bootstrap snapshot
- remains visible after a recovery `getSession()` call

Suggested snapshot shape:

```tsx
cards: [
  {
    type: "card.upsert",
    card_type: "imaging_card",
    payload: {
      type: "imaging_card",
      data: {
        folder_name: "093",
        total_images: 176,
        images: [
          {"image_name": "I1000000.png", "image_base64": "..."},
        ],
      },
    },
    source_channel: "state",
  },
]
```

- [ ] **Step 2: Run the workspace-page tests to verify failure**

Run:

```powershell
cd frontend
npm run test -- --run ../tests/frontend/workspace-page.test.tsx
```

Expected: FAIL before the snapshot path reliably preserves imaging preview payloads end to end.

- [ ] **Step 3: Finish any remaining integration fixes**

If the page test still fails after Tasks 1-3, fix the integration at the narrowest point:

- snapshot hydration in `state_snapshot.py`
- card rendering assumptions in `card-renderers.tsx`
- card selection behavior in `clinical-cards-panel.tsx` only if required

Do not introduce Streamlit-style popup state or rewrite the workspace shell.

- [ ] **Step 4: Run the full targeted regression suite**

Run:

```powershell
pytest tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py tests/backend/test_chat_stream_route.py -v
```

```powershell
cd frontend
npm run test -- --run ../tests/frontend/clinical-cards-panel.test.tsx ../tests/frontend/workspace-page.test.tsx
```

Expected: PASS with imaging previews visible in both card-level and workspace-level tests.

- [ ] **Step 5: Checkpoint the integration boundary**

Files to verify:

- `tests/frontend/workspace-page.test.tsx`
- any remaining frontend files touched to complete snapshot-driven preview rendering

## Completion Criteria

- `imaging_card` previews are visible in the React UI
- previews survive both SSE delivery and session snapshot recovery
- payload size is bounded by a preview limit
- non-whitelisted binary fields are still stripped
- no Streamlit-specific UI state model is reintroduced
