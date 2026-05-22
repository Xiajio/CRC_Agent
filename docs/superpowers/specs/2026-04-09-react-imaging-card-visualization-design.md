# React Imaging Card Visualization Design

**Date:** 2026-04-09  
**Status:** Approved for planning  
**Goal:** Restore true imaging-card visualization in the current React + FastAPI platform so the frontend shows real preview images instead of only summary counts.

## 1. Context

The legacy Streamlit app in `app/app/views/` rendered imaging cards as visual UI, including image previews and card-level actions. The new platform already has:

- backend SSE + snapshot delivery in `backend/`
- React card rendering in `frontend/src/features/cards/`
- fixture data that already contains `image_base64` previews in `tests/fixtures/graph_ticks/database_case.json`

The current failure is not that the platform cannot render cards. The real issue is that the backend adapter layer strips media-bearing fields before cards reach the frontend:

- `backend/api/adapters/event_normalizer.py`
- `backend/api/adapters/state_snapshot.py`

As a result, the frontend only sees summary text such as total image count.

## 2. Platform Principles

This work must follow the new platform boundaries instead of recreating Streamlit behavior.

- Preserve the capability of legacy cards, not the Streamlit implementation details.
- Keep LangGraph and card production in `src/` as-is unless a real production bug requires source changes.
- Treat `backend/` as the transport and sanitization layer for card payloads.
- Treat `frontend/` as the rendering layer for standardized card payloads.
- Avoid moving back to `session_state`, popup flags, or Streamlit-specific UI control flow.

## 3. Scope

### In scope for phase 1

- Restore visible imaging previews for `imaging_card`
- Preserve previews in both stream events and session snapshots
- Keep payload size bounded by sending preview subsets instead of all available images
- Add regression coverage at adapter, route, component, and page levels

### Out of scope for phase 1

- Full parity with every legacy Streamlit card interaction
- Full-image browsing for all 176+ images in a study
- A generic media service migration for every card type
- Recreating Streamlit popup/show-hide state machines

## 4. Current-State Findings

### 4.1 Existing frontend capability

`frontend/src/features/cards/card-renderers.tsx` already renders an image gallery when card payloads include `image_base64`. The imaging card renderer is therefore already capable of showing previews.

### 4.2 Existing backend behavior

`event_normalizer.py` and `state_snapshot.py` each define a generic `_strip_binary()` helper that removes keys containing:

- `base64`
- `blob`
- `binary`
- `bytes`
- `artifact`

This blanket removal is correct for large opaque payloads, but it also deletes the exact preview fields the React imaging card needs.

### 4.3 Existing test evidence

- `tests/frontend/clinical-cards-panel.test.tsx` already expects thumbnail rendering when `image_base64` exists.
- `tests/fixtures/graph_ticks/database_case.json` already carries preview-ready image data.

This confirms the regression sits between card production and card rendering.

## 5. Proposed Design

## 5.1 Data contract for phase 1

Keep the existing imaging-card shape so the React renderer does not need a protocol rewrite:

```json
{
  "type": "imaging_card",
  "data": {
    "folder_name": "093",
    "total_images": 176,
    "images": [
      {
        "image_name": "I1000000.png",
        "image_base64": "<preview>",
        "image_path": "<optional>",
        "image_url": "<optional-future>"
      }
    ]
  },
  "text_summary": "Imaging sample: patient 093, total 176 images."
}
```

For phase 1, `images` remains the field consumed by React. No new `preview_images` field is required.

## 5.2 Adapter-layer media sanitization

Replace the duplicated generic sanitization with a shared adapter utility, for example:

- `backend/api/adapters/card_payload_sanitizer.py`

Responsibilities:

- keep the existing generic binary stripping behavior for most payloads
- apply a whitelist for `imaging_card`
- preserve only the preview-safe fields needed by the frontend
- cap preserved preview images to a fixed limit such as `8`

Recommended whitelist for `imaging_card.data.images[*]`:

- `image_name`
- `image_base64`
- `image_url`
- `image_path` only if it is still useful for debugging or future server-side URL derivation

Everything else should continue to be sanitized.

## 5.3 Shared use in both delivery paths

The same sanitization rules must be applied in both:

- `backend/api/adapters/event_normalizer.py`
- `backend/api/adapters/state_snapshot.py`

If only one path changes, the user will either:

- see previews during streaming but lose them after refresh, or
- recover previews from snapshots but never see them in live streaming

Both paths must stay consistent.

## 5.4 Frontend rendering shape

The React imaging card should keep a three-part structure:

1. Summary block  
   Patient id, total image count, preview count, source folder.

2. Preview block  
   One large selected image plus a thumbnail strip.

3. Fallback block  
   If no preview images survive sanitization, show a clear empty state instead of only raw counts.

Phase 1 may keep the current component structure in `card-renderers.tsx` and only make targeted UX hardening changes.

## 6. Limits and Safety

Phase 1 must explicitly bound payload size.

- Preserve only the first `4-8` previews per imaging card
- Continue stripping non-whitelisted binary fields
- Do not stream all full-resolution study images over SSE

This keeps the current architecture usable while restoring visible output.

## 7. Testing Strategy

### Backend adapter tests

- `tests/backend/test_event_normalizer.py`
- `tests/backend/test_state_snapshot.py`

Verify:

- imaging previews are preserved for whitelisted fields
- preview count is capped
- unrelated binary fields are still removed

### Backend route test

- `tests/backend/test_chat_stream_route.py`

Verify:

- fixture-backed SSE responses include an `imaging_card` event with preview image data

### Frontend component test

- `tests/frontend/clinical-cards-panel.test.tsx`

Verify:

- thumbnail rendering
- primary preview rendering
- thumbnail switching
- empty-state fallback

### Frontend page test

- `tests/frontend/workspace-page.test.tsx`

Verify:

- imaging previews render from bootstrap snapshots
- imaging previews remain visible after recovery snapshots

## 8. Acceptance Criteria

Phase 1 is done when:

1. A session that emits `imaging_card` shows real preview images in the React card panel.
2. The card still shows patient id, total image count, and source folder.
3. Refreshing the page keeps the preview images through snapshot recovery.
4. The stream and snapshot paths share one sanitization contract.
5. Other card types keep their current behavior.

## 9. Follow-up Work

Phase 2 can migrate from inline `image_base64` previews to asset-backed media URLs using `/api/assets/{asset_id}` or a dedicated derived-preview endpoint. That should only happen after phase 1 restores visible parity for the imaging card.
