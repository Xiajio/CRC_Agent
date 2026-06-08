# Whole Body Anatomy Overview Design

## Goal

Add a whole-body overview to the existing colorectal anatomy locator so clinicians can see the tumor's body-level context before reading the precise colorectal segment highlight.

## Approved Direction

Use option A: extend the existing `AnatomyHighlightPanel` with a paired visualization.

- Show a simple front-facing whole-body SVG overview.
- Highlight only the lower abdomen and pelvis when colorectal location evidence exists.
- Keep the existing colorectal segment map in the same card.
- Link both views to the same resolved anatomy state.
- Do not expand the product into a full multi-organ body atlas.

## Current Context

The frontend already has:

- `resolveAnatomyRegions(detail)` for structured and text-based colorectal location resolution.
- `ColorectalAnatomyMap` for precise colorectal segment highlighting.
- `AnatomyHighlightPanel` embedded in the doctor consultation sidebar and multimodal view.
- Motion tokens, GSAP scoped animation helpers, and reduced-motion handling.

This feature should reuse those seams rather than introducing a parallel anatomy model.

## User Experience

The anatomy card becomes a two-level locator:

1. Whole-body overview: a quiet, compact human silhouette with a lower-abdomen/pelvis active region.
2. Colorectal segment map: the current detailed map showing cecum, colon, rectosigmoid, rectum, and anus segments.

When the patient detail resolves to one or more colorectal segments, the whole-body lower-abdomen region is active. When no location signal is available, the overview remains inactive and the card keeps the current "location not available" copy.

Clicking the whole-body active region sends a general colorectal-location prompt. Clicking a precise colorectal segment keeps the existing segment-specific prompt.

## Component Design

Add a focused component, tentatively `WholeBodyAnatomyOverview`, under `frontend/src/features/anatomy/`.

Public props:

- `active`: whether any colorectal location signal exists.
- `disabled`: whether prompt interactions are disabled.
- `onRegionSelect`: callback for the lower-abdomen/pelvis overview region.

The component renders SVG only. It does not parse patient data and does not own prompt text. `AnatomyHighlightPanel` remains responsible for data resolution and prompt dispatch.

`AnatomyHighlightPanel` will:

- Compute `hasResolvedRegion = resolved.regionCodes.length > 0`.
- Render `WholeBodyAnatomyOverview` before or beside `ColorectalAnatomyMap`.
- Use the whole-body click to call a new general prompt builder.
- Keep the existing legend and source label.

## Data Flow

No backend or API change is needed.

`PatientRegistryDetail` already supplies `tumor_location`, `tumor_region_code`, and `tumor_region_codes`. The existing resolver continues to produce the source of truth. The whole-body view only derives active/inactive state from the resolver output.

## Interaction And Accessibility

- The whole-body lower-abdomen region is keyboard focusable when prompt actions are available.
- It uses `role="button"`, an accessible Chinese label, `aria-pressed`, and `aria-disabled` consistently with `ColorectalAnatomyMap`.
- Reduced-motion users see the same active state without animation.
- No hover-only information is required.

## Visual Treatment

- Use the existing clinical card density and restrained palette.
- The body outline should be schematic and clinical, not decorative or anatomical-photo-like.
- The active lower-abdomen region uses the existing success/accent highlight treatment.
- Motion is limited to transform and opacity, using the existing motion tokens.
- The card must remain stable in the narrow doctor sidebar and the multimodal left column.

## Testing

Use test-first implementation.

Add or update frontend tests for:

- `AnatomyHighlightPanel` rendering the whole-body overview.
- Whole-body lower-abdomen region active when a colorectal segment is resolved.
- Whole-body region inactive when no location signal exists.
- Whole-body click submitting a general anatomy prompt with patient context.
- Existing colorectal segment click behavior remaining unchanged.

Update motion/CSS contract tests if new classes or animation hooks are added.

## Non-Goals

- No whole-body multi-organ diagnosis model.
- No new backend schema.
- No new dependencies.
- No image-generation asset requirement.
- No replacement of the existing colorectal segment map.
- No route or navigation changes.

## Acceptance Criteria

- The doctor consultation sidebar and multimodal view show a whole-body overview inside the existing anatomy card.
- Resolved colorectal locations activate both the whole-body lower-abdomen overview and the precise colorectal segment map.
- Clicking the whole-body active region sends the approved general prompt path.
- Clicking a specific segment still sends the existing segment prompt path.
- Tests and build pass.
