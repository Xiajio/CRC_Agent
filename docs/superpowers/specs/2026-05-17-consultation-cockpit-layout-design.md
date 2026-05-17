# Consultation Cockpit Layout Design

Date: 2026-05-17

## Goal

Optimize the existing doctor consultation page so the center consultation workspace is the primary visual area. The page should feel like a focused clinical cockpit: patient context on the left, conversation and cards in the center, workflow progress and evidence on the right, and the event stream as a secondary status band.

## Current Surface

The existing doctor consultation page is built from:

- `DoctorSceneShell`
- `ClinicalPatientSummary`
- `ClinicalUploads`
- `DoctorConsultationView`
- `ConversationPanel`
- `ClinicalCardsPanel`
- `RoadmapPanel`
- `ExecutionPlanPanel`
- `ClinicalEventStream`

The current desktop grid uses a large right column and a bottom event area. This makes the conversation compete with the roadmap, execution plan, and event stream.

## Approved Direction

The selected direction is "consultation cockpit first":

- Keep the existing four information zones.
- Make the center consultation column the strongest visual area.
- Narrow the left patient-context column.
- Narrow the right workflow/evidence column from its current oversized desktop width.
- Convert the event stream into a compact status band below the main center area instead of a competing large panel.
- Keep changes focused on layout and styling for the first implementation pass.

## Desktop Layout

For wide desktop viewports, use a three-column grid with a compact event row:

- Left column: patient summary and uploads, about 260-280 px.
- Center column: conversation and medical cards, `minmax(0, 1fr)`, visually dominant.
- Right column: roadmap, execution plan, references, and status, about 340-380 px.
- Event band: spans the left and center columns below the main row; the right column remains vertically grouped for workflow/evidence.

The conversation card should have the strongest hierarchy through size, stable height, and clear composer placement. It should not be visually buried under equal-weight cards.

## Responsive Layout

Use existing breakpoints as much as possible:

- At mid-size desktop/tablet widths, reduce fixed side widths before stacking.
- Below the current tablet breakpoint, stack the right column below the center column.
- On mobile, use a single column order: consultation, patient summary, workflow/evidence, cards, event stream.
- Avoid horizontal overflow and fixed min-widths that force cramped columns.

## Component Scope

The first implementation should preserve component boundaries:

- No new business state.
- No new backend API.
- No major component split unless a small wrapper is needed for layout clarity.
- Existing tests should remain valid, with targeted updates only when layout-specific expectations change.

## Visual Rules

- Keep the restrained clinical palette.
- Avoid decorative hero or marketing layout patterns.
- Keep cards at 8 px radius or less.
- Do not nest cards inside cards.
- Use denser but readable spacing for dashboard panels.
- Preserve code-native UI text and controls.
- Do not introduce placeholder feature labels beyond existing product copy.

## Acceptance Criteria

1. On the doctor consultation tab, the center conversation area is visually dominant on desktop.
2. The right workflow/evidence column no longer consumes 500 px on desktop.
3. The event stream is secondary and does not compete with the conversation area.
4. No desktop horizontal overflow at 1280 px width.
5. The page stacks coherently at tablet and mobile widths.
6. Existing patient and doctor scene switching still works.
7. Existing frontend unit tests pass, or any failures are explained and fixed.
8. Browser QA verifies page identity, nonblank render, no framework overlay, console health, and at least one scene/navigation interaction.

## Out Of Scope

- Human review task workflow.
- New patient timeline/version UI.
- New multimodal result drilldown.
- Backend API changes.
- New generated visual assets.
