# Clinical Visual Polish Design

## Goal

Refine the clinical workspace visual system without changing business logic or navigation structure. The interface should feel quieter, more polished, and more useful in empty or low-data states.

## Approved Direction

Use the conservative polish approach:

- Keep the existing consultation cockpit layout.
- Improve visual hierarchy through shared card, header, empty-state, composer, and top-nav treatment.
- Add one small reusable empty-state component instead of broad component restructuring.
- Avoid decorative-heavy, marketing-like, or one-color visual treatment.

## Scope

### Visual System

- Clinical cards keep 8px radius but gain a softer border and more deliberate shadow.
- Panel headers use consistent height, icon containers, and muted title rhythm.
- Composer and status row receive a cleaner working-surface treatment.

### Empty States

- Introduce `ClinicalEmptyState` for low-data surfaces.
- Empty states have an icon, short title, and concise message.
- Empty states remain compact in side rails and event stream areas.

### Top Navigation

- Preserve the dark clinical header and current navigation labels.
- Refine active tab treatment, status pill, reset button, avatar, and spacing.
- Mobile wrapping must remain readable and stable.

## Non-Goals

- No route changes.
- No backend or data-model changes.
- No new dependency.
- No large component rewrite.
- No generated visual assets.

## Validation

- Unit tests for `ClinicalEmptyState`.
- Existing component tests continue to pass.
- CSS contract test covers the visual-system classes.
- Build passes.
- Browser validation covers desktop and mobile consultation workspace.
