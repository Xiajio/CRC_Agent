# Apple-Inspired Clinical Workspace Design

Date: 2026-05-18

## Goal

Raise the doctor consultation workspace from a functional dashboard to a premium clinical product surface. The design should borrow Apple-like principles of focus, restraint, material layering, and exact typography without copying Apple branding or turning the clinical workspace into a marketing page.

The approved direction is **Apple Product Stage + controlled Liquid Glass**:

- The consultation conversation becomes the main product stage.
- Navigation, status, and utility controls become a light functional layer above the content.
- Supporting clinical context remains available but visually secondary.
- Empty states become quiet placeholders rather than repeated "暂无" panels.
- Icons move from CSS-drawn symbols to a precise, consistent vector icon system.

## Reference Principles

Sources reviewed:

- Apple homepage: product-first composition, direct copy, large negative space, minimal navigation.
- Apple OS overview: focus on content, dynamic controls, and design continuity across device sizes.
- Apple Newsroom software design update: Liquid Glass as a translucent functional layer, navigation and sidebars that preserve context, and controls that give way to content.
- Apple Human Interface Guidelines: clear hierarchy, predictable controls, adaptive layout, and familiar interaction patterns.

These are translated into this product rather than copied literally. The workspace remains a medical decision surface, so legibility, density, and clinical trust outrank decorative effects.

## Current Issues To Solve

- The top navigation is visually heavy and competes with the consultation area.
- Most panels have similar visual weight, so the page feels like a generic admin grid.
- Empty states are clearer after the last polish pass, but still read as repeated component blocks instead of a premium system.
- CSS-drawn icons are fragile and not refined enough for a high-end interface.
- The layout lacks one signature focal point. It is organized, but not yet memorable.

## Visual Direction

### Composition

The page should read as three layers:

1. **Glass command layer**: thin top navigation, scene controls, SSE status, profile switch.
2. **Clinical stage**: large central consultation surface containing conversation, status, composer, and generated cards.
3. **Context rails**: patient context and workflow evidence panels that stay useful but subdued.

The center stage should feel like the product. On desktop, it should own the largest uninterrupted white surface. On mobile, it should appear first after navigation.

### Material System

Use a restrained material stack:

- App background: cool white to very light blue-gray, not beige or dark slate.
- Top command layer: translucent ink-blue glass with blur, subtle border, and controlled shadow.
- Primary stage: true white surface with very soft blue-gray border and a light depth shadow.
- Secondary rails: lower-contrast surfaces, smaller elevation, and reduced header strength.
- Status pills: compact translucent capsules with semantic color only where needed.

Glass effects must remain readable. Avoid strong blur behind text, noisy gradients, or decorative glow blobs.

### Typography

Use the existing font stack unless the project already has a better system font stack. The design should feel closer to Apple system UI:

- Navigation labels: compact, medium weight, no oversized tabs.
- Panel titles: 14-15 px equivalent, strong but not loud.
- Stage title and key state labels: 15-16 px equivalent with precise line height.
- Body and empty-state copy: 13-14 px equivalent, cool gray, generous line height.
- Buttons and controls: deliberate font size and weight, never browser-default.

Letter spacing remains 0. Font size must not scale with viewport width.

### Color Tokens

Target token direction:

- `clinical-bg`: `#f4f7fb` or similar cool white-blue.
- `clinical-ink`: `#071b2f`.
- `clinical-ink-soft`: `rgba(7, 27, 47, 0.72)`.
- `clinical-surface`: `#ffffff`.
- `clinical-surface-glass`: translucent white or ink depending on layer.
- `clinical-border`: `#d9e5f1`.
- `clinical-border-soft`: `rgba(158, 179, 204, 0.28)`.
- `clinical-accent`: cyan-blue for active navigation only.
- `clinical-safe`: green for connection/ready states only.
- `clinical-attention`: amber/red only for clinical risk or blocked states.

The interface must not become a one-hue blue dashboard. Use semantic green sparingly and keep neutral surfaces dominant.

## Component Design

### Top Navigation

Replace the heavy dark bar feel with a thinner command layer:

- Height target: about 52-56 px desktop.
- Background: ink-blue translucent material with blur and a bottom hairline.
- Brand mark and product name remain left aligned and compact.
- Navigation uses small segmented text controls rather than full-height slabs.
- Active tab uses a subtle glass fill plus a thin cyan underline or glow line.
- Reset, SSE, and profile controls group on the right as small capsules.
- On mobile, navigation wraps into two clean rows without oversized tab blocks.

### Consultation Stage

The conversation card becomes the visual anchor:

- Larger uninterrupted white area.
- Header is calm: icon, title, and optional running state only.
- Empty conversation state sits inside the stage with low-contrast iconography and short copy.
- Composer is visually integrated into the stage bottom, like a refined input tray.
- Send button is an icon button with premium alignment and clear disabled/active states.
- The status row should be compact and not split the stage visually.

### Context Rails

Left patient context and upload panels:

- Reduce header weight.
- Use thinner dividers and softer background.
- Empty states use compact inline placeholders.
- Panels should feel like supporting metadata, not equal product cards.

Right workflow/evidence rail:

- Preserve roadmap, execution plan, and references.
- Use tighter vertical rhythm.
- Use small vector icons and low-contrast headers.
- The rail should look like an inspector/sidebar from a pro app, not a stack of standalone cards.

### Empty States

Empty states should become quiet placeholders:

- Replace large repeated icons with smaller, optically centered vector icons.
- Use one concise title and one line of explanation where needed.
- Add subtle skeleton/ghost lines only when they clarify future content shape.
- Avoid repeating "暂无" as the dominant visible word across the screen.
- Keep clinical meaning specific: "等待患者摘要", "等待执行计划", "等待参考资料".

### Icons

Move away from CSS-drawn pseudo-icons for the premium pass:

- Prefer a consistent SVG/lucide-style icon set with 1.75-2 px stroke.
- Use `currentColor` and shared icon sizing tokens.
- All icon containers use stable 28-34 px boxes with optical centering.
- Filled and outline icons should not be mixed casually.
- Header icons, empty-state icons, status icons, and action icons need separate variants.

## Responsive Rules

Desktop:

- Keep three-zone layout: context rail, center stage, workflow rail.
- Center stage has strongest width and depth.
- Top command layer remains one row where width allows.

Tablet:

- Context rails reduce first, then stack below the stage.
- Workflow evidence can become a lower inspector band.
- Composer remains attached to the stage.

Mobile:

- Order: top command, consultation stage, generated cards, patient context, workflow/evidence, event stream.
- Tabs must not overflow horizontally.
- Controls must meet accessible tap targets without becoming oversized.
- Empty states must remain compact and not dominate the first viewport.

## Interaction And Motion

Use restrained motion:

- Active tab and scene changes: 120-180 ms fade/slide.
- Empty-to-content transition: soft opacity and vertical settle.
- Status pulse only for active streaming or connection state.
- Respect reduced-motion preferences.

No decorative animated blobs, orbs, or large marketing-style reveal effects.

## Implementation Scope

This design pass is front-end only:

- Update CSS tokens and layout treatments.
- Refine `ClinicalTopNav`.
- Refine `ClinicalEmptyState` and replace CSS icons with SVG/icon component variants.
- Adjust consultation stage, composer, card, rail, and panel styling.
- Add or update tests for design-system hooks and icon alignment.

No backend changes, route changes, data-model changes, or new clinical logic.

## Acceptance Criteria

1. The first desktop viewport clearly presents the consultation stage as the primary surface.
2. Top navigation reads as a light command layer instead of a heavy page header.
3. Side rails are useful but visually secondary.
4. Empty states look intentional and premium, not like repeated placeholder paragraphs.
5. Icons are optically centered, consistent, and not CSS-shadow drawings.
6. Desktop, tablet, and mobile views have no obvious overlap, clipping, or cramped text.
7. Browser validation checks page identity, nonblank render, no framework overlay, console health, and navigation interaction.
8. Playwright or Browser screenshots are captured for desktop and mobile comparison.
9. Full frontend tests and build pass.

## Non-Goals

- Do not mimic Apple trademarks, product imagery, or exact Apple page layouts.
- Do not create a landing page or marketing hero.
- Do not reduce clinical density so much that the workspace becomes decorative.
- Do not add new large visual assets unless a later concept pass explicitly calls for them.
- Do not introduce new dependencies unless existing icon coverage is insufficient and the user approves.

## Review Checklist Before Implementation

- The user approves this spec as the target direction.
- The implementation plan lists exact files and tests.
- A before/after screenshot comparison is planned.
- Any visual concept image or mockup, if generated later, is treated as the source of truth for implementation fidelity.
