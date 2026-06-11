# Patient ChatGPT-Style Polish Design

## Context

The patient workspace currently uses the `patient-care` light theme with a clinical workstation structure. It has useful capabilities, including patient profile capture, report upload, conversation, light brand logo, and patient-safe session state. The problem is visual and structural: the patient side still feels like a small clinical cockpit, while the desired product experience is closer to a quiet medical Q&A entry point.

The reference direction is the ChatGPT home screen: low noise, centered input, limited default chrome, and a clear primary action. The patient side should learn from that layout discipline without becoming a generic chat toy. It must remain precise, trustworthy, and clinically appropriate.

## Goal

Turn the patient workspace into a centered, lightweight medical assistant entry point with sharper visual boundaries. The first screen should make it obvious that the patient can ask a question, upload a report, or complete profile information, without being confronted by a three-column workstation.

## Non-Goals

- Do not redesign the doctor cockpit.
- Do not change backend APIs or streaming contracts.
- Do not build a full patient portal with notifications, longitudinal plans, or account management.
- Do not introduce 3D, complex motion, or marketing-page hero treatment.
- Do not make the patient side cute, soft, or cartoon-like as a primary visual direction.

## Recommended Approach

Use a "focused patient Q&A home" approach:

1. Default the patient workspace to a new assistant-first tab.
2. Show a centered conversation home when there are no messages.
3. Keep patient profile and upload as secondary tabs or quick actions.
4. Sharpen the light theme through strict boundary, shadow, radius, text, and green-token rules.

This is preferred over a minor visual tweak because the current default profile-first layout is the source of much of the workstation feeling. It is also preferred over a full patient portal because the current product need is focused polish, not a new subsystem.

## Patient Navigation

Patient navigation should contain three user-facing tabs:

- Ask assistant
- My information
- Upload report

The default tab must be Ask assistant. The current profile and upload capabilities remain available. The Ask assistant tab renders the conversation experience and should be the default landing surface for the patient scene.

## First Screen

When the patient has no active messages, the central surface should show:

- A concise welcome prompt, such as "What would you like to understand today?"
- A large primary composer.
- Three to four quick action buttons:
  - Explain my report
  - Upload a report
  - Add symptom details
  - Understand treatment options

The screen should not default to three columns of profile, conversation, and upload content. Supporting patient information can remain available behind tabs or inline expansion.

## Layout Requirements

Desktop:

- The patient assistant surface is centered.
- The main content width is constrained to a comfortable reading and input width.
- Profile and upload panels are not both shown by default.
- The layout should no longer rely on a mandatory left rail plus center plus right rail for the default patient state.

Tablet and mobile:

- Navigation compresses without hiding core actions.
- Quick actions wrap cleanly.
- The composer remains the strongest interaction target.
- No text overflow, clipped buttons, or column squeeze is acceptable.

## Visual Boundary Requirements

The patient side must avoid the current "light border on light background" blur.

Rules:

- Do not use transparent pale borders for patient cards.
- Use explicit solid borders where borders are needed.
- Prefer these values for the patient light system:
  - Page background: `#ffffff`
  - Secondary surface: `#f8f9fa`
  - Border: `#e5e7eb`
- Cards and panels must have visibly clean edges against the page background.
- Do not rely on tinted translucent strokes to separate major surfaces.

## Shadow Requirements

The patient side must not use heavy or misty shadows for primary cards.

Rules:

- Default patient cards should use no outer shadow or a minimal hard shadow.
- Maximum allowed common shadow: `0 1px 3px rgba(0, 0, 0, 0.05)`.
- The main composer may use the lightest shadow in the system to signal focus.
- Do not use large blur radii, colored shadows, or glow-like shadows in the patient theme.

## Radius Requirements

The interface should be calm and modern without becoming visually blunt.

Rules:

- Large cards: 8px to 12px.
- Standard buttons: 8px to 10px.
- Small tags: 6px to 8px.
- The primary composer and focused quick actions may use a pill-like radius.
- Do not apply large rounded corners uniformly across all controls.

## Text And Icon Contrast

The patient side must feel clearer and more anchored.

Rules:

- Primary text should use near-black or deep ink, such as `#111827`.
- Secondary text should use `#4b5563`.
- Helper text may use `#6b7280`, but not lighter for important instructions.
- Placeholder text must be readable, not decorative.
- Icons and status dots must have enough contrast to read as deliberate marks.

## Green Token Rules

The patient theme should use fewer greens with clearer meaning.

Rules:

- One primary green is used for current selection, primary action, and send controls.
- A separate success or safe green may be used only for success and safety states.
- Informational labels, inactive tabs, and decorative surfaces should not use green.
- Non-state surfaces should return to neutral black, white, and gray.

## Composer Requirements

The composer is the primary visual anchor of the patient home.

It must support:

- Patient text input.
- Send action.
- Upload report entry point or adjacent quick action.
- Clear disabled and loading states.
- Patient-facing placeholder copy: "Describe your question, or upload a report for help understanding it."

The composer should have a clean border, readable placeholder, restrained shadow, and stronger contrast than surrounding secondary controls.

## Quick Actions

Quick actions should use patient language, not internal workflow language. They should either prefill the composer or invoke the existing card prompt path where appropriate.

Initial quick actions:

- Explain my report.
- Upload a report.
- Add symptom details.
- Understand treatment options.

## Validation Criteria

- Patient scene defaults to Ask assistant.
- Profile and upload remain reachable.
- Empty patient conversation renders a centered assistant home instead of a dense workstation.
- Patient visual boundaries are sharper: no pale translucent card borders, no misty shadows, no over-rounded major cards.
- Patient secondary text and placeholders are readable.
- Green usage is consolidated and semantic.
- Desktop, tablet, and mobile layouts do not overlap, clip, or overflow.
- Doctor cockpit remains functionally and visually unaffected.

## Risks And Mitigations

Risk: The patient side becomes too generic and loses medical trust.
Mitigation: Keep safety status, upload/report language, and patient profile access visible, but secondary to the main assistant entry.

Risk: Visual sharpening makes the patient side feel too cold.
Mitigation: Use spacing, clear copy, and a restrained primary green rather than large radius, soft shadows, or washed-out colors.

Risk: Changing the default patient tab breaks assumptions in tests.
Mitigation: Update patient navigation tests first, then adjust workspace rendering tests to assert the new default and continued access to profile/upload.
