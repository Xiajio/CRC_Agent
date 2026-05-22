# Patient Profile To Patient Card Auto-generation Design

**Date:** 2026-04-20  
**Status:** In review  
**Goal:** Automatically generate and incrementally refresh a patient-side `patient_card` from `patient_profile`, `findings`, triage snapshots, and upload-derived medical data, while preserving explicit conflict visibility instead of silently overwriting inconsistent facts.

## 1. Context

The current workspace already has:

- a patient-side right inspector panel
- a reusable `patient_card` frontend renderer
- state-level `patient_profile` updates during assessment
- triage state and symptom snapshots during patient questioning
- upload-derived `medical_card` and lightweight patient snapshot extraction

What is missing is the linking layer between these pieces. The system can collect patient facts over time, but the patient scene does not currently receive an automatically maintained patient background card that fills itself in as information accumulates.

Today, `patient_card` is mainly produced by database and historical-case flows. Patient-side dialogue updates `patient_profile`, but that state is not projected into a visible patient card. This leaves the patient inspector underused and prevents the UI from reflecting growing certainty, missing information, or source conflicts.

## 2. Problem

The patient scene needs a continuously updating patient summary card, but the available data arrives from multiple places and at different levels of certainty:

- structured `patient_profile` updates from assessment
- partially structured triage state
- upload-derived `medical_card` fields
- free-text-derived `findings`

If the system simply picks the newest value, it hides uncertainty and source disagreement. If it only uses `patient_profile`, too many fields remain empty. If each node hand-builds a card, the implementation will drift and become inconsistent.

The product requirement is therefore:

- show a full patient card skeleton in the patient scene
- incrementally fill fields as the conversation progresses
- keep unresolved fields visible as `待确认`
- keep inconsistent fields visible as `待确认（来源不一致）`
- avoid building a second frontend card framework

## 3. Goals

### 3.1 In scope

- auto-generate a patient-side `patient_card`
- refresh the card incrementally as patient facts change
- aggregate from `patient_profile`, `findings`, triage snapshots, and upload-derived medical data
- preserve conflicts instead of overwriting them
- keep the existing `patient_card` card type and right-panel rendering path
- distinguish patient self-report cards from database cards
- restore the generated card after refresh via session snapshot recovery

### 3.2 Out of scope

- redesigning the whole patient right panel
- replacing `patient_profile` with a broader state model
- turning conflict resolution into an interactive confirmation workflow
- adding doctor-only controls to patient self-report cards
- rebuilding the frontend card system around per-field widgets

## 4. Confirmed Product Decisions

The following decisions are already resolved:

- The patient card should update incrementally as information is collected.
- The card should show a full skeleton rather than hiding empty sections.
- Unfilled fields should display `待确认`.
- Conflicting sources should display `待确认（来源不一致）`.
- The card may aggregate from `patient_profile`, `findings`, triage snapshots, and upload parsing results.
- Conflicts must be preserved instead of resolved through silent precedence rules.
- The existing `patient_card` renderer and right inspector should be reused instead of introducing a second card type.

## 5. Current-state Findings

### 5.1 `patient_card` already exists, but mainly for database flows

The frontend already supports `patient_card` rendering and labels it as `患者画像`. The patient scene right inspector can display it today.

However, the main production path for `patient_card` comes from database tooling and formatter-based historical case retrieval, not from live patient dialogue.

### 5.2 `patient_profile` is already updated during assessment

Assessment nodes already create and update `patient_profile`, including fields such as:

- `tumor_type`
- `pathology_confirmed`
- `tnm_staging`
- `mmr_status`
- `chief_complaint`
- `age`
- `gender`
- `ecog_score`

This provides a strong structured foundation, but it is not enough on its own to fill the entire patient card.

### 5.3 Triage and upload flows provide additional useful facts

The triage flow can populate symptom-oriented facts such as chief symptoms and symptom duration. Upload parsing can populate a medical summary, diagnosis block, and staging block. Those sources are useful for patient card completeness, but they are currently not projected into a single patient-side card.

### 5.4 The right inspector path does not need architectural changes

The patient scene already routes visible cards into the right panel. The missing work is data projection and selective renderer enhancement, not inspector architecture.

## 6. Proposed Architecture

### 6.1 Introduce a dedicated patient card projector

Add a single backend projector responsible for constructing patient-side `patient_card` payloads from multiple sources.

Recommended location:

- `src/services/patient_card_projector.py`

Responsibilities:

- normalize raw values across sources
- aggregate candidates per logical field
- determine each field's status
- build a consistent `patient_card` payload

This projector should be the only place where:

- field-level source merging happens
- field conflict rules live
- `pending / confirmed / conflict` status is computed

Nodes should not handcraft patient cards independently.

### 6.2 Treat the card as a projection, not a primary state model

`patient_card` should be derived from canonical state rather than becoming its own parallel truth source.

That means:

- `patient_profile` remains the structured profile source
- `findings` remain extracted dialogue facts
- triage snapshots remain symptom-tracking state
- uploads remain medical-card-derived evidence
- `patient_card` is a synthesized presentation artifact

This avoids dual-write drift between facts and UI output.

## 7. Field Model And Merge Rules

### 7.1 Reuse the existing `patient_card` skeleton

The generated card should continue to use the existing `patient_card.data` shape:

- `patient_info`
- `diagnosis_block`
- `staging_block`
- `history_block`

This keeps the existing frontend renderer reusable.

### 7.2 Add field-level status metadata

Extend the payload with metadata describing per-field certainty:

- `confirmed`
- `pending`
- `conflict`

Recommended auxiliary sections:

- `field_meta`
- `source_candidates`
- `card_meta`

Payload contract:

- `data` always carries canonical raw field values or `null`
- `field_meta` carries field status and display text
- `source_candidates` carries provenance and conflicting alternatives
- placeholder text such as `待确认` never becomes the canonical `data` value
- if a field status is `conflict`, the canonical `data` value for that field must be serialized as `null`

### 7.3 Field status rules

For each field:

1. No usable value from any source -> `pending`
2. One usable value, or multiple sources normalize to the same value -> `confirmed`
3. Multiple usable values normalize differently -> `conflict`

Conflict does not block card refresh. The card should still update incrementally while marking only the inconsistent fields as unresolved.

Conflict serialization rule:

- every `conflict` field serializes `data.<field> = null`
- the conflicting candidates live only in `source_candidates`
- the user-facing string lives only in `field_meta.display`
- projector, stream recovery, and any later recomputation must follow the same rule so the payload stays deterministic

### 7.4 Field-specific normalization and equality rules

The projector must not use a single generic equality rule for every field type. It should normalize and compare by field category:

- boolean fields such as `diagnosis_block.confirmed` and `history_block.biopsy_confirmed`
  - normalize to `true`, `false`, or `null`
- scalar enum/text fields such as `gender`, `primary_site`, `mmr_status`, and stage fragments
  - normalize by trimming, collapsing aliases, and mapping to one canonical string
- numeric fields such as `age`, `ecog`, and future lab values
  - normalize to numeric values when extraction is reliable, otherwise treat as text candidates
- free-text detail fields such as `chief_complaint`, `family_history_details`, and `biopsy_details`
  - normalize lightly by trimming whitespace and punctuation noise only
- list/set fields such as `risk_factors`
  - normalize to deduplicated sorted sets before comparison

Snapshot-specific rule:

- triage-derived fields use the latest effective triage snapshot only
- historical triage snapshots are not merged together for equality comparison
- a corrected later snapshot replaces earlier snapshot-derived values for the same field

### 7.5 No silent override rule

There is no hidden latest wins or upload always wins display rule. A field with conflicting evidence must visibly surface the inconsistency.

Candidate values remain available in raw payload metadata for debugging, disclosure, and future UX improvements.

## 8. Field Mapping

### 8.1 `patient_info`

- `gender`
  - `patient_profile.gender`
  - `medical_card.data.patient_summary.gender`
  - normalization target: `男`, `女`, or `未知`
- `age`
  - `patient_profile.age`
  - `medical_card.data.patient_summary.age`
  - normalization target: integer age when parseable
- `ecog`
  - `patient_profile.ecog_score`
- `cea`
  - future structured `findings` CEA extraction when available

### 8.2 `diagnosis_block`

- `confirmed`
  - `patient_profile.pathology_confirmed`
  - `findings.pathology_confirmed`
  - `medical_card.data.diagnosis_block.confirmed`
  - `medical_card.data.diagnosis_block.pathology`
- `primary_site`
  - `patient_profile.tumor_type`
  - `findings.tumor_location`
  - `medical_card.data.diagnosis_block.location`
- `mmr_status`
  - `patient_profile.mmr_status`
  - `findings.mmr_status`
  - `medical_card.data.diagnosis_block.mmr_status`

### 8.3 `staging_block`

- `clinical_stage`
  - `patient_profile.tnm_staging.stage_group`
  - `findings.clinical_stage`
  - `medical_card.data.staging_block.clinical_stage`
- `ct_stage`
  - `patient_profile.tnm_staging.cT`
  - `findings.tnm_staging.cT`
  - `medical_card.data.staging_block.t_stage`
- `cn_stage`
  - `patient_profile.tnm_staging.cN`
  - `findings.tnm_staging.cN`
  - `medical_card.data.staging_block.n_stage`
- `cm_stage`
  - `patient_profile.tnm_staging.cM`
  - `findings.tnm_staging.cM`
  - `medical_card.data.staging_block.m_stage`

### 8.4 `history_block`

- `chief_complaint`
  - `patient_profile.chief_complaint`
  - latest triage `symptom_snapshot.chief_symptoms`
  - `medical_card.data.patient_summary.chief_complaint`
- `symptom_duration`
  - latest triage `symptom_snapshot.duration`
  - `findings.symptom_duration`
- `family_history`
  - structured `findings.family_history`
- `family_history_details`
  - structured `findings.family_history_details`
- `biopsy_confirmed`
  - `patient_profile.pathology_confirmed`
  - `findings.pathology_confirmed`
  - upload diagnosis/pathology summary
- `biopsy_details`
  - structured `findings.biopsy_details`
  - upload diagnosis/pathology text when available
- `risk_factors`
  - structured `findings.risk_factors`

Not every field will be available in phase 1. That is acceptable as long as the card shows the full skeleton and unresolved fields render as `待确认`.

## 9. Card Payload Shape

### 9.1 Proposed payload

```json
{
  "type": "patient_card",
  "title": "患者画像",
  "patient_id": "current",
  "data": {
    "patient_info": {
      "gender": "女",
      "age": 52,
      "ecog": null,
      "cea": null
    },
    "diagnosis_block": {
      "confirmed": null,
      "primary_site": "直肠",
      "mmr_status": "dMMR"
    },
    "staging_block": {
      "clinical_stage": "III期",
      "ct_stage": "cT3",
      "cn_stage": "cN1",
      "cm_stage": "cM0"
    },
    "history_block": {
      "chief_complaint": "便血",
      "symptom_duration": "3天",
      "family_history": null,
      "family_history_details": null,
      "biopsy_confirmed": null,
      "biopsy_details": null,
      "risk_factors": []
    }
  },
  "field_meta": {
    "diagnosis_block": {
      "confirmed": {
        "status": "conflict",
        "display": "待确认（来源不一致）"
      }
    },
    "history_block": {
      "family_history": {
        "status": "pending",
        "display": "待确认"
      }
    }
  },
  "source_candidates": {
    "diagnosis_block.confirmed": [
      {
        "value": true,
        "display_value": "已病理确认",
        "source_type": "upload",
        "source_path": "medical_card.data.diagnosis_block.pathology"
      },
      {
        "value": false,
        "display_value": "未确认",
        "source_type": "dialogue",
        "source_path": "findings.pathology_confirmed"
      }
    ]
  },
  "card_meta": {
    "source_mode": "patient_self_report",
    "completion_ratio": 0.58,
    "conflict_count": 1,
    "projection_version": 1
  }
}
```

### 9.2 Semantics

- `data`
  - canonical raw values using the existing patient card shape
  - never stores placeholder display strings such as `待确认`
  - uses `null` for every conflicted field, regardless of original field type
- `field_meta`
  - field-level status and final user-facing display text
- `source_candidates`
  - optional raw provenance for fields needing auditability
- `card_meta.source_mode`
  - allows frontend behavior changes for patient self-report cards
- `card_meta.completion_ratio`
  - optional summary metric for future UI
- `card_meta.conflict_count`
  - quick indicator of how many unresolved fields remain

### 9.3 Stable identity and replacement semantics

This feature reuses the existing `patient_card` type, so replacement semantics must be explicit.

- the self-report card identity is `type == "patient_card"` plus `card_meta.source_mode == "patient_self_report"`
- within one live session state, later emissions of the self-report `patient_card` replace earlier self-report `patient_card` payloads
- the projector must emit the full current self-report card, not partial patches
- emitters do not merge card payload fragments on the frontend; each emission is a full replacement snapshot
- database-origin `patient_card` payloads remain separate in meaning and must not be rewritten by the self-report projector

The implementation should preserve current adapter/store behavior by ensuring one effective patient self-report card survives after pruning and recovery.

## 10. Emission Points

The same projector should be invoked from every place where patient facts materially change.

### 10.1 Assessment nodes

When assessment updates `patient_profile`, rebuild the patient self-report `patient_card` in the node output so the UI updates immediately.

### 10.2 Triage nodes

When outpatient triage updates the effective symptom snapshot or symptom-related findings, rebuild the patient self-report `patient_card` so the right panel fills in during questioning.

### 10.3 Upload service

When uploads produce a `medical_card`, rebuild the patient self-report `patient_card` using upload-derived diagnosis and staging data.

### 10.4 Snapshot recovery fallback

During session snapshot construction, always recompute the patient self-report projection from the recovered canonical state. Do not rely on a previously cached self-report `patient_card` being fresh enough.

Recovery rule:

- recompute patient self-report `patient_card` from `patient_profile`, recovered `findings`, latest effective triage snapshot, and recovered upload-derived medical data
- do not recompute legacy database-origin `patient_card` payloads through this projector

This guarantees correct refresh behavior and prevents stale projected cards from surviving across reloads.

## 11. Frontend Changes

### 11.1 Reuse the existing `patient_card` renderer

Do not introduce a new card type. Extend `renderPatientCard()` to understand:

- `field_meta`
- `card_meta.source_mode`

### 11.2 Gate full skeleton behavior to patient self-report cards

The new skeleton and placeholder behavior applies only when:

- `card_meta.source_mode == "patient_self_report"`

Display rules for patient self-report cards:

- `confirmed` -> display the raw value
- `pending` -> display `待确认`
- `conflict` -> display `待确认（来源不一致）`

Legacy database cards keep current rendering behavior when:

- `field_meta` is absent
- or `card_meta.source_mode != "patient_self_report"`

This prevents regressions in existing richer database cards.

### 11.3 Hide doctor-oriented quick actions for patient self-report cards

The current `patient_card` renderer includes doctor-style quick actions such as treatment plan generation. These should be hidden when:

- `card_meta.source_mode == "patient_self_report"`

Database-origin patient cards can keep their current quick-action behavior.

### 11.4 Preserve raw disclosure instead of adding conflict UI complexity

Conflict source details should remain in the raw payload disclosure for phase 1. The card body itself should stay simple and focused on status visibility.

## 12. Adapter And Transport Implications

No new transport is needed.

- the card remains a normal `patient_card`
- existing card extractor and stream/event normalization paths continue to work
- session snapshot recovery continues to deliver cards through the standard card collection mechanism

Compatibility contract:

- `field_meta`, `source_candidates`, and `card_meta` are additive JSON members inside the existing card payload
- they do not require new event types, new top-level transport fields, or a custom streaming path
- existing adapters already preserve arbitrary mapping payloads inside cards, so the implementation only needs to ensure these keys are not stripped during formatting or normalization

This feature should fit into the current adapter system rather than introducing custom streaming logic.

## 13. Testing Strategy

### 13.1 Projector unit tests

Add direct tests for:

- no source value -> `pending`
- single source value -> `confirmed`
- equal normalized values from multiple sources -> `confirmed`
- conflicting normalized values -> `conflict`
- normalization of age, stage, MMR status, and boolean pathology confirmation
- list normalization for `risk_factors`
- latest triage snapshot replacing older snapshot-derived values

### 13.2 Node integration tests

Cover:

- assessment updates causing immediate patient card generation
- triage updates incrementally filling chief complaint and symptom duration
- upload-derived medical card data flowing into the patient card
- corrected later state changing a field from `conflict` back to `confirmed`
- source withdrawal or correction changing a field from `confirmed` back to `pending`

### 13.3 Adapter tests

Verify:

- patient self-report `patient_card` is extracted and emitted correctly
- replacement semantics keep the latest self-report card active
- out-of-order multi-emitter updates do not leave duplicate active self-report cards
- session snapshot rebuild restores the recomputed projected patient card
- legacy database `patient_card` payloads are not overwritten by self-report recovery

### 13.4 Frontend tests

Cover:

- full skeleton rendering for patient self-report cards
- `待确认` display for pending fields
- `待确认（来源不一致）` display for conflict fields
- hidden quick actions when `source_mode == "patient_self_report"`
- legacy database `patient_card` rendering remains unchanged when self-report metadata is absent

Testing files should be added in tracked locations, not in the ignored root `tests/` tree.

## 14. Risks

- Over-aggregation may create noisy cards if field mapping is too loose.
- Upload parsing and dialogue extraction may disagree frequently, increasing visible conflict count.
- If multiple nodes project cards differently, drift will reappear.
- The existing `patient_card` renderer was originally shaped for richer database cards, so phase 1 must keep the UI changes minimal.

## 15. Recommendation

Implement this as a derived projection with one shared projector module and a minimally enhanced frontend renderer.

That approach:

- satisfies the incremental patient-side UX requirement
- preserves explicit uncertainty and conflict
- avoids dual-write state drift
- keeps the implementation aligned with the existing card system
- allows future improvement without changing the transport or inspector architecture
