# Routing and Review Policy Consolidation Design

**Date:** 2026-04-10  
**Status:** Approved for planning  
**Goal:** Improve behavioral stability by consolidating routing and review decisions into two single-source policy layers while preserving the current LangGraph structure and most existing node behavior.

## 1. Context

The current CRC agent has grown a strong but fragmented control surface:

- intent classification decides first-hop routing
- router functions decide follow-up routing
- planner and plan-driven routing decide execution routing
- assessment and staging helpers decide clinical shortcuts
- critic and evaluator each influence final approval or retry

This fragmentation is now the main source of instability. Two failures are especially costly:

- the same input can reach different paths in different turns or state conditions
- the same clinical case can pass review in one run and fail in another

These are control-plane problems, not just prompt-writing problems. The system currently spreads decision authority across prompt text, router functions, graph-level helper functions, and node-local fallback logic.

## 2. Design Principles

- Preserve the current graph shape as much as possible.
- Do not rewrite the node inventory or re-invent the graph runtime.
- Move final routing authority into one routing policy layer.
- Move final review authority into one review policy layer.
- Keep nodes responsible for producing signals, not final control decisions.
- Restrict policy inputs to stable, normalized facts rather than raw prompts or free-form history.
- Use shadow comparison before cutover.
- Treat degraded fallbacks as explicit review signals, not silent success.

## 3. Scope

### In scope

- Add a new `src/policies/` package
- Normalize `CRCAgentState` into stable policy-facing facts
- Consolidate routing decisions for:
  - `route_after_intent`
  - `dynamic_router`
  - `route_assessment`
- Consolidate review decisions for:
  - `route_by_critic_v2`
  - `route_after_evaluator`
- Introduce structured degraded-signal handling for critic and evaluator fallback paths
- Add shadow-mode diagnostics, policy tests, and acceptance fixtures

### Out of scope

- Rewriting the prompt taxonomy in Phase 1
- Replacing the current LangGraph topology
- Reworking `clinical_entry_resolver`, `outpatient_triage`, or staging behavior beyond adapter cleanup
- Removing all fast-path behavior in Phase 1
- Building a fully declarative policy DSL

## 4. Current-State Findings

### 4.1 Routing authority is distributed across too many places

Routing decisions are currently split across:

- `intent_nodes.py`
- `router.py`
- `graph_builder.py`
- `planner.py`
- node-local helper branches

As a result, the system can route differently depending on whether the same semantic condition is encountered:

- before planning
- during plan execution
- after assessment
- after a node-specific shortcut

### 4.2 Review authority is similarly fragmented

The decision loop currently uses:

- `decision`
- `critic`
- `citation`
- `evaluator`
- graph-level routing after critic and evaluator

These components do not only emit signals. They also perform control decisions. This makes approval behavior sensitive to:

- parser recovery
- heuristic fast review
- default pass behavior
- retry caps
- stable-RAG shortcuts

### 4.3 Some current control inputs are not stable facts

Certain existing toggles, such as `fast_pass_mode`, may reflect model- or prompt-derived behavior rather than purely normalized state facts. Such fields should not enter the single-source policy layer as raw truth inputs unless their provenance is stable and deterministic.

### 4.4 Fallback success can currently be silent

Critic and evaluator both have downgrade/fallback paths. Today, some of these paths can effectively become success-like control signals without preserving enough detail about:

- why the fallback happened
- what value was substituted
- whether the result should be treated as weaker than a normal pass

### 4.5 A few narrow routers can remain temporary exceptions

The following functions are narrow enough to remain out of policy consolidation in Phase 1:

- `route_after_clinical_entry`
- `route_after_outpatient_triage`
- `node_staging_router`

However, they must be explicitly marked as transitional exceptions so they do not evolve into a second policy system.

## 5. Proposed Architecture

### 5.1 New package layout

Phase 1 adds a policy layer without replacing existing node entrypoints:

- `src/policies/types.py`
- `src/policies/constants.py`
- `src/policies/turn_facts.py`
- `src/policies/routing_policy.py`
- `src/policies/review_policy.py`
- `src/policies/diagnostics.py`
- `src/policies/__init__.py`

### 5.2 Control-plane model

The system should separate control into four layers:

1. `state`  
   Raw execution state owned by the graph and nodes.

2. `facts`  
   Stable normalized facts derived from state.

3. `signals`  
   Node-produced outputs that represent review or execution observations.

4. `policy decisions`  
   Final route or review actions derived from facts and signals.

### 5.3 Adapter model

Existing router entrypoints remain in place as adapters:

- they can still read `CRCAgentState`
- they build facts and policy inputs
- they call the policy layer
- they return `target` strings expected by the graph

This preserves the current graph interfaces while centralizing the decision rules.

## 6. Policy Data Model

### 6.1 `TurnFacts`

`TurnFacts` should contain only stable, normalized, policy-relevant facts. It must not contain:

- prompt text
- full free-form history
- model reasoning text
- unstable, prompt-derived shortcuts treated as raw truth

Recommended contents:

- intent facts
  - `user_intent`
  - `sub_tasks`
  - `multi_task_mode`
- orchestration facts
  - `has_plan`
  - `pending_step_tool`
  - `pending_step_target`
  - `has_parallel_group`
  - `active_inquiry`
  - `active_field`
  - `pending_patient_data`
  - `pending_patient_id`
- clinical gate facts
  - `encounter_track`
  - `is_outpatient_triage`
  - `has_missing_critical_data`
  - `missing_critical_data_count`
  - `pathology_confirmed`
  - `tumor_location`
  - `patient_profile_locked`
  - `needs_full_decision`
- decision loop facts
  - `decision_exists`
  - `decision_strategy`
  - `iteration_count`
  - `rejection_count`
  - `evaluation_retry_count`
- review gate facts
  - `critic_verdict`
  - `citation_coverage_score`
  - `citation_needs_more_sources`
  - `stable_guideline_rag_support`
  - `evaluator_verdict`
  - `evaluator_scores`
  - `evaluator_actionable_retry`
  - `evaluator_degraded`

`TurnFacts` should retain the normalized evaluator outcome fields needed for cross-turn consistency checks, while `ReviewSignal` retains per-turn signal detail and degraded metadata used by the review policy.

### 6.2 `DerivedRoutingFlags`

These are computed from stable facts rather than read directly from state as legacy toggles:

- `should_shortcut_to_general_chat`
- `should_shortcut_to_knowledge`
- `should_force_clinical_entry`
- `should_end_turn_for_inquiry`
- `can_fast_pass_decision`

This is where current transitional behavior such as fast-pass rules should move if they can be deterministically derived from facts.

### 6.3 `DegradedSignal`

Fallback states must be explicit and structured:

```python
@dataclass(frozen=True)
class DegradedSignal:
    is_degraded: bool
    reason: str | None
    fallback_value: str | None
```

Recommended reasons include:

- `parse_error`
- `timeout`
- `llm_exception`
- `heuristic_fallback`
- `score_below_threshold`

### 6.4 `ReviewSignal`

Critic and evaluator should each produce a structured review signal that is separate from final graph routing:

- `source`
- `verdict`
- `retryable`
- `reasons`
- `degraded`

### 6.5 Final decisions

Routing and review each produce one final decision object:

- `RouteDecision(target, rule_name, rationale)`
- `ReviewDecision(action, rule_name, rationale)`

## 7. Routing Policy Design

### 7.1 Phase 1 routing entrypoints

The routing policy should expose:

- `decide_after_intent(facts, flags)`
- `decide_dynamic(facts, flags)`
- `decide_after_assessment(facts, flags)`

### 7.2 Behavioral objective

The routing policy must make path selection stable for the same normalized facts, regardless of:

- where the request is in the graph
- which router helper called the decision
- whether legacy state fields are present in different shapes

### 7.3 Initial rule consolidation targets

Phase 1 should consolidate decisions currently made by:

- `route_after_intent`
- `dynamic_router`
- `route_assessment`

### 7.4 Transitional exceptions

The following remain outside the main routing policy in Phase 1:

- `route_after_clinical_entry`
- `route_after_outpatient_triage`
- `node_staging_router`

These functions must include a clear Phase 1 comment stating that they are narrow transitional exceptions and do not participate in the consolidated policy layer.

## 8. Review Policy Design

### 8.1 Phase 1 review entrypoints

The review policy should expose:

- `decide_after_critic(facts, critic_signal)`
- `decide_after_evaluator(facts, critic_signal, evaluator_signal)`

### 8.2 Behavioral objective

The review policy must ensure that final approval behavior is stable for the same facts and review signals.

It must distinguish:

- normal pass
- degraded pass
- retryable failure
- non-retryable failure
- retry-cap finalization

### 8.3 Initial rule consolidation targets

Phase 1 should consolidate decisions currently made by:

- `route_by_critic_v2`
- `route_after_evaluator`

### 8.4 Node responsibility after consolidation

After consolidation:

- `node_critic` emits `ReviewSignal`
- `node_llm_judge` emits `ReviewSignal`
- only `review_policy` decides `retry_decision` vs `finalize`

## 9. Shadow Mode and Diagnostics

### 9.1 Migration strategy

Phase 1 should use shadow comparison before cutover.

Each adapter keeps the legacy behavior while also computing the policy decision and recording any divergence.

### 9.2 Minimum diagnostics payload

The system should record, at minimum:

- `legacy_route`
- `policy_route`
- `route_diverged`
- `legacy_review_action`
- `policy_review_action`
- `review_diverged`
- `divergence_reason`
- `policy_rule_name`

### 9.3 Purpose of shadow mode

Shadow mode is required to answer:

- which rules still diverge most often
- whether divergence comes from fact normalization or policy logic
- whether the new policy is safe to cut over

## 10. Phase 1 Implementation Sequence

1. Create the `src/policies/` package and type definitions.
2. Implement `build_turn_facts(state)` and `derive_routing_flags(facts)`.
3. Add fact-layer unit tests before any cutover.
4. Implement `routing_policy` for:
   - `after_intent`
   - `dynamic`
   - `after_assessment`
5. Add routing golden fixtures and pure policy tests.
6. Add shadow-mode comparison inside existing routing adapters without changing graph behavior.
7. Cut over `route_after_intent` to `routing_policy`.
8. Keep shadow comparison enabled.
9. Cut over `dynamic_router` to `routing_policy` while preserving current plan-priority semantics.
10. Move `route_assessment` out of `graph_builder.py` into an adapter layer, then cut it over to `routing_policy`.
11. Implement structured `ReviewSignal` and `DegradedSignal` production in critic and evaluator.
12. Implement `review_policy`.
13. Add shadow-mode comparison for review actions.
14. Cut over `route_by_critic_v2` to `review_policy`.
15. Cut over `route_after_evaluator` to `review_policy`.
16. Mark the three transitional routers with explicit Phase 1 comments and boundary notes.
17. Update internal docs so future work understands that nodes emit signals and policies own final control decisions.

## 11. Testing Strategy

### 11.1 Fact determinism tests

`build_turn_facts(state)` must be deterministic for the same input fixture across repeated runs.

Coverage should include:

- missing fields
- empty fields
- legacy shape compatibility
- field-priority normalization
- degraded-reason normalization

### 11.2 Routing golden-case tests

Policy tests should lock expected route targets and rule names for representative cases such as:

- general chat
- knowledge query
- clinical assessment
- treatment decision
- case database query
- active inquiry
- missing critical data
- pending plan step
- parallel subagents

### 11.3 Review golden-case tests

Policy tests should lock outcomes for:

- normal pass
- critic rejection with retry
- evaluator fail with actionable retry
- stable guideline RAG support leading to finalize
- critic degraded due to parse error
- evaluator degraded due to timeout
- retry-cap finalization

### 11.4 Shadow divergence tests

Adapter tests should verify that shadow diagnostics are written consistently when legacy and policy decisions diverge.

## 12. Acceptance Criteria

- The same `CRCAgentState` fixture always produces the same `TurnFacts`.
- Routing golden fixtures produce stable `RouteDecision.target` and `RouteDecision.rule_name`.
- `route_after_intent` cutover preserves intended first-hop behavior for approved fixtures.
- `dynamic_router` cutover preserves current plan-first semantics while reducing unexplained branch variation.
- `route_assessment` cutover stabilizes missing-data and clinical shortcut handling.
- Critic and evaluator fallback paths always emit explicit degraded signals.
- The system can distinguish normal pass from degraded pass in review decisions.
- Review golden fixtures produce stable retry vs finalize outcomes for the same case.
- Shadow diagnostics expose old decision, new decision, rule name, and divergence reason.
- After Phase 1, only two consolidated control sources remain for the main flow:
  - `routing_policy`
  - `review_policy`

## 13. Risks and Mitigations

### 13.1 Risk: policy layer becomes a new glue mess

Mitigation:

- policy functions may only consume normalized facts and signals
- adapters may read raw state
- policies may not read raw `CRCAgentState`

### 13.2 Risk: fast-pass behavior is silently preserved as unstable legacy logic

Mitigation:

- do not carry `fast_pass_mode` into `TurnFacts` as a raw truth field unless provenance is stable
- prefer deriving fast-pass eligibility from normalized facts

### 13.3 Risk: fallback success still acts like silent approval

Mitigation:

- require structured `DegradedSignal`
- make review policy treat degraded pass differently from normal pass

### 13.4 Risk: transitional routers grow into a second control plane

Mitigation:

- annotate them as Phase 1 exceptions
- restrict their responsibilities
- keep them out of future policy growth

## 14. Post-Phase-1 Outlook

Phase 2 should focus on prompt stability and prompt assembly:

- standardizing prompt inputs against the new fact layer
- reducing context injection drift
- separating policy text from task prompts
- improving prompt-level regression coverage

Phase 1 should not attempt to solve prompt architecture and control-plane consolidation in one step.
