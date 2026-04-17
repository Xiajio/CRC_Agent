# Routing and Review Policy Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate routing and review control decisions into two stable policy layers so the same CRC input follows the same path and review outcome under the same normalized facts.

**Architecture:** Keep the current LangGraph topology and node inventory, but insert a policy layer between raw `CRCAgentState` and final routing/review decisions. Existing router functions and review routers remain as adapters that normalize state into `TurnFacts`, call `routing_policy` or `review_policy`, and return the graph-compatible target string. Nodes continue to emit findings and review signals; policy owns final control decisions.

**Tech Stack:** Python, LangGraph, Pydantic, FastAPI, pytest

---

**Repository note:** This workspace is not currently a git repository, so commit steps are written as checkpoints but cannot be executed until repository metadata is restored.

## File Structure

### Backend files to create

- `D:\亿铸智能体\LangG_New\src\policies\__init__.py`
  - Public export surface for policy-layer types and helpers.
- `D:\亿铸智能体\LangG_New\src\policies\types.py`
  - Define `TurnFacts`, `DerivedRoutingFlags`, `DegradedSignal`, `RouteDecision`, `ReviewSignal`, and `ReviewDecision`.
- `D:\亿铸智能体\LangG_New\src\policies\constants.py`
  - Hold thresholds and retry-cap constants that should stop living inside node-local logic.
- `D:\亿铸智能体\LangG_New\src\policies\turn_facts.py`
  - Normalize `CRCAgentState` into policy-safe facts and derived flags.
- `D:\亿铸智能体\LangG_New\src\policies\routing_policy.py`
  - Single source of truth for `after_intent`, `dynamic`, and `after_assessment` routing decisions.
- `D:\亿铸智能体\LangG_New\src\policies\review_policy.py`
  - Single source of truth for `after_critic` and `after_evaluator` review actions.
- `D:\亿铸智能体\LangG_New\src\policies\diagnostics.py`
  - Shadow-mode divergence recording helpers for legacy-vs-policy comparison.

### Backend files to modify

- `D:\亿铸智能体\LangG_New\src\state.py`
  - Keep state fields aligned with the facts/signals the policy layer must read.
- `D:\亿铸智能体\LangG_New\src\nodes\router.py`
  - Convert `route_after_intent` and `dynamic_router` into policy adapters and mark remaining transitional routers.
- `D:\亿铸智能体\LangG_New\src\graph_builder.py`
  - Remove inline `route_assessment` logic from the graph definition and point to an adapter-backed function.
- `D:\亿铸智能体\LangG_New\src\nodes\decision_nodes.py`
  - Emit `ReviewSignal`-compatible critic output, add degraded metadata, and adapt `route_by_critic_v2`.
- `D:\亿铸智能体\LangG_New\src\nodes\evaluation_nodes.py`
  - Emit `ReviewSignal`-compatible evaluator output, add degraded metadata, and adapt `route_after_evaluator`.
- `D:\亿铸智能体\LangG_New\src\nodes\citation_nodes.py`
  - Keep citation helpers aligned with the new facts consumed by review policy if helper extraction is reused.

### Backend tests to create

- `D:\亿铸智能体\LangG_New\tests\backend\policies\test_turn_facts.py`
  - Determinism and normalization coverage for `TurnFacts` and `DerivedRoutingFlags`.
- `D:\亿铸智能体\LangG_New\tests\backend\policies\test_routing_policy.py`
  - Golden-case route selection and rule-name coverage.
- `D:\亿铸智能体\LangG_New\tests\backend\policies\test_review_policy.py`
  - Golden-case retry/finalize behavior and degraded-pass handling.
- `D:\亿铸智能体\LangG_New\tests\backend\policies\test_policy_diagnostics.py`
  - Shadow divergence recording coverage.

### Backend tests to modify

- `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
  - Preserve legacy routing behavior where intended and add adapter-level shadow checks.
- `D:\亿铸智能体\LangG_New\tests\backend\test_assessment_node.py`
  - Keep `assessment -> inquiry / decision / diagnosis` behavior locked while the routing adapter changes.
- `D:\亿铸智能体\LangG_New\tests\backend\test_state_model_source.py`
  - Verify no duplicate state fields are introduced during policy rollout.
- `D:\亿铸智能体\LangG_New\tests\backend\test_state_snapshot.py`
  - Extend snapshot coverage if policy diagnostics or review fields need to survive state capture.

## Task 1: Create Policy Types and Constants

**Files:**
- Create: `D:\亿铸智能体\LangG_New\src\policies\types.py`
- Create: `D:\亿铸智能体\LangG_New\src\policies\constants.py`
- Create: `D:\亿铸智能体\LangG_New\src\policies\__init__.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_turn_facts.py`

- [ ] **Step 1: Write the failing type import test**

```python
from src.policies.types import (
    TurnFacts,
    DerivedRoutingFlags,
    DegradedSignal,
    RouteDecision,
    ReviewSignal,
    ReviewDecision,
)


def test_policy_types_are_importable():
    assert TurnFacts is not None
    assert ReviewDecision is not None
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `pytest tests/backend/policies/test_turn_facts.py -k importable -v`  
Expected: FAIL with import error because the policy package does not exist yet.

- [ ] **Step 3: Create the policy types and constants**

Define frozen dataclasses or Pydantic models for:

```python
TurnFacts
DerivedRoutingFlags
DegradedSignal
RouteDecision
ReviewSignal
ReviewDecision
```

Create explicit constants for:

```python
MAX_DECISION_RETRIES = 3
MAX_EVALUATION_RETRIES = 2
STABLE_GUIDELINE_MIN_COVERAGE = 80
STABLE_GUIDELINE_MIN_INLINE_ANCHORS = 2
```

Use names that match existing semantics before changing thresholds.

- [ ] **Step 4: Re-run the import test**

Run: `pytest tests/backend/policies/test_turn_facts.py -k importable -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/policies/__init__.py src/policies/types.py src/policies/constants.py tests/backend/policies/test_turn_facts.py
git commit -m "feat: add policy-layer types and constants"
```

## Task 2: Implement Deterministic Turn-Fact Normalization

**Files:**
- Create: `D:\亿铸智能体\LangG_New\src\policies\turn_facts.py`
- Modify: `D:\亿铸智能体\LangG_New\src\state.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_state_model_source.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_turn_facts.py`

- [ ] **Step 1: Write failing determinism tests**

```python
def test_build_turn_facts_is_deterministic_for_same_state():
    state = CRCAgentState(
        findings={"user_intent": "clinical_assessment", "active_inquiry": True},
        missing_critical_data=["TNM Staging"],
        critic_verdict="REJECTED",
        evaluation_report={"verdict": "FAIL", "safety": 2, "completeness": 2},
    )

    first = build_turn_facts(state)
    second = build_turn_facts(state)

    assert first == second


def test_derive_routing_flags_does_not_depend_on_fast_pass_mode_raw_field():
    state = CRCAgentState(
        findings={"user_intent": "clinical_assessment", "fast_pass_mode": True},
        patient_profile={"is_locked": True, "tnm_staging": {"cT": "T3", "cN": "N1", "cM": "M0"}},
    )

    facts = build_turn_facts(state)

    assert hasattr(facts, "decision_strategy")
    assert "fast_pass_mode" not in facts.__dict__
```

- [ ] **Step 2: Run the determinism tests and confirm failure**

Run: `pytest tests/backend/policies/test_turn_facts.py -k "deterministic or fast_pass" -v`  
Expected: FAIL because the normalization helpers do not exist yet.

- [ ] **Step 3: Implement `build_turn_facts` and `derive_routing_flags`**

Implement explicit normalization helpers for:

```python
user_intent
sub_tasks
multi_task_mode
has_plan
pending_step_tool
pending_step_target
has_parallel_group
active_inquiry
active_field
pending_patient_data
pending_patient_id
encounter_track
has_missing_critical_data
missing_critical_data_count
pathology_confirmed
tumor_location
patient_profile_locked
needs_full_decision
decision_exists
decision_strategy
iteration_count
rejection_count
evaluation_retry_count
critic_verdict
citation_coverage_score
citation_needs_more_sources
stable_guideline_rag_support
evaluator_verdict
evaluator_scores
evaluator_actionable_retry
evaluator_degraded
```

Do not read prompt text or free-form history. Keep `fast_pass_mode` out of `TurnFacts`; derive `can_fast_pass_decision` from stable facts only.

- [ ] **Step 4: Add duplicate-field coverage if state fields are introduced**

Extend `tests/backend/test_state_model_source.py` with assertions like:

```python
assert _count_class_field_declarations(class_node, "evaluation_report") == 1
assert _count_class_field_declarations(class_node, "citation_report") == 1
```

Only add assertions for fields you touch in `src/state.py`.

- [ ] **Step 5: Re-run the fact tests**

Run: `pytest tests/backend/policies/test_turn_facts.py tests/backend/test_state_model_source.py -v`  
Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add src/policies/turn_facts.py src/state.py tests/backend/policies/test_turn_facts.py tests/backend/test_state_model_source.py
git commit -m "feat: add deterministic turn-fact normalization"
```

## Task 3: Implement Pure Routing Policy Decisions

**Files:**
- Create: `D:\亿铸智能体\LangG_New\src\policies\routing_policy.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_routing_policy.py`

- [ ] **Step 1: Write failing golden-case routing tests**

```python
def test_decide_after_intent_routes_general_chat_directly():
    facts = TurnFacts(user_intent="general_chat", has_plan=False, ...)
    flags = DerivedRoutingFlags(should_shortcut_to_general_chat=True, ...)

    decision = decide_after_intent(facts, flags)

    assert decision.target == "general_chat"
    assert decision.rule_name == "intent_general_chat"


def test_decide_dynamic_prioritizes_parallel_plan_group():
    facts = TurnFacts(user_intent="clinical_assessment", has_plan=True, has_parallel_group=True, ...)
    flags = DerivedRoutingFlags(...)

    decision = decide_dynamic(facts, flags)

    assert decision.target == "parallel_subagents"
    assert decision.rule_name == "plan_parallel_group"


def test_decide_after_assessment_routes_missing_data_to_chat_main():
    facts = TurnFacts(has_missing_critical_data=True, active_inquiry=False, ...)
    flags = DerivedRoutingFlags(should_end_turn_for_inquiry=False, ...)

    decision = decide_after_assessment(facts, flags)

    assert decision.target == "chat_main"
```

- [ ] **Step 2: Run the routing-policy tests and confirm failure**

Run: `pytest tests/backend/policies/test_routing_policy.py -v`  
Expected: FAIL because `routing_policy.py` does not exist yet.

- [ ] **Step 3: Implement the pure routing policy**

Add:

```python
def decide_after_intent(facts: TurnFacts, flags: DerivedRoutingFlags) -> RouteDecision: ...
def decide_dynamic(facts: TurnFacts, flags: DerivedRoutingFlags) -> RouteDecision: ...
def decide_after_assessment(facts: TurnFacts, flags: DerivedRoutingFlags) -> RouteDecision: ...
```

Preserve current semantics:

- general chat and off-topic still short-circuit
- knowledge queries still direct-route when not multi-task
- pending plan steps still beat fallback routing
- parallel groups still beat single plan steps
- missing critical data still interrupts clinical progression

Do not cut over adapters yet.

- [ ] **Step 4: Re-run the routing-policy tests**

Run: `pytest tests/backend/policies/test_routing_policy.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/policies/routing_policy.py tests/backend/policies/test_routing_policy.py
git commit -m "feat: add pure routing policy decisions"
```

## Task 4: Add Shadow Diagnostics for Legacy-vs-Policy Routing

**Files:**
- Create: `D:\亿铸智能体\LangG_New\src\policies\diagnostics.py`
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\router.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_policy_diagnostics.py`

- [ ] **Step 1: Write failing diagnostics tests**

```python
def test_record_route_divergence_marks_mismatch():
    payload = record_route_divergence(
        legacy_target="assessment",
        policy_target="chat_main",
        rule_name="assessment_missing_data",
    )

    assert payload["route_diverged"] is True
    assert payload["policy_rule_name"] == "assessment_missing_data"
```

- [ ] **Step 2: Run the diagnostics tests and confirm failure**

Run: `pytest tests/backend/policies/test_policy_diagnostics.py -v`  
Expected: FAIL because diagnostics helpers do not exist yet.

- [ ] **Step 3: Implement divergence-record helpers**

Create helpers like:

```python
def record_route_divergence(*, legacy_target: str, policy_target: str, rule_name: str, reason: str | None = None) -> dict: ...
def record_review_divergence(*, legacy_action: str, policy_action: str, rule_name: str, reason: str | None = None) -> dict: ...
```

Return plain dictionaries so adapters can merge them into state diagnostics or structured logs without extra coupling.

- [ ] **Step 4: Re-run the diagnostics tests**

Run: `pytest tests/backend/policies/test_policy_diagnostics.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/policies/diagnostics.py tests/backend/policies/test_policy_diagnostics.py
git commit -m "feat: add policy shadow diagnostics helpers"
```

## Task 5: Cut `route_after_intent` to Routing Policy via Adapter

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\router.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`

- [ ] **Step 1: Write or extend the failing adapter tests**

```python
def test_route_after_intent_uses_policy_result_for_clinical_assessment():
    state = CRCAgentState(findings={"user_intent": "clinical_assessment"})
    assert route_after_intent(state) == "clinical_entry_resolver"


def test_route_after_intent_keeps_knowledge_query_direct():
    state = CRCAgentState(findings={"user_intent": "knowledge_query"})
    assert route_after_intent(state) == "knowledge"
```

- [ ] **Step 2: Run the route-after-intent tests**

Run: `pytest tests/backend/test_router_dynamic.py -k route_after_intent -v`  
Expected: PASS or mixed result before cutover; keep the test in place as a protection rail.

- [ ] **Step 3: Convert `route_after_intent` into a policy adapter**

Implementation shape:

```python
facts = build_turn_facts(state)
flags = derive_routing_flags(facts)
policy = decide_after_intent(facts, flags)
legacy = _legacy_route_after_intent_adapter(state)
_record_shadow(...)
return policy.target
```

Do not remove shadow comparison after cutover.

- [ ] **Step 4: Re-run the router tests**

Run: `pytest tests/backend/test_router_dynamic.py -k route_after_intent -v`  
Expected: PASS with stable direct-route behavior.

- [ ] **Step 5: Checkpoint**

```bash
git add src/nodes/router.py tests/backend/test_router_dynamic.py
git commit -m "refactor: route intent adapter through routing policy"
```

## Task 6: Cut `dynamic_router` and `route_assessment` to Routing Policy

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\router.py`
- Modify: `D:\亿铸智能体\LangG_New\src\graph_builder.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_assessment_node.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_assessment_node.py`

- [ ] **Step 1: Add or extend failing tests for assessment and dynamic routing**

```python
def test_dynamic_router_keeps_parallel_groups_first():
    ...


def test_route_after_assessment_routes_inquiry_pending_to_end_turn():
    state = CRCAgentState(
        findings={"active_inquiry": True},
        missing_critical_data=["TNM Staging"],
    )
    assert route_after_assessment(state) == "chat_main"
```

- [ ] **Step 2: Run the dynamic and assessment tests**

Run: `pytest tests/backend/test_router_dynamic.py tests/backend/test_assessment_node.py -v`  
Expected: PASS or mixed result before cutover; these tests become the cutover safety net.

- [ ] **Step 3: Move `route_assessment` out of `graph_builder.py`**

Create an adapter function in `src/nodes/router.py` or a small companion adapter module. Keep `graph_builder.py` responsible for graph wiring only.

- [ ] **Step 4: Adapt `dynamic_router` and `route_assessment` to call `routing_policy`**

Preserve:

- pending plan step precedence
- parallel group precedence
- active inquiry behavior
- missing critical data interruption
- current clinical fast-pass semantics through derived flags, not raw `fast_pass_mode`

- [ ] **Step 5: Add explicit transitional comments**

In `router.py` and `assessment_nodes.py`, add comments to:

- `route_after_clinical_entry`
- `route_after_outpatient_triage`
- `node_staging_router`

Use wording like:

```python
# Phase 1 transitional router: narrow-scope exception.
# Do not expand this into a second policy layer.
```

- [ ] **Step 6: Re-run the routing and assessment tests**

Run: `pytest tests/backend/test_router_dynamic.py tests/backend/test_assessment_node.py -v`  
Expected: PASS

- [ ] **Step 7: Checkpoint**

```bash
git add src/nodes/router.py src/graph_builder.py src/nodes/assessment_nodes.py tests/backend/test_router_dynamic.py tests/backend/test_assessment_node.py
git commit -m "refactor: route dynamic and assessment adapters through policy"
```

## Task 7: Emit Structured Critic Review Signals

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\decision_nodes.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_review_policy.py`

- [ ] **Step 1: Write the failing critic-signal tests**

```python
def test_build_critic_signal_marks_parse_fallback_as_degraded():
    signal = build_critic_signal(
        verdict="APPROVED",
        feedback="审核出错自动放行: parser failed",
        degraded_reason="parse_error",
    )

    assert signal.degraded.is_degraded is True
    assert signal.degraded.reason == "parse_error"
```

- [ ] **Step 2: Run the critic-signal tests and confirm failure**

Run: `pytest tests/backend/policies/test_review_policy.py -k critic_signal -v`  
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Add critic signal helpers and degraded metadata**

Add helpers such as:

```python
def build_critic_signal(... ) -> ReviewSignal: ...
```

Update fallback paths in `node_critic` so they preserve:

- verdict
- retryability
- degraded flag
- degraded reason
- fallback value

Do not let fallback approval look identical to normal approval.

- [ ] **Step 4: Re-run the critic-signal tests**

Run: `pytest tests/backend/policies/test_review_policy.py -k critic_signal -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/nodes/decision_nodes.py tests/backend/policies/test_review_policy.py
git commit -m "feat: emit structured critic review signals"
```

## Task 8: Emit Structured Evaluator Review Signals

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\evaluation_nodes.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_review_policy.py`

- [ ] **Step 1: Write the failing evaluator-signal tests**

```python
def test_build_evaluator_signal_captures_scores_and_retryability():
    signal = build_evaluator_signal(
        report={
            "factual_accuracy": 2,
            "citation_accuracy": 3,
            "completeness": 2,
            "safety": 4,
            "verdict": "FAIL",
        },
        actionable_retry=True,
        degraded_reason=None,
    )

    assert signal.verdict == "FAIL"
    assert signal.retryable is True
    assert signal.reasons == ("factual_accuracy<3", "completeness<3")
```

- [ ] **Step 2: Run the evaluator-signal tests and confirm failure**

Run: `pytest tests/backend/policies/test_review_policy.py -k evaluator_signal -v`  
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Add evaluator signal helpers and degraded metadata**

Add helpers such as:

```python
def build_evaluator_signal(... ) -> ReviewSignal: ...
```

Preserve in normalized form:

- evaluator verdict
- evaluator scores
- actionable retry
- degraded status and reason

Keep existing heuristic fast-review behavior unchanged for now; only expose its outcome as a structured signal.

- [ ] **Step 4: Re-run the evaluator-signal tests**

Run: `pytest tests/backend/policies/test_review_policy.py -k evaluator_signal -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/nodes/evaluation_nodes.py tests/backend/policies/test_review_policy.py
git commit -m "feat: emit structured evaluator review signals"
```

## Task 9: Implement Pure Review Policy Decisions

**Files:**
- Create: `D:\亿铸智能体\LangG_New\src\policies\review_policy.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_review_policy.py`

- [ ] **Step 1: Write failing review-policy golden tests**

```python
def test_decide_after_critic_retries_on_rejected_signal():
    facts = TurnFacts(critic_verdict="REJECTED", iteration_count=1, ...)
    signal = ReviewSignal(source="critic", verdict="REJECTED", retryable=True, ...)

    decision = decide_after_critic(facts, signal)

    assert decision.action == "retry_decision"


def test_decide_after_evaluator_finalizes_on_stable_guideline_support():
    facts = TurnFacts(
        stable_guideline_rag_support=True,
        evaluation_retry_count=0,
        evaluator_verdict="FAIL",
        evaluator_actionable_retry=True,
        ...
    )
    evaluator_signal = ReviewSignal(source="evaluator", verdict="FAIL", retryable=True, ...)

    decision = decide_after_evaluator(facts, None, evaluator_signal)

    assert decision.action == "finalize"
```

- [ ] **Step 2: Run the review-policy tests and confirm failure**

Run: `pytest tests/backend/policies/test_review_policy.py -v`  
Expected: FAIL because `review_policy.py` does not exist yet or helpers are incomplete.

- [ ] **Step 3: Implement the pure review policy**

Add:

```python
def decide_after_critic(facts: TurnFacts, critic_signal: ReviewSignal | None) -> ReviewDecision: ...
def decide_after_evaluator(
    facts: TurnFacts,
    critic_signal: ReviewSignal | None,
    evaluator_signal: ReviewSignal | None,
) -> ReviewDecision: ...
```

Preserve current semantics first:

- rejected critic can retry
- retry caps still finalize
- stable guideline support can bypass retry
- degraded pass is not identical to normal pass

- [ ] **Step 4: Re-run the review-policy tests**

Run: `pytest tests/backend/policies/test_review_policy.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/policies/review_policy.py tests/backend/policies/test_review_policy.py
git commit -m "feat: add pure review policy decisions"
```

## Task 10: Cut `route_by_critic_v2` and `route_after_evaluator` to Review Policy

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\decision_nodes.py`
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\evaluation_nodes.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_review_policy.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_review_policy.py`

- [ ] **Step 1: Add adapter-level cutover tests**

```python
def test_route_by_critic_v2_uses_review_policy_retry():
    state = CRCAgentState(critic_verdict="REJECTED", iteration_count=1)
    assert route_by_critic_v2(state) == "decision"


def test_route_after_evaluator_uses_review_policy_finalize():
    state = CRCAgentState(
        evaluation_report={"verdict": "PASS", "factual_accuracy": 4, "citation_accuracy": 4, "completeness": 4, "safety": 4},
        evaluation_retry_count=0,
    )
    assert route_after_evaluator(state) == "finalize"
```

- [ ] **Step 2: Run the adapter tests**

Run: `pytest tests/backend/policies/test_review_policy.py -k "route_by_critic_v2 or route_after_evaluator" -v`  
Expected: PASS or mixed result before cutover; use as guardrails.

- [ ] **Step 3: Convert review routers into adapters**

Implementation shape:

```python
facts = build_turn_facts(state)
critic_signal = build_critic_signal_from_state(state)
evaluator_signal = build_evaluator_signal_from_state(state)
policy = decide_after_critic(...) or decide_after_evaluator(...)
legacy = _legacy_review_route(...)
_record_shadow(...)
return "decision" if policy.action == "retry_decision" else "finalize"
```

- [ ] **Step 4: Re-run the review tests**

Run: `pytest tests/backend/policies/test_review_policy.py -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add src/nodes/decision_nodes.py src/nodes/evaluation_nodes.py tests/backend/policies/test_review_policy.py
git commit -m "refactor: route review adapters through review policy"
```

## Task 11: Run Focused Regression Suite and Document Cutover Readiness

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\docs\superpowers\plans\2026-04-10-routing-review-policy-consolidation.md`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_turn_facts.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_routing_policy.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\policies\test_review_policy.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_assessment_node.py`

- [ ] **Step 1: Run the focused policy and routing regression suite**

Run:

```bash
pytest \
  tests/backend/policies/test_turn_facts.py \
  tests/backend/policies/test_routing_policy.py \
  tests/backend/policies/test_review_policy.py \
  tests/backend/test_router_dynamic.py \
  tests/backend/test_assessment_node.py \
  -v
```

Expected: PASS

- [ ] **Step 2: Run a broader state/regression smoke check**

Run:

```bash
pytest \
  tests/backend/test_state_model_source.py \
  tests/backend/test_state_snapshot.py \
  tests/backend/test_clinical_entry_resolver.py \
  -v
```

Expected: PASS

- [ ] **Step 3: Record cutover-readiness notes in the plan**

Add a short completion note under this task documenting:

- whether shadow divergence remained
- which transitional routers still exist
- whether any thresholds were intentionally left unchanged

**Completion Note (2026-04-10):**

- Shadow divergence remains intentionally active in the adapters for observation, but the focused regression suite and broader smoke check both passed after the Phase 1 cutover.
- Transitional routers still intentionally outside the consolidated policy layer are `route_after_clinical_entry`, `route_after_outpatient_triage`, and `node_staging_router`.
- Thresholds intentionally left unchanged in Phase 1 include `MAX_DECISION_RETRIES = 3`, `MAX_EVALUATION_RETRIES = 2`, `STABLE_GUIDELINE_MIN_COVERAGE = 80`, and `STABLE_GUIDELINE_MIN_INLINE_ANCHORS = 2`.
- Focused verification result: `77 passed` across the policy/routing/state suite.
- Broader smoke verification result: `20 passed` across `test_state_model_source.py`, `test_state_snapshot.py`, and `test_clinical_entry_resolver.py`.

- [ ] **Step 4: Checkpoint**

```bash
git add docs/superpowers/plans/2026-04-10-routing-review-policy-consolidation.md
git commit -m "docs: record policy consolidation cutover readiness"
```

## Implementation Notes

- Use `superpowers:test-driven-development` discipline inside each task even if the implementation is small.
- Do not expand the scope into prompt refactoring in Phase 1.
- Do not move `route_after_clinical_entry`, `route_after_outpatient_triage`, or `node_staging_router` into the consolidated policy layer yet.
- Do not let `routing_policy` or `review_policy` read raw `CRCAgentState`.
- Keep all legacy-to-policy shadow comparisons active until the focused regression suite is green.
