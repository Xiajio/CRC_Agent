# Outpatient Triage vs CRC Clinical Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split symptom-led GI outpatient triage from CRC case-completion questioning without breaking the current intent classifier, planner flow, or non-clinical routes.

**Architecture:** Keep the current intent classifier as the first routing layer. Add a thin `clinical_entry_resolver` layer that decides whether a clinical request enters the new `outpatient_triage` track or the existing CRC clinical pipeline. Reuse the current planner, diagnosis, staging, and decision chain for CRC work, and reuse the current frontend conversation/card/roadmap surfaces with one new `triage_card` type.

**Tech Stack:** Python, LangGraph, FastAPI, Pydantic, pytest, React, TypeScript, Vitest, Vite

---

**Repository note:** This workspace is not currently a git repository, so commit steps are written as checkpoints but cannot be executed until repository metadata is restored.

## File Structure

### Backend files to modify

- `D:\亿铸智能体\LangG_New\src\state.py`
  - Add new clinical-track and triage-related fields to the state model and any helper validation needed.
- `D:\亿铸智能体\LangG_New\src\prompts\intent_prompts.py`
  - Clarify prompt guidance so `clinical_assessment` remains a clinical entry intent instead of implying immediate CRC-only follow-up.
- `D:\亿铸智能体\LangG_New\src\nodes\intent_nodes.py`
  - Preserve current intent writes while ensuring new triage-related fields are reset safely when appropriate.
- `D:\亿铸智能体\LangG_New\src\nodes\router.py`
  - Add the clinical-entry routing layer and planner tool mapping for `outpatient_triage`.
- `D:\亿铸智能体\LangG_New\src\graph_builder.py`
  - Insert the new resolver node and triage node into the graph while preserving existing direct routes.
- `D:\亿铸智能体\LangG_New\src\nodes\assessment_nodes.py`
  - Narrow the node to CRC case-completion and support one-time direct-entry explanation.

### Backend files to create

- `D:\亿铸智能体\LangG_New\src\nodes\clinical_entry_nodes.py`
  - Implement `clinical_entry_resolver`.
- `D:\亿铸智能体\LangG_New\src\nodes\outpatient_triage_nodes.py`
  - Implement outpatient triage logic and structured outputs.

### Backend tests to modify or create

- `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
  - Extend routing coverage for the new entry gate.
- `D:\亿铸智能体\LangG_New\tests\backend\test_event_normalizer.py`
  - Verify new triage card and triage findings surface correctly if needed.
- `D:\亿铸智能体\LangG_New\tests\backend\test_state_snapshot.py`
  - Verify new state fields survive snapshots.
- `D:\亿铸智能体\LangG_New\tests\backend\test_chat_stream_route.py`
  - Add stream-level assertions once the new card and route are available.
- `D:\亿铸智能体\LangG_New\tests\backend\test_clinical_entry_resolver.py`
  - New focused tests for track resolution.
- `D:\亿铸智能体\LangG_New\tests\backend\test_outpatient_triage_node.py`
  - New focused tests for triage output and non-CRC behavior.

### Frontend files to modify

- `D:\亿铸智能体\LangG_New\frontend\src\features\cards\card-renderers.tsx`
  - Add `triage_card` title and renderer.
- `D:\亿铸智能体\LangG_New\frontend\src\app\store\stream-reducer.ts`
  - Register `triage_card` as an inline card with proper priority.
- `D:\亿铸智能体\LangG_New\frontend\src\app\api\types.ts`
  - Keep types aligned if any test fixture or state field coverage needs expansion.

### Frontend tests to modify

- `D:\亿铸智能体\LangG_New\tests\frontend\clinical-cards-panel.test.tsx`
  - Add `triage_card` rendering assertions.
- `D:\亿铸智能体\LangG_New\tests\frontend\stream-reducer.test.ts`
  - Add inline card priority and merge behavior for `triage_card`.
- `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`
  - Verify triage results surface correctly in the page shell.

## Task 1: Add State Fields for Track and Triage Outputs

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\state.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_state_snapshot.py`

- [ ] **Step 1: Write the failing snapshot test for new track fields**

```python
def test_build_snapshot_preserves_triage_track_fields():
    state = CRCAgentState(
        findings={
            "encounter_track": "outpatient_triage",
            "clinical_entry_reason": "none",
            "triage_risk_level": "medium",
            "triage_disposition": "urgent_gi_clinic",
            "triage_suggested_tests": ["血常规", "肠镜"],
        }
    )

    snapshot = build_snapshot(state)

    assert snapshot.findings["encounter_track"] == "outpatient_triage"
    assert snapshot.findings["triage_risk_level"] == "medium"
```

- [ ] **Step 2: Run the backend snapshot test and verify failure**

Run: `pytest tests/backend/test_state_snapshot.py -k triage_track -v`  
Expected: FAIL because the test does not exist yet or the new fields are not asserted.

- [ ] **Step 3: Extend `CRCAgentState` findings conventions in `src/state.py`**

Add the following tracked keys to the state contract through documentation/comments and any helper model shapes needed:

```python
TRIAGE_FINDING_KEYS = {
    "encounter_track",
    "clinical_entry_reason",
    "known_crc_signals",
    "triage_risk_level",
    "triage_disposition",
    "triage_suggested_tests",
    "triage_summary",
    "symptom_snapshot",
    "entry_explanation_shown",
}
```

Do not add a separate top-level reducer unless the implementation truly needs one. Keep the change minimal and compatible with the existing shallow findings merge.

- [ ] **Step 4: Re-run the snapshot test**

Run: `pytest tests/backend/test_state_snapshot.py -k triage_track -v`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

Git checkpoint command if repository metadata is restored:

```bash
git add src/state.py tests/backend/test_state_snapshot.py
git commit -m "feat: add triage routing state fields"
```

## Task 2: Add the Clinical Entry Resolver

**Files:**
- Create: `D:\亿铸智能体\LangG_New\src\nodes\clinical_entry_nodes.py`
- Modify: `D:\亿铸智能体\LangG_New\src\graph_builder.py`
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\router.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_clinical_entry_resolver.py`

- [ ] **Step 1: Write the failing resolver tests**

```python
def test_clinical_assessment_without_crc_signals_routes_to_outpatient_triage():
    state = CRCAgentState(findings={"user_intent": "clinical_assessment"})
    updates = resolve_clinical_entry(state)
    assert updates["findings"]["encounter_track"] == "outpatient_triage"


def test_clinical_assessment_with_system_crc_signal_routes_to_crc_track():
    state = CRCAgentState(
        findings={"user_intent": "clinical_assessment", "pathology_confirmed": True}
    )
    updates = resolve_clinical_entry(state)
    assert updates["findings"]["encounter_track"] == "crc_clinical"
    assert updates["findings"]["clinical_entry_reason"] == "system_known_crc_signal"


def test_treatment_decision_always_routes_to_crc_track():
    state = CRCAgentState(findings={"user_intent": "treatment_decision"})
    updates = resolve_clinical_entry(state)
    assert updates["findings"]["encounter_track"] == "crc_clinical"
```

- [ ] **Step 2: Run the resolver tests and verify failure**

Run: `pytest tests/backend/test_clinical_entry_resolver.py -v`  
Expected: FAIL because the module and function do not exist yet.

- [ ] **Step 3: Implement `clinical_entry_nodes.py` with focused helpers**

Create:

- `collect_known_crc_signals(state: CRCAgentState) -> dict`
- `resolve_clinical_entry(state: CRCAgentState) -> dict`
- `route_after_clinical_entry(state: CRCAgentState) -> str`

Minimal logic:

```python
if intent == "treatment_decision":
    return crc_track("treatment_decision_request")
if has_current_turn_strong_signal or has_system_known_signal:
    return crc_track(reason)
return outpatient_track()
```

Do not add LLM calls here.

- [ ] **Step 4: Wire the node into `graph_builder.py`**

Add the node registration and route:

```python
builder.add_node("clinical_entry_resolver", node_clinical_entry_resolver())
```

Then route:

```python
"clinical_entry_resolver": NodeName.CLINICAL_ENTRY_RESOLVER
```

- [ ] **Step 5: Update `router.py` for the new planner tool mapping**

Add support for:

- `outpatient_triage`
- `crc_clinical_assessment`

So the planner can explicitly target the correct node later.

- [ ] **Step 6: Re-run the resolver tests**

Run: `pytest tests/backend/test_clinical_entry_resolver.py tests/backend/test_router_dynamic.py -v`  
Expected: PASS

- [ ] **Step 7: Checkpoint**

```bash
git add src/nodes/clinical_entry_nodes.py src/graph_builder.py src/nodes/router.py tests/backend/test_clinical_entry_resolver.py tests/backend/test_router_dynamic.py
git commit -m "feat: add clinical entry resolver"
```

## Task 3: Preserve Existing Intent Behavior While Routing Clinical Intents Through the Resolver

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\prompts\intent_prompts.py`
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\intent_nodes.py`
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\router.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`

- [ ] **Step 1: Add a failing routing test for non-clinical compatibility**

```python
def test_route_after_intent_keeps_knowledge_query_direct():
    state = CRCAgentState(findings={"user_intent": "knowledge_query"})
    assert route_after_intent(state) == "knowledge"


def test_route_after_intent_sends_clinical_assessment_to_entry_resolver():
    state = CRCAgentState(findings={"user_intent": "clinical_assessment"})
    assert route_after_intent(state) == "clinical_entry_resolver"
```

- [ ] **Step 2: Run the routing tests and verify failure**

Run: `pytest tests/backend/test_router_dynamic.py -k route_after_intent -v`  
Expected: FAIL because `clinical_assessment` still routes directly to planner/assessment.

- [ ] **Step 3: Update the intent prompt language**

In `src/prompts/intent_prompts.py`, clarify in the prompt text:

- `clinical_assessment` means a clinical entry request
- track selection between outpatient triage and CRC follow-up happens later

Do not add a brand-new top-level intent category yet.

- [ ] **Step 4: Update `route_after_intent` and any reset logic**

In `src/nodes/router.py` and `src/nodes/intent_nodes.py`:

- send `clinical_assessment` and `treatment_decision` to `clinical_entry_resolver`
- keep all current non-clinical direct routes unchanged
- reset stale triage fields in `_base_findings()` when switching to clearly non-clinical intents

- [ ] **Step 5: Re-run the routing tests**

Run: `pytest tests/backend/test_router_dynamic.py -v`  
Expected: PASS for both old direct-route tests and new clinical-entry tests.

- [ ] **Step 6: Checkpoint**

```bash
git add src/prompts/intent_prompts.py src/nodes/intent_nodes.py src/nodes/router.py tests/backend/test_router_dynamic.py
git commit -m "feat: route clinical intents through entry resolver"
```

## Task 4: Implement the Outpatient Triage Node

**Files:**
- Create: `D:\亿铸智能体\LangG_New\src\nodes\outpatient_triage_nodes.py`
- Modify: `D:\亿铸智能体\LangG_New\src\graph_builder.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_outpatient_triage_node.py`

- [ ] **Step 1: Write the failing triage-node tests**

```python
def test_outpatient_triage_returns_risk_disposition_and_tests():
    state = CRCAgentState(messages=[HumanMessage(content="我腹痛两周，还便血")])
    result = node_outpatient_triage(model=stub_model)(state)

    assert result["findings"]["encounter_track"] == "outpatient_triage"
    assert result["findings"]["triage_risk_level"] in {"medium", "high"}
    assert "triage_suggested_tests" in result["findings"]


def test_outpatient_triage_does_not_set_crc_missing_data():
    state = CRCAgentState(messages=[HumanMessage(content="我腹泻一周")])
    result = node_outpatient_triage(model=stub_model)(state)

    assert result.get("missing_critical_data") in (None, [])
```

- [ ] **Step 2: Run the triage-node tests and verify failure**

Run: `pytest tests/backend/test_outpatient_triage_node.py -v`  
Expected: FAIL because the node does not exist.

- [ ] **Step 3: Implement `outpatient_triage_nodes.py` minimally**

Create a small structured-output node that:

- reads the latest symptom-led user text
- produces:
  - `triage_risk_level`
  - `triage_disposition`
  - `triage_suggested_tests`
  - `triage_summary`
  - `symptom_snapshot`
- returns one assistant message
- does not write CRC completion fields such as TNM requirements

Use a small schema like:

```python
class OutpatientTriageResult(BaseModel):
    risk_level: Literal["low", "medium", "high"]
    disposition: Literal["emergency", "urgent_gi_clinic", "routine_gi_clinic", "observe", "enter_crc_flow"]
    suggested_tests: list[str]
    summary: str
    follow_up_question: str | None = None
```

- [ ] **Step 4: Register the node in `graph_builder.py`**

Add the node and connect `clinical_entry_resolver -> outpatient_triage -> memory/end_turn`.

- [ ] **Step 5: Re-run the outpatient triage tests**

Run: `pytest tests/backend/test_outpatient_triage_node.py -v`  
Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add src/nodes/outpatient_triage_nodes.py src/graph_builder.py tests/backend/test_outpatient_triage_node.py
git commit -m "feat: add outpatient triage node"
```

## Task 5: Narrow `assessment` to CRC Case Completion and Add One-Time Direct-Entry Explanation

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\src\nodes\assessment_nodes.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_outpatient_triage_node.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_event_normalizer.py`

- [ ] **Step 1: Add the failing assessment test**

```python
def test_crc_direct_entry_explanation_is_prepended_once():
    state = CRCAgentState(
        findings={
            "encounter_track": "crc_clinical",
            "clinical_entry_reason": "system_known_crc_signal",
            "entry_explanation_shown": False,
        },
        messages=[HumanMessage(content="我腹痛")]
    )

    result = node_assessment(model=stub_model, tools=[])(state)

    message = result["messages"][0].content
    assert "已有异常检查线索" in message
    assert result["findings"]["entry_explanation_shown"] is True
```

- [ ] **Step 2: Run the assessment test and verify failure**

Run: `pytest tests/backend/test_outpatient_triage_node.py -k direct_entry -v`  
Expected: FAIL because `assessment` does not prepend the explanation yet.

- [ ] **Step 3: Add the one-time explanation logic in `assessment_nodes.py`**

Implement a helper:

```python
def _prepend_crc_entry_explanation_if_needed(state, inquiry_message):
    ...
```

Conditions:

- `clinical_entry_reason == "system_known_crc_signal"`
- `entry_explanation_shown == False`

Output:

- prepend the explanation text to the first CRC inquiry
- set `entry_explanation_shown = True`

- [ ] **Step 4: Verify `assessment` no longer behaves like outpatient triage**

Search for symptom-led triage wording in `assessment_nodes.py` and remove any logic that treats symptom-only requests as the normal first question path.

- [ ] **Step 5: Re-run the targeted tests**

Run: `pytest tests/backend/test_outpatient_triage_node.py tests/backend/test_event_normalizer.py -v`  
Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add src/nodes/assessment_nodes.py tests/backend/test_outpatient_triage_node.py tests/backend/test_event_normalizer.py
git commit -m "feat: add crc direct-entry explanation"
```

## Task 6: Surface Triage Results as a First-Class Card and Snapshot-Friendly Findings

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_event_normalizer.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_state_snapshot.py`
- Modify: `D:\亿铸智能体\LangG_New\tests\backend\test_chat_stream_route.py`

- [ ] **Step 1: Add a failing normalizer test for `triage_card`**

```python
def test_normalize_tick_emits_triage_card_upsert():
    events = normalize_tick(
        "outpatient_triage",
        {
            "cards": {
                "triage_card": {
                    "risk_level": "medium",
                    "disposition": "urgent_gi_clinic",
                }
            }
        },
        messages=[],
    )

    assert any(getattr(event, "card_type", None) == "triage_card" for event in events)
```

- [ ] **Step 2: Run the backend card/snapshot tests and verify failure**

Run: `pytest tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py -k triage -v`  
Expected: FAIL because no triage card exists yet.

- [ ] **Step 3: Ensure the node returns a card-shaped payload**

When `outpatient_triage` completes, emit a card payload in the same shape used by existing card upserts:

```python
{
    "triage_card": {
        "title": "门诊分诊",
        "risk_level": "...",
        "disposition": "...",
        "chief_symptoms": [...],
        "suggested_tests": [...],
    }
}
```

Re-use existing card extraction/normalization pathways instead of inventing a new event type.

- [ ] **Step 4: Add or update route-level stream coverage**

In `tests/backend/test_chat_stream_route.py`, add one fixture-backed or stub-backed flow that confirms `triage_card` reaches the stream.

- [ ] **Step 5: Re-run the backend stream/card tests**

Run: `pytest tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py tests/backend/test_chat_stream_route.py -v`  
Expected: PASS where environment dependencies are available.

- [ ] **Step 6: Checkpoint**

```bash
git add tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py tests/backend/test_chat_stream_route.py
git commit -m "feat: stream outpatient triage card"
```

## Task 7: Render `triage_card` in the Frontend

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\features\cards\card-renderers.tsx`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\clinical-cards-panel.test.tsx`

- [ ] **Step 1: Write the failing frontend card test**

```tsx
it("renders triage card risk, disposition, and suggested tests", () => {
  render(
    <ClinicalCardsPanel
      cards={{
        triage_card: {
          title: "门诊分诊",
          risk_level: "medium",
          disposition: "urgent_gi_clinic",
          suggested_tests: ["血常规", "肠镜"],
        },
      }}
      selectedCardType="triage_card"
    />
  );

  expect(screen.getByText("门诊分诊")).toBeInTheDocument();
  expect(screen.getByText("urgent_gi_clinic")).toBeInTheDocument();
  expect(screen.getByText("血常规")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the frontend card test and verify failure**

Run: `npm run test -- --run ../tests/frontend/clinical-cards-panel.test.tsx`  
Expected: FAIL because `triage_card` is not rendered yet.

- [ ] **Step 3: Implement the new card renderer**

In `card-renderers.tsx`:

- add `triage_card` to `cardTitle()`
- add `renderTriageCard(payload)`
- render:
  - risk
  - disposition
  - symptom summary
  - suggested tests
  - urgent-care warning if present

Minimal renderer shape:

```tsx
function renderTriageCard(payload: JsonObject) {
  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">门诊分诊</p>
        ...
      </div>
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}
```

- [ ] **Step 4: Re-run the frontend card test**

Run: `npm run test -- --run ../tests/frontend/clinical-cards-panel.test.tsx`  
Expected: PASS

- [ ] **Step 5: Checkpoint**

```bash
git add frontend/src/features/cards/card-renderers.tsx tests/frontend/clinical-cards-panel.test.tsx
git commit -m "feat: render triage card in frontend"
```

## Task 8: Register `triage_card` Inline Priority and Page-Level Behavior

**Files:**
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\app\store\stream-reducer.ts`
- Modify: `D:\亿铸智能体\LangG_New\frontend\src\app\api\types.ts`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\stream-reducer.test.ts`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`

- [ ] **Step 1: Add the failing reducer test**

```ts
it("keeps triage_card as the highest-priority non-decision inline card", () => {
  const next = reduceStreamEvent(
    createInitialSessionState(),
    {
      type: "card.upsert",
      card_type: "triage_card",
      payload: { risk_level: "medium" },
      source_channel: "state",
    }
  );

  expect(next.pendingInlineCards?.[0]?.cardType).toBe("triage_card");
});
```

- [ ] **Step 2: Run the reducer and page tests and verify failure**

Run: `npm run test -- --run ../tests/frontend/stream-reducer.test.ts ../tests/frontend/workspace-page.test.tsx`  
Expected: FAIL because `triage_card` is not in the allowlist/priority map.

- [ ] **Step 3: Register `triage_card` in `stream-reducer.ts`**

Update:

- `INLINE_CARD_TYPES`
- `INLINE_CARD_PRIORITY`

Recommended priority:

```ts
decision_card: 6,
triage_card: 5,
patient_card: 4,
imaging_card: 3,
tumor_detection_card: 2,
radiomics_report_card: 1,
```

Only adjust existing values if the tests demonstrate that a rebalance is required.

- [ ] **Step 4: Add a page-level triage visibility test**

In `workspace-page.test.tsx`, simulate:

- one assistant message
- one `triage_card` upsert

Then assert the page shows the card content in the side panel or inline card path expected by current behavior.

- [ ] **Step 5: Re-run the reducer and page tests**

Run: `npm run test -- --run ../tests/frontend/stream-reducer.test.ts ../tests/frontend/workspace-page.test.tsx`  
Expected: PASS

- [ ] **Step 6: Checkpoint**

```bash
git add frontend/src/app/store/stream-reducer.ts frontend/src/app/api/types.ts tests/frontend/stream-reducer.test.ts tests/frontend/workspace-page.test.tsx
git commit -m "feat: support triage card inline priority"
```

## Task 9: End-to-End Regression Sweep for Both Tracks

**Files:**
- Modify as needed based on failures from earlier tasks
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_clinical_entry_resolver.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_outpatient_triage_node.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_router_dynamic.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_event_normalizer.py`
- Test: `D:\亿铸智能体\LangG_New\tests\backend\test_state_snapshot.py`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\clinical-cards-panel.test.tsx`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\stream-reducer.test.ts`
- Test: `D:\亿铸智能体\LangG_New\tests\frontend\workspace-page.test.tsx`

- [ ] **Step 1: Run the focused backend suite**

Run:

```bash
pytest tests/backend/test_clinical_entry_resolver.py tests/backend/test_outpatient_triage_node.py tests/backend/test_router_dynamic.py tests/backend/test_event_normalizer.py tests/backend/test_state_snapshot.py -v
```

Expected: PASS

- [ ] **Step 2: Run the focused frontend suite**

Run:

```bash
npm run test -- --run ../tests/frontend/clinical-cards-panel.test.tsx ../tests/frontend/stream-reducer.test.ts ../tests/frontend/workspace-page.test.tsx
```

Expected: PASS

- [ ] **Step 3: Run one real dialogue retest**

Use the existing manual retest approach:

- symptom-only input should stay in outpatient triage
- known CRC-signal input should bypass triage and explain why

Expected:

- no duplicate replies
- correct track-specific questioning

- [ ] **Step 4: Record any environment caveats**

If any route tests require unavailable dependencies such as `fastapi`, document the exact blocker and do not claim those tests passed.

- [ ] **Step 5: Final checkpoint**

```bash
git add src frontend tests
git commit -m "feat: split outpatient triage from crc clinical routing"
```

## Execution Notes

- Keep changes incremental. Do not try to land backend routing, triage node, narrowed assessment, and frontend triage card in one untested patch.
- Preserve all non-clinical direct routes first; regressions there would be higher cost than delayed triage polish.
- Prefer adding focused tests over rewriting broad fixture suites.
- When updating `assessment`, remove only behavior that clearly belongs to outpatient triage. Do not destabilize diagnosis, staging, or decision logic beyond what is required for the new entry model.
