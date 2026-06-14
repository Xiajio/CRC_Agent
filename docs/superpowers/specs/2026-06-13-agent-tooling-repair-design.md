# Agent Tooling Repair Design

**Date:** 2026-06-13  
**Status:** Draft for review  
**Goal:** Make the LangG agent tool surface deterministic, reachable, and testable without changing the clinical graph topology or rewriting model-facing workflows.

## 1. Context

The current agent has several independent tool surfaces:

- graph-level registries in `src/tools/__init__.py`
- database-node-local tools in `src/tools/database_tools.py::ATOMIC_DATABASE_TOOLS`
- runtime executor tools loaded by `src/nodes/tools_executor.py`
- planner-level abstract `PlanStep.tool_needed` values in `src/state.py` and `src/prompts/planner_prompts.py`
- routing mappings in `src/policies/tool_targets.py`

This split is understandable because different nodes have different safety and dependency requirements. The problem is that these layers are not contract-tested against each other. A tool can exist but be unreachable, a planner value can be allowed but unsupported, and a node can silently use a different registry than the graph-level one.

The current tooling map is documented in:

```text
docs/superpowers/specs/2026-06-13-agent-tooling-map-design.md
```

This repair spec turns that inventory into a prioritized stabilization design.

## 2. Problem Statement

LangG currently has a tool-control-plane consistency problem.

The most important symptoms are:

- `search_treatment_recommendations` is listed as a valid planner tool type, but the plan-driven knowledge path can still dispatch generic search through `search_clinical_guidelines` instead of the treatment-specific tool.
- `parallel_subagents` receives graph-level tools, but graph-level registries do not include database tools. A case-database subtask can therefore be routed to a database intent while lacking the same tool set that `node_case_database()` binds locally.
- `calculator` is a valid `PlanStep.tool_needed` value and routes to `tool_executor`, but no concrete calculator tool is registered in the active tool registries.
- `src/tools/basic_tools.py` defines `echo` and `word_count`, but they are not imported by the central registry and are effectively orphaned.
- `list_database_tools()` is stale relative to `ATOMIC_DATABASE_TOOLS`, which makes database tool discovery misleading.
- `list_all_tools()` and graph-level registries differ in ways that are not explicitly modeled, so the difference can be intentional in one place and accidental in another.
- Directly importing all registries appears slow enough to time out in a short smoke command, which suggests the tool registry may be doing heavy work at import time.

The desired end state is not "all tools everywhere." The desired end state is explicit, tested reachability: every planner abstraction has a route, every route has the expected concrete tools, and every intentionally isolated node-local tool set is documented and contract-tested.

## 3. Scope

### In scope

- Add a single source of truth for tool metadata or an equivalent contract layer.
- Align planner `tool_needed` values with actual route targets and available tools.
- Fix plan-driven RAG dispatch for treatment, staging, drug, guideline TOC, and chapter reads.
- Ensure case-database subtasks in parallel execution receive the authoritative database tool set.
- Resolve unsupported or orphaned tools by either registering them intentionally or removing their advertised planner surface.
- Add focused backend tests that validate registry, route, and node reachability contracts without loading heavy ML assets.
- Keep current doctor and patient graph topology intact.

### Out of scope

- Rewriting the clinical prompts beyond necessary tool-name corrections.
- Changing the medical reasoning policy or diagnosis/treatment decision flow.
- Replacing LangGraph structure.
- Reworking radiology, pathology, tumor screening, or radiomics model internals.
- Changing frontend behavior.
- Adding broad new external tools.
- Expanding patient-mode access to doctor-only tools.

## 4. Design Principles

- Tool access must be explicit by scope: graph-level, node-local, tool-executor-only, doctor-only, patient-safe.
- Planner values are not concrete tool names, so they need a tested mapping layer.
- Registry functions should be cheap to import and should not initialize heavy models.
- Node-local tool sets are acceptable when the node has special safety or data-access requirements, but the exception must be documented in code and tests.
- Fixes should preserve compatibility first, then simplify.
- Patient graph restrictions should remain conservative.

## 5. Proposed Approach

Use a manifest-first contract, but apply it incrementally.

The repair should not begin with a broad refactor. Instead, introduce a lightweight metadata layer that describes the current intended state, then make the smallest code changes needed for the tests to pass.

Recommended path:

1. Add contract tests that encode expected current behavior and expose the known gaps.
2. Fix the direct gaps in RAG dispatch, database subagent tooling, stale database discovery, and unsupported planner values.
3. Introduce or complete a manifest helper only where it reduces duplication between registries, prompts, routing, and tests.
4. Keep public registry function names stable: `list_tools()`, `list_tools_with_web_search()`, and `list_all_tools()`.

## 6. Tool Manifest Contract

Add a lightweight module:

```text
src/tools/manifest.py
```

The manifest should define metadata for tool reachability. It does not need to instantiate every LangChain tool at import time.

Suggested fields:

```text
ToolSpec
- name: concrete tool name
- category: rag | web | database | imaging | pathology | tumor | utility | formatting
- registries: graph | graph_web | executor | database_node | optional
- planner_aliases: abstract PlanStep.tool_needed values that can reach this tool
- route_target: knowledge | case_database | rad_agent | path_agent | web_search | tool_executor | decision
- graph_scope: doctor | patient | both | node_local
- requires_web: bool
- heavy_dependency: bool
- patient_safe: bool
- factory_ref: optional lazy factory name
```

The first implementation may use plain dictionaries or dataclasses. Pydantic is not required.

The key rule: metadata can be imported cheaply. Concrete tool construction remains in existing modules and registry functions.

## 7. Required Fixes

### 7.1 RAG dispatch alignment

`src/nodes/knowledge_nodes.py` should dispatch plan-driven RAG steps by explicit tool intent, not by a generic search branch.

Minimum expected mapping:

| Planner value | Expected concrete tool |
| --- | --- |
| `search_treatment_recommendations` | `search_treatment_recommendations` |
| `search_staging_criteria` | `search_staging_criteria` |
| `search_drug_information` | `search_drug_information` |
| `search_clinical_guidelines` | `search_clinical_guidelines` |
| `list_guideline_toc`, `toc` | `list_guideline_toc` |
| `read_guideline_chapter`, `read`, `chapter` | `read_guideline_chapter` |
| `web_search`, `web` | `web_search` when web search is enabled |

Acceptance signal:

- A test can build a planned knowledge step with `tool_needed="search_treatment_recommendations"` and verify the treatment-specific tool path is selected.
- Generic `search` should either remain a guideline search alias or be removed from planner guidance if it is too ambiguous.

### 7.2 Parallel case-database tooling

`src/nodes/parallel_subagents.py` should not rely only on graph-level tools for case-database subtasks.

Recommended minimal fix:

- When a parallel step is assigned to `case_database`, pass `ATOMIC_DATABASE_TOOLS` or merge them into the step-specific tool set.
- Keep the standalone `node_case_database()` binding unchanged.

Alternative:

- Move database tool selection into the manifest and let `parallel_subagents` request tools by `route_target="case_database"`.

Acceptance signal:

- A parallel step with `tool_needed="case_database_query"` resolves to the same database tool names used by `node_case_database()`.

### 7.3 Unsupported calculator planner value

`calculator` is currently advertised but unsupported.

Recommended fix:

- Remove `calculator` from `PlanStep.tool_needed` accepted values and from planner prompt guidance unless a concrete arithmetic tool is needed now.
- Keep `tool_executor` available for concrete tools that actually exist.

Alternative:

- Add a small safe arithmetic-only calculator tool to `src/tools/basic_tools.py` and register it in `list_all_tools()` only.

Acceptance signal:

- There is no planner-accepted value that routes to a missing concrete capability.

### 7.4 Orphaned basic tools

`src/tools/basic_tools.py` should be made intentional.

Recommended fix:

- Register `word_count` in `list_all_tools()` if useful for executor diagnostics, and leave it out of graph-level clinical tools.
- Keep or remove `echo`, but do not leave it silently orphaned. If kept, document it as diagnostic-only and executor-only.

Acceptance signal:

- Every module under `src/tools/` that defines LangChain tools is either referenced by a registry/manifest or explicitly marked internal.

### 7.5 Database discovery drift

`list_database_tools()` should derive from `ATOMIC_DATABASE_TOOLS` or be renamed as legacy.

Recommended fix:

- Make `list_database_tools()` return the names and descriptions of `ATOMIC_DATABASE_TOOLS`.
- If legacy names are still needed for callers, create a clearly named compatibility helper instead of mixing old and new discovery.

Acceptance signal:

- `set(item["name"] for item in list_database_tools())` equals `{tool.name for tool in ATOMIC_DATABASE_TOOLS}` unless documented compatibility entries are intentionally included.

### 7.6 Registry performance guard

Registry import and metadata inspection should stay lightweight.

Recommended fix:

- Keep heavy ML model loading behind tool invocation, status calls, or explicit lazy wrappers.
- Add a smoke test or simple timing guard for manifest import and registry metadata construction.

Acceptance signal:

- Importing `src.tools.manifest` does not load radiology/pathology/tumor model weights.
- A registry smoke command completes within a practical local threshold without network access.

## 8. Tests

Add focused backend tests, likely under:

```text
tests/backend/test_agent_tool_registry_contract.py
```

Minimum cases:

- `PlanStep` accepted `tool_needed` values all map through `src/policies/tool_targets.py`.
- Planner prompt documented tool values match the accepted values in `src/state.py`, except for explicitly documented aliases.
- Each route target has the expected concrete tool family:
  - `knowledge`: RAG tools
  - `case_database`: `ATOMIC_DATABASE_TOOLS`
  - `rad_agent`: imaging/radiomics tools
  - `path_agent`: CLAM pathology tools
  - `web_search`: `web_search` when enabled
  - `tool_executor`: only registered executor tools
- `search_treatment_recommendations` is not downgraded to generic guideline search in the planned knowledge path.
- `list_database_tools()` matches `ATOMIC_DATABASE_TOOLS`.
- No advertised planner value points to an empty concrete tool set.

Test constraints:

- Tests must not require web access.
- Tests must not require real model weights.
- Tests should prefer metadata and monkeypatched tool objects over invoking clinical tools.

## 9. Rollout Plan

### Phase 1: Contract tests

Add tests for the known mismatches first. These tests should fail before the repair, except for cases where current behavior is already correct.

### Phase 2: Minimal code fixes

Fix:

- knowledge RAG dispatch
- parallel database subagent tool injection
- unsupported `calculator`
- stale `list_database_tools()`
- orphaned `basic_tools.py` registration or documentation

### Phase 3: Manifest consolidation

Introduce `src/tools/manifest.py` if Phase 2 still leaves duplicated hand-maintained lists. Use it to drive tests and, where practical, registry helpers.

### Phase 4: Documentation refresh

Update:

- planner prompt tool list
- agent tooling map if registry behavior changes
- any developer-facing docs that mention database or executor tools

## 10. Acceptance Criteria

The repair is complete when:

- Every accepted `PlanStep.tool_needed` value has a tested route target.
- Every route target has an explicit concrete tool set or an explicit no-tool policy.
- `search_treatment_recommendations` reaches the treatment recommendation RAG tool in plan-driven knowledge execution.
- Parallel case-database subtasks can access the authoritative database tools.
- `calculator` is either backed by a concrete safe tool or removed from the advertised planner surface.
- `list_database_tools()` no longer reports stale tool names.
- Orphaned utility tools are either registered intentionally or documented as internal.
- Focused backend contract tests pass locally.
- No heavy ML model initialization is required during metadata import or contract tests.

Suggested verification command after implementation:

```powershell
pytest tests/backend/test_agent_tool_registry_contract.py -q
```

If the implementation touches RAG dispatch, also run the nearest existing RAG or node tests. If it touches path/tool loading, run the existing path-security tests as a regression check.

## 11. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Accidentally broadening patient-mode tool access | Keep `patient_safe` and graph scope explicit; add tests for patient graph restrictions |
| Heavy imports during tests | Keep manifest metadata separate from concrete tool construction |
| Breaking treatment decision RAG behavior | Limit changes to planned knowledge dispatch; keep `node_decision()` isolated RAG path compatible |
| Database tool duplication | Treat `ATOMIC_DATABASE_TOOLS` as authoritative until manifest consolidation is proven |
| Prompt and validator drift | Add a prompt/validator consistency test |
| Ambiguous `search` planner alias | Either document it as guideline search or remove it from planner guidance |

## 12. Open Questions

- Should generic `search` remain a valid planner alias, or should all planner RAG intents become explicit?
- Should utility tools like `word_count` be available to the clinical tool executor, or should they stay test-only?
- Is a safe arithmetic calculator clinically useful enough to implement, or is removing the planner value better?
- Should `search_latest_research` remain executor-only, or should it be included in graph-level web tools when web search is enabled?

## 13. Recommendation

Proceed with Phase 1 and Phase 2 first. The immediate correctness issues do not require a large registry rewrite.

Use `ATOMIC_DATABASE_TOOLS` as the authoritative database source, fix RAG dispatch explicitly, remove or back unsupported planner values, and add contract tests that make future drift visible.

Only then decide whether to make `src/tools/manifest.py` the source for all registry construction.
