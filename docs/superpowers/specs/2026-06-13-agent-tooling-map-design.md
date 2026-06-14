# LangG Agent Tooling Map - 2026-06-13

This document captures the current tool surface of the LangG agent project. It focuses on how tools are defined, registered, injected into graphs, and actually reached at runtime.

## Scope

Reviewed files:

- `src/tools/__init__.py`
- `src/tools/*.py`
- `src/graph_builder.py`
- `src/state.py`
- `src/prompts/planner_prompts.py`
- `src/policies/tool_targets.py`
- `src/policies/routing_policy.py`
- `src/nodes/knowledge_nodes.py`
- `src/nodes/database_nodes.py`
- `src/nodes/radiology_nodes.py`
- `src/nodes/pathology_nodes.py`
- `src/nodes/decision_nodes.py`
- `src/nodes/parallel_subagents.py`

Runtime settings observed in `.env`:

- `LLM_MODE=API`
- `LLM_MODEL=deepseek-v4-flash`
- `WEB_SEARCH_ENABLED=true`
- `CHECKPOINT_KIND=memory`

## Registry Layers

The project currently has three registry functions in `src/tools/__init__.py`:

| Registry | Intended contents | Runtime use |
| --- | --- | --- |
| `list_tools()` | Clinical tools, tumor screening, tumor localization, radiomics, pathology CLAM, enhanced RAG | Used by graph construction when web search is disabled |
| `list_tools_with_web_search()` | `list_tools()` plus clinical web-search tools | Used by graph construction when `WEB_SEARCH_ENABLED=true` |
| `list_all_tools()` | Clinical tools, enhanced RAG, all web search tools, database tools, card formatter, imaging/pathology tool variants | Used by `node_tool_executor`, not the normal graph tool injection path |

Important separate registry:

- `src/tools/database_tools.py::ATOMIC_DATABASE_TOOLS` is bound directly in `node_case_database()`. This is the authoritative database tool set for the database node, regardless of the graph-level `tools` list.

## Graph Injection

`src/graph_builder.py::_load_agent_tools()` loads the graph-level tool list:

- If `settings.web_search.enabled` is true, it uses `list_tools_with_web_search()`.
- Otherwise it uses `list_tools()`.

With the current `.env`, the doctor and patient graphs receive `list_tools_with_web_search()`.

Doctor graph nodes that consume tools:

| Node | Tool access pattern |
| --- | --- |
| `knowledge` | Receives graph-level tools; chooses local RAG plus optional web tools |
| `case_database` | Receives graph-level tools but primarily uses `ATOMIC_DATABASE_TOOLS` directly |
| `rad_agent` | Receives graph-level tools and selects imaging/radiomics tools by name |
| `path_agent` | Receives graph-level tools and selects CLAM tools by name |
| `web_search` | Receives graph-level tools and looks for `web_search` |
| `tool_executor` | Does not use graph-level tools; dynamically calls `list_all_tools()` |
| `parallel_subagents` | Receives graph-level tools and filters them by step type |
| `decision` | Receives graph-level tools and selects RAG/web evidence tools |
| `assessment` / `diagnosis` / staging | Receive graph-level tools, but most core logic is structured state and LLM driven |

Patient graph nodes are intentionally narrower:

| Node | Tool access pattern |
| --- | --- |
| `knowledge` | Receives graph-level tools |
| `assessment` | Receives graph-level tools |
| `chat_main` / `general_chat` | No direct broad tool execution |

Patient planner routing also remaps advanced targets such as `case_database`, `rad_agent`, `path_agent`, `web_search`, and `tool_executor` to safer paths like `knowledge` or `chat_main`.

## Planner Tool Types

`PlanStep.tool_needed` is not the same as a concrete LangChain tool name. It is an abstract execution type validated in `src/state.py::PlanStep`.

Allowed standard values and aliases:

| Planner value | Meaning | Main route |
| --- | --- | --- |
| `list_guideline_toc`, `toc` | Browse guideline structure | `knowledge` |
| `read_guideline_chapter`, `read`, `chapter` | Read guideline chapter | `knowledge` |
| `search_treatment_recommendations`, `search` | Retrieve treatment evidence | `knowledge` or `decision`, depending on intent |
| `database_query`, `case_database_query` | Query case database | `case_database` |
| `web_search`, `web` | Online search | `knowledge` or `web_search` |
| `ask_user` | Ask for missing clinical input | `assessment` |
| `calculator` | Generic computation | `tool_executor` |
| `imaging_analysis`, `tumor_detection`, `radiology`, `tumor_screening`, `ct_analysis` | Imaging AI analysis | `rad_agent` |
| `pathology_analysis`, `pathology`, `clam` | Pathology AI analysis | `path_agent` |

Routing is normalized through:

- `src/policies/tool_targets.py::classify_pending_step_target()`
- `src/policies/turn_facts.py::build_turn_facts()`
- `src/policies/routing_policy.py::decide_dynamic()`

## Available Tool Inventory

### Clinical Text Tools

Defined in `src/tools/clinical_tools.py`:

- `PatientHistoryParserTool`
- `PolypDetectionTool`
- `PathologyParserTool`
- `VolumeCTSegmentorTool`
- `RectalMRStagerTool`
- `MolecularGuidelineTool`

Reachability:

- Included in graph-level `list_tools()` and `list_tools_with_web_search()`.
- Available to LLM-bound or sub-agent flows that receive graph-level tools.

### RAG / Guideline Tools

Defined in `src/tools/rag_tools.py`:

- `search_clinical_guidelines`
- `search_treatment_recommendations`
- `search_staging_criteria`
- `search_drug_information`
- `search_by_guideline_source`
- `hybrid_guideline_search`
- `list_guideline_toc`
- `read_guideline_chapter`

Default enhanced set:

- `search_clinical_guidelines`
- `search_treatment_recommendations`
- `search_staging_criteria`
- `search_drug_information`
- `list_guideline_toc`
- `read_guideline_chapter`

Non-default RAG tools:

- `search_by_guideline_source`
- `hybrid_guideline_search`

Reachability:

- Enhanced set is graph-level.
- Full RAG set is only available through `get_all_rag_tools()` or `list_all_tools()`.
- Decision node prioritizes `search_treatment_recommendations`, then `search_clinical_guidelines`.

### Web Search Tools

Defined in `src/tools/web_search_tools.py`:

- `web_search`
- `search_clinical_evidence`
- `search_drug_online`
- `search_guideline_updates`
- `search_latest_research`

Clinical web-search set:

- `web_search`
- `search_clinical_evidence`
- `search_drug_online`
- `search_guideline_updates`

Reachability:

- With `WEB_SEARCH_ENABLED=true`, the clinical web-search set is graph-level.
- `search_latest_research` is only in `get_all_web_search_tools()` and `list_all_tools()`, so it is not normally injected into the graph.

### Case Database Tools

Defined in `src/tools/database_tools.py::ATOMIC_DATABASE_TOOLS`:

- `get_patient_case_info`
- `summarize_patient_existing_info`
- `upsert_patient_info`
- `get_patient_imaging`
- `get_patient_pathology_slides`
- `get_database_statistics`
- `search_cases`
- `list_imaging_folders`
- `get_random_case`
- `perform_comprehensive_tumor_check`

Reachability:

- Directly bound to the LLM in `node_case_database()`.
- Also included in `list_all_tools()`.
- Not included in graph-level `list_tools()` or `list_tools_with_web_search()`.

### Imaging / Tumor Detection Tools

Defined in `src/tools/tumor_screening_tools.py`:

- `tumor_screening_tool`
- `quick_tumor_check`
- `get_tumor_screening_status`
- `perform_comprehensive_tumor_check`

Defined in `src/tools/tumor_localization_tools.py`:

- `tumor_localization_tool`
- `batch_tumor_localization`
- `get_localization_status`

Defined in `src/tools/radiomics_tools.py`:

- `unet_segmentation_tool`
- `radiomics_feature_extraction_tool`
- `lasso_feature_selection_tool`
- `comprehensive_radiomics_analysis`

Reachability:

- Included in graph-level `list_tools()` and `list_tools_with_web_search()`.
- `node_rad_agent()` selects tools by name based on requested analysis mode.
- `perform_comprehensive_tumor_check` is also exposed through database tools.

### Pathology CLAM Tools

Defined in `src/tools/pathology_clam_tools.py`:

- `pathology_slide_classify`
- `quick_pathology_check`
- `get_pathology_clam_status`
- `perform_comprehensive_pathology_analysis`

Reachability:

- Included in graph-level `list_tools()` and `list_tools_with_web_search()`.
- `node_pathology_agent()` selects status, quick, full, or patient-level comprehensive analysis by request shape.

### Basic Tools

Defined in `src/tools/basic_tools.py`:

- `echo`
- `word_count`

Reachability:

- They are not imported into `src/tools/__init__.py`.
- They are effectively not part of the current agent runtime unless imported manually elsewhere.

## Main Usage Patterns

### Knowledge retrieval

`node_knowledge_retrieval()` uses:

- `search_clinical_guidelines` as the primary local RAG tool.
- `list_guideline_toc` and `read_guideline_chapter` for plan-driven active context.
- `web_search` when plan type or local sufficiency requires online search.

Current caveat:

- Plan-driven `search_treatment_recommendations` is described as supported, but the implementation routes generic `"search"` steps through `search_clinical_guidelines`. This is a name-to-tool mismatch.

### Treatment decision

`node_decision()` uses:

- `search_treatment_recommendations` first.
- `search_clinical_guidelines` as fallback.
- `web_search` and `search_clinical_evidence` if available.

Planner intentionally filters guideline-search steps for `treatment_decision`, because decision-level sub-agent retrieval is the single intended RAG entry point for treatment plans.

### Database query

`node_case_database()` binds `ATOMIC_DATABASE_TOOLS` directly:

- This gives strong control over case, imaging, pathology preview, stats, and database-write requests.
- It also means database tool availability is not controlled by the graph-level tool registry.

### Imaging analysis

`node_rad_agent()` is code-routed rather than free-form LLM tool selection:

- Request mode is detected by keywords.
- Patient ID is resolved from state or text.
- The node selects detection, segmentation, radiomics, or comprehensive analysis.

This is more predictable than exposing all image tools for arbitrary model selection.

### Pathology analysis

`node_pathology_agent()` is also code-routed:

- Status query calls `get_pathology_clam_status`.
- Slide path calls quick or full slide analysis.
- Patient ID calls `perform_comprehensive_pathology_analysis`.

### Parallel sub-agents

`node_parallel_subagents()` filters graph-level tools by step type:

- `web` steps get web-like tools.
- `database/case` steps look for names containing database, case, patient, or imaging.
- Other steps get graph-level tools.

Because graph-level tools do not include `ATOMIC_DATABASE_TOOLS`, case-database parallel subtasks may not have the same tool access as `node_case_database()`.

## First-Round Findings

### P1: Registry and reachability are split

The project has at least four effective tool surfaces:

- Graph-level tools from `list_tools()` / `list_tools_with_web_search()`.
- Database-node tools from `ATOMIC_DATABASE_TOOLS`.
- Tool-executor tools from `list_all_tools()`.
- Planner abstract tool types from `PlanStep.tool_needed`.

This makes it hard to answer "what tools can the agent use?" without specifying graph, node, and route.

### P1: Planner names do not always map to concrete tool calls

The main example is `search_treatment_recommendations`:

- It is allowed in `PlanStep.tool_needed`.
- It is documented as usable in plan-driven knowledge retrieval.
- But `node_knowledge_retrieval()` currently falls back to `search_clinical_guidelines` for generic search paths.

### P1: Parallel case-database tasks may be under-tooled

`node_parallel_subagents()` receives graph-level tools, but case database tools are not graph-level by default. This means a plan with `assignee=case_database` inside a parallel group may not receive the same tools as the dedicated database node.

### P2: `calculator` is allowed but no concrete calculator tool is registered

`calculator` routes to `tool_executor`, but `list_all_tools()` does not currently expose a calculator. A planner-created calculator step is likely to fail or degrade.

### P2: `basic_tools.py` is orphaned

`echo` and `word_count` exist but are not part of the main registries.

### P2: `list_database_tools()` is stale

It returns legacy names such as `search_case_database` and `get_case_by_id`, while the actual tools are in `ATOMIC_DATABASE_TOOLS`.

### P2: Import-time behavior should be measured

A direct attempt to import and print all three registries timed out after 20 seconds in this environment. The tools may still work during app startup, but cold-start cost should be measured and reduced if reproducible.

### P3: `search_latest_research` is not graph-level

It exists in the full web-search registry, but the default clinical web-search set excludes it. That may be intentional; if latest research is expected during normal doctor workflow, it should be promoted or explicitly routed.

## Recommended Next Pass

1. Create a single declarative tool manifest that records:
   - concrete tool name
   - category
   - registry membership
   - allowed planner aliases
   - reachable nodes
   - dependencies
   - whether it is safe for patient graph, doctor graph, or tool executor

2. Add tests that assert registry consistency:
   - every `PlanStep.get_valid_tool_types()` value maps to a route or concrete tool
   - every planner prompt tool name exists in `PlanStep.get_valid_tool_types()`
   - every `classify_pending_step_target()` target is reachable in the relevant graph
   - no documented tool is orphaned unless explicitly marked internal

3. Align `node_knowledge_retrieval()` with concrete RAG tool names:
   - call `search_treatment_recommendations` when the plan asks for it
   - call `search_staging_criteria` and `search_drug_information` when requested
   - preserve `search_clinical_guidelines` as the general fallback

4. Decide whether database tools should be:
   - graph-level tools, or
   - node-local only

   Then update `node_parallel_subagents()` accordingly.

5. Either add a real calculator tool or remove `calculator` from allowed planner values and prompts.

6. Replace `list_database_tools()` with names derived from `ATOMIC_DATABASE_TOOLS`.

7. Add a lightweight registry smoke test that imports registries and checks names without loading model weights.

