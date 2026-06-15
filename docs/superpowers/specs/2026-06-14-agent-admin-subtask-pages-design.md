# Agent Admin Subtask Pages Design

Date: 2026-06-14
Status: Draft for review

## Goal

Expand the Agent Admin console subtask pages from similar single-panel cards into purpose-built read-only workspaces. Each subtask should have a distinct layout, clear operational purpose, and enough visible structure for an operator to inspect the large-model agent's state, memory, rules, tools, learning readiness, trace, and evidence.

This spec refines the first-phase Agent Admin design. It does not add write controls, rule editing, tool toggles, or autonomous learning execution.

## Current Context

The current Agent Admin surface already has the right shell direction:

- the workspace surface switcher separates Patient, Doctor, and Agent Admin;
- the Agent Admin page uses the company-red theme;
- the backend header uses the light company logo variant;
- the subtask rail has nine tasks;
- selecting a subtask replaces the active page instead of changing only a small inspector.

The remaining product issue is density and differentiation. Most subtasks currently render a simple heading plus a small list, so they feel repetitive and do not yet express the function of a real agent control plane.

## Design Direction

Use a shared shell with purpose-built page bodies.

Recommended approach:

- keep one persistent Agent Admin shell: top nav, context strip, and left subtask rail;
- render only one active subtask page at a time;
- give every subtask a specific workbench layout, not a generic card stack;
- reuse common components for metrics, status chips, tables, timelines, and empty states;
- keep all phase-one actions read-only or disabled with explicit reasons.

Rejected alternatives:

- A single dashboard with collapsible sections would keep the page simple but would make the subtask rail redundant.
- Building all admin APIs before improving the frontend would slow the visual iteration and leave the current UI feeling incomplete.
- Rendering all subtask sections on one page and scrolling to anchors would conflict with the user's request that other pages disappear after a subtask click.

## Global Page Shell

The page shell remains consistent across all subtasks:

- Top nav:
  - company logo using the light logo variant;
  - brand label `智能体后台`;
  - quick tabs for high-priority pages such as `巡检总览`, `运行链路`, `学习准备`, and `只读边界`;
  - workspace surface switcher for Patient, Doctor, and Agent Admin.
- Context strip:
  - current watched side: Patient, Doctor, or Both;
  - patient session id;
  - doctor session id;
  - runtime mode;
  - active run id;
  - last snapshot version.
- Main layout:
  - left subtask rail;
  - right active task page;
  - inactive task pages are not rendered in the main content.

Each task page uses this internal structure:

- Page header: task name, operational description, status chip, primary icon.
- Scoped summary: metrics or readiness indicators relevant to that task.
- Primary workbench: the main table, timeline, comparison view, or pipeline view.
- Secondary detail region: inspector, detail drawer, side panel, or bottom log when useful.
- Empty, loading, error, and unavailable states at panel level.

## Shared Data Boundaries

Phase one should prefer existing frontend state first:

- patient and doctor `SessionState`;
- `sessionId`;
- `snapshotVersion`;
- `runtime`;
- `activeRunId`;
- `statusNode`;
- `messages`;
- `cards`;
- `plan`;
- `references`;
- `critic`;
- `contextState`;
- `contextMaintenance`;
- `lastError`.

Static manifests can back pages that do not yet have full APIs:

- tool inventory manifest;
- rule catalog manifest;
- learning readiness checklist.

Future admin APIs can replace or enrich these manifests without changing page intent:

- `/api/admin/agent/overview`;
- `/api/admin/agent/tools`;
- `/api/admin/agent/rules`;
- `/api/admin/agent/sessions/{session_id}/state`.

The admin surface must never expose API keys, hidden reasoning, raw binary assets, or mutable rule/tool controls in phase one.

## Subtask 1: 巡检总览

Purpose:

Give operators a command-center view of the agent's health across patient and doctor sessions.

Layout:

- Header:
  - title `全局健康`;
  - current observation scope;
  - `只读观测` status.
- KPI strip:
  - patient session status;
  - doctor session status;
  - active run state;
  - memory summary status;
  - evidence count;
  - tool/rule manifest status.
- Main workbench:
  - left: `运行健康时间线`, showing recent trace, plan, context maintenance, reference append, and completion events;
  - center: `当前图状态`, showing active graph node, planned next node, and simplified DAG step list;
  - right: `风险队列`, showing stale snapshot, missing session, failed recovery, missing web search config, missing memory summary, or admin API authorization issues.
- Bottom band:
  - `最近变化`: latest memory update, latest reference, latest plan update, latest error;
  - quick jump buttons to `会话`, `记忆`, `Trace`, and `证据`.

Functions:

- Switch watched scope between Patient, Doctor, and Both.
- Click a risk item to open the relevant subtask page.
- Show panel-level unavailable states when a session has not been created.
- Keep all health actions read-only.

Data:

- patient and doctor `SessionState`;
- local trace event cache when available;
- future overview endpoint.

## Subtask 2: 会话

Purpose:

Inspect and compare the patient and doctor sessions without starting a new graph run.

Layout:

- Header:
  - title `会话观测`;
  - watched scope selector;
  - sync freshness indicator.
- Primary split view:
  - left card: patient session identity and runtime;
  - right card: doctor session identity and runtime;
  - center comparison column: status differences between the two sessions.
- Session card fields:
  - session id;
  - snapshot version;
  - runtime mode;
  - runner mode;
  - active run id;
  - current node;
  - stream state;
  - abort state;
  - last user message timestamp when available;
  - last assistant response length;
  - cards count;
  - references count;
  - last error.
- Bottom table:
  - recent session events;
  - recovery state;
  - stale snapshot warnings.

Functions:

- Toggle focus between patient, doctor, and compare mode.
- Highlight mismatched snapshot versions.
- Show missing-session empty state separately for patient and doctor.
- Provide disabled future controls for `打开会话详情` and `刷新快照`, labelled as future admin API work.

Data:

- patient and doctor `SessionState`;
- future session-state endpoint.

## Subtask 3: 记忆

Purpose:

Show what the agent retained in session/context memory, why it appears there, and whether it is summary, permanent, dynamic, or anchor-event information.

Layout:

- Header:
  - title `上下文记忆`;
  - scope selector: Patient, Doctor, Compare;
  - clear boundary copy: `这是会话上下文记忆，不是模型权重训练`.
- Top metrics:
  - summary memory status;
  - permanent fact count;
  - dynamic fact count;
  - anchor event count;
  - maintenance status.
- Workbench:
  - left segmented list:
    - `摘要记忆`;
    - `永久事实`;
    - `动态事实`;
    - `锚点事件`;
    - `维护日志`;
  - center table:
    - fact or summary text;
    - memory type;
    - source session;
    - retention reason;
    - confidence or availability;
    - recency;
    - status: active, stale, unavailable, unstructured;
  - right inspector:
    - selected memory item detail;
    - source evidence or message reference when available;
    - raw structured summary excerpt;
    - context maintenance note.

Functions:

- Search memory text locally.
- Filter by memory type.
- Compare patient and doctor memory side by side.
- Mark unstructured fields clearly instead of hiding them.
- Keep delete, pin, merge, and edit controls absent in phase one.

Data:

- `contextState.summary_memory`;
- `contextState.structured_summary.immutable_info`;
- `contextState.structured_summary.dynamic_info`;
- `contextState.structured_summary.anchor_events`;
- `summary_memory_cursor`;
- `contextMaintenance`.

## Subtask 4: 规则

Purpose:

Make permanent context rules, routing policies, safety policies, memory rules, and prompt-owned behavior visible as a read-only catalog.

Layout:

- Header:
  - title `永久上下文规则`;
  - catalog freshness state;
  - read-only badge.
- Left tree:
  - `路由规则`;
  - `安全规则`;
  - `评估规则`;
  - `记忆规则`;
  - `上下文维护`;
  - `Planner Prompt`;
  - `Knowledge Prompt`;
  - `Intent Prompt`.
- Center rule list:
  - rule id;
  - title;
  - group;
  - type: prompt, policy, memory, safety, tool-routing;
  - owner module;
  - risk level;
  - editable: false;
  - last known source path.
- Right inspector:
  - rule description;
  - trigger condition;
  - source file path;
  - short prompt or policy excerpt when safe;
  - related graph nodes;
  - future versioning metadata shown as disabled read-only information, not an action.

Functions:

- Search by rule id, title, module, or source path.
- Filter by group and rule type.
- Click a rule to open inspector.
- Show prompt excerpts only when they are intentionally surfaced in metadata.
- Do not expose editing, rollback, or save controls in phase one.

Data:

- static rule catalog manifest;
- future `/api/admin/agent/rules` endpoint.

## Subtask 5: 工具

Purpose:

Show which tools exist, where they are reachable, what dependencies they require, and whether they are safe for patient, doctor, or admin contexts.

Layout:

- Header:
  - title `工具可用性`;
  - manifest status;
  - graph construction safety note.
- Top metric strip:
  - graph-level tools;
  - executor tools;
  - database tools;
  - web-search tools;
  - unavailable tools;
  - tools requiring config.
- Workbench:
  - left filters:
    - category;
    - registry;
    - reachable graph node;
    - safety scope;
    - dependency status;
  - center inventory table:
    - tool name;
    - category;
    - registries;
    - reachable nodes;
    - patient safe;
    - doctor safe;
    - admin safe;
    - dependencies;
    - status;
    - notes;
  - right dependency inspector:
    - selected tool detail;
    - dependency checks;
    - expected input/output shape summary;
    - known gaps.
- Bottom reachability map:
  - graph node to tool availability matrix.

Functions:

- Search by tool name.
- Filter by `Graph-level`, `Executor`, `Database`, and `Web search`.
- Mark `search_latest_research` as available for research search but not automatically graph-level unless current runtime confirms it.
- Avoid constructing heavyweight tools just to render inventory.
- Keep enable/disable switches absent in phase one.

Data:

- static tool manifest;
- future `/api/admin/agent/tools` endpoint;
- current `WEB_SEARCH_ENABLED` when exposed.

## Subtask 6: 学习

Purpose:

Prepare the operator view for daily autonomous paper collection and learning while making it clear that phase one does not run the scheduler or ingest papers automatically.

Layout:

- Header:
  - title `自主学习准备`;
  - status: disabled, missing config, or ready preview;
  - phase boundary note.
- Readiness strip:
  - scheduler configured;
  - source connectors configured;
  - topic queue available;
  - approval mode configured;
  - target store configured;
  - last run available.
- Main pipeline:
  - `发现论文`;
  - `去重`;
  - `打分`;
  - `摘要`;
  - `人工审核`;
  - `写入知识库`;
  - `生成学习报告`.
- Left panel:
  - source readiness: PubMed, arXiv, Crossref, local database, web search;
  - config requirement for each source.
- Center table:
  - topic queue;
  - disease/domain;
  - keywords;
  - priority;
  - schedule window;
  - approval mode;
  - target store.
- Right panel:
  - disabled controls preview:
    - run now;
    - pause schedule;
    - add topic;
    - approve artifact;
  - each disabled control has a reason.
- Bottom panel:
  - future learned artifacts list:
    - paper title;
    - summary status;
    - evidence quality;
    - reviewer;
    - target collection.

Functions:

- Show why autonomous learning is not active.
- Preview the operational workflow before backend scheduling exists.
- Surface `search_latest_research` as a candidate capability.
- Keep all schedule and ingestion controls disabled in phase one.

Data:

- static readiness checklist;
- `WEB_SEARCH_ENABLED` when available;
- future learning job table and artifact store.

## Subtask 7: Trace

Purpose:

Debug the agent's execution path, trace events, timings, retrieval calls, tool calls, and failure points.

Layout:

- Header:
  - title `执行时间线`;
  - active run id;
  - trace id when available;
  - status: idle, running, done, failed, aborted.
- Top controls:
  - event type filter;
  - node filter;
  - error-only toggle;
  - disabled copy/export controls until trace payload is durable.
- Workbench:
  - left vertical timeline:
    - `trace.start`;
    - `status.node`;
    - `plan.update`;
    - `references.append`;
    - `critic.verdict`;
    - `context.maintenance`;
    - `trace.step`;
    - `trace.summary`;
    - `done`;
  - center DAG path:
    - current node;
    - previous nodes;
    - skipped nodes;
    - retry or abort marker;
  - right latency panel:
    - node timings;
    - stage timings;
    - retrieval timings;
    - tool timings;
    - total time.
- Bottom event table:
  - timestamp;
  - event type;
  - graph node;
  - duration;
  - payload summary;
  - error summary;
  - related evidence count.

Functions:

- Highlight the active or failed node.
- Let an operator filter down to only errors, retrieval, or tool calls.
- Show missing timing fields as unavailable, not zero.
- Avoid displaying hidden chain-of-thought.

Data:

- frontend SSE event cache;
- `activeRunId`;
- `statusNode`;
- `plan`;
- `references`;
- future `node_timings`, `stage_timings`, `retrieval_timings`, and `step_history`.

## Subtask 8: 证据

Purpose:

Inspect references, retrieved evidence, RAG traces, source confidence, and where evidence was used in the generated response.

Layout:

- Header:
  - title `证据池`;
  - watched session;
  - reference count;
  - citation coverage status.
- Metric strip:
  - references;
  - retrieved chunks;
  - high-confidence items;
  - items missing source metadata;
  - citations used in answer;
  - stale evidence warnings.
- Workbench:
  - left filters:
    - source type;
    - retrieval profile;
    - confidence band;
    - used in answer;
    - session scope;
  - center evidence table:
    - title;
    - source;
    - page or section;
    - snippet;
    - retrieval profile;
    - confidence;
    - used in response;
    - linked trace event;
  - right preview:
    - selected evidence detail;
    - longer snippet;
    - source path or URL;
    - citation/report status;
    - related memory item when available.
- Bottom RAG pipeline:
  - query;
  - retrieval;
  - rerank;
  - citation assembly;
  - final response usage.

Functions:

- Search evidence title and snippet.
- Filter by confidence or source type.
- Mark fallback rows clearly when only references are available and full RAG trace is not exposed.
- Keep source editing and manual citation changes absent in phase one.

Data:

- `references`;
- future `retrieved_evidence`;
- future `rag_trace`;
- future citation report.

## Subtask 9: 设置只读

Purpose:

Make the admin surface's security, permissions, feature flags, and mutation boundary explicit.

Layout:

- Header:
  - title `只读边界`;
  - authorization status;
  - mutation-disabled status.
- Status cards:
  - auth mode;
  - admin token requirement;
  - current access state;
  - feature flags;
  - write actions disabled;
  - audit logging availability.
- Permissions matrix:
  - view sessions;
  - view memory;
  - view rule catalog;
  - view tool manifest;
  - view learning readiness;
  - edit rules;
  - enable tools;
  - run learning jobs;
  - write memory;
  - delete evidence.
- Boundary panel:
  - no rule writes;
  - no tool state writes;
  - no memory writes;
  - no scheduler execution;
  - no graph scene creation;
  - no hidden reasoning exposure.
- Future controls preview:
  - rule versioning;
  - tool policy toggles;
  - learning schedule;
  - audit export;
  - all controls disabled with reasons.

Functions:

- Explain why controls are unavailable.
- Surface admin-token errors as a page-level warning.
- Keep security and feature flags visible without enabling mutations.

Data:

- runtime auth mode when exposed;
- static feature boundary manifest;
- future audit log endpoint.

## Shared Components

Implementing this design should avoid nine unrelated page implementations. Use shared primitives with task-specific composition:

- `AgentAdminTaskPageShell`: header, status chip, icon, and page-level states.
- `AgentAdminMetricStrip`: responsive metric row.
- `AgentAdminStatusChip`: consistent status color and label.
- `AgentAdminSplitWorkbench`: left filters, center content, right inspector.
- `AgentAdminSessionCard`: session identity and runtime summary.
- `AgentAdminComparisonMatrix`: patient/doctor field comparison.
- `AgentAdminFactTable`: memory facts and summaries.
- `AgentAdminRuleCatalog`: rule tree, list, and inspector.
- `AgentAdminToolInventory`: filters, table, and dependency inspector.
- `AgentAdminLearningPipeline`: scheduler/readiness pipeline.
- `AgentAdminTraceTimeline`: event timeline and latency view.
- `AgentAdminEvidenceTable`: evidence list and preview.
- `AgentAdminReadOnlyNotice`: phase-one mutation boundary.
- `AgentAdminEmptyState` and `AgentAdminPanelError`: per-panel fallback states.

## Visual Requirements

- Use company red as an accent, not as a full-page wash.
- Keep the page operational and dense: more table, timeline, matrix, and pipeline structures; fewer generic cards.
- Avoid nested cards.
- Avoid using one large centered panel for each page.
- Preserve strong contrast for red text and borders.
- Make page bodies visually distinct:
  - overview uses health timeline and risk queue;
  - sessions uses comparison layout;
  - memory uses segmented memory table plus inspector;
  - rules uses catalog tree plus rule inspector;
  - tools uses inventory table plus reachability matrix;
  - learning uses pipeline and readiness controls;
  - trace uses timeline and latency bars;
  - evidence uses table and source preview;
  - read-only uses permission matrix.
- On mobile:
  - rail can stack or become a compact vertical list above content;
  - workbench columns stack;
  - tables use compact rows or horizontal-safe wrappers;
  - no text overlaps controls.

## Interaction Rules

- Clicking a subtask rail item sets the active task page.
- Only the active task page should be visible in the main content.
- Top nav quick tabs should map to the same active task state as the rail.
- Switching to Agent Admin must not create a third graph scene.
- Patient and Doctor session state remains owned by `WorkspacePage`.
- Page-level filters are local UI state and reset only when the task changes unless preserving them improves usability.
- Disabled future controls must include a visible reason.

## Testing Requirements

Frontend unit tests:

- Agent Admin uses `brandLogoVariant="light"`.
- Rail click switches `data-task-id`.
- After switching to a subtask, previous page-specific content is absent from the main content.
- Each subtask page renders a unique landmark or title:
  - `全局健康`;
  - `会话观测`;
  - `上下文记忆`;
  - `永久上下文规则`;
  - `工具可用性`;
  - `自主学习准备`;
  - `执行时间线`;
  - `证据池`;
  - `只读边界`.
- Top nav quick tabs and rail stay synchronized.
- Missing patient or doctor session shows empty state without throwing.

Visual checks:

- desktop view has no overlapping content;
- mobile view has no horizontal overflow from rail, tables, or long session ids;
- red theme remains distinct from patient and doctor themes;
- company logo is visible and not distorted.

Future backend tests:

- tools endpoint returns static inventory without constructing heavyweight tools;
- rules endpoint returns read-only catalog;
- session state endpoint sanitizes sensitive fields;
- `/api/admin/*` is protected by admin bearer token when bearer auth is enabled.

## Acceptance Criteria

This subtask-page design is satisfied when:

1. Every rail item opens a visually distinct page body.
2. The main content contains only the selected subtask page.
3. Each page has a clear operational purpose and at least one non-generic primary view: timeline, comparison, table, matrix, pipeline, or inspector.
4. Read-only boundaries are visible wherever a future control might otherwise imply mutation.
5. Existing patient and doctor workflows are unaffected.
6. The design can be implemented incrementally using shared Agent Admin components without rewriting the workspace shell.

## Implementation Boundary

This spec intentionally stops before implementation. After review, the next step is an implementation plan that splits work into focused subtasks:

- shared page shell and component primitives;
- page-specific layouts;
- navigation synchronization tests;
- responsive and visual QA;
- optional static manifests for richer rules/tools/learning data.
