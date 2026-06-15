# Agent Admin Memory Automation Design

Date: 2026-06-15
Status: Draft for review

## Goal

Upgrade the Agent Admin `记忆` subtask page into an automation-oriented memory workbench.

The page should show two things at the same time:

- what the large-model agent currently retains as session/context memory;
- how the system automatically maintains that memory through collection, summarization, structuring, synchronization, and stale checks.

This is a frontend-first refinement of the existing Agent Admin subtask page. It does not add mutation controls, new autonomous learning jobs, model-weight training, or a durable cross-session global memory store.

## Selected Direction

Use the `自动化流水线型` layout.

This direction was selected because it best communicates the operational nature of memory:

- memory is not just a static list;
- memory has an automated maintenance pipeline;
- each retained item should have a source, retention reason, and status;
- operators need to understand whether memory is available, structured, stale, or waiting for maintenance.

Rejected directions:

- `记忆图谱型`: visually stronger, but it needs reliable relationship edges between facts, events, users, evidence, and decisions. The current frontend state does not consistently provide those edges.
- `审计表格型`: stable and dense, but it feels like a generic table and does not make automation visible enough for an agent backend.

## Current Context

The current memory page already shows:

- session/context memory boundary copy;
- patient and doctor session ids;
- typed memory rows from `summary_memory`, `immutable_info`, `dynamic_info`, and `anchor_events`;
- memory facts from structured summary values;
- context maintenance status and messages.

The product gap is visual and operational clarity. The page still reads mostly as a typed list, so it does not clearly express:

- the automated memory lifecycle;
- whether memory extraction is complete, partial, stale, or unstructured;
- how patient and doctor memory differ;
- why an item is retained;
- which state fields drive each visible result.

## Scope

In scope:

- Replace the memory subtask body with a purpose-built automation workbench.
- Keep the existing Agent Admin shell, company-red theme, top navigation, and subtask rail.
- Keep the page read-only.
- Use existing `SessionState` fields and local derivation helpers.
- Show patient and doctor memory together by default.
- Make unstructured or missing fields visible instead of hiding them.
- Add tests for memory metrics, lifecycle rows, fact rows, and empty/unstructured states.

Out of scope:

- Editing, deleting, merging, pinning, or approving memory.
- Starting real background maintenance jobs from the UI.
- Adding backend write APIs.
- Showing hidden reasoning or raw private model traces.
- Persisting a cross-session global knowledge base.
- Building a relationship graph view in phase one.

## Page Structure

The memory page keeps the existing task page header:

- title: `上下文记忆`;
- description: `summary memory / 永久事实 / dynamic facts / anchor events / context maintenance`;
- boundary message: `这是会话上下文记忆，不是模型权重训练`.

Below the header, the page is divided into four regions.

## Region 1: Memory Health Metrics

Render a compact KPI strip at the top of the memory page.

Metrics:

- `摘要状态`
  - `已生成`: at least one watched session has non-empty `contextState.summary_memory`;
  - `待生成`: neither session has summary memory;
  - `待刷新`: summary memory exists but the session snapshot has advanced past the visible cursor when cursor data is available.
- `永久事实`
  - total normalized count from `structured_summary.immutable_info` across patient and doctor sessions.
- `动态事实`
  - total normalized count from `structured_summary.dynamic_info`.
- `锚点事件`
  - total normalized count from `structured_summary.anchor_events`.
- `维护状态`
  - derived from `contextMaintenance.status`;
  - if patient and doctor differ, show the higher-attention state in this order: `failed`, `running`, `completed`, `idle`.

Metric tone:

- red for active/available memory and completed automation;
- warning for failed, stale, or partially structured memory;
- neutral for idle or unavailable memory.

## Region 2: Memory Layer Navigation

Render a left-side layer panel that explains the memory model.

Layers:

- `摘要记忆`
  - source: `contextState.summary_memory`;
  - retention reason: compressed session context.
- `永久事实`
  - source: `structured_summary.immutable_info`;
  - retention reason: stable patient facts or stable clinical context.
- `动态事实`
  - source: `structured_summary.dynamic_info`;
  - retention reason: changing symptoms, preferences, recent decisions, or temporary state.
- `锚点事件`
  - source: `structured_summary.anchor_events`;
  - retention reason: clinically important milestones.
- `维护日志`
  - source: `contextMaintenance`;
  - retention reason: automation status and maintenance errors.

Each layer row shows:

- label;
- patient count;
- doctor count;
- availability state: `active`, `empty`, `unstructured`, or `stale`;
- a short source key such as `summary_memory` or `immutable_info`.

This first version does not need interactive filtering. The panel can be a visual index that mirrors the rows below.

## Region 3: Automation Lifecycle Pipeline

Render the main center panel as a lifecycle timeline.

Pipeline stages:

1. `收集`
   - indicates whether messages or session state exist.
   - input fields: `messages`, `sessionId`, `snapshotVersion`.
2. `摘要`
   - indicates whether `summary_memory` exists.
   - input fields: `contextState.summary_memory`, `summary_memory_cursor`.
3. `结构化`
   - indicates whether structured facts exist.
   - input fields: `immutable_info`, `dynamic_info`, `anchor_events`.
4. `同步`
   - indicates whether the visible memory belongs to a known snapshot.
   - input fields: `snapshotVersion`, `summary_memory_cursor`.
5. `过期检查`
   - indicates whether maintenance is idle, running, completed, or failed.
   - input fields: `contextMaintenance.status`, `contextMaintenance.message`, `contextMaintenance.error`.

Each stage row shows:

- stage name;
- status icon;
- status label: `ready`, `active`, `waiting`, `warning`, or `failed`;
- short explanation;
- patient state;
- doctor state.

The lifecycle is derived only from available frontend state. It must not claim a real scheduler or background job is running unless `contextMaintenance.status === "running"`.

## Region 4: Current Memory Visualization

Render a bottom table or dense row list that shows actual retained memory items.

Columns:

- `记忆内容`
  - summary text or normalized fact text.
- `类型`
  - `摘要记忆`, `永久事实`, `动态事实`, `锚点事件`, or `维护日志`.
- `来源`
  - `患者`, `医生`, or both if the same normalized text appears in both sessions.
- `保留原因`
  - fixed by type:
    - summary memory: `压缩会话上下文`;
    - permanent fact: `稳定事实`;
    - dynamic fact: `近期变化`;
    - anchor event: `关键事件`;
    - maintenance log: `自动维护状态`.
- `状态`
  - `active`: non-empty structured value;
  - `empty`: source field exists but has no normalized values;
  - `unstructured`: source exists but cannot be normalized into readable values;
  - `stale`: cursor and snapshot suggest the summary may lag behind;
  - `failed`: maintenance error is present.

Rows should include both patient and doctor memory. Empty states should be explicit, for example:

- `暂无摘要记忆`;
- `永久事实未结构化`;
- `动态事实未结构化`;
- `暂无锚点事件`;
- `维护日志为空`.

## Data Mapping

Use existing frontend state first.

Inputs:

- `patient.contextState.summary_memory`;
- `doctor.contextState.summary_memory`;
- `patient.contextState.structured_summary`;
- `doctor.contextState.structured_summary`;
- `structured_summary.immutable_info`;
- `structured_summary.dynamic_info`;
- `structured_summary.anchor_events`;
- `summary_memory_cursor`;
- `contextMaintenance.status`;
- `contextMaintenance.message`;
- `contextMaintenance.error`;
- `sessionId`;
- `snapshotVersion`;
- `messages`.

Normalization rules:

- Accept arrays, dictionaries, primitive strings, booleans, numbers, and object records.
- For dictionaries, preserve the key in the displayed value: `key: value`.
- For object records, prefer readable fields in this order: `title`, `summary`, `event`, `description`, `text`.
- If no readable field exists, fall back to JSON stringification.
- Do not hide unsupported shapes; show an `unstructured` row when the source is present but unreadable.

## Component Changes

Expected frontend units:

- `buildMemoryAutomationSummary()`
  - derives metric cards and the combined maintenance state.
- `buildMemoryLayerRows()`
  - derives layer navigation rows and patient/doctor counts.
- `buildMemoryLifecycleRows()`
  - derives the automation pipeline rows.
- `buildMemoryVisualizationRows()`
  - derives the bottom memory table rows.
- `MemoryPage`
  - composes the metrics, layer panel, lifecycle pipeline, source inspector, and visualization rows.

Existing helpers such as `normalizeMemoryFactValues`, `buildMemoryRows`, and `buildMemoryFactRows` can either be reused or replaced by the new helpers, as long as existing behavior remains covered by tests.

## Visual Requirements

- Use the existing `agent-admin` company-red theme.
- Keep the workbench dense and operational, not marketing-like.
- Use icons for state and lifecycle stages.
- Avoid cards nested inside cards.
- Use stable responsive dimensions so lifecycle rows and table rows do not shift when text changes.
- On desktop, use a three-column workbench:
  - left: memory layer navigation;
  - center: automation lifecycle;
  - right: source and maintenance inspector.
- On mobile, stack the regions:
  - metrics;
  - lifecycle;
  - memory layers;
  - visualization rows;
  - inspector.
- Do not introduce decorative gradient orbs, bokeh, or unrelated imagery.

## Read-Only Boundary

The memory page must remain observational.

Do not render enabled controls for:

- delete memory;
- edit memory;
- merge memory;
- pin memory;
- approve memory;
- run maintenance now.

If future controls are visually needed, render them disabled with explicit copy explaining that phase one is read-only.

## Empty And Error States

The page should degrade visibly:

- no session id: show `会话未创建`;
- no summary: show `暂无摘要记忆`;
- no structured summary: show `结构化摘要未生成`;
- malformed structured summary: show `结构化字段不可读`;
- maintenance failed: show the error message when available;
- maintenance running: show a running state without claiming completion.

These states should appear at panel level, not as a blank page.

## Testing

Add or update frontend tests for:

- memory page renders the boundary copy;
- metrics count patient and doctor structured values;
- dictionary-shaped `immutable_info`, `dynamic_info`, and `anchor_events` are counted and displayed;
- lifecycle stages render `收集`, `摘要`, `结构化`, `同步`, and `过期检查`;
- maintenance `failed` state is visible;
- missing memory renders explicit empty-state rows;
- clicking the `记忆` subtask still renders only the memory page and does not leave other subtask content visible.

Recommended command:

```bash
npm --prefix frontend run test -- --run src/features/agent-admin/agent-admin-view.test.tsx src/features/agent-admin/agent-admin-components.test.tsx
```

Before completion, also run the frontend build if production CSS or TypeScript surfaces changed:

```bash
npm --prefix frontend run build
```

## Acceptance Criteria

- The memory subtask visually communicates an automated memory lifecycle, not only a fact list.
- Operators can see current memory content, memory type, source side, retention reason, and status.
- Operators can see whether patient and doctor memory differ.
- Missing, stale, failed, or unstructured memory states are explicit.
- The page remains read-only.
- The implementation uses existing state fields and does not require a new backend API.
- Focused agent-admin tests pass.
- The page remains responsive without horizontal overflow on desktop and mobile.

## Future Extensions

After phase one, the page can be extended with:

- selectable memory rows with a right-side detail drawer;
- local search and type filtering;
- relationship graph view once stable memory edges are available;
- maintenance run history if the backend exposes durable events;
- manual approval workflow for memory edits if product policy allows writes;
- cross-session durable knowledge store, explicitly separated from session/context memory.
