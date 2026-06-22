# Agent Tool Manifest Admin API Design

Date: 2026-06-22
Status: Approved for implementation planning

## Goal

Create a single observable tool manifest for LangG and expose it through a read-only admin API so the agent-admin console can show the current backend tool surface instead of a stale static frontend inventory.

This design covers the first runtime-facing control-plane slice only:

- backend manifest metadata
- read-only `GET /api/admin/tools`
- existing Bearer admin-token protection
- frontend agent-admin dynamic display
- contract tests for manifest coverage and API safety

It does not implement tool enable/disable controls, tool execution, health probes, new authentication channels, or changes to the doctor/patient graph permissions.

## Current Context

The project currently has several effective tool surfaces:

- graph-level registries in `src/tools/__init__.py`
- database-node tools in `src/tools/database_tools.py::ATOMIC_DATABASE_TOOLS`
- executor-only tools from `list_all_tools()`
- planner-level aliases in `src/state.py::PlanStep`
- agent-admin static frontend inventory in `frontend/src/features/agent-admin/agent-admin-model.ts`

This split makes runtime tool visibility hard to trust. Existing repair work already improved RAG dispatch and parallel database subagent access, but the frontend still does not read a backend source of truth.

Existing authentication infrastructure should be reused. `BearerAuthMiddleware` already accepts `API_BEARER_TOKEN` and `API_ADMIN_BEARER_TOKEN`, and `load_runtime_settings()` falls back from `API_ADMIN_BEARER_TOKEN` to `API_BEARER_TOKEN` when no separate admin token is configured. Therefore, the new endpoint should use the existing `Authorization: Bearer <token>` path and should not introduce an `X-Admin-Token` header.

## Architecture

Add a lightweight tool control-plane module:

```text
src/tools/manifest.py
```

This module defines metadata only. It must not instantiate LangChain tools, load model weights, touch the network, or run health checks. Existing runtime registry functions remain in place:

- `list_tools()`
- `list_tools_with_web_search()`
- `list_all_tools()`
- `ATOMIC_DATABASE_TOOLS`

The manifest becomes the single source for audit, admin display, and manifest contract tests. Runtime registries can be migrated toward the manifest later, but that is outside this first slice.

Add a backend admin route:

```text
GET /api/admin/tools
```

The route returns a safe projection of the manifest plus runtime availability derived from current settings. It does not execute tools.

Add the endpoint to `backend/app.py::_requires_admin_token()`:

```python
if method == "GET" and path == "/api/admin/tools":
    return True
```

With the existing fallback behavior, local development and demos keep working when only `API_BEARER_TOKEN` is set. Production deployments can separate admin access by setting `API_ADMIN_BEARER_TOKEN`.

## Manifest Contract

Internal metadata should use a small immutable structure. A dataclass is sufficient.

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: ToolCategory
    registries: tuple[ToolRegistry, ...]
    route_targets: tuple[RouteTarget, ...]
    graph_scope: GraphScope
    planner_aliases: tuple[str, ...] = ()
    requires_web: bool = False
    patient_safe: bool = False
    heavy_dependency: bool = False
    state: ToolState = "available"
    notes: str = ""
```

Suggested allowed values:

- `category`: `clinical`, `rag`, `web`, `database`, `imaging`, `pathology`, `tumor`, `utility`, `formatting`
- `registries`: `graph`, `graph_web`, `executor`, `database_node`, `optional`
- `route_targets`: `knowledge`, `case_database`, `rad_agent`, `path_agent`, `web_search`, `tool_executor`, `decision`
- `graph_scope`: `doctor`, `patient`, `both`, `node_local`, `executor_only`
- `state`: `available`, `candidate`, `internal`, `disabled`

The first manifest should cover:

- clinical tools
- enhanced RAG tools
- clinical web-search tools
- database node tools
- imaging, tumor, radiomics, and pathology tools
- executor-only utility tools such as `echo` and `word_count`, if retained
- candidate tools such as `search_latest_research`, if not injected into graph-level tools

The manifest should avoid internal implementation fields in the first version. Do not include factory references, module paths, model paths, local file paths, API keys, tokens, or provider credentials.

## Admin API Response

The API returns a safe projection:

```json
{
  "tools": [
    {
      "name": "search_treatment_recommendations",
      "category": "rag",
      "registries": ["graph", "graph_web"],
      "route_targets": ["knowledge", "decision"],
      "graph_scope": "both",
      "planner_aliases": ["search_treatment_recommendations"],
      "requires_web": false,
      "available": true,
      "state": "available"
    }
  ],
  "groups": [
    {
      "category": "rag",
      "count": 6,
      "available_count": 6
    }
  ],
  "runtime": {
    "web_search_enabled": true,
    "auth": "admin",
    "source": "src.tools.manifest"
  }
}
```

Runtime availability is intentionally shallow:

- `state == "disabled"` means `available=false`
- `requires_web == true` and web search disabled means `available=false`
- otherwise `available=true`

This endpoint reports declared/configuration-level reachability. It does not prove that an external API, model file, GPU, or data source is healthy.

## Frontend Integration

Add frontend API types:

- `AdminToolManifestResponse`
- `AdminToolItem`
- `AdminToolGroup`

Add `getAdminTools()` to `frontend/src/app/api/client.ts`. It should call `/api/admin/tools` with the existing default headers from `createApiClient()`. No new auth header or token path should be added.

The agent-admin tools page should use API data as the primary source:

- loading state: show that the runtime manifest is being read
- success state: render `tools` and `groups` from the API
- error state: show `runtime manifest unavailable` with status details, then use a clearly labeled fallback inventory if needed

The current static `TOOL_INVENTORY` should be renamed or treated as fallback-only data. It must not be presented as runtime truth when the API fails.

Recommended UI groupings:

- total tools
- available tools
- web-limited tools
- candidate or disabled tools
- inventory rows by `name`, `category`, `state`, `available`, and `registries`
- reachability summaries by `route_targets` and `graph_scope`

Production caveat: if `API_ADMIN_BEARER_TOKEN` is separated from `API_BEARER_TOKEN`, the browser must have an admin-capable token to call this endpoint. The first version does not add admin login or a server-side proxy for that token.

## Backend Tests

Extend the existing authentication test pattern rather than creating a new auth harness. `tests/backend/test_auth_security.py` already builds an isolated FastAPI app with `BearerAuthMiddleware`; the helper can add a simple `GET /api/admin/tools` route and include that path in the existing admin endpoint parameterized tests.

Expected auth cases:

- distinct admin token: user token gets `403`
- distinct admin token: admin token gets success
- no separate admin token: user token succeeds through fallback
- invalid or missing Bearer token still gets `401`

Add manifest contract tests, likely in:

```text
tests/backend/test_agent_tool_manifest_contract.py
```

These tests complement the existing runtime registry contract tests. Existing registry tests validate that planner aliases and routes can reach expected runtime tools. New manifest tests validate that the manifest declares and covers all known tool surfaces without stale or unknown entries.

Minimum manifest checks:

- importing `src.tools.manifest` does not instantiate heavy tools
- tool names are unique
- enum-like fields use allowed values
- every accepted `PlanStep.get_valid_tool_types()` value is covered by a manifest `planner_aliases` entry or an explicit exception
- database coverage is bidirectional:
  - every `ATOMIC_DATABASE_TOOLS` name appears in the manifest with `database_node`
  - every manifest item with `database_node` exists in `ATOMIC_DATABASE_TOOLS`
- graph, graph-web, executor, and candidate memberships explain current registry differences
- sensitive fields are not present in the API response
- web-required tools become unavailable when web search is disabled through test settings

The bidirectional database check is required to prevent both missing real database tools and stale manifest declarations.

## Frontend Tests

Update frontend tests around the dynamic source:

- `frontend/src/app/api/client.test.ts`
  - `getAdminTools()` calls `/api/admin/tools`
  - default `Authorization` headers are preserved
- `agent-admin-view.test.tsx` or a focused tools-page test
  - successful API manifest renders as the inventory table
  - failed API call shows fallback as fallback, not runtime data
  - old hard-coded names such as `query_case_database` and `get_patient_registry` are removed from runtime assertions unless intentionally present in fallback fixtures

## Migration Plan

1. Add `src/tools/manifest.py` and manifest tests.
2. Add admin route and register it in the FastAPI app.
3. Mark `GET /api/admin/tools` as admin-protected via `_requires_admin_token()`.
4. Add API response types and `getAdminTools()` on the frontend.
5. Switch agent-admin tools page to API-first rendering with fallback labeling.
6. Update existing tests that assert stale static tool names.
7. Refresh documentation to mark the 2026-06-13 tooling map as historical and point to the runtime manifest API.

## Non-Goals

- Do not rewrite the LangGraph topology.
- Do not change actual doctor or patient graph tool permissions.
- Do not add write APIs for enabling or disabling tools.
- Do not add an `X-Admin-Token` header.
- Do not expose factory references, local paths, model paths, API keys, or tokens.
- Do not run network calls, model loading, GPU checks, or health probes in the manifest endpoint.

## Acceptance Criteria

- `GET /api/admin/tools` returns a safe manifest projection under existing Bearer admin protection.
- Local fallback from `API_ADMIN_BEARER_TOKEN` to `API_BEARER_TOKEN` remains covered by tests.
- Agent-admin tools page uses backend manifest data as the primary source.
- API failures are visible and fallback data is labeled as fallback.
- Manifest tests prove known tool surfaces are covered without stale database declarations.
- API response tests prove sensitive fields are not exposed.
- Manifest import and API response generation remain lightweight and do not load model weights or require network access.

## Implementation Notes

Prefer small, reversible changes:

- Keep manifest metadata declarative.
- Keep runtime registry functions stable.
- Build API projections from manifest helper functions instead of formatting directly in the route.
- Add backend tests before frontend wiring where possible.
- Treat candidate tools such as `search_latest_research` as visible but not graph-enabled unless a separate decision promotes them.
