# Production Readiness Roadmap Design

## Goal

把当前 `FastAPI BFF + POST SSE + LangGraph` 架构从本地/内部演示形态推进到可生产部署形态。本文只定义生产化路线图、阶段边界、风险控制和验证要求，不修改代码、不引入运行时依赖、不改变当前部署行为。

## Current State

当前系统的核心交互是前端通过 `POST /api/sessions/{session_id}/messages/stream` 发起一轮对话，后端用 `StreamingResponse(text/event-stream)` 持续发送应用语义事件。前端 `frontend/src/app/api/stream.ts` 使用 `fetch()` 和 `ReadableStream.getReader()` 手动解析 SSE block，只读取 `data:` 行。

后端 `backend/api/services/graph_service.py` 的 `encode_sse_event()` 只输出 `event:` 和 `data:`，不输出 SSE `id:`。`GraphService.stream_turn()` 在单轮开始时通过 `session_store.try_acquire_run_lock()` 获取会话运行锁，在流成功、错误或取消路径释放锁。取消路径会恢复 pending context 并关闭 LangGraph stream。

会话 metadata 可选 `memory` 或 `sqlite`。`SqliteSessionStore` 继承 `InMemorySessionStore`，只镜像 session metadata 到 SQLite；active run lock 仍是进程内 `threading.Lock`。文档已说明 active run/run locks 只在当前进程有效，第一版不支持 multi-worker 共享锁。

认证目前是 BFF 层 Bearer token。浏览器端通过 `VITE_API_BEARER_TOKEN` 构造 `Authorization` header；后端区分 user token 与 admin token，并对病例 upsert、患者删除等管理端点要求 admin token。CORS 使用显式 origin allowlist，`allow_credentials=False`。

## Problem Statement

当前架构对本地开发、fixture demo 和单进程内部工具是合理的，但生产部署会遇到四类确定问题：

1. **断连续传缺失**  
   POST SSE 不能使用浏览器原生 `EventSource` 自动重连。当前协议没有 `id:`、没有 `Last-Event-ID`、没有 `after_seq` 回放端点。网络中断时前端只会报告可恢复错误，无法接回同一个 run。

2. **运行锁只在单进程有效**  
   多 worker 或多实例部署时，每个进程都有自己的 `_run_locks` 字典，同一个 `session_id` 可被多个进程同时运行，可能交叉写同一个 LangGraph `thread_id` checkpoint。

3. **浏览器长期 Bearer token 不适合多用户生产**  
   Vite 注入的 token 对浏览器用户可见。它可以服务内部工具或本地演示，但不能作为真实多用户长期密钥。admin token 更不能下发到浏览器。

4. **SSE 对反代和容量敏感**  
   每个活跃 turn 持有长连接和后端协程。Nginx、云 LB、CDN 若开启 response buffering 或 idle timeout 太短，会让 SSE 延迟或断流。生产必须显式配置 buffering、timeout、连接上限和容量指标。

## Non-Goals

本路线图不要求第一阶段立即实现以下能力：

- 不把 POST SSE 立即改成 WebSocket。
- 不在第一阶段实现完整断连续传。
- 不在第一阶段引入 Redis、OAuth/OIDC 或新的部署平台。
- 不把 SQLite 设计成多实例分布式锁后端。
- 不改变当前前端 reducer 的事件语义。
- 不承诺横向扩展前的多 worker 安全性。

## Design Principles

- **阶段化落地**：每一阶段都应可独立测试、可回滚、能解释生产收益。
- **先固化边界，再扩展能力**：先让当前单进程模型在文档、启动脚本和检查项中明确，再引入分布式能力。
- **运行所有权清晰**：一个 `run_id` 必须有明确 owner；连接断开不应模糊地决定 graph 是否继续。
- **锁必须有 TTL**：任何跨进程锁都必须能在进程崩溃后自动释放，并支持续租。
- **协议演进兼容**：新增 SSE `id:`、`schema_version`、`run_id`/`seq` 不应破坏旧事件 payload 消费。
- **浏览器不持有长期管理密钥**：管理能力必须留在服务端可信边界内。

## Phase 0: Current Deployment Boundary

### Objective

把当前架构的部署前提写清楚，让部署人员不会误以为 `SESSION_STORE_BACKEND=sqlite` 已经支持多 worker，也不会让反代破坏 SSE。

### Changes

- 在 README 增加生产边界说明：
  - 当前推荐 `uvicorn` 单 worker。
  - `SESSION_STORE_BACKEND=sqlite` 只恢复 metadata，不提供跨进程运行锁。
  - POST SSE 当前无中途续传；断开后的用户体验是报错或刷新后读取已完成 snapshot。
  - 浏览器 `VITE_API_BEARER_TOKEN` 仅适合内部/本地工具。
- 增加反代示例片段：
  - `/api/sessions/*/messages/stream` 关闭 response buffering。
  - proxy read timeout 大于最大预期 turn 时长。
  - idle timeout 大于 heartbeat interval。
  - 禁用会压缩或缓存 SSE 的中间层行为。
- 增加容量估算公式：
  - 活跃 SSE 连接数约等于并发用户数乘以平均活跃 turn 占比。
  - worker 协程、LLM 并发、checkpoint 后端吞吐分别作为容量上限。

### Validation

- 文档审查确认没有“SQLite 支持多 worker 锁”的暗示。
- 本地 `pytest tests/backend/test_graph_service_streaming.py tests/backend/test_sqlite_session_store.py -q` 继续通过。
- 前端 `vitest src/app/api/stream.test.ts` 继续通过。

## Phase 1: Low-Risk Stability Repairs

### Objective

不改变协议、不引入新基础设施，先减少单进程部署中的易触发错误。

### Changes

- 加固 `InMemorySessionStore.try_acquire_run_lock()` 和 `release_run_lock()`：
  - 在 `_store_lock` 内读取 session 和 lock。
  - 使用 `.get()` 处理 session 不存在。
  - session 或 lock 缺失时返回 `False`，不抛 `KeyError`。
- 加固 pending context 操作：
  - 对 `restore_context_messages()`、`drain_context_messages()` 等当前直接下标访问的方法做一致性审查。
  - 仅修补会影响 stream cleanup 的路径。
- 降低 abort 后快速重发的 409：
  - 首选前端对 `409 Session is busy` 做一次短延迟重试，延迟范围 100-250ms。
  - 如果重试后仍 409，则保留现有可恢复错误。
  - 不在本阶段做后端抢占式取消，因为当前 `GraphService` 不持有每个 run 的长期 task registry。

### Data Flow

1. 用户提交新 prompt。
2. 前端 abort 旧 stream。
3. 新 stream 请求到达后端。
4. 如果旧 generator cleanup 还未释放锁，后端返回 409。
5. 前端等待短延迟后重试一次。
6. 若 cleanup 已完成，新请求正常进入；若仍 busy，用户看到可恢复错误。

### Validation

- 后端新增测试覆盖：
  - session 不存在时 `try_acquire_run_lock()` 返回 `False`。
  - release 时 session 已缺失不会抛异常。
  - cleanup finally 重复释放锁不会出错。
- 前端新增测试覆盖：
  - `streamTurn` 首次 409、短延迟后成功。
  - 第二次仍 409 时保留 `STREAM_REQUEST_FAILED` 行为。
  - 非 409 错误不重试。

## Phase 2: Redis Distributed Run Lock

### Objective

支持多 worker 或多实例部署时，同一个 session 同一时刻只有一个 active graph run。

### Architecture

引入一个运行锁抽象，不再让 graph service 直接依赖进程内 lock 细节。生产环境使用 Redis；本地和测试仍可使用 memory lock。

建议接口：

```python
class RunLockStore(Protocol):
    def try_acquire(self, session_id: str, run_id: str, ttl_ms: int) -> bool: ...
    def renew(self, session_id: str, run_id: str, ttl_ms: int) -> bool: ...
    def release(self, session_id: str, run_id: str) -> bool: ...
```

Redis key 格式：

```text
langg:run-lock:{session_id} -> run_id
```

获取锁：

```text
SET langg:run-lock:{session_id} {run_id} NX PX {ttl_ms}
```

释放锁必须比较 value，只有当前 `run_id` 匹配才能删除。续租也必须比较 value，避免旧 owner 续租新 owner 的锁。

### Runtime Behavior

- 每个 graph run 获取锁后启动续租任务。
- 续租周期建议为 TTL 的 1/3。
- 正常完成、错误、取消时释放锁。
- 进程崩溃时由 Redis TTL 自动释放。
- 如果续租失败，当前 run 应停止向 checkpoint 写入并返回可恢复错误；否则会产生双 owner 风险。

### Configuration

新增配置建议：

```text
RUN_LOCK_BACKEND=memory|redis
RUN_LOCK_REDIS_URL=redis://...
RUN_LOCK_TTL_MS=60000
RUN_LOCK_RENEW_INTERVAL_MS=20000
```

部署约束：

- `RUN_LOCK_BACKEND=memory` 只允许单 worker。
- 多 worker 或多实例必须设置 `RUN_LOCK_BACKEND=redis`。
- Redis 不替代 LangGraph checkpoint。checkpoint 仍需要选择 memory/sqlite/postgres/redis 中适合生产的后端。

### Validation

- 单进程 memory lock 测试保持不变。
- Redis 集成测试使用独立 test database 或容器化 Redis。
- 并发两个请求同一 session：
  - 第一个获取锁。
  - 第二个返回 409 或等待策略定义结果。
  - checkpoint 中不出现交叉写入。
- 进程模拟崩溃后，TTL 到期可重新获取锁。

## Phase 3: SSE Event Sequencing And Resume

### Objective

网络中断后，客户端可从最后收到的事件序号继续读取同一个 run 的事件，而不是要求用户重发整轮。

### Protocol

SSE 增加标准 `id:`：

```text
id: {run_id}:{seq}
event: message.delta
data: {...}
```

事件 payload 增加可选字段：

```json
{
  "schema_version": 1,
  "run_id": "run_x",
  "seq": 42,
  "type": "message.delta"
}
```

兼容策略：

- 旧前端忽略 `id:`，继续只读 `data:`。
- 新前端记录最后成功处理的 `{run_id, seq}`。
- `schema_version` 初始为 `1`，只在破坏性 payload 变更时递增。

### Execution Ownership

把 graph 执行和 SSE 连接解耦：

- `RunManager` 持有 active run task。
- graph task 将事件写入 run event buffer。
- SSE 连接只负责从 buffer 读取并向客户端发送。
- 客户端断开不默认取消 graph task。
- 用户显式 abort 或发起 superseding turn 时，才取消旧 run。

### Event Buffer

Phase 3 可以先用进程内 ring buffer，Phase 3 production-ready 目标使用 Redis Stream。

Redis Stream key：

```text
langg:run-events:{run_id}
```

字段：

```json
{
  "seq": 42,
  "event_type": "message.delta",
  "payload": "{json}",
  "created_at": "2026-06-13T..."
}
```

保留策略：

- 成功完成后保留短 TTL，例如 15-60 分钟。
- aborted/error run 也保留短 TTL，供客户端收到最终状态。
- 过期后续传端点返回 410 Gone，前端提示刷新 snapshot 或重发。

### API

新增续传端点：

```text
GET /api/sessions/{session_id}/runs/{run_id}/stream?after_seq={seq}
```

行为：

- 校验 session 与 run 归属。
- 先回放 `seq > after_seq` 的 buffered events。
- 再接 live events。
- 如果 run 已完成且 buffer 全部回放完，发送 `done` 后结束。
- 如果 run 不存在或 buffer 过期，返回 404 或 410。

当前 POST endpoint 可以保留：

```text
POST /api/sessions/{session_id}/messages/stream
```

它负责创建 run 并返回首条 SSE；响应中包含 `run_id` 和 `seq`。断线后前端切换到 GET resume endpoint。

### Frontend

- `stream.ts` 解析 `id:`，并将 `lastEventId` 暴露给调用层。
- `use-workspace-streaming-turn.ts` 在网络错误时判断是否有 active `run_id` 和 `lastSeq`。
- 如果有，则调用 resume endpoint。
- 重连采用指数退避，但设置最大尝试次数和总时长。
- 收到 410 时降级为刷新 session snapshot。

### Validation

- 单元测试：
  - parser 能读取 `id:` 且不破坏旧事件。
  - 后端 encode 支持 seq。
  - buffer 能按 `after_seq` 回放。
- 集成测试：
  - 模拟客户端在 `message.delta` 后断开。
  - graph task 继续完成。
  - resume endpoint 回放缺失事件并最终发送 `done`。
- 负向测试：
  - 过期 run 返回 410。
  - 错误 session 不能读取其他 session run。

## Phase 4: Short-Lived Browser Tokens And OAuth/OIDC

### Objective

浏览器不再持有长期 API token 或 admin token。多用户生产环境由可信身份提供方签发用户身份，BFF 管理 session 和权限。

### Architecture

建议从两步演进：

1. **短期 BFF session token**
   - 前端通过登录或受控 bootstrap 获取短期 token。
   - token TTL 以分钟计，支持 refresh。
   - admin 操作不通过浏览器 admin token 直连，而由后端根据用户角色判断。

2. **OAuth/OIDC**
   - 接入医院/组织身份源。
   - BFF 校验 ID token 或通过授权码流程建立 server-side session。
   - RBAC 映射到普通用户、医生、管理员、审计只读等角色。

### Security Boundaries

- `API_ADMIN_BEARER_TOKEN` 只作为服务端到服务端或运维脚本凭据。
- 前端构建产物不得包含 admin token。
- 管理端点必须校验用户角色，而不是只校验浏览器传来的静态 token。
- 患者数据访问必须绑定用户权限、场景和审计日志。

### API Changes

新增认证状态端点：

```text
GET /api/auth/me
POST /api/auth/session
POST /api/auth/refresh
POST /api/auth/logout
```

前端 API client 从静态 build-time token 迁移到 runtime token provider：

```ts
type AuthTokenProvider = () => Promise<string | null>;
```

### Validation

- 无 token 访问 API 返回 401。
- 普通用户访问 admin endpoint 返回 403。
- admin 用户访问 admin endpoint 成功。
- 前端构建产物扫描不包含 `API_ADMIN_BEARER_TOKEN` 或固定生产 token。
- token 过期时前端 refresh；refresh 失败则回登录态。

## Phase 5: Deployment Operations Baseline

### Objective

让生产部署可观测、可容量评估、可安全滚动升级。

### Reverse Proxy Requirements

Nginx 示例方向：

```nginx
location ~ ^/api/sessions/.*/messages/stream$ {
    proxy_pass http://langg_backend;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    add_header X-Accel-Buffering no;
}
```

实际部署需按平台调整，核心要求是不缓冲 SSE、不缓存 SSE、idle timeout 大于 heartbeat interval 和最大无业务事件间隔。

### Health Checks

建议区分：

- **Liveness**：进程存在并能返回基础响应。
- **Readiness**：配置有效、session store 可用、run lock backend 可用、checkpoint backend 可用。
- **Dependency health**：Redis、checkpoint、LLM provider、RAG index 可分别报告 degraded。

### Metrics

至少需要：

- active SSE connections
- active graph runs
- session lock acquire success/failure
- lock renewal failure
- stream disconnect count
- resume success/failure
- run duration histogram
- event buffer size and eviction count
- 401/403 count
- LLM latency and error rate

### Rollout Constraints

- Phase 0/1 可单实例滚动。
- Phase 2 开始必须先部署 Redis 和配置 readiness。
- Phase 3 涉及协议演进，需支持旧前端一段时间。
- Phase 4 涉及认证迁移，必须提供回滚路径和管理员访问兜底。

## Migration Order

推荐顺序：

1. Phase 0：文档和部署边界。
2. Phase 1：单进程稳定性修补。
3. Phase 2：Redis run lock，先单实例启用，再多 worker 灰度。
4. Phase 3：SSE sequencing，先发 `id:` 和 payload seq，再上 resume endpoint，再启用前端自动 resume。
5. Phase 4：短期 token，再 OIDC。
6. Phase 5：贯穿每阶段补齐指标、健康检查和反代配置。

不推荐先做 Phase 3 再做 Phase 2，因为续传要求 run ownership 稳定；没有分布式锁时，多 worker 下无法可靠判断哪个 task 拥有 run。

## Testing Strategy

### Unit Tests

- SSE encoding and parsing。
- session lock acquire/release。
- Redis lock compare-and-release。
- run event buffer append/replay。
- auth role checks。

### Integration Tests

- FastAPI stream endpoint contract。
- disconnect and resume。
- concurrent same-session run rejection。
- token expiration and refresh。
- proxy-like delayed chunk behavior。

### End-to-End Tests

- 患者端提交 prompt，后端流式返回，前端完成渲染。
- 网络中断后 resume 并补齐消息。
- 普通用户无法调用管理操作。
- 管理员操作带审计记录。

### Load Tests

- 目标并发 SSE 连接数。
- 平均 run 时长和 p95 run 时长。
- Redis lock renewal under load。
- LLM provider rate limit 下的降级行为。

## Open Decisions

这些问题不阻塞 spec，但会影响 Phase 2+ 的具体实现计划：

- Redis 是否已是目标部署环境可用组件。
- LangGraph checkpoint 生产后端选择 Postgres 还是 Redis。
- OAuth/OIDC 身份源是医院内网 IdP、企业 SSO，还是应用自建账号。
- 断线后 graph 是否默认继续跑完，还是只在特定场景继续。
- 同一 session 新 prompt 到来时，是取消旧 run、排队，还是返回 busy。

## Acceptance Criteria

该路线图完成后，应满足：

- 当前单 worker 前提在部署文档中明确，没有 SQLite 多 worker 锁的误导。
- 第一阶段可在不引入新依赖的情况下提升 abort/retry 稳定性。
- 多 worker 生产化有明确 Redis lock 设计和 TTL/续租策略。
- SSE 续传有明确协议、buffer、API 和前端重连路径。
- 浏览器长期 token 和 admin token 风险有清晰迁移方案。
- 反代、健康检查、指标和容量评估进入上线清单。

