# Patient Identity Capture For Registry Design

**Date:** 2026-04-21  
**Status:** In review  
**Goal:** 在患者端新增一条“身份信息补录并写入 SQL 患者库”的固定链路，让患者在当前 session 已绑定的 draft patient 上填写 `患者名称` 和 `患者编号`，并在保存时完成“患者编号唯一”校验。

## 1. 背景

当前患者端已经有两条相对独立的能力：

- 创建患者 session 时，后端会自动在 patient registry 中创建一条 draft patient，并把 `patient_id` 绑定到当前 session。
- 患者端问诊、分诊、上传资料会不断补充医学信息，但这些信息主要服务于 `patient_card`、medical card 和后续医生判断。

目前缺少的是一条明确的“身份补录”链路：

- 患者端没有填写 `患者名称` 和 `患者编号` 的固定入口。
- draft patient 只有内部主键 `id`，缺少面向业务的患者编号。
- 患者编号没有唯一校验，因此无法稳定进入 SQL 患者库作为业务识别信息。

本次需求不是重做建档，而是在现有 draft patient 基础上，补一条“患者端补录身份信息”的固定路径。

## 2. 当前代码现状

已确认的现状如下：

1. 患者 session 创建时，`backend/api/routes/sessions.py` 会调用 `PatientRegistryService.create_draft_patient(...)` 创建 draft patient，并把 `patient_id` 绑定到 session。
2. `backend/api/services/patient_registry_service.py` 的 `patients` 表当前已有内部主键和医学快照字段，但没有：
   - `patient_name`
   - `patient_number`
   - `identity_locked`
3. 前端患者右侧已有固定模块式结构，适合承载一张常驻的身份信息卡，而不是把这套表单做成消息卡片。
4. 当前自动生成的患者画像卡 `patient_card` 属于医学画像，不适合混入身份主数据编辑。

## 3. 范围

### 3.1 范围内

- 在患者端右侧新增常驻身份信息卡片
- 支持填写 `患者名称` 和 `患者编号`
- 只对 `患者编号` 做唯一校验
- 将填写结果写入当前 session 绑定的 draft patient 记录
- 保存成功后在患者端锁定
- 在患者端 recovery / session 刷新时恢复该身份状态
- 为后续医生端数据库维护保留清晰的后端字段

### 3.2 范围外

- 重做患者建档流程
- 要求患者必须先填身份信息才能继续问诊
- 在患者端支持二次编辑
- 在本次需求中扩展复杂的患者合并/冲突处理工作流
- 在本次交付中新增医生端 registry 列表/详情展示或编辑入口

## 4. 已确认的产品决策

以下决策已经确认：

- `患者名称` 由患者手动填写
- `患者编号` 由患者手动填写
- 只校验 `患者编号` 唯一，不校验 `患者名称`
- 入口位于患者端右侧，长期可见
- 默认先显示 `填写患者信息` 按钮，不直接展开表单
- 由患者主动点击按钮后再展开填写
- 若 `患者编号` 已存在，则阻止保存并提示：`患者编号已存在，请更换`
- 保存成功后患者端锁定，不允许再次修改
- 后续如需修改，只能在医生端数据库入口中处理

## 5. 设计原则

### 5.1 基于现有 draft patient 回填

本次方案不新建第二条患者记录，不重做 session 建档逻辑，而是把 `患者名称` 和 `患者编号` 回填到当前 session 已绑定的 draft patient 上。

### 5.2 身份信息与医学画像分离

`patient_card` 继续承载医学画像字段；`患者名称` 和 `患者编号` 属于患者身份主数据，不进入 `patient_card` payload。

### 5.3 患者端只负责首次填写

患者端职责是首次补录并锁定。锁定后不再在患者端提供编辑入口。

### 5.4 唯一性基于归一化值

`患者编号` 的唯一规则必须在 spec 中固定，不能把“是否忽略大小写/空白”的判断留给实现时自由发挥。

## 6. 总体方案

### 6.1 右侧固定身份信息卡

在患者端右侧新增一张固定模块卡片，例如“补充身份信息”：

- 未填写时：
  - 显示说明文案
  - 显示 `填写患者信息` 按钮
- 点击按钮后：
  - 原位展开表单
  - 字段为：
    - `患者名称`
    - `患者编号`
- 保存成功后：
  - 切换为只读显示
  - 明确提示：`如需修改，请在医生端数据库中处理`

这张卡不走消息流，不做成 clinical card，而是患者右侧固定 UI 模块。

### 6.2 不阻塞问诊主流程

即使患者尚未填写身份信息，也不阻塞当前问诊、分诊、上传资料流程。  
当前 session 仍然挂在自动创建的 draft patient 上，只是该记录尚未补齐业务身份字段。

### 6.3 新增 session 级身份补录接口

建议新增专门的 session 级接口，例如：

- `POST /api/sessions/{session_id}/identity`

输入：

- `patient_name`
- `patient_number`

隐式上下文：

- 当前 `session_id`
- 当前 session 已绑定的 draft `patient_id`

服务端处理流程：

1. 校验 session 存在
2. 校验该 session 属于 `patient` scene
3. 读取当前 session 绑定的 patient
4. 若该 patient 已经 `identity_locked = 1`，拒绝再次写入
5. 校验 `patient_name` 非空
6. 校验 `patient_number` 非空
7. 对 `patient_number` 执行归一化
8. 校验归一化后的编号未被其他 patient 占用
9. 写入 `patient_name`
10. 写入原始 `patient_number`
11. 写入归一化后的 `patient_number_normalized`
12. 将 `identity_locked = 1`
13. bump snapshot version
14. 返回最新 session snapshot

### 6.4 重复校验规则

重复校验只看 `patient_number`，但判重必须基于归一化值。

固定规则：

- 允许不同患者名称重复
- `patient_number` 写入前先做归一化
- 归一化规则固定为：
  - 去掉首尾空白
  - 保留中间字符，不额外删除中间空格或分隔符
  - 英文字母统一转为大写
- 不允许两个 patient 拥有相同的 `patient_number_normalized`
- 保存当前 patient 时，若发现另一条 patient 记录已使用相同归一化编号，则返回冲突错误

重复时的前端提示固定为：

- `患者编号已存在，请更换`

### 6.5 患者端锁定规则

患者端保存成功后进入锁定态：

- 显示只读的 `患者名称`
- 显示只读的 `患者编号`
- 不再显示编辑按钮
- 如需修改，只提示去医生端数据库处理

锁定真相源以后端 `identity_locked` 为准，而不是前端本地状态。

## 7. 前端交互设计

### 7.1 状态机

患者端身份信息卡建议使用以下状态：

- `empty`
  - 未填写
- `editing`
  - 已展开表单
- `saving`
  - 正在提交
- `error`
  - 提交失败，展示错误
- `saved`
  - 已成功保存并锁定

### 7.2 默认展示

未填写时展示：

- 一段简短说明
- `填写患者信息` 按钮

点击按钮后进入 `editing`。

### 7.3 编辑态

编辑态展示：

- `患者名称`
- `患者编号`
- `保存`
- `取消`

取消只回到未展开状态，不写入后端。

### 7.4 错误态

错误类型至少覆盖：

- `patient_number_conflict`
  - 提示：`患者编号已存在，请更换`
- `identity_locked`
  - 提示：`当前身份信息已锁定，请在医生端数据库中修改`
- 通用失败
  - 使用统一失败文案

错误发生后保持在 `editing`，便于用户修正并重新提交。

### 7.5 已保存态

保存成功后展示：

- `患者名称`
- `患者编号`
- 锁定提示文案

不再显示编辑按钮。

## 8. 后端设计

### 8.1 数据库字段

`patients` 表新增字段：

- `patient_name TEXT NULL`
- `patient_number TEXT NULL`
- `patient_number_normalized TEXT NULL`
- `identity_locked INTEGER NOT NULL DEFAULT 0`

推荐约束：

- `CREATE UNIQUE INDEX IF NOT EXISTS idx_patients_patient_number_normalized_unique ON patients(patient_number_normalized) WHERE patient_number_normalized IS NOT NULL`

执行位置：

- 索引与新增列一起在 `PatientRegistryService._initialize()` 中完成初始化
- 不额外引入独立迁移脚本，保持与当前 registry service 的自启动建表/补列模式一致

使用部分唯一索引的原因：

- 允许 draft patient 初始编号为空
- 只对已填写归一化编号的记录生效

字段职责：

- `patient_number`
  - 保留患者原始输入，用于展示
- `patient_number_normalized`
  - 仅用于唯一校验和查询比对

### 8.2 后端 service 能力

在 `PatientRegistryService` 中新增明确职责的方法：

- `set_patient_identity(...)`
  - 给指定 patient 写入 `patient_name/patient_number`
  - 生成并写入 `patient_number_normalized`
  - 做唯一校验
  - 成功后锁定
- `get_patient_identity(...)`
  - 返回患者身份字段与锁定状态
- `patient_number_exists(...)`
  - 基于归一化后的编号做查询
  - 供 service 内部使用
- `normalize_patient_number(...)`
  - 封装唯一规则，避免多处实现不一致

错误分支应明确为业务错误，而不是模糊的通用异常。

并发场景处理：

- 即使前置做了 `patient_number_exists(...)` 检查，最终写入仍可能被数据库唯一索引拦截
- service 层需要捕获 `sqlite3.IntegrityError`
- 捕获后统一转换成业务错误，例如 `PatientNumberConflictError`
- route 层再把该业务错误映射为 `409`

`get_patient_identity(...)` 返回 shape 固定为：

```json
{
  "patient_name": null,
  "patient_number": null,
  "identity_locked": false
}
```

未填写的 draft patient 也返回同一 shape，避免前端区分多种空态结构。

### 8.3 API contract

建议新增 session 级接口：

- `POST /api/sessions/{session_id}/identity`

请求示例：

```json
{
  "patient_name": "张三",
  "patient_number": "P-2026-0001"
}
```

成功返回：

- 最新 `SessionResponse`

冲突返回：

- `409`
- `detail = "PATIENT_NUMBER_ALREADY_EXISTS"`

锁定返回：

- `409`
- `detail = "PATIENT_IDENTITY_LOCKED"`

非法 scene 返回：

- `409`
- `detail = "NOT_PATIENT_SESSION"`

### 8.4 Session Snapshot

为支持刷新恢复，身份信息应进入 `RecoverySnapshot`，而不是挂在 `SessionResponse` 顶层。

推荐字段：

- `patient_identity`
  - `patient_name`
  - `patient_number`
  - `identity_locked`

组装位置：

- `build_recovery_snapshot(...)` 保持现有签名，不直接依赖 registry service
- 在 route 层的 `_build_session_response(...)` 中，先构建 snapshot，再基于 `meta.patient_id` 调用 `patient_registry_service.get_patient_identity(...)`
- 最后把结果注入 `snapshot.patient_identity`

这样患者端右侧卡片只依赖 session snapshot 即可恢复，同时不扩大 `state_snapshot.py` 的职责边界。

## 9. 与现有能力的关系

### 9.1 与问诊/分诊/上传流程的关系

身份补录链路不阻塞以下流程：

- 问诊
- 分诊
- 上传资料

它们继续作用于当前 draft patient，不需要等待身份补录完成。

### 9.2 与 patient_card 的关系

本次明确保持职责分离：

- 不把 `patient_name/patient_number` 塞进当前 `patient_card` payload
- 身份信息与医学画像保持分离

### 9.3 与医生端数据库的关系

患者端只负责首次填写并锁定。

本次交付只要求后端数据层为后续医生端维护保留清晰字段，不要求在本次实现中同步交付医生端 registry 的展示或编辑入口。

后续医生端能力边界：

- 后续可基于同一批数据库字段查看 `patient_name`
- 后续可基于同一批数据库字段查看 `patient_number`
- 后续如需维护，走医生端数据库入口，不走患者端复用链路

## 10. 错误处理

至少需要覆盖以下异常：

- session 不存在
- 当前 session 不是 patient scene
- 当前 session 未绑定 patient
- `患者名称` 为空
- `患者编号` 为空
- `患者编号` 冲突
- 当前 patient 已锁定

错误处理原则：

- 后端返回清晰的业务错误
- 前端把冲突和锁定态转成固定文案
- 其他失败统一落到通用失败提示

## 11. 测试建议

### 11.1 后端

建议覆盖：

- patient session 创建后已绑定 draft patient
- 首次写入 `patient_name/patient_number` 成功
- `patient_number` 归一化后冲突会阻止保存
- 大小写不同但归一化后相同的编号会冲突
- 首尾空白不同但归一化后相同的编号会冲突
- 已锁定 patient 再次写入会被拒绝
- session snapshot 能带回 identity 信息
- 非 patient scene 调用被拒绝

### 11.2 前端

建议覆盖：

- 患者端右侧显示“填写患者信息”入口
- 点击后展开表单
- 保存成功后切换为只读
- 编号冲突时显示 `患者编号已存在，请更换`
- 锁定后不再显示编辑入口
- 页面刷新后仍能恢复已保存状态

## 12. 实施建议

建议实现顺序：

1. 扩展 patient registry 数据库与 schema
2. 增加 service 方法和 session-level identity API
3. 扩展 session snapshot 返回 identity 信息
4. 在患者端右侧新增固定身份信息卡
5. 接通保存、冲突提示、锁定显示
6. 最后补充回归测试与联调验证

这样可以先把后端真相源和唯一校验建立起来，再做前端交互，避免前端先行产生临时状态协议。

## 13. 结论

本需求不是新建患者，而是在现有自动创建的 draft patient 基础上补一条“患者端身份补录”链路。

推荐方案是：

- 患者端右侧常驻身份信息卡
- 点击后填写 `患者名称` 和 `患者编号`
- 仅校验 `患者编号` 唯一
- 唯一规则基于 `patient_number_normalized`
- 保存成功后锁定
- 刷新后通过 session snapshot 恢复

这条方案对当前 session / registry 架构侵入最小，职责边界也最清晰。
