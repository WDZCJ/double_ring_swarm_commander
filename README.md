# 双环 Agent 架构 · 智能巡飞弹集群决策系统

> 外环 Agent 负责理解意图、规划与调度，内环 Jev（本地 Laya 决策模型）负责快速响应。运行时可通过 REST API / Web UI 与系统实时交互：加载场景、启动任务、暂停/恢复、调速、下发新任务（replace 语义）、注入临机事件（弹损毁/通信失联/情报更新）、管理监控任务等。

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/WDZCJ/double_ring_swarm_commander.git
cd double_ring_swarm_commander

# 2. 创建虚拟环境 & 安装依赖
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt

# 3. 配置环境变量（复制示例即可运行 mock 模式）
copy .env.example .env        # Windows
# cp .env.example .env       # macOS / Linux

# 编辑 .env，填写 DeepSeek API Key 与本地 Laya 服务：
#   LLM_MODE=real
#   LLM_BASE_URL=https://api.deepseek.com/v1
#   LLM_API_KEY=<你的 key>
#   LLM_MODEL=deepseek-chat
#   JEV_MODE=real
#   JEV_BASE_URL=http://localhost:8000

# 4. 启动 Web 可视化 Demo
python -m munition_system.demos.demo_web --host 127.0.0.1 --port 8080

# 5. 浏览器打开 http://127.0.0.1:8080
#    默认不会自动调用 LLM；在 Web UI 上先"加载场景"，再"启动任务"。
```

运行测试：

```bash
pytest -q
```

---

## 设计思路

系统采用**双层闭环架构**：

```
                  ┌──────────────────────────┐
                  │      外环 Agent（LLM）     │
                  │ DeepSeek / OpenAI 兼容    │
                  │ 意图理解 · ReAct 规划      │
                  │ 监控任务管理 · 重规划        │
                  └─────────────────┬────────┘
                                    │ tools / command
           运行时交互 ←──────────────┼──────────────→ CommandBus
                                    ▼
                  ┌──────────────────────────┐
                  │    内环 Jev / Laya 模型    │
                  │ 本地 HTTP 服务 / mock      │
                  │ 全局 schema · 定时监控任务   │
                  │ 动作映射 · 升级判定          │
                  └─────────────────┬────────┘
                                    │ ActionCommand / Escalation
                                    ▼
                  ┌──────────────────────────┐
                  │    仿真层 simulation       │
                  │ World · Munitions · Zones │
                  │ 事件注入 · 威胁向量          │
                  └──────────────────────────┘
```

### 线程边界（关键不变量）

| 线程 | 写者 | 只读 |
|---|---|---|
| Runner 线程（唯一） | World · Plan · state · event_log | 只读 runner 自身字段 |
| API 线程（uvicorn） | CommandBus.submit() | `web_state.get_current_state()` 深拷贝快照 |
| WS 广播线程 | 无 | 转发队列中的消息 |

**只有 Runner 线程执行写操作。** API 线程只通过 CommandBus 排队命令，Runner 在 tick 边界 `drain()` 并逐条 `apply`，每条命令的结果（applied / rejected + result / error）写回 Command。这避免了跨线程竞态，是整套系统可靠性的核心保证。

### 命令模型（`CommandBus`）

10 种命令：`NEW_INTENT / REPLAN / INJECT_EVENT / MONITORING_* / MUNITION_COMMAND / SAFE_MODE / PAUSE / RESUME / STOP / SPEED`。

- **重命令节流**：`NEW_INTENT / REPLAN` 每 tick 最多执行 1 条（`max_heavy=1`），其余留在队列
- **队列上限 100**：满了返回 429 `QueueFullError`
- **异步等待**：POST 默认 202 + `status=pending`；带 `?wait_ms=N` 时同步阻塞等待
- **pause 时仍交互**：暂停期间 Runner 仍在 100ms 轮询里 drain 命令

### 事件驱动重规划

破坏性临机事件（`DamageEvent` 弹损毁 / `CommLossEvent` 通信失联）注入后，不在命令执行时立刻重规划——那时世界状态还没应用事件。正确时序是：

```
tick N  drain → _cmd_inject_event → 事件排队 tick N+1 触发 + 挂起 _pending_event_replan
tick N+1 world.step() → EventInjector.check() → 事件生效 → _on_event_fired 标记就绪
tick N+1 _tick 尾部 → 发现 ready → _cmd_replan → LLM 看到损失后的真实态势 → 产出新 plan
```

这条路径走 `trigger_external`，绕过了 demo 的 `DEMO_SUPPRESS_REPLAN` 自动重规划抑制。

### 全局 schema vs 定时监控任务

| 维度 | 全局 schema | 定时监控任务 |
|---|---|---|
| 执行频率 | 每 tick | `schedule_every_ticks` 指定周期 |
| 监控范围 | 集群级持续监控 | 特定主体（单弹/弹组/全局）的精细化问题 |
| 产出 | 路由 → ActionCommand 或 Escalation | 高置信 Choice → ActionCommand；低置信/答案命中升级策略 → Escalation |
| 生命周期 | `plan_version` 变化时整体替换 | 独立 create / pause / resume / cancel |
| 例子 | "集群返航状态 Score" | "M1 弹药耗尽？" "B 组完成度 Score" |

### 安全模式

- 默认关闭；可通过 Web UI / `POST /api/safe-mode` 手动触发
- 连续 5 次重规划失败（Replanner 内部计数）自动进入
- 触发后全员返航：`enter_safe_mode` 给所有弹下发 `action=return_to_base`，立即应用到 World，safe_mode 标志置 True
- 外部触发（`trigger_external`）**不**累计连续失败计数，防止用户频繁操作误触安全模式

### 重规划计数语义

`_cmd_replan`（外部重规划，包括事件驱动）会递增 `_replan_count`，这样 Web UI 和 API 返回的 `replan_count` 反映真实重规划次数，即使这些规划不走自动路径。

### Fallback 策略

- **ReAct 失败 / max_iterations 耗尽** → `_safe_fallback_plan`（全员返航安全降级）
- **空 assignments 不再被当作有效 plan**：`_try_parse_plan` 强制 assignments 非空；LLM 输出 `{"assignments":[]}` 视为过早终止，让 ReAct 继续迭代或命中 max_iterations
- **Laya 不可用（503 / timeout / network）** → `JevUnavailable` 抛到 orchestrator，当前 demo 模式下由 `_handle_replan` 处理

---

## 目录结构

```
munition_system/
├── common/          # 类型定义 / 协议 / 配置 / 校验
├── simulation/      # World / Munition / Zone / EventInjector / SituationalApi
├── jev/             # 内环 Jev：Laya 客户端（mock + real /api/v1/predict）
├── agent/           # 外环 ReActLoop / Replanner / SchemaDesigner / Tools / MonitoringManager
├── runner/          # DualLoopRunner（串起三层）/ Scenario / Telemetry
├── web/             # FastAPI Web 服务 / Control.CommandBus / Interaction 路由 / 静态 UI
└── demos/           # demo_web（带 API 和 WS 的可视化 Demo）

config/
└── scenarios/       # YAML 场景定义

tests/               # pytest（174 项）
```

---

## REST API（免鉴权）

启动后可通过 curl / Postman / Web UI 调用：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/scenarios` | 列出内置场景 |
| POST | `/api/scenarios/load` | `{scenario, max_ticks?}` |
| POST | `/api/run/start` | `{scenario?, intent_text?, max_ticks?}` |
| POST | `/api/runtime/pause\|resume\|stop` | 运行时开关 |
| POST | `/api/runtime/speed` | `{speed: 1.0}` （0.1–20） |
| POST | `/api/intents` | `{raw_text\|goal, mode:"replace"}` — **新任务整体替换 plan** |
| POST | `/api/replan` | 手动触发重规划 |
| POST | `/api/events/inject` | `{event_type, munition_id?\|munition_ids?\|target_id?, tick?}` — 破坏性事件默认自动重规划 |
| POST | `/api/monitoring/tasks` | 注册新监控任务 |
| POST | `/api/monitoring/tasks/{id}/pause\|resume\|cancel` | 管理任务 |
| PUT | `/api/monitoring/tasks/{id}/schema` | 更新 schema |
| GET | `/api/state\|plan\|intents\|telemetry\|events\|runtime\|commands` | 只读快照 |

所有命令默认异步返回 `202 {command_id, status:"pending"}`；带 `?wait_ms=2000` 时同步等待最多 N 毫秒。

WebSocket `/ws` 广播实时帧：`init / tick / phase / plan / llm_round / jev_call / monitoring_tasks / schema / event_injected / event_fired / replanned / run_end`。

---

## 配置（`.env`）

复制 `.env.example` 为 `.env`：

```env
# === LLM（外环）===
LLM_MODE=real          # mock 或 real
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=<你的 API Key>
LLM_MODEL=deepseek-chat

# === Jev / Laya（内环）===
JEV_MODE=real          # mock 或 real
JEV_BASE_URL=http://localhost:8000     # 本地 Laya 服务
JEV_API_KEY=

# === Demo 运行参数 ===
DEMO_TICK_SLEEP=0.8    # 每个 tick 之间停秒（让可视化清晰）
DEMO_COV_DELTA=0.005   # 覆盖率每 tick 增量
DEMO_SUPPRESS_REPLAN=true  # Demo 自动抑制重规划，避免 UI 上频繁切换
```

| 组合 | 效果 |
|---|---|
| LLM=real + Jev=real | **全真实**：DeepSeek 规划 + 本地 Laya 决策 |
| LLM=real + Jev=mock | DeepSeek 规划 + 启发式决策 |
| LLM=mock + Jev=real | 预制 LLM 响应 + Laya 决策（离线 / 单测用） |
| LLM=mock + Jev=mock | 完全离线（18 个 demo / 单测用） |

---

## 场景

内置 3 个场景（在 `config/scenarios/` 下）：

| key | 说明 | 规模 |
|---|---|---|
| `cooperative_search` | A/B 两组协同搜索东西两区 | 18 发 |
| `cooperative_attack` | A 前出诱饵，B 按 A 状态决策 | 2 发 |
| `adhoc_events` | 编组缺员 + 目标消失 + 部分失联三连发 | 6 发 |

可自行扩展 YAML 场景（字段参考 `Scenario` dataclass）。

---

## 运行时可观测性

| 来源 | 位置 | 内容 |
|---|---|---|
| REST `/api/state` | `web.state` | tick / phase / plan_version / scenario / world / schema / intents / event_log / telemetry / runtime |
| WS `/ws` | 广播通道 | 所有 Web UI 渲染数据 |
| `event_log` | `runner._event_log` deque(200) | command / intent_replaced / replanned / event_injected / event_fired / replan_armed / safe_mode |
| telemetry | `runner.telemetry` | tick_records / tool_calls / escalations / replans |

### 事件驱动重规划时序

```
tick N   drain → _cmd_inject_event → 事件排队 tick=N+1 + _pending_event_replan
tick N+1 world.step() → EventInjector.check() → _on_event_fired → _event_replan_ready=True
tick N+1 _tick 尾部 → ready=True → _try_event_replan → _cmd_replan → plan_v+1
```

这条路径**绕过 Demo 的自动重规划抑制**（`DEMO_SUPPRESS_REPLAN`），走 `_cmd_replan` → `trigger_external` 外部重规划，且递增 `replan_count`。

---

## 注意事项

- `.env`（含 API Key）、`任务调度引擎-双环决策架构.pptx`、`推进思路.md` 已在 `.gitignore` 中排除，不会被提交
- 内环本地 Laya 服务端点：`POST {base_url}/api/v1/predict`，body 格式与响应见 `jev/real_client.py` 注释
- 真实模式下 DeepSeek 偶发 503 `Service is too busy`，系统会自动 fallback 到"全员返航"安全降级（plan.assignments 为空）
- 全真实 Laya 本地服务可在 http://localhost:8000/docs 查 OpenAPI

---

## License

本仓库内容仅用于研发协作，请勿直接用于实战部署。
