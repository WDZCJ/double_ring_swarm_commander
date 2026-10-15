"""运行时交互 REST API（web/interaction.py）。

在系统运行过程中，外部可通过这些接口与"双环"系统交互：
- 选择/加载场景、启动任务（run/start）
- 下发新任务（intents，replace 语义）、注入临机事件
- 暂停/恢复/调速/停止、强制安全模式
- 管理定时监控任务、给单发巡飞弹下指令
- 轮询命令执行结果

设计约束（与 web/control.py 一致）：
- 写操作一律转成命令提交到 CommandBus，由 runner 线程在 tick 边界执行
- 只读接口读取 web.state 已发布的快照，绝不跨线程直接访问 runner 内部对象
- 运行时开关（pause/resume/stop/speed）直接操作 runner 标志，即时生效
- 不需要鉴权
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

from munition_system.web import state as web_state
from munition_system.web.control import CommandBus, CommandType, QueueFullError

router = APIRouter(prefix="/api", tags=["interaction"])

# 由 demo_web 启动时注入的会话对象（duck typing，见 demo_web.DemoSession）
_session: Any | None = None


def bind(session: Any) -> None:
    """绑定运行时会话（由 demos/demo_web.py 调用）。"""
    global _session
    _session = session


def get_session() -> Any | None:
    return _session


def _require_session() -> Any:
    if _session is None:
        raise HTTPException(
            status_code=503,
            detail="交互接口未初始化：请通过 python -m munition_system.demos.demo_web 启动",
        )
    return _session


def _require_runner() -> Any:
    session = _require_session()
    runner = getattr(session, "runner", None)
    if runner is None:
        raise HTTPException(status_code=409, detail="当前未加载场景/未创建运行器")
    return runner


def _bus() -> CommandBus:
    session = _require_session()
    bus = getattr(session, "bus", None)
    if bus is None:
        raise HTTPException(status_code=503, detail="命令总线不可用")
    return bus


async def _submit(ctype: CommandType, payload: dict, wait_ms: int = 0) -> dict:
    """提交命令；wait_ms>0 时同步等待结果（to_thread 避免阻塞事件循环）。"""
    bus = _bus()
    try:
        cmd = bus.submit(ctype, payload)
    except QueueFullError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if wait_ms and wait_ms > 0:
        done = await asyncio.to_thread(
            bus.wait, cmd.command_id, min(int(wait_ms), 30000) / 1000.0
        )
        if done is not None:
            return done.to_dict()
    return {"command_id": cmd.command_id, "type": cmd.type.value, "status": cmd.status}


def _snapshot() -> dict:
    return web_state.get_current_state()


# === 场景 / 运行控制 ===

@router.get("/scenarios")
async def list_scenarios() -> dict:
    """列出可加载的场景。"""
    session = _require_session()
    return {"scenarios": session.scenarios()}


@router.post("/scenarios/load")
async def load_scenario(body: dict = Body(default={})) -> dict:
    """加载（或切换）场景：重置世界与监控任务，创建新的运行器。

    body: {scenario: str, max_ticks?: int}
    """
    session = _require_session()
    key = str(body.get("scenario") or "").strip()
    if not key:
        raise HTTPException(status_code=422, detail="缺少 scenario")
    try:
        info = await asyncio.to_thread(session.load_scenario, key, body.get("max_ticks"))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"未知场景 {key}")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "loaded", **info}


@router.post("/run/start")
async def start_run(body: dict = Body(default={})) -> dict:
    """启动任务（在后台线程运行主循环）。

    body: {scenario?: str, intent_text?: str, max_ticks?: int}
    若给出 scenario，则先加载该场景再启动。
    """
    session = _require_session()
    scenario_key = body.get("scenario")
    if scenario_key:
        try:
            await asyncio.to_thread(session.load_scenario, str(scenario_key), body.get("max_ticks"))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"未知场景 {scenario_key}")
    try:
        info = await asyncio.to_thread(
            session.start, body.get("intent_text"), body.get("max_ticks")
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "started", **info}


# === 运行时开关 ===

@router.get("/runtime")
async def get_runtime() -> dict:
    return _snapshot().get("runtime", {})


@router.post("/runtime/pause")
async def pause() -> dict:
    runner = _require_runner()
    runner.request_pause()
    runner.publish()
    return {"status": "pausing", "runtime": _snapshot().get("runtime", {})}


@router.post("/runtime/resume")
async def resume() -> dict:
    runner = _require_runner()
    runner.request_resume()
    runner.publish()
    return {"status": "resuming", "runtime": _snapshot().get("runtime", {})}


@router.post("/runtime/stop")
async def stop() -> dict:
    runner = _require_runner()
    runner.request_stop()
    runner.request_resume()  # 解除暂停以便主循环尽快退出
    return {"status": "stopping"}


@router.post("/runtime/speed")
async def set_speed(body: dict = Body(default={})) -> dict:
    runner = _require_runner()
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="speed 必须是数字")
    value = runner.set_speed(speed)
    return {"status": "ok", "speed": value}


@router.post("/safe-mode")
async def safe_mode(body: dict = Body(default={})) -> dict:
    """强制进入安全模式（全员返航）。"""
    return await _submit(
        CommandType.SAFE_MODE,
        {"reason": body.get("reason") or "api_safe_mode"},
        int(body.get("wait_ms") or 0),
    )


# === 新任务 / 重规划 ===

@router.post("/intents")
async def new_intent(body: dict = Body(default={})) -> dict:
    """下发新任务（replace 语义：整体覆盖并重规划）。

    body: {raw_text|goal, constraints?, success_criteria?, mode:"replace", wait_ms?}
    """
    mode = str(body.get("mode") or "replace").lower()
    if mode != "replace":
        raise HTTPException(status_code=422, detail=f"不支持的 mode={mode}（当前仅支持 replace）")
    raw_text = str(body.get("raw_text") or body.get("goal") or "").strip()
    if not raw_text:
        raise HTTPException(status_code=422, detail="必须提供 raw_text 或 goal")
    payload = {
        "raw_text": raw_text,
        "goal": body.get("goal"),
        "constraints": body.get("constraints") or [],
        "success_criteria": body.get("success_criteria") or "",
    }
    return await _submit(CommandType.NEW_INTENT, payload, int(body.get("wait_ms") or 0))


@router.post("/replan")
async def replan(body: dict = Body(default={})) -> dict:
    """触发重规划（不改变意图，仅重新规划）。"""
    return await _submit(
        CommandType.REPLAN,
        {"reason": body.get("reason") or "api_replan"},
        int(body.get("wait_ms") or 0),
    )


@router.get("/intents")
async def list_intents() -> dict:
    """已下发的指挥员意图历史（含 replace 之后的新意图）。"""
    return {"intents": _snapshot().get("intents", [])}


@router.get("/plan")
async def get_plan() -> dict:
    return {"plan": _snapshot().get("plan"), "plan_version": _snapshot().get("plan_version", 0),
            "schema": _snapshot().get("schema")}


# === 事件注入 ===

@router.post("/events/inject")
async def inject_event(body: dict = Body(default={})) -> dict:
    """注入临机事件（damage / intel_update / comm_loss）。

    body: {event_type, tick?, munition_id?, munition_ids?, target_id?, status?,
           auto_replan?, wait_ms?}
    默认 tick = 当前 tick + 1。
    """
    payload = dict(body)
    if not (payload.get("event_type") or payload.get("type")):
        raise HTTPException(status_code=422, detail="缺少 event_type")
    return await _submit(CommandType.INJECT_EVENT, payload, int(body.get("wait_ms") or 0))


@router.get("/events")
async def list_events(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    """运行时事件日志（注入/触发/命令执行）。"""
    log = _snapshot().get("event_log", [])
    return {"events": log[-limit:]}


# === 监控任务管理 ===

@router.get("/monitoring/tasks")
async def list_tasks(
    status: str | None = Query(default=None),
    subject: str | None = Query(default=None),
) -> dict:
    tasks = _snapshot().get("monitoring_tasks", [])
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    if subject:
        tasks = [t for t in tasks if t.get("subject") == subject]
    return {"tasks": tasks}


@router.get("/telemetry")
async def get_telemetry() -> dict:
    return _snapshot().get("telemetry", {})


@router.get("/monitoring/tasks/{task_id}")
async def inspect_task(task_id: str) -> dict:
    for t in _snapshot().get("monitoring_tasks", []):
        if t.get("task_id") == task_id:
            return {"task": t}
    raise HTTPException(status_code=404, detail=f"监控任务 {task_id} 不存在")


@router.post("/monitoring/tasks")
async def create_task(body: dict = Body(default={})) -> dict:
    """创建定时监控任务。

    body: {subject, schema, schedule_every_ticks?, escalation_policy?, metadata?, wait_ms?}
    """
    if not body.get("subject") or not isinstance(body.get("schema"), dict):
        raise HTTPException(status_code=422, detail="subject 与 schema 必填")
    return await _submit(CommandType.MONITORING_CREATE, dict(body), int(body.get("wait_ms") or 0))


@router.post("/monitoring/tasks/{task_id}/pause")
async def pause_task(task_id: str, body: dict = Body(default={})) -> dict:
    return await _submit(CommandType.MONITORING_PAUSE, {"task_id": task_id},
                         int((body or {}).get("wait_ms") or 0))


@router.post("/monitoring/tasks/{task_id}/resume")
async def resume_task(task_id: str, body: dict = Body(default={})) -> dict:
    return await _submit(CommandType.MONITORING_RESUME, {"task_id": task_id},
                         int((body or {}).get("wait_ms") or 0))


@router.post("/monitoring/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, body: dict = Body(default={})) -> dict:
    return await _submit(CommandType.MONITORING_CANCEL, {"task_id": task_id},
                         int((body or {}).get("wait_ms") or 0))


@router.put("/monitoring/tasks/{task_id}/schema")
async def update_task_schema(task_id: str, body: dict = Body(default={})) -> dict:
    new_schema = body.get("new_schema") or body.get("schema")
    if not isinstance(new_schema, dict):
        raise HTTPException(status_code=422, detail="new_schema 必须是对象")
    return await _submit(
        CommandType.MONITORING_UPDATE_SCHEMA,
        {"task_id": task_id, "new_schema": new_schema},
        int(body.get("wait_ms") or 0),
    )


# === 单弹指令 ===

@router.post("/munitions/{munition_id}/command")
async def munition_command(munition_id: str, body: dict = Body(default={})) -> dict:
    """给指定巡飞弹下发动作指令。

    body: {action, args?, wait_ms?}
    """
    action = str(body.get("action") or "").strip()
    if not action:
        raise HTTPException(status_code=422, detail="缺少 action")
    return await _submit(
        CommandType.MUNITION_COMMAND,
        {"munition_id": munition_id, "action": action, "args": body.get("args") or {}},
        int(body.get("wait_ms") or 0),
    )


# === 通用命令 & 命令查询 ===

@router.post("/commands")
async def submit_command(body: dict = Body(default={})) -> dict:
    """通用命令入口。body: {type, payload?, wait_ms?}"""
    ctype = body.get("type")
    if not ctype:
        raise HTTPException(status_code=422, detail="缺少 type")
    try:
        ctype = CommandType(str(ctype))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"未知命令类型 {ctype}")
    payload = body.get("payload") or {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload 必须是对象")
    return await _submit(ctype, payload, int(body.get("wait_ms") or 0))


@router.get("/commands")
async def list_commands(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    return {"commands": _bus().recent(limit)}


@router.get("/commands/{command_id}")
async def get_command(command_id: str) -> dict:
    cmd = _bus().get(command_id)
    if cmd is None:
        raise HTTPException(status_code=404, detail=f"命令 {command_id} 不存在")
    return cmd.to_dict()
