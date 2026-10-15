"""三层联调主循环（runner/loop.py）。

DualLoopRunner 把仿真层（simulation）、内环决策层（jev）、外环 Agent 层（agent）
串起来，逐 tick 推进并收集 telemetry。

流程：
1. 装配：World + SituationalApi + Jev(client/routing/orchestrator) +
   MonitoringTaskManager + LLM + Tools + ReActLoop + SchemaDesigner +
   PlanTracker + Replanner
2. ReAct 规划：调 ReActLoop.run 产出 BattlePlan
3. 部署：assign_tasks + 全局 schema
4. Tick 循环：
   a. 内环决策（全局 schema + 监控任务）→ ActionCommand[] + Escalation[]
   b. 应用动作 + 推进仿真
   c. 检查进度（plan_tracker）
   d. 处理 escalation（重规划 or 再 ReAct）
   e. 收集 telemetry
5. 返回 RunnerResult

支持 mock 模式（无 LLM key/无网络）：llm_responses 注入预制响应序列。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from munition_system.agent.llm_client import MockLLMClient, make_llm_client
from munition_system.agent.monitoring_manager import MonitoringTaskManagerImpl
from munition_system.agent.plan_tracker import PlanTracker
from munition_system.agent.react_loop import ReActLoop
from munition_system.agent.replanner import Replanner, SafeModeTriggered
from munition_system.agent.schema_designer import SchemaDesigner
from munition_system.agent.tools import build_all_tools
from munition_system.common.config import Config
from munition_system.common.contracts import LLMResponse, Tool, ToolContext
from munition_system.common.types import (
    ActionCommand,
    BattlePlan,
    CommLossEvent,
    CommanderIntent,
    DamageEvent,
    Escalation,
    Event,
    IntelUpdateEvent,
    MunitionState,
)
from munition_system.jev.client import make_client
from munition_system.jev.orchestrator import DecisionOrchestrator
from munition_system.jev.routing import Routing
from munition_system.runner.scenario import Scenario
from munition_system.runner.telemetry import Telemetry
from munition_system.simulation.munition import apply_action
from munition_system.simulation.situational_api import MockSituationalApi
from munition_system.simulation.world import World
from munition_system.web.control import Command, CommandBus, CommandType


@dataclass
class RunnerResult:
    """运行结果。"""
    plan: BattlePlan | None = None
    final_phase: str = "done"
    replan_count: int = 0
    escalation_count: int = 0
    safe_mode: bool = False
    warnings: list[str] = field(default_factory=list)
    telemetry: Telemetry | None = None


class DualLoopRunner:
    """三层联调运行器。"""

    def __init__(
        self,
        config: Config,
        scenario: Scenario,
        *,
        llm_responses: list[LLMResponse] | None = None,
        control: CommandBus | None = None,
        event_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self._config = config
        self._scenario = scenario
        self._telemetry = Telemetry()

        # 1) 仿真层
        self._world = World(
            grid_dim=scenario.grid_dim,
            seed=config.sim.seed,
            munitions=scenario.munitions,
            threats=scenario.threats,
            zones=scenario.zones,
        )
        # 注入事件
        for ev in scenario.events:
            self._world.event_injector.schedule(ev)
        self._situational_api = MockSituationalApi(self._world)

        # 2) 内环 Jev
        self._jev_client = make_client(config.jev)
        self._routing = Routing(config.jev.routing)

        # 3) 监控任务管理器（带 escalation 回调记录 telemetry）
        self._monitoring_manager = MonitoringTaskManagerImpl(
            max_concurrent_tasks=config.agent.monitoring.max_concurrent_tasks,
            on_escalation=self._on_task_escalation,
        )

        self._jev_orchestrator = DecisionOrchestrator(
            client=self._jev_client,
            situational_api=self._situational_api,
            routing=self._routing,
            monitoring_manager=self._monitoring_manager,
        )

        # 4) LLM（mock 注入预制响应 或 real）
        if llm_responses is not None:
            self._llm = MockLLMClient()
            self._llm.set_next_responses(llm_responses)
        else:
            self._llm = make_llm_client(config.llm)

        # 5) Replanner
        self._replanner = Replanner(
            self._monitoring_manager,
            safe_mode_after_consecutive_failures=config.agent.replanner.safe_mode_after_consecutive_failures,
        )

        # 6) Tools（包装一层记录 telemetry）
        raw_tools = build_all_tools(
            jev_orchestrator=self._jev_orchestrator,
            monitoring_manager=self._monitoring_manager,
            replanner=self._replanner,
        )
        self._tools = [_TelemetryTool(t, self._telemetry) for t in raw_tools]

        # 7) Agent 组件
        self._react_loop = ReActLoop(
            llm=self._llm,
            tools=self._tools,
            monitoring_manager=self._monitoring_manager,
            max_iterations=config.llm.react_loop.max_iterations,
            fallback_to_safe_intent=config.llm.react_loop.fallback_to_safe_intent,
        )
        self._schema_designer = SchemaDesigner(self._llm)
        check_tool = next((t for t in self._tools if t.name == "check_task_progress"), None)
        self._plan_tracker = PlanTracker(
            check_tool,  # type: ignore[arg-type]
            check_interval_ticks=config.agent.plan_tracker.check_interval_ticks,
        )

        # 运行状态
        self._plan: BattlePlan | None = None
        self._pending_escalations: list[Escalation] = []
        self._replan_count = 0
        self._safe_mode = False
        self._warnings: list[str] = []

        # === 运行时交互控制（可选；不传则行为与原先完全一致）===
        self._control = control
        self._event_sink = event_sink
        self._paused = False
        self._stop_requested = False
        self._running = False
        self._speed = 1.0
        self._current_tick = 0
        self._phase_label = "init"
        self._intents: list[dict] = []
        self._event_log: deque[dict] = deque(maxlen=200)
        self._scenario_name = getattr(scenario, "name", "") or ""
        # 事件驱动重规划：注入破坏性事件后，待其应用到世界再触发重规划
        self._pending_event_replan: str | None = None
        self._event_replan_ready = False
        # 监听仿真事件触发（用于事件日志与前端推送）
        self._world.event_injector.add_listener(self._on_event_fired)

    @property
    def telemetry(self) -> Telemetry:
        return self._telemetry

    # === 运行时控制（线程安全：API 线程写标志，runner 线程读）===

    @property
    def control(self) -> CommandBus | None:
        return self._control

    @property
    def plan(self) -> BattlePlan | None:
        return self._plan

    @property
    def safe_mode(self) -> bool:
        return self._safe_mode

    @property
    def replan_count(self) -> int:
        return self._replan_count

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def running(self) -> bool:
        return self._running

    @property
    def speed(self) -> float:
        return self._speed

    def request_pause(self) -> None:
        self._paused = True

    def request_resume(self) -> None:
        self._paused = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def set_speed(self, value: float) -> float:
        """设置运行倍速（0.1x ~ 20x）。仅影响 demo 的 tick 间隔。"""
        self._speed = max(0.1, min(20.0, float(value)))
        return self._speed

    def publish(self) -> None:
        """主动发布快照并向前端推送一帧态势（场景加载后、运行开始时调用）。"""
        self._publish_snapshot()
        self._emit_situation_frame()

    # === 主入口 ===

    def run(
        self,
        intent: CommanderIntent,
        *,
        max_ticks: int = 100,
    ) -> RunnerResult:
        """运行三层联调。"""
        self._running = True
        try:
            if not self._intents:
                self._record_intent(intent, tick=0, plan_version=1)

            # 0) 先把初始态势推给前端（地图/弹位/区域/威胁），
            #    避免在 LLM 规划期间界面空白、看不出任务已启动
            self._set_phase("初始态势")
            self.publish()

            # 1) ReAct 规划
            self._set_phase("ReAct 规划中")
            ctx = self._make_ctx(tick=0)
            result = self._react_loop.run(intent, ctx, plan_version=1)
            if result.plan is None:
                self._warnings.append("ReAct 未产出 plan")
                return self._build_result("react_failed")
            self._plan = result.plan
            self._warnings.extend(result.warnings)

            # 2) 部署
            self._set_phase("部署中")
            self._deploy_plan(ctx)
            self._set_phase("运行中")
            self._publish_snapshot()

            # 3) Tick 循环
            for tick in range(max_ticks):
                # 暂停门：暂停时轮询等待，仍可消费命令（可下发新任务/注入事件）与停止
                if self._wait_while_paused():
                    return self._build_result("stopped")
                if self._stop_requested:
                    return self._build_result("stopped")

                self._current_tick = tick
                ctx = self._make_ctx(tick=tick)
                self._drain_commands(ctx)
                if self._stop_requested:
                    return self._build_result("stopped")

                done = self._tick(ctx)
                self._publish_snapshot()
                if done:
                    return self._build_result("done")
                if self._safe_mode:
                    return self._build_result("safe_mode")

            return self._build_result("max_ticks_reached")
        finally:
            self._running = False
            self._publish_snapshot()

    def _wait_while_paused(self) -> bool:
        """暂停时轮询等待。返回 True 表示收到停止请求。

        暂停期间仍会 drain 命令（允许用户下发新意图/注入事件），因此"交互"
        在暂停状态下依然可用。
        """
        if not self._paused:
            return False
        self._set_phase("已暂停")
        while self._paused and not self._stop_requested:
            self._drain_commands(self._make_ctx(tick=self._current_tick))
            self._publish_snapshot()
            time.sleep(0.1)
        if not self._stop_requested:
            self._set_phase("运行中")
        return self._stop_requested

    # === 部署 ===

    def _deploy_plan(self, ctx: ToolContext) -> None:
        plan = self._plan
        if plan is None:
            return
        # assign_tasks
        assign_tool = self._get_tool("assign_tasks")
        if assign_tool is not None:
            assign_tool.call(
                {"assignments": [a.__dict__ for a in plan.assignments]},
                ctx,
            )
        # 全局 schema
        ws_summary = self._situational_api.get_subject_state("global")
        schema = self._schema_designer.design(plan, ws_summary, version=1)
        self._jev_orchestrator.deploy_global_schema(schema)
        self._replanner.set_plan_version(plan.plan_version)

    # === 单 tick ===

    def _tick(self, ctx: ToolContext) -> bool:
        """推进一个 tick。返回 True 表示任务完成。"""
        tick = ctx.tick

        # 1) 内环决策
        inner_actions, escalations = self._jev_orchestrator.tick(tick)
        self._pending_escalations.extend(escalations)

        # 2) 应用动作 + 推进仿真
        ws_before = self._situational_api.get_world_state()
        self._world.step(inner_actions)
        ws_after = self._situational_api.get_world_state()

        # 3) 收集本 tick 的 Jev confidence（从 escalation 的 low_confidence_answers 提取）
        confidences: list[float] = []
        for e in escalations:
            for a in e.low_confidence_answers:
                confidences.append(a.confidence)

        # 4) telemetry
        self._telemetry.record_tick(
            tick=tick,
            actions=inner_actions,
            escalations=escalations,
            world_summary=_world_summary(ws_after),
            confidences=confidences,
        )

        # 4.5) 注入的破坏性事件已在本 tick 应用到世界 → 事件驱动重规划
        if self._event_replan_ready:
            self._event_replan_ready = False
            reason = self._pending_event_replan or "event_injected"
            self._pending_event_replan = None
            self._try_event_replan(reason, ctx)
            return False

        # 5) 检查进度
        if self._plan is not None:
            report = self._plan_tracker.track(tick, self._plan, ctx)
            if report is not None:
                if report.failed:
                    self._handle_replan(f"progress_failed:{report.reason}", ctx)
                    return False
                if report.stuck:
                    self._handle_replan(f"progress_stuck:{report.reason}", ctx)
                    return False
                if report.completion >= 1.0:
                    return True

        # 6) 处理 escalation
        if self._pending_escalations:
            self._handle_escalations(ctx)

        return False

    # === escalation 处理 ===

    def _handle_escalations(self, ctx: ToolContext) -> None:
        has_task_esc = any(
            e.reason == "monitoring_task_escalation"
            for e in self._pending_escalations
        )
        if has_task_esc:
            self._handle_replan("monitoring_task_escalation", ctx)
            return
        # 低置信/边界态 → 再 ReAct
        if self._plan is not None:
            ctx2 = self._make_ctx(tick=ctx.tick)
            result = self._react_loop.run(
                self._plan.intent,
                ctx2,
                pending_escalations=self._pending_escalations,
                plan_version=self._plan.plan_version,
            )
            if result.plan is not None:
                self._plan = result.plan
                self._deploy_plan(ctx2)
        self._pending_escalations.clear()

    # === 重规划 ===

    def _handle_replan(self, reason: str, ctx: ToolContext) -> None:
        self._replan_count += 1
        consecutive = self._replanner.consecutive_failures + 1
        try:
            new_version = self._replanner.trigger(reason, ctx=ctx)
        except SafeModeTriggered as e:
            self._safe_mode = True
            self._warnings.append(str(e))
            actions = self._replanner.enter_safe_mode(ctx)
            self._world.step(actions)
            self._telemetry.record_replan(ctx.tick, reason, new_version=0, consecutive_failures=consecutive)
            return

        self._telemetry.record_replan(ctx.tick, reason, new_version, consecutive)

        if self._plan is not None:
            ctx2 = self._make_ctx(tick=ctx.tick)
            result = self._react_loop.run(
                self._plan.intent,
                ctx2,
                pending_escalations=self._pending_escalations,
                plan_version=new_version,
            )
            if result.plan is not None:
                self._plan = result.plan
                self._replanner.on_plan_success()
                self._deploy_plan(ctx2)
            else:
                self._warnings.append(f"重规划失败: {reason}")
        self._pending_escalations.clear()

    # === 辅助 ===

    def _make_ctx(self, *, tick: int) -> ToolContext:
        return ToolContext(
            tick=tick,
            plan_version=self._plan.plan_version if self._plan else None,
            world=self._world,
            situational_api=self._situational_api,
            monitoring_manager=self._monitoring_manager,
            plan=self._plan,
        )

    def _get_tool(self, name: str):
        for t in self._tools:
            if t.name == name:
                return t
        return None

    def _on_task_escalation(self, task_id: str, answer, tick: int) -> None:
        """监控任务 escalation 回调：记录 telemetry。"""
        self._telemetry.record_task_change(
            tick=tick,
            task_id=task_id,
            old_status="active",
            new_status="escalated",
            reason="escalation_policy_triggered",
        )

    def _build_result(self, phase: str) -> RunnerResult:
        self._telemetry.set_final_status({
            "phase": phase,
            "safe_mode": self._safe_mode,
            "replan_count": self._replan_count,
        })
        self._set_phase(f"结束: {phase}")
        return RunnerResult(
            plan=self._plan,
            final_phase=phase,
            replan_count=self._replan_count,
            escalation_count=self._telemetry.escalation_count,
            safe_mode=self._safe_mode,
            warnings=list(self._warnings),
            telemetry=self._telemetry,
        )

    # === 运行时命令处理（仅 runner 线程调用）===

    def _drain_commands(self, ctx: ToolContext) -> None:
        """在 tick 边界消费命令队列并执行（每条命令被 applied/rejected 记录）。"""
        if self._control is None:
            return
        for cmd in self._control.drain(max_heavy=1):
            try:
                result = self._apply_command(cmd, ctx)
                self._control.complete(
                    cmd, status="applied", result=result, applied_tick=ctx.tick,
                )
                self._log_event({
                    "type": "command",
                    "command_id": cmd.command_id,
                    "command_type": cmd.type.value,
                    "status": "applied",
                    "tick": ctx.tick,
                    "result": result,
                })
            except Exception as e:  # noqa: BLE001 - 单条命令失败不影响主循环
                self._control.complete(
                    cmd, status="rejected", error=str(e), applied_tick=ctx.tick,
                )
                self._log_event({
                    "type": "command",
                    "command_id": cmd.command_id,
                    "command_type": cmd.type.value,
                    "status": "rejected",
                    "tick": ctx.tick,
                    "error": str(e),
                })

    def _apply_command(self, cmd: Command, ctx: ToolContext) -> dict:
        """分发单条命令到对应 handler。返回 result dict（失败时抛异常）。"""
        handler = {
            CommandType.NEW_INTENT: self._cmd_new_intent,
            CommandType.REPLAN: self._cmd_replan,
            CommandType.INJECT_EVENT: self._cmd_inject_event,
            CommandType.MONITORING_CREATE: self._cmd_monitoring_create,
            CommandType.MONITORING_PAUSE: self._cmd_monitoring_pause,
            CommandType.MONITORING_RESUME: self._cmd_monitoring_resume,
            CommandType.MONITORING_CANCEL: self._cmd_monitoring_cancel,
            CommandType.MONITORING_UPDATE_SCHEMA: self._cmd_monitoring_update_schema,
            CommandType.MUNITION_COMMAND: self._cmd_munition_command,
            CommandType.SAFE_MODE: self._cmd_safe_mode,
        }[cmd.type]
        return handler(cmd.payload, ctx)

    # --- 新任务（replace 语义）---

    def _cmd_new_intent(self, payload: dict, ctx: ToolContext) -> dict:
        """下发新任务：整体覆盖并重规划（plan_version+1，旧监控任务自动取消）。

        旧意图保留在 self._intents 历史中以便审计。
        """
        intent = _parse_intent(payload)
        new_version = self._replanner.trigger_external("external_new_intent", ctx=ctx)
        self._record_intent(intent, tick=ctx.tick, plan_version=new_version)

        ctx2 = self._make_ctx(tick=ctx.tick)
        result = self._react_loop.run(intent, ctx2, plan_version=new_version)
        if result.plan is None:
            raise RuntimeError("新意图重规划未产出 plan（ReAct 失败）")
        self._plan = result.plan
        self._warnings.extend(result.warnings)
        self._deploy_plan(ctx2)
        self._log_event({
            "type": "intent_replaced",
            "tick": ctx.tick,
            "plan_version": new_version,
            "raw_text": intent.raw_text,
        })
        return {
            "plan_version": new_version,
            "assignments": len(self._plan.assignments),
            "raw_text": intent.raw_text,
        }

    # --- 重规划 ---

    def _cmd_replan(self, payload: dict, ctx: ToolContext) -> dict:
        if self._plan is None:
            raise RuntimeError("尚未产出 plan，无法重规划")
        reason = str(payload.get("reason") or "external_replan")
        new_version = self._replanner.trigger_external(reason, ctx=ctx)
        self._replan_count += 1
        ctx2 = self._make_ctx(tick=ctx.tick)
        result = self._react_loop.run(
            self._plan.intent,
            ctx2,
            pending_escalations=list(self._pending_escalations),
            plan_version=new_version,
        )
        if result.plan is None:
            raise RuntimeError("重规划未产出 plan（ReAct 失败）")
        self._plan = result.plan
        self._warnings.extend(result.warnings)
        self._deploy_plan(ctx2)
        self._log_event({
            "type": "replanned",
            "tick": ctx.tick,
            "reason": reason,
            "plan_version": new_version,
        })
        return {"plan_version": new_version, "reason": reason}

    # --- 注入事件 ---

    def _cmd_inject_event(self, payload: dict, ctx: ToolContext) -> dict:
        ev = _parse_event(payload, default_tick=ctx.tick + 1)
        self._world.event_injector.schedule(ev)
        info = _event_info(ev)
        self._log_event({"type": "event_injected", "tick": ctx.tick, **info})

        # 破坏性临机事件（弹损毁/通信失联）默认触发事件驱动重规划：
        # 待事件在下一 tick 应用到世界后，再据此重规划（不能在此刻重规划，
        # 否则 LLM 看到的是事件生效前的态势）。
        disruptive = isinstance(ev, (DamageEvent, CommLossEvent))
        auto = bool(payload.get("auto_replan")) or disruptive
        out: dict = {"injected": info, "auto_replan": auto, "replan_armed": auto}
        if auto:
            self._pending_event_replan = f"event_injected:{info['event_type']}"
            self._log_event({
                "type": "replan_armed",
                "tick": ctx.tick,
                "reason": self._pending_event_replan,
            })
        return out

    def _try_event_replan(self, reason: str, ctx: ToolContext) -> None:
        """事件驱动重规划（绕过 demo 的自动重规划抑制，走外部重规划路径）。"""
        if self._plan is None:
            return
        try:
            self._cmd_replan({"reason": reason}, ctx)
        except Exception as e:  # noqa: BLE001 - 重规划失败不阻塞主循环
            self._warnings.append(f"事件驱动重规划失败: {reason}: {e}")

    # --- 监控任务管理（复用已注册工具，保证与 Agent 侧行为一致）---

    def _cmd_monitoring_create(self, payload: dict, ctx: ToolContext) -> dict:
        return self._call_tool("create_monitoring_task", payload, ctx)

    def _cmd_monitoring_pause(self, payload: dict, ctx: ToolContext) -> dict:
        return self._call_tool("pause_monitoring_task", payload, ctx)

    def _cmd_monitoring_resume(self, payload: dict, ctx: ToolContext) -> dict:
        return self._call_tool("resume_monitoring_task", payload, ctx)

    def _cmd_monitoring_cancel(self, payload: dict, ctx: ToolContext) -> dict:
        return self._call_tool("cancel_monitoring_task", payload, ctx)

    def _cmd_monitoring_update_schema(self, payload: dict, ctx: ToolContext) -> dict:
        task_id = str(payload.get("task_id", ""))
        args = {"task_id": task_id, "new_schema": payload.get("new_schema") or payload.get("schema")}
        return self._call_tool("update_monitoring_task_schema", args, ctx)

    # --- 单弹指令 ---

    def _cmd_munition_command(self, payload: dict, ctx: ToolContext) -> dict:
        munition_id = str(payload.get("munition_id", ""))
        action = str(payload.get("action", ""))
        if not munition_id or not action:
            raise ValueError("munition_id 与 action 必填")
        m = self._world.munitions.get(munition_id)
        if m is None:
            raise KeyError(f"巡飞弹 {munition_id} 不存在")
        cmd = ActionCommand(
            munition_id=munition_id,
            action=action,
            args=dict(payload.get("args") or {}),
        )
        new_m, warns = apply_action(m, cmd)
        self._world.munitions[munition_id] = new_m
        self._log_event({
            "type": "munition_command",
            "tick": ctx.tick,
            "munition_id": munition_id,
            "action": action,
            "state": new_m.state.value,
        })
        return {
            "munition_id": munition_id,
            "action": action,
            "state": new_m.state.value,
            "warnings": [w.message for w in warns],
        }

    # --- 安全模式 ---

    def _cmd_safe_mode(self, payload: dict, ctx: ToolContext) -> dict:
        self._safe_mode = True
        actions = self._replanner.enter_safe_mode(ctx)
        self._world.step(actions)
        reason = str(payload.get("reason") or "external_safe_mode")
        self._warnings.append(f"外部触发安全模式：{reason}")
        self._log_event({
            "type": "safe_mode",
            "tick": ctx.tick,
            "reason": reason,
            "actions": len(actions),
        })
        return {"safe_mode": True, "actions": len(actions), "reason": reason}

    def _call_tool(self, name: str, args: dict, ctx: ToolContext) -> dict:
        tool = self._get_tool(name)
        if tool is None:
            raise RuntimeError(f"工具 {name} 未注册")
        res = tool.call(dict(args), ctx)
        if not res.ok:
            raise RuntimeError(res.error or f"工具 {name} 执行失败")
        return res.data if isinstance(res.data, dict) else {"result": res.data}

    # === 快照发布 & 事件日志 ===

    def _set_phase(self, phase: str) -> None:
        self._phase_label = phase

    def _record_intent(self, intent: CommanderIntent, *, tick: int, plan_version: int) -> None:
        from munition_system.web import state as web_state
        self._intents.append(web_state.serialize_intent(
            intent, tick=tick, plan_version=plan_version,
        ))

    def _log_event(self, data: dict) -> None:
        self._event_log.append(data)
        if self._event_sink is not None:
            try:
                self._event_sink(dict(data))
            except Exception:  # noqa: BLE001 - 推送失败不影响主循环
                pass

    def _on_event_fired(self, ev: Event) -> None:
        """仿真事件触发回调（EventInjector 监听器）。"""
        self._log_event({"type": "event_fired", **_event_info(ev)})
        # 已排队的破坏性事件应用到世界后，标记可以触发事件驱动重规划
        if self._pending_event_replan is not None and isinstance(ev, (DamageEvent, CommLossEvent)):
            self._event_replan_ready = True

    def _runtime_dict(self) -> dict:
        return {
            "running": self._running,
            "paused": self._paused,
            "safe_mode": self._safe_mode,
            "speed": self._speed,
            "stop_requested": self._stop_requested,
        }

    def _publish_snapshot(self) -> None:
        """把当前状态快照发布到 web.state（供 REST 只读端点使用）。"""
        if self._control is None:
            return
        from munition_system.web import state as web_state
        try:
            ws = self._situational_api.get_world_state()
            tasks = [
                web_state.serialize_monitoring_task(t)
                for t in self._monitoring_manager.list_monitoring_tasks()
            ]
            web_state.update_state(
                tick=self._current_tick,
                phase=self._phase_label,
                plan_version=self._plan.plan_version if self._plan else 0,
                replan_count=self._replan_count,
                scenario=self._scenario_name,
                world=web_state.serialize_world(ws),
                monitoring_tasks=tasks,
                plan=web_state.serialize_plan(self._plan),
                schema=web_state.serialize_schema(self._jev_orchestrator.get_global_schema()),
                intents=list(self._intents),
                event_log=list(self._event_log),
                telemetry=self._telemetry.summary(),
                runtime=self._runtime_dict(),
            )
        except Exception:  # noqa: BLE001 - 快照发布失败不影响主循环
            pass

    def _emit_situation_frame(self) -> None:
        """经 event_sink 推送一帧 tick 态势（地图/弹位/区域/威胁）到前端。

        与 _publish_snapshot 的区别：后者只更新 web REST 快照，本方法会主动
        通过 WebSocket 广播，使前端立刻绘制当前态势（无需等待 LLM 规划与首个 tick）。
        """
        if self._event_sink is None:
            return
        from munition_system.web import state as web_state
        try:
            ws = self._situational_api.get_world_state()
            tasks = [
                web_state.serialize_monitoring_task(t)
                for t in self._monitoring_manager.list_monitoring_tasks()
            ]
            self._event_sink({
                "type": "tick",
                "tick": self._current_tick,
                "plan_version": self._plan.plan_version if self._plan else 0,
                "replan_count": self._replan_count,
                "world": web_state.serialize_world(ws),
                "monitoring_tasks": tasks,
            })
        except Exception:  # noqa: BLE001 - 推送失败不影响主循环
            pass



# === Tool 包装器：记录 telemetry ===

class _TelemetryTool:
    """包装 Tool，记录每次调用到 telemetry。"""

    def __init__(self, tool: Tool, telemetry: Telemetry) -> None:
        self._tool = tool
        self._telemetry = telemetry
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.input_schema

    def call(self, args: dict, ctx: ToolContext):
        result = self._tool.call(args, ctx)
        self._telemetry.record_tool_call(
            tick=ctx.tick,
            name=self.name,
            args=args,
            ok=result.ok,
            error=result.error,
        )
        return result


def _world_summary(ws) -> dict:
    """紧凑的世界状态摘要。"""
    state_counts: dict[str, int] = {}
    for m in ws.munitions:
        s = m.state.value
        state_counts[s] = state_counts.get(s, 0) + 1
    return {
        "tick": ws.tick,
        "n_munitions": len(ws.munitions),
        "n_threats": len(ws.threats),
        "states": state_counts,
        "zones": [{"id": z.id, "cov": round(z.coverage, 2)} for z in ws.zones],
    }


# === 运行时命令辅助 ===

def _parse_intent(payload: dict) -> CommanderIntent:
    """从命令 payload 解析指挥员意图（raw_text 或 goal 至少一个）。"""
    raw_text = str(payload.get("raw_text") or payload.get("goal") or "").strip()
    if not raw_text:
        raise ValueError("必须提供 raw_text 或 goal")
    goal = str(payload.get("goal") or raw_text).strip()
    constraints = payload.get("constraints") or []
    if isinstance(constraints, str):
        constraints = [constraints]
    return CommanderIntent(
        raw_text=raw_text,
        goal=goal,
        constraints=[str(c) for c in constraints],
        success_criteria=str(payload.get("success_criteria") or ""),
    )


def _parse_event(payload: dict, *, default_tick: int) -> Event:
    """从 payload 解析仿真事件。event_type ∈ damage | intel_update | comm_loss。"""
    etype = str(payload.get("event_type") or payload.get("type") or "").lower()
    tick = int(payload.get("tick", default_tick))
    tick = max(0, tick)
    if etype == "damage":
        mid = payload.get("munition_id")
        if not mid:
            raise ValueError("damage 事件需要 munition_id")
        return DamageEvent(munition_id=str(mid), tick=tick)
    if etype == "intel_update":
        return IntelUpdateEvent(
            target_id=payload.get("target_id"),
            status=str(payload.get("status") or ""),
            tick=tick,
        )
    if etype == "comm_loss":
        ids = payload.get("munition_ids") or []
        if not ids:
            raise ValueError("comm_loss 事件需要 munition_ids")
        return CommLossEvent(munition_ids=[str(x) for x in ids], tick=tick)
    raise ValueError(f"未知事件类型: {etype!r}（支持 damage/intel_update/comm_loss）")


def _event_info(ev: Event) -> dict:
    """事件的可序列化摘要。"""
    return {
        "event_type": type(ev).__name__,
        "tick": getattr(ev, "tick", None),
        "munition_id": getattr(ev, "munition_id", None),
        "munition_ids": list(getattr(ev, "munition_ids", []) or []),
        "target_id": getattr(ev, "target_id", None),
        "status": getattr(ev, "status", None),
    }
