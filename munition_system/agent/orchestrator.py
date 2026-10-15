"""外环 Agent 顶层编排器（agent/orchestrator.py）。

状态机：
PARSE_INTENT → REACT_LOOP → DEPLOY_SCHEMA/CREATE_MONITORING_TASKS → MONITOR_LOOP
                    ↑                                            ↓
                    └──────────── REPLAN ←── ESCALATION/EVENT ───┘

职责：
- 解析指挥员意图
- 运行 ReAct 循环产出 BattlePlan
- 部署全局 schema + 创建监控任务
- tick 循环：推进仿真 + 执行内环 + 检查进度 + 处理升级/重规划
- 安全模式：连续 5 次重规划失败 → 全员 RTB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from munition_system.common.contracts import (
    SituationalApi,
    ToolContext,
)
from munition_system.common.types import (
    ActionCommand,
    BattlePlan,
    CommanderIntent,
    Escalation,
    Event,
    MunitionState,
)

if TYPE_CHECKING:
    from munition_system.agent.monitoring_manager import MonitoringTaskManagerImpl
    from munition_system.agent.plan_tracker import PlanTracker
    from munition_system.agent.react_loop import ReActLoop
    from munition_system.agent.replanner import Replanner, SafeModeTriggered
    from munition_system.agent.schema_designer import SchemaDesigner
    from munition_system.common.contracts import Tool
    from munition_system.jev.orchestrator import DecisionOrchestrator
    from munition_system.simulation.world import World


class AgentPhase(str, Enum):
    PARSE_INTENT = "parse_intent"
    REACT_LOOP = "react_loop"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    REPLAN = "replan"
    SAFE_MODE = "safe_mode"
    DONE = "done"


@dataclass
class AgentRunResult:
    """Agent 运行结果。"""
    plan: BattlePlan | None = None
    final_phase: AgentPhase = AgentPhase.DONE
    replan_count: int = 0
    escalation_count: int = 0
    safe_mode: bool = False
    warnings: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """外环 Agent 顶层编排器。"""

    def __init__(
        self,
        *,
        world: "World",
        situational_api: SituationalApi,
        jev_orchestrator: "DecisionOrchestrator",
        monitoring_manager: "MonitoringTaskManagerImpl",
        llm: Any,
        tools: list["Tool"],
        react_loop: "ReActLoop",
        schema_designer: "SchemaDesigner",
        plan_tracker: "PlanTracker",
        replanner: "Replanner",
    ) -> None:
        self._world = world
        self._situational_api = situational_api
        self._jev_orchestrator = jev_orchestrator
        self._monitoring_manager = monitoring_manager
        self._llm = llm
        self._tools = tools
        self._react_loop = react_loop
        self._schema_designer = schema_designer
        self._plan_tracker = plan_tracker
        self._replanner = replanner

        self._phase = AgentPhase.PARSE_INTENT
        self._plan: BattlePlan | None = None
        self._pending_escalations: list[Escalation] = []
        self._replan_count = 0
        self._escalation_count = 0
        self._safe_mode = False
        self._warnings: list[str] = []

        # 注册 escalation 回调
        self._monitoring_manager._on_escalation = self._on_escalation  # type: ignore[attr-defined]

    # === 主入口 ===

    def run(
        self,
        intent: CommanderIntent,
        *,
        max_ticks: int = 100,
    ) -> AgentRunResult:
        """运行 Agent 主循环。

        流程：
        1. REACT_LOOP 产出 BattlePlan
        2. DEPLOY：全局 schema + 监控任务
        3. MONITOR_LOOP：逐 tick 推进，处理 escalation / progress / replan
        """
        # 1) ReAct 规划
        self._phase = AgentPhase.REACT_LOOP
        ctx = self._make_ctx(tick=0)
        result = self._react_loop.run(intent, ctx, plan_version=1)
        if result.plan is None:
            self._warnings.append("ReAct 未产出 plan")
            return self._build_result()
        self._plan = result.plan
        self._warnings.extend(result.warnings)

        # 2) 部署：assign_tasks + 全局 schema
        self._phase = AgentPhase.DEPLOY
        self._deploy_plan(ctx)

        # 3) 监控循环
        self._phase = AgentPhase.MONITOR
        for tick in range(max_ticks):
            ctx = self._make_ctx(tick=tick)
            done = self._monitor_tick(ctx)
            if done:
                break
            if self._safe_mode:
                break

        return self._build_result()

    # === 部署 ===

    def _deploy_plan(self, ctx: ToolContext) -> None:
        """应用 plan：assign_tasks + 部署全局 schema。"""
        plan = self._plan
        if plan is None:
            return
        # assign_tasks（通过工具调用，复用 simulation 工具）
        assign_tool = self._get_tool("assign_tasks")
        if assign_tool is not None:
            assign_tool.call(
                {"assignments": [a.__dict__ for a in plan.assignments]},
                ctx,
            )
        # 设计并部署全局 schema
        ws_summary = self._situational_api.get_subject_state("global")
        schema = self._schema_designer.design(plan, ws_summary, version=1)
        self._jev_orchestrator.deploy_global_schema(schema)
        # 设置 replanner 当前 plan version
        self._replanner.set_plan_version(plan.plan_version)

    # === 单 tick 监控 ===

    def _monitor_tick(self, ctx: ToolContext) -> bool:
        """推进一个 tick。返回 True 表示任务完成。"""
        tick = ctx.tick

        # 1) 内环决策：全局 schema + 监控任务
        inner_actions, escalations = self._jev_orchestrator.tick(tick)
        for e in escalations:
            self._pending_escalations.append(e)
            self._escalation_count += 1

        # 2) 应用内环动作 + 推进仿真
        self._world.step(inner_actions)

        # 3) 检查任务进度
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
                    self._phase = AgentPhase.DONE
                    return True

        # 4) 处理待处理 escalation（触发重规划或 ReAct 介入）
        if self._pending_escalations:
            self._handle_escalations(ctx)

        return False

    # === 升级处理 ===

    def _handle_escalations(self, ctx: ToolContext) -> None:
        """处理待处理 escalation。

        - 若有 monitoring_task_escalation → 触发重规划
        - 否则 → 再跑一次 ReAct（让 LLM 看 escalation 后决策）
        """
        has_task_esc = any(
            e.reason == "monitoring_task_escalation"
            for e in self._pending_escalations
        )
        if has_task_esc:
            self._handle_replan("monitoring_task_escalation", ctx)
            return
        # 低置信/边界态 escalation → 让 Agent 再思考
        ctx2 = self._make_ctx(tick=ctx.tick)
        if self._plan is not None:
            result = self._react_loop.run(
                self._plan.intent,
                ctx2,
                pending_escalations=self._pending_escalations,
                plan_version=self._plan.plan_version,
            )
            if result.plan is not None and result.plan is not self._plan:
                self._plan = result.plan
                self._deploy_plan(ctx2)
        self._pending_escalations.clear()

    # === 重规划 ===

    def _handle_replan(self, reason: str, ctx: ToolContext) -> None:
        """触发重规划。"""
        self._phase = AgentPhase.REPLAN
        self._replan_count += 1
        try:
            new_version = self._replanner.trigger(reason, ctx=ctx)
        except SafeModeTriggered as e:
            self._safe_mode = True
            self._phase = AgentPhase.SAFE_MODE
            self._warnings.append(str(e))
            # 全员 RTB
            actions = self._replanner.enter_safe_mode(ctx)
            self._world.step(actions)
            return

        # 重新 ReAct 规划
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
        """构造 ToolContext。"""
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

    def _on_escalation(self, task_id: str, answer, tick: int) -> None:
        """监控任务 escalation 回调。"""
        # 由 jev orchestrator 产出的 Escalation 已在 tick() 返回中处理
        # 这里仅记录计数
        self._escalation_count += 1

    def _build_result(self) -> AgentRunResult:
        return AgentRunResult(
            plan=self._plan,
            final_phase=self._phase,
            replan_count=self._replan_count,
            escalation_count=self._escalation_count,
            safe_mode=self._safe_mode,
            warnings=list(self._warnings),
        )
