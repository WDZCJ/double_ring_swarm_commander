"""任务完成度闭环跟踪（agent/plan_tracker.py）。

每 N tick（默认 5）调一次 check_task_progress 工具，输出 ProgressReport：
完成度%、卡住原因、是否需要重规划。
完成→通知 Agent 终止任务；卡住/失败→触发 replanner。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from munition_system.common.contracts import ToolContext
    from munition_system.common.types import BattlePlan
    from munition_system.simulation.tools_impl import CheckTaskProgressTool


@dataclass
class ProgressReport:
    """任务进度报告。"""
    plan_version: int
    completion: float  # 0.0~1.0
    stuck: bool = False
    failed: bool = False
    reason: str = ""
    satisfied_conditions: list[str] = None  # type: ignore[assignment]


class PlanTracker:
    """任务完成度闭环跟踪。"""

    def __init__(
        self,
        check_tool: "CheckTaskProgressTool",
        *,
        check_interval_ticks: int = 5,
    ) -> None:
        self._check_tool = check_tool
        self._check_interval_ticks = check_interval_ticks
        self._last_check_tick = -1

    def should_check(self, tick: int) -> bool:
        """本 tick 是否应检查进度。"""
        return (tick - self._last_check_tick) >= self._check_interval_ticks

    def track(
        self,
        tick: int,
        plan: "BattlePlan",
        ctx: "ToolContext",
    ) -> ProgressReport | None:
        """检查进度。返回 ProgressReport 或 None（未到检查周期）。"""
        if not self.should_check(tick):
            return None
        self._last_check_tick = tick
        result = self._check_tool.call({"plan_version": plan.plan_version}, ctx)
        if not result.ok or not isinstance(result.data, dict):
            return None
        d = result.data
        return ProgressReport(
            plan_version=int(d.get("plan_version", plan.plan_version)),
            completion=float(d.get("completion", 0.0)),
            stuck=bool(d.get("stuck", False)),
            failed=bool(d.get("failed", False)),
            reason=str(d.get("reason", "")),
            satisfied_conditions=list(d.get("satisfied_conditions", [])),
        )
