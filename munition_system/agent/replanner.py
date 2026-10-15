"""重规划器（agent/replanner.py）。

触发条件（任一，由 Agent orchestrator 判定后调用 trigger）：
1. 内环 Escalation 中同问题连续 ≥3 tick 低置信
2. 仿真注入 DamageEvent/IntelUpdateEvent/CommLossEvent
3. check_task_progress 显示卡住/失败
4. 任意 MonitoringTask 进入 ESCALATED
5. LLM 自己判定需要重规划

执行：
- 递增 consecutive_failures
- 若 consecutive_failures ≥ max_failures（默认 5）→ 触发安全模式（全员 RTB）
- 否则：cancel_tasks_for_plan(new_version) + 返回 new_plan_version
- 重规划成功后由 Agent 调 on_plan_success() 重置计数
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from munition_system.common.types import ActionCommand, MunitionState

if TYPE_CHECKING:
    from munition_system.agent.monitoring_manager import MonitoringTaskManagerImpl
    from munition_system.common.contracts import ToolContext


class SafeModeTriggered(RuntimeError):
    """连续重规划失败达上限，进入安全模式（全员 RTB）。"""

    def __init__(self, failures: int) -> None:
        super().__init__(
            f"连续 {failures} 次重规划失败，进入安全模式（全员 RTB）"
        )
        self.failures = failures


class Replanner:
    """重规划器。"""

    def __init__(
        self,
        monitoring_manager: "MonitoringTaskManagerImpl",
        *,
        safe_mode_after_consecutive_failures: int = 5,
    ) -> None:
        self._monitoring_manager = monitoring_manager
        self._max_failures = safe_mode_after_consecutive_failures
        self._consecutive_failures = 0
        self._current_plan_version = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def current_plan_version(self) -> int:
        return self._current_plan_version

    def set_plan_version(self, version: int) -> None:
        """设置当前 plan 版本（由 Agent orchestrator 在新 plan 产出后调用）。"""
        self._current_plan_version = version

    def trigger(self, reason: str, *, ctx: "ToolContext | None" = None) -> int:
        """触发重规划。

        - 递增 consecutive_failures
        - 达上限 → 抛 SafeModeTriggered，调用方负责执行全员 RTB
        - 否则取消旧 plan 的所有监控任务，返回 new_plan_version
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_failures:
            raise SafeModeTriggered(self._consecutive_failures)

        new_version = self._current_plan_version + 1
        # 取消所有 owner_plan_version 不匹配的任务
        self._monitoring_manager.cancel_tasks_for_plan(new_version)
        return new_version

    def trigger_external(self, reason: str, *, ctx: "ToolContext | None" = None) -> int:
        """外部（用户 API / Web UI）触发的重规划。

        与 trigger() 的区别：不递增 consecutive_failures——避免用户频繁交互
        被误判为"连续重规划失败"而触发安全模式。
        仍然取消旧 plan 版本关联的监控任务，返回 new_plan_version。
        """
        new_version = self._current_plan_version + 1
        self._monitoring_manager.cancel_tasks_for_plan(new_version)
        return new_version

    def on_plan_success(self) -> None:
        """重规划成功后重置失败计数。"""
        self._consecutive_failures = 0

    def enter_safe_mode(self, ctx: "ToolContext") -> list[ActionCommand]:
        """安全模式：全员 RTB。

        返回 ActionCommand 列表（所有非终态弹 → ReturnToBase）。
        """
        actions: list[ActionCommand] = []
        if ctx.situational_api is None:
            return actions
        ws = ctx.situational_api.get_world_state()
        for m in ws.munitions:
            if m.state in (MunitionState.DESTROYED, MunitionState.LOST_COMMS, MunitionState.RTB):
                continue
            actions.append(ActionCommand(
                munition_id=m.id,
                action="ReturnToBase",
                args={},
            ))
        return actions
