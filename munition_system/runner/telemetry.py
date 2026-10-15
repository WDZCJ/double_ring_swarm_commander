"""可观测收集器（runner/telemetry.py）。

收集三层联调过程中的运行数据，供 demo 末尾打印与人工审批：
- tick 轨迹：每 tick 的动作/escalation/世界状态摘要
- Agent 工具调用序列
- 监控任务状态变化日志
- Jev 决策置信度直方图（所有 answer 的 confidence 分布）
- escalation 次数 / 重规划次数 / 最终任务状态
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickRecord:
    """单 tick 运行记录。"""
    tick: int
    actions: list[dict] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)
    world_summary: dict = field(default_factory=dict)
    confidences: list[float] = field(default_factory=list)  # 本 tick 所有 Jev answer 的 confidence


@dataclass
class ToolCallRecord:
    """单次工具调用记录。"""
    tick: int
    name: str
    args: dict
    ok: bool
    error: str | None = None


@dataclass
class TaskStatusChange:
    """监控任务状态变化记录。"""
    tick: int
    task_id: str
    old_status: str
    new_status: str
    reason: str = ""


@dataclass
class ReplanRecord:
    """重规划记录。"""
    tick: int
    reason: str
    new_plan_version: int
    consecutive_failures: int


class Telemetry:
    """可观测收集器。"""

    def __init__(self) -> None:
        self.tick_records: list[TickRecord] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.task_changes: list[TaskStatusChange] = []
        self.replans: list[ReplanRecord] = []
        self.final_status: dict = {}
        self._all_confidences: list[float] = []

    # === 记录方法 ===

    def record_tick(
        self,
        tick: int,
        actions: list[Any],
        escalations: list[Any],
        world_summary: dict,
        confidences: list[float],
    ) -> None:
        """记录一个 tick 的运行数据。"""
        self.tick_records.append(TickRecord(
            tick=tick,
            actions=[_action_to_dict(a) for a in actions],
            escalations=[_escalation_to_dict(e) for e in escalations],
            world_summary=world_summary,
            confidences=list(confidences),
        ))
        self._all_confidences.extend(confidences)

    def record_tool_call(
        self,
        tick: int,
        name: str,
        args: dict,
        ok: bool,
        error: str | None = None,
    ) -> None:
        self.tool_calls.append(ToolCallRecord(
            tick=tick, name=name, args=args, ok=ok, error=error,
        ))

    def record_task_change(
        self,
        tick: int,
        task_id: str,
        old_status: str,
        new_status: str,
        reason: str = "",
    ) -> None:
        self.task_changes.append(TaskStatusChange(
            tick=tick, task_id=task_id,
            old_status=old_status, new_status=new_status, reason=reason,
        ))

    def record_replan(
        self,
        tick: int,
        reason: str,
        new_plan_version: int,
        consecutive_failures: int,
    ) -> None:
        self.replans.append(ReplanRecord(
            tick=tick, reason=reason,
            new_plan_version=new_plan_version,
            consecutive_failures=consecutive_failures,
        ))

    def set_final_status(self, status: dict) -> None:
        self.final_status = status

    # === 汇总方法 ===

    @property
    def escalation_count(self) -> int:
        return sum(len(r.escalations) for r in self.tick_records)

    @property
    def replan_count(self) -> int:
        return len(self.replans)

    @property
    def safe_mode(self) -> bool:
        return bool(self.final_status.get("safe_mode", False))

    def confidence_histogram(self, bins: int = 10) -> dict[str, int]:
        """置信度直方图：把 [0,1] 分成 bins 个区间，统计每区间数量。"""
        if not self._all_confidences:
            return {}
        counts: Counter[int] = Counter()
        for c in self._all_confidences:
            idx = min(int(c * bins), bins - 1)
            counts[idx] += 1
        return {f"{i / bins:.1f}-{(i + 1) / bins:.1f}": counts.get(i, 0) for i in range(bins)}

    def summary(self) -> dict:
        """生成可打印的摘要 dict。"""
        return {
            "total_ticks": len(self.tick_records),
            "total_tool_calls": len(self.tool_calls),
            "escalation_count": self.escalation_count,
            "replan_count": self.replan_count,
            "safe_mode": self.safe_mode,
            "task_changes": len(self.task_changes),
            "confidence_histogram": self.confidence_histogram(),
            "final_phase": self.final_status.get("phase", "unknown"),
        }


# === 序列化辅助 ===

def _action_to_dict(a: Any) -> dict:
    if isinstance(a, dict):
        return a
    if hasattr(a, "__dict__"):
        d = dict(a.__dict__)
        # args 可能是 dict
        return d
    return {"action": str(a)}


def _escalation_to_dict(e: Any) -> dict:
    if isinstance(e, dict):
        return e
    return {
        "reason": getattr(e, "reason", ""),
        "source_task": getattr(e, "source_monitoring_task_id", None),
        "schema_version": getattr(e, "schema_version", 0),
        "n_low_conf": len(getattr(e, "low_confidence_answers", [])),
    }
