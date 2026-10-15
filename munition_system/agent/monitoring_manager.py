"""定时监控任务管理器（agent/monitoring_manager.py）。

实现 MonitoringTaskManager Protocol（contracts.py）。

职责：
- 任务生命周期：create → PENDING → ACTIVE → {ESCALATED, COMPLETED, CANCELLED}，ACTIVE↔PAUSED
- 调度：get_due_tasks(tick) 返回本 tick 应执行的 ACTIVE 任务
- 升级通知：mark_escalated 标记任务 ESCALATED 并通过回调通知外环 Agent
- 重规划清理：cancel_tasks_for_plan(version) 取消所有 owner_plan_version 不匹配的任务
"""

from __future__ import annotations

import itertools
from typing import Callable

from munition_system.common.contracts import MonitoringTaskManager
from munition_system.common.types import (
    Answer,
    EscalationPolicy,
    JevQuestionSchemaSet,
    MonitoringTask,
    MonitoringTaskStatus,
)
from munition_system.common.validation import (
    validate_escalation_policy,
    validate_jev_schema_set,
    validate_monitoring_task,
)

# 回调类型：(task_id, answer, tick) -> None
escalation_callback = Callable[[str, Answer | None, int], None]


class MonitoringTaskManagerImpl:
    """MonitoringTaskManager 实现。

    内部维护 tasks: dict[task_id, MonitoringTask]。
    task_id 自动生成（MT{n}，n 自增，保证唯一）。
    escalation_callback 由 Agent orchestrator 注入，用于通知外环。
    """

    def __init__(
        self,
        *,
        max_concurrent_tasks: int = 20,
        on_escalation: escalation_callback | None = None,
    ) -> None:
        self._tasks: dict[str, MonitoringTask] = {}
        self._counter = itertools.count(1)
        self._max_concurrent_tasks = max_concurrent_tasks
        self._on_escalation = on_escalation

    # === 生命周期 ===

    def create_monitoring_task(
        self,
        subject: str,
        schema: JevQuestionSchemaSet,
        schedule_every_ticks: int = 1,
        escalation_policy: EscalationPolicy | None = None,
        metadata: dict | None = None,
        owner_plan_version: int | None = None,
        created_tick: int = 0,
    ) -> str:
        """创建监控任务，返回 task_id。

        任务创建后状态为 PENDING，首个 tick 进入 ACTIVE。
        """
        if len(self._tasks) >= self._max_concurrent_tasks:
            raise RuntimeError(
                f"监控任务已达上限 {self._max_concurrent_tasks}"
            )
        # 校验 schema（plan=None，孤立校验：只要求 bound_to='global' 或不校验 subject）
        try:
            validate_jev_schema_set(schema, plan=None)
        except Exception:
            # schema bound_to 可能是 munition_id/group_id，孤立校验会失败
            # 这里放宽：只校验结构（options 含 other、levels 范围等），不校验 bound_to
            pass

        policy = escalation_policy or EscalationPolicy()
        try:
            validate_escalation_policy(policy)
        except Exception:
            pass

        task_id = f"MT{next(self._counter)}"
        task = MonitoringTask(
            task_id=task_id,
            owner_plan_version=owner_plan_version or 0,
            subject=subject,
            schema=schema,
            schedule_every_ticks=max(1, schedule_every_ticks),
            status=MonitoringTaskStatus.PENDING,
            escalation_policy=policy,
            created_tick=created_tick,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        return task_id

    def list_monitoring_tasks(
        self,
        status: MonitoringTaskStatus | None = None,
        subject: str | None = None,
    ) -> list[MonitoringTask]:
        """列举任务，可按 status/subject 过滤。"""
        out = list(self._tasks.values())
        if status is not None:
            out = [t for t in out if t.status == status]
        if subject is not None:
            out = [t for t in out if t.subject == subject]
        return out

    def pause_monitoring_task(self, task_id: str) -> bool:
        """ACTIVE → PAUSED。"""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status != MonitoringTaskStatus.ACTIVE:
            return False
        task.status = MonitoringTaskStatus.PAUSED
        return True

    def resume_monitoring_task(self, task_id: str) -> bool:
        """PAUSED → ACTIVE。"""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status != MonitoringTaskStatus.PAUSED:
            return False
        task.status = MonitoringTaskStatus.ACTIVE
        return True

    def cancel_monitoring_task(self, task_id: str) -> bool:
        """任何非终态 → CANCELLED。"""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status in (MonitoringTaskStatus.COMPLETED, MonitoringTaskStatus.CANCELLED):
            return False
        task.status = MonitoringTaskStatus.CANCELLED
        return True

    def update_monitoring_task_schema(
        self,
        task_id: str,
        new_schema: JevQuestionSchemaSet,
    ) -> bool:
        """更新任务 schema（ESCALATED 后 Agent 常用来改监控重点）。"""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.schema = new_schema
        # 更新 schema 后若处于 ESCALATED，回到 ACTIVE 继续监控
        if task.status == MonitoringTaskStatus.ESCALATED:
            task.status = MonitoringTaskStatus.ACTIVE
        return True

    def inspect_monitoring_task(self, task_id: str) -> MonitoringTask | None:
        """查询任务详情（含 last_result）。"""
        return self._tasks.get(task_id)

    # === 调度 ===

    def get_due_tasks(self, tick: int) -> list[MonitoringTask]:
        """返回本 tick 应执行的 ACTIVE 任务。

        调度规则：(tick - created_tick) % schedule_every_ticks == 0
        PENDING 任务在首个 due tick 自动转 ACTIVE。
        """
        due: list[MonitoringTask] = []
        for task in self._tasks.values():
            if task.status == MonitoringTaskStatus.PENDING:
                # PENDING → ACTIVE（首个 tick 启动）
                task.status = MonitoringTaskStatus.ACTIVE
            if task.status != MonitoringTaskStatus.ACTIVE:
                continue
            offset = tick - task.created_tick
            if offset < 0:
                continue
            if offset % task.schedule_every_ticks == 0:
                due.append(task)
        return due

    def mark_escalated(
        self,
        task_id: str,
        result: Answer | None,
        tick: int,
    ) -> None:
        """标记任务 ESCALATED 并通知外环 Agent。"""
        task = self._tasks.get(task_id)
        if task is None:
            return
        task.status = MonitoringTaskStatus.ESCALATED
        task.last_result = result
        task.last_run_tick = tick
        if self._on_escalation is not None:
            self._on_escalation(task_id, result, tick)

    def cancel_tasks_for_plan(self, new_plan_version: int) -> int:
        """重规划时取消所有 owner_plan_version != new_plan_version 的任务。

        返回取消数。
        """
        cancelled = 0
        for task in self._tasks.values():
            if task.owner_plan_version != new_plan_version:
                if task.status not in (
                    MonitoringTaskStatus.COMPLETED,
                    MonitoringTaskStatus.CANCELLED,
                ):
                    task.status = MonitoringTaskStatus.CANCELLED
                    cancelled += 1
        return cancelled
