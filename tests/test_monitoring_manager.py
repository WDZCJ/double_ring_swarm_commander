"""MonitoringTaskManager 单测（tests/test_monitoring_manager.py）。

覆盖：
- 任务生命周期：create → PENDING → ACTIVE → {ESCALATED, COMPLETED, CANCELLED}，ACTIVE↔PAUSED
- get_due_tasks 调度（schedule_every_ticks 匹配）
- mark_escalated 通知回调
- cancel_tasks_for_plan 重规划清理
- 上限保护
"""

from __future__ import annotations

import pytest

from munition_system.agent.monitoring_manager import MonitoringTaskManagerImpl
from munition_system.common.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    EscalationPolicy,
    JevQuestionSchemaSet,
    MonitoringTaskStatus,
    NoulQuestion,
)


def _make_schema(plan_version: int = 1) -> JevQuestionSchemaSet:
    return JevQuestionSchemaSet(
        version=1,
        plan_version=plan_version,
        questions=[
            ChoiceQuestion(
                id="q_status",
                prompt="状态",
                options=["normal", "abnormal", "other"],
                bound_to="global",
            )
        ],
    )


class TestLifecycle:
    """任务生命周期。"""

    def test_create_task_is_pending(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema())
        task = mgr.inspect_monitoring_task(tid)
        assert task is not None
        assert task.status == MonitoringTaskStatus.PENDING
        assert task.subject == "global"

    def test_first_tick_moves_to_active(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        due = mgr.get_due_tasks(0)
        assert len(due) == 1
        assert due[0].task_id == tid
        assert due[0].status == MonitoringTaskStatus.ACTIVE

    def test_pause_resume(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        mgr.get_due_tasks(0)  # → ACTIVE
        assert mgr.pause_monitoring_task(tid) is True
        task = mgr.inspect_monitoring_task(tid)
        assert task.status == MonitoringTaskStatus.PAUSED
        assert mgr.resume_monitoring_task(tid) is True
        assert task.status == MonitoringTaskStatus.ACTIVE

    def test_pause_non_active_fails(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema())
        # PENDING 不能直接 pause
        assert mgr.pause_monitoring_task(tid) is False

    def test_cancel(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        mgr.get_due_tasks(0)
        assert mgr.cancel_monitoring_task(tid) is True
        assert mgr.inspect_monitoring_task(tid).status == MonitoringTaskStatus.CANCELLED

    def test_update_schema_resets_escalated(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        mgr.get_due_tasks(0)
        mgr.mark_escalated(tid, None, 5)
        assert mgr.inspect_monitoring_task(tid).status == MonitoringTaskStatus.ESCALATED
        # 更新 schema → 回到 ACTIVE
        new_schema = _make_schema(plan_version=1)
        new_schema.version = 2
        assert mgr.update_monitoring_task_schema(tid, new_schema) is True
        assert mgr.inspect_monitoring_task(tid).status == MonitoringTaskStatus.ACTIVE


class TestScheduling:
    """get_due_tasks 调度。"""

    def test_schedule_every_ticks(self):
        mgr = MonitoringTaskManagerImpl()
        # schedule=2：tick 0,2,4 执行
        tid = mgr.create_monitoring_task(
            "global", _make_schema(), schedule_every_ticks=2, created_tick=0
        )
        assert len(mgr.get_due_tasks(0)) == 1
        assert len(mgr.get_due_tasks(1)) == 0
        assert len(mgr.get_due_tasks(2)) == 1
        assert len(mgr.get_due_tasks(3)) == 0
        assert len(mgr.get_due_tasks(4)) == 1

    def test_paused_task_not_due(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        mgr.get_due_tasks(0)  # → ACTIVE
        mgr.pause_monitoring_task(tid)
        assert len(mgr.get_due_tasks(1)) == 0

    def test_cancelled_task_not_due(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        mgr.get_due_tasks(0)
        mgr.cancel_monitoring_task(tid)
        assert len(mgr.get_due_tasks(1)) == 0


class TestEscalation:
    """mark_escalated + 回调。"""

    def test_mark_escalated(self):
        notifications = []
        mgr = MonitoringTaskManagerImpl(on_escalation=lambda tid, ans, t: notifications.append((tid, t)))
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        mgr.get_due_tasks(0)
        ans = ChoiceAnswer(question_id="q", probabilities={"a": 1.0}, chosen="a", confidence=0.9)
        mgr.mark_escalated(tid, ans, 5)
        task = mgr.inspect_monitoring_task(tid)
        assert task.status == MonitoringTaskStatus.ESCALATED
        assert task.last_run_tick == 5
        assert task.last_result is ans
        assert notifications == [(tid, 5)]


class TestReplanCleanup:
    """cancel_tasks_for_plan。"""

    def test_cancel_other_plan_versions(self):
        mgr = MonitoringTaskManagerImpl()
        # plan v1 的任务
        t1 = mgr.create_monitoring_task("global", _make_schema(plan_version=1), created_tick=0, owner_plan_version=1)
        t2 = mgr.create_monitoring_task("M1", _make_schema(plan_version=1), created_tick=0, owner_plan_version=1)
        mgr.get_due_tasks(0)
        # 重规划到 v2
        cancelled = mgr.cancel_tasks_for_plan(2)
        assert cancelled == 2
        assert mgr.inspect_monitoring_task(t1).status == MonitoringTaskStatus.CANCELLED
        assert mgr.inspect_monitoring_task(t2).status == MonitoringTaskStatus.CANCELLED

    def test_keep_matching_plan_version(self):
        mgr = MonitoringTaskManagerImpl()
        t1 = mgr.create_monitoring_task("global", _make_schema(plan_version=2), created_tick=0, owner_plan_version=2)
        mgr.get_due_tasks(0)
        cancelled = mgr.cancel_tasks_for_plan(2)
        assert cancelled == 0
        assert mgr.inspect_monitoring_task(t1).status == MonitoringTaskStatus.ACTIVE


class TestLimits:
    """上限保护。"""

    def test_max_concurrent_tasks(self):
        mgr = MonitoringTaskManagerImpl(max_concurrent_tasks=2)
        mgr.create_monitoring_task("global", _make_schema())
        mgr.create_monitoring_task("M1", _make_schema())
        with pytest.raises(RuntimeError):
            mgr.create_monitoring_task("M2", _make_schema())


class TestListFilter:
    """list 过滤。"""

    def test_list_by_subject(self):
        mgr = MonitoringTaskManagerImpl()
        mgr.create_monitoring_task("global", _make_schema())
        mgr.create_monitoring_task("M1", _make_schema())
        assert len(mgr.list_monitoring_tasks(subject="global")) == 1
        assert len(mgr.list_monitoring_tasks(subject="M1")) == 1

    def test_list_by_status(self):
        mgr = MonitoringTaskManagerImpl()
        tid = mgr.create_monitoring_task("global", _make_schema(), created_tick=0)
        mgr.get_due_tasks(0)  # → ACTIVE
        assert len(mgr.list_monitoring_tasks(status=MonitoringTaskStatus.ACTIVE)) == 1
        mgr.cancel_monitoring_task(tid)
        assert len(mgr.list_monitoring_tasks(status=MonitoringTaskStatus.CANCELLED)) == 1
        assert len(mgr.list_monitoring_tasks(status=MonitoringTaskStatus.ACTIVE)) == 0
