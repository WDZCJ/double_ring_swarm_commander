"""Replanner 单测（tests/test_replanner.py）。

覆盖：
- trigger 递增 consecutive_failures
- 连续 N 次失败 → SafeModeTriggered
- on_plan_success 重置计数
- cancel_tasks_for_plan 被调用
- enter_safe_mode 全员 RTB
"""

from __future__ import annotations

import pytest

from munition_system.agent.monitoring_manager import MonitoringTaskManagerImpl
from munition_system.agent.replanner import Replanner, SafeModeTriggered
from munition_system.common.contracts import ToolContext
from munition_system.common.types import (
    ActionCommand,
    BattlePlan,
    CommanderIntent,
    JevQuestionSchemaSet,
    Munition,
    MunitionState,
)
from munition_system.simulation.world import World


def _make_world() -> World:
    w = World()
    w.add_munition(Munition(
        id="M1", role="strike", position=(0, 0, 0),
        state=MunitionState.SEARCHING, fuel=100.0, target_zone="Z1",
    ))
    w.add_munition(Munition(
        id="M2", role="search", position=(1, 1, 1),
        state=MunitionState.SEARCHING, fuel=100.0, target_zone="Z2",
    ))
    return w


class TestReplanner:
    def test_trigger_increments_failures(self):
        mgr = MonitoringTaskManagerImpl()
        r = Replanner(mgr, safe_mode_after_consecutive_failures=5)
        r.set_plan_version(1)
        assert r.consecutive_failures == 0
        r.trigger("test")
        assert r.consecutive_failures == 1

    def test_trigger_returns_new_version(self):
        mgr = MonitoringTaskManagerImpl()
        r = Replanner(mgr)
        r.set_plan_version(3)
        v = r.trigger("test")
        assert v == 4

    def test_trigger_cancels_old_tasks(self):
        mgr = MonitoringTaskManagerImpl()
        schema = JevQuestionSchemaSet(
            version=1, plan_version=1,
            questions=[],
        )
        # 直接插入一个 v1 的任务
        from munition_system.common.types import MonitoringTask, MonitoringTaskStatus
        mgr._tasks["MT1"] = MonitoringTask(
            task_id="MT1", owner_plan_version=1, subject="global",
            schema=schema, status=MonitoringTaskStatus.ACTIVE,
        )
        r = Replanner(mgr)
        r.set_plan_version(1)
        r.trigger("test")
        assert mgr.inspect_monitoring_task("MT1").status == MonitoringTaskStatus.CANCELLED

    def test_safe_mode_after_max_failures(self):
        mgr = MonitoringTaskManagerImpl()
        r = Replanner(mgr, safe_mode_after_consecutive_failures=3)
        r.set_plan_version(1)
        r.trigger("f1")
        r.trigger("f2")
        with pytest.raises(SafeModeTriggered) as exc_info:
            r.trigger("f3")
        assert exc_info.value.failures == 3

    def test_on_plan_success_resets(self):
        mgr = MonitoringTaskManagerImpl()
        r = Replanner(mgr)
        r.set_plan_version(1)
        r.trigger("f1")
        r.trigger("f2")
        assert r.consecutive_failures == 2
        r.on_plan_success()
        assert r.consecutive_failures == 0

    def test_enter_safe_mode_all_rtb(self):
        mgr = MonitoringTaskManagerImpl()
        r = Replanner(mgr)
        world = _make_world()
        from munition_system.simulation.situational_api import MockSituationalApi
        api = MockSituationalApi(world)
        ctx = ToolContext(tick=0, world=world, situational_api=api)
        actions = r.enter_safe_mode(ctx)
        assert len(actions) == 2
        assert all(a.action == "ReturnToBase" for a in actions)
        assert {a.munition_id for a in actions} == {"M1", "M2"}
