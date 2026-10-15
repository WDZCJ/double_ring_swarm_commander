"""状态机转移 + tick 推进单测（tests/test_simulation_tick.py）。

覆盖：
- 巡飞弹状态机合法/非法转移
- World.step 推进 + WorldState 快照不变量
- 威胁向量恒为 (0,0,0) 或长度 3 int 元组（用户硬偏好）
- 事件注入（Damage/CommLoss/IntelUpdate）
- 工具 Mock 基本调用
- 定时监控任务（MonitoringTask）字段默认值
"""

from __future__ import annotations

import pytest

from munition_system.common.contracts import ToolContext
from munition_system.common.types import (
    ActionCommand,
    ActionType,
    CommLossEvent,
    DamageEvent,
    IntelUpdateEvent,
    MonitoringTask,
    MonitoringTaskStatus,
    Munition,
    MunitionState,
    Threat,
    ThreatVector,
)
from munition_system.common.validation import (
    assert_threat_vector,
    check_threat_vector,
    validate_world_state,
)
from munition_system.simulation.environment import VoxelGrid, make_zone
from munition_system.simulation.events import EventInjector
from munition_system.simulation.munition import (
    apply_action,
    make_munition,
)
from munition_system.simulation.situational_api import MockSituationalApi
from munition_system.simulation.threat import (
    aggregate_lethality,
    make_threat,
    no_threat,
    threats_in_range,
)
from munition_system.simulation.tools_impl import (
    AbortMunitionTaskTool,
    AnalyzeTargetTool,
    AssessThreatTool,
    CheckTaskProgressTool,
    DesignGroupingTool,
    GetBattlefieldSituationTool,
    GetSwarmStatusTool,
    PlanRoutesTool,
    SendDirectCommandTool,
)
from munition_system.simulation.world import World


# === 威胁向量不变量（用户硬偏好） ===

class TestThreatVector:
    def test_default_no_threat_is_zero_tuple(self):
        """无威胁时默认 (0,0,0) 而非 None。"""
        m = make_munition("M1")
        assert m.threat_vector == (0, 0, 0)
        assert isinstance(m.threat_vector, tuple)
        assert len(m.threat_vector) == 3

    def test_check_threat_vector_rejects_none(self):
        assert not check_threat_vector(None)

    def test_check_threat_vector_rejects_wrong_length(self):
        assert not check_threat_vector((1, 2))
        assert not check_threat_vector((1, 2, 3, 4))

    def test_check_threat_vector_rejects_non_int(self):
        assert not check_threat_vector((1, 2, "x"))  # type: ignore[arg-type]
        assert not check_threat_vector((1.0, 2, 3))  # 浮点不算
        assert not check_threat_vector([1, 2, 3])  # list 不算

    def test_assert_threat_vector_raises_on_none(self):
        with pytest.raises(Exception):
            assert_threat_vector(None)

    def test_aggregate_empty_returns_zero_tuple(self):
        assert aggregate_lethality([]) == (0, 0, 0)

    def test_aggregate_takes_max_per_component(self):
        t1 = make_threat("T1", (0, 0, 0), "sam", (1, 5, 2))
        t2 = make_threat("T2", (0, 0, 0), "radar", (3, 1, 4))
        out = aggregate_lethality([t1, t2])
        assert out == (3, 5, 4)

    def test_no_threat_helper(self):
        assert no_threat() == (0, 0, 0)


# === 巡飞弹状态机 ===

class TestMunitionStateMachine:
    def test_idle_to_assigned_legal(self):
        m = make_munition("M1", state=MunitionState.IDLE)
        cmd = ActionCommand(munition_id="M1", action=ActionType.REASSIGN_ROLE.value, args={"role": "decoy"})
        new_m, ws = apply_action(m, cmd)
        assert new_m.state == MunitionState.ASSIGNED
        assert new_m.role == "decoy"
        assert ws == []

    def test_idle_to_engaging_illegal(self):
        """IDLE 不能直接 ENGAGING（需经 ASSIGNED）。"""
        m = make_munition("M1", state=MunitionState.IDLE)
        cmd = ActionCommand(munition_id="M1", action=ActionType.ENGAGE_TARGET.value)
        new_m, ws = apply_action(m, cmd)
        assert new_m.state == MunitionState.IDLE  # 自留原态
        assert any(w.code == "illegal_transition" for w in ws)

    def test_destroyed_absorbing_state(self):
        """DESTROYED 是终态，拒绝任何动作。"""
        m = make_munition("M1", state=MunitionState.DESTROYED)
        cmd = ActionCommand(munition_id="M1", action=ActionType.MOVE_TO.value, args={"waypoint": [5, 5, 0]})
        new_m, ws = apply_action(m, cmd)
        assert new_m.state == MunitionState.DESTROYED
        assert any(w.code == "illegal_transition" for w in ws)

    def test_move_to_advances_position(self):
        m = make_munition("M1", state=MunitionState.SEARCHING, position=(0, 0, 0), fuel=1.0)
        cmd = ActionCommand(
            munition_id="M1", action=ActionType.MOVE_TO.value,
            args={"waypoint": [3, 4, 0]},
        )
        new_m, ws = apply_action(m, cmd)
        assert new_m.position == (3, 4, 0)
        # 曼哈顿距离=7，每单位 0.01 燃料
        assert new_m.fuel < m.fuel
        assert all(w.code != "low_fuel" for w in ws)  # 燃料充足

    def test_move_to_no_waypoint_warns(self):
        m = make_munition("M1", state=MunitionState.SEARCHING)
        cmd = ActionCommand(munition_id="M1", action=ActionType.MOVE_TO.value, args={})
        new_m, ws = apply_action(m, cmd)
        assert any(w.code == "no_route" for w in ws)
        assert new_m.position == m.position  # 保持原位

    def test_unknown_action_warns(self):
        m = make_munition("M1", state=MunitionState.IDLE)
        cmd = ActionCommand(munition_id="M1", action="SomeUnknownAction", args={})
        new_m, ws = apply_action(m, cmd)
        assert any(w.code == "unknown_action" for w in ws)
        assert new_m.state == m.state

    def test_adjust_altitude_changes_z(self):
        m = make_munition("M1", state=MunitionState.SEARCHING, position=(5, 5, 0))
        cmd = ActionCommand(munition_id="M1", action=ActionType.ADJUST_ALTITUDE.value, args={"altitude": 3})
        new_m, _ = apply_action(m, cmd)
        assert new_m.position == (5, 5, 3)


# === World tick 推进 ===

class TestWorldStep:
    def _make_world(self):
        m1 = make_munition("M1", state=MunitionState.IDLE, position=(0, 0, 0))
        m2 = make_munition("M2", state=MunitionState.IDLE, position=(5, 5, 0))
        z1 = make_zone("Z1", (0, 10, 0, 10, 0, 5))
        return World(munitions=[m1, m2], zones=[z1])

    def test_step_increments_tick(self):
        w = self._make_world()
        ws0 = w.snapshot()
        assert ws0.tick == 0
        ws1 = w.step([])
        assert ws1.tick == 1
        ws2 = w.step([])
        assert ws2.tick == 2

    def test_step_applies_actions(self):
        w = self._make_world()
        cmd = ActionCommand(
            munition_id="M1", action=ActionType.REASSIGN_ROLE.value,
            args={"role": "decoy"},
        )
        ws = w.step([cmd])
        m1 = next(m for m in ws.munitions if m.id == "M1")
        assert m1.state == MunitionState.ASSIGNED
        assert m1.role == "decoy"

    def test_step_updates_threat_vector_zero_when_no_threats(self):
        """无威胁时巡飞弹 threat_vector 始终为 (0,0,0)。"""
        w = self._make_world()
        ws = w.step([])
        for m in ws.munitions:
            assert m.threat_vector == (0, 0, 0)

    def test_step_updates_threat_vector_with_threats(self):
        w = self._make_world()
        t = make_threat("T1", (1, 1, 0), "sam", (2, 3, 1))
        w.add_threat(t)
        ws = w.step([])
        m1 = next(m for m in ws.munitions if m.id == "M1")
        # M1 在 (0,0,0)，距 (1,1,0) 距离=2，半径 8 内
        assert m1.threat_vector == (2, 3, 1)
        m2 = next(m for m in ws.munitions if m.id == "M2")
        # M2 在 (5,5,0)，距 (1,1,0) 距离=8，边界上
        assert check_threat_vector(m2.threat_vector)

    def test_step_fires_scheduled_events(self):
        w = self._make_world()
        w.event_injector.schedule(DamageEvent(munition_id="M1", tick=1))
        # tick 0 不触发
        w.step([])
        m1 = w.get_munition("M1")
        assert m1.state != MunitionState.DESTROYED
        # tick 1 触发
        w.step([])
        m1 = w.get_munition("M1")
        assert m1.state == MunitionState.DESTROYED
        assert m1.threat_vector == (0, 0, 0)  # 终态弹威胁向量清零

    def test_step_comms_loss_event(self):
        w = self._make_world()
        w.event_injector.schedule(CommLossEvent(munition_ids=["M1"], tick=0))
        ws = w.step([])
        m1 = next(m for m in ws.munitions if m.id == "M1")
        assert m1.state == MunitionState.LOST_COMMS
        assert "M1" in ws.comms_blackout_ids

    def test_step_intel_update_disappears_threat(self):
        w = self._make_world()
        t = make_threat("T1", (0, 0, 0), "radar", (1, 1, 1))
        w.add_threat(t)
        w.event_injector.schedule(IntelUpdateEvent(target_id="T1", status="disappeared", tick=0))
        ws = w.step([])
        assert all(t.id != "T1" for t in ws.threats)

    def test_snapshot_validates(self):
        """快照必须通过 validate_world_state。"""
        w = self._make_world()
        ws = w.step([])
        validate_world_state(ws)  # 不抛异常即通过


# === 覆盖率推进 ===

class TestCoverageAdvance:
    def test_searching_munition_advances_zone_coverage(self):
        m1 = make_munition("M1", state=MunitionState.SEARCHING, position=(0, 0, 0), target_zone="Z1")
        z1 = make_zone("Z1", (0, 10, 0, 10, 0, 5))
        w = World(munitions=[m1], zones=[z1])
        assert w.get_zone_coverage("Z1") == 0.0
        w.step([])
        assert w.get_zone_coverage("Z1") == pytest.approx(0.05)
        w.step([])
        assert w.get_zone_coverage("Z1") == pytest.approx(0.10)

    def test_non_searching_munition_does_not_advance_coverage(self):
        m1 = make_munition("M1", state=MunitionState.IDLE, target_zone="Z1")
        z1 = make_zone("Z1", (0, 10, 0, 10, 0, 5))
        w = World(munitions=[m1], zones=[z1])
        w.step([])
        assert w.get_zone_coverage("Z1") == 0.0


# === 体素网格 ===

class TestVoxelGrid:
    def test_in_bounds(self):
        g = VoxelGrid((10, 10, 5))
        assert g.in_bounds((0, 0, 0))
        assert g.in_bounds((9, 9, 4))
        assert not g.in_bounds((10, 0, 0))
        assert not g.in_bounds((-1, 0, 0))

    def test_neighbors_clipped_to_bounds(self):
        g = VoxelGrid((5, 5, 5))
        nb = g.neighbors((0, 0, 0))
        assert (1, 0, 0) in nb
        assert (0, 1, 0) in nb
        assert (0, 0, 1) in nb
        assert (-1, 0, 0) not in nb  # 越界裁剪

    def test_distance_manhattan(self):
        assert VoxelGrid.distance((0, 0, 0), (3, 4, 0)) == 7


# === 事件注入器 ===

class TestEventInjector:
    def test_fifo_within_same_tick(self):
        inj = EventInjector()
        e1 = DamageEvent(munition_id="M1", tick=2)
        e2 = CommLossEvent(munition_ids=["M2"], tick=2)
        inj.schedule(e1)
        inj.schedule(e2)
        fired = inj.check(2)
        assert fired == [e1, e2]  # 同 tick 按 schedule 顺序

    def test_check_returns_empty_when_no_events(self):
        inj = EventInjector()
        assert inj.check(5) == []

    def test_check_only_fires_due_events(self):
        inj = EventInjector()
        inj.schedule(DamageEvent(munition_id="M1", tick=5))
        assert inj.check(3) == []
        assert len(inj.check(5)) == 1

    def test_peek_next_tick(self):
        inj = EventInjector()
        assert inj.peek_next_tick() is None
        inj.schedule(DamageEvent(munition_id="M1", tick=5))
        inj.schedule(DamageEvent(munition_id="M2", tick=3))
        assert inj.peek_next_tick() == 3

    def test_listener_notified(self):
        inj = EventInjector()
        received = []
        inj.add_listener(lambda ev: received.append(ev))
        inj.schedule(DamageEvent(munition_id="M1", tick=0))
        inj.check(0)
        assert len(received) == 1


# === 工具 Mock 调用 ===

class TestToolsBasic:
    def _make_ctx(self, world: World, plan=None) -> ToolContext:
        return ToolContext(
            tick=world.tick,
            world=world,
            situational_api=MockSituationalApi(world),
            plan=plan,
        )

    def test_get_swarm_status(self):
        w = World(munitions=[make_munition("M1"), make_munition("M2", position=(1, 1, 0))])
        tool = GetSwarmStatusTool()
        r = tool.call({}, self._make_ctx(w))
        assert r.ok
        assert len(r.data) == 2
        # 无威胁时 threat_vector=[0,0,0]
        assert all(m["threat_vector"] == [0, 0, 0] for m in r.data)

    def test_plan_routes_basic(self):
        w = World()
        tool = PlanRoutesTool()
        r = tool.call({"start": [0, 0, 0], "end": [3, 3, 0]}, self._make_ctx(w))
        assert r.ok
        assert len(r.data) == 1
        # threat_level 永远长度 3
        assert len(r.data[0]["threat_level"]) == 3

    def test_analyze_target(self):
        w = World()
        w.add_threat(make_threat("TGT1", (5, 5, 0), "sam", (1, 1, 1)))
        tool = AnalyzeTargetTool()
        r = tool.call({"target_id": "TGT1"}, self._make_ctx(w))
        assert r.ok
        assert r.data["target_id"] == "TGT1"
        assert "weakness" in r.data["defense_weakness"] or "防御" in r.data["defense_weakness"]

    def test_assess_threat_no_threats_returns_zero_vector(self):
        w = World()
        tool = AssessThreatTool()
        r = tool.call({"zone_id": None}, self._make_ctx(w))
        # 无 zone_id + 无 route → 应当报错
        assert not r.ok

    def test_assess_threat_with_zone(self):
        w = World(zones=[make_zone("Z1", (0, 10, 0, 10, 0, 5))])
        w.add_threat(make_threat("T1", (5, 5, 0), "sam", (2, 2, 2)))
        tool = AssessThreatTool()
        r = tool.call({"zone_id": "Z1"}, self._make_ctx(w))
        assert r.ok
        assert r.data["threat_vector"] == [2, 2, 2]
        assert r.data["level"] == "medium"

    def test_design_grouping_even_split(self):
        w = World()
        tool = DesignGroupingTool()
        r = tool.call({"available_munitions": 6, "zones": ["ZA", "ZB"]}, self._make_ctx(w))
        assert r.ok
        assert len(r.data["groups"]) == 2
        # 每组 3 个
        assert all(len(members) == 3 for members in r.data["groups"].values())

    def test_design_grouping_odd_split(self):
        w = World()
        tool = DesignGroupingTool()
        r = tool.call({"available_munitions": 7, "zones": ["ZA", "ZB"]}, self._make_ctx(w))
        assert r.ok
        counts = [len(m) for m in r.data["groups"].values()]
        assert sorted(counts) == [3, 4]  # 余数分给前组

    def test_design_grouping_rejects_over_18(self):
        w = World()
        tool = DesignGroupingTool()
        r = tool.call({"available_munitions": 19, "zones": ["ZA"]}, self._make_ctx(w))
        assert not r.ok

    def test_abort_munition_task(self):
        m = make_munition("M1", state=MunitionState.SEARCHING)
        w = World(munitions=[m])
        tool = AbortMunitionTaskTool()
        r = tool.call({"munition_id": "M1"}, self._make_ctx(w))
        assert r.ok
        assert w.get_munition("M1").state == MunitionState.RTB

    def test_send_direct_command(self):
        m = make_munition("M1", state=MunitionState.IDLE)
        w = World(munitions=[m])
        tool = SendDirectCommandTool()
        r = tool.call({
            "munition_id": "M1",
            "action": ActionType.REASSIGN_ROLE.value,
            "args": {"role": "decoy"},
        }, self._make_ctx(w))
        assert r.ok
        assert w.get_munition("M1").state == MunitionState.ASSIGNED

    def test_check_task_progress(self):
        from munition_system.common.types import BattlePlan, CommanderIntent, TaskAssignment
        plan = BattlePlan(
            intent=CommanderIntent(raw_text="r", goal="g", constraints=[], success_criteria="搜索覆盖完成"),
            assignments=[TaskAssignment(munition_id="M1", role="search", zone_id="Z1", capability="search")],
            groups={"G1": ["M1"]},
            execution_steps=[], success_conditions=["搜索覆盖完成"],
            monitoring_points=[], plan_version=1,
        )
        m = make_munition("M1", state=MunitionState.SEARCHING, target_zone="Z1")
        z = make_zone("Z1", (0, 10, 0, 10, 0, 5))
        w = World(munitions=[m], zones=[z])
        ctx = self._make_ctx(w, plan=plan)
        ctx.plan_version = 1
        tool = CheckTaskProgressTool()
        r = tool.call({"plan_version": 1}, ctx)
        assert r.ok
        assert "completion" in r.data


# === MonitoringTask 默认值 ===

class TestMonitoringTaskDefaults:
    def test_default_status_is_pending(self):
        from munition_system.common.types import JevQuestionSchemaSet, NoulQuestion
        schema = JevQuestionSchemaSet(
            version=1, plan_version=1,
            questions=[NoulQuestion(id="Q1", prompt="ok?", bound_to="global")],
        )
        mt = MonitoringTask(task_id="MT1", owner_plan_version=1, subject="M1", schema=schema)
        assert mt.status == MonitoringTaskStatus.PENDING
        assert mt.schedule_every_ticks == 1
        assert mt.last_result is None
        assert mt.last_run_tick == -1
        assert mt.created_tick == 0
        assert mt.metadata == {}
