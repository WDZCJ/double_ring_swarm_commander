"""tick 推进 + WorldState 快照（simulation/world.py）。

World 持有可变状态（munitions/threats/zones/event_injector/voxel_grid），
step(actions) 应用动作→推进→触发事件→返回不可变快照。
这是后续替换为真实硬件适配层的接入点之一。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from munition_system.common.types import (
    ActionCommand,
    CommLossEvent,
    DamageEvent,
    Event,
    IntelUpdateEvent,
    Munition,
    MunitionState,
    Threat,
    ThreatVector,
    WorldState,
    Zone,
)
from munition_system.common.validation import validate_world_state
from munition_system.simulation.environment import VoxelGrid, ZoneRegistry
from munition_system.simulation.events import EventInjector
from munition_system.simulation.munition import MunitionWarning, apply_action
from munition_system.simulation.threat import aggregate_lethality, threats_in_range


@dataclass
class TickReport:
    """单 tick 推进的报告：应用的动作 + 事件 + 警告 + 触发的威胁向量更新。"""

    tick: int
    actions_applied: list[ActionCommand]
    events_fired: list[Event]
    warnings: list[MunitionWarning]
    threat_updates: dict[str, ThreatVector] = field(default_factory=dict)
    """threat_updates: munition_id -> 新威胁向量（被周边威胁更新）"""


class World:
    """tick-based 仿真内核。

    状态持有：
    - tick：当前 tick
    - munitions：可变 dict[id, Munition]
    - threats：可变 list[Threat]
    - zones：ZoneRegistry
    - grid：VoxelGrid
    - event_injector：EventInjector
    - rng：种子化随机（跨运行一致）
    - listeners：tick 推进监听器（Agent orchestrator 用此接收事件）
    """

    def __init__(
        self,
        *,
        grid_dim: tuple[int, int, int] = (40, 40, 8),
        seed: int = 42,
        munitions: list[Munition] | None = None,
        threats: list[Threat] | None = None,
        zones: list[Zone] | None = None,
    ) -> None:
        self.tick: int = 0
        self.grid = VoxelGrid(grid_dim)
        self.rng = random.Random(seed)
        self.munitions: dict[str, Munition] = {}
        if munitions:
            for m in munitions:
                self.add_munition(m)
        self.threats: list[Threat] = list(threats or [])
        self.zones = ZoneRegistry()
        if zones:
            for z in zones:
                self.zones.add(z)
        self.event_injector = EventInjector()
        self._listeners: list[Callable[[TickReport], None]] = []

    # === 状态变更 ===

    def add_munition(self, m: Munition) -> None:
        if m.id in self.munitions:
            raise ValueError(f"巡飞弹 {m.id} 已存在")
        self.munitions[m.id] = m

    def remove_munition(self, munition_id: str) -> None:
        self.munitions.pop(munition_id, None)

    def add_threat(self, t: Threat) -> None:
        self.threats.append(t)

    def remove_threat(self, threat_id: str) -> None:
        self.threats = [t for t in self.threats if t.id != threat_id]

    def add_zone(self, z: Zone) -> None:
        self.zones.add(z)

    def add_tick_listener(self, listener: Callable[[TickReport], None]) -> None:
        """注册 tick 推进监听器（Agent orchestrator 用此接收 TickReport）。"""
        self._listeners.append(listener)

    # === tick 推进 ===

    def step(self, actions: list[ActionCommand]) -> WorldState:
        """推进一个 tick：应用动作→更新威胁→推进 coverage→触发事件→返回快照。

        步骤：
        1) 应用所有 ActionCommand 到对应巡飞弹（按 munition_id 索引）
        2) 推进威胁扫描：每个巡飞弹的 threat_vector 被周边威胁更新
           （无威胁时设为 (0,0,0) 而非 None）
        3) 推进 zone.coverage：处于搜索态的巡飞弹为目标 zone 增加覆盖
        4) 触发 EventInjector.check(tick)：应用损毁/情报更新/失联事件
        5) 构造不可变 WorldState 快照返回
        """
        report = TickReport(tick=self.tick, actions_applied=list(actions), events_fired=[], warnings=[])
        # 1) 应用动作
        for cmd in actions:
            m = self.munitions.get(cmd.munition_id)
            if m is None:
                report.warnings.append(MunitionWarning(
                    cmd.munition_id, "unknown_action",
                    f"巡飞弹 {cmd.munition_id} 不存在"
                ))
                continue
            new_m, ws = apply_action(m, cmd)
            self.munitions[cmd.munition_id] = new_m
            report.warnings.extend(ws)

        # 2) 威胁扫描更新
        report.threat_updates = self._update_threat_vectors()

        # 3) zone.coverage 推进（搜索态弹为目标 zone 增加覆盖）
        self._advance_coverage()

        # 3.5) 位置推进：搜索态弹朝 target_zone 中心逐步移动
        self._advance_positions()

        # 4) 触发事件
        events = self.event_injector.check(self.tick)
        report.events_fired = events
        for ev in events:
            self._apply_event(ev, report)

        # 通知监听器
        for listener in self._listeners:
            listener(report)

        # 5) 构造快照
        self.tick += 1
        return self.snapshot()

    def _update_threat_vectors(self) -> dict[str, ThreatVector]:
        """扫描每个巡飞弹周边威胁，更新 threat_vector（无威胁=(0,0,0)）。"""
        updates: dict[str, ThreatVector] = {}
        for mid, m in self.munitions.items():
            # 终态弹不更新威胁向量
            if m.state in (MunitionState.DESTROYED, MunitionState.LOST_COMMS):
                continue
            nearby = threats_in_range(self.threats, m.position, radius=8)
            tv = aggregate_lethality(nearby)  # 无威胁时返回 (0,0,0)
            from dataclasses import replace
            self.munitions[mid] = replace(m, threat_vector=tv)
            updates[mid] = tv
        return updates

    def _advance_coverage(self) -> None:
        """搜索态弹为目标 zone 增加覆盖率。"""
        for m in self.munitions.values():
            if m.state != MunitionState.SEARCHING:
                continue
            if not m.target_zone:
                continue
            # 每个搜索态弹每 tick 为目标 zone 增加 0.05 覆盖率
            self.zones.increment_coverage(m.target_zone, delta=0.05)

    def _advance_positions(self) -> None:
        """搜索态弹朝 target_zone 中心逐步移动（每 tick 移动 1 单位）。

        当 ActionCommand 没有显式 waypoint 时，此方法保证巡飞弹会向目标区域推进。
        """
        from dataclasses import replace
        for mid, m in self.munitions.items():
            if m.state not in (MunitionState.SEARCHING, MunitionState.ASSIGNED, MunitionState.ENGAGING):
                continue
            if not m.target_zone:
                continue
            zone = self.zones.get(m.target_zone)
            if zone is None:
                continue
            xmin, xmax, ymin, ymax, zmin, zmax = zone.bounds
            # 目标区域中心
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            cz = (zmin + zmax) // 2
            px, py, pz = m.position
            # 已到达中心则不移动
            if px == cx and py == cy and pz == cz:
                continue
            # 每 tick 沿主轴方向移动 1 单位（曼哈顿步进）
            new_x = px + (1 if cx > px else (-1 if cx < px else 0))
            new_y = py + (1 if cy > py else (-1 if cy < py else 0))
            new_z = pz + (1 if cz > pz else (-1 if cz < pz else 0))
            new_position = (new_x, new_y, new_z)
            # 边界检查
            grid_x, grid_y, grid_z = self.grid.dim
            if 0 <= new_x < grid_x and 0 <= new_y < grid_y and 0 <= new_z < grid_z:
                self.munitions[mid] = replace(m, position=new_position)

    def _apply_event(self, ev: Event, report: TickReport) -> None:
        """应用事件到世界状态。"""
        from dataclasses import replace
        if isinstance(ev, DamageEvent):
            m = self.munitions.get(ev.munition_id)
            if m is not None:
                self.munitions[ev.munition_id] = replace(m, state=MunitionState.DESTROYED, threat_vector=(0,0,0))
        elif isinstance(ev, CommLossEvent):
            for mid in ev.munition_ids:
                m = self.munitions.get(mid)
                if m is not None:
                    self.munitions[mid] = replace(m, state=MunitionState.LOST_COMMS)
        elif isinstance(ev, IntelUpdateEvent):
            # 情报更新：target_id 对应 threat 消失
            if ev.target_id is not None and ev.status == "disappeared":
                self.remove_threat(ev.target_id)

    # === 快照 ===

    def snapshot(self) -> WorldState:
        """构造不可变 WorldState 快照。"""
        ws = WorldState(
            tick=self.tick,
            munitions=list(self.munitions.values()),
            threats=list(self.threats),
            zones=self.zones.all_zones(),
            comms_blackout_ids={
                mid for mid, m in self.munitions.items()
                if m.state == MunitionState.LOST_COMMS
            },
        )
        validate_world_state(ws)  # 推进后再次校验不变量
        return ws

    # === 查询 ===

    def get_munition(self, munition_id: str) -> Munition | None:
        return self.munitions.get(munition_id)

    def get_munitions(self, ids: list[str] | None = None) -> list[Munition]:
        if ids is None:
            return list(self.munitions.values())
        return [self.munitions[i] for i in ids if i in self.munitions]

    def get_threats_in_zone(self, zone_id: str | None = None) -> list[Threat]:
        """返回 zone 内的威胁列表（无威胁时返回空 list，永不为 None）。"""
        if zone_id is None:
            return list(self.threats)
        z = self.zones.get(zone_id)
        if z is None:
            return []
        xmin, xmax, ymin, ymax, zmin, zmax = z.bounds
        out: list[Threat] = []
        for t in self.threats:
            x, y, zz = t.position
            if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= zz <= zmax:
                out.append(t)
        return out

    def get_zone_coverage(self, zone_id: str) -> float:
        z = self.zones.get(zone_id)
        return z.coverage if z else 0.0
