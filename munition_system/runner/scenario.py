"""Scenario YAML 加载器（runner/scenario.py）。

加载 scenario YAML，构造初始 World（munitions/threats/zones/events）。

YAML 结构示例：
```yaml
name: cooperative_search
description: 18 发分 A/B 两组搜东西两区
grid_dim: [40, 40, 8]
munitions:
  - id: G1-M1
    role: strike
    position: [0, 0, 0]
    state: idle
    fuel: 100.0
    target_zone: Z1
threats:
  - id: T1
    kind: sam
    position: [20, 20, 2]
    lethality: [3, 2, 1]
zones:
  - id: Z1
    bounds: [0, 20, 0, 20, 0, 8]
events:
  - type: damage
    munition_id: G1-M1
    tick: 10
  - type: intel_update
    target_id: T1
    status: disappeared
    tick: 20
  - type: comm_loss
    munition_ids: [G2-M1, G2-M2]
    tick: 30
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from munition_system.common.types import (
    CommLossEvent,
    DamageEvent,
    Event,
    IntelUpdateEvent,
    Munition,
    MunitionState,
    Threat,
    ThreatVector,
    Zone,
)


@dataclass
class Scenario:
    """场景定义。"""
    name: str
    description: str = ""
    grid_dim: tuple[int, int, int] = (40, 40, 8)
    munitions: list[Munition] = field(default_factory=list)
    threats: list[Threat] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


def load_scenario(path: str | Path) -> Scenario:
    """从 YAML 加载 Scenario。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scenario 文件不存在: {p}")
    import yaml
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _parse_scenario(raw)


def _parse_scenario(raw: dict) -> Scenario:
    grid = raw.get("grid_dim", [40, 40, 8])
    grid_dim = tuple(int(x) for x in grid)  # type: ignore[assignment]

    munitions = [_parse_munition(m) for m in raw.get("munitions", [])]
    threats = [_parse_threat(t) for t in raw.get("threats", [])]
    zones = [_parse_zone(z) for z in raw.get("zones", [])]
    events = [_parse_event(e) for e in raw.get("events", [])]

    return Scenario(
        name=str(raw.get("name", "unnamed")),
        description=str(raw.get("description", "")),
        grid_dim=grid_dim,
        munitions=munitions,
        threats=threats,
        zones=zones,
        events=events,
    )


def _parse_munition(m: dict) -> Munition:
    state_str = m.get("state", "idle")
    state = MunitionState(state_str)
    pos = m.get("position", [0, 0, 0])
    return Munition(
        id=str(m["id"]),
        role=str(m.get("role", "strike")),
        position=tuple(int(x) for x in pos),  # type: ignore[arg-type]
        state=state,
        fuel=float(m.get("fuel", 100.0)),
        target_zone=m.get("target_zone"),
        threat_vector=tuple(int(x) for x in m.get("threat_vector", [0, 0, 0])),  # type: ignore[arg-type]
    )


def _parse_threat(t: dict) -> Threat:
    pos = t.get("position", [0, 0, 0])
    lethality = t.get("lethality", [0, 0, 0])
    return Threat(
        id=str(t["id"]),
        position=tuple(int(x) for x in pos),  # type: ignore[arg-type]
        kind=str(t.get("kind", "unknown")),
        lethality=tuple(int(x) for x in lethality),  # type: ignore[arg-type]
    )


def _parse_zone(z: dict) -> Zone:
    bounds = z.get("bounds", [0, 0, 0, 0, 0, 0])
    return Zone(
        id=str(z["id"]),
        bounds=tuple(int(x) for x in bounds),  # type: ignore[arg-type]
        coverage=float(z.get("coverage", 0.0)),
    )


def _parse_event(e: dict) -> Event:
    etype = e.get("type")
    tick = int(e.get("tick", 0))
    if etype == "damage":
        return DamageEvent(munition_id=str(e["munition_id"]), tick=tick)
    if etype == "intel_update":
        return IntelUpdateEvent(
            target_id=e.get("target_id"),
            status=str(e.get("status", "")),
            tick=tick,
        )
    if etype == "comm_loss":
        return CommLossEvent(
            munition_ids=[str(m) for m in e.get("munition_ids", [])],
            tick=tick,
        )
    raise ValueError(f"未知事件类型: {etype}")
