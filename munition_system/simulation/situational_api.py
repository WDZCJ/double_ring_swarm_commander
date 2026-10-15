"""实现 SituationalApi Protocol（simulation/situational_api.py）。

MockSituationalApi 包装 World，实现 SituationalApi Protocol。
这是后续替换为真实硬件适配层的接入点——接口不变，实现替换。
"""

from __future__ import annotations

import json
from typing import Any

from munition_system.common.contracts import SituationalApi
from munition_system.common.types import Munition, Threat, WorldState
from munition_system.simulation.world import World


class MockSituationalApi:
    """基于 World 的态势 API mock 实现。

    满足 SituationalApi Protocol（结构子类型，不需显式继承）。
    关键不变量：get_threats_in_zone 永远返回 list（无威胁时为空 list），
    Munition.threat_vector 无威胁时为 (0,0,0) 而非 None。
    """

    def __init__(self, world: World) -> None:
        self._world = world

    def get_world_state(self) -> WorldState:
        return self._world.snapshot()

    def get_munition_status(self, munition_id: str) -> Munition | None:
        return self._world.get_munition(munition_id)

    def get_threats_in_zone(self, zone_id: str | None = None) -> list[Threat]:
        # 永远返回 list，无威胁时为空 list
        return self._world.get_threats_in_zone(zone_id)

    def get_zone_coverage(self, zone_id: str) -> float:
        return self._world.get_zone_coverage(zone_id)

    def get_swarm_status(self, munition_ids: list[str] | None = None) -> list[Munition]:
        return self._world.get_munitions(munition_ids)

    def get_subject_state(self, subject: str) -> str | dict:
        """取 subject 对应 WorldState 子集，序列化为 Jev 输入文本。

        subject 取值：
        - "global"：返回整个 WorldState 的紧凑 JSON 摘要
        - munition_id：返回该弹的状态摘要（位置/状态/燃料/威胁向量/目标 zone）
        - group_id：返回该组所有弹的摘要
        """
        if subject == "global":
            return self._global_state_summary()
        # 单弹
        m = self._world.get_munition(subject)
        if m is not None:
            return self._munition_summary(m)
        # 组：在 BattlePlan.groups 中按 group_id 索引——但 MockSituationalApi
        # 不持有 plan，简化为：组 id 形如 "G<n>"，直接把所有同组前缀的弹聚合
        # 实际场景中 Jev orchestrator 会先查 plan.groups 拿到成员列表再聚合
        # 这里做兼容：若 subject 在 world.munitions 中找不到，按 group 处理
        return self._group_summary(subject)

    def _global_state_summary(self) -> str:
        ws = self._world.snapshot()
        compact = {
            "tick": ws.tick,
            "munitions": [
                {
                    "id": m.id, "state": m.state.value, "role": m.role,
                    "pos": list(m.position), "fuel": round(m.fuel, 2),
                    "tv": list(m.threat_vector),  # (0,0,0) 而非 None
                    "zone": m.target_zone,
                }
                for m in ws.munitions
            ],
            "threats": [
                {"id": t.id, "kind": t.kind, "pos": list(t.position), "lethality": list(t.lethality)}
                for t in ws.threats
            ],
            "zones": [{"id": z.id, "cov": round(z.coverage, 2)} for z in ws.zones],
            "comms_blackout": sorted(ws.comms_blackout_ids),
        }
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    def _munition_summary(self, m: Munition) -> str:
        compact = {
            "id": m.id, "state": m.state.value, "role": m.role,
            "pos": list(m.position), "fuel": round(m.fuel, 2),
            "tv": list(m.threat_vector),  # 永远长度 3
            "zone": m.target_zone,
        }
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    def _group_summary(self, group_id: str) -> str:
        # 组 id 形如 "G1"，约定组内弹 id 形如 "{group_id}-M1"
        # 这是 mock 的简化方案；真实场景由 plan.groups 决定成员
        members = [
            m for mid, m in self._world.munitions.items()
            if mid.startswith(group_id + "-") or mid == group_id
        ]
        if not members:
            return json.dumps({"group": group_id, "members": []}, ensure_ascii=False)
        compact = {
            "group": group_id,
            "members": [
                {
                    "id": m.id, "state": m.state.value,
                    "pos": list(m.position), "tv": list(m.threat_vector),
                    "zone": m.target_zone,
                }
                for m in members
            ],
        }
        return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
