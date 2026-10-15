"""Threat 辅助逻辑 + 向量聚合（simulation/threat.py）。

Threat dataclass 已在 common/types.py 定义；本模块提供工厂与聚合函数。
关键不变量：聚合后的 ThreatVector 恒为 tuple[int,int,int]，无威胁时 (0,0,0)。
"""

from __future__ import annotations

from munition_system.common.types import Threat, ThreatVector
from munition_system.common.validation import assert_threat_vector


def make_threat(
    threat_id: str,
    position: tuple[int, int, int],
    kind: str,
    lethality: ThreatVector,
) -> Threat:
    """创建 Threat 实例，校验 lethality 严格 int 三元组。"""
    assert_threat_vector(lethality, context=f"Threat {threat_id}.lethality")
    if not isinstance(position, tuple) or len(position) != 3:
        raise ValueError(f"Threat {threat_id} position 必须长度 3 元组，得到 {position!r}")
    return Threat(id=threat_id, position=position, kind=kind, lethality=lethality)


def aggregate_lethality(threats: list[Threat]) -> ThreatVector:
    """聚合多个威胁的 lethality 向量为单个 ThreatVector。

    按分量取 max（最严重的威胁等级主导），保证结果恒为 tuple[int,int,int]。
    无威胁时返回 (0,0,0) 而非 None。
    """
    if not threats:
        return (0, 0, 0)
    x = max(t.lethality[0] for t in threats)
    y = max(t.lethality[1] for t in threats)
    z = max(t.lethality[2] for t in threats)
    out = (int(x), int(y), int(z))
    assert_threat_vector(out, context="aggregate_lethality result")
    return out


def threats_in_range(
    threats: list[Threat],
    center: tuple[int, int, int],
    radius: int,
) -> list[Threat]:
    """返回距 center 欧氏距离 ≤ radius 的威胁列表（3D 整数坐标）。"""
    cx, cy, cz = center
    out: list[Threat] = []
    for t in threats:
        tx, ty, tz = t.position
        # 整数距离平方比较，避免浮点
        d2 = (tx - cx) ** 2 + (ty - cy) ** 2 + (tz - cz) ** 2
        if d2 <= radius * radius:
            out.append(t)
    return out


def no_threat() -> ThreatVector:
    """返回无威胁向量 (0,0,0)。语义化辅助，避免散落字面量。"""
    return (0, 0, 0)
