"""巡飞弹状态机（simulation/munition.py）。

状态转移图（核心路径）：
    IDLE → ASSIGNED → SEARCHING → ENGAGING
旁路（可从任意状态进入）：
    RTB（returning）/ LOST_COMMS / DESTROYED

非法动作→自留原态 + 产出 MunitionWarning（不抛异常，由调用方决定如何上报）。
移动逻辑：按 route 推进位置；燃料按距离消耗。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from munition_system.common.types import (
    ActionCommand,
    ActionType,
    Munition,
    MunitionState,
)


class MunitionWarning:
    """巡飞弹动作警告（非法转移/燃料不足/无 route 等）。"""

    def __init__(self, munition_id: str, code: str, message: str) -> None:
        self.munition_id = munition_id
        self.code = code  # illegal_transition | no_route | low_fuel | unknown_action
        self.message = message

    def __repr__(self) -> str:
        return f"MunitionWarning({self.munition_id!r}, {self.code!r}, {self.message!r})"


# 合法的"核心"前进转移（不含旁路）
_FORWARD_TRANSITIONS: dict[MunitionState, set[MunitionState]] = {
    MunitionState.IDLE: {MunitionState.ASSIGNED},
    MunitionState.ASSIGNED: {MunitionState.SEARCHING, MunitionState.ENGAGING},
    MunitionState.SEARCHING: {MunitionState.ENGAGING, MunitionState.ASSIGNED},
    MunitionState.ENGAGING: {MunitionState.SEARCHING},  # 重新搜索
}

# 终态/旁路：可从任意状态进入
_TERMINAL_OR_BYPASS: set[MunitionState] = {
    MunitionState.RTB,
    MunitionState.LOST_COMMS,
    MunitionState.DESTROYED,
}

# 真正的终态（一旦进入不可离开）
_ABSORBING: set[MunitionState] = {
    MunitionState.DESTROYED,
    MunitionState.LOST_COMMS,
}


def _can_transition(src: MunitionState, dst: MunitionState) -> bool:
    """判断 src→dst 是否合法。"""
    if src in _ABSORBING:
        return False  # 终态不可离开
    if src == dst:
        return True  # 自留原态始终合法
    if dst in _TERMINAL_OR_BYPASS:
        return True  # 旁路始终可进入
    if dst in _FORWARD_TRANSITIONS.get(src, set()):
        return True
    return False


# Action → 目标状态映射（基础动作集）
_ACTION_TARGET_STATE: dict[str, MunitionState] = {
    ActionType.MOVE_TO.value: MunitionState.SEARCHING,  # 移动中=搜索态
    ActionType.HOLD.value: None,  # 保持当前态，不变
    ActionType.ENGAGE_TARGET.value: MunitionState.ENGAGING,
    ActionType.RETURN_TO_BASE.value: MunitionState.RTB,
    ActionType.EVADE_LOW_LEVEL.value: MunitionState.EVADING,
    ActionType.ABORT_MISSION.value: MunitionState.RTB,
    ActionType.HOLD_FIRE.value: None,  # 保持当前态
    ActionType.REALLOCATE_TO_ZONE.value: MunitionState.SEARCHING,
    ActionType.ADJUST_ALTITUDE.value: None,  # 不改状态
    ActionType.REASSIGN_ROLE.value: MunitionState.ASSIGNED,
}


def apply_action(
    m: Munition,
    cmd: ActionCommand,
) -> tuple[Munition, list[MunitionWarning]]:
    """对巡飞弹应用动作指令。

    返回 (new_munition, warnings)。非法转移→自留原态 + MunitionWarning。
    燃料不足→警告但仍执行（应急场景由 Agent 决策，不硬阻断）。
    """
    warnings: list[MunitionWarning] = []

    # 终态弹不接受任何动作
    if m.state in _ABSORBING:
        warnings.append(MunitionWarning(
            m.id, "illegal_transition",
            f"巡飞弹 {m.id} 已处终态 {m.state.value}，拒绝动作 {cmd.action}"
        ))
        return m, warnings

    # 动作目标状态
    target_state = _ACTION_TARGET_STATE.get(cmd.action)
    if target_state is None and cmd.action not in _ACTION_TARGET_STATE:
        # 未知动作：自留原态 + 警告（开放枚举——P3 真实系统接入时再校验）
        warnings.append(MunitionWarning(
            m.id, "unknown_action",
            f"巡飞弹 {m.id} 收到未知动作 {cmd.action!r}，保持 {m.state.value}"
        ))
        return m, warnings

    # 校验转移合法性
    if target_state is not None and not _can_transition(m.state, target_state):
        warnings.append(MunitionWarning(
            m.id, "illegal_transition",
            f"巡飞弹 {m.id} 非法转移 {m.state.value}→{target_state.value}（动作 {cmd.action}）"
        ))
        # 自留原态
        target_state = None

    # 位置推进（MoveTo/ReallocateToZone 需要 args.waypoint 或 args.route）
    new_position = m.position
    new_fuel = m.fuel
    if cmd.action in (ActionType.MOVE_TO.value, ActionType.REALLOCATE_TO_ZONE.value):
        waypoint = cmd.args.get("waypoint")
        if isinstance(waypoint, (list, tuple)) and len(waypoint) == 3:
            wp = tuple(int(x) for x in waypoint)
            # 距离 = 欧氏距离的整数近似（曼哈顿距离作为燃料消耗保守估计）
            dist = sum(abs(a - b) for a, b in zip(m.position, wp))
            new_position = wp
            new_fuel = max(0.0, m.fuel - dist * 0.01)  # 每单位距离消耗 0.01 燃料
            if new_fuel <= 0.0 and m.fuel > 0.0:
                warnings.append(MunitionWarning(
                    m.id, "low_fuel",
                    f"巡飞弹 {m.id} 燃料耗尽（剩余 {new_fuel}）"
                ))
        elif cmd.args.get("route"):
            # route 是 [(x,y,z), ...]，取下一航点
            route = cmd.args["route"]
            if isinstance(route, list) and route:
                nxt = route[0]
                if isinstance(nxt, (list, tuple)) and len(nxt) == 3:
                    wp = tuple(int(x) for x in nxt)
                    dist = sum(abs(a - b) for a, b in zip(m.position, wp))
                    new_position = wp
                    new_fuel = max(0.0, m.fuel - dist * 0.01)
        else:
            # 无 waypoint/route 但 MOVE_TO：保持原位 + 警告
            warnings.append(MunitionWarning(
                m.id, "no_route",
                f"巡飞弹 {m.id} 收到 {cmd.action} 但无 waypoint/route"
            ))

    # AdjustAltitude：仅改 z 坐标
    if cmd.action == ActionType.ADJUST_ALTITUDE.value:
        new_alt = cmd.args.get("altitude")
        if isinstance(new_alt, int) and not isinstance(new_alt, bool):
            new_position = (m.position[0], m.position[1], new_alt)
        else:
            warnings.append(MunitionWarning(
                m.id, "no_route",
                f"巡飞弹 {m.id} AdjustAltitude 缺少 altitude 参数"
            ))

    # 构造新 Munition（dataclass.replace 保持不可变风格）
    updates: dict[str, Any] = {"position": new_position, "fuel": new_fuel}
    if target_state is not None:
        updates["state"] = target_state
    # ReassignRole：更新 role
    if cmd.action == ActionType.REASSIGN_ROLE.value:
        new_role = cmd.args.get("role")
        if isinstance(new_role, str) and new_role:
            updates["role"] = new_role
        else:
            warnings.append(MunitionWarning(
                m.id, "no_route",
                f"巡飞弹 {m.id} ReassignRole 缺少 role 参数"
            ))
    # ReallocateToZone：更新 target_zone
    if cmd.action == ActionType.REALLOCATE_TO_ZONE.value:
        new_zone = cmd.args.get("zone_id")
        if isinstance(new_zone, str):
            updates["target_zone"] = new_zone

    return replace(m, **updates), warnings


def make_munition(
    munition_id: str,
    role: str = "unassigned",
    position: tuple[int, int, int] = (0, 0, 0),
    state: MunitionState = MunitionState.IDLE,
    fuel: float = 1.0,
    target_zone: str | None = None,
    threat_vector: tuple[int, int, int] = (0, 0, 0),
) -> Munition:
    """创建巡飞弹实例。threat_vector 默认 (0,0,0) 而非 None。"""
    return Munition(
        id=munition_id,
        role=role,
        position=position,
        state=state,
        fuel=float(fuel),
        target_zone=target_zone,
        threat_vector=tuple(threat_vector),
    )
