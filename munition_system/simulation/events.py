"""事件 + 注入器（simulation/events.py）。

Damage/IntelUpdate/CommLoss 三类事件已 在 common/types.py 定义；
EventInjector 持有按 tick 排序的事件队列，每 tick 弹出应用并通知 Agent。
Agent 也可反向注入事件（如决定取消某弹任务→注入 DamageEvent）。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

from munition_system.common.types import (
    CommLossEvent,
    DamageEvent,
    Event,
    IntelUpdateEvent,
)


# 事件通知回调：把触发的事件告知 Agent orchestrator
EventListener = Callable[[Event], None]


@dataclass
class _ScheduledEvent:
    """堆元素：(tick, seq, event)。seq 保证同 tick 内 FIFO。"""
    tick: int
    seq: int
    event: Event

    def __lt__(self, other: "_ScheduledEvent") -> bool:
        return (self.tick, self.seq) < (other.tick, other.seq)


class EventInjector:
    """按 tick 排序的事件队列。

    schedule(event)：加入队列（tick 由 event.tick 决定）
    check(tick)：弹出并返回所有 tick<=当前 tick 的事件
    Agent 反向注入：调 schedule 即可
    """

    def __init__(self) -> None:
        self._heap: list[_ScheduledEvent] = []
        self._seq: int = 0  # 自增序号，保证同 tick FIFO
        self._listeners: list[EventListener] = []

    def schedule(self, event: Event) -> None:
        """加入事件队列。event.tick 必须非负。"""
        if event.tick < 0:
            raise ValueError(f"Event.tick 不能为负，得到 {event.tick}")
        heapq.heappush(self._heap, _ScheduledEvent(event.tick, self._seq, event))
        self._seq += 1

    def schedule_at(self, tick: int, event_cls: type, **kwargs: object) -> Event:
        """便捷构造：在 tick 触发 event_cls(**kwargs, tick=tick)。"""
        # event_cls 是 DamageEvent/IntelUpdateEvent/CommLossEvent 之一
        ev: Event = event_cls(**kwargs, tick=tick)  # type: ignore[arg-type]
        self.schedule(ev)
        return ev

    def add_listener(self, listener: EventListener) -> None:
        """注册事件监听器（Agent orchestrator 用此接收事件）。"""
        self._listeners.append(listener)

    def check(self, tick: int) -> list[Event]:
        """弹出所有 tick<=当前 tick 的事件，按 (tick, seq) 排序返回。"""
        fired: list[Event] = []
        while self._heap and self._heap[0].tick <= tick:
            se = heapq.heappop(self._heap)
            fired.append(se.event)
            # 通知监听器
            for listener in self._listeners:
                listener(se.event)
        return fired

    def peek_next_tick(self) -> int | None:
        """返回队列中下一个事件的 tick（不弹出）；空队列返回 None。"""
        return self._heap[0].tick if self._heap else None

    def clear(self) -> None:
        self._heap.clear()
        self._seq = 0

    def __len__(self) -> int:
        return len(self._heap)


def make_damage_event(munition_id: str, tick: int) -> DamageEvent:
    return DamageEvent(munition_id=munition_id, tick=tick)


def make_intel_update_event(target_id: str | None, status: str, tick: int) -> IntelUpdateEvent:
    return IntelUpdateEvent(target_id=target_id, status=status, tick=tick)


def make_comm_loss_event(munition_ids: list[str], tick: int) -> CommLossEvent:
    return CommLossEvent(munition_ids=list(munition_ids), tick=tick)
