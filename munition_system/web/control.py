"""运行时交互控制总线（web/control.py）。

系统运行过程中，外部（HTTP API / Web UI）通过 CommandBus 向正在运行的主循环
提交命令（如下发新任务、注入事件、管理监控任务），实现"人机在环"交互。

线程模型（关键约束）：
- API/Web 线程：只负责 submit 命令 + 读快照，绝不直接改 World/监控任务状态
- runner 线程：唯一的状态写者，在每个 tick 边界 drain 命令并执行 handler

命令语义：
- 重命令（new_intent/replan）每 tick 最多执行 1 条——避免单 tick 内多次 ReAct
  规划把主循环拖死；轻命令（注入事件/监控任务管理/单弹指令）不限制
- 运行时开关（pause/resume/speed/stop）不走队列，直接设置线程标志即时生效

状态机：pending → applied | rejected
队列上限 100，超限 submit 抛 QueueFullError（API 层映射 429）。
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class CommandType(str, Enum):
    """命令类型枚举。"""
    NEW_INTENT = "new_intent"            # 下发新任务（replace 语义：整体覆盖重规划）
    REPLAN = "replan"                    # 触发重规划
    INJECT_EVENT = "inject_event"        # 注入仿真事件（损毁/情报更新/失联）
    MONITORING_CREATE = "monitoring_create"
    MONITORING_PAUSE = "monitoring_pause"
    MONITORING_RESUME = "monitoring_resume"
    MONITORING_CANCEL = "monitoring_cancel"
    MONITORING_UPDATE_SCHEMA = "monitoring_update_schema"
    MUNITION_COMMAND = "munition_command"  # 直接给某发巡飞弹下动作指令
    SAFE_MODE = "safe_mode"              # 强制进入安全模式（全员返航）


# 每 tick 最多执行 1 条的重命令（会触发 LLM/ReAct 规划）
HEAVY_COMMANDS: frozenset[CommandType] = frozenset({
    CommandType.NEW_INTENT,
    CommandType.REPLAN,
})


class QueueFullError(RuntimeError):
    """命令队列已满。"""

    def __init__(self, limit: int) -> None:
        super().__init__(f"命令队列已满（上限 {limit}）")
        self.limit = limit


@dataclass
class Command:
    """一条运行时命令及其执行记录。"""
    command_id: str
    type: CommandType
    payload: dict
    status: str = "pending"          # pending | applied | rejected
    submitted_at: float = field(default_factory=time.time)
    applied_tick: int | None = None
    result: dict | None = None
    error: str | None = None
    completed_at: float | None = None

    @property
    def is_heavy(self) -> bool:
        return self.type in HEAVY_COMMANDS

    def to_dict(self) -> dict:
        return {
            "command_id": self.command_id,
            "type": self.type.value,
            "payload": self.payload,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "applied_tick": self.applied_tick,
            "result": self.result,
            "error": self.error,
            "completed_at": self.completed_at,
        }


class CommandBus:
    """线程安全命令总线：API 线程 submit，runner 线程 drain。"""

    def __init__(self, *, max_queue: int = 100, log_size: int = 500) -> None:
        self._lock = threading.Lock()
        self._pending: deque[Command] = deque()
        self._records: dict[str, Command] = {}
        self._log: deque[Command] = deque(maxlen=log_size)
        self._max_queue = max_queue
        self._counter = itertools.count(1)

    # === 提交 ===

    def submit(self, type: CommandType | str, payload: dict | None = None) -> Command:
        """提交一条命令（线程安全）。队列满时抛 QueueFullError。"""
        ctype = type if isinstance(type, CommandType) else CommandType(type)
        with self._lock:
            if len(self._pending) >= self._max_queue:
                raise QueueFullError(self._max_queue)
            cmd = Command(
                command_id=f"cmd-{next(self._counter)}",
                type=ctype,
                payload=dict(payload or {}),
            )
            self._pending.append(cmd)
            self._records[cmd.command_id] = cmd
            self._log.append(cmd)
            return cmd

    # === 消费（仅 runner 线程调用）===

    def drain(self, *, max_heavy: int = 1) -> list[Command]:
        """取出本 tick 待执行的命令。

        - 全部轻命令 + 至多 max_heavy 条重命令被取出，其余重命令留在队列（保持原序）
        - 已被处理的命令（status != pending）跳过
        """
        with self._lock:
            take: list[Command] = []
            remain: deque[Command] = deque()
            heavy = 0
            for cmd in self._pending:
                if cmd.status != "pending":
                    continue
                if cmd.is_heavy:
                    if heavy < max_heavy:
                        heavy += 1
                        take.append(cmd)
                    else:
                        remain.append(cmd)
                else:
                    take.append(cmd)
            self._pending = remain
            return take

    def complete(
        self,
        cmd: Command,
        *,
        status: str,
        result: dict | None = None,
        applied_tick: int | None = None,
        error: str | None = None,
    ) -> None:
        """登记命令执行结果（applied / rejected）。"""
        with self._lock:
            cmd.status = status
            cmd.result = result
            cmd.error = error
            cmd.applied_tick = applied_tick
            cmd.completed_at = time.time()

    # === 查询 ===

    def get(self, command_id: str) -> Command | None:
        with self._lock:
            return self._records.get(command_id)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def recent(self, limit: int = 50) -> list[dict]:
        """最近提交的命令记录（按时间倒序）。"""
        with self._lock:
            items = list(self._log)[-limit:]
        return [c.to_dict() for c in reversed(items)]

    # === 同步等待（供 ?wait_ms= 使用，API 层经 asyncio.to_thread 调用）===

    def wait(self, command_id: str, timeout: float) -> Command | None:
        """阻塞等待命令完成，超时返回当前记录。"""
        cmd = self.get(command_id)
        if cmd is None:
            return None
        deadline = time.time() + max(0.0, timeout)
        while time.time() < deadline:
            if cmd.status != "pending":
                return cmd
            time.sleep(0.02)
        return self.get(command_id)
