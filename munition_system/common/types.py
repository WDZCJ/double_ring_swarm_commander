"""核心数据模型（types.py）。

严格按方案第四节签名实现，含 MonitoringTask/EscalationPolicy/MonitoringTaskStatus
等定时监控任务相关类型。本文件仅定义数据结构，不做行为；行为由各层模块实现。

不变量（由 validation.py 运行时强制）：
- ThreatVector 恒为长度 3 的 int 元组；无威胁时为 (0, 0, 0) 而非 None
- ChoiceQuestion.options 数量 ≤ 255 且必须含 "other"
- ScoreQuestion.levels ∈ [2, 10]
- JevQuestionSchemaSet 中所有 question.bound_to 必须在 plan.groups 或
  assignments.munition_id 中存在（"global" 永远合法）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

# === 基础类型别名 ===

# 威胁向量：永远长度 3 的 int 元组；无威胁=(0,0,0)，永不为 None
ThreatVector = tuple[int, int, int]


# === 巡飞弹与战场实体 ===

class MunitionState(str, Enum):
    """巡飞弹状态机取值。"""
    IDLE = "idle"
    ASSIGNED = "assigned"
    SEARCHING = "searching"
    ENGAGING = "engaging"
    EVADING = "evading"
    RTB = "returning"
    DESTROYED = "destroyed"
    LOST_COMMS = "lost_comms"


class ActionType(str, Enum):
    """基础动作集（开放枚举，Agent 可扩展；后续接真实系统时由真实指令集校验）。

    本原型不卡死动作空间——Agent 在 ReAct 循环中可下发不在本枚举中的字符串
    动作（如扩展动作），由仿真层/真实系统在执行时校验。
    """
    MOVE_TO = "MoveTo"
    HOLD = "Hold"
    ENGAGE_TARGET = "EngageTarget"
    RETURN_TO_BASE = "ReturnToBase"
    REASSIGN_ROLE = "ReassignRole"
    ADJUST_ALTITUDE = "AdjustAltitude"
    EVADE_LOW_LEVEL = "EvadeLowLevel"
    ABORT_MISSION = "AbortMission"
    REALLOCATE_TO_ZONE = "ReallocateToZone"
    HOLD_FIRE = "HoldFire"


@dataclass
class Munition:
    """巡飞弹实体。"""
    id: str
    role: str
    position: tuple[int, int, int]
    state: MunitionState
    fuel: float
    target_zone: str | None
    threat_vector: ThreatVector = (0, 0, 0)  # 默认无威胁；永不为 None


@dataclass
class Threat:
    """威胁实体。lethality 永远是长度 3 的 int 向量。"""
    id: str
    position: tuple[int, int, int]
    kind: str  # radar / sam / small_arms / ...
    lethality: ThreatVector  # 永远 [x, y, z]


@dataclass
class Zone:
    """任务区域。bounds 为 (x_min, x_max, y_min, y_max, z_min, z_max)。"""
    id: str
    bounds: tuple[int, int, int, int, int, int]
    coverage: float = 0.0  # 0.0~1.0


@dataclass
class WorldState:
    """tick 时刻的不可变世界快照。"""
    tick: int
    munitions: list[Munition]
    threats: list[Threat]
    zones: list[Zone]
    comms_blackout_ids: set[str]


# === 外环 Agent 输出 ===

@dataclass
class CommanderIntent:
    """指挥员意图（NL 原文 + 结构化）。"""
    raw_text: str  # 原始 NL
    goal: str  # 结构化目标
    constraints: list[str]
    success_criteria: str


@dataclass
class TaskAssignment:
    """单弹任务分配。route 由航线规划工具产出。"""
    munition_id: str
    role: str
    zone_id: str | None
    capability: str  # search/decoy/strike/escort/...
    route: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass
class BattlePlan:
    """外环 Agent ReAct 循环产出的作战方案。

    Agent 自主思考设计——assignments/execution_steps/success_conditions/
    monitoring_points 都是 Agent 创造性输出，不由模板生成。
    """
    intent: CommanderIntent
    assignments: list[TaskAssignment]  # ≤ 18 条
    groups: dict[str, list[str]]  # group_id -> [munition_id, ...]
    execution_steps: list[str]  # Agent 描述的执行编排（可被解释）
    success_conditions: list[str]  # Agent 自己界定的完成判据
    monitoring_points: list[str]  # Agent 自己识别出的需持续监控事项
    plan_version: int


# === Jev 问题 schema（Agent 自由设计）===

@dataclass
class ChoiceQuestion:
    """选择题：在给定 options 中选一个（含 'other' 兜底）。"""
    id: str
    prompt: str
    options: list[str]  # ≤ 255，必须含 "other"
    bound_to: str  # munition_id | group_id | "global"


@dataclass
class ScoreQuestion:
    """评分题：在 levels 个等级上打分。"""
    id: str
    prompt: str
    levels: int  # 2–10
    desc_low: str
    desc_high: str
    bound_to: str


@dataclass
class NoulQuestion:
    """是非题（Noul）：返回概率（如"任务是否进展正常"）。"""
    id: str
    prompt: str
    bound_to: str


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


@dataclass
class JevQuestionSchemaSet:
    """Agent 下发给内环的 Jev 问题集。

    全局 schema（deploy_jev_schema）每 tick 执行；
    MonitoringTask.schema 是任务专属 schema，按 schedule_every_ticks 周期执行。
    escalate_on_confidence_below：低于此置信度触发上报外环。
    """
    version: int
    plan_version: int
    questions: list[Question]  # Agent 自由设计
    escalate_on_confidence_below: float = 0.5


# === Jev 答案 ===

@dataclass
class ChoiceAnswer:
    """选择题答案：options 的概率分布 + 选中的 + 校准置信度。"""
    question_id: str
    probabilities: dict[str, float]
    chosen: str
    confidence: float


@dataclass
class ScoreAnswer:
    """评分题答案：值 + 校准置信度。"""
    question_id: str
    value: float
    confidence: float


@dataclass
class NoulAnswer:
    """是非题答案：概率 + 校准置信度。"""
    question_id: str
    probability: float
    confidence: float


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


# === 内环输出 ===

@dataclass
class ActionCommand:
    """内环下发给巡飞弹的动作指令。

    action 为 ActionType 字符串值或 Agent 扩展动作（开放枚举）；
    args 由具体动作解释（如 {"target_id": "T1"}）。
    """
    munition_id: str
    action: str  # ActionType 字符串值或扩展动作
    args: dict = field(default_factory=dict)


@dataclass
class Escalation:
    """内环上报外环的升级事件。

    source_monitoring_task_id 为 None 表示来自全局 schema，
    否则为对应 MonitoringTask.task_id。
    """
    schema_version: int
    low_confidence_answers: list[Answer]
    reason: str  # low_confidence | no_clear_winner | boundary_state | monitoring_task_escalation
    source_monitoring_task_id: str | None = None


# === 定时监控任务（核心新增）===

class MonitoringTaskStatus(str, Enum):
    """定时监控任务生命周期状态。

    PENDING → ACTIVE → {ESCALATED, COMPLETED, CANCELLED}；ACTIVE ↔ PAUSED 可双向。
    """
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    ESCALATED = "escalated"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class EscalationPolicy:
    """监控任务的升级策略：满足任一条件即触发上报外环 Agent。

    - on_answers：Choice/Noul 答案值命中即上报（如 ["destroyed", "escalate"]）
    - on_confidence_below：低置信上报
    - on_no_clear_winner：Choice 选项无明确胜出（top-2 概率差 < 阈值）时上报
    """
    on_answers: list[str] = field(default_factory=list)
    on_confidence_below: float = 0.5
    on_no_clear_winner: bool = True


@dataclass
class MonitoringTask:
    """定时监控任务：用 Jev 周期性监控特定 subject，按 escalation_policy 上报。

    Agent 可启动多个并发监控任务，由 MonitoringTaskManager 管理生命周期。
    这是"系统不死板"的关键机制——Agent 不必死板轮询，按需介入。
    """
    task_id: str
    owner_plan_version: int
    subject: str  # munition_id | group_id | "global"
    schema: JevQuestionSchemaSet  # 此任务专属 schema
    schedule_every_ticks: int = 1  # 调度周期：1=每 tick，2=隔 tick
    status: MonitoringTaskStatus = MonitoringTaskStatus.PENDING
    escalation_policy: EscalationPolicy = field(default_factory=EscalationPolicy)
    last_result: Answer | None = None
    last_run_tick: int = -1
    created_tick: int = 0
    metadata: dict = field(default_factory=dict)  # Agent 自由附加说明（如"等待 A 探路结果"）


# === 事件（仿真注入 / 临机事件）===

@dataclass
class DamageEvent:
    """巡飞弹损毁事件。"""
    munition_id: str
    tick: int


@dataclass
class IntelUpdateEvent:
    """情报更新事件（如目标消失、新目标出现）。"""
    target_id: str | None  # None=无对应目标（如全局情报更新）
    status: str
    tick: int


@dataclass
class CommLossEvent:
    """通讯失联事件（影响一组巡飞弹）。"""
    munition_ids: list[str]
    tick: int


Event = Union[DamageEvent, IntelUpdateEvent, CommLossEvent]
