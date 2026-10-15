"""Protocol 契约（contracts.py）。

定义五个 Protocol 边界——决定后续真实硬件/真实 Jev/不同 LLM/真实算法的替换点：
- Tool：Agent 工具协议（外环调用底座能力）
- JevClient：内环 Jev 决策客户端（mock / real）
- SituationalApi：态势 API（仿真 mock / 真实硬件适配）
- LLMClient：外环 LLM 客户端（OpenAI 兼容）
- MonitoringTaskManager：定时监控任务管理器（生命周期 + 调度）

Protocol 用 typing.Protocol 定义（PEP 544），实现方不需继承，结构子类型即可。
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from munition_system.common.types import (
    Answer,
    BattlePlan,
    Event,
    JevQuestionSchemaSet,
    MonitoringTask,
    MonitoringTaskStatus,
    Munition,
    Question,
    TaskAssignment,
    Threat,
    WorldState,
)


# === Tool Context / Result ===

class ToolContext:
    """工具调用上下文：工具实现可访问的运行时句柄。

    由 runner/loop.py 在每 tick 构造并注入。工具实现按需读取字段——
    未提供的字段在工具不使用时为 None 也无妨。
    """

    def __init__(
        self,
        *,
        tick: int = 0,
        plan_version: int | None = None,
        world: Any | None = None,  # simulation.world.World，避免循环导入用 Any
        situational_api: "SituationalApi | None" = None,
        monitoring_manager: "MonitoringTaskManager | None" = None,
        plan: BattlePlan | None = None,
    ) -> None:
        self.tick = tick
        self.plan_version = plan_version
        self.world = world
        self.situational_api = situational_api
        self.monitoring_manager = monitoring_manager
        self.plan = plan


class ToolResult:
    """工具调用结果：data + warnings（不抛异常，用 warnings 上报问题）。"""

    def __init__(
        self,
        *,
        ok: bool = True,
        data: Any = None,
        warnings: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        self.ok = ok
        self.data = data
        self.warnings = warnings or []
        self.error = error


class Tool(Protocol):
    """Agent 工具协议：外环 Agent 通过此抽象访问底层能力。

    所有 Mock 实现放在 simulation/tools_impl.py，真实算法/真实硬件通过实现
    同一 Protocol 替换，接口不变。
    """

    name: str
    description: str  # 给 LLM 看的工具说明
    input_schema: dict  # JSON Schema，约束 LLM 调用参数

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        ...


# === JevClient ===

class JevClient(Protocol):
    """内环 Jev 决策客户端协议。

    decide()：输入 state（任意结构，由 situational_api.get_subject_state 产出）
    + questions（Agent 设计的问题集），返回带校准置信度的 Answer 列表。

    实现：
    - mock（jev/mock_client.py）：脚本驱动 + 启发式兜底 + 种子化随机
    - real（jev/real_client.py）：HTTP POST 到 TypeSafe AI 端点
    """

    def decide(
        self,
        state: str | dict | list,
        questions: list[Question],
    ) -> list[Answer]:
        ...


# === SituationalApi ===

class SituationalApi(Protocol):
    """态势 API 协议：仿真 mock 与真实硬件适配的替换点。

    关键不变量：get_threats_in_zone 永远返回 list（无威胁时为空 list 或
    threat_vector=(0,0,0) 的元素），永不为 None。
    """

    def get_world_state(self) -> WorldState:
        ...

    def get_munition_status(self, munition_id: str) -> Munition | None:
        ...

    def get_threats_in_zone(self, zone_id: str | None = None) -> list[Threat]:
        ...

    def get_zone_coverage(self, zone_id: str) -> float:
        ...

    def get_subject_state(
        self,
        subject: str,
    ) -> str | dict:
        """取 subject 对应的 WorldState 子集，序列化为 Jev 输入。

        subject 取值：munition_id / group_id / "global"。
        """

    def get_swarm_status(
        self,
        munition_ids: list[str] | None = None,
    ) -> list[Munition]:
        ...


# === LLMClient ===

class LLMResponse:
    """LLM 响应：内容 + 工具调用决策。

    - content：自然语言思考文本（ReAct 的 Think 步输出）
    - tool_calls：LLM 选择调用的工具列表（name + args）
    - finish_reason：stop | tool_calls | length
    """

    def __init__(
        self,
        *,
        content: str = "",
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason


class LLMClient(Protocol):
    """外环 LLM 客户端协议（OpenAI 兼容）。

    chat()：输入 messages + tools（JSON Schema 列表），返回 LLMResponse。
    P3 实现真实 OpenAI 兼容 client；P5 demo 可降级为离线预制响应（无网络/无 key）。
    """

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
        seed: int = 42,
    ) -> LLMResponse:
        ...


# === MonitoringTaskManager ===

class MonitoringTaskManager(Protocol):
    """定时监控任务管理器协议（核心新增）。

    外环 Agent 通过此管理多个并发监控任务的生命周期；内环 Jev orchestrator
    通过 get_due_tasks(tick) 获取本 tick 应执行的任务。

    通知机制：on_escalation 回调把 ESCALATED 任务告知 Agent orchestrator。
    """

    def create_monitoring_task(
        self,
        subject: str,
        schema: JevQuestionSchemaSet,
        schedule_every_ticks: int = 1,
        escalation_policy: Any | None = None,
        metadata: dict | None = None,
        owner_plan_version: int | None = None,
        created_tick: int = 0,
    ) -> str:
        """创建任务，返回 task_id（自动生成保证唯一）。"""

    def list_monitoring_tasks(
        self,
        status: MonitoringTaskStatus | None = None,
        subject: str | None = None,
    ) -> list[MonitoringTask]:
        ...

    def pause_monitoring_task(self, task_id: str) -> bool:
        ...

    def resume_monitoring_task(self, task_id: str) -> bool:
        ...

    def cancel_monitoring_task(self, task_id: str) -> bool:
        ...

    def update_monitoring_task_schema(
        self,
        task_id: str,
        new_schema: JevQuestionSchemaSet,
    ) -> bool:
        ...

    def inspect_monitoring_task(self, task_id: str) -> MonitoringTask | None:
        ...

    def get_due_tasks(self, tick: int) -> list[MonitoringTask]:
        """返回本 tick 应执行的 ACTIVE 任务（schedule_every_ticks 匹配）。"""

    def mark_escalated(
        self,
        task_id: str,
        result: Answer,
        tick: int,
    ) -> None:
        """内环执行任务后，若触发 escalation_policy，标记任务 ESCALATED 并通知 Agent。"""

    def cancel_tasks_for_plan(self, new_plan_version: int) -> int:
        """重规划时自动取消所有 owner_plan_version 不匹配的任务，返回取消数。"""


# === 工具类型别名（供 tools_impl 注册表使用） ===

ToolFactory = Callable[[Any], Tool]
"""工具工厂：注入依赖（SituationalApi/World/MonitoringTaskManager）构造工具实例。"""
