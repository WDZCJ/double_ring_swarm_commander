"""三层联调运行器（runner/）。

P4 交付：把仿真层（simulation）、内环决策层（jev）、外环 Agent 层（agent）
串起来的主循环 + 可观测。

模块：
- telemetry.py：可观测收集（tick 轨迹/工具调用/任务状态/置信度直方图）
- scenario.py：scenario YAML 加载器（munitions/threats/zones/events）
- loop.py：三层联调主循环（装配 + 逐 tick 推进 + 收集 telemetry）
"""

from munition_system.runner.loop import DualLoopRunner, RunnerResult
from munition_system.runner.scenario import Scenario, load_scenario
from munition_system.runner.telemetry import Telemetry, TickRecord

__all__ = [
    "DualLoopRunner",
    "RunnerResult",
    "Scenario",
    "Telemetry",
    "TickRecord",
    "load_scenario",
]
