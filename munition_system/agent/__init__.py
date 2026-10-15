"""外环 Agent 层（agent/）。

ReAct 循环驱动的决策 Agent：
- llm_client.py：OpenAI 兼容 LLM 客户端（real + mock）
- monitoring_manager.py：定时监控任务管理器（生命周期 + escalation 回调）
- tools.py：Agent 层工具（deploy_jev_schema/replan/7 个监控任务工具）+ 注册表
- schema_designer.py：Jev schema 设计器（LLM + 最小化兜底 + 校验重试）
- plan_tracker.py：任务完成度闭环跟踪
- replanner.py：重规划触发 + 安全模式
- react_loop.py：ReAct 循环（Observe→Think→Act→Reflect）
- orchestrator.py：顶层状态机编排器
"""

from munition_system.agent.llm_client import (
    LLMUnavailable,
    MockLLMClient,
    RealLLMClient,
    make_llm_client,
)
from munition_system.agent.monitoring_manager import (
    MonitoringTaskManagerImpl,
    escalation_callback,
)
from munition_system.agent.orchestrator import AgentOrchestrator, AgentPhase
from munition_system.agent.plan_tracker import PlanTracker
from munition_system.agent.react_loop import ReActLoop, ReActResult
from munition_system.agent.replanner import Replanner, SafeModeTriggered
from munition_system.agent.schema_designer import SchemaDesigner
from munition_system.agent.tools import (
    CancelMonitoringTaskTool,
    CreateMonitoringTaskTool,
    DeployJevSchemaTool,
    InspectMonitoringTaskTool,
    ListMonitoringTasksTool,
    PauseMonitoringTaskTool,
    ReplanTool,
    ResumeMonitoringTaskTool,
    UpdateMonitoringTaskSchemaTool,
    build_all_tools,
)

__all__ = [
    "AgentOrchestrator",
    "AgentPhase",
    "CancelMonitoringTaskTool",
    "CreateMonitoringTaskTool",
    "DeployJevSchemaTool",
    "InspectMonitoringTaskTool",
    "LLMUnavailable",
    "ListMonitoringTasksTool",
    "MockLLMClient",
    "MonitoringTaskManagerImpl",
    "PauseMonitoringTaskTool",
    "PlanTracker",
    "ReActLoop",
    "ReActResult",
    "RealLLMClient",
    "ReplanTool",
    "Replanner",
    "ResumeMonitoringTaskTool",
    "SafeModeTriggered",
    "SchemaDesigner",
    "UpdateMonitoringTaskSchemaTool",
    "build_all_tools",
    "escalation_callback",
    "make_llm_client",
]
