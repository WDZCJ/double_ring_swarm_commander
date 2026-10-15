"""内环 Jev 决策层（jev/）。

本项目中用 **Laya** 替代 Jev 作为内环决策模型（API 调用，非本地部署）。
Laya 是开源 System 1 决策模型，question 类型 choice/score/noul 与方案中的
Jev question schema 完全一致，返回校准概率——完美替代 Jev。

模块组成：
- client.py：工厂 make_client + Laya 序列化辅助
- real_client.py：LayaRealClient（httpx POST 到 Vercel TypeSafe API）
- mock_client.py：LayaMockClient（三层策略确定性 mock）
- routing.py：置信度路由 + Answer→ActionCommand 映射
- orchestrator.py：DecisionOrchestrator（多任务并发，按方案 6.5）
"""

from munition_system.jev.client import (
    DEFAULT_LAYA_BASE_URL,
    DEFAULT_LAYA_MODEL,
    laya_answer_to_answer,
    make_client,
    question_to_laya,
    serialize_state,
)
from munition_system.jev.mock_client import LayaMockClient
from munition_system.jev.orchestrator import DecisionOrchestrator
from munition_system.jev.real_client import JevUnavailable, LayaRealClient
from munition_system.jev.routing import Routing

__all__ = [
    "DEFAULT_LAYA_BASE_URL",
    "DEFAULT_LAYA_MODEL",
    "DecisionOrchestrator",
    "JevUnavailable",
    "LayaMockClient",
    "LayaRealClient",
    "Routing",
    "laya_answer_to_answer",
    "make_client",
    "question_to_laya",
    "serialize_state",
]
