"""SchemaDesigner 单测（tests/test_schema_designer.py）。

覆盖：
- LLM 成功返回合法 schema → 通过校验
- LLM 返回非法 schema → 重试，最终降级为最小化 schema
- 最小化 schema 只含一个 Noul question
"""

from __future__ import annotations

from munition_system.agent.schema_designer import SchemaDesigner
from munition_system.common.contracts import LLMResponse
from munition_system.common.types import (
    BattlePlan,
    CommanderIntent,
    NoulQuestion,
)
from munition_system.agent.llm_client import MockLLMClient


def _make_plan() -> BattlePlan:
    return BattlePlan(
        intent=CommanderIntent(
            raw_text="搜索 A 区",
            goal="搜索",
            constraints=["≤18 弹"],
            success_criteria="覆盖率≥0.95",
        ),
        assignments=[],
        groups={},
        execution_steps=[],
        success_conditions=["覆盖率≥0.95"],
        monitoring_points=["A 区覆盖率"],
        plan_version=1,
    )


class TestSchemaDesigner:
    def test_valid_schema_from_llm(self):
        llm = MockLLMClient()
        llm.set_next_responses([LLMResponse(content='''{
            "version": 1, "plan_version": 1,
            "questions": [{"type":"noul","id":"q1","prompt":"正常?","bound_to":"global"}],
            "escalate_on_confidence_below": 0.5
        }''')])
        designer = SchemaDesigner(llm)
        schema = designer.design(_make_plan(), "summary", version=1)
        assert schema.version == 1
        assert len(schema.questions) == 1
        assert isinstance(schema.questions[0], NoulQuestion)

    def test_invalid_schema_falls_back(self):
        """LLM 连续返回非法 schema → 降级为最小化 schema。"""
        llm = MockLLMClient()
        # 3 次非法（options 不含 other）
        invalid = '{"version":1,"plan_version":1,"questions":[{"type":"choice","id":"q1","prompt":"?","options":["a","b"],"bound_to":"global"}]}'
        llm.set_next_responses([
            LLMResponse(content=invalid),
            LLMResponse(content=invalid),
            LLMResponse(content=invalid),
        ])
        designer = SchemaDesigner(llm, max_retries=2)
        schema = designer.design(_make_plan(), "summary", version=1)
        # 降级为最小化：1 个 Noul
        assert len(schema.questions) == 1
        assert isinstance(schema.questions[0], NoulQuestion)
        assert schema.questions[0].bound_to == "global"

    def test_retry_then_success(self):
        """第一次失败，第二次成功。"""
        llm = MockLLMClient()
        invalid = '{"version":1,"plan_version":1,"questions":[{"type":"choice","id":"q1","prompt":"?","options":["a"],"bound_to":"global"}]}'
        valid = '{"version":1,"plan_version":1,"questions":[{"type":"noul","id":"q2","prompt":"ok?","bound_to":"global"}]}'
        llm.set_next_responses([
            LLMResponse(content=invalid),
            LLMResponse(content=valid),
        ])
        designer = SchemaDesigner(llm, max_retries=2)
        schema = designer.design(_make_plan(), "summary", version=1)
        assert len(schema.questions) == 1
        assert schema.questions[0].id == "q2"
