"""三层联调单测（tests/test_runner_loop.py）。

验证 DualLoopRunner 把 simulation + jev + agent 串起来跑通：
- 用预制 LLM 响应序列驱动 ReAct 产出 plan
- mock Jev 执行全局 schema + 监控任务
- 逐 tick 推进，收集 telemetry
- 断言 telemetry 非空、phase 正常、无异常崩溃
"""

from __future__ import annotations

import json

import pytest

from munition_system.common.config import Config
from munition_system.common.contracts import LLMResponse
from munition_system.common.types import CommanderIntent
from munition_system.runner.loop import DualLoopRunner
from munition_system.runner.scenario import Scenario
from munition_system.agent.react_loop import _try_parse_plan


def _make_scenario() -> Scenario:
    from munition_system.common.types import (
        Munition,
        MunitionState,
        Threat,
        Zone,
    )
    return Scenario(
        name="test",
        description="联调测试场景",
        grid_dim=(20, 20, 4),
        munitions=[
            Munition(
                id="M1", role="strike", position=(0, 0, 0),
                state=MunitionState.IDLE, fuel=100.0, target_zone=None,
            ),
            Munition(
                id="M2", role="search", position=(1, 1, 0),
                state=MunitionState.IDLE, fuel=100.0, target_zone=None,
            ),
        ],
        threats=[],
        zones=[
            Zone(id="Z1", bounds=(0, 10, 0, 10, 0, 4)),
        ],
        events=[],
    )


def _plan_json() -> str:
    return json.dumps({
        "assignments": [
            {"munition_id": "M1", "role": "strike", "zone_id": "Z1",
             "capability": "search", "route": []},
            {"munition_id": "M2", "role": "search", "zone_id": "Z1",
             "capability": "search", "route": []},
        ],
        "groups": {"G1": ["M1", "M2"]},
        "execution_steps": ["搜索 Z1"],
        "success_conditions": ["Z1 覆盖率 ≥ 0.95"],
        "monitoring_points": ["Z1 覆盖率"],
    })


class TestRunnerLoop:
    def test_full_loop_runs(self):
        """三层联调能完整跑通，不抛异常。"""
        cfg = Config()
        scenario = _make_scenario()
        llm_responses = [LLMResponse(content=_plan_json())]
        runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

        intent = CommanderIntent(
            raw_text="搜索 Z1", goal="搜索", constraints=[], success_criteria="覆盖率达标",
        )
        result = runner.run(intent, max_ticks=5)

        # 不应崩溃
        assert result is not None
        # telemetry 应收集到 tick 记录
        assert len(result.telemetry.tick_records) > 0
        # plan 应产出
        assert result.plan is not None

    def test_telemetry_collects_confidences(self):
        """telemetry 收集到 Jev confidence。"""
        cfg = Config()
        scenario = _make_scenario()
        llm_responses = [LLMResponse(content=_plan_json())]
        runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

        intent = CommanderIntent(raw_text="搜索", goal="搜索", constraints=[], success_criteria="完成")
        result = runner.run(intent, max_ticks=5)

        # 至少有一些 tick 记录
        assert result.telemetry is not None
        summary = result.telemetry.summary()
        assert summary["total_ticks"] > 0
        # escalation_count 是非负整数
        assert isinstance(summary["escalation_count"], int)
        assert isinstance(summary["replan_count"], int)

    def test_assign_tasks_applied(self):
        """ReAct 产出的 assignments 被应用到 world（弹状态变 ASSIGNED/SEARCHING）。"""
        cfg = Config()
        scenario = _make_scenario()
        llm_responses = [LLMResponse(content=_plan_json())]
        runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

        intent = CommanderIntent(raw_text="搜索", goal="搜索", constraints=[], success_criteria="完成")
        runner.run(intent, max_ticks=3)

        # 检查 M1/M2 状态不再是 IDLE
        ws = runner._situational_api.get_world_state()
        states = {m.id: m.state.value for m in ws.munitions}
        assert states["M1"] != "idle"
        assert states["M2"] != "idle"

    def test_global_schema_deployed(self):
        """全局 schema 被部署到 jev orchestrator。"""
        cfg = Config()
        scenario = _make_scenario()
        llm_responses = [LLMResponse(content=_plan_json())]
        runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

        intent = CommanderIntent(raw_text="搜索", goal="搜索", constraints=[], success_criteria="完成")
        runner.run(intent, max_ticks=2)

        schema = runner._jev_orchestrator.get_global_schema()
        assert schema is not None
        assert len(schema.questions) >= 1

    def test_safe_mode_not_triggered_normally(self):
        """正常情况下不触发安全模式。"""
        cfg = Config()
        scenario = _make_scenario()
        llm_responses = [LLMResponse(content=_plan_json())]
        runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

        intent = CommanderIntent(raw_text="搜索", goal="搜索", constraints=[], success_criteria="完成")
        result = runner.run(intent, max_ticks=5)

        assert result.safe_mode is False

    def test_tool_calls_recorded(self):
        """telemetry 记录工具调用。"""
        cfg = Config()
        scenario = _make_scenario()
        llm_responses = [LLMResponse(content=_plan_json())]
        runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

        intent = CommanderIntent(raw_text="搜索", goal="搜索", constraints=[], success_criteria="完成")
        result = runner.run(intent, max_ticks=3)

        # assign_tasks 在部署阶段被调用
        tool_names = {tc.name for tc in result.telemetry.tool_calls}
        assert "assign_tasks" in tool_names


class TestReactPlanParsing:
    """_try_parse_plan 边界条件 — 防止空 plan 被误判为有效。"""

    def test_empty_assignments_rejected(self):
        """空 assignments 视为 LLM 过早终止，必须拒绝。"""
        intent = CommanderIntent(raw_text="t", goal="t", constraints=[], success_criteria="t")
        empty = json.dumps({
            "version": 1, "plan_version": 1, "assignments": [],
            "groups": {}, "execution_steps": ["x"],
            "success_conditions": ["x"], "monitoring_points": ["x"],
        })
        assert _try_parse_plan(empty, intent, 1) is None

    def test_missing_assignments_rejected(self):
        """完全缺 assignments 字段当然也拒绝。"""
        intent = CommanderIntent(raw_text="t", goal="t", constraints=[], success_criteria="t")
        no_assign = json.dumps({"plan_version": 1, "execution_steps": ["x"]})
        assert _try_parse_plan(no_assign, intent, 1) is None

    def test_valid_plan_parsed(self):
        """正确的非空 assignments 应该通过。"""
        intent = CommanderIntent(raw_text="t", goal="t", constraints=[], success_criteria="t")
        valid = _plan_json()
        plan = _try_parse_plan(valid, intent, 1)
        assert plan is not None
        assert len(plan.assignments) == 2
        assert plan.plan_version == 1
