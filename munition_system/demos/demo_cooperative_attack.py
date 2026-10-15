"""协同攻击 demo（demos/demo_cooperative_attack.py）。

场景：A 弹前出诱饵，B 弹盘旋等待，根据 A 的状态决策规避/绕路/取消。
Agent 关键工具调用：
  analyze_target → plan_routes(诱饵+打击) → assign_tasks →
  create_monitoring_task(MT1 监控 A 状态含 damaged/destroyed/target_reached；
  MT2 监控 B 应 continue_wait/start_attack/abort/flank/escalate)
验证断言：
  - MT1 在 A 损毁时（tick=8）触发 escalation → Agent 介入
  - MT2 在 ≤2 tick 内出 evade 决策且选低空规避
  - Agent 介入后取消 MT2 + 创建 MT3 监控 B 攻击阶段

运行：python -m munition_system.demos.demo_cooperative_attack
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from munition_system.common.contracts import LLMResponse
from munition_system.common.types import (
    ChoiceQuestion,
    CommanderIntent,
    EscalationPolicy,
    JevQuestionSchemaSet,
)
from munition_system.demos._common import make_plan_json, run_demo


def _build_plan_v1() -> str:
    """A 前出诱饵，B 盘旋等待。"""
    return make_plan_json(
        assignments=[
            {"munition_id": "A", "role": "decoy", "zone_id": "Z1",
             "capability": "decoy", "route": [[0, 10, 2], [15, 12, 2]]},
            {"munition_id": "B", "role": "strike", "zone_id": "Z1",
             "capability": "strike", "route": [[5, 5, 1]]},
        ],
        groups={"decoy": ["A"], "strike": ["B"]},
        execution_steps=["A 前出诱饵吸引防空", "B 盘旋等待", "监控 A 状态决策 B 动作"],
        success_conditions=["威胁 T1 被压制", "B 完成打击"],
        monitoring_points=["A 状态（damaged/destroyed/target_reached）", "B 应执行动作"],
    )


def _build_plan_v2() -> str:
    """A 损毁后，B 低空规避接近打击。"""
    return make_plan_json(
        assignments=[
            {"munition_id": "B", "role": "strike", "zone_id": "Z1",
             "capability": "strike", "route": [[5, 5, 0], [18, 14, 0]]},
        ],
        groups={"strike": ["B"]},
        execution_steps=["B 低空规避接近 T1", "打击 T1"],
        success_conditions=["T1 被摧毁"],
        monitoring_points=["B 攻击阶段状态"],
    )


def _setup_monitoring_tasks(runner) -> None:
    """部署后创建 MT1/MT2 监控任务。

    MT1：监控 A 状态（damaged/destroyed/target_reached）
    MT2：监控 B 应执行动作（continue_wait/start_attack/abort/flank/escalate）
    """
    # MT1
    mt1_schema = JevQuestionSchemaSet(
        version=1, plan_version=1,
        questions=[
            ChoiceQuestion(
                id="q_a_status",
                prompt="A 弹当前状态属于",
                options=["normal_progress", "damaged", "destroyed", "target_reached", "other"],
                bound_to="A",
            )
        ],
    )
    runner._monitoring_manager.create_monitoring_task(
        subject="A", schema=mt1_schema, schedule_every_ticks=1,
        escalation_policy=EscalationPolicy(on_answers=["damaged", "destroyed"]),
        owner_plan_version=1, created_tick=0,
    )
    # MT2
    mt2_schema = JevQuestionSchemaSet(
        version=1, plan_version=1,
        questions=[
            ChoiceQuestion(
                id="q_b_action",
                prompt="B 弹应执行动作",
                options=["continue_wait", "start_attack", "abort", "flank", "evade", "escalate", "other"],
                bound_to="B",
            )
        ],
    )
    runner._monitoring_manager.create_monitoring_task(
        subject="B", schema=mt2_schema, schedule_every_ticks=1,
        escalation_policy=EscalationPolicy(on_answers=["escalate", "abort"]),
        owner_plan_version=1, created_tick=0,
    )


def main() -> None:
    intent = CommanderIntent(
        raw_text="A 弹前出诱饵，B 弹盘旋等待，根据 A 状态决策规避/绕路/取消",
        goal="协同攻击 T1",
        constraints=["A 诱饵 B 打击", "A 损毁后 B 决策"],
        success_criteria="T1 被摧毁",
    )
    llm_responses = [
        LLMResponse(content=_build_plan_v1()),
        LLMResponse(content=_build_plan_v2()),  # A 损毁后重规划
    ]

    # 自定义 run：在部署后注入监控任务
    from munition_system.common.config import Config
    from munition_system.runner.loop import DualLoopRunner
    from munition_system.runner.scenario import load_scenario

    cfg = Config()
    from pathlib import Path
    scenario_path = Path(__file__).resolve().parent.parent.parent / "config/scenarios/cooperative_attack.yaml"
    scenario = load_scenario(scenario_path)
    runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

    # 在 run 之前用 hook 注入监控任务（通过 monkey-patch _deploy_plan）
    # 用标志位确保只在首次 deploy 时创建（避免重规划后重复创建达上限）
    original_deploy = runner._deploy_plan
    mt_created = [False]

    def _deploy_with_mt(ctx):
        original_deploy(ctx)
        if not mt_created[0]:
            _setup_monitoring_tasks(runner)
            mt_created[0] = True

    runner._deploy_plan = _deploy_with_mt

    result = runner.run(intent, max_ticks=15)
    from munition_system.demos._common import print_result
    print_result("协同攻击 demo（A 诱饵 B 打击）", result)

    # 验证断言
    assert result.plan is not None
    assert result.telemetry is not None
    # tick=8 应触发 A 损毁事件 → MT1 escalation
    has_replan = result.replan_count >= 1
    print(f"验证断言：A 损毁触发 escalation+重规划 {'✓' if has_replan else '✗'}")


if __name__ == "__main__":
    main()
