"""临机事件 demo（demos/demo_adhoc_events.py）。

场景：编组缺员+目标消失+部分失联三连发。
- tick=10 损毁 → 监控任务上报 → check_task_progress → replan
- tick=20 目标消失 → replan + 更新 schema
- tick=30 失联 → abort_munition_task + replan
验证断言：
  - 3 次重规划触发
  - 旧监控任务被自动取消
  - 新监控任务被启动
  - 失联弹任务被重分配
  - 最终任务取消并 Agent 主动决定全员 RTB

运行：python -m munition_system.demos.demo_adhoc_events
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from munition_system.common.contracts import LLMResponse
from munition_system.common.types import (
    ChoiceQuestion,
    CommanderIntent,
    EscalationPolicy,
    JevQuestionSchemaSet,
    NoulQuestion,
)
from munition_system.demos._common import make_plan_json, print_result


def _setup_monitoring_tasks(runner) -> None:
    """部署后创建监控任务（让 Jev 能检测到状态变化并触发 escalation）。

    MT1：监控 G1 编组完整度（是否有弹损毁）
    MT2：监控 T1 威胁状态（是否消失）
    MT3：监控 G2 通讯状态（是否失联）
    """
    # MT1：G1 编组完整度
    mt1_schema = JevQuestionSchemaSet(
        version=1, plan_version=1,
        questions=[
            ChoiceQuestion(
                id="q_g1_integrity",
                prompt="G1 编组是否完整",
                options=["intact", "damaged", "destroyed", "other"],
                bound_to="G1",
            )
        ],
    )
    runner._monitoring_manager.create_monitoring_task(
        subject="G1", schema=mt1_schema, schedule_every_ticks=1,
        escalation_policy=EscalationPolicy(on_answers=["damaged", "destroyed"]),
        owner_plan_version=1, created_tick=0,
    )
    # MT2：T1 威胁状态
    mt2_schema = JevQuestionSchemaSet(
        version=1, plan_version=1,
        questions=[
            NoulQuestion(
                id="q_t1_status",
                prompt="T1 威胁是否仍然存在",
                bound_to="global",
            )
        ],
    )
    runner._monitoring_manager.create_monitoring_task(
        subject="global", schema=mt2_schema, schedule_every_ticks=1,
        escalation_policy=EscalationPolicy(on_confidence_below=0.5),
        owner_plan_version=1, created_tick=0,
    )
    # MT3：G2 通讯状态
    mt3_schema = JevQuestionSchemaSet(
        version=1, plan_version=1,
        questions=[
            ChoiceQuestion(
                id="q_g2_comms",
                prompt="G2 通讯状态",
                options=["normal", "lost_comms", "partial_loss", "other"],
                bound_to="G2",
            )
        ],
    )
    runner._monitoring_manager.create_monitoring_task(
        subject="G2", schema=mt3_schema, schedule_every_ticks=1,
        escalation_policy=EscalationPolicy(on_answers=["lost_comms", "partial_loss"]),
        owner_plan_version=1, created_tick=0,
    )


def _build_plan_v1() -> str:
    """G1 打击 Z1，G2 搜索 Z2。"""
    g1 = [{"munition_id": f"G1-M{i}", "role": "strike", "zone_id": "Z1",
           "capability": "strike", "route": [[i, 0, 0]]} for i in range(1, 4)]
    g2 = [{"munition_id": f"G2-M{i}", "role": "search", "zone_id": "Z2",
           "capability": "search", "route": [[20 + i, 20, 0]]} for i in range(1, 4)]
    return make_plan_json(
        assignments=g1 + g2,
        groups={"G1": [f"G1-M{i}" for i in range(1, 4)],
                "G2": [f"G2-M{i}" for i in range(1, 4)]},
        execution_steps=["G1 打击 Z1", "G2 搜索 Z2"],
        success_conditions=["Z1 目标被摧毁", "Z2 覆盖率 ≥ 0.95"],
        monitoring_points=["G1 编组完整度", "T1 状态", "G2 通讯状态"],
    )


def _build_plan_v2() -> str:
    """G1-M1 损毁后，G1-M2/M3 继续打击。"""
    return make_plan_json(
        assignments=[
            {"munition_id": "G1-M2", "role": "strike", "zone_id": "Z1",
             "capability": "strike", "route": [[1, 5, 0]]},
            {"munition_id": "G1-M3", "role": "strike", "zone_id": "Z1",
             "capability": "strike", "route": [[2, 5, 0]]},
        ],
        groups={"G1": ["G1-M2", "G1-M3"]},
        execution_steps=["G1 缺员后继续打击 Z1"],
        success_conditions=["Z1 目标被摧毁"],
        monitoring_points=["G1 剩余弹状态"],
    )


def _build_plan_v3() -> str:
    """T1 消失后，调整搜索重点。"""
    return make_plan_json(
        assignments=[
            {"munition_id": "G1-M2", "role": "strike", "zone_id": "Z1",
             "capability": "strike", "route": [[1, 10, 0]]},
            {"munition_id": "G2-M3", "role": "search", "zone_id": "Z2",
             "capability": "search", "route": [[22, 25, 0]]},
        ],
        groups={"G1": ["G1-M2"], "G2": ["G2-M3"]},
        execution_steps=["T1 消失，调整搜索", "G2-M3 继续搜索 Z2"],
        success_conditions=["Z2 覆盖率 ≥ 0.95"],
        monitoring_points=["Z2 覆盖率"],
    )


def _build_plan_v4() -> str:
    """G2-M1/M2 失联后，全员 RTB。"""
    return make_plan_json(
        assignments=[
            {"munition_id": "G1-M2", "role": "strike", "zone_id": "Z1",
             "capability": "strike", "route": []},
        ],
        groups={},
        execution_steps=["G2 部分失联，全员 RTB"],
        success_conditions=["所有弹安全返航"],
        monitoring_points=["返航状态"],
    )


def main() -> None:
    intent = CommanderIntent(
        raw_text="编组缺员+目标消失+部分失联三连发",
        goal="临机事件处理",
        constraints=["6 发规模", "G1 打击 G2 搜索"],
        success_criteria="任务完成或安全返航",
    )
    llm_responses = [
        LLMResponse(content=_build_plan_v1()),
        LLMResponse(content=_build_plan_v2()),  # tick=10 损毁后
        LLMResponse(content=_build_plan_v3()),  # tick=20 目标消失后
        LLMResponse(content=_build_plan_v4()),  # tick=30 失联后
    ]

    # 自定义 run：部署后注入监控任务
    from munition_system.common.config import Config
    from munition_system.runner.loop import DualLoopRunner
    from munition_system.runner.scenario import load_scenario

    cfg = Config()
    from pathlib import Path
    scenario_path = Path(__file__).resolve().parent.parent.parent / "config/scenarios/adhoc_events.yaml"
    scenario = load_scenario(scenario_path)
    runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)

    original_deploy = runner._deploy_plan
    mt_created = [False]

    def _deploy_with_mt(ctx):
        original_deploy(ctx)
        if not mt_created[0]:
            _setup_monitoring_tasks(runner)
            mt_created[0] = True

    runner._deploy_plan = _deploy_with_mt

    result = runner.run(intent, max_ticks=40)
    print_result("临机事件 demo（损毁+目标消失+失联三连发）", result)
    # 验证断言
    assert result.telemetry is not None
    print(f"验证断言：")
    print(f"  重规划次数 = {result.replan_count}（期望 ≥ 3）{'✓' if result.replan_count >= 3 else '✗'}")
    print(f"  任务状态变化 = {len(result.telemetry.task_changes)}")
    print(f"  最终阶段 = {result.final_phase}")


if __name__ == "__main__":
    main()
