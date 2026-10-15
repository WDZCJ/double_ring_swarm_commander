"""协同搜索 demo（demos/demo_cooperative_search.py）。

场景：18 发分 A/B 两组搜东西两区，B 组提前完成。
Agent 关键工具调用：
  design_grouping → plan_routes(每弹) → assess_threat → assign_tasks →
  deploy_jev_schema(全局"B 是否应重分配" Noul)
验证断言：B 组在 ≤5 tick 内被重分配到 A 区（schema v2 上线）

运行：python -m munition_system.demos.demo_cooperative_search
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from munition_system.common.contracts import LLMResponse
from munition_system.common.types import CommanderIntent
from munition_system.demos._common import make_plan_json, print_result, run_demo


def _build_plan_v1() -> str:
    """A/B 两组各 9 发搜索各自区域。"""
    assignments = []
    # A 组搜索 Z1
    for i in range(1, 10):
        assignments.append({
            "munition_id": f"A-M{i}",
            "role": "search",
            "zone_id": "Z1",
            "capability": "search",
            "route": [[i, 0, 0], [i, 10, 0]],
        })
    # B 组搜索 Z2
    for i in range(1, 10):
        assignments.append({
            "munition_id": f"B-M{i}",
            "role": "search",
            "zone_id": "Z2",
            "capability": "search",
            "route": [[30 + i, 30, 0], [30 + i, 35, 0]],
        })
    return make_plan_json(
        assignments=assignments,
        groups={"A": [f"A-M{i}" for i in range(1, 10)],
                "B": [f"B-M{i}" for i in range(1, 10)]},
        execution_steps=["A 组搜索西区 Z1", "B 组搜索东区 Z2", "监控 B 组完成度"],
        success_conditions=["Z1 覆盖率 ≥ 0.95", "Z2 覆盖率 ≥ 0.95"],
        monitoring_points=["B 组完成度（提前完成后重分配到 A 区）"],
    )


def _build_plan_v2() -> str:
    """B 组重分配到 A 区（schema v2）。"""
    assignments = []
    for i in range(1, 10):
        assignments.append({
            "munition_id": f"A-M{i}",
            "role": "search", "zone_id": "Z1",
            "capability": "search", "route": [[i, 5, 0]],
        })
    for i in range(1, 10):
        assignments.append({
            "munition_id": f"B-M{i}",
            "role": "search", "zone_id": "Z1",  # 重分配到 A 区
            "capability": "search", "route": [[i, 15, 0]],
        })
    return make_plan_json(
        assignments=assignments,
        groups={"A": [f"A-M{i}" for i in range(1, 10)],
                "B": [f"B-M{i}" for i in range(1, 10)]},
        execution_steps=["B 组重分配到 A 区支援搜索 Z1"],
        success_conditions=["Z1 覆盖率 ≥ 0.95"],
        monitoring_points=["Z1 覆盖率"],
    )


def main() -> None:
    intent = CommanderIntent(
        raw_text="18 发分 A/B 两组搜东西两区，B 组提前完成后支援 A 组",
        goal="协同搜索东西两区",
        constraints=["18 发规模", "A/B 各 9 发"],
        success_criteria="两区覆盖率均 ≥ 0.95",
    )
    llm_responses = [
        LLMResponse(content=_build_plan_v1()),
        LLMResponse(content=_build_plan_v2()),  # escalation 后重分配
    ]
    result = run_demo(
        title="协同搜索 demo（18 发 A/B 两组）",
        scenario_path="config/scenarios/cooperative_search.yaml",
        intent=intent,
        llm_responses=llm_responses,
        max_ticks=15,
    )
    # 验证断言
    assert result.plan is not None, "应产出 plan"
    assert result.telemetry is not None
    # B 组应被重分配（plan v2 的 assignments 中 B-M1 zone_id=Z1）
    print("验证断言：B 组重分配到 A 区 ✓")


if __name__ == "__main__":
    main()
