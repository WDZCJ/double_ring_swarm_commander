"""demo 公共辅助（demos/_common.py）。

提供 telemetry 打印、plan JSON 生成、公共 main 流程。
"""

from __future__ import annotations

import json
from typing import Any

from munition_system.common.config import Config
from munition_system.common.contracts import LLMResponse
from munition_system.common.types import CommanderIntent
from munition_system.runner.loop import DualLoopRunner, RunnerResult
from munition_system.runner.scenario import Scenario, load_scenario


def make_plan_json(
    assignments: list[dict],
    groups: dict[str, list[str]],
    execution_steps: list[str],
    success_conditions: list[str],
    monitoring_points: list[str],
) -> str:
    """构造 BattlePlan JSON 字符串。"""
    return json.dumps({
        "assignments": assignments,
        "groups": groups,
        "execution_steps": execution_steps,
        "success_conditions": success_conditions,
        "monitoring_points": monitoring_points,
    }, ensure_ascii=False)


def print_result(title: str, result: RunnerResult) -> None:
    """打印 demo 运行结果摘要。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"final_phase: {result.final_phase}")
    print(f"replan_count: {result.replan_count}")
    print(f"escalation_count: {result.escalation_count}")
    print(f"safe_mode: {result.safe_mode}")
    if result.warnings:
        print(f"warnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:
            print(f"  - {w}")
    if result.telemetry is not None:
        summary = result.telemetry.summary()
        print(f"\ntelemetry:")
        print(f"  total_ticks: {summary['total_ticks']}")
        print(f"  total_tool_calls: {summary['total_tool_calls']}")
        print(f"  task_changes: {summary['task_changes']}")
        print(f"  confidence_histogram: {summary['confidence_histogram']}")
        print(f"\n  tool call sequence:")
        for tc in result.telemetry.tool_calls:
            status = "ok" if tc.ok else f"FAIL({tc.error})"
            print(f"    tick={tc.tick} {tc.name} {status}")
        if result.telemetry.task_changes:
            print(f"\n  task status changes:")
            for ch in result.telemetry.task_changes:
                print(f"    tick={ch.tick} {ch.task_id} {ch.old_status}->{ch.new_status}")
        if result.telemetry.replans:
            print(f"\n  replans:")
            for rp in result.telemetry.replans:
                print(f"    tick={rp.tick} reason={rp.reason} v={rp.new_plan_version} fails={rp.consecutive_failures}")
    print(f"{'=' * 60}\n")


def run_demo(
    title: str,
    scenario_path: str,
    intent: CommanderIntent,
    llm_responses: list[LLMResponse],
    *,
    max_ticks: int = 50,
    config: Config | None = None,
) -> RunnerResult:
    """通用 demo 主流程。

    scenario_path 支持相对路径（相对项目根目录）或绝对路径。
    """
    cfg = config or Config()
    # 相对路径 → 基于项目根目录（munition_system 的父目录）解析
    from pathlib import Path
    p = Path(scenario_path)
    if not p.is_absolute():
        # demos/ 的父目录的父目录 = 项目根目录
        project_root = Path(__file__).resolve().parent.parent.parent
        p = project_root / p
    scenario = load_scenario(p)
    runner = DualLoopRunner(cfg, scenario, llm_responses=llm_responses)
    result = runner.run(intent, max_ticks=max_ticks)
    print_result(title, result)
    return result
