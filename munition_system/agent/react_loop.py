"""ReAct 循环（agent/react_loop.py）。

单次循环：
1. Observe：调 get_swarm_status + get_battlefield_situation + list_monitoring_tasks（ESCALATED）
2. Think：LLM 推理（system prompt + tool descriptions + 当前观察 + 待处理 escalation）
3. Act：执行 LLM 选择的工具调用 → ToolResult
4. Reflect：把 ToolResult 加入 LLM 上下文 → 回到 Think
5. 终止：LLM 产出 BattlePlan（assignments + execution_steps + success_conditions + monitoring_points）
   或 max_iterations 耗尽

System prompt 强调：自主思考、可超出预料、确保任务可完成、善用监控任务并发感知态势。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from munition_system.common.contracts import (
    LLMClient,
    Tool,
    ToolContext,
    ToolResult,
)
from munition_system.common.types import (
    BattlePlan,
    CommanderIntent,
    Escalation,
    TaskAssignment,
)

if TYPE_CHECKING:
    from munition_system.agent.monitoring_manager import MonitoringTaskManagerImpl

# ReAct system prompt
_REACT_SYSTEM_PROMPT = """你是巡飞弹集群作战的外环决策 Agent，采用 ReAct 循环自主规划。

核心原则：
1. 自主思考：不依赖预设模板，根据态势创造性决策
2. 确保任务可完成：所有决策必须服务于指挥员意图的成功达成
3. 善用监控任务：用 create_monitoring_task 启动并发定时监控，不必死板轮询
4. 灵活应对：态势变化时主动 replan 或 update_monitoring_task_schema
5. 高效收敛：观察足够即输出 plan，不要无意义循环

部署阶段会自动处理（不要在 ReAct 阶段重复调用）：
- assign_tasks：用你输出 plan.assignments 自动部署
- deploy_jev_schema：用 schema_designer 基于你 plan.monitoring_points 自动设计并部署全局 schema

ReAct 阶段你应该聚焦：
- 调用 get_swarm_status / get_battlefield_situation 观察态势
- 调用 design_grouping / plan_routes / assess_threat 等做规划辅助
- 调用 create_monitoring_task 建立定时监控（部署阶段不会自动建监控任务）
- 一旦态势清楚、监控任务部署完毕，立即输出 BattlePlan JSON 终止循环

重规划场景（pending_escalations 非空时）：
- 简短观察 + 一次性输出新 BattlePlan，不要重新跑完整 ReAct
- 复用原 plan.assignments.route，只调整必要部分

可用工具：通过 function calling 调用。
最终产出 BattlePlan：包含 assignments、execution_steps、success_conditions、monitoring_points。

当你认为规划完成时，输出一段 JSON（只输出 JSON，不要其他文字）：
{"intent_raw":"...","goal":"...","constraints":[...],"success_criteria":"...",
 "assignments":[{"munition_id":"...","role":"...","zone_id":"...","capability":"...","route":[[x,y,z],...]}],
 "groups":{"G1":["M1",...]},
 "execution_steps":["..."],
 "success_conditions":["..."],
 "monitoring_points":["..."]}
"""


@dataclass
class ReActResult:
    """ReAct 循环结果。"""
    plan: BattlePlan | None = None
    tool_calls: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    iterations: int = 0
    finished: bool = False


class ReActLoop:
    """ReAct 循环：Observe→Think→Act→Reflect。"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: list[Tool],
        monitoring_manager: "MonitoringTaskManagerImpl",
        max_iterations: int = 20,
        fallback_to_safe_intent: bool = True,
    ) -> None:
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._monitoring_manager = monitoring_manager
        self._max_iterations = max_iterations
        self._fallback = fallback_to_safe_intent

    def run(
        self,
        intent: CommanderIntent,
        ctx: ToolContext,
        *,
        pending_escalations: list[Escalation] | None = None,
        plan_version: int = 1,
    ) -> ReActResult:
        """执行 ReAct 循环，产出 BattlePlan 或 None。"""
        pending_escalations = pending_escalations or []
        messages: list[dict] = [
            {"role": "system", "content": _REACT_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_initial_user_prompt(intent, pending_escalations)},
        ]
        tool_schemas = _tools_to_openai_schema(list(self._tools.values()))
        tool_call_log: list[dict] = []
        warnings: list[str] = []

        for iteration in range(self._max_iterations):
            # Think
            try:
                resp = self._llm.chat(messages, tools=tool_schemas)
            except Exception as e:
                warnings.append(f"LLM 调用失败: {e}")
                break

            # 如果 LLM 直接输出了 plan JSON（content 中）
            plan = _try_parse_plan(resp.content, intent, plan_version)
            if plan is not None:
                tool_call_log.append({"type": "plan", "content": resp.content[:200]})
                return ReActResult(
                    plan=plan,
                    tool_calls=tool_call_log,
                    warnings=warnings,
                    iterations=iteration + 1,
                    finished=True,
                )

            # Act：执行工具调用
            if not resp.tool_calls:
                # 无工具调用且无 plan → 终止（LLM 认为无法继续）
                if resp.content.strip():
                    messages.append({"role": "assistant", "content": resp.content})
                break

            # 批量执行 tool_calls，组装 OpenAI 兼容消息：
            # - 一条 assistant 消息含全部 tool_calls（带 id/type/function）
            # - N 条 tool 消息（tool_call_id 与上面 id 配对）
            openai_tool_calls: list[dict] = []
            tool_results: list[tuple[str, str, dict, Any]] = []
            for i, tc in enumerate(resp.tool_calls):
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                tool_call_id = f"call_{iteration}_{i}"
                openai_tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                })
                tool = self._tools.get(name)
                if tool is None:
                    warnings.append(f"未知工具: {name}")
                    tool_results.append((tool_call_id, name, args, None))
                    continue
                result = tool.call(args, ctx)
                tool_call_log.append({
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                    "ok": result.ok,
                })
                tool_results.append((tool_call_id, name, args, result))

            # 一条 assistant 消息（含全部 OpenAI 格式 tool_calls）
            messages.append({
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": openai_tool_calls,
            })
            # N 条 tool 消息（每个 tool_call 一条，id 配对）
            for tool_call_id, name, _args, result in tool_results:
                if result is None:
                    content = json.dumps(
                        {"error": f"未知工具 {name}"}, ensure_ascii=False
                    )
                else:
                    content = json.dumps(
                        {
                            "ok": result.ok,
                            "data": result.data,
                            "warnings": result.warnings,
                            "error": result.error,
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                })

        # 耗尽迭代或异常 → 降级
        if self._fallback:
            return ReActResult(
                plan=_safe_fallback_plan(intent, plan_version),
                tool_calls=tool_call_log,
                warnings=warnings + ["reached_max_iterations_or_llm_failure_fallback_to_safe"],
                iterations=self._max_iterations,
                finished=True,
            )
        return ReActResult(
            plan=None,
            tool_calls=tool_call_log,
            warnings=warnings,
            iterations=self._max_iterations,
            finished=False,
        )

    def _build_initial_user_prompt(
        self,
        intent: CommanderIntent,
        pending_escalations: list[Escalation],
    ) -> str:
        parts = [
            f"指挥员意图：{intent.raw_text}",
            f"目标：{intent.goal}",
            f"约束：{intent.constraints}",
            f"成功判据：{intent.success_criteria}",
        ]
        if pending_escalations:
            parts.append(f"待处理升级事件：{len(pending_escalations)} 条")
            for e in pending_escalations:
                parts.append(
                    f"  - reason={e.reason}, source_task={e.source_monitoring_task_id}, "
                    f"low_conf_answers={len(e.low_confidence_answers)}"
                )
        parts.append("请开始规划。先观察态势，再调用工具，最后输出 BattlePlan JSON。")
        return "\n".join(parts)


# === 辅助函数 ===

def _tools_to_openai_schema(tools: list[Tool]) -> list[dict]:
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        })
    return out


def _try_parse_plan(content: str, intent: CommanderIntent, plan_version: int) -> BattlePlan | None:
    """尝试从 LLM content 解析 BattlePlan JSON。

    必须同时满足：
    1. 有 assignments 字段
    2. assignments 非空（空数组视作 LLM 过早终止，拒绝以让 ReAct 继续迭代）
    """
    if not content or "{" not in content:
        return None
    # 提取 JSON
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return None
    # 必须有 assignments 字段且非空
    if "assignments" not in data:
        return None
    raw_assignments = data.get("assignments")
    if not raw_assignments:
        return None  # 空 assignments → LLM 过早终止，拒绝
    assignments = [
        TaskAssignment(
            munition_id=str(a.get("munition_id", "")),
            role=str(a.get("role", "")),
            zone_id=a.get("zone_id"),
            capability=str(a.get("capability", "")),
            route=[tuple(wp) for wp in a.get("route", []) if isinstance(wp, (list, tuple))],
        )
        for a in raw_assignments
    ]
    return BattlePlan(
        intent=intent,
        assignments=assignments,
        groups=data.get("groups", {}),
        execution_steps=list(data.get("execution_steps", [])),
        success_conditions=list(data.get("success_conditions", [])),
        monitoring_points=list(data.get("monitoring_points", [])),
        plan_version=plan_version,
    )


def _safe_fallback_plan(intent: CommanderIntent, plan_version: int) -> BattlePlan:
    """安全降级 plan：全员 RTB。"""
    return BattlePlan(
        intent=intent,
        assignments=[],
        groups={},
        execution_steps=["全员返航（安全降级）"],
        success_conditions=["所有巡飞弹安全返航"],
        monitoring_points=["集群返航状态"],
        plan_version=plan_version,
    )
