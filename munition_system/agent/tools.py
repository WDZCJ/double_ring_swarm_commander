"""Agent 层工具（agent/tools.py）。

实现 simulation 层无法独立完成的工具（依赖 agent 层状态）：
- deploy_jev_schema：下发全局 Jev 监控 schema 到内环编排器
- replan：触发重规划（委托 Replanner）
- 7 个监控任务工具：create/list/pause/resume/cancel/update_schema/inspect

build_all_tools() 聚合 simulation 层工具 + agent 层工具，返回完整 Tool 列表。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from munition_system.common.contracts import Tool, ToolContext, ToolResult
from munition_system.common.types import (
    EscalationPolicy,
    JevQuestionSchemaSet,
    MonitoringTaskStatus,
)
from munition_system.simulation.tools_impl import (
    _BaseTool,
    make_default_tools,
)

if TYPE_CHECKING:
    from munition_system.agent.monitoring_manager import MonitoringTaskManagerImpl
    from munition_system.agent.replanner import Replanner
    from munition_system.jev.orchestrator import DecisionOrchestrator


# === deploy_jev_schema ===

class DeployJevSchemaTool(_BaseTool):
    """下发全局 Jev 监控 schema 到内环编排器。"""

    name = "deploy_jev_schema"
    description = (
        "下发全局 Jev 监控 schema（每 tick 执行）。\n"
        "schema 格式：{\"version\":N,\"plan_version\":M,\"questions\":[...],"
        "\"escalate_on_confidence_below\":0.5}\n"
        "question.type 必须是 choice | score | noul：\n"
        "  choice: {type:\"choice\",id,prompt,options:[...含 \"other\"...],bound_to}\n"
        "  score:  {type:\"score\",id,prompt,levels:5,desc_low,desc_high,bound_to}\n"
        "  noul:   {type:\"noul\",id,prompt,bound_to}\n"
        "bound_to ∈ plan.groups 的 key / assignments 的 munition_id / \"global\"。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "schema": {"type": "object", "description": "JevQuestionSchemaSet dict"},
        },
        "required": ["schema"],
    }

    def __init__(self, jev_orchestrator: "DecisionOrchestrator") -> None:
        self._jev_orchestrator = jev_orchestrator

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        schema_raw = args.get("schema")
        if not isinstance(schema_raw, dict):
            return ToolResult(ok=False, error="schema 必须是 dict")
        try:
            schema = _parse_schema(schema_raw)
        except Exception as e:
            return ToolResult(ok=False, error=f"schema 解析失败: {e}")
        self._jev_orchestrator.deploy_global_schema(schema)
        return ToolResult(
            ok=True,
            data={"schema_version": schema.version, "plan_version": schema.plan_version},
        )


# === replan ===

class ReplanTool(_BaseTool):
    """触发重规划。"""

    name = "replan"
    description = "触发重规划（重新整体规划，自动取消旧监控任务）。"
    input_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "重规划原因"},
        },
        "required": ["reason"],
    }

    def __init__(self, replanner: "Replanner") -> None:
        self._replanner = replanner

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        reason = str(args.get("reason", "agent_requested"))
        try:
            new_version = self._replanner.trigger(reason, ctx=ctx)
            return ToolResult(
                ok=True,
                data={"new_plan_version": new_version, "reason": reason},
            )
        except SafeModeTriggered as e:
            return ToolResult(
                ok=False,
                error="safe_mode",
                data={"message": str(e)},
            )


# === 监控任务 7 工具 ===

class CreateMonitoringTaskTool(_BaseTool):
    """启动定时监控任务。"""

    name = "create_monitoring_task"
    description = (
        "启动定时监控任务（Jev 周期性监控 subject，按 escalation_policy 上报）。\n"
        "schema 格式同 deploy_jev_schema：question.type ∈ {choice,score,noul}，"
        "choice 需 options 含 \"other\"；score 需 levels/desc_low/desc_high；"
        "noul 只需 id/prompt/bound_to。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "munition_id/group_id/global"},
            "schema": {"type": "object", "description": "JevQuestionSchemaSet dict"},
            "schedule_every_ticks": {"type": "integer", "description": "调度周期（1=每 tick）"},
            "escalation_policy": {"type": "object", "description": "升级策略"},
            "metadata": {"type": "object", "description": "附加说明"},
        },
        "required": ["subject", "schema"],
    }

    def __init__(self, manager: "MonitoringTaskManagerImpl") -> None:
        self._manager = manager

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        subject = args.get("subject")
        schema_raw = args.get("schema")
        if not subject or not isinstance(schema_raw, dict):
            return ToolResult(ok=False, error="subject 和 schema 必填")
        try:
            schema = _parse_schema(schema_raw)
        except Exception as e:
            return ToolResult(ok=False, error=f"schema 解析失败: {e}")
        policy_raw = args.get("escalation_policy") or {}
        policy = EscalationPolicy(
            on_answers=list(policy_raw.get("on_answers", [])),
            on_confidence_below=float(policy_raw.get("on_confidence_below", 0.5)),
            on_no_clear_winner=bool(policy_raw.get("on_no_clear_winner", True)),
        )
        task_id = self._manager.create_monitoring_task(
            subject=subject,
            schema=schema,
            schedule_every_ticks=int(args.get("schedule_every_ticks", 1)),
            escalation_policy=policy,
            metadata=args.get("metadata"),
            owner_plan_version=ctx.plan_version,
            created_tick=ctx.tick,
        )
        return ToolResult(ok=True, data={"task_id": task_id})


class ListMonitoringTasksTool(_BaseTool):
    """列举监控任务。"""

    name = "list_monitoring_tasks"
    description = "列举监控任务及状态。可按 status/subject 过滤。"
    input_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "pending/active/paused/escalated/completed/cancelled"},
            "subject": {"type": "string"},
        },
    }

    def __init__(self, manager: "MonitoringTaskManagerImpl") -> None:
        self._manager = manager

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        status_str = args.get("status")
        status = MonitoringTaskStatus(status_str) if status_str else None
        tasks = self._manager.list_monitoring_tasks(
            status=status,
            subject=args.get("subject"),
        )
        data = [_task_to_dict(t) for t in tasks]
        return ToolResult(ok=True, data=data)


class PauseMonitoringTaskTool(_BaseTool):
    name = "pause_monitoring_task"
    description = "暂停监控任务。"
    input_schema = {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    }

    def __init__(self, manager: "MonitoringTaskManagerImpl") -> None:
        self._manager = manager

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        ok = self._manager.pause_monitoring_task(str(args.get("task_id", "")))
        return ToolResult(ok=ok, data={"ok": ok})


class ResumeMonitoringTaskTool(_BaseTool):
    name = "resume_monitoring_task"
    description = "恢复暂停的监控任务。"
    input_schema = {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    }

    def __init__(self, manager: "MonitoringTaskManagerImpl") -> None:
        self._manager = manager

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        ok = self._manager.resume_monitoring_task(str(args.get("task_id", "")))
        return ToolResult(ok=ok, data={"ok": ok})


class CancelMonitoringTaskTool(_BaseTool):
    name = "cancel_monitoring_task"
    description = "取消监控任务。"
    input_schema = {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    }

    def __init__(self, manager: "MonitoringTaskManagerImpl") -> None:
        self._manager = manager

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        ok = self._manager.cancel_monitoring_task(str(args.get("task_id", "")))
        return ToolResult(ok=ok, data={"ok": ok})


class UpdateMonitoringTaskSchemaTool(_BaseTool):
    name = "update_monitoring_task_schema"
    description = "更新监控任务的 schema（ESCALATED 后常用来改监控重点）。"
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "new_schema": {"type": "object", "description": "新 JevQuestionSchemaSet dict"},
        },
        "required": ["task_id", "new_schema"],
    }

    def __init__(self, manager: "MonitoringTaskManagerImpl") -> None:
        self._manager = manager

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        task_id = str(args.get("task_id", ""))
        new_schema_raw = args.get("new_schema")
        if not isinstance(new_schema_raw, dict):
            return ToolResult(ok=False, error="new_schema 必须是 dict")
        try:
            new_schema = _parse_schema(new_schema_raw)
        except Exception as e:
            return ToolResult(ok=False, error=f"schema 解析失败: {e}")
        ok = self._manager.update_monitoring_task_schema(task_id, new_schema)
        return ToolResult(ok=ok, data={"ok": ok})


class InspectMonitoringTaskTool(_BaseTool):
    name = "inspect_monitoring_task"
    description = "查询监控任务详情（含 last_result）。"
    input_schema = {
        "type": "object",
        "properties": {"task_id": {"type": "string"}},
        "required": ["task_id"],
    }

    def __init__(self, manager: "MonitoringTaskManagerImpl") -> None:
        self._manager = manager

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        task = self._manager.inspect_monitoring_task(str(args.get("task_id", "")))
        if task is None:
            return ToolResult(ok=False, error="task 不存在")
        return ToolResult(ok=True, data=_task_to_dict(task))


# === 注册表 ===

def build_all_tools(
    *,
    jev_orchestrator: "DecisionOrchestrator",
    monitoring_manager: "MonitoringTaskManagerImpl",
    replanner: "Replanner",
) -> list[Tool]:
    """聚合 simulation 层工具 + agent 层工具。"""
    tools = make_default_tools()  # simulation 层 10 个工具
    tools.extend([
        DeployJevSchemaTool(jev_orchestrator),
        ReplanTool(replanner),
        CreateMonitoringTaskTool(monitoring_manager),
        ListMonitoringTasksTool(monitoring_manager),
        PauseMonitoringTaskTool(monitoring_manager),
        ResumeMonitoringTaskTool(monitoring_manager),
        CancelMonitoringTaskTool(monitoring_manager),
        UpdateMonitoringTaskSchemaTool(monitoring_manager),
        InspectMonitoringTaskTool(monitoring_manager),
    ])
    return tools


# === 辅助 ===

def _parse_schema(raw: dict) -> JevQuestionSchemaSet:
    """从 dict 解析 JevQuestionSchemaSet。

    对 LLM 常见错误做容错：
    - type "float"/"rating"/"number"/"scale" → score（缺 levels/desc_* 补默认）
    - type "bool"/"boolean"/"yes_no" → noul
    - choice options 缺 "other" 自动补
    - bound_to 缺省时默认 "global"
    - id/prompt 缺失才报错
    """
    from munition_system.common.types import (
        ChoiceQuestion,
        NoulQuestion,
        ScoreQuestion,
    )
    questions: list = []
    for q_raw in raw.get("questions", []):
        q_type = (q_raw.get("type") or "").lower()
        qid = q_raw.get("id") or ""
        prompt = q_raw.get("prompt") or ""
        bound_to = q_raw.get("bound_to") or "global"
        if not qid or not prompt:
            raise ValueError(f"question 缺 id/prompt: {q_raw}")

        if q_type == "choice":
            opts = list(q_raw.get("options") or [])
            if "other" not in opts:
                opts.append("other")
            questions.append(ChoiceQuestion(
                id=qid, prompt=prompt, options=opts, bound_to=bound_to,
            ))
        elif q_type in ("score", "float", "rating", "number", "scale"):
            levels = int(q_raw.get("levels", 5))
            levels = max(2, min(10, levels))
            desc_low = q_raw.get("desc_low") or "低"
            desc_high = q_raw.get("desc_high") or "高"
            questions.append(ScoreQuestion(
                id=qid, prompt=prompt, levels=levels,
                desc_low=desc_low, desc_high=desc_high, bound_to=bound_to,
            ))
        elif q_type in ("noul", "bool", "boolean", "yes_no", "yesno"):
            questions.append(NoulQuestion(
                id=qid, prompt=prompt, bound_to=bound_to,
            ))
        else:
            # 未知 type → 当 noul 兜底（让 demo 不崩，LLM 收到错误后下次纠正）
            questions.append(NoulQuestion(
                id=qid, prompt=prompt, bound_to=bound_to,
            ))
    return JevQuestionSchemaSet(
        version=int(raw.get("version", 1)),
        plan_version=int(raw.get("plan_version", 0)),
        questions=questions,
        escalate_on_confidence_below=float(raw.get("escalate_on_confidence_below", 0.5)),
    )


def _task_to_dict(task) -> dict:
    """把 MonitoringTask 序列化为 dict（给 LLM 看）。"""
    from dataclasses import asdict
    d = asdict(task)
    # status 转字符串
    d["status"] = task.status.value if hasattr(task.status, "value") else str(task.status)
    # escalation_policy 展平
    if hasattr(task, "escalation_policy"):
        d["escalation_policy"] = asdict(task.escalation_policy)
    return d


class SafeModeTriggered(RuntimeError):
    """重规划连续失败达到上限，触发安全模式。"""
