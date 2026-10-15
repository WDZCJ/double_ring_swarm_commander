"""Web 可视化状态管理（web/state.py）。

维护 runner 的当前状态快照，供 REST API 和 WebSocket 推送使用。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

_current_state: dict[str, Any] = {
    "tick": 0,
    "phase": "init",
    "plan_version": 0,
    "replan_count": 0,
    "world": None,
    "monitoring_tasks": [],
    "llm_rounds": [],
    "jev_calls": [],
    "escalations": [],
    "actions": [],
    "plan": None,
    "schema": None,
    # === 运行时交互扩展字段（由 runner 每 tick 发布）===
    "scenario": None,       # 当前加载的场景 key
    "intents": [],          # 指挥员意图历史（含 replace 后的新意图）
    "event_log": [],        # 事件环形缓冲（注入/触发/命令执行）
    "telemetry": {},        # telemetry 摘要
    "runtime": {            # 运行时开关状态
        "running": False,
        "paused": False,
        "safe_mode": False,
        "speed": 1.0,
        "stop_requested": False,
    },
}


def update_state(**kwargs: Any) -> None:
    """更新当前状态。"""
    _current_state.update(kwargs)


def get_current_state() -> dict[str, Any]:
    """获取当前状态快照（深拷贝）。"""
    return json.loads(json.dumps(_current_state, ensure_ascii=False, default=str))


def reset_state() -> None:
    """重置状态。"""
    _current_state.update({
        "tick": 0,
        "phase": "init",
        "plan_version": 0,
        "replan_count": 0,
        "world": None,
        "monitoring_tasks": [],
        "llm_rounds": [],
        "jev_calls": [],
        "escalations": [],
        "actions": [],
        "plan": None,
        "schema": None,
        "scenario": None,
        "intents": [],
        "event_log": [],
        "telemetry": {},
        "runtime": {
            "running": False,
            "paused": False,
            "safe_mode": False,
            "speed": 1.0,
            "stop_requested": False,
        },
    })


# === 数据序列化辅助 ===

def serialize_munition(m: Any) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "position": list(m.position),
        "state": m.state.value if hasattr(m.state, "value") else str(m.state),
        "fuel": round(m.fuel, 1),
        "target_zone": m.target_zone,
        "threat_vector": list(m.threat_vector),
    }


def serialize_threat(t: Any) -> dict:
    return {
        "id": t.id,
        "position": list(t.position),
        "kind": t.kind,
        "lethality": list(t.lethality),
    }


def serialize_zone(z: Any) -> dict:
    return {
        "id": z.id,
        "bounds": list(z.bounds),
        "coverage": round(z.coverage, 3),
    }


def serialize_world(ws: Any) -> dict:
    return {
        "tick": ws.tick,
        "munitions": [serialize_munition(m) for m in ws.munitions],
        "threats": [serialize_threat(t) for t in ws.threats],
        "zones": [serialize_zone(z) for z in ws.zones],
        "comms_blackout_ids": list(ws.comms_blackout_ids),
    }


def serialize_monitoring_task(t: Any) -> dict:
    last_result = None
    if t.last_result is not None:
        r = t.last_result
        last_result = {
            "type": type(r).__name__,
            "question_id": getattr(r, "question_id", ""),
            "confidence": round(getattr(r, "confidence", 0), 3),
        }
        if hasattr(r, "chosen"):
            last_result["chosen"] = r.chosen
            last_result["probabilities"] = {k: round(v, 3) for k, v in r.probabilities.items()}
        elif hasattr(r, "probability"):
            last_result["probability"] = round(r.probability, 3)
        elif hasattr(r, "value"):
            last_result["value"] = round(r.value, 3)

    return {
        "task_id": t.task_id,
        "subject": t.subject,
        "status": t.status.value if hasattr(t.status, "value") else str(t.status),
        "schedule_every_ticks": t.schedule_every_ticks,
        "last_run_tick": t.last_run_tick,
        "created_tick": t.created_tick,
        "owner_plan_version": t.owner_plan_version,
        "last_result": last_result,
        "metadata": t.metadata,
    }


def serialize_llm_round(round_no: int, resp: Any, ctx_label: str = "") -> dict:
    content = (getattr(resp, "content", "") or "").strip()
    tool_calls = getattr(resp, "tool_calls", None) or []
    return {
        "round": round_no,
        "ctx_label": ctx_label,
        "content": content,
        "tool_calls": [
            {"name": tc.get("name", "?"), "args": tc.get("args", {})}
            for tc in tool_calls
        ],
        "has_plan": bool(content and '"assignments"' in content),
    }


def serialize_jev_call(
    tick: int,
    task_id: str | None,
    subject: str,
    state: Any,
    questions: list[Any],
    answers: list[Any],
) -> dict:
    return {
        "tick": tick,
        "task_id": task_id,
        "subject": subject,
        "source": f"监控任务 {task_id}" if task_id else "全局 schema",
        "questions": [
            {
                "type": type(q).__name__,
                "id": q.id,
                "prompt": q.prompt,
                "bound_to": q.bound_to,
            }
            for q in questions
        ],
        "answers": [
            {
                "type": type(a).__name__,
                "question_id": a.question_id,
                "confidence": round(a.confidence, 3),
                **({"chosen": a.chosen, "probabilities": {k: round(v, 3) for k, v in a.probabilities.items()}} if hasattr(a, "chosen") else {}),
                **({"probability": round(a.probability, 3)} if hasattr(a, "probability") else {}),
                **({"value": round(a.value, 3)} if hasattr(a, "value") else {}),
            }
            for a in answers
        ],
    }


def serialize_plan(plan: Any) -> dict | None:
    if plan is None:
        return None
    return {
        "plan_version": plan.plan_version,
        "assignments": [asdict(a) for a in plan.assignments],
        "groups": plan.groups,
        "execution_steps": plan.execution_steps,
        "success_conditions": plan.success_conditions,
        "monitoring_points": plan.monitoring_points,
    }


def serialize_schema(schema: Any) -> dict | None:
    """序列化 JevQuestionSchemaSet（全局 schema 或监控任务 schema）。"""
    if schema is None:
        return None
    return {
        "version": schema.version,
        "plan_version": schema.plan_version,
        "escalate_on_confidence_below": schema.escalate_on_confidence_below,
        "questions": [
            {
                "type": type(q).__name__,
                "id": q.id,
                "prompt": q.prompt,
                "bound_to": q.bound_to,
                **( {"options": list(q.options)} if hasattr(q, "options") else {} ),
                **( {"levels": q.levels, "desc_low": q.desc_low, "desc_high": q.desc_high} if hasattr(q, "levels") else {} ),
            }
            for q in schema.questions
        ],
    }


def serialize_event(ev: Any) -> dict:
    """序列化仿真事件（DamageEvent/IntelUpdateEvent/CommLossEvent）。"""
    return {
        "event_type": type(ev).__name__,
        "tick": getattr(ev, "tick", None),
        "munition_id": getattr(ev, "munition_id", None),
        "munition_ids": list(getattr(ev, "munition_ids", []) or []),
        "target_id": getattr(ev, "target_id", None),
        "status": getattr(ev, "status", None),
    }


def serialize_intent(intent: Any, *, tick: int = 0, plan_version: int = 0) -> dict:
    """序列化指挥员意图（含来源 tick 与 plan 版本）。"""
    return {
        "raw_text": getattr(intent, "raw_text", ""),
        "goal": getattr(intent, "goal", ""),
        "constraints": list(getattr(intent, "constraints", []) or []),
        "success_criteria": getattr(intent, "success_criteria", ""),
        "tick": tick,
        "plan_version": plan_version,
    }
