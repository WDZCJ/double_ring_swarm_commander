"""Jev schema 设计器（agent/schema_designer.py）。

Agent 通过 LLM 自主设计 Jev 问题集（Choice/Score/Noul），用于监控任务执行。
流程：
1. 输入：BattlePlan + 当前 WorldState（紧凑摘要）
2. LLM 调用（system prompt 内联）：要求根据 monitoring_points 设计问题
3. 结构校验：option≤255 含 'other'、levels∈[2,10]、bound_to 在 plan 中存在
4. 校验失败 → LLM 重试 ≤2 次 → 仍失败则降级为最小化 schema（只问"任务是否进展正常" Noul）

自由生成（不用模板库），结构校验保证合法性。
"""

from __future__ import annotations

import json
from typing import Any

from munition_system.common.contracts import LLMClient
from munition_system.common.types import (
    BattlePlan,
    ChoiceQuestion,
    JevQuestionSchemaSet,
    NoulQuestion,
    ScoreQuestion,
)
from munition_system.common.validation import validate_jev_schema_set

# system prompt 内联（避免创建 prompts/ 文件）
_SCHEMA_DESIGN_SYSTEM_PROMPT = """你是巡飞弹集群作战的内环监控 schema 设计师。
根据指挥员意图、作战方案和当前态势，设计 Jev 决策问题集，用于内环持续监控任务执行。

问题类型：
- choice：选择题，options 必须包含 "other" 兜底
- score：评分题，levels ∈ [2,10]，给出 desc_low/desc_high
- noul：是非题，返回概率

要求：
1. 每个问题 bound_to 必须是 plan.groups 的 key、assignments 的 munition_id 或 "global"
2. 问题应聚焦 monitoring_points 中列出的监控事项
3. 输出严格 JSON：{"version": N, "plan_version": M, "questions": [...]}
4. 每个 question 格式：
   choice: {"type":"choice","id":"...","prompt":"...","options":["a","b","other"],"bound_to":"..."}
   score: {"type":"score","id":"...","prompt":"...","levels":5,"desc_low":"...","desc_high":"...","bound_to":"..."}
   noul: {"type":"noul","id":"...","prompt":"...","bound_to":"..."}
5. escalate_on_confidence_below 默认 0.5

只输出 JSON，不要其他文字。
"""

# 最小化兜底 schema（校验失败时用）
_MINIMAL_FALLBACK_QUESTIONS = [
    NoulQuestion(
        id="q_is_normal",
        prompt="任务是否进展正常？",
        bound_to="global",
    )
]


class SchemaDesigner:
    """Jev schema 设计器。

    依赖 LLMClient。设计失败时降级为最小化 schema。
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def design(
        self,
        plan: BattlePlan,
        world_state_summary: str,
        *,
        version: int = 1,
    ) -> JevQuestionSchemaSet:
        """设计 Jev schema。

        重试 ≤ max_retries 次，仍失败则降级为最小化 schema。
        """
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                schema = self._design_once(plan, world_state_summary, version)
                validate_jev_schema_set(schema, plan=plan)
                return schema
            except Exception as e:
                last_error = str(e)
                continue
        # 降级
        return self._minimal_schema(plan, version)

    def _design_once(
        self,
        plan: BattlePlan,
        world_state_summary: str,
        version: int,
    ) -> JevQuestionSchemaSet:
        """单次 LLM 调用设计 schema。"""
        user_prompt = (
            f"指挥员意图：{plan.intent.raw_text}\n"
            f"目标：{plan.intent.goal}\n"
            f"约束：{plan.intent.constraints}\n"
            f"成功判据：{plan.success_conditions}\n"
            f"监控要点：{plan.monitoring_points}\n"
            f"编组：{plan.groups}\n"
            f"任务分配：{[(a.munition_id, a.role, a.zone_id) for a in plan.assignments]}\n"
            f"当前态势：{world_state_summary}\n"
            f"请设计 version={version}, plan_version={plan.plan_version} 的 Jev schema。"
        )
        messages = [
            {"role": "system", "content": _SCHEMA_DESIGN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        resp = self._llm.chat(messages)
        raw = resp.content.strip()
        # 提取 JSON（可能被 markdown 代码块包裹）
        raw = _extract_json(raw)
        data = json.loads(raw)
        return _parse_schema_data(data, plan.plan_version)

    def _minimal_schema(self, plan: BattlePlan, version: int) -> JevQuestionSchemaSet:
        """最小化兜底 schema：只问"任务是否进展正常"。"""
        return JevQuestionSchemaSet(
            version=version,
            plan_version=plan.plan_version,
            questions=list(_MINIMAL_FALLBACK_QUESTIONS),
            escalate_on_confidence_below=0.5,
        )


def _extract_json(text: str) -> str:
    """从文本中提取 JSON（处理 markdown 代码块）。"""
    text = text.strip()
    if text.startswith("```"):
        # 去掉 ```json ... ```
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    # 找第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _parse_schema_data(data: dict, plan_version: int) -> JevQuestionSchemaSet:
    """从 LLM 返回的 dict 解析 JevQuestionSchemaSet。"""
    questions: list = []
    for q_raw in data.get("questions", []):
        q_type = q_raw.get("type")
        if q_type == "choice":
            questions.append(ChoiceQuestion(
                id=str(q_raw["id"]),
                prompt=str(q_raw["prompt"]),
                options=list(q_raw["options"]),
                bound_to=str(q_raw["bound_to"]),
            ))
        elif q_type == "score":
            questions.append(ScoreQuestion(
                id=str(q_raw["id"]),
                prompt=str(q_raw["prompt"]),
                levels=int(q_raw["levels"]),
                desc_low=str(q_raw["desc_low"]),
                desc_high=str(q_raw["desc_high"]),
                bound_to=str(q_raw["bound_to"]),
            ))
        elif q_type == "noul":
            questions.append(NoulQuestion(
                id=str(q_raw["id"]),
                prompt=str(q_raw["prompt"]),
                bound_to=str(q_raw["bound_to"]),
            ))
        else:
            raise ValueError(f"未知 question type: {q_type}")
    return JevQuestionSchemaSet(
        version=int(data.get("version", 1)),
        plan_version=plan_version,
        questions=questions,
        escalate_on_confidence_below=float(data.get("escalate_on_confidence_below", 0.5)),
    )
