"""结构合法性 + 不变量断言（validation.py）。

仅做结构校验，不限制 ChoiceQuestion 内容——Agent 自由设计监控策略。
所有 check_* 函数返回 bool；assert_* 版本在失败时抛 ValidationError。
聚合校验入口：validate_battle_plan + validate_jev_schema_set。
"""

from __future__ import annotations

from typing import Iterable

from munition_system.common.types import (
    Answer,
    BattlePlan,
    ChoiceAnswer,
    ChoiceQuestion,
    Escalation,
    EscalationPolicy,
    JevQuestionSchemaSet,
    MonitoringTask,
    Munition,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Threat,
    ThreatVector,
    WorldState,
    Zone,
)


class ValidationError(Exception):
    """结构合法性校验失败。"""


# === ThreatVector 不变量 ===

def check_threat_vector(tv: ThreatVector) -> bool:
    """威胁向量恒为长度 3 的 int 元组；无威胁=(0,0,0)。永不为 None。"""
    if tv is None:
        return False
    if not isinstance(tv, tuple):
        return False
    if len(tv) != 3:
        return False
    return all(isinstance(x, int) and not isinstance(x, bool) for x in tv)


def assert_threat_vector(tv: ThreatVector, context: str = "threat_vector") -> None:
    if not check_threat_vector(tv):
        raise ValidationError(
            f"{context} 必须是长度 3 的 int 元组，得到 {tv!r}（type={type(tv).__name__}）"
        )


# === Munition / Threat / Zone / WorldState ===

def validate_munition(m: Munition) -> None:
    if not m.id:
        raise ValidationError("Munition.id 不能为空")
    if not isinstance(m.position, tuple) or len(m.position) != 3:
        raise ValidationError(f"Munition {m.id} position 必须是长度 3 元组，得到 {m.position!r}")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in m.position):
        raise ValidationError(f"Munition {m.id} position 必须全 int，得到 {m.position!r}")
    if m.fuel < 0:
        raise ValidationError(f"Munition {m.id} fuel 不能为负，得到 {m.fuel}")
    # threat_vector 严格 int 三元组；无威胁=(0,0,0) 而非 None
    assert_threat_vector(m.threat_vector, context=f"Munition {m.id}.threat_vector")


def validate_threat(t: Threat) -> None:
    if not t.id:
        raise ValidationError("Threat.id 不能为空")
    if not isinstance(t.position, tuple) or len(t.position) != 3:
        raise ValidationError(f"Threat {t.id} position 必须是长度 3 元组，得到 {t.position!r}")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in t.position):
        raise ValidationError(f"Threat {t.id} position 必须全 int，得到 {t.position!r}")
    if not t.kind:
        raise ValidationError(f"Threat {t.id} kind 不能为空")
    # lethality 永远 [x,y,z]
    assert_threat_vector(t.lethality, context=f"Threat {t.id}.lethality")


def validate_zone(z: Zone) -> None:
    if not z.id:
        raise ValidationError("Zone.id 不能为空")
    if not isinstance(z.bounds, tuple) or len(z.bounds) != 6:
        raise ValidationError(
            f"Zone {z.id} bounds 必须是长度 6 元组 (xmin,xmax,ymin,ymax,zmin,zmax)，得到 {z.bounds!r}"
        )
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in z.bounds):
        raise ValidationError(f"Zone {z.id} bounds 必须全 int，得到 {z.bounds!r}")
    xmin, xmax, ymin, ymax, zmin, zmax = z.bounds
    if xmin > xmax or ymin > ymax or zmin > zmax:
        raise ValidationError(f"Zone {z.id} bounds 区间反向：{z.bounds!r}")
    if not (0.0 <= z.coverage <= 1.0):
        raise ValidationError(f"Zone {z.id} coverage 必须在 [0,1]，得到 {z.coverage}")


def validate_world_state(ws: WorldState) -> None:
    if ws.tick < 0:
        raise ValidationError(f"WorldState.tick 不能为负，得到 {ws.tick}")
    for m in ws.munitions:
        validate_munition(m)
    for t in ws.threats:
        validate_threat(t)
    for z in ws.zones:
        validate_zone(z)
    if not isinstance(ws.comms_blackout_ids, set):
        raise ValidationError(
            f"WorldState.comms_blackout_ids 必须是 set，得到 {type(ws.comms_blackout_ids).__name__}"
        )


# === Question 不变量 ===

def _check_bound_to(bound_to: str, valid_subjects: Iterable[str]) -> None:
    """bound_to 必须在 plan.groups.keys() 或 assignments.munition_id 中存在，或为 'global'。"""
    if not bound_to:
        raise ValidationError("bound_to 不能为空")
    valid = set(valid_subjects)
    valid.add("global")
    if bound_to not in valid:
        raise ValidationError(
            f"bound_to={bound_to!r} 不在 plan 中（合法值：global / 任一 munition_id / 任一 group_id）"
        )


def validate_choice_question(q: ChoiceQuestion, valid_subjects: Iterable[str]) -> None:
    if not q.id:
        raise ValidationError("ChoiceQuestion.id 不能为空")
    if not q.prompt:
        raise ValidationError(f"ChoiceQuestion {q.id} prompt 不能为空")
    # options ≤ 255，必须含 "other"
    if len(q.options) == 0:
        raise ValidationError(f"ChoiceQuestion {q.id} options 不能为空")
    if len(q.options) > 255:
        raise ValidationError(
            f"ChoiceQuestion {q.id} options 数量 {len(q.options)} 超过上限 255"
        )
    if "other" not in q.options:
        raise ValidationError(f"ChoiceQuestion {q.id} options 必须含 'other'，得到 {q.options!r}")
    # option 唯一性
    if len(set(q.options)) != len(q.options):
        raise ValidationError(f"ChoiceQuestion {q.id} options 存在重复：{q.options!r}")
    _check_bound_to(q.bound_to, valid_subjects)


def validate_score_question(q: ScoreQuestion, valid_subjects: Iterable[str]) -> None:
    if not q.id:
        raise ValidationError("ScoreQuestion.id 不能为空")
    if not q.prompt:
        raise ValidationError(f"ScoreQuestion {q.id} prompt 不能为空")
    # levels ∈ [2, 10]
    if not isinstance(q.levels, int) or isinstance(q.levels, bool):
        raise ValidationError(
            f"ScoreQuestion {q.id} levels 必须是 int，得到 {q.levels!r}（type={type(q.levels).__name__}）"
        )
    if not (2 <= q.levels <= 10):
        raise ValidationError(f"ScoreQuestion {q.id} levels={q.levels} 不在 [2,10]")
    if not q.desc_low or not q.desc_high:
        raise ValidationError(f"ScoreQuestion {q.id} desc_low/desc_high 不能为空")
    _check_bound_to(q.bound_to, valid_subjects)


def validate_noul_question(q: NoulQuestion, valid_subjects: Iterable[str]) -> None:
    if not q.id:
        raise ValidationError("NoulQuestion.id 不能为空")
    if not q.prompt:
        raise ValidationError(f"NoulQuestion {q.id} prompt 不能为空")
    _check_bound_to(q.bound_to, valid_subjects)


# === BattlePlan / JevQuestionSchemaSet 聚合校验 ===

def validate_battle_plan(plan: BattlePlan) -> None:
    """校验作战方案的结构合法性。

    不限制 assignments/execution_steps/success_conditions/monitoring_points 的内容——
    那是 Agent 创造性的输出，仅校验结构边界（≤18 条 assignments 等）。
    """
    if plan.plan_version < 1:
        raise ValidationError(f"BattlePlan.plan_version 必须 ≥1，得到 {plan.plan_version}")
    # assignments ≤ 18（最大集群规模）
    if len(plan.assignments) > 18:
        raise ValidationError(
            f"assignments 数量 {len(plan.assignments)} 超过集群上限 18"
        )
    # munition_id 唯一性（一次 plan 同一弹不能被分配多次）
    ids = [a.munition_id for a in plan.assignments]
    if len(set(ids)) != len(ids):
        raise ValidationError(f"assignments 中存在重复 munition_id：{ids!r}")
    for a in plan.assignments:
        if not a.munition_id:
            raise ValidationError("TaskAssignment.munition_id 不能为空")
        if not a.role:
            raise ValidationError(f"TaskAssignment({a.munition_id}).role 不能为空")
        if not a.capability:
            raise ValidationError(f"TaskAssignment({a.munition_id}).capability 不能为空")
        for p in a.route:
            if not isinstance(p, tuple) or len(p) != 3:
                raise ValidationError(
                    f"TaskAssignment({a.munition_id}) route 点必须是长度 3 元组，得到 {p!r}"
                )
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in p):
                raise ValidationError(
                    f"TaskAssignment({a.munition_id}) route 点必须全 int，得到 {p!r}"
                )
    # groups 中每个 munition_id 必须在 assignments 中存在
    assigned_ids = set(ids)
    for gid, members in plan.groups.items():
        if not gid:
            raise ValidationError("groups 中存在空 group_id")
        if not members:
            raise ValidationError(f"group {gid!r} 成员列表为空")
        for mid in members:
            if mid not in assigned_ids:
                raise ValidationError(
                    f"group {gid!r} 中 munition_id={mid!r} 不在 assignments 中"
                )
        if len(set(members)) != len(members):
            raise ValidationError(f"group {gid!r} 中存在重复 munition_id：{members!r}")
    if not plan.intent.goal:
        raise ValidationError("CommanderIntent.goal 不能为空")
    if not plan.intent.success_criteria:
        raise ValidationError("CommanderIntent.success_criteria 不能为空")


def _collect_valid_subjects(plan: BattlePlan | None) -> set[str]:
    """从 plan 提取所有合法的 bound_to 目标：所有 munition_id + 所有 group_id。"""
    subjects: set[str] = set()
    if plan is not None:
        for a in plan.assignments:
            subjects.add(a.munition_id)
        for gid in plan.groups.keys():
            subjects.add(gid)
    return subjects


def validate_jev_schema_set(
    schema_set: JevQuestionSchemaSet,
    plan: BattlePlan | None = None,
) -> None:
    """校验 Jev 问题集结构。

    - 不限制 questions 内容（Agent 自由设计）
    - 校验：option≤255 含 'other'、levels∈[2,10]、bound_to 在 plan 中存在
    - plan=None 时只校验 bound_to='global'（孤立 schema，如单元测试用）
    """
    if schema_set.version < 1:
        raise ValidationError(f"JevQuestionSchemaSet.version 必须 ≥1，得到 {schema_set.version}")
    if plan is not None and schema_set.plan_version != plan.plan_version:
        raise ValidationError(
            f"schema plan_version={schema_set.plan_version} 与 BattlePlan.plan_version={plan.plan_version} 不匹配"
        )
    if not (0.0 <= schema_set.escalate_on_confidence_below <= 1.0):
        raise ValidationError(
            f"escalate_on_confidence_below={schema_set.escalate_on_confidence_below} 不在 [0,1]"
        )
    if not schema_set.questions:
        raise ValidationError("JevQuestionSchemaSet.questions 不能为空")
    valid_subjects = _collect_valid_subjects(plan)
    qids: list[str] = []
    for q in schema_set.questions:
        if isinstance(q, ChoiceQuestion):
            validate_choice_question(q, valid_subjects)
        elif isinstance(q, ScoreQuestion):
            validate_score_question(q, valid_subjects)
        elif isinstance(q, NoulQuestion):
            validate_noul_question(q, valid_subjects)
        else:
            raise ValidationError(f"未知 question 类型：{type(q).__name__}")
        qids.append(q.id)
    if len(set(qids)) != len(qids):
        raise ValidationError(f"question id 存在重复：{qids!r}")


# === Answer / Escalation / MonitoringTask ===

def validate_answer(a: Answer) -> None:
    if not (0.0 <= a.confidence <= 1.0):
        raise ValidationError(f"Answer {a.question_id} confidence={a.confidence} 不在 [0,1]")
    if isinstance(a, ChoiceAnswer):
        if not a.probabilities:
            raise ValidationError(f"ChoiceAnswer {a.question_id} probabilities 不能为空")
        if a.chosen not in a.probabilities:
            raise ValidationError(
                f"ChoiceAnswer {a.question_id} chosen={a.chosen!r} 不在 probabilities 中"
            )
        total = sum(a.probabilities.values())
        if not (0.99 <= total <= 1.01):
            raise ValidationError(
                f"ChoiceAnswer {a.question_id} probabilities 概率和={total:.4f} 不在 [0.99,1.01]"
            )
        for k, v in a.probabilities.items():
            if not (0.0 <= v <= 1.0):
                raise ValidationError(
                    f"ChoiceAnswer {a.question_id} probability[{k!r}]={v} 不在 [0,1]"
                )
    elif isinstance(a, ScoreAnswer):
        if not isinstance(a.value, (int, float)) or isinstance(a.value, bool):
            raise ValidationError(f"ScoreAnswer {a.question_id} value 必须数值，得到 {a.value!r}")
    elif isinstance(a, NoulAnswer):
        if not (0.0 <= a.probability <= 1.0):
            raise ValidationError(
                f"NoulAnswer {a.question_id} probability={a.probability} 不在 [0,1]"
            )
    else:
        raise ValidationError(f"未知 answer 类型：{type(a).__name__}")


def validate_escalation_policy(p: EscalationPolicy) -> None:
    if not (0.0 <= p.on_confidence_below <= 1.0):
        raise ValidationError(f"EscalationPolicy.on_confidence_below={p.on_confidence_below} 不在 [0,1]")
    for v in p.on_answers:
        if not isinstance(v, str) or not v:
            raise ValidationError(f"EscalationPolicy.on_answers 元素必须非空字符串，得到 {v!r}")


def validate_monitoring_task(
    task: MonitoringTask,
    plan: BattlePlan | None = None,
) -> None:
    """校验定时监控任务。

    - owner_plan_version 应与 plan.plan_version 一致（plan 给定时）
    - schedule_every_ticks ≥ 1
    - schema 自身校验（bound_to 在 task.subject 视角下应合法——
      为简化，schema 中的 bound_to 通常 == task.subject 或 'global'，
      但不强制相等，只要求 bound_to 在 plan 中存在）
    """
    if not task.task_id:
        raise ValidationError("MonitoringTask.task_id 不能为空")
    if task.owner_plan_version < 1:
        raise ValidationError(
            f"MonitoringTask {task.task_id} owner_plan_version 必须 ≥1，得到 {task.owner_plan_version}"
        )
    if plan is not None and task.owner_plan_version != plan.plan_version:
        raise ValidationError(
            f"MonitoringTask {task.task_id} owner_plan_version={task.owner_plan_version} "
            f"与 plan.plan_version={plan.plan_version} 不匹配"
        )
    if not task.subject:
        raise ValidationError(f"MonitoringTask {task.task_id} subject 不能为空")
    if task.schedule_every_ticks < 1:
        raise ValidationError(
            f"MonitoringTask {task.task_id} schedule_every_ticks={task.schedule_every_ticks} 必须 ≥1"
        )
    validate_escalation_policy(task.escalation_policy)
    validate_jev_schema_set(task.schema, plan=plan)
    if task.last_run_tick < -1:
        raise ValidationError(
            f"MonitoringTask {task.task_id} last_run_tick={task.last_run_tick} 必须 ≥ -1"
        )
    if task.created_tick < 0:
        raise ValidationError(
            f"MonitoringTask {task.task_id} created_tick={task.created_tick} 必须 ≥ 0"
        )
    if task.last_result is not None:
        validate_answer(task.last_result)


def validate_escalation(e: Escalation) -> None:
    if e.schema_version < 1:
        raise ValidationError(f"Escalation.schema_version 必须 ≥1，得到 {e.schema_version}")
    if not e.reason:
        raise ValidationError("Escalation.reason 不能为空")
    for a in e.low_confidence_answers:
        validate_answer(a)
