"""内环决策编排器（jev/orchestrator.py）。

DecisionOrchestrator 每 tick 执行（按方案 6.5）：
1. 取 WorldState 快照
2. 执行全局 schema（按 question.bound_to 分组，每组调 decide）
   - 调 JevClient.decide(state, questions) → Answer[]
   - 交 Routing.route → (ActionCommand[], Escalation | None)
3. 执行所有 active 监控任务（MonitoringTaskManager.get_due_tasks）
   - 对每个任务：取 subject state + 任务 schema → decide → 写回 last_result/last_run_tick
   - 检查 escalation_policy：触发则标记 ESCALATED + 通知外环
   - 高置信答案 → map_actions → ActionCommand[]
4. 输出合并 (ActionCommand[], Escalation[]) 给仿真层执行 + 外环 Agent 处理

无状态于 schema/task 之外：task 注册表由 Manager 维护，编排器只负责执行；
schema/task 切换时无需迁移状态。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from munition_system.common.contracts import (
    JevClient,
    MonitoringTaskManager,
    SituationalApi,
)
from munition_system.common.types import (
    ActionCommand,
    Answer,
    ChoiceAnswer,
    Escalation,
    JevQuestionSchemaSet,
    MonitoringTask,
    Question,
)

if TYPE_CHECKING:
    from munition_system.jev.routing import Routing


class DecisionOrchestrator:
    """内环决策编排器：每 tick 整合全局 schema + 所有 active 监控任务。

    构造依赖：
    - client: JevClient（mock 或 real Laya）
    - situational_api: SituationalApi（取 subject state）
    - routing: Routing（置信度路由 + action 映射）
    - monitoring_manager: MonitoringTaskManager | None（P3 实现，P2 可为 None）
    """

    def __init__(
        self,
        *,
        client: JevClient,
        situational_api: SituationalApi,
        routing: "Routing",
        monitoring_manager: MonitoringTaskManager | None = None,
    ) -> None:
        self._client = client
        self._situational_api = situational_api
        self._routing = routing
        self._monitoring_manager = monitoring_manager
        self._global_schema: JevQuestionSchemaSet | None = None

    def deploy_global_schema(self, schema: JevQuestionSchemaSet) -> None:
        """部署全局 schema（Agent 调用，每 tick 执行）。

        替换之前的全局 schema。全局 schema 用于集群级持续监控
        （如"B 组完成度 Score"、"是否需要重分配 Noul"）。
        """
        self._global_schema = schema

    def get_global_schema(self) -> JevQuestionSchemaSet | None:
        """查询当前全局 schema（Agent inspect 用）。"""
        return self._global_schema

    def tick(self, tick: int) -> tuple[list[ActionCommand], list[Escalation]]:
        """推进一个 tick 的内环决策。

        流程（方案 6.5）：
        1. 全局 schema 执行（如有）
        2. 所有 active 监控任务执行（如有 manager）
        3. 返回合并 (actions, escalations)
        """
        all_actions: list[ActionCommand] = []
        all_escalations: list[Escalation] = []

        # 1) 全局 schema（每 tick 都跑）
        if self._global_schema is not None:
            actions, esc = self._run_global_schema(tick)
            all_actions.extend(actions)
            if esc is not None:
                all_escalations.append(esc)

        # 2) 监控任务（按 schedule_every_ticks 周期执行）
        if self._monitoring_manager is not None:
            due_tasks = self._monitoring_manager.get_due_tasks(tick)
            for task in due_tasks:
                actions, escs = self._run_task(task, tick)
                all_actions.extend(actions)
                all_escalations.extend(escs)

        return all_actions, all_escalations

    # === 全局 schema 执行 ===

    def _run_global_schema(
        self,
        tick: int,
    ) -> tuple[list[ActionCommand], Escalation | None]:
        """执行全局 schema：按 bound_to 分组，每组调 decide。

        全局 schema 的 questions 可能 bound_to 不同（如 "global"、
        某个 munition_id、某个 group_id）。按 bound_to 分组，每组取一次
        subject state，调一次 decide。最后合并所有 answers 交 routing.route。
        """
        schema = self._global_schema
        assert schema is not None

        all_answers, all_questions = self._decide_grouped(schema.questions, tick, task_id=None)

        return self._routing.route(
            all_answers,
            all_questions,
            subject="global",
            schema_version=schema.version,
            source_monitoring_task_id=None,
        )

    # === 监控任务执行 ===

    def _run_task(
        self,
        task: MonitoringTask,
        tick: int,
    ) -> tuple[list[ActionCommand], list[Escalation]]:
        """执行单个监控任务。

        1. 调 decide 得到 answers
        2. 写回 task.last_result / task.last_run_tick
        3. 高置信 answers → map_actions → ActionCommand[]
        4. 检查 escalation_policy：触发则标记 ESCALATED + 产生 Escalation
        """
        answers, _ = self._decide_grouped(
            task.schema.questions,
            tick,
            task_id=task.task_id,
            subject_override=task.subject,
        )

        # 写回 task（dataclass 可变字段）
        task.last_run_tick = tick
        task.last_result = answers[0] if answers else None

        # action 映射（只做高置信 Choice → ActionCommand，escalation 由 policy 处理）
        actions = self._routing.map_actions(
            answers,
            task.schema.questions,
            subject=task.subject,
        )

        # 检查 escalation_policy
        esc = self._check_policy(task, answers)
        escalations: list[Escalation] = []
        if esc is not None:
            escalations.append(esc)
            # 标记任务 ESCALATED（通知外环 Agent）
            if self._monitoring_manager is not None:
                self._monitoring_manager.mark_escalated(
                    task.task_id,
                    answers[0] if answers else None,  # type: ignore[arg-type]
                    tick,
                )

        return actions, escalations

    def _check_policy(
        self,
        task: MonitoringTask,
        answers: list[Answer],
    ) -> Escalation | None:
        """检查 escalation_policy 是否触发。

        - on_answers：Choice answer.chosen 命中即触发
        - on_confidence_below：answer.confidence 低于此值触发
        - on_no_clear_winner：Choice top-2 概率差 < no_clear_winner_gap 触发
        """
        policy = task.escalation_policy
        triggered: list[Answer] = []

        for ans in answers:
            # on_answers（Choice chosen 命中）
            if isinstance(ans, ChoiceAnswer) and ans.chosen in policy.on_answers:
                triggered.append(ans)
                continue
            # on_confidence_below
            if ans.confidence < policy.on_confidence_below:
                triggered.append(ans)
                continue
            # on_no_clear_winner（Choice top-2 差小于阈值）
            if policy.on_no_clear_winner and isinstance(ans, ChoiceAnswer):
                sorted_probs = sorted(ans.probabilities.values(), reverse=True)
                top1 = sorted_probs[0] if sorted_probs else 0.0
                top2 = sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
                gap = top1 - top2
                if gap < self._routing.no_clear_winner_gap:
                    triggered.append(ans)

        if not triggered:
            return None

        return Escalation(
            schema_version=task.schema.version,
            low_confidence_answers=triggered,
            reason="monitoring_task_escalation",
            source_monitoring_task_id=task.task_id,
        )

    # === decide 分组执行 ===

    def _decide_grouped(
        self,
        questions: list[Question],
        tick: int,
        *,
        task_id: str | None,
        subject_override: str | None = None,
    ) -> tuple[list[Answer], list[Question]]:
        """按 question.bound_to 分组，每组调一次 decide。

        subject_override：监控任务用 task.subject 覆盖 bound_to（任务专一监控）。
        返回 (所有 answers 合并, 所有 questions 合并)。
        """
        if not questions:
            return [], []

        groups: dict[str, list[Question]] = {}
        for q in questions:
            bound = subject_override or q.bound_to
            groups.setdefault(bound, []).append(q)

        all_answers: list[Answer] = []
        all_questions: list[Question] = []
        for bound, qs in groups.items():
            state = self._situational_api.get_subject_state(bound)
            self._set_client_context(tick, task_id)
            answers = self._client.decide(state, qs)
            all_answers.extend(answers)
            all_questions.extend(qs)

        return all_answers, all_questions

    def _set_client_context(self, tick: int, task_id: str | None) -> None:
        """设置 mock client 上下文（duck typing）。

        LayaMockClient 有 set_context 方法；LayaRealClient 没有（无影响）。
        orchestrator 用 hasattr 检查，避免依赖具体类型。
        """
        client = self._client
        set_ctx = getattr(client, "set_context", None)
        if callable(set_ctx):
            set_ctx(tick=tick, task_id=task_id)
