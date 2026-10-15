"""置信度路由 + Answer→ActionCommand 映射（jev/routing.py）。

Routing 接收 Answer[] + Question[]，按置信度阈值判断：
- 高置信（≥high_confidence）→ 转 ActionCommand 执行
- 低置信（<low_confidence）→ Escalation（reason="low_confidence"）
- Choice top-2 概率差 < no_clear_winner_gap → Escalation（reason="no_clear_winner"）
- Noul prob∈[0.4, 0.6] → Escalation（reason="boundary_state"）

两个方法：
- route()：完整路由（全局 schema 用），返回 (ActionCommand[], Escalation | None)
- map_actions()：只做 action 映射（监控任务用，escalation 由 escalation_policy 处理）

ActionCommand 映射：基于 ChoiceAnswer.chosen 关键词匹配。
Score/Noul 不直接映射为 action（用于评估和 escalation）。
"""

from __future__ import annotations

from munition_system.common.config import RoutingConfig
from munition_system.common.types import (
    ActionCommand,
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Escalation,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

# Choice chosen 值到 ActionType 的关键词映射（按优先级排序，先匹配先返回）
_CHOICE_ACTION_KEYWORDS: list[tuple[str, str]] = [
    # abort/return 优先（紧急动作）
    ("abort", "AbortMission"),
    ("rtb", "ReturnToBase"),
    ("return", "ReturnToBase"),
    # evade
    ("evade", "EvadeLowLevel"),
    ("low_level", "EvadeLowLevel"),
    ("dodge", "EvadeLowLevel"),
    # hold fire
    ("hold_fire", "HoldFire"),
    ("ceasefire", "HoldFire"),
    # hold/wait
    ("continue_wait", "Hold"),
    ("standby", "Hold"),
    ("wait", "Hold"),
    ("hold", "Hold"),
    # engage/attack
    ("start_attack", "EngageTarget"),
    ("engage", "EngageTarget"),
    ("attack", "EngageTarget"),
    ("strike", "EngageTarget"),
    ("intercept", "EngageTarget"),
    # reallocate/reassign
    ("reallocate", "ReallocateToZone"),
    ("reassign", "ReassignRole"),
    # altitude
    ("adjust_altitude", "AdjustAltitude"),
    ("climb", "AdjustAltitude"),
    ("descend", "AdjustAltitude"),
    # move/flank
    ("flank_route", "MoveTo"),
    ("flank", "MoveTo"),
    ("relocate", "MoveTo"),
    ("advance", "MoveTo"),
    ("move", "MoveTo"),
]


class Routing:
    """置信度路由 + Answer→ActionCommand 映射。

    阈值（默认，可配）：
    - high_confidence: 0.85
    - low_confidence: 0.5
    - no_clear_winner_gap: 0.15
    """

    def __init__(self, config: RoutingConfig | None = None) -> None:
        cfg = config or RoutingConfig()
        self.high_confidence = cfg.high_confidence
        self.low_confidence = cfg.low_confidence
        self.no_clear_winner_gap = cfg.no_clear_winner_gap

    def route(
        self,
        answers: list[Answer],
        questions: list[Question],
        subject: str = "global",
        *,
        schema_version: int = 0,
        source_monitoring_task_id: str | None = None,
    ) -> tuple[list[ActionCommand], Escalation | None]:
        """完整路由：全局 schema 用，返回 (ActionCommand[], Escalation | None)。

        参数：
        - answers：JevClient.decide 返回的答案
        - questions：对应的 question（用于查类型）
        - subject：默认 ActionCommand.munition_id（全局 schema 用 "global"）
        - schema_version / source_monitoring_task_id：写入 Escalation 字段

        规则：
        - 高置信 Choice → ActionCommand（关键词映射）
        - 低置信 → Escalation(reason="low_confidence")
        - Choice no_clear_winner → Escalation(reason="no_clear_winner")
        - Noul boundary (prob∈[0.4,0.6]) → Escalation(reason="boundary_state")
        - 优先级：no_clear_winner > boundary_state > low_confidence
        """
        q_map = {q.id: q for q in questions}
        actions: list[ActionCommand] = []
        no_winner_answers: list[Answer] = []
        boundary_answers: list[Answer] = []
        low_conf_answers: list[Answer] = []

        for ans in answers:
            q = q_map.get(ans.question_id)
            if q is None:
                continue

            if isinstance(ans, ChoiceAnswer) and isinstance(q, ChoiceQuestion):
                sorted_probs = sorted(ans.probabilities.values(), reverse=True)
                top1 = sorted_probs[0] if sorted_probs else 0.0
                top2 = sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
                gap = top1 - top2
                if gap < self.no_clear_winner_gap:
                    no_winner_answers.append(ans)
                    continue
                if ans.confidence >= self.high_confidence:
                    action = self._map_choice_to_action(ans.chosen, subject)
                    if action is not None:
                        actions.append(action)
                elif ans.confidence < self.low_confidence:
                    low_conf_answers.append(ans)

            elif isinstance(ans, ScoreAnswer) and isinstance(q, ScoreQuestion):
                if ans.confidence < self.low_confidence:
                    low_conf_answers.append(ans)

            elif isinstance(ans, NoulAnswer) and isinstance(q, NoulQuestion):
                if 0.4 <= ans.probability <= 0.6:
                    boundary_answers.append(ans)
                elif ans.confidence < self.low_confidence:
                    low_conf_answers.append(ans)

        # 构造 escalation（优先级：no_clear_winner > boundary_state > low_confidence）
        if no_winner_answers:
            return actions, Escalation(
                schema_version=schema_version,
                low_confidence_answers=no_winner_answers,
                reason="no_clear_winner",
                source_monitoring_task_id=source_monitoring_task_id,
            )
        if boundary_answers:
            return actions, Escalation(
                schema_version=schema_version,
                low_confidence_answers=boundary_answers,
                reason="boundary_state",
                source_monitoring_task_id=source_monitoring_task_id,
            )
        if low_conf_answers:
            return actions, Escalation(
                schema_version=schema_version,
                low_confidence_answers=low_conf_answers,
                reason="low_confidence",
                source_monitoring_task_id=source_monitoring_task_id,
            )

        return actions, None

    def map_actions(
        self,
        answers: list[Answer],
        questions: list[Question],
        subject: str = "global",
    ) -> list[ActionCommand]:
        """只做 action 映射（监控任务用），不做 escalation 判断。

        监控任务的 escalation 由 escalation_policy 独立检查（orchestrator._check_policy）。
        这里只把高置信 Choice answer 映射为 ActionCommand。
        """
        q_map = {q.id: q for q in questions}
        actions: list[ActionCommand] = []
        for ans in answers:
            q = q_map.get(ans.question_id)
            if q is None:
                continue
            if isinstance(ans, ChoiceAnswer) and isinstance(q, ChoiceQuestion):
                if ans.confidence >= self.high_confidence:
                    action = self._map_choice_to_action(ans.chosen, subject)
                    if action is not None:
                        actions.append(action)
        return actions

    def _map_choice_to_action(self, chosen: str, subject: str) -> ActionCommand | None:
        """关键词映射 Choice chosen → ActionCommand。"""
        chosen_lower = chosen.lower()
        # 含 "escalate" → 不产生 action（触发 escalation 由调用方处理）
        if "escalate" in chosen_lower:
            return None
        for keyword, action_type in _CHOICE_ACTION_KEYWORDS:
            if keyword in chosen_lower:
                return ActionCommand(munition_id=subject, action=action_type)
        return None
