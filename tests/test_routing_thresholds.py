"""Routing 置信度路由阈值单测（tests/test_routing_thresholds.py）。

覆盖三类阈值边界：
- high_confidence（≥0.85）→ ActionCommand 执行
- low_confidence（<0.5）→ Escalation(reason="low_confidence")
- no_clear_winner_gap（<0.15）→ Escalation(reason="no_clear_winner")
- Noul boundary（prob∈[0.4,0.6]）→ Escalation(reason="boundary_state")

另测：
- map_actions 只做 action 映射不产 escalation
- Choice chosen 关键词映射（hold/attack/evade/abort/return/escalate）
- Score answer 不映射为 action
- 优先级：no_clear_winner > boundary_state > low_confidence
"""

from __future__ import annotations

import pytest

from munition_system.common.config import RoutingConfig
from munition_system.common.types import (
    ActionCommand,
    ChoiceAnswer,
    ChoiceQuestion,
    Escalation,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from munition_system.jev.routing import Routing


# === Fixtures ===

def _make_routing(
    *,
    high: float = 0.85,
    low: float = 0.5,
    gap: float = 0.15,
) -> Routing:
    return Routing(RoutingConfig(
        high_confidence=high,
        low_confidence=low,
        no_clear_winner_gap=gap,
    ))


def _choice_q(qid: str = "q_c", opts: list[str] | None = None) -> ChoiceQuestion:
    if opts is None:
        opts = ["hold", "attack", "evade", "abort", "return", "escalate", "other"]
    return ChoiceQuestion(id=qid, prompt="test", options=opts, bound_to="M1")


def _noul_q(qid: str = "q_n") -> NoulQuestion:
    return NoulQuestion(id=qid, prompt="test", bound_to="global")


def _score_q(qid: str = "q_s", levels: int = 5) -> ScoreQuestion:
    return ScoreQuestion(
        id=qid, prompt="test", levels=levels,
        desc_low="low", desc_high="high", bound_to="M1",
    )


def _choice_ans(
    qid: str = "q_c",
    chosen: str = "attack",
    conf: float = 0.9,
    probs: dict[str, float] | None = None,
) -> ChoiceAnswer:
    if probs is None:
        # 默认：chosen 高，其他均匀低
        probs = {opt: 0.02 for opt in _choice_q().options}
        probs[chosen] = conf
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
    return ChoiceAnswer(
        question_id=qid,
        probabilities=probs,
        chosen=chosen,
        confidence=conf,
    )


def _noul_ans(qid: str = "q_n", prob: float = 0.9) -> NoulAnswer:
    return NoulAnswer(question_id=qid, probability=prob, confidence=prob)


def _score_ans(qid: str = "q_s", value: float = 4.0, levels: int = 5) -> ScoreAnswer:
    denom = max(levels - 1, 1)
    return ScoreAnswer(
        question_id=qid,
        value=value,
        confidence=max(0.0, min(1.0, value / denom)),
    )


# === 高置信 → ActionCommand ===

class TestHighConfidence:
    """confidence ≥ high_confidence → 产 ActionCommand。"""

    def test_high_conf_choice_yields_action(self):
        routing = _make_routing()
        ans = _choice_ans(chosen="attack", conf=0.9)
        qs = [_choice_q()]
        actions, esc = routing.route([ans], qs, subject="M1")
        assert len(actions) == 1
        assert actions[0].action == "EngageTarget"
        assert actions[0].munition_id == "M1"
        assert esc is None

    def test_exact_threshold_yields_action(self):
        """confidence 恰好等于 high_confidence → 产 action（≥ 含等号）。"""
        routing = _make_routing(high=0.85)
        ans = _choice_ans(chosen="hold", conf=0.85)
        qs = [_choice_q()]
        actions, _ = routing.route([ans], qs, subject="M1")
        assert len(actions) == 1
        assert actions[0].action == "Hold"

    def test_just_below_high_no_action(self):
        """confidence 略低于 high_confidence → 不产 action（中间区间静默）。"""
        routing = _make_routing(high=0.85)
        ans = _choice_ans(chosen="attack", conf=0.84)
        qs = [_choice_q()]
        actions, esc = routing.route([ans], qs, subject="M1")
        assert len(actions) == 0
        assert esc is None  # 中间区间：不 action 也不 escalation


# === 低置信 → Escalation ===

class TestLowConfidence:
    """confidence < low_confidence → Escalation(reason="low_confidence")。"""

    def test_low_conf_choice_yields_escalation(self):
        routing = _make_routing(low=0.5)
        ans = _choice_ans(chosen="attack", conf=0.3)
        qs = [_choice_q()]
        actions, esc = routing.route([ans], qs, subject="M1")
        assert len(actions) == 0
        assert esc is not None
        assert esc.reason == "low_confidence"
        assert len(esc.low_confidence_answers) == 1

    def test_just_above_low_no_escalation(self):
        """confidence 略高于 low_confidence → 无 escalation（中间区间）。"""
        routing = _make_routing(low=0.5)
        ans = _choice_ans(chosen="attack", conf=0.51)
        qs = [_choice_q()]
        _, esc = routing.route([ans], qs, subject="M1")
        assert esc is None

    def test_low_conf_score_yields_escalation(self):
        routing = _make_routing(low=0.5)
        ans = _score_ans(value=0.5, levels=5)  # conf=0.5/4=0.125 < 0.5
        qs = [_score_q()]
        _, esc = routing.route([ans], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "low_confidence"

    def test_low_conf_noul_yields_escalation(self):
        routing = _make_routing(low=0.5)
        ans = _noul_ans(prob=0.1)  # conf=0.1 < 0.5
        qs = [_noul_q()]
        _, esc = routing.route([ans], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "low_confidence"


# === no_clear_winner → Escalation ===

class TestNoClearWinner:
    """Choice top-2 概率差 < no_clear_winner_gap → Escalation(reason="no_clear_winner")。"""

    def test_close_top2_yields_no_clear_winner(self):
        routing = _make_routing(gap=0.15)
        # top-2 差 0.10 < 0.15（other=0.45, attack=0.35）
        probs = {"attack": 0.35, "hold": 0.2, "other": 0.45}
        ans = ChoiceAnswer(
            question_id="q_c",
            probabilities=probs,
            chosen="other",
            confidence=0.45,
        )
        qs = [_choice_q()]
        actions, esc = routing.route([ans], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "no_clear_winner"
        assert len(actions) == 0

    def test_clear_winner_no_escalation(self):
        routing = _make_routing(gap=0.15)
        # top-2 差 0.8 >> 0.15，且高置信
        probs = {"attack": 0.9, "hold": 0.05, "other": 0.05}
        ans = ChoiceAnswer(
            question_id="q_c",
            probabilities=probs,
            chosen="attack",
            confidence=0.9,
        )
        qs = [_choice_q()]
        actions, esc = routing.route([ans], qs, subject="M1")
        assert esc is None
        assert len(actions) == 1

    def test_no_clear_winner_priority_over_low_confidence(self):
        """no_clear_winner 优先于 low_confidence。"""
        routing = _make_routing(gap=0.15, low=0.5)
        # top-2 差 0.0 < 0.15（other=0.35, attack=0.35）+ 低置信 0.47
        probs = {"attack": 0.35, "hold": 0.30, "other": 0.35}
        ans = ChoiceAnswer(
            question_id="q_c",
            probabilities=probs,
            chosen="other",
            confidence=0.47,
        )
        qs = [_choice_q()]
        _, esc = routing.route([ans], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "no_clear_winner"


# === Noul boundary → Escalation ===

class TestNoulBoundary:
    """Noul prob∈[0.4, 0.6] → Escalation(reason="boundary_state")。"""

    def test_boundary_prob_yields_boundary_state(self):
        routing = _make_routing()
        ans = _noul_ans(prob=0.5)
        qs = [_noul_q()]
        _, esc = routing.route([ans], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "boundary_state"

    def test_high_prob_noul_no_escalation(self):
        routing = _make_routing()
        ans = _noul_ans(prob=0.95)
        qs = [_noul_q()]
        actions, esc = routing.route([ans], qs, subject="M1")
        assert esc is None
        assert len(actions) == 0  # Noul 不映射为 action

    def test_boundary_inclusive_0_4(self):
        """prob=0.4 属于边界态（含等号）。"""
        routing = _make_routing()
        ans = _noul_ans(prob=0.4)
        qs = [_noul_q()]
        _, esc = routing.route([ans], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "boundary_state"

    def test_boundary_inclusive_0_6(self):
        """prob=0.6 属于边界态（含等号）。"""
        routing = _make_routing()
        ans = _noul_ans(prob=0.6)
        qs = [_noul_q()]
        _, esc = routing.route([ans], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "boundary_state"

    def test_just_outside_boundary_no_escalation(self):
        """prob=0.39 或 0.61 不属于边界态。"""
        routing = _make_routing()
        qs = [_noul_q()]

        ans_low = _noul_ans(prob=0.39)
        _, esc1 = routing.route([ans_low], qs, subject="M1")
        # prob=0.39 < low_confidence=0.5 → low_confidence
        assert esc1 is not None
        assert esc1.reason == "low_confidence"

        ans_high = _noul_ans(prob=0.61)
        _, esc2 = routing.route([ans_high], qs, subject="M1")
        assert esc2 is None  # 0.61 > 0.6 且 > 0.5，无 escalation


# === Choice 关键词映射 ===

class TestChoiceActionMapping:
    """Choice chosen 关键词映射到 ActionType。"""

    @pytest.mark.parametrize("chosen,expected_action", [
        ("hold", "Hold"),
        ("wait", "Hold"),
        ("continue_wait", "Hold"),
        ("standby", "Hold"),
        ("attack", "EngageTarget"),
        ("start_attack", "EngageTarget"),
        ("engage", "EngageTarget"),
        ("strike", "EngageTarget"),
        ("evade", "EvadeLowLevel"),
        ("dodge", "EvadeLowLevel"),
        ("abort", "AbortMission"),
        ("return", "ReturnToBase"),
        ("rtb", "ReturnToBase"),
        ("reallocate", "ReallocateToZone"),
        ("reassign", "ReassignRole"),
        ("flank_route", "MoveTo"),
        ("move", "MoveTo"),
        ("hold_fire", "HoldFire"),
        ("ceasefire", "HoldFire"),
        ("adjust_altitude", "AdjustAltitude"),
        ("climb", "AdjustAltitude"),
    ])
    def test_keyword_mapping(self, chosen, expected_action):
        routing = _make_routing()
        ans = _choice_ans(chosen=chosen, conf=0.9)
        qs = [_choice_q()]
        actions, _ = routing.route([ans], qs, subject="M1")
        assert len(actions) == 1
        assert actions[0].action == expected_action

    def test_escalate_chosen_no_action(self):
        """chosen 含 'escalate' → 不产 action（触发 escalation 由调用方处理）。"""
        routing = _make_routing()
        # escalate 是低置信，不产 action
        probs = {"escalate_to_outer": 0.9, "other": 0.1}
        ans = ChoiceAnswer(
            question_id="q_c",
            probabilities=probs,
            chosen="escalate_to_outer",
            confidence=0.9,
        )
        qs = [_choice_q()]
        actions, esc = routing.route([ans], qs, subject="M1")
        # 高置信但不映射为 action
        assert len(actions) == 0
        assert esc is None

    def test_unknown_chosen_no_action(self):
        """chosen 不匹配任何关键词 → 不产 action。"""
        routing = _make_routing()
        probs = {"unknown_action": 0.9, "other": 0.1}
        ans = ChoiceAnswer(
            question_id="q_c",
            probabilities=probs,
            chosen="unknown_action",
            confidence=0.9,
        )
        qs = [_choice_q(opts=["unknown_action", "other"])]
        actions, _ = routing.route([ans], qs, subject="M1")
        assert len(actions) == 0


# === map_actions（监控任务专用）===

class TestMapActions:
    """map_actions 只做 action 映射，不产 escalation。"""

    def test_map_actions_high_conf_yields_action(self):
        routing = _make_routing()
        ans = _choice_ans(chosen="attack", conf=0.9)
        qs = [_choice_q()]
        actions = routing.map_actions([ans], qs, subject="M1")
        assert len(actions) == 1
        assert actions[0].action == "EngageTarget"

    def test_map_actions_low_conf_no_action(self):
        routing = _make_routing()
        ans = _choice_ans(chosen="attack", conf=0.3)
        qs = [_choice_q()]
        actions = routing.map_actions([ans], qs, subject="M1")
        assert len(actions) == 0

    def test_map_actions_noul_no_action(self):
        routing = _make_routing()
        ans = _noul_ans(prob=0.95)
        qs = [_noul_q()]
        actions = routing.map_actions([ans], qs, subject="M1")
        assert len(actions) == 0

    def test_map_actions_score_no_action(self):
        routing = _make_routing()
        ans = _score_ans(value=4.0, levels=5)
        qs = [_score_q()]
        actions = routing.map_actions([ans], qs, subject="M1")
        assert len(actions) == 0


# === 优先级与合并 ===

class TestPriorityAndMerge:
    """优先级：no_clear_winner > boundary_state > low_confidence。"""

    def test_multiple_answers_merge_escalation(self):
        """多个低置信 answer 合并到一个 Escalation。"""
        routing = _make_routing(low=0.5)
        a1 = _choice_ans(chosen="attack", conf=0.3)
        a2 = _noul_ans(prob=0.1)
        qs = [_choice_q(), _noul_q()]
        _, esc = routing.route([a1, a2], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "low_confidence"
        assert len(esc.low_confidence_answers) == 2

    def test_no_clear_winner_priority(self):
        """同时有 no_clear_winner 和 low_confidence → no_clear_winner 优先。"""
        routing = _make_routing(gap=0.15, low=0.5)
        # answer1: no_clear_winner（top-2 差 0.0 < 0.15：other=0.35, attack=0.35）
        probs1 = {"attack": 0.35, "hold": 0.30, "other": 0.35}
        a1 = ChoiceAnswer("q_c", probs1, "other", 0.47)
        # answer2: low_confidence
        a2 = _noul_ans(prob=0.1)
        qs = [_choice_q(), _noul_q()]
        _, esc = routing.route([a1, a2], qs, subject="M1")
        assert esc is not None
        assert esc.reason == "no_clear_winner"

    def test_schema_version_and_task_id_propagated(self):
        routing = _make_routing(low=0.5)
        ans = _choice_ans(chosen="attack", conf=0.3)
        qs = [_choice_q()]
        _, esc = routing.route(
            [ans], qs, subject="M1",
            schema_version=7,
            source_monitoring_task_id="MT1",
        )
        assert esc is not None
        assert esc.schema_version == 7
        assert esc.source_monitoring_task_id == "MT1"

    def test_empty_answers_no_escalation(self):
        routing = _make_routing()
        actions, esc = routing.route([], [], subject="M1")
        assert actions == []
        assert esc is None
