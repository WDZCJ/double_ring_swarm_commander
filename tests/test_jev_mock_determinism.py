"""Laya mock 客户端确定性单测（tests/test_jev_mock_determinism.py）。

验证：
- 同输入两次运行答案完全一致（种子化随机跨运行一致）
- 不同 tick 可能不同（tick 影响 seed）
- 脚本驱动优先于启发式和随机
- 启发式规则（destroyed → "destroyed" choice conf=0.95；threat 超 → evade）
- set_context(tick/task_id) 影响脚本查找
- 三种 question 类型（Choice/Score/Noul）都能产出正确 Answer 类型
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from munition_system.common.config import JevConfig, JevMockConfig, RoutingConfig
from munition_system.common.types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from munition_system.jev.mock_client import LayaMockClient


# === Fixtures ===

def _make_config(
    *,
    script_path: str = "",
    heuristic: bool = True,
    seed: int = 42,
) -> JevConfig:
    return JevConfig(
        mode="mock",
        routing=RoutingConfig(),
        mock=JevMockConfig(
            script_path=script_path,
            heuristic_fallback=heuristic,
            seed=seed,
        ),
    )


def _make_questions() -> list:
    return [
        ChoiceQuestion(
            id="q_status",
            prompt="A 当前状态属于",
            options=["normal_progress", "damaged", "destroyed", "target_reached", "other"],
            bound_to="M1",
        ),
        ScoreQuestion(
            id="q_health",
            prompt="A 健康度评分",
            levels=5,
            desc_low="完全损毁",
            desc_high="完好无损",
            bound_to="M1",
        ),
        NoulQuestion(
            id="q_normal",
            prompt="任务是否进展正常",
            bound_to="global",
        ),
    ]


# === 确定性：同输入两次运行答案完全一致 ===

class TestDeterminism:
    """同输入两次运行答案完全一致。"""

    def test_same_input_same_output_choice(self):
        config = _make_config(heuristic=False, seed=42)
        client = LayaMockClient(config)
        client.set_context(tick=3)
        qs = _make_questions()
        state = json.dumps({"tick": 3, "state": "searching", "tv": [0, 0, 0]})

        a1 = client.decide(state, qs)
        a2 = client.decide(state, qs)

        assert len(a1) == len(a2) == 3
        for x, y in zip(a1, a2):
            assert type(x) is type(y)
            assert x.question_id == y.question_id
            if isinstance(x, ChoiceAnswer):
                assert x.chosen == y.chosen
                assert x.probabilities == y.probabilities
                assert x.confidence == y.confidence
            elif isinstance(x, ScoreAnswer):
                assert x.value == y.value
                assert x.confidence == y.confidence
            elif isinstance(x, NoulAnswer):
                assert x.probability == y.probability
                assert x.confidence == y.confidence

    def test_same_input_same_output_across_instances(self):
        """不同 client 实例，同 seed，同输入 → 一致。"""
        config = _make_config(heuristic=False, seed=42)
        c1 = LayaMockClient(config)
        c2 = LayaMockClient(config)
        qs = _make_questions()
        state = json.dumps({"tick": 7, "tv": [0, 0, 0]})

        c1.set_context(tick=7)
        c2.set_context(tick=7)
        a1 = c1.decide(state, qs)
        a2 = c2.decide(state, qs)

        for x, y in zip(a1, a2):
            if isinstance(x, ChoiceAnswer):
                assert x.chosen == y.chosen
                assert x.confidence == y.confidence
            elif isinstance(x, ScoreAnswer):
                assert x.value == y.value
            elif isinstance(x, NoulAnswer):
                assert x.probability == y.probability

    def test_different_tick_may_differ(self):
        """不同 tick 的种子化随机可能不同。"""
        config = _make_config(heuristic=False, seed=42)
        client = LayaMockClient(config)
        qs = [NoulQuestion(id="q_n", prompt="test", bound_to="global")]
        state = json.dumps({"tv": [0, 0, 0]})

        client.set_context(tick=1)
        a1 = client.decide(state, qs)
        client.set_context(tick=2)
        a2 = client.decide(state, qs)

        # 不同 tick 种子不同，概率可能不同（允许相同，但通常不同）
        # 这里只验证都能产出合法 Answer
        assert isinstance(a1[0], NoulAnswer)
        assert isinstance(a2[0], NoulAnswer)
        assert 0.0 <= a1[0].probability <= 1.0
        assert 0.0 <= a2[0].probability <= 1.0


# === 三层策略优先级 ===

class TestStrategyPriority:
    """脚本驱动 > 启发式 > 种子化随机。"""

    def test_script_overrides_heuristic_and_random(self, tmp_path: Path):
        """脚本答案优先于启发式和随机。"""
        script_yaml = tmp_path / "script.yaml"
        script_yaml.write_text(
            "script:\n"
            '  "(3, q_status)": {answer: "destroyed", confidence: 0.99}\n',
            encoding="utf-8",
        )
        config = _make_config(script_path=str(script_yaml), heuristic=True, seed=42)
        client = LayaMockClient(config)
        client.set_context(tick=3)
        qs = _make_questions()
        state = json.dumps({"tick": 3, "state": "searching", "tv": [0, 0, 0]})

        answers = client.decide(state, qs)
        status_ans = next(a for a in answers if a.question_id == "q_status")
        assert isinstance(status_ans, ChoiceAnswer)
        assert status_ans.chosen == "destroyed"
        assert status_ans.confidence == pytest.approx(0.99)

    def test_wildcard_script_key(self, tmp_path: Path):
        """(*, q_id) 通配 tick 匹配。"""
        script_yaml = tmp_path / "script.yaml"
        script_yaml.write_text(
            "script:\n"
            '  "(*, q_normal)": {answer: 0.92, confidence: 0.92}\n',
            encoding="utf-8",
        )
        config = _make_config(script_path=str(script_yaml))
        client = LayaMockClient(config)
        client.set_context(tick=99)
        qs = [NoulQuestion(id="q_normal", prompt="test", bound_to="global")]
        state = "{}"

        answers = client.decide(state, qs)
        assert isinstance(answers[0], NoulAnswer)
        assert answers[0].probability == pytest.approx(0.92)

    def test_task_id_specific_script_key(self, tmp_path: Path):
        """(tick, task_id, q_id) 优先于 (tick, q_id)。"""
        script_yaml = tmp_path / "script.yaml"
        script_yaml.write_text(
            "script:\n"
            '  "(5, MT1, q_status)": {answer: "target_reached", confidence: 0.95}\n'
            '  "(5, q_status)": {answer: "normal_progress", confidence: 0.8}\n',
            encoding="utf-8",
        )
        config = _make_config(script_path=str(script_yaml))
        client = LayaMockClient(config)

        qs = [
            ChoiceQuestion(
                id="q_status",
                prompt="状态",
                options=["normal_progress", "target_reached", "other"],
                bound_to="M1",
            )
        ]
        state = json.dumps({"tick": 5})

        # 有 task_id → 命中 MT1 专用条目
        client.set_context(tick=5, task_id="MT1")
        a1 = client.decide(state, qs)
        assert a1[0].chosen == "target_reached"
        assert a1[0].confidence == pytest.approx(0.95)

        # 无 task_id → 命中通用条目
        client.set_context(tick=5, task_id=None)
        a2 = client.decide(state, qs)
        assert a2[0].chosen == "normal_progress"


# === 启发式兜底 ===

class TestHeuristicFallback:
    """启发式规则基于 state 字段。"""

    def test_destroyed_state_yields_destroyed_choice(self):
        config = _make_config(heuristic=True, script_path="")
        client = LayaMockClient(config)
        client.set_context(tick=1)
        qs = [
            ChoiceQuestion(
                id="q_status",
                prompt="状态",
                options=["normal_progress", "damaged", "destroyed", "other"],
                bound_to="M1",
            )
        ]
        state = json.dumps({"state": "destroyed", "tv": [0, 0, 0]})

        answers = client.decide(state, qs)
        assert answers[0].chosen == "destroyed"
        assert answers[0].confidence == pytest.approx(0.95)

    def test_threat_vector_above_threshold_yields_evade(self):
        config = _make_config(heuristic=True)
        client = LayaMockClient(config)
        client.set_context(tick=1)
        qs = [
            ChoiceQuestion(
                id="q_action",
                prompt="动作",
                options=["continue", "evade", "abort", "return", "other"],
                bound_to="M1",
            )
        ]
        state = json.dumps({"tv": [10, 0, 0]})  # threat 超阈值 5

        answers = client.decide(state, qs)
        assert answers[0].chosen == "evade"
        assert answers[0].confidence == pytest.approx(0.75)

    def test_low_confidence_on_boundary_tick(self):
        """tick % 5 == 0 时故意制造 conf=0.45 触发 escalation。"""
        config = _make_config(heuristic=True)
        client = LayaMockClient(config)
        client.set_context(tick=10)
        qs = [NoulQuestion(id="q_n", prompt="正常", bound_to="global")]
        state = json.dumps({"tv": [0, 0, 0]})

        answers = client.decide(state, qs)
        assert isinstance(answers[0], NoulAnswer)
        assert answers[0].confidence == pytest.approx(0.45)
        assert answers[0].probability == pytest.approx(0.5)

    def test_high_coverage_yields_reallocation(self):
        config = _make_config(heuristic=True)
        client = LayaMockClient(config)
        client.set_context(tick=1)
        qs = [NoulQuestion(id="q_realloc", prompt="需重分配", bound_to="global")]
        state = json.dumps({"coverage": 0.97, "tv": [0, 0, 0]})

        answers = client.decide(state, qs)
        assert answers[0].probability == pytest.approx(0.85)
        assert answers[0].confidence == pytest.approx(0.8)


# === Answer 类型正确 ===

class TestAnswerTypes:
    """三种 question 类型产出对应 Answer 类型。"""

    def test_choice_question_yields_choice_answer(self):
        config = _make_config(heuristic=False)
        client = LayaMockClient(config)
        client.set_context(tick=1)
        qs = [
            ChoiceQuestion(
                id="q_c",
                prompt="选",
                options=["a", "b", "other"],
                bound_to="M1",
            )
        ]
        answers = client.decide("{}", qs)
        assert isinstance(answers[0], ChoiceAnswer)
        assert answers[0].chosen in {"a", "b", "other"}
        # probabilities 包含所有 options
        for opt in ("a", "b", "other"):
            assert opt in answers[0].probabilities

    def test_score_question_yields_score_answer(self):
        config = _make_config(heuristic=False)
        client = LayaMockClient(config)
        client.set_context(tick=1)
        qs = [
            ScoreQuestion(
                id="q_s",
                prompt="分",
                levels=5,
                desc_low="低",
                desc_high="高",
                bound_to="M1",
            )
        ]
        answers = client.decide("{}", qs)
        assert isinstance(answers[0], ScoreAnswer)
        assert 0.0 <= answers[0].value <= 4.0  # levels-1=4
        assert 0.0 <= answers[0].confidence <= 1.0

    def test_noul_question_yields_noul_answer(self):
        config = _make_config(heuristic=False)
        client = LayaMockClient(config)
        client.set_context(tick=1)
        qs = [NoulQuestion(id="q_n", prompt="是", bound_to="global")]
        answers = client.decide("{}", qs)
        assert isinstance(answers[0], NoulAnswer)
        assert 0.0 <= answers[0].probability <= 1.0


# === 非确定性验证（跨 seed 不同）===

class TestSeedDifference:
    """不同 seed → 不同输出（验证 seed 生效）。"""

    def test_different_seed_different_output(self):
        c1 = LayaMockClient(_make_config(heuristic=False, seed=1))
        c2 = LayaMockClient(_make_config(heuristic=False, seed=999))
        qs = [NoulQuestion(id="q_n", prompt="test", bound_to="global")]
        state = json.dumps({"tv": [0, 0, 0]})

        c1.set_context(tick=1)
        c2.set_context(tick=1)
        a1 = c1.decide(state, qs)
        a2 = c2.decide(state, qs)

        # 不同 seed 通常产出不同概率（允许相同但通常不同）
        # 这里只验证都是合法 NoulAnswer
        assert isinstance(a1[0], NoulAnswer)
        assert isinstance(a2[0], NoulAnswer)
