"""Laya 确定性 mock 客户端（jev/mock_client.py）。

LayaMockClient 实现 JevClient Protocol，三层策略（确定性优先）：
1. 脚本驱动：从 YAML 读 script 表，键=(tick, question_id) 或 (tick, task_id, question_id)
2. 启发式兜底：基于 state 规则（munition 状态/threat_vector/coverage 等）
3. 种子化随机：random.Random(seed=tick*31+hash(qid)+base_seed) 跨运行一致

故意制造 <0.5 置信情形以验证 escalation 路径（方案第八节）。

set_context(tick, task_id)：orchestrator 在调 decide 前设置上下文（duck typing，
非 Protocol 方法；real client 无此方法，orchestrator 用 hasattr 检查）。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from munition_system.common.config import JevConfig
from munition_system.common.types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


class LayaMockClient:
    """Laya 确定性 mock 客户端。

    满足 JevClient Protocol（结构子类型）。

    三层策略（确定性优先）：
    1. 脚本驱动（script_path 指向 YAML）
    2. 启发式兜底（heuristic_fallback=True）
    3. 种子化随机（seed）
    """

    def __init__(self, config: JevConfig) -> None:
        self._script_path = config.mock.script_path
        self._heuristic_fallback = config.mock.heuristic_fallback
        self._base_seed = config.mock.seed
        self._script: dict[str, dict] | None = None
        if self._script_path:
            self._script = self._load_script(self._script_path)
        # 上下文（由 orchestrator 通过 set_context 设置）
        self._tick: int = -1
        self._task_id: str | None = None

    def set_context(self, *, tick: int = -1, task_id: str | None = None) -> None:
        """设置当前调用上下文（tick + task_id），供脚本查找用。

        orchestrator 在调 decide 前调用（duck typing）。
        """
        self._tick = tick
        self._task_id = task_id

    def decide(
        self,
        state: str | dict | list,
        questions: list[Question],
    ) -> list[Answer]:
        """三层策略决策（确定性优先）。"""
        state_dict = self._parse_state(state)
        # 如果 set_context 没设置 tick，尝试从 state 解析
        tick = self._tick
        if tick < 0 and isinstance(state_dict, dict):
            tick = int(state_dict.get("tick", 0))

        answers: list[Answer] = []
        for q in questions:
            ans = self._answer_one(q, state_dict, tick)
            answers.append(ans)
        return answers

    # === 三层策略 ===

    def _answer_one(
        self,
        q: Question,
        state_dict: dict | None,
        tick: int,
    ) -> Answer:
        """对单个 question 应用三层策略。"""
        # 1) 脚本驱动
        if self._script is not None:
            ans = self._script_lookup(q, tick)
            if ans is not None:
                return ans
        # 2) 启发式兜底
        if self._heuristic_fallback:
            ans = self._heuristic_answer(q, state_dict, tick)
            if ans is not None:
                return ans
        # 3) 种子化随机
        return self._seeded_random_answer(q, tick)

    # === 脚本驱动 ===

    def _load_script(self, path: str) -> dict[str, dict]:
        """加载 scenario YAML 的 script 段。

        YAML 结构：
        script:
          "(tick, q_id)": {answer: ..., confidence: ...}
          "(tick, task_id, q_id)": {answer: ..., confidence: ...}
          "(*, q_id)": {answer: ..., confidence: ...}  # 通配 tick
        """
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            data = yaml_safe_load(f) or {}
        script = data.get("script", {}) if isinstance(data, dict) else {}
        return script

    def _script_lookup(self, q: Question, tick: int) -> Answer | None:
        """查脚本表，键优先级：(tick, task_id, q_id) > (tick, q_id) > (*, q_id)。"""
        if self._script is None:
            return None
        candidates: list[str] = []
        if self._task_id:
            candidates.append(f"({tick}, {self._task_id}, {q.id})")
        candidates.append(f"({tick}, {q.id})")
        candidates.append(f"(*, {q.id})")

        for key in candidates:
            entry = self._script.get(key)
            if entry:
                return self._build_answer_from_entry(q, entry)
        return None

    def _build_answer_from_entry(self, q: Question, entry: dict) -> Answer:
        """从脚本条目构造 Answer。"""
        conf = float(entry.get("confidence", 0.8))
        ans_val = entry.get("answer")
        return self._build_answer(q, ans_val, conf)

    # === 启发式兜底 ===

    def _heuristic_answer(
        self,
        q: Question,
        state_dict: dict | None,
        tick: int,
    ) -> Answer | None:
        """基于 state 规则的启发式答案。

        规则（方案第八节）：
        - munition 状态=DESTROYED → 状态 choice 选 "destroyed" conf=0.95
        - threat_vector 任一分量超阈值 → evade choice conf=0.75
        - 搜索完成度>0.95 → reallocation prob=0.85/conf=0.8
        - 边界附近故意 conf=0.45 触发上报
        """
        if state_dict is None:
            return None

        # 解析 state 里的关键字段
        m_state = state_dict.get("state")
        tv = state_dict.get("tv") or state_dict.get("threat_vector")
        tv_list = [0, 0, 0]
        if isinstance(tv, (list, tuple)):
            tv_list = [int(x) for x in tv]
        elif isinstance(tv, dict):
            tv_list = [int(tv.get(k, 0)) for k in ("x", "y", "z")]

        # 规则1：munition DESTROYED → choice 选 "destroyed"
        if m_state == "destroyed":
            if isinstance(q, ChoiceQuestion) and "destroyed" in q.options:
                return self._build_choice(q, "destroyed", 0.95)
            if isinstance(q, NoulQuestion):
                return NoulAnswer(q.id, probability=0.1, confidence=0.95)

        # 规则2：threat_vector 任一分量超阈值 → evade/abort/return
        threat_threshold = 5
        if any(abs(x) > threat_threshold for x in tv_list):
            if isinstance(q, ChoiceQuestion):
                for opt in q.options:
                    opt_lower = opt.lower()
                    if "evade" in opt_lower or "abort" in opt_lower or "return" in opt_lower:
                        return self._build_choice(q, opt, 0.75)
            if isinstance(q, NoulQuestion):
                # "威胁是否严重" → True
                return NoulAnswer(q.id, probability=0.9, confidence=0.75)

        # 规则3：搜索完成度高 → reallocation
        coverage = state_dict.get("coverage") or state_dict.get("cov")
        if coverage is not None:
            try:
                cov_val = float(coverage)
                if cov_val > 0.95:
                    if isinstance(q, NoulQuestion):
                        return NoulAnswer(q.id, probability=0.85, confidence=0.8)
                    if isinstance(q, ChoiceQuestion):
                        for opt in q.options:
                            if "reallocat" in opt.lower() or "reassign" in opt.lower():
                                return self._build_choice(q, opt, 0.8)
            except (TypeError, ValueError):
                pass

        # 规则4：边界附近故意低置信（conf=0.45）触发上报
        # 用 tick % 5 == 0 模拟"边界附近"周期
        if tick > 0 and tick % 5 == 0:
            if isinstance(q, NoulQuestion):
                return NoulAnswer(q.id, probability=0.5, confidence=0.45)
            if isinstance(q, ChoiceQuestion) and "other" in q.options:
                return self._build_choice(q, "other", 0.45)

        return None

    # === 种子化随机 ===

    def _seeded_random_answer(self, q: Question, tick: int) -> Answer:
        """种子化随机答案（跨运行一致）。

        seed = tick*31 + hash(qid) + base_seed
        使用 md5 hash 保证跨运行/跨平台一致（不依赖 Python hash 随机化）。
        """
        seed = tick * 31 + self._hash_qid(q.id) + self._base_seed
        rng = random.Random(seed)

        if isinstance(q, ChoiceQuestion):
            opts = q.options
            probs = {opt: rng.random() for opt in opts}
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            chosen = max(probs, key=probs.get)  # type: ignore[arg-type]
            conf = probs[chosen]
            return ChoiceAnswer(
                question_id=q.id,
                probabilities=probs,
                chosen=chosen,
                confidence=conf,
            )

        if isinstance(q, ScoreQuestion):
            value = rng.uniform(0, q.levels - 1)
            denom = max(q.levels - 1, 1)
            conf = value / denom
            return ScoreAnswer(
                question_id=q.id,
                value=value,
                confidence=max(0.0, min(1.0, conf)),
            )

        if isinstance(q, NoulQuestion):
            prob = rng.random()
            return NoulAnswer(
                question_id=q.id,
                probability=prob,
                confidence=prob,
            )

        raise TypeError(f"未知 Question 类型: {type(q).__name__}")

    # === 辅助构造 ===

    def _build_answer(self, q: Question, ans_val: Any, conf: float) -> Answer:
        """根据 question 类型从标量值构造 Answer。"""
        if isinstance(q, ChoiceQuestion):
            chosen = str(ans_val) if ans_val is not None else "other"
            if chosen not in q.options:
                chosen = "other"
            return self._build_choice(q, chosen, conf)

        if isinstance(q, ScoreQuestion):
            value = float(ans_val) if ans_val is not None else 0.0
            denom = max(q.levels - 1, 1)
            return ScoreAnswer(
                question_id=q.id,
                value=value,
                confidence=max(0.0, min(1.0, value / denom)),
            )

        if isinstance(q, NoulQuestion):
            prob = float(ans_val) if ans_val is not None else 0.5
            return NoulAnswer(
                question_id=q.id,
                probability=prob,
                confidence=prob,
            )

        raise TypeError(f"未知 Question 类型: {type(q).__name__}")

    def _build_choice(
        self,
        q: ChoiceQuestion,
        chosen: str,
        conf: float,
    ) -> ChoiceAnswer:
        """构造 ChoiceAnswer：chosen=conf，其他均匀低权重。"""
        probs = {opt: 0.05 for opt in q.options}
        probs[chosen] = conf
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return ChoiceAnswer(
            question_id=q.id,
            probabilities=probs,
            chosen=chosen,
            confidence=conf,
        )

    # === 解析辅助 ===

    def _parse_state(self, state: str | dict | list) -> dict | None:
        """解析 state 为 dict（如果是 JSON 字符串）。"""
        if isinstance(state, dict):
            return state
        if isinstance(state, list):
            return {"items": state}
        if isinstance(state, str):
            try:
                parsed = json.loads(state)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def _hash_qid(self, qid: str) -> int:
        """稳定 hash（跨运行一致，不依赖 Python hash 随机化）。"""
        h = hashlib.md5(qid.encode("utf-8")).hexdigest()
        return int(h[:8], 16)


def yaml_safe_load(f) -> Any:
    """lazy import yaml 并 safe_load。"""
    import yaml
    return yaml.safe_load(f)
