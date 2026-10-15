"""JevClient 工厂 + Laya 序列化辅助（jev/client.py）。

工厂 make_client(config) 据 config.jev.mode 返回 LayaRealClient 或 LayaMockClient。
序列化辅助：
- serialize_state：state -> str（Laya 接受字符串）
- question_to_laya：Question -> (q_id, laya_question_dict)
- laya_answer_to_answer：Laya answer dict -> Answer

Laya API（替代 Jev，API 调用而非本地部署）：
- POST {base_url}/v1/systemone
- model: convaiinnovations/laya
- 详见 real_client.py
"""

from __future__ import annotations

import json
from typing import Any

from munition_system.common.config import JevConfig
from munition_system.common.contracts import JevClient
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

# Laya 默认配置（本地部署：默认指向 http://localhost:8000）
# 通过 JEV_BASE_URL 覆盖；本地 Laya 不需要 model 字段（响应自带 laya-rl-agent）。
DEFAULT_LAYA_MODEL = "convaiinnovations/laya"
DEFAULT_LAYA_BASE_URL = "http://localhost:8000"


def serialize_state(state: str | dict | list) -> str:
    """序列化 state 为 Laya 接受的字符串。

    Laya 的 state 字段接受任意文本/JSON 字符串。
    - str 直接返回
    - dict/list 用 json.dumps（紧凑，ensure_ascii=False）
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"))


def question_to_laya(q: Question) -> tuple[str, dict]:
    """把 Question 序列化为 Laya question dict。

    返回 (q_id, {type, instructions, criteria})。

    Laya question 格式：
    - choice: {type:"choice", instructions:prompt, criteria:{opt: desc, ...}}
      （criteria 必须含 "other"，正好 ChoiceQuestion.options 必须含 "other"）
    - score: {type:"score", instructions:prompt, criteria:[desc_low, ..., desc_high]}
      （criteria 为有序 list，长度=levels）
    - noul: {type:"noul", instructions:prompt}（无 criteria）
    """
    if isinstance(q, ChoiceQuestion):
        # criteria: {opt: opt for opt in options}（用 opt 本身作为描述）
        criteria: dict[str, str] = {opt: opt for opt in q.options}
        return q.id, {
            "type": "choice",
            "instructions": q.prompt,
            "criteria": criteria,
        }
    if isinstance(q, ScoreQuestion):
        # criteria: 有序 list，长度=levels。首尾用 desc_low/desc_high，中间合成。
        levels_list: list[str] = []
        for i in range(q.levels):
            if i == 0:
                levels_list.append(q.desc_low)
            elif i == q.levels - 1:
                levels_list.append(q.desc_high)
            else:
                levels_list.append(f"level {i} (between low and high)")
        return q.id, {
            "type": "score",
            "instructions": q.prompt,
            "criteria": levels_list,
        }
    if isinstance(q, NoulQuestion):
        return q.id, {
            "type": "noul",
            "instructions": q.prompt,
        }
    raise TypeError(f"未知 Question 类型: {type(q).__name__}")


def laya_answer_to_answer(q: Question, laya_ans: dict) -> Answer:
    """把 Laya answer dict 反序列化为 Answer。

    本地 Laya（/api/v1/predict）answer 格式：
    - choice: {type:"choice", choice:"opt", probabilities:{...},
               confidence:<原始小值>, answer_confidence:<校准值>}
    - score: {type:"score", score:1.84, probabilities:{...}, legend:{...},
              confidence:<原始小值>, answer_confidence:<top prob>}
    - noul:  {type:"noul", noul:0.98,
               confidence:<≈noul>, answer_confidence:<≈noul>}

    confidence 取值优先级：
    1) answer_confidence（本地 Laya 直接给出，最可靠）
    2) fallback 旧方式（兼容 Vercel TypeSafe 风格）：
       - choice: probabilities[chosen]
       - score:  score/(levels-1)
       - noul:   noul
    """
    ans_type = laya_ans.get("type")
    # 优先用 answer_confidence（本地 Laya 直接给出校准置信度）
    raw_answer_conf = laya_ans.get("answer_confidence")

    if isinstance(q, ChoiceQuestion):
        chosen = laya_ans.get("choice", "other")
        probabilities = dict(laya_ans.get("probabilities", {}))
        # 确保 probabilities 包含所有 options（补默认 0.0）
        for opt in q.options:
            if opt not in probabilities:
                probabilities[opt] = 0.0
        if raw_answer_conf is not None:
            confidence = max(0.0, min(1.0, float(raw_answer_conf)))
        else:
            confidence = float(probabilities.get(chosen, 0.0))
        return ChoiceAnswer(
            question_id=q.id,
            probabilities=probabilities,
            chosen=chosen,
            confidence=confidence,
        )

    if isinstance(q, ScoreQuestion):
        value = float(laya_ans.get("score", 0.0))
        if raw_answer_conf is not None:
            confidence = max(0.0, min(1.0, float(raw_answer_conf)))
        else:
            # 归一化 confidence：score 范围 [0, levels-1]
            denom = max(q.levels - 1, 1)
            confidence = max(0.0, min(1.0, value / denom))
        return ScoreAnswer(
            question_id=q.id,
            value=value,
            confidence=confidence,
        )

    if isinstance(q, NoulQuestion):
        probability = float(laya_ans.get("noul", 0.5))
        if raw_answer_conf is not None:
            confidence = max(0.0, min(1.0, float(raw_answer_conf)))
        else:
            confidence = probability  # noul 的 confidence 就是概率本身
        return NoulAnswer(
            question_id=q.id,
            probability=probability,
            confidence=confidence,
        )

    raise TypeError(f"未知 Question 类型: {type(q).__name__}")


def make_client(config: JevConfig) -> JevClient:
    """工厂：据 config.jev.mode 返回 JevClient 实现。

    - mode="mock"（默认）：LayaMockClient（确定性，三层策略）
    - mode="real"：LayaRealClient（httpx POST 到 Laya API）
    """
    mode = config.mode.lower()
    if mode == "real":
        from munition_system.jev.real_client import LayaRealClient
        return LayaRealClient(config)
    if mode == "mock":
        from munition_system.jev.mock_client import LayaMockClient
        return LayaMockClient(config)
    raise ValueError(f"未知 jev.mode: {config.mode}（应为 mock 或 real）")
