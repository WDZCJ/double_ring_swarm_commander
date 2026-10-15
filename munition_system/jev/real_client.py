"""Laya 真实客户端（jev/real_client.py）。

LayaRealClient 实现 JevClient Protocol，通过 httpx POST 调用本地部署的 Laya
决策模型服务（替代 Jev，HTTP API 调用而非本地推理）。

本地 Laya 服务接口（实测，http://localhost:8000，OpenAPI 0.1.0）：
- Endpoint: POST {base_url}/api/v1/predict
- Auth: 无（本地部署默认不鉴权；如需可经反代加 Bearer）
- Body: {state: str, questions: {q_id: {type, instructions, criteria?}}, use_cache?: bool}
- Response:
    {
      "model": "laya-rl-agent",
      "answers": {
        "q_id": {
          "type": "choice"|"score"|"noul",
          # choice:
          "choice": "...", "probabilities": {...},
          # score:
          "score": 1.6, "probabilities": {"0":...,"1":...}, "legend": {...},
          # noul:
          "noul": 0.9,
          # 公共（关键）:
          "confidence": <原始 confidence 字段，偏小>,
          "answer_confidence": <校准后的置信度，优先用这个>,
          "action": {"act_probability": 1.0}
        }
      },
      "usage": {...}, "meta": {"elapsed_ms": ...}
    }

默认配置：
- base_url: http://localhost:8000（DEFAULT_LAYA_BASE_URL，可通过 JEV_BASE_URL 覆盖）
- 不需要 api_key（如配置则忽略，兼容反代加 Bearer 的场景）

错误处理：超时/HTTP 错误→抛 JevUnavailable，编排器转 mock 或全量上报。
"""

from __future__ import annotations

from typing import Any

from munition_system.common.config import JevConfig
from munition_system.common.types import Answer, Question
from munition_system.jev.client import (
    DEFAULT_LAYA_BASE_URL,
    laya_answer_to_answer,
    question_to_laya,
    serialize_state,
)


class JevUnavailable(RuntimeError):
    """Laya 本地服务不可用（超时/HTTP 错误/响应格式错误）。

    编排器捕获后可降级为 mock client 或全量上报外环。
    """


class LayaRealClient:
    """Laya 本地真实客户端：httpx POST 到 /api/v1/predict。

    满足 JevClient Protocol（结构子类型，不需显式继承）。
    """

    def __init__(
        self,
        config: JevConfig,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = (config.base_url or DEFAULT_LAYA_BASE_URL).rstrip("/")
        # 本地 Laya 默认无鉴权；api_key 仅当配置时作为 Bearer 发送（兼容反代场景）
        self._api_key = config.api_key or ""
        self._timeout = timeout

    def decide(
        self,
        state: str | dict | list,
        questions: list[Question],
    ) -> list[Answer]:
        """调本地 Laya API 决策。

        1. 序列化 state + questions 为 Laya 请求体
        2. POST 到 {base_url}/api/v1/predict
        3. 解析响应 → Answer[]

        错误→抛 JevUnavailable。
        """
        # 序列化 questions（保留 q_id 与原 question 的映射）
        laya_questions: dict[str, dict] = {}
        q_map: dict[str, Question] = {}
        for q in questions:
            q_id, lq = question_to_laya(q)
            laya_questions[q_id] = lq
            q_map[q_id] = q

        body: dict[str, Any] = {
            "state": serialize_state(state),
            "questions": laya_questions,
            "use_cache": True,
        }

        # lazy import httpx，避免无网络环境安装失败
        try:
            import httpx
        except ImportError as e:
            raise JevUnavailable("httpx 未安装（pip install httpx）") from e

        url = f"{self._base_url}/api/v1/predict"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=body, headers=headers)
                resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise JevUnavailable(f"Laya 本地服务超时: {e}") from e
        except httpx.HTTPStatusError as e:
            raise JevUnavailable(
                f"Laya HTTP 错误 {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.HTTPError as e:
            raise JevUnavailable(f"Laya 网络错误: {e}") from e

        # 解析响应
        data = resp.json()
        laya_answers: dict[str, dict] = data.get("answers", {})

        answers: list[Answer] = []
        for q_id, q in q_map.items():
            laya_ans = laya_answers.get(q_id)
            if laya_ans is None:
                # 该问题无响应→用默认低置信 answer（noul=0.5）
                laya_ans = {"type": "noul", "noul": 0.5, "answer_confidence": 0.5}
            answers.append(laya_answer_to_answer(q, laya_ans))

        return answers
