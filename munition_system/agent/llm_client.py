"""外环 LLM 客户端（agent/llm_client.py）。

实现 LLMClient Protocol（OpenAI 兼容 chat completions）。
- RealLLMClient：httpx POST 到 OpenAI 兼容端点（base_url/api_key/model 从环境变量取）
- MockLLMClient：离线预制响应（无网络/无 key 时用，P5 demo 降级）

严格以配置契约为中心：base_url/api_key/model/seed/temperature 全部来自 LLMConfig，
不凭空假设、不随意替换模型名。
"""

from __future__ import annotations

import json
from typing import Any

from munition_system.common.config import LLMConfig
from munition_system.common.contracts import LLMClient, LLMResponse


class LLMUnavailable(RuntimeError):
    """LLM 不可用（超时/HTTP 错误/响应格式错误）。"""


class RealLLMClient:
    """OpenAI 兼容 LLM 客户端（httpx POST 到 /v1/chat/completions）。

    满足 LLMClient Protocol（结构子类型）。
    base_url/api_key/model/seed 从 LLMConfig（环境变量）读取。
    """

    def __init__(self, config: LLMConfig, *, timeout: float = 60.0) -> None:
        self._base_url = (config.base_url or "").rstrip("/")
        self._api_key = config.api_key
        self._model = config.model
        self._seed = config.seed
        self._temperature = config.temperature
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        """调 OpenAI 兼容 chat completions。

        支持 tool_calling：响应中若有 tool_calls 则解析为 list[dict]。
        """
        if not self._base_url:
            raise LLMUnavailable("未配置 LLM base_url（LLM_BASE_URL）")
        if not self._api_key:
            raise LLMUnavailable("未配置 LLM api_key（LLM_API_KEY）")
        if not self._model:
            raise LLMUnavailable("未配置 LLM model（LLM_MODEL）")

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "seed": seed if seed is not None else self._seed,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        try:
            import httpx
        except ImportError as e:
            raise LLMUnavailable("httpx 未安装（pip install httpx）") from e

        # 拼接 chat completions endpoint：
        # DeepSeek/OpenAI 兼容协议约定 base_url 应含 /v1（如 https://api.deepseek.com/v1），
        # endpoint = base_url + "/chat/completions"。
        # 若用户给的 base_url 未含 /v1（如 https://api.deepseek.com），自动补一次。
        if self._base_url.endswith("/v1"):
            url = f"{self._base_url}/chat/completions"
        else:
            url = f"{self._base_url}/v1/chat/completions"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise LLMUnavailable(f"LLM 超时: {e}") from e
        except httpx.HTTPStatusError as e:
            raise LLMUnavailable(
                f"LLM HTTP 错误 {e.response.status_code}: {e.response.text[:200]}"
            ) from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"LLM 网络错误: {e}") from e

        data = resp.json()
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> LLMResponse:
        """解析 OpenAI 响应 → LLMResponse。"""
        try:
            choice = data["choices"][0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            raw_tool_calls = msg.get("tool_calls") or []
            finish_reason = choice.get("finish_reason", "stop")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMUnavailable(f"LLM 响应格式错误: {e}") from e

        tool_calls: list[dict] = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"name": name, "args": args})

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )


class MockLLMClient:
    """离线 Mock LLM 客户端（无网络/无 key 时用，P5 demo 降级）。

    通过 set_next_responses() 注入预制响应序列；chat() 按序消费。
    未配置预制响应时返回空 content + finish_reason=stop（让 ReAct 循环终止）。
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._responses: list[LLMResponse] = []
        self._call_count = 0

    def set_next_responses(self, responses: list[LLMResponse]) -> None:
        """注入预制响应序列（按序消费）。"""
        self._responses = list(responses)
        self._call_count = 0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        self._call_count += 1
        if self._responses:
            return self._responses.pop(0)
        # 无预制响应 → 空 content，让 ReAct 循环正常终止
        return LLMResponse(content="", finish_reason="stop")


def make_llm_client(config: LLMConfig) -> LLMClient:
    """工厂：据配置返回 LLMClient 实现。

    - 有 base_url + api_key + model → RealLLMClient
    - 否则 → MockLLMClient（离线降级）
    """
    if config.base_url and config.api_key and config.model:
        return RealLLMClient(config)
    return MockLLMClient(config)
