"""配置 dataclass + 加载器（config.py）。

按方案第十二节 config.yaml 结构定义，从环境变量与 YAML 两处取值：
- 环境变量（.env）：敏感信息与运行时切换开关（API key、JEV_MODE 等）
- YAML：结构与阈值（路由阈值、ReAct 循环上限、监控任务上限等）
- YAML 中 ${VAR} 占位会被环境变量替换

加载入口：load_config(path="config/config.yaml") -> Config
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# === 子配置 dataclass ===

@dataclass
class ReactLoopConfig:
    """ReAct 循环参数。"""
    max_iterations: int = 20  # 单次规划最多 20 步 ReAct
    fallback_to_safe_intent: bool = True  # 卡死时降级为安全意图（全员 RTB）


@dataclass
class LLMConfig:
    """外环 Agent LLM 配置。

    base_url/api_key/model/seed 从环境变量取；temperature 从 YAML 取（略高给 Agent 创造性）。
    """
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    seed: int = 42
    temperature: float = 0.3
    react_loop: ReactLoopConfig = field(default_factory=ReactLoopConfig)


@dataclass
class RoutingConfig:
    """Jev 答案路由阈值。

    - high_confidence：≥ 此值直接执行
    - low_confidence：< 此值上报外环
    - no_clear_winner_gap：Choice top-2 概率差 < 此值视为无明确胜出，上报
    """
    high_confidence: float = 0.85
    low_confidence: float = 0.5
    no_clear_winner_gap: float = 0.15


@dataclass
class JevMockConfig:
    """Jev mock 客户端配置。

    三层策略（确定性优先）：
    1) script_path：脚本驱动（按 (tick, question_id) 或 (tick, task_id, question_id) 取预制答案）
    2) heuristic_fallback：启发式兜底（基于 state 规则）
    3) seed：种子化随机（跨运行一致）
    """
    script_path: str = ""
    heuristic_fallback: bool = True
    seed: int = 42


@dataclass
class JevConfig:
    """内环 Jev 配置。mode=mock 用确定性 mock，mode=real 调真实 HTTP。"""
    mode: str = "mock"  # mock | real
    base_url: str = ""
    api_key: str = ""
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    mock: JevMockConfig = field(default_factory=JevMockConfig)


@dataclass
class SimConfig:
    """仿真内核配置。grid_dim 体素网格规模（3D 整数坐标）。"""
    grid_dim: tuple[int, int, int] = (40, 40, 8)
    max_munitions: int = 18
    tick_ms: int = 200  # 单 tick 真实毫秒数（仅用于显示/调速，不参与逻辑）
    seed: int = 42


@dataclass
class PlanTrackerConfig:
    """任务完成度闭环跟踪。"""
    check_interval_ticks: int = 5  # 每 N tick 调一次 check_task_progress


@dataclass
class ReplannerConfig:
    """重规划触发与安全模式。"""
    safe_mode_after_consecutive_failures: int = 5  # 连续 N 次重规划失败→安全模式（全员 RTB）


@dataclass
class MonitoringConfig:
    """定时监控任务默认配置。"""
    default_schedule_every_ticks: int = 1
    max_concurrent_tasks: int = 20
    default_escalation_confidence: float = 0.5


@dataclass
class AgentConfig:
    """外环 Agent 子配置。"""
    plan_tracker: PlanTrackerConfig = field(default_factory=PlanTrackerConfig)
    replanner: ReplannerConfig = field(default_factory=ReplannerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


@dataclass
class Config:
    """顶层配置聚合。

    由 load_config() 构造。后续各层通过此对象读取参数。
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    jev: JevConfig = field(default_factory=JevConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


# === 加载器 ===

def _expand_env_placeholders(value: Any) -> Any:
    """递归把字符串中 ${VAR} 替换为 os.environ[VAR]（未设置则空串）。"""
    if isinstance(value, str):
        # 简单 ${VAR} 替换
        out = value
        # 用占位符扫描
        i = 0
        while i < len(out):
            if out[i:i + 2] == "${":
                end = out.find("}", i + 2)
                if end == -1:
                    break
                var = out[i + 2:end]
                out = out[:i] + os.environ.get(var, "") + out[end + 1:]
                # 不前进 i，因为替换后可能有新的占位符（一般不会，但保守起见）
            else:
                i += 1
        return out
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(v) for v in value]
    return value


def _coerce_tuple_int(value: Any, length: int) -> tuple[int, ...]:
    """把 list[3] 强制为 tuple[int,int,int]。"""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"期望长度 {length} 的 list/tuple，得到 {value!r}")
    return tuple(int(x) for x in value)


def _build_llm_config(raw: dict[str, Any]) -> LLMConfig:
    react_raw = raw.get("react_loop", {}) or {}
    react = ReactLoopConfig(
        max_iterations=int(react_raw.get("max_iterations", 20)),
        fallback_to_safe_intent=bool(react_raw.get("fallback_to_safe_intent", True)),
    )
    return LLMConfig(
        base_url=os.environ.get("LLM_BASE_URL", ""),
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ.get("LLM_MODEL", ""),
        seed=int(os.environ.get("LLM_SEED", 42)),
        temperature=float(raw.get("temperature", 0.3)),
        react_loop=react,
    )


def _build_jev_config(raw: dict[str, Any]) -> JevConfig:
    routing_raw = raw.get("routing", {}) or {}
    routing = RoutingConfig(
        high_confidence=float(routing_raw.get("high_confidence", 0.85)),
        low_confidence=float(routing_raw.get("low_confidence", 0.5)),
        no_clear_winner_gap=float(routing_raw.get("no_clear_winner_gap", 0.15)),
    )
    mock_raw = raw.get("mock", {}) or {}
    mock = JevMockConfig(
        script_path=str(mock_raw.get("script_path", "")),
        heuristic_fallback=bool(mock_raw.get("heuristic_fallback", True)),
        seed=int(mock_raw.get("seed", 42)),
    )
    return JevConfig(
        mode=str(raw.get("mode", os.environ.get("JEV_MODE", "mock"))),
        base_url=os.environ.get("JEV_BASE_URL", ""),
        api_key=os.environ.get("JEV_API_KEY", ""),
        routing=routing,
        mock=mock,
    )


def _build_sim_config(raw: dict[str, Any]) -> SimConfig:
    return SimConfig(
        grid_dim=_coerce_tuple_int(raw.get("grid_dim", [40, 40, 8]), 3),
        max_munitions=int(raw.get("max_munitions", 18)),
        tick_ms=int(os.environ.get("SIM_TICK_MS", raw.get("tick_ms", 200))),
        seed=int(os.environ.get("SIM_SEED", raw.get("seed", 42))),
    )


def _build_agent_config(raw: dict[str, Any]) -> AgentConfig:
    pt_raw = raw.get("plan_tracker", {}) or {}
    rp_raw = raw.get("replanner", {}) or {}
    mo_raw = raw.get("monitoring", {}) or {}
    return AgentConfig(
        plan_tracker=PlanTrackerConfig(
            check_interval_ticks=int(pt_raw.get("check_interval_ticks", 5)),
        ),
        replanner=ReplannerConfig(
            safe_mode_after_consecutive_failures=int(
                rp_raw.get("safe_mode_after_consecutive_failures", 5)
            ),
        ),
        monitoring=MonitoringConfig(
            default_schedule_every_ticks=int(mo_raw.get("default_schedule_every_ticks", 1)),
            max_concurrent_tasks=int(mo_raw.get("max_concurrent_tasks", 20)),
            default_escalation_confidence=float(mo_raw.get("default_escalation_confidence", 0.5)),
        ),
    )


def load_config(
    path: str | Path = "config/config.yaml",
    *,
    env_file: str | Path | None = ".env",
) -> Config:
    """加载配置：先读 .env，再读 YAML（${VAR} 占位被环境变量替换）。

    - YAML 不存在时返回全默认 Config（便于单元测试）
    - 环境变量优先级 > YAML 字面值（敏感信息只走环境变量）
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)

    p = Path(path)
    if not p.exists():
        # YAML 不存在时仍走环境变量分支（便于单元测试与无 YAML 部署）
        raw: dict[str, Any] = {}
    else:
        with p.open("r", encoding="utf-8") as f:
            raw_yaml = yaml.safe_load(f) or {}
        raw = _expand_env_placeholders(raw_yaml)

    return Config(
        llm=_build_llm_config(raw.get("llm", {}) or {}),
        jev=_build_jev_config(raw.get("jev", {}) or {}),
        sim=_build_sim_config(raw.get("sim", {}) or {}),
        agent=_build_agent_config(raw.get("agent", {}) or {}),
    )
