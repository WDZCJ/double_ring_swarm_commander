"""运行时交互 API 单测（tests/test_interaction_api.py）。

覆盖：
1. CommandBus 单元行为（提交/drain 重命令节流/完成/队列上限/查询）
2. interaction REST 路由（场景加载、运行启动、命令查询、参数校验、运行时开关）
3. runner 的命令 handler（新任务 replace、注入事件、监控任务管理、单弹指令、安全模式）
4. DemoSession + TestClient 端到端（离线 mock LLM，真实主循环线程）
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from munition_system.agent.llm_client import MockLLMClient
from munition_system.common.config import Config
from munition_system.common.contracts import LLMResponse
from munition_system.common.types import (
    CommanderIntent,
    DamageEvent,
    Munition,
    MunitionState,
    Zone,
)
from munition_system.runner.loop import DualLoopRunner
from munition_system.runner.scenario import Scenario
from munition_system.web import interaction
from munition_system.web import state as web_state
from munition_system.web.control import CommandBus, CommandType, QueueFullError
from munition_system.web.server import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# === 公共构造 ===

def _make_scenario() -> Scenario:
    return Scenario(
        name="interaction_test",
        description="交互测试场景",
        grid_dim=(20, 20, 4),
        munitions=[
            Munition(id="M1", role="strike", position=(0, 0, 0),
                     state=MunitionState.IDLE, fuel=100.0, target_zone=None),
            Munition(id="M2", role="search", position=(1, 1, 0),
                     state=MunitionState.IDLE, fuel=100.0, target_zone=None),
        ],
        threats=[],
        zones=[Zone(id="Z1", bounds=(0, 10, 0, 10, 0, 4))],
        events=[],
    )


def _plan_json() -> str:
    return json.dumps({
        "assignments": [
            {"munition_id": "M1", "role": "strike", "zone_id": "Z1",
             "capability": "search", "route": []},
            {"munition_id": "M2", "role": "search", "zone_id": "Z1",
             "capability": "search", "route": []},
        ],
        "groups": {"G1": ["M1", "M2"]},
        "execution_steps": ["搜索 Z1"],
        "success_conditions": ["Z1 覆盖率 ≥ 0.95"],
        "monitoring_points": ["Z1 覆盖率"],
    })


def _make_runner(*, responses: list[LLMResponse] | None = None):
    bus = CommandBus()
    runner = DualLoopRunner(
        Config(), _make_scenario(),
        llm_responses=responses if responses is not None else [LLMResponse(content=_plan_json())],
        control=bus,
    )
    return runner, bus


def _simple_schema() -> dict:
    return {
        "version": 1,
        "plan_version": 1,
        "questions": [
            {"type": "choice", "id": "q1", "prompt": "任务是否正常?",
             "options": ["normal", "abnormal"], "bound_to": "global"},
        ],
    }


# === 1. CommandBus 单元测试 ===

class TestCommandBus:
    def test_submit_and_get(self):
        bus = CommandBus()
        cmd = bus.submit(CommandType.REPLAN, {"reason": "x"})
        assert cmd.status == "pending"
        assert bus.get(cmd.command_id) is cmd
        assert bus.pending_count() == 1

    def test_drain_throttles_heavy_commands(self):
        """同一 tick 内只放行 1 条重命令，其余留在队列（保持顺序）。"""
        bus = CommandBus()
        h1 = bus.submit(CommandType.NEW_INTENT, {"raw_text": "a"})
        h2 = bus.submit(CommandType.NEW_INTENT, {"raw_text": "b"})
        light = bus.submit(CommandType.MONITORING_CANCEL, {"task_id": "MT1"})

        taken = bus.drain(max_heavy=1)
        taken_ids = {c.command_id for c in taken}
        assert h1.command_id in taken_ids       # 第一条重命令放行
        assert h2.command_id not in taken_ids   # 第二条重命令留在队列
        assert light.command_id in taken_ids    # 轻命令全部放行
        assert [c.command_id for c in bus._pending] == [h2.command_id]

        taken2 = bus.drain(max_heavy=1)
        assert [c.command_id for c in taken2] == [h2.command_id]

    def test_complete_and_recent(self):
        bus = CommandBus()
        cmd = bus.submit(CommandType.SAFE_MODE, {})
        bus.complete(cmd, status="applied", result={"safe_mode": True}, applied_tick=3)
        assert cmd.status == "applied"
        assert cmd.applied_tick == 3
        recent = bus.recent()
        assert recent[0]["command_id"] == cmd.command_id
        assert recent[0]["status"] == "applied"

    def test_queue_full(self):
        bus = CommandBus(max_queue=3)
        for _ in range(3):
            bus.submit(CommandType.SAFE_MODE, {})
        with pytest.raises(QueueFullError):
            bus.submit(CommandType.SAFE_MODE, {})

    def test_wait_returns_when_completed(self):
        bus = CommandBus()
        cmd = bus.submit(CommandType.SAFE_MODE, {})
        bus.complete(cmd, status="applied", result={})
        done = bus.wait(cmd.command_id, timeout=0.5)
        assert done is not None and done.status == "applied"

    def test_unknown_type_rejected(self):
        bus = CommandBus()
        with pytest.raises(ValueError):
            bus.submit("not_a_type", {})


# === 2. REST 路由测试（假会话，验证路由与校验）===

class _FakeRunner:
    def __init__(self) -> None:
        self.paused = False
        self.stop_requested = False
        self.speed = 1.0

    def request_pause(self):
        self.paused = True

    def request_resume(self):
        self.paused = False

    def request_stop(self):
        self.stop_requested = True

    def set_speed(self, v):
        self.speed = float(v)
        return self.speed

    def publish(self):
        pass


class _FakeSession:
    def __init__(self) -> None:
        self.bus = CommandBus()
        self.runner = None

    def scenarios(self):
        return [{"key": "s1", "description": "假场景", "max_ticks": 5, "intent_text": "搜索"}]

    def load_scenario(self, key, max_ticks=None):
        if key != "s1":
            raise KeyError(key)
        self.runner = _FakeRunner()
        return {"scenario": key, "grid_dim": [20, 20, 4], "munitions": 2,
                "max_ticks": max_ticks or 5}

    def start(self, intent_text=None, max_ticks=None):
        if self.runner is None:
            raise RuntimeError("请先加载场景")
        return {"scenario": "s1", "max_ticks": max_ticks or 5, "intent": intent_text or "搜索"}


@pytest.fixture
def api_client():
    web_state.reset_state()
    session = _FakeSession()
    interaction.bind(session)
    with TestClient(app) as client:
        yield client, session
    interaction.bind(None)


class TestInteractionRoutes:
    def test_list_scenarios(self, api_client):
        client, _ = api_client
        r = client.get("/api/scenarios")
        assert r.status_code == 200
        assert r.json()["scenarios"][0]["key"] == "s1"

    def test_load_scenario_and_unknown(self, api_client):
        client, _ = api_client
        r = client.post("/api/scenarios/load", json={"scenario": "s1"})
        assert r.status_code == 200
        assert r.json()["status"] == "loaded"

        r = client.post("/api/scenarios/load", json={"scenario": "nope"})
        assert r.status_code == 404

        r = client.post("/api/scenarios/load", json={})
        assert r.status_code == 422

    def test_start_run_requires_scenario(self, api_client):
        client, session = api_client
        r = client.post("/api/run/start", json={})
        assert r.status_code == 409

        client.post("/api/scenarios/load", json={"scenario": "s1"})
        r = client.post("/api/run/start", json={})
        assert r.status_code == 200
        assert r.json()["status"] == "started"

    def test_intent_mode_must_be_replace(self, api_client):
        client, _ = api_client
        r = client.post("/api/intents", json={"raw_text": "新任务", "mode": "append"})
        assert r.status_code == 422

        r = client.post("/api/intents", json={"raw_text": "新任务"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "pending"
        assert body["type"] == "new_intent"

    def test_intent_requires_text(self, api_client):
        client, _ = api_client
        r = client.post("/api/intents", json={})
        assert r.status_code == 422

    def test_command_query(self, api_client):
        client, session = api_client
        r = client.post("/api/monitoring/tasks/MT1/pause", json={})
        assert r.status_code == 200
        cmd_id = r.json()["command_id"]

        r = client.get(f"/api/commands/{cmd_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "pending"

        r = client.get("/api/commands")
        assert r.status_code == 200
        assert len(r.json()["commands"]) == 1

        r = client.get("/api/commands/cmd-does-not-exist")
        assert r.status_code == 404

    def test_generic_command_unknown_type(self, api_client):
        client, _ = api_client
        r = client.post("/api/commands", json={"type": "bogus", "payload": {}})
        assert r.status_code == 422

    def test_inject_event_requires_type(self, api_client):
        client, _ = api_client
        r = client.post("/api/events/inject", json={"munition_id": "M1"})
        assert r.status_code == 422

        r = client.post("/api/events/inject", json={"event_type": "damage", "munition_id": "M1"})
        assert r.status_code == 200
        assert r.json()["status"] == "pending"

    def test_runtime_switches_call_runner(self, api_client):
        client, session = api_client
        client.post("/api/scenarios/load", json={"scenario": "s1"})
        assert client.post("/api/runtime/pause").status_code == 200
        assert session.runner.paused is True
        assert client.post("/api/runtime/resume").status_code == 200
        assert session.runner.paused is False
        assert client.post("/api/runtime/speed", json={"speed": 3}).json()["speed"] == 3.0
        assert client.post("/api/runtime/stop").status_code == 200
        assert session.runner.stop_requested is True

    def test_runtime_requires_runner(self, api_client):
        client, _ = api_client
        assert client.post("/api/runtime/pause").status_code == 409

    def test_monitoring_task_validation(self, api_client):
        client, _ = api_client
        r = client.post("/api/monitoring/tasks", json={"subject": "M1"})
        assert r.status_code == 422

    def test_readonly_snapshot_endpoints(self, api_client):
        client, _ = api_client
        web_state.update_state(tick=7, plan_version=2, intents=[{"raw_text": "a"}],
                              telemetry={"total_ticks": 7})
        assert client.get("/api/plan").json()["plan_version"] == 2
        assert client.get("/api/intents").json()["intents"][0]["raw_text"] == "a"
        assert client.get("/api/telemetry").json()["total_ticks"] == 7
        assert client.get("/api/events").json()["events"] == []
        assert client.get("/api/monitoring/tasks").json()["tasks"] == []
        assert client.get("/api/monitoring/tasks/MTX").status_code == 404

    def test_not_bound_returns_503(self):
        interaction.bind(None)
        with TestClient(app) as client:
            assert client.get("/api/scenarios").status_code == 503


# === 3. runner 命令 handler 测试（真实 runner，离线 mock LLM）===

class TestRunnerCommands:
    def _ctx(self, runner, tick=2):
        return runner._make_ctx(tick=tick)

    def test_new_intent_replaces_plan(self):
        runner, bus = _make_runner()
        intent = CommanderIntent(raw_text="搜索 Z1", goal="搜索", constraints=[], success_criteria="x")
        runner.run(intent, max_ticks=3)
        assert runner.plan is not None
        old_version = runner.plan.plan_version

        cmd = bus.submit(CommandType.NEW_INTENT, {"raw_text": "改为全员返航", "success_criteria": "返航"})
        runner._drain_commands(self._ctx(runner))

        assert cmd.status == "applied", cmd.error
        assert cmd.result["plan_version"] == old_version + 1
        assert len(runner._intents) == 2           # 旧意图 + 新意图
        assert runner._intents[-1]["raw_text"] == "改为全员返航"

    def test_new_intent_missing_text_rejected(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        cmd = bus.submit(CommandType.NEW_INTENT, {})
        runner._drain_commands(self._ctx(runner))
        assert cmd.status == "rejected"
        assert "raw_text" in (cmd.error or "")

    def test_inject_event(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        cmd = bus.submit(CommandType.INJECT_EVENT, {"event_type": "damage", "munition_id": "M1"})
        runner._drain_commands(self._ctx(runner, tick=3))
        assert cmd.status == "applied", cmd.error
        assert cmd.result["injected"]["event_type"] == "DamageEvent"
        # 事件已进入注入队列，且默认 tick = ctx.tick + 1
        assert len(runner._world.event_injector) == 1

    def test_inject_event_bad_type_rejected(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        cmd = bus.submit(CommandType.INJECT_EVENT, {"event_type": "nope"})
        runner._drain_commands(self._ctx(runner))
        assert cmd.status == "rejected"

    def test_inject_damage_event_arms_event_driven_replan(self):
        """注入弹损毁事件 → 事件应用到世界后自动触发重规划（绕过 demo 抑制）。"""
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        old_version = runner.plan.plan_version

        cmd = bus.submit(CommandType.INJECT_EVENT, {"event_type": "damage", "munition_id": "M1"})
        runner._drain_commands(self._ctx(runner, tick=3))
        assert cmd.status == "applied", cmd.error
        assert cmd.result["replan_armed"] is True
        assert runner._pending_event_replan is not None

        # 事件被世界应用后的回调 → 标记可触发事件驱动重规划
        runner._on_event_fired(DamageEvent(munition_id="M1", tick=4))
        assert runner._event_replan_ready is True

        # 事件驱动重规划：plan_version+1 且 replan_count+1
        runner._try_event_replan(runner._pending_event_replan, runner._make_ctx(tick=4))
        assert runner.plan.plan_version == old_version + 1
        assert runner.replan_count == 1

    def test_inject_comm_loss_arms_replan_intel_does_not(self):
        """通信失联同样触发事件驱动重规划；情报更新不触发。"""
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)

        loss = bus.submit(CommandType.INJECT_EVENT, {"event_type": "comm_loss", "munition_ids": ["M1"]})
        runner._drain_commands(self._ctx(runner, tick=2))
        assert loss.status == "applied", loss.error
        assert loss.result["replan_armed"] is True

        runner._pending_event_replan = None
        intel = bus.submit(CommandType.INJECT_EVENT,
                           {"event_type": "intel_update", "target_id": "T1", "status": "disappeared"})
        runner._drain_commands(self._ctx(runner, tick=3))
        assert intel.status == "applied", intel.error
        assert intel.result["replan_armed"] is False
        assert runner._pending_event_replan is None

    def test_monitoring_create_pause_cancel(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)

        create = bus.submit(CommandType.MONITORING_CREATE,
                            {"subject": "M1", "schema": _simple_schema(), "schedule_every_ticks": 1})
        runner._drain_commands(self._ctx(runner))
        assert create.status == "applied", create.error
        task_id = create.result["task_id"]

        # 任务存在于管理器
        assert runner._monitoring_manager.inspect_monitoring_task(task_id) is not None

        cancel = bus.submit(CommandType.MONITORING_CANCEL, {"task_id": task_id})
        runner._drain_commands(self._ctx(runner))
        assert cancel.status == "applied", cancel.error
        assert runner._monitoring_manager.inspect_monitoring_task(task_id).status.value == "cancelled"

    def test_munition_command(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        cmd = bus.submit(CommandType.MUNITION_COMMAND,
                         {"munition_id": "M1", "action": "ReturnToBase"})
        runner._drain_commands(self._ctx(runner))
        assert cmd.status == "applied", cmd.error
        assert runner._world.munitions["M1"].state == MunitionState.RTB

    def test_munition_command_unknown_munition_rejected(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        cmd = bus.submit(CommandType.MUNITION_COMMAND, {"munition_id": "NOPE", "action": "Hold"})
        runner._drain_commands(self._ctx(runner))
        assert cmd.status == "rejected"

    def test_safe_mode_command(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        cmd = bus.submit(CommandType.SAFE_MODE, {})
        runner._drain_commands(self._ctx(runner))
        assert cmd.status == "applied", cmd.error
        assert runner.safe_mode is True

    def test_heavy_command_throttled_across_ticks(self):
        """一次 drain 只执行 1 条重命令，下一条留待下一 tick。"""
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="a", goal="a", constraints=[], success_criteria=""), max_ticks=1)
        c1 = bus.submit(CommandType.REPLAN, {"reason": "r1"})
        c2 = bus.submit(CommandType.REPLAN, {"reason": "r2"})
        runner._drain_commands(self._ctx(runner, tick=5))
        assert c1.status == "applied"
        assert c2.status == "pending"
        runner._drain_commands(self._ctx(runner, tick=6))
        assert c2.status == "applied"

    def test_snapshot_published_with_control(self):
        runner, bus = _make_runner()
        runner.run(CommanderIntent(raw_text="搜索", goal="搜索", constraints=[], success_criteria=""), max_ticks=2)
        snap = web_state.get_current_state()
        assert snap["plan_version"] >= 1
        assert snap["scenario"] == "interaction_test"
        assert snap["runtime"]["running"] is False
        assert len(snap["intents"]) == 1

    def test_no_control_still_runs(self):
        """不注入 control 时行为与原先一致（向后兼容）。"""
        runner = DualLoopRunner(Config(), _make_scenario(),
                                llm_responses=[LLMResponse(content=_plan_json())])
        result = runner.run(CommanderIntent(raw_text="搜索", goal="搜索", constraints=[],
                                            success_criteria="x"), max_ticks=3)
        assert result.plan is not None
        assert len(result.telemetry.tick_records) > 0


# === 4. DemoSession 端到端（离线真实主循环）===

class TestDemoSessionEndToEnd:
    @pytest.fixture(autouse=True)
    def _slow_ticks(self, monkeypatch):
        # 保持覆盖率不饱和（cov_delta=0），使主循环跑满 max_ticks，
        # 从而留出稳定的"运行中交互"时间窗；tick 间隔设小以缩短测试时长。
        monkeypatch.setenv("DEMO_TICK_SLEEP", "0.05")
        monkeypatch.setenv("DEMO_COV_DELTA", "0.0")

    def test_load_start_pause_stop(self):
        from munition_system.demos.demo_web import DemoSession

        web_state.reset_state()
        session = DemoSession(project_root=PROJECT_ROOT, config=Config(), quiet=True)
        interaction.bind(session)

        try:
            with TestClient(app) as client:
                assert client.post("/api/scenarios/load",
                                   json={"scenario": "cooperative_search"}).status_code == 200
                assert client.post("/api/run/start", json={}).status_code == 200

                # 等待主循环推进至少一个 tick
                tick = 0
                for _ in range(100):
                    tick = client.get("/api/state").json().get("tick", 0)
                    if tick >= 1:
                        break
                    time.sleep(0.05)
                assert tick >= 1, "主循环未推进"

                # 暂停应即时生效并可在暂停态下发新任务
                assert client.post("/api/runtime/pause").status_code == 200
                r = client.post("/api/intents", json={"raw_text": "暂停态下发的新任务", "wait_ms": 2000})
                assert r.status_code == 200
                assert r.json()["type"] == "new_intent"

                # 恢复后通过 API 查询意图历史
                assert client.post("/api/runtime/resume").status_code == 200
                intents = client.get("/api/intents").json()["intents"]
                assert any(i["raw_text"] == "暂停态下发的新任务" for i in intents)

                # 停止
                assert client.post("/api/runtime/stop").status_code == 200
                for _ in range(60):
                    if not client.get("/api/state").json()["runtime"]["running"]:
                        break
                    time.sleep(0.05)
                assert client.get("/api/state").json()["runtime"]["running"] is False
        finally:
            session._stop_current()
            interaction.bind(None)
