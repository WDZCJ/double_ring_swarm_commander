"""Web 可视化 demo（demos/demo_web.py）。

与 demo_live.py 的区别：
- 启动 FastAPI Web 服务（默认 127.0.0.1:8080）
- 通过 WebSocket 实时推送运行数据到浏览器
- 支持在运行过程中通过 REST API / Web UI 与系统交互：
  选择场景加载、启动任务、暂停/恢复/调速/停止、下发新任务（replace 语义）、
  注入临机事件、管理监控任务、给单弹下指令、强制安全模式

运行：python -m munition_system.demos.demo_web
然后浏览器打开 http://127.0.0.1:8080

线程模型：
- main 线程：启动 Web 服务后驻留（等待 Web UI/API 操作）
- runner 线程：运行 DualLoopRunner 主循环（唯一状态写者）
- server 线程：uvicorn 事件循环，仅提交命令 + 读快照
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from munition_system.common.config import load_config
from munition_system.common.types import CommanderIntent
from munition_system.runner.loop import DualLoopRunner, RunnerResult
from munition_system.runner.scenario import load_scenario
from munition_system.web import interaction
from munition_system.web import state as web_state
from munition_system.web.control import CommandBus
from munition_system.web.server import queue_broadcast, start_server_in_thread

_SCENARIOS = {
    "cooperative_search": {
        "file": "config/scenarios/cooperative_search.yaml",
        "description": "18 发分 A/B 两组协同搜索东西两区",
        "intent": CommanderIntent(
            raw_text="18 发分 A/B 两组搜东西两区，B 组提前完成后支援 A 组",
            goal="协同搜索东西两区",
            constraints=["18 发规模", "A/B 各 9 发"],
            success_criteria="两区覆盖率均 ≥ 0.95",
        ),
        "max_ticks": 30,
    },
    "cooperative_attack": {
        "file": "config/scenarios/cooperative_attack.yaml",
        "description": "A 弹前出诱饵，B 弹盘旋等待，按 A 状态决策",
        "intent": CommanderIntent(
            raw_text="A 弹前出诱饵，B 弹盘旋等待，根据 A 状态决策规避/绕路/取消",
            goal="协同攻击 T1",
            constraints=["A 诱饵 B 打击", "A 损毁后 B 决策"],
            success_criteria="T1 被摧毁",
        ),
        "max_ticks": 20,
    },
    "adhoc_events": {
        "file": "config/scenarios/adhoc_events.yaml",
        "description": "编组缺员+目标消失+部分失联三连发",
        "intent": CommanderIntent(
            raw_text="编组缺员+目标消失+部分失联三连发",
            goal="临机事件处理",
            constraints=["6 发规模", "G1 打击 G2 搜索"],
            success_criteria="任务完成或安全返航",
        ),
        "max_ticks": 40,
    },
}


def _resolve_scenario_path(project_root: Path, file: str) -> Path:
    p = Path(file)
    return p if p.is_absolute() else project_root / p


def _build_intent(intent_text: str | None, default_intent: CommanderIntent) -> CommanderIntent:
    """构造本次运行的指挥员意图（未给定文本则用场景默认意图）。"""
    if not intent_text or not intent_text.strip():
        return default_intent
    text = intent_text.strip()
    return CommanderIntent(
        raw_text=text,
        goal=text,
        constraints=list(default_intent.constraints),
        success_criteria=default_intent.success_criteria,
    )


class DemoSession:
    """Web demo 会话：持有当前运行器 + 命令总线，供 REST API 交互。

    对外暴露 interaction 模块所需的最小接口：
    - bus / runner 属性
    - scenarios() / load_scenario(key, max_ticks) / start(intent_text, max_ticks)
    """

    def __init__(
        self,
        *,
        project_root: Path,
        config,
        quiet: bool = False,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.quiet = quiet
        self.bus = CommandBus()
        self.runner: DualLoopRunner | None = None
        self.scenario_key: str | None = None
        self.max_ticks: int | None = None
        self.last_result: RunnerResult | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    # --- 场景 ---

    def scenarios(self) -> list[dict]:
        return [
            {
                "key": key,
                "description": sc.get("description", ""),
                "max_ticks": sc["max_ticks"],
                "intent_text": sc["intent"].raw_text,
            }
            for key, sc in _SCENARIOS.items()
        ]

    def load_scenario(self, key: str, max_ticks: int | None = None) -> dict:
        """加载（或切换）场景，创建新的运行器。若已有任务运行则先停止。"""
        if key not in _SCENARIOS:
            raise KeyError(key)
        with self._lock:
            self._stop_current()
            sc = _SCENARIOS[key]
            scenario = load_scenario(_resolve_scenario_path(self.project_root, sc["file"]))
            runner = DualLoopRunner(
                self.config,
                scenario,
                control=self.bus,
                event_sink=queue_broadcast,
            )
            _attach_hooks(runner, quiet=self.quiet)
            self.runner = runner
            self.scenario_key = key
            self.max_ticks = max_ticks or sc["max_ticks"]

        web_state.reset_state()
        web_state.update_state(scenario=key, phase="场景已加载")
        queue_broadcast({"type": "init", "grid_dim": list(scenario.grid_dim), "scenario": key})
        queue_broadcast({"type": "phase", "phase": f"场景已加载: {key}"})
        with self._lock:
            self.runner.publish()
        return {
            "scenario": key,
            "grid_dim": list(scenario.grid_dim),
            "munitions": len(scenario.munitions),
            "max_ticks": self.max_ticks,
        }

    def _stop_current(self) -> None:
        """停止并等待当前运行器线程退出（持有 self._lock 时调用）。"""
        if self.runner is not None:
            self.runner.request_stop()
            self.runner.request_resume()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("上一个任务仍在停止中，请稍后重试")

    # --- 运行 ---

    def start(self, intent_text: str | None = None, max_ticks: int | None = None) -> dict:
        """在后台线程启动主循环。"""
        with self._lock:
            if self.runner is None or self.scenario_key is None:
                raise RuntimeError("请先加载场景")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("已有任务在运行，请先停止")
            sc = _SCENARIOS[self.scenario_key]
            intent = _build_intent(intent_text, sc["intent"])
            ticks = max_ticks or self.max_ticks or sc["max_ticks"]
            runner = self.runner
            runner.request_resume()
            t = threading.Thread(
                target=self._run_runner,
                args=(runner, intent, ticks),
                daemon=True,
                name="demo-runner",
            )
            self._thread = t
        t.start()
        return {"scenario": self.scenario_key, "max_ticks": ticks, "intent": intent.raw_text}

    def _run_runner(self, runner: DualLoopRunner, intent: CommanderIntent, max_ticks: int) -> None:
        try:
            result = runner.run(intent, max_ticks=max_ticks)
            self.last_result = result
            queue_broadcast({"type": "phase", "phase": f"运行结束: {result.final_phase}"})
            queue_broadcast({
                "type": "run_end",
                "phase": result.final_phase,
                "replan_count": result.replan_count,
                "safe_mode": result.safe_mode,
                "warnings": result.warnings[-5:],
            })
            if not self.quiet:
                print(f"\n[结束] phase={result.final_phase} replan={result.replan_count} "
                      f"escalations={result.escalation_count}")
        except Exception as e:  # noqa: BLE001 - 运行异常需上报到前端而非静默
            queue_broadcast({"type": "phase", "phase": f"运行异常: {e}"})
            queue_broadcast({"type": "run_end", "phase": "error", "error": str(e)})
            if not self.quiet:
                print(f"\n[异常] {e}")


def _attach_hooks(runner: DualLoopRunner, *, quiet: bool) -> None:
    """挂载可视化 hook：把 LLM/Jev/ReAct/tick 过程广播到 Web 前端。"""
    # === Hook: LLM chat ===
    llm_round = {"n": 0}
    original_chat = runner._llm.chat

    def chat_with_log(messages, tools=None, temperature=None, seed=None):
        resp = original_chat(messages, tools=tools, temperature=temperature, seed=seed)
        if tools is not None:
            llm_round["n"] += 1
            label = "ReAct 规划"
        else:
            label = "schema 设计"

        round_data = {
            "type": "llm_round",
            "round": llm_round["n"],
            "ctx_label": label,
            "content": (getattr(resp, "content", "") or "").strip(),
            "tool_calls": [
                {"name": tc.get("name", "?"), "args": tc.get("args", {})}
                for tc in (getattr(resp, "tool_calls", None) or [])
            ],
            "has_plan": bool(getattr(resp, "content", "") and '"assignments"' in getattr(resp, "content", "")),
        }
        queue_broadcast(round_data)

        if not quiet:
            print(f"\n[LLM Round {llm_round['n']}] {label}")
            if round_data["content"]:
                print(f"  思考: {round_data['content'][:200]}...")
            if round_data["tool_calls"]:
                print(f"  工具: {[t['name'] for t in round_data['tool_calls']]}")
        return resp

    runner._llm.chat = chat_with_log  # type: ignore[assignment]

    # === Hook: Jev decide ===
    laya_ctx = {"tick": 0, "task_id": None}
    original_set_ctx = runner._jev_orchestrator._set_client_context

    def set_ctx_with_log(tick, task_id):
        laya_ctx["tick"] = tick
        laya_ctx["task_id"] = task_id
        original_set_ctx(tick, task_id)

    runner._jev_orchestrator._set_client_context = set_ctx_with_log  # type: ignore[assignment]

    original_decide = runner._jev_client.decide

    def decide_with_log(state, questions):
        answers = original_decide(state, questions)
        subject = "global"
        if laya_ctx["task_id"]:
            for t in runner._monitoring_manager.list_monitoring_tasks():
                if t.task_id == laya_ctx["task_id"]:
                    subject = t.subject
                    break

        jev_data = {
            "type": "jev_call",
            "tick": laya_ctx["tick"],
            "task_id": laya_ctx["task_id"],
            "subject": subject,
            "questions": [
                {
                    "type": type(q).__name__,
                    "id": q.id,
                    "prompt": q.prompt,
                    "bound_to": q.bound_to,
                }
                for q in questions
            ],
            "answers": [
                {
                    "type": type(a).__name__,
                    "question_id": a.question_id,
                    "confidence": round(a.confidence, 3),
                    **({"chosen": a.chosen, "probabilities": {k: round(v, 3) for k, v in a.probabilities.items()}} if hasattr(a, "chosen") else {}),
                    **({"probability": round(a.probability, 3)} if hasattr(a, "probability") else {}),
                    **({"value": round(a.value, 3)} if hasattr(a, "value") else {}),
                }
                for a in answers
            ],
        }
        queue_broadcast(jev_data)
        return answers

    runner._jev_client.decide = decide_with_log  # type: ignore[assignment]

    # === Hook: ReAct run ===
    original_react_run = runner._react_loop.run

    def react_run_with_frame(intent_arg, ctx, *, pending_escalations=None, plan_version=1):
        label = "重规划 ReAct" if plan_version > 1 else "首次 ReAct 规划"
        llm_round["n"] = 0
        queue_broadcast({"type": "phase", "phase": label})
        if not quiet:
            print(f"\n[阶段] {label} 开始...")
        t0 = time.time()
        result = original_react_run(intent_arg, ctx, pending_escalations=pending_escalations, plan_version=plan_version)
        elapsed = time.time() - t0
        queue_broadcast({"type": "phase", "phase": f"{label} 完成 ({elapsed:.1f}s)"})
        if result.plan is not None:
            queue_broadcast({
                "type": "plan",
                "plan_version": result.plan.plan_version,
                "assignments_count": len(result.plan.assignments),
                "groups": list(result.plan.groups.keys()),
            })
        if not quiet:
            print(f"[阶段] {label} 完成，耗时 {elapsed:.1f}s，iterations={result.iterations}")
        return result

    runner._react_loop.run = react_run_with_frame  # type: ignore[assignment]

    # === Hook: deploy ===
    original_deploy = runner._deploy_plan

    def deploy_with_frame(ctx):
        original_deploy(ctx)
        schema = runner._jev_orchestrator.get_global_schema()
        plan = runner._plan
        queue_broadcast({"type": "phase", "phase": "部署阶段"})
        if schema is not None:
            queue_broadcast({
                "type": "schema",
                "version": schema.version,
                "questions_count": len(schema.questions),
            })
        if plan is not None:
            queue_broadcast({
                "type": "plan",
                "plan_version": plan.plan_version,
                "assignments_count": len(plan.assignments),
                "groups": list(plan.groups.keys()),
            })
        if not quiet:
            print(f"[部署] plan_v{plan.plan_version if plan else '?'} schema_v{schema.version if schema else '?'}")
        time.sleep(0.5)

    runner._deploy_plan = deploy_with_frame  # type: ignore[assignment]

    # === Hook: tick ===
    original_tick = runner._tick
    sleep_sec = float(os.environ.get("DEMO_TICK_SLEEP", "0.8"))

    def tick_with_frame(ctx):
        done = original_tick(ctx)

        ws = runner._situational_api.get_world_state()
        tasks = [web_state.serialize_monitoring_task(t) for t in runner._monitoring_manager.list_monitoring_tasks()]

        tick_data = {
            "type": "tick",
            "tick": ctx.tick,
            "plan_version": runner._plan.plan_version if runner._plan else 0,
            "replan_count": runner._replan_count,
            "world": web_state.serialize_world(ws),
            "monitoring_tasks": tasks,
        }
        queue_broadcast(tick_data)
        queue_broadcast({"type": "monitoring_tasks", "tasks": tasks})

        if not quiet:
            print(f"[Tick {ctx.tick:3d}] plan_v{tick_data['plan_version']} "
                  f"munitions={len(ws.munitions)} threats={len(ws.threats)} tasks={len(tasks)}")

        # 状态快照由 runner 在 tick 末尾统一发布；此处仅推送前端帧
        if not done:
            time.sleep(sleep_sec / max(0.1, runner.speed))
        return done

    runner._tick = tick_with_frame  # type: ignore[assignment]

    # === Hook: escalation ===
    original_handle_esc = runner._handle_escalations

    def handle_escalations_filtered(ctx):
        if os.environ.get("DEMO_SUPPRESS_REPLAN", "true").lower() in ("true", "1", "yes"):
            runner._pending_escalations.clear()
        else:
            original_handle_esc(ctx)

    runner._handle_escalations = handle_escalations_filtered  # type: ignore[assignment]

    # === Hook: coverage speed ===
    cov_delta = float(os.environ.get("DEMO_COV_DELTA", "0.005"))

    def advance_cov_slow():
        from munition_system.common.types import MunitionState as _MS
        for m in runner._world.munitions.values():
            if m.state != _MS.SEARCHING or not m.target_zone:
                continue
            runner._world.zones.increment_coverage(m.target_zone, delta=cov_delta)

    runner._world._advance_coverage = advance_cov_slow  # type: ignore[assignment]

    # === Hook: replan ===
    original_handle_replan = runner._handle_replan

    def handle_replan_filtered(reason: str, ctx):
        if os.environ.get("DEMO_SUPPRESS_REPLAN", "true").lower() in ("true", "1", "yes"):
            return
        if reason.startswith("progress_stuck"):
            return
        original_handle_replan(reason, ctx)

    runner._handle_replan = handle_replan_filtered  # type: ignore[assignment]


def _serve_forever() -> None:
    """main 线程驻留，保持 Web 服务在线（可用 Ctrl+C 退出）。"""
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[用户中断] 退出。")


def run_demo(
    scenario_key: str,
    max_ticks: int,
    host: str = "127.0.0.1",
    port: int = 8080,
    quiet: bool = False,
    autostart: bool = False,
) -> RunnerResult | None:
    """运行 Web 可视化 demo。

    默认不自动启动：仅启动 Web 服务，等待在 Web UI 上加载场景并点击"启动任务"
    （避免一运行就调用 LLM 开始规划）。autostart=True 时才自动加载并启动。
    函数在 Web 服务关闭（Ctrl+C）前不返回。
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    cfg = load_config(
        path=project_root / "config" / "config.yaml",
        env_file=project_root / ".env",
    )

    llm_ok = bool(cfg.llm.base_url and cfg.llm.api_key and cfg.llm.model)
    print("=" * 70)
    print(f"[Web Demo] 项目根: {project_root}")
    print(f"[Web Demo] LLM: {'real' if llm_ok else 'mock(降级!)'}  model={cfg.llm.model or '(空)'}")
    print(f"[Web Demo] Jev: {cfg.jev.mode}  base_url={cfg.jev.base_url or '(空)'}")
    print(f"[Web Demo] Web: http://{host}:{port}  （无需鉴权）")
    print("=" * 70)

    session = DemoSession(project_root=project_root, config=cfg, quiet=quiet)
    interaction.bind(session)

    start_server_in_thread(host, port)
    time.sleep(1)  # 等待服务启动

    if autostart and scenario_key in _SCENARIOS:
        session.load_scenario(scenario_key, max_ticks)
        session.start()
    else:
        print("[Web Demo] 等待通过 Web UI / REST API 加载场景并启动任务...")

    _serve_forever()
    return session.last_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Web 可视化 demo")
    parser.add_argument("--scenario", default="cooperative_search", choices=list(_SCENARIOS.keys()))
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--quiet", action="store_true", help="关闭终端打印")
    parser.add_argument("--autostart", action="store_true", help="启动服务后自动加载并运行默认场景（默认关闭）")
    args = parser.parse_args()

    scenario_key = os.environ.get("DEMO_SCENARIO", args.scenario)
    if scenario_key not in _SCENARIOS:
        print(f"未知场景 {scenario_key}")
        sys.exit(1)

    sc = _SCENARIOS[scenario_key]
    max_ticks = args.max_ticks or int(os.environ.get("DEMO_MAX_TICKS", str(sc["max_ticks"])))

    run_demo(
        scenario_key,
        max_ticks,
        args.host,
        args.port,
        args.quiet,
        autostart=args.autostart,
    )


if __name__ == "__main__":
    main()
