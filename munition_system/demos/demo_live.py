"""实时可视化 demo（demos/demo_live.py）。

与 mock demo 的区别：
- 使用真实 DeepSeek LLM（外环 ReAct 规划与 schema 设计都由 LLM 完成）
- 使用本地部署的 Laya（内环 Jev 决策，HTTP /api/v1/predict）
- 打印 LLM 每一轮的完整输出（content 思考内容 + tool_calls 工具调用）
- 打印每次 Laya 调用的完整输入（state + questions）与输出（answers 含置信度/概率）
- 每帧打印一帧态势：弹位/状态/威胁/zone覆盖率/监控任务/最近决策
- 帧之间 sleep（DEMO_TICK_SLEEP 秒，默认 0.8s），让肉眼可见状态演化
- 逐帧追加打印（不清屏），历史保留，便于回看全过程
- 默认场景：协同搜索（cooperative_search.yaml），18 发 A/B 两组

环境变量（在 .env 中配置）：
- LLM_BASE_URL / LLM_API_KEY / LLM_MODEL（DeepSeek）
- JEV_BASE_URL / JEV_MODE=real（本地 Laya）
- DEMO_TICK_SLEEP=0.8（帧间停顿秒数；0=最快跑完，便于回归）
- DEMO_CLEAR_SCREEN=false（默认 false=逐帧追加打印，历史保留；true=每帧清屏覆盖）
- DEMO_SUPPRESS_REPLAN=true（Demo 模式抑制重规划，让 coverage 逐帧推进）
- DEMO_COV_DELTA=0.005（每搜索弹每 tick coverage 增量；越小越慢填满）
- DEMO_SCENARIO=cooperative_search|cooperative_attack|adhoc_events
- DEMO_MAX_TICKS=30

运行：python -m munition_system.demos.demo_live
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from munition_system.common.config import load_config
from munition_system.common.types import (
    ChoiceAnswer,
    ChoiceQuestion,
    CommanderIntent,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from munition_system.demos._common import print_result
from munition_system.runner.loop import DualLoopRunner
from munition_system.runner.scenario import load_scenario


# === 可视化辅助 ===

def _clear_screen() -> None:
    # 默认 false：不清屏，逐帧追加打印（历史帧保留，可一直回看）
    # 如需每帧刷新覆盖，设 DEMO_CLEAR_SCREEN=true
    if os.environ.get("DEMO_CLEAR_SCREEN", "false").lower() in ("true", "1", "yes"):
        os.system("cls" if os.name == "nt" else "clear")


def _bar(p: float, width: int = 20) -> str:
    """生成 [██████░░░░] 风格进度条，p∈[0,1]。"""
    filled = int(p * width + 0.5)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _state_short(s: str) -> str:
    """状态名缩进显示对齐。"""
    return f"{s:11s}"


def _answer_brief(ans) -> str:
    """监控任务 last_result 的一行简介。"""
    if ans is None:
        return "（无）"
    if isinstance(ans, ChoiceAnswer):
        return f"choice → {ans.chosen} (conf={ans.confidence:.2f})"
    if isinstance(ans, NoulAnswer):
        return f"noul   → p={ans.probability:.2f} (conf={ans.confidence:.2f})"
    if isinstance(ans, ScoreAnswer):
        return f"score  → v={ans.value:.2f} (conf={ans.confidence:.2f})"
    return str(ans)


def _print_header(title: str, width: int = 70) -> None:
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_llm_round(round_no: int, resp, ctx_label: str = "", width: int = 70) -> None:
    """打印 LLM 单轮输出（content 思考内容 + tool_calls 工具调用）。"""
    print("=" * width)
    label = f"  [LLM 第 {round_no} 轮输出]"
    if ctx_label:
        label += f"  ({ctx_label})"
    print(label)
    print("=" * width)

    content = (getattr(resp, "content", "") or "").strip()
    if content:
        print("  ● 思考内容:")
        for line in content.splitlines():
            print(f"    {line}")
    else:
        print("  ● 思考内容: （空）")

    tool_calls = getattr(resp, "tool_calls", None) or []
    if tool_calls:
        print(f"\n  ● 工具调用（{len(tool_calls)} 个）:")
        for i, tc in enumerate(tool_calls, 1):
            name = tc.get("name", "?")
            args = tc.get("args", {}) or {}
            try:
                args_str = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_str = str(args)
            print(f"    {i}. {name}  args={args_str}")
    else:
        print("\n  ● 工具调用: 无（本轮未调用工具）")

    if content and '"assignments"' in content:
        print("\n  ● 本轮结论: 输出 BattlePlan JSON")
    elif not tool_calls:
        print("\n  ● 本轮结论: 无工具调用，本轮结束")
    else:
        print("\n  ● 本轮结论: 继续 → 执行工具 → 下一轮")
    print("=" * width)


def _dump_state_readable(state) -> str:
    """把 Jev 输入 state 格式化为易读多行文本：顶层字段逐行，列表元素（dict）压成一行。"""
    try:
        obj = json.loads(state) if isinstance(state, str) else state
    except Exception:
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
    if not isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False, default=str, indent=2)
    lines: list[str] = []
    for k, v in obj.items():
        if isinstance(v, list) and v and all(isinstance(it, dict) for it in v):
            lines.append(f"{k}: [{len(v)} 项]")
            for it in v:
                lines.append("  " + json.dumps(it, ensure_ascii=False, default=str))
        else:
            lines.append(f"{k}: " + json.dumps(v, ensure_ascii=False, default=str))
    return "\n".join(lines)


def _print_laya_call(tick: int, task_id, subject: str, state, questions, answers, width: int = 70) -> None:
    """打印一次 Laya 调用的完整输入（state + questions）与输出（answers）。"""
    print("=" * width)
    print("  [内环 Laya 调用]")
    src = f"监控任务 {task_id}" if task_id else "全局 schema"
    print(f"  tick={tick}  subject={subject}  来源={src}")
    print("=" * width)

    # 输入 state
    print("  ● 输入 state（发给 Laya 的态势）:")
    for line in _dump_state_readable(state).splitlines():
        print(f"    {line}")

    # 问题
    print(f"\n  ● 问题（{len(questions)} 个）:")
    for i, q in enumerate(questions, 1):
        if isinstance(q, ChoiceQuestion):
            print(f"    {i}. [choice] id={q.id} bound_to={q.bound_to}")
            print(f"       prompt: {q.prompt}")
            print(f"       options: {q.options}")
        elif isinstance(q, ScoreQuestion):
            print(f"    {i}. [score]  id={q.id} bound_to={q.bound_to} levels={q.levels}")
            print(f"       prompt: {q.prompt}")
            print(f"       低/高描述: {q.desc_low} … {q.desc_high}")
        elif isinstance(q, NoulQuestion):
            print(f"    {i}. [noul]   id={q.id} bound_to={q.bound_to}")
            print(f"       prompt: {q.prompt}")

    # 输出 answers
    print(f"\n  ● Laya 输出（{len(answers)} 个 answer）:")
    for i, a in enumerate(answers, 1):
        if isinstance(a, ChoiceAnswer):
            probs = ", ".join(
                f"{k}={v:.2f}"
                for k, v in sorted(a.probabilities.items(), key=lambda kv: -kv[1])
            )
            print(f"    {i}. [choice] id={a.question_id} → chosen={a.chosen}  conf={a.confidence:.2f}")
            print(f"       probs: {{{probs}}}")
        elif isinstance(a, ScoreAnswer):
            print(f"    {i}. [score]  id={a.question_id} → value={a.value:.2f}  conf={a.confidence:.2f}")
        elif isinstance(a, NoulAnswer):
            print(f"    {i}. [noul]   id={a.question_id} → p={a.probability:.2f}  conf={a.confidence:.2f}")
    print("=" * width)


def _resolve_subject(task_id, questions, manager) -> str:
    """推导本次 Laya 调用的 subject（监控任务查 task.subject，否则用 questions.bound_to）。"""
    if task_id and manager is not None:
        for t in manager.list_monitoring_tasks():
            if t.task_id == task_id:
                return t.subject
    bounds = sorted({q.bound_to for q in questions})
    return ",".join(bounds) if bounds else "?"


def _print_planning_frame(label: str, intent_text: str, width: int = 70) -> None:
    """LLM 规划/重规划阶段的等待帧。"""
    _clear_screen()
    _print_header(f"智能巡飞弹集群决策系统 · 实时 demo", width=width)
    print()
    print(f"  阶段: {label}")
    print(f"  指挥员意图: {intent_text}")
    print()
    print("  · 调用 DeepSeek LLM (ReAct 循环) 中...")
    print("  · LLM 将: 观察态势 → 调用工具 → 输出 BattlePlan JSON")
    print()
    print("  [等待 LLM 响应，可能需数秒到十几秒]")
    print("=" * width)


def _print_tick_frame(runner: DualLoopRunner, tick: int, phase: str, width: int = 70) -> None:
    """单帧态势打印：执行完 tick 之后调用。"""
    _clear_screen()
    _print_header(f"智能巡飞弹集群决策系统 · 实时 demo", width=width)
    print(f"  TICK {tick:03d}    phase: {phase}    replan_count: {runner._replan_count}")
    print("=" * width)

    # 从 telemetry 取本 tick 的执行结果
    telemetry = runner._telemetry
    rec = telemetry.tick_records[-1] if telemetry.tick_records else None

    # 巡飞弹
    ws = runner._situational_api.get_world_state()
    print("\n  [巡飞弹]")
    print(f"  {'id':10s} {'state':12s} {'pos':14s} {'fuel':6s} {'威胁':16s}")
    for m in ws.munitions:
        x, y, z = m.position
        tv = m.threat_vector
        tv_str = f"({tv[0]},{tv[1]},{tv[2]})" if any(tv) else "无"
        print(f"  {m.id:10s} {_state_short(m.state.value)} "
              f"({x:2d},{y:2d},{z:2d})    {m.fuel:5.1f}  {tv_str}")
    # 威胁
    if ws.threats:
        print("\n  [威胁]")
        for t in ws.threats:
            x, y, z = t.position
            print(f"  {t.id:6s} [{t.kind:8s}] pos=({x:2d},{y:2d},{z:2d}) "
                  f"杀伤={list(t.lethality)}")
    # 区域覆盖率
    if ws.zones:
        print("\n  [区域覆盖率]")
        for z in ws.zones:
            bar = _bar(z.coverage)
            print(f"  {z.id}: [{bar}] {z.coverage * 100:5.1f}%")
    # 监控任务
    mm = runner._monitoring_manager
    tasks = mm.list_monitoring_tasks()
    if tasks:
        print("\n  [定时监控任务]")
        for t in tasks:
            print(f"  {t.task_id:6s} [{t.status.value:10s}] subject={t.subject:8s} "
                  f"{_answer_brief(t.last_result)}")
    # 本 tick 内环动作
    if rec and rec.actions:
        print("\n  [本 tick 内环下发动作]")
        for a in rec.actions[:8]:
            mid = a.get("munition_id", "?")
            action = a.get("action", "?")
            args = a.get("args", {})
            args_str = " ".join(f"{k}={v}" for k, v in args.items()) if args else ""
            print(f"  · {mid:8s} → {action}  {args_str}")
        if len(rec.actions) > 8:
            print(f"  · ... 共 {len(rec.actions)} 条")
    # 本 tick escalation
    if rec and rec.escalations:
        print("\n  [本 tick Escalation]")
        for e in rec.escalations:
            reason = e.get("reason", "?")
            src = e.get("source_task", "?")
            n_low = e.get("n_low_conf", 0)
            print(f"  · reason={reason} source={src} low_conf_answers={n_low}")
    # 本 tick 置信度
    if rec and rec.confidences:
        hist = {}
        for c in rec.confidences:
            bucket = "高" if c >= 0.85 else ("低" if c < 0.5 else "中")
            hist[bucket] = hist.get(bucket, 0) + 1
        parts = [f"{k}:{v}" for k, v in hist.items()]
        print(f"\n  Jev 置信度分布: {' '.join(parts)} (共 {len(rec.confidences)} 个 answer)")

    # 重规划日志
    if telemetry.replans:
        last = telemetry.replans[-1]
        print(f"\n  最近重规划: tick={last.tick} reason={last.reason} "
              f"new_v={last.new_plan_version} fails={last.consecutive_failures}")

    print("\n" + "=" * width)
    sleep_sec = float(os.environ.get("DEMO_TICK_SLEEP", "0.8"))
    print(f"  帧间停顿 {sleep_sec}s · Ctrl+C 中止")
    print("=" * width)


def _print_final_summary(runner: DualLoopRunner, intent: CommanderIntent, max_ticks: int) -> None:
    """打印运行结束后的摘要。"""
    print()
    _print_header("运行结束 · 最终摘要")
    print(f"  指挥员意图: {intent.raw_text}")
    print(f"  最大 tick 上限: {max_ticks}")
    print(f"  LLM model: {runner._config.llm.model}")
    print(f"  Jev mode: {runner._config.jev.mode} (base_url={runner._config.jev.base_url})")
    print("=" * 70)


# === 主入口 ===

_SCENARIOS = {
    "cooperative_search": {
        "file": "config/scenarios/cooperative_search.yaml",
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
        "intent": CommanderIntent(
            raw_text="编组缺员+目标消失+部分失联三连发",
            goal="临机事件处理",
            constraints=["6 发规模", "G1 打击 G2 搜索"],
            success_criteria="任务完成或安全返航",
        ),
        "max_ticks": 40,
    },
}


def main() -> None:
    scenario_key = os.environ.get("DEMO_SCENARIO", "cooperative_search")
    if scenario_key not in _SCENARIOS:
        print(f"未知场景 {scenario_key}，可选: {list(_SCENARIOS)}")
        sys.exit(1)
    sc = _SCENARIOS[scenario_key]
    intent = sc["intent"]
    max_ticks = int(os.environ.get("DEMO_MAX_TICKS", str(sc["max_ticks"])))

    # 项目根：.env 与 config/ 都相对项目根解析，保证在任意 cwd 下直接运行也能读到真实配置
    project_root = Path(__file__).resolve().parent.parent.parent

    # 加载配置：显式传绝对路径，避免 cwd 不在项目根时找不到 .env / config.yaml
    # 若 .env 未命中，LLM 配置为空 → make_llm_client 降级 MockLLMClient → 每轮 content 恒为空
    cfg = load_config(
        path=project_root / "config" / "config.yaml",
        env_file=project_root / ".env",
    )

    # 启动诊断：显式打印当前 LLM / Jev 是否走真实实现
    llm_ok = bool(cfg.llm.base_url and cfg.llm.api_key and cfg.llm.model)
    print("=" * 70)
    print(f"[启动] 项目根: {project_root}")
    print(f"[启动] LLM: {'real' if llm_ok else 'mock(降级!)'}  "
          f"model={cfg.llm.model or '(空)'}  base_url={cfg.llm.base_url or '(空)'}")
    print(f"[启动] Jev: {cfg.jev.mode}  base_url={cfg.jev.base_url or '(空)'}")
    if not llm_ok:
        print("[启动] ⚠ LLM 配置为空，请确认从项目根运行（python -m munition_system.demos.demo_live），"
              "且 .env 存在并含 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL")
    print("=" * 70)

    # 装配场景与 runner（不传 llm_responses → 走真实 LLM 路径）
    scenario_path = sc["file"]
    p = Path(scenario_path)
    if not p.is_absolute():
        p = project_root / p
    scenario = load_scenario(p)
    runner = DualLoopRunner(cfg, scenario, llm_responses=None)

    # === Monkey-patch runner 的几个阶段，插入可视化帧 ===

    sleep_sec = float(os.environ.get("DEMO_TICK_SLEEP", "0.8"))

    # 0) LLM 每轮输出打印：包装 llm.chat，每次调用后打印 content + tool_calls
    llm_round = {"n": 0}
    original_chat = runner._llm.chat

    def chat_with_log(messages, tools=None, temperature=None, seed=None):
        resp = original_chat(messages, tools=tools, temperature=temperature, seed=seed)
        if tools is not None:
            # ReAct 循环调用（带 tools）→ 计为第 N 轮
            llm_round["n"] += 1
            _print_llm_round(llm_round["n"], resp, ctx_label="ReAct 规划")
        else:
            # schema 设计等无 tools 调用（部署阶段）
            _print_llm_round(llm_round["n"] + 1, resp, ctx_label="schema 设计")
        return resp

    runner._llm.chat = chat_with_log  # type: ignore[assignment]

    # 0b) Laya 每次调用的输入/输出打印：包装 jev_client.decide
    #     内环每 tick 会多次调 decide（全局 schema 按 bound_to 分组 + 各监控任务各一次）
    #     用 orchestrator._set_client_context 记录当前 (tick, task_id) 供打印用
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
        subject = _resolve_subject(laya_ctx["task_id"], questions, runner._monitoring_manager)
        _print_laya_call(laya_ctx["tick"], laya_ctx["task_id"], subject, state, questions, answers)
        return answers

    runner._jev_client.decide = decide_with_log  # type: ignore[assignment]

    # 1) ReAct 规划阶段：在 _react_loop.run 前后打印等待帧
    original_react_run = runner._react_loop.run

    def react_run_with_frame(intent_arg, ctx, *, pending_escalations=None, plan_version=1):
        label = "重规划 ReAct" if plan_version > 1 else "首次 ReAct 规划"
        llm_round["n"] = 0  # 每轮规划会话内重新计数
        _print_planning_frame(label, intent_arg.raw_text)
        t0 = time.time()
        result = original_react_run(intent_arg, ctx,
                                    pending_escalations=pending_escalations,
                                    plan_version=plan_version)
        elapsed = time.time() - t0
        # 规划完成提示
        _clear_screen()
        _print_header(f"智能巡飞弹集群决策系统 · 实时 demo")
        print(f"  阶段: {label} 完成（耗时 {elapsed:.1f}s）")
        print(f"  产出 plan: {'是' if result.plan is not None else '否'}"
              f"  · iterations={result.iterations}"
              f"  · tool_calls={len(result.tool_calls)}")
        if result.warnings:
            print(f"  warnings: {len(result.warnings)}")
            for w in result.warnings[:3]:
                print(f"    - {w}")
        if result.plan is not None:
            print(f"\n  assignments: {len(result.plan.assignments)} 条")
            print(f"  groups: {list(result.plan.groups.keys())}")
            print(f"  monitoring_points: {result.plan.monitoring_points}")
        print("\n  2s 后开始 tick 循环...")
        print("=" * 70)
        time.sleep(2.0)
        return result

    runner._react_loop.run = react_run_with_frame  # type: ignore[assignment]

    # 2) tick 阶段：每个 tick 执行后打印一帧
    original_tick = runner._tick

    def tick_with_frame(ctx):
        done = original_tick(ctx)
        phase = f"plan_v{ctx.plan_version}" if ctx.plan_version else "init"
        _print_tick_frame(runner, ctx.tick, phase)
        if not done:
            time.sleep(sleep_sec)
        return done

    runner._tick = tick_with_frame  # type: ignore[assignment]

    # 3) 部署阶段：打印 schema 部署信息
    original_deploy = runner._deploy_plan

    def deploy_with_frame(ctx):
        original_deploy(ctx)
        schema = runner._jev_orchestrator.get_global_schema()
        _clear_screen()
        _print_header("部署阶段")
        plan = runner._plan
        if plan is not None:
            print(f"  plan_version: {plan.plan_version}")
            print(f"  assignments: {len(plan.assignments)} 条")
            print(f"  groups: {list(plan.groups.keys())}")
        if schema is not None:
            print(f"  全局 schema: version={schema.version}, "
                  f"questions={len(schema.questions)}")
            for q in schema.questions:
                print(f"    - [{type(q).__name__.replace('Question', '')}] "
                      f"id={q.id} bound_to={q.bound_to}")
        print("\n  1.5s 后进入监控循环...")
        print("=" * 70)
        time.sleep(1.5)

    runner._deploy_plan = deploy_with_frame  # type: ignore[assignment]

    # 4) 抑制重规划：Laya 对 searching 态常返回低置信 → 每tick触发 DeepSeek 重规划（30s/tick）
    #    Demo 模式下抑制所有 escalation/replan，让 coverage 逐帧可见推进
    #    escalation 仍记录在 telemetry（tick 帧可见），只是不触发 DeepSeek 再规划
    suppress_replan = os.environ.get("DEMO_SUPPRESS_REPLAN", "true").lower() in ("true", "1", "yes")

    original_handle_esc = runner._handle_escalations

    def handle_escalations_filtered(ctx):
        if suppress_replan:
            runner._pending_escalations.clear()
        else:
            original_handle_esc(ctx)

    runner._handle_escalations = handle_escalations_filtered  # type: ignore[assignment]

    # 5) 减慢 coverage 推进速度，让逐帧变化更可见
    #    原始 0.05/弹/tick × 9 弹 = 0.45/tick → 2-3 tick 即满；改为 0.005 → ~23 tick 填满
    cov_delta = float(os.environ.get("DEMO_COV_DELTA", "0.005"))

    def advance_cov_slow():
        from munition_system.common.types import MunitionState as _MS
        for m in runner._world.munitions.values():
            if m.state != _MS.SEARCHING or not m.target_zone:
                continue
            runner._world.zones.increment_coverage(m.target_zone, delta=cov_delta)

    runner._world._advance_coverage = advance_cov_slow  # type: ignore[assignment]

    # 6) 抑制 progress_stuck 触发的重规划（coverage 缓慢推进时 0<cov<0.5 会误判 stuck）
    original_handle_replan = runner._handle_replan

    def handle_replan_filtered(reason: str, ctx):
        if suppress_replan:
            return  # Demo 模式：跳过所有重规划
        if reason.startswith("progress_stuck"):
            return  # 跳过 stuck 重规划，让 coverage 继续推进
        original_handle_replan(reason, ctx)

    runner._handle_replan = handle_replan_filtered  # type: ignore[assignment]

    # === 运行 ===
    try:
        result = runner.run(intent, max_ticks=max_ticks)
    except KeyboardInterrupt:
        print("\n[用户中断，输出当前结果]")
        # 构造部分结果
        from munition_system.runner.loop import RunnerResult
        result = RunnerResult(
            plan=runner._plan,
            final_phase="interrupted",
            replan_count=runner._replan_count,
            escalation_count=runner._telemetry.escalation_count,
            safe_mode=runner._safe_mode,
            warnings=list(runner._warnings),
            telemetry=runner._telemetry,
        )

    _print_final_summary(runner, intent, max_ticks)
    print_result(f"实时 demo · 场景={scenario_key} · DeepSeek + 本地 Laya", result)


if __name__ == "__main__":
    main()
