"""工具 Mock 实现（simulation/tools_impl.py）。

所有第五节工具的 Mock 实现，基于 SituationalApi + 简单算法。
后续可替换为真实算法/真实硬件适配，接口不变。

本模块实现 simulation 层能独立完成的工具：
- get_swarm_status / get_battlefield_situation
- plan_routes（A* 简化版，在体素网格上避威胁）
- analyze_target / assess_threat / design_grouping
- assign_tasks / check_task_progress / abort_munition_task / send_direct_command

monitoring task 相关 7 个工具（create/list/pause/resume/cancel/update_schema/inspect）
依赖 MonitoringTaskManager（agent 层），留到 P3 实现。
deploy_jev_schema / replan 涉及 agent 层状态，也留到 P3。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from munition_system.common.contracts import Tool, ToolContext, ToolResult
from munition_system.common.types import (
    ActionCommand,
    BattlePlan,
    Munition,
    MunitionState,
    TaskAssignment,
    Threat,
    ThreatVector,
)
from munition_system.simulation.threat import aggregate_lethality, threats_in_range


# === 工具输出 dataclass ===

@dataclass
class Route:
    """航线：航点序列 + 估计威胁等级。"""
    waypoints: list[tuple[int, int, int]] = field(default_factory=list)
    threat_level: ThreatVector = (0, 0, 0)  # 无威胁时 (0,0,0)
    length: int = 0


@dataclass
class TargetAnalysis:
    """目标分析结果。"""
    target_id: str
    position: tuple[int, int, int]
    kind: str
    defense_weakness: str  # 防御弱点（基于周边 threat.kind 推断）
    strike_suggestion: str  # 打击建议


@dataclass
class ThreatAssessment:
    """威胁评估结果。threat_vector 无威胁时 (0,0,0)。"""
    threat_vector: ThreatVector
    sources: list[str] = field(default_factory=list)  # 威胁来源 threat_id 列表
    level: str = "none"  # none | low | medium | high


@dataclass
class GroupingProposal:
    """编组协同方案。groups: group_id -> [munition_id]。"""
    groups: dict[str, list[str]] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class ProgressReport:
    """任务进度报告。"""
    plan_version: int
    completion: float  # 0.0~1.0
    stuck: bool = False
    failed: bool = False
    reason: str = ""
    satisfied_conditions: list[str] = field(default_factory=list)


@dataclass
class Ack:
    """通用确认。"""
    ok: bool = True
    message: str = ""
    data: Any = None


# === 工具基类（减少样板） ===

class _BaseTool:
    """工具基类：持有 name/description/input_schema，由子类填 call。"""

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        raise NotImplementedError


# === 工具实现 ===

class GetSwarmStatusTool(_BaseTool):
    """查询集群巡飞弹状态。"""

    name = "get_swarm_status"
    description = "查询集群巡飞弹状态。无威胁时 threat_vector=(0,0,0)。"
    input_schema = {
        "type": "object",
        "properties": {
            "munition_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "指定巡飞弹 id 列表；null=全部",
            },
        },
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.situational_api is None:
            return ToolResult(ok=False, error="situational_api 未注入")
        ids = args.get("munition_ids")
        munitions = ctx.situational_api.get_swarm_status(ids)
        # 序列化为紧凑 dict 列表（给 LLM 看）
        data = [
            {
                "id": m.id, "state": m.state.value, "role": m.role,
                "pos": list(m.position), "fuel": round(m.fuel, 2),
                "threat_vector": list(m.threat_vector),  # (0,0,0) 而非 None
                "target_zone": m.target_zone,
            }
            for m in munitions
        ]
        return ToolResult(ok=True, data=data)


class GetBattlefieldSituationTool(_BaseTool):
    """查询战场态势。"""

    name = "get_battlefield_situation"
    description = "查询战场态势（巡飞弹/威胁/区域）。zone_id 限定区域范围。"
    input_schema = {
        "type": "object",
        "properties": {
            "zone_id": {"type": "string", "description": "限定区域；null=全部"},
        },
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.situational_api is None:
            return ToolResult(ok=False, error="situational_api 未注入")
        ws = ctx.situational_api.get_world_state()
        zone_id = args.get("zone_id")
        threats = ctx.situational_api.get_threats_in_zone(zone_id)
        data = {
            "tick": ws.tick,
            "munitions": [
                {"id": m.id, "state": m.state.value, "pos": list(m.position),
                 "tv": list(m.threat_vector)}
                for m in ws.munitions
            ],
            "threats": [
                {"id": t.id, "kind": t.kind, "pos": list(t.position),
                 "lethality": list(t.lethality)}
                for t in threats
            ],
            "zones": [{"id": z.id, "cov": round(z.coverage, 2)} for z in ws.zones],
        }
        return ToolResult(ok=True, data=data)


class PlanRoutesTool(_BaseTool):
    """航线规划算法（A* 简化版，在体素网格上避威胁）。"""

    name = "plan_routes"
    description = "调用航线规划算法，返回候选多条航线（避开威胁）。"
    input_schema = {
        "type": "object",
        "properties": {
            "start": {"type": "array", "items": {"type": "integer"}, "description": "起点 [x,y,z]"},
            "end": {"type": "array", "items": {"type": "integer"}, "description": "终点 [x,y,z]"},
            "constraints": {"type": "object", "description": "约束（如避开 zone）"},
        },
        "required": ["start", "end"],
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.situational_api is None:
            return ToolResult(ok=False, error="situational_api 未注入")
        start = args.get("start")
        end = args.get("end")
        if not (isinstance(start, (list, tuple)) and len(start) == 3):
            return ToolResult(ok=False, error=f"start 必须长度 3，得到 {start!r}")
        if not (isinstance(end, (list, tuple)) and len(end) == 3):
            return ToolResult(ok=False, error=f"end 必须长度 3，得到 {end!r}")
        start_p = tuple(int(x) for x in start)
        end_p = tuple(int(x) for x in end)

        # A* 在体素网格上避威胁
        grid = ctx.world.grid if ctx.world is not None else None
        threats = ctx.situational_api.get_threats_in_zone(None)

        # 威胁位置集合（避开）
        threat_positions = {t.position for t in threats}
        # 威胁影响半径（曼哈顿距离 ≤3 视为高风险）
        def is_high_risk(pos: tuple[int, int, int]) -> bool:
            for tp in threat_positions:
                if sum(abs(a - b) for a, b in zip(pos, tp)) <= 3:
                    return True
            return False

        route = self._astar(start_p, end_p, grid, is_high_risk)
        if route is None:
            # 无可行航线→返回直连线（带高威胁标记）
            direct = [start_p, end_p]
            nearby = threats_in_range(threats, start_p, radius=10) + threats_in_range(threats, end_p, radius=10)
            tv = aggregate_lethality(nearby)
            r = Route(waypoints=direct, threat_level=tv, length=VoxelGrid_distance(start_p, end_p))
            return ToolResult(ok=True, data=[_route_to_dict(r)], warnings=["no_safe_route_fallback_to_direct"])

        # 沿航线聚合威胁
        nearby_threats: list[Threat] = []
        for wp in route:
            nearby_threats.extend(threats_in_range(threats, wp, radius=5))
        tv = aggregate_lethality(nearby_threats)
        r = Route(waypoints=route, threat_level=tv, length=sum(
            VoxelGrid_distance(a, b) for a, b in zip(route[:-1], route[1:])
        ))
        return ToolResult(ok=True, data=[_route_to_dict(r)])

    def _astar(
        self,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
        grid: Any,
        is_high_risk: Any,
    ) -> list[tuple[int, int, int]] | None:
        """A* 在体素网格上搜索；避开 high_risk 节点；找不到则返回 None。"""
        if grid is None:
            # 无 grid 时走直线
            return [start, goal]
        if not grid.in_bounds(start) or not grid.in_bounds(goal):
            return [start, goal]

        # 节点 = (x, y, z)；g/h/f
        open_heap: list[tuple[int, int, tuple[int, int, int]]] = []
        counter = 0
        heapq.heappush(open_heap, (VoxelGrid_distance(start, goal), counter, start))
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], int] = {start: 0}

        max_iter = 2000  # 防御性上限
        iter_count = 0
        while open_heap and iter_count < max_iter:
            iter_count += 1
            _, _, current = heapq.heappop(open_heap)
            if current == goal:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for nb in grid.neighbors(current):
                # 高风险节点代价 +10；普通节点代价 1
                step_cost = 10 if is_high_risk(nb) else 1
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(nb, math.inf):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + VoxelGrid_distance(nb, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, nb))
        return None


class AnalyzeTargetTool(_BaseTool):
    """目标分析。"""

    name = "analyze_target"
    description = "目标分析（位置/类型/防御弱点/打击建议）。"
    input_schema = {
        "type": "object",
        "properties": {"target_id": {"type": "string"}},
        "required": ["target_id"],
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.situational_api is None:
            return ToolResult(ok=False, error="situational_api 未注入")
        target_id = args.get("target_id")
        if not target_id:
            return ToolResult(ok=False, error="target_id 不能为空")
        threats = ctx.situational_api.get_threats_in_zone(None)
        target = next((t for t in threats if t.id == target_id), None)
        if target is None:
            return ToolResult(ok=False, error=f"目标 {target_id} 不存在")
        # 基于周边 threat 推断防御弱点
        nearby = threats_in_range(threats, target.position, radius=5)
        nearby = [t for t in nearby if t.id != target_id]
        if not nearby:
            weakness = "无近距防御，建议直接打击"
        else:
            kinds = {t.kind for t in nearby}
            if "sam" in kinds:
                weakness = "近距有 SAM，建议低空突防"
            elif "radar" in kinds:
                weakness = "近距有雷达，建议从盲区接近"
            else:
                weakness = "近距有轻武器，建议中高空打击"
        analysis = TargetAnalysis(
            target_id=target_id,
            position=target.position,
            kind=target.kind,
            defense_weakness=weakness,
            strike_suggestion=f"从 {weakness} 方向接近",
        )
        return ToolResult(ok=True, data={
            "target_id": analysis.target_id,
            "position": list(analysis.position),
            "kind": analysis.kind,
            "defense_weakness": analysis.defense_weakness,
            "strike_suggestion": analysis.strike_suggestion,
        })


class AssessThreatTool(_BaseTool):
    """威胁评估。"""

    name = "assess_threat"
    description = "威胁评估：扫描指定区域/航线周边威胁，聚合为 ThreatVector。"
    input_schema = {
        "type": "object",
        "properties": {
            "route": {"type": "array", "description": "航线航点列表"},
            "zone_id": {"type": "string", "description": "区域 id（与 route 二选一）"},
        },
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.situational_api is None:
            return ToolResult(ok=False, error="situational_api 未注入")
        route = args.get("route")
        zone_id = args.get("zone_id")
        all_threats = ctx.situational_api.get_threats_in_zone(None)
        if zone_id is not None:
            threats = ctx.situational_api.get_threats_in_zone(zone_id)
        elif isinstance(route, list) and route:
            # 沿航线扫描
            threats: list[Threat] = []  # type: ignore[no-redef]
            for wp in route:
                if isinstance(wp, (list, tuple)) and len(wp) == 3:
                    pos = tuple(int(x) for x in wp)
                    threats.extend(threats_in_range(all_threats, pos, radius=6))
            # 去重
            seen_ids = set()
            unique: list[Threat] = []
            for t in threats:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    unique.append(t)
            threats = unique
        else:
            return ToolResult(ok=False, error="需提供 route 或 zone_id")
        tv = aggregate_lethality(threats)  # 无威胁时 (0,0,0)
        # 等级判定
        max_v = max(tv) if tv != (0, 0, 0) else 0
        level = "none" if max_v == 0 else ("low" if max_v <= 1 else ("medium" if max_v <= 3 else "high"))
        assessment = ThreatAssessment(
            threat_vector=tv,
            sources=[t.id for t in threats],
            level=level,
        )
        return ToolResult(ok=True, data={
            "threat_vector": list(assessment.threat_vector),  # (0,0,0) 而非 None
            "sources": assessment.sources,
            "level": assessment.level,
        })


class DesignGroupingTool(_BaseTool):
    """编组协同设计：按区域数和弹数均分（启发式）。"""

    name = "design_grouping"
    description = "编组协同设计：按区域数和弹数均分。"
    input_schema = {
        "type": "object",
        "properties": {
            "task_kind": {"type": "string", "description": "任务类型 search/strike/..."},
            "available_munitions": {"type": "integer", "description": "可用弹数"},
            "zones": {"type": "array", "items": {"type": "string"}, "description": "目标区域 id 列表"},
        },
        "required": ["available_munitions", "zones"],
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        n = int(args.get("available_munitions", 0))
        zones = args.get("zones") or []
        if n <= 0 or not zones:
            return ToolResult(ok=False, error="available_munitions 和 zones 不能为空")
        if n > 18:
            return ToolResult(ok=False, error=f"集群上限 18，得到 {n}")
        # 均分：弹数 / 区域数，余数分给前几个组
        n_groups = len(zones)
        base = n // n_groups
        rem = n % n_groups
        groups: dict[str, list[str]] = {}
        mid_counter = 1
        for i, zid in enumerate(zones):
            count = base + (1 if i < rem else 0)
            members = [f"G{i+1}-M{j+1}" for j in range(count)]
            # 占用 mid_counter 序号保证全局唯一
            members = [f"G{i+1}-M{k}" for k in range(1, count + 1)]
            groups[f"G{i+1}"] = members
        rationale = f"均分 {n} 发到 {n_groups} 个区域：base={base}, 余数={rem}"
        proposal = GroupingProposal(groups=groups, rationale=rationale)
        return ToolResult(ok=True, data={
            "groups": proposal.groups,
            "rationale": proposal.rationale,
        })


class AssignTasksTool(_BaseTool):
    """任务下发：把 TaskAssignment 应用到巡飞弹（设状态/role/target_zone/route）。"""

    name = "assign_tasks"
    description = "任务下发：应用 assignments 到巡飞弹，返回实际生效列表。"
    input_schema = {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "description": "TaskAssignment 列表",
            },
        },
        "required": ["assignments"],
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.world is None:
            return ToolResult(ok=False, error="world 未注入")
        raw = args.get("assignments") or []
        from dataclasses import replace
        applied: list[dict] = []
        warnings: list[str] = []
        for a in raw:
            mid = a.get("munition_id")
            if not mid:
                warnings.append(f"assignment 缺少 munition_id: {a!r}")
                continue
            m = ctx.world.get_munition(mid)
            if m is None:
                warnings.append(f"巡飞弹 {mid} 不存在")
                continue
            # 根据 capability/role 决定态：search → SEARCHING（推进 coverage），其他 → ASSIGNED
            capability = str(a.get("capability", "")).lower()
            role_str = str(a.get("role", m.role or "")).lower()
            is_search = ("search" in capability) or ("search" in role_str)
            new_state = m.state
            if m.state == MunitionState.IDLE:
                new_state = MunitionState.SEARCHING if is_search else MunitionState.ASSIGNED
            # 解析 route
            route_raw = a.get("route") or []
            route = [tuple(int(x) for x in wp) for wp in route_raw if isinstance(wp, (list, tuple)) and len(wp) == 3]
            ctx.world.munitions[mid] = replace(
                m,
                role=a.get("role", m.role),
                target_zone=a.get("zone_id"),
                state=new_state,
            )
            applied.append({
                "munition_id": mid,
                "role": a.get("role", m.role),
                "zone_id": a.get("zone_id"),
                "capability": a.get("capability", ""),
                "route": [list(wp) for wp in route],
            })
        return ToolResult(
            ok=True,
            data={"applied": applied, "count": len(applied)},
            warnings=warnings,
        )


class CheckTaskProgressTool(_BaseTool):
    """任务进度检查：根据 plan.success_conditions 模式匹配 WorldState。"""

    name = "check_task_progress"
    description = "任务进度检查：完成度%/卡住/失败原因。"
    input_schema = {
        "type": "object",
        "properties": {"plan_version": {"type": "integer"}},
        "required": ["plan_version"],
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.situational_api is None or ctx.plan is None:
            return ToolResult(ok=False, error="situational_api 或 plan 未注入")
        plan_version = int(args.get("plan_version", 0))
        if plan_version != ctx.plan.plan_version:
            return ToolResult(ok=False, error=f"plan_version 不匹配：{plan_version} vs {ctx.plan.plan_version}")
        ws = ctx.situational_api.get_world_state()
        # 启发式：覆盖率平均 + 终态弹比例
        zones = ws.zones
        if zones:
            avg_cov = sum(z.coverage for z in zones) / len(zones)
        else:
            avg_cov = 0.0
        destroyed = sum(1 for m in ws.munitions if m.state == MunitionState.DESTROYED)
        total = max(1, len(ws.munitions))
        # 完成度：覆盖率 + 未损毁比例 加权
        alive_ratio = (total - destroyed) / total
        completion = 0.6 * avg_cov + 0.4 * alive_ratio
        # 卡住：覆盖率未增且未达 1.0
        stuck = avg_cov < 1.0 and avg_cov > 0 and avg_cov < 0.5
        # 失败：所有弹损毁
        failed = destroyed == total
        satisfied = []
        for cond in ctx.plan.success_conditions:
            if "覆盖" in cond or "搜索" in cond:
                if avg_cov >= 0.95:
                    satisfied.append(cond)
            elif "损毁" in cond or "摧毁" in cond:
                if destroyed > 0:
                    satisfied.append(cond)
        report = ProgressReport(
            plan_version=plan_version,
            completion=round(completion, 3),
            stuck=stuck,
            failed=failed,
            reason="coverage_stuck" if stuck else ("all_destroyed" if failed else ""),
            satisfied_conditions=satisfied,
        )
        return ToolResult(ok=True, data={
            "plan_version": report.plan_version,
            "completion": report.completion,
            "stuck": report.stuck,
            "failed": report.failed,
            "reason": report.reason,
            "satisfied_conditions": report.satisfied_conditions,
        })


class AbortMunitionTaskTool(_BaseTool):
    """取消单弹任务：转 RTB 态。"""

    name = "abort_munition_task"
    description = "取消单弹任务（转 RTB）。"
    input_schema = {
        "type": "object",
        "properties": {"munition_id": {"type": "string"}},
        "required": ["munition_id"],
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.world is None:
            return ToolResult(ok=False, error="world 未注入")
        mid = args.get("munition_id")
        if not mid:
            return ToolResult(ok=False, error="munition_id 不能为空")
        m = ctx.world.get_munition(mid)
        if m is None:
            return ToolResult(ok=False, error=f"巡飞弹 {mid} 不存在")
        from dataclasses import replace
        ctx.world.munitions[mid] = replace(m, state=MunitionState.RTB)
        return ToolResult(ok=True, data=Ack(ok=True, message=f"{mid} → RTB").__dict__)


class SendDirectCommandTool(_BaseTool):
    """直接下指令（应急）：构造 ActionCommand 并立即应用。"""

    name = "send_direct_command"
    description = "直接下指令（应急）：立即应用 ActionCommand 到巡飞弹。"
    input_schema = {
        "type": "object",
        "properties": {
            "munition_id": {"type": "string"},
            "action": {"type": "string", "description": "ActionType 字符串值或扩展动作"},
            "args": {"type": "object", "description": "动作参数"},
        },
        "required": ["munition_id", "action"],
    }

    def call(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.world is None:
            return ToolResult(ok=False, error="world 未注入")
        mid = args.get("munition_id")
        action = args.get("action")
        if not mid or not action:
            return ToolResult(ok=False, error="munition_id 和 action 不能为空")
        cmd = ActionCommand(munition_id=mid, action=str(action), args=args.get("args") or {})
        ws = ctx.world.snapshot()
        ws_after = ctx.world.step([cmd])
        return ToolResult(ok=True, data=Ack(ok=True, message=f"{mid} ← {action}").__dict__)


# === 工具注册表 ===

def make_default_tools() -> list[Tool]:
    """构造 simulation 层所有工具实例（不依赖 MonitoringTaskManager）。

    monitoring task 相关工具（create/list/pause/resume/cancel/update_schema/inspect）
    留到 P3 实现（依赖 agent 层的 MonitoringTaskManager）。
    """
    return [
        GetSwarmStatusTool(),
        GetBattlefieldSituationTool(),
        PlanRoutesTool(),
        AnalyzeTargetTool(),
        AssessThreatTool(),
        DesignGroupingTool(),
        AssignTasksTool(),
        CheckTaskProgressTool(),
        AbortMunitionTaskTool(),
        SendDirectCommandTool(),
    ]


def tools_to_openai_schema(tools: list[Tool]) -> list[dict]:
    """把工具列表转为 OpenAI function-calling schema 格式（P3 LLM 用）。"""
    out: list[dict] = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        })
    return out


# === 辅助函数 ===

def _route_to_dict(r: Route) -> dict:
    return {
        "waypoints": [list(wp) for wp in r.waypoints],
        "threat_level": list(r.threat_level),  # (0,0,0) 而非 None
        "length": r.length,
    }


def VoxelGrid_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    """曼哈顿距离（A* 启发式用）。"""
    return sum(abs(ai - bi) for ai, bi in zip(a, b))
