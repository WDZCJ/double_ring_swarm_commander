"""体素网格 + 区域管理（simulation/environment.py）。

3D 整数坐标系，与用户 3D 体素数据结构偏好一致。
Zone 是网格内的矩形子区域，bounds=(xmin,xmax,ymin,ymax,zmin,zmax)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from munition_system.common.types import Zone
from munition_system.common.validation import validate_zone


class VoxelGrid:
    """3D 整数体素网格。

    grid_dim=(Nx, Ny, Nz)，坐标范围 [0, Nx) × [0, Ny) × [0, Nz)。
    提供 in_bounds / neighbors / distance 等基础查询。
    """

    def __init__(self, grid_dim: tuple[int, int, int] = (40, 40, 8)) -> None:
        if len(grid_dim) != 3 or not all(isinstance(d, int) and d > 0 for d in grid_dim):
            raise ValueError(f"grid_dim 必须是 3 个正 int 元组，得到 {grid_dim!r}")
        self.dim = tuple(grid_dim)
        self.Nx, self.Ny, self.Nz = self.dim

    def in_bounds(self, pos: tuple[int, int, int]) -> bool:
        if not (isinstance(pos, tuple) and len(pos) == 3):
            return False
        x, y, z = pos
        return 0 <= x < self.Nx and 0 <= y < self.Ny and 0 <= z < self.Nz

    def neighbors(self, pos: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        """6-邻接（上下左右前后）+ 边界裁剪。"""
        if not self.in_bounds(pos):
            return []
        x, y, z = pos
        candidates = [
            (x + 1, y, z), (x - 1, y, z),
            (x, y + 1, z), (x, y - 1, z),
            (x, y, z + 1), (x, y, z - 1),
        ]
        return [p for p in candidates if self.in_bounds(p)]

    @staticmethod
    def distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
        """欧氏距离的整数近似（曼哈顿距离，便于 A* 启发式）。"""
        return sum(abs(ai - bi) for ai, bi in zip(a, b))

    @staticmethod
    def euclidean_sq(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
        """欧氏距离平方（用于威胁半径比较，避免浮点）。"""
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


@dataclass
class ZoneRegistry:
    """区域注册表：zone_id -> Zone。提供查询/包含/覆盖率更新。"""

    zones: dict[str, Zone] = field(default_factory=dict)

    def add(self, zone: Zone) -> None:
        validate_zone(zone)
        if zone.id in self.zones:
            raise ValueError(f"Zone {zone.id} 已存在")
        self.zones[zone.id] = zone

    def get(self, zone_id: str) -> Zone | None:
        return self.zones.get(zone_id)

    def contains(self, zone_id: str, pos: tuple[int, int, int]) -> bool:
        """判断 pos 是否落在 zone_id 的 bounds 内。"""
        z = self.zones.get(zone_id)
        if z is None:
            return False
        xmin, xmax, ymin, ymax, zmin, zmax = z.bounds
        x, y, zz = pos
        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= zz <= zmax

    def update_coverage(self, zone_id: str, coverage: float) -> None:
        """更新区域覆盖率（0.0~1.0）。"""
        z = self.zones.get(zone_id)
        if z is None:
            return
        coverage = max(0.0, min(1.0, float(coverage)))
        # dataclass 不可变替换：用 dataclasses.replace
        from dataclasses import replace
        self.zones[zone_id] = replace(z, coverage=coverage)

    def increment_coverage(self, zone_id: str, delta: float) -> None:
        z = self.zones.get(zone_id)
        if z is None:
            return
        new_cov = max(0.0, min(1.0, z.coverage + delta))
        from dataclasses import replace
        self.zones[zone_id] = replace(z, coverage=new_cov)

    def all_zones(self) -> list[Zone]:
        return list(self.zones.values())


def make_zone(
    zone_id: str,
    bounds: tuple[int, int, int, int, int, int],
    coverage: float = 0.0,
) -> Zone:
    """创建 Zone 实例（带校验）。"""
    z = Zone(id=zone_id, bounds=tuple(bounds), coverage=coverage)  # type: ignore[arg-type]
    validate_zone(z)
    return z
