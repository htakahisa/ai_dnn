"""Shared geometry for defender search ability support."""

from dataclasses import dataclass

from game_core import (
    FLASH_MAX_FLIGHT_TICKS, FLASH_SPEED_CELLS_PER_TICK, RECON_REVEAL_SIZE,
)


@dataclass(frozen=True)
class SupportAbilityPlan:
    aim: tuple[int, int]
    enemy_name: str
    impact: tuple[int, int] | None = None


def line_cells(start, end):
    """Use the same Bresenham steps as abilities_los._line_cells."""
    y0, x0 = map(int, start)
    y1, x1 = map(int, end)
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    cells = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def clear_los(grid, start, end, smoke_cells=()):
    return all(grid[r, c] != 1 and (r, c) not in smoke_cells
               for r, c in line_cells(start, end))


def projectile_impact(grid, start, aim, role):
    """Predict where the real game will burst a FLASH or RECON projectile."""
    start = tuple(map(int, start))
    aim = tuple(map(int, aim))
    if start == aim:
        return None
    scale = max(grid.shape) * 3
    far = (start[0] + (aim[0] - start[0]) * scale,
           start[1] + (aim[1] - start[1]) * scale)
    path = [start]
    for r, c in line_cells(start, far)[1:]:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            break
        if grid[r, c] == 1:
            break
        path.append((r, c))
    if len(path) <= 1:
        return None
    if role == "FLASH":
        return path[min(len(path) - 1,
                        FLASH_SPEED_CELLS_PER_TICK * FLASH_MAX_FLIGHT_TICKS)]
    return path[-1]


def _smoke_cells(grid, center):
    r, c = center
    return {
        (rr, cc)
        for rr in range(r - 1, r + 2)
        for cc in range(c - 1, c + 2)
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
        and grid[rr, cc] != 1
    }


def find_support_ability_plan(
    grid, caster, allies, enemies, role, smoke_cells=(), max_aim_range=8,
):
    """Find a useful cast for an ally's *current* sighting, without own LOS.

    FLASH/RECON require a simulated burst that affects the spotted enemy.
    SMOKE must preserve every ally's current firing line and cover an approach
    behind the enemy. No last-seen or hidden enemy coordinates are used.
    """
    if role not in ("FLASH", "RECON", "SMOKE"):
        return None
    caster_pos = tuple(map(int, caster.pos))
    smoke_cells = set(smoke_cells)
    live_allies = [ally for ally in allies if ally is not caster
                   and getattr(ally, "is_alive", True)]
    engagements = [
        (ally, enemy) for enemy in enemies if getattr(enemy, "is_alive", True)
        for ally in live_allies
        if clear_los(grid, ally.pos, enemy.pos, smoke_cells)
    ]
    targets = [enemy for enemy in enemies
               if any(seen is enemy for _, seen in engagements)
               and not clear_los(grid, caster_pos, enemy.pos, smoke_cells)]
    targets.sort(key=lambda enemy: (
        not getattr(enemy, "has_spike", False),
        max(abs(int(enemy.pos[0]) - caster_pos[0]),
            abs(int(enemy.pos[1]) - caster_pos[1])),
        str(enemy.name),
    ))

    for enemy in targets:
        if role == "FLASH" and getattr(enemy, "blind_remaining", 0) > 0:
            continue
        if role == "RECON" and getattr(enemy, "reveal_remaining", 0) > 0:
            continue
        enemy_pos = tuple(map(int, enemy.pos))
        candidates = []
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                aim = (enemy_pos[0] + dr, enemy_pos[1] + dc)
                if (not 0 <= aim[0] < grid.shape[0]
                        or not 0 <= aim[1] < grid.shape[1]
                        or grid[aim] == 1 or aim == caster_pos
                        or (role != "SMOKE" and
                            max(abs(aim[0] - caster_pos[0]),
                                abs(aim[1] - caster_pos[1])) > max_aim_range)):
                    continue
                if role == "SMOKE":
                    if not 2 <= max(abs(dr), abs(dc)) <= 4:
                        continue
                    cells = _smoke_cells(grid, aim)
                    if (len(cells - smoke_cells) < 3
                            or caster_pos in cells or enemy_pos in cells
                            or any(tuple(ally.pos) in cells for ally in live_allies)
                            or any(cells.intersection(line_cells(ally.pos, seen.pos))
                                   for ally, seen in engagements)):
                        continue
                    shooter = min(
                        (ally for ally, seen in engagements if seen is enemy),
                        key=lambda ally: max(abs(ally.pos[0] - enemy_pos[0]),
                                             abs(ally.pos[1] - enemy_pos[1])),
                    )
                    forward = (enemy_pos[0] - int(shooter.pos[0]),
                               enemy_pos[1] - int(shooter.pos[1]))
                    projection = dr * forward[0] + dc * forward[1]
                    if projection <= 0:
                        continue
                    score = (projection / max(1, max(abs(forward[0]), abs(forward[1]))),
                             len(cells - smoke_cells), -abs(dr) - abs(dc))
                    candidates.append((score, aim, None))
                    continue
                aim_delta = (aim[0] - caster_pos[0], aim[1] - caster_pos[1])
                enemy_delta = (enemy_pos[0] - caster_pos[0],
                               enemy_pos[1] - caster_pos[1])
                dot = aim_delta[0] * enemy_delta[0] + aim_delta[1] * enemy_delta[1]
                if (dot <= 0 or 2 * dot * dot <
                        (aim_delta[0] ** 2 + aim_delta[1] ** 2)
                        * (enemy_delta[0] ** 2 + enemy_delta[1] ** 2)):
                    continue
                impact = projectile_impact(grid, caster_pos, aim, role)
                if impact is None:
                    continue
                if role == "FLASH":
                    effective = clear_los(grid, impact, enemy_pos, smoke_cells)
                else:
                    effective = max(abs(impact[0] - enemy_pos[0]),
                                    abs(impact[1] - enemy_pos[1])) <= RECON_REVEAL_SIZE // 2
                if effective:
                    score = (-max(abs(impact[0] - enemy_pos[0]),
                                  abs(impact[1] - enemy_pos[1])),
                             -max(abs(dr), abs(dc)),
                             -abs(dr) - abs(dc))
                    candidates.append((score, aim, impact))
        if candidates:
            _, aim, impact = max(candidates, key=lambda entry: entry[0])
            return SupportAbilityPlan(aim, str(enemy.name), impact)
    return None
