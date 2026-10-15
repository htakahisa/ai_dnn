"""Low-priority, shared search-phase orb outing (training and runtime)."""

from dataclasses import dataclass

import numpy as np

from ov1_defender_search_routes import CARDINAL, deployment_step, reachable_cells
from ov1_ultimate_training import valid_orb_cells, ultimate_points_needed


IDLE_ORB_ENABLED = True
IDLE_ORB_WAIT_TICKS = 8
IDLE_ORB_MIN_REMAINING_TICKS = 20
IDLE_ORB_MAX_OUTING_TICKS = 40
IDLE_ORB_GUARD_RADIUS = 3
IDLE_ORB_PREFERRED_ABILITY = "FLASH"
IDLE_ORB_VERSION = "quiet_team_outing_flash_priority_v2"


def distance_map(grid, goal):
    distances = np.full(grid.shape, -1, dtype=np.int32)
    for cell, _ in reachable_cells(grid, goal):
        if cell == goal:
            distances[cell] = 0
        else:
            distances[cell] = 1 + min(
                distances[cell[0] + dr, cell[1] + dc]
                for dr, dc in CARDINAL
                if 0 <= cell[0] + dr < grid.shape[0]
                and 0 <= cell[1] + dc < grid.shape[1]
                and distances[cell[0] + dr, cell[1] + dc] >= 0
            )
    return distances


def facing_towards(start, target):
    dr, dc = target[0] - start[0], target[1] - start[1]
    return ("N" if dr < 0 else "S" if dr > 0 else "") + (
        "W" if dc < 0 else "E" if dc > 0 else ""
    ) or "N"


def guard_facing(grid, pos, orb):
    """Watch an open outward corridor, rather than an adjacent wall."""
    candidates = []
    for dr, dc in CARDINAL:
        clear = 0
        for step in range(1, 6):
            r, c = pos[0] + dr * step, pos[1] + dc * step
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or grid[r, c] == 1:
                break
            clear += 1
        outward = dr * (pos[0] - orb[0]) + dc * (pos[1] - orb[1])
        candidates.append(((clear > 0, outward > 0, clear, outward), (dr, dc)))
    _, (dr, dc) = max(candidates)
    return facing_towards(pos, (pos[0] + dr, pos[1] + dc))


@dataclass(frozen=True)
class OrbPlan:
    move: tuple
    facing: str
    collect: bool = False


class IdleOrbOuting:
    """One outing per round. Contact always yields to the existing combat policy.

    Home assignments are never modified. Only static geometry, live allies,
    available orbs and the caller's *observed* threat flag are used here.
    """

    def __init__(self):
        self.phase = "waiting"
        self.quiet_since = None
        self.started = None
        self.orb = None
        self.collector = None
        self.goals = {}
        self.maps = {}

    def update(self, grid, allies, homes, available_orbs, tick, remaining,
               threat=False, setup=False, planted=False):
        alive = [a for a in allies if a.is_alive]
        if not IDLE_ORB_ENABLED or setup or planted or not alive:
            return False
        if threat:
            self.quiet_since = None
            if self.phase == "outbound":
                self.phase = "returning"
            return False
        if self.phase == "done":
            return False
        if self.phase == "waiting":
            if self.quiet_since is None:
                self.quiet_since = tick
            if tick - self.quiet_since < IDLE_ORB_WAIT_TICKS:
                return False
            # Finish initial deployment before leaving the normal defense posts.
            if not all(a.name in homes and tuple(a.pos) == tuple(homes[a.name]) for a in alive):
                return False
            if remaining < IDLE_ORB_MIN_REMAINING_TICKS:
                return False
            candidates = valid_orb_cells(grid)
            if not candidates:
                self.phase = "done"
                return False
            # Select the central orb from the static map, even if already taken.
            self.orb = min(candidates, key=lambda p: (
                (p[0] - (grid.shape[0] - 1) / 2) ** 2
                + (p[1] - (grid.shape[1] - 1) / 2) ** 2, p,
            ))
            if self.orb not in available_orbs:
                self.phase = "done"
                return False
            orb_dist = distance_map(grid, self.orb)
            eligible = [a for a in alive if ultimate_points_needed(a) > 0
                        and orb_dist[tuple(a.pos)] >= 0]
            if not eligible:
                self.phase = "done"
                return False
            chosen = min(eligible, key=lambda a: (
                str(getattr(a, "ability_name", getattr(a, "role", ""))).upper()
                != IDLE_ORB_PREFERRED_ABILITY,
                ultimate_points_needed(a), int(orb_dist[tuple(a.pos)]), a.name,
            ))
            self.collector = chosen.name
            self.goals = {chosen.name: self.orb}
            # Keep the collector's shortest approach corridor free of guards.
            corridor = {self.orb}
            cell = tuple(chosen.pos)
            while orb_dist[cell] > 0:
                corridor.add(cell)
                dr, dc = deployment_step(grid, orb_dist, cell)
                cell = (cell[0] + dr, cell[1] + dc)
            guards = [a for a in alive if a.name != chosen.name]
            cells = [p for p, _ in reachable_cells(grid, self.orb)
                     if 1 < orb_dist[p] <= IDLE_ORB_GUARD_RADIUS
                     and p not in corridor]
            used = {self.orb}
            for guard in sorted(guards, key=lambda a: a.name):
                guard_dist = distance_map(grid, tuple(guard.pos))
                choices = [p for p in cells if p not in used and guard_dist[p] >= 0]
                if not choices:
                    # No safe formation: keep everyone at their normal posts.
                    self.phase = "done"
                    return False
                goal = min(choices, key=lambda p: (
                    -min(abs(p[0] - q[0]) + abs(p[1] - q[1]) for q in used),
                    int(guard_dist[p]), p,
                ))
                used.add(goal)
                self.goals[guard.name] = goal
            self.maps = {name: distance_map(grid, goal) for name, goal in self.goals.items()}
            self.started = tick
            self.phase = "outbound"
        if self.phase == "outbound":
            collector = next((a for a in alive if a.name == self.collector), None)
            if (self.orb not in available_orbs or collector is None
                    or ultimate_points_needed(collector) <= 0
                    or tick - self.started >= IDLE_ORB_MAX_OUTING_TICKS
                    or remaining < IDLE_ORB_MIN_REMAINING_TICKS):
                self.phase = "returning"
        if self.phase == "returning":
            self.goals = {a.name: tuple(homes[a.name]) for a in alive if a.name in homes}
            self.maps = {name: distance_map(grid, goal) for name, goal in self.goals.items()}
            if all(a.name in self.goals and tuple(a.pos) == self.goals[a.name] for a in alive):
                self.phase = "done"
                return False
        return True

    def plan(self, unit, grid, occupied=()):
        if unit.name not in self.maps:
            return None
        start = tuple(map(int, unit.pos))
        dist = self.maps[unit.name]
        # Guards must leave the orb itself and the collector's approach open.
        blocked = set(occupied)
        if self.phase == "outbound" and unit.name != self.collector:
            blocked.add(self.orb)
        move = deployment_step(grid, dist, start, blocked)
        nxt = (start[0] + move[0], start[1] + move[1])
        if self.phase == "outbound" and unit.name != self.collector:
            # Look outwards from the pickup, including while moving to cover.
            facing = guard_facing(grid, nxt, self.orb)
        else:
            facing = facing_towards(start, nxt) if move != (0, 0) else getattr(unit, "facing", "N")
        return OrbPlan(move, facing, self.phase == "outbound"
                       and unit.name == self.collector and start == self.orb)
