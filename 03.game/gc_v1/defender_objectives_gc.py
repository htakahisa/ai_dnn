"""Shared objective features for GC defender training and inference.

The helpers expose only team/objective timing at inference.  Training may use
the returned state to build teacher labels, while production remains driven by
the learned policy.
"""

from __future__ import annotations

from collections import deque

import numpy as np


RETAKE_COORDINATION_DIM = 6
RETAKE_UTILITY_CONTEXT_DIM = 7


def bfs_distance_map(grid, goal):
    """Return four-neighbour shortest-path distances from ``goal``."""
    height, width = grid.shape
    dist = np.full((height, width), -1, dtype=np.int32)
    gr, gc = map(int, goal)
    if not (0 <= gr < height and 0 <= gc < width) or int(grid[gr, gc]) == 1:
        return dist
    dist[gr, gc] = 0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < height
                and 0 <= nc < width
                and int(grid[nr, nc]) != 1
                and dist[nr, nc] < 0
            ):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def path_distance(dist_map, unit):
    r, c = map(int, unit.pos)
    value = int(dist_map[r, c])
    return value if value >= 0 else 10**9


def shortest_legal_action(dist_map, pos, mask, move_deltas):
    """Choose a legal action that strictly reduces shortest-path distance."""
    r, c = map(int, pos)
    current = int(dist_map[r, c])
    if current < 0:
        return None
    candidates = []
    for action, (dr, dc) in move_deltas.items():
        if action >= len(mask) or not mask[action] or (dr, dc) == (0, 0):
            continue
        nr, nc = r + int(dr), c + int(dc)
        if not (0 <= nr < dist_map.shape[0] and 0 <= nc < dist_map.shape[1]):
            continue
        distance = int(dist_map[nr, nc])
        if 0 <= distance < current:
            candidates.append((distance, int(action)))
    return min(candidates)[1] if candidates else None


def retake_approach_sector(pos, objective):
    """Group nearby firing positions by their bearing from the spike."""
    dr = int(pos[0]) - int(objective[0])
    dc = int(pos[1]) - int(objective[1])
    if dr == 0 and dc == 0:
        return "center"
    if abs(dr) >= abs(dc):
        return "north" if dr < 0 else "south"
    return "west" if dc < 0 else "east"


def assign_retake_lane_targets(
    grid, objective, defenders, objective_dist_map, remaining_ticks,
    *, entry_radius, defuse_ticks, safety_margin, max_detour_ticks=8,
):
    """Choose feasible entry positions from different sides of the objective.

    This supplies a training target and an observation feature. The deployed
    policy still chooses its own movement action.
    """
    objective = tuple(map(int, objective))
    frontier = {}
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            distance = int(objective_dist_map[r, c])
            if 2 <= distance <= int(entry_radius):
                sector = retake_approach_sector((r, c), objective)
                frontier.setdefault(sector, []).append((r, c))

    targets = {}
    sector_uses = {}
    budget = float(remaining_ticks) - int(defuse_ticks) - int(safety_margin)
    alive = [unit for unit in defenders if getattr(unit, "is_alive", True)]
    for unit in sorted(alive, key=lambda ally: (path_distance(objective_dist_map, ally), ally.name)):
        name = unit.name
        if path_distance(objective_dist_map, unit) <= 1:
            targets[name] = tuple(map(int, unit.pos))
            continue
        from_unit = bfs_distance_map(grid, unit.pos)
        options = []
        for sector, cells in frontier.items():
            reachable = []
            for cell in cells:
                travel = int(from_unit[cell])
                if travel < 0:
                    continue
                total = travel + max(0, int(objective_dist_map[cell]) - 1)
                if total <= budget:
                    reachable.append((total, travel, cell))
            if reachable:
                total, travel, cell = min(reachable)
                options.append((sector, total, travel, cell))
        if not options:
            targets[name] = objective
            continue
        shortest = min(option[1] for option in options)
        feasible = [
            option for option in options
            if option[1] <= shortest + int(max_detour_ticks)
        ]
        sector, _total, _travel, target = min(
            feasible,
            key=lambda option: (
                sector_uses.get(option[0], 0), option[1], option[0], option[3]
            ),
        )
        targets[name] = target
        sector_uses[sector] = sector_uses.get(sector, 0) + 1
    return targets


def retake_coordination_state(
    char,
    allies,
    dist_map,
    remaining_ticks,
    *,
    entry_radius,
    defuse_ticks,
    safety_margin,
):
    """Recompute shortest-path retake ETAs for the current tick.

    An ally is worth waiting for only while it can still reach the entry area
    and leave enough time for a full defuse plus the safety margin.
    """
    alive = [ally for ally in allies if getattr(ally, "is_alive", True)]
    self_distance = path_distance(dist_map, char)
    self_entry_eta = max(0, self_distance - int(entry_radius))
    self_defuse_eta = max(0, self_distance - 1) + int(defuse_ticks)

    eligible = []
    joined = []
    pending = []
    for ally in alive:
        if ally is char or getattr(ally, "name", None) == getattr(char, "name", None):
            continue
        distance = path_distance(dist_map, ally)
        entry_eta = max(0, distance - int(entry_radius))
        if entry_eta + int(defuse_ticks) + int(safety_margin) > remaining_ticks:
            continue
        eligible.append(ally)
        if distance <= int(entry_radius):
            joined.append(ally)
        else:
            pending.append((entry_eta, ally))

    max_pending_eta = max((eta for eta, _ally in pending), default=0)
    must_commit = remaining_ticks <= self_defuse_eta + int(safety_margin)
    return {
        "self_distance": self_distance,
        "self_entry_eta": self_entry_eta,
        "self_defuse_eta": self_defuse_eta,
        "eligible_allies": eligible,
        "joined_allies": joined,
        "pending_allies": [ally for _eta, ally in pending],
        "max_pending_eta": max_pending_eta,
        "team_ready": not pending,
        "must_commit": bool(must_commit),
        "should_wait": bool(pending) and not must_commit,
    }


def retake_coordination_features(state, normalizer):
    scale = max(1.0, float(normalizer))
    return np.asarray(
        (
            min(1.0, state["self_entry_eta"] / scale),
            min(1.0, len(state["eligible_allies"]) / 4.0),
            min(1.0, len(state["pending_allies"]) / 4.0),
            min(1.0, state["max_pending_eta"] / scale),
            float(state["team_ready"]),
            float(state["must_commit"]),
        ),
        dtype=np.float32,
    )


def retake_utility_state(
    allies, enemies, objective, smoke_cells, entry_radius, distance_map=None,
):
    """Observable utility already in effect and charges near the spike."""
    pr, pc = map(int, objective)

    def ready_to_cast(ally):
        r, c = map(int, ally.pos)
        if distance_map is not None:
            distance = int(distance_map[r, c])
            return 0 <= distance <= int(entry_radius)
        return max(abs(r - pr), abs(c - pc)) <= int(entry_radius)

    ready = [
        ally for ally in allies
        if getattr(ally, "is_alive", True)
        and ready_to_cast(ally)
    ]
    remaining = {"SMOKE": 0, "FLASH": 0, "RECON": 0}
    charges = {"SMOKE": "smoke_charges", "FLASH": "flash_charges", "RECON": "recon_charges"}
    for ally in ready:
        ability = str(getattr(ally, "ability_name", "")).upper()
        if ability in remaining and int(getattr(ally, charges[ability], 0)) > 0:
            remaining[ability] += 1
    alive_enemies = [enemy for enemy in enemies if getattr(enemy, "is_alive", True)]
    return {
        "ready": ready,
        "remaining": remaining,
        "smoke_active": any(
            max(abs(int(cell[0]) - pr), abs(int(cell[1]) - pc)) <= 1
            for cell in smoke_cells
        ),
        "blind_active": any(
            int(getattr(enemy, "blind_remaining", 0)) > 0
            for enemy in alive_enemies
        ),
        "reveal_active": any(
            int(getattr(enemy, "reveal_remaining", 0)) > 0
            for enemy in alive_enemies
        ),
    }


def retake_utility_features(state):
    remaining = state["remaining"]
    return np.asarray(
        (
            min(1.0, remaining["SMOKE"] / 5.0),
            min(1.0, remaining["FLASH"] / 5.0),
            min(1.0, remaining["RECON"] / 5.0),
            float(state["smoke_active"]),
            float(state["blind_active"]),
            float(state["reveal_active"]),
            min(1.0, len(state["ready"]) / 5.0),
        ),
        dtype=np.float32,
    )


def nearest_orb_assignment(chars, available_orbs, grid, cache, max_radius):
    """Assign one nearby ult-hungry defender to one orb by path distance."""
    candidates = []
    for raw_orb in available_orbs or ():
        orb = tuple(map(int, raw_orb))
        dist_map = cache.setdefault(orb, bfs_distance_map(grid, orb))
        for char in chars:
            if (
                not getattr(char, "is_alive", True)
                or getattr(char, "ultimate_cost", 0) <= 0
                or getattr(char, "ultimate_points", 0)
                >= getattr(char, "ultimate_cost", 0)
            ):
                continue
            distance = path_distance(dist_map, char)
            if distance <= int(max_radius):
                candidates.append(
                    (distance, str(getattr(char, "name", "")), orb, char, dist_map)
                )
    return min(candidates, key=lambda row: row[:3]) if candidates else None
