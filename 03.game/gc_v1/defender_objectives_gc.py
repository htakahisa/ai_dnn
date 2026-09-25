"""Shared objective features for GC defender training and inference.

The helpers expose only team/objective timing at inference.  Training may use
the returned state to build teacher labels, while production remains driven by
the learned policy.
"""

from __future__ import annotations

from collections import deque

import numpy as np


RETAKE_COORDINATION_DIM = 6


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

