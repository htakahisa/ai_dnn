"""Shared deployment routes for training and runtime (no policy dependency)."""

from collections import deque

import numpy as np


CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DEPLOYMENT_VERSION = "uppercase_setup_frontier_queue_v2"
DEPLOYMENT_TRANSIT_WAIT_COST = 3  # Prefer a short queue to a long detour.


def reachable_cells(grid, start, occupied=()):
    """Yield reachable cells and their first step, in walking-distance order."""
    start = tuple(map(int, start))
    occupied = set(occupied) - {start}
    queue = deque([(start, (0, 0))])
    visited = {start}
    while queue:
        cell, first = queue.popleft()
        yield cell, first
        for dr, dc in CARDINAL:
            nxt = (cell[0] + dr, cell[1] + dc)
            r, c = nxt
            if (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                    and grid[r, c] != 1 and nxt not in occupied
                    and nxt not in visited):
                visited.add(nxt)
                queue.append((nxt, (dr, dc) if cell == start else first))


def setup_distance_map(setup_grid, normal_dist_map, spawn):
    """Route to the reachable frontier closest to the final uppercase marker.

    Distance to the final marker follows actual corridors in the normal map.
    All equally close frontier cells are valid waiting positions.
    """
    cells = [cell for cell, _ in reachable_cells(setup_grid, spawn)
             if normal_dist_map[cell] >= 0]
    if not cells:
        raise RuntimeError(f"Defense destination is unreachable from spawn={spawn}")
    best = min(int(normal_dist_map[cell]) for cell in cells)
    goals = [cell for cell in cells if normal_dist_map[cell] == best]
    dist = np.full(setup_grid.shape, -1, dtype=np.int32)
    queue = deque(goals)
    for cell in goals:
        dist[cell] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if (0 <= nr < dist.shape[0] and 0 <= nc < dist.shape[1]
                    and setup_grid[nr, nc] != 1 and dist[nr, nc] < 0):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def moving_deployment_cells(units, distance_maps):
    """Distinguish allies still deploying from allies holding their posts."""
    return {
        tuple(map(int, unit.pos)) for unit in units
        if unit.is_alive and unit.name in distance_maps
        and distance_maps[unit.name] is not None
        and distance_maps[unit.name][tuple(map(int, unit.pos))] > 0
    }


def deployment_step(grid, dist_map, start, occupied=(), transit_cells=()):
    """Take a free route or queue behind a moving ally instead of oscillating.

    Transit cells are virtual route endpoints, never legal movement cells.
    Compare waiting there with the full detour cost. Stationary occupants
    remain hard obstacles, including allies already at their defense posts.
    """
    start = tuple(map(int, start))
    current = int(dist_map[start])
    if current <= 0:
        return (0, 0)
    occupied = set(occupied) - {start}
    transit = set(transit_cells) & occupied
    queue = deque([(start, (0, 0), 0)])
    visited = {start}
    best_rank, best_move = None, (0, 0)
    fallback_dist, fallback = current, (0, 0)

    def consider(cost, first):
        nonlocal best_rank, best_move
        next_cell = (start[0] + first[0], start[1] + first[1])
        rank = (cost, int(dist_map[next_cell]))
        if best_rank is None or rank < best_rank:
            best_rank, best_move = rank, first

    while queue:
        cell, first, steps = queue.popleft()
        distance = int(dist_map[cell])
        if 0 <= distance < fallback_dist:
            fallback_dist, fallback = distance, first
        if distance == 0:
            consider(steps, first)
        for dr, dc in CARDINAL:
            nxt = (cell[0] + dr, cell[1] + dc)
            r, c = nxt
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or grid[nxt] == 1:
                continue
            if nxt in transit and dist_map[nxt] >= 0:
                consider(steps + 1 + int(dist_map[nxt]) + DEPLOYMENT_TRANSIT_WAIT_COST,
                         first)  # At the queue head, first=(0, 0): wait.
            if nxt in occupied or nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, (dr, dc) if cell == start else first, steps + 1))
    return best_move if best_rank is not None else fallback
