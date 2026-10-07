"""Versioned assembly routes: unrestricted v1 and rear-only approaches in v2."""

from collections import deque
from functools import lru_cache

import numpy as np

from concon_v1.co1_attacker_common import bfs_distance_map
from concon_v1.co1_retake_config import COORDINATION_VERSION

STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def assembly_navigation(scenario, version=COORDINATION_VERSION):
    if version == 1:
        return (np.zeros(scenario.grid.shape, dtype=bool),
                {point: bfs_distance_map(scenario.grid, point) for point in scenario.rally_points})
    if version != 2:
        raise ValueError(f"unsupported retake coordination version: {version}")
    return _navigation(scenario.grid.shape, scenario.grid.astype(np.int32).tobytes(),
                       getattr(scenario, "map_name", None), tuple(scenario.rally_points))


@lru_cache(maxsize=8)
def _navigation(shape, terrain, site, rally):
    grid = np.frombuffer(terrain, dtype=np.int32).reshape(shape)
    front = np.zeros(shape, dtype=bool)
    ordinary = {point: bfs_distance_map(grid, point) for point in rally}
    # The static entry lanes from A to every plant tile describe the site side
    # without consulting any hidden opponent positions or a blurred live plant.
    for r, c in zip(*np.where(grid == 2)):
        if ("L" if c < shape[1] / 2 else "R") != site:
            continue
        to_site = bfs_distance_map(grid, (r, c))
        for point, to_A in ordinary.items():
            if to_site[point] >= 0:
                front |= ((to_site >= 0) & (to_A >= 0) & (to_site + to_A == to_site[point]))
    for point in rally:
        front[point] = False
    # A pocket surrounded by entry lanes is also on the site side. Mark it
    # before directed routing so an actor already there can retreat through a
    # lane rather than being trapped by the rear -> front restriction.
    rear = np.zeros(shape, dtype=bool)
    pending = deque(rally)
    for point in rally:
        rear[point] = True
    while pending:
        r, c = pending.popleft()
        for dr, dc in STEPS:
            point = r + dr, c + dc
            if (0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]
                    and grid[point] != 1 and not front[point] and not rear[point]):
                rear[point] = True
                pending.append(point)
    front |= (grid != 1) & ~rear
    routes = {}
    for goal in rally:
        distances = np.full(shape, -1, dtype=np.int32)
        distances[goal] = 0
        pending = deque([goal])
        while pending:
            r, c = pending.popleft()
            for dr, dc in STEPS:
                source = r + dr, c + dc
                if not (0 <= source[0] < shape[0] and 0 <= source[1] < shape[1]):
                    continue
                if grid[source] == 1 or distances[source] >= 0:
                    continue
                # Reverse traversal: front -> rear is permitted for retreat,
                # rear -> front is forbidden while assembling.
                if front[r, c] and not front[source]:
                    continue
                distances[source] = distances[r, c] + 1
                pending.append(source)
        distances.setflags(write=False)
        routes[goal] = distances
    front.setflags(write=False)
    return front, routes


def assembly_step_allowed(front, source, destination):
    return bool(front[source] or not front[destination])
