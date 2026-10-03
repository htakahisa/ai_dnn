"""Fnatic-owned plant patterns and corresponding post-plant positions."""

from collections import deque
from functools import lru_cache

import numpy as np

from .map_data_guard_plant_fnatic import NEW_MAZE_STR as PLANT_MAZE
from .map_data_guard_fnatic import NEW_MAZE_STR as GUARD_MAZE


def region(cell, grid):
    """Map columns: left third A, middle third Mid, right third B."""
    if cell[1] < grid.shape[1] / 3:
        return 'A'
    if cell[1] >= grid.shape[1] * 2 / 3:
        return 'B'
    return 'MID'


def parse_grid(text):
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError('Fnatic map rows must have equal widths')
    return np.array([[int(cell) for cell in row] for row in rows], dtype=np.int32)


def distances(start, grid, blocked=()):
    """Walking distances with optional occupied-cell exclusions."""
    walls = (grid == 1).tobytes()
    blocked = frozenset(blocked) - {start}
    # Return a fresh dictionary so a caller cannot alter a later AI decision.
    return _cached_distances(start, grid.shape, walls, blocked).copy()


@lru_cache(maxsize=1024)
def _cached_distances(start, shape, walls, blocked):
    grid = np.frombuffer(walls, dtype=np.bool_).reshape(shape)
    return _walking_distances(start, grid, blocked)


def _walking_distances(start, grid, blocked=()):
    """Uncached BFS, also used when comparing optimized and original matches."""
    result = {start: 0}
    blocked = set(blocked) - {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cell = (r + dr, c + dc)
            if (cell in result or cell in blocked or not (0 <= cell[0] < grid.shape[0]
                                      and 0 <= cell[1] < grid.shape[1])
                    or grid[cell] == 1):
                continue
            result[cell] = result[(r, c)] + 1
            queue.append(cell)
    return result


class TacticalPositions:
    def __init__(self, plant_map=PLANT_MAZE, guard_map=GUARD_MAZE):
        self.plant_grid = parse_grid(plant_map)
        self.guard_grid = parse_grid(guard_map)
        self.plants = self._patterns(self.plant_grid)
        self.guards = self._patterns(self.guard_grid)
        if not self.plants or self.plants.keys() != self.guards.keys():
            raise ValueError('Fnatic plant and guard pattern IDs must correspond')
        self.plant_cells = tuple(p for cells in self.plants.values() for p in cells)

    @staticmethod
    def _patterns(grid):
        return {marker: tuple(tuple(map(int, p)) for p in zip(*np.where(grid == marker)))
                for marker in range(5, 10) if np.any(grid == marker)}

    def validate(self, grid):
        for tactical_grid in (self.plant_grid, self.guard_grid):
            if tactical_grid.shape != grid.shape:
                raise ValueError('Fnatic tactical maps must match game map dimensions')
            if not np.array_equal(tactical_grid == 1, grid == 1):
                raise ValueError('Fnatic tactical map walls must match the game map')
        if any(grid[cell] != 2 for cell in self.plant_cells):
            raise ValueError('Fnatic plant positions must be on plantable cells (2)')

    def pattern_for(self, anchor, grid):
        for marker, cells in self.plants.items():
            if anchor in cells:
                return marker
        lengths = distances(anchor, grid)
        reachable = [(lengths[p], marker) for marker, cells in self.plants.items()
                     for p in cells if p in lengths]
        return min(reachable)[1] if reachable else None

    def guard_candidates(self, marker, team_size, grid):
        registered = self.guards[marker]
        candidates = list(registered)
        seen = set(candidates)
        queue = deque(candidates)
        while queue and len(candidates) < team_size:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cell = (r + dr, c + dc)
                if (cell in seen or not (0 <= cell[0] < grid.shape[0]
                                          and 0 <= cell[1] < grid.shape[1])
                        or grid[cell] == 1):
                    continue
                seen.add(cell)
                queue.append(cell)
                candidates.append(cell)
                if len(candidates) >= team_size:
                    break
        return candidates
