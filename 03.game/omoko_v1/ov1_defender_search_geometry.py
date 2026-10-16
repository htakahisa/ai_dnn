"""Bounded caches for search training; dynamic smoke is never cached."""

from collections import OrderedDict

import ov1_common_rl


class SearchGeometryCache:
    def __init__(self, distance_capacity=256, los_capacity=8192):
        self.distance_capacity = distance_capacity
        self.los_capacity = los_capacity
        self._grid_signature = None
        self._distances = OrderedDict()
        self._lines = OrderedDict()

    def _check_grid(self, grid):
        # Detect replacement as well as in-place edits, including test maps.
        signature = (grid.shape, grid.dtype.str, grid.tobytes())
        if signature != self._grid_signature:
            self._distances.clear()
            self._lines.clear()
            self._grid_signature = signature

    @staticmethod
    def _remember(cache, key, value, capacity):
        cache[key] = value
        if len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def distance_map(self, grid, goal):
        self._check_grid(grid)
        goal = tuple(map(int, goal))
        if goal in self._distances:
            self._distances.move_to_end(goal)
            return self._distances[goal]
        result = ov1_common_rl.bfs_distance_map(grid, goal)
        result.setflags(write=False)
        return self._remember(self._distances, goal, result, self.distance_capacity)

    def has_los(self, grid, start, end, smoke_cells=None):
        self._check_grid(grid)
        # Preserve direction: Bresenham tie handling is not always symmetric.
        key = (tuple(map(int, start)), tuple(map(int, end)))
        if key in self._lines:
            self._lines.move_to_end(key)
            wall_clear, cells = self._lines[key]
        else:
            cells = tuple(ov1_common_rl.line_cells(*key))
            wall_clear = all(grid[r, c] != 1 for r, c in cells)
            self._remember(self._lines, key, (wall_clear, cells), self.los_capacity)
        if not wall_clear:
            return False
        # Match the existing search LOS rule, including adjacent endpoints.
        if smoke_cells and len(cells) > 2:
            return not any(cell in smoke_cells for cell in cells)
        return True
