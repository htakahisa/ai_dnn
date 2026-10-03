"""Bounded shortest-distance cache for controller and collision helpers."""

from collections import deque
from functools import lru_cache
import numpy as np


def distance_map(grid, goal, moves=((-1, 0), (1, 0), (0, -1), (0, 1))):
    board = np.asarray(grid)
    # Return a copy: callers may annotate their own distance field.
    return _distance_map(board.shape, (board == 1).tobytes(), tuple(map(int, goal)), tuple(moves)).copy()


@lru_cache(maxsize=512)
def _distance_map(shape, walls, goal, moves):
    rows, columns = shape
    blocked = np.frombuffer(walls, dtype=np.bool_).reshape(shape)
    distances = np.full(shape, -1, np.int32)
    if not blocked[goal]:
        distances[goal] = 0
        queue = deque([goal])
        while queue:
            r, c = queue.popleft()
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < columns and not blocked[nr, nc] and distances[nr, nc] == -1:
                    distances[nr, nc] = distances[r, c] + 1
                    queue.append((nr, nc))
    distances.setflags(write=False)
    return distances


def walking_distance(grid, start, target, blocked=(), moves=((-1, 0), (1, 0), (0, -1), (0, 1))):
    board = np.asarray(grid)
    return _distance(board.shape, (board == 1).tobytes(), tuple(map(int, start)), tuple(map(int, target)),
                     frozenset(blocked), tuple(moves))


@lru_cache(maxsize=4096)
def _distance(shape, walls, start, target, blocked, moves):
    if start == target:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    rows, columns = shape
    while queue:
        (r, c), distance = queue.popleft()
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            pos = nr, nc
            if (pos in seen or not (0 <= nr < rows and 0 <= nc < columns)
                    or walls[nr * columns + nc] or pos in blocked):
                continue
            if pos == target:
                return distance + 1
            seen.add(pos)
            queue.append((pos, distance + 1))
    return float("inf")
