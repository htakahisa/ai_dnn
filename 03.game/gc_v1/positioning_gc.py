"""GC tactical map data shared by training and inference (no action policy)."""

import numpy as np
from collections import deque

try:
    from .map_data_guard_plant_gc import NEW_MAZE_STR as PLANT_MAZE
    from .map_data_guard_gc import NEW_MAZE_STR as GUARD_MAZE
except ImportError:
    from map_data_guard_plant_gc import NEW_MAZE_STR as PLANT_MAZE
    from map_data_guard_gc import NEW_MAZE_STR as GUARD_MAZE

POSITIONING_VERSION = 2  # v2 distinguishes actual defuser LOS from partial spike LOS.


def parse_grid(text):
    return np.array([[int(ch) for ch in line.strip()]
                     for line in text.splitlines() if line.strip()], dtype=np.int32)


PLANT_GRID = parse_grid(PLANT_MAZE)
GUARD_GRID = parse_grid(GUARD_MAZE)
PLANT_PATTERNS = {marker: [tuple(map(int, p)) for p in zip(*np.where(PLANT_GRID == marker))]
                  for marker in range(5, 10) if np.any(PLANT_GRID == marker)}
REGISTERED_PLANT_CELLS = [p for cells in PLANT_PATTERNS.values() for p in cells]


def team_plant_target(game):
    """Read our shared tactical intent, independent of a viewer's IQ noise."""
    owner = getattr(game, "real_game", game)
    target = getattr(owner, "target_plant_pos", None)
    return tuple(map(int, target)) if target is not None else None


def set_team_plant_target(game, target):
    """Publish our decision to the match and the current perception view."""
    target = tuple(map(int, target))
    owner = getattr(game, "real_game", game)
    owner.target_plant_pos = target
    game.target_plant_pos = target


def validate_tactical_maps(grid):
    if grid.shape != PLANT_GRID.shape or grid.shape != GUARD_GRID.shape:
        raise ValueError("GC positioning maps must match the game map dimensions")
    if not PLANT_PATTERNS:
        raise ValueError("GC plant patterns 5-9 are missing")
    for marker, plants in PLANT_PATTERNS.items():
        guards = list(zip(*np.where(GUARD_GRID == marker)))
        if not guards:
            raise ValueError(f"GC plant pattern {marker} has no guard positions")
        for pos in plants:
            if int(grid[pos]) not in (2, 5):
                raise ValueError(f"GC plant pattern {marker}: {pos} is not plantable")
        for pos in guards:
            if int(grid[pos]) == 1:
                raise ValueError(f"GC guard pattern {marker}: {pos} is a wall")


def preferred_plant_cells(grid, cells):
    """Filter strategic target candidates; emergency plant cells remain legal."""
    candidates = [tuple(map(int, p)) for p in cells]
    preferred = [p for p in candidates if p in REGISTERED_PLANT_CELLS
                 and int(grid[p]) in (2, 5)]
    return preferred or candidates


def guard_candidates(grid, marker, team_size):
    """Keep registered guards; fill shortages with nearest walkable cells.

    Every living teammate needs a distinct, reachable target even when the
    tactical map contains fewer positions than players.
    """
    candidates = [tuple(map(int, p)) for p in zip(*np.where(GUARD_GRID == marker))]
    if not candidates:
        raise ValueError(f"GC guard pattern {marker} has no positions")
    seen = set(candidates)
    queue = deque(candidates)
    while queue and len(candidates) < team_size:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            pos = (r + dr, c + dc)
            if (pos in seen or not (0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1])
                    or int(grid[pos]) == 1):
                continue
            seen.add(pos)
            queue.append(pos)
            candidates.append(pos)
            if len(candidates) >= team_size:
                break
    if len(candidates) < team_size:
        raise ValueError(f"GC guard pattern {marker} has too few reachable positions")
    return candidates
