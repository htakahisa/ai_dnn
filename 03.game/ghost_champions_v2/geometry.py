"""Static geometry only. No hidden game objects or simulation LOS methods."""
from collections import deque
from functools import lru_cache
import math
import numpy as np
from grid_lines import line_cells

CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def valid(grid, point):
    r, c = point
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] != 1


def los(grid, start, end, smoke=(), occupied=()):
    cells = line_cells(start, end)
    if any(not valid(grid, p) for p in cells):
        return False
    return ((len(cells) <= 2 or not set(cells).intersection(smoke))
            and not set(cells[1:-1]).intersection(occupied))


def route(grid, start, goals, occupied=(), center=None, radius=None):
    """Occupancy-aware BFS; enforce leash once its owner is inside it."""
    start = tuple(start)
    blocked = set(occupied) - {start}
    goals = set(goals) - blocked
    if not goals:
        return [start]
    restrict = center is not None and math.dist(start, center) <= radius
    parents = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell in goals:
            result = [cell]
            while parents[result[-1]] is not None:
                result.append(parents[result[-1]])
            return result[::-1]
        for dr, dc in CARDINAL:
            nxt = cell[0]+dr, cell[1]+dc
            if (nxt not in parents and nxt not in blocked and valid(grid, nxt)
                    and (not restrict or math.dist(nxt, center) <= radius)):
                parents[nxt] = cell
                queue.append(nxt)
    return [start]


def watch_cells(grid, spike):
    return [p for r in range(spike[0]-1, spike[0]+2)
            for c in range(spike[1]-1, spike[1]+2)
            if valid(grid, p := (r, c))]


def angle_at(center, first, second):
    a = (first[0]-center[0], first[1]-center[1])
    b = (second[0]-center[0], second[1]-center[1])
    denominator = math.hypot(*a)*math.hypot(*b)
    if not denominator:
        return 0.0
    return math.degrees(math.acos(max(-1, min(1, sum(x*y for x,y in zip(a,b))/denominator))))


def projectile_path(grid, start, aim):
    """Same directional, wall-limited path construction as AbilityLosMixin."""
    dr, dc = aim[0]-start[0], aim[1]-start[1]
    if dr == dc == 0:
        return [tuple(start)]
    scale = max(grid.shape)*3
    end = start[0]+dr*scale, start[1]+dc*scale
    # Iterate the engine's Bresenham rule only to the first wall/edge. Building
    # and globally caching thousands of off-map cells for every candidate aim
    # would consume large amounts of memory during a multi-opponent evaluation.
    y, x = start
    ey, ex = end
    dx, dy = abs(ex-x), -abs(ey-y)
    sx, sy = (1 if x < ex else -1), (1 if y < ey else -1)
    error = dx+dy
    path = [tuple(start)]
    while (y,x) != end:
        twice = 2*error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy
        cell = y,x
        if not valid(grid, cell):
            break
        path.append(cell)
    return path


def nearest_floor(grid, point):
    return min((tuple(map(int, p)) for p in np.argwhere(grid != 1)),
               key=lambda p: (math.dist(p, point), p))


def axis_for(point, width):
    # Match BattleLogicMixin._round_tactic_snapshot's A-Mid-B boundaries.
    return "A" if point[1] < width/3 else "B" if point[1] > width*2/3 else "Mid"


@lru_cache(maxsize=2048)
def _recon_aim(shape, walls, start, axis, reference):
    grid = np.frombuffer(walls, dtype=np.bool_).reshape(shape)
    # Candidate aims are public floor tiles. Score actual impact, not aim cell.
    candidates = set()
    for r, c in np.argwhere(~grid):
        if axis_for((r,c), shape[1]) == axis:
            candidates.add((int(r), int(c)))
    best = None
    for aim in sorted(candidates):
        path = projectile_path(grid, start, aim)
        if len(path) <= 1:
            continue
        end = path[-1]
        # Covers a 9x9 square through walls; reference is a public map anchor.
        coverage = sum(max(abs(end[0]-p[0]), abs(end[1]-p[1])) <= 4 for p in candidates)
        score = (max(abs(end[0]-reference[0]), abs(end[1]-reference[1])) <= 4,
                 -math.dist(end, reference), coverage, len(path))
        if best is None or score > best[0]:
            best = score, aim
    return best[1] if best else None


def recon_aim(grid, start, axis, reference):
    return _recon_aim(grid.shape, (grid == 1).tobytes(), tuple(start), axis, tuple(reference))
