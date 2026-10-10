"""Bounded geometry cache; callers always receive their own mutable path."""

from functools import lru_cache

WALL_LOS_VERSION = 'symmetric_supercover_v1'


@lru_cache(maxsize=4096)
def wall_line_cells(start, end):
    """All cells touched by a center-to-center ray, including corner neighbors.

    Keep projectile/Bresenham paths separate: wall occlusion must be symmetric.
    """
    y, x = start
    ey, ex = end
    nx, ny = abs(ex-x), abs(ey-y)
    sx, sy = (1 if ex>x else -1), (1 if ey>y else -1)
    ix = iy = 0
    cells = [(y,x)]
    while ix < nx or iy < ny:
        horizontal, vertical = (1+2*ix)*ny, (1+2*iy)*nx
        if horizontal == vertical:
            cells.extend(((y,x+sx),(y+sy,x)))
            x, y = x+sx, y+sy
            ix, iy = ix+1, iy+1
        elif horizontal < vertical:
            x, ix = x+sx, ix+1
        else:
            y, iy = y+sy, iy+1
        cells.append((y,x))
    return tuple(cells)


def line_cells(start, end):
    return list(_line_cells(tuple(map(int, start)), tuple(map(int, end))))


@lru_cache(maxsize=4096)
def _line_cells(start, end):
    return tuple(iter_line_cells(start, end))


def iter_line_cells(start, end):
    """Yield the same directed Bresenham cells, allowing callers to stop early."""
    y0, x0 = start
    y1, x1 = end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    while True:
        yield y0, x0
        if x0 == x1 and y0 == y1:
            return
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy
