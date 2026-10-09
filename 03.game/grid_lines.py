"""Bounded geometry cache; callers always receive their own mutable path."""

from functools import lru_cache


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
