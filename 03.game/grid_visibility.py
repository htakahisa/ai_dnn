"""Exact batched Bresenham visibility with bounded immutable-result caches."""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=128)
def _rays(shape, origin):
    """Map geometry is independent of walls and moving smoke effects."""
    targets_r, targets_c = np.indices(shape)
    r = np.full(shape, origin[0], dtype=np.int32)
    c = np.full(shape, origin[1], dtype=np.int32)
    dx = np.abs(targets_c - origin[1])
    dy = -np.abs(targets_r - origin[0])
    sx = np.where(origin[1] < targets_c, 1, -1)
    sy = np.where(origin[0] < targets_r, 1, -1)
    error = dx + dy
    pending = np.ones(shape, bool)
    rays = []
    while pending.any():
        rays.append((r * shape[1] + c).ravel().copy())
        pending &= (r != targets_r) | (c != targets_c)
        e2 = error * 2
        move_c = pending & (e2 >= dy)
        move_r = pending & (e2 <= dx)
        error += np.where(move_c, dy, 0) + np.where(move_r, dx, 0)
        c += move_c * sx
        r += move_r * sy
    result = np.asarray(rays, dtype=np.int32)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=256)
def _clear_mask(shape, walls, origin, smoke):
    rays = _rays(shape, origin)
    clear = ~np.frombuffer(walls, dtype=np.bool_)[rays].any(axis=0).reshape(shape)
    if smoke:
        smoke_grid = np.zeros(shape, bool)
        for sr, sc in smoke:
            if 0 <= sr < shape[0] and 0 <= sc < shape[1]:
                smoke_grid[sr, sc] = True
        smoky = smoke_grid.ravel()[rays].any(axis=0).reshape(shape)
        rr, cc = np.indices(shape)
        nearby = np.maximum(abs(rr - origin[0]), abs(cc - origin[1])) <= 1
        clear &= ~smoky | nearby
    clear.setflags(write=False)
    return clear


def visible_cells(grid, viewers, smoke_cells):
    """Same forward-half-plane and smoke rules as the FRC public sensor."""
    board = np.asarray(grid)
    walls = (board == 1).tobytes()
    smoke = frozenset(smoke_cells)
    rr, cc = np.indices(board.shape)
    visible = np.zeros(board.shape, bool)
    for origin, direction in viewers:
        if not (0 <= origin[0] < board.shape[0] and 0 <= origin[1] < board.shape[1]):
            raise ValueError("viewer must be inside the grid")
        clear = _clear_mask(board.shape, walls, tuple(origin), smoke)
        forward = direction[0] * (rr - origin[0]) + direction[1] * (cc - origin[1]) >= -1e-9
        visible |= clear & forward
    # The sensor never marks a smoke cell visible, even at adjacent range.
    for r, c in smoke:
        if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
            visible[r, c] = False
    return set(map(tuple, np.argwhere(visible).tolist()))
