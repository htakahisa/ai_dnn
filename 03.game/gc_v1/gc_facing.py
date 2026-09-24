"""Shared facing representation for Ghost Champions v1.

Facing is an action dimension in the GC policies.  Keeping the encoding in one
module is important because training and live inference must decode the same
action index and use the same geometric convention as battle_logic.py.
"""

from __future__ import annotations

import math


FACING_DIRS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
FACING_VECTORS = {
    "N": (0.0, -1.0),
    "NE": (0.70710678, -0.70710678),
    "E": (1.0, 0.0),
    "SE": (0.70710678, 0.70710678),
    "S": (0.0, 1.0),
    "SW": (-0.70710678, 0.70710678),
    "W": (-1.0, 0.0),
    "NW": (-0.70710678, -0.70710678),
}


def facing_from_delta(dr: int, dc: int, fallback: str | None = None) -> str | None:
    if dr == 0 and dc == 0:
        return fallback
    return max(
        FACING_DIRS,
        key=lambda d: FACING_VECTORS[d][0] * dc + FACING_VECTORS[d][1] * dr,
    )


def facing_towards(source, target) -> str | None:
    dr = float(target[0] - source[0])
    dc = float(target[1] - source[1])
    if math.hypot(dr, dc) == 0:
        return None
    return facing_from_delta(int(round(dr)), int(round(dc)))


def nearest_alive_enemy_facing(char, characters) -> str | None:
    """Face the nearest alive enemy using training-state information.

    GC combat ranges use Chebyshev distance, so the pre-aim teacher uses the
    same metric.  The name tie-break keeps labels deterministic when multiple
    enemies are equally close.  This helper is intended for training only;
    production controllers must infer facing from their observations.
    """
    source = tuple(char.pos)
    enemies = [
        other
        for other in characters
        if getattr(other, "is_alive", True)
        and getattr(other, "team", None) != getattr(char, "team", None)
        and tuple(other.pos) != source
    ]
    if not enemies:
        return None
    nearest = min(
        enemies,
        key=lambda other: (
            max(
                abs(int(other.pos[0]) - int(source[0])),
                abs(int(other.pos[1]) - int(source[1])),
            ),
            str(getattr(other, "name", "")),
        ),
    )
    return facing_towards(source, tuple(nearest.pos))


def encode_action(base_action: int, facing: str) -> int:
    return int(base_action) * len(FACING_DIRS) + FACING_DIRS.index(facing)


def decode_action(action: int) -> tuple[int, str]:
    base, facing_idx = divmod(int(action), len(FACING_DIRS))
    return base, FACING_DIRS[facing_idx]


def append_facing_onehot(values, facing: str):
    encoded = [1.0 if facing == direction else 0.0 for direction in FACING_DIRS]
    if hasattr(values, "shape"):
        values[-len(FACING_DIRS):] = encoded
    else:
        values.extend(encoded)
