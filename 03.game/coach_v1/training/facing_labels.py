"""Training-only facing labels derived from legal, current team reports."""

from __future__ import annotations

import math

from coach_v1.common.constants import FACING_DELTAS
from coach_v1.common.types import Facing
from coach_v1.perception.team_perception import TeamPerceptionSnapshot


def acceptable_facings_from_snapshot(
    snapshot: TeamPerceptionSnapshot, *, slot: int = 0,
) -> tuple[Facing, ...]:
    """Accept every heading within 45 degrees of any shared sighting."""
    origin = snapshot.allies[slot].position
    accepted = []
    for facing, (dr, dc) in FACING_DELTAS.items():
        for sighting in snapshot.sightings:
            delta_r = sighting.reported_position[0] - origin[0]
            delta_c = sighting.reported_position[1] - origin[1]
            distance = math.hypot(delta_r, delta_c)
            if (distance == 0 or (dr * delta_r + dc * delta_c) /
                    (math.hypot(dr, dc) * distance) >= math.cos(math.pi / 4) - 1e-9):
                accepted.append(facing)
                break
    return tuple(accepted) or (snapshot.allies[slot].facing,)
