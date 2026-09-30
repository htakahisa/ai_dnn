"""Opt-in left-site attacker policy with a v2 post-plant handoff."""

from __future__ import annotations

import numpy as np

from coach_v1.common.versions import (
    COACH_OBSERVATION_VERSION, ORB_COACH_OBSERVATION_VERSION,
)
from coach_v1.models.coach_model import CoachPolicy
from coach_v1.observation.coach_encoder import (
    COACH_GRID_CHANNELS, COACH_VECTOR_FIELDS, ORB_COACH_VECTOR_FIELDS,
    CoachObservation,
)


class LeftSiteAttackPolicy:
    """Use the specialist until plant; keep the incumbent for post-plant play."""

    def __init__(self, specialist: CoachPolicy, incumbent: CoachPolicy) -> None:
        if (specialist.encoder.version != ORB_COACH_OBSERVATION_VERSION
                or incumbent.encoder.version != COACH_OBSERVATION_VERSION
                or specialist.encoder.map_hash != incumbent.encoder.map_hash
                or specialist.encoder.watch_points_hash != incumbent.encoder.watch_points_hash):
            raise ValueError("incompatible left-site and incumbent coaches")
        self.specialist = specialist
        self.incumbent = incumbent
        self.encoder = specialist.encoder
        self._old_indices = tuple(ORB_COACH_VECTOR_FIELDS.index(field)
                                  for field in COACH_VECTOR_FIELDS)

    def reset_round(self) -> None:
        self.specialist.reset_round()
        self.incumbent.reset_round()

    def act(self, observation: CoachObservation):
        if observation.version != self.encoder.version:
            raise ValueError("left-site coach requires the orb-aware observation")
        planted = observation.vector[ORB_COACH_VECTOR_FIELDS.index("spike_planted")]
        if not planted:
            return self.specialist.act(observation)
        old = CoachObservation(
            np.ascontiguousarray(observation.grid[:len(COACH_GRID_CHANNELS)]),
            np.ascontiguousarray(observation.vector[list(self._old_indices)]),
            self.incumbent.encoder.version,
            observation.map_hash,
            observation.watch_points_hash,
        )
        return self.incumbent.act(old)
