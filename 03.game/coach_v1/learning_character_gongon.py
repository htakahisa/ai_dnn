"""Actor-only loader for gongon."""

from __future__ import annotations

from pathlib import Path

from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS
from coach_v1.learning_character_base import CharacterPolicy
from coach_v1.observation.character_encoder import CharacterInputError, CharacterObservation
from coach_v1.observation.coach_encoder import LEGACY_COACH_VECTOR_FIELDS


_ATTACKER_INDEX = LEGACY_COACH_VECTOR_FIELDS.index("side_attacker")
_DEFENDER_INDEX = LEGACY_COACH_VECTOR_FIELDS.index("side_defender")


class GongonPolicy(CharacterPolicy):
    def __init__(self, path: Path | None = None, *, device: str = "cpu",
                 defender_path: Path | None = None,
                 watch_points_path: Path | None = None) -> None:
        super().__init__(1, path, device=device,
                         watch_points_path=watch_points_path)
        self._defender = (CharacterPolicy(
            1, defender_path or CHARACTER_CHECKPOINT_PATHS["gongon"] / "defender_best.pt",
            device=device, watch_points_path=watch_points_path,
        ) if path is None or defender_path is not None else None)

    def act(self, observation: CharacterObservation):
        if self._defender is None:
            return super().act(observation)
        if (not isinstance(observation, CharacterObservation)
                or observation.vector.shape != (106,)):
            raise CharacterInputError("character actor observation mismatch")
        attacker = observation.vector[_ATTACKER_INDEX]
        defender = observation.vector[_DEFENDER_INDEX]
        if attacker == 1 and defender == 0:
            return super().act(observation)
        if attacker == 0 and defender == 1:
            return self._defender.act(observation)
        raise CharacterInputError("character side observation mismatch")
