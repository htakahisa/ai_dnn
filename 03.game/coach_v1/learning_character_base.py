"""Shared actor loader for one fixed-roster character checkpoint."""

from __future__ import annotations

from pathlib import Path

from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS, FIXED_ROSTER
from coach_v1.common.versions import CHARACTER_OBSERVATION_VERSION
from coach_v1.models import CharacterModel, select_character_action
from coach_v1.observation.character_encoder import CharacterInputError, CharacterObservation, CharacterObservationEncoder
from coach_v1.training.character_environment import CharacterAction
from coach_v1.training.character_trainer import load_character_model


class CharacterPolicy:
    def __init__(self, slot: int, path: Path | None = None, *, device: str = "cpu") -> None:
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < len(FIXED_ROSTER):
            raise ValueError("invalid fixed-roster slot")
        checkpoint = Path(path) if path is not None else CHARACTER_CHECKPOINT_PATHS[FIXED_ROSTER[slot].checkpoint_id] / "best.pt"
        self.model: CharacterModel = load_character_model(checkpoint, slot=slot, device=device)
        self.slot = slot
        self.device = device
        self.encoder = CharacterObservationEncoder()

    def act(self, observation: CharacterObservation) -> CharacterAction:
        if (not isinstance(observation, CharacterObservation)
                or observation.version != CHARACTER_OBSERVATION_VERSION
                or observation.map_hash != self.encoder.map_hash
                or observation.watch_points_hash != self.encoder.watch_points_hash
                or observation.grid.shape != (29, 26, 44)
                or observation.vector.shape != (106,)
                or observation.vector[84 + self.slot] != 1
                or observation.vector[84:89].sum() != 1):
            raise CharacterInputError("character actor observation mismatch")
        return select_character_action(self.model, observation, device=self.device)
