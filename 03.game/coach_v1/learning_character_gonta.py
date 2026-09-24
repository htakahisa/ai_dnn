"""Actor-only loader for gonta."""

from __future__ import annotations

from pathlib import Path

from coach_v1.learning_character_base import CharacterPolicy


class GontaPolicy(CharacterPolicy):
    def __init__(self, path: Path | None = None, *, device: str = "cpu") -> None:
        super().__init__(2, path, device=device)
