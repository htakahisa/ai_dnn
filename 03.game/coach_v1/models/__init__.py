"""Network implementations shared by matching character train/learning modules."""

from .character_model import CharacterModel, CharacterModelConfig, select_character_action

__all__ = ["CharacterModel", "CharacterModelConfig", "select_character_action"]
