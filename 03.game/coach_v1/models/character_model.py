"""Actor-only character policy: facing, normal ability use, and target."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from coach_v1.common.types import Facing
from coach_v1.observation.character_encoder import CharacterInputError, CharacterObservation

if TYPE_CHECKING:
    from coach_v1.training.character_environment import CharacterAction


@dataclass(frozen=True)
class CharacterModelConfig:
    grid_channels: int = 29
    vector_features: int = 106
    hidden_channels: int = 32
    hidden_features: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


class CharacterModel(nn.Module):
    """One independent instance and checkpoint per fixed-roster character."""

    def __init__(self, config: CharacterModelConfig = CharacterModelConfig()) -> None:
        super().__init__()
        if min(config.grid_channels, config.vector_features,
               config.hidden_channels, config.hidden_features) <= 0:
            raise ValueError("model dimensions must be positive")
        self.config = config
        width = config.hidden_channels
        features = config.hidden_features
        self.spatial = nn.Sequential(
            nn.Conv2d(config.grid_channels, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.vector = nn.Sequential(nn.Linear(config.vector_features, features), nn.ReLU())
        self.joint = nn.Sequential(nn.Linear(width + features, features), nn.ReLU())
        self.facing_head = nn.Linear(features, len(Facing))
        self.ability_head = nn.Linear(features, 2)
        self.target_head = nn.Conv2d(width, 1, 1)
        self.target_context = nn.Linear(features, width)

    def forward(self, grid: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if grid.ndim != 4 or grid.shape[1] != self.config.grid_channels:
            raise ValueError("invalid character grid shape")
        if vector.ndim != 2 or vector.shape != (grid.shape[0], self.config.vector_features):
            raise ValueError("invalid character vector shape")
        spatial = self.spatial(grid)
        encoded = self.joint(torch.cat((spatial.mean(dim=(2, 3)), self.vector(vector)), dim=1))
        context = self.target_context(encoded)[:, :, None, None]
        target = (self.target_head(spatial) + (spatial * context).sum(dim=1, keepdim=True)
                  / math.sqrt(self.config.hidden_channels)).flatten(1)
        return self.facing_head(encoded), self.ability_head(encoded), target


def select_character_action(model: CharacterModel, observation: CharacterObservation,
                            *, device: str = "cpu") -> CharacterAction:
    """Choose a legal action from learned logits; mask only game-impossible choices."""
    from coach_v1.training.character_environment import CharacterAction
    if not isinstance(observation, CharacterObservation):
        raise TypeError("character actor requires a safe CharacterObservation")
    mask = observation.mask
    if not mask.facing.any() or not mask.ability_use[0]:
        raise CharacterInputError("character cannot act")
    model.eval()
    with torch.no_grad():
        grid = torch.as_tensor(np.asarray(observation.grid).copy(), device=device).unsqueeze(0)
        vector = torch.as_tensor(np.asarray(observation.vector).copy(), device=device).unsqueeze(0)
        facing, ability, target = model(grid, vector)
        facing = facing[0].masked_fill(~torch.as_tensor(mask.facing.copy(), device=device), -torch.inf)
        ability = ability[0].masked_fill(~torch.as_tensor(mask.ability_use.copy(), device=device), -torch.inf)
        facing_value = tuple(Facing)[int(facing.argmax().item())]
        use_ability = bool(ability.argmax().item())
        if not use_ability:
            return CharacterAction(facing_value)
        legal_targets = torch.as_tensor(mask.ability_target.reshape(-1).copy(), device=device)
        if not legal_targets.any():
            raise CharacterInputError("ability target mask is empty")
        index = int(target[0].masked_fill(~legal_targets, -torch.inf).argmax().item())
        columns = mask.ability_target.shape[1]
        return CharacterAction(facing_value, True, (index // columns, index % columns))
