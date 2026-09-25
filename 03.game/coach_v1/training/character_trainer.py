"""Supervised character training over actor-safe observations and outcome labels.

Labels may be produced by a training simulator, but only CharacterObservation
reaches the network. No game, scenario, or critic object is accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn

from coach_v1.common.checkpoint import (
    CheckpointMetadata, build_checkpoint_metadata, build_checkpoint_payload,
    validate_checkpoint_compatibility,
)
from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS, FIXED_ROSTER
from coach_v1.common.types import Facing, ModelTarget
from coach_v1.models.character_model import CharacterModel, CharacterModelConfig
from coach_v1.observation.character_encoder import CharacterObservation, CharacterObservationEncoder


@dataclass(frozen=True)
class CharacterTrainingExample:
    observation: CharacterObservation
    facing: Facing
    use_ability: bool
    target: tuple[int, int] | None = None
    ability_effective: bool | None = None
    acceptable_facings: tuple[Facing, ...] | None = None


@dataclass(frozen=True)
class CharacterMetrics:
    samples: int
    loss: float
    facing_accuracy: float
    ability_accuracy: float
    unused_ability_accuracy: float
    target_accuracy: float | None
    ability_effective_rate: float | None

    def to_dict(self) -> dict:
        return vars(self).copy()


def _validate_examples(examples: Sequence[CharacterTrainingExample], slot: int,
                       encoder: CharacterObservationEncoder) -> None:
    if not examples:
        raise ValueError("training and validation sets must be nonempty")
    for example in examples:
        if not isinstance(example, CharacterTrainingExample):
            raise TypeError("expected CharacterTrainingExample")
        observation = example.observation
        if not isinstance(observation, CharacterObservation):
            raise TypeError("trainer accepts only CharacterObservation")
        if (observation.grid.shape != (29, 26, 44) or observation.vector.shape != (106,)
                or observation.mask.facing.shape != (8,) or observation.mask.ability_use.shape != (2,)
                or observation.mask.ability_target.shape != (26, 44)):
            raise ValueError("character observation shape mismatch")
        if observation.map_hash != encoder.map_hash or observation.watch_points_hash != encoder.watch_points_hash:
            raise ValueError("character observation map or watch-point hash mismatch")
        if observation.version != "character-observation-v1":
            raise ValueError("character observation version mismatch")
        if not np.isfinite(observation.grid).all() or not np.isfinite(observation.vector).all():
            raise ValueError("non-finite character observation")
        if observation.vector[84 + slot] != 1 or int(observation.vector[84:89].sum()) != 1:
            raise ValueError("observation belongs to another character slot")
        if not isinstance(example.facing, Facing) or not observation.mask.facing[tuple(Facing).index(example.facing)]:
            raise ValueError("masked facing label")
        if example.acceptable_facings is not None:
            accepted = example.acceptable_facings
            if (not isinstance(accepted, tuple) or not accepted
                    or any(not isinstance(item, Facing) or not observation.mask.facing[tuple(Facing).index(item)]
                           for item in accepted)
                    or len(set(accepted)) != len(accepted)
                    or example.facing not in accepted):
                raise ValueError("invalid acceptable facing labels")
        if not isinstance(example.use_ability, bool) or not observation.mask.ability_use[int(example.use_ability)]:
            raise ValueError("masked ability-use label")
        if example.use_ability:
            target = example.target
            if (not isinstance(target, tuple) or len(target) != 2
                    or any(not isinstance(value, int) or isinstance(value, bool) for value in target)
                    or not (0 <= target[0] < 26 and 0 <= target[1] < 44)
                    or not observation.mask.ability_target[target]):
                raise ValueError("masked ability target label")
        elif example.target is not None:
            raise ValueError("target requires ability use")
        if example.ability_effective is not None and (not example.use_ability or not isinstance(example.ability_effective, bool)):
            raise ValueError("ability effectiveness requires a used ability")


class CharacterTrainer:
    """Train one slot; validation loss chooses best and every epoch saves latest."""

    def __init__(self, slot: int, *, seed: int, config: CharacterModelConfig = CharacterModelConfig(),
                 learning_rate: float = 1e-3, directory: Path | None = None,
                 device: str = "cpu") -> None:
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < len(FIXED_ROSTER):
            raise ValueError("invalid fixed-roster slot")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        self.slot = slot
        self.seed = seed
        self.config = config
        self.device = device
        self.encoder = CharacterObservationEncoder()
        self.directory = Path(directory) if directory is not None else CHARACTER_CHECKPOINT_PATHS[FIXED_ROSTER[slot].checkpoint_id]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = CharacterModel(config).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.training_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.history: list[dict] = []

    def _expected_metadata(self) -> CheckpointMetadata:
        return build_checkpoint_metadata(
            target=ModelTarget.character(FIXED_ROSTER[self.slot].checkpoint_id),
            map_hash=self.encoder.map_hash,
            watch_points_hash=self.encoder.watch_points_hash,
            model_config=self.config.to_dict(), training_seed=self.seed,
            training_step=self.training_step,
        )

    def _batch(self, examples: Sequence[CharacterTrainingExample]):
        grid = torch.as_tensor(np.stack([e.observation.grid for e in examples]).copy(), device=self.device)
        vector = torch.as_tensor(np.stack([e.observation.vector for e in examples]).copy(), device=self.device)
        facing_mask = torch.as_tensor(np.stack([e.observation.mask.facing for e in examples]).copy(), device=self.device)
        use_mask = torch.as_tensor(np.stack([e.observation.mask.ability_use for e in examples]).copy(), device=self.device)
        target_mask = torch.as_tensor(np.stack([e.observation.mask.ability_target.reshape(-1) for e in examples]).copy(), device=self.device)
        facing_label = torch.tensor([tuple(Facing).index(e.facing) for e in examples], device=self.device)
        use_label = torch.tensor([int(e.use_ability) for e in examples], device=self.device)
        target_label = torch.tensor([e.target[0] * 44 + e.target[1] if e.target else 0 for e in examples], device=self.device)
        return grid, vector, facing_mask, use_mask, target_mask, facing_label, use_label, target_label

    def _loss_and_predictions(self, examples: Sequence[CharacterTrainingExample]):
        grid, vector, facing_mask, use_mask, target_mask, facing_label, use_label, target_label = self._batch(examples)
        facing, ability, target = self.model(grid, vector)
        facing = facing.masked_fill(~facing_mask, -torch.inf)
        ability = ability.masked_fill(~use_mask, -torch.inf)
        target = target.masked_fill(~target_mask, -torch.inf)
        if any(e.acceptable_facings is not None for e in examples):
            accepted = torch.zeros_like(facing, dtype=torch.bool)
            for row, example in enumerate(examples):
                for direction in example.acceptable_facings or (example.facing,):
                    accepted[row, tuple(Facing).index(direction)] = True
            facing_loss = -torch.logsumexp(
                nn.functional.log_softmax(facing, dim=1).masked_fill(~accepted, -torch.inf), dim=1,
            ).mean()
        else:
            facing_loss = nn.functional.cross_entropy(facing, facing_label)
        ability_loss = nn.functional.cross_entropy(ability, use_label)
        used = use_label.bool()
        target_loss = nn.functional.cross_entropy(target[used], target_label[used]) if used.any() else facing_loss.new_zeros(())
        return facing_loss + 2.0 * ability_loss + 0.5 * target_loss, facing.argmax(1), ability.argmax(1), target.argmax(1)

    def evaluate(self, examples: Sequence[CharacterTrainingExample], *, batch_size: int = 16) -> CharacterMetrics:
        _validate_examples(examples, self.slot, self.encoder)
        self.model.eval()
        total_loss = facing_correct = ability_correct = unused_correct = unused_count = 0
        target_correct = target_count = effective = effective_count = 0
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = examples[start:start + batch_size]
                loss, facing, ability, target = self._loss_and_predictions(batch)
                total_loss += float(loss.item()) * len(batch)
                for i, example in enumerate(batch):
                    accepted = example.acceptable_facings or (example.facing,)
                    facing_correct += int(tuple(Facing)[facing[i].item()] in accepted)
                    predicted_use = bool(ability[i].item())
                    ability_correct += int(predicted_use == example.use_ability)
                    if not example.use_ability:
                        unused_count += 1
                        unused_correct += int(not predicted_use)
                    else:
                        target_count += 1
                        target_correct += int(target[i].item() == example.target[0] * 44 + example.target[1])
                        if example.ability_effective is not None:
                            effective_count += 1
                            effective += int(example.ability_effective)
        return CharacterMetrics(
            len(examples), total_loss / len(examples), facing_correct / len(examples),
            ability_correct / len(examples), unused_correct / unused_count if unused_count else 0.0,
            target_correct / target_count if target_count else None,
            effective / effective_count if effective_count else None,
        )

    def fit(self, train: Sequence[CharacterTrainingExample], validation: Sequence[CharacterTrainingExample],
            *, epochs: int, batch_size: int = 16) -> list[dict]:
        _validate_examples(train, self.slot, self.encoder)
        _validate_examples(validation, self.slot, self.encoder)
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(epochs):
            self.model.train()
            order = np.random.default_rng(self.seed + self.epoch).permutation(len(train))
            for start in range(0, len(order), batch_size):
                batch = [train[int(index)] for index in order[start:start + batch_size]]
                loss, _, _, _ = self._loss_and_predictions(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.training_step += 1
            self.epoch += 1
            metrics = self.evaluate(validation, batch_size=batch_size)
            self.history.append({"epoch": self.epoch, "step": self.training_step, **metrics.to_dict()})
            improved = metrics.loss < self.best_loss
            if improved:
                self.best_loss = metrics.loss
            self.save("latest.pt")
            if improved:
                self.save("best.pt")
        return self.history

    def save(self, filename: str) -> Path:
        if filename not in ("latest.pt", "best.pt"):
            raise ValueError("checkpoint filename must be best.pt or latest.pt")
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = build_checkpoint_payload(
            metadata=self._expected_metadata(), model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )
        payload.update(epoch=self.epoch, best_loss=self.best_loss, history=self.history)
        path = self.directory / filename
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def resume(self, filename: str = "latest.pt") -> None:
        payload = torch.load(self.directory / filename, map_location=self.device, weights_only=False)
        metadata = CheckpointMetadata.from_dict(payload["metadata"])
        validate_checkpoint_compatibility(metadata, self._expected_metadata())
        self.encoder.validate_checkpoint(metadata, slot=self.slot)
        if metadata.training_seed != self.seed:
            raise ValueError("resume seed mismatch")
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        if payload["optimizer_state_dict"] is None:
            raise ValueError("checkpoint lacks optimizer state")
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.training_step = metadata.training_step
        self.epoch = int(payload["epoch"])
        self.best_loss = float(payload["best_loss"])
        self.history = list(payload["history"])


def load_character_model(path: Path, *, slot: int, device: str = "cpu") -> CharacterModel:
    """Load only the matching character's model after full metadata checks."""
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = CheckpointMetadata.from_dict(payload["metadata"])
    config = CharacterModelConfig(**metadata.model_config)
    encoder = CharacterObservationEncoder()
    expected = build_checkpoint_metadata(
        target=ModelTarget.character(FIXED_ROSTER[slot].checkpoint_id),
        map_hash=encoder.map_hash,
        watch_points_hash=encoder.watch_points_hash,
        model_config=config.to_dict(), training_seed=metadata.training_seed, training_step=0,
    )
    validate_checkpoint_compatibility(metadata, expected)
    encoder.validate_checkpoint(metadata, slot=slot)
    model = CharacterModel(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model
