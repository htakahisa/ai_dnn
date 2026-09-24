"""Train the gonta character model from actor-safe labeled examples."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_trainer import CharacterTrainer, CharacterTrainingExample


def train_gonta(
    train: Sequence[CharacterTrainingExample],
    validation: Sequence[CharacterTrainingExample],
    *,
    seed: int,
    epochs: int,
    batch_size: int = 16,
    directory: Path | None = None,
    config: CharacterModelConfig = CharacterModelConfig(),
    learning_rate: float = 1e-3,
    resume: bool = False,
) -> CharacterTrainer:
    trainer = CharacterTrainer(2, seed=seed, config=config, learning_rate=learning_rate, directory=directory)
    if resume:
        trainer.resume()
    trainer.fit(train, validation, epochs=epochs, batch_size=batch_size)
    return trainer
