"""Test more training examples and facing-aware checkpoint selection.

The 80-example validation set selects checkpoints. A separate 160-example
holdout checks whether that choice generalizes beyond the selection set.
"""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, FIXED_ROSTER, REPORTS_DIR
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


def _scores(trainer: CharacterTrainer, path: Path, slot: int,
            validation: list, holdout: list) -> dict:
    trainer.model = load_character_model(path, slot=slot)
    return {
        "validation": trainer.evaluate(validation).to_dict(),
        "holdout": trainer.evaluate(holdout).to_dict(),
    }


def run(*, training_count: int = 360, epochs: int = 80) -> dict:
    if training_count <= 120 or epochs != 80:
        raise ValueError("comparison requires more than 120 examples and 80 epochs")
    torch.set_num_threads(1)
    config = CharacterModelConfig(hidden_channels=8, hidden_features=64)
    report = {
        "experiment": "more-examples-and-facing-selection",
        "training_count": training_count,
        "validation_count": 80,
        "holdout_count": 160,
        "epochs": epochs,
        "characters": {},
    }
    for slot, roster in enumerate(FIXED_ROSTER):
        training = build_character_curriculum(slot, seed=1000 + slot, count=training_count)
        validation = build_character_curriculum(slot, seed=2000 + slot, count=80)
        holdout = build_character_curriculum(slot, seed=4000 + slot, count=160)
        train_examples = [item.example for item in training]
        validation_examples = [item.example for item in validation]
        holdout_examples = [item.example for item in holdout]
        directory = CHECKPOINTS_DIR / "experiments" / f"task09_data{training_count}" / roster.checkpoint_id
        trainer = CharacterTrainer(slot, seed=3000 + slot, config=config,
                                   learning_rate=0.002, directory=directory)
        best_facing = -1.0
        best_facing_loss = float("inf")
        best_facing_epoch = 0
        for _ in range(epochs):
            trainer.fit(train_examples, validation_examples, epochs=1, batch_size=16)
            current = trainer.history[-1]
            if (current["facing_accuracy"] > best_facing
                    or (current["facing_accuracy"] == best_facing
                        and current["loss"] < best_facing_loss)):
                best_facing = current["facing_accuracy"]
                best_facing_loss = current["loss"]
                best_facing_epoch = current["epoch"]
                copyfile(directory / "latest.pt", directory / "facing_best.pt")

        original_dir = CHECKPOINTS_DIR / "experiments" / "task09_epochs80" / roster.checkpoint_id
        candidates = {
            "baseline_loss_best": original_dir / "best.pt",
            "baseline_epoch80": original_dir / "latest.pt",
            "more_data_loss_best": directory / "best.pt",
            "more_data_facing_best": directory / "facing_best.pt",
            "more_data_last": directory / "latest.pt",
        }
        report["characters"][roster.checkpoint_id] = {
            "baseline_training_count": 120,
            "facing_best_epoch": best_facing_epoch,
            "loss_best_epoch": min(trainer.history, key=lambda item: item["loss"])["epoch"],
            "training_categories": {category: sum(item.category == category for item in training)
                                    for category in ("point", "jitter", "random")},
            "holdout_categories": {category: sum(item.category == category for item in holdout)
                                   for category in ("point", "jitter", "random")},
            "scores": {name: _scores(trainer, path, slot, validation_examples, holdout_examples)
                       for name, path in candidates.items()},
            "history": trainer.history,
            "checkpoint_directory": str(directory),
        }
        selected = report["characters"][roster.checkpoint_id]["scores"]
        print(roster.checkpoint_id,
              "holdout facing baseline/new loss/new facing",
              *(round(selected[name]["holdout"]["facing_accuracy"], 3)
                for name in ("baseline_loss_best", "more_data_loss_best", "more_data_facing_best")),
              flush=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09_data_selection_experiment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    run()
