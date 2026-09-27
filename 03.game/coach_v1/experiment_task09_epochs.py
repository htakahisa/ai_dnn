"""Compare longer character training with the Task 09 data held fixed."""

from __future__ import annotations

import json

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, FIXED_ROSTER, REPORTS_DIR
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


def run(*, epochs: int = 80) -> dict:
    if epochs <= 16:
        raise ValueError("experiment must train longer than the Task 09 baseline")
    torch.set_num_threads(1)
    baseline = json.loads((REPORTS_DIR / "task09_character_curriculum.json").read_text(encoding="utf-8"))
    report = {
        "experiment": "epochs-only",
        "baseline_epochs": 16,
        "epochs": epochs,
        "training_examples_per_character": 120,
        "validation_examples_per_character": 80,
        "characters": {},
    }
    for slot, roster in enumerate(FIXED_ROSTER):
        training = build_character_curriculum(slot, seed=1000 + slot, count=120)
        validation = build_character_curriculum(slot, seed=2000 + slot, count=80)
        train_examples = [item.example for item in training]
        validation_examples = [item.example for item in validation]
        trainer = CharacterTrainer(
            slot, seed=3000 + slot,
            config=CharacterModelConfig(hidden_channels=8, hidden_features=64),
            learning_rate=0.002,
            directory=CHECKPOINTS_DIR / "experiments" / "task09_epochs80" / roster.checkpoint_id,
        )
        trainer.fit(train_examples, validation_examples, epochs=epochs, batch_size=16)
        last = trainer.evaluate(validation_examples).to_dict()
        trainer.model = load_character_model(trainer.directory / "best.pt", slot=slot)
        best = trainer.evaluate(validation_examples).to_dict()
        by_category = {}
        for category in ("point", "jitter", "random"):
            examples = [item.example for item in validation if item.category == category]
            by_category[category] = trainer.evaluate(examples).to_dict() if examples else None
        report["characters"][roster.checkpoint_id] = {
            "baseline": baseline["characters"][roster.checkpoint_id]["validation"],
            "epoch_16": trainer.history[15],
            "epoch_40": trainer.history[39] if epochs >= 40 else None,
            "epoch_80": trainer.history[79] if epochs >= 80 else None,
            "last": last,
            "best_epoch": min(trainer.history, key=lambda item: item["loss"])["epoch"],
            "best_validation": best,
            "best_training": trainer.evaluate(train_examples).to_dict(),
            "best_by_category": by_category,
            "checkpoint_directory": str(trainer.directory),
        }
        print(roster.checkpoint_id,
              "baseline", baseline["characters"][roster.checkpoint_id]["validation"]["facing_accuracy"],
              "epoch_80", last["facing_accuracy"],
              "best", best["facing_accuracy"],
              "best_epoch", report["characters"][roster.checkpoint_id]["best_epoch"],
              flush=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09_epoch_experiment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    run()
