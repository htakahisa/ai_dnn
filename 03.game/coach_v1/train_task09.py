"""Reproducible staged curriculum run for the five character checkpoints."""

from __future__ import annotations

import json

import torch

from coach_v1.common.constants import FIXED_ROSTER, REPORTS_DIR
from coach_v1.models import CharacterModelConfig, select_character_action
from coach_v1.training.character_trainer import load_character_model
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.train_character_gorimaru import train_gorimaru
from coach_v1.train_character_gongon import train_gongon
from coach_v1.train_character_gonta import train_gonta
from coach_v1.train_character_kunta import train_kunta
from coach_v1.train_character_kurimaru import train_kurimaru


_TRAINERS = (train_gorimaru, train_gongon, train_gonta, train_kunta, train_kurimaru)


def run(*, training_count: int = 360, validation_count: int = 80,
        epochs: int = 80) -> dict:
    torch.set_num_threads(1)
    report = {"kind": "staged-supervised-curriculum", "epochs": epochs, "characters": {}}
    for slot, roster in enumerate(FIXED_ROSTER):
        training = build_character_curriculum(slot, seed=1000 + slot, count=training_count)
        validation = build_character_curriculum(slot, seed=2000 + slot, count=validation_count)
        trainer = _TRAINERS[slot](
            [item.example for item in training], [item.example for item in validation],
            seed=3000 + slot, epochs=epochs, batch_size=16,
            config=CharacterModelConfig(hidden_channels=8, hidden_features=64),
            learning_rate=0.002,
        )
        trainer.model = load_character_model(trainer.directory / "best.pt", slot=slot)
        predicted_use = [select_character_action(trainer.model, item.example.observation).use_ability
                         for item in validation]
        positive = sum(item.example.use_ability for item in validation)
        true_positive = sum(prediction and item.example.use_ability
                            for prediction, item in zip(predicted_use, validation))
        categories = {}
        for category in ("point", "jitter", "random"):
            examples = [item.example for item in validation if item.category == category]
            categories[category] = trainer.evaluate(examples).to_dict() if examples else None
        report["characters"][roster.checkpoint_id] = {
            "training_categories": {category: sum(item.category == category for item in training)
                                    for category in ("point", "jitter", "random")},
            "validation_categories": {category: sum(item.category == category for item in validation)
                                      for category in ("point", "jitter", "random")},
            "validation": trainer.evaluate([item.example for item in validation]).to_dict(),
            "validation_ability_use_labels": sum(item.example.use_ability for item in validation),
            "validation_predicted_ability_uses": sum(predicted_use),
            "validation_ability_use_recall": true_positive / positive if positive else None,
            "validation_sightings": sum(item.sightings for item in validation),
            "by_category": categories,
            "best_loss": trainer.best_loss,
            "training_step": trainer.training_step,
            "checkpoint_directory": str(trainer.directory),
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09_character_curriculum.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
