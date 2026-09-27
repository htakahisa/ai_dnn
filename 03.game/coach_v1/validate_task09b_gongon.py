"""Independent point/jitter/random validation for Gongon's routed checkpoints."""

from __future__ import annotations

import json

import torch

from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


def run() -> dict:
    torch.set_num_threads(1)
    records = build_character_curriculum(1, seed=4001, count=600, multi_facing=True)
    directory = CHARACTER_CHECKPOINT_PATHS["gongon"]
    trainer = CharacterTrainer(1, seed=3001)
    result = {"seed": 4001, "samples": len(records), "checkpoints": {}}
    for name, path in (("attacker", directory / "best.pt"),
                       ("defender", directory / "defender_best.pt")):
        trainer.model = load_character_model(path, slot=1)
        result["checkpoints"][name] = {
            f"{side.value}_{category}": trainer.evaluate([
                row.example for row in records
                if row.side is side and row.category == category
            ]).to_dict()
            for side in (Side.ATTACKER, Side.DEFENDER)
            for category in ("point", "jitter", "random")
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09b_gongon_distribution_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    for side, groups in report["checkpoints"].items():
        print(side, {category: (row["samples"], row["facing_accuracy"])
                     for category, row in groups.items()})
