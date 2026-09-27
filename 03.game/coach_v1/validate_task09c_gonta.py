"""Independent watch-point and random-position validation for Gonta."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


BASELINE = CHECKPOINTS_DIR / "experiments" / "task09c_gonta" / "baseline_best.pt"


def run(candidate: Path) -> dict:
    torch.set_num_threads(1)
    records = build_character_curriculum(2, seed=4002, count=600, multi_facing=True)
    trainer = CharacterTrainer(2, seed=3002)
    result = {"seed": 4002, "samples": len(records), "checkpoints": {}}
    for name, path in (("baseline", BASELINE), ("candidate", candidate)):
        trainer.model = load_character_model(path, slot=2)
        result["checkpoints"][name] = {
            f"{side.value}_{category}": trainer.evaluate([
                row.example for row in records
                if row.side is side and row.category == category
            ]).to_dict()
            for side in (Side.ATTACKER, Side.DEFENDER)
            for category in ("point", "jitter", "random")
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09c_gonta_distribution_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    report = run(args.candidate)
    for name, groups in report["checkpoints"].items():
        print(name, {group: (row["samples"], row["facing_accuracy"], row["ability_accuracy"])
                     for group, row in groups.items()})
