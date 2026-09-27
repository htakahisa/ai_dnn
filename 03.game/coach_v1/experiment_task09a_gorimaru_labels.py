"""Compare Gorimaru facing labels on fixed, independent game observations."""

from __future__ import annotations

import json
from pathlib import Path
import random
from shutil import copyfile

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model
from coach_v1.training.gorimaru_rollout import collect_gorimaru_rollout


SCENARIOS = ("near", "west", "east", "crossfire", "natural")
EXPERIMENT = CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_labels"
BASELINE = CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_rollout" / "baseline_best.pt"
# Frozen Task 09-A official before the attacker adjustment was promoted.
CURRENT = CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_rollout" / "epoch70_latest.pt"


def collect(start: int, stop: int):
    return tuple(item for seed in range(start, stop)
                 for side in (Side.ATTACKER, Side.DEFENDER)
                 for scenario in SCENARIOS
                 for item in collect_gorimaru_rollout(
                     side=side, seed=seed, ticks=20, encounter=scenario,
                     actor_checkpoint=CURRENT))


def balanced_sample(records, *, per_group: int, seed: int):
    rng = random.Random(seed)
    selected = []
    for side in (Side.ATTACKER, Side.DEFENDER):
        for scenario in SCENARIOS:
            group = [row for row in records if row.side is side and row.encounter == scenario]
            selected.extend(rng.sample(group, min(per_group, len(group))))
    return tuple(selected)


def score(trainer: CharacterTrainer, path: Path, records) -> dict:
    trainer.model = load_character_model(path, slot=0)
    def subset(rows):
        if not rows:
            return {"samples": 0}
        metrics = trainer.evaluate([row.example for row in rows]).to_dict()
        metrics["mean_accepted_facings"] = sum(
            len(row.example.acceptable_facings or (row.example.facing,)) for row in rows
        ) / len(rows)
        return metrics
    return {
        "all": subset(records),
        "sighted": subset([row for row in records if row.has_sighting]),
        "by_side_scenario": {
            f"{side.value}_{scenario}": subset([
                row for row in records if row.side is side and row.encounter == scenario
                and row.has_sighting
            ])
            for side in (Side.ATTACKER, Side.DEFENDER) for scenario in SCENARIOS
        },
    }


def run() -> dict:
    torch.set_num_threads(1)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    train_records = collect(500, 512)
    validation_records = collect(600, 605)
    holdout_records = collect(700, 710)
    train_rollout = balanced_sample(train_records, per_group=60, seed=31)
    validation_rollout = balanced_sample(validation_records, per_group=15, seed=32)
    staged = build_character_curriculum(0, seed=1000, count=1400, multi_facing=True)
    staged_validation = build_character_curriculum(0, seed=2000, count=300,
                                                   multi_facing=True)
    staged_holdout = build_character_curriculum(0, seed=3000, count=200,
                                                multi_facing=True)
    train = [item.example for item in staged] + [item.example for item in train_rollout]
    validation = ([item.example for item in staged_validation]
                  + [item.example for item in validation_rollout])

    copyfile(CURRENT, EXPERIMENT / "latest.pt")
    copyfile(CURRENT, EXPERIMENT / "best.pt")
    trainer = CharacterTrainer(
        0, seed=3000, config=CharacterModelConfig(hidden_channels=8, hidden_features=64),
        learning_rate=0.001, directory=EXPERIMENT,
    )
    trainer.resume()
    trainer.best_loss = float("inf")
    paths = {"old": BASELINE, "current": CURRENT}
    for epochs in (5, 5, 10):
        trainer.fit(train, validation, epochs=epochs, batch_size=16)
        milestone = trainer.epoch
        for variant in ("best", "latest"):
            path = EXPERIMENT / f"epoch{milestone}_{variant}.pt"
            copyfile(EXPERIMENT / f"{variant}.pt", path)
            paths[f"epoch{milestone}_{variant}"] = path
        print("trained epoch", milestone, flush=True)

    result = {
        "teacher": "all current legal shared sightings; any facing within 45 degrees",
        "selection": "validation total loss; holdout seeds remain unused in training",
        "train_seeds": [500, 511], "validation_seeds": [600, 604],
        "holdout_seeds": [700, 709],
        "train_rollout_collected": len(train_records),
        "train_rollout_samples": len(train_rollout),
        "validation_rollout_samples": len(validation_rollout),
        "holdout_rollout_samples": len(holdout_records),
        "staged_train_samples": len(staged),
        "staged_validation_samples": len(staged_validation),
        "staged_holdout_samples": len(staged_holdout),
        "watchpoint_training_share": len(staged) / len(train),
        "checkpoints": {},
        "history": trainer.history[-20:],
    }
    for name, path in paths.items():
        result["checkpoints"][name] = {
            "path": str(path),
            "validation": score(trainer, path, validation_records),
            "holdout": score(trainer, path, holdout_records),
            "staged_holdout": trainer.evaluate([row.example for row in staged_holdout]).to_dict(),
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09a_gorimaru_labels.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    report = run()
    for name, item in report["checkpoints"].items():
        sighted = item["holdout"]["sighted"]
        print(name, sighted["samples"], round(sighted["facing_accuracy"], 3), flush=True)
