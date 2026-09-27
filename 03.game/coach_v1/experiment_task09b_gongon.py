"""Train and compare only Gongon's facing model on disjoint legal observations."""

from __future__ import annotations

from collections import Counter
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
from coach_v1.training.gongon_rollout import collect_gongon_rollout


SCENARIOS = ("near", "west", "east", "crossfire", "natural")
EXPERIMENT = CHECKPOINTS_DIR / "experiments" / "task09b_gongon"
BASELINE = EXPERIMENT / "baseline_best.pt"


def collect(start: int, stop: int):
    return tuple(row for seed in range(start, stop)
                 for side in (Side.ATTACKER, Side.DEFENDER)
                 for scenario in SCENARIOS
                 for row in collect_gongon_rollout(
                     side=side, seed=seed, encounter=scenario, actor_checkpoint=BASELINE))


def balanced_sample(records, *, per_group: int, seed: int):
    rng = random.Random(seed)
    selected = []
    for side in (Side.ATTACKER, Side.DEFENDER):
        for scenario in SCENARIOS:
            group = [row for row in records if row.side is side and row.encounter == scenario]
            selected.extend(rng.sample(group, min(per_group, len(group))))
    return tuple(selected)


def score(trainer: CharacterTrainer, path: Path, records) -> dict:
    trainer.model = load_character_model(path, slot=1)

    def subset(rows):
        return trainer.evaluate([row.example for row in rows]).to_dict() if rows else {"samples": 0}

    return {
        "sighted": subset([row for row in records if row.has_sighting]),
        "by_side_scenario": {
            f"{side.value}_{scenario}": subset([
                row for row in records if row.has_sighting
                and row.side is side and row.encounter == scenario
            ])
            for side in (Side.ATTACKER, Side.DEFENDER) for scenario in SCENARIOS
        },
    }


def run() -> dict:
    torch.set_num_threads(1)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    training_records = collect(500, 512)
    validation_records = collect(600, 605)
    holdout_records = collect(700, 710)
    rollout_train = balanced_sample(training_records, per_group=60, seed=31)
    rollout_validation = balanced_sample(validation_records, per_group=15, seed=32)
    staged_train = build_character_curriculum(1, seed=1001, count=1400, multi_facing=True)
    staged_validation = build_character_curriculum(1, seed=2001, count=300, multi_facing=True)
    staged_holdout = build_character_curriculum(1, seed=3001, count=200, multi_facing=True)
    train = [row.example for row in staged_train] + [row.example for row in rollout_train]
    validation = ([row.example for row in staged_validation]
                  + [row.example for row in rollout_validation])
    copyfile(BASELINE, EXPERIMENT / "latest.pt")
    copyfile(BASELINE, EXPERIMENT / "best.pt")
    trainer = CharacterTrainer(
        1, seed=3001, config=CharacterModelConfig(hidden_channels=8, hidden_features=64),
        directory=EXPERIMENT,
    )
    trainer.resume()
    trainer.best_loss = float("inf")
    paths = {"baseline": BASELINE}
    for extra in (5, 5, 10):
        trainer.fit(train, validation, epochs=extra, batch_size=16)
        for variant in ("best", "latest"):
            path = EXPERIMENT / f"epoch{trainer.epoch}_{variant}.pt"
            copyfile(EXPERIMENT / f"{variant}.pt", path)
            paths[path.stem] = path
        print("trained epoch", trainer.epoch, flush=True)
    result = {
        "slot": 1,
        "ability": "HUNT has no active action; use is masked",
        "train_seeds": [500, 511], "validation_seeds": [600, 604],
        "holdout_seeds": [700, 709],
        "teacher": "current legal team sightings, within 45 degrees; no sighting gives zero facing loss",
        "training_samples": len(train),
        "watchpoint_training_share": len(staged_train) / len(train),
        "rollout_training_groups": dict(Counter(
            f"{row.side.value}_{row.encounter}" for row in rollout_train)),
        "validation_samples": len(validation),
        "holdout_sighted": sum(row.has_sighting for row in holdout_records),
        "checkpoints": {},
    }
    for name, path in paths.items():
        result["checkpoints"][name] = {
            "path": str(path),
            "validation": score(trainer, path, validation_records),
            "holdout": score(trainer, path, holdout_records),
            "staged_holdout": trainer.evaluate([row.example for row in staged_holdout]).to_dict(),
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09b_gongon_training.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    for name, item in report["checkpoints"].items():
        sighted = item["holdout"]["sighted"]
        print(name, sighted["samples"], sighted["facing_accuracy"], flush=True)
