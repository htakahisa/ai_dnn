"""Test extra attacker close-contact examples without changing the 70% watch-point share."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import random
from shutil import copyfile
import sys

import torch

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.experiment_task09b_gongon import (
    BASELINE, EXPERIMENT, SCENARIOS, collect, score,
)
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer
from coach_v1.training.gongon_rollout import collect_gongon_rollout


def _select(records):
    rng = random.Random(310)
    selected = []
    for side in (Side.ATTACKER, Side.DEFENDER):
        for scenario in SCENARIOS:
            quota = (180 if side is Side.ATTACKER and scenario == "near"
                     else 30 if side is Side.ATTACKER else 60)
            group = [row for row in records if row.side is side and row.encounter == scenario]
            if side is Side.ATTACKER and scenario == "near":
                sighted = [row for row in group if row.has_sighting]
                unseen = [row for row in group if not row.has_sighting]
                chosen = rng.sample(sighted, min(quota, len(sighted)))
                chosen += rng.sample(unseen, min(quota - len(chosen), len(unseen)))
            else:
                chosen = rng.sample(group, min(quota, len(group)))
            selected.extend(chosen)
    return tuple(selected)


def run(*, fixed: bool = False) -> dict:
    torch.set_num_threads(1)
    records = collect(500, 512)
    extra = tuple(row for seed in range(500 if fixed else 512, 540)
                  for row in collect_gongon_rollout(
                      side=Side.ATTACKER, seed=seed,
                      encounter="near_fixed" if fixed else "near",
                      actor_checkpoint=BASELINE))
    if fixed:
        records = tuple(row for row in records
                        if not (row.side is Side.ATTACKER and row.encounter == "near"))
        extra = tuple(replace(row, encounter="near") for row in extra)
    rollout = _select(records + extra)
    if len(rollout) != 600:
        raise ValueError(f"expected 600 rollout examples, got {len(rollout)}")
    validation_records = collect(600, 605)
    holdout_records = collect(700, 710)
    staged_train = build_character_curriculum(1, seed=1001, count=1400,
                                              multi_facing=True)
    staged_validation = build_character_curriculum(1, seed=2001, count=300,
                                                   multi_facing=True)
    staged_holdout = build_character_curriculum(1, seed=3001, count=200,
                                                multi_facing=True)
    rng = random.Random(32)
    validation_rollout = []
    for side in (Side.ATTACKER, Side.DEFENDER):
        for scenario in SCENARIOS:
            group = [row for row in validation_records
                     if row.side is side and row.encounter == scenario]
            validation_rollout.extend(rng.sample(group, min(15, len(group))))
    train = [row.example for row in staged_train] + [row.example for row in rollout]
    validation = ([row.example for row in staged_validation]
                  + [row.example for row in validation_rollout])
    directory = EXPERIMENT / ("fixed_near_priority" if fixed else "near_priority")
    directory.mkdir(parents=True, exist_ok=True)
    copyfile(BASELINE, directory / "best.pt")
    copyfile(BASELINE, directory / "latest.pt")
    trainer = CharacterTrainer(
        1, seed=3001, config=CharacterModelConfig(hidden_channels=8, hidden_features=64),
        directory=directory,
    )
    trainer.resume()
    trainer.best_loss = float("inf")
    paths = {"baseline": BASELINE}
    for extra_epochs in (5, 5, 10):
        trainer.fit(train, validation, epochs=extra_epochs, batch_size=16)
        for variant in ("best", "latest"):
            path = directory / f"epoch{trainer.epoch}_{variant}.pt"
            copyfile(directory / f"{variant}.pt", path)
            paths[path.stem] = path
        print("trained epoch", trainer.epoch, flush=True)
    result = {
        "train_seeds": [500, 539], "validation_seeds": [600, 604],
        "holdout_seeds": [700, 709],
        "attacker_near_staging": "fixed" if fixed else "random nearby",
        "training_samples": len(train),
        "watchpoint_training_share": len(staged_train) / len(train),
        "rollout_training_groups": dict(Counter(
            f"{row.side.value}_{row.encounter}" for row in rollout)),
        "attacker_near_sighted": sum(row.has_sighting for row in rollout
                                     if row.side is Side.ATTACKER and row.encounter == "near"),
        "checkpoints": {},
    }
    for name, path in paths.items():
        result["checkpoints"][name] = {
            "path": str(path),
            "validation": score(trainer, path, validation_records),
            "holdout": score(trainer, path, holdout_records),
            "staged_holdout": trainer.evaluate(
                [row.example for row in staged_holdout]).to_dict(),
        }
    report_name = ("task09b_gongon_fixed_near_training.json" if fixed else
                   "task09b_gongon_near_training.json")
    (REPORTS_DIR / report_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run(fixed="--fixed" in sys.argv[1:])
    for name, item in report["checkpoints"].items():
        print(name, item["holdout"]["sighted"]["facing_accuracy"], flush=True)
