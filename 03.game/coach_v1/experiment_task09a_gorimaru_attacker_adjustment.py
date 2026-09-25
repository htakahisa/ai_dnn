"""Ablate unsighted facing labels and attacker encounter sampling."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from shutil import copyfile

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.experiment_task09a_gorimaru_labels import (
    CURRENT, SCENARIOS, balanced_sample, score,
)
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer
from coach_v1.training.gorimaru_rollout import collect_gorimaru_rollout


EXPERIMENT = CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_attacker_adjustment"


def _collect_masked(start: int, stop: int):
    return tuple(row for seed in range(start, stop)
                 for side in (Side.ATTACKER, Side.DEFENDER)
                 for scenario in SCENARIOS
                 for row in collect_gorimaru_rollout(
                     side=side, seed=seed, ticks=20, encounter=scenario,
                     mask_unsighted_facing=True, actor_checkpoint=CURRENT))


def _near_priority(records):
    rng = random.Random(310)
    quotas = {
        (Side.ATTACKER, "near"): 180,
        **{(Side.ATTACKER, scenario): 35 for scenario in SCENARIOS if scenario != "near"},
        **{(Side.DEFENDER, scenario): 56 for scenario in SCENARIOS},
    }
    selected = []
    for (side, scenario), quota in quotas.items():
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


def _fit_arm(name: str, rollout, staged_train, validation, validation_records,
             holdout_records, staged_holdout) -> dict:
    directory = EXPERIMENT / name
    directory.mkdir(parents=True, exist_ok=True)
    copyfile(CURRENT, directory / "latest.pt")
    copyfile(CURRENT, directory / "best.pt")
    trainer = CharacterTrainer(
        0, seed=3000, config=CharacterModelConfig(hidden_channels=8, hidden_features=64),
        learning_rate=0.001, directory=directory,
    )
    trainer.resume()
    trainer.best_loss = float("inf")
    paths = {}
    train = [row.example for row in staged_train] + [row.example for row in rollout]
    for extra in (5, 5, 10):
        trainer.fit(train, validation, epochs=extra, batch_size=16)
        for variant in ("best", "latest"):
            path = directory / f"epoch{trainer.epoch}_{variant}.pt"
            copyfile(directory / f"{variant}.pt", path)
            paths[path.stem] = path
        print(name, "epoch", trainer.epoch, flush=True)
    return {
        "train_rollout_samples": len(rollout),
        "train_rollout_sighted": sum(row.has_sighting for row in rollout),
        "train_rollout_by_side_scenario": dict(Counter(
            f"{row.side.value}_{row.encounter}" for row in rollout)),
        "watchpoint_training_share": len(staged_train) / len(train),
        "checkpoints": {
            key: {
                "path": str(path),
                "validation": score(trainer, path, validation_records),
                "holdout": score(trainer, path, holdout_records),
                "staged_holdout": trainer.evaluate(
                    [row.example for row in staged_holdout]).to_dict(),
            }
            for key, path in paths.items()
        },
    }


def run() -> dict:
    torch.set_num_threads(1)
    records = _collect_masked(500, 512)
    extra_near = tuple(row for seed in range(512, 540)
                       for row in collect_gorimaru_rollout(
                           side=Side.ATTACKER, seed=seed, ticks=20,
                           encounter="near", mask_unsighted_facing=True,
                           actor_checkpoint=CURRENT))
    validation_records = _collect_masked(600, 605)
    holdout_records = _collect_masked(700, 710)
    staged_train = build_character_curriculum(0, seed=1000, count=1400,
                                              multi_facing=True)
    staged_validation = build_character_curriculum(0, seed=2000, count=300,
                                                   multi_facing=True)
    staged_holdout = build_character_curriculum(0, seed=3000, count=200,
                                                multi_facing=True)
    validation = ([row.example for row in staged_validation]
                  + [row.example for row in balanced_sample(
                      validation_records, per_group=15, seed=32)])
    arms = {
        "mask_only": balanced_sample(records, per_group=60, seed=31),
        "near_priority": _near_priority(records + extra_near),
    }
    result = {
        "start_checkpoint": str(CURRENT),
        "change": "no facing loss without current legal sighting; near attacker quota in second arm",
        "train_seeds": [500, 539], "validation_seeds": [600, 604],
        "holdout_seeds": [700, 709],
        "staged_train_samples": len(staged_train),
        "extra_attacker_near_collected": len(extra_near),
        "arms": {},
    }
    for name, rollout in arms.items():
        result["arms"][name] = _fit_arm(
            name, rollout, staged_train, validation, validation_records,
            holdout_records, staged_holdout,
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09a_gorimaru_attacker_adjustment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    result = run()
    for name, arm in result["arms"].items():
        print(name, arm["train_rollout_by_side_scenario"], flush=True)
        for checkpoint, item in arm["checkpoints"].items():
            metrics = item["holdout"]["by_side_scenario"]["attacker_near"]
            print(checkpoint, metrics.get("samples"),
                  metrics.get("facing_accuracy"), flush=True)
