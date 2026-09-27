"""Compare Kurimaru facing-only training with baseline FLASH distillation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from shutil import copyfile

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.learning_character_kurimaru import KurimaruPolicy
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer
from coach_v1.experiment_task09e_kurimaru import BASELINE, balanced_sample, collect, score


EXPERIMENT = CHECKPOINTS_DIR / "experiments" / "task09e_kurimaru" / "facing_only"


def run(*, distill_ability: bool = False) -> dict:
    torch.set_num_threads(1)
    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)
    experiment = (EXPERIMENT.parent / "ability_distill") if distill_ability else EXPERIMENT
    experiment.mkdir(parents=True, exist_ok=True)
    training_records = collect(500, 512)
    validation_records = collect(600, 605)
    holdout_records = collect(700, 710)
    rollout_train = balanced_sample(training_records, per_group=60, seed=31)
    rollout_validation = balanced_sample(validation_records, per_group=15, seed=32)
    staged_train = build_character_curriculum(4, seed=1004, count=1400, multi_facing=True)
    staged_validation = build_character_curriculum(4, seed=2004, count=300, multi_facing=True)
    staged_holdout = build_character_curriculum(4, seed=3004, count=200, multi_facing=True)
    train = [row.example for row in staged_train] + [row.example for row in rollout_train]
    validation = ([row.example for row in staged_validation]
                  + [row.example for row in rollout_validation])
    if len(rollout_train) != 600 or len(train) != 2000:
        raise ValueError("expected 1400 watch-point examples and 600 rollout examples")
    if distill_ability:
        baseline_actor = KurimaruPolicy(BASELINE)
        def preserve_ability(examples):
            return [replace(example, use_ability=action.use_ability, target=action.target)
                    for example in examples
                    for action in (baseline_actor.act(example.observation),)]
        train = preserve_ability(train)
        validation = preserve_ability(validation)
    copyfile(BASELINE, experiment / "latest.pt")
    copyfile(BASELINE, experiment / "best.pt")
    trainer = CharacterTrainer(
        4, seed=3004, config=CharacterModelConfig(hidden_channels=8, hidden_features=64),
        learning_rate=2e-4 if distill_ability else 1e-3,
        directory=experiment,
    )
    trainer.resume()
    if distill_ability:
        for group in trainer.optimizer.param_groups:
            group["lr"] = 2e-4
    else:
        for name, parameter in trainer.model.named_parameters():
            parameter.requires_grad_(name.startswith("facing_head."))
    trainer.best_loss = float("inf")
    paths = {"baseline": BASELINE}
    for extra in (5, 5, 10):
        trainer.fit(train, validation, epochs=extra, batch_size=16)
        for variant in ("best", "latest"):
            path = experiment / f"epoch{trainer.epoch}_{variant}.pt"
            copyfile(experiment / f"{variant}.pt", path)
            paths[path.stem] = path
        print("trained epoch", trainer.epoch,
              "with ability distillation" if distill_ability else "with facing head only",
              flush=True)
    result = {
        "slot": 4,
        "method": ("old policy ability labels, learning rate 2e-4"
                   if distill_ability else "all model parameters except facing_head"),
        "training_samples": len(train),
        "watchpoint_training_share": len(staged_train) / len(train),
        "rollout_training_groups": dict(Counter(
            f"{row.side.value}_{row.encounter}" for row in rollout_train)),
        "holdout_sighted": sum(row.has_sighting for row in holdout_records),
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
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_name = ("task09e_kurimaru_ability_distill_training.json" if distill_ability
                   else "task09e_kurimaru_facing_only_training.json")
    (REPORTS_DIR / report_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill-ability", action="store_true")
    report = run(distill_ability=parser.parse_args().distill_ability)
    for name, item in report["checkpoints"].items():
        print(name, item["holdout"]["sighted"], flush=True)
