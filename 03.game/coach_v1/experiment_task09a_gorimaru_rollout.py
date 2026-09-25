"""Fine tune Gorimaru on legally perceived 5v5 encounters only."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import torch

from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS, CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model
from coach_v1.training.gorimaru_rollout import collect_gorimaru_rollout


_OFFICIAL = CHARACTER_CHECKPOINT_PATHS["gorimaru"]
_EXPERIMENT = CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_rollout"
_CONFIG = CharacterModelConfig(hidden_channels=8, hidden_features=64)


def _collect(start: int, stop: int, baseline: Path):
    return tuple(item for seed in range(start, stop)
                 for side in (Side.ATTACKER, Side.DEFENDER)
                 for item in collect_gorimaru_rollout(
                     side=side, seed=seed, ticks=20, actor_checkpoint=baseline))


def _metrics(trainer: CharacterTrainer, path, rollout, staged):
    trainer.model = load_character_model(path, slot=0)
    def score(records):
        return trainer.evaluate([item.example for item in records]).to_dict() if records else None
    return {
        "rollout_all": score(rollout),
        "rollout_attacker": score([item for item in rollout if item.side is Side.ATTACKER]),
        "rollout_defender": score([item for item in rollout if item.side is Side.DEFENDER]),
        "rollout_sighted": score([item for item in rollout if item.has_sighting]),
        "staged": score(staged),
    }


def run() -> dict:
    torch.set_num_threads(1)
    _EXPERIMENT.mkdir(parents=True, exist_ok=True)
    baseline = _EXPERIMENT / "baseline_best.pt"
    if not baseline.exists():
        copyfile(_OFFICIAL / "best.pt", baseline)
    train_rollout = _collect(100, 140, baseline)
    validation_rollout = _collect(200, 210, baseline)
    holdout_rollout = _collect(300, 310, baseline)
    staged_train = build_character_curriculum(0, seed=1000, count=360)
    staged_validation = build_character_curriculum(0, seed=2000, count=80)
    train = [item.example for item in staged_train] + [item.example for item in train_rollout]
    validation = ([item.example for item in staged_validation]
                  + [item.example for item in validation_rollout])

    copyfile(baseline, _EXPERIMENT / "latest.pt")
    copyfile(baseline, _EXPERIMENT / "best.pt")
    trainer = CharacterTrainer(0, seed=3000, config=_CONFIG,
                               learning_rate=0.002, directory=_EXPERIMENT)
    trainer.resume()
    trainer.best_loss = float("inf")  # New validation distribution.
    paths = {"official_best": baseline}
    for extra_epochs in (20, 20, 40):
        trainer.fit(train, validation, epochs=extra_epochs, batch_size=16)
        milestone = trainer.epoch
        for variant in ("best", "latest"):
            name = f"epoch{milestone}_{variant}"
            path = _EXPERIMENT / f"{name}.pt"
            copyfile(_EXPERIMENT / f"{variant}.pt", path)
            paths[name] = path
        print("reached epoch", milestone, "validation facing",
              trainer.history[-1]["facing_accuracy"], flush=True)

    result = {
        "character": "gorimaru",
        "training": "360 staged examples plus legal 5v5 rollout observations",
        "train_rollout_examples": len(train_rollout),
        "train_rollout_sighted": sum(item.has_sighting for item in train_rollout),
        "validation_rollout_examples": len(validation_rollout),
        "validation_rollout_sighted": sum(item.has_sighting for item in validation_rollout),
        "holdout_rollout_examples": len(holdout_rollout),
        "holdout_rollout_sighted": sum(item.has_sighting for item in holdout_rollout),
        "checkpoints": {},
        "history": trainer.history,
    }
    for name, path in paths.items():
        result["checkpoints"][name] = {
            "path": str(path),
            "validation": _metrics(trainer, path, validation_rollout, staged_validation),
            "holdout": _metrics(trainer, path, holdout_rollout, staged_validation),
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09a_gorimaru_rollout_training.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    report = run()
    for name, item in report["checkpoints"].items():
        print(name, "holdout sighted facing",
              round(item["holdout"]["rollout_sighted"]["facing_accuracy"], 3),
              flush=True)
