"""Continue only Gorimaru training and audit whether 80 epochs was a plateau."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import torch

from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS, CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import CurriculumExample, build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


_SLOT = 0
_SEED = 3000
_CONFIG = CharacterModelConfig(hidden_channels=8, hidden_features=64)
_EXPERIMENT_DIR = CHECKPOINTS_DIR / "experiments" / "task09a_gorimaru_epochs"
_OFFICIAL_DIR = CHARACTER_CHECKPOINT_PATHS["gorimaru"]


def _evaluate(trainer: CharacterTrainer, path: Path,
              records: tuple[CurriculumExample, ...]) -> dict:
    trainer.model = load_character_model(path, slot=_SLOT)
    all_examples = [record.example for record in records]
    def score(subset: list[CurriculumExample]):
        return trainer.evaluate([record.example for record in subset]).to_dict() if subset else None
    return {
        "all": trainer.evaluate(all_examples).to_dict(),
        "attacker": score([record for record in records if record.side.value == "attacker"]),
        "defender": score([record for record in records if record.side.value == "defender"]),
        "sighted": score([record for record in records if record.sightings]),
        "unseen": score([record for record in records if not record.sightings]),
        "point": score([record for record in records if record.category == "point"]),
        "jitter": score([record for record in records if record.category == "jitter"]),
        "random": score([record for record in records if record.category == "random"]),
    }


def run() -> dict:
    torch.set_num_threads(1)
    training = build_character_curriculum(_SLOT, seed=1000, count=360)
    validation = build_character_curriculum(_SLOT, seed=2000, count=80)
    holdout = (build_character_curriculum(_SLOT, seed=4000, count=160)
               + build_character_curriculum(_SLOT, seed=5000, count=160))
    _EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_latest = _EXPERIMENT_DIR / "baseline_epoch80.pt"
    baseline_best = _EXPERIMENT_DIR / "baseline_best.pt"
    if not baseline_latest.exists():
        copyfile(_OFFICIAL_DIR / "latest.pt", baseline_latest)
    if not baseline_best.exists():
        copyfile(_OFFICIAL_DIR / "best.pt", baseline_best)
    copyfile(baseline_latest, _EXPERIMENT_DIR / "latest.pt")
    copyfile(baseline_best, _EXPERIMENT_DIR / "best.pt")
    trainer = CharacterTrainer(_SLOT, seed=_SEED, config=_CONFIG,
                               learning_rate=0.002, directory=_EXPERIMENT_DIR)
    trainer.resume()
    if trainer.epoch != 80:
        raise ValueError("Gorimaru official latest checkpoint is not epoch 80")
    checkpoint_paths = {
        "official_best": baseline_best,
        "official_epoch80": baseline_latest,
    }
    for milestone in (160, 320):
        trainer.fit([record.example for record in training],
                    [record.example for record in validation],
                    epochs=milestone - trainer.epoch, batch_size=16)
        for variant in ("best", "latest"):
            name = f"epoch{milestone}_{variant}"
            path = _EXPERIMENT_DIR / f"{name}.pt"
            copyfile(_EXPERIMENT_DIR / f"{variant}.pt", path)
            checkpoint_paths[name] = path
        print("reached epoch", milestone,
              "validation facing", trainer.history[-1]["facing_accuracy"], flush=True)
    report = {
        "character": "gorimaru",
        "experiment": "fixed-360-examples-continue-epochs-80-to-320",
        "training_examples": len(training),
        "validation_examples": len(validation),
        "independent_holdout_examples": len(holdout),
        "checkpoints": {},
        "history": trainer.history,
    }
    for name, path in checkpoint_paths.items():
        report["checkpoints"][name] = {
            "path": str(path),
            "validation": _evaluate(trainer, path, validation),
            "holdout": _evaluate(trainer, path, holdout),
        }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09a_gorimaru_epochs.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


def compare_real_game(*, checkpoint_paths: dict[str, Path] | None = None,
                      report_name: str = "task09a_gorimaru_real_game.json") -> dict:
    """Paired near encounters; only Gorimaru's checkpoint differs."""
    from coach_v1.common.types import Side
    from coach_v1.evaluate_task10_rollout import evaluate

    paths = checkpoint_paths or {
        "official_best": _EXPERIMENT_DIR / "baseline_best.pt",
        "official_epoch80": _EXPERIMENT_DIR / "baseline_epoch80.pt",
        "epoch160_latest": _EXPERIMENT_DIR / "epoch160_latest.pt",
        "epoch320_latest": _EXPERIMENT_DIR / "epoch320_latest.pt",
    }
    result = {"encounter": "near", "coach": "fixed STAY/HOLD dummy",
              "seeds": list(range(10)), "ticks_per_seed": 20, "checkpoints": {}}
    for name, path in paths.items():
        by_side = {}
        for side in (Side.ATTACKER, Side.DEFENDER):
            runs = [evaluate(side, ticks=20, seed=seed, near=True,
                             checkpoint_overrides={0: path}) for seed in range(10)]
            aligned = sum(run["per_slot"]["0"]["unforced_aligned_45"] for run in runs)
            opportunities = sum(run["per_slot"]["0"]["unforced_sighting_actions"] for run in runs)
            by_side[side.value] = {
                "aligned_45": aligned,
                "sighting_actions": opportunities,
                "rate": aligned / opportunities if opportunities else None,
                "runs": runs,
            }
            print(name, side.value, aligned, opportunities, flush=True)
        result["checkpoints"][name] = {"path": str(path), "sides": by_side}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / report_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    result = run()
    for name, item in result["checkpoints"].items():
        print(name,
              "validation", round(item["validation"]["all"]["facing_accuracy"], 3),
              "holdout", round(item["holdout"]["all"]["facing_accuracy"], 3),
              flush=True)
