"""Retrain character actors for the two adopted Task 16 watch points.

Keep the former checkpoints intact. The coach actors are retrained separately
after these character candidates pass evaluation and are promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS, FIXED_ROSTER, REPORTS_DIR
from coach_v1.common.watch_point_versions import (
    LEGACY_WATCH_POINTS_HASH as OLD_WATCH_HASH,
    TASK16_WATCH_POINTS_HASH as NEW_WATCH_HASH,
)
from coach_v1.models import CharacterModelConfig
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


def run(*, directory: Path, train_count: int = 720, validation_count: int = 160,
        holdout_count: int = 160, epochs: int = 8) -> dict:
    torch.set_num_threads(1)
    report = {"old_watch_hash": OLD_WATCH_HASH, "new_watch_hash": NEW_WATCH_HASH,
              "train_count": train_count, "validation_count": validation_count,
              "holdout_count": holdout_count, "epochs": epochs, "characters": {}}
    for slot, roster in enumerate(FIXED_ROSTER):
        source = CHARACTER_CHECKPOINT_PATHS[roster.checkpoint_id] / "best.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata["watch_points_hash"] != OLD_WATCH_HASH:
            raise ValueError(f"unexpected source watch-point hash: {source}")
        config = CharacterModelConfig(**metadata["model_config"])
        target = directory / "characters" / roster.checkpoint_id
        if (target / "latest.pt").exists():
            raise FileExistsError(f"candidate already exists: {target}")
        train = build_character_curriculum(slot, seed=2600 + slot, count=train_count)
        validation = build_character_curriculum(slot, seed=3600 + slot,
                                                count=validation_count)
        holdout = build_character_curriculum(slot, seed=4600 + slot, count=holdout_count)
        trainer = CharacterTrainer(slot, seed=5600 + slot, config=config,
                                   learning_rate=0.0001, directory=target)
        if trainer.encoder.watch_points_hash != NEW_WATCH_HASH:
            raise ValueError("watch-point configuration changed during retraining")
        trainer.model.load_state_dict(payload["model_state_dict"], strict=True)
        baseline = trainer.evaluate([item.example for item in holdout]).to_dict()
        trainer.fit([item.example for item in train],
                    [item.example for item in validation], epochs=epochs)
        trainer.model = load_character_model(target / "best.pt", slot=slot)
        after = trainer.evaluate([item.example for item in holdout]).to_dict()
        report["characters"][roster.checkpoint_id] = {
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "candidate": str(target / "best.pt"),
            "baseline_holdout": baseline, "candidate_holdout": after,
            "best_epoch": torch.load(target / "best.pt", map_location="cpu",
                                     weights_only=False)["epoch"],
        }
        print(json.dumps({"character": roster.checkpoint_id,
                          "baseline": baseline, "candidate": after}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parent / "checkpoints" /
                        "experiments" / "task16_watch_added")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task16_watch_character_retrain.json")
    parser.add_argument("--train-count", type=int, default=720)
    parser.add_argument("--validation-count", type=int, default=160)
    parser.add_argument("--holdout-count", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    report = run(directory=args.directory, train_count=args.train_count,
                 validation_count=args.validation_count,
                 holdout_count=args.holdout_count, epochs=args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
