"""Fine-tune both coach actors after the Task 16 watch-point addition.

Source actor and training weights are loaded directly under a checked old
configuration hash; all saved candidates use the current configuration hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from coach_v1.common.constants import COACH_CHECKPOINT_PATHS, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.learning_coach import load_coach_policy
from coach_v1.models.coach_model import CoachModelConfig
from coach_v1.training.attacker_imitation import fit_imitation as fit_attacker_imitation
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_imitation import (
    fit_balanced_imitation, validate_defender_actor,
)
from coach_v1.train_coach_defender import STAGE_TICKS
from coach_v1.training.coach_environment import DEFENDER_STAGES

from coach_v1.retrain_task16_watch_points import OLD_WATCH_HASH, NEW_WATCH_HASH


def _initialize(side: Side, target: Path) -> tuple[CoachTrainer, Path]:
    source = COACH_CHECKPOINT_PATHS[side.value] / ("task12" if side is Side.ATTACKER else "task13")
    actor = torch.load(source / "latest.pt", map_location="cpu", weights_only=False)
    training = torch.load(source / "training_latest.pt", map_location="cpu", weights_only=False)
    actor_meta, training_meta = actor["metadata"], training["metadata"]
    if (actor_meta["watch_points_hash"] != OLD_WATCH_HASH
            or training_meta["watch_points_hash"] != OLD_WATCH_HASH
            or actor_meta["training_step"] != training_meta["training_step"]
            or actor_meta["target_id"] != side.value
            or training_meta["target_id"] != side.value):
        raise ValueError(f"source coach provenance mismatch: {source}")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"candidate directory is not empty: {target}")
    config = CoachModelConfig(**actor_meta["model_config"])
    trainer = CoachTrainer(side, seed=actor_meta["training_seed"],
                           directory=target, config=config)
    if trainer.encoder.watch_points_hash != NEW_WATCH_HASH:
        raise ValueError("watch-point configuration changed during retraining")
    trainer.actor.load_state_dict(actor["model_state_dict"], strict=True)
    trainer.critic.load_state_dict(training["critic_state_dict"], strict=True)
    trainer.optimizer.load_state_dict(training["optimizer_state_dict"])
    trainer.training_step = actor_meta["training_step"]
    trainer.episode = int(training["episode"])
    trainer.history = list(training["history"])
    return trainer, source


def run(*, directory: Path) -> dict:
    torch.set_num_threads(1)
    report = {"old_watch_hash": OLD_WATCH_HASH, "new_watch_hash": NEW_WATCH_HASH,
              "coaches": {}}
    for side in (Side.ATTACKER, Side.DEFENDER):
        trainer, source = _initialize(side, directory / "coach" / side.value)
        old_step = trainer.training_step
        if side is Side.ATTACKER:
            fit_attacker_imitation(trainer, episodes=4, stage="entry", max_ticks=40)
            fit_attacker_imitation(trainer, episodes=8, stage="full_round",
                                   max_ticks=160, on_policy=True)
            validation = None
        else:
            validation_before = validate_defender_actor(
                trainer, seeds=tuple(range(3600, 3604)), max_ticks=120)
            fit_balanced_imitation(
                trainer, cycles=2,
                stage_ticks=dict(zip(DEFENDER_STAGES, STAGE_TICKS)),
                samples_per_bucket=16,
            )
            validation_after = validate_defender_actor(
                trainer, seeds=tuple(range(3600, 3604)), max_ticks=120)
            validation = {"before": validation_before, "after": validation_after}
        candidate = trainer.directory / "latest.pt"
        load_coach_policy(side, candidate)
        report["coaches"][side.value] = {
            "source": str(source / "latest.pt"),
            "source_sha256": hashlib.sha256((source / "latest.pt").read_bytes()).hexdigest(),
            "candidate": str(candidate), "old_training_step": old_step,
            "new_training_step": trainer.training_step,
            "new_episodes": trainer.episode - int(torch.load(
                source / "training_latest.pt", map_location="cpu",
                weights_only=False)["episode"]),
            "validation": validation,
        }
        print(json.dumps({"side": side.value,
                          "steps_added": trainer.training_step - old_step,
                          "validation": validation}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parent / "checkpoints" /
                        "experiments" / "task16_watch_added")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task16_watch_coach_retrain.json")
    args = parser.parse_args()
    report = run(directory=args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
