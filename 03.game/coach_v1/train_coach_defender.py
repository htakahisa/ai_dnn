"""Task 13 defender-only search and retake curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from coach_v1.common.constants import COACH_CHECKPOINT_PATHS
from coach_v1.common.types import Side
from coach_v1.training.coach_environment import DEFENDER_STAGES
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_imitation import (
    fit_balanced_imitation,
    fit_imitation,
    validate_defender_actor,
)


DEFAULT_DIRECTORY = COACH_CHECKPOINT_PATHS[Side.DEFENDER.value] / "task13"
SOURCE_DIRECTORY = COACH_CHECKPOINT_PATHS[Side.DEFENDER.value] / "observation_v2"
DEFAULT_EPISODES = (8, 8, 8, 8, 10, 12, 16, 24)
DEFAULT_IMITATION_EPISODES = (8, 8, 8, 8, 10, 12, 16, 24)
STAGE_TICKS = (20, 35, 30, 45, 45, 45, 35, 120)


def train(*, directory: Path = DEFAULT_DIRECTORY, source: Path = SOURCE_DIRECTORY,
          episodes: tuple[int, ...] = DEFAULT_EPISODES,
          imitation_episodes: tuple[int, ...] = DEFAULT_IMITATION_EPISODES,
          aggregation_episodes: int = 0,
          balanced_cycles: int = 0, balanced_samples: int = 24,
          validation_seed: int = 1000, validation_episodes: int = 4,
          validation_max_ticks: int = STAGE_TICKS[-1],
          seed: int = 11, save_rollouts: bool = False) -> list[dict]:
    if (len(episodes) != len(DEFENDER_STAGES)
            or len(imitation_episodes) != len(DEFENDER_STAGES)
            or any(value < 0 for value in episodes + imitation_episodes)
            or aggregation_episodes < 0 or balanced_cycles < 0
            or balanced_samples <= 0 or validation_episodes <= 0
            or validation_max_ticks <= 0
            or not any(episodes) and not any(imitation_episodes)
            and aggregation_episodes == 0 and balanced_cycles == 0):
        raise ValueError("one nonnegative PPO and imitation target per defender stage is required")
    directory, source = Path(directory), Path(source)
    if not (directory / "latest.pt").exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("target directory is nonempty but has no actor checkpoint")
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("latest.pt", "training_latest.pt"):
            shutil.copy2(source / name, directory / name)
    trainer = CoachTrainer(Side.DEFENDER, seed=seed, directory=directory)
    trainer.resume()
    for stage, ppo_target, ticks in zip(DEFENDER_STAGES, episodes, STAGE_TICKS):
        completed = sum(item.get("stage") == stage for item in trainer.history)
        if completed < ppo_target:
            trainer.fit(episodes=ppo_target - completed, stage=stage,
                        max_ticks=ticks, save_rollouts=save_rollouts)
    # Cycle the curriculum one episode at a time.  Training every demonstration
    # for one stage in a single block caused later retake stages to erase setup
    # and long-distance movement learned by earlier stages.
    while True:
        pending = False
        for stage, target, ticks in zip(
                DEFENDER_STAGES, imitation_episodes, STAGE_TICKS):
            label = "imitation_" + stage
            completed = sum(item.get("stage") == label for item in trainer.history)
            if completed < target:
                fit_imitation(trainer, episodes=1, stage=stage, max_ticks=ticks)
                pending = True
        if not pending:
            break
    completed = sum(item.get("stage") == "aggregation_full_round"
                    for item in trainer.history)
    if completed < aggregation_episodes:
        fit_imitation(trainer, episodes=aggregation_episodes - completed,
                      stage="full_round", max_ticks=STAGE_TICKS[-1], on_policy=True)
    completed = sum(item.get("stage") == "balanced_imitation"
                    for item in trainer.history)
    for _ in range(max(0, balanced_cycles - completed)):
        fit_balanced_imitation(
            trainer, cycles=1,
            stage_ticks=dict(zip(DEFENDER_STAGES, STAGE_TICKS)),
            samples_per_bucket=balanced_samples,
        )
        validation = validate_defender_actor(
            trainer,
            seeds=tuple(range(validation_seed,
                              validation_seed + validation_episodes)),
            max_ticks=validation_max_ticks,
        )
        previous_scores = [
            float(item["validation"]["selection_score"])
            for item in trainer.history[:-1]
            if item.get("stage") == "balanced_imitation" and "validation" in item
        ]
        trainer.history[-1]["validation"] = validation
        trainer.history[-1]["validation_best"] = (
            not previous_scores
            or validation["selection_score"] > max(previous_scores)
        )
        trainer.save()
        if trainer.history[-1]["validation_best"]:
            shutil.copy2(directory / "latest.pt", directory / "best.pt")
            shutil.copy2(directory / "training_latest.pt",
                         directory / "training_best.pt")
    return trainer.history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--source", type=Path, default=SOURCE_DIRECTORY)
    parser.add_argument("--episodes-per-stage", type=int)
    parser.add_argument("--imitation-per-stage", type=int)
    parser.add_argument("--stage-target", action="append", default=[],
                        metavar="STAGE=COUNT")
    parser.add_argument("--imitation-stage-target", action="append", default=[],
                        metavar="STAGE=COUNT")
    parser.add_argument("--aggregation-episodes", type=int, default=0,
                        help="cumulative on-policy full-round demonstration target")
    parser.add_argument("--balanced-cycles", type=int, default=0,
                        help="cumulative mixed-stage imitation cycle target")
    parser.add_argument("--balanced-samples", type=int, default=24,
                        help="samples drawn per curriculum/phase bucket")
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--validation-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-rollouts", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    ppo = list((args.episodes_per_stage,) * len(DEFENDER_STAGES)
               if args.episodes_per_stage is not None else DEFAULT_EPISODES)
    imitation = list((args.imitation_per_stage,) * len(DEFENDER_STAGES)
                     if args.imitation_per_stage is not None
                     else DEFAULT_IMITATION_EPISODES)
    for values, overrides in ((ppo, args.stage_target),
                              (imitation, args.imitation_stage_target)):
        for item in overrides:
            name, separator, value = item.partition("=")
            if not separator or name not in DEFENDER_STAGES:
                parser.error("stage target must be STAGE=COUNT")
            try:
                values[DEFENDER_STAGES.index(name)] = int(value)
            except ValueError:
                parser.error("stage target count must be an integer")
    history = train(directory=args.directory, source=args.source,
                    episodes=tuple(ppo), imitation_episodes=tuple(imitation),
                    aggregation_episodes=args.aggregation_episodes,
                    balanced_cycles=args.balanced_cycles,
                    balanced_samples=args.balanced_samples,
                    validation_seed=args.validation_seed,
                    validation_episodes=args.validation_episodes,
                    seed=args.seed, save_rollouts=args.save_rollouts)
    print(json.dumps({"directory": str(args.directory), "episodes": len(history),
                      "last": history[-1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
