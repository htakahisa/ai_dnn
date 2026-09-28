"""Task 12 attacker-only curriculum, continuing the v2 coach checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from coach_v1.common.constants import COACH_CHECKPOINT_PATHS
from coach_v1.common.types import Side
from coach_v1.training.coach_environment import ATTACKER_STAGES
from coach_v1.training.attacker_imitation import fit_imitation
from coach_v1.training.coach_trainer import CoachTrainer


DEFAULT_DIRECTORY = COACH_CHECKPOINT_PATHS[Side.ATTACKER.value] / "task12"
SOURCE_DIRECTORY = COACH_CHECKPOINT_PATHS[Side.ATTACKER.value] / "observation_v2"
DEFAULT_EPISODES = (12, 12, 12, 12, 12, 12, 16, 16, 24)
STAGE_TICKS = (80, 40, 40, 35, 60, 40, 18, 55, 160)


def train(*, directory: Path = DEFAULT_DIRECTORY,
          source: Path = SOURCE_DIRECTORY,
          episodes: tuple[int, ...] = DEFAULT_EPISODES,
          seed: int = 11, save_rollouts: bool = False,
          imitation_episodes: int = 0,
          aggregation_episodes: int = 0,
          retrieve_imitation_episodes: int = 0,
          retrieve_aggregation_episodes: int = 0) -> list[dict]:
    if len(episodes) != len(ATTACKER_STAGES) or any(n <= 0 for n in episodes):
        raise ValueError("one positive episode target per attacker stage is required")
    if min(imitation_episodes, aggregation_episodes,
           retrieve_imitation_episodes, retrieve_aggregation_episodes) < 0:
        raise ValueError("demonstration episode targets cannot be negative")
    directory = Path(directory)
    source = Path(source)
    if not (directory / "latest.pt").exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("target directory is nonempty but has no actor checkpoint")
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("latest.pt", "training_latest.pt"):
            shutil.copy2(source / name, directory / name)
    trainer = CoachTrainer(Side.ATTACKER, seed=seed, directory=directory)
    trainer.resume()
    for stage, target, ticks in zip(ATTACKER_STAGES, episodes, STAGE_TICKS):
        completed = sum(item.get("stage") == stage for item in trainer.history)
        if completed < target:
            trainer.fit(episodes=target - completed, stage=stage, max_ticks=ticks,
                        save_rollouts=save_rollouts)
    completed = sum(item.get("stage") == "imitation_full_round" for item in trainer.history)
    if completed < imitation_episodes:
        fit_imitation(trainer, episodes=imitation_episodes - completed)
    completed = sum(item.get("stage") == "aggregation_full_round" for item in trainer.history)
    if completed < aggregation_episodes:
        fit_imitation(trainer, episodes=aggregation_episodes - completed, on_policy=True)
    completed = sum(item.get("stage") == "imitation_retrieve" for item in trainer.history)
    if completed < retrieve_imitation_episodes:
        fit_imitation(trainer, episodes=retrieve_imitation_episodes - completed,
                      stage="retrieve", max_ticks=40)
    completed = sum(item.get("stage") == "aggregation_retrieve" for item in trainer.history)
    if completed < retrieve_aggregation_episodes:
        fit_imitation(trainer, episodes=retrieve_aggregation_episodes - completed,
                      stage="retrieve", max_ticks=40, on_policy=True)
    return trainer.history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--source", type=Path, default=SOURCE_DIRECTORY)
    parser.add_argument("--episodes-per-stage", type=int, default=None)
    parser.add_argument("--stage-target", action="append", default=[],
                        metavar="STAGE=COUNT",
                        help="override one cumulative stage episode target")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-rollouts", action="store_true")
    parser.add_argument("--imitation-episodes", type=int, default=0)
    parser.add_argument("--aggregation-episodes", type=int, default=0)
    parser.add_argument("--retrieve-imitation-episodes", type=int, default=0)
    parser.add_argument("--retrieve-aggregation-episodes", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    targets = ((args.episodes_per_stage,) * len(ATTACKER_STAGES)
               if args.episodes_per_stage is not None else DEFAULT_EPISODES)
    targets = list(targets)
    for item in args.stage_target:
        name, separator, value = item.partition("=")
        if not separator or name not in ATTACKER_STAGES:
            parser.error("--stage-target must be STAGE=COUNT")
        try:
            targets[ATTACKER_STAGES.index(name)] = int(value)
        except ValueError:
            parser.error("--stage-target count must be an integer")
    history = train(directory=args.directory, source=args.source,
                    episodes=tuple(targets), seed=args.seed, save_rollouts=args.save_rollouts,
                    imitation_episodes=args.imitation_episodes,
                    aggregation_episodes=args.aggregation_episodes,
                    retrieve_imitation_episodes=args.retrieve_imitation_episodes,
                    retrieve_aggregation_episodes=args.retrieve_aggregation_episodes)
    print(json.dumps({"directory": str(args.directory), "episodes": len(history),
                      "last": history[-1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
