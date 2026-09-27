"""Train a side-specific coach checkpoint through its matching trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coach_v1.common.types import Side
from coach_v1.models.coach_model import CoachModelConfig
from coach_v1.training.coach_trainer import CoachTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=[side.value for side in Side])
    parser.add_argument("--stage", default="2v1", choices=("2v1", "2v2", "3v3", "5v5"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-rollouts", action="store_true")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--collision-penalty", type=float, default=0.01)
    parser.add_argument("--action-feedback", action="store_true",
                        help="experimental previous movement outcome input; requires its own checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(1)
    trainer = CoachTrainer(Side(args.side), seed=args.seed, directory=args.directory,
                           config=CoachModelConfig(action_feedback=args.action_feedback))
    if args.resume:
        trainer.resume()
    history = trainer.fit(episodes=args.episodes, stage=args.stage,
                          max_ticks=args.max_ticks,
                          save_rollouts=not args.no_rollouts,
                          collision_penalty=args.collision_penalty)
    print(json.dumps(history[-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
