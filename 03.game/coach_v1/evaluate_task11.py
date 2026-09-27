"""Evaluate a coach against a legal random policy on identical episode seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from coach_v1.common.checkpoint import CheckpointMetadata
from coach_v1.common.constants import COACH_CHECKPOINT_PATHS
from coach_v1.common.types import MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.learning_coach import load_coach_policy
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.training.coach_environment import CoachTrainingEnvironment, _STAY


def evaluate(side: Side, *, stage: str, seeds: range, max_ticks: int,
             checkpoint: Path | None = None, include_stay: bool = False,
             collision_penalty: float = 0.01) -> dict:
    checkpoint = Path(checkpoint) if checkpoint is not None else COACH_CHECKPOINT_PATHS[side.value] / "latest.pt"
    policy = load_coach_policy(side, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = CheckpointMetadata.from_dict(payload["metadata"])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    results = {}
    for mode in (("trained", "random", "stay") if include_stay else ("trained", "random")):
        totals = {"reward": 0.0, "new_clear_cells": 0.0,
                  "invalid_moves": 0.0, "round_win": 0.0, "ticks": 0.0}
        for seed in seeds:
            environment = CoachTrainingEnvironment(side, seed=seed, stage=stage,
                                                    max_ticks=max_ticks,
                                                    collision_penalty=collision_penalty)
            state = environment.reset()
            policy.reset_round()
            rng = random.Random(seed + 1409)
            while state is not None:
                if mode == "trained":
                    actions = policy.act(state.observation)
                elif mode == "stay":
                    actions = _STAY
                else:
                    actions = tuple(CoachInstruction(
                        rng.choice([action for i, action in enumerate(MovementAction)
                                    if state.mask.movement[slot, i]]),
                        rng.choice([action for i, action in enumerate(ObjectiveAction)
                                    if state.mask.objective[slot, i]]),
                        rng.choice(tuple(TacticalIntent)),
                    ) for slot in range(5))
                transition = environment.step(actions)
                totals["reward"] += transition.reward
                totals["ticks"] += 1
                for name, value in transition.metrics.items():
                    totals[name] = totals.get(name, 0.0) + value
                state = transition.next_state
        count = len(seeds)
        results[mode] = {key: value / count for key, value in totals.items()}
    return {"side": side.value, "stage": stage, "seeds": list(seeds),
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "training_step": metadata.training_step,
            "collision_penalty": collision_penalty, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=[side.value for side in Side], required=True)
    parser.add_argument("--stage", choices=("2v1", "2v2", "3v3", "5v5"), default="2v1")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-ticks", type=int, default=40)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--include-stay", action="store_true")
    parser.add_argument("--collision-penalty", type=float, default=0.01)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = evaluate(Side(args.side), stage=args.stage,
                      seeds=range(args.seed, args.seed + args.episodes),
                      max_ticks=args.max_ticks, checkpoint=args.checkpoint,
                      include_stay=args.include_stay,
                      collision_penalty=args.collision_penalty)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
