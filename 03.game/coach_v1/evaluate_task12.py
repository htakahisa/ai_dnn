"""Paired attacker Task 12 checks against legal random and a simple rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from coach_v1.common.types import MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.learning_coach_attacker import load_attacker_coach
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.training.coach_environment import ATTACKER_STAGES, CoachTrainingEnvironment
from coach_v1.training.attacker_teacher import teacher_actions
from coach_v1.train_coach_attacker import DEFAULT_DIRECTORY, STAGE_TICKS


_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))


def rule_actions(observation, mask) -> tuple[CoachInstruction, ...]:
    """Deliberately simple actor-only baseline; no game truth or route search."""
    grid, slots = observation.grid, observation.vector[-70:].reshape(5, 14)
    dropped = np.argwhere(grid[20] > 0)
    sites = np.argwhere(grid[2] > 0)
    planted = bool(np.any(grid[21] > 0))
    actions = []
    for slot in range(5):
        if not slots[slot, 0] or planted:
            move = MovementAction.STAY
        else:
            row = round(float(slots[slot, 4]) * 25)
            col = round(float(slots[slot, 5]) * 43)
            targets = dropped if len(dropped) else sites
            choices = [(min(abs(row + dr - int(target[0])) +
                            abs(col + dc - int(target[1])) for target in targets), index)
                       for index, (dr, dc) in enumerate(_DELTAS)
                       if mask.movement[slot, index]]
            move = tuple(MovementAction)[min(choices)[1]]
        objective = (ObjectiveAction.PLANT
                     if mask.objective[slot, tuple(ObjectiveAction).index(ObjectiveAction.PLANT)]
                     else ObjectiveAction.NONE)
        actions.append(CoachInstruction(move, objective, TacticalIntent.ADVANCE))
    return tuple(actions)


def evaluate(*, checkpoint: Path = DEFAULT_DIRECTORY / "latest.pt",
             stages: tuple[str, ...] = ATTACKER_STAGES,
             seeds: range = range(100, 104), include_teacher: bool = False) -> dict:
    if not seeds or any(stage not in ATTACKER_STAGES for stage in stages):
        raise ValueError("nonempty seeds and valid attacker stages required")
    checkpoint = Path(checkpoint)
    policy = load_attacker_coach(checkpoint)
    report = {"checkpoint": str(checkpoint),
              "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "seeds": list(seeds),
              "metric_definitions": {
                  "plant_rate": "episodes with a completed plant / episodes",
                  "round_win_rate": "attacker round wins / episodes",
                  "solo_entry_rate": "site entries without an ally within Chebyshev distance 3 / site entries",
                  "collision_rate": "ally-blocked moves / move requests",
                  "trade_distance": "mean nearest living ally Chebyshev distance per tick",
                  "multi_angle_rate": "ticks with two legal sight lines to one enemy separated by at least 30 degrees / ticks",
                  "ability_entry_rate": "site entries within 15 ticks of an allied ability / site entries",
                  "invalid_move_rate": "failed move requests / move requests",
              }, "stages": {}}
    for stage in stages:
        ticks = STAGE_TICKS[ATTACKER_STAGES.index(stage)]
        results = {}
        for mode in (("trained", "random", "rule", "teacher") if include_teacher
                     else ("trained", "random", "rule")):
            totals: dict[str, float] = {}
            for seed in seeds:
                environment = CoachTrainingEnvironment(
                    Side.ATTACKER, seed=seed, stage=stage, max_ticks=ticks,
                    observation_version=policy.encoder.version)
                state = environment.reset()
                policy.reset_round()
                rng = random.Random(seed + 1409)
                while state is not None:
                    if mode == "trained":
                        actions = policy.act(state.observation)
                    elif mode == "rule":
                        actions = rule_actions(state.observation, state.mask)
                    elif mode == "teacher":
                        actions = teacher_actions(state.observation, state.mask)
                    else:
                        actions = tuple(CoachInstruction(
                            rng.choice([action for i, action in enumerate(MovementAction)
                                        if state.mask.movement[slot, i]]),
                            rng.choice([action for i, action in enumerate(ObjectiveAction)
                                        if state.mask.objective[slot, i]]),
                            rng.choice(tuple(TacticalIntent)),
                        ) for slot in range(5))
                    transition = environment.step(actions)
                    totals["reward"] = totals.get("reward", 0.0) + transition.reward
                    totals["ticks"] = totals.get("ticks", 0.0) + 1
                    for name, value in transition.metrics.items():
                        totals[name] = totals.get(name, 0.0) + value
                    state = transition.next_state
            n = len(seeds)
            summary = {key: value / n for key, value in totals.items()}
            summary["plant_rate"] = summary.get("plant", 0.0)
            summary["round_win_rate"] = summary.get("attacker_win", 0.0)
            summary["solo_entry_rate"] = (totals.get("solo_entries", 0.0) /
                                           max(1.0, totals.get("site_entries", 0.0)))
            summary["collision_rate"] = (totals.get("blocked_ally", 0.0) /
                                         max(1.0, totals.get("move_requests", 0.0)))
            summary["trade_distance"] = (totals.get("trade_distance_sum", 0.0) /
                                         max(1.0, totals.get("trade_distance_samples", 0.0)))
            summary["multi_angle_rate"] = (totals.get("multi_angle_ticks", 0.0) /
                                           max(1.0, totals.get("ticks", 0.0)))
            summary["ability_entry_rate"] = (totals.get("ability_after_entry", 0.0) /
                                             max(1.0, totals.get("site_entries", 0.0)))
            summary["invalid_move_rate"] = (totals.get("invalid_moves", 0.0) /
                                            max(1.0, totals.get("move_requests", 0.0)))
            results[mode] = summary
        report["stages"][stage] = results
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DIRECTORY / "latest.pt")
    parser.add_argument("--stages", nargs="+", choices=ATTACKER_STAGES,
                        default=list(ATTACKER_STAGES))
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--include-teacher", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    result = evaluate(checkpoint=args.checkpoint, stages=tuple(args.stages),
                      seeds=range(args.seed, args.seed + args.episodes),
                      include_teacher=args.include_teacher)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
