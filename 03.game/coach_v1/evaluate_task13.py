"""Evaluate the Task 13 defender policy against legal random and simple rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from coach_v1.common.types import MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.learning_coach_defender import load_defender_coach
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import COACH_GRID_CHANNELS
from coach_v1.train_coach_defender import DEFAULT_DIRECTORY, STAGE_TICKS
from coach_v1.training.coach_environment import DEFENDER_STAGES, CoachTrainingEnvironment


_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))


def rule_actions(observation, mask) -> tuple[CoachInstruction, ...]:
    """A simple actor-only baseline with no route search or hidden truth."""
    grid, slots = observation.grid, observation.vector[-70:].reshape(5, 14)
    planted = np.argwhere(grid[COACH_GRID_CHANNELS.index("spike_planted")] > 0)
    sightings = np.argwhere(grid[COACH_GRID_CHANNELS.index("current_enemy_sighting")] > 0)
    target = (tuple(map(int, planted[0])) if len(planted)
              else tuple(map(int, sightings[0])) if len(sightings) else None)
    actions = []
    for slot in range(5):
        objective = (ObjectiveAction.DEFUSE
                     if mask.objective[slot, tuple(ObjectiveAction).index(ObjectiveAction.DEFUSE)]
                     else ObjectiveAction.NONE)
        move = MovementAction.STAY
        if slots[slot, 0] and target is not None and objective is ObjectiveAction.NONE:
            row = round(float(slots[slot, 4]) * 25)
            column = round(float(slots[slot, 5]) * 43)
            choices = [(abs(row + dr - target[0]) + abs(column + dc - target[1]), index)
                       for index, (dr, dc) in enumerate(_DELTAS)
                       if mask.movement[slot, index]]
            move = tuple(MovementAction)[min(choices)[1]]
        intent = (TacticalIntent.SUPPORT_ENTRY if len(planted)
                  else TacticalIntent.CLEAR_AREA)
        actions.append(CoachInstruction(move, objective, intent))
    return tuple(actions)


def evaluate(*, checkpoint: Path = DEFAULT_DIRECTORY / "latest.pt",
             stages: tuple[str, ...] = DEFENDER_STAGES,
             seeds: range = range(200, 204)) -> dict:
    if not seeds or any(stage not in DEFENDER_STAGES for stage in stages):
        raise ValueError("nonempty seeds and valid defender stages required")
    checkpoint = Path(checkpoint)
    policy = load_defender_coach(checkpoint)
    report = {
        "checkpoint": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seeds": list(seeds),
        "metric_definitions": {
            "plant_prevention_rate": "defender wins before a plant / episodes",
            "retake_success_rate": "completed defuses / planted episode starts",
            "solo_retake_rate": "planted ticks with exactly one defender near site / planted ticks",
            "group_up_rate": "planted ticks with at least two defenders near site / planted ticks",
            "defuse_guard_count": "mean nearby escorts while a defender is defusing",
            "overrotation_rate": "no-sighting ticks with four or more move commands / no-sighting ticks",
            "unconfirmed_response_rate": "no-sighting ticks that clear new cells / no-sighting ticks",
            "invalid_move_rate": "failed move requests / move requests",
        },
        "stages": {},
    }
    for stage in stages:
        ticks = STAGE_TICKS[DEFENDER_STAGES.index(stage)]
        modes = {}
        for mode in ("trained", "random", "rule"):
            totals: dict[str, float] = {}
            for seed in seeds:
                environment = CoachTrainingEnvironment(
                    Side.DEFENDER, seed=seed, stage=stage, max_ticks=ticks,
                    observation_version=policy.encoder.version,
                )
                state = environment.reset()
                policy.reset_round()
                rng = random.Random(seed + 1513)
                while state is not None:
                    if mode == "trained":
                        actions = policy.act(state.observation)
                    elif mode == "rule":
                        actions = rule_actions(state.observation, state.mask)
                    else:
                        actions = tuple(CoachInstruction(
                            rng.choice([action for index, action in enumerate(MovementAction)
                                        if state.mask.movement[slot, index]]),
                            rng.choice([action for index, action in enumerate(ObjectiveAction)
                                        if state.mask.objective[slot, index]]),
                            rng.choice(tuple(TacticalIntent)),
                        ) for slot in range(5))
                    transition = environment.step(actions)
                    totals["reward"] = totals.get("reward", 0.0) + transition.reward
                    totals["ticks"] = totals.get("ticks", 0.0) + 1
                    for name, value in transition.metrics.items():
                        totals[name] = totals.get(name, 0.0) + value
                    state = transition.next_state
            count = len(seeds)
            summary = {name: value / count for name, value in totals.items()}
            summary["plant_prevention_rate"] = totals.get("plant_prevention_win", 0.0) / count
            summary["retake_success_rate"] = (totals.get("retake_success", 0.0) /
                                                max(1.0, totals.get("retake_opportunity", 0.0)))
            summary["solo_retake_rate"] = (totals.get("solo_retake_ticks", 0.0) /
                                             max(1.0, totals.get("planted_ticks", 0.0)))
            summary["group_up_rate"] = (totals.get("group_up_ticks", 0.0) /
                                         max(1.0, totals.get("planted_ticks", 0.0)))
            summary["defuse_guard_count"] = (totals.get("defuse_guard_count_sum", 0.0) /
                                               max(1.0, totals.get("defuse_guard_samples", 0.0)))
            summary["overrotation_rate"] = (totals.get("overrotation_ticks", 0.0) /
                                              max(1.0, totals.get("no_sighting_ticks", 0.0)))
            summary["unconfirmed_response_rate"] = (
                totals.get("unconfirmed_response_ticks", 0.0) /
                max(1.0, totals.get("no_sighting_ticks", 0.0))
            )
            summary["invalid_move_rate"] = (totals.get("invalid_moves", 0.0) /
                                              max(1.0, totals.get("move_requests", 0.0)))
            modes[mode] = summary
        report["stages"][stage] = modes
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DIRECTORY / "latest.pt")
    parser.add_argument("--stages", nargs="+", choices=DEFENDER_STAGES,
                        default=list(DEFENDER_STAGES))
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    result = evaluate(checkpoint=args.checkpoint, stages=tuple(args.stages),
                      seeds=range(args.seed, args.seed + args.episodes))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
