"""Trace coach choices against real movement for one Task 11 curriculum round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coach_v1.common.constants import COACH_CHECKPOINT_PATHS, FIXED_ROSTER
from coach_v1.common.types import MovementAction, Side
from coach_v1.learning_coach import load_coach_policy
from coach_v1.models.coach_model import action_feedback
from coach_v1.training.coach_environment import CoachTrainingEnvironment


def trace(side: Side, stage: str, checkpoint: Path, *, seed: int,
          max_ticks: int) -> dict:
    environment = CoachTrainingEnvironment(side, seed=seed, stage=stage,
                                           max_ticks=max_ticks)
    policy = load_coach_policy(side, checkpoint)
    state = environment.reset()
    policy.reset_round()
    ticks = []
    while state is not None:
        alive = [index for index in range(5)
                 if state.observation.vector[-70:].reshape(5, 14)[index, 0]]
        positions = [tuple(next(c.pos for c in environment.game.chars
                                if c.team == ("A" if side is Side.ATTACKER else "D")
                                and c.name == FIXED_ROSTER[index].character_name))
                     for index in alive]
        with torch.no_grad():
            grid = torch.as_tensor(state.observation.grid.copy())[None]
            vector = torch.as_tensor(state.observation.vector.copy())[None]
            movement_mask = torch.as_tensor(state.mask.movement.copy())
            feedback = (torch.as_tensor(action_feedback(
                state.observation, policy._previous_observation,
                policy._previous_actions))[None]
                if policy.model.config.action_feedback else None)
            current_logits, _, _, _ = policy.model(grid, vector, policy.hidden,
                                                    feedback=feedback)
            fresh_logits, _, _, _ = policy.model(grid, vector, None,
                                                  feedback=feedback)
            current = current_logits[0].masked_fill(~movement_mask, -torch.inf)
            fresh = fresh_logits[0].masked_fill(~movement_mask, -torch.inf)
            stay_probability = torch.softmax(current, dim=-1)[:, 0].tolist()
        actions = policy.act(state.observation)
        before_logs = len(environment.controller.action_log)
        transition = environment.step(actions)
        logs = environment.controller.action_log[before_logs:]
        ticks.append({
            "tick": len(ticks), "alive_slots": alive, "positions": positions,
            "movement": [actions[index].movement.value for index in alive],
            "fresh_memory_movement": [tuple(MovementAction)[int(fresh[index].argmax())].value
                                      for index in alive],
            "stay_probability": [round(stay_probability[index], 4) for index in alive],
            "legal_moves": [int(state.mask.movement[index].sum()) for index in alive],
            "actions": [{"slot": log.slot, "result": log.action,
                         "start": log.start, "requested": log.requested_position}
                        for log in logs],
            "metrics": transition.metrics,
        })
        state = transition.next_state
    return {"side": side.value, "stage": stage, "seed": seed,
            "checkpoint": str(checkpoint), "ticks": ticks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", required=True, choices=[item.value for item in Side])
    parser.add_argument("--stage", required=True, choices=("2v1", "2v2", "3v3", "5v5"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-ticks", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    side = Side(args.side)
    checkpoint = args.checkpoint or COACH_CHECKPOINT_PATHS[side.value] / "latest.pt"
    result = trace(side, args.stage, checkpoint, seed=args.seed,
                   max_ticks=args.max_ticks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"ticks": len(result["ticks"]), "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
