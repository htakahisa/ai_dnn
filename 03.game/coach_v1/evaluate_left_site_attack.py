"""Compare a left-site attack candidate with the incumbent on unused seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coach_v1.common.types import Side
from coach_v1.learning_coach_attacker import load_attacker_coach
from coach_v1.training.coach_environment import CoachTrainingEnvironment


def evaluate(path: Path, *, seed: int, episodes: int) -> dict:
    policy = load_attacker_coach(path)
    results = []
    for index in range(episodes):
        environment = CoachTrainingEnvironment(
            Side.ATTACKER, seed=seed + index, stage="left_full_round",
            max_ticks=100, observation_version=policy.encoder.version)
        state = environment.reset()
        policy.reset_round()
        totals = {"left_plant": 0.0, "survivors": 0.0,
                  "invalid_moves": 0.0, "carrier_progress": 0.0}
        ticks = 0
        while state is not None:
            transition = environment.step(policy.act(state.observation))
            ticks += 1
            totals["left_plant"] += transition.metrics["left_plant"]
            totals["survivors"] += transition.metrics["left_plant_survivors"]
            totals["invalid_moves"] += transition.metrics["invalid_moves"]
            totals["carrier_progress"] += transition.metrics["carrier_progress"]
            state = transition.next_state
        results.append({"seed": seed + index, "ticks": ticks, **totals})
    return {"checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "results": results,
            "left_plants": sum(item["left_plant"] for item in results),
            "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--seed", type=int, default=2300)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    incumbent = args.incumbent or Path(
        "coach_v1/checkpoints/coach/attacker/task12/latest.pt")
    report = {
        "stage": "left_full_round", "seed": args.seed,
        "incumbent": evaluate(incumbent, seed=args.seed, episodes=args.episodes),
        "candidate": evaluate(args.candidate, seed=args.seed, episodes=args.episodes),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: report[name]["left_plants"]
                      for name in ("incumbent", "candidate")}))


if __name__ == "__main__":
    main()
