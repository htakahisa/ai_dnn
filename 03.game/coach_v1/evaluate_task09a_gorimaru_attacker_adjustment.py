"""Paired real-game check for attacker facing adjustment candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.evaluate_task10_rollout import evaluate
from coach_v1.experiment_task09a_gorimaru_attacker_adjustment import EXPERIMENT
from coach_v1.experiment_task09a_gorimaru_labels import CURRENT


PATHS = {
    "current": CURRENT,
    "mask_only_90_best": EXPERIMENT / "mask_only" / "epoch90_best.pt",
    "near_priority_80_best": EXPERIMENT / "near_priority" / "epoch80_best.pt",
    "near_priority_90_latest": EXPERIMENT / "near_priority" / "epoch90_latest.pt",
}


def run(*, start: int, stop: int, output: Path,
        names: tuple[str, ...] | None = None) -> dict:
    result = {"seeds": [start, stop - 1], "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
              "models": {}}
    for name, path in PATHS.items():
        if names is not None and name not in names:
            continue
        games = [evaluate(side, ticks=20, seed=seed, near=True,
                          checkpoint_overrides={0: path})
                 for seed in range(start, stop)
                 for side in (Side.ATTACKER, Side.DEFENDER)]
        summary = {
            side.value: {
                "unforced_aligned_45": sum(
                    game["per_slot"]["0"]["unforced_aligned_45"]
                    for game in games if game["side"] == side.value),
                "unforced_sighting_actions": sum(
                    game["per_slot"]["0"]["unforced_sighting_actions"]
                    for game in games if game["side"] == side.value),
                "sighting_actions": sum(
                    game["per_slot"]["0"]["sighting_actions"]
                    for game in games if game["side"] == side.value),
                "ability_requests_team": sum(
                    game["ability_requests"] for game in games
                    if game["side"] == side.value),
                "ability_successes_team": sum(
                    game["ability_successes"] for game in games
                    if game["side"] == side.value),
            }
            for side in (Side.ATTACKER, Side.DEFENDER)
        }
        result["models"][name] = {"path": str(path), "summary": summary, "games": games}
        print(name, summary, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--stop", type=int, default=40)
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task09a_gorimaru_attacker_adjustment_real_game.json")
    parser.add_argument("--model", choices=tuple(PATHS), action="append")
    args = parser.parse_args()
    run(start=args.start, stop=args.stop, output=args.output,
        names=tuple(args.model) if args.model else None)
