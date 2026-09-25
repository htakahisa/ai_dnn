"""Paired near-encounter game checks for Gorimaru facing candidates."""

from __future__ import annotations

import json
import os

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.evaluate_task10_rollout import evaluate
from coach_v1.experiment_task09a_gorimaru_labels import CURRENT, EXPERIMENT


def run() -> dict:
    paths = {
        "current": CURRENT,
        "epoch75_best": EXPERIMENT / "epoch75_best.pt",
        "epoch80_best": EXPERIMENT / "epoch80_best.pt",
        "epoch90_best": EXPERIMENT / "epoch90_best.pt",
        "epoch90_latest": EXPERIMENT / "epoch90_latest.pt",
    }
    result = {"seeds": [0, 19], "encounter": "Task10 near STAY/HOLD",
              "python_hash_seed": os.environ.get("PYTHONHASHSEED"), "models": {}}
    for name, path in paths.items():
        games = [evaluate(side, ticks=20, seed=seed, near=True,
                          checkpoint_overrides={0: path})
                 for seed in range(20)
                 for side in (Side.ATTACKER, Side.DEFENDER)]
        result["models"][name] = {
            "path": str(path),
            "by_side": {
                side.value: {
                    "sighting_actions": sum(game["per_slot"]["0"]["sighting_actions"]
                                            for game in games if game["side"] == side.value),
                    "aligned_45": sum(game["per_slot"]["0"]["aligned_45"]
                                      for game in games if game["side"] == side.value),
                    "unforced_sighting_actions": sum(
                        game["per_slot"]["0"]["unforced_sighting_actions"]
                        for game in games if game["side"] == side.value),
                    "unforced_aligned_45": sum(
                        game["per_slot"]["0"]["unforced_aligned_45"]
                        for game in games if game["side"] == side.value),
                    "ability_requests": sum(game["ability_requests"] for game in games
                                            if game["side"] == side.value),
                    "ability_successes": sum(game["ability_successes"] for game in games
                                             if game["side"] == side.value),
                }
                for side in (Side.ATTACKER, Side.DEFENDER)
            },
            "games": games,
            "by_seed_half": {
                f"{side.value}_{start}_{start + 9}": {
                    "sighting_actions": sum(game["per_slot"]["0"]["sighting_actions"]
                                            for game in games if game["side"] == side.value
                                            and start <= game["seed"] < start + 10),
                    "aligned_45": sum(game["per_slot"]["0"]["aligned_45"]
                                      for game in games if game["side"] == side.value
                                      and start <= game["seed"] < start + 10),
                    "unforced_sighting_actions": sum(
                        game["per_slot"]["0"]["unforced_sighting_actions"]
                        for game in games if game["side"] == side.value
                        and start <= game["seed"] < start + 10),
                    "unforced_aligned_45": sum(
                        game["per_slot"]["0"]["unforced_aligned_45"]
                        for game in games if game["side"] == side.value
                        and start <= game["seed"] < start + 10),
                }
                for side in (Side.ATTACKER, Side.DEFENDER) for start in (0, 10)
            },
        }
        print(name, result["models"][name]["by_side"], flush=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09a_gorimaru_labels_real_game.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    run()
