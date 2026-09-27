"""Inspect Gorimaru attacker facing failures using legal reports only."""

from __future__ import annotations

import json
from collections import defaultdict

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.evaluate_task10_rollout import evaluate
from coach_v1.experiment_task09a_gorimaru_labels import CURRENT, EXPERIMENT


def _nearest_sector(offsets) -> str:
    dr, dc = min(offsets, key=lambda pair: abs(pair[0]) + abs(pair[1]))
    vertical = "N" if dr < 0 else "S" if dr > 0 else ""
    horizontal = "W" if dc < 0 else "E" if dc > 0 else ""
    return vertical + horizontal or "same_cell"


def _summary(games) -> dict:
    groups = defaultdict(lambda: [0, 0])
    for game in games:
        for event in game["sighting_trace"]:
            if event["forced"]:
                continue
            keys = (
                "all",
                "one_report" if len(event["reported_offsets"]) == 1 else "multiple_reports",
                "nearest_" + _nearest_sector(event["reported_offsets"]),
            )
            for key in keys:
                groups[key][0] += int(event["aligned_45"])
                groups[key][1] += 1
    return {key: {"aligned": value[0], "actions": value[1],
                  "rate": value[0] / value[1]}
            for key, value in sorted(groups.items())}


def run() -> dict:
    paths = {
        "current": CURRENT,
        "multi_label_epoch80": EXPERIMENT / "epoch80_best.pt",
    }
    result = {"seeds": [0, 19], "side": "attacker", "encounter": "near",
              "models": {}}
    for name, path in paths.items():
        games = [evaluate(Side.ATTACKER, ticks=20, seed=seed, near=True,
                          checkpoint_overrides={0: path}, trace_slot=0)
                 for seed in range(20)]
        result["models"][name] = {"path": str(path), "summary": _summary(games),
                                  "games": games}
        print(name, result["models"][name]["summary"], flush=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "task09a_gorimaru_attacker_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    run()
