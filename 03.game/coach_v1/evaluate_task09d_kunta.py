"""Paired Kunta checkpoint evaluation in the Task 10 game loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.evaluate_task10_rollout import evaluate


BASELINE = CHECKPOINTS_DIR / "experiments" / "task09d_kunta" / "baseline_best.pt"


def compare(candidate: Path, *, start: int, stop: int) -> dict:
    result = {"candidate": str(candidate), "seeds": [start, stop - 1],
              "encounter": "near", "by_side": {}}
    for side in (Side.ATTACKER, Side.DEFENDER):
        variants = {}
        for name, path in (("baseline", BASELINE), ("candidate", candidate)):
            runs = [evaluate(side, ticks=20, seed=seed, near=True,
                             checkpoint_overrides={3: path})
                    for seed in range(start, stop)]
            slot = [run["per_slot"]["3"] for run in runs]
            variants[name] = {
                "unforced_aligned_45": sum(row["unforced_aligned_45"] for row in slot),
                "unforced_sighting_actions": sum(row["unforced_sighting_actions"] for row in slot),
                "ability_requests": sum(row["ability_requests"] for row in slot),
                "ability_successes": sum(row["ability_successes"] for row in slot),
                "flash_enemy_hits": sum(run["flash_enemy_hits_by_slot"]["3"] for run in runs),
            }
        result["by_side"][side.value] = variants
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / f"task09d_kunta_real_game_{candidate.stem}_{start}_{stop - 1}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(compare(args.candidate, start=args.start, stop=args.stop),
                     ensure_ascii=False, indent=2))
