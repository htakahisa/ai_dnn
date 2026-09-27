"""Compare Gongon checkpoints in the Task 10 real-game near encounter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.evaluate_task10_rollout import evaluate


BASELINE = CHECKPOINTS_DIR / "experiments" / "task09b_gongon" / "baseline_best.pt"


def compare(candidate: Path, *, start: int, stop: int,
            verify_routed: bool = False) -> dict:
    result = {"candidate": str(candidate), "seeds": [start, stop - 1], "by_side": {}}
    for side in (Side.ATTACKER, Side.DEFENDER):
        variants = {}
        for name, path in (("baseline", BASELINE), ("candidate", candidate)):
            runs = [evaluate(side, ticks=20, seed=seed, near=True,
                             checkpoint_overrides={1: path})
                    for seed in range(start, stop)]
            slot = [run["per_slot"]["1"] for run in runs]
            variants[name] = {
                "unforced_aligned_45": sum(row["unforced_aligned_45"] for row in slot),
                "unforced_sighting_actions": sum(row["unforced_sighting_actions"] for row in slot),
                "sighting_actions": sum(row["sighting_actions"] for row in slot),
                "ability_requests": sum(row["ability_requests"] for row in slot),
            }
        if verify_routed:
            runs = [evaluate(side, ticks=20, seed=seed, near=True,
                             gongon_routed=True)
                    for seed in range(start, stop)]
            slot = [run["per_slot"]["1"] for run in runs]
            variants["routed"] = {
                "unforced_aligned_45": sum(row["unforced_aligned_45"] for row in slot),
                "unforced_sighting_actions": sum(row["unforced_sighting_actions"] for row in slot),
                "sighting_actions": sum(row["sighting_actions"] for row in slot),
                "ability_requests": sum(row["ability_requests"] for row in slot),
            }
            expected = variants["baseline" if side is Side.ATTACKER else "candidate"]
            if variants["routed"] != expected:
                raise AssertionError(f"Gongon side routing changed {side.value} actions")
        result["by_side"][side.value] = variants
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_routed" if verify_routed else ""
    (REPORTS_DIR / f"task09b_gongon_real_game_{candidate.parent.name}_{candidate.stem}_{start}_{stop - 1}{suffix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=20)
    parser.add_argument("--verify-routed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(compare(args.candidate, start=args.start, stop=args.stop,
                             verify_routed=args.verify_routed),
                     ensure_ascii=False, indent=2))
