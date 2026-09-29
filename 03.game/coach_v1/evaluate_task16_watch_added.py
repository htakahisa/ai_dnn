"""Evaluate the added-point candidate against the four existing opponent AIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.opponent_pool import OpponentKind, OpponentPool, load_opponent_pool
from coach_v1.task15_self_play import evaluate_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    candidate = root / "checkpoints" / "experiments" / "task16_watch_added" / "coach"
    parser.add_argument("--attacker-checkpoint", type=Path,
                        default=candidate / "attacker" / "latest.pt")
    parser.add_argument("--defender-checkpoint", type=Path,
                        default=candidate / "defender" / "latest.pt")
    parser.add_argument("--seed", type=int, default=1510)
    parser.add_argument("--matches-per-opponent", type=int, default=2)
    parser.add_argument("--only-historical", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task16_watch_added_eval.json")
    args = parser.parse_args()
    if args.matches_per_opponent < 2:
        parser.error("at least two matches are needed for both starting sides")
    torch.set_num_threads(1)
    complete = load_opponent_pool()
    selected_kind = (OpponentKind.HISTORICAL_COACH if args.only_historical
                     else OpponentKind.EXISTING_AI)
    pool = OpponentPool(tuple(spec for spec in complete.opponents
                              if spec.kind is selected_kind),
                        require_coverage=False)
    result = evaluate_pool(
        attacker_checkpoint=args.attacker_checkpoint,
        defender_checkpoint=args.defender_checkpoint,
        pool=pool, seeds=range(args.seed, args.seed + args.matches_per_opponent),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result.aggregates(), ensure_ascii=False))


if __name__ == "__main__":
    main()
