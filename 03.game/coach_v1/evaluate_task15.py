"""Run Task 15 multi-opponent full-match evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coach_v1.common.constants import REPORTS_DIR
from coach_v1.learning_coach_attacker import DEFAULT_DIRECTORY as ATTACKER_DIRECTORY
from coach_v1.learning_coach_defender import DEFAULT_DIRECTORY as DEFENDER_DIRECTORY
from coach_v1.opponent_pool import DEFAULT_POOL_PATH, load_opponent_pool
from coach_v1.task15_self_play import PoolEvaluation, decide_promotion, evaluate_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacker-checkpoint", type=Path,
                        default=ATTACKER_DIRECTORY / "latest.pt")
    parser.add_argument("--defender-checkpoint", type=Path,
                        default=DEFENDER_DIRECTORY / "latest.pt")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--seed", type=int, default=1500)
    parser.add_argument("--matches-per-opponent", type=int, default=2)
    parser.add_argument("--max-difficulty", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "task15_opponent_pool.json")
    parser.add_argument(
        "--incumbent-report", type=Path,
        help="paired prior evaluation; adds a non-regression promotion decision",
    )
    args = parser.parse_args()
    if args.matches_per_opponent < 2:
        parser.error("at least two matches are required to alternate starting sides")
    torch.set_num_threads(1)
    evaluation = evaluate_pool(
        attacker_checkpoint=args.attacker_checkpoint,
        defender_checkpoint=args.defender_checkpoint,
        pool=load_opponent_pool(args.pool),
        seeds=range(args.seed, args.seed + args.matches_per_opponent),
        max_difficulty=args.max_difficulty,
        device=args.device,
    )
    payload = evaluation.to_dict()
    if args.incumbent_report is not None:
        incumbent = PoolEvaluation.from_dict(json.loads(
            args.incumbent_report.read_text(encoding="utf-8")
        ))
        historical = tuple(
            item.opponent_id for item in load_opponent_pool(args.pool).opponents
            if item.kind.value == "historical_coach"
        )
        payload["promotion"] = decide_promotion(
            evaluation, incumbent, historical_opponents=historical,
        ).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
