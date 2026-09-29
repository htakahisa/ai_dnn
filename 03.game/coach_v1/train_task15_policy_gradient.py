"""Continue Task 15 candidates using full-match round-win rewards."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.opponent_pool import DEFAULT_POOL_PATH, load_opponent_pool
from coach_v1.training.opponent_pool_policy_gradient import (
    prepare_reward_trainers, train_reward_schedule,
)


DEFAULT_ROOT = CHECKPOINTS_DIR / "experiments" / "task15_pool_round_reward"
ATTACKER_SOURCE = CHECKPOINTS_DIR / "coach" / "attacker" / "task12"
DEFENDER_SOURCE = CHECKPOINTS_DIR / "coach" / "defender" / "task13"
DEFAULT_SCHEDULE = ("omoko_v1", "touyama_v2", "gc_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--opponent", action="append", dest="opponents")
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path,
                        default=REPORTS_DIR / "task15_pool_round_reward.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    attacker, defender = prepare_reward_trainers(
        root=args.root,
        attacker_source=ATTACKER_SOURCE,
        defender_source=DEFENDER_SOURCE,
        seed=11,
        device=args.device,
    )
    results = train_reward_schedule(
        attacker_trainer=attacker,
        defender_trainer=defender,
        pool=load_opponent_pool(args.pool),
        opponent_ids=tuple(args.opponents or DEFAULT_SCHEDULE),
        seed=args.seed,
        temperature=args.temperature,
    )
    payload = {"matches": [asdict(item) for item in results]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
