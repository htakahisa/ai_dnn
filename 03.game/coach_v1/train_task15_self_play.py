"""Continue both coach models on real opponent-pool states using safe DAgger."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.opponent_pool import DEFAULT_POOL_PATH, load_opponent_pool
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.opponent_pool_imitation import (
    prepare_candidate_directory, train_pool_schedule,
)


DEFAULT_ROOT = CHECKPOINTS_DIR / "experiments" / "task15_pool_dagger"
ATTACKER_SOURCE = CHECKPOINTS_DIR / "coach" / "attacker" / "task12"
DEFENDER_SOURCE = CHECKPOINTS_DIR / "coach" / "defender" / "task13"
DEFAULT_SCHEDULE = ("omoko_v1", "touyama_v2", "gc_v1")


def train(
    *,
    root: Path = DEFAULT_ROOT,
    attacker_source: Path = ATTACKER_SOURCE,
    defender_source: Path = DEFENDER_SOURCE,
    pool_path: Path = DEFAULT_POOL_PATH,
    opponents: tuple[str, ...] = DEFAULT_SCHEDULE,
    seed: int = 1600,
    samples_per_bucket: int = 64,
    epochs: int = 2,
    device: str = "cpu",
):
    root = Path(root)
    attacker_dir, defender_dir = root / "attacker", root / "defender"
    prepare_candidate_directory(source=attacker_source, target=attacker_dir)
    prepare_candidate_directory(source=defender_source, target=defender_dir)
    attacker = CoachTrainer(Side.ATTACKER, seed=11, directory=attacker_dir,
                            device=device)
    defender = CoachTrainer(Side.DEFENDER, seed=11, directory=defender_dir,
                            device=device)
    attacker.resume()
    defender.resume()
    return train_pool_schedule(
        attacker_trainer=attacker, defender_trainer=defender,
        pool=load_opponent_pool(pool_path), opponent_ids=opponents,
        seed=seed, samples_per_bucket=samples_per_bucket, epochs=epochs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--opponent", action="append", dest="opponents")
    parser.add_argument("--seed", type=int, default=1600)
    parser.add_argument("--samples-per-bucket", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path,
                        default=REPORTS_DIR / "task15_pool_training.json")
    args = parser.parse_args()
    if args.samples_per_bucket <= 0 or args.epochs <= 0:
        parser.error("samples and epochs must be positive")
    torch.set_num_threads(1)
    results = train(
        root=args.root, pool_path=args.pool,
        opponents=tuple(args.opponents or DEFAULT_SCHEDULE), seed=args.seed,
        samples_per_bucket=args.samples_per_bucket, epochs=args.epochs,
        device=args.device,
    )
    payload = {"matches": [asdict(item) for item in results]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
