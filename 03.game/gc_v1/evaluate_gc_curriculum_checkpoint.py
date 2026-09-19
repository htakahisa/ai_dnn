"""Evaluate one saved joint positioning checkpoint without changing its weights."""
import argparse
import json
from pathlib import Path

import torch

import train_attacker_gc_real_curriculum as trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("latest", "best_by_eval"), default="best_by_eval")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    sources = {p: args.models_dir / f"dqn_attacker_{p}_gc_{args.kind}.pt" for p in trainer.PHASES}
    checkpoints = {p: torch.load(path, map_location="cpu", weights_only=False)
                   for p, path in sources.items()}
    if len({c.get("episode") for c in checkpoints.values()}) != 1:
        raise ValueError("checkpoint set is not synchronized")
    policies = {
        "carry": trainer.runtime.AttackerCarryDuelingDQN(checkpoints["carry"]["obs_dim"]),
        "escort": trainer.escort_runtime.DuelingQNetwork(checkpoints["escort"]["obs_dim"], 6),
        "guard": trainer.guard_runtime.AttackerGuardDuelingDQN(),
    }
    for phase in trainer.PHASES:
        policies[phase].load_state_dict(checkpoints[phase]["model_state_dict"])
    session = trainer.CurriculumSession(sources, policies, 0.99)
    for phase in trainer.PHASES:
        session.controllers[phase].positioning_version = checkpoints[phase].get("positioning_version", 0)
    report = trainer.evaluate_multi(session, args.episodes, args.seeds, (100, 60, 40))
    report["checkpoint_episode"] = checkpoints["carry"].get("episode")
    content = json.dumps(report, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
