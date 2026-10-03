"""Held-out seeded round evaluation, side splits and effect ablations."""

import argparse
import json
import math
from pathlib import Path
import torch

from frc_v1.baseline import FrcBaseline
from frc_v1.environment import FrcRoundEnvironment
from frc_v1.model import FrcPolicy

OPPONENTS = {"Fnatic2023": "fnatic_v3", "Ghost Champions": "ghost_champions_v1",
             "Touyama Gaming": "touyama_gaming_v2", "Omoko Gaming": "omoko_gaming_v1"}


def wilson(wins, total):
    if total == 0:
        return [0.0, 1.0]
    z, p = 1.96, wins / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    distance = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - distance), min(1, center + distance)]


def evaluate(args):
    if args.rounds < 1:
        raise ValueError("evaluation rounds must be positive")
    torch.set_num_threads(args.threads)
    sides = ("A", "D") if args.side == "both" else (args.side,)
    if args.checkpoint and (args.mode != "learned" or args.side == "both"):
        raise ValueError("--checkpoint requires --mode learned and one --side")
    report = {"seed": args.seed, "policy_mode": args.mode, "effect_override": args.effects,
              "evaluation_type": "single_round", "checkpoint_override": args.checkpoint, "results": []}
    for roster in args.opponents:
        for side in sides:
            actor = FrcBaseline() if args.mode == "baseline" else FrcPolicy.load(
                args.checkpoint or Path(args.checkpoints) / f"{side}_policy.pt", side=side)
            effect_mode = args.effects or getattr(actor, "effects_mode", "all")
            env = FrcRoundEnvironment(side, seed=args.seed, opponent=OPPONENTS[roster], opponent_roster=roster,
                                      effects_mode=effect_mode)
            outcomes = []
            for _ in range(args.rounds):
                observation = env.reset()
                while True:
                    decision = actor.act(observation, env.controller.snapshot, env.controller.belief)
                    result = env.step(decision)
                    observation = result.observation
                    if result.terminated:
                        outcomes.append(result.metrics)
                        break
            wins = sum(item["winner"] == side for item in outcomes)
            entry_samples = [item for item in outcomes if item["first_contact_slot"] is not None]
            summary = {"opponent": roster, "side": side, "effects": effect_mode, "rounds": args.rounds,
                "wins": wins, "win_rate": wins / args.rounds, "wilson_95": wilson(wins, args.rounds),
                "lohen_first_contact_rate": sum(item["lohen_first_contact"] for item in entry_samples) / max(1, len(entry_samples)),
                "outcomes": outcomes}
            report["results"].append(summary)
            print(json.dumps({key: value for key, value in summary.items() if key != "outcomes"}, ensure_ascii=False), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("learned", "baseline"), default="learned")
    parser.add_argument("--side", choices=("A", "D", "both"), default="both")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100001)
    parser.add_argument("--opponents", choices=tuple(OPPONENTS), nargs="+", default=["Fnatic2023"])
    parser.add_argument("--effects", choices=("all", "none", "flight", "warning"))
    parser.add_argument("--checkpoints", default="frc_v1/checkpoints")
    parser.add_argument("--checkpoint", help="evaluate one side's candidate checkpoint without replacing the live model")
    parser.add_argument("--output", default="frc_v1/evaluation/report.json")
    parser.add_argument("--threads", type=int, default=1)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
