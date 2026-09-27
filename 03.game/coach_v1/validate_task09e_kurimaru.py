"""Independent watch-point and random-position validation for Kurimaru."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from coach_v1.common.constants import CHECKPOINTS_DIR, REPORTS_DIR
from coach_v1.common.types import Side
from coach_v1.learning_character_kurimaru import KurimaruPolicy
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer, load_character_model


BASELINE = CHECKPOINTS_DIR / "experiments" / "task09e_kurimaru" / "baseline_best.pt"


def run(candidate: Path, *, seed: int = 4004, count: int = 600,
        output: Path | None = None) -> dict:
    torch.set_num_threads(1)
    records = build_character_curriculum(4, seed=seed, count=count, multi_facing=True)
    trainer = CharacterTrainer(4, seed=3004)
    result = {"seed": seed, "samples": len(records), "checkpoints": {}}
    for name, path in (("baseline", BASELINE), ("candidate", candidate)):
        trainer.model = load_character_model(path, slot=4)
        result["checkpoints"][name] = {
            f"{side.value}_{category}": trainer.evaluate([
                row.example for row in records
                if row.side is side and row.category == category
            ]).to_dict()
            for side in (Side.ATTACKER, Side.DEFENDER)
            for category in ("point", "jitter", "random")
        }
    baseline_actor = KurimaruPolicy(BASELINE)
    candidate_actor = KurimaruPolicy(candidate)
    paired = {"same_use": 0, "same_target_when_both_use": 0,
              "both_use": 0, "baseline_use": 0, "candidate_use": 0}
    for row in records:
        observation = row.example.observation
        old = baseline_actor.act(observation)
        new = candidate_actor.act(observation)
        paired["same_use"] += old.use_ability == new.use_ability
        paired["baseline_use"] += old.use_ability
        paired["candidate_use"] += new.use_ability
        if old.use_ability and new.use_ability:
            paired["both_use"] += 1
            paired["same_target_when_both_use"] += old.target == new.target
            if not observation.mask.ability_target[new.target]:
                raise AssertionError("candidate chose a masked FLASH target")
    result["paired_actions"] = paired
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = output or REPORTS_DIR / "task09e_kurimaru_distribution_validation.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--seed", type=int, default=4004)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.candidate, seed=args.seed, count=args.count, output=args.output)
    for name, groups in report["checkpoints"].items():
        print(name, {group: (row["samples"], row["facing_accuracy"], row["ability_accuracy"])
                     for group, row in groups.items()})
