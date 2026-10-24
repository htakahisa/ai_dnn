"""Compare recon/smoke responses with the prior runtime; no training."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch

AI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AI_DIR.parent))
sys.path.insert(0, str(AI_DIR))

import torch
import omoko_v1.ov1_attacker_controller as attacker
import ov1_learning_attacker_escort as escort
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import play_match
from omoko_v1.ability_advisor_v1.ov1_learn_attacker_ability_advisor import FixedAbilityAdvisor

SEEDS = (100000, 100001, 100002, 100003, 100004)
ROUNDS_PER_MATCH = 6
OPPONENT = "frc_v1"
OUTPUT = AI_DIR / "logs" / "ov1_recon_response_comparison.json"
HOLDOUT_SEEDS = (100010, 100011, 100012)
HOLDOUT_OUTPUT = AI_DIR / "logs" / "ov1_recon_response_holdout.json"


def prior_nearest(self, grid, chars, my_team, from_pos, max_range=None, **kwargs):
    best, distance = None, None
    for enemy in chars:
        if not enemy.is_alive or enemy.team == my_team:
            continue
        pos = tuple(enemy.pos)
        gap = escort._chebyshev(from_pos, pos)
        if max_range is not None and gap > max_range:
            continue
        if not escort._has_los_walls_only(grid, from_pos, pos):
            continue
        if distance is None or gap < distance:
            best, distance = enemy, gap
    return best, distance


def compare(seeds=SEEDS, output=OUTPUT):
    torch.set_num_threads(1)
    effect = escort.Ov1LearningAttackerEscortController._team_effect_active
    report = dict(opponent=OPPONENT,seeds=seeds,rounds=ROUNDS_PER_MATCH,
                  ability_advisor=False,carry_escape=True,runs=[])
    for seed in seeds:
        for variant in ("prior_runtime", "recon_smoke_fix"):
            with ExitStack() as stack:
                if variant == "prior_runtime":
                    stack.enter_context(patch.object(attacker, "recon_response", return_value=None))
                    stack.enter_context(patch.object(escort.Ov1LearningAttackerEscortController,
                                                      "_nearest_visible_enemy", prior_nearest))
                    stack.enter_context(patch.object(escort.Ov1LearningAttackerEscortController,
                        "_team_effect_active", lambda self, chars, team, smoke_cells=(): effect(self,chars,team,())))
                result = play_match(OPPONENT,seed,ROUNDS_PER_MATCH,FixedAbilityAdvisor(None, enabled=False))
            report["runs"].append(dict(variant=variant,seed=seed,result=result))
            print(json.dumps(dict(variant=variant,seed=seed,wins=result["attacker_wins"],
                                  plants=result["plants"])),flush=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def main():
    compare()
    compare(HOLDOUT_SEEDS, HOLDOUT_OUTPUT)


if __name__ == "__main__":
    main()
