"""Run a complete headless FRC match, including the normal side swap."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import random

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "learned"), default="baseline")
    parser.add_argument("--checkpoints", default="frc_v1/checkpoints")
    parser.add_argument("--seed", type=int, default=100001)
    parser.add_argument("--output", default="frc_v1/evaluation/full_match.json")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    from map_data import NEW_MAZE_STR
    from run_game import VisualFPSBattle, _build_team_ai
    from team_ai import DualRoleTeamAI
    from party_presets import get_preset
    from frc_v1.controller import FrcAttackerController, FrcDefenderController
    from frc_v1.model import FrcPolicy
    if args.mode == "baseline":
        own_ai = _build_team_ai("frc_v1_baseline")
    else:
        path = Path(args.checkpoints)
        own_ai = DualRoleTeamAI("FRC v1",
            lambda: FrcAttackerController(actor=FrcPolicy.load(path / "A_policy.pt", side="A")),
            lambda: FrcDefenderController(actor=FrcPolicy.load(path / "D_policy.pt", side="D")))
    own, other = get_preset("Furina Classic"), get_preset("Fnatic2023")
    with contextlib.redirect_stdout(io.StringIO()):
        game = VisualFPSBattle(NEW_MAZE_STR, own_ai, _build_team_ai("fnatic_v3"), headless=True,
            attacker_roster=list(own.players), defender_roster=list(other.players),
            spike_holder_name=own.spike_holder, defender_spike_holder_name=other.spike_holder,
            attacker_igl_name=own.igl, defender_igl_name=other.igl,
            attacker_team_name=own.name, defender_team_name=other.name)
        game.run_headless_loop()
    frc_is_attacker = game.attacker_team_name == own.name
    own_score, other_score = ((game.attacker_wins, game.defender_wins) if frc_is_attacker else
                              (game.defender_wins, game.attacker_wins))
    result = {"mode": args.mode, "seed": args.seed, "frc_score": own_score, "opponent_score": other_score,
              "match_over": game.match_over, "sides_swapped": game.sides_swapped,
              "rounds": game.current_round, "winner": own.name if own_score > other_score else other.name}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
