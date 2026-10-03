"""Watch a full rendered FRC match using an explicit candidate checkpoint directory."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from frc_v1.evaluate import OPPONENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True,
                        help="directory containing A_policy.pt and D_policy.pt")
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="Fnatic2023")
    parser.add_argument("--seed", type=int, default=100001)
    parser.add_argument("--output", help="result JSON; defaults to visual_match.json beside the checkpoints")
    args = parser.parse_args()

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from frc_v1.controller import FrcAttackerController, FrcDefenderController
    from frc_v1.model import FrcPolicy
    from map_data import NEW_MAZE_STR
    from party_presets import get_preset
    from run_game import VisualFPSBattle, _build_team_ai
    from team_ai import DualRoleTeamAI

    directory = Path(args.checkpoints)
    attacker = FrcPolicy.load(directory / "A_policy.pt", side="A")
    defender = FrcPolicy.load(directory / "D_policy.pt", side="D")
    own = get_preset("Furina Classic")
    other = get_preset(args.opponent)
    own_ai = DualRoleTeamAI("FRC v1 candidate",
        lambda: FrcAttackerController(actor=attacker),
        lambda: FrcDefenderController(actor=defender))
    game = VisualFPSBattle(NEW_MAZE_STR, own_ai, _build_team_ai(OPPONENTS[args.opponent]),
        headless=False,
        attacker_roster=list(own.players), defender_roster=list(other.players),
        spike_holder_name=own.spike_holder, defender_spike_holder_name=other.spike_holder,
        attacker_igl_name=own.igl, defender_igl_name=other.igl,
        attacker_team_name=own.name, defender_team_name=other.name)
    game.run()

    frc_is_attacker = game.attacker_team_name == own.name
    frc_score, other_score = ((game.attacker_wins, game.defender_wins) if frc_is_attacker else
                              (game.defender_wins, game.attacker_wins))
    result = {"mode": "learned", "checkpoints": str(directory), "opponent": other.name,
              "seed": args.seed, "frc_score": frc_score, "opponent_score": other_score,
              "match_over": game.match_over, "sides_swapped": game.sides_swapped,
              "rounds": game.current_round,
              "winner": (own.name if frc_score > other_score else other.name if other_score > frc_score else None)
              if game.match_over else None}
    output = Path(args.output) if args.output else directory / "visual_match.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
