"""Run the newly trained generic defender with any valid five-player preset."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import argparse
from toruAI_v4.tv4_scenario import Scenario, OPPONENTS
from toruAI_v4.tv4_defender_controller import DEFENDER_BEST
from toruAI_v4.tv4_game_controller import ToruV4GameDefenderController
from toruAI_v4.tv4_train_defender_analysis import legacy_root, seed_all, relocate_debug_logs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), default="fnatic_v3")
    parser.add_argument("--defender-preset", default="Gorigons")
    parser.add_argument("--render", action="store_true", help="Show the existing game window")
    parser.add_argument("--best-dir", type=Path, default=DEFENDER_BEST)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    from party_presets import get_preset
    from controllers import DefaultAttackerController
    from team_ai import DualRoleTeamAI
    from simulation_runtime import cpu_inference
    scenario = Scenario()
    own, enemy = get_preset(args.defender_preset), get_preset(OPPONENTS[args.opponent][1])
    if own is None or len(own.players) != 5 or set(own.players) & set(enemy.players):
        raise ValueError("Choose a valid five-player defender preset disjoint from the attackers")
    defender = ToruV4GameDefenderController(best_dir=args.best_dir.resolve())
    ai = DualRoleTeamAI("Toru AI v4", DefaultAttackerController, lambda: defender)
    with legacy_root(), cpu_inference(enabled=not args.render):
        from run_game import VisualFPSBattle, _build_team_ai
        seed_all(args.seed)
        game = VisualFPSBattle(scenario.maze, _build_team_ai(OPPONENTS[args.opponent][0], device="cpu"), ai,
            headless=not args.render, attacker_roster=list(enemy.players), defender_roster=list(own.players),
            spike_holder_name=enemy.spike_holder, defender_spike_holder_name=own.spike_holder,
            attacker_igl_name=enemy.igl, defender_igl_name=own.igl,
            attacker_team_name=enemy.name, defender_team_name=own.name, disable_side_swap=True)
        relocate_debug_logs(game.attacker_controller, None)
        game.run()


if __name__ == "__main__":
    main()
