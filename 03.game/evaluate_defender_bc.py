import argparse
from controllers import DefaultAttackerController,DefaultDefenderController
from policy_defender_controller import PolicyDefenderController
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters

def main():
    p=argparse.ArgumentParser(); p.add_argument("--matches",type=int,default=20); p.add_argument("--model",default="policy_fnatic_defender_final.pt"); a=p.parse_args(); wins=0; rounds_a=rounds_d=0
    for i in range(a.matches):
        ta=DualRoleTeamAI("logic",lambda:DefaultAttackerController(),lambda:DefaultDefenderController())
        td=DualRoleTeamAI("fnaticD",lambda:DefaultAttackerController(),lambda:PolicyDefenderController(a.model))
        ar,dr=build_two_balanced_rosters(); g=VisualFPSBattle(NEW_MAZE_STR,ta,td,headless=True,disable_side_swap=True,attacker_roster=ar,defender_roster=dr); g.run_headless_loop(); rounds_a+=g.attacker_wins; rounds_d+=g.defender_wins; wins+=int(g.defender_wins>g.attacker_wins); print(i+1,g.attacker_wins,g.defender_wins)
    print(f"Defender win rate: {wins/a.matches*100:.2f}% | avg A {rounds_a/a.matches:.2f} - D {rounds_d/a.matches:.2f}")
if __name__=="__main__": main()
