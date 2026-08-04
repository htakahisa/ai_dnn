from controllers import DefaultAttackerController,DefaultDefenderController
from policy_defender_controller import PolicyDefenderController
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters

def main():
    a=DualRoleTeamAI("Logic",lambda:DefaultAttackerController(),lambda:DefaultDefenderController())
    d=DualRoleTeamAI("Fnatic Defender",lambda:DefaultAttackerController(),lambda:PolicyDefenderController())
    ar,dr=build_two_balanced_rosters(); game=VisualFPSBattle(NEW_MAZE_STR,a,d,headless=False,disable_side_swap=True,attacker_roster=ar,defender_roster=dr); game.run()
if __name__=="__main__": main()
