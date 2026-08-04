from __future__ import annotations
import json
from pathlib import Path
from controllers import DefaultAttackerController,DefaultDefenderController
from defender_policy_common import get_defender_observation,valid_destination
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI

TARGET=60000; OUT=Path("demos/defender_rule_based_demos.json")
class RecordingDefender(DefaultDefenderController):
    def __init__(self,buf): super().__init__(); self.buf=buf; self.game=None
    def set_game(self,g): self.game=g
    def decide_move(self,char,state):
        obs=get_defender_observation(self.game,char); res=super().decide_move(char,state)
        action={"char":char.name,"team":char.team,"move":list(map(int,res[0] if isinstance(res,tuple) else res)),"ability":None,"special":None}
        if isinstance(res,tuple) and isinstance(res[1],dict): action["ability"]=res[1].get("ability"); action["ability_target"]=list(map(int,res[1].get("target",char.pos)))
        elif isinstance(res,tuple) and isinstance(res[1],str): action["special"]=res[1]
        valid=True
        if action["special"] is None and action["ability"] is None and action["move"]!=list(map(int,char.pos)): valid=valid_destination(self.game,char,*action["move"])
        if len(self.buf)<TARGET: self.buf.append({"observation":obs,"action":action,"teacher_action_valid":bool(valid)})
        return res

def main():
    buf=[]; matches=0
    while len(buf)<TARGET and matches<30:
        matches+=1; rec=RecordingDefender(buf)
        ateam=DualRoleTeamAI("logic-A",lambda:DefaultAttackerController(),lambda:DefaultDefenderController())
        dteam=DualRoleTeamAI("record-D",lambda:DefaultAttackerController(),lambda:rec)
        ar,dr=build_two_balanced_rosters(); game=VisualFPSBattle(NEW_MAZE_STR,ateam,dteam,headless=True,disable_side_swap=True,attacker_roster=ar,defender_roster=dr); game.run_headless_loop(); print(matches,len(buf))
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(buf,ensure_ascii=False),encoding="utf-8"); print("saved",OUT,len(buf))
if __name__=="__main__": main()
