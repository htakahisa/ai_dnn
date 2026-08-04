"""初期版Defender DAgger。
まずBCモデルと実ゲーム接続を確認後に使用してください。
モデルが訪れた状態でDefaultDefenderControllerの行動を教師ラベルとして追加し、
集約JSONを再学習用に作ります。
"""
import json, random, shutil
from pathlib import Path
from controllers import DefaultAttackerController,DefaultDefenderController
from policy_defender_controller import PolicyDefenderController
from defender_policy_common import get_defender_observation,valid_destination
from run_game import VisualFPSBattle
from team_ai import DualRoleTeamAI
from map_data import NEW_MAZE_STR
from roster_utils import build_two_balanced_rosters
BASE=Path("demos/defender_rule_based_demos.json"); AGG=Path("demos/defender_dagger_aggregated.json"); TARGET=12000
class DaggerDefender:
    def __init__(self,model,buf,beta=.5): self.policy=PolicyDefenderController(model); self.expert=DefaultDefenderController(); self.buf=buf; self.beta=beta; self.game=None
    def set_game(self,g): self.game=g; self.policy.set_game(g)
    def reset_round(self): self.policy.reset_round()
    def decide_move(self,char,state):
        obs=get_defender_observation(self.game,char); exp=self.expert.decide_move(char,state)
        action={"char":char.name,"team":char.team,"move":list(map(int,exp[0] if isinstance(exp,tuple) else exp)),"ability":None,"special":None}
        if isinstance(exp,tuple) and len(exp)==2 and isinstance(exp[1],dict):
            action["ability"]=exp[1].get("ability")
            target=exp[1].get("target",char.pos)
            action["ability_target"]=[int(target[0]),int(target[1])]
        elif isinstance(exp,tuple) and len(exp)==2 and isinstance(exp[1],str):
            action["special"]=exp[1]
        valid=True
        if action["ability"] is None and action["special"] is None and action["move"]!=list(map(int,char.pos)):
            valid=valid_destination(self.game,char,*action["move"])
        if len(self.buf)<TARGET: self.buf.append({"observation":obs,"action":action,"teacher_action_valid":bool(valid)})
        return exp if random.random()<self.beta else self.policy.decide_move(char,state)
def main():
    if not BASE.exists(): raise FileNotFoundError(BASE)
    agg=json.loads(BASE.read_text(encoding="utf-8")); new=[]; ctrl=DaggerDefender("policy_fnatic_defender_final.pt",new,.5)
    while len(new)<TARGET:
        ta=DualRoleTeamAI("logic",lambda:DefaultAttackerController(),lambda:DefaultDefenderController()); td=DualRoleTeamAI("dagger",lambda:DefaultAttackerController(),lambda:ctrl)
        ar,dr=build_two_balanced_rosters(); g=VisualFPSBattle(NEW_MAZE_STR,ta,td,headless=True,disable_side_swap=True,attacker_roster=ar,defender_roster=dr); g.run_headless_loop(); print(len(new),TARGET)
    agg.extend(new); AGG.write_text(json.dumps(agg,ensure_ascii=False),encoding="utf-8"); print("saved",AGG,len(agg)); print("次にtrain_defender_bc.pyの入力をAGGへ変更して継続学習してください")
if __name__=="__main__": main()
