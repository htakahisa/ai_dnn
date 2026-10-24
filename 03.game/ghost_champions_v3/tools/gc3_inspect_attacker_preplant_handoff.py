"""Frozen movement/congestion and real postplant handoff diagnostics, without learning."""
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout
import argparse
import io
import json
import sys
import types
from unittest.mock import patch
import numpy as np
import torch

AI_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "attacker_preplant_omoko_rl_resources_v2"
CHECKPOINTS = {label:AI_ROOT/"data"/RUN_NAME/(label+".pt") for label in ("best", "latest")}
OPPONENT_CODES = ("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC")
EVALUATION_GROUPS = (("fixed", 20261017+100000, 1000), ("holdout", 2030102100, 10000))
ROUNDS_PER_OPPONENT = 4
TORCH_THREADS = 1
LONG_IDLE_DECISIONS = 5
OUTPUT_ROOT = AI_ROOT/"logs/validation/gc3_preplant_handoff_audit"

sys.path.insert(0, str(AI_ROOT.parent))
from ghost_champions_v3.preplant.gc3_policy_attacker_history import load_preplant_policy
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import play_preplant, PreplantRewards
from ghost_champions_v3.preplant import gc3_rollout_attacker_preplant as rollout
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash, source_signature
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_engine_actions import MODES, CARDINAL, capabilities
from ghost_champions_v3.gc3_rollout import phase_start_snapshot, module_digest
from ghost_champions_v3.gc3_reward_phase import alignment
from ghost_champions_v3.gc3_io import write_json
from grid_paths import distance_map


def aggregate(rounds):
    plants = [r for r in rounds if r["stats"]["planted"]]
    result = dict(rounds=len(rounds), plants=len(plants),
        preplant_successes=sum(r["stats"]["success"] for r in rounds),
        actual_wins=sum(r["stats"]["actual_round_success"] for r in rounds),
        movement= dict(sum((Counter(r["movement"]) for r in rounds), Counter())),
        postplant_defeat_reasons=dict(Counter(r["stats"]["actual_round_reason"] for r in plants if not r["stats"]["actual_round_success"])))
    result["handoff_by_result"] = {}
    for win in (False, True):
        selected=[r["handoff"] for r in plants if r["stats"]["actual_round_success"] == win]
        result["handoff_by_result"]["win" if win else "loss"] = dict(count=len(selected),
            **{key:float(np.mean(values)) if (values:=[r[key] for r in selected if r[key] is not None and (key!="enemy_nearest_plant" or r[key]<100000)]) else None for key in
               ("own_alive", "enemy_alive", "own_hp", "enemy_hp", "own_within_5", "own_plant_los", "own_charges", "enemy_nearest_plant")})
    return result


class RoundAudit:
    def __init__(self):
        self.game=None;self.frame=0;self.trace=[];self.post_ticks=[];self.pending={}
        self.handoff=None;self.movements=Counter();self.distances=None

    def attach(self, game, reward):
        self.game=game;self.distances=reward.goal_distances
        original_move=game.move_character
        original_tick=game._simulate_tick
        def move(char):
            before=tuple(char.pos);name=str(char.name)
            self.pending.pop(name,None)
            occupants={tuple(c.pos):(str(c.name),c.team) for c in game.chars if c.is_alive and c is not char}
            result=original_move(char)
            row=self.pending.pop(name,None)
            if row is not None:
                row["executed_position"]=list(char.pos)
                row["actually_moved"]=tuple(char.pos)!=before
                wanted=tuple(row["requested_position"])
                if row["mode"] in ("MOVE","MOVE_WATCH") and wanted!=before and not row["actually_moved"]:
                    reason="occupied_ally" if occupants.get(wanted,(None,None))[1]==char.team else "occupied_enemy" if wanted in occupants else "other"
                    row["rejected_move_reason"]=reason
                if row["phase"]=="preplant" and not row["setup"]:
                    self.movements["decisions"]+=1
                    self.movements["ally_adjacent_decisions"]+=bool(row["ally_neighbors"])
                    self.movements["all_floor_neighbors_ally_blocked"]+=row["all_floor_neighbors_ally_blocked"]
                    self.movements["rejected_"+row.get("rejected_move_reason","none")]+=bool(row.get("rejected_move_reason"))
                    if row["carrying"]:
                        self.movements["carrier_decisions"]+=1
                        self.movements["carrier_goal_progress_ally_blocked"]+=row["goal_progress_ally_blocked"]
                        if row["goal_progress_ally_blocked"] and not row["actually_moved"] and row["mode"] in ("MOVE","MOVE_WATCH","WATCH"):
                            self.movements["carrier_wait_while_ally_blocks_progress"]+=1
                    if row["can_plant"]:
                        self.movements["can_plant_decisions"]+=1
                        self.movements["can_plant_but_chose_other"]+=row["mode"]!="PLANT"
                    self.movements["plant_channel_interrupted"]+=row["plant_timer"]>0 and row["mode"]!="PLANT" and row["can_act"]
                self.trace.append(row)
            return result
        def tick():
            self.frame+=1
            result=original_tick()
            if game.is_planted:
                own=[c for c in game.chars if c.team=="A" and c.is_alive]
                enemy=[c for c in game.chars if c.team=="D" and c.is_alive]
                defusers=[c for c in enemy if c.defuse_timer>0]
                self.post_ticks.append(dict(frame=self.frame, own_alive=len(own),enemy_alive=len(enemy),
                    defusers=[dict(name=str(c.name),position=list(c.pos),progress=int(c.defuse_timer),
                        own_clear_shot=[str(a.name) for a in own if game.check_shot_line_of_sight(a,c)],
                        own_facing_shot=[str(a.name) for a in own if not getattr(a,"blind_remaining",0) and alignment(a,c.pos)>.7 and game.check_shot_line_of_sight(a,c)],
                        clear_line_details=[dict(name=str(a.name),position=list(a.pos),facing=a.facing,
                            alignment=alignment(a,c.pos),blind=int(getattr(a,"blind_remaining",0)))
                            for a in own if game.check_shot_line_of_sight(a,c)]) for c in defusers]))
            return result
        game.move_character=move
        game._simulate_tick=tick

    def decision(self, char, state, command, row, *, phase):
        if self.game is None or char.team!="A":return
        origin=tuple(char.pos);grid=np.asarray(state["grid"])
        friends={tuple(c.pos):str(c.name) for c in state["chars"] if c.team==char.team and c.name!=char.name and c.is_alive}
        neighbors=[];blocked=[];open_floor=[]
        goal=row.get("goal_site")
        field=self.distances.get(goal) if goal is not None else None
        for dr,dc in CARDINAL:
            p=origin[0]+dr,origin[1]+dc
            if not (0<=p[0]<grid.shape[0] and 0<=p[1]<grid.shape[1]) or grid[p]==1:continue
            open_floor.append(p)
            if p in friends:neighbors.append(dict(position=list(p),name=friends[p]))
            if field is not None and field[p]<field[origin] and p in friends:blocked.append(list(p))
        caps=capabilities(char,state)
        mode=MODES[row["selections"]["mode"]]
        requested=command[0] if isinstance(command,tuple) else command
        self.pending[str(char.name)]=dict(frame=self.frame, tick=int(state.get("battle_tick",0)),phase=phase,
            setup=bool(state.get("defender_setup_active")),name=str(char.name),position=list(origin),
            requested_position=list(map(int,requested)),mode=mode,goal_site=goal,carrying=bool(char.has_spike),
            goal_distance=int(field[origin]) if field is not None else None,can_act=caps["can_act"],can_move=caps["can_move"],
            visible_enemy=any(c.team!=char.team and c.is_alive and getattr(c,"position_known",False) for c in state["chars"]),
            ally_neighbors=neighbors,goal_progress_ally_blocked=bool(blocked),blocked_progress_cells=blocked,
            all_floor_neighbors_ally_blocked=bool(open_floor) and all(p in friends for p in open_floor),
            can_plant=bool(row["masks"]["mode"][MODES.index("PLANT")]),plant_timer=int(char.plant_timer))

    def capture_handoff(self, game):
        if not game.is_planted:return
        own=[c for c in game.chars if c.team=="A" and c.is_alive]
        enemy=[c for c in game.chars if c.team=="D" and c.is_alive]
        field=distance_map(game.grid,tuple(game.planted_pos))
        distances=[int(field[tuple(c.pos)]) for c in enemy if field[tuple(c.pos)]>=0]
        self.handoff=dict(snapshot=phase_start_snapshot(game),plant=list(game.planted_pos),
            own_alive=len(own),enemy_alive=len(enemy),own_hp=sum(c.hp+getattr(c,"shield_hp",0) for c in own),
            enemy_hp=sum(c.hp+getattr(c,"shield_hp",0) for c in enemy),
            own_within_5=sum(bool(0<=field[tuple(c.pos)]<=5) for c in own),
            own_plant_los=sum(game.check_cell_line_of_sight(tuple(c.pos),tuple(game.planted_pos),block_smoke=False) for c in own),
            own_charges=sum(getattr(c,name+"_charges",0) for c in own for name in ("recon","flash","smoke","ramp","dance","ash")),
            enemy_nearest_plant=min(distances) if distances else None)

    def finish(self):
        by_actor=defaultdict(list)
        for row in self.trace:
            if row["phase"]=="preplant" and not row["setup"]:by_actor[row["name"]].append(row)
        idle_runs=[];oscillations=0
        for name,rows in by_actor.items():
            streak=[]
            def save():
                if len(streak)>=LONG_IDLE_DECISIONS:
                    idle_runs.append(dict(name=name,start=streak[0]["frame"],end=streak[-1]["frame"],
                        length=len(streak),position=streak[0]["position"],carrier=sum(r["carrying"] for r in streak),
                        blocked_by_ally=sum(r["goal_progress_ally_blocked"] for r in streak),
                        full_ally_block=sum(r["all_floor_neighbors_ally_blocked"] for r in streak)))
            for i,row in enumerate(rows):
                is_idle=(row["can_move"] and not row["actually_moved"] and not row["visible_enemy"] and row["mode"] in ("MOVE","WATCH","MOVE_WATCH"))
                if streak and (not is_idle or row["position"]!=streak[-1]["position"]):save();streak=[]
                if is_idle:streak.append(row)
                if i and row["actually_moved"] and rows[i-1]["actually_moved"] and row["position"]==rows[i-1]["executed_position"] and row["executed_position"]==rows[i-1]["position"]:
                    oscillations+=1
            save()
        self.movements["long_noncombat_idle_runs"]=len(idle_runs)
        self.movements["long_noncombat_idle_decisions"]=sum(r["length"] for r in idle_runs)
        self.movements["long_idle_with_no_ally_blocked_progress"]=sum(r["length"]-r["blocked_by_ally"] for r in idle_runs)
        self.movements["carrier_long_noncombat_idle_decisions"]=sum(r["carrier"] for r in idle_runs)
        self.movements["backtracking_pairs"]=oscillations
        return idle_runs


@contextmanager
def inspect_round(policy,audit):
    original_select=policy.select
    original_init=PreplantRewards.__init__
    original_guard=FreshPolicy.select
    original_complete=rollout.complete_round
    def select(self,char,state,history):
        command,row=original_select(char,state,history)
        audit.decision(char,state,command,row,phase="preplant")
        return command,row
    def init(self,game,*args,**kwargs):
        original_init(self,game,*args,**kwargs);audit.attach(game,self)
    def guard(self,char,state):
        command,row=original_guard(self,char,state)
        if state.get("is_planted"):audit.decision(char,state,command,row,phase="postplant")
        return command,row
    def complete(game,*args,**kwargs):
        audit.capture_handoff(game)
        return original_complete(game,*args,**kwargs)
    policy.select=types.MethodType(select,policy)
    try:
        with patch.object(PreplantRewards,"__init__",init),patch.object(FreshPolicy,"select",guard),patch.object(rollout,"complete_round",complete):
            yield
    finally:
        policy.select=original_select


def inspect_model(label):
    checkpoint=CHECKPOINTS[label];checkpoint_hash=file_hash(checkpoint)
    policy,metadata=load_preplant_policy(checkpoint)
    policy.requires_grad_(False).eval();weights=module_digest([policy])
    rounds=[]
    report=dict(stage="evaluation_only",optimizer_updates=0,model=label,step=metadata["step"],
        checkpoint=str(checkpoint),checkpoint_hash=checkpoint_hash,source_hash=source_signature(),complete=False,
        diagnostic_only=True,policy_inputs_unchanged=True,groups=EVALUATION_GROUPS)
    for group,base,stride in EVALUATION_GROUPS:
        for i,code in enumerate(OPPONENT_CODES):
            history=MatchHistory()
            for repeat in range(ROUNDS_PER_OPPONENT):
                audit=RoundAudit();seed=base+i*stride+repeat
                with torch.no_grad(),inspect_round(policy,audit),redirect_stdout(io.StringIO()):
                    _,stats=play_preplant(code,seed,policy=policy,history=history)
                idle=audit.finish()
                detail=dict(stats=stats,movement=dict(audit.movements),handoff=audit.handoff,
                            idle_runs=idle,trace=audit.trace,postplant_ticks=audit.post_ticks)
                filename=f"{group}_{code}_{repeat}.json"
                write_json(OUTPUT_ROOT/label/filename,detail)
                rounds.append(dict(group=group,detail=str(OUTPUT_ROOT/label/filename),stats=stats,
                                   movement=dict(audit.movements),handoff=audit.handoff))
            report.update(rounds=rounds,summary=aggregate(rounds))
            write_json(OUTPUT_ROOT/(label+".json"),report)
            print(label,group,code,json.dumps(aggregate(rounds)["movement"]),flush=True)
    if checkpoint_hash!=file_hash(checkpoint) or weights!=module_digest([policy]):
        raise RuntimeError("Diagnostic altered frozen student weights")
    fixed_path=AI_ROOT/"logs"/RUN_NAME/f"evaluation_{metadata['step']}.json"
    if fixed_path.exists():
        saved={r["seed"]:r for r in json.loads(fixed_path.read_text(encoding="utf-8"))["rounds"]}
        fields=("planted","plant","reason","ticks","orb_collections","actual_round_success","actual_round_reason","ultimate_points_start","ultimate_points_round_end")
        matches=all(all(r["stats"][key]==saved[r["stats"]["seed"]][key] for key in fields) for r in rounds if r["group"]=="fixed")
        if not matches:raise RuntimeError("Diagnostic hooks changed fixed scenario outcomes")
        report["fixed_scenarios_match_training_evaluation"]=True
    report.update(complete=True,checkpoint_unchanged=True,weights_unchanged=True)
    write_json(OUTPUT_ROOT/(label+".json"),report)
    print("TOTAL",label,json.dumps(report["summary"]),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=tuple(CHECKPOINTS),help="Temporary diagnostic override")
    args=parser.parse_args();torch.set_num_threads(TORCH_THREADS)
    for label in (args.model,) if args.model else CHECKPOINTS:inspect_model(label)


if __name__=="__main__":main()
