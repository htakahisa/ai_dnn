"""Read-only GC v3 training audit: matched-seed episodes, commands, effects and PPO data."""
from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import json
import math
import sys
from unittest.mock import patch
import numpy as np
import torch

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT.parent))
OUTPUT_ROOT = AI_ROOT/"logs/training_audit_fixed_v1"
ATTACKER_RUN = "attacker_postplant_fixed_v1"
DEFENDER_RUN = "defender_search_fixed_v1"
OPPONENT_CODES = ("TYG","OMG","FRC","FNC","GG","SPS")
ROUNDS_PER_OPPONENT = 2
EVALUATION_SEED = 20261011+100000
REPEAT_CHECK_OPPONENT = "TYG"
TORCH_THREADS = 1

from ghost_champions_v3.gc3_policy import FreshPolicy, LEGACY_SCHEMA_HASH
from ghost_champions_v3.gc3_engine_actions import MODES, FACING_DIRS, facing_towards
from ghost_champions_v3.gc3_spatial_policy import parse_map
from ghost_champions_v3.gc3_map_attacker_postplant import NEW_MAZE_STR as GUARD_MAP
from ghost_champions_v3.gc3_map_defender_initial import DEFENDER_SETUP_POSITION_MAP_GC_STR as SETUP_MAP
from ghost_champions_v3 import gc3_rollout as rollout
from ghost_champions_v3.gc3_io import write_json
from grid_paths import distance_map

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Probe:
    def __init__(self, policy, phase):
        self.policy,self.phase = policy,phase
        self.team = "A" if phase=="postplant" else "D"
        self.modes,self.casts,self.accepted,self.rejected,self.contributions = (Counter() for _ in range(5))
        self.counts = Counter()
        self.start = None
        self.spike_distances = []
        self.candidate_distances = []
        self.head_entropy = []
        self.events = []
        self.rows_to_check = []

    def after_setup(self, game):
        grid=game.grid
        field=distance_map(grid,game.planted_pos)
        own=[c for c in game.chars if c.team==self.team]
        enemy=[c for c in game.chars if c.team!=self.team]
        self.start=dict(plant=game.planted_pos, own_positions=[list(c.pos) for c in own],
                        own_spike_distances=[int(field[tuple(c.pos)]) for c in own],
                        enemy_spike_distances=[int(field[tuple(c.pos)]) for c in enemy])
        return game.planted_pos

    def select(self, original, char, state):
        result,record=original(char,state)
        mode=MODES[record["selections"]["mode"]]
        self.modes[mode]+=1
        self.counts["decisions"]+=1
        self.counts["ability_available"]+=bool(record["masks"]["mode"][3])
        self.counts["known_enemy_decisions"]+=any(c.team!=char.team and c.is_alive and getattr(c,"position_known",False)
                                                   for c in state["chars"])
        if len(self.rows_to_check)<32:
            self.rows_to_check.append(record)
        with torch.no_grad():
            context=self.policy.features(torch.from_numpy(record["obs"]))
            if "watch" in record["selections"]:
                distribution=self.policy.distribution(context,record,"watch")
                probabilities=distribution.probs.detach().numpy()
                directions=np.zeros(8)
                source=result[0] if mode=="MOVE_WATCH" else char.pos
                for index in np.flatnonzero(probabilities):
                    direction=facing_towards(source,np.unravel_index(index,state["grid"].shape),char.facing)
                    directions[FACING_DIRS.index(direction)]+=float(probabilities[index])
                p=directions[directions>0]
                self.head_entropy.append(dict(cell=float(distribution.entropy()),
                                               direction=float(-(p*np.log(p)).sum())))
        self.events.append(dict(name=str(char.name),tick=int(state.get("battle_tick",0)),
                                position=list(map(int,char.pos)),mode=mode,command=result,
                                ability=getattr(char,"ability_name",None),known_enemies=
                                sum(c.team!=char.team and c.is_alive and getattr(c,"position_known",False) for c in state["chars"])))
        return result,record

    def cast(self, original, game, owner, payload):
        relevant=owner.team==self.team and (self.phase!="postplant" or self.start is not None)
        kind=payload.get("ability","?") if isinstance(payload,dict) else "?"
        path_length=None
        if relevant:
            self.casts[kind]+=1
            if kind in ("FLASH","RECON") and isinstance(payload.get("target"),(list,tuple)):
                path_length=len(game._projectile_path(tuple(owner.pos),tuple(payload["target"])))
        accepted=original(game,owner,payload)
        if relevant:
            (self.accepted if accepted else self.rejected)[kind]+=1
            self.counts["immediately_blocked_projectiles"]+=bool(path_length is not None and path_length<=1)
            if self.events and self.events[-1]["name"]==str(owner.name):
                self.events[-1].update(cast_accepted=bool(accepted),projectile_path_length=path_length)
        return accepted

    def contribution(self, original, tracker, assister, victim, tick, method):
        if (self.phase!="postplant" or self.start is not None) and assister is not None and victim is not None and assister.team==self.team and victim.team!=self.team:
            self.contributions[method]+=1
        return original(tracker,assister,victim,tick,method)

    def tick(self, original, game):
        if self.phase=="postplant" and self.start is None:
            return original(game)
        previous={str(c.name):tuple(c.pos) for c in game.chars if c.team==self.team}
        result=original(game)
        own=[c for c in game.chars if c.team==self.team and c.is_alive]
        candidate=parse_map(GUARD_MAP if self.phase=="postplant" else SETUP_MAP)
        points=np.argwhere(candidate==5)
        # Geometry probes are never passed to a policy or used as a reward.
        for char in own:
            self.counts["alive_agent_ticks"]+=1
            self.counts["moved_agent_ticks"]+=previous.get(str(char.name))!=tuple(char.pos)
            if len(points):
                near=min(max(abs(char.pos[0]-int(p[0])),abs(char.pos[1]-int(p[1]))) for p in points)
                self.candidate_distances.append(near)
            if game.is_planted:
                near=max(abs(char.pos[i]-game.planted_pos[i]) for i in (0,1))
                self.spike_distances.append(near)
                self.counts["within_5_spike"]+=near<=5
        for shot in game.last_shots or []:
            if shot["shooter"].team==self.team:
                self.counts["shots"]+=1
                self.counts["hits"]+=bool(shot["hit"])
                self.counts["damage"]+=int(shot["damage"])
        return result

    def report(self):
        with torch.no_grad():
            measured=self.policy.evaluate_records(self.rows_to_check)[0].numpy()
            observations=torch.from_numpy(np.stack([row["obs"] for row in self.rows_to_check]))
            values=self.policy.critic(self.policy.features(observations)).numpy()
            first_saturation = second_saturation = None
            if self.policy.schema_hash == LEGACY_SCHEMA_HASH:
                first=self.policy.features[0](observations)
                second=self.policy.features[2](torch.tanh(first))
                first_saturation=float((first.abs()>3).float().mean())
                second_saturation=float((second.abs()>3).float().mean())
        error=max((abs(float(value)-row["logp"]) for value,row in zip(measured,self.rows_to_check)),default=0.)
        return dict(modes=dict(self.modes),casts=dict(self.casts),accepted=dict(self.accepted),
                    rejected=dict(self.rejected),contributions=dict(self.contributions),counts=dict(self.counts),
                    start=self.start,mean_spike_distance=float(np.mean(self.spike_distances)) if self.spike_distances else None,
                    mean_candidate_distance=float(np.mean(self.candidate_distances)) if self.candidate_distances else None,
                    watch_cell_entropy=float(np.mean([r["cell"] for r in self.head_entropy])) if self.head_entropy else None,
                    watch_direction_entropy=float(np.mean([r["direction"] for r in self.head_entropy])) if self.head_entropy else None,
                    logp_max_error=error,
                    architecture="tanh-v1" if self.policy.schema_hash==LEGACY_SCHEMA_HASH else "split-layernorm-silu-v2",
                    first_layer_saturation=first_saturation,
                    second_layer_saturation=second_saturation,
                    sampled_value_std=float(np.std(values)))

def play(label, policy, phase, code, seed):
    from run_game import VisualFPSBattle
    from analytics.combat_tracker import CombatTracker
    probe=Probe(policy,phase)
    original_select=policy.select
    original_tick=VisualFPSBattle._simulate_tick
    original_cast=VisualFPSBattle.execute_ai_ability
    original_contribution=CombatTracker.record_contribution
    original_setup=rollout.setup_postplant
    def setup(game,seed):
        original_setup(game,seed)
        return probe.after_setup(game)
    with patch.object(policy,"select",lambda char,state:probe.select(original_select,char,state)), \
         patch.object(VisualFPSBattle,"_simulate_tick",lambda game:probe.tick(original_tick,game)), \
         patch.object(VisualFPSBattle,"execute_ai_ability",lambda game,owner,payload:probe.cast(original_cast,game,owner,payload)), \
         patch.object(CombatTracker,"record_contribution",lambda tracker,*args:probe.contribution(original_contribution,tracker,*args)), \
         patch.object(rollout,"setup_postplant",setup):
        rows,stats=rollout.play_phase(code,seed,phase=phase,policy=policy)
    report=probe.report()
    report.update(model=label,phase=phase,stats=stats,
                  reward_nonzero=sum(row["reward"]!=0 for row in rows),transitions=len(rows),
                  value_explained_variance=1-float(np.var([row["return"]-row["value"] for row in rows]))/
                  max(1e-12,float(np.var([row["return"] for row in rows]))))
    del rows
    with (OUTPUT_ROOT/"decisions.jsonl").open("a",encoding="utf-8") as handle:
        for event in probe.events:
            handle.write(json.dumps(dict(model=label,phase=phase,opponent=code,seed=seed,**event),ensure_ascii=False,
                                    default=lambda value:value.item() if isinstance(value,np.generic) else value.tolist())+"\n")
    return report

def aggregate(rows):
    results={}
    for label in dict.fromkeys(row["model"] for row in rows):
        group=[row for row in rows if row["model"]==label]
        result=dict(rounds=len(group),successes=sum(row["stats"]["success"] for row in group))
        for field in ("modes","casts","accepted","rejected","contributions","counts"):
            counter=Counter()
            for row in group: counter.update(row[field])
            result[field]=dict(counter)
        for field in ("mean_spike_distance","mean_candidate_distance","watch_cell_entropy","watch_direction_entropy",
                      "first_layer_saturation","second_layer_saturation","sampled_value_std"):
            vals=[row[field] for row in group if row[field] is not None]
            result[field]=float(np.mean(vals)) if vals else None
        result["logp_max_error"]=max(row["logp_max_error"] for row in group)
        result["initial_nearest_own_spike"]=float(np.mean([min(row["start"]["own_spike_distances"]) for row in group if row["start"]])) if any(row["start"] for row in group) else None
        result["initial_nearest_enemy_spike"]=float(np.mean([min(row["start"]["enemy_spike_distances"]) for row in group if row["start"]])) if any(row["start"] for row in group) else None
        results[label]=result
    return results

def main():
    torch.set_num_threads(TORCH_THREADS)
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT_ROOT/"decisions.jsonl").exists():
        raise RuntimeError("Audit output already exists; choose a new OUTPUT_ROOT")
    paths={label:AI_ROOT/"data"/ATTACKER_RUN/(label+".pt") for label in ("initial","best","latest")}
    paths["defender_latest"]=AI_ROOT/"data"/DEFENDER_RUN/"latest.pt"
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("Audit checkpoints are missing; set ATTACKER_RUN/DEFENDER_RUN to completed runs")
    preserved=list(paths.values())+[AI_ROOT/name for name in ("gc3_map_attacker_postplant.py","gc3_map_defender_initial.py")]
    before={str(path):file_hash(path) for path in preserved}
    policies={label:FreshPolicy.load(path,allow_legacy=True)[0] for label,path in paths.items()}
    policy_digests={label:rollout.module_digest([policy]) for label,policy in policies.items()}
    rows=[]
    with (OUTPUT_ROOT/"engine.log").open("w",encoding="utf-8") as engine:
        for label,policy in policies.items():
            phase="search" if label=="defender_latest" else "postplant"
            for index,code in enumerate(OPPONENT_CODES):
                for repeat in range(ROUNDS_PER_OPPONENT):
                    seed=EVALUATION_SEED+index*1000+repeat
                    with redirect_stdout(engine):
                        row=play(label,policy,phase,code,seed)
                    rows.append(row)
                    write_json(OUTPUT_ROOT/f"episode_{len(rows):03d}.json",row)
                    write_json(OUTPUT_ROOT/"progress.json",dict(finished=len(rows),planned=len(policies)*len(OPPONENT_CODES)*ROUNDS_PER_OPPONENT))
                    print(label,code,repeat,"success",row["stats"]["success"],"casts",row["casts"],"rejected",row["rejected"],flush=True)
        repeat_rows=[]
        for repeat in range(2):
            with redirect_stdout(engine):
                repeat_rows.append(play("best",policies["best"],"postplant",REPEAT_CHECK_OPPONENT,EVALUATION_SEED))
    immutable=all(file_hash(path)==before[str(path)] for path in preserved)
    weights_unchanged=all(rollout.module_digest([policy])==policy_digests[label] for label,policy in policies.items())
    report=dict(episodes=rows,summary=aggregate(rows),repeat_checks=repeat_rows,
                checkpoint_and_maps_unchanged=immutable,policy_weights_unchanged=weights_unchanged,
                optimizer_updates=0,training_executed=False)
    write_json(OUTPUT_ROOT/"report.json",report)
    assert immutable and weights_unchanged
    print(json.dumps(report["summary"],ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
