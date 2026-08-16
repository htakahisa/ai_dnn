"""GC Defender Opening Macro v2 - opening-only trainer.

Flow per episode:
  learned Setup -> 24 live Opening ticks -> stop

No Search/Retake/full-round reward is used.
Attackers come from real party presets and use the normal Toru attacker AI.

Stage 2A:
- Selection only is learned.
- SMOKE auto-executes once selected.
- FLASH / RECON move to Origin, then auto-execute immediately on arrival.
- WAIT / EXECUTE / CANCEL is intentionally not learned in this stage.

GC utility slots:
  Xdll RECON / SyouTa SMOKE / Absol FLASH / eKo RECON / SugarZ3ro SMOKE
"""

from __future__ import annotations
import argparse, inspect, json, random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from map_data import NEW_MAZE_STR
from controllers import DefaultAttackerController
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI
from party_presets import all_preset_names, get_preset

try:
    from gc_v1.learning_defender_setup_gc_runtime import LearningDefenderSetupGCRuntime
except ImportError:
    from learning_defender_setup_gc_runtime import LearningDefenderSetupGCRuntime

try:
    from gc_v1.map_data_defender_opening_ability_gc import OPENING_PATTERN_IDS, ABILITY_PATTERN_LAYERS
except ImportError:
    from map_data_defender_opening_ability_gc import OPENING_PATTERN_IDS, ABILITY_PATTERN_LAYERS


GC_ROSTER = ["Xdll", "SyouTa", "Absol", "eKo", "SugarZ3ro"]
SLOT_ABILITY = {
    "Xdll": "RECON", "SyouTa": "SMOKE", "Absol": "FLASH",
    "eKo": "RECON", "SugarZ3ro": "SMOKE",
}
GC_PRESET = "Ghost Champions"
CARDINAL = ((-1,0),(1,0),(0,-1),(0,1))
OPENING_TICKS = 24

LR = 2.5e-4
BATCH = 128
REPLAY_CAP = 100_000
LEARN_START = 500
EPS_START, EPS_END, EPS_DECAY = 0.90, 0.05, 3000

R_EXECUTE = 0.18
R_ORIGIN = 0.08
P_IMPOSSIBLE = -0.20
P_UNUSED = -0.10
P_DUP_TARGET = -0.07
W_ALIVE = 0.18
W_HP = 0.003

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "defender_opening_macro_gc_v2_stage2a_data"
BEST = DATA_DIR / "dqn_defender_opening_macro_gc_v2_stage2a_best.pt"
LATEST = DATA_DIR / "dqn_defender_opening_macro_gc_v2_stage2a_latest.pt"
FINAL = DATA_DIR / "dqn_defender_opening_macro_gc_v2_stage2a_final.pt"
INTERRUPT = DATA_DIR / "dqn_defender_opening_macro_gc_v2_stage2a_interrupt.pt"
LOG = DATA_DIR / "training_log.jsonl"


def _rows(text):
    rows = [x.strip() for x in str(text).strip().splitlines() if x.strip()]
    if not rows or any(len(x) != len(rows[0]) for x in rows):
        raise ValueError("invalid Opening ability map")
    return rows


def _collect(text):
    result = {int(pid): [] for pid in OPENING_PATTERN_IDS}
    for r, row in enumerate(_rows(text)):
        for c, cell in enumerate(row):
            if cell.isdigit() and int(cell) in result:
                result[int(cell)].append((r, c))
    return {k:v for k,v in result.items() if v}


def load_patterns():
    result = {}
    for ability, layers in ABILITY_PATTERN_LAYERS.items():
        ability = str(ability).upper()
        cells = {name: _collect(text) for name, text in layers.items()}
        out = {}
        for pid in map(int, OPENING_PATTERN_IDS):
            origins = cells.get("origin", {}).get(pid, [])
            targets = cells.get("target", {}).get(pid, [])
            valid = bool(targets) if ability == "SMOKE" else bool(origins and targets)
            if valid:
                out[pid] = {"id":pid, "ability":ability, "origins":origins, "targets":targets}
        result[ability] = out
    return result


PATTERNS = load_patterns()
CATALOG = {n:[None] + sorted(PATTERNS[SLOT_ABILITY[n]]) for n in GC_ROSTER}
ACTION_DIMS = {n:len(CATALOG[n]) for n in GC_ROSTER}

# global 10 + players 5*8 + plans 5*8
OBS_DIM = 90


def ability_charge(char, ability):
    attr = {"SMOKE":"smoke_charges","FLASH":"flash_charges","RECON":"recon_charges"}[ability]
    return int(getattr(char, attr, 0) or 0)


def bfs_dist(grid, goals):
    h,w = grid.shape
    d = np.full((h,w), -1, np.int32)
    q = deque()
    for r,c in goals:
        r,c=int(r),int(c)
        if 0<=r<h and 0<=c<w and grid[r,c] != 1:
            d[r,c]=0; q.append((r,c))
    while q:
        r,c=q.popleft(); nd=int(d[r,c])+1
        for dr,dc in CARDINAL:
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and grid[nr,nc] != 1 and d[nr,nc] < 0:
                d[nr,nc]=nd; q.append((nr,nc))
    return d


def bfs_step(grid, char, chars, goal):
    dm = bfs_dist(grid, [goal])
    r,c=map(int,char.pos); cur=int(dm[r,c])
    if cur <= 0: return [r,c]
    occupied={tuple(map(int,x.pos)) for x in chars if x is not char and getattr(x,"is_alive",True)}
    cand=[]
    for dr,dc in CARDINAL:
        nr,nc=r+dr,c+dc
        if 0<=nr<grid.shape[0] and 0<=nc<grid.shape[1] and grid[nr,nc] != 1 and (nr,nc) not in occupied:
            dd=int(dm[nr,nc])
            if 0<=dd<cur: cand.append((dd,nr,nc))
    if not cand: return [r,c]
    _,nr,nc=min(cand)
    return [nr,nc]


def nearest_origin(char, pattern, grid):
    best=None
    for o in pattern["origins"]:
        dm=bfs_dist(grid,[o]); r,c=map(int,char.pos); d=int(dm[r,c])
        if d>=0 and (best is None or d<best[0]): best=(d,tuple(o))
    return (None,999) if best is None else (best[1],best[0])


def nearest_target(origin, pattern):
    targets=[tuple(x) for x in pattern["targets"]]
    if not targets: return None
    if origin is None: return targets[0]
    return min(targets,key=lambda p:max(abs(p[0]-origin[0]),abs(p[1]-origin[1])))


def alive(chars, team):
    return [c for c in chars if getattr(c,"team",None)==team and getattr(c,"is_alive",True)]


def hp_sum(chars, team):
    return float(sum(max(0,float(getattr(c,"hp",0))) for c in alive(chars,team)))


@dataclass
class Plan:
    slot:str
    ability:str
    pid:int
    origin:tuple|None
    target:tuple
    state:str="PLANNED"
    origin_reached:bool=False
    executed_tick:int|None=None
    cancel_reason:str|None=None


def build_obs(game, plans, tick):
    chars=list(game.chars); by={str(c.name):c for c in chars}; h,w=game.grid.shape
    g=[
        min(tick/max(1,OPENING_TICKS),1.0),
        len(alive(chars,"D"))/5, len(alive(chars,"A"))/5,
        hp_sum(chars,"D")/500, hp_sum(chars,"A")/500,
        float(bool(getattr(game,"is_planted",False))),
        float(getattr(game,"spike_pos",None) is not None),
        sum(p.state=="EXECUTED" for p in plans.values())/5,
        sum(p.state=="CANCELLED" for p in plans.values())/5,
        sum(p.origin_reached for p in plans.values())/5,
    ]
    pf=[]
    for n in GC_ROSTER:
        c=by.get(n); ab=SLOT_ABILITY[n]
        if c is None: pf += [0.0]*8; continue
        r,col=map(int,c.pos)
        pf += [
            float(getattr(c,"is_alive",True)), r/max(1,h-1), col/max(1,w-1),
            min(max(float(getattr(c,"hp",0))/100,0),1.5),
            float(ability_charge(c,ab)>0),
            float(ab=="SMOKE"),float(ab=="FLASH"),float(ab=="RECON"),
        ]
    pl=[]
    for n in GC_ROSTER:
        p=plans.get(n); c=by.get(n)
        dist=0.0
        if p and p.origin is not None and c is not None:
            dm=bfs_dist(game.grid,[p.origin]); r,col=map(int,c.pos); dd=int(dm[r,col])
            if dd>=0: dist=min(dd/24,1.0)
        pl += [
            float(p is not None and p.state not in {"EXECUTED","CANCELLED"}),
            float(p is not None and p.state=="EXECUTED"),
            float(p is not None and p.state=="CANCELLED"),
            float(p is not None and p.state=="READY"),
            float(p is not None and p.origin_reached),
            dist,
            float(p.pid)/9 if p else 0.0,
            float(p is not None and p.target is not None),
        ]
    obs=np.asarray(g+pf+pl,np.float32)
    if obs.shape!=(OBS_DIM,): raise RuntimeError(f"OBS {obs.shape} != {(OBS_DIM,)}")
    return obs


class SelectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(OBS_DIM,256),nn.ReLU(),nn.Linear(256,192),nn.ReLU())
        self.heads=nn.ModuleDict({n:nn.Linear(192,ACTION_DIMS[n]) for n in GC_ROSTER})
    def forward(self,x):
        h=self.trunk(x); return {n:m(h) for n,m in self.heads.items()}


@dataclass
class Transition:
    obs:np.ndarray
    action:int
    reward:float
    slot:str


class Replay:
    def __init__(self,cap): self.data=deque(maxlen=cap)
    def __len__(self): return len(self.data)
    def add(self,t): self.data.append(t)
    def sample(self,n,rng): return rng.sample(list(self.data),n)


def eps_value(ep):
    f=min(1,max(0,ep/EPS_DECAY))
    return EPS_END+(EPS_START-EPS_END)*(1-f)


def eps_action(q,eps,rng):
    return rng.randrange(len(q)) if rng.random()<eps else int(np.argmax(q))


class DefenderOpeningController:
    def __init__(self,setup,sel,device,eps,rng,training):
        self.setup=setup; self.sel=sel; self.device=device
        self.eps=eps; self.rng=rng; self.training=training; self.game=None
        self.tick=0; self.selected=False; self.plans={}
        self.sel_steps=[]; self.local_reward=0.0

    def set_game(self,game):
        self.game=game
        if hasattr(self.setup,"set_game"): self.setup.set_game(game)

    def reset_round(self):
        self.tick=0; self.selected=False; self.plans={}
        self.sel_steps=[]; self.local_reward=0.0
        if hasattr(self.setup,"reset_round"): self.setup.reset_round()

    def setup_active(self):
        phase=getattr(self.game,"defender_setup_phase",None)
        return bool(phase is not None and phase.active)

    def select_plans(self):
        if self.selected:return
        self.selected=True
        obs=build_obs(self.game,self.plans,self.tick)
        by={str(c.name):c for c in self.game.chars}
        with torch.no_grad():
            x=torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            heads={k:v[0].cpu().numpy() for k,v in self.sel(x).items()}
        for n in GC_ROSTER:
            c=by.get(n)
            if c is None or not getattr(c,"is_alive",True):continue
            a=eps_action(heads[n],self.eps if self.training else 0,self.rng)
            pid=CATALOG[n][a]
            if self.training:self.sel_steps.append((n,obs.copy(),a))
            if pid is None:continue
            ab=SLOT_ABILITY[n]; pat=PATTERNS[ab][pid]
            if ability_charge(c,ab)<=0:
                self.local_reward += P_IMPOSSIBLE; continue
            if ab=="SMOKE": origin=None; target=nearest_target(None,pat)
            else:
                origin,d=nearest_origin(c,pat,self.game.grid); target=nearest_target(origin,pat)
                if origin is None or d>=999:
                    self.local_reward += P_IMPOSSIBLE; continue
            if target is None:
                self.local_reward += P_IMPOSSIBLE; continue
            self.plans[n]=Plan(n,ab,pid,origin,target)
        dup=Counter((p.ability,p.target) for p in self.plans.values())
        self.local_reward += sum(max(0,v-1) for v in dup.values())*P_DUP_TARGET

    def decide_move(self,char,game_state):
        if self.setup_active():
            chars=list(self.game.chars)
            if not getattr(self.setup,"round_initialized",False): self.setup.initialize_round(chars)
            return self.setup.decide_setup_move(char,chars)

        base=list(map(int,char.pos))
        n=str(getattr(char,"name",""))
        if n not in GC_ROSTER:return base
        self.select_plans()
        p=self.plans.get(n)
        if p is None or p.state in {"EXECUTED","CANCELLED"}:return base
        if not getattr(char,"is_alive",True):
            p.state="CANCELLED";p.cancel_reason="DEAD";return base
        if ability_charge(char,p.ability)<=0:
            p.state="CANCELLED";p.cancel_reason="NO_CHARGE";return base

        if p.ability!="SMOKE":
            dm=bfs_dist(self.game.grid,[p.origin]); r,c=map(int,char.pos); d=int(dm[r,c])
            if d<0:
                p.state="CANCELLED";p.cancel_reason="UNREACHABLE";self.local_reward+=P_IMPOSSIBLE;return base
            if d>0:
                p.state="MOVING"; return bfs_step(self.game.grid,char,self.game.chars,p.origin)
            if not p.origin_reached:
                p.origin_reached=True; self.local_reward += R_ORIGIN

        # Stage 2A: no timing policy yet.
        # Once the selected plan is executable, fire it immediately.
        p.state="READY"
        p.state="EXECUTED"
        p.executed_tick=self.tick
        self.local_reward += R_EXECUTE
        return (base,{"ability":p.ability,"target":tuple(map(int,p.target))})


def opponent_names():
    out=[]
    for n in all_preset_names():
        if n==GC_PRESET:continue
        p=get_preset(n)
        if p is not None and len(tuple(p.players))==5:out.append(n)
    if not out:raise RuntimeError("No opponent presets")
    return out


def attacker_ai_key():
    for key in ("toru_ai_v3.1","toru_ai_v3"):
        try:_build_team_ai(key);return key
        except Exception:pass
    raise RuntimeError("No Toru attacker AI key")


ATTACKER_AI=attacker_ai_key()


def build_game(ctrl,rng):
    name=rng.choice(opponent_names()); p=get_preset(name)
    attacker=_build_team_ai(ATTACKER_AI)
    defender=DualRoleTeamAI("GC-Opening-v2-D",attacker_factory=lambda:DefaultAttackerController(),defender_factory=lambda:ctrl,use_iq_perception=False)
    kw:dict[str,Any]={"headless":True,"attacker_roster":list(p.players),"defender_roster":list(GC_ROSTER)}
    sig=inspect.signature(VisualFPSBattle.__init__)
    if "disable_side_swap" in sig.parameters:kw["disable_side_swap"]=True
    if "spike_holder_name" in sig.parameters and p.spike_holder is not None:kw["spike_holder_name"]=p.spike_holder
    if "attacker_igl_name" in sig.parameters:kw["attacker_igl_name"]=p.igl
    if "attacker_team_name" in sig.parameters:kw["attacker_team_name"]=p.name
    if "defender_team_name" in sig.parameters:kw["defender_team_name"]=GC_PRESET
    game=VisualFPSBattle(NEW_MAZE_STR,attacker,defender,**kw);ctrl.set_game(game)
    return game,name


def step_setup(game):
    if hasattr(game,"_run_defender_setup_tick"):game._run_defender_setup_tick()
    else:
        for c in game.chars:
            if c.is_alive and c.team=="D":game._move_character_during_defender_setup(c)
        game.defender_setup_phase.advance_tick()


def step_live(game):
    if hasattr(game,"_build_occupancy_counts"):game._build_occupancy_counts()
    try:
        order=game._move_order() if hasattr(game,"_move_order") else game.chars
        for c in order:
            if c.is_alive:game.move_character(c)
    finally:
        if hasattr(game,"_clear_occupancy_counts"):game._clear_occupancy_counts()
    game.process_battle()
    if hasattr(game,"_advance_combo_announcement"):game._advance_combo_announcement()


def run_episode(sel,device,eps,rng,training):
    setup=LearningDefenderSetupGCRuntime(device=str(device),verbose=False)
    ctrl=DefenderOpeningController(setup,sel,device,eps,rng,training)
    game,opp=build_game(ctrl,rng)

    guard=0
    while getattr(game.defender_setup_phase,"active",False):
        step_setup(game);guard+=1
        if guard>100:raise RuntimeError("Setup did not terminate")

    ad0,aa0=len(alive(game.chars,"D")),len(alive(game.chars,"A"))
    hd0,ha0=hp_sum(game.chars,"D"),hp_sum(game.chars,"A")

    ticks=0
    for t in range(OPENING_TICKS):
        if game.round_over or game.match_over:break
        ctrl.tick=t;step_live(game);ticks+=1

    ad,aa=len(alive(game.chars,"D")),len(alive(game.chars,"A"))
    hd,ha=hp_sum(game.chars,"D"),hp_sum(game.chars,"A")
    alive_swing=(ad-aa)-(ad0-aa0);hp_swing=(hd-ha)-(hd0-ha0)
    reward=ctrl.local_reward+alive_swing*W_ALIVE+hp_swing*W_HP

    unused=0
    for p in ctrl.plans.values():
        if p.state not in {"EXECUTED","CANCELLED"}:reward+=P_UNUSED;unused+=1

    sel_t=[Transition(o,a,float(reward),n) for n,o,a in ctrl.sel_steps]
    return {
        "reward":float(reward),"local":float(ctrl.local_reward),
        "alive_swing":int(alive_swing),"hp_swing":float(hp_swing),"ticks":ticks,
        "opponent":opp,"plans":len(ctrl.plans),"unused":unused,
        "states":dict(Counter(p.state for p in ctrl.plans.values())),
        "executed_by_ability":dict(Counter(p.ability for p in ctrl.plans.values() if p.state=="EXECUTED")),
        "patterns":{n:(ctrl.plans[n].pid if n in ctrl.plans else None) for n in GC_ROSTER},
        "sel_t":sel_t,
    }


def optimize_sel(net,opt,replay,device,rng):
    if len(replay)<max(BATCH,LEARN_START):return None
    batch=replay.sample(BATCH,rng);opt.zero_grad(set_to_none=True);losses=[]
    for slot in GC_ROSTER:
        sub=[t for t in batch if t.slot==slot]
        if not sub:continue
        o=torch.from_numpy(np.stack([t.obs for t in sub])).float().to(device)
        a=torch.tensor([t.action for t in sub],dtype=torch.long,device=device)
        r=torch.tensor([t.reward for t in sub],dtype=torch.float32,device=device)
        q=net(o)[slot].gather(1,a[:,None]).squeeze(1)
        losses.append(F.smooth_l1_loss(q,r))
    if not losses:return None
    loss=torch.stack(losses).mean();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
    return float(loss.item())


def evaluate(sel,device,episodes,seed):
    sel.eval();rng=random.Random(seed)
    rows=[run_episode(sel,device,0,rng,False) for _ in range(episodes)]
    states=Counter();abilities=Counter();opps=Counter();patterns=Counter()
    for x in rows:
        states.update(x["states"]);abilities.update(x["executed_by_ability"]);opps[x["opponent"]]+=1
        for n,pid in x["patterns"].items():patterns[f"{n}:{pid}"]+=1
    out={
        "avg_opening_reward":float(np.mean([x["reward"] for x in rows])),
        "avg_local_reward":float(np.mean([x["local"] for x in rows])),
        "avg_alive_swing":float(np.mean([x["alive_swing"] for x in rows])),
        "avg_hp_swing":float(np.mean([x["hp_swing"] for x in rows])),
        "avg_ticks":float(np.mean([x["ticks"] for x in rows])),
        "avg_plans":float(np.mean([x["plans"] for x in rows])),
        "avg_unused":float(np.mean([x["unused"] for x in rows])),
        "states":dict(states),"executed_by_ability":dict(abilities),
        "patterns":dict(patterns),"opponents":dict(opps),
    }
    sel.train();return out


def save(path,sel,so,episode,best):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "model_type":"gc_defender_opening_macro_v2_stage2a_selection_only",
        "obs_dim":OBS_DIM,
        "slot_abilities":SLOT_ABILITY,
        "slot_action_dims":ACTION_DIMS,
        "slot_catalogs":CATALOG,
        "opening_horizon_ticks":OPENING_TICKS,
        "auto_execute_when_ready":True,
        "selection_state_dict":sel.state_dict(),
        "selection_optimizer_state_dict":so.state_dict(),
        "episode":episode,
        "best_opening_reward":best,
    },path)


def train(args):
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    rng=random.Random(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    sel=SelectionNet().to(device)
    so=torch.optim.AdamW(sel.parameters(),lr=LR)
    sr=Replay(REPLAY_CAP);best=-1e9
    sloss=deque(maxlen=100)

    print(f"device={device}")
    print(f"OBS_DIM={OBS_DIM}")
    print(f"GC slots={SLOT_ABILITY}")
    print(f"selection action dims={ACTION_DIMS}")
    print(f"opening horizon={OPENING_TICKS} Tick")
    print(f"opponent pool={len(opponent_names())} real teams / ai={ATTACKER_AI}")
    print("mode=OPENING_STAGE2A_SELECTION_ONLY (Setup -> select patterns -> auto execute -> stop)")
    print("Execution DQN/Search/Retake/final-round reward are disabled")

    try:
        for ep in range(1,args.episodes+1):
            eps=eps_value(ep);x=run_episode(sel,device,eps,rng,True)
            for t in x["sel_t"]:sr.add(t)
            a=optimize_sel(sel,so,sr,device,rng)
            if a is not None:sloss.append(a)

            if ep<=10 or ep%args.log_every==0:
                print(
                    f"[{ep:6d}/{args.episodes}] eps={eps:.3f} | "
                    f"R={x['reward']:+.3f} (local={x['local']:+.3f}, "
                    f"alive={x['alive_swing']:+d}, hp={x['hp_swing']:+.0f}) | "
                    f"plans={x['plans']} unused={x['unused']} | opp={x['opponent']} | "
                    f"replay={len(sr)} | "
                    f"loss={np.mean(sloss) if sloss else 0:.4f}"
                )

            if ep%args.save_every==0:save(LATEST,sel,so,ep,best)
            if ep%args.eval_every==0:
                m=evaluate(sel,device,args.eval_episodes,args.seed+ep*1000)
                print("[EVAL]",m)
                with LOG.open("a",encoding="utf-8") as f:f.write(json.dumps({"episode":ep,**m},ensure_ascii=False)+"\n")
                if m["avg_opening_reward"]>best:
                    best=m["avg_opening_reward"];save(BEST,sel,so,ep,best)
                    print(f"[BEST] opening_reward={best:+.4f}")

        save(FINAL,sel,so,args.episodes,best)
    except KeyboardInterrupt:
        save(INTERRUPT,sel,so,locals().get("ep",0),best)
        print(f"\n[INTERRUPT] saved: {INTERRUPT}")
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=3000)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--eval-episodes",type=int,default=30)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--log-every",type=int,default=25)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="auto")
    train(p.parse_args())


if __name__=="__main__":
    main()
