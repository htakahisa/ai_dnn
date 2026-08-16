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
    from character_stats import characters as CHARACTER_STATS
except Exception:
    try:
        from character_stats import CHARACTER_STATS
    except Exception:
        CHARACTER_STATS = None

try:
    from gc_v1.learning_defender_setup_gc_runtime import LearningDefenderSetupGCRuntime, SETUP_VARIATIONS, VARIATION_DIM
except ImportError:
    from learning_defender_setup_gc_runtime import LearningDefenderSetupGCRuntime, SETUP_VARIATIONS, VARIATION_DIM

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

# Flash / Recon must reach Origin with some time left to execute.
# Static map BFS only; live teammate/enemy blocking can still add delay.
MAX_ORIGIN_BFS_DISTANCE = 14

LR = 2.5e-4
BATCH = 128
REPLAY_CAP = 100_000
LEARN_START = 500
EPS_START, EPS_END, EPS_DECAY = 0.90, 0.05, 3000

# ------------------------------------------------------------------
# Stage 2A reward philosophy
# ------------------------------------------------------------------
# A legal/pre-approved set play that is successfully completed is "good enough".
# Whether the enemy happened to be there is mostly result noise at Opening time.
#
# Important: style reward is normalized by the number of selected plans later,
# so selecting 4 utilities is NOT automatically better than selecting 2.
R_EXECUTE = 0.08
R_RECON_EXECUTE = R_EXECUTE
R_ORIGIN = 0.00

# Recon hit/miss is kept as diagnostics only. It is NOT added to training reward.
R_RECON_NEW_UNIQUE = 0.0
R_RECON_HIT = 0.0
P_RECON_ZERO_EFFECT = 0.0

# Structural mistakes remain meaningful penalties.
P_IMPOSSIBLE = -0.20
P_UNUSED = -0.10
P_DUP_TARGET = -0.07

# Opening result is only a gentle tie-breaker.
# Clamp later so one lucky/unlucky fight cannot dominate set-play learning.
# Result is intentionally much weaker than set-play/style reward.
# Typical clean set-play completion is around +0.08, while the ENTIRE combat
# result contribution is capped to +/-0.02.
W_ALIVE = 0.015
W_HP = 0.00005
OUTCOME_ALIVE_CLIP = 1.0
OUTCOME_HP_CLIP = 100.0
OUTCOME_TOTAL_CLIP = 0.02

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "defender_opening_macro_gc_v2_data"
BEST = DATA_DIR / "dqn_defender_opening_macro_gc_v2_best.pt"
LATEST = DATA_DIR / "dqn_defender_opening_macro_gc_v2_latest.pt"
FINAL = DATA_DIR / "dqn_defender_opening_macro_gc_v2_final.pt"
INTERRUPT = DATA_DIR / "dqn_defender_opening_macro_gc_v2_interrupt.pt"
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

# Opponent-known-at-round-start features:
# role counts 4
# team averages: Hit / IQ / dodge / reaction / HS = 5
# max IQ = 1
# => 10
OPP_FEATURE_DIM = 10

# global 10 + opponent 10 + Setup variation 3 + players 5*8 + plans 5*8
OBS_DIM = 100 + VARIATION_DIM


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
    """Replan every Tick with current alive player positions blocked."""
    start=tuple(map(int,char.pos))
    goal=tuple(map(int,goal))
    if start==goal:
        return [start[0],start[1]]

    h,w=grid.shape
    blocked={
        tuple(map(int,other.pos))
        for other in chars
        if other is not char and getattr(other,"is_alive",True)
    }

    if goal in blocked:
        return [start[0],start[1]]

    q=deque([start])
    prev={start:None}

    while q:
        cur=q.popleft()
        if cur==goal:
            break
        r,c=cur
        for dr,dc in CARDINAL:
            nr,nc=r+dr,c+dc
            nxt=(nr,nc)
            if not (0<=nr<h and 0<=nc<w):
                continue
            if grid[nr,nc]==1:
                continue
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt]=cur
            q.append(nxt)

    if goal not in prev:
        return [start[0],start[1]]

    cur=goal
    while prev[cur] is not None and prev[cur]!=start:
        cur=prev[cur]
    return [int(cur[0]),int(cur[1])]


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
    origin_bfs_distance:int|None=None
    origin_candidates:list|None=None


def _stat_lookup(name):
    """Return one character stat mapping/object, tolerating current project layouts."""
    if CHARACTER_STATS is None:
        return None

    if isinstance(CHARACTER_STATS, dict):
        return CHARACTER_STATS.get(name)

    try:
        for item in CHARACTER_STATS:
            item_name = (
                item.get("name")
                if isinstance(item, dict)
                else getattr(item, "name", None)
            )
            if item_name == name:
                return item
    except Exception:
        pass

    return None


def _value(item, *keys, default=0.0):
    if item is None:
        return float(default)

    for key in keys:
        if isinstance(item, dict):
            if key in item:
                try:
                    return float(item[key])
                except Exception:
                    pass
        else:
            if hasattr(item, key):
                try:
                    return float(getattr(item, key))
                except Exception:
                    pass

    return float(default)


def _role_from_char(char):
    role = str(getattr(char, "ability_name", "") or "").upper()
    if role in {"SMOKE", "FLASH", "RECON", "HUNT", "TIGER"}:
        if role == "HUNT":
            role = "TIGER"
        return role
    return "OTHER"


def opponent_features(game):
    """Known opponent composition/stats only; no hidden tactical state.

    The opponent roster is known before the round, so these are legal macro inputs.
    Values are normalized to roughly 0..1 ranges.
    """
    attackers = [
        c for c in game.chars
        if getattr(c, "team", None) == "A"
    ]

    role_counts = {
        "SMOKE": 0,
        "FLASH": 0,
        "RECON": 0,
        "TIGER": 0,
    }
    for c in attackers:
        role = _role_from_char(c)
        if role in role_counts:
            role_counts[role] += 1

    hit = []
    iq = []
    dodge = []
    reaction = []
    hs = []

    for c in attackers:
        item = _stat_lookup(str(getattr(c, "name", "")))

        # Prefer live character attrs, then character_stats.
        hit.append(
            float(getattr(c, "hit_rate", getattr(c, "hit", _value(item, "hit_rate", "hit", "Hit", default=0.0))))
        )
        iq.append(
            float(getattr(c, "iq", _value(item, "iq", "IQ", default=0.0)))
        )
        dodge.append(
            float(getattr(c, "dodge", getattr(c, "evasion", _value(item, "dodge", "evasion", "回避", default=0.0))))
        )
        reaction.append(
            float(getattr(c, "reaction", getattr(c, "reaction_speed", _value(item, "reaction", "reaction_speed", "反応", default=0.0))))
        )
        hs.append(
            float(getattr(c, "hs_rate", getattr(c, "headshot_rate", _value(item, "hs_rate", "headshot_rate", "hs", "HS", default=0.0))))
        )

    def avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    # Existing game values are approximately:
    # hit/dodge/hs = fractions, IQ ~= 50..200+, reaction ~= 50..200.
    return np.asarray([
        role_counts["SMOKE"] / 5.0,
        role_counts["FLASH"] / 5.0,
        role_counts["RECON"] / 5.0,
        role_counts["TIGER"] / 5.0,
        np.clip(avg(hit), 0.0, 1.5) / 1.5,
        np.clip(avg(iq), 0.0, 250.0) / 250.0,
        np.clip(avg(dodge), 0.0, 1.0),
        np.clip(avg(reaction), 0.0, 250.0) / 250.0,
        np.clip(avg(hs), 0.0, 1.0),
        np.clip(max(iq) if iq else 0.0, 0.0, 250.0) / 250.0,
    ], dtype=np.float32)


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
    opp = opponent_features(game).tolist()
    variation_index = int(getattr(game, "gc_setup_variation_index", 0))
    variation_one_hot = [1.0 if i == variation_index else 0.0 for i in range(VARIATION_DIM)]
    obs=np.asarray(g+opp+variation_one_hot+pf+pl,np.float32)
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


def eps_action(q,eps,rng,valid_mask=None):
    q = np.asarray(q, dtype=np.float32)

    if valid_mask is None:
        valid = np.ones(len(q), dtype=np.bool_)
    else:
        valid = np.asarray(valid_mask, dtype=np.bool_)

    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        raise RuntimeError("Opening selection has no valid action")

    if rng.random() < eps:
        return int(rng.choice(indices.tolist()))

    masked = q.copy()
    masked[~valid] = -1e30
    return int(np.argmax(masked))


def selection_valid_mask(slot, char, game):
    """NONE + Opening horizon内に実行可能なPatternだけを許可する。"""
    catalog = CATALOG[slot]
    ability = SLOT_ABILITY[slot]
    mask = np.zeros(len(catalog), dtype=np.bool_)

    # NONE is always legal.
    mask[0] = True

    for action, pid in enumerate(catalog[1:], start=1):
        pattern = PATTERNS[ability][pid]

        if ability == "SMOKE":
            # Smoke has no Origin requirement.
            mask[action] = bool(pattern.get("targets"))
            continue

        origin, dist = nearest_origin(char, pattern, game.grid)
        mask[action] = (
            origin is not None
            and dist <= MAX_ORIGIN_BFS_DISTANCE
            and bool(pattern.get("targets"))
        )

    return mask


def opening_reachability_debug(game):
    """Human-readable current Setup-end -> Opening pattern BFS distances."""
    by = {str(c.name): c for c in game.chars}
    result = {}

    for slot in GC_ROSTER:
        char = by.get(slot)
        ability = SLOT_ABILITY[slot]
        rows = []

        if char is None:
            result[slot] = rows
            continue

        for pid in CATALOG[slot][1:]:
            pattern = PATTERNS[ability][pid]

            if ability == "SMOKE":
                rows.append((pid, 0, True))
            else:
                _, dist = nearest_origin(char, pattern, game.grid)
                rows.append(
                    (
                        pid,
                        int(dist),
                        bool(dist <= MAX_ORIGIN_BFS_DISTANCE),
                    )
                )

        result[slot] = rows

    return result


def dynamic_nearest_origin(char,plan,game):
    candidates=list(plan.origin_candidates or [])
    if not candidates:return plan.origin
    start=tuple(map(int,char.pos));h,w=game.grid.shape
    blocked={tuple(map(int,o.pos)) for o in game.chars if o is not char and getattr(o,"is_alive",True)}
    best=None;best_dist=10**9
    for goal in candidates:
        goal=tuple(map(int,goal))
        if goal in blocked and goal!=start:continue
        q=deque([start]);dist={start:0}
        while q:
            cur=q.popleft()
            if cur==goal:break
            r,c=cur
            for dr,dc in CARDINAL:
                nxt=(r+dr,c+dc);nr,nc=nxt
                if not (0<=nr<h and 0<=nc<w):continue
                if game.grid[nr,nc]==1 or nxt in dist:continue
                if nxt in blocked and nxt!=goal:continue
                dist[nxt]=dist[cur]+1;q.append(nxt)
        if goal in dist and dist[goal]<best_dist:
            best,best_dist=goal,dist[goal]
    return best


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
            valid_mask = selection_valid_mask(n, c, self.game)
            a=eps_action(
                heads[n],
                self.eps if self.training else 0,
                self.rng,
                valid_mask=valid_mask,
            )
            pid=CATALOG[n][a]
            if self.training:self.sel_steps.append((n,obs.copy(),a))
            if pid is None:continue
            ab=SLOT_ABILITY[n]; pat=PATTERNS[ab][pid]
            if ability_charge(c,ab)<=0:
                self.local_reward += P_IMPOSSIBLE; continue
            if ab=="SMOKE":
                origin=None; d=0; target=nearest_target(None,pat)
            else:
                origin,d=nearest_origin(c,pat,self.game.grid); target=nearest_target(origin,pat)
                if origin is None or d>=999:
                    self.local_reward += P_IMPOSSIBLE; continue
            if target is None:
                self.local_reward += P_IMPOSSIBLE; continue
            self.plans[n]=Plan(
                n,ab,pid,origin,target,
                origin_bfs_distance=(None if ab=="SMOKE" else int(d)),
                origin_candidates=(None if ab=="SMOKE" else [tuple(map(int,p)) for p in pat.get("origins",[])]),
            )
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
            new_origin=dynamic_nearest_origin(char,p,self.game)
            if new_origin is not None:p.origin=new_origin
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


def build_game(ctrl, rng, opponent_name=None, variation_index=None):
    name = str(opponent_name) if opponent_name is not None else rng.choice(opponent_names())
    p = get_preset(name)
    if p is None:
        raise RuntimeError(f"Unknown opponent preset: {name}")
    if variation_index is None:
        variation_index = rng.randrange(VARIATION_DIM)
    variation_index = int(variation_index)
    variation_name = SETUP_VARIATIONS[variation_index]
    if hasattr(ctrl.setup, "set_context"):
        ctrl.setup.set_context(opponent_name=name, variation=variation_index)
    attacker=_build_team_ai(ATTACKER_AI)
    defender=DualRoleTeamAI("GC-Opening-v2-D",attacker_factory=lambda:DefaultAttackerController(),defender_factory=lambda:ctrl,use_iq_perception=False)
    kw:dict[str,Any]={"headless":True,"attacker_roster":list(p.players),"defender_roster":list(GC_ROSTER)}
    sig=inspect.signature(VisualFPSBattle.__init__)
    if "disable_side_swap" in sig.parameters:kw["disable_side_swap"]=True
    if "spike_holder_name" in sig.parameters and p.spike_holder is not None:kw["spike_holder_name"]=p.spike_holder
    if "attacker_igl_name" in sig.parameters:kw["attacker_igl_name"]=p.igl
    if "attacker_team_name" in sig.parameters:kw["attacker_team_name"]=p.name
    if "defender_team_name" in sig.parameters:kw["defender_team_name"]=GC_PRESET
    game=VisualFPSBattle(NEW_MAZE_STR,attacker,defender,**kw)
    game.gc_setup_variation_index = variation_index
    game.gc_setup_variation = variation_name
    game.gc_opponent_name = name
    ctrl.set_game(game)
    return game,name,variation_name


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



def pattern_origin_distance_diagnostics(game):
    by={str(c.name):c for c in game.chars};out={}
    for slot in GC_ROSTER:
        c=by.get(slot);ability=SLOT_ABILITY[slot];out[slot]={}
        if c is None:continue
        for pid in CATALOG[slot][1:]:
            pat=PATTERNS[ability][pid]
            if ability=="SMOKE":out[slot][pid]=0
            else:
                _,d=nearest_origin(c,pat,game.grid)
                out[slot][pid]=None if d>=999 else int(d)
    return out


def install_recon_effect_tracker(game):
    """Measure actual GC Recon effects by wrapping the game's own explosion."""
    tracker = {
        "team_unique_names": set(),
        "hits": 0,
        "casts_with_effect": 0,
        "casts_zero_effect": 0,
        "by_caster": {
            "Xdll": {"casts": 0, "hits": 0, "new_unique": 0, "unique_names": set()},
            "eKo": {"casts": 0, "hits": 0, "new_unique": 0, "unique_names": set()},
        },
    }

    original = getattr(game, "_explode_recon", None)
    if original is None:
        raise RuntimeError("Game has no _explode_recon; cannot score Recon effect")

    def tracked_explode_recon(projectile, impact=None):
        owner = str(projectile.get("owner", ""))
        owner_team = projectile.get("team")
        enemies = [
            c for c in game.chars
            if getattr(c, "is_alive", True)
            and getattr(c, "team", None) != owner_team
        ]
        before = {
            str(c.name): float(getattr(c, "reveal_remaining", 0) or 0)
            for c in enemies
        }

        result = original(projectile, impact)

        affected = []
        for c in enemies:
            name = str(c.name)
            after = float(getattr(c, "reveal_remaining", 0) or 0)
            if after > before.get(name, 0.0):
                affected.append(name)

        # Ignore attacker Recon completely. Only GC Xdll/eKo count.
        if owner not in tracker["by_caster"]:
            return result

        caster = tracker["by_caster"][owner]
        caster["casts"] += 1
        caster["hits"] += len(affected)
        caster_new = [n for n in affected if n not in caster["unique_names"]]
        caster["new_unique"] += len(caster_new)
        caster["unique_names"].update(caster_new)

        if affected:
            tracker["casts_with_effect"] += 1
            tracker["hits"] += len(affected)
            tracker["team_unique_names"].update(affected)
        else:
            tracker["casts_zero_effect"] += 1

        return result

    game._explode_recon = tracked_explode_recon
    return tracker

def run_episode(sel,device,eps,rng,training,opponent_name=None,variation_index=None):
    setup=LearningDefenderSetupGCRuntime(device=str(device),verbose=False)
    ctrl=DefenderOpeningController(setup,sel,device,eps,rng,training)
    game,opp,variation=build_game(ctrl,rng,opponent_name=opponent_name,variation_index=variation_index)

    guard=0
    while getattr(game.defender_setup_phase,"active",False):
        step_setup(game);guard+=1
        if guard>100:raise RuntimeError("Setup did not terminate")

    pattern_origin_distances=pattern_origin_distance_diagnostics(game)
    recon_tracker=install_recon_effect_tracker(game)

    ad0,aa0=len(alive(game.chars,"D")),len(alive(game.chars,"A"))
    hd0,ha0=hp_sum(game.chars,"D"),hp_sum(game.chars,"A")

    ticks=0
    for t in range(OPENING_TICKS):
        if game.round_over or game.match_over:break
        ctrl.tick=t;step_live(game);ticks+=1

    ad,aa=len(alive(game.chars,"D")),len(alive(game.chars,"A"))
    hd,ha=hp_sum(game.chars,"D"),hp_sum(game.chars,"A")
    alive_swing=(ad-aa)-(ad0-aa0);hp_swing=(hd-ha)-(hd0-ha0)

    recon_unique=len(recon_tracker["team_unique_names"])
    recon_hits=int(recon_tracker["hits"])
    recon_zero=int(recon_tracker["casts_zero_effect"])

    # Diagnostic only. Recon hit/miss must not decide whether the set play was
    # "correct", because Opening has no information that can reliably predict it.
    recon_effect_reward=0.0

    # Normalize set-play execution so "more utility" is not inherently better.
    selected_plan_count=len(ctrl.plans)
    if selected_plan_count > 0:
        style_reward=float(ctrl.local_reward) / float(selected_plan_count)
    else:
        style_reward=0.0

    # Result is deliberately weak and clipped: only a long-run tie-breaker
    # between otherwise plausible set plays.
    clipped_alive=float(np.clip(alive_swing,-OUTCOME_ALIVE_CLIP,OUTCOME_ALIVE_CLIP))
    clipped_hp=float(np.clip(hp_swing,-OUTCOME_HP_CLIP,OUTCOME_HP_CLIP))
    raw_outcome_reward=clipped_alive*W_ALIVE + clipped_hp*W_HP
    outcome_reward=float(np.clip(
        raw_outcome_reward,
        -OUTCOME_TOTAL_CLIP,
        OUTCOME_TOTAL_CLIP,
    ))

    reward=style_reward + outcome_reward

    unused=0
    for p in ctrl.plans.values():
        if p.state not in {"EXECUTED","CANCELLED"}:reward+=P_UNUSED;unused+=1

    sel_t=[Transition(o,a,float(reward),n) for n,o,a in ctrl.sel_steps]
    return {
        "reward":float(reward),"local":float(ctrl.local_reward),
        "style_reward":float(style_reward),
        "raw_outcome_reward":float(raw_outcome_reward),
        "outcome_reward":float(outcome_reward),
        "recon_effect_reward":float(recon_effect_reward),
        "recon_unique_revealed":int(recon_unique),
        "recon_hits":int(recon_hits),
        "recon_zero_effect_casts":int(recon_zero),
        "recon_by_caster":{
            caster:{
                "casts":int(stats["casts"]),
                "hits":int(stats["hits"]),
                "new_unique":int(stats["new_unique"]),
            }
            for caster,stats in recon_tracker["by_caster"].items()
        },
        "alive_swing":int(alive_swing),"hp_swing":float(hp_swing),"ticks":ticks,
        "opponent":opp,"setup_variation":variation,"setup_assignments":dict(getattr(setup,"assignments",{})),"plans":len(ctrl.plans),"unused":unused,
        "states":dict(Counter(p.state for p in ctrl.plans.values())),
        "executed_by_ability":dict(Counter(p.ability for p in ctrl.plans.values() if p.state=="EXECUTED")),
        "patterns":{n:(ctrl.plans[n].pid if n in ctrl.plans else None) for n in GC_ROSTER},
        "pattern_origin_distances":pattern_origin_distances,
        "plan_details":{
            n:{
                "pid":ctrl.plans[n].pid,
                "ability":ctrl.plans[n].ability,
                "state":ctrl.plans[n].state,
                "origin_bfs_distance":ctrl.plans[n].origin_bfs_distance,
                "executed_tick":ctrl.plans[n].executed_tick,
            } for n in ctrl.plans
        },
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


def evaluate(sel,device,episodes,seed,detail=True):
    sel.eval(); rng=random.Random(seed)

    matrix=[(opp,vi) for opp in opponent_names() for vi in range(VARIATION_DIM)]
    requested=max(int(episodes),len(matrix))
    schedule=(matrix*((requested+len(matrix)-1)//len(matrix)))[:requested]
    rows=[
        run_episode(
            sel,device,0,rng,False,
            opponent_name=opp,
            variation_index=vi,
        )
        for opp,vi in schedule
    ]

    states=Counter()
    abilities=Counter()
    patterns=Counter()
    variations=Counter()

    for x in rows:
        states.update(x["states"])
        abilities.update(x["executed_by_ability"])
        variations[x["setup_variation"]]+=1
        for n,pid in x["patterns"].items():
            patterns[f"{n}:{pid}"]+=1

    recon_casts=sum(
        x["recon_by_caster"]["Xdll"]["casts"]
        + x["recon_by_caster"]["eKo"]["casts"]
        for x in rows
    )
    recon_effective_casts=sum(
        (
            x["recon_by_caster"]["Xdll"]["casts"]
            + x["recon_by_caster"]["eKo"]["casts"]
            - x["recon_zero_effect_casts"]
        )
        for x in rows
    )

    base={
        "eval_count":len(rows),
        "eval_matrix_size":len(matrix),
        "avg_opening_reward":float(np.mean([x["reward"] for x in rows])),
        "avg_local_reward":float(np.mean([x["local"] for x in rows])),
        "avg_style_reward":float(np.mean([x["style_reward"] for x in rows])),
        "avg_raw_outcome_reward":float(np.mean([x["raw_outcome_reward"] for x in rows])),
        "avg_outcome_reward":float(np.mean([x["outcome_reward"] for x in rows])),
        "avg_recon_effect_reward":float(np.mean([x["recon_effect_reward"] for x in rows])),
        "avg_recon_unique_revealed":float(np.mean([x["recon_unique_revealed"] for x in rows])),
        "avg_recon_hits":float(np.mean([x["recon_hits"] for x in rows])),
        "recon_zero_effect_casts":int(sum(x["recon_zero_effect_casts"] for x in rows)),
        "recon_effective_cast_rate":float(recon_effective_casts/max(1,recon_casts)),
        "recon_by_caster":{
            caster:{
                "casts":sum(x["recon_by_caster"][caster]["casts"] for x in rows),
                "hits":sum(x["recon_by_caster"][caster]["hits"] for x in rows),
                "new_unique":sum(x["recon_by_caster"][caster]["new_unique"] for x in rows),
            }
            for caster in ("Xdll","eKo")
        },
        "avg_alive_swing":float(np.mean([x["alive_swing"] for x in rows])),
        "avg_hp_swing":float(np.mean([x["hp_swing"] for x in rows])),
        "avg_ticks":float(np.mean([x["ticks"] for x in rows])),
        "avg_plans":float(np.mean([x["plans"] for x in rows])),
        "avg_unused":float(np.mean([x["unused"] for x in rows])),
        "states":dict(states),
        "executed_by_ability":dict(abilities),
        "patterns":dict(patterns),
        "setup_variations":dict(variations),
    }

    if not detail:
        sel.train()
        return base

    opponent_patterns={}
    variation_patterns={}
    opponent_variation={}
    opps=Counter()
    ds={n:[] for n in GC_ROSTER}
    de={n:[] for n in GC_ROSTER}
    df={n:[] for n in GC_ROSTER}
    pattern_samples={n:{} for n in GC_ROSTER}

    for x in rows:
        opps[x["opponent"]]+=1
        sig=" | ".join(f"{n}={x['patterns'].get(n)}" for n in GC_ROSTER)
        opponent_patterns.setdefault(x["opponent"],Counter())[sig]+=1
        variation_patterns.setdefault(x["setup_variation"],Counter())[sig]+=1

        key=f"{x['opponent']} | {x['setup_variation']}"
        cell=opponent_variation.setdefault(
            key,
            {"rows":[],"patterns":Counter(),"abilities":Counter(),"setup_assignments":Counter()}
        )
        cell["rows"].append(x)
        cell["patterns"][sig]+=1
        cell["abilities"].update(x["executed_by_ability"])
        setup_sig=" | ".join(
            f"{n}={x['setup_assignments'].get(n)}"
            for n in GC_ROSTER
        )
        cell["setup_assignments"][setup_sig]+=1

        for n,per in x["pattern_origin_distances"].items():
            for pid,dist in per.items():
                if dist is not None:
                    pattern_samples[n].setdefault(pid,[]).append(float(dist))

        for n,d in x["plan_details"].items():
            dist=d.get("origin_bfs_distance")
            if dist is None:
                continue
            ds[n].append(float(dist))
            if d.get("state")=="EXECUTED":
                de[n].append(float(dist))
            else:
                df[n].append(float(dist))

    avg=lambda xs: None if not xs else float(np.mean(xs))

    origin_distance={
        n:{
            "selected_avg":avg(ds[n]),
            "executed_avg":avg(de[n]),
            "failed_avg":avg(df[n]),
            "selected_n":len(ds[n]),
            "executed_n":len(de[n]),
            "failed_n":len(df[n]),
        }
        for n in GC_ROSTER
    }

    pattern_origin_distance={
        n:{
            pid:{
                "avg":float(np.mean(v)),
                "min":int(min(v)),
                "max":int(max(v)),
                "n":len(v),
                "selectable_rate":float(np.mean([
                    d<=MAX_ORIGIN_BFS_DISTANCE for d in v
                ])),
            }
            for pid,v in per.items()
        }
        for n,per in pattern_samples.items()
    }

    def summarize(sub):
        executed=sum(sum(x["executed_by_ability"].values()) for x in sub)
        planned=sum(x["plans"] for x in sub)
        rc=sum(
            x["recon_by_caster"]["Xdll"]["casts"]
            + x["recon_by_caster"]["eKo"]["casts"]
            for x in sub
        )
        reff=sum(
            (
                x["recon_by_caster"]["Xdll"]["casts"]
                + x["recon_by_caster"]["eKo"]["casts"]
                - x["recon_zero_effect_casts"]
            )
            for x in sub
        )
        return {
            "n":len(sub),
            "avg_opening_reward":float(np.mean([x["reward"] for x in sub])),
            "avg_local_reward":float(np.mean([x["local"] for x in sub])),
            "avg_style_reward":float(np.mean([x["style_reward"] for x in sub])),
            "avg_raw_outcome_reward":float(np.mean([x["raw_outcome_reward"] for x in sub])),
            "avg_outcome_reward":float(np.mean([x["outcome_reward"] for x in sub])),
            "avg_recon_effect_reward":float(np.mean([x["recon_effect_reward"] for x in sub])),
            "avg_recon_unique_revealed":float(np.mean([x["recon_unique_revealed"] for x in sub])),
            "avg_recon_hits":float(np.mean([x["recon_hits"] for x in sub])),
            "recon_effective_cast_rate":float(reff/max(1,rc)),
            "recon_by_caster":{
                caster:{
                    "casts":sum(x["recon_by_caster"][caster]["casts"] for x in sub),
                    "hits":sum(x["recon_by_caster"][caster]["hits"] for x in sub),
                    "new_unique":sum(x["recon_by_caster"][caster]["new_unique"] for x in sub),
                }
                for caster in ("Xdll","eKo")
            },
            "avg_alive_swing":float(np.mean([x["alive_swing"] for x in sub])),
            "avg_hp_swing":float(np.mean([x["hp_swing"] for x in sub])),
            "avg_plans":float(np.mean([x["plans"] for x in sub])),
            "avg_unused":float(np.mean([x["unused"] for x in sub])),
            "execution_rate":float(executed/max(1,planned)),
            "executed_by_ability":dict(
                sum((Counter(x["executed_by_ability"]) for x in sub),Counter())
            ),
        }

    by_variation={
        v:summarize([x for x in rows if x["setup_variation"]==v])
        for v in SETUP_VARIATIONS
    }
    by_opponent={
        opp:{
            v:summarize([
                x for x in rows
                if x["opponent"]==opp and x["setup_variation"]==v
            ])
            for v in SETUP_VARIATIONS
        }
        for opp in opponent_names()
    }

    matrix_detail={}
    for key,cell in opponent_variation.items():
        s=summarize(cell["rows"])
        s["patterns"]=dict(cell["patterns"])
        s["setup_assignments"]=dict(cell["setup_assignments"])
        matrix_detail[key]=s

    base.update({
        "by_variation":by_variation,
        "by_opponent":by_opponent,
        "opponent_variation":matrix_detail,
        "origin_distance":origin_distance,
        "pattern_origin_distance":pattern_origin_distance,
        "variation_patterns":{k:dict(v) for k,v in variation_patterns.items()},
        "opponent_patterns":{k:dict(v) for k,v in opponent_patterns.items()},
        "opponents":dict(opps),
    })

    sel.train()
    return base


def save(path,sel,so,episode,best):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "model_type":"gc_defender_opening_macro_v2_stage2a_opponent_aware",
        "obs_dim":OBS_DIM,
        "opponent_feature_dim":OPP_FEATURE_DIM,
        "setup_variations":list(SETUP_VARIATIONS),
        "setup_variation_dim":VARIATION_DIM,
        "opponent_features":[
            "smoke_count","flash_count","recon_count","tiger_count",
            "avg_hit","avg_iq","avg_dodge","avg_reaction","avg_hs","max_iq",
        ],
        "slot_abilities":SLOT_ABILITY,
        "slot_action_dims":ACTION_DIMS,
        "slot_catalogs":CATALOG,
        "opening_horizon_ticks":OPENING_TICKS,
        "max_origin_bfs_distance":MAX_ORIGIN_BFS_DISTANCE,
        "dynamic_opening_bfs":True,
        "dynamic_pattern_origin_retarget":True,
        "auto_execute_when_ready":True,
        "reward_philosophy":"setplay_style_primary_outcome_soft",
        "style_reward":{
            "execute_per_plan":R_EXECUTE,
            "normalized_by_selected_plans":True,
            "origin":R_ORIGIN,
        },
        "outcome_reward":{
            "alive_weight":W_ALIVE,
            "hp_weight":W_HP,
            "alive_clip":OUTCOME_ALIVE_CLIP,
            "hp_clip":OUTCOME_HP_CLIP,
            "total_clip":OUTCOME_TOTAL_CLIP,
        },
        "recon_effect_reward":{
            "training_enabled":False,
            "diagnostics_only":True,
        },
        "selection_state_dict":sel.state_dict(),
        "selection_optimizer_state_dict":so.state_dict(),
        "episode":episode,
        "best_opening_reward":best,
    },path)



def print_eval_compact(ep, metrics):
    xdll=metrics["recon_by_caster"]["Xdll"]
    eko=metrics["recon_by_caster"]["eKo"]
    print(
        f"[EVAL {ep}] "
        f"R={metrics['avg_opening_reward']:+.3f} "
        f"style={metrics['avg_style_reward']:+.3f} "
        f"outcome={metrics['avg_outcome_reward']:+.3f} "
        f"unused={metrics['avg_unused']:.2f} | "
        f"alive={metrics['avg_alive_swing']:+.3f} "
        f"hp={metrics['avg_hp_swing']:+.1f} | "
        f"Recon Xdll={xdll['hits']}/{xdll['casts']} "
        f"eKo={eko['hits']}/{eko['casts']}"
    )


def apply_fast_preset(args):
    if not getattr(args, "fast", False):
        return args
    args.eval_every = max(1000, int(args.episodes // 2))
    args.eval_episodes = 45
    args.save_every = 1000
    args.log_every = 0
    return args

def train(args):
    args=apply_fast_preset(args)
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
    print("opponent features=roles + avg(hit/iq/dodge/reaction/hs) + maxIQ")
    print(f"selection action dims={ACTION_DIMS}")
    print(f"opening horizon={OPENING_TICKS} Tick")
    print(f"Flash/Recon selectable only when Origin BFS<={MAX_ORIGIN_BFS_DISTANCE}")
    print("Opening movement=dynamic occupied-cell BFS each Tick")
    print("Pattern Origin=dynamic re-selection every Tick")
    print("eval diagnostics=Setup variation -> patterns/abilities -> execution rate -> alive/hp swing")
    print(f"opponent pool={len(opponent_names())} real teams / ai={ATTACKER_AI}")
    print("mode=OPENING_STAGE2A_CONTEXTUAL (opponent + Setup variation -> selection -> auto execute -> stop)")
    print("Execution DQN/Search/Retake/final-round reward are disabled")
    if args.fast:
        print(
            f"FAST mode=ON | episode log=OFF | "
            f"eval every={args.eval_every}E x {args.eval_episodes} | "
            f"save every={args.save_every}E | full diagnostics=FINAL only"
        )
    else:
        print("intermediate EVAL=compact only; full diagnostics=FINAL only")
    print(
        "reward=set-play style primary / result is a very soft clipped tie-breaker "
        f"(execute={R_EXECUTE:+.2f}, aliveW={W_ALIVE:.3f}, "
        f"hpW={W_HP:.5f}, outcomeCap=+/-{OUTCOME_TOTAL_CLIP:.2f})"
    )
    print("Recon hit/miss=diagnostics only; no training reward/penalty")

    try:
        for ep in range(1,args.episodes+1):
            eps=eps_value(ep);x=run_episode(sel,device,eps,rng,True)
            for t in x["sel_t"]:sr.add(t)
            a=optimize_sel(sel,so,sr,device,rng)
            if a is not None:sloss.append(a)

            if args.log_every>0 and (ep<=10 or ep%args.log_every==0):
                print(
                    f"[{ep:6d}/{args.episodes}] eps={eps:.3f} | "
                    f"R={x['reward']:+.3f} (local={x['local']:+.3f}, "
                    f"alive={x['alive_swing']:+d}, hp={x['hp_swing']:+.0f}) | "
                    f"plans={x['plans']} unused={x['unused']} | opp={x['opponent']} | setup={x['setup_variation']} | "
                    f"replay={len(sr)} | "
                    f"loss={np.mean(sloss) if sloss else 0:.4f}"
                )

            if ep%args.save_every==0:save(LATEST,sel,so,ep,best)
            if ep%args.eval_every==0:
                m=evaluate(
                    sel,device,args.eval_episodes,
                    args.seed+ep*1000,
                    detail=False,
                )
                print_eval_compact(ep,m)
                if not args.fast:
                    compact_log={
                        "episode":ep,
                        "avg_opening_reward":m["avg_opening_reward"],
                        "avg_style_reward":m["avg_style_reward"],
                        "avg_outcome_reward":m["avg_outcome_reward"],
                        "avg_alive_swing":m["avg_alive_swing"],
                        "avg_hp_swing":m["avg_hp_swing"],
                        "avg_unused":m["avg_unused"],
                        "recon_effective_cast_rate":m["recon_effective_cast_rate"],
                        "recon_by_caster":m["recon_by_caster"],
                    }
                    with LOG.open("a",encoding="utf-8") as f:
                        f.write(json.dumps(compact_log,ensure_ascii=False)+"\n")
                if m["avg_opening_reward"]>best:
                    best=m["avg_opening_reward"];save(BEST,sel,so,ep,best)
                    print(f"[BEST] opening_reward={best:+.4f}")

        save(FINAL,sel,so,args.episodes,best)
        print("="*72)
        print("FINAL DETAILED EVAL")
        print("="*72)
        final_eval=evaluate(
            sel,device,args.eval_episodes,
            args.seed+args.episodes*1000+777,
            detail=True,
        )
        print("[FINAL EVAL]",final_eval)
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
    p.add_argument(
        "--fast",
        action="store_true",
        help="5000E向け高速学習: episodeログOFF、途中EVAL最小、詳細EVALは最後だけ",
    )
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="auto")
    train(p.parse_args())


if __name__=="__main__":
    main()
