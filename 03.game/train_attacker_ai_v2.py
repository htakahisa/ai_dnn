# train_attacker_ai_v2.py
"""共有DQNで5人のアタッカーを学習するAI ver2.0用環境。

- 5人全員が同じDueling DQNを共有
- 移動、待機/プラント、近/遠4方向アビリティをDQNが選択
- 自動迂回、固定護衛、ルールベースのアビリティ判断は不使用
- 壁、同一マス禁止、残数などはゲームルールとして検証

実行例:
    python train_attacker_ai_v2.py --episodes 5000
    python train_attacker_ai_v2.py --resume attacker_ai_v2_data/training_state_latest.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from map_data import NEW_MAZE_STR

Position = Tuple[int, int]
TEAM_ATTACKER = "A"
TEAM_DEFENDER = "D"

ABILITY_NONE = "NONE"
ABILITY_FLASH = "FLASH"
ABILITY_SMOKE = "SMOKE"
ABILITY_RECON = "RECON"
ABILITY_HUNT = "HUNT"

# ---------------------------------------------------------------------
# 学習環境・実戦推論で共有する観測定義
# ---------------------------------------------------------------------

ABILITY_TO_INDEX = {
    ABILITY_NONE: 0,
    ABILITY_FLASH: 1,
    ABILITY_SMOKE: 2,
    ABILITY_RECON: 3,
    ABILITY_HUNT: 4,
}

# 味方1人分:
# alive, relative_row, relative_col, hp, has_spike, moved
TEAMMATE_SLOT_DIM = 6

# 敵1人分:
# exists, visible, remembered, relative_row, relative_col,
# hp, memory_age, revealed
ENEMY_SLOT_DIM = 8

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_WAIT_OR_PLANT = 4
ACTION_ABILITY_UP_NEAR = 5
ACTION_ABILITY_DOWN_NEAR = 6
ACTION_ABILITY_LEFT_NEAR = 7
ACTION_ABILITY_RIGHT_NEAR = 8
ACTION_ABILITY_UP_FAR = 9
ACTION_ABILITY_DOWN_FAR = 10
ACTION_ABILITY_LEFT_FAR = 11
ACTION_ABILITY_RIGHT_FAR = 12
N_ACTIONS = 13

MOVE_DELTAS: Dict[int, Position] = {
    ACTION_UP: (-1, 0), ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1), ACTION_RIGHT: (0, 1),
}
ABILITY_ACTIONS: Dict[int, Tuple[int, int, int]] = {
    ACTION_ABILITY_UP_NEAR: (-1, 0, 4),
    ACTION_ABILITY_DOWN_NEAR: (1, 0, 4),
    ACTION_ABILITY_LEFT_NEAR: (0, -1, 4),
    ACTION_ABILITY_RIGHT_NEAR: (0, 1, 4),
    ACTION_ABILITY_UP_FAR: (-1, 0, 10),
    ACTION_ABILITY_DOWN_FAR: (1, 0, 10),
    ACTION_ABILITY_LEFT_FAR: (0, -1, 10),
    ACTION_ABILITY_RIGHT_FAR: (0, 1, 10),
}
ACTION_NAMES = {
    0: "MOVE_UP", 1: "MOVE_DOWN", 2: "MOVE_LEFT", 3: "MOVE_RIGHT",
    4: "WAIT_OR_PLANT", 5: "ABILITY_UP_NEAR", 6: "ABILITY_DOWN_NEAR",
    7: "ABILITY_LEFT_NEAR", 8: "ABILITY_RIGHT_NEAR",
    9: "ABILITY_UP_FAR", 10: "ABILITY_DOWN_FAR",
    11: "ABILITY_LEFT_FAR", 12: "ABILITY_RIGHT_FAR",
}

N_ATTACKERS = 5
N_DEFENDERS = 5
MAX_HP = 100
BODY_DAMAGE = 40
HEADSHOT_DAMAGE = 160
ROUND_DURATION_TICKS = 90
PLANT_REQUIRED_TICKS = 4
DEFUSE_REQUIRED_TICKS = 9
SPIKE_DETONATION_TICKS = 45
SMOKE_DURATION_TICKS = 15
SMOKE_RADIUS = 1
BLIND_DURATION_TICKS = 3
REVEAL_DURATION_TICKS = 5
RECON_RADIUS = 4
MOVING_ACCURACY_MULTIPLIER = 0.50
MOVING_TARGET_HIT_MULTIPLIER = 0.70
BLIND_ACCURACY_MULTIPLIER = 0.30
REVEALED_DODGE_MULTIPLIER = 0.50
ENEMY_MEMORY_TICKS = 15
K_ENEMIES = 5
MAX_EPISODE_STEPS = 145

BASE_OBS_DIM = 120
ABILITY_VECTOR_DIM = len(ABILITY_TO_INDEX)

OBS_DIM = BASE_OBS_DIM + ABILITY_VECTOR_DIM

DEFAULT_EPISODES = 5000
DEFAULT_SEED = 42
GAMMA = 0.99
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 256
REPLAY_CAPACITY = 300_000
LEARNING_STARTS = 5_000
TRAIN_EVERY_AGENT_STEPS = 4
TARGET_UPDATE_INTERVAL = 2_000
GRADIENT_CLIP_NORM = 10.0
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_AGENT_STEPS = 350_000
EVAL_INTERVAL_EPISODES = 100
EVAL_EPISODES = 30
SAVE_INTERVAL_EPISODES = 100

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "attacker_ai_v2_data"
LATEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v2_latest.pt"
BEST_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v2_best.pt"
FINAL_MODEL_PATH = MODEL_DIR / "dqn_attacker_ai_v2_final.pt"
TRAINING_STATE_PATH = MODEL_DIR / "training_state_latest.pt"
CONFIG_PATH = MODEL_DIR / "training_config.json"

# 報酬
R_STEP = -0.01
R_WAIT = -0.10

R_TOWARD = 0.8
R_AWAY = -0.4
R_OUT = -1.0
R_WALL = -0.8
R_ALLY = -1.2
R_CONFLICT = -1.0
R_DAMAGE_DEALT = 0.18
R_DAMAGE_TAKEN = -0.10

R_KILL = 10.0
R_HS_KILL = 2.0

R_DEATH = -18.0
R_TEAMMATE_DEATH = -2.0
R_ENTER_PLANT_SITE = 15.0
R_PLANT_PROGRESS = 4.0
R_PLANT_CARRIER = 80.0
R_PLANT_TEAM = 20.0
R_INVALID_PLANT = -1.0
R_PLANT_INTERRUPTED = -3.0
R_ABILITY_OK = 1.0
R_ABILITY_NO_CHARGE = -1.0
R_ABILITY_INVALID = -0.8
R_ABILITY_WASTED = -0.5
R_FLASH_ENEMY = 4.0
R_FLASH_ALLY = -4.0
R_RECON_ENEMY = 3.0
R_SMOKE_USEFUL = 2.0
R_DEFUSE_START = -2.0
R_DEFUSE_PROGRESS = -1.2
R_DEFUSER_KILL = 8.0
R_WIN = 35.0
R_LOSS = -45.0
R_DETONATE = 80.0
R_ELIMINATE = 5.0


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def parse_grid(text: str) -> np.ndarray:
    rows = [x.strip() for x in text.strip().splitlines() if x.strip()]
    if not rows or any(len(x) != len(rows[0]) for x in rows):
        raise ValueError("NEW_MAZE_STRの形が不正です")
    return np.asarray([[int(ch) for ch in row] for row in rows], dtype=np.int8)


def cheb(a: Position, b: Position) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))


def line_cells(a: Position, b: Position) -> List[Position]:
    r0, c0 = a
    r1, c1 = b
    out = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    if dc > dr:
        err = dc // 2
        while c0 != c1:
            out.append((r0, c0))
            err -= dr
            if err < 0:
                r0 += sr
                err += dc
            c0 += sc
    else:
        err = dr // 2
        while r0 != r1:
            out.append((r0, c0))
            err -= dc
            if err < 0:
                c0 += sc
                err += dr
            r0 += sr

    out.append((r1, c1))
    return out


def bresenham_cells(
    start: Position,
    end: Position,
) -> List[Position]:
    return line_cells(start, end)


def safe_normalized_distance(
    distance: float,
    scale: float,
) -> float:
    if not np.isfinite(distance):
        return 1.0

    safe_scale = max(float(scale), 1.0)
    return float(min(max(float(distance) / safe_scale, 0.0), 1.0))

def has_los(a: Position, b: Position, grid: np.ndarray, smokes: Sequence["SmokeArea"]) -> bool:
    for i,(r,c) in enumerate(line_cells(a,b)):
        if i == 0: continue
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]): return False
        if grid[r,c] == 1: return False
        if any(s.remaining > 0 and s.contains((r,c)) for s in smokes): return False
    return True


def bfs(goal: Position, grid: np.ndarray) -> np.ndarray:
    h,w=grid.shape; d=np.full((h,w),np.inf,dtype=np.float32)
    if not (0<=goal[0]<h and 0<=goal[1]<w) or grid[goal]==1: return d
    q: Deque[Position]=deque([goal]); d[goal]=0
    while q:
        r,c=q.popleft()
        for dr,dc in MOVE_DELTAS.values():
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and grid[nr,nc]!=1 and d[nr,nc]>d[r,c]+1:
                d[nr,nc]=d[r,c]+1; q.append((nr,nc))
    return d


def epsilon_by_steps(steps: int) -> float:
    f=min(max(steps/EPSILON_DECAY_AGENT_STEPS,0.0),1.0)
    return EPSILON_START + f*(EPSILON_END-EPSILON_START)


@dataclass
class CharacterState:
    uid: int
    name: str
    team: str
    pos: Position
    hp: int = MAX_HP
    alive: bool = True
    accuracy: float = 0.58
    dodge: float = 0.15
    hs: float = 0.30
    reaction: float = 100.0
    ability: str = ABILITY_FLASH
    charges: int = 1
    has_spike: bool = False
    planting: bool = False
    plant_timer: int = 0
    defuse_timer: int = 0
    blind: int = 0
    reveal: int = 0
    moved: bool = False
    last_action: int = ACTION_WAIT_OR_PLANT
    move_failed: bool = False
    fail_reason: str = ""
    kills: int = 0
    deaths: int = 0

    def tick_reset(self) -> None:
        self.moved=False; self.move_failed=False; self.fail_reason=""


@dataclass
class SmokeArea:
    center: Position
    remaining: int
    owner_uid: int
    radius: int = SMOKE_RADIUS
    def contains(self, p: Position) -> bool: return cheb(self.center,p) <= self.radius


@dataclass
class MemoryEntry:
    pos: Position
    hp: int
    age: int
    visible: bool
    revealed: bool


class EnemyMemoryTracker:
    def __init__(self, memory_ticks: int=ENEMY_MEMORY_TICKS, k: int=K_ENEMIES):
        self.memory_ticks=memory_ticks; self.k=k; self.memory: Dict[int,MemoryEntry]={}
    def reset(self) -> None: self.memory.clear()
    def update(self, observer: CharacterState, enemies: Sequence[CharacterState], grid: np.ndarray, smokes: Sequence[SmokeArea]) -> None:
        visible: Set[int]=set()
        for e in enemies:
            if not e.alive: self.memory.pop(e.uid,None); continue
            seen=observer.alive and observer.blind<=0 and (e.reveal>0 or has_los(observer.pos,e.pos,grid,smokes))
            if seen:
                self.memory[e.uid]=MemoryEntry(e.pos,e.hp,0,True,e.reveal>0); visible.add(e.uid)
        for uid in list(self.memory):
            if uid in visible: continue
            x=self.memory[uid]; x.age+=1; x.visible=False; x.revealed=False
            if x.age>self.memory_ticks: del self.memory[uid]
    def features(self, pos: Position, h: int, w: int) -> List[float]:
        items=sorted(self.memory.values(),key=lambda x:cheb(pos,x.pos)); out=[]
        for i in range(self.k):
            if i>=len(items): out.extend([0.0]*8); continue
            x=items[i]
            out += [1.0,float(x.visible),float(not x.visible),
                    (x.pos[0]-pos[0])/max(h-1,1),(x.pos[1]-pos[1])/max(w-1,1),
                    max(x.hp,0)/MAX_HP,min(x.age/max(self.memory_ticks,1),1.0),float(x.revealed)]
        return out


class ReplayBuffer:
    def __init__(self, capacity: int=REPLAY_CAPACITY):
        self.capacity=capacity; self.s=np.empty((capacity,OBS_DIM),np.float32)
        self.a=np.empty(capacity,np.int64); self.r=np.empty(capacity,np.float32)
        self.ns=np.empty((capacity,OBS_DIM),np.float32); self.d=np.empty(capacity,np.float32)
        self.i=0; self.size=0
    def __len__(self): return self.size
    def add(self,s,a,r,ns,d):
        i=self.i; self.s[i]=s; self.a[i]=a; self.r[i]=r; self.ns[i]=ns; self.d[i]=float(d)
        self.i=(i+1)%self.capacity; self.size=min(self.size+1,self.capacity)
    def sample(self,n: int,device: torch.device):
        idx=np.random.randint(0,self.size,size=n)
        return tuple(torch.as_tensor(x[idx],device=device) for x in (self.s,self.a,self.r,self.ns,self.d))


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int=OBS_DIM, n_actions: int=N_ACTIONS):
        super().__init__()
        self.feature=nn.Sequential(nn.Linear(obs_dim,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.value=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        self.adv=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,n_actions))
    def forward(self,x):
        f=self.feature(x); v=self.value(f); a=self.adv(f); return v+a-a.mean(dim=1,keepdim=True)


class DefenderPolicy:
    """学習相手用。アタッカーの選択を補正・上書きしない。"""
    def actions(self, env: "AttackerAIv2Env") -> Dict[int,int]:
        out={}
        for d in env.defenders:
            if not d.alive: continue
            visible=[a for a in env.attackers if a.alive and (a.reveal>0 or has_los(d.pos,a.pos,env.grid,env.smokes))]
            if visible: out[d.uid]=ACTION_WAIT_OR_PLANT; continue
            obj=env.spike_pos if env.planted else env.spike_target()
            if obj is None: out[d.uid]=random.randrange(4); continue
            dm=bfs(obj,env.grid); cand=[]
            for act,(dr,dc) in MOVE_DELTAS.items():
                p=(d.pos[0]+dr,d.pos[1]+dc)
                if env.walkable(p): cand.append((dm[p],act))
            out[d.uid]=min(cand,key=lambda x:x[0])[1] if cand else ACTION_WAIT_OR_PLANT
        return out


class AttackerAIv2Env:
    def __init__(self, seed: int=DEFAULT_SEED):
        self.grid=parse_grid(NEW_MAZE_STR); self.h,self.w=self.grid.shape; self.rng=random.Random(seed)
        self.a_spawns=self.cells(3); self.d_spawns=self.cells(4); self.plant_cells=self.cells(2)
        self.walk_cells=[(r,c) for r in range(self.h) for c in range(self.w) if self.grid[r,c]!=1]
        if not self.a_spawns or not self.plant_cells: raise ValueError("map_data.pyに3または2のセルがありません")
        if not self.d_spawns: self.d_spawns=[p for p in self.walk_cells if p[0]<self.h//2]
        self.def_policy=DefenderPolicy(); self.memories={}; self.attackers=[]; self.defenders=[]; self.smokes=[]
        self.tick=0; self.planted=False; self.spike_pos=None; self.spike_timer=0; self.drop_pos=None
        self.target_plant=self.plant_cells[0]; self.done=False; self.winner=None; self.end_reason=""; self.defuser_uid=None

    def cells(self,v: int) -> List[Position]:
        rr,cc=np.where(self.grid==v); return list(zip(rr.tolist(),cc.tolist()))
    def inside(self,p): return 0<=p[0]<self.h and 0<=p[1]<self.w
    def walkable(self,p): return self.inside(p) and self.grid[p]!=1
    def sample_positions(self,pool,n,excluded=None):
        excluded=set(excluded or []); cand=[p for p in pool if p not in excluded]
        if len(cand)<n: cand += [p for p in self.walk_cells if p not in excluded and p not in cand]
        return self.rng.sample(cand,n)
    def stats(self):
        return dict(accuracy=self.rng.uniform(.48,.68),dodge=self.rng.uniform(.08,.28),hs=self.rng.uniform(.20,.45),reaction=self.rng.uniform(70,150))

    def reset(self, seed: Optional[int]=None) -> Dict[int,np.ndarray]:
        if seed is not None: self.rng.seed(seed)
        self.tick=0; self.planted=False; self.spike_pos=None; self.spike_timer=0; self.drop_pos=None
        self.done=False; self.winner=None; self.end_reason=""; self.defuser_uid=None; self.smokes=[]
        ap=self.sample_positions(self.a_spawns,5); dp=self.sample_positions(self.d_spawns,5,set(ap))
        abilities=[ABILITY_FLASH,ABILITY_SMOKE,ABILITY_RECON,ABILITY_HUNT,ABILITY_FLASH]
        self.attackers=[]
        for i,p in enumerate(ap):
            s=self.stats(); ab=abilities[i]; ch=CharacterState(i,f"Attacker_{i+1}",TEAM_ATTACKER,p,ability=ab,charges=0 if ab==ABILITY_HUNT else 1,**s)
            if ab==ABILITY_HUNT: ch.accuracy+=.10; ch.hs=min(1,ch.hs+.05)
            self.attackers.append(ch)
        self.defenders=[CharacterState(100+i,f"Defender_{i+1}",TEAM_DEFENDER,p,ability=ABILITY_NONE,charges=0,**self.stats()) for i,p in enumerate(dp)]
        carrier=self.attackers[0]; carrier.has_spike=True; self.target_plant=self.rng.choice(self.plant_cells)
        self.memories={a.uid:EnemyMemoryTracker() for a in self.attackers}
        return self.observations(True)

    def objective(self,a):
        if self.planted and self.spike_pos is not None: return self.spike_pos
        if a.has_spike: return self.target_plant
        if self.drop_pos is not None: return self.drop_pos
        holder=next((x for x in self.attackers if x.alive and x.has_spike),None)
        return holder.pos if holder else self.target_plant
    def spike_target(self):
        if self.planted: return self.spike_pos
        holder=next((x for x in self.attackers if x.alive and x.has_spike),None)
        return holder.pos if holder else self.drop_pos
    def distance(self,a,b): return float(bfs(b,self.grid)[a])

    def observations(self, update: bool) -> Dict[int,np.ndarray]:
        return {a.uid:self.observation(a,update) for a in self.attackers if a.alive}

    def observation(self,a: CharacterState,update=True) -> np.ndarray:
        m=self.memories[a.uid]
        if update: m.update(a,self.defenders,self.grid,self.smokes)
        r,c=a.pos; obj=self.objective(a)
        out=[r/max(self.h-1,1),c/max(self.w-1,1),a.hp/MAX_HP,float(a.alive),float(a.moved),
             min(a.blind/BLIND_DURATION_TICKS,1),min(a.reveal/REVEAL_DURATION_TICKS,1),float(a.has_spike),
             float(a.planting),min(a.plant_timer/PLANT_REQUIRED_TICKS,1)]
        # ability one-hot: NONE, FLASH, SMOKE, RECON, HUNT
        names=[ABILITY_NONE,ABILITY_FLASH,ABILITY_SMOKE,ABILITY_RECON,ABILITY_HUNT]
        out += [float(a.ability==x) for x in names] + [float(a.charges>0)]
        out += [float(a.move_failed),float(a.fail_reason=="OUT"),float(a.fail_reason=="WALL"),float(a.fail_reason in {"ALLY","CONFLICT"})]
        out += [float(i==a.last_action) for i in range(N_ACTIONS)]
        occupied={x.pos for x in self.attackers+self.defenders if x.alive and x.uid!=a.uid}
        out += [float(not self.walkable((r+dr,c+dc)) or (r+dr,c+dc) in occupied) for dr,dc in MOVE_DELTAS.values()]
        dist=self.distance(a.pos,obj)
        out += [(obj[0]-r)/max(self.h-1,1),(obj[1]-c)/max(self.w-1,1),1.0 if not np.isfinite(dist) else min(dist/(self.h+self.w),1)]
        out += [float(not self.planted),float(self.planted),float(self.drop_pos is not None),min(self.tick/ROUND_DURATION_TICKS,1),min(self.spike_timer/SPIKE_DETONATION_TICKS,1) if self.planted else 0]
        mates=sorted([x for x in self.attackers if x.uid!=a.uid],key=lambda x:cheb(a.pos,x.pos))
        for x in mates[:4]: out += [float(x.alive),(x.pos[0]-r)/max(self.h-1,1),(x.pos[1]-c)/max(self.w-1,1),max(x.hp,0)/MAX_HP,float(x.has_spike),float(x.moved)]
        while len(out)<76: out += [0.0]
        out += m.features(a.pos,self.h,self.w)
        # smoke: own tile + 4方向で最初に煙がある近さ
        out.append(float(any(s.contains(a.pos) for s in self.smokes)))
        for dr,dc in MOVE_DELTAS.values():
            found=0.0
            for k in range(1,max(self.h,self.w)+1):
                p=(r+dr*k,c+dc*k)
                if not self.inside(p): break
                if any(s.contains(p) for s in self.smokes): found=1-min(k/max(self.h,self.w),1); break
            out.append(found)
        active=[d for d in self.defenders if d.alive and d.defuse_timer>0]
        if active:
            d=max(active,key=lambda x:x.defuse_timer)
            out += [1.0,min(d.defuse_timer/DEFUSE_REQUIRED_TICKS,1),(d.pos[0]-r)/max(self.h-1,1),(d.pos[1]-c)/max(self.w-1,1)]
        else: out += [0.0]*4
        arr=np.asarray(out,np.float32)
        if arr.shape!=(OBS_DIM,): raise RuntimeError(f"OBS_DIM mismatch: {arr.shape}, expected {(OBS_DIM,)}")
        return arr

    def step(self, actions: Dict[int,int]):
        before=self.observations(True); alive_before=[a for a in self.attackers if a.alive]
        rewards={a.uid:R_STEP for a in alive_before}; self.tick+=1
        for x in self.attackers+self.defenders: x.tick_reset()
        acts={a.uid:int(actions.get(a.uid,ACTION_WAIT_OR_PLANT)) if 0<=int(actions.get(a.uid,4))<N_ACTIONS else 4 for a in alive_before}
        for a in alive_before: a.last_action=acts[a.uid]
        dacts=self.def_policy.actions(self)
        old={a.uid:self.distance(a.pos,self.objective(a)) for a in alive_before}
        was_in_site={a.uid:(a.pos in self.plant_cells) for a in alive_before}
        self.resolve_moves({u:x for u,x in acts.items() if x in MOVE_DELTAS},{u:x for u,x in dacts.items() if x in MOVE_DELTAS},rewards)
        for a in alive_before:
            if (
                a.alive
                and a.has_spike
                and not was_in_site[a.uid]
                and a.pos in self.plant_cells
            ):
                rewards[a.uid]+=R_ENTER_PLANT_SITE

            if a.alive and a.moved:
                old_dist=old[a.uid]
                new_dist=self.distance(a.pos,self.objective(a))
                if np.isfinite(old_dist) and np.isfinite(new_dist):
                    diff=old_dist-new_dist
                    if diff>0:
                        rewards[a.uid]+=diff*R_TOWARD
                    elif diff<0:
                        rewards[a.uid]+=abs(diff)*R_AWAY
        for a in sorted(alive_before,key=lambda x:x.reaction,reverse=True):
            if a.alive and acts[a.uid] in ABILITY_ACTIONS: self.use_ability(a,acts[a.uid],rewards)
        for a in alive_before:
            if not a.alive: continue
            if acts[a.uid]==ACTION_WAIT_OR_PLANT: self.wait_or_plant(a,rewards)
            elif a.planting: a.planting=False; a.plant_timer=0; rewards[a.uid]+=R_PLANT_INTERRUPTED
        self.handle_defuse(rewards); self.combat(rewards); self.deaths_and_spike(); self.update_timers(); self.check_end(rewards)
        nxt=self.observations(True)
        return nxt,rewards,self.done,{"before_obs":before,"actions":acts,"tick":self.tick,"winner":self.winner,"end_reason":self.end_reason,"is_planted":self.planted}

    def resolve_moves(self,aacts,dacts,rewards):
        chars={x.uid:x for x in self.attackers+self.defenders if x.alive}; current={u:x.pos for u,x in chars.items()}; prop={}
        allacts=dict(aacts); allacts.update(dacts)
        for uid,act in allacts.items():
            x=chars[uid]; dr,dc=MOVE_DELTAS[act]; p=(x.pos[0]+dr,x.pos[1]+dc)
            if not self.inside(p): x.move_failed=True; x.fail_reason="OUT"; rewards[uid]=rewards.get(uid,0)+R_OUT if x.team==TEAM_ATTACKER else 0; continue
            if self.grid[p]==1: x.move_failed=True; x.fail_reason="WALL"; rewards[uid]=rewards.get(uid,0)+R_WALL if x.team==TEAM_ATTACKER else 0; continue
            prop[uid]=p
        conflicts=set(); targets={}
        for uid,p in prop.items(): targets.setdefault(p,[]).append(uid)
        for ids in targets.values():
            if len(ids)>1: conflicts.update(ids)
        for uid,p in list(prop.items()):
            for ouid,op in current.items():
                if uid!=ouid and p==op and prop.get(ouid)==current[uid]: conflicts|={uid,ouid}
        for uid in conflicts:
            x=chars[uid]; x.move_failed=True; x.fail_reason="CONFLICT"; prop.pop(uid,None)
            if x.team==TEAM_ATTACKER: rewards[uid]+=R_CONFLICT
        for uid,p in list(prop.items()):
            blocker=next((ouid for ouid,op in current.items() if ouid!=uid and op==p),None)
            if blocker is not None and blocker not in prop:
                x=chars[uid]; x.move_failed=True; x.fail_reason="ALLY"; prop.pop(uid)
                if x.team==TEAM_ATTACKER: rewards[uid]+=R_ALLY
        for uid,p in prop.items(): chars[uid].pos=p; chars[uid].moved=True

    def ability_target(self,origin,action):
        dr,dc,n=ABILITY_ACTIONS[action]; p=origin
        for _ in range(n):
            q=(p[0]+dr,p[1]+dc)
            if not self.walkable(q): break
            p=q
        return p

    def use_ability(self,a,action,rewards):
        if a.ability in {ABILITY_NONE,ABILITY_HUNT}: rewards[a.uid]+=R_ABILITY_WASTED; return
        if a.charges<=0: rewards[a.uid]+=R_ABILITY_NO_CHARGE; return
        target=self.ability_target(a.pos,action)
        if target==a.pos: rewards[a.uid]+=R_ABILITY_INVALID; return
        a.charges-=1; rewards[a.uid]+=R_ABILITY_OK
        if a.ability==ABILITY_SMOKE:
            self.smokes.append(SmokeArea(target,SMOKE_DURATION_TICKS,a.uid))
            useful=any(any(cheb(cell,target)<=SMOKE_RADIUS for cell in line_cells(d.pos,self.target_plant)) for d in self.defenders if d.alive)
            rewards[a.uid]+=R_SMOKE_USEFUL if useful else R_ABILITY_WASTED
        elif a.ability==ABILITY_FLASH:
            eh=ah=0
            for x in self.attackers+self.defenders:
                if not x.alive or x.uid==a.uid or cheb(target,x.pos)>4 or not has_los(target,x.pos,self.grid,self.smokes): continue
                x.blind=max(x.blind,BLIND_DURATION_TICKS)
                if x.team==a.team: ah+=1
                else: eh+=1
            rewards[a.uid]+=eh*R_FLASH_ENEMY+ah*R_FLASH_ALLY
            if eh==0 and ah==0: rewards[a.uid]+=R_ABILITY_WASTED
        elif a.ability==ABILITY_RECON:
            n=0
            for d in self.defenders:
                if d.alive and cheb(target,d.pos)<=RECON_RADIUS: d.reveal=max(d.reveal,REVEAL_DURATION_TICKS); n+=1
            rewards[a.uid]+=n*R_RECON_ENEMY
            if n==0: rewards[a.uid]+=R_ABILITY_WASTED

    def wait_or_plant(self,a,rewards):
        if self.planted: rewards[a.uid]+=R_WAIT; return
        if not a.has_spike:
            if self.drop_pos is not None and a.pos==self.drop_pos: self.give_spike(a); self.drop_pos=None
            else: rewards[a.uid]+=R_WAIT
            return
        if a.pos not in self.plant_cells: a.planting=False; a.plant_timer=0; rewards[a.uid]+=R_INVALID_PLANT; return
        a.planting=True; a.plant_timer+=1; rewards[a.uid]+=R_PLANT_PROGRESS
        if a.plant_timer>=PLANT_REQUIRED_TICKS:
            self.planted=True; self.spike_pos=a.pos; self.spike_timer=SPIKE_DETONATION_TICKS; a.has_spike=False; a.planting=False; a.plant_timer=0; self.drop_pos=None
            rewards[a.uid]+=R_PLANT_CARRIER
            for t in self.attackers:
                if t.alive and t.uid!=a.uid: rewards[t.uid]=rewards.get(t.uid,0)+R_PLANT_TEAM

    def handle_defuse(self,rewards):
        self.defuser_uid=None
        if not self.planted or self.spike_pos is None:
            for d in self.defenders: d.defuse_timer=0
            return
        cand=[d for d in self.defenders if d.alive and cheb(d.pos,self.spike_pos)<=1]
        if not cand:
            for d in self.defenders: d.defuse_timer=0
            return
        d=max(cand,key=lambda x:(x.defuse_timer,x.reaction)); self.defuser_uid=d.uid
        for x in self.defenders:
            if x.uid!=d.uid: x.defuse_timer=0
        d.defuse_timer+=1
        for a in self.attackers:
            if a.alive: rewards[a.uid]=rewards.get(a.uid,0)+(R_DEFUSE_START if d.defuse_timer==1 else 0)+R_DEFUSE_PROGRESS
        if d.defuse_timer>=DEFUSE_REQUIRED_TICKS: self.done=True; self.winner=TEAM_DEFENDER; self.end_reason="DEFUSED"

    def combat(self,rewards):
        shooters=[x for x in self.attackers+self.defenders if x.alive]; self.rng.shuffle(shooters); shooters.sort(key=lambda x:x.reaction,reverse=True)
        for s in shooters:
            if not s.alive: continue
            enemies=self.defenders if s.team==TEAM_ATTACKER else self.attackers
            vis=[t for t in enemies if t.alive and (t.reveal>0 or has_los(s.pos,t.pos,self.grid,self.smokes))]
            if not vis: continue
            t=min(vis,key=lambda x:(cheb(s.pos,x.pos),x.hp)); chance=s.accuracy
            if s.blind>0: chance*=BLIND_ACCURACY_MULTIPLIER
            if s.moved: chance*=MOVING_ACCURACY_MULTIPLIER
            if t.moved: chance*=MOVING_TARGET_HIT_MULTIPLIER
            dodge=t.dodge*(REVEALED_DODGE_MULTIPLIER if t.reveal>0 else 1); chance*=max(0,1-dodge)
            if self.rng.random()>=min(max(chance,0),1): continue
            hs=self.rng.random()<s.hs; dmg=HEADSHOT_DAMAGE if hs else BODY_DAMAGE; actual=min(t.hp,dmg); t.hp-=dmg
            if s.team==TEAM_ATTACKER: rewards[s.uid]=rewards.get(s.uid,0)+actual*R_DAMAGE_DEALT
            else:
                alive=max(sum(a.alive for a in self.attackers),1)
                for a in self.attackers:
                    if a.alive: rewards[a.uid]=rewards.get(a.uid,0)+actual*R_DAMAGE_TAKEN/alive
            if t.hp<=0:
                t.alive=False; t.deaths+=1; s.kills+=1
                if s.team==TEAM_ATTACKER:
                    rewards[s.uid]+=R_KILL+(R_HS_KILL if hs else 0)+(R_DEFUSER_KILL if t.uid==self.defuser_uid else 0)
                else:
                    rewards[t.uid]=rewards.get(t.uid,0)+R_DEATH
                    for a in self.attackers:
                        if a.alive: rewards[a.uid]=rewards.get(a.uid,0)+R_TEAMMATE_DEATH

    def deaths_and_spike(self):
        for a in self.attackers:
            if not a.alive and a.has_spike: a.has_spike=False; a.planting=False; a.plant_timer=0; self.drop_pos=a.pos
        if self.drop_pos is not None:
            pick=[a for a in self.attackers if a.alive and a.pos==self.drop_pos]
            if pick: self.give_spike(max(pick,key=lambda x:x.reaction)); self.drop_pos=None
    def give_spike(self,a):
        for x in self.attackers: x.has_spike=False
        a.has_spike=True
    def update_timers(self):
        for x in self.attackers+self.defenders:
            x.blind=max(0,x.blind-1); x.reveal=max(0,x.reveal-1)
        for s in self.smokes: s.remaining-=1
        self.smokes=[s for s in self.smokes if s.remaining>0]
        if self.planted and not self.done: self.spike_timer-=1
    def check_end(self,rewards):
        if not self.done:
            aa=[a for a in self.attackers if a.alive]; dd=[d for d in self.defenders if d.alive]
            if not dd and self.planted:
                self.done=True; self.winner=TEAM_ATTACKER; self.end_reason="DEFENDERS_ELIMINATED"
                for a in aa: rewards[a.uid]=rewards.get(a.uid,0)+R_ELIMINATE
            elif not aa: self.done=True; self.winner=TEAM_DEFENDER; self.end_reason="ATTACKERS_ELIMINATED"
            elif self.planted and self.spike_timer<=0: self.done=True; self.winner=TEAM_ATTACKER; self.end_reason="SPIKE_DETONATED"; [rewards.__setitem__(a.uid,rewards.get(a.uid,0)+R_DETONATE) for a in aa]
            elif not self.planted and self.tick>=ROUND_DURATION_TICKS: self.done=True; self.winner=TEAM_DEFENDER; self.end_reason="ROUND_TIMEOUT"
            elif self.tick>=MAX_EPISODE_STEPS: self.done=True; self.winner=TEAM_DEFENDER; self.end_reason="SAFETY_TIMEOUT"
        if self.done:
            for a in self.attackers: rewards[a.uid]=rewards.get(a.uid,0)+(R_WIN if self.winner==TEAM_ATTACKER else R_LOSS)


def choose_actions(model,obs,epsilon,device,rng):
    if not obs: return {}
    uids=list(obs); batch=np.stack([obs[u] for u in uids])
    with torch.no_grad(): q=model(torch.as_tensor(batch,dtype=torch.float32,device=device)).cpu().numpy()
    return {u:(rng.randrange(N_ACTIONS) if rng.random()<epsilon else int(np.argmax(q[i]))) for i,u in enumerate(uids)}


def optimize(policy,target,opt,buf,device):
    if len(buf)<max(BATCH_SIZE,LEARNING_STARTS): return None
    s,a,r,ns,d=buf.sample(BATCH_SIZE,device); q=policy(s).gather(1,a.long().unsqueeze(1)).squeeze(1)
    with torch.no_grad(): na=policy(ns).argmax(1,keepdim=True); nq=target(ns).gather(1,na).squeeze(1); y=r+GAMMA*(1-d)*nq
    loss=nn.functional.smooth_l1_loss(q,y); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(),GRADIENT_CLIP_NORM); opt.step()
    return float(loss.item())


def save_model(path,model): path.parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),path)

def evaluate(model,device,episodes=EVAL_EPISODES,seed=10000):
    env=AttackerAIv2Env(seed); rng=random.Random(seed); wins=plants=det=0; ticks=0; reasons={}
    was=model.training; model.eval()
    for ep in range(episodes):
        obs=env.reset(seed+ep); done=False; planted=False
        while not done:
            obs,_,done,_=env.step(choose_actions(model,obs,0,device,rng)); planted|=env.planted
        wins+=env.winner==TEAM_ATTACKER; plants+=planted; det+=env.end_reason=="SPIKE_DETONATED"; ticks+=env.tick; reasons[env.end_reason]=reasons.get(env.end_reason,0)+1
    if was: model.train()
    return {"win_rate":wins/episodes,"plant_rate":plants/episodes,"detonation_rate":det/episodes,"avg_ticks":ticks/episodes,"reasons":reasons}


def train(episodes=DEFAULT_EPISODES,seed=DEFAULT_SEED,device_name=None,resume=None,tensorboard=True):
    seed_everything(seed); device=torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu")); MODEL_DIR.mkdir(parents=True,exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({"obs_dim":OBS_DIM,"n_actions":N_ACTIONS,"action_names":ACTION_NAMES},ensure_ascii=False,indent=2),encoding="utf-8")
    env=AttackerAIv2Env(seed); policy=DuelingQNetwork().to(device); target=DuelingQNetwork().to(device); target.load_state_dict(policy.state_dict()); target.eval()
    opt=optim.AdamW(policy.parameters(),lr=LEARNING_RATE,eps=1e-5); buf=ReplayBuffer(); start=1; steps=0; best=-math.inf
    if resume:
        cp=torch.load(resume,map_location=device)
        if cp.get("obs_dim")!=OBS_DIM or cp.get("n_actions")!=N_ACTIONS: raise ValueError("checkpointのOBS_DIM/N_ACTIONSが不一致")
        policy.load_state_dict(cp["policy"]); target.load_state_dict(cp["target"]); opt.load_state_dict(cp["optimizer"]); start=cp["episode"]+1; steps=cp["steps"]; best=cp.get("best",best)
    writer=SummaryWriter(str(MODEL_DIR/"tensorboard")) if tensorboard and SummaryWriter is not None else None;
    if tensorboard and SummaryWriter is None:
        print("[WARN] tensorboard未導入のためログを無効化します")
    rng=random.Random(seed+99); recent=deque(maxlen=100); wins=deque(maxlen=100)
    print(f"device={device} episodes={episodes} OBS_DIM={OBS_DIM} N_ACTIONS={N_ACTIONS}")
    try:
        for ep in range(start,episodes+1):
            obs=env.reset(seed+ep); done=False; total=0; losses=[]
            while not done:
                acts=choose_actions(policy,obs,epsilon_by_steps(steps),device,rng); nxt,rews,done,info=env.step(acts); before=info["before_obs"]
                for uid,act in acts.items():
                    ns=nxt.get(uid,np.zeros(OBS_DIM,np.float32)); terminal=done or uid not in nxt; rew=float(rews.get(uid,0)); buf.add(before[uid],act,rew,ns,terminal); total+=rew; steps+=1
                    if steps>=LEARNING_STARTS and steps%TRAIN_EVERY_AGENT_STEPS==0:
                        loss=optimize(policy,target,opt,buf,device)
                        if loss is not None: losses.append(loss)
                    if steps%TARGET_UPDATE_INTERVAL==0: target.load_state_dict(policy.state_dict())
                obs=nxt
            recent.append(total); wins.append(float(env.winner==TEAM_ATTACKER)); ml=float(np.mean(losses)) if losses else 0
            if writer:
                writer.add_scalar("train/team_reward",total,ep); writer.add_scalar("train/win",wins[-1],ep); writer.add_scalar("train/loss",ml,ep); writer.add_scalar("train/epsilon",epsilon_by_steps(steps),ep)
            if ep%10==0 or ep==start: print(f"ep {ep}/{episodes} win100={np.mean(wins):.3f} reward100={np.mean(recent):.1f} eps={epsilon_by_steps(steps):.3f} loss={ml:.4f} end={env.end_reason}")
            if ep%SAVE_INTERVAL_EPISODES==0:
                save_model(LATEST_MODEL_PATH,policy); torch.save({"episode":ep,"steps":steps,"best":best,"obs_dim":OBS_DIM,"n_actions":N_ACTIONS,"policy":policy.state_dict(),"target":target.state_dict(),"optimizer":opt.state_dict()},TRAINING_STATE_PATH)
            if ep%EVAL_INTERVAL_EPISODES==0:
                result=evaluate(policy,device); print("[EVAL]",result)
                if writer:
                    writer.add_scalar("eval/win_rate",result["win_rate"],ep); writer.add_scalar("eval/plant_rate",result["plant_rate"],ep)
                if result["win_rate"]>best: best=result["win_rate"]; save_model(BEST_MODEL_PATH,policy); print("[BEST]",best)
        save_model(FINAL_MODEL_PATH,policy); save_model(LATEST_MODEL_PATH,policy)
    except KeyboardInterrupt:
        print("中断されたため保存します")
        save_model(LATEST_MODEL_PATH,policy)
    finally:
        if writer: writer.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--episodes",type=int,default=DEFAULT_EPISODES); p.add_argument("--seed",type=int,default=DEFAULT_SEED); p.add_argument("--device",choices=["cpu","cuda","mps"],default=None); p.add_argument("--resume",type=Path,default=None); p.add_argument("--no-tensorboard",action="store_true"); a=p.parse_args()
    if a.resume and not a.resume.is_file(): raise SystemExit(f"resume file not found: {a.resume}")
    train(a.episodes,a.seed,a.device,a.resume,not a.no_tensorboard)


if __name__ == "__main__": main()