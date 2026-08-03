from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

ACTION_NAMES = ["MOVE_UP","MOVE_DOWN","MOVE_LEFT","MOVE_RIGHT","SMOKE","FLASH","RECON","STOP","DEFUSE"]
NUM_ACTIONS = len(ACTION_NAMES)
DIRECTION_BY_ACTION = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
ABILITY_BY_ACTION = {4:"SMOKE",5:"FLASH",6:"RECON"}
LOCAL_RADIUS = 3
LOCAL_SIZE = 7
ALLY_COUNT = ENEMY_COUNT = 5

class DefenderPolicyNetwork(nn.Module):
    def __init__(self, obs_size:int, num_actions:int=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size,512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(.20),
            nn.Linear(512,256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(256,128), nn.ReLU(), nn.Linear(128,num_actions),
        )
    def forward(self,x): return self.net(x)

def alive(c): return bool(getattr(c,"is_alive",True))

def occupied(game, viewer=None):
    return {(int(c.pos[0]),int(c.pos[1])):c for c in getattr(game,"chars",[]) if alive(c) and c is not viewer}

def valid_destination(game, viewer, r, c):
    g=game.grid; r=int(r); c=int(c)
    return 0<=r<g.shape[0] and 0<=c<g.shape[1] and g[r,c]!=1 and (r,c) not in occupied(game,viewer)

def valid_move_mask(game, viewer):
    r,c=map(int,viewer.pos)
    return [int(valid_destination(game,viewer,r+dr,c+dc)) for dr,dc in DIRECTION_BY_ACTION.values()]

def local_map(game, viewer):
    g=game.grid; h,w=g.shape; vr,vc=map(int,viewer.pos); chars=occupied(game,None)
    pp=getattr(game,"planted_pos",None); pp=None if pp is None else tuple(map(int,pp))
    out=[]
    for dr in range(-LOCAL_RADIUS,LOCAL_RADIUS+1):
        row=[]
        for dc in range(-LOCAL_RADIUS,LOCAL_RADIUS+1):
            r,c=vr+dr,vc+dc; p=(r,c)
            if not (0<=r<h and 0<=c<w): v=6
            elif p==(vr,vc): v=5
            elif p in chars:
                v=3 if getattr(chars[p],"team",None)==viewer.team else 4
            elif pp is not None and p==pp: v=7
            elif g[r,c]==1: v=1
            elif g[r,c]==2: v=2
            else: v=0
            row.append(v)
        out.append(row)
    return out

def get_defender_observation(game, viewer):
    vr,vc=map(int,viewer.pos); team=viewer.team; enemy="A" if team=="D" else "D"
    planted=getattr(game,"planted_pos",None)
    planted_rel=[0,0] if planted is None else [int(planted[0])-vr,int(planted[1])-vc]
    allies=[viewer]+[c for c in game.chars if c.team==team and c is not viewer]
    obs={
        "grid":game.grid.flatten().tolist(), "viewer_pos":[vr,vc], "local_map":local_map(game,viewer),
        "valid_move_mask":valid_move_mask(game,viewer), "allies":[], "enemies":[],
        "is_planted":int(bool(getattr(game,"is_planted",False))), "planted_pos":planted_rel,
        "distance_to_planted":float(abs(planted_rel[0])+abs(planted_rel[1])) if planted is not None else 0.0,
        "detonate_timer":float(getattr(game,"detonate_timer",0.0)),
        "round_timer":float(getattr(game,"round_timer",0.0)),
        "viewer_is_defuser":int(getattr(game,"active_defuser_name",None)==getattr(viewer,"name",None)),
        "ally_is_defusing":int(any(getattr(game,"active_defuser_name",None)==getattr(c,"name",None) for c in allies)),
        "alive_allies":sum(1 for c in game.chars if c.team==team and alive(c)),
        "alive_enemies":sum(1 for c in game.chars if c.team==enemy and alive(c)),
    }
    for c in allies[:5]:
        obs["allies"].append({"rel_pos":[int(c.pos[0])-vr,int(c.pos[1])-vc],"hp":int(c.hp),"alive":int(alive(c)),
            "recon":int(getattr(c,"recon_charges",0)>0),"flash":int(getattr(c,"flash_charges",0)>0),"smoke":int(getattr(c,"smoke_charges",0)>0)})
    for c in [x for x in game.chars if x.team==enemy and alive(x)][:5]:
        obs["enemies"].append({"rel_pos":[int(c.pos[0])-vr,int(c.pos[1])-vc],"hp":int(c.hp)})
    return obs

def _fixed(values,n):
    a=np.zeros(n,dtype=np.float32); x=np.asarray(values,dtype=np.float32).reshape(-1); a[:min(n,len(x))]=x[:n]; return a

def defender_observation_to_vector(obs):
    grid=np.clip(np.asarray(obs["grid"],dtype=np.float32),0,7)/7
    vp=np.asarray(obs.get("viewer_pos",[0,0]),dtype=np.float32)/np.array([25,35],dtype=np.float32)
    lm=np.clip(_fixed(obs.get("local_map",[]),49),0,7)/7
    vm=np.clip(_fixed(obs.get("valid_move_mask",[1]*4),4),0,1)
    av=np.zeros(5*7,dtype=np.float32)
    for i,a in enumerate(obs.get("allies",[])[:5]):
        p=a.get("rel_pos",[0,0]); av[i*7:(i+1)*7]=[p[0]/25,p[1]/35,a.get("hp",0)/100,a.get("alive",0),a.get("recon",0),a.get("flash",0),a.get("smoke",0)]
    ev=np.zeros(5*3,dtype=np.float32)
    for i,e in enumerate(obs.get("enemies",[])[:5]):
        p=e.get("rel_pos",[0,0]); ev[i*3:(i+1)*3]=[p[0]/25,p[1]/35,e.get("hp",0)/100]
    pp=np.asarray(obs.get("planted_pos",[0,0]),dtype=np.float32)/np.array([25,35],dtype=np.float32)
    state=np.array([obs.get("is_planted",0),obs.get("distance_to_planted",0)/60,obs.get("detonate_timer",0)/100,
        obs.get("round_timer",0)/100,obs.get("viewer_is_defuser",0),obs.get("ally_is_defusing",0),
        obs.get("alive_allies",0)/5,obs.get("alive_enemies",0)/5],dtype=np.float32)
    return np.concatenate([grid,vp,lm,vm,av,ev,pp,state]).astype(np.float32)

def encode_defender_action(action, obs):
    if action.get("ability") in ("SMOKE","FLASH","RECON"): return {"SMOKE":4,"FLASH":5,"RECON":6}[action["ability"]]
    if action.get("special")=="DEFUSE": return 8
    cur=obs.get("viewer_pos",[0,0]); dst=action.get("move",cur); dr,dc=int(dst[0])-int(cur[0]),int(dst[1])-int(cur[1])
    return {(-1,0):0,(1,0):1,(0,-1):2,(0,1):3}.get((dr,dc),7)

def action_mask(game,char,game_state):
    mask=torch.ones(NUM_ACTIONS,dtype=torch.bool)
    for i,v in enumerate(valid_move_mask(game,char)): mask[i]=bool(v)
    mask[4]=getattr(char,"smoke_charges",0)>0; mask[5]=getattr(char,"flash_charges",0)>0; mask[6]=getattr(char,"recon_charges",0)>0; mask[7]=True
    pp=game_state.get("planted_pos"); r,c=map(int,char.pos)
    mask[8]=bool(game_state.get("is_planted",False) and pp is not None and max(abs(int(pp[0])-r),abs(int(pp[1])-c))<=1)
    return mask

def masked_probs(logits,mask):
    m=mask.to(logits.device).unsqueeze(0); return torch.softmax(logits.masked_fill(~m,torch.finfo(logits.dtype).min),dim=1)

def load_defender_policy(path,device):
    ck=torch.load(Path(path),map_location=device); sd=ck.get("model_state_dict",ck); obs=int(ck.get("obs_size",sd["net.0.weight"].shape[1])); n=int(ck.get("num_actions",NUM_ACTIONS))
    model=DefenderPolicyNetwork(obs,n).to(device); model.load_state_dict(sd); model.eval(); return model,obs
