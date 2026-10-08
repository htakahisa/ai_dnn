"""Versioned public observations for the GC v2 residual policy."""
import hashlib
import json
import numpy as np
from gc_v1.roster_observation_gc import enemy_roster_features, ENEMY_ROSTER_DIM
from gc_v1.gc_facing import FACING_DIRS
from game_core import SPIKE_DETONATION_TICKS,ROUND_DURATION_TICKS,FLASH_BLIND_TICKS
from ..hazards import HAZARD_BLOCKS
from .unit_observation import UNIT_BLOCKS, UNIT_FIELDS, SELF_FIELDS, BASE_STATS, LIVE_STATS, player_abilities_status, self_capabilities

OPPONENTS=("TYG","OMG","FRC","FNC","GG","SPS","unknown","observed_mid_heavy")
ABILITIES=("RECON","FLASH","SMOKE","DANCE","ASH","HUNT","UNKNOWN")
LEGACY_V2_BLOCKS=(("opponent",8),("observed_axes",6),("surveyed_axes",3),("spike",8),
        ("self",8),("ability",7),("proposal",10),("local_map",49),
        ("units",70),("public_defuse",2),("self_status",12),("enemy_roster",ENEMY_ROSTER_DIM))
LEGACY_V3_BLOCKS=LEGACY_V2_BLOCKS+HAZARD_BLOCKS
BLOCKS=LEGACY_V3_BLOCKS+UNIT_BLOCKS
OBS_DIM=sum(n for _,n in BLOCKS)
OBSERVATION_SCHEMA=dict(version=4,dim=OBS_DIM,blocks=BLOCKS,enemy_roster_version=1,public_effect_version=1,
                       unit_status_version=1,unit_status_fields=UNIT_FIELDS,self_capability_fields=SELF_FIELDS,
                       unit_base_scales=BASE_STATS,unit_live_scales=LIVE_STATS)
SCHEMA_HASH=hashlib.sha256(json.dumps(OBSERVATION_SCHEMA,sort_keys=True).encode()).hexdigest()


def public_defuse_features(state):
    """Use the public tap notification, including when the defuser is unseen."""
    taps=[(progress,required) for progress,required in (state.get("defender_defuse_info") or {}).values()
          if state.get("is_planted") and progress>0]
    return [len(taps)/5,max((min(1.,progress/max(1,required)) for progress,required in taps),default=0.)]


def proposal_kind(proposal,origin):
    command=proposal[1] if isinstance(proposal,tuple) and len(proposal)>1 else None
    if isinstance(command,dict) and (command.get("ability") or command.get("ultimate")):
        return 2
    if isinstance(command,str) and command in ("PLANT","COLLECT_ORB"):
        return 3
    destination=proposal[0] if isinstance(proposal,tuple) else proposal
    return 0 if tuple(destination)!=tuple(origin) else 1


class ObservationEncoder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.planted_at=None

    def encode(self,char,state,tactics,proposal):
        grid=state["grid"]
        height,width=grid.shape
        tick=int(state.get("battle_tick",0))
        planted=bool(state.get("is_planted"))
        if planted and self.planted_at is None:
            self.planted_at=tick
        point=tactics.planted_position(state) if planted else state.get("spike_pos")
        target=tactics.target_plant_pos
        allies=sorted((c for c in state.get("chars",()) if c.team==char.team),key=lambda c:str(c.name))
        enemies=sorted((c for c in state.get("chars",()) if c.team!=char.team),key=lambda c:str(c.name))
        opponent=np.zeros(len(OPPONENTS),np.float32)
        opponent[OPPONENTS.index(tactics.opponent)]=1
        observed=[tactics.observed_defenders_by_axis[a]/5 for a in ("A","Mid","B")]
        observed += [tactics.max_observed_defenders_by_axis[a]/5 for a in ("A","Mid","B")]
        spike=[planted,(tick-self.planted_at)/SPIKE_DETONATION_TICKS if planted else 0,
               bool(point),point[0]/height if point else 0,point[1]/width if point else 0,
               bool(target),target[0]/height if target else 0,target[1]/width if target else 0]
        own=[char.pos[0]/height,char.pos[1]/width,getattr(char,"hp",100)/100,
             bool(getattr(char,"has_spike",False)),sum(c.is_alive for c in allies)/5,
             sum(c.is_alive for c in enemies)/5,tick/ROUND_DURATION_TICKS,getattr(char,"plant_timer",0)/10]
        role=np.zeros(len(ABILITIES),np.float32)
        ability=getattr(char,"ability_name","UNKNOWN")
        role[ABILITIES.index(ability) if ability in ABILITIES else -1]=1
        kind=proposal_kind(proposal,char.pos)
        intent=np.zeros(10,np.float32)
        intent[kind]=1
        destination=proposal[0] if isinstance(proposal,tuple) else proposal
        intent[4:6]=[(destination[0]-char.pos[0])/2,(destination[1]-char.pos[1])/2]
        command=proposal[1] if isinstance(proposal,tuple) and len(proposal)>1 else None
        cast=command.get("ability") if isinstance(command,dict) else None
        intent[6:9]=[cast==a for a in ("RECON","FLASH","SMOKE")]
        intent[9]=getattr(char,ability.lower()+"_charges",0)/3
        local=[]
        for dr in range(-3,4):
            for dc in range(-3,4):
                r,c=char.pos[0]+dr,char.pos[1]+dc
                local.append(-1 if not (0<=r<height and 0<=c<width) or grid[r,c]==1 else
                             1 if grid[r,c]==2 else 0)
        units=[]
        for team in (allies,enemies):
            for i in range(5):
                if i>=len(team):
                    units.extend([0]*7)
                    continue
                other=team[i]
                known=other.team==char.team or getattr(other,"position_known",False)
                units.extend([1,bool(other.is_alive),known,
                              other.pos[0]/height if known else 0,other.pos[1]/width if known else 0,
                              getattr(other,"hp",0)/100 if known else 0,
                              getattr(other,"reveal_remaining",0)/15 if known else 0])
        facing=[getattr(char,"facing",None)==direction for direction in FACING_DIRS]
        status=[*facing,getattr(char,"blind_remaining",0)/FLASH_BLIND_TICKS,getattr(char,"electric_remaining",0)/10,
                bool(getattr(char,"moved_last_tick",False)),getattr(char,"shield_hp",0)/100]
        obs=np.concatenate([opponent,observed,[a in tactics.surveyed_axes for a in ("A","Mid","B")],
                            spike,own,role,intent,local,units,public_defuse_features(state),status,
                            enemy_roster_features(game_state=state),tactics.hazards.features(char),
                            player_abilities_status(char,state),self_capabilities(char,state)]).astype(np.float32)
        if obs.shape!=(OBS_DIM,) or not np.isfinite(obs).all():
            raise ValueError("Invalid GC v2 RL observation")
        return obs
