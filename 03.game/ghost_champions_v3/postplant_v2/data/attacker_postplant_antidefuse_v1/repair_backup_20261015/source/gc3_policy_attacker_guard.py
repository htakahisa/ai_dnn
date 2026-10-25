"""Warm-start the existing guard; add public objective inputs and isolated value fitting."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from ..gc3_policy import FreshPolicy,SCHEMA_HASH as BASE_SCHEMA
from ..gc3_observation import OBS_DIM,map_hints,encode
from ..gc3_engine_actions import legal_actions,decode_command,MODES
from ..gc3_io import replace_file
from .gc3_features_attacker_guard import FEATURE_WIDTH,guard_features,tap_state

POLICY_TYPE='gc3_postplant_objective_guard_v2'
SCHEMA_HASH=hashlib.sha256((BASE_SCHEMA+'public-tap-spike-geometry-residual-v1').encode()).hexdigest()


class ObjectiveFeatures(nn.Module):
    def __init__(self,base,hidden):
        super().__init__();self.base=base
        self.objective=nn.Sequential(nn.Linear(FEATURE_WIDTH,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        nn.init.zeros_(self.objective[-1].weight);nn.init.zeros_(self.objective[-1].bias)

    def forward(self,obs):
        return self.base(obs[...,:OBS_DIM])+self.objective(obs[...,OBS_DIM:])


class GuardPolicy(FreshPolicy):
    def __init__(self,role='attacker',hidden=128):
        if role!='attacker':raise ValueError('Objective guard is attacker-only')
        super().__init__(role,hidden)
        self.features=ObjectiveFeatures(self.features,hidden);self.schema_hash=SCHEMA_HASH;self.previous={}

    @classmethod
    def from_guard(cls,base):
        if base.role!='attacker' or base.schema_hash!=BASE_SCHEMA:raise ValueError('Use the current trained GC3 guard')
        result=cls(base.role,base.hidden)
        transferred={('features.base.'+key[len('features.'):] if key.startswith('features.') else key):value
                     for key,value in base.state_dict().items()}
        missing,unexpected=result.load_state_dict(transferred,strict=False)
        if unexpected or set(missing)!={'features.objective.0.weight','features.objective.0.bias',
                'features.objective.2.weight','features.objective.2.bias'}:raise ValueError('Unexpected guard transfer')
        return result.eval()

    def reset_memory(self):self.previous.clear()

    def contexts(self,rows):return self.features(torch.from_numpy(np.stack([r['obs'] for r in rows])))
    def actor_parameters(self):return [p for name,p in self.named_parameters() if not name.startswith('critic.')]
    def value_parameters(self):return list(self.critic.parameters())

    def select(self,char,state):
        if not state.get('is_planted'):raise ValueError('The objective guard acts only after an actual plant')
        hints=map_hints(char,state);masks,_=legal_actions(char,state)
        extra=guard_features(char,state,self.previous.get(str(char.name)))
        active,progress,_,_=tap_state(state);self.previous[str(char.name)]=(active,progress)
        row=dict(obs=np.concatenate((encode(char,state,hints),extra)),hints=hints,masks=masks,full_masks=deepcopy(masks),
            phase='postplant',selections={},ability=char.ability_name,ultimate=char.ultimate_name)
        with torch.no_grad():
            context=self.features(torch.from_numpy(row['obs']));logp=0.
            def draw(kind):
                nonlocal logp
                dist=self.distribution(context,row,kind);choice=dist.sample()
                row['selections'][kind]=int(choice);logp+=float(dist.log_prob(choice))
            draw('mode');mode=MODES[row['selections']['mode']]
            if mode in ('MOVE','MOVE_WATCH'):draw('position')
            if mode in ('WATCH','MOVE_WATCH'):draw('watch')
            if mode=='ABILITY':
                if char.ability_name=='DANCE':draw('ally')
                elif char.ability_name!='RAMP':draw('target')
            if mode=='ULTIMATE':
                if char.ultimate_name in ('NEON','ESCAPE'):
                    row['masks']['target']=masks['ultimate_target'];draw('target')
                elif char.ultimate_name in ('RAID','TUNNEL'):draw('watch')
            row.update(logp=logp,value=float(self.critic(context)))
        return decode_command(char,state,row['selections']),row

    def evaluate_records(self,records):
        # Parent uses the same masks, hint temperatures and sampled branches.
        logp,entropy,_=super().evaluate_records(records)
        context=self.contexts(records)
        return logp,entropy,self.critic(context.detach()).squeeze(-1)

    def all_probabilities(self,records):
        context=self.contexts(records);result={}
        for kind in ('mode','position','watch','target','ultimate_target','ally'):
            masks=np.stack([r['full_masks'][kind] for r in records]).copy()
            # Physically unavailable branches have zero mode probability.
            # Give those unused distributions a harmless valid dummy mask.
            empty=~masks.any(-1);masks[empty,0]=True
            actual='target' if kind=='ultimate_target' else kind
            batch=dict(phase='postplant',hints=np.stack([r['hints'] for r in records]),masks={actual:masks})
            result[kind]=self.distribution(context,batch,actual).probs
        return result

    def save(self,path,**metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
        torch.save(dict(policy_type=POLICY_TYPE,schema_hash=SCHEMA_HASH,role=self.role,hidden=self.hidden,
            model_state_dict=self.state_dict(),**metadata),temporary);replace_file(temporary,path)

    @classmethod
    def load(cls,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('policy_type')!=POLICY_TYPE or saved.get('schema_hash')!=SCHEMA_HASH:raise ValueError('Objective guard checkpoint mismatch')
        with torch.random.fork_rng(devices=[]):
            result=cls(saved['role'],saved['hidden']);result.load_state_dict(saved['model_state_dict'])
        return result.eval(),saved
