"""Own public macro directives as learned inputs; no tactical action overrides."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from ..gc3_observation import ARENA_SHAPE
from ..gc3_io import replace_file
from ..preplant.gc3_policy_attacker_site import SiteHistoryPolicy,RETRIEVAL_SCHEMA_HASH
from ..preplant.gc3_features_attacker_navigation import public_layout,walking_distances,proximity
from ..preplant.gc3_policy_attacker_history import MOVE_OFFSETS

INSTRUCTIONS=("WAIT","ADVANCE","RETREAT")
INSTRUCTION_LABELS=("待機","進軍","撤退")
DIRECTIVE_WIDTH=32
PLAN_WIDTH=33
MICRO_POLICY_TYPE="gc3_macro_directed_micro_v2"
MICRO_SCHEMA_HASH=hashlib.sha256((RETRIEVAL_SCHEMA_HASH+"own-macro-directives-spawn-retreat-v3").encode()).hexdigest()


def plan_features(plan):
    """Previous/current own plan; no newly sampled action enters its own input."""
    plan=plan or dict(destinations=[None]*5,instructions=[None]*5,plant=None)
    values=[]
    for destination,instruction in zip(plan['destinations'],plan['instructions']):
        values.extend((destination is not None,*(np.asarray(destination)/ARENA_SHAPE if destination is not None else (0.,0.)),
                       *(instruction==name for name in INSTRUCTIONS)))
    plant=plan['plant']
    values.extend((plant is not None,*(np.asarray(plant)/ARENA_SHAPE if plant is not None else (0.,0.))))
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(PLAN_WIDTH,) or not np.isfinite(result).all():raise ValueError("Invalid macro plan")
    return result


def directive_features(observation,directive):
    if directive is None:return np.zeros(DIRECTIVE_WIDTH,dtype=np.float32)
    if directive['instruction'] not in INSTRUCTIONS:raise ValueError("Unknown macro instruction")
    board,_=public_layout(observation)
    walls=np.asarray(board[0]>.5,dtype=np.bool_)
    origin=tuple(np.rint(observation[:2]*ARENA_SHAPE).astype(int))
    values=[]
    for key in ('destination','plant'):
        target_key='retreat_target' if key=='destination' and directive['instruction']=='RETREAT' else key
        if target_key not in directive:raise ValueError('RETREAT requires the public own spawn coordinate')
        point=tuple(directive[target_key])
        if len(point)!=2 or any(not isinstance(v,(int,np.integer)) for v in point) or not (
                0<=point[0]<ARENA_SHAPE[0] and 0<=point[1]<ARENA_SHAPE[1]) or walls[point]:
            raise ValueError("Invalid own macro target coordinate")
        target=np.zeros(ARENA_SHAPE,dtype=np.bool_);target[point]=True
        distances=walking_distances(walls.tobytes(),target.tobytes())
        base=proximity(int(distances[origin]))
        values.extend((1.,point[0]/ARENA_SHAPE[0],point[1]/ARENA_SHAPE[1],base))
        for dr,dc in MOVE_OFFSETS:
            p=origin[0]+dr,origin[1]+dc
            valid=0<=p[0]<ARENA_SHAPE[0] and 0<=p[1]<ARENA_SHAPE[1]
            near=proximity(int(distances[p])) if valid else 0.
            values.extend((near,near-base))
    values.extend((*(directive['instruction']==name for name in INSTRUCTIONS),1.))
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(DIRECTIVE_WIDTH,) or not np.isfinite(result).all():raise ValueError("Invalid directive inputs")
    return result


class DirectedMicroPolicy(SiteHistoryPolicy):
    def __init__(self,role='attacker',hidden=128,**kwargs):
        super().__init__(role,hidden,navigation=True,retrieval=True)
        self.directive=nn.Sequential(nn.Linear(DIRECTIVE_WIDTH,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        nn.init.zeros_(self.directive[-1].weight);nn.init.zeros_(self.directive[-1].bias)
        self.schema_hash=MICRO_SCHEMA_HASH
        self._pending_directive=None
        self.directives_trained=False

    @classmethod
    def from_micro(cls,policy):
        """Explicit optional conversion; imports never load or save a model."""
        if policy.schema_hash!=RETRIEVAL_SCHEMA_HASH:raise ValueError("Use the current retrieval micro architecture")
        result=cls(policy.role,policy.hidden)
        missing,unexpected=result.load_state_dict(policy.state_dict(),strict=False)
        expected={'directive.0.weight','directive.0.bias','directive.2.weight','directive.2.bias'}
        if unexpected or set(missing)!=expected:raise ValueError("Unexpected directed micro architecture")
        return result.eval()

    def contexts(self,rows,*,cache=False):
        if self._pending_directive is not None:
            for row in rows:
                if 'macro_directive' not in row:row['macro_directive']=deepcopy(self._pending_directive)
        encoded=np.stack([directive_features(row['obs'],row.get('macro_directive')) for row in rows])
        present=torch.tensor([row.get('macro_directive') is not None for row in rows],dtype=torch.float32)[:,None]
        return super().contexts(rows,cache=cache)+self.directive(torch.from_numpy(encoded))*present

    def event_features(self,events):
        directives=[(e.get('command') or {}).get('macro_directive') if isinstance(e.get('command'),dict) else None for e in events]
        encoded=np.stack([directive_features(e['obs'],d) for e,d in zip(events,directives)])
        present=torch.tensor([d is not None for d in directives],dtype=torch.float32)[:,None]
        return super().event_features(events)+self.directive(torch.from_numpy(encoded))*present

    def select_directed(self,char,state,history,directive):
        if self._pending_directive is not None:raise RuntimeError("Nested directed policy selection")
        self._pending_directive=deepcopy(directive)
        try:return super().select(char,state,history)
        finally:self._pending_directive=None

    def save(self,path,**metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        metadata.setdefault('directives_trained',self.directives_trained)
        torch.save(dict(policy_type=MICRO_POLICY_TYPE,schema_hash=MICRO_SCHEMA_HASH,role=self.role,hidden=self.hidden,
            model_state_dict=self.state_dict(),**metadata),temporary)
        replace_file(temporary,path)

    @classmethod
    def load(cls,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('policy_type')!=MICRO_POLICY_TYPE or saved.get('schema_hash')!=MICRO_SCHEMA_HASH or saved.get('role')!='attacker':
            raise ValueError("Need a macro-v2 directive-conditioned micro checkpoint")
        with torch.random.fork_rng(devices=[]):
            policy=cls(saved['role'],saved['hidden']);policy.load_state_dict(saved['model_state_dict'])
        policy.directives_trained=bool(saved.get('directives_trained'))
        return policy.eval(),saved
