"""Sample five destinations, five orders and one team plant cell from public history."""
from copy import deepcopy
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn
from ..preplant.gc3_policy_attacker_history import HistoryPolicy,DATA_SCHEMA_HASH
from ..preplant.gc3_observation_attacker_history import encode_public,EXTRA_WIDTH
from ..gc3_observation import CELLS
from ..gc3_spatial_policy import sampled_distribution
from ..gc3_io import replace_file
from ..macro_v1.gc3_policy_attacker_macro import MacroHistoryView
from .gc3_directives_attacker_macro import INSTRUCTIONS,PLAN_WIDTH,plan_features
from .gc3_candidates_attacker_macro import MacroCandidates

POLICY_TYPE='gc3_attacker_macro_v2'
SCHEMA_HASH=hashlib.sha256((DATA_SCHEMA_HASH+'five-destinations-five-orders-one-plant-causal-v2').encode()).hexdigest()


class MacroPolicy(HistoryPolicy):
    def __init__(self,role='attacker',hidden=128):
        super().__init__(role,hidden)
        del self.mode,self.ally,self.spatial,self.move
        self.destination=nn.Linear(hidden,5*CELLS)
        self.instruction=nn.Linear(hidden,5*len(INSTRUCTIONS))
        self.plant=nn.Linear(hidden,CELLS)
        self.plan_input=nn.Linear(PLAN_WIDTH,hidden)
        self.schema_hash=SCHEMA_HASH

    def contexts(self,rows,*,cache=False):
        plans=np.stack([plan_features(r.get('current_plan')) for r in rows])
        return super().contexts(rows,cache=cache)+self.plan_input(torch.from_numpy(plans))

    def event_features(self,events):
        plans=np.stack([plan_features(e.get('macro_plan')) for e in events])
        return super().event_features(events)+self.plan_input(torch.from_numpy(plans))

    def distributions(self,context,masks):
        size=context.shape[:-1]
        return dict(destination=sampled_distribution(self.destination(context).reshape(*size,5,CELLS),masks['destination']),
            instruction=sampled_distribution(self.instruction(context).reshape(*size,5,len(INSTRUCTIONS)),masks['instruction']),
            plant=sampled_distribution(self.plant(context),masks['plant']))

    def select_macro(self,char,state,history,current_plan,candidates):
        candidates.validate_arena(state['grid'])
        own=sorted((c for c in state['chars'] if c.team==char.team),key=lambda c:str(c.name))[:5]
        roster=[str(c.name) for c in own]
        active=np.array([i<len(own) and own[i].is_alive for i in range(5)],dtype=np.bool_)
        destination=np.tile(candidates.destination_mask,(5,1))
        instruction=np.ones((5,len(INSTRUCTIONS)),dtype=np.bool_)
        for slot in np.flatnonzero(~active):
            # Dummy dead/missing slots contribute neither actions nor PPO terms.
            destination[slot]=False;destination[slot,0]=True
            instruction[slot]=False;instruction[slot,0]=True
        observation=encode_public(char,state,history)
        observation[-EXTRA_WIDTH+6]=2.
        row=dict(obs=observation,phase='macro',history=history,history_index=len(history.events),
            current_plan=deepcopy(current_plan),active=active,roster=roster,
            masks=dict(destination=destination,instruction=instruction,plant=candidates.plant_mask.copy()),selections={})
        with torch.no_grad():
            context=self.contexts([row],cache=True)[0]
            distributions=self.distributions(context,row['masks'])
            chosen={key:dist.sample() for key,dist in distributions.items()}
            logp=(distributions['destination'].log_prob(chosen['destination'])*torch.from_numpy(active)).sum()
            logp+=(distributions['instruction'].log_prob(chosen['instruction'])*torch.from_numpy(active)).sum()
            logp+=distributions['plant'].log_prob(chosen['plant'])
            row.update(selections={key:value.tolist() for key,value in chosen.items()},
                       logp=float(logp),value=float(self.critic(context)))
        plan=dict(destinations=[MacroCandidates.coordinate(value) if active[i] else None for i,value in enumerate(row['selections']['destination'])],
            instructions=[INSTRUCTIONS[value] if active[i] else None for i,value in enumerate(row['selections']['instruction'])],
            plant=MacroCandidates.coordinate(row['selections']['plant']))
        return plan,row

    def evaluate_records(self,rows):
        context=self.contexts(rows)
        masks={key:np.stack([r['masks'][key] for r in rows]) for key in ('destination','instruction','plant')}
        distributions=self.distributions(context,masks)
        active=torch.from_numpy(np.stack([r['active'] for r in rows]))
        logp=context.new_zeros(len(rows));entropy=logp.clone()
        for key,dist in distributions.items():
            choices=torch.as_tensor([r['selections'][key] for r in rows],dtype=torch.long)
            term,ent=dist.log_prob(choices),dist.entropy()
            if key!='plant':term,ent=(term*active).sum(-1),(ent*active).sum(-1)
            logp=logp+term;entropy=entropy+ent
        # Value fitting must not move the BC actor's public/history encoder.
        # Keep the architecture and checkpoint schema identical for BC reuse.
        return logp,entropy,self.critic(context.detach()).squeeze(-1)

    def actor_parameters(self):
        return [parameter for name,parameter in self.named_parameters() if not name.startswith('critic.')]

    def value_parameters(self):
        return list(self.critic.parameters())

    def save(self,path,**metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        torch.save(dict(policy_type=POLICY_TYPE,schema_hash=SCHEMA_HASH,role=self.role,hidden=self.hidden,
                        model_state_dict=self.state_dict(),**metadata),temporary)
        replace_file(temporary,path)

    @classmethod
    def load(cls,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('policy_type')!=POLICY_TYPE or saved.get('schema_hash')!=SCHEMA_HASH or saved.get('role')!='attacker':
            raise ValueError('Attacker macro v2 checkpoint mismatch; v1 weights cannot be used')
        with torch.random.fork_rng(devices=[]):
            model=cls(saved['role'],saved['hidden']);model.load_state_dict(saved['model_state_dict'])
        return model.eval(),saved
