"""Small attacker-only actor/critic; incompatible observation schemas fail closed."""
from pathlib import Path
import torch
from torch import nn
from .observation import OBS_DIM,OBSERVATION_SCHEMA,SCHEMA_HASH
from .actions import ACTION_DIM,ACTIONS


class ResidualPolicy(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.hidden=hidden
        self.features=nn.Sequential(nn.Linear(OBS_DIM,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.actor=nn.Linear(hidden,ACTION_DIM)
        self.critic=nn.Linear(hidden,1)
        with torch.no_grad():
            self.actor.bias.fill_(-2)
            self.actor.bias[:4].fill_(2)

    def forward(self,obs,mask):
        features=self.features(obs)
        logits=self.actor(features).masked_fill(~mask,-1e9)
        return torch.distributions.Categorical(logits=logits),self.critic(features).squeeze(-1)

    def save(self,path,**metadata):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(policy_type="gc_v2_residual",schema=OBSERVATION_SCHEMA,schema_hash=SCHEMA_HASH,
                     actions=ACTIONS,hidden=self.hidden,model_state_dict=self.state_dict(),**metadata)
        temp=path.with_suffix(path.suffix+".tmp")
        torch.save(payload,temp)
        temp.replace(path)

    @classmethod
    def load(cls,path):
        payload=torch.load(path,map_location="cpu",weights_only=True)
        if payload.get("schema_hash")!=SCHEMA_HASH or tuple(payload.get("actions",()))!=ACTIONS:
            raise ValueError("GC v2 residual checkpoint schema mismatch (player ability/status inputs require v4); "
                             "recollect BC in a new run. For legacy viewing only, use tools/upgrade_observation_checkpoint.py")
        # Loading an optional model must not consume the match RNG stream.
        with torch.random.fork_rng(devices=[]):
            model=cls(payload["hidden"])
            model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model,payload
