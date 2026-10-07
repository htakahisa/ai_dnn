"""Opt-in residual controller. Its defender factory remains exactly GC v1."""
import os
import numpy as np
import torch
from ..controller import GhostChampionsV2AttackerController
from .observation import ObservationEncoder,proposal_kind,OBSERVATION_SCHEMA
from .policy import ResidualPolicy
from .actions import ACTIONS,candidates,commit_recon


class LearnedAttackerController(GhostChampionsV2AttackerController):
    def __init__(self,*,policy=None,checkpoint=None,recorder=None,mode="greedy",**kwargs):
        self.encoder=ObservationEncoder()
        self.recorder=recorder
        self.mode=mode
        self.policy=policy
        self.last_policy_action=None
        if policy is None and mode!="teacher":
            self.policy,_=ResidualPolicy.load(checkpoint or os.environ["GC_V2_RL_CHECKPOINT"])
        super().__init__(**kwargs)

    def reset_round(self):
        super().reset_round()
        self.encoder.reset()
        self.last_policy_action=None

    def decide_move(self,char,state):
        name=str(char.name)
        old_cast=self.tactics.last_cast.get(name)
        old_plant=self.tactics.plant_attempt
        proposal=super().decide_move(char,state)
        if state.get("defender_setup_active") or not char.is_alive:
            return proposal
        kind=proposal_kind(proposal,char.pos)
        command=proposal[1] if isinstance(proposal,tuple) and len(proposal)>1 else None
        request=self.tactics.last_recon_request.get(name)
        generated=(isinstance(command,dict) and command.get("ability")=="RECON"
                   and self.tactics.last_cast.get(name)==self.tactics.tick and request is not None)
        if generated:
            # Evaluate alternatives without committing an unexecuted rule cast.
            self.tactics.recon_sequences[name]-=1
            self.tactics.last_recon_request.pop(name,None)
            if old_cast is None:
                self.tactics.last_cast.pop(name,None)
            else:
                self.tactics.last_cast[name]=old_cast
        obs=self.encoder.encode(char,state,self.tactics,proposal)
        choices,mask=candidates(char,state,self.tactics,proposal)
        if self.policy is None:
            selected,logp,value=kind,0.,0.
        else:
            with torch.no_grad():
                distribution,value_tensor=self.policy(torch.from_numpy(obs).unsqueeze(0),torch.from_numpy(mask).unsqueeze(0))
                selected=int(distribution.sample().item()) if self.mode=="stochastic" else int(distribution.logits.argmax(-1).item())
                logp=float(distribution.log_prob(torch.tensor([selected])).item())
                value=float(value_tensor.item())
        result=choices[selected]
        if generated and selected==kind:
            self.tactics.recon_sequences[name]+=1
            self.tactics.last_cast[name]=self.tactics.tick
            self.tactics.last_recon_request[name]=request
        elif 16<=selected<=18:
            commit_recon(char,state,self.tactics,("A","Mid","B")[selected-16],result)
        if selected!=kind:
            self.tactics.plant_attempt=old_plant
            self.tactics.remember_result(char,result)
        self.last_policy_action=ACTIONS[selected]
        if self.recorder is not None:
            self.recorder.record(name,obs,mask,selected,logp,value,kind)
            self.recorder.observed_names.update(self.tactics.sightings)
        return result

    def attacker_snapshot(self):
        snapshot=super().attacker_snapshot()
        snapshot["residual_policy"]=dict(schema_version=OBSERVATION_SCHEMA["version"],action=self.last_policy_action)
        return snapshot
