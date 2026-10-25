"""The postplant policy still chooses every command by sampling."""
from ..gc3_policy import ScratchController
from .gc3_policy_attacker_guard import GuardPolicy


class GuardController(ScratchController):
    def __init__(self,*,policy=None,checkpoint=None,recorder=None):
        if policy is None:policy,_=GuardPolicy.load(checkpoint)
        super().__init__(role='attacker',seed=0,policy=policy,checkpoint=checkpoint,recorder=recorder,record_phase='postplant')

    def reset_round(self):
        super().reset_round();self.policy.reset_memory()
