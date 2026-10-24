"""GC v3 attack controller: fresh policy in every phase."""
from . import ATTACKER_SEED
from .gc3_policy import FreshPolicy, ScratchController, SCHEMA_HASH
GuardPolicy = FreshPolicy


class GhostChampionsV3AttackerController(ScratchController):
    def __init__(self, *, policy=None, checkpoint=None, recorder=None, record_phase=None):
        super().__init__(role="attacker", seed=ATTACKER_SEED, policy=policy, checkpoint=checkpoint,
                         recorder=recorder, record_phase=record_phase)
