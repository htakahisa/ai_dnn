"""GC v3 defense controller: fresh policy in setup, search and retake."""
from . import DEFENDER_SEED
from .gc3_policy import FreshPolicy, ScratchController, SCHEMA_HASH
DefenderSearchPolicy = FreshPolicy


class GhostChampionsV3SearchController(ScratchController):
    def __init__(self, *, policy=None, checkpoint=None, recorder=None, record_phase=None):
        super().__init__(role="defender", seed=DEFENDER_SEED, policy=policy, checkpoint=checkpoint,
                         recorder=recorder, record_phase=record_phase)
