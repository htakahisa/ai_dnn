"""Touyama v3 game factory entrypoint."""
from touyama_v3.tv3_game_controller import TouyamaV3GameAttackerController

class Tv3TouyamaAttackerController(TouyamaV3GameAttackerController):
    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
