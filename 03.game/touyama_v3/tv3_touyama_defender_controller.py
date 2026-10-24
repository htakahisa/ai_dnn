"""Touyama v3 game factory entrypoint."""
from touyama_v3.tv3_game_controller import TouyamaV3GameDefenderController

class Tv3TouyamaDefenderController(TouyamaV3GameDefenderController):
    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
