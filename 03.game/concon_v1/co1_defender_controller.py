"""Use learned ConCon positioning before planting and standard retake after."""

from pathlib import Path

from controllers import BaseController, DefaultDefenderController
from concon_v1.co1_defender_scenario import get_scenario
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController


class ConconDefenderController(BaseController):
    def __init__(self, model_path=None, search_controller=None):
        super().__init__()
        self.default_controller = DefaultDefenderController()
        path = model_path if model_path is not None else get_scenario().runtime_model_path
        self.search_controller = search_controller if search_controller is not None else (ConconDefenderSearchController(model_path=path)
                                  if model_path is not None or Path(path).exists() else None)

    def set_game(self, game):
        self.game = game
        for controller in (self.default_controller, self.search_controller):
            if controller is not None and hasattr(controller, "set_game"):
                controller.set_game(game)

    def reset_round(self):
        reset_round = getattr(self.default_controller, "reset_round", None)
        if callable(reset_round):
            reset_round()
        if self.search_controller is not None:
            self.search_controller.reset_round()

    def decide_move(self, char, game_state):
        if self.search_controller is not None and not game_state.get("is_planted"):
            return self.search_controller.decide_move(char, game_state)
        return self.default_controller.decide_move(char, game_state)


Ov1DefenderController = ConconDefenderController
