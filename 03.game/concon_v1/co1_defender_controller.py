"""ConCon-specific defender adapter using the game's standard defender AI."""

from controllers import BaseController, DefaultDefenderController


class ConconDefenderController(BaseController):
    def __init__(self):
        super().__init__()
        self.default_controller = DefaultDefenderController()

    def reset_round(self):
        reset_round = getattr(self.default_controller, "reset_round", None)
        if callable(reset_round):
            reset_round()

    def decide_move(self, char, game_state):
        return self.default_controller.decide_move(char, game_state)


Ov1DefenderController = ConconDefenderController