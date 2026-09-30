"""Runtime adapter for the concon_v1 A1 route-learning controller."""

from controllers import BaseController, DefaultAttackerController

from concon_v1.co1_learn_attacker_A1 import (
    DEFAULT_MODEL_PATH,
    ConconAttackerA1Controller,
)


class ConconAttackerController(BaseController):
    """Use the learned route before planting and default behavior otherwise."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, seed=None):
        super().__init__()
        self.route_controller = ConconAttackerA1Controller(model_path=model_path, seed=seed)
        self.default_controller = DefaultAttackerController()

    def set_game(self, game):
        self.game = game
        self.route_controller.set_game(game)

    def reset_round(self):
        self.route_controller.reset_round()
        self.default_controller.reset_round()

    def decide_move(self, char, game_state):
        chars = game_state.get("chars", [])
        carrier = next(
            (
                other for other in chars
                if getattr(other, "is_alive", True)
                and getattr(other, "team", None) == "A"
                and getattr(other, "has_spike", False)
            ),
            None,
        )
        if game_state.get("is_planted") or carrier is None or game_state.get("spike_pos") is not None:
            return self.default_controller.decide_move(char, game_state)
        return self.route_controller.decide_move(char, game_state)


Ov1AttackerController = ConconAttackerController