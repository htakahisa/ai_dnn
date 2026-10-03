"""Runtime adapter for the concon_v1 route-learning controller."""

from controllers import BaseController, DefaultAttackerController

from concon_v1.co1_learn_attacker import (
    DEFAULT_MODEL_PATH,
    ConconAttackerRouteController,
)
from concon_v1.co1_attacker_retrieve import ConconAttackerRetrieveController
from concon_v1.co1_attacker_sighting import TeamEnemySightings
from concon_v1.co1_attacker_abilities import FixedSmokePlan, FixedFlashPlan
from concon_v1.co1_attacker_scenarios import get_scenario


class ConconAttackerController(BaseController):
    """Use the learned route before planting and default behavior otherwise."""

    def __init__(self, model_path=None, seed=None, checkpoint_bytes=None,
                 route_controller=None, map_name="A1"):
        super().__init__()
        self.route_controller = route_controller if route_controller is not None else ConconAttackerRouteController(
            model_path=model_path, seed=seed, checkpoint_bytes=checkpoint_bytes, map_name=map_name
        )
        self.retrieve_controller = ConconAttackerRetrieveController()
        self.default_controller = DefaultAttackerController()
        self.enemy_sightings = TeamEnemySightings()
        scenario = getattr(self.route_controller, "scenario", get_scenario(map_name))
        self.fixed_smokes = FixedSmokePlan(scenario)
        self.fixed_flashes = FixedFlashPlan(scenario)
        self.retrieve_controller.allow_smoke = not scenario.smoke_points
        self.retrieve_controller.allow_flash = not scenario.flash_points

    def set_game(self, game):
        self.game = game
        self.route_controller.set_game(game)
        self.retrieve_controller.set_game(game)

    def reset_round(self):
        self.route_controller.reset_round()
        self.retrieve_controller.reset_round()
        self.default_controller.reset_round()
        self.enemy_sightings.reset_round()
        self.fixed_smokes.reset_round()
        self.fixed_flashes.reset_round()

    def decide_move(self, char, game_state):
        if not game_state.get("is_planted"):
            smoke = self.fixed_smokes.choose(char, getattr(self, "game", None))
            if smoke is not None:
                return list(char.pos), smoke
            flash = self.fixed_flashes.choose(char, getattr(self, "game", None))
            if flash is not None:
                return list(char.pos), flash
        chars = game_state.get("chars", [])
        tick = int(game_state.get("battle_tick", 0))
        self.enemy_sightings.observe(
            char, chars, getattr(self, "game", None), game_state["grid"], tick,
        )
        carrier = next(
            (
                other for other in chars
                if getattr(other, "is_alive", True)
                and getattr(other, "team", None) == "A"
                and getattr(other, "has_spike", False)
            ),
            None,
        )
        if game_state.get("is_planted"):
            result = self.default_controller.decide_move(char, game_state)
        elif game_state.get("spike_pos") is not None:
            result = self.retrieve_controller.decide_move(char, game_state)
        elif carrier is None:
            result = self.default_controller.decide_move(char, game_state)
        else:
            result = self.route_controller.decide_move(char, game_state)
        return self.enemy_sightings.guard_move(result, char.pos, tick)


Ov1AttackerController = ConconAttackerController
