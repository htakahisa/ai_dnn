"""Runtime adapter for the concon_v1 route-learning controller."""

import random
import hashlib

from controllers import BaseController, DefaultAttackerController

from concon_v1.co1_learn_attacker import (
    DEFAULT_MODEL_PATH,
    ConconAttackerRouteController,
)
from concon_v1.co1_attacker_retrieve import ConconAttackerRetrieveController
from concon_v1.co1_attacker_sighting import TeamEnemySightings
from concon_v1.co1_attacker_abilities import FixedSmokePlan, FixedFlashPlan, FixedReconPlan
from concon_v1.co1_attacker_scenarios import get_scenario
from concon_v1.co1_attacker_selection import (
    SELECTION_PATH, load_selection, route_candidates, defending_opponent,
)


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
        self.postplant_controller = None
        self.enemy_sightings = TeamEnemySightings()
        scenario = getattr(self.route_controller, "scenario", get_scenario(map_name))
        self.fixed_smokes = FixedSmokePlan(scenario)
        self.fixed_flashes = FixedFlashPlan(scenario)
        self.fixed_recons = FixedReconPlan(scenario)
        self.retrieve_controller.allow_smoke = not scenario.smoke_points
        self.retrieve_controller.allow_flash = True
        self.retrieve_controller.allow_recon = True

    def set_game(self, game):
        self.game = game
        self.route_controller.set_game(game)
        self.retrieve_controller.set_game(game)
        if getattr(self, "postplant_controller", None) is not None:
            self.postplant_controller.set_game(game)

    def reset_round(self):
        self.route_controller.reset_round()
        self.retrieve_controller.reset_round()
        self.default_controller.reset_round()
        if getattr(self, "postplant_controller", None) is not None:
            self.postplant_controller.reset_round()
        self.enemy_sightings.reset_round()
        self.fixed_smokes.reset_round()
        self.fixed_flashes.reset_round()
        self.fixed_recons.reset_round()

    def decide_move(self, char, game_state):
        if not game_state.get("is_planted") and not game_state.get("defender_setup_active"):
            smoke = self.fixed_smokes.choose(char, getattr(self, "game", None))
            if smoke is not None:
                return list(char.pos), smoke
            flash = self.fixed_flashes.choose(char, getattr(self, "game", None))
            if flash is not None:
                return list(char.pos), flash
            recon = self.fixed_recons.choose(char, getattr(self, "game", None))
            if recon is not None:
                return list(char.pos), recon
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
            controller = getattr(self, "postplant_controller", None) or self.default_controller
            result = controller.decide_move(char, game_state)
        elif game_state.get("spike_pos") is not None:
            result = self.retrieve_controller.decide_move(char, game_state)
        elif carrier is None:
            result = self.default_controller.decide_move(char, game_state)
        else:
            result = self.route_controller.decide_move(char, game_state)
        return self.enemy_sightings.guard_move(result, char.pos, tick)


class ConconRoundAttackerController(BaseController):
    """Randomly choose an evaluated route for the opposing AI each round.

    Frozen models are loaded once. Route, utility and retrieval state belong
    to the selected scenario; all five attackers share that selection.
    Postplant factories construct controllers with the BaseController API.
    """

    def __init__(self, map_names=("A1", "A2", "A3", "A4"), seed=None,
                 postplant_factories=None, selection_path=SELECTION_PATH):
        super().__init__()
        names = (map_names,) if isinstance(map_names, str) else tuple(map_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("attacker map candidates must be non-empty and unique")
        self.rng = random.Random(seed)
        self.map_names = names
        self.controllers = {}
        self.model_hashes = {}
        self.selection_report = load_selection(selection_path) if selection_path is not None else None
        for name in names:
            scenario = get_scenario(name)
            model_path = scenario.model_path
            if not model_path.is_file():
                model_path = scenario.save_dir / scenario.checkpoint_filename("latest")
            checkpoint_bytes = model_path.read_bytes()
            self.model_hashes[name] = hashlib.sha256(checkpoint_bytes).hexdigest()
            self.controllers[name] = ConconAttackerController(
                map_name=name, model_path=model_path,
                checkpoint_bytes=checkpoint_bytes,
                seed=self.rng.randrange(2 ** 32),
            )
        self.postplant_factories = {
            site: tuple(factories)
            for site, factories in (postplant_factories or {}).items()
        }
        if any(site not in ("left", "right") for site in self.postplant_factories):
            raise ValueError("postplant sites must be left or right")
        if any(not callable(factory) for factories in self.postplant_factories.values()
               for factory in factories):
            raise ValueError("postplant candidates must be controller factories")
        self._postplant_cache = {}
        self.current_map_name = None
        self.current_controller = None
        self._postplant_selected = False

    def set_game(self, game):
        self.game = game
        if self.current_controller is not None:
            self.current_controller.set_game(game)

    def reset_round(self):
        opponent = defending_opponent(getattr(self, "game", None))
        self.current_candidates = route_candidates(self.selection_report, opponent, self.map_names, self.model_hashes)
        self.current_map_name = self.rng.choice(self.current_candidates)
        self.current_controller = self.controllers[self.current_map_name]
        self.current_controller.postplant_controller = None
        if getattr(self, "game", None) is not None:
            self.current_controller.set_game(self.game)
        self.current_controller.reset_round()
        self._postplant_selected = False

    def decide_move(self, char, game_state):
        # Also support runtimes that first bind/act before calling reset_round.
        if self.current_controller is None:
            self.reset_round()
        if game_state.get("is_planted") and not self._postplant_selected:
            position = game_state.get("planted_pos")
            if position is not None:
                site = "left" if position[1] < len(game_state["grid"][0]) / 2 else "right"
                factories = self.postplant_factories.get(site, ())
                if factories:
                    index = self.rng.randrange(len(factories))
                    key = (site, index)
                    if key not in self._postplant_cache:
                        self._postplant_cache[key] = factories[index]()
                    controller = self._postplant_cache[key]
                    if getattr(self, "game", None) is not None:
                        controller.set_game(self.game)
                    controller.reset_round()
                    self.current_controller.postplant_controller = controller
                self._postplant_selected = True
        return self.current_controller.decide_move(char, game_state)


Ov1AttackerController = ConconAttackerController
