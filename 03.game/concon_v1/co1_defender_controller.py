"""Use learned ConCon search and site-specific retake checkpoints."""

from pathlib import Path

from controllers import BaseController, DefaultDefenderController
from concon_v1.co1_defender_scenario import get_scenario
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
from concon_v1.co1_retake_scenarios import get_scenario as retake_scenario, plant_site
from concon_v1.co1_retake_models import game_opponent, runtime_model_path


class ConconDefenderController(BaseController):
    def __init__(self, model_path=None, search_controller=None, retake_model_paths=None):
        super().__init__()
        self.default_controller = DefaultDefenderController()
        self.retake_controllers = {}
        self._explicit_retake_paths = retake_model_paths is not None
        self._loaded_retake_paths = {}
        paths = ({site: retake_scenario(site).model_path("best") for site in ("L", "R")}
                 if retake_model_paths is None else retake_model_paths)
        if set(paths) != {"L", "R"}:
            raise ValueError("retake model paths must specify both L and R")
        self.retake_model_paths = {site: Path(path).resolve() for site, path in paths.items()}
        self._missing_retake_warned = set()
        path = model_path if model_path is not None else get_scenario().runtime_model_path
        self.search_controller = search_controller if search_controller is not None else (ConconDefenderSearchController(model_path=path)
                                  if model_path is not None or Path(path).exists() else None)

    def set_game(self, game):
        self.game = game
        for controller in (self.default_controller, self.search_controller, *self.retake_controllers.values()):
            if controller is not None and hasattr(controller, "set_game"):
                controller.set_game(game)

    def reset_round(self):
        reset_round = getattr(self.default_controller, "reset_round", None)
        if callable(reset_round):
            reset_round()
        if self.search_controller is not None:
            self.search_controller.reset_round()
        for controller in self.retake_controllers.values():
            controller.reset_round()

    def decide_move(self, char, game_state):
        if self.search_controller is not None and not game_state.get("is_planted"):
            return self.search_controller.decide_move(char, game_state)
        if game_state.get("is_planted") and game_state.get("planted_pos") is not None:
            site = plant_site(game_state["planted_pos"], game_state["grid"])
            path = (self.retake_model_paths[site] if self._explicit_retake_paths else
                    runtime_model_path(site, game_opponent(getattr(self, "game", None))).resolve())
            if self._loaded_retake_paths.get(site) != path:
                self.retake_controllers.pop(site, None)
            if site not in self.retake_controllers and path.is_file():
                self.retake_controllers[site] = ConconDefenderRetakeController(site, model_path=path)
                self._loaded_retake_paths[site] = path
                if hasattr(self, "game"):
                    self.retake_controllers[site].set_game(self.game)
            if site in self.retake_controllers:
                return self.retake_controllers[site].decide_move(char, game_state)
            if site not in self._missing_retake_warned:
                print(f"[ConCon defender retake] site={site} checkpoint missing: {path}; "
                      "using default defender controller", flush=True)
                self._missing_retake_warned.add(site)
        return self.default_controller.decide_move(char, game_state)


Ov1DefenderController = ConconDefenderController
