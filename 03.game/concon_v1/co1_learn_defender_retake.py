"""Production retake inference: identical observations and masks to training."""

from pathlib import Path

import torch

from controllers import BaseController
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_retake_scenarios import get_scenario, validate_checkpoint, normalize_ability_distances
from concon_v1.co1_retake_common import RetakeDQN, build_inputs, decode_action, GORIGONS
from concon_v1.co1_retake_coordination import RetakeAssembly


class ConconDefenderRetakeController(BaseController):
    def __init__(self, map_name="L", model=None, model_path=None, ability_distance=None):
        super().__init__()
        self.scenario = get_scenario(map_name)
        self.model_path, self.checkpoint_episode = None, None
        self.coordination_version = 2
        if model is None:
            path = Path(model_path) if model_path is not None else self.scenario.model_path()
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if ability_distance is None:
                ability_distance = checkpoint.get("ability_distances", checkpoint.get("ability_distance", 6))
            validate_checkpoint(checkpoint, self.scenario, ability_distance, allow_legacy_coordination=True)
            self.coordination_version = checkpoint["coordination_version"]
            model = RetakeDQN(self.scenario, foundation=checkpoint.get("foundation_version") == 1)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval().requires_grad_(False)
            self.model_path, self.checkpoint_episode = path.resolve(), int(checkpoint.get("episode", 0))
            print(f"[ConCon defender retake] site={map_name} model={self.model_path} "
                  f"episode={self.checkpoint_episode} coordination_version={self.coordination_version} "
                  "epsilon=0.000", flush=True)
        self.ability_distances = normalize_ability_distances(6 if ability_distance is None else ability_distance)
        self.model = model
        self.reset_round()

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.assignments = {name: slot for name, slot in zip(GORIGONS.players, "abcde")}
        self.sightings = {}
        self._stationary = {}
        self.combat_history = {}
        self.assembly = RetakeAssembly(self.scenario, version=self.coordination_version)

    stationary_ticks = ConconDefenderSearchController.stationary_ticks

    def choose_action(self, char, observation, mask, context):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            values = self.model(torch.as_tensor(observation, device=device).unsqueeze(0))[0]
            return int(values.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf).argmax())

    def decide_move(self, char, game_state):
        observation, mask, context = build_inputs(self, char, game_state)
        return decode_action(self.choose_action(char, observation, mask, context), char, context)
