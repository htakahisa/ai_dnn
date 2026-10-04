"""Shared postplant inference for all registered ConCon guard patterns."""

import io
import random
from pathlib import Path

import numpy as np
import torch

from controllers import BaseController
from concon_v1.co1_guard_scenarios import get_scenario, validate_checkpoint
from concon_v1.co1_guard_common import (
    GuardDQN, ACTION_DIM, observation_dim, build_inputs, decode_action, GORIGONS,
)


class ConconGuardController(BaseController):
    def __init__(self, map_name="L", model_path=None, checkpoint_bytes=None, model=None, seed=0):
        super().__init__()
        self.scenario = get_scenario(map_name)
        self.rng = random.Random(seed)
        if model is None:
            path = Path(model_path) if model_path is not None else self.scenario.model_path
            source = io.BytesIO(checkpoint_bytes) if checkpoint_bytes is not None else path
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, self.scenario)
            if (checkpoint.get("obs_dim") != observation_dim(self.scenario)
                    or checkpoint.get("n_actions") != ACTION_DIM
                    or tuple(checkpoint.get("training_roster", ())) != GORIGONS.players):
                raise ValueError("guard checkpoint dimensions/roster do not match")
            with torch.random.fork_rng(devices=[]):
                model = GuardDQN(self.scenario)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        self.model = model
        self.reset_round()

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self.assignments = {}
        self.sightings = {}
        self._stationary = {}
        self._assigned = False

    def prepare_assignments(self, chars):
        if self._assigned:
            return
        names = [other.name for other in chars if other.team == "A" and other.is_alive]
        if len(names) > 5 or any(name not in GORIGONS.players for name in names):
            raise ValueError("guard requires up to five surviving Gorigons attackers")
        slots = list("abcde")
        self.rng.shuffle(slots)
        self.assignments = dict(zip(names, slots))
        self._assigned = True

    def stationary_ticks(self, char, tick):
        position = tuple(char.pos)
        previous = self._stationary.get(char.name)
        if previous is not None and previous[0] == tick:
            return previous[2]
        count = 0
        if (previous is not None and previous[0] == tick - 1 and previous[1] == position
                and not getattr(char, "moved_this_tick", False)):
            count = previous[2] + 1
        self._stationary[char.name] = (tick, position, count)
        return count

    def policy_inputs(self, char, state):
        self.prepare_assignments(state.get("chars", []))
        return build_inputs(self, char, state)

    def choose_action(self, char, observation, mask, context):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            values = self.model(torch.as_tensor(observation, device=device).unsqueeze(0))[0]
            values = values.masked_fill(~torch.as_tensor(mask, device=device), -torch.inf)
            return int(values.argmax().item())

    def decide_move(self, char, game_state):
        if not game_state.get("is_planted"):
            return list(char.pos)
        observation, mask, context = self.policy_inputs(char, game_state)
        action = self.choose_action(char, observation, mask, context)
        return decode_action(action, tuple(char.pos), context["targets"])


def guard_factory(map_name="L", model_path=None):
    """Factory for CONCON_ATTACKER_POSTPLANT_MODELS, loaded only after planting."""
    return lambda: ConconGuardController(map_name=map_name, model_path=model_path)
