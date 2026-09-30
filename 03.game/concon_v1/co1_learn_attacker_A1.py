"""Inference controller matching co1_train_attacker_A1's observation and actions."""

import random
from pathlib import Path

import numpy as np
import torch

from concon_v1.co1_train_attacker_A1 import (
    ACTION_PLANT,
    CARDINAL_MOVES,
    DEFAULT_SAVE_DIR,
    GRID,
    LEFT_PLANT_CELLS,
    OBS_DIM,
    ACTION_DIM,
    RouteProgress,
    SharedRouteDQN,
    build_action_mask,
    build_observation,
    choose_split_assignment,
)


DEFAULT_MODEL_PATH = DEFAULT_SAVE_DIR / "co1_attacker_A1_best.pt"


class ConconAttackerA1Controller:
    """DQN route follower for one five-character attacking team.

    reset_round() samples a fresh 2:3, 3:2, 0:5, or 5:0 assignment. The model
    chooses every movement action; BFS is used only for waypoint choice and
    distance features/rewards shared with the training environment.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, seed=None):
        self.model_path = Path(model_path)
        self.rng = random.Random(seed)
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if checkpoint.get("obs_dim") != OBS_DIM or checkpoint.get("n_actions") != ACTION_DIM:
            raise ValueError("checkpoint observation/action dimensions do not match this controller")
        self.model = SharedRouteDQN(OBS_DIM, ACTION_DIM)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self._routes = {}
        self._pattern_index = None
        self._groups = {}

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self._routes.clear()
        self._groups.clear()
        self._pattern_index = None

    def _prepare_round(self, chars):
        attackers = [char for char in chars if getattr(char, "team", None) == "A"]
        if len(attackers) != 5:
            raise ValueError(f"expected five attackers, got {len(attackers)}")
        self._pattern_index, groups = choose_split_assignment(self.rng, len(attackers))
        self._groups = {char.name: group for char, group in zip(attackers, groups)}
        self._routes = {
            char.name: RouteProgress(group, self._pattern_index, char.pos, GRID)
            for char, group in zip(attackers, groups)
        }

    def decide_move(self, char, game_state):
        chars = game_state.get("chars", [])
        if self._pattern_index is None:
            self._prepare_round(chars)
        route = self._routes.get(char.name)
        if route is None:
            raise ValueError(f"attacker {char.name!r} was not assigned a route")

        grid = np.asarray(game_state.get("grid", GRID), dtype=np.int32)
        position = tuple(map(int, char.pos))
        if route.advance_if_reached(position, grid):
            pass
        allies = [
            other.pos for other in chars
            if other is not char and getattr(other, "team", None) == char.team
            and getattr(other, "is_alive", True)
        ]
        is_carrier = bool(getattr(char, "has_spike", False))
        observation = build_observation(
            route,
            position,
            is_carrier,
            allies,
            getattr(char, "plant_timer", 0),
            game_state.get("battle_tick", 0),
            grid,
        )
        mask = build_action_mask(
            grid, position, allies, is_carrier, route.at_plant_stage, route.goal,
        )
        with torch.no_grad():
            values = self.model(torch.as_tensor(observation).unsqueeze(0))[0]
            values[~torch.as_tensor(mask)] = -torch.inf
            action = int(values.argmax().item())

        if action == ACTION_PLANT:
            return list(char.pos), "PLANT"
        if action < len(CARDINAL_MOVES):
            row_delta, col_delta = CARDINAL_MOVES[action]
            return [position[0] + row_delta, position[1] + col_delta]
        return list(char.pos)
