"""Inference controller matching the shared route model's observation and actions."""

import io
import random
from pathlib import Path

import numpy as np
import torch

from controllers import BaseController
from concon_v1.co1_attacker_abilities import choose_ability
from concon_v1.co1_attacker_common import (
    ACTION_PLANT,
    CARDINAL_MOVES,
    GRID,
    GORIGONS,
    OBS_DIM,
    ACTION_DIM,
    RouteProgress,
    SharedRouteDQN,
    advance_team_routes,
    build_action_mask,
    build_observation,
    choose_split_assignment,
    choose_team_fire_target,
    facing_for_fire_target,
    plant_stage_action_mask,
)

from concon_v1.co1_attacker_scenarios import get_scenario, validate_checkpoint_scenario


DEFAULT_MODEL_PATH = get_scenario("A1").model_path
ENEMY_SIGHT_STOP_TICKS = 2  # 敵を視認したときに停止するtick


def _sees_enemy(char, chars, grid, game):
    """Visual sight is broader than an unobstructed firing line."""
    for enemy in chars:
        if (not getattr(enemy, "is_alive", True)
                or getattr(enemy, "team", None) == char.team):
            continue
        if game is not None and hasattr(game, "check_line_of_sight"):
            if game.check_line_of_sight(char, enemy):
                return True
        elif BaseController.has_line_of_sight(char.pos, enemy.pos, grid):
            return True
    return False


def preplant_contact_action(char, game_state, game, route_goal,
                            last_enemy_seen_tick, enemy_was_visible):
    """Use the same perceived contact behavior in training and inference."""
    chars = game_state.get("chars", [])
    grid = np.asarray(game_state.get("grid", GRID), dtype=np.int32)
    smoke_cells = game_state.get("smoke_cells", ())
    target = choose_team_fire_target(char, chars, grid, smoke_cells, game)
    tick = int(game_state.get("battle_tick", 0))
    visible = target is not None or _sees_enemy(char, chars, grid, game)
    was_visible = enemy_was_visible.get(char.name, False)
    if visible and not was_visible:
        last_enemy_seen_tick[char.name] = tick
    enemy_was_visible[char.name] = visible
    first_seen = last_enemy_seen_tick.get(char.name)
    if first_seen is not None and 0 <= tick - first_seen < ENEMY_SIGHT_STOP_TICKS:
        if target is not None:
            facing = facing_for_fire_target(char, target, chars, grid, smoke_cells, game)
            return list(char.pos), {"facing": facing}
        return list(char.pos)
    last_enemy_seen_tick.pop(char.name, None)
    ability = choose_ability(char, game, route_goal=route_goal)
    if ability is not None:
        return list(char.pos), ability
    return None


class ConconAttackerRouteController:
    """DQN route follower for one five-character attacking team.

    reset_round() samples a fresh 2:3, 3:2, 0:5, or 5:0 assignment. The model
    chooses every movement action; BFS is used only for waypoint choice and
    distance features/rewards shared with the training environment.
    """

    def __init__(self, model_path=None, seed=None, checkpoint_bytes=None,
                 model=None, map_name="A1"):
        self.scenario = get_scenario(map_name)
        self.model_path = Path(model_path) if model_path is not None else self.scenario.model_path
        self.rng = random.Random(seed)
        if model is None and checkpoint_bytes is None and not self.model_path.is_file():
            raise FileNotFoundError(
                f"ConCon {self.scenario.map_name} model not found: {self.model_path}. "
                f"Train it with: python co1_train_attacker.py -map {self.scenario.map_name}"
            )
        if model is None:
            source = io.BytesIO(checkpoint_bytes) if checkpoint_bytes is not None else self.model_path
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            validate_checkpoint_scenario(checkpoint, self.scenario)
            if checkpoint.get("obs_dim") != self.scenario.obs_dim or checkpoint.get("n_actions") != ACTION_DIM:
                raise ValueError("checkpoint observation/action dimensions do not match this controller")
            if (tuple(checkpoint.get("training_roster", ())) != GORIGONS.players
                    or checkpoint.get("spike_carrier") != GORIGONS.spike_holder):
                raise ValueError("model was not trained for Gorigons / ごんた; retrain co1_train_attacker.py")
            model = SharedRouteDQN(self.scenario.obs_dim, ACTION_DIM)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        self.model = model
        self._routes = {}
        self._pattern_index = None
        self._groups = {}
        self._a_completed_groups = set()
        self._last_enemy_seen_tick = {}
        self._enemy_was_visible = {}

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self._routes.clear()
        self._groups.clear()
        self._a_completed_groups.clear()
        self._last_enemy_seen_tick.clear()
        self._enemy_was_visible.clear()
        self._pattern_index = None

    def _prepare_round(self, chars):
        attackers = [char for char in chars if getattr(char, "team", None) == "A"]
        if len(attackers) != 5:
            raise ValueError(f"expected five attackers, got {len(attackers)}")
        if tuple(str(char.name) for char in attackers) != GORIGONS.players:
            raise ValueError("ConCon requires the Gorigons attacker roster in preset order")
        self._pattern_index, groups = choose_split_assignment(
            self.rng, len(attackers), a_point_count=len(self.scenario.waypoint_points["a"]),
        )
        self._groups = {char.name: group for char, group in zip(attackers, groups)}
        self._routes = {
            char.name: RouteProgress(group, self._pattern_index, char.pos,
                                     scenario=self.scenario)
            for char, group in zip(attackers, groups)
        }

    def _prepare_route(self, char, game_state):
        chars = game_state.get("chars", [])
        if self._pattern_index is None:
            self._prepare_round(chars)
        route = self._routes.get(char.name)
        if route is None:
            raise ValueError(f"attacker {char.name!r} was not assigned a route")

        grid = np.asarray(game_state.get("grid", route.scenario.grid), dtype=np.int32)
        position = tuple(map(int, char.pos))
        attackers = [other for other in chars if getattr(other, "team", None) == "A"]
        alive = [bool(getattr(other, "is_alive", True)) for other in attackers]
        carrier_index = next((i for i, other in enumerate(attackers)
                              if alive[i] and getattr(other, "has_spike", False)), None)
        advance_team_routes(
            [self._routes[other.name] for other in attackers],
            [tuple(map(int, other.pos)) for other in attackers],
            alive, self._a_completed_groups, grid, carrier_index,
        )
        return route

    def policy_inputs(self, char, game_state):
        """Build the policy input from the same perceived state in every mode."""
        chars = game_state.get("chars", [])
        route = self._routes[char.name]
        grid = np.asarray(game_state.get("grid", route.scenario.grid), dtype=np.int32)
        position = tuple(map(int, char.pos))
        attackers = [other for other in chars if getattr(other, "team", None) == "A"]
        alive = [bool(getattr(other, "is_alive", True)) for other in attackers]
        carrier_index = next((i for i, other in enumerate(attackers)
                              if alive[i] and getattr(other, "has_spike", False)), None)
        is_carrier = bool(getattr(char, "has_spike", False))
        allies = [
            other.pos for other in chars
            if other is not char and getattr(other, "team", None) == char.team
            and getattr(other, "is_alive", True)
        ]
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
            route.distance_map,
            route.stage == 0 and position == route.goal
            and bool({self._groups[other.name] for other, is_alive in zip(attackers, alive)
                      if is_alive} - self._a_completed_groups),
        )
        if route.at_plant_stage and not is_carrier:
            mask = plant_stage_action_mask(
                grid, position, allies, attackers[carrier_index].pos if carrier_index is not None else None,
                self._routes[attackers[carrier_index].name].goal if carrier_index is not None else None,
            )
        return observation, mask

    def _choose_policy_action(self, char, observation, mask):
        with torch.no_grad():
            values = self.model(torch.as_tensor(observation).unsqueeze(0))[0]
            values[~torch.as_tensor(mask)] = -torch.inf
            return int(values.argmax().item())

    def decide_move(self, char, game_state):
        route = self._prepare_route(char, game_state)
        position = tuple(map(int, char.pos))
        planting = char.has_spike and route.at_plant_stage and position == route.goal
        if not planting:
            contact = preplant_contact_action(
                char, game_state, getattr(self, "game", None), route.goal,
                self._last_enemy_seen_tick, self._enemy_was_visible,
            )
            if contact is not None:
                return contact
        observation, mask = self.policy_inputs(char, game_state)
        action = self._choose_policy_action(char, observation, mask)

        if action == ACTION_PLANT:
            return list(char.pos), "PLANT"
        if action < len(CARDINAL_MOVES):
            row_delta, col_delta = CARDINAL_MOVES[action]
            return [position[0] + row_delta, position[1] + col_delta]
        return list(char.pos)
