"""Real 5v5 pre-plant training environment for the ConCon A1 route policy."""

import contextlib
from functools import wraps
import io
import os
import random
from pathlib import Path

from concon_v1.co1_attacker_retrieve import ConconAttackerRetrieveController
from concon_v1.co1_attacker_sighting import TeamEnemySightings
from concon_v1.co1_learn_attacker_A1 import preplant_contact_action
from concon_v1.co1_train_attacker_A1 import (
    ACTION_PLANT, ACTION_WAIT, CARDINAL_MOVES, GRID, GORIGONS,
    MAX_TICKS, RouteProgress, advance_team_routes, build_action_mask,
    build_observation, choose_split_assignment, plant_stage_action_mask,
)


OPPONENTS = {
    "omoko_v1": ("omoko_gaming_v1", "Omoko Gaming"),
    "touyama_v2": ("touyama_gaming_v2", "Touyama Gaming"),
    "fnatic_v3": ("fnatic_v3", "Fnatic2023"),
    "gc_v1": ("gc_v1", "Ghost Champions"),
    "toru_ai_v3.1": ("toru_ai_v3.1", "Team Elites"),
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_from_project_root(function):
    """Resolve legacy opponent model paths independently of the launch folder."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous_directory = Path.cwd()
        os.chdir(PROJECT_ROOT)
        try:
            return function(*args, **kwargs)
        finally:
            os.chdir(previous_directory)
    return wrapped


def plant_advantage_reward(attackers_alive, defenders_alive, planted=False):
    """Value the team advantage at a successful plant, and only then."""
    if not planted:
        return 0.0
    return 0.5 * (int(attackers_alive) - int(defenders_alive))


class TrainingAttackerController:
    def __init__(self, env):
        self.env = env
        self._last_enemy_seen_tick = {}
        self._enemy_was_visible = {}

    def set_game(self, game):
        self.game = game

    def reset_round(self):
        self._last_enemy_seen_tick.clear()
        self._enemy_was_visible.clear()

    def decide_move(self, char, game_state):
        tick = int(game_state.get("battle_tick", 0))
        sightings = self.env.enemy_sightings
        sightings.observe(char, game_state.get("chars", []), self.game,
                          game_state["grid"], tick)
        if game_state.get("spike_pos") is not None and not game_state.get("is_planted"):
            self.env.retrieve_controller.set_game(self.game)
            result = self.env.retrieve_controller.decide_move(char, game_state)
            return sightings.guard_move(result, char.pos, tick)
        index = self.env.attacker_indices[char.name]
        route = self.env.routes[index]
        planting = char.has_spike and route.at_plant_stage and tuple(char.pos) == route.goal
        if not planting:
            contact = preplant_contact_action(
                char, game_state, self.game, route.goal,
                self._last_enemy_seen_tick, self._enemy_was_visible,
            )
            if contact is not None:
                return sightings.guard_move(contact, char.pos, tick)
        self.env.policy_action_applied[index] = True
        action = self.env.actions[index]
        if action == ACTION_PLANT:
            return list(char.pos), "PLANT"
        if action < len(CARDINAL_MOVES):
            dr, dc = CARDINAL_MOVES[action]
            result = [int(char.pos[0]) + dr, int(char.pos[1]) + dc]
        else:
            result = list(char.pos)
        return sightings.guard_move(result, char.pos, tick)


class BattleRouteEnv:
    """One actual 5v5 round per episode; defenders use production AI."""

    def __init__(self, seed=0, opponents=None):
        self.rng = random.Random(seed)
        self.opponents = tuple(opponents or OPPONENTS)
        if not self.opponents or any(name not in OPPONENTS for name in self.opponents):
            raise ValueError("at least one known opponent is required")
        self.game = None
        self.reset()

    @_run_from_project_root
    def reset(self):
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from team_ai import DualRoleTeamAI
        from controllers import DefaultDefenderController

        self.opponent = self.rng.choice(self.opponents)
        ai_key, roster_name = OPPONENTS[self.opponent]
        defenders = get_preset(roster_name)
        attacker_ai = DualRoleTeamAI(
            "ConCon A1 training",
            attacker_factory=lambda: TrainingAttackerController(self),
            defender_factory=DefaultDefenderController,
            use_iq_perception=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle, _build_team_ai
            self.game = VisualFPSBattle(
                NEW_MAZE_STR, attacker_ai, _build_team_ai(ai_key),
                headless=True,
                attacker_roster=list(GORIGONS.players),
                defender_roster=list(defenders.players),
                spike_holder_name=GORIGONS.spike_holder,
                defender_spike_holder_name=defenders.spike_holder,
                attacker_igl_name=GORIGONS.igl,
                defender_igl_name=defenders.igl,
                disable_side_swap=True,
            )
            while self.game.defender_setup_phase.active:
                self.game._run_defender_setup_tick()
        self.game.stop_after_round = True
        self.game.analytics_tracker = None
        self.retrieve_controller = ConconAttackerRetrieveController()
        self.retrieve_controller.set_game(self.game)
        self.enemy_sightings = TeamEnemySightings()
        self.attackers = [char for char in self.game.chars if char.team == "A"]
        if tuple(char.name for char in self.attackers) != GORIGONS.players:
            raise ValueError("the training attacker roster must be Gorigons in preset order")
        self.attacker_indices = {char.name: i for i, char in enumerate(self.attackers)}
        self.pattern_index, groups = choose_split_assignment(self.rng)
        self.routes = [RouteProgress(group, self.pattern_index, char.pos)
                       for group, char in zip(groups, self.attackers)]
        self._a_completed_groups = set()
        self.actions = [ACTION_WAIT] * len(self.attackers)
        self.policy_action_applied = [False] * len(self.attackers)
        self.elapsed_ticks = 0
        self.done = False
        self.success = False
        self.retrieve_active = False
        self.had_spike_drop = False
        self.spike_recovered = False
        self._sync()
        return self._collect()

    def _sync(self):
        self.positions = [tuple(map(int, char.pos)) for char in self.attackers]
        self.alive = [bool(char.is_alive) for char in self.attackers]
        self.plant_progress = int(getattr(self.attackers[GORIGONS.players.index(
            GORIGONS.spike_holder)], "plant_timer", 0))

    def _collect(self):
        self._sync()
        self.retrieve_active = bool(self.game.spike_pos is not None and not self.game.is_planted)
        carrier_index = next((i for i, char in enumerate(self.attackers)
                              if char.is_alive and char.has_spike), None)
        advance_team_routes(self.routes, self.positions, self.alive,
                            self._a_completed_groups, GRID, carrier_index)
        observations, masks = [], []
        perception = self.game.current_attacker_team_ai.perception_engine
        for index, (char, route) in enumerate(zip(self.attackers, self.routes)):
            view = perception.build_game_view(viewer=char, game=self.game)
            actor = view.perceived_character_for(char)
            allies = [other.pos for other in view.chars
                      if other is not actor and other.team == "A" and other.is_alive]
            is_carrier = bool(actor.has_spike and actor.is_alive)
            observations.append(build_observation(
                route, actor.pos, is_carrier, allies,
                getattr(actor, "plant_timer", 0), self.game.battle_tick, view.grid,
            ))
            mask = build_action_mask(
                view.grid, actor.pos, allies, is_carrier, route.at_plant_stage,
                route.goal, route.distance_map,
                route.stage == 0 and tuple(actor.pos) == route.goal
                and bool({self.routes[i].group for i, alive in enumerate(self.alive)
                          if alive} - self._a_completed_groups),
            )
            if route.at_plant_stage and not is_carrier:
                mask = plant_stage_action_mask(
                    view.grid, actor.pos, allies,
                    view.perceived_character_for(self.attackers[carrier_index]).pos
                    if carrier_index is not None else None,
                    self.routes[carrier_index].goal if carrier_index is not None else None,
                )
            if not char.is_alive or self.retrieve_active:
                mask[:] = False
                mask[ACTION_WAIT] = True
            masks.append(mask)
        return observations, masks

    @_run_from_project_root
    def step(self, actions, *, current=None):
        observations, masks = current if current is not None else self._collect()
        route_active_before = not self.retrieve_active
        was_retrieving = self.retrieve_active
        self.actions = [int(action) if masks[i][int(action)] else ACTION_WAIT
                        for i, action in enumerate(actions)]
        self.policy_action_applied = [False] * len(self.attackers)
        previous_stages = [route.stage for route in self.routes]
        previous_distances = [int(route.distance_map[pos])
                              for route, pos in zip(self.routes, self.positions)]
        self.game._prepare_team_controllers_tick()
        self.game._build_occupancy_counts()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for char in self.game._move_order():
                    if char.is_alive:
                        self.game.move_character(char)
                self.game.process_battle()
                self.game._advance_combo_announcement()
        finally:
            self.game._clear_occupancy_counts()
        self.elapsed_ticks += 1
        next_observations, next_masks = self._collect()
        self.route_active_before_step = route_active_before
        self.had_spike_drop |= was_retrieving or self.retrieve_active
        self.spike_recovered |= bool(
            was_retrieving and not self.retrieve_active
            and any(char.is_alive and char.has_spike for char in self.attackers)
        )
        after_attackers = sum(self.alive)
        after_defenders = sum(c.is_alive for c in self.game.chars if c.team == "D")
        planted = bool(self.game.is_planted)
        team_reward = plant_advantage_reward(
            after_attackers, after_defenders, planted,
        )
        rewards = [-0.005 + team_reward] * len(self.attackers)
        for index, (route, pos) in enumerate(zip(self.routes, self.positions)):
            if route.stage != previous_stages[index]:
                rewards[index] += 0.25
            elif previous_distances[index] >= 0:
                distance = int(route.distance_map[pos])
                if distance >= 0:
                    rewards[index] += 0.04 * (previous_distances[index] - distance)
        if route_active_before and self.retrieve_active:
            # Preserve the carrier-loss signal while keeping the round alive.
            rewards = [reward - 3.0 for reward in rewards]
        self.success = planted
        won_by_elimination = bool(self.game.round_over and self.game.attacker_wins)
        # The game ends pre-plant rounds on timeout or either team's wipe.
        # A dropped spike is an intermediate retrieve phase, not a loss.
        self.done = self.success or self.game.round_over
        if self.success or won_by_elimination:
            rewards = [reward + 10.0 for reward in rewards]
        elif self.done:
            rewards = [reward - 3.0 for reward in rewards]
        return observations, masks, rewards, next_observations, next_masks, self.done
