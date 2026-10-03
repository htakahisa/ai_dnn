"""Real 5v5 pre-plant training environment for the ConCon route policy."""

import contextlib
import copy
from functools import wraps
import io
import os
import random
from pathlib import Path

import numpy as np

from concon_v1.co1_attacker_controller import ConconAttackerController
from concon_v1.co1_learn_attacker import ConconAttackerRouteController
from concon_v1.co1_attacker_common import (
    ACTION_WAIT, GORIGONS, OBS_DIM, ACTION_DIM, SharedRouteDQN, _choose_action,
)

from concon_v1.co1_attacker_scenarios import get_scenario

OPPONENTS = {
    "omoko_v1": ("omoko_gaming_v1", "Omoko Gaming"),
    "touyama_v2": ("touyama_gaming_v2", "Touyama Gaming"),
    "fnatic_v3": ("fnatic_v3", "Fnatic2023"),
    "gc_v1": ("gc_v1", "Ghost Champions"),
    "toru_ai_v3.1": ("toru_ai_v3.1", "Team Elites"),
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANT_SUCCESS_REWARD = 10.0
ELIMINATION_WIN_REWARD = 7.0


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


class TrainingRouteController(ConconAttackerRouteController):
    """Production route controller with exploration and decision recording."""

    def __init__(self, env):
        super().__init__(model=env.model, map_name=env.scenario)
        self.env = env
        self.rng = env.route_rng

    def _choose_policy_action(self, char, observation, mask):
        index = self.env.attacker_indices[char.name]
        if self.env.forced_actions is None:
            action = _choose_action(self.model, observation, mask,
                                    self.env.epsilon, self.env.action_rng,
                                    route=self._routes[char.name], position=char.pos)
        else:
            requested = int(self.env.forced_actions[index])
            action = requested if mask[requested] else ACTION_WAIT
        self.env.actions[index] = action
        self.env.policy_action_applied[index] = True
        self.env._tick_observations[index] = observation
        self.env._tick_masks[index] = mask
        route = self._routes[char.name]
        self.env._decision_routes[index] = (
            route.stage, int(route.distance_map[tuple(map(int, char.pos))]),
        )
        return action


class BattleRouteEnv:
    """Production first-round games ending at plant completion or round end."""

    def __init__(self, seed=0, opponents=None, model=None, map_name="A1", *, learn_setup=False):
        self.scenario = get_scenario(map_name)
        self.rng = random.Random(seed)
        self.route_rng = random.Random(seed)
        self.action_rng = random.Random(seed)
        self.model = model if model is not None else SharedRouteDQN(obs_dim=self.scenario.obs_dim)
        self.learn_setup = learn_setup
        self.opponents = tuple(opponents or OPPONENTS)
        if not self.opponents or any(name not in OPPONENTS for name in self.opponents):
            raise ValueError("at least one known opponent is required")
        self.game = None
        self.reset()

    @_run_from_project_root
    def reset(self):
        from party_presets import get_preset
        from run_game import VisualFPSBattle, _build_team_ai

        self.opponent = self.rng.choice(self.opponents)
        ai_key, roster_name = OPPONENTS[self.opponent]
        defenders = get_preset(roster_name)
        attacker_ai = _build_team_ai("concon_v1")
        attacker_ai.attacker_factory = lambda: ConconAttackerController(
            route_controller=TrainingRouteController(self)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.game = VisualFPSBattle(
                self.scenario.game_map, attacker_ai, _build_team_ai(ai_key), headless=True,
                attacker_roster=list(GORIGONS.players),
                defender_roster=list(defenders.players),
                spike_holder_name=GORIGONS.spike_holder,
                defender_spike_holder_name=defenders.spike_holder,
                attacker_igl_name=GORIGONS.igl, defender_igl_name=defenders.igl,
                attacker_team_name=GORIGONS.name, defender_team_name=defenders.name,
                disable_side_swap=True,
            )
            self.game.stop_after_round = True
            self.game.analytics_tracker = None
        self.controller = self.game.attacker_controller.inner_controller
        self.route_controller = self.controller.route_controller
        self.attackers = [char for char in self.game.chars if char.team == "A"]
        if tuple(char.name for char in self.attackers) != GORIGONS.players:
            raise ValueError("the training attacker roster must be Gorigons in preset order")
        self.attacker_indices = {char.name: i for i, char in enumerate(self.attackers)}
        self.forced_actions = None
        self.epsilon = 0.0
        self._reset_decision_recording()
        # Setup now calls the attacker policy too, so all policy state must
        # exist before advancing it (including on subsequent resets).
        with contextlib.redirect_stdout(io.StringIO()):
            while not self.learn_setup and self.game.defender_setup_phase.active:
                self.game.step_tick()
        # Evaluation can start after setup; training returns its decisions via step().
        self._reset_decision_recording()
        self.elapsed_ticks = 0
        self.done = False
        self.success = False
        self.retrieve_active = False
        self.had_spike_drop = False
        self.spike_recovered = False
        return self._collect()

    def _reset_decision_recording(self, current=None):
        self.actions = [ACTION_WAIT] * len(self.attackers)
        self.policy_action_applied = [False] * len(self.attackers)
        self._decision_routes = {}
        # Only actual policy decisions will enter replay. Others are placeholders.
        self._tick_observations, self._tick_masks = current if current is not None else (
            [np.zeros(self.scenario.obs_dim, dtype=np.float32) for _ in self.attackers],
            [np.eye(ACTION_DIM, dtype=bool)[ACTION_WAIT].copy() for _ in self.attackers],
        )

    @property
    def routes(self):
        routes = self.route_controller._routes or self._preview_routes
        return [routes[char.name] for char in self.attackers]

    def _sync(self):
        self.positions = [tuple(map(int, char.pos)) for char in self.attackers]
        self.alive = [bool(char.is_alive) for char in self.attackers]
        self.retrieve_active = bool(self.game.spike_pos is not None and not self.game.is_planted)

    def _collect(self):
        """Preview bootstrap inputs without changing live routes or perception caches.

        Actions use inputs captured inside the production decide_move call instead.
        """
        self._sync()
        observations, masks, preview_stages = [], [], []
        perception = copy.copy(self.game.current_attacker_team_ai.perception_engine)
        perception._cache = {}
        perception._last_tick = None
        perception._defuse_touched_viewers = set(perception._defuse_touched_viewers)
        for char in self.attackers:
            if self.game.defender_setup_phase.active:
                from iq_perception import build_team_position_view
                view = build_team_position_view(self.game, char.team)
            else:
                view = perception.build_game_view(viewer=char, game=self.game)
            actor = view.perceived_character_for(char)
            preview = copy.copy(self.route_controller)
            # Route progress changes by assigning fields/new distance maps.
            # Share the read-only terrain and existing maps, but keep each
            # viewer's progress separate from the live controller and others.
            preview._routes = {
                name: copy.copy(route) for name, route in self.route_controller._routes.items()
            }
            preview._a_completed_groups = set(self.route_controller._a_completed_groups)
            preview.rng = random.Random()
            preview.rng.setstate(self.route_rng.getstate())
            state = {"grid": view.grid, "chars": view.chars,
                     "battle_tick": self.game.battle_tick,
                     "defender_setup_active": self.game.defender_setup_phase.active}
            preview.set_game(view)
            preview._prepare_route(actor, state)
            observation, mask = preview.policy_inputs(actor, state)
            if not char.is_alive or self.retrieve_active or self.game.is_planted:
                mask[:] = False
                mask[ACTION_WAIT] = True
            observations.append(observation)
            masks.append(mask)
            preview_stages.append(preview._routes[char.name].stage)
            self._preview_routes = preview._routes
        self._preview_stages = preview_stages
        return observations, masks

    @_run_from_project_root
    def step(self, actions=None, *, current=None, epsilon=0.0, action_rng=None):
        if self.done:
            raise RuntimeError("reset the environment before stepping a finished round")
        self._sync()
        was_retrieving = self.retrieve_active
        was_planted = bool(self.game.is_planted)
        self.route_active_before_step = not was_retrieving and not was_planted
        self.forced_actions = actions
        self.epsilon = epsilon
        if action_rng is not None:
            self.action_rng = action_rng
        self._reset_decision_recording(current)
        with contextlib.redirect_stdout(io.StringIO()):
            self.game.step_tick()
        self.elapsed_ticks += 1
        next_observations, next_masks = self._collect()
        self.had_spike_drop |= was_retrieving or self.retrieve_active
        self.spike_recovered |= bool(
            was_retrieving and not self.retrieve_active
            and any(char.is_alive and char.has_spike for char in self.attackers)
        )
        planted = bool(self.game.is_planted)
        newly_planted = planted and not was_planted
        team_reward = plant_advantage_reward(
            sum(self.alive), sum(c.is_alive for c in self.game.chars if c.team == "D"),
            newly_planted,
        )
        rewards = [-0.005 + team_reward] * len(self.attackers)
        for index, (stage, distance_before) in self._decision_routes.items():
            route = self.route_controller._routes[self.attackers[index].name]
            # Arrival may advance only the bootstrap preview until the next
            # live decision. Reward the same stage change the target observes.
            if self._preview_stages[index] != stage:
                rewards[index] += 0.25
            elif distance_before >= 0:
                distance = int(route.distance_map[self.positions[index]])
                if distance >= 0:
                    rewards[index] += 0.04 * (distance_before - distance)
        if self.route_active_before_step and self.retrieve_active:
            rewards = [reward - 3.0 for reward in rewards]
        self.success = planted
        won_by_elimination = bool(self.game.round_over and self.game.attacker_wins and not planted)
        self.done = bool(planted or self.game.round_over)
        if newly_planted:
            rewards = [reward + PLANT_SUCCESS_REWARD for reward in rewards]
        elif won_by_elimination:
            rewards = [reward + ELIMINATION_WIN_REWARD for reward in rewards]
        elif self.done and not planted:
            rewards = [reward - 3.0 for reward in rewards]
        return (self._tick_observations, self._tick_masks, rewards,
                next_observations, next_masks, self.done)
