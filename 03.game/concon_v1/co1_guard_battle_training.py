"""Real postplant rollouts with production IQ perception and dedicated opponent AIs."""

import contextlib
import io
import random

import numpy as np

from concon_v1.co1_battle_training import OPPONENTS, _run_from_project_root
from concon_v1.co1_guard_scenarios import get_scenario
from concon_v1.co1_guard_common import (
    GuardDQN, WAIT_ACTION, ACTION_DIM, ABILITIES, observation_dim, aim_alignment,
    bfs_distance_map, GORIGONS,
)
from concon_v1.co1_learn_guard import ConconGuardController
from concon_v1.co1_attacker_sighting import facing_towards
from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR
from concon_v1.co1_guard_rewards import GAMMA, DEATH_PENALTY, ROUND_REWARD, decision_reward

START_MODES = ("hold", "transition", "smoke", "pressure", "pressure_tap")


class TrainingGuardController(ConconGuardController):
    def __init__(self, env):
        super().__init__(map_name=env.scenario, model=env.model, seed=env.rng.randrange(2**32))
        self.env = env

    def choose_action(self, char, observation, mask, context):
        index = self.env.indices[char.name]
        self.env.finish_pending(index, observation, mask, False)
        valid = np.flatnonzero(mask)
        if self.env.forced_actions is not None:
            requested = self.env.forced_actions[index]
            action = int(requested) if mask[int(requested)] else WAIT_ACTION
        elif (not (self.model.navigation and not context["tap"] and not context["fireable"]
                   and not context.get("impaired", False) and not mask[40:].any())
              and self.env.action_rng.random() < self.env.epsilon):
            action = int(self.env.action_rng.choice(valid.tolist()))
        else:
            action = super().choose_action(char, observation, mask, context)
        self.env.pending[index] = {"obs": observation.astype(np.float16), "action": action,
                                   "reward": 0.0, "duration": 0}
        self.env.decisions[index] = (action, context)
        return action


class GuardBattleEnv:
    def __init__(self, seed=0, opponents=None, model=None, map_name="L", start_modes=START_MODES):
        self.scenario = get_scenario(map_name)
        self.rng = random.Random(seed)
        self.action_rng = random.Random(seed + 1)
        self.model = model if model is not None else GuardDQN(self.scenario)
        self.opponents = tuple(opponents or OPPONENTS)
        self.start_modes = tuple(start_modes)
        if not self.opponents or any(name not in OPPONENTS for name in self.opponents):
            raise ValueError("guard requires known opponents")
        if not self.start_modes or any(mode not in START_MODES for mode in self.start_modes):
            raise ValueError("unknown guard start mode")
        self.game = None
        self.epsilon = 0.0
        self.forced_actions = None

    def _sample_cells(self, distance_map, minimum, maximum, count, occupied):
        candidates = [(int(r), int(c)) for r, c in zip(*np.where(
            (distance_map >= minimum) & (distance_map <= maximum)))
            if (int(r), int(c)) not in occupied and self.scenario.grid[r, c] != 1]
        if len(candidates) < count:
            raise ValueError("not enough reachable cells for guard start state")
        return self.rng.sample(candidates, count)

    @_run_from_project_root
    def reset(self, start_mode=None, attacker_count=None, defender_count=None, opponent=None):
        from controllers import DefaultDefenderController
        from party_presets import get_preset
        with contextlib.redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle, _build_team_ai
        from team_ai import DualRoleTeamAI
        from game_core import SPIKE_DETONATION_TICKS
        if opponent is not None and opponent not in self.opponents:
            raise ValueError("opponent must belong to the configured guard opponents")
        self.opponent = opponent if opponent is not None else self.rng.choice(self.opponents)
        self.start_mode = start_mode or self.rng.choice(self.start_modes)
        if self.start_mode not in START_MODES:
            raise ValueError("unknown guard start mode")
        for count in (attacker_count, defender_count):
            if count is not None and not 1 <= count <= 5:
                raise ValueError("survivor counts must be between one and five")
        ai_key, roster_name = OPPONENTS[self.opponent]
        defenders = get_preset(roster_name)
        self.indices = {name: index for index, name in enumerate(GORIGONS.players)}
        self.pending = [None] * 5
        self.decisions = {}
        self.transitions = []
        self.forced_actions = None
        self.done = False
        self.elapsed_ticks = 0
        self._quiet_history = {}
        self.metrics = {"tap_ticks": 0, "blocked_tap_decisions": 0,
                        "approach_decisions": 0, "recon_on_tap": 0,
                        "stationary_fire_decisions": 0, "two_tick_fire_decisions": 0,
                        "quiet_decisions": 0, "quiet_utility_decisions": 0, "quiet_at_goal_decisions": 0,
                        "quiet_leave_goal_decisions": 0, "quiet_reversals": 0,
                        "quiet_bad_facing_decisions": 0, "fireable_decisions": 0,
                        "moving_fire_decisions": 0, "bad_fire_facing_decisions": 0,
                        "impaired_decisions": 0, "exposed_wait_decisions": 0,
                        "cover_moves": 0, "counter_utility_decisions": 0}
        attacker_ai = DualRoleTeamAI(
            "ConCon guard", attacker_factory=lambda: TrainingGuardController(self),
            defender_factory=DefaultDefenderController, use_iq_perception=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.game = VisualFPSBattle(
                GAME_MAZE_STR, attacker_ai, _build_team_ai(ai_key), headless=True,
                attacker_roster=list(GORIGONS.players), defender_roster=list(defenders.players),
                spike_holder_name=GORIGONS.spike_holder,
                defender_spike_holder_name=defenders.spike_holder,
                attacker_igl_name=GORIGONS.igl, defender_igl_name=defenders.igl,
                attacker_team_name=GORIGONS.name, defender_team_name=defenders.name,
                disable_side_swap=True,
            )
        self.game.stop_after_round = True
        self.game.analytics_tracker = None
        self.game.defender_setup_phase.finish()
        self.controller = self.game.attacker_controller.inner_controller
        self.attackers = [char for char in self.game.chars if char.team == "A"]
        self.defenders = [char for char in self.game.chars if char.team == "D"]
        if tuple(char.name for char in self.attackers) != GORIGONS.players:
            raise ValueError("unexpected guard training roster")
        attacker_count = attacker_count if attacker_count is not None else self.rng.randint(1, 5)
        defender_count = defender_count if defender_count is not None else self.rng.randint(1, 5)
        for team, count in ((self.attackers, attacker_count), (self.defenders, defender_count)):
            survivors = set(self.rng.sample([char.name for char in team], count))
            for char in team:
                char.is_alive = char.name in survivors
                char.hp = max(1, int(char.max_hp * self.rng.uniform(0.6, 1))) if char.is_alive else 0
                char.has_spike = False
                char.is_planting = False
                char.plant_timer = char.defuse_timer = 0
                char.moved_this_tick = False
                char.moved_last_tick = False
                char.forced_facing_next_tick = None
                char.los_revealed = False
                char.reveal_remaining = 0
                # Train ready, partly charged and spent ultimate states.
                char.ultimate_points = self.rng.choice((0, char.ultimate_cost // 2, char.ultimate_cost))
                # Spent utility is part of the postplant start distribution.
                for ability in ABILITIES:
                    attribute = ability.lower() + "_charges"
                    if getattr(char, attribute, 0) and self.rng.random() < 0.4:
                        setattr(char, attribute, 0)
        self.controller.prepare_assignments(self.game.chars)
        self.game.is_planted = True
        self.game.planted_pos = list(self.rng.choice(self.scenario.plant_cells))
        self.game.target_plant_pos = list(self.game.planted_pos)
        self.game.spike_pos = None
        self.game.detonate_timer = SPIKE_DETONATION_TICKS
        distance_map = bfs_distance_map(self.scenario.grid, self.game.planted_pos)
        alive_attackers = [char for char in self.attackers if char.is_alive]
        alive_defenders = [char for char in self.defenders if char.is_alive]
        if self.start_mode == "hold":
            attacker_positions = [self.scenario.positions[self.controller.assignments[char.name]]
                                  for char in alive_attackers]
        else:
            attacker_positions = self._sample_cells(distance_map, 2, 6, len(alive_attackers), set())
        for char, position in zip(alive_attackers, attacker_positions):
            char.pos = list(position)
            aim = self.scenario.facing_points[self.controller.assignments[char.name]]
            char.facing = facing_towards(position, aim) or "N"
        occupied = set(attacker_positions)
        if self.start_mode in ("smoke", "pressure_tap"):
            # The defender model continues normally after this already-started tap.
            nearby = [(int(r), int(c)) for r, c in zip(*np.where(distance_map >= 0))
                      if max(abs(r - self.game.planted_pos[0]), abs(c - self.game.planted_pos[1])) <= 1
                      and (int(r), int(c)) not in occupied]
            defuser = alive_defenders[0]
            defuser.pos = list(self.rng.choice(nearby))
            defuser.defuse_timer = 1
            self.game.active_defuser_name = defuser.name
            occupied.add(tuple(defuser.pos))
            rest = self._sample_cells(distance_map, 2, 8, len(alive_defenders) - 1, occupied)
            for char, position in zip(alive_defenders[1:], rest):
                char.pos = list(position)
            # A sampled active smoke represents utility thrown before this snapshot.
            r, c = self.game.planted_pos
            self.game.smokes = [{
                "cells": {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                          if 0 <= rr < self.game.height and 0 <= cc < self.game.width
                          and self.game.grid[rr, cc] != 1},
                "remaining_ticks": 10, "owner": "postplant_snapshot", "team": "D",
                "center": (r, c),
            }]
        else:
            positions = self._sample_cells(distance_map, 8, 20, len(alive_defenders), occupied)
            for char, position in zip(alive_defenders, positions):
                char.pos = list(position)
        for char in alive_defenders:
            char.facing = facing_towards(char.pos, self.game.planted_pos) or "S"
        if self.start_mode in ("pressure", "pressure_tap"):
            # Snapshots after enemy utility, with normal IQ and opponent AI
            # thereafter. Include recon through smoke, where smoke is no cover.
            if self.start_mode == "pressure":
                positions = self._sample_cells(distance_map, 2, 8, len(alive_defenders), occupied)
                for char, position in zip(alive_defenders, positions):
                    char.pos = list(position)
                    char.facing = facing_towards(position, self.game.planted_pos) or "S"
            else:
                alive_defenders[0].defuse_timer = self.rng.randint(1, 4)
            perception = self.game.current_attacker_team_ai.perception_engine
            perception.clear_cache()
            # Preserve only what the team could perceive before the effect.
            for viewer in alive_attackers:
                view = perception.build_game_view(viewer=viewer, game=self.game)
                for enemy in view.chars:
                    if (enemy.team != viewer.team and enemy.is_alive and enemy.position_known
                            and 0 <= enemy.pos[0] < self.game.height
                            and 0 <= enemy.pos[1] < self.game.width):
                        self.controller.sightings[enemy.name] = (tuple(enemy.pos), self.game.battle_tick)
            effect = self.rng.choice(("blind_remaining", "reveal_remaining", "electric_remaining"))
            for char in alive_attackers:
                setattr(char, effect, self.rng.randint(2, 4))
        self.game.current_attacker_team_ai.perception_engine.clear_cache()
        self.game.current_defender_team_ai.perception_engine.clear_cache()
        return {"opponent": self.opponent, "start_mode": self.start_mode,
                "attacker_alive": attacker_count, "defender_alive": defender_count}

    def finish_pending(self, index, observation, mask, terminal):
        pending = self.pending[index]
        if pending is None:
            return
        self.transitions.append((pending["obs"], pending["action"], pending["reward"],
                                 observation.astype(np.float16), mask.copy(), float(terminal),
                                 pending["duration"]))
        self.pending[index] = None

    def _terminal_inputs(self):
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[WAIT_ACTION] = True
        return np.zeros(observation_dim(self.scenario), dtype=np.float16), mask

    def step(self, epsilon=0.0, actions=None):
        if self.done or self.game is None:
            raise RuntimeError("reset before stepping an active guard episode")
        self.epsilon = float(epsilon)
        self.forced_actions = actions
        if actions is not None and (len(actions) != 5 or any(not 0 <= int(a) < ACTION_DIM for a in actions)):
            raise ValueError("provide five valid guard action indices")
        self.decisions = {}
        self.transitions = []
        hp_before = [char.hp for char in self.defenders]
        alive_before = [char.is_alive for char in self.attackers]
        tap_before = any(char.is_alive and char.defuse_timer > 0 for char in self.defenders)
        with contextlib.redirect_stdout(io.StringIO()):
            self.game.step_tick()
        self.elapsed_ticks += 1
        self.metrics["tap_ticks"] += int(tap_before)
        self.done = bool(self.game.round_over or self.game.match_over)
        if not self.done and self.elapsed_ticks >= 100:
            self.done = True
        terminal_reward = ROUND_REWARD if self.game.attacker_wins else -ROUND_REWARD
        team_damage = sum(max(0, before - char.hp) for before, char in zip(hp_before, self.defenders))
        rewards = [0.0] * 5
        for index, (action, context) in self.decisions.items():
            char = self.attackers[index]
            position = tuple(char.pos)
            operation = action // 8
            facing = char.facing  # engine can override facing after incoming fire
            approach = context["tap"] and (not context["fireable"] or context.get("impaired", False))
            goal = context["spike"] if approach else context["goal"]
            previous_distance = context["distance_spike"] if approach else context["distance_goal"]
            new_distance = int(bfs_distance_map(self.scenario.grid, goal)[position])
            reward = decision_reward(action, context, position, facing, new_distance)
            reward += min(team_damage / 1000, 0.1)
            stationary = position == context["position"]
            if context.get("impaired", False):
                self.metrics["impaired_decisions"] += 1
                self.metrics["counter_utility_decisions"] += int(operation >= 5)
                before = context["exposures"][context["position"]]
                after = context["exposures"].get(position, before)
                self.metrics["cover_moves"] += int(not stationary and after < before)
                self.metrics["exposed_wait_decisions"] += int(stationary and operation < 5 and before > 0)
            elif context["tap"] and not context["fireable"]:
                self.metrics["blocked_tap_decisions"] += 1
                if not stationary and new_distance < previous_distance:
                    self.metrics["approach_decisions"] += 1
                if 88 <= action < 112:
                    self.metrics["recon_on_tap"] += 1
            elif context["fireable"]:
                self.metrics["fireable_decisions"] += 1
                alignment = aim_alignment(position, context["target"], facing)
                self.metrics["moving_fire_decisions"] += int(not stationary)
                self.metrics["bad_fire_facing_decisions"] += int(alignment <= 0)
                if stationary and alignment > 0.7:
                    self.metrics["stationary_fire_decisions"] += 1
                    if context["stopped"] >= 1:
                        self.metrics["two_tick_fire_decisions"] += 1
            else:
                self.metrics["quiet_decisions"] += 1
                self.metrics["quiet_utility_decisions"] += int(operation >= 5)
                self.metrics["quiet_bad_facing_decisions"] += int(
                    aim_alignment(position, context["aim"], facing) <= 0)
                if context["position"] == context["goal"]:
                    self.metrics["quiet_at_goal_decisions"] += 1
                    self.metrics["quiet_leave_goal_decisions"] += int(not stationary)
                previous = self._quiet_history.get(index)
                if (not stationary and previous is not None
                        and previous[0] == context["tick"] - 1
                        and previous[1] == position and previous[2] == context["position"]):
                    self.metrics["quiet_reversals"] += 1
                self._quiet_history[index] = (context["tick"], context["position"], position)
            if context["tap"] or context["fireable"] or context.get("impaired", False):
                self._quiet_history.pop(index, None)
            rewards[index] = reward
        # Keep the final decision of a dead actor until the round ends. Otherwise
        # dying early avoids the defeat reward. Count every elapsed tick, even
        # when death or an engine status prevents another policy decision.
        for index, char in enumerate(self.attackers):
            pending = self.pending[index]
            if pending is not None:
                reward = rewards[index]
                if alive_before[index] and not char.is_alive:
                    reward -= DEATH_PENALTY
                if self.done:
                    reward += terminal_reward
                pending["reward"] += GAMMA ** pending["duration"] * reward
                pending["duration"] += 1
                rewards[index] = reward
        terminal_observation, terminal_mask = self._terminal_inputs()
        for index, char in enumerate(self.attackers):
            if self.done:
                self.finish_pending(index, terminal_observation, terminal_mask, True)
        return self.transitions, rewards, self.done

    def result(self):
        if not self.done:
            raise RuntimeError("guard episode has not finished")
        game = self.game
        reason = ("defused" if game.is_defused else "detonated" if game.detonate_timer <= 0
                  else "defender_eliminated" if game.attacker_wins else "timeout")
        return {"opponent": self.opponent, "start_mode": self.start_mode,
                "winner": "A" if game.attacker_wins else "D", "end_reason": reason,
                "ticks": self.elapsed_ticks,
                "attacker_alive": sum(char.is_alive for char in self.attackers),
                "defender_alive": sum(char.is_alive for char in self.defenders), **self.metrics}
