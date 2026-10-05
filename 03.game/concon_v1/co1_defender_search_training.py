"""Real IQ search rollouts ending at planting or a pre-plant round outcome."""

import contextlib
import io
import random

import numpy as np
import torch

from concon_v1.co1_defender_common import DefenderSearchDQN
from concon_v1.co1_battle_training import OPPONENTS, _run_from_project_root
from concon_v1.co1_defender_scenario import get_scenario
from concon_v1.co1_defender_controller import ConconDefenderController
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_defender_search_common import ACTION_DIM, observation_dim, GORIGONS, SUPPORT_DISTANCE
from concon_v1.co1_defender_search_rewards import (
    GAMMA, ROUND_REWARD, DEATH_PENALTY, prepare_reward_context, decision_reward, search_score, facing_target,
)
START_MODES = ("round",)


class SearchTrainingAdapter(ConconDefenderController):
    def decide_move(self, char, game_state):
        if game_state.get("is_planted"):
            # An attacker may complete planting midway through movement order.
            # No defender retake action may run even in that final search tick.
            return list(char.pos), {"facing": char.facing}
        return super().decide_move(char, game_state)


class TrainingSearchController(ConconDefenderSearchController):
    def __init__(self, env):
        super().__init__(model=env.model, seed=env.rng.randrange(2**32))
        self.env = env

    def choose_action(self, char, observation, mask, context):
        index = self.env.indices[char.name]
        self.env.finish_pending(index, observation, mask, False)
        if context["active"] and not context["setup"]:
            if self.env.forced_actions is not None:
                action = int(self.env.forced_actions[index])
                if not mask[action]:
                    raise ValueError(f"requested search action {action} is not legal for {char.name}")
            elif self.env.action_rng.random() < self.env.epsilon:
                action = int(self.env.action_rng.choice(np.flatnonzero(mask).tolist()))
            else:
                action = super().choose_action(char, observation, mask, context)
            self.env.pending[index] = dict(obs=observation.astype(np.float16), action=action, reward=0., duration=0)
        else:
            # Exploration must not replace the frozen quiet/setup basic skill.
            action = super().choose_action(char, observation, mask, context)
        reward_context = prepare_reward_context(context, self.scenario.grid, self.env.support_distance)
        if context["active"] and not context["fireable"] and not context["post_kill_hold"] and not context["disclosed"]:
            with torch.no_grad():
                device = next(self.model.parameters()).device
                basic = torch.as_tensor(observation[:self.model.basic_dim], device=device).unsqueeze(0)
                values = DefenderSearchDQN.forward(self.model, basic)[0]
                values = values.masked_fill(~torch.as_tensor(mask[:40], device=device), -torch.inf)
                reward_context["foundation_move"] = int(values.argmax()) // 8
        self.env.decisions[index] = (action, reward_context)
        return action


class DefenderSearchEnv:
    def __init__(self, model, seed=0, opponents=None, support_distance=SUPPORT_DISTANCE):
        self.model, self.scenario = model, get_scenario()
        self.rng, self.action_rng = random.Random(seed), random.Random(seed + 1)
        self.opponents = tuple(OPPONENTS if opponents is None else opponents)
        if not self.opponents or any(name not in OPPONENTS for name in self.opponents):
            raise ValueError("choose known defender search opponents")
        if support_distance < 1:
            raise ValueError("support distance must be positive")
        self.support_distance = support_distance
        self.game = None

    @_run_from_project_root
    def reset(self, start_mode="round", opponent=None):
        if start_mode not in START_MODES:
            raise ValueError("search training/evaluation must start a normal 5v5 round from spawn")
        if opponent is not None and opponent not in self.opponents:
            raise ValueError("choose an opponent from this environment's roster")
        from controllers import DefaultAttackerController
        from party_presets import get_preset
        from team_ai import DualRoleTeamAI
        from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR
        with contextlib.redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle, _build_team_ai
        self.opponent = self.rng.choice(self.opponents) if opponent is None else opponent
        ai_key, roster_name = OPPONENTS[self.opponent]
        attackers = get_preset(roster_name)
        self.indices = {name: index for index, name in enumerate(GORIGONS.players)}
        self.pending = [None] * 5
        self.decisions, self.transitions = {}, []
        self.done, self.epsilon, self.forced_actions = False, 0., None
        self.elapsed_ticks, self.search_ticks, self.start_mode = 0, 0, start_mode
        self.planted = False
        self.plant_attacker_alive = self.plant_defender_alive = None
        self.end_reason = None
        self.metrics = dict(fire_decisions=0, moving_fire_decisions=0, aligned_fire_decisions=0,
            normal_fire_decisions=0, aligned_normal_fire_decisions=0,
            post_kill_decisions=0, stationary_aligned_post_kill=0, support_opportunities=0, support_progress=0,
            memory_return_decisions=0, aligned_memory_returns=0, flash_casts=0, smoke_casts=0,
            recon_casts=0, enemy_flash_ticks=0, ally_flash_ticks=0, distant_post_departures=0,
            quiet_decisions=0, quiet_post_departures=0)
        self.metrics.update(memory_motion_decisions=0, memory_navigation_errors=0)
        self.controller = TrainingSearchController(self)
        self.adapter = SearchTrainingAdapter(search_controller=self.controller)
        defender_ai = DualRoleTeamAI("ConCon search", DefaultAttackerController,
                                     lambda: self.adapter, use_iq_perception=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.game = VisualFPSBattle(GAME_MAZE_STR, _build_team_ai(ai_key), defender_ai,
                headless=True, attacker_roster=list(attackers.players), defender_roster=list(GORIGONS.players),
                spike_holder_name=attackers.spike_holder, defender_spike_holder_name=GORIGONS.spike_holder,
                attacker_igl_name=attackers.igl, defender_igl_name=GORIGONS.igl,
                attacker_team_name=attackers.name, defender_team_name=GORIGONS.name, disable_side_swap=True)
        self.game.stop_after_round, self.game.analytics_tracker = True, None
        self.attackers = [char for char in self.game.chars if char.team == "A"]
        self.defenders = [char for char in self.game.chars if char.team == "D"]
        self.controller.prepare_assignments(self.game.chars)
        self.game.current_attacker_team_ai.perception_engine.clear_cache()
        self.game.current_defender_team_ai.perception_engine.clear_cache()
        return dict(opponent=self.opponent, start_mode=start_mode,
                    attacker_alive=sum(char.is_alive for char in self.attackers),
                    defender_alive=sum(char.is_alive for char in self.defenders))

    def finish_pending(self, index, observation, mask, terminal):
        pending = self.pending[index]
        if pending is not None:
            self.transitions.append((pending["obs"], pending["action"], pending["reward"],
                observation.astype(np.float16), mask.copy(), float(terminal), pending["duration"]))
            self.pending[index] = None

    @_run_from_project_root
    def step(self, epsilon=0., actions=None):
        if self.game is None or self.done:
            raise RuntimeError("reset an active search episode before stepping")
        if actions is not None and (len(actions) != 5 or any(not 0 <= int(action) < ACTION_DIM for action in actions)):
            raise ValueError("provide five valid search actions")
        self.epsilon, self.forced_actions = float(epsilon), actions
        self.decisions, self.transitions = {}, []
        setup_before = self.game.defender_setup_phase.active
        hp_before = [char.hp for char in self.defenders]
        alive_before = [char.is_alive for char in self.defenders]
        enemy_hp_before = sum(char.hp for char in self.attackers)
        kills_before = [char.round_kills for char in self.defenders]
        charges_before = [{ability: getattr(char, ability.lower() + "_charges", 0)
                           for ability in ("SMOKE", "FLASH", "RECON")} for char in self.defenders]
        with contextlib.redirect_stdout(io.StringIO()):
            self.game.step_tick()
        self.elapsed_ticks += 1
        self.search_ticks += int(not setup_before and not self.planted)
        just_planted = self.game.is_planted and not self.planted
        self.planted |= bool(self.game.is_planted)
        attacker_alive = sum(char.is_alive for char in self.attackers)
        defender_alive = sum(char.is_alive for char in self.defenders)
        if just_planted:
            self.plant_attacker_alive, self.plant_defender_alive = attacker_alive, defender_alive
        self.end_reason = ("attacker_eliminated" if attacker_alive == 0 else "planted" if self.planted
                           else "defender_eliminated" if defender_alive == 0 else "timeout"
                           if self.game.round_over and not self.game.attacker_wins else "truncated"
                           if self.game.round_over or self.game.match_over else None)
        self.done = self.end_reason is not None
        team_damage = max(0., enemy_hp_before - sum(char.hp for char in self.attackers)) / 1000
        rewards = [0.] * 5
        from concon_v1.co1_guard_common import aim_alignment
        for index, (action, context) in self.decisions.items():
            if context["setup"]:
                continue
            char = self.defenders[index]
            position, previous = tuple(char.pos), context["position"]
            moved = position != previous
            target = facing_target(context, position)
            reward = decision_reward(action, context, position, char.facing)
            reward += min(team_damage, .15)
            reward += .2 * max(0, char.round_kills - kills_before[index])
            reward -= .002 * max(0, hp_before[index] - char.hp)
            if context["fireable"]:
                self.metrics["fire_decisions"] += 1
                self.metrics["moving_fire_decisions"] += int(moved and not context["neutralized"])
                self.metrics["aligned_fire_decisions"] += int(not moved and target is not None and aim_alignment(position, target, char.facing) >= .7)
                if not context["neutralized"]:
                    self.metrics["normal_fire_decisions"] += 1
                    self.metrics["aligned_normal_fire_decisions"] += int(not moved and target is not None and aim_alignment(position, target, char.facing) >= .7)
            if context["post_kill_hold"]:
                self.metrics["post_kill_decisions"] += 1
                self.metrics["stationary_aligned_post_kill"] += int(not moved and target is not None and aim_alignment(position, target, char.facing) >= .7)
            if "foundation_move" in context:
                self.metrics["memory_motion_decisions"] += 1
                operation = action // 8 if action < 40 else 4
                self.metrics["memory_navigation_errors"] += int(operation != context["foundation_move"])
            if context["support_goal"] is not None:
                self.metrics["support_opportunities"] += 1
                distances = context["reward_distances"]
                self.metrics["support_progress"] += int(distances[position] < distances[previous])
            elif context["active"] and not context["fireable"] and not context["post_kill_hold"] and previous == context["goal"]:
                self.metrics["distant_post_departures"] += int(moved)
            distances = context["reward_distances"]
            returning = context["support_goal"] is None and distances[position] < distances[previous]
            if context["active"] and target is not None and not context["fireable"] and moved and returning:
                self.metrics["memory_return_decisions"] += 1
                self.metrics["aligned_memory_returns"] += int(aim_alignment(position, target, char.facing) >= .7)
            if not context["active"]:
                self.metrics["quiet_decisions"] += 1
                self.metrics["quiet_post_departures"] += int(previous == context["goal"] and moved)
            for ability in charges_before[index]:
                self.metrics[ability.lower() + "_casts"] += max(0, charges_before[index][ability] - getattr(char, ability.lower() + "_charges", 0))
            rewards[index] = reward
        if not setup_before and not self.planted:
            self.metrics["enemy_flash_ticks"] += sum(char.is_alive and char.blind_remaining > 0 for char in self.attackers)
            self.metrics["ally_flash_ticks"] += sum(char.is_alive and char.blind_remaining > 0 for char in self.defenders)
        score = search_score(self.end_reason, defender_alive, attacker_alive)
        terminal_reward = ROUND_REWARD * (2 * score - 1)
        # Dead actors also receive the search endpoint reward. Planting ends
        # the episode immediately; no retake decisions or rewards are collected.
        for index, char in enumerate(self.defenders):
            pending = self.pending[index]
            if pending is None:
                continue
            reward = rewards[index]
            if alive_before[index] and not char.is_alive:
                reward -= DEATH_PENALTY
            if self.done:
                reward += terminal_reward
            pending["reward"] += GAMMA ** pending["duration"] * reward
            pending["duration"] += 1
            rewards[index] = reward
        if self.done:
            observation = np.zeros(observation_dim(self.scenario), dtype=np.float16)
            mask = np.zeros(ACTION_DIM, dtype=bool)
            mask[32] = True
            for index in range(5):
                self.finish_pending(index, observation, mask, True)
        return self.transitions, rewards, self.done

    def result(self):
        if not self.done:
            raise RuntimeError("finish the search episode before reading its result")
        attacker_alive = sum(char.is_alive for char in self.attackers)
        defender_alive = sum(char.is_alive for char in self.defenders)
        won = self.end_reason in ("attacker_eliminated", "timeout")
        return dict(opponent=self.opponent, start_mode=self.start_mode,
                    winner="D" if won else "A" if self.end_reason == "defender_eliminated" else None,
                    end_reason=self.end_reason, ticks=self.elapsed_ticks, search_ticks=self.search_ticks,
                    planted=self.planted, attacker_alive=sum(char.is_alive for char in self.attackers),
                    defender_alive=defender_alive,
                    plant_attacker_alive=getattr(self, "plant_attacker_alive", None),
                    plant_defender_alive=getattr(self, "plant_defender_alive", None),
                    search_score=search_score(self.end_reason, defender_alive, attacker_alive), **self.metrics)
