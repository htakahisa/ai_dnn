"""Normal IQ 5v5 rounds; record retake transitions only after a real plant."""

import contextlib
import io

import numpy as np

from concon_v1.co1_battle_training import _run_from_project_root
from concon_v1.co1_defender_search_training import DefenderSearchEnv
from concon_v1.co1_defender_controller import ConconDefenderController
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
from concon_v1.co1_retake_scenarios import plant_site, normalize_site_ability_distances
from concon_v1.co1_retake_common import (
    ACTION_DIM, DEFUSE_ACTION, GAMMA, observation_dim, decision_reward,
)


class TrainingRetakeController(ConconDefenderRetakeController):
    def __init__(self, env, site):
        super().__init__(site, model=env.models[site], ability_distance=env.ability_distances[site])
        self.env = env

    def choose_action(self, char, observation, mask, context):
        index = self.env.indices[char.name]
        site = self.scenario.map_name
        self.env.finish_retake_pending(index, observation, mask, False)
        epsilon = self.env.epsilon[site] if isinstance(self.env.epsilon, dict) else self.env.epsilon
        if self.env.action_rng.random() < epsilon:
            action = self.env.action_rng.choice(np.flatnonzero(mask).tolist())
        else:
            action = super().choose_action(char, observation, mask, context)
        self.env.retake_pending[index] = dict(site=site, obs=observation.astype(np.float16),
                                              action=action, mask=mask.copy(), reward=0., duration=0)
        context["hp_before"] = char.hp
        context["kills_before"] = char.round_kills
        context["enemy_hp_before"] = sum(enemy.hp for enemy in self.env.attackers)
        self.env.retake_decisions[index] = action, context
        return action


class RetakeTrainingAdapter(ConconDefenderController):
    def __init__(self, search, retakes):
        super().__init__(search_controller=search)
        self.retakes = retakes

    def set_game(self, game):
        super().set_game(game)
        for controller in self.retakes.values():
            controller.set_game(game)

    def reset_round(self):
        super().reset_round()
        for controller in self.retakes.values():
            controller.reset_round()

    def decide_move(self, char, state):
        if state.get("is_planted"):
            site = plant_site(state["planted_pos"], self.search_controller.scenario.grid)
            if site not in self.retakes:
                # Unselected live sites use the existing production model only.
                controller = ConconDefenderRetakeController(site)
                controller.set_game(self.game)
                self.retakes[site] = controller
            return self.retakes[site].decide_move(char, state)
        return self.search_controller.decide_move(char, state)


class DefenderRetakeEnv(DefenderSearchEnv):
    def __init__(self, models, search_model, seed=0, opponents=None, ability_distance=6):
        super().__init__(search_model, seed=seed, opponents=opponents)
        if not models or not set(models) <= {"L", "R"}:
            raise ValueError("retake requires at least one L/R site model")
        self.models, self.ability_distances = models, normalize_site_ability_distances(ability_distance)
        self.frc_attack_attempts = 0

    @_run_from_project_root
    def reset(self, start_mode="round", opponent=None):
        result = super().reset(start_mode, opponent)
        if self.opponent == "frc_v1":
            # Fresh one-round games otherwise repeat FRC's first attack plan.
            # Advance on every attempt, including rounds excluded from retakes.
            self.game.current_round = self.frc_attack_attempts % 5 + 1
            self.frc_attack_attempts += 1
        # Replace the recording search controller with frozen production search.
        self.controller = ConconDefenderSearchController(model=self.model)
        self.retakes = {site: TrainingRetakeController(self, site) for site in self.models}
        self.adapter = RetakeTrainingAdapter(self.controller, self.retakes)
        team_ai = self.game.current_defender_team_ai
        team_ai.defender_factory = lambda: self.adapter
        team_ai._defender_controller = None
        self.game.defender_controller = team_ai.get_defender_controller()
        self.case_file = None
        self._reset_retake_recording()
        return result

    def _reset_retake_recording(self):
        self.retake_pending = [None] * 5
        self.retake_decisions, self.transitions = {}, []
        self.site, self.retake_ticks, self.retake_decision_count = None, 0, 0
        self.metrics = dict(fire_decisions=0, moving_fire_decisions=0, waiting_decisions=0,
                            safe_waits=0, defuse_decisions=0, smoke_defuse_decisions=0,
                            smoke_casts=0, flash_casts=0, recon_casts=0, ultimate_casts=0)

    @_run_from_project_root
    def reset_case(self, path, tensor_cache=None):
        """Continue a saved real plant, replacing only the defender policy."""
        from concon_v1.co1_retake_cases import load_case
        from concon_v1.co1_attacker_common import GORIGONS
        from concon_v1.co1_retake_scenarios import get_scenario
        from iq_controller_adapter import IQAwareController
        with contextlib.redirect_stdout(io.StringIO()):
            game, metadata = load_case(path, restore_rng=False, tensor_cache=tensor_cache)
        if metadata["opponent"] not in self.opponents:
            raise ValueError("case opponent is outside the training roster")
        actual_site = plant_site(game.planted_pos, game.grid)
        if actual_site not in self.models:
            raise ValueError("case site is outside the selected training sites")
        if metadata["site"] != actual_site or not np.array_equal(game.grid, get_scenario(actual_site).grid):
            raise ValueError("case site or terrain differs from the training scenario")
        defenders = [char for char in game.chars if char.team == "D"]
        if tuple(char.name for char in defenders) != GORIGONS.players:
            raise ValueError("case defender roster must be Gorigons in preset order")
        team_ai = game.current_defender_team_ai
        if not team_ai.use_iq_perception or not isinstance(game.defender_controller, IQAwareController):
            raise ValueError("case must retain production defender IQ perception")
        self.game, self.opponent = game, metadata["opponent"]
        self.attackers, self.defenders = [char for char in game.chars if char.team == "A"], defenders
        self.indices = {char.name: index for index, char in enumerate(defenders)}
        self.done, self.epsilon, self.end_reason = False, 0., None
        self.elapsed_ticks, self.search_ticks, self.start_mode = 0, 0, "case"
        self.planted = True
        self.plant_attacker_alive = sum(char.is_alive for char in self.attackers)
        self.plant_defender_alive = sum(char.is_alive for char in defenders)
        # Search is not executed here. The current frozen search remains available
        # for the adapter; the attacker's controller and IQ memories are untouched.
        self.controller = ConconDefenderSearchController(model=self.model)
        self.retakes = {site: TrainingRetakeController(self, site) for site in self.models}
        self.adapter = RetakeTrainingAdapter(self.controller, self.retakes)
        team_ai.defender_factory = lambda: self.adapter
        team_ai._defender_controller = None
        game.defender_controller = team_ai.get_defender_controller()
        self._reset_retake_recording()
        self.site, self.case_file = actual_site, str(path)
        return metadata

    def finish_retake_pending(self, index, observation, mask, terminal):
        pending = self.retake_pending[index]
        if pending is not None:
            self.transitions.append((pending["site"], (pending["obs"], pending["action"], pending["reward"],
                observation.astype(np.float16), mask.copy(), pending.get("mask", np.zeros(ACTION_DIM, dtype=bool)),
                float(terminal), pending["duration"])))
            self.retake_pending[index] = None

    @_run_from_project_root
    def step(self, epsilon=0.):
        if self.game is None or self.done:
            raise RuntimeError("reset an active retake episode before stepping")
        self.epsilon = ({site: float(epsilon[site]) for site in ("L", "R")}
                        if isinstance(epsilon, dict) else float(epsilon))
        self.retake_decisions, self.transitions = {}, []
        charges = [{ability: getattr(char, ability + "_charges", 0) for ability in ("smoke", "flash", "recon")}
                   for char in self.defenders]
        points = [getattr(char, "ultimate_points", 0) for char in self.defenders]
        was_planted = bool(self.game.is_planted)
        with contextlib.redirect_stdout(io.StringIO()):
            self.game.step_tick()
        self.elapsed_ticks += 1
        just_planted = self.game.is_planted and not self.planted
        self.planted |= bool(self.game.is_planted)
        if just_planted:
            self.site = plant_site(self.game.planted_pos, self.scenario.grid)
            self.plant_attacker_alive = sum(char.is_alive for char in self.attackers)
            self.plant_defender_alive = sum(char.is_alive for char in self.defenders)
        self.retake_ticks += int(was_planted or bool(self.retake_decisions))
        self.retake_decision_count += len(self.retake_decisions)
        self.done = bool(self.game.round_over or self.game.match_over)
        self.end_reason = ("defused" if self.game.is_defused else "detonated"
                           if self.planted and self.game.detonate_timer <= 0 else "defender_eliminated"
                           if not any(char.is_alive for char in self.defenders) else "attacker_eliminated"
                           if not self.planted and not any(char.is_alive for char in self.attackers)
                           else "timeout") if self.done else None
        rewards = [0.] * 5
        for index, (action, context) in self.retake_decisions.items():
            char = self.defenders[index]
            moved = tuple(char.pos) != context["position"]
            reward = decision_reward(action, context, tuple(char.pos), char.facing)
            reward += .003 * max(0, context["enemy_hp_before"] - sum(enemy.hp for enemy in self.attackers))
            reward += .3 * max(0, char.round_kills - context["kills_before"])
            reward -= .003 * max(0, context["hp_before"] - char.hp)
            self.metrics["fire_decisions"] += int(context["fireable"])
            self.metrics["moving_fire_decisions"] += int(context["fireable"] and moved and not context["neutralized"])
            self.metrics["waiting_decisions"] += int(context["waiting"])
            self.metrics["safe_waits"] += int(context["waiting"] and context["safe"] and not moved)
            self.metrics["defuse_decisions"] += int(action == DEFUSE_ACTION)
            self.metrics["smoke_defuse_decisions"] += int(action == DEFUSE_ACTION and context["smoke_defuse"])
            for ability, count in charges[index].items():
                self.metrics[ability + "_casts"] += max(0, count - getattr(char, ability + "_charges", 0))
            self.metrics["ultimate_casts"] += int(getattr(char, "ultimate_points", 0) < points[index])
            rewards[index] = reward
        for index, char in enumerate(self.defenders):
            pending = self.retake_pending[index]
            if pending is None:
                continue
            reward = rewards[index] - (1.5 if not char.is_alive else 0)
            if self.done:
                reward += 10. if self.game.is_defused else -10.
            pending["reward"] += GAMMA ** pending["duration"] * reward
            pending["duration"] += 1
            if self.done or not char.is_alive:
                scenario = self.retakes[pending["site"]].scenario
                mask = np.zeros(ACTION_DIM, dtype=bool)
                mask[32] = True
                self.finish_retake_pending(index, np.zeros(observation_dim(scenario), dtype=np.float16), mask, True)
        return self.transitions, rewards, self.done

    def result(self):
        if not self.done:
            raise RuntimeError("finish the round before reading its result")
        return dict(opponent=self.opponent, planted=self.planted, site=self.site,
                    excluded_from_retake=not self.planted or self.retake_decision_count == 0,
                    retake_decisions=self.retake_decision_count, end_reason=self.end_reason,
                    defused=bool(self.game.is_defused), ticks=self.elapsed_ticks, retake_ticks=self.retake_ticks,
                    plant_attacker_alive=self.plant_attacker_alive, plant_defender_alive=self.plant_defender_alive,
                    attacker_alive=sum(char.is_alive for char in self.attackers),
                    defender_alive=sum(char.is_alive for char in self.defenders),
                    case_file=self.case_file, **self.metrics)
