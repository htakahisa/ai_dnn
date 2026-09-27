"""Real-game coach curriculum with a strict actor/critic information split."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

from game_core import DEFUSE_REQUIRED_TICKS
from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI

from coach_v1.common.constants import FIXED_ROSTER, MAP_COLUMNS, MAP_ROWS
from coach_v1.common.types import MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.learning_character_base import CharacterPolicy
from coach_v1.learning_character_gongon import GongonPolicy
from coach_v1.models.coach_model import CoachActionMask, legal_action_mask
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import CoachObservation, CoachObservationEncoder
from coach_v1.perception.belief_memory import BeliefMemory
from coach_v1.perception.team_perception import TeamPerceptionBuilder
from coach_v1.training.scenario_generator import ScenarioGenerator
from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.common.watch_points import load_watch_points


CURRICULUM = {"2v1": (2, 1), "2v2": (2, 2), "3v3": (3, 3), "5v5": (5, 5)}
CURRICULUM_ELAPSED_TICKS = 45
_STAY = tuple(CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE,
                               TacticalIntent.HOLD) for _ in range(5))


class _QueuedCoach:
    def __init__(self) -> None:
        self.next_actions: tuple[CoachInstruction, ...] | None = None
        self.last_observation: CoachObservation | None = None

    def act(self, observation: CoachObservation) -> tuple[CoachInstruction, ...]:
        self.last_observation = observation
        actions = self.next_actions or _STAY
        self.next_actions = None
        return actions


@dataclass(frozen=True)
class CoachTrainingState:
    observation: CoachObservation
    mask: CoachActionMask
    # Training-only truth. Never pass this value to coach or character actors.
    critic_enemy_truth: np.ndarray  # float32 [1, 26, 44]


@dataclass(frozen=True)
class CoachTransition:
    next_state: CoachTrainingState | None
    reward: float
    done: bool
    metrics: dict[str, float]


class CoachTrainingEnvironment:
    """Runs frozen character policies and one learned coach in a real round."""

    def __init__(self, side: Side, *, seed: int, stage: str = "2v1",
                 max_ticks: int = 40, collision_penalty: float = 0.01) -> None:
        if (not isinstance(side, Side) or stage not in CURRICULUM or max_ticks <= 0
                or not np.isfinite(collision_penalty) or collision_penalty < 0):
            raise ValueError("invalid coach curriculum configuration")
        self.side, self.seed, self.stage, self.max_ticks = side, seed, stage, max_ticks
        self.collision_penalty = float(collision_penalty)
        self.sensor = TeamPerceptionBuilder()
        self.encoder = CoachObservationEncoder()
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        self.memory = BeliefMemory(config.for_side(side))
        # These checkpoint-backed character policies remain frozen throughout coach training.
        # Loading frozen checkpoints constructs temporary modules. Preserve
        # the trainer's torch RNG so a resumed run samples the same actions.
        with torch.random.fork_rng(devices=[]):
            self.characters = {slot: GongonPolicy() if slot == 1 else CharacterPolicy(slot)
                               for slot in range(5)}
        for policy in self.characters.values():
            policy.model.requires_grad_(False)
        self.queue = _QueuedCoach()
        self.controller = TeamExecutionCoordinator(side, self.queue, self.characters)
        self.game = None
        self.ticks = 0
        self.state: CoachTrainingState | None = None

    def reset(self, *, episode: int = 0) -> CoachTrainingState:
        random.seed(self.seed + episode)
        self.memory.reset()
        self.queue.next_actions = None
        self.controller.reset_round()
        self.ticks = 0
        opponent = _build_team_ai("default")
        own = DualRoleTeamAI("coach_v1_training", lambda: self.controller,
                             lambda: self.controller)
        roster = [entry.character_name for entry in FIXED_ROSTER]
        self.game = VisualFPSBattle(
            NEW_MAZE_STR,
            own if self.side is Side.ATTACKER else opponent,
            own if self.side is Side.DEFENDER else opponent,
            headless=True,
            attacker_roster=roster if self.side is Side.ATTACKER else None,
            defender_roster=roster if self.side is Side.DEFENDER else None,
            disable_side_swap=True,
        )
        while self.game.defender_setup_phase.active:
            self.game._run_defender_setup_tick()
        # Live curriculum starts with a fresh belief on both sides. Setup
        # decisions may have populated the defender coordinator's memory.
        self.controller.reset_round()
        self.controller.action_log.clear()
        # The curriculum changes only training-game truth, before sensing.
        own_code = "A" if self.side is Side.ATTACKER else "D"
        own = [c for c in self.game.chars if c.team == own_code]
        enemies = [c for c in self.game.chars if c.team != own_code]
        allied_count, enemy_count = CURRICULUM[self.stage]
        for character in own[allied_count:] + enemies[enemy_count:]:
            character.is_alive = False
            character.hp = 0
        # These are limited mid-round situations, not round-start teleports.
        self.game.battle_tick = CURRICULUM_ELAPSED_TICKS
        self.game.round_timer = max(1, self.game.round_timer - CURRICULUM_ELAPSED_TICKS)
        snapshot = self.sensor.build(game=self.game, side=self.side)
        situation = _situation(self.side, snapshot)
        # Episode-local scenario RNG makes episode N reproducible whether fit
        # runs continuously or resumes from a checkpoint before that episode.
        scenario = ScenarioGenerator(self.seed + episode).generate(
            actor_side=self.side, situation=situation,
            elapsed_ticks=CURRICULUM_ELAPSED_TICKS,
            enemy_count=enemy_count, currently_visible=snapshot.currently_visible,
            occupied_positions=(ally.position for ally in snapshot.allies if ally.is_alive),
        )
        for character, placement in zip(enemies[:enemy_count], scenario.enemies):
            character.pos = list(placement.position)
        self.state = self._observe()
        return self.state

    def _observe(self) -> CoachTrainingState:
        # move_character applies last tick's forced facing immediately before
        # requesting the first controller action. Mirror that public state so
        # the stored policy input matches the actual actor call.
        own_code = "A" if self.side is Side.ATTACKER else "D"
        first = next((c for c in self.game._move_order() if c.team == own_code), None)
        if first is not None and getattr(first, "forced_facing_next_tick", None):
            first.facing = first.forced_facing_next_tick
        snapshot = self.sensor.build(game=self.game, side=self.side)
        belief = self.memory.update(snapshot)
        observation = self.encoder.encode(snapshot, belief,
                                          situation=_situation(self.side, snapshot))
        truth = np.zeros((1, MAP_ROWS, MAP_COLUMNS), dtype=np.float32)
        for enemy in self.game.chars:
            if enemy.team != own_code and enemy.is_alive:
                truth[0, int(enemy.pos[0]), int(enemy.pos[1])] += 1.0 / 5.0
        truth.setflags(write=False)
        return CoachTrainingState(observation, legal_action_mask(observation), truth)

    def step(self, actions: tuple[CoachInstruction, ...]) -> CoachTransition:
        if self.state is None or self.ticks >= self.max_ticks or self.game.round_over:
            raise RuntimeError("reset before stepping or episode already ended")
        if len(actions) != 5 or any(not isinstance(action, CoachInstruction) for action in actions):
            raise ValueError("one coach instruction is required per slot")
        for slot, action in enumerate(actions):
            if (not self.state.mask.movement[slot, tuple(MovementAction).index(action.movement)]
                    or not self.state.mask.objective[slot, tuple(ObjectiveAction).index(action.objective)]):
                raise ValueError("coach selected a masked action")
        before = self.state
        self.queue.next_actions = actions
        previous_clear = np.count_nonzero(before.observation.grid[13] == 0)
        own_code = "A" if self.side is Side.ATTACKER else "D"
        round_allies = [c for c in self.game.chars if c.team == own_code]
        before_alive = sum(c.is_alive for c in round_allies)
        before_plant = bool(self.game.is_planted)
        before_defuse = bool(self.game.is_defused)
        before_round = self.game.current_round
        before_logs = len(self.controller.action_log)
        alive_slots = [slot for slot, ally in enumerate(self.controller.sensor.build(
            game=self.game, side=self.side).allies) if ally.is_alive]
        move_commands = sum(actions[slot].movement is not MovementAction.STAY
                            for slot in alive_slots)
        move_requests = executed_moves = invalid_moves = 0
        blocked_ally = blocked_enemy = blocked_other = ability_actions = 0
        self.game._build_occupancy_counts()
        try:
            # The pre-step actor observation is captured before any moves.
            # Process this team first so the logged policy input is exactly
            # what the coordinator receives on either side.
            ordered = self.game._move_order()
            for character in ([c for c in ordered if c.team == own_code]
                              + [c for c in ordered if c.team != own_code]):
                if character.is_alive:
                    log_count = len(self.controller.action_log)
                    occupied = {tuple(other.pos): other.team for other in self.game.chars
                                if other is not character and other.is_alive}
                    self.game.move_character(character)
                    if character.team != own_code or len(self.controller.action_log) == log_count:
                        continue
                    log = self.controller.action_log[-1]
                    ability_actions += log.action == "ABILITY"
                    if log.action != "MOVE" or log.requested_position == log.start:
                        continue
                    move_requests += 1
                    if tuple(character.pos) != log.start:
                        executed_moves += 1
                        continue
                    invalid_moves += 1
                    blocker = occupied.get(log.requested_position)
                    if blocker == own_code:
                        blocked_ally += 1
                    elif blocker is not None:
                        blocked_enemy += 1
                    else:
                        blocked_other += 1
        finally:
            self.game._clear_occupancy_counts()
        if self.queue.next_actions is not None:
            raise RuntimeError("coach was not invoked by the game")
        if (not np.array_equal(self.queue.last_observation.grid, before.observation.grid)
                or not np.array_equal(self.queue.last_observation.vector, before.observation.vector)):
            raise RuntimeError("training observation differs from actor observation: "
                               f"grid={np.argwhere(self.queue.last_observation.grid != before.observation.grid)[:5].tolist()} "
                               f"vector={np.argwhere(self.queue.last_observation.vector != before.observation.vector).flatten().tolist()}")
        logs = self.controller.action_log[before_logs:]
        if len(logs) != len(alive_slots):
            raise RuntimeError("game did not execute one character action per living ally")
        planted_after_move = bool(self.game.is_planted)
        # Headless mode may replace the entire round inside process_battle.
        # Preserve the just-finished round's legal visibility before that.
        post_move_visible = self.sensor.build(game=self.game, side=self.side).currently_visible
        self.game.process_battle()
        self.ticks += 1
        round_ended = bool(self.game.round_over or self.game.current_round != before_round)
        done = bool(round_ended or self.ticks >= self.max_ticks)
        # A time-limit truncation still has a valid next observation. Count
        # the final tick's newly cleared cells before ending the episode.
        next_state = None if round_ended else self._observe()
        if next_state is None:
            new_clear = sum(
                visible and before.observation.grid[13, row, column] == 0
                for row, cells in enumerate(post_move_visible)
                for column, visible in enumerate(cells)
            )
        else:
            new_clear = max(0, previous_clear - np.count_nonzero(next_state.observation.grid[13] == 0))
        # Headless process_battle can initialize the next round immediately.
        # The old Character references still hold the completed round result.
        after_alive = sum(c.is_alive for c in round_allies)
        defuse_completed = (before_plant and self.side is Side.DEFENDER
                            and any(c.is_alive and c.defuse_timer >= DEFUSE_REQUIRED_TICKS
                                    for c in round_allies))
        win = 0.0
        if round_ended:
            attacker_won = self.game.attacker_wins > 0
            win = 1.0 if attacker_won == (self.side is Side.ATTACKER) else -1.0
        reward_clear = min(new_clear, 20) * 0.002
        reward_invalid = -invalid_moves * self.collision_penalty
        reward_death = -max(0, before_alive - after_alive) * 0.05
        reward_objective = (
            (0.1 if not before_plant and planted_after_move and self.side is Side.ATTACKER else 0)
            + (0.1 if not before_defuse and defuse_completed else 0)
        )
        reward = win + reward_clear + reward_invalid + reward_death + reward_objective
        self.state = None if done else next_state
        return CoachTransition(self.state, reward, done, {
            "new_clear_cells": float(new_clear), "invalid_moves": float(invalid_moves),
            "round_win": win, "move_commands": float(move_commands),
            "move_requests": float(move_requests),
            "executed_moves": float(executed_moves),
            "blocked_ally": float(blocked_ally),
            "blocked_enemy": float(blocked_enemy),
            "blocked_other": float(blocked_other),
            "ability_actions": float(ability_actions),
            "reward_clear": float(reward_clear),
            "reward_invalid": float(reward_invalid),
            "reward_death": float(reward_death),
            "reward_objective": float(reward_objective),
            "reward_win": win,
        })


def _situation(side: Side, snapshot) -> str:
    if side is Side.DEFENDER:
        return "retake" if snapshot.spike.is_planted else "search"
    if snapshot.spike.is_planted:
        return "guard"
    if snapshot.spike.dropped_position is not None:
        return "retrieve"
    return "carry"
