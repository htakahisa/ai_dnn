"""Real-game coach curriculum with a strict actor/critic information split."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch

from game_core import DEFUSE_REQUIRED_TICKS
from map_data import NEW_MAZE_STR
from map_data_defender_setup import DEFENDER_SETUP_TICKS
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI

from coach_v1.common.constants import FACING_DELTAS, FIXED_ROSTER, MAP_COLUMNS, MAP_ROWS
from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.common.versions import COACH_OBSERVATION_VERSION
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
ATTACKER_STAGES = (
    "rally", "entry", "utility_entry", "multi_peek", "escort",
    "retrieve", "plant", "post_plant", "full_round",
)
ATTACKER_STAGE_ELAPSED = {
    "rally": 20, "entry": 65, "utility_entry": 65,
    "multi_peek": 70, "escort": 45, "retrieve": 65,
    "plant": 85, "post_plant": 85, "full_round": 0,
}
CURRICULUM_ELAPSED_TICKS = 45
_STAY = tuple(CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE,
                               TacticalIntent.HOLD) for _ in range(5))


class _QueuedCoach:
    def __init__(self, encoder: CoachObservationEncoder) -> None:
        self.encoder = encoder
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
                 max_ticks: int = 40, collision_penalty: float = 0.01,
                 observation_version: str = COACH_OBSERVATION_VERSION) -> None:
        if (not isinstance(side, Side) or stage not in CURRICULUM
                and not (side is Side.ATTACKER and stage in ATTACKER_STAGES)
                or max_ticks <= 0
                or not np.isfinite(collision_penalty) or collision_penalty < 0):
            raise ValueError("invalid coach curriculum configuration")
        self.side, self.seed, self.stage, self.max_ticks = side, seed, stage, max_ticks
        self.collision_penalty = float(collision_penalty)
        self.sensor = TeamPerceptionBuilder()
        self.encoder = CoachObservationEncoder(version=observation_version)
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
        self.queue = _QueuedCoach(self.encoder)
        self.controller = TeamExecutionCoordinator(side, self.queue, self.characters)
        self.game = None
        self.ticks = 0
        self.state: CoachTrainingState | None = None
        self.scenario = None  # training-only sampled truth and provenance
        self._pending_moves: tuple = ()
        self._ability_window = 0

    def reset(self, *, episode: int = 0) -> CoachTrainingState:
        random.seed(self.seed + episode)
        self.memory.reset()
        self.queue.next_actions = None
        self.controller.reset_round()
        self.ticks = 0
        self._pending_moves = ()
        self._ability_window = 0
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
        allied_count, enemy_count = CURRICULUM.get(self.stage, (5, 5))
        for character in own[allied_count:] + enemies[enemy_count:]:
            character.is_alive = False
            character.hp = 0
        elapsed = (ATTACKER_STAGE_ELAPSED[self.stage] if self.stage in ATTACKER_STAGES
                   else CURRICULUM_ELAPSED_TICKS)
        if self.stage in ATTACKER_STAGES:
            self._set_attacker_situation(own, episode)
        # Full-round training starts at the real opening; other stages are
        # training-only plausible mid-round states.
        self.game.battle_tick = elapsed
        self.game.round_timer = max(1, self.game.round_timer - elapsed)
        snapshot = self.sensor.build(game=self.game, side=self.side)
        situation = _situation(self.side, snapshot)
        # Episode-local scenario RNG makes episode N reproducible whether fit
        # runs continuously or resumes from a checkpoint before that episode.
        scenario = ScenarioGenerator(self.seed + episode).generate(
            actor_side=self.side, situation=situation,
            # Defender setup occurs before the first LIVE tick. Its legal
            # travel time counts for initial defender placement only.
            elapsed_ticks=DEFENDER_SETUP_TICKS if self.stage == "full_round" else elapsed,
            enemy_count=enemy_count, currently_visible=snapshot.currently_visible,
            occupied_positions=(ally.position for ally in snapshot.allies if ally.is_alive),
        )
        for character, placement in zip(enemies[:enemy_count], scenario.enemies):
            character.pos = list(placement.position)
        self.scenario = scenario
        self._prepare_tick()
        self.state = self._observe()
        return self.state

    def _set_attacker_situation(self, allies: list, episode: int) -> None:
        """Seed public objective states and allied positions for training only."""
        if self.stage == "full_round":
            return
        rows = NEW_MAZE_STR.strip().splitlines()
        sites = [(r, c) for r, row in enumerate(rows)
                 for c, tile in enumerate(row) if tile == "2"]
        site = sites[(self.seed + episode) % len(sites)]
        self.game.target_plant_pos = site
        # Fixed map, training-only breadth-first distances; no route or
        # tactical branch is available to the inference policy.
        from coach_v1.training.scenario_generator import _distances
        distances = _distances(rows, (site,))
        bands = {
            "rally": (20, 30), "entry": (7, 13),
            "utility_entry": (7, 13), "multi_peek": (4, 10),
            "escort": (12, 20), "retrieve": (7, 14),
            "plant": (1, 5), "post_plant": (1, 7),
        }
        low, high = bands[self.stage]
        candidates = sorted((position for position, distance in distances.items()
                             if low <= distance <= high),
                            key=lambda cell: (distances[cell], cell[0], cell[1]))
        if len(candidates) < len(allies):
            raise RuntimeError("attacker stage has too few reachable allied cells")
        # Spread five slots across the band instead of stacking them on one tile.
        step = max(1, len(candidates) // len(allies))
        chosen = [candidates[i * step] for i in range(len(allies))]
        if self.stage == "plant":
            carrier = next(ally for ally in allies if ally.has_spike)
            chosen[allies.index(carrier)] = site
        for ally, position in zip(allies, chosen):
            ally.pos = list(position)
        if self.stage == "utility_entry":
            # A legal smoke cast just before this mid-round snapshot makes
            # the ability-assisted entry stage distinct from plain entry.
            if not self.game.execute_ai_ability(
                    allies[0], {"ability": "SMOKE", "target": site}):
                raise RuntimeError("training smoke setup failed")
            self.game.smoke_thrown_this_tick = False
            self._ability_window = 15
        if self.stage == "retrieve":
            for ally in allies:
                ally.has_spike = False
            self.game.spike_pos = site
        elif self.stage == "post_plant":
            for ally in allies:
                ally.has_spike = False
            self.game.is_planted = True
            self.game.planted_pos = site
            self.game.detonate_timer = 35

    def _prepare_tick(self) -> None:
        """Advance opponents preceding our first actor in the real move order."""
        ordered = tuple(self.game._move_order())
        own_code = "A" if self.side is Side.ATTACKER else "D"
        first_own = next((index for index, character in enumerate(ordered)
                          if character.team == own_code), len(ordered))
        self.game._build_occupancy_counts()
        try:
            for character in ordered[:first_own]:
                if character.is_alive:
                    self.game.move_character(character)
        finally:
            self.game._clear_occupancy_counts()
        self._pending_moves = ordered[first_own:]

    def _observe(self) -> CoachTrainingState:
        # move_character applies last tick's forced facing immediately before
        # requesting the first controller action. Mirror that public state so
        # the stored policy input matches the actual actor call.
        own_code = "A" if self.side is Side.ATTACKER else "D"
        first = next((c for c in self._pending_moves if c.team == own_code), None)
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
        before_drop = self.game.spike_pos is not None
        before_defuse = bool(self.game.is_defused)
        before_round = self.game.current_round
        before_logs = len(self.controller.action_log)
        before_positions = {id(ally): tuple(ally.pos) for ally in round_allies if ally.is_alive}
        alive_slots = [slot for slot, ally in enumerate(self.controller.sensor.build(
            game=self.game, side=self.side).allies) if ally.is_alive]
        move_commands = 0
        move_requests = executed_moves = invalid_moves = 0
        blocked_ally = blocked_enemy = blocked_other = ability_actions = 0
        self.game._build_occupancy_counts()
        try:
            # Opponents before our first actor already moved in _prepare_tick.
            # Continue the original game order without moving them twice.
            for character in self._pending_moves:
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
            self._pending_moves = ()
        if self.queue.next_actions is not None:
            raise RuntimeError("coach was not invoked by the game")
        if (not np.array_equal(self.queue.last_observation.grid, before.observation.grid)
                or not np.array_equal(self.queue.last_observation.vector, before.observation.vector)):
            raise RuntimeError("training observation differs from actor observation: "
                               f"grid={np.argwhere(self.queue.last_observation.grid != before.observation.grid)[:5].tolist()} "
                               f"vector={np.argwhere(self.queue.last_observation.vector != before.observation.vector).flatten().tolist()}")
        logs = self.controller.action_log[before_logs:]
        # An earlier opponent in the real move order may kill an ally before
        # that ally's turn. Only slots that actually acted belong in counts.
        if len(logs) > len(alive_slots) or len({log.slot for log in logs}) != len(logs):
            raise RuntimeError("game produced duplicate or unexpected allied actions")
        move_commands = sum(actions[log.slot].movement is not MovementAction.STAY
                            for log in logs)
        planted_after_move = bool(self.game.is_planted)
        live_allies = [ally for ally in round_allies if ally.is_alive]
        positions = [tuple(ally.pos) for ally in live_allies]
        trade_distances = [min(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                               for j, b in enumerate(positions) if i != j)
                           for i, a in enumerate(positions)] if len(positions) > 1 else []
        entries = [ally for ally in live_allies
                   if self.game.grid[tuple(ally.pos)] == 2
                   and self.game.grid[before_positions[id(ally)]] != 2]
        solo_entries = sum(not any(other is not ally and
                                  max(abs(other.pos[0] - ally.pos[0]),
                                      abs(other.pos[1] - ally.pos[1])) <= 3
                                  for other in live_allies) for ally in entries)
        ability_after_entry = len(entries) if self._ability_window > 0 or ability_actions else 0
        self._ability_window = 15 if ability_actions else max(0, self._ability_window - 1)
        # Training metric only: use game truth to check two legal sight lines
        # to the same enemy from directions separated by at least 30 degrees.
        multi_angle = _has_multiple_enemy_angles(self.game, live_allies)
        # Headless mode may replace the entire round inside process_battle.
        # Preserve the just-finished round's legal visibility before that.
        post_move_visible = self.sensor.build(game=self.game, side=self.side).currently_visible
        self.game.process_battle()
        self.ticks += 1
        round_ended = bool(self.game.round_over or self.game.current_round != before_round)
        done = bool(round_ended or self.ticks >= self.max_ticks)
        # A time-limit truncation still has a valid next observation. Count
        # the final tick's newly cleared cells before ending the episode.
        if not round_ended and self.ticks < self.max_ticks:
            self._prepare_tick()
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
            "plant": float(not before_plant and planted_after_move),
            "spike_recovered": float(before_drop and self.game.spike_pos is None
                                      and any(ally.has_spike for ally in round_allies)),
            "attacker_win": float(round_ended and self.game.attacker_wins > 0),
            "ally_deaths": float(max(0, before_alive - after_alive)),
            "site_entries": float(len(entries)),
            "solo_entries": float(solo_entries),
            "trade_distance_sum": float(sum(trade_distances)),
            "trade_distance_samples": float(len(trade_distances)),
            "multi_angle_ticks": float(multi_angle),
            "ability_after_entry": float(ability_after_entry),
            "ability_preseeded": float(self.stage == "utility_entry" and self.ticks == 1),
        })


def _situation(side: Side, snapshot) -> str:
    if side is Side.DEFENDER:
        return "retake" if snapshot.spike.is_planted else "search"
    if snapshot.spike.is_planted:
        return "guard"
    if snapshot.spike.dropped_position is not None:
        return "retrieve"
    return "carry"


def _has_multiple_enemy_angles(game, allies: list) -> bool:
    if not allies:
        return False
    enemy_team = "D" if allies[0].team == "A" else "A"
    for enemy in (character for character in game.chars
                  if character.team == enemy_team and character.is_alive):
        target = tuple(enemy.pos)
        directions = []
        for ally in allies:
            if getattr(ally, "blind_remaining", 0.0) > 0:
                continue
            origin = tuple(ally.pos)
            row, col = target[0] - origin[0], target[1] - origin[1]
            facing_row, facing_col = FACING_DELTAS[Facing(ally.facing)]
            if (row * facing_row + col * facing_col < 0
                    or not game.check_cell_line_of_sight(origin, target, block_smoke=True)):
                continue
            directions.append((row, col))
        for index, (ar, ac) in enumerate(directions):
            for br, bc in directions[index + 1:]:
                if (ar == 0 and ac == 0) or (br == 0 and bc == 0):
                    continue
                cross = ar * bc - ac * br
                if 4 * cross * cross >= (ar * ar + ac * ac) * (br * br + bc * bc):
                    return True
    return False
