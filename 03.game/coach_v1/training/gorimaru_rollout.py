"""Training-only Gorimaru examples from legal 5v5 game observations.

The game stages encounters, while facing labels use only the team's reported
enemy sightings. The actor receives the same safe observation as runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai
from team_ai import DualRoleTeamAI

from coach_v1.common.constants import FIXED_ROSTER
from coach_v1.common.types import Facing, Side
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.evaluate_task10_rollout import StayCoach
from coach_v1.learning_character_base import CharacterPolicy
from coach_v1.training.character_trainer import CharacterTrainingExample
from coach_v1.training.facing_labels import acceptable_facings_from_snapshot


_DIRECTIONS = {(-1, 0): Facing.N, (-1, 1): Facing.NE, (0, 1): Facing.E,
               (1, 1): Facing.SE, (1, 0): Facing.S, (1, -1): Facing.SW,
               (0, -1): Facing.W, (-1, -1): Facing.NW}


@dataclass(frozen=True)
class GorimaruRolloutExample:
    example: CharacterTrainingExample
    side: Side
    seed: int
    has_sighting: bool
    encounter: str = "near"


class _Recorder:
    def __init__(self, actor: CharacterPolicy) -> None:
        self.actor = actor
        self.last_observation = None
        self.last_action = None

    def act(self, observation):
        self.last_observation = observation
        self.last_action = self.actor.act(observation)
        return self.last_action


def _direction(origin: tuple[int, int], target: tuple[int, int], fallback: Facing) -> Facing:
    dr = (target[0] > origin[0]) - (target[0] < origin[0])
    dc = (target[1] > origin[1]) - (target[1] < origin[1])
    return _DIRECTIONS.get((dr, dc), fallback)


def label_facing_from_snapshot(snapshot) -> Facing:
    """Use only legal shared reports, never the training game's enemy truth."""
    own_position = snapshot.allies[0].position
    if not snapshot.sightings:
        return snapshot.allies[0].facing
    selected = min(
        snapshot.sightings,
        key=lambda item: (abs(item.reported_position[0] - own_position[0])
                          + abs(item.reported_position[1] - own_position[1]),
                          item.enemy_id),
    )
    return _direction(own_position, selected.reported_position,
                      snapshot.allies[0].facing)


def _stage_encounter(game, own_team: str, encounter: str) -> None:
    """Place both teams on known legal corridor cells for evaluation/training."""
    if encounter not in {"west", "east", "crossfire"}:
        raise ValueError(f"unsupported encounter: {encounter}")
    allies = [character for character in game.chars if character.team == own_team]
    opponents = [character for character in game.chars if character.team != own_team]
    own_columns = (17, 18, 19, 20, 21)
    enemy_columns = {
        "west": (10, 11, 12, 13, 14),
        "east": (25, 26, 27, 28, 29),
        "crossfire": (11, 13, 25, 27, 29),
    }[encounter]
    for index, (character, column) in enumerate(zip(allies, own_columns)):
        assert game.grid[22, column] != 1
        character.pos = [22, column]
        character.facing = "W" if encounter == "west" or (encounter == "crossfire" and index % 2 == 0) else "E"
    for character, column in zip(opponents, enemy_columns):
        assert game.grid[22, column] != 1
        character.pos = [22, column]


def collect_gorimaru_rollout(*, side: Side, seed: int, ticks: int = 20,
                             near: bool = True,
                             encounter: str | None = None,
                             mask_unsighted_facing: bool = False,
                             actor_checkpoint: Path | None = None) -> tuple[GorimaruRolloutExample, ...]:
    """Collect one round; labels never read actual enemy positions."""
    random.seed(seed)
    recorder = _Recorder(CharacterPolicy(0, actor_checkpoint))
    actors = {slot: (recorder if slot == 0 else CharacterPolicy(slot))
              for slot in range(5)}
    coordinator = TeamExecutionCoordinator(side, StayCoach(), actors)
    team = DualRoleTeamAI("coach_v1_gorimaru_data", lambda: coordinator, lambda: coordinator)
    opponent = _build_team_ai("default")
    roster = [item.character_name for item in FIXED_ROSTER]
    game = VisualFPSBattle(
        NEW_MAZE_STR,
        team if side is Side.ATTACKER else opponent,
        team if side is Side.DEFENDER else opponent,
        headless=True,
        attacker_roster=roster if side is Side.ATTACKER else None,
        defender_roster=roster if side is Side.DEFENDER else None,
        disable_side_swap=True,
    )
    own_team = "A" if side is Side.ATTACKER else "D"
    initial_round = game.current_round
    while game.defender_setup_phase.active:
        game._run_defender_setup_tick()
    selected_encounter = encounter or ("near" if near else "natural")
    if selected_encounter in {"west", "east", "crossfire"}:
        _stage_encounter(game, own_team, selected_encounter)
    elif selected_encounter == "near":
        enemy_row = 21 if side is Side.ATTACKER else 2
        columns = [column for column in range(15, 24) if game.grid[enemy_row, column] != 1]
        selected = random.sample(columns, 5)
        opponents = [character for character in game.chars if character.team != own_team]
        for character, column in zip(opponents, selected):
            character.pos = [enemy_row, column]
    elif selected_encounter != "natural":
        raise ValueError(f"unsupported encounter: {selected_encounter}")
    records = []
    for _ in range(ticks):
        if game.current_round != initial_round or game.match_over:
            break
        game._build_occupancy_counts()
        try:
            for character in game._move_order():
                if not character.is_alive:
                    continue
                before = len(coordinator.action_log)
                game.move_character(character)
                if (character.team != own_team or len(coordinator.action_log) == before
                        or coordinator.action_log[-1].slot != 0):
                    continue
                snapshot = coordinator._snapshot
                facing = label_facing_from_snapshot(snapshot)
                # This opt-in keeps ability supervision but gives facing zero
                # loss when no legal report exists. The default reproduces
                # earlier Task 09-A experiments.
                accepted = (tuple(Facing) if mask_unsighted_facing and not snapshot.sightings
                            else acceptable_facings_from_snapshot(snapshot))
                if facing not in accepted:
                    facing = accepted[0]
                action = recorder.last_action
                records.append(GorimaruRolloutExample(
                    CharacterTrainingExample(recorder.last_observation, facing,
                                             action.use_ability, action.target,
                                             acceptable_facings=accepted),
                    side, seed, bool(snapshot.sightings), selected_encounter,
                ))
        finally:
            game._clear_occupancy_counts()
        game.process_battle()
    return tuple(records)
