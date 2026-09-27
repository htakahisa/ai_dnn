"""Training-only Gonta examples from legal five-player team observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

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


@dataclass(frozen=True)
class GontaRolloutExample:
    example: CharacterTrainingExample
    side: Side
    seed: int
    has_sighting: bool
    encounter: str


class _Recorder:
    def __init__(self, actor: CharacterPolicy) -> None:
        self.actor = actor
        self.last_observation = None

    def act(self, observation):
        self.last_observation = observation
        return self.actor.act(observation)


def label_gonta(snapshot, observation) -> CharacterTrainingExample:
    """Use current shared reports only; private enemy positions are unavailable here."""
    accepted = (acceptable_facings_from_snapshot(snapshot, slot=2)
                if snapshot.sightings else tuple(Facing))
    use = bool(snapshot.sightings and observation.mask.ability_use[1])
    target = None
    if use:
        origin = snapshot.allies[2].position
        report = min(snapshot.sightings, key=lambda item: (
            abs(item.reported_position[0] - origin[0])
            + abs(item.reported_position[1] - origin[1]), item.enemy_id,
        )).reported_position
        legal = np.argwhere(observation.mask.ability_target)
        chosen = min(legal, key=lambda cell: (
            abs(int(cell[0]) - report[0]) + abs(int(cell[1]) - report[1]),
            int(cell[0]), int(cell[1]),
        ))
        target = int(chosen[0]), int(chosen[1])
    return CharacterTrainingExample(observation, accepted[0], use, target,
                                    acceptable_facings=accepted)


def collect_gonta_rollout(*, side: Side, seed: int, encounter: str,
                          ticks: int = 20, actor_checkpoint: Path | None = None
                          ) -> tuple[GontaRolloutExample, ...]:
    if encounter not in {"near", "near_fixed", "west", "east", "crossfire", "natural"}:
        raise ValueError(f"unsupported encounter: {encounter}")
    random.seed(seed)
    recorder = _Recorder(CharacterPolicy(2, actor_checkpoint))
    actors = {slot: (recorder if slot == 2 else CharacterPolicy(slot)) for slot in range(5)}
    coordinator = TeamExecutionCoordinator(side, StayCoach(), actors)
    team = DualRoleTeamAI("coach_v1_gonta_data", lambda: coordinator, lambda: coordinator)
    opponent = _build_team_ai("default")
    roster = [item.character_name for item in FIXED_ROSTER]
    game = VisualFPSBattle(
        NEW_MAZE_STR, team if side is Side.ATTACKER else opponent,
        team if side is Side.DEFENDER else opponent, headless=True,
        attacker_roster=roster if side is Side.ATTACKER else None,
        defender_roster=roster if side is Side.DEFENDER else None,
        disable_side_swap=True,
    )
    own_team = "A" if side is Side.ATTACKER else "D"
    initial_round = game.current_round
    while game.defender_setup_phase.active:
        game._run_defender_setup_tick()
    if encounter in {"west", "east", "crossfire"}:
        allies = [char for char in game.chars if char.team == own_team]
        enemies = [char for char in game.chars if char.team != own_team]
        columns = {
            "west": (10, 11, 12, 13, 14),
            "east": (25, 26, 27, 28, 29),
            "crossfire": (11, 13, 25, 27, 29),
        }[encounter]
        for index, (char, column) in enumerate(zip(allies, (17, 18, 19, 20, 21))):
            assert game.grid[22, column] != 1
            char.pos = [22, column]
            char.facing = "W" if encounter == "west" or (encounter == "crossfire" and index % 2 == 0) else "E"
        for char, column in zip(enemies, columns):
            assert game.grid[22, column] != 1
            char.pos = [22, column]
    elif encounter in {"near", "near_fixed"}:
        row = 21 if side is Side.ATTACKER else 2
        columns = (list(range(18, 23)) if encounter == "near_fixed" else
                   [column for column in range(15, 24) if game.grid[row, column] != 1])
        if encounter == "near":
            columns = random.sample(columns, 5)
        for char, column in zip((char for char in game.chars if char.team != own_team), columns):
            char.pos = [row, column]
    records = []
    for _ in range(ticks):
        if game.current_round != initial_round or game.match_over:
            break
        game._build_occupancy_counts()
        try:
            for char in game._move_order():
                if not char.is_alive:
                    continue
                before = len(coordinator.action_log)
                game.move_character(char)
                if (char.team != own_team or len(coordinator.action_log) == before
                        or coordinator.action_log[-1].slot != 2):
                    continue
                snapshot = coordinator._snapshot
                records.append(GontaRolloutExample(
                    label_gonta(snapshot, recorder.last_observation),
                    side, seed, bool(snapshot.sightings), encounter,
                ))
        finally:
            game._clear_occupancy_counts()
        game.process_battle()
    return tuple(records)
