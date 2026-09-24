"""Training-only staged 1v1/2v1 character examples on the fixed map.

Enemy truth is used to stage a game. Labels are then derived only from the
team's legal report, watch-point metadata, and the character action mask.
This is a supervised curriculum, not a measurement of live-game ability value.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import random

import numpy as np

from abilities_los import AbilityLosMixin
from map_data import NEW_MAZE_STR

from coach_v1.common.constants import FIXED_ROSTER, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.common.watch_points import load_watch_points
from coach_v1.perception import BeliefMemory, TeamPerceptionBuilder
from coach_v1.training.character_environment import CharacterEnvironment, CurriculumStage
from coach_v1.training.character_trainer import CharacterTrainingExample
from coach_v1.training.scenario_generator import ScenarioGenerator
from coach_v1.observation.character_encoder import CoachInstruction


_ROWS = tuple(NEW_MAZE_STR.strip().splitlines())
_GRID = np.asarray([[int(cell) for cell in row] for row in _ROWS], dtype=np.int8)
_FACINGS = {( -1, 0): Facing.N, (-1, 1): Facing.NE, (0, 1): Facing.E,
            (1, 1): Facing.SE, (1, 0): Facing.S, (1, -1): Facing.SW,
            (0, -1): Facing.W, (-1, -1): Facing.NW}


@dataclass(frozen=True)
class CurriculumExample:
    example: CharacterTrainingExample
    category: str
    side: Side
    situation: str
    sightings: int


class _StagedGame(AbilityLosMixin):
    def __init__(self, characters: list, tick: int):
        self.grid = _GRID
        self.chars = characters
        self.smokes = []
        self.current_round = 1
        self.battle_tick = tick
        self.defender_setup_phase = SimpleNamespace(active=False, ticks_remaining=0)
        self.spike_pos = None
        self.is_planted = False
        self.planted_pos = None


def _character(name: str, team: str, position: tuple[int, int], *,
               facing: Facing = Facing.N, alive: bool = True, blind: bool = False):
    return SimpleNamespace(
        name=name, team=team, pos=list(position), facing=facing.value,
        is_alive=alive, hp=100 if alive else 0, effective_iq=200,
        blind_remaining=int(blind), reveal_remaining=0, has_spike=False,
        smoke_charges=1, recon_charges=1, flash_charges=1,
    )


def _direction(origin: tuple[int, int], target: tuple[int, int], fallback: Facing) -> Facing:
    dr = (target[0] > origin[0]) - (target[0] < origin[0])
    dc = (target[1] > origin[1]) - (target[1] < origin[1])
    return _FACINGS.get((dr, dc), fallback)


def _nearby(target: tuple[int, int], rng: random.Random, excluded: set[tuple[int, int]]) -> tuple[int, int]:
    candidates = [(r, c) for r in range(max(0, target[0] - 4), min(26, target[0] + 5))
                  for c in range(max(0, target[1] - 4), min(44, target[1] + 5))
                  if _GRID[r, c] != 1 and (r, c) not in excluded
                  and 2 <= abs(r - target[0]) + abs(c - target[1]) <= 5]
    if not candidates:
        raise ValueError("no walkable curriculum staging cell")
    return rng.choice(candidates)


def build_character_curriculum(slot: int, *, seed: int, count: int) -> tuple[CurriculumExample, ...]:
    """Generate independent staged rounds using the 70/20/10 scenario sampler."""
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < 5:
        raise ValueError("invalid fixed-roster slot")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    generator = ScenarioGenerator(seed)
    points = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
    builder = TeamPerceptionBuilder()
    environment = CharacterEnvironment()
    result = []
    for index in range(count):
        side = Side.ATTACKER if index % 2 == 0 else Side.DEFENDER
        situation = ("carry", "retrieve", "guard")[index // 2 % 3] if side is Side.ATTACKER else ("search", "retake")[index // 2 % 2]
        own_team = "A" if side is Side.ATTACKER else "D"
        enemy_team = "D" if side is Side.ATTACKER else "A"
        # A blinded setup team supplies the legal empty visibility mask.
        setup = [_character(roster.character_name, own_team, (23, 18 + i),
                            alive=i == slot, blind=True)
                 for i, roster in enumerate(FIXED_ROSTER)]
        setup_snapshot = builder.build(game=_StagedGame(setup, index), side=side)
        scenario = generator.generate(
            actor_side=side, situation=situation, elapsed_ticks=100,
            enemy_count=1, currently_visible=setup_snapshot.currently_visible,
            occupied_positions=(ally.position for ally in setup_snapshot.allies),
        )
        placement = scenario.enemies[0]
        actor_position = _nearby(placement.position, rng, {placement.position})
        toward = _direction(actor_position, placement.position, Facing.N)
        facing = toward if index % 3 else tuple(Facing)[(tuple(Facing).index(toward) + 4) % 8]
        allies = [_character(roster.character_name, own_team, (23, 18 + i), alive=False)
                  for i, roster in enumerate(FIXED_ROSTER)]
        allies[slot] = _character(FIXED_ROSTER[slot].character_name, own_team,
                                  actor_position, facing=facing)
        stage = CurriculumStage.ONE_V_ONE
        if index % 4 == 0:
            support_slot = (slot + 1) % 5
            support_position = _nearby(placement.position, rng, {placement.position, actor_position})
            allies[support_slot] = _character(FIXED_ROSTER[support_slot].character_name,
                                              own_team, support_position,
                                              facing=_direction(support_position, placement.position, Facing.N))
            stage = CurriculumStage.TWO_V_ONE
        game = _StagedGame(allies + [_character("training_enemy", enemy_team, placement.position)], index)
        snapshot = builder.build(game=game, side=side)
        belief = BeliefMemory(points.for_side(side)).update(snapshot)
        request_utility = index // 2 % 2 == 1
        instruction = CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE,
                                       TacticalIntent.UTILITY_REQUEST if request_utility else TacticalIntent.CLEAR_AREA)
        step = environment.prepare(snapshot, belief, situation=situation, slot=slot,
                                   instruction=instruction, curriculum=stage)
        sighting = snapshot.sighting_for("training_enemy")
        relevant_points = [point for point in points.for_side(side) if point.supports_situation(situation)]
        nearest = min(relevant_points,
                      key=lambda point: (abs(point.position[0] - actor_position[0])
                                         + abs(point.position[1] - actor_position[1]), point.point_id))
        label_facing = (_direction(actor_position, sighting.reported_position, nearest.facing)
                        if sighting is not None else nearest.facing)
        use = bool(sighting is not None and request_utility and step.observation.mask.ability_use[1])
        target = None
        if use:
            legal = np.argwhere(step.observation.mask.ability_target)
            report = sighting.reported_position
            selected = min(legal, key=lambda cell: (abs(int(cell[0]) - report[0])
                                                    + abs(int(cell[1]) - report[1]),
                                                    int(cell[0]), int(cell[1])))
            target = int(selected[0]), int(selected[1])
        result.append(CurriculumExample(
            CharacterTrainingExample(step.observation, label_facing, use, target),
            placement.category, side, situation, int(sighting is not None),
        ))
    return tuple(result)
