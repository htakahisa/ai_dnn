"""One team decision per tick, adapted to the game's per-character controller API.

Only the perception builder and this controller hold a live game reference.
Coach and character actors receive copied, actor-safe observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from coach_v1.common.constants import FIXED_ROSTER, ROSTER_SIZE, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import Side
from coach_v1.common.watch_points import load_watch_points
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import CoachObservation, CoachObservationEncoder
from coach_v1.perception.belief_memory import BeliefMemory
from coach_v1.perception.team_perception import TeamPerceptionBuilder, _tick_key
from coach_v1.training.character_environment import CharacterAction, CharacterEnvironment
from map_data import NEW_MAZE_STR


class CoachActor(Protocol):
    def act(self, observation: CoachObservation) -> Sequence[CoachInstruction]: ...


class CharacterActor(Protocol):
    def act(self, observation: object) -> CharacterAction: ...


@dataclass(frozen=True)
class ActionLog:
    round_number: int
    phase: str
    tick: int
    side: Side
    slot: int
    action: str
    start: tuple[int, int]
    requested_position: tuple[int, int]
    facing: str | None
    ability: str | None
    ability_target: tuple[int, int] | None


class TeamExecutionCoordinator:
    """Controller for one side of the fixed five-character roster.

    The caller supplies a coach actor and five character actors. The coach
    interface remains injectable until Task 11 supplies trained checkpoints.
    """

    handles_team_perception = True

    def __init__(self, side: Side, coach: CoachActor,
                 characters: Mapping[int, CharacterActor]) -> None:
        if not isinstance(side, Side):
            raise ValueError("side must be attacker or defender")
        if set(characters) != set(range(ROSTER_SIZE)):
            raise ValueError("one character actor is required for every roster slot")
        self.side = side
        self.coach = coach
        self.characters = dict(characters)
        self.game = None
        self.sensor = TeamPerceptionBuilder()
        self.encoder = CoachObservationEncoder()
        self.character_environment = CharacterEnvironment()
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        self.memory = BeliefMemory(config.for_side(side))
        self._cache_key = None
        self._snapshot = None
        self._belief = None
        self._instructions = None
        self.action_log: list[ActionLog] = []

    def set_game(self, game) -> None:
        expected = np.array([[int(cell) for cell in line]
                             for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        if not np.array_equal(np.asarray(getattr(game, "grid", None)), expected):
            raise ValueError("coach_v1 requires the current fixed map")
        if game is not self.game:
            self.reset_round()
        self.game = game

    def reset_round(self) -> None:
        self._cache_key = None
        self._snapshot = None
        self._belief = None
        self._instructions = None
        self.memory.reset()

    def decide_move(self, char, game_state):
        if self.game is None:
            raise RuntimeError("set_game must be called before decide_move")
        expected_team = "A" if self.side is Side.ATTACKER else "D"
        if getattr(char, "team", None) != expected_team:
            raise ValueError("character belongs to the other side")
        try:
            slot = next(i for i, item in enumerate(FIXED_ROSTER)
                        if item.character_name == char.name)
        except StopIteration as exc:
            raise ValueError("character is outside the fixed roster") from exc
        if not any(member is char for member in self.game.chars):
            raise ValueError("character is not in the bound game")

        key = (self.side, _tick_key(self.game))
        if key != self._cache_key:
            if self._cache_key is not None and key[1].round_number != self._cache_key[1].round_number:
                self.memory.reset()
            snapshot = self.sensor.build(game=self.game, side=self.side)
            belief = self.memory.update(snapshot)
            situation = _situation(snapshot)
            observation = self.encoder.encode(snapshot, belief, situation=situation)
            instructions = tuple(self.coach.act(observation))
            if len(instructions) != ROSTER_SIZE or any(
                not isinstance(item, CoachInstruction) for item in instructions
            ):
                raise ValueError("coach must return five CoachInstruction values")
            self._snapshot = snapshot
            self._belief = belief
            self._instructions = instructions
            self._cache_key = key

        snapshot = self._snapshot
        ally = snapshot.allies[slot]
        if not ally.is_alive:
            return ally.position, {"facing": ally.facing.value}

        step = self.character_environment.prepare(
            snapshot, self._belief, situation=_situation(snapshot),
            slot=slot, instruction=self._instructions[slot],
        )
        action = self.characters[slot].act(step.observation)
        result = self.character_environment.resolve(step, action)
        requested = tuple(result[0])
        payload = result[1]
        action_name = payload if isinstance(payload, str) else (
            "ABILITY" if "ability" in payload else "MOVE"
        )
        # The current game applies a facing payload on MOVE, but its ABILITY
        # and objective branches return before that code. Apply the actor's
        # facing here, after validation and before the game consumes the tuple.
        if action_name != "MOVE" and not getattr(char, "facing_forced_this_tick", False):
            char.facing = action.facing.value
        self.action_log.append(ActionLog(
            round_number=snapshot.tick.round_number,
            phase=snapshot.tick.phase, tick=snapshot.tick.tick,
            side=self.side, slot=slot, action=action_name,
            start=ally.position, requested_position=requested,
            facing=action.facing.value,
            ability=payload.get("ability") if isinstance(payload, dict) else None,
            ability_target=payload.get("target") if isinstance(payload, dict) else None,
        ))
        return result


def _situation(snapshot) -> str:
    if snapshot.side is Side.DEFENDER:
        return "retake" if snapshot.spike.is_planted else "search"
    if snapshot.spike.is_planted:
        return "guard"
    if snapshot.spike.dropped_position is not None:
        return "retrieve"
    return "carry"
