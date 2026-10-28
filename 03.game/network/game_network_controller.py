"""Server-side controller that consumes validated remote move and hold orders."""

from __future__ import annotations

import sys
from threading import Lock
from typing import Any
from pathlib import Path

# The normal execution directory for this shared game package is ``network/``.
GAME_ROOT = Path(__file__).resolve().parent.parent
if str(GAME_ROOT) not in sys.path:
    sys.path.insert(0, str(GAME_ROOT))

from controllers import UserInputController


def network_character_id(team: str, name: str) -> str:
    """Return an attacker/defender-qualified network character identifier."""
    return f"{team}:{name}"


class NetworkUserInputController(UserInputController):
    """Use the existing pathfinder while keeping remote commands thread-safe.

    The server calls ``submit_actions`` after validating team ownership and
    board bounds. The transport layer never mutates game characters directly.
    """

    def __init__(self, team: str):
        super().__init__()
        if team not in ("A", "D"):
            raise ValueError("team must be A or D")
        self.team = team
        self._command_lock = Lock()
        self._follow_predecessors: dict[str, str] = {}
        self._follow_origins: dict[str, tuple[int, int]] = {}
        self._follow_leader: str | None = None

    def prepare_follow_positions(self, chars) -> None:
        """Capture every position before the authoritative tick starts."""
        with self._command_lock:
            self._follow_origins = {
                network_character_id(char.team, str(char.name)): tuple(map(int, char.pos))
                for char in chars if char.is_alive
            }

    def follow_order(self) -> list[str]:
        with self._command_lock:
            leader = self._follow_leader
            if leader is None:
                return []
            order = [leader]
            while True:
                follower = next(
                    (name for name, predecessor in self._follow_predecessors.items()
                     if predecessor == order[-1]),
                    None,
                )
                if follower is None:
                    return order
                order.append(follower)

    def submit_actions(self, actions: list[dict[str, Any]], chars, grid) -> None:
        by_name = {
            network_character_id(char.team, str(char.name)): char
            for char in chars if char.team == self.team
        }
        with self._command_lock:
            moves = [action for action in actions if action["kind"] == "move"]
            if len(moves) > 1 and len({tuple(action["target"]) for action in moves}) == 1:
                names = [str(action["character"]) for action in moves]
                self._follow_leader = names[0]
                self._follow_predecessors = {
                    name: names[index - 1]
                    for index, name in enumerate(names) if index > 0
                }
            elif moves:
                self._follow_leader = None
                self._follow_predecessors.clear()
            else:
                self._follow_leader = None
                self._follow_predecessors.clear()
            for action in actions:
                name = str(action["character"])
                char = by_name.get(name)
                if char is None or not char.is_alive:
                    continue
                if action["kind"] == "move":
                    target = tuple(map(int, action["target"]))
                    if (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]
                            and int(grid[target]) != 1):
                        self.targets[str(char.name)] = target
                elif action["kind"] == "hold":
                    self.targets.pop(str(char.name), None)

    def decide_move(self, char, game_state):
        with self._command_lock:
            name = network_character_id(char.team, str(char.name))
            predecessor = self._follow_predecessors.get(name)
            target = self.targets.get(char.name)
            previous_position = self._follow_origins.get(predecessor)
            if predecessor and target is not None and previous_position is not None:
                origin = tuple(map(int, char.pos))
                planted_pos = game_state.get("planted_pos")
                can_defuse = (
                    char.team == "D" and game_state.get("is_planted")
                    and planted_pos is not None
                    and max(abs(origin[0] - int(planted_pos[0])),
                            abs(origin[1] - int(planted_pos[1]))) <= 1
                )
                if not can_defuse:
                    distance = abs(origin[0] - previous_position[0]) + abs(
                        origin[1] - previous_position[1]
                    )
                    if distance == 1:
                        result = list(previous_position)
                    else:
                        result = self.move_towards_target(
                            origin, previous_position, game_state["grid"],
                            chars=game_state.get("chars", []), moving_char=char,
                            wait_for_allies=True,
                        )
                    facing = self.facings.get(char.name)
                    if facing is None:
                        return result
                    if not getattr(char, "facing_forced_this_tick", False):
                        char.facing = facing
                    return result if isinstance(result, tuple) else (result, {"facing": facing})
            return super().decide_move(char, game_state)

    def reset_round(self):
        with self._command_lock:
            super().reset_round()
            self._follow_predecessors.clear()
            self._follow_origins.clear()
            self._follow_leader = None

