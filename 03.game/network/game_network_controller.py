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

    def submit_actions(self, actions: list[dict[str, Any]], chars, grid) -> None:
        by_name = {str(char.name): char for char in chars if char.team == self.team}
        with self._command_lock:
            for action in actions:
                name = str(action["character"])
                char = by_name.get(name)
                if char is None or not char.is_alive:
                    continue
                if action["kind"] == "move":
                    target = tuple(map(int, action["target"]))
                    if (0 <= target[0] < grid.shape[0] and 0 <= target[1] < grid.shape[1]
                            and int(grid[target]) != 1):
                        self.targets[name] = target
                elif action["kind"] == "hold":
                    self.targets.pop(name, None)

    def decide_move(self, char, game_state):
        with self._command_lock:
            return super().decide_move(char, game_state)

