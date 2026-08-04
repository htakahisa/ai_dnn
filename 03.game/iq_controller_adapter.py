from __future__ import annotations
from typing import Any
import numpy as np

from iq_perception import IQPerceptionEngine


class IQAwareController:
    def __init__(self, inner_controller: Any, perception_engine: IQPerceptionEngine | None = None):
        self.inner = inner_controller
        self.perception_engine = perception_engine or IQPerceptionEngine()
        self.real_game = None

    @property
    def inner_controller(self):
        return self.inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def set_game(self, game):
        self.real_game = game
        if hasattr(self.inner, "set_game"):
            self.inner.set_game(game)

    def reset_round(self):
        self.perception_engine.clear_cache()
        if hasattr(self.inner, "reset_round"):
            self.inner.reset_round()

    def _valid_real_move(self, char, destination):
        if self.real_game is None:
            return True
        if not isinstance(destination, (list, tuple, np.ndarray)) or len(destination) != 2:
            return True
        try:
            r, c = int(destination[0]), int(destination[1])
        except Exception:
            return False
        grid = self.real_game.grid
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False
        if int(grid[r, c]) == 1:
            return False
        return not any(
            other is not char
            and bool(getattr(other, "is_alive", True))
            and int(other.pos[0]) == r
            and int(other.pos[1]) == c
            for other in self.real_game.chars
        )

    def _sanitize(self, char, result):
        if isinstance(result, tuple) and len(result) >= 1:
            if not self._valid_real_move(char, result[0]):
                return (list(char.pos), *result[1:])
            return result
        if isinstance(result, (list, tuple, np.ndarray)) and len(result) == 2:
            if not self._valid_real_move(char, result):
                return list(char.pos)
        return result

    def decide_move(self, char, game_state):
        if self.real_game is None:
            raise RuntimeError("IQAwareController.set_game(game)が未実行です")

        view = self.perception_engine.build_game_view(viewer=char, game=self.real_game)
        perceived_char = view.perceived_character_for(char)
        perceived_state = self.perception_engine.build_perceived_state(
            viewer=char,
            game_state=game_state,
            game_view=view,
        )
        if hasattr(self.inner, "set_game"):
            self.inner.set_game(view)
        result = self.inner.decide_move(perceived_char, perceived_state)
        return self._sanitize(char, result)
