"""Shared, short-lived enemy sightings for ConCon attackers."""

from controllers import BaseController
from game_core import FACING_VECTORS


SIGHTING_MEMORY_TICKS = 12


def facing_towards(source, target):
    dr = int(target[0]) - int(source[0])
    dc = int(target[1]) - int(source[1])
    if dr == 0 and dc == 0:
        return None
    return max(FACING_VECTORS, key=lambda direction: (
        FACING_VECTORS[direction][0] * dc
        + FACING_VECTORS[direction][1] * dr
    ))


class TeamEnemySightings:
    def __init__(self):
        self.reset_round()

    def reset_round(self):
        self.last_seen = {}

    def _prune(self, tick):
        for name, (_, seen_tick) in list(self.last_seen.items()):
            if tick < seen_tick or tick - seen_tick >= SIGHTING_MEMORY_TICKS:
                del self.last_seen[name]

    def observe(self, observer, chars, game, grid, tick):
        """Share positions that this teammate can see in its own game view."""
        self._prune(tick)
        for enemy in chars:
            if (not getattr(enemy, "is_alive", True)
                    or getattr(enemy, "team", None) == observer.team):
                continue
            if game is not None and hasattr(game, "check_line_of_sight"):
                visible = game.check_line_of_sight(observer, enemy)
            else:
                visible = BaseController.has_line_of_sight(observer.pos, enemy.pos, grid)
            if visible:
                self.last_seen[str(enemy.name)] = (
                    tuple(map(int, enemy.pos)), tick,
                )

    def facing(self, position, tick):
        self._prune(tick)
        position = tuple(map(int, position))
        candidates = [
            (max(abs(pos[0] - position[0]), abs(pos[1] - position[1])),
             tick - seen_tick, name, pos)
            for name, (pos, seen_tick) in self.last_seen.items()
            if pos != position
        ]
        if not candidates:
            return None
        return facing_towards(position, min(candidates)[3])

    def guard_move(self, result, position, tick):
        """Aim toward a recent sighting while making an ordinary move or wait."""
        destination = result
        if isinstance(result, tuple) and len(result) == 2:
            destination, action = result
            if action != "MOVE":
                return result
        if not isinstance(destination, (list, tuple)) or len(destination) != 2:
            return result
        facing = self.facing(destination, tick) or self.facing(position, tick)
        if facing is None:
            return result
        return list(destination), {"facing": facing}
