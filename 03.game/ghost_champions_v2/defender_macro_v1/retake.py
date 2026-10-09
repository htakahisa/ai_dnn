"""Commit one defender to an observed smoke-protected defuse opportunity."""
from game_core import DEFUSE_REQUIRED_TICKS
from ghost_champions_v2.geometry import route, valid


class CoveredRetake:
    def __init__(self):
        self.reset_round()

    def reset_round(self):
        self.runner = None
        self.commits = 0

    def coordinate(self, char, state, result):
        if not char.is_alive or not state.get("is_planted") or state.get("planted_pos") is None:
            return result
        if isinstance(result, tuple) and len(result) > 1:
            payload = result[1]
            if not isinstance(payload, dict) or any(k != "facing" for k in payload):
                return result
        if getattr(char, "defuse_timer", 0) > 0:
            return result
        chars = state.get("chars", ())
        allies = [c for c in chars if c.team == char.team and c.is_alive]
        active = [c for c in allies if getattr(c, "defuse_timer", 0) > 0]
        if active:
            return result
        smoke = set(map(tuple, state.get("smoke_cells", ())))
        if not smoke:
            return result
        grid, spike = state["grid"], tuple(state["planted_pos"])
        known_enemies = [e for e in chars if e.team != char.team and e.is_alive
                         and getattr(e, "position_known", True) and valid(grid, tuple(e.pos))]
        goals = [(r, c) for r in range(spike[0]-1, spike[0]+2) for c in range(spike[1]-1, spike[1]+2)
                 if valid(grid, (r, c)) and (r, c) in smoke
                 and not any(max(abs(e.pos[0]-r), abs(e.pos[1]-c)) <= 1 for e in known_enemies)]
        if not goals:
            return result
        if self.runner not in {str(c.name) for c in allies}:
            candidates = []
            for ally in allies:
                blocked = {tuple(c.pos) for c in chars if c is not ally and c.is_alive
                           and getattr(c, "position_known", True)}
                path = route(grid, tuple(ally.pos), goals, blocked)
                if len(path) > 1 or tuple(ally.pos) in goals:
                    candidates.append((len(path)-1, str(ally.name)))
            self.runner = min(candidates)[1] if candidates else None
        if str(char.name) != self.runner:
            return result
        blocked = {tuple(c.pos) for c in chars if c is not char and c.is_alive
                   and getattr(c, "position_known", True)}
        path = route(grid, tuple(char.pos), goals, blocked)
        distance = len(path)-1
        if (tuple(char.pos) not in goals and distance == 0) or distance > 6:
            return result
        if float(state.get("detonate_timer", 0)) < distance + DEFUSE_REQUIRED_TICKS:
            return result
        self.commits += 1
        if tuple(char.pos) in goals:
            return list(char.pos), "DEFUSE"
        return (list(path[1]), result[1]) if isinstance(result, tuple) else list(path[1])
