"""Keep a learned patrol direction from overriding an observed engagement."""
from gc_v1.gc_facing import facing_towards
from ghost_champions_v2.geometry import los, valid


def visible_targets(char, state):
    grid = state["grid"]
    if not valid(grid, tuple(char.pos)):
        return []
    chars = state.get("chars", ())
    occupied = {tuple(c.pos) for c in chars
                if c.is_alive and getattr(c, "position_known", True) and c is not char}
    smoke = set(map(tuple, state.get("smoke_cells", ())))
    return [enemy for enemy in chars
            if enemy.team != char.team and enemy.is_alive
            and getattr(enemy, "position_known", True)
            and valid(grid, tuple(enemy.pos))
            and los(grid, tuple(char.pos), tuple(enemy.pos), smoke, occupied)]


class CombatFacing:
    def __init__(self):
        self.reset_round()

    def reset_round(self):
        self.corrections = 0

    def coordinate(self, char, state, result, locked_facing=None):
        if not char.is_alive or state.get("defender_setup_active") or state.get("is_planted"):
            return result
        facing = locked_facing
        if facing is None:
            targets = visible_targets(char, state)
            if not targets:
                return result
            target = min(targets, key=lambda e: (
                max(abs(e.pos[0]-char.pos[0]), abs(e.pos[1]-char.pos[1])),
                float(getattr(e, "hp", 100)), str(e.name)))
            facing = facing_towards(char.pos, target.pos)
        if facing is None:
            return result
        if isinstance(result, tuple) and len(result) > 1:
            payload = result[1]
            # DEFUSE/COLLECT commands must keep their command representation.
            if not isinstance(payload, dict):
                char.facing = facing
                return result
            if payload.get("facing", getattr(char, "facing", None)) != facing:
                self.corrections += 1
            char.facing = facing
            return (result[0], dict(payload, facing=facing), *result[2:])
        if getattr(char, "facing", None) != facing:
            self.corrections += 1
        char.facing = facing
        return list(result), {"facing": facing}
