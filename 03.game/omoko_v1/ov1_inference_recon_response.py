"""Attacker recon reactions; defenders keep their search/retake decisions."""

from ov1_inference_carry_escape import escape_disclosed_fire
from ov1_inference_damage_avoidance import danger_cells
from ov1_inference_team_tactics import facing_towards, known_enemies, shot_clear, tactical_state
from grid_lines import line_cells

CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def recon_response(char, game_state, memory):
    if not getattr(char, "is_alive", True) or game_state.get("defender_setup_active"):
        return None
    # Disclosure alone does not justify interrupting defense. Keep search /
    # retake movement, utility and defusing with their existing policies, even
    # when a disclosed enemy has a clear shot. Shared facing and engine fire
    # still use public sightings; destruction-area avoidance is independent.
    if char.team == "D":
        memory.clear()
        return None
    state = tactical_state(game_state)
    enemies = known_enemies(char, state)
    if getattr(char, "reveal_remaining", 0) > 0:
        escape = escape_disclosed_fire(char, game_state, memory)
        if escape is not None:
            return escape
        if enemies:
            # Known attackers cannot shoot through the current wall/body cover.
            enemy = min(enemies, key=lambda e: abs(e.pos[0]-char.pos[0])+abs(e.pos[1]-char.pos[1]))
            return list(char.pos), {"facing": facing_towards(char.pos, enemy.pos) or char.facing}
        # A recon hit does not reveal the shooter's position to its victim.
        # Without a disclosed attacker, seek nearby wall cover using only map
        # geometry. Do not read hidden enemy coordinates to choose a direction.
        grid = state["grid"]
        occupied = {tuple(c.pos) for c in state["chars"] if c.name != char.name and c.is_alive
                    and (c.team == char.team or getattr(c, "position_known", True))}
        danger = danger_cells(char, game_state)

        def wall(cell):
            return not (0 <= cell[0] < grid.shape[0] and 0 <= cell[1] < grid.shape[1]) or grid[cell] == 1

        start = tuple(char.pos)
        candidates = [(start[0]+dr, start[1]+dc) for dr, dc in CARDINAL]
        candidates = [c for c in candidates if not wall(c) and c not in occupied and c not in danger]
        previous = memory.get("unknown_previous")
        def rank(cell):
            return (-sum(wall((cell[0]+dr,cell[1]+dc)) for dr,dc in CARDINAL), cell == previous, cell)
        step = min(candidates, key=rank, default=start)
        memory["unknown_previous"] = start
        char._navigation_step_limit = 1
        return list(step), {"facing": getattr(char, "facing", "N")}
    memory.clear()
    # Preserve planting and postplant positioning/defusing. These actors still
    # face disclosed enemies through shared tactics and the engine auto-fires
    # whenever their accepted action permits it. Do not stall the objective
    # throughout a distant recon window just to request stationary fire.
    if getattr(char, "has_spike", False) or game_state.get("is_planted"):
        return None
    smoke = set(state.get("smoke_cells") or ())
    targets = [e for e in enemies if getattr(e, "reveal_remaining", 0) > 0
               and len(line_cells(tuple(char.pos), tuple(e.pos))) > 2
               and smoke.intersection(line_cells(tuple(char.pos), tuple(e.pos)))
               and shot_clear(char, e, state)]
    if targets:
        enemy = min(targets, key=lambda e: abs(e.pos[0]-char.pos[0])+abs(e.pos[1]-char.pos[1]))
        return list(char.pos), {"facing": facing_towards(char.pos, enemy.pos) or char.facing}
    return None
