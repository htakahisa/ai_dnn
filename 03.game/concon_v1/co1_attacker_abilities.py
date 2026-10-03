"""Shared tactical ability decisions for A1 training and inference."""

from game_core import FACING_VECTORS, FLASH_MAX_FLIGHT_TICKS, FLASH_SPEED_CELLS_PER_TICK


def _visible_enemies(game, allies):
    return [enemy for enemy in game.chars
            if enemy.is_alive and enemy.team != allies[0].team
            and any(game.check_line_of_sight(ally, enemy) for ally in allies)]


def _impact_aim(game, caster, center, *, radius=0, require_flash_hit=None, flash=False):
    """Find an aim whose actual projectile endpoint reaches the desired area."""
    start = tuple(caster.pos)
    best = None
    for row in range(game.grid.shape[0]):
        for col in range(game.grid.shape[1]):
            if game.grid[row, col] == 1 or (row, col) == start:
                continue
            path = game._projectile_path(start, (row, col))
            if len(path) <= 1:
                continue
            impact = path[min(len(path) - 1,
                              FLASH_MAX_FLIGHT_TICKS * FLASH_SPEED_CELLS_PER_TICK)] \
                if flash or require_flash_hit is not None else path[-1]
            distance = max(abs(impact[0] - center[0]), abs(impact[1] - center[1]))
            if distance > radius:
                continue
            if require_flash_hit is not None and not game.check_cell_line_of_sight(
                    tuple(require_flash_hit.pos), impact, block_smoke=True):
                continue
            score = (distance, len(path), abs(row - center[0]) + abs(col - center[1]))
            if best is None or score < best[0]:
                best = (score, (row, col))
    return best[1] if best else None


class FixedSmokePlan:
    """Spend real smoke charges at each marked point at most once per round."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.reset_round()

    def reset_round(self):
        self.used_points = set()

    def choose(self, char, game):
        if (not self.scenario.smoke_points or game is None or not char.is_alive
                or char.ability_name != "SMOKE" or not char.smoke_charges):
            return None
        # Confirm actual casts; rejected requests do not consume a point.
        self.used_points.update(tuple(smoke["center"]) for smoke in game.smokes
                                if smoke.get("team") == char.team)
        from concon_v1.co1_attacker_common import bfs_distance_map

        allies = [ally for ally in game.chars if ally.is_alive and ally.team == char.team]
        for point in self.scenario.smoke_points:
            if point in self.used_points:
                continue
            distances = bfs_distance_map(game.grid, point)
            if any(0 <= distances[tuple(ally.pos)] <= self.scenario.smoke_trigger_bfs_distance
                   for ally in allies):
                return {"ability": "SMOKE", "target": point}
        return None


class FixedFlashPlan:
    """Trigger a real flash only when its projectile can reach the marked cell."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.reset_round()

    def reset_round(self):
        self.used_points = set()
        self.pending = {}

    def choose(self, char, game):
        if not self.scenario.flash_points or game is None:
            return None
        # Charge consumption confirms casts even after their projectiles disappear.
        by_name = {ally.name: ally for ally in game.chars if ally.team == char.team}
        for point, (name, charges) in list(self.pending.items()):
            owner = by_name.get(name)
            if owner is not None and owner.flash_charges < charges:
                self.used_points.add(point)
                del self.pending[point]
        if not char.is_alive or char.ability_name != "FLASH" or not char.flash_charges:
            return None
        from concon_v1.co1_attacker_common import bfs_distance_map

        allies = [ally for ally in game.chars if ally.is_alive and ally.team == char.team]
        for point in self.scenario.flash_points:
            if point in self.used_points:
                continue
            distances = bfs_distance_map(game.grid, point)
            if not any(0 <= distances[tuple(ally.pos)] <= self.scenario.flash_trigger_bfs_distance
                       for ally in allies):
                continue
            aim = _impact_aim(game, char, point, flash=True)
            if aim is None:
                continue
            self.pending[point] = (char.name, char.flash_charges)
            return {"ability": "FLASH", "target": aim}
        return None


def choose_ability(char, game, *, route_goal=None, allow_smoke=True, allow_flash=True):
    """Return a supported ability payload, using only team sight and smoke history."""
    if game is None or not char.is_alive:
        return None
    allies = [ally for ally in game.chars if ally.is_alive and ally.team == char.team]
    if not allies:
        return None
    visible = _visible_enemies(game, allies)
    visible.sort(key=lambda enemy: (
        max(abs(enemy.pos[0] - char.pos[0]), abs(enemy.pos[1] - char.pos[1])),
        str(enemy.name),
    ))
    smoke_cells = game._smoke_cells()

    if allow_smoke and char.ability_name == "SMOKE" and char.smoke_charges and visible:
        enemy = next((enemy for enemy in visible if tuple(enemy.pos) not in smoke_cells), None)
        if enemy is not None:
            return {"ability": "SMOKE", "target": tuple(enemy.pos)}

    if char.ability_name == "RECON" and char.recon_charges:
        if not any(projectile.get("team") == char.team
                   for projectile in game.recon_projectiles):
            for enemy in visible:
                if enemy.reveal_remaining > 0:
                    continue
                aim = _impact_aim(game, char, tuple(enemy.pos), radius=4)
                if aim is not None:
                    return {"ability": "RECON", "target": aim}
        allied_smokes = [smoke for smoke in game.smokes
                         if smoke.get("team") == char.team and smoke.get("remaining_ticks", 0) > 0]
        for smoke in allied_smokes:
            center = tuple(smoke["center"])
            # A reveal already running in the smoke needs no second dart.
            if any(enemy.is_alive and enemy.team != char.team
                   and tuple(enemy.pos) in smoke["cells"]
                   and enemy.reveal_remaining > 0 for enemy in game.chars):
                continue
            if any(projectile.get("team") == char.team for projectile in game.recon_projectiles):
                break
            # Smoke is 3x3 and recon reveals 9x9: within 3 cells of the
            # center covers every smoke cell, even at opposite edges.
            aim = _impact_aim(game, char, center, radius=3)
            if aim is not None:
                return {"ability": "RECON", "target": aim}

    if allow_flash and char.ability_name == "FLASH" and char.flash_charges:
        for ally in allies:
            if ally is char:
                continue
            for enemy in visible:
                if (not game.check_line_of_sight(ally, enemy)
                        or not game.check_shot_line_of_sight(ally, enemy)):
                    continue
                aim = _impact_aim(game, char, tuple(enemy.pos),
                                  radius=max(game.grid.shape), require_flash_hit=enemy)
                if aim is not None:
                    return {"ability": "FLASH", "target": aim}

    if char.ultimate_points < char.ultimate_cost:
        return None
    name = char.ultimate_name
    if name == "MONITOR" and visible:
        return {"ultimate": name}
    if name == "NEON" and visible:
        return {"ultimate": name, "target": tuple(visible[0].pos)}
    if name == "BALEMOON" and visible and any(
            max(abs(enemy.pos[0] - char.pos[0]), abs(enemy.pos[1] - char.pos[1])) <= 2
            for enemy in visible):
        return {"ultimate": name}
    if name in ("TUNNEL", "RAID") and visible:
        enemy = visible[0]
        direction = min(FACING_VECTORS, key=lambda key: -(
            FACING_VECTORS[key][0] * (enemy.pos[1] - char.pos[1])
            + FACING_VECTORS[key][1] * (enemy.pos[0] - char.pos[0])))
        if name == "TUNNEL" and tuple(enemy.pos) in game._tunnel_cells(tuple(char.pos), direction):
            return {"ultimate": name, "facing": direction}
        if name == "RAID" and not game._ramp_blocks_movement(char):
            return {"ultimate": name, "facing": direction}
    if name == "ESCAPE" and route_goal is not None:
        goal = tuple(route_goal)
        if (goal != tuple(char.pos) and game.grid[goal] != 1
                and not any(ally.is_alive and tuple(ally.pos) == goal for ally in game.chars)):
            return {"ultimate": name, "target": goal}
    return None
