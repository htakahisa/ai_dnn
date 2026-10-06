"""Ability and ultimate effects, projectiles, smoke, and line of sight."""

from collections import deque
from grid_lines import line_cells

from game_core import (
    absorb_shield_damage,
    FLASH_SPEED_CELLS_PER_TICK,
    FLASH_MAX_FLIGHT_TICKS,
    RECON_SPEED_CELLS_PER_TICK,
    FLASH_BURST_DURATION_TICKS,
    RECON_BURST_DISPLAY_TICKS,
    BLIND_DURATION_TICKS,
    ESCAPE_WARP_DELAY_TICKS,
    RECON_REVEAL_SIZE,
    REVEAL_DURATION_TICKS,
    SMOKE_DURATION_TICKS,
    FACING_VECTORS,
    MONITOR_COLLISION_REVEAL_TICKS,
    MONITOR_DRONE_HP,
    RAID_DISTANCE_CELLS,
    RAID_TRAIL_TICKS,
    TUNNEL_BLIND_TICKS,
    TUNNEL_ACTIVE_TICKS,
    TUNNEL_HALF_WIDTH,
    TUNNEL_WARNING_TICKS,
    RAMP_CHAIN_DISTANCE_CELLS,
    RAMP_ELECTRIC_TICKS,
    NEON_RADIUS_CELLS,
    NEON_WARNING_TICKS,
    NEON_ACTIVE_TICKS,
    NEON_DAMAGE_PER_TICK,
    DANCE_HEAL_HP,
    DANCE_MAX_HP,
    SERENADE_REVEAL_TICKS,
    DANCE_SPARKLE_TICKS,
    ASH_RANGE_CELLS,
    DESTRUCTION_AREA_TICKS,
    BALEMOON_WARNING_TICKS,
    CONTRACT_DAMAGE_PER_TICK,
)

ULTIMATE_FACING_STEPS = {
    "N": (-1, 0),
    "NE": (-1, 1),
    "E": (0, 1),
    "SE": (1, 1),
    "S": (1, 0),
    "SW": (1, -1),
    "W": (0, -1),
    "NW": (-1, -1),
}


class MonitorDrone:
    """Shootable Seeker ultimate unit."""

    def __init__(self, owner, index, target_name=None):
        self.name = f"{owner.name}:MONITOR:{index}"
        self.owner_name = owner.name
        self.team = owner.team
        self.pos = list(owner.pos)
        self.hp = MONITOR_DRONE_HP
        self.max_hp = MONITOR_DRONE_HP
        self.is_alive = True
        self.is_ultimate_drone = True
        self.target_name = target_name
        self.dodge_rate = 0.0
        self.moved_this_tick = False
        self.defuse_timer = 0
        self.reveal_remaining = 0
        self.los_revealed = False
        self.forced_facing_next_tick = None


class AbilityLosMixin:

    def _save_ultimate_points(self, owner):
        saved = self.match_stats.setdefault(
            owner.name,
            {"kills": owner.kills, "deaths": owner.deaths},
        )
        saved["ultimate_points"] = int(owner.ultimate_points)

    def _spend_ultimate(self, owner):
        owner.ultimate_points = 0
        self._save_ultimate_points(owner)

    def _serenade_has_living_teams(self, owner):
        return (any(char.is_alive and char.team == owner.team for char in self.chars)
                and any(char.is_alive and char.team != owner.team for char in self.chars))

    def execute_ai_ultimate(self, owner, ultimate_action, *, during_battle=False):
        """Execute an ultimate requested by a controller.

        Controllers use ``{"ultimate": "RAID|ESCAPE|MONITOR|TUNNEL|NEON|SERENADE", ...}``.
        ESCAPE and NEON additionally require a cell in ``target``.
        RAID and TUNNEL can set ``facing`` for the direction of the cast.
        """
        if not isinstance(ultimate_action, dict):
            return False

        ultimate_name = str(ultimate_action.get("ultimate", "")).upper()
        if ultimate_name != owner.ultimate_name:
            return False
        if owner.ultimate_points < owner.ultimate_cost:
            return False
        if ultimate_name == "SERENADE":
            if (owner.is_alive or getattr(self, "round_over", False) or
                    getattr(self, "match_over", False) or
                    not self._serenade_has_living_teams(owner)):
                return False
            self.serenade_flash_remaining = 1
            self.serenade_flash_applied_tick = self.battle_tick + (0 if during_battle else 1)
            for enemy in self.chars:
                if enemy.is_alive and enemy.team != owner.team:
                    enemy.reveal_remaining = max(enemy.reveal_remaining, SERENADE_REVEAL_TICKS)
                    if not during_battle:
                        enemy.serenade_reveal_pending = max(
                            getattr(enemy, "serenade_reveal_pending", 0), SERENADE_REVEAL_TICKS)
                    tracker = getattr(self, "analytics_tracker", None)
                    if tracker is not None:
                        tracker.record_contribution(owner, enemy, self.battle_tick, "serenade")
            self._spend_ultimate(owner)
            return True
        if not owner.is_alive:
            return False
        if ultimate_name == "BALEMOON":
            if not hasattr(self, "balemoon_warnings"):
                self.balemoon_warnings = []
            center = tuple(owner.pos)
            self.balemoon_warnings.append({
                "pos": center, "cells": self._destruction_area_cells(center, radius=2),
                "remaining_ticks": BALEMOON_WARNING_TICKS,
                "owner": owner.name, "team": owner.team,
            })
            self._spend_ultimate(owner)
            return True
        if ultimate_name in ("RAID", "ESCAPE") and self._ramp_blocks_movement(owner):
            return False

        if ultimate_name == "NEON":
            target = ultimate_action.get("target")
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                return False
            row, col = int(target[0]), int(target[1])
            if not (0 <= row < self.height and 0 <= col < self.width):
                return False
            if self.grid[row, col] == 1:
                return False
            cells = {
                (rr, cc)
                for rr in range(max(0, row - NEON_RADIUS_CELLS), min(self.height, row + NEON_RADIUS_CELLS + 1))
                for cc in range(max(0, col - NEON_RADIUS_CELLS), min(self.width, col + NEON_RADIUS_CELLS + 1))
                if self.grid[rr, cc] != 1
            }
            if not hasattr(self, "neon_bursts"):
                self.neon_bursts = []
            self.neon_bursts.append({
                "pos": (row, col), "cells": cells, "phase": "warning",
                "remaining_ticks": NEON_WARNING_TICKS,
                "owner": owner.name, "team": owner.team,
            })
            self._spend_ultimate(owner)
            return True

        if ultimate_name == "RAID":
            facing = ultimate_action.get("facing", owner.facing)
            step = ULTIMATE_FACING_STEPS.get(facing)
            if step is None:
                return False
            destination = tuple(owner.pos)
            old_pos = tuple(owner.pos)
            path = []
            for distance in range(1, RAID_DISTANCE_CELLS + 1):
                candidate = (
                    old_pos[0] + step[0] * distance,
                    old_pos[1] + step[1] * distance,
                )
                if not (
                    0 <= candidate[0] < self.height and 0 <= candidate[1] < self.width
                ):
                    break
                if self.grid[candidate[0], candidate[1]] == 1:
                    break
                if self._is_position_occupied(owner, candidate, old_pos):
                    break
                destination = candidate
                path.append(candidate)
            if destination == old_pos:
                return False
            owner.facing = facing
            owner.moved_this_tick = True
            self._spend_ultimate(owner)
            traversed = [old_pos]
            for destination in path:
                previous = tuple(owner.pos)
                owner.pos = list(destination)
                self._update_occupancy_after_move(previous, destination)
                traversed.append(destination)
                self._trigger_cell_effects(owner)
                if self._ramp_blocks_movement(owner):
                    break
            if not hasattr(self, "ultimate_trails"):
                self.ultimate_trails = []
            self.ultimate_trails.append({
                "start": old_pos, "end": tuple(owner.pos), "cells": traversed,
                "direction": step, "team": owner.team, "owner": owner.name,
                "remaining_ticks": RAID_TRAIL_TICKS,
                "created_tick": int(getattr(self, "battle_tick", 0)) + 1,
            })
            return True

        if ultimate_name == "ESCAPE":
            if any(
                portal.get("owner") == owner.name
                for portal in getattr(self, "escape_portals", [])
            ):
                return False
            target = ultimate_action.get("target")
            if not isinstance(target, (list, tuple)) or len(target) != 2:
                return False
            destination = (int(target[0]), int(target[1]))
            old_pos = tuple(owner.pos)
            if not (
                0 <= destination[0] < self.height and 0 <= destination[1] < self.width
            ):
                return False
            if self.grid[destination[0], destination[1]] == 1:
                return False
            if self._is_position_occupied(owner, destination, old_pos):
                return False
            if not hasattr(self, "escape_portals"):
                self.escape_portals = []
            self.escape_portals.append(
                {
                    "pos": destination,
                    "remaining_ticks": ESCAPE_WARP_DELAY_TICKS,
                    "owner": owner.name,
                    "team": owner.team,
                }
            )
            owner.moved_this_tick = False
            self._spend_ultimate(owner)
            return True

        if ultimate_name == "MONITOR":
            enemies = sorted(
                (
                    char
                    for char in self.chars
                    if char.is_alive and char.team != owner.team
                ),
                key=lambda char: (
                    max(
                        abs(int(char.pos[0]) - int(owner.pos[0])),
                        abs(int(char.pos[1]) - int(owner.pos[1])),
                    ),
                    str(char.name),
                ),
            )
            targets = [enemy.name for enemy in enemies[:2]]
            if len(targets) == 1:
                targets.append(targets[0])
            while len(targets) < 2:
                targets.append(None)
            start_index = int(getattr(self, "monitor_drone_serial", 0))
            self.monitor_drone_serial = start_index + 2
            self.monitor_drones.extend(
                MonitorDrone(owner, start_index + index, targets[index])
                for index in range(2)
            )
            self._spend_ultimate(owner)
            return True

        if ultimate_name == "TUNNEL":
            facing = ultimate_action.get("facing", owner.facing)
            cells = self._tunnel_cells(tuple(owner.pos), facing)
            if not cells:
                return False
            owner.facing = facing
            self.tunnel_bursts.append(
                {
                    "cells": cells,
                    "phase": "warning",
                    "remaining_ticks": TUNNEL_WARNING_TICKS,
                    "owner": owner.name,
                    "team": owner.team,
                }
            )
            self._spend_ultimate(owner)
            return True

        return False

    def _destruction_area_cells(self, center, *, radius):
        row, col = center
        return {(rr, cc)
                 for rr in range(max(0, row-radius), min(self.height, row+radius+1))
                 for cc in range(max(0, col-radius), min(self.width, col+radius+1))
                 if self.grid[rr, cc] != 1}

    def _create_destruction_area(self, owner, center, *, radius, level):
        cells = self._destruction_area_cells(center, radius=radius)
        if not hasattr(self, "destruction_areas"):
            self.destruction_areas = []
        self.destruction_areas.append({"cells": cells, "pos": center, "level": level,
                                       "remaining_ticks": DESTRUCTION_AREA_TICKS,
                                       "owner": owner.name, "team": owner.team})

    def _advance_balemoon_warnings(self):
        warnings = []
        owners = {char.name: char for char in self.chars}
        for warning in getattr(self, "balemoon_warnings", []):
            if warning["remaining_ticks"] > 0:
                warning["remaining_ticks"] -= 1
                warnings.append(warning)
                continue
            owner = owners.get(warning["owner"])
            if owner is None:
                continue
            self._create_destruction_area(owner, warning["pos"], radius=2, level=10)
            if owner.is_alive:
                owner.hp = owner.max_hp
            for idol in list(self.chars):
                if idol.is_alive and getattr(idol, "role", None) == "アイドル":
                    self._kill_character(owner, idol, credit_kill=idol.team != owner.team)
        self.balemoon_warnings = warnings

    def _trigger_destruction_areas(self, char):
        if not char.is_alive or getattr(char, "life_contract_remaining", 0) > 0:
            return
        areas = [area for area in getattr(self, "destruction_areas", [])
                 if area["remaining_ticks"] > 0 and area["team"] != char.team
                 and tuple(char.pos) in area["cells"]]
        if areas:
            area = max(areas, key=lambda item: item["level"])
            char.life_contract_remaining = area["level"]
            char.life_contract_owner = area["owner"]

    def _trigger_cell_effects(self, char, *, during_battle=False):
        self._trigger_destruction_areas(char)
        self._trigger_ramp_traps(char, during_battle=during_battle)

    def _advance_destruction_areas(self):
        self.destruction_areas = [area for area in getattr(self, "destruction_areas", [])
                                  if area["remaining_ticks"] > 0]
        owners = {char.name: char for char in self.chars}
        for char in self.chars:
            self._trigger_destruction_areas(char)
            if not char.is_alive or getattr(char, "life_contract_remaining", 0) <= 0:
                continue
            old_max = char.max_hp
            owner = owners.get(char.life_contract_owner)
            hp_damage = absorb_shield_damage(char, CONTRACT_DAMAGE_PER_TICK, owner)
            char.max_hp = max(0, old_max - hp_damage)
            char.contract_max_hp_lost = getattr(char, "contract_max_hp_lost", 0) + old_max - char.max_hp
            char.hp = max(0, min(char.max_hp, char.hp - hp_damage))
            char.life_contract_remaining -= 1
            tracker = getattr(self, "analytics_tracker", None)
            if tracker is not None and owner is not None:
                tracker.record_contribution(owner, char, self.battle_tick, "damage")
            if char.hp <= 0:
                self._kill_character(owner or char, char, credit_kill=owner is not None)
        for area in self.destruction_areas:
            area["remaining_ticks"] -= 1

    def _advance_visual_effects(self):
        for char in self.chars:
            if getattr(char, "heal_sparkle_applied_tick", None) != self.battle_tick:
                char.heal_sparkle_remaining = max(0, getattr(char, "heal_sparkle_remaining", 0) - 1)
        if getattr(self, "serenade_flash_applied_tick", None) != self.battle_tick:
            self.serenade_flash_remaining = max(0, getattr(self, "serenade_flash_remaining", 0) - 1)

    def _advance_raid_trails(self):
        trails = []
        for trail in getattr(self, "ultimate_trails", []):
            if trail.get("created_tick") != self.battle_tick:
                trail["remaining_ticks"] -= 1
            if trail["remaining_ticks"] > 0:
                trails.append(trail)
        self.ultimate_trails = trails

    def _remove_dead_ramp_traps(self):
        live_owners = {char.name for char in self.chars if char.is_alive}
        self.ramp_traps = [
            trap for trap in getattr(self, "ramp_traps", [])
            if trap["owner"] in live_owners
        ]

    def _ramp_blocks_movement(self, char):
        if getattr(char, "electric_remaining", 0) <= 0:
            return False
        applied_tick = getattr(char, "electric_applied_tick", None)
        return applied_tick is None or self.battle_tick + 1 < applied_tick + RAMP_ELECTRIC_TICKS

    def _ramp_reachable_cells(self, start):
        """Flood up to five cardinal steps through walkable cells."""
        distances = {start: 0}
        queue = deque([start])
        while queue:
            row, col = queue.popleft()
            distance = distances[(row, col)]
            if distance >= RAMP_CHAIN_DISTANCE_CELLS:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                cell = (row + dr, col + dc)
                if cell in distances:
                    continue
                if not (0 <= cell[0] < self.height and 0 <= cell[1] < self.width):
                    continue
                if self.grid[cell[0], cell[1]] == 1:
                    continue
                distances[cell] = distance + 1
                queue.append(cell)
        return distances.keys()

    def _trigger_ramp_traps(self, char, *, during_battle=False):
        if not getattr(self, "ramp_traps", None):
            return
        self._remove_dead_ramp_traps()
        if not char.is_alive:
            return
        triggered = [
            trap for trap in self.ramp_traps
            if trap["team"] != char.team and trap["pos"] == tuple(char.pos)
        ]
        if not triggered:
            return
        self.ramp_traps = [trap for trap in self.ramp_traps if trap not in triggered]
        # A connected component is visited once, even if chains form cycles.
        affected = {char.name}
        queue = deque([char])
        enemies = [enemy for enemy in self.chars if enemy.is_alive and enemy.team == char.team]
        while queue:
            source = queue.popleft()
            source.electric_remaining = RAMP_ELECTRIC_TICKS
            # Movement precedes process_battle's tick increment and status decay.
            source.electric_applied_tick = int(getattr(self, "battle_tick", 0)) + (0 if during_battle else 1)
            source.reveal_remaining = max(source.reveal_remaining, RAMP_ELECTRIC_TICKS)
            tracker = getattr(self, "analytics_tracker", None)
            if tracker is not None:
                owner = next((owner for owner in self.chars if owner.name == triggered[0]["owner"]), None)
                tracker.record_contribution(owner, source, source.electric_applied_tick, "ramp")
            reachable = self._ramp_reachable_cells(tuple(source.pos))
            for enemy in enemies:
                if enemy.name not in affected and tuple(enemy.pos) in reachable:
                    affected.add(enemy.name)
                    queue.append(enemy)

    def _advance_engineer_effects(self):
        self._remove_dead_ramp_traps()
        for char in self.chars:
            if getattr(char, "electric_applied_tick", None) != self.battle_tick:
                char.electric_remaining = max(0, getattr(char, "electric_remaining", 0) - 1)
            char.reveal_remaining = max(char.reveal_remaining, char.electric_remaining)
        remaining_bursts = []
        owners = {char.name: char for char in self.chars}
        for burst in getattr(self, "neon_bursts", []):
            remaining = int(burst["remaining_ticks"])
            if burst["phase"] == "warning":
                if remaining > 0:
                    burst["remaining_ticks"] = remaining - 1
                    remaining_bursts.append(burst)
                    continue
                burst["phase"] = "active"
                remaining = NEON_ACTIVE_TICKS
            if remaining <= 0:
                continue
            owner = owners.get(burst["owner"])
            for char in self.chars:
                if char.is_alive and char.team != burst["team"] and tuple(char.pos) in burst["cells"]:
                    damage = absorb_shield_damage(char, NEON_DAMAGE_PER_TICK, owner)
                    tracker = getattr(self, "analytics_tracker", None)
                    if tracker is not None:
                        tracker.record_contribution(owner, char, self.battle_tick, "damage")
                    # Respect the same once-per-round lethal-hit passive as gunfire.
                    if char.hp <= damage and char.hp >= char.max_hp and getattr(char, "iron_will_charges", 0) > 0:
                        char.iron_will_charges -= 1
                        char.hp = 1
                    else:
                        char.hp = max(0, char.hp - damage)
                        if char.hp <= 0 and owner is not None:
                            self._kill_character(owner, char)
            burst["remaining_ticks"] = remaining - 1
            remaining_bursts.append(burst)
        self.neon_bursts = remaining_bursts

    def _tunnel_cells(self, start, facing):
        """Wide, wall-piercing Paranoia-style corridor in facing direction."""
        vector = FACING_VECTORS.get(facing)
        if vector is None:
            return set()
        fx, fy = vector
        sr, sc = start
        cells = set()
        for row in range(self.height):
            for col in range(self.width):
                dr, dc = row - sr, col - sc
                forward = dc * fx + dr * fy
                sideways = abs(dc * (-fy) + dr * fx)
                if forward > 0.0 and sideways <= TUNNEL_HALF_WIDTH:
                    cells.add((row, col))
        return cells

    def _drone_next_step(self, start, goal):
        """One wall-aware cardinal step. Players and other drones are passable."""
        if start == goal:
            return start
        queue = deque([start])
        parent = {start: None}
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (row + dr, col + dc)
                if nxt in parent:
                    continue
                if not (0 <= nxt[0] < self.height and 0 <= nxt[1] < self.width):
                    continue
                if self.grid[nxt[0], nxt[1]] == 1:
                    continue
                parent[nxt] = (row, col)
                if nxt == goal:
                    step = nxt
                    while parent[step] is not None and parent[step] != start:
                        step = parent[step]
                    return step
                queue.append(nxt)
        return start

    def _advance_escape_portals(self):
        """Hold ESCAPE's caster in place, then warp after ten full ticks."""
        remaining_portals = []
        chars_by_name = {char.name: char for char in self.chars}
        for portal in getattr(self, "escape_portals", []):
            owner = chars_by_name.get(portal.get("owner"))
            if owner is None or not owner.is_alive:
                continue

            remaining = int(portal.get("remaining_ticks", 0))
            if remaining > 0:
                portal["remaining_ticks"] = remaining - 1
                remaining_portals.append(portal)
                continue

            if getattr(owner, "electric_remaining", 0) > 0:
                remaining_portals.append(portal)
                continue

            old_pos = tuple(owner.pos)
            destination = tuple(portal["pos"])
            smoke_cells = self._smoke_cells()
            owner.was_in_smoke_before_move = old_pos in smoke_cells
            owner.pos = [destination[0], destination[1]]
            owner.moved_this_tick = destination != old_pos
            owner.entered_smoke_this_tick = (
                destination in smoke_cells and old_pos not in smoke_cells
            )
            owner.exited_smoke_this_tick = (
                old_pos in smoke_cells and destination not in smoke_cells
            )
            owner.stopped_after_move_this_tick = False
            self._trigger_cell_effects(owner, during_battle=True)

        self.escape_portals = remaining_portals

    def _advance_tunnel_bursts(self):
        """Advance TUNNEL's five-tick warning and three-tick active phases."""
        remaining_bursts = []
        for burst in self.tunnel_bursts:
            phase = burst.get("phase", "warning")
            remaining = int(burst.get("remaining_ticks", 0))

            if phase == "warning":
                if remaining > 0:
                    burst["remaining_ticks"] = remaining - 1
                    remaining_bursts.append(burst)
                    continue
                burst["phase"] = "active"
                burst["remaining_ticks"] = TUNNEL_ACTIVE_TICKS
                phase = "active"
                remaining = TUNNEL_ACTIVE_TICKS

            if phase == "active":
                if remaining <= 0:
                    continue
                for char in self.chars:
                    if (
                        char.is_alive
                        and char.team != burst.get("team")
                        and tuple(char.pos) in burst["cells"]
                    ):
                        char.blind_remaining = max(
                            char.blind_remaining,
                            TUNNEL_BLIND_TICKS,
                        )
                        # Application occurs after the status decrement for this
                        # battle tick, so no extra compensation tick is needed.
                burst["remaining_ticks"] = remaining - 1
                remaining_bursts.append(burst)

        self.tunnel_bursts = remaining_bursts

    def _advance_monitor_drones(self):
        live_drones = [drone for drone in self.monitor_drones if drone.is_alive]
        reserved_targets = {
            drone.target_name for drone in live_drones if drone.target_name is not None
        }
        for drone in live_drones:
            enemies = [
                char for char in self.chars if char.is_alive and char.team != drone.team
            ]
            target = next(
                (enemy for enemy in enemies if enemy.name == drone.target_name),
                None,
            )
            if target is None and enemies:
                candidates = [
                    enemy for enemy in enemies if enemy.name not in reserved_targets
                ] or enemies
                target = min(
                    candidates,
                    key=lambda enemy: (
                        max(
                            abs(int(enemy.pos[0]) - int(drone.pos[0])),
                            abs(int(enemy.pos[1]) - int(drone.pos[1])),
                        ),
                        str(enemy.name),
                    ),
                )
                drone.target_name = target.name
                reserved_targets.add(target.name)

            old_pos = tuple(drone.pos)
            if target is not None:
                new_pos = self._drone_next_step(old_pos, tuple(target.pos))
                drone.pos = [new_pos[0], new_pos[1]]
            drone.moved_this_tick = tuple(drone.pos) != old_pos

            collided = next(
                (enemy for enemy in enemies if tuple(enemy.pos) == tuple(drone.pos)),
                None,
            )
            if collided is not None:
                collided.reveal_remaining = max(
                    collided.reveal_remaining,
                    MONITOR_COLLISION_REVEAL_TICKS,
                )
                drone.is_alive = False
                continue

            for enemy in enemies:
                if self.check_cell_line_of_sight(
                    tuple(drone.pos), tuple(enemy.pos), block_smoke=True
                ):
                    enemy.reveal_remaining = max(enemy.reveal_remaining, 1)

        self.monitor_drones = [drone for drone in self.monitor_drones if drone.is_alive]

    def execute_ai_ability(self, owner, ability_action):
        """AIコントローラーから受け取ったアビリティ要求を実行する。"""
        if not owner.is_alive or not isinstance(ability_action, dict):
            return False

        ability_name = str(ability_action.get("ability", "")).upper()
        if ability_name == "DANCE":
            if owner.ability_name != "DANCE" or getattr(owner, "dance_charges", 0) <= 0:
                return False
            target_name = ability_action.get("target_name")
            target_cell = ability_action.get("target")
            if target_name is None and isinstance(target_cell, str):
                target_name = target_cell
            target = next((ally for ally in self.chars
                           if ally is not owner and ally.team == owner.team and ally.is_alive
                           and ((target_name is not None and ally.name == target_name)
                                or (target_name is None and isinstance(target_cell, (list, tuple))
                                    and len(target_cell) == 2 and tuple(ally.pos) == tuple(target_cell)))), None)
            heal_cap = min(DANCE_MAX_HP, target.max_hp) if target is not None and getattr(target, "contract_max_hp_lost", 0) > 0 else DANCE_MAX_HP
            if target is None or target.hp >= heal_cap:
                return False
            target.hp = min(heal_cap, target.hp + DANCE_HEAL_HP)
            target.heal_sparkle_remaining = DANCE_SPARKLE_TICKS
            target.heal_sparkle_applied_tick = self.battle_tick + 1
            owner.dance_charges -= 1
            return True
        if ability_name == "RAMP":
            if owner.ability_name != "RAMP" or owner.ramp_charges <= 0:
                return False
            self._remove_dead_ramp_traps()
            position = tuple(owner.pos)
            if any(trap["pos"] == position and trap["team"] == owner.team for trap in self.ramp_traps):
                return False
            self.ramp_traps.append({"pos": position, "owner": owner.name, "team": owner.team})
            owner.ramp_charges -= 1
            return True
        target = ability_action.get("target")
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return False

        r, c = int(target[0]), int(target[1])
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        if self.grid[r, c] == 1:
            return False
        if owner.ability_name != ability_name:
            return False

        if ability_name == "ASH" and owner.ash_charges > 0:
            dr, dc = r-owner.pos[0], c-owner.pos[1]
            if dr*dr + dc*dc > ASH_RANGE_CELLS*ASH_RANGE_CELLS:
                return False
            path = [tuple(owner.pos)]
            for cell in self._line_cells(tuple(owner.pos), (r, c))[1:]:
                if self.grid[cell[0], cell[1]] == 1:
                    break
                path.append(cell)
            if not hasattr(self, "ash_projectiles"):
                self.ash_projectiles = []
            self.ash_projectiles.append({"owner": owner.name, "team": owner.team,
                                         "path": path, "progress": 0})
            owner.ash_charges -= 1
            return True

        if ability_name == "SMOKE" and owner.smoke_charges > 0:
            cells = {
                (rr, cc)
                for rr in range(r - 1, r + 2)
                for cc in range(c - 1, c + 2)
                if 0 <= rr < self.height
                and 0 <= cc < self.width
                and self.grid[rr, cc] != 1
            }
            self.smokes.append(
                {
                    "cells": cells,
                    "remaining_ticks": SMOKE_DURATION_TICKS,
                    "owner": owner.name,
                    "team": owner.team,
                    "center": (r, c),
                }
            )
            owner.smoke_charges -= 1
            self.smoke_thrown_this_tick = True
            return True

        if ability_name == "FLASH" and owner.flash_charges > 0:
            path = self._projectile_path(tuple(owner.pos), (r, c))
            if len(path) <= 1:
                return False
            self.flash_projectiles.append(
                {
                    "owner": owner.name,
                    "team": owner.team,
                    "path": path,
                    "progress": 0,
                    "ticks_alive": 0,
                }
            )
            owner.flash_charges -= 1
            return True

        if ability_name == "RECON" and owner.recon_charges > 0:
            path = self._projectile_path(tuple(owner.pos), (r, c))
            if len(path) <= 1:
                return False
            self.recon_projectiles.append(
                {
                    "owner": owner.name,
                    "team": owner.team,
                    "path": path,
                    "progress": 0,
                }
            )
            owner.recon_charges -= 1
            return True

        return False

    def _smoke_cells(self):
        cells = set()
        for smoke in self.smokes:
            cells.update(smoke["cells"])
        return cells

    def _line_cells(self, start, end):
        """2マス間を結ぶBresenham線上のセルを順番に返す。"""
        return line_cells(start, end)

    def _smoke_allows_line(self, line_cells, smoke_cells):
        if not line_cells:
            return True

        # 同じマス、または1マス隣なら射線を通す
        if len(line_cells) <= 2:
            return True

        # 距離がある場合、始点・終点・途中のどこかが
        # スモーク内なら射線を遮断する
        return not any(cell in smoke_cells for cell in line_cells)

    def check_cell_line_of_sight(self, start, end, block_smoke=True):
        line_cells = self._line_cells(start, end)

        grid_height, grid_width = self.grid.shape
        for r, c in line_cells:
            # 境界チェックを追加してIndexErrorを防止
            if r < 0 or r >= grid_height or c < 0 or c >= grid_width:
                continue
            if self.grid[r, c] == 1:
                return False

        if block_smoke and not self._smoke_allows_line(line_cells, self._smoke_cells()):
            return False

        return True

    def check_line_of_sight(self, p1, p2):
        """壁とスモーク規則を考慮して、2人の間に射線が通るか判定する。"""
        line_cells = self._line_cells(tuple(p1.pos), tuple(p2.pos))

        grid_height, grid_width = self.grid.shape
        for r, c in line_cells:
            # 境界チェックを追加してIndexErrorを防止
            if r < 0 or r >= grid_height or c < 0 or c >= grid_width:
                continue
            if self.grid[r, c] == 1:
                return False

        return self._smoke_allows_line(line_cells, self._smoke_cells())

    def can_ignore_smoke_for_shot(self, shooter, target):
        """Return whether this shot may treat smoke as transparent.

        An awakened shooter can always see through smoke.  Independently, a
        target currently revealed by recon is visible to every teammate even
        through smoke.  This intentionally affects only smoke: walls and
        living characters on the firing line are still checked separately.
        """
        return bool(
            getattr(shooter, "sees_through_smoke", False)
            or getattr(target, "reveal_remaining", 0) > 0
        )

    def check_shot_line_of_sight(self, shooter, target):
        """射撃専用の射線判定。

        通常の壁・スモーク判定に加えて、射手と標的の間にいる
        生存プレイヤーを遮蔽物として扱う。味方・敵のどちらでも遮る。
        射手自身と標的自身のマスは遮蔽物判定から除外する。

        視認用 check_line_of_sight() の仕様は変更しない。
        """
        ignore_smoke = self.can_ignore_smoke_for_shot(shooter, target)
        if not self.check_cell_line_of_sight(
            tuple(shooter.pos), tuple(target.pos), block_smoke=not ignore_smoke
        ):
            return False

        line_cells = self._line_cells(tuple(shooter.pos), tuple(target.pos))
        if len(line_cells) <= 2:
            return True

        intermediate_cells = set(line_cells[1:-1])
        if not intermediate_cells:
            return True

        for char in self.chars:
            if char is shooter or char is target or not char.is_alive:
                continue
            if tuple(char.pos) in intermediate_cells:
                return False

        return True

    def get_viewer_team(self):
        controllers = self.get_user_controllers()
        return controllers[0][1] if controllers else None

    def is_visible_to_team(self, target, viewer_team):
        if viewer_team is None or target.team == viewer_team:
            return True
        return self._is_revealed(target)

    def _can_reveal_by_sight(self, viewer, target):
        if not viewer.is_alive or not target.is_alive or viewer.team == target.team:
            return False
        if getattr(viewer, "blind_remaining", 0) > 0:
            return False
        facing = FACING_VECTORS.get(getattr(viewer, "facing", None))
        if facing is None:
            return False
        dc = target.pos[1] - viewer.pos[1]
        dr = target.pos[0] - viewer.pos[0]
        if facing[0] * dc + facing[1] * dr < 0:
            return False
        return self.check_line_of_sight(viewer, target)

    def _update_los_reveal(self):
        """敵同士の射線が通っている間、双方をリビール状態として扱う。"""
        for char in self.chars:
            char.los_revealed = False
        alive = [char for char in self.chars if char.is_alive]
        for i, first in enumerate(alive):
            for second in alive[i + 1 :]:
                if self._can_reveal_by_sight(first, second):
                    second.los_revealed = True
                if self._can_reveal_by_sight(second, first):
                    first.los_revealed = True

    def _is_revealed(self, char):
        return char.reveal_remaining > 0 or char.los_revealed

    def _current_los_revealed_names(self):
        """現在、敵との射線が通っているキャラクター名の集合を返す。"""
        revealed = set()
        alive = [char for char in self.chars if char.is_alive]
        for i, first in enumerate(alive):
            for second in alive[i + 1 :]:
                if self._can_reveal_by_sight(first, second):
                    revealed.add(second.name)
                if self._can_reveal_by_sight(second, first):
                    revealed.add(first.name)
        return revealed

    def _is_revealed_for_shot(self, char, current_los_revealed_names):
        """射撃判定用のリビール状態。

        リコン由来のリビールは即時適用する。
        射線由来は「前Tickでもリビール済み」かつ「現在も射線が通る」場合だけ
        回避率低下を適用する。これにより初めて視認したTickは通常回避率で撃たれ、
        その射撃後から射線リビール状態になる。
        """
        recon_revealed = char.reveal_remaining > 0
        persistent_los_revealed = (
            char.los_revealed and char.name in current_los_revealed_names
        )
        return recon_revealed or persistent_los_revealed

    def _projectile_path(self, start, aimed_cell):
        """指定マスを方向として、壁またはマップ端まで伸びる投射経路を作る。"""
        sr, sc = start
        ar, ac = aimed_cell
        dr, dc = ar - sr, ac - sc
        if dr == 0 and dc == 0:
            return [start]
        scale = max(self.height, self.width) * 3
        far = (sr + dr * scale, sc + dc * scale)
        raw = self._line_cells(start, far)
        path = [start]
        for rr, cc in raw[1:]:
            if not (0 <= rr < self.height and 0 <= cc < self.width):
                break
            if self.grid[rr, cc] == 1:
                break
            path.append((rr, cc))
        return path

    def _explode_flash(self, projectile, impact=None):
        impact = (
            impact
            or projectile["path"][
                min(projectile["progress"], len(projectile["path"]) - 1)
            ]
        )
        self.flash_bursts.append(
            {
                "pos": impact,
                "remaining_ticks": FLASH_BURST_DURATION_TICKS,
                "owner": projectile.get("owner"),
                "team": projectile.get("team"),
            }
        )
        owner_team = projectile.get("team")
        for char in self.chars:
            if not char.is_alive or char.team == owner_team:
                continue
            if self.check_cell_line_of_sight(tuple(char.pos), impact, block_smoke=True):
                char.blind_remaining = max(char.blind_remaining, BLIND_DURATION_TICKS)
                tracker = getattr(self, "analytics_tracker", None)
                if tracker is not None:
                    owner = next(
                        (
                            c
                            for c in self.chars
                            if str(c.name) == str(projectile.get("owner"))
                        ),
                        None,
                    )
                    tracker.record_contribution(owner, char, self.battle_tick, "flash")

    def _explode_recon(self, projectile, impact=None):
        impact = (
            impact
            or projectile["path"][
                min(projectile["progress"], len(projectile["path"]) - 1)
            ]
        )
        ir, ic = impact
        # 9x9: 着弾地点を中心に上下左右へ4マスずつ。
        radius = RECON_REVEAL_SIZE // 2
        cells = {
            (rr, cc)
            for rr in range(ir - radius, ir + radius + 1)
            for cc in range(ic - radius, ic + radius + 1)
            if 0 <= rr < self.height and 0 <= cc < self.width
        }
        self.recon_bursts.append(
            {
                "cells": cells,
                "remaining_ticks": RECON_BURST_DISPLAY_TICKS,
                "owner": projectile.get("owner"),
                "team": projectile.get("team"),
            }
        )
        owner_team = projectile.get("team")
        for char in self.chars:
            if char.is_alive and char.team != owner_team and tuple(char.pos) in cells:
                char.reveal_remaining = max(
                    char.reveal_remaining, REVEAL_DURATION_TICKS
                )
                tracker = getattr(self, "analytics_tracker", None)
                if tracker is not None:
                    owner = next(
                        (
                            c
                            for c in self.chars
                            if str(c.name) == str(projectile.get("owner"))
                        ),
                        None,
                    )
                    tracker.record_contribution(owner, char, self.battle_tick, "recon")

    def _advance_ash_projectiles(self):
        remaining = []
        owners = {char.name: char for char in self.chars}
        for projectile in getattr(self, "ash_projectiles", []):
            projectile["progress"] = min(projectile["progress"] + FLASH_SPEED_CELLS_PER_TICK,
                                          len(projectile["path"]) - 1)
            if projectile["progress"] == len(projectile["path"]) - 1:
                owner = owners.get(projectile["owner"])
                if owner is not None:
                    self._create_destruction_area(owner, projectile["path"][-1], radius=1, level=5)
            else:
                remaining.append(projectile)
        self.ash_projectiles = remaining

    def _advance_flash_projectiles(self):
        remaining = []
        for projectile in self.flash_projectiles:
            projectile["ticks_alive"] += 1
            next_progress = projectile["progress"] + FLASH_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(projectile["path"]) - 1
            projectile["progress"] = min(next_progress, len(projectile["path"]) - 1)
            if hit_wall_or_edge or projectile["ticks_alive"] >= FLASH_MAX_FLIGHT_TICKS:
                self._explode_flash(projectile)
            else:
                remaining.append(projectile)
        self.flash_projectiles = remaining

    def _advance_recon_projectiles(self):
        remaining = []
        for projectile in self.recon_projectiles:
            next_progress = projectile["progress"] + RECON_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(projectile["path"]) - 1
            projectile["progress"] = min(next_progress, len(projectile["path"]) - 1)
            if hit_wall_or_edge:
                self._explode_recon(projectile)
            else:
                remaining.append(projectile)
        self.recon_projectiles = remaining
