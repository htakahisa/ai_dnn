"""Ability and ultimate effects, projectiles, smoke, and line of sight."""

from collections import deque

from game_core import (
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
    TUNNEL_BLIND_TICKS,
    TUNNEL_ACTIVE_TICKS,
    TUNNEL_HALF_WIDTH,
    TUNNEL_WARNING_TICKS,
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

    def execute_ai_ultimate(self, owner, ultimate_action):
        """Execute an ultimate requested by a controller.

        Controllers use ``{"ultimate": "RAID|ESCAPE|MONITOR|TUNNEL", ...}``.
        ESCAPE additionally requires a destination in ``target``.
        """
        if not owner.is_alive or not isinstance(ultimate_action, dict):
            return False

        ultimate_name = str(ultimate_action.get("ultimate", "")).upper()
        if ultimate_name != owner.ultimate_name:
            return False
        if owner.ultimate_points < owner.ultimate_cost:
            return False

        if ultimate_name == "RAID":
            step = ULTIMATE_FACING_STEPS.get(owner.facing)
            if step is None:
                return False
            destination = tuple(owner.pos)
            old_pos = tuple(owner.pos)
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
            if destination == old_pos:
                return False
            owner.pos = [destination[0], destination[1]]
            owner.moved_this_tick = True
            self._update_occupancy_after_move(old_pos, destination)
            self._spend_ultimate(owner)
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
            cells = self._tunnel_cells(tuple(owner.pos), owner.facing)
            if not cells:
                return False
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
        y0, x0 = int(start[0]), int(start[1])
        y1, x1 = int(end[0]), int(end[1])
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        cells = []
        while True:
            cells.append((y0, x0))
            if x0 == x1 and y0 == y1:
                return cells
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

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
        if target.reveal_remaining > 0:
            return True
        return any(
            ally.is_alive
            and ally.team == viewer_team
            and self.check_line_of_sight(ally, target)
            for ally in self.chars
        )

    def _update_los_reveal(self):
        """敵同士の射線が通っている間、双方をリビール状態として扱う。"""
        for char in self.chars:
            char.los_revealed = False
        alive = [char for char in self.chars if char.is_alive]
        for i, first in enumerate(alive):
            for second in alive[i + 1 :]:
                if first.team != second.team and self.check_line_of_sight(
                    first, second
                ):
                    first.los_revealed = True
                    second.los_revealed = True

    def _is_revealed(self, char):
        return char.reveal_remaining > 0 or char.los_revealed

    def _current_los_revealed_names(self):
        """現在、敵との射線が通っているキャラクター名の集合を返す。"""
        revealed = set()
        alive = [char for char in self.chars if char.is_alive]
        for i, first in enumerate(alive):
            for second in alive[i + 1 :]:
                if first.team != second.team and self.check_line_of_sight(
                    first, second
                ):
                    revealed.add(first.name)
                    revealed.add(second.name)
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
