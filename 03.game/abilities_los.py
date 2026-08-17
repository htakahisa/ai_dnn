"""Ability projectiles, smoke behavior, line of sight, flash, and recon."""

from game_core import (
    FLASH_SPEED_CELLS_PER_TICK,
    FLASH_MAX_FLIGHT_TICKS,
    RECON_SPEED_CELLS_PER_TICK,
    FLASH_BURST_DURATION_TICKS,
    RECON_BURST_DISPLAY_TICKS,
    BLIND_DURATION_TICKS,
    RECON_REVEAL_SIZE,
    REVEAL_DURATION_TICKS,
    SMOKE_DURATION_TICKS,
)


class AbilityLosMixin:

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

        for r, c in line_cells:
            if self.grid[r, c] == 1:
                return False

        if block_smoke and not self._smoke_allows_line(line_cells, self._smoke_cells()):
            return False

        return True

    def check_line_of_sight(self, p1, p2):
        """壁とスモーク規則を考慮して、2人の間に射線が通るか判定する。"""
        line_cells = self._line_cells(tuple(p1.pos), tuple(p2.pos))

        for r, c in line_cells:
            if self.grid[r, c] == 1:
                return False

        return self._smoke_allows_line(line_cells, self._smoke_cells())

    def check_shot_line_of_sight(self, shooter, target):
        """射撃専用の射線判定。

        通常の壁・スモーク判定に加えて、射手と標的の間にいる
        生存プレイヤーを遮蔽物として扱う。味方・敵のどちらでも遮る。
        射手自身と標的自身のマスは遮蔽物判定から除外する。

        視認用 check_line_of_sight() の仕様は変更しない。
        """
        if not self.check_line_of_sight(shooter, target):
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
            {"pos": impact, "remaining_ticks": FLASH_BURST_DURATION_TICKS}
        )
        owner_team = projectile.get("team")
        for char in self.chars:
            if not char.is_alive or char.team == owner_team:
                continue
            if self.check_cell_line_of_sight(tuple(char.pos), impact, block_smoke=True):
                char.blind_remaining = max(char.blind_remaining, BLIND_DURATION_TICKS)

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
            {"cells": cells, "remaining_ticks": RECON_BURST_DISPLAY_TICKS}
        )
        owner_team = projectile.get("team")
        for char in self.chars:
            if char.is_alive and char.team != owner_team and tuple(char.pos) in cells:
                char.reveal_remaining = max(
                    char.reveal_remaining, REVEAL_DURATION_TICKS
                )

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
