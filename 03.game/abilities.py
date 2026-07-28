# abilities.py
"""アビリティ(フラッシュ/スモーク/リコン)の効果実装。
発動トリガー(いつ・どこへ使うか)はコントローラー側(UserInputController等)の責務とし、
ここでは「使用が指示されたら何が起こるか」だけを扱う。
"""

import random

# 💡変更: アビリティの物理パラメータ(速度・射程tick・持続時間・範囲)は
# game_core.py からそのままimportし、実ゲームと完全に一致させる。
from game_core import (
    WINNING_ROUNDS,
    TICK_TIME,
    MAX_HP,
    BODY_DAMAGE,
    HEADSHOT_DAMAGE,
    SHOOT_INTERVAL_TICKS,
    SIDE_PANEL_WIDTH,
    PLANT_REQUIRED_TICKS,
    DEFUSE_REQUIRED_TICKS,
    SMOKE_DURATION_TICKS,
    MOVING_ACCURACY,
    MOVING_TARGET_HIT_MULTIPLIER,
    BLIND_DURATION_TICKS,
    FLASH_BURST_DURATION_TICKS,
    BLIND_ACCURACY_MULTIPLIER,
    FLASH_SPEED_CELLS_PER_TICK,
    FLASH_MAX_FLIGHT_TICKS,
    RECON_SPEED_CELLS_PER_TICK,
    REVEAL_DURATION_TICKS,
    REVEALED_DODGE_MULTIPLIER,
    RECON_REVEAL_SIZE,
    COMBO_DISPLAY_TICKS,
    COMBO_BANNER_HEIGHT,
    ROUND_DURATION_TICKS,
    SPIKE_DETONATION_TICKS,
    RECON_BURST_DISPLAY_TICKS,
    SMOKE_WARNING_TICKS,
    ROUND_TRANSITION_TICKS,
    ABILITY_TYPES,
    FLASH_BLIND_TICKS,
    SMOKE_RADIUS,
    RECON_RADIUS,
)


class AbilityMixin:
    # -----------------------------------------------------------------
    # 割り当て
    # -----------------------------------------------------------------
    def assign_abilities(self):
        """attacker側のキャラにラウンドごとランダムでアビリティを1つずつ割り当てる(1人1種類)。
        defenderには割り当てない(defender学習時に別途対応)。"""
        attackers = [c for c in self.chars if c.team == "A"]
        pool = ABILITY_TYPES * ((len(attackers) // len(ABILITY_TYPES)) + 1)
        random.shuffle(pool)

        for char, ability in zip(attackers, pool):
            char.ability_type = ability
            char.flash_charges = 1 if ability == "flash" else 0
            char.smoke_charges = 1 if ability == "smoke" else 0
            char.recon_charges = 1 if ability == "recon" else 0

    # -----------------------------------------------------------------
    # 発動(コントローラーから呼ばれる想定の入口)
    # -----------------------------------------------------------------
    def use_ability(self, char, target_cell):
        """所持アビリティを発動する。チャージが無ければ何もせずFalseを返す。"""
        if char.ability_type == "flash" and char.flash_charges > 0:
            self._launch_flash(char, target_cell)
            char.flash_charges -= 1
            return True
        if char.ability_type == "recon" and char.recon_charges > 0:
            self._launch_recon(char, target_cell)
            char.recon_charges -= 1
            return True
        if char.ability_type == "smoke" and char.smoke_charges > 0:
            self._place_smoke(char, target_cell)
            char.smoke_charges -= 1
            return True
        return False

    # -----------------------------------------------------------------
    # 発射・設置
    # -----------------------------------------------------------------
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

    def _launch_flash(self, char, target_cell):
        path = self._projectile_path(tuple(char.pos), tuple(target_cell))
        self.flash_projectiles.append({
            "path": path, "progress": 0, "ticks_alive": 0, "team": char.team,
        })

    def _launch_recon(self, char, target_cell):
        path = self._projectile_path(tuple(char.pos), tuple(target_cell))
        self.recon_projectiles.append({
            "path": path, "progress": 0, "team": char.team,
        })

    def _place_smoke(self, char, target_cell):
        tr, tc = target_cell
        cells = {
            (rr, cc)
            for rr in range(tr - SMOKE_RADIUS, tr + SMOKE_RADIUS + 1)
            for cc in range(tc - SMOKE_RADIUS, tc + SMOKE_RADIUS + 1)
            if 0 <= rr < self.height and 0 <= cc < self.width and self.grid[rr, cc] != 1
        }
        self.smokes.append({"cells": cells, "remaining_ticks": SMOKE_DURATION_TICKS})

    # -----------------------------------------------------------------
    # 着弾効果
    # -----------------------------------------------------------------
    def _explode_flash(self, projectile, impact=None):
        impact = impact or projectile["path"][min(projectile["progress"], len(projectile["path"]) - 1)]
        self.flash_bursts.append({"pos": impact, "remaining_ticks": FLASH_BURST_DURATION_TICKS})
        owner_team = projectile.get("team")
        for char in self.chars:
            if not char.is_alive or char.team == owner_team:
                continue
            if self._check_cell_line_of_sight(tuple(char.pos), impact, block_smoke=True):
                char.blind_remaining = max(char.blind_remaining, BLIND_DURATION_TICKS)

    def _explode_recon(self, projectile, impact=None):
        impact = impact or projectile["path"][min(projectile["progress"], len(projectile["path"]) - 1)]
        ir, ic = impact
        radius = RECON_REVEAL_SIZE // 2
        cells = {
            (rr, cc)
            for rr in range(ir - radius, ir + radius + 1)
            for cc in range(ic - radius, ic + radius + 1)
            if 0 <= rr < self.height and 0 <= cc < self.width
        }
        self.recon_bursts.append({"cells": cells, "remaining_ticks": RECON_BURST_DISPLAY_TICKS})
        owner_team = projectile.get("team")
        for char in self.chars:
            if char.is_alive and char.team != owner_team and tuple(char.pos) in cells:
                char.reveal_remaining = max(char.reveal_remaining, REVEAL_DURATION_TICKS)

    # -----------------------------------------------------------------
    # 毎tick進行(process_battleから呼ぶ)
    # -----------------------------------------------------------------
    def _advance_ability_effects(self):
        for char in self.chars:
            char.blind_remaining = max(0, char.blind_remaining - 1)
            char.reveal_remaining = max(0, char.reveal_remaining - 1)

        for burst in self.flash_bursts:
            burst["remaining_ticks"] -= 1
        self.flash_bursts = [b for b in self.flash_bursts if b["remaining_ticks"] > 0]

        for burst in self.recon_bursts:
            burst["remaining_ticks"] -= 1
        self.recon_bursts = [b for b in self.recon_bursts if b["remaining_ticks"] > 0]

        remaining = []
        for p in self.flash_projectiles:
            p["ticks_alive"] += 1
            next_progress = p["progress"] + FLASH_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge or p["ticks_alive"] >= FLASH_MAX_FLIGHT_TICKS:
                self._explode_flash(p)
            else:
                remaining.append(p)
        self.flash_projectiles = remaining

        remaining = []
        for p in self.recon_projectiles:
            next_progress = p["progress"] + RECON_SPEED_CELLS_PER_TICK
            hit_wall_or_edge = next_progress >= len(p["path"]) - 1
            p["progress"] = min(next_progress, len(p["path"]) - 1)
            if hit_wall_or_edge:
                self._explode_recon(p)
            else:
                remaining.append(p)
        self.recon_projectiles = remaining

        for smoke in self.smokes:
            smoke["remaining_ticks"] -= 1
        self.smokes = [s for s in self.smokes if s["remaining_ticks"] > 0]

    # -----------------------------------------------------------------
    # スモークによる視線判定(check_line_of_sightから利用)
    # -----------------------------------------------------------------
    def _smoke_cells(self):
        cells = set()
        for smoke in self.smokes:
            cells.update(smoke["cells"])
        return cells

    def _smoke_allows_line(self, line_cells, smoke_cells):
        if not line_cells:
            return True
        start_in_smoke = line_cells[0] in smoke_cells
        end_in_smoke = line_cells[-1] in smoke_cells
        if start_in_smoke and end_in_smoke:
            return True
        if start_in_smoke != end_in_smoke:
            return False
        return not any(cell in smoke_cells for cell in line_cells[1:-1])

    def _check_cell_line_of_sight(self, start, end, block_smoke=True):
        line_cells = self._line_cells(start, end)
        for r, c in line_cells:
            if self.grid[r, c] == 1:
                return False
        if block_smoke and not self._smoke_allows_line(line_cells, self._smoke_cells()):
            return False
        return True