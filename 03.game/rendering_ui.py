"""Tkinter input handling and all visual rendering."""

import math

from controllers import UserInputController
from game_core import (
    COMBO_BANNER_HEIGHT, SIDE_PANEL_WIDTH, SMOKE_DURATION_SECONDS,
    COMBO_DISPLAY_TICKS, PLANT_REQUIRED_SECONDS, DEFUSE_REQUIRED_SECONDS,
)

class RenderingUIMixin:
    def get_user_controllers(self):
        """ユーザー操作のコントローラーとそのチームのペアを返す"""
        result = []
        if isinstance(self.attacker_controller, UserInputController):
            result.append((self.attacker_controller, "A"))
        if isinstance(self.defender_controller, UserInputController):
            result.append((self.defender_controller, "D"))
        return result


    def on_canvas_click(self, event):
        if event.y < COMBO_BANNER_HEIGHT:
            return
        event.y -= COMBO_BANNER_HEIGHT
        selected = self._selected_user_character()

        # プラント可能な選択キャラクターにだけ表示されるPLANTボタン。
        plant_bounds = self._plant_button_bounds()
        if selected and plant_bounds and plant_bounds[0] <= event.x <= plant_bounds[2] and plant_bounds[1] <= event.y <= plant_bounds[3]:
            selected.is_planting = not selected.is_planting
            selected.plant_timer = 0
            self.ability_mode = None
            self.draw()
            return

        # 選択中キャラクターは、ロールに対応したアビリティ一つだけ使用できる。
        if selected and selected.ability_name in ("SMOKE", "FLASH", "RECON"):
            ability_name = selected.ability_name
            panel = self._ability_button_bounds(ability_name)
            if panel and panel[0] <= event.x <= panel[2] and panel[1] <= event.y <= panel[3]:
                same_is_armed = self.ability_mode == (ability_name, selected.team, selected.name)
                if same_is_armed:
                    self.ability_mode = None
                else:
                    charges = {
                        "SMOKE": selected.smoke_charges,
                        "FLASH": selected.flash_charges,
                        "RECON": selected.recon_charges,
                    }[ability_name]
                    if charges > 0:
                        self.ability_mode = (ability_name, selected.team, selected.name)
                self.draw()
                return

        map_x = event.x - self.map_offset_x
        if map_x < 0 or map_x >= self.map_pixel_width:
            return
        c = map_x // self.cell_size
        r = event.y // self.cell_size
        if not (0 <= r < self.height and 0 <= c < self.width):
            return

        if self.ability_mode:
            ability_name, team, owner_name = self.ability_mode
            owner = next((ch for ch in self.chars if ch.name == owner_name and ch.is_alive), None)
            if owner and owner.team == team and owner.ability_name == ability_name and self.grid[r, c] != 1:
                if ability_name == "SMOKE" and owner.smoke_charges > 0:
                    cells = {(rr, cc) for rr in range(r-1, r+2) for cc in range(c-1, c+2)
                             if 0 <= rr < self.height and 0 <= cc < self.width and self.grid[rr, cc] != 1}
                    self.smokes.append({"cells": cells, "remaining_seconds": SMOKE_DURATION_SECONDS, "owner": owner.name})
                    owner.smoke_charges -= 1
                elif ability_name == "FLASH" and owner.flash_charges > 0:
                    path = self._projectile_path(tuple(owner.pos), (r, c))
                    if len(path) > 1:
                        self.flash_projectiles.append({
                            "owner": owner.name, "team": owner.team, "path": path,
                            "progress": 0, "ticks_alive": 0
                        })
                        owner.flash_charges -= 1
                elif ability_name == "RECON" and owner.recon_charges > 0:
                    path = self._projectile_path(tuple(owner.pos), (r, c))
                    if len(path) > 1:
                        self.recon_projectiles.append({
                            "owner": owner.name, "team": owner.team, "path": path, "progress": 0
                        })
                        owner.recon_charges -= 1
            self.ability_mode = None
            self.draw()
            return

        for ctrl, team in self.get_user_controllers():
            ctrl.handle_click(r, c, self.grid, self.chars, team)
        self.draw()


    def _selected_user_character(self):
        for ctrl, team in self.get_user_controllers():
            if ctrl.selected_char:
                return next((ch for ch in self.chars if ch.name == ctrl.selected_char and ch.team == team and ch.is_alive), None)
        return None


    def _can_selected_plant(self):
        selected = self._selected_user_character()
        if not selected or not selected.is_alive or selected.team != "A":
            return False
        if self.is_planted or not selected.has_spike:
            return False
        r, c = selected.pos
        return self.grid[r, c] == 2


    def _bottom_control_layout(self):
        total_w, h = min(690, self.map_pixel_width - 20), 68
        x1 = self.map_offset_x + (self.map_pixel_width - total_w) / 2
        y1 = self.map_pixel_height + 28
        if self._can_selected_plant():
            gap, plant_w = 10, 150
            return (x1, y1, x1 + total_w - plant_w - gap, y1 + h), (x1 + total_w - plant_w, y1, x1 + total_w, y1 + h)
        return (x1, y1, x1 + total_w, y1 + h), None


    def _ability_panel_bounds(self):
        if not self._selected_user_character():
            return None
        return self._bottom_control_layout()[0]


    def _plant_button_bounds(self):
        if not self._can_selected_plant():
            return None
        return self._bottom_control_layout()[1]


    def _ability_button_bounds(self, ability_name):
        panel = self._ability_panel_bounds()
        selected = self._selected_user_character()
        if not panel or not selected or selected.ability_name != ability_name:
            return None
        return panel


    def _map_x(self, x):
        return self.map_offset_x + x


    def _draw_compact_ability_icon(self, ability_name, cx, cy, available):
        """構えるUIと同じ意匠の小型アビリティアイコンを描く。"""
        active_color = {"SMOKE": "#e67e22", "FLASH": "#f1c40f", "RECON": "#65d8e8", "HUNT": "#e74c3c"}[ability_name]
        fill = active_color if available else "#59616c"
        outline = "#f8c471" if available else "#7f8c8d"
        if ability_name == "SMOKE":
            for ox, oy, radius in [(-4, 2, 4), (0, -2, 5), (5, 2, 4), (1, 5, 4)]:
                self.canvas.create_oval(cx+ox-radius, cy+oy-radius, cx+ox+radius, cy+oy+radius,
                                        fill=fill, outline=outline, width=1)
        elif ability_name == "FLASH":
            self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6, fill=fill, outline=outline, width=1)
            for dx, dy in [(0,-9),(0,9),(-9,0),(9,0)]:
                self.canvas.create_line(cx, cy, cx+dx, cy+dy, fill=fill, width=1)
        elif ability_name == "RECON":
            self.canvas.create_polygon(cx-8, cy+3, cx+5, cy-5, cx+8, cy-2, cx-4, cy+6,
                                       fill=fill, outline=outline)
        else:  # HUNT
            self.canvas.create_text(cx, cy, text="H", fill=fill, font=("Arial", 10, "bold"))


    def _draw_team_panel(self, team, x0, title, accent):
        """左右パネルへHP・K/D・アビリティ・現在の戦闘ステータスを表示する。"""
        panel_w = SIDE_PANEL_WIDTH
        panel_h = self.map_pixel_height + self.ability_area_height
        self.canvas.create_rectangle(
            x0, 0, x0 + panel_w, panel_h,
            fill="#111722", outline="#2a3444"
        )
        self.canvas.create_rectangle(x0, 0, x0 + panel_w, 42, fill=accent, outline="")
        self.canvas.create_text(
            x0 + panel_w / 2, 21,
            text=title, fill="white", font=("Arial", 13, "bold")
        )

        chars = [c for c in self.chars if c.team == team][:5]
        row_h = 108
        card_h = 100

        for i, char in enumerate(chars):
            y = 48 + i * row_h
            row_fill = "#1b2432" if char.is_alive else "#17191e"
            muted = "#aeb8c6" if char.is_alive else "#666b73"
            name_fill = "white" if char.is_alive else "#777b83"

            self.canvas.create_rectangle(
                x0 + 10, y, x0 + panel_w - 10, y + card_h,
                fill=row_fill, outline="#323e50"
            )

            # 名前・K/D
            self.canvas.create_text(
                x0 + 18, y + 13,
                text=char.display_name, anchor="w",
                fill=name_fill, font=("Arial", 9, "bold")
            )
            self.canvas.create_text(
                x0 + panel_w - 17, y + 13,
                text=f"K {char.kills}  D {char.deaths}", anchor="e",
                fill="#d6dde8" if char.is_alive else "#777b83",
                font=("Arial", 8, "bold")
            )

            # HPバー
            hp_ratio = char.hp / char.max_hp if char.is_alive and char.max_hp > 0 else 0.0
            bar_x1, bar_x2 = x0 + 18, x0 + panel_w - 18
            self.canvas.create_rectangle(bar_x1, y + 25, bar_x2, y + 37, fill="#343a45", outline="")
            self.canvas.create_rectangle(
                bar_x1, y + 25,
                bar_x1 + (bar_x2 - bar_x1) * hp_ratio, y + 37,
                fill=accent, outline=""
            )
            hp_text = f"HP {char.hp}/{char.max_hp}" if char.is_alive else "DEAD"
            self.canvas.create_text(
                (bar_x1 + bar_x2) / 2, y + 31,
                text=hp_text, fill="white", font=("Arial", 7, "bold")
            )

            # 現在値を表示するため、タイガー・コンボ・覚醒後の4能力から総合戦闘力も毎回再計算される。
            accuracy_pct = round(char.accuracy * 100)
            dodge_pct = round(char.dodge_rate * 100)
            hs_pct = round(char.hs_rate * 100)
            iq_text = f"{char.iq:g}"
            reaction_text = f"{char.reaction:g}"
            power_text = str(math.floor(char.combat_power))

            self.canvas.create_text(
                x0 + 18, y + 49,
                text=f"命中精度 {accuracy_pct}%   判断力(IQ) {iq_text}",
                anchor="w", fill=muted, font=("Arial", 7, "bold")
            )
            self.canvas.create_text(
                x0 + 18, y + 63,
                text=f"キャラコン(回避) {dodge_pct}%   HS {hs_pct}%",
                anchor="w", fill=muted, font=("Arial", 7, "bold")
            )
            self.canvas.create_text(
                x0 + 18, y + 77,
                text=f"反応速度 {reaction_text}",
                anchor="w", fill=muted, font=("Arial", 7, "bold")
            )
            self.canvas.create_text(
                x0 + panel_w - 18, y + 77,
                text=f"総合戦闘力 {power_text}",
                anchor="e", fill="#f5c76b" if char.is_alive else "#777b83",
                font=("Arial", 8, "bold")
            )

            # ロールに対応するアビリティ
            ability = char.ability_name
            charges = {
                "SMOKE": char.smoke_charges,
                "FLASH": char.flash_charges,
                "RECON": char.recon_charges,
                "HUNT": 1,
            }[ability]
            available = char.is_alive and (charges > 0 or ability == "HUNT")
            self._draw_compact_ability_icon(ability, x0 + 27, y + 91, available)
            label = {"SMOKE": "SMOKE", "FLASH": "FLASH", "RECON": "RECON", "HUNT": "HUNT +10%"}[ability]
            status = "PASSIVE" if ability == "HUNT" else f"残り {charges}"
            self.canvas.create_text(
                x0 + 43, y + 91,
                text=f"{label}  {status}", anchor="w",
                fill=muted, font=("Arial", 7, "bold")
            )


    def draw(self):
        if self.headless:
            return
        self.canvas.delete("all")
        self._draw_team_panel("A", 0, "ATTACKERS", "#c0392b")
        self._draw_team_panel("D", self.map_offset_x + self.map_pixel_width, "DEFENDERS", "#27ae60")

        color_map = {"0":"white", "1":"#34495e", "2":"#fff9c4", "3":"#ffcccc", "4":"#ccffcc"}
        for r in range(self.height):
            for c in range(self.width):
                x1 = self._map_x(c * self.cell_size)
                color = color_map.get(str(self.grid[r, c]), "white")
                self.canvas.create_rectangle(x1, r*self.cell_size, x1+self.cell_size, (r+1)*self.cell_size, fill=color, outline="#eee")

        # 煙らしく見えるように、半透明風の円を重ねて雲状に描画する。
        # Tkinter CanvasはRGBA非対応なのでstippleを使う。
        for smoke_index, smoke in enumerate(self.smokes):
            warning = smoke["remaining_seconds"] <= 3 * self._tick_seconds()
            if warning and self.battle_tick % 2 == 0:
                continue
            for sr, sc in smoke["cells"]:
                x1 = self._map_x(sc*self.cell_size)
                y1 = sr*self.cell_size
                pad = max(2, self.cell_size // 10)
                self.canvas.create_oval(
                    x1-pad, y1-pad, x1+self.cell_size+pad, y1+self.cell_size+pad,
                    fill="#d97706", outline="#f59e0b", width=1, stipple="gray50"
                )
                # 小さな煙の塊をずらして重ねる
                offsets = [
                    (-0.18, -0.12, 0.72), (0.20, -0.18, 0.66),
                    (-0.10, 0.22, 0.68), (0.24, 0.20, 0.62),
                ]
                for ox, oy, scale in offsets:
                    size = self.cell_size * scale
                    cx = x1 + self.cell_size * (0.5 + ox)
                    cy = y1 + self.cell_size * (0.5 + oy)
                    self.canvas.create_oval(
                        cx-size/2, cy-size/2, cx+size/2, cy+size/2,
                        fill="#f59e0b", outline="", stipple="gray50"
                    )

        if not self.is_planted and self.target_plant_pos:
            tr, tc = self.target_plant_pos
            x1 = self._map_x(tc*self.cell_size)
            self.canvas.create_rectangle(x1, tr*self.cell_size, x1+self.cell_size, (tr+1)*self.cell_size, fill="#f39c12", outline="#d35400")
        if self.is_planted and self.planted_pos:
            pr, pc = self.planted_pos
            x1 = self._map_x(pc*self.cell_size)
            self.canvas.create_oval(x1+2, pr*self.cell_size+2, x1+self.cell_size-2, (pr+1)*self.cell_size-2, fill="red", outline="")
        if self.spike_pos:
            sr, sc = self.spike_pos
            x1 = self._map_x(sc*self.cell_size)
            self.canvas.create_oval(x1+2, sr*self.cell_size+2, x1+self.cell_size-2, (sr+1)*self.cell_size-2, fill="black", outline="")

        if self.last_shot:
            shooter = self.last_shot["shooter"]
            target = self.last_shot["target"]
            self.canvas.create_line(
                self._map_x((shooter.pos[1]+0.5)*self.cell_size), (shooter.pos[0]+0.5)*self.cell_size,
                self._map_x((target.pos[1]+0.5)*self.cell_size), (target.pos[0]+0.5)*self.cell_size,
                fill="#ff3b30" if self.last_shot["hit"] else "#aab2bd", width=2, dash=() if self.last_shot["hit"] else (4, 3),
            )

        # 飛翔中フラッシュ：投擲済み経路を点線、現在位置を小さな光点で表示。
        for projectile in self.flash_projectiles:
            path = projectile["path"]
            end_index = min(projectile["progress"], len(path) - 1)
            if end_index > 0:
                coords = []
                for rr, cc in path[:end_index + 1]:
                    coords.extend([self._map_x((cc + 0.5) * self.cell_size), (rr + 0.5) * self.cell_size])
                if len(coords) >= 4:
                    self.canvas.create_line(*coords, fill="#f4d03f", width=2, dash=(3, 4))
            rr, cc = path[end_index]
            cx = self._map_x((cc + 0.5) * self.cell_size)
            cy = (rr + 0.5) * self.cell_size
            self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4, fill="#fff4a3", outline="#d4ac0d")

        # 飛翔中リコン。フラッシュ同様に点線で進行方向を示す。
        for projectile in self.recon_projectiles:
            path = projectile["path"]
            end_index = min(projectile["progress"], len(path) - 1)
            if end_index > 0:
                coords = []
                for rr, cc in path[:end_index + 1]:
                    coords.extend([self._map_x((cc + 0.5) * self.cell_size), (rr + 0.5) * self.cell_size])
                if len(coords) >= 4:
                    self.canvas.create_line(*coords, fill="#65d8e8", width=2, dash=(3, 4))
            rr, cc = path[end_index]
            cx = self._map_x((cc + 0.5) * self.cell_size)
            cy = (rr + 0.5) * self.cell_size
            self.canvas.create_polygon(cx-7, cy+3, cx+5, cy-5, cx+8, cy-2, cx-4, cy+6,
                                       fill="#9eeaf4", outline="#2aa9bd")

        for burst in self.recon_bursts:
            for rr, cc in burst["cells"]:
                x1 = self._map_x(cc * self.cell_size)
                y1 = rr * self.cell_size
                self.canvas.create_rectangle(x1, y1, x1+self.cell_size, y1+self.cell_size,
                                             fill="#6ed7e8", outline="", stipple="gray50")

        for burst in self.flash_bursts:
            rr, cc = burst["pos"]
            cx = self._map_x((cc + 0.5) * self.cell_size)
            cy = (rr + 0.5) * self.cell_size
            radius = self.cell_size * 0.65
            self.canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                                    fill="#fff7bf", outline="#f1c40f", width=2, stipple="gray50")

        selected_names = {ctrl.selected_char for ctrl, _ in self.get_user_controllers() if ctrl.selected_char is not None}
        viewer_team = self.get_viewer_team()
        visible_chars = [
            c for c in self.chars
            if self.is_visible_to_team(c, viewer_team) and (c.is_alive or c.just_died)
        ]
        for char in visible_chars:
            row, col = char.pos
            x1 = self._map_x(col*self.cell_size)
            cx, cy = x1 + self.cell_size/2, (row+0.5)*self.cell_size
            if char.just_died:
                self.canvas.create_text(cx, cy, text="X", fill="orange", font=("Arial", 12, "bold"))
                char.just_died = False
                continue

            bg = "#2980b9" if (char.defuse_timer > 0 and self.is_planted) else char.bg_color
            outline_color = "yellow" if char.name in selected_names else ""
            outline_width = 3 if char.name in selected_names else 1
            self.canvas.create_oval(x1, row*self.cell_size, x1+self.cell_size, (row+1)*self.cell_size, fill=bg, outline=outline_color, width=outline_width)
            if char.blind_remaining > 0:
                # 視認性を壊さない薄い二重リングと小さな印でブラインド状態を表示。
                self.canvas.create_oval(x1+2, row*self.cell_size+2, x1+self.cell_size-2, (row+1)*self.cell_size-2,
                                        outline="#fff2a8", width=2, dash=(2, 2))
                self.canvas.create_text(cx, cy, text="✦", fill="#fff7c2", font=("Arial", 9, "bold"))
            if self._is_revealed(char):
                self.canvas.create_rectangle(x1+3, row*self.cell_size+3, x1+self.cell_size-3, (row+1)*self.cell_size-3,
                                             outline="#7de3f2", width=2, dash=(4, 2))
                self.canvas.create_text(cx, cy+7, text="◇", fill="#b8f4fb", font=("Arial", 8, "bold"))
            if char.has_spike:
                self.canvas.create_oval(x1+3, row*self.cell_size+3, x1+self.cell_size-3, (row+1)*self.cell_size-3, fill="black", outline="")

            # キャラクター上部の名前・HPパネル
            panel_w = max(56, min(150, 16 + len(char.display_name) * 7))
            panel_h = 23
            px1 = cx - panel_w / 2
            py2 = row*self.cell_size - 3
            py1 = py2 - panel_h
            has_adjacent = any(
                other is not char and other.is_alive
                and max(abs(other.pos[0] - char.pos[0]), abs(other.pos[1] - char.pos[1])) <= 1
                for other in visible_chars
            )
            # 隣接時は札を半透明風(stipple)にして重なりの圧迫感を減らす。
            self.canvas.create_rectangle(px1, py1, px1+panel_w, py2, fill="#101820", outline=char.bg_color,
                                         width=1, stipple="gray50" if has_adjacent else "")
            name_color = "#d6d9de" if has_adjacent else ("yellow" if char.has_spike else "white")
            self.canvas.create_text(cx, py1+8, text=char.display_name, fill=name_color, font=("Arial", 8, "bold"))
            hp_ratio = char.hp / char.max_hp
            self.canvas.create_rectangle(px1+4, py2-7, px1+panel_w-4, py2-3, fill="#3a404a", outline="")
            self.canvas.create_rectangle(px1+4, py2-7, px1+4+(panel_w-8)*hp_ratio, py2-3, fill=char.bg_color, outline="")

        # アビリティ専用の下部領域。マップとは完全に分離する。
        bottom_y = self.map_pixel_height
        self.canvas.create_rectangle(
            self.map_offset_x, bottom_y,
            self.map_offset_x + self.map_pixel_width, bottom_y + self.ability_area_height,
            fill="#0b1018", outline="#2a3444", width=2
        )
        self.canvas.create_text(
            self.map_offset_x + 18, bottom_y + 18,
            text="ABILITIES", anchor="w", fill="#8b98a9", font=("Arial", 10, "bold")
        )

        bounds = self._ability_panel_bounds()
        selected = self._selected_user_character()
        if bounds and selected:
            ability_name = selected.ability_name
            x1, y1, x2, y2 = bounds
            accent = {"SMOKE": "#e67e22", "FLASH": "#f1c40f", "RECON": "#65d8e8", "HUNT": "#e74c3c"}[ability_name]
            charges = {
                "SMOKE": selected.smoke_charges,
                "FLASH": selected.flash_charges,
                "RECON": selected.recon_charges,
                "HUNT": 1,
            }[ability_name]
            armed = self.ability_mode == (ability_name, selected.team, selected.name)
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#151c27",
                                         outline=accent if armed or ability_name == "HUNT" else "#536273", width=2)
            icon_cx, icon_cy = x1 + 42, (y1 + y2) / 2
            self._draw_compact_ability_icon(ability_name, icon_cx, icon_cy, charges > 0)

            text_cx = (x1 + 70 + x2) / 2
            if ability_name == "HUNT":
                state = "HUNT / ハンター（常時発動）"
                help_text = "Hit% +10ポイント・HS% +5ポイント"
            else:
                label = {"SMOKE": "SMOKE", "FLASH": "FLASH", "RECON": "RECON"}[ability_name]
                state = "構え中：再クリックでキャンセル" if armed else f"{label}  残り {charges}"
                help_text = ("方向を指定" if armed and ability_name in ("FLASH", "RECON")
                             else ("マスを選択" if armed else f"クリックして{label}を構える"))
            self.canvas.create_text(text_cx, y1+22, text=state,
                                    fill=accent if charges else "#777", font=("Arial", 10, "bold"))
            self.canvas.create_text(text_cx, y1+48, text=help_text, fill="white", font=("Arial", 9))

            plant_bounds = self._plant_button_bounds()
            if plant_bounds:
                px1, py1, px2, py2 = plant_bounds
                planting = selected.is_planting
                self.canvas.create_rectangle(px1, py1, px2, py2, fill="#2a1d0d",
                                             outline="#f39c12" if planting else "#8a6a32", width=2)
                self.canvas.create_text((px1+px2)/2, py1+22,
                                        text="PLANTING..." if planting else "PLANT",
                                        fill="#ffd27a", font=("Arial", 11, "bold"))
                progress = min(1.0, selected.plant_timer / max(0.001, PLANT_REQUIRED_SECONDS))
                self.canvas.create_rectangle(px1+14, py2-20, px2-14, py2-12, fill="#4b3a22", outline="")
                self.canvas.create_rectangle(px1+14, py2-20, px1+14+(px2-px1-28)*progress, py2-12,
                                             fill="#f39c12", outline="")

        # 既存ゲーム画面を下へずらし、告知がマップへ重ならない専用領域を確保する。
        self.canvas.move("all", 0, COMBO_BANNER_HEIGHT)
        self._draw_combo_announcement_banner()


    def _draw_combo_announcement_banner(self):
        """コンボと覚醒イベントを共通の上部パネルへ描画する。"""
        total_w = self.map_pixel_width + SIDE_PANEL_WIDTH * 2
        self.canvas.create_rectangle(0, 0, total_w, COMBO_BANNER_HEIGHT, fill="#090d14", outline="#2a3444", width=2)

        queue = getattr(self, "announcement_queue", [])
        if self.combo_announcement_index >= len(queue):
            self.canvas.create_text(
                total_w / 2, COMBO_BANNER_HEIGHT / 2,
                text=f"ROUND {self.current_round}", fill="#768394",
                font=("Arial", 14, "bold")
            )
            return

        announcement = queue[self.combo_announcement_index]
        is_awakening = announcement.get("type") == "awakening"
        team = announcement.get("team")
        accent = "#c0392b" if team == "A" else "#27ae60"
        team_text = "ATTACKERS" if team == "A" else "DEFENDERS"
        category_text = "覚醒イベント" if is_awakening else "プレイヤーコンボ"
        title_color = "#ff8f70" if is_awakening else "#ffd66b"

        self.canvas.create_rectangle(12, 10, total_w-12, COMBO_BANNER_HEIGHT-10, fill="#151c27", outline=accent, width=3)
        self.canvas.create_text(
            30, 25, text=category_text, anchor="w",
            fill=accent, font=("Arial", 10, "bold")
        )
        self.canvas.create_text(
            total_w/2, 25, text=announcement.get("name", "名称未設定"),
            fill=title_color, font=("Arial", 17, "bold")
        )
        names = " × ".join(announcement.get("display_players", announcement.get("players", ())))
        self.canvas.create_text(
            total_w/2, 51, text=f"{team_text}  |  {names}",
            fill="white", font=("Arial", 11, "bold")
        )
        self.canvas.create_text(
            total_w/2, 73, text=announcement.get("effect_text", "特殊効果"),
            fill="#b9c6d8", font=("Arial", 10)
        )

