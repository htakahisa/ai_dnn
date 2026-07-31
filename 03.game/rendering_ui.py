"""Canvas rendering and mouse input handling."""

from controllers import UserInputController


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
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if not (0 <= r < self.height and 0 <= c < self.width):
            return

        user_controllers = self.get_user_controllers()
        if not user_controllers:
            return

        for ctrl, team in user_controllers:
            ctrl.handle_click(r, c, self.grid, self.chars, team)

        self.draw()

    def on_canvas_right_click(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if not (0 <= r < self.height and 0 <= c < self.width):
            return

        user_controllers = self.get_user_controllers()
        if not user_controllers:
            return

        for ctrl, team in user_controllers:
            if hasattr(ctrl, "handle_right_click"):
                ctrl.handle_right_click(r, c, self.grid, self.chars, team)

        self.draw()

    def draw(self):
        if self.headless: return  # 描画スキップ
        self.canvas.delete("all")
        color_map = {"0":"white", "1":"#34495e", "2":"#fff9c4", "3":"#ffcccc", "4":"#ccffcc"}

        for r in range(self.height):
            for c in range(self.width):
                color = color_map.get(str(self.grid[r, c]), "white")
                self.canvas.create_rectangle(c*self.cell_size, r*self.cell_size, (c+1)*self.cell_size, (r+1)*self.cell_size, fill=color, outline="#eee")

        # -----------------------------------------------------------------
        # 💡 [追加] アビリティの描画処理
        # -----------------------------------------------------------------
        for smoke in self.smokes:
            for r, c in smoke["cells"]:
                self.canvas.create_rectangle(
                    c * self.cell_size, r * self.cell_size,
                    (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                    fill="#7f8c8d", outline="#95a5a6", stipple="gray50"
                )

        for burst in self.recon_bursts:
            for r, c in burst["cells"]:
                self.canvas.create_rectangle(
                    c * self.cell_size, r * self.cell_size,
                    (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                    fill="#3498db", outline="#2980b9", stipple="gray25"
                )

        for burst in self.flash_bursts:
            r, c = burst["pos"]
            cx, cy = (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size
            rad = self.cell_size * 0.8
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="#f1c40f", outline="#f39c12")

        for p in self.flash_projectiles:
            if p["path"] and p["progress"] < len(p["path"]):
                r, c = p["path"][p["progress"]]
                cx, cy = (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size
                rad = self.cell_size * 0.3
                self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="#f1c40f", outline="black")

        for p in self.recon_projectiles:
            if p["path"] and p["progress"] < len(p["path"]):
                r, c = p["path"][p["progress"]]
                cx, cy = (c + 0.5) * self.cell_size, (r + 0.5) * self.cell_size
                rad = self.cell_size * 0.3
                self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="#2980b9", outline="black")
        # -----------------------------------------------------------------

        if not self.is_planted and self.target_plant_pos:
            tr, tc = self.target_plant_pos
            self.canvas.create_rectangle(tc*self.cell_size, tr*self.cell_size, (tc+1)*self.cell_size, (tr+1)*self.cell_size, fill="#f39c12", outline="#d35400")

        if self.is_planted and self.planted_pos:
            pr, pc = self.planted_pos
            self.canvas.create_oval(pc*self.cell_size+2, pr*self.cell_size+2, (pc+1)*self.cell_size-2, (pr+1)*self.cell_size-2, fill="red", outline="")

        if self.spike_pos:
            sr, sc = self.spike_pos
            self.canvas.create_oval(sc*self.cell_size+2, sr*self.cell_size+2, (sc+1)*self.cell_size-2, (sr+1)*self.cell_size-2, fill="black", outline="")

        for c1, c2 in self.last_engagements:
            self.canvas.create_line((c1.pos[1]+0.5)*self.cell_size, (c1.pos[0]+0.5)*self.cell_size,
                                    (c2.pos[1]+0.5)*self.cell_size, (c2.pos[0]+0.5)*self.cell_size, fill="red", width=1)

        selected_names = {ctrl.selected_char for ctrl, _ in self.get_user_controllers() if ctrl.selected_char is not None}

        for c in self.chars:
            if not c.is_alive and not c.just_died: continue
            row, col = c.pos
            cx, cy = (col+0.5)*self.cell_size, (row+0.5)*self.cell_size
            if c.just_died:
                self.canvas.create_text(cx, cy, text="X", fill="orange", font=("Arial", 12, "bold"))
                c.just_died = False
            else:
                bg = "#2980b9" if (getattr(c, 'defuse_timer', 0) > 0 and self.is_planted) else c.bg_color
                outline_color = "yellow" if c.name in selected_names else ""
                outline_width = 3 if c.name in selected_names else 1
                self.canvas.create_oval(col*self.cell_size, row*self.cell_size, (col+1)*self.cell_size, (row+1)*self.cell_size, fill=bg, outline=outline_color, width=outline_width)

                if c.has_spike:
                    self.canvas.create_oval(col*self.cell_size+1, row*self.cell_size+1, (col+1)*self.cell_size-1, (row+1)*self.cell_size-1, fill="black", outline="")
                    self.canvas.create_text(cx, cy, text=c.name, fill="yellow", font=("Arial", 6, "bold"))
                else:
                    self.canvas.create_text(cx, cy, text=c.name, fill=c.text_color, font=("Arial", 6, "bold"))