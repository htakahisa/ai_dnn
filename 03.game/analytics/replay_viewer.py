"""Small state-based replay viewer for recorded competition matches."""

from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk

# analytics/ から起動しても、プロジェクトルートの map_data.py を
# 解決できるようにする。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from map_data import NEW_MAZE_STR
from ability_effects import (draw_raid_wind, draw_heal_sparkle, draw_destruction_areas,
                            draw_contract_status, draw_serenade_flash,
                            draw_ash_projectiles, draw_balemoon_warnings,
                            TUNNEL_WARNING_COLOR, TUNNEL_ACTIVE_COLOR)
from game_core import FACING_VECTORS, FACING_INDICATOR_LENGTH_RATIO
from public_effects import displayed_projectile_path, displayed_area_cells


class ReplayViewer(tk.Toplevel):
    def __init__(
        self,
        parent,
        replay_frames,
        *,
        title="Match Replay",
        map_options=None,
        round_records=None,
    ):
        super().__init__(parent)
        self.title(title)
        self.geometry("980x720")
        self.frames = list(replay_frames or [])
        self.round_records = list(round_records or [])
        self.map_options = list(map_options or [])
        self.index = 0
        self.playing = False
        self._updating_timeline = False
        self.speed = tk.DoubleVar(value=1.0)
        self.status = tk.StringVar(value="")
        self.view_mode = tk.StringVar(value="ALL")
        initial_round = self.frames[0].get("round", 1) if self.frames else 1
        self.round_var = tk.StringVar(value=str(initial_round))

        rows = [
            line.strip() for line in NEW_MAZE_STR.strip().splitlines() if line.strip()
        ]
        self.grid = [list(map(int, row)) for row in rows]
        self.cell = 18

        self.canvas = tk.Canvas(self, bg="#10141c", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        controls = ttk.Frame(self, padding=(8, 0, 8, 8))
        controls.pack(fill=tk.X)
        toolbar = ttk.Frame(controls)
        toolbar.pack(fill=tk.X)
        timeline_row = ttk.Frame(controls)
        timeline_row.pack(fill=tk.X, pady=(4, 0))
        if len(self.map_options) > 1:
            ttk.Label(toolbar, text="Map").pack(side=tk.LEFT, padx=(0, 4))
            self.map_var = tk.StringVar()
            self.map_combo = ttk.Combobox(
                toolbar,
                textvariable=self.map_var,
                values=[
                    f"Map {getattr(item, 'number', index + 1)}"
                    for index, item in enumerate(self.map_options)
                ],
                state="readonly",
                width=10,
            )
            self.map_combo.pack(side=tk.LEFT, padx=(0, 12))
            self.map_combo.current(0)
            self.map_combo.bind("<<ComboboxSelected>>", self._map_changed)
        self.play_button = ttk.Button(toolbar, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Round").pack(side=tk.LEFT, padx=(8, 3))
        self.round_entry = ttk.Entry(toolbar, textvariable=self.round_var, width=5)
        self.round_entry.pack(side=tk.LEFT)
        self.round_entry.bind("<Return>", lambda _event: self.toggle_play())
        ttk.Label(toolbar, text="View").pack(side=tk.LEFT, padx=(12, 4))
        for label, mode in (("All", "ALL"), ("Attackers", "A"), ("Defenders", "D")):
            ttk.Button(
                toolbar,
                text=label,
                command=lambda selected=mode: self.set_view_mode(selected),
            ).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="|<", command=lambda: self.seek(0)).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(toolbar, text="Round -", command=lambda: self.seek_round(-1)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(toolbar, text="Round +", command=lambda: self.seek_round(1)).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Label(toolbar, text="Speed").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(
            toolbar,
            textvariable=self.speed,
            values=(0.25, 0.5, 1.0, 2.0, 4.0),
            width=5,
            state="readonly",
        ).pack(side=tk.LEFT)
        self.timeline_marks = tk.Canvas(
            timeline_row,
            height=58,
            highlightthickness=0,
            bg="#e5e7eb",
        )
        self.timeline_marks.pack(fill=tk.X, expand=True)
        self.timeline_marks.bind("<Button-1>", self._timeline_clicked)
        self.timeline_marks.bind(
            "<Configure>", lambda _event: self._draw_timeline_marks()
        )
        self.timeline = ttk.Scale(
            timeline_row,
            from_=0,
            to=max(0, len(self.frames) - 1),
            orient=tk.HORIZONTAL,
            command=self._scale_changed,
        )
        self.timeline.pack(fill=tk.X, expand=True, padx=(0, 12))
        ttk.Label(toolbar, textvariable=self.status, width=34).pack(side=tk.RIGHT)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._draw_timeline_marks()
        self._draw_current_round_marker()
        self.draw_frame()

    def _round_starts(self):
        starts = {}
        for index, frame in enumerate(self.frames):
            round_number = frame.get("round")
            if round_number is not None and round_number not in starts:
                starts[round_number] = index
        return starts

    def _round_specials(self):
        return {
            int(record.get("round_number")): str(record.get("special", "")).upper()
            for record in self.round_records
            if record.get("round_number") is not None
        }

    def _team_round_winners(self):
        """Return round winners using the replay's team names when available."""
        if not self.map_options:
            return {}
        selected = getattr(self, "map_combo", None)
        option_index = selected.current() if selected is not None else 0
        if not 0 <= option_index < len(self.map_options):
            option_index = 0
        match_map = self.map_options[option_index]
        team1 = str(getattr(match_map, "team1", "Team A"))
        team2 = str(getattr(match_map, "team2", "Team B"))
        winners = {}
        for record in getattr(match_map, "round_records", self.round_records):
            number = record.get("round_number")
            winner = str(record.get("winner", "")).lower()
            attacker = record.get("attacker_team")
            if number is None:
                continue
            if winner == "attacker" and attacker:
                winners[int(number)] = str(attacker)
            elif winner == "defender" and attacker:
                winners[int(number)] = team2 if str(attacker) == team1 else team1
            elif winner in {team1.lower(), team2.lower()}:
                winners[int(number)] = team1 if winner == team1.lower() else team2
        if not winners:
            # Some short/manual replays omit round_records. Their frames still
            # contain cumulative attacker/defender round scores.
            previous = {"attacker": 0, "defender": 0}
            initial_attacker = str(getattr(match_map, "initial_attacker", team1))
            grouped = {}
            for frame in self.frames:
                number = frame.get("round")
                if number is not None:
                    grouped[number] = frame
            for number in sorted(grouped):
                frame = grouped[number]
                attack_score = int(frame.get("attacker_wins", 0) or 0)
                defend_score = int(frame.get("defender_wins", 0) or 0)
                attacking_team = initial_attacker if int(number) <= 12 else (
                    team2 if initial_attacker == team1 else team1
                )
                if attack_score > previous["attacker"]:
                    winners[int(number)] = attacking_team
                elif defend_score > previous["defender"]:
                    winners[int(number)] = team2 if attacking_team == team1 else team1
                previous = {"attacker": attack_score, "defender": defend_score}
        return {"teams": (team1, team2), "winners": winners}

    def _draw_timeline_marks(self):
        self.timeline_marks.delete("all")
        if not self.frames:
            return
        self.timeline_marks.update_idletasks()
        width = max(1, self.timeline_marks.winfo_width())
        total = max(1, len(self.frames) - 1)
        starts = self._round_starts()
        rounds = sorted(starts.items())
        specials = self._round_specials()
        scoreboard = self._team_round_winners()
        teams = scoreboard.get("teams", ("Team A", "Team B"))
        winners = scoreboard.get("winners", {})
        left_pad = 86
        right_pad = 8
        usable_width = max(1, width - left_pad - right_pad)
        round_centers = {}
        for round_number, start in rounds:
            round_centers[round_number] = left_pad + (
                start / total * usable_width
            )
        for row, team in enumerate(teams):
            y = 27 + row * 23
            label = str(team)
            if len(label) > 12:
                label = label[:11] + "…"
            self.timeline_marks.create_text(
                left_pad - 7, y, text=label, anchor="e", fill="#111827",
                font=("Arial", 8, "bold"),
            )
        for offset, (round_number, start) in enumerate(rounds):
            end = (
                rounds[offset + 1][1] if offset + 1 < len(rounds) else len(self.frames)
            )
            left = left_pad + start / total * usable_width
            right = left_pad + max(start + 1, end - 1) / total * usable_width
            special = specials.get(int(round_number), "")
            color = (
                "#facc15"
                if special == "ACE"
                else "#ef4444" if special == "CLUTCH" else "#94a3b8"
            )
            self.timeline_marks.create_rectangle(
                left, 2, right, 20, fill=color, outline=""
            )
            self.timeline_marks.create_text(
                (left + right) / 2,
                7,
                text=str(round_number),
                fill="#111827",
                font=("Arial", 8, "bold"),
            )
            winner = winners.get(int(round_number))
            for row, team in enumerate(teams):
                y = 27 + row * 23
                if winner is None:
                    marker, color = "–", "#9ca3af"
                elif winner == team:
                    marker, color = "●", "#16a34a"
                else:
                    marker, color = "–", "#9ca3af"
                self.timeline_marks.create_text(
                    round_centers[round_number], y, text=marker, fill=color,
                    font=("Arial", 13, "bold"),
                )
        self._draw_current_round_marker()

    def _draw_current_round_marker(self):
        if not hasattr(self, "timeline_marks"):
            return
        self.timeline_marks.delete("current_round_marker")
        if not self.frames:
            return
        starts = self._round_starts()
        current_round = self.frames[self.index].get("round")
        if current_round not in starts:
            return
        width = max(1, self.timeline_marks.winfo_width())
        total = max(1, len(self.frames) - 1)
        x = 86 + starts[current_round] / total * max(1, width - 94)
        self.timeline_marks.create_text(
            x, 17, text="▼", fill="#111827", font=("Arial", 9, "bold"),
            tags=("current_round_marker",),
        )

    def _timeline_clicked(self, event):
        if not self.frames:
            return
        width = max(1, self.timeline_marks.winfo_width())
        usable_width = max(1, width - 94)
        position = max(0, min(1, (event.x - 86) / usable_width))
        self.seek(position * max(0, len(self.frames) - 1))

    def seek_round(self, delta):
        starts = self._round_starts()
        if not starts:
            return
        rounds = sorted(starts)
        current_round = self.frames[self.index].get("round")
        try:
            current_index = rounds.index(current_round)
        except ValueError:
            current_index = 0
        target_index = max(0, min(len(rounds) - 1, current_index + int(delta)))
        self.seek(starts[rounds[target_index]])

    def _map_changed(self, _event=None):
        selected = self.map_combo.current()
        if not (0 <= selected < len(self.map_options)):
            return
        self.playing = False
        self.play_button.configure(text="Play")
        self.frames = list(
            getattr(self.map_options[selected], "replay_frames", []) or []
        )
        self.round_records = list(
            getattr(self.map_options[selected], "round_records", []) or []
        )
        self.index = 0
        if self.frames:
            self.round_var.set(str(self.frames[0].get("round", 1)))
        else:
            self.round_var.set("")
        self.timeline.configure(to=max(0, len(self.frames) - 1))
        self._updating_timeline = True
        try:
            self.timeline.set(0)
        finally:
            self._updating_timeline = False
        self._draw_timeline_marks()
        self._draw_current_round_marker()
        self.draw_frame()

    def close(self):
        self.playing = False
        self.destroy()

    def toggle_play(self):
        if not self.frames:
            return
        if not self.playing:
            try:
                requested_round = int(self.round_var.get().strip())
            except ValueError:
                self.status.set("Enter a valid round number")
                return
            round_start = self._round_starts().get(requested_round)
            if round_start is None:
                self.status.set(f"Round {requested_round} is not in this replay")
                return
            self.seek(round_start)
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.after(1, self.advance)

    def advance(self):
        if not self.playing or not self.winfo_exists():
            return
        if self.index >= len(self.frames) - 1:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.index += 1
        self.timeline.set(self.index)
        self.draw_frame()
        delay = max(10, int(100 / max(0.25, float(self.speed.get()))))
        self.after(delay, self.advance)

    def seek(self, value):
        if not self.frames:
            return
        self.index = max(0, min(len(self.frames) - 1, int(value)))
        self._updating_timeline = True
        try:
            self.timeline.set(self.index)
        finally:
            self._updating_timeline = False
        self.draw_frame()

    def _scale_changed(self, value):
        if not self.playing and not self._updating_timeline:
            self.seek(float(value))

    def set_view_mode(self, mode):
        mode = str(mode).upper()
        if mode not in {"ALL", "A", "D"}:
            return
        self.view_mode.set(mode)
        self.draw_frame()

    def _char_visible(self, char):
        mode = self.view_mode.get()
        if mode == "ALL" or char.get("team") == mode:
            return True
        visible_to = char.get("visible_to")
        if isinstance(visible_to, (list, tuple, set)):
            return mode in visible_to
        # Old replay files only have the historical global reveal flag. It is
        # not team-perfect, but remains a useful backwards-compatible view.
        return bool(char.get("revealed", False))

    def _draw_neon_bolt(self, row, col):
        x, y, size = col*self.cell, row*self.cell, self.cell
        points = (x+size*.60, y+size*.08, x+size*.35, y+size*.43,
                  x+size*.65, y+size*.48, x+size*.40, y+size*.92)
        self.canvas.create_line(*points, fill="#218dff", width=5)
        self.canvas.create_line(*points, fill="#e5fcff", width=2)

    def draw_frame(self):
        self.canvas.delete("all")
        frame = self.frames[self.index] if self.frames else {}
        if frame.get("round") is not None and hasattr(self, "round_var"):
            self.round_var.set(str(frame["round"]))
        for r, row in enumerate(self.grid):
            for c, value in enumerate(row):
                if value == 1:
                    fill = "#34495e"
                elif value == 2:
                    # Keep the original game's yellow plant-position overlay.
                    fill = "#fff9c4"
                else:
                    fill = "#f8fafc"
                self.canvas.create_rectangle(
                    c * self.cell,
                    r * self.cell,
                    (c + 1) * self.cell,
                    (r + 1) * self.cell,
                    fill=fill,
                    outline="#cbd5e1",
                )
        target = frame.get("target_plant_pos")
        if target:
            r, c = target
            self.canvas.create_rectangle(
                c * self.cell + 2,
                r * self.cell + 2,
                (c + 1) * self.cell - 2,
                (r + 1) * self.cell - 2,
                outline="#eab308",
                width=2,
            )
        for r, c in frame.get("available_orbs", []):
            cx, cy = (c + 0.5) * self.cell, (r + 0.5) * self.cell
            radius = self.cell * 0.34
            self.canvas.create_polygon(
                cx, cy - radius, cx + radius, cy, cx, cy + radius,
                cx - radius, cy, fill="#b06cff", outline="#5b2c83", width=2,
            )
        for smoke in frame.get("smokes", []):
            if (int(smoke.get("remaining_ticks", 0)) <= 3
                    and int(frame.get("tick", 0)) % 2 == 0):
                continue
            for r, c in smoke.get("cells", []):
                self.canvas.create_oval(
                    c * self.cell + 1,
                    r * self.cell + 1,
                    (c + 1) * self.cell - 1,
                    (r + 1) * self.cell - 1,
                    fill="#d97706",
                    outline="#f59e0b",
                )
        for burst in frame.get("recon_bursts", []):
            for r, c in displayed_area_cells(burst):
                self.canvas.create_rectangle(
                    c * self.cell,
                    r * self.cell,
                    (c + 1) * self.cell,
                    (r + 1) * self.cell,
                    fill="#6ed7e8",
                    outline="",
                    stipple="gray50",
                )
        for burst in frame.get("flash_bursts", []):
            if burst.get("pos"):
                r, c = burst["pos"]
                self.canvas.create_oval(
                    c * self.cell + self.cell * 0.175,
                    r * self.cell + self.cell * 0.175,
                    c * self.cell + self.cell * 0.825,
                    r * self.cell + self.cell * 0.825,
                    fill="#fff7bf",
                    outline="#f1c40f",
                    width=2,
                    stipple="gray50",
                )
        for kind, projectiles, color, outline, trail_color in (
            ("flash", frame.get("flash_projectiles", []), "#fff4a3", "#d4ac0d", "#f4d03f"),
            ("recon", frame.get("recon_projectiles", []), "#9eeaf4", "#2aa9bd", "#65d8e8"),
        ):
            for projectile in projectiles:
                path = displayed_projectile_path(projectile)
                if not path:
                    continue
                if len(path) > 1:
                    coords = [
                        coordinate
                        for r, c in path
                        for coordinate in ((c + 0.5) * self.cell, (r + 0.5) * self.cell)
                    ]
                    self.canvas.create_line(
                        *coords, fill=trail_color,
                        width=2, dash=(3, 4),
                    )
                r, c = path[-1]
                cx, cy = (c + 0.5) * self.cell, (r + 0.5) * self.cell
                if kind == "recon":
                    self.canvas.create_polygon(
                        cx - 7, cy + 3, cx + 5, cy - 5, cx + 8, cy - 2,
                        cx - 4, cy + 6, fill=color, outline=outline,
                    )
                else:
                    self.canvas.create_oval(
                        cx - 4, cy - 4, cx + 4, cy + 4,
                        fill=color, outline=outline,
                    )
        for marker, color in (
            (frame.get("spike_pos"), "#111827"),
            (frame.get("planted_pos"), "#7c2d12"),
        ):
            if marker:
                r, c = marker
                self.canvas.create_polygon(
                    c * self.cell + self.cell // 2,
                    r * self.cell + 3,
                    c * self.cell + self.cell - 3,
                    r * self.cell + self.cell - 4,
                    c * self.cell + 3,
                    r * self.cell + self.cell - 4,
                    fill=color,
                    outline="#fbbf24",
                )
        for portal in frame.get("escape_portals", []):
            if not portal.get("pos"):
                continue
            r, c = portal["pos"]
            self.canvas.create_oval(
                c * self.cell + 1,
                r * self.cell + 1,
                (c + 1) * self.cell - 1,
                (r + 1) * self.cell - 1,
                fill="#dff8ff",
                outline="#38bdf8",
                width=2,
            )
            self.canvas.create_oval(
                c * self.cell + 6,
                r * self.cell + 6,
                (c + 1) * self.cell - 6,
                (r + 1) * self.cell - 6,
                fill="#f8fdff",
                outline="#bae6fd",
            )
        for burst in frame.get("tunnel_bursts", []):
            warning = burst.get("phase") == "warning"
            for row, col in burst.get("cells", []):
                self.canvas.create_rectangle(col*self.cell, row*self.cell, (col+1)*self.cell, (row+1)*self.cell,
                    fill=TUNNEL_WARNING_COLOR if warning else TUNNEL_ACTIVE_COLOR,
                    outline="", stipple="gray25" if warning else "gray50")
        draw_destruction_areas(self.canvas, frame.get("destruction_areas", []), self.cell)
        draw_balemoon_warnings(self.canvas, frame.get("balemoon_warnings", []), self.cell)
        draw_ash_projectiles(self.canvas, frame.get("ash_projectiles", []), self.cell)
        for trail in frame.get("ultimate_trails", []):
            draw_raid_wind(self.canvas, trail, self.cell)
        for trap in frame.get("ramp_traps", []):
            if self.view_mode.get() in ("A", "D") and trap.get("team") != self.view_mode.get():
                continue
            r, c = trap["pos"]
            self.canvas.create_oval(c*self.cell+4, r*self.cell+4,
                (c+1)*self.cell-4, (r+1)*self.cell-4,
                fill="#1773d1", outline="#7edcff", width=2)
        for burst in frame.get("neon_bursts", []):
            warning = burst.get("phase") == "warning"
            for r, c in burst.get("cells", []):
                self.canvas.create_rectangle(c*self.cell, r*self.cell,
                    (c+1)*self.cell, (r+1)*self.cell,
                    fill="#b5e5ff" if warning else "#167eff", outline="#9ddcff", stipple="gray25")
                if not warning:
                    self._draw_neon_bolt(r, c)
        for drone in frame.get("monitor_drones", []):
            r, c = drone.get("pos", (0, 0))
            cx, cy = (c + 0.5) * self.cell, (r + 0.5) * self.cell
            radius = self.cell * 0.33
            self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="#58d3f7", outline="#12394a", width=2,
            )
            self.canvas.create_text(
                cx, cy, text=str(int(drone.get("hp", 0))), fill="#08202a",
                font=("Arial", 6, "bold"),
            )
        for char in frame.get("chars", []):
            if not self._char_visible(char):
                continue
            r, c = char.get("pos") or (0, 0)
            if not char.get("alive", False):
                self.canvas.create_text(
                    (c + 0.5) * self.cell,
                    (r + 0.5) * self.cell,
                    text="X",
                    fill="#ef4444",
                    font=("Arial", 12, "bold"),
                )
                continue
            color = "#ef4444" if char.get("team") == "A" else "#22c55e"
            self.canvas.create_oval(
                c * self.cell + 2,
                r * self.cell + 2,
                (c + 1) * self.cell - 2,
                (r + 1) * self.cell - 2,
                fill=color,
                outline="#111827",
                width=1,
            )
            if char.get("blind", 0) > 0:
                self.canvas.create_oval(
                    c * self.cell + 2, r * self.cell + 2,
                    (c + 1) * self.cell - 2, (r + 1) * self.cell - 2,
                    outline="#fff2a8", width=2, dash=(2, 2),
                )
                self.canvas.create_text(
                    (c + 0.5) * self.cell, (r + 0.5) * self.cell,
                    text="F", fill="#fff7c2", font=("Arial", 8, "bold"),
                )
            if char.get("revealed", False):
                self.canvas.create_rectangle(
                    c * self.cell + 3, r * self.cell + 3,
                    (c + 1) * self.cell - 3, (r + 1) * self.cell - 3,
                    outline="#7de3f2", width=2, dash=(4, 2),
                )
                self.canvas.create_text(
                    (c + 0.5) * self.cell, (r + 0.5) * self.cell + 6,
                    text="R", fill="#b8f4fb", font=("Arial", 7, "bold"),
                )
            if char.get("ability_seal", 0) > 0:
                self.canvas.create_text(
                    (c + 0.5) * self.cell, (r + 0.5) * self.cell - 6,
                    text="LOCK", fill="#e4a6ff", font=("Arial", 5, "bold"),
                )
            if char.get("fate_loom_root", 0) > 0:
                self.canvas.create_text(
                    (c + 0.5) * self.cell, (r + 0.5) * self.cell + 6,
                    text="ROOT", fill="#ffcf80", font=("Arial", 5, "bold"),
                )
            if char.get("active_awakening"):
                cx, cy = (c + 0.5) * self.cell, (r + 0.5) * self.cell
                radius = self.cell * 0.4
                self.canvas.create_polygon(
                    cx, cy - radius, cx + radius * 0.45, cy - radius * 0.1,
                    cx + radius * 0.25, cy + radius, cx - radius * 0.3, cy + radius * 0.6,
                    cx - radius * 0.5, cy - radius * 0.1,
                    fill="#e8890c", outline="#ffd54f", width=1,
                )
            facing = FACING_VECTORS.get(char.get("facing"))
            if facing:
                center_x = (c + 0.5) * self.cell
                center_y = (r + 0.5) * self.cell
                length = self.cell * FACING_INDICATOR_LENGTH_RATIO
                self.canvas.create_line(
                    center_x,
                    center_y,
                    center_x + facing[0] * length,
                    center_y + facing[1] * length,
                    fill="#111827",
                    width=2,
                    arrow=tk.LAST,
                    arrowshape=(5, 6, 3),
                )
            if char.get("has_spike"):
                self.canvas.create_text(
                    (c + 0.5) * self.cell,
                    (r + 0.5) * self.cell,
                    text="S",
                    fill="white",
                    font=("Arial", 9, "bold"),
                )
            shield_hp = char.get("shield_hp", 0)
            max_shield_hp = char.get("max_shield_hp", 0)
            if shield_hp > 0 and max_shield_hp > 0:
                shield_ratio = min(1, shield_hp / max_shield_hp)
                self.canvas.create_rectangle(
                    c * self.cell + 2, (r + 1) * self.cell - 4,
                    c * self.cell + 2 + (self.cell - 4) * shield_ratio,
                    (r + 1) * self.cell - 2, fill="#66d9ff", outline=""
                )
            if char.get("electric", 0) > 0:
                self._draw_neon_bolt(r, c)
            if char.get("heal_sparkle", 0) > 0:
                draw_heal_sparkle(self.canvas, r, c, self.cell, char["heal_sparkle"])
            if char.get("life_contract", 0) > 0:
                draw_contract_status(self.canvas, r, c, self.cell, char["life_contract"])
            self.canvas.create_text(
                (c + 0.5) * self.cell,
                r * self.cell - 2,
                text=char.get("name", ""),
                anchor="s",
                fill="#111827",
                font=("Arial", 7),
            )
        if frame.get("serenade_flash", 0) > 0:
            draw_serenade_flash(self.canvas, len(self.grid[0]), len(self.grid), self.cell)
        self.status.set(
            f"Frame {self.index + 1}/{len(self.frames)}  "
            f"R{frame.get('round', '-')}/T{frame.get('tick', '-')}  "
            f"Score {frame.get('attacker_wins', 0)}-{frame.get('defender_wins', 0)}  "
            f"View {self.view_mode.get()}"
        )
        if hasattr(self, "timeline_marks"):
            self._draw_current_round_marker()
