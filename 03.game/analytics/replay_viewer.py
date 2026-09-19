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


class ReplayViewer(tk.Toplevel):
    def __init__(self, parent, replay_frames, *, title="Match Replay", map_options=None):
        super().__init__(parent)
        self.title(title)
        self.geometry("980x720")
        self.frames = list(replay_frames or [])
        self.map_options = list(map_options or [])
        self.index = 0
        self.playing = False
        self._updating_timeline = False
        self.speed = tk.DoubleVar(value=1.0)
        self.status = tk.StringVar(value="")
        self.view_mode = tk.StringVar(value="ALL")

        rows = [line.strip() for line in NEW_MAZE_STR.strip().splitlines() if line.strip()]
        self.grid = [list(map(int, row)) for row in rows]
        self.cell = 18

        self.canvas = tk.Canvas(self, bg="#10141c", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        controls = ttk.Frame(self, padding=(8, 0, 8, 8))
        controls.pack(fill=tk.X)
        if len(self.map_options) > 1:
            ttk.Label(controls, text="Map").pack(side=tk.LEFT, padx=(0, 4))
            self.map_var = tk.StringVar()
            self.map_combo = ttk.Combobox(
                controls,
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
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT)
        ttk.Label(controls, text="View").pack(side=tk.LEFT, padx=(12, 4))
        for label, mode in (("All", "ALL"), ("Attackers", "A"), ("Defenders", "D")):
            ttk.Button(
                controls, text=label,
                command=lambda selected=mode: self.set_view_mode(selected),
            ).pack(side=tk.LEFT, padx=1)
        ttk.Button(controls, text="|<", command=lambda: self.seek(0)).pack(side=tk.LEFT, padx=4)
        ttk.Label(controls, text="Speed").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(
            controls, textvariable=self.speed, values=(0.25, 0.5, 1.0, 2.0, 4.0),
            width=5, state="readonly",
        ).pack(side=tk.LEFT)
        self.timeline = ttk.Scale(
            controls, from_=0, to=max(0, len(self.frames) - 1),
            orient=tk.HORIZONTAL, command=self._scale_changed,
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)
        ttk.Label(controls, textvariable=self.status, width=34).pack(side=tk.RIGHT)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.draw_frame()

    def _map_changed(self, _event=None):
        selected = self.map_combo.current()
        if not (0 <= selected < len(self.map_options)):
            return
        self.playing = False
        self.play_button.configure(text="Play")
        self.frames = list(
            getattr(self.map_options[selected], "replay_frames", []) or []
        )
        self.index = 0
        self.timeline.configure(to=max(0, len(self.frames) - 1))
        self._updating_timeline = True
        try:
            self.timeline.set(0)
        finally:
            self._updating_timeline = False
        self.draw_frame()

    def close(self):
        self.playing = False
        self.destroy()

    def toggle_play(self):
        if not self.frames:
            return
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

    def draw_frame(self):
        self.canvas.delete("all")
        frame = self.frames[self.index] if self.frames else {}
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
                    c * self.cell, r * self.cell, (c + 1) * self.cell,
                    (r + 1) * self.cell, fill=fill, outline="#cbd5e1",
                )
        target = frame.get("target_plant_pos")
        if target:
            r, c = target
            self.canvas.create_rectangle(
                c * self.cell + 2, r * self.cell + 2,
                (c + 1) * self.cell - 2, (r + 1) * self.cell - 2,
                outline="#eab308", width=2,
            )
        for smoke in frame.get("smokes", []):
            for r, c in smoke.get("cells", []):
                self.canvas.create_oval(
                    c * self.cell + 1, r * self.cell + 1,
                    (c + 1) * self.cell - 1, (r + 1) * self.cell - 1,
                    fill="#d97706", outline="#f59e0b",
                )
        for burst in frame.get("recon_bursts", []):
            for r, c in burst.get("cells", []):
                self.canvas.create_rectangle(
                    c * self.cell + 4, r * self.cell + 4,
                    (c + 1) * self.cell - 4, (r + 1) * self.cell - 4,
                    fill="#38bdf8", outline="",
                )
        for burst in frame.get("flash_bursts", []):
            if burst.get("pos"):
                r, c = burst["pos"]
                self.canvas.create_oval(
                    c * self.cell - 3, r * self.cell - 3,
                    (c + 1) * self.cell + 3, (r + 1) * self.cell + 3,
                    fill="#fef08a", outline="#facc15", width=2,
                )
        for projectiles, color in (
            (frame.get("flash_projectiles", []), "#fde047"),
            (frame.get("recon_projectiles", []), "#67e8f9"),
        ):
            for projectile in projectiles:
                path = projectile.get("path", [])
                if not path:
                    continue
                progress = min(int(projectile.get("progress", 0)), len(path) - 1)
                r, c = path[progress]
                self.canvas.create_oval(
                    c * self.cell + 5, r * self.cell + 5,
                    (c + 1) * self.cell - 5, (r + 1) * self.cell - 5,
                    fill=color, outline="#111827",
                )
        for marker, color in ((frame.get("spike_pos"), "#111827"), (frame.get("planted_pos"), "#7c2d12")):
            if marker:
                r, c = marker
                self.canvas.create_polygon(
                    c * self.cell + self.cell // 2, r * self.cell + 3,
                    c * self.cell + self.cell - 3, r * self.cell + self.cell - 4,
                    c * self.cell + 3, r * self.cell + self.cell - 4,
                    fill=color, outline="#fbbf24",
                )
        for char in frame.get("chars", []):
            if not self._char_visible(char):
                continue
            r, c = char.get("pos") or (0, 0)
            if not char.get("alive", False):
                self.canvas.create_text(
                    (c + 0.5) * self.cell, (r + 0.5) * self.cell,
                    text="X", fill="#ef4444", font=("Arial", 12, "bold"),
                )
                continue
            color = "#ef4444" if char.get("team") == "A" else "#22c55e"
            self.canvas.create_oval(
                c * self.cell + 2, r * self.cell + 2,
                (c + 1) * self.cell - 2, (r + 1) * self.cell - 2,
                fill=color, outline="#111827", width=1,
            )
            if char.get("has_spike"):
                self.canvas.create_text(
                    (c + 0.5) * self.cell, (r + 0.5) * self.cell,
                    text="S", fill="white", font=("Arial", 9, "bold"),
                )
            self.canvas.create_text(
                (c + 0.5) * self.cell, r * self.cell - 2,
                text=char.get("name", ""), anchor="s", fill="#111827",
                font=("Arial", 7),
            )
        self.status.set(
            f"Frame {self.index + 1}/{len(self.frames)}  "
            f"R{frame.get('round', '-')}/T{frame.get('tick', '-')}  "
            f"Score {frame.get('attacker_wins', 0)}-{frame.get('defender_wins', 0)}  "
            f"View {self.view_mode.get()}"
        )
