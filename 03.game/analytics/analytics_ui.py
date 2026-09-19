"""Tkinter analytics viewer for competition result JSON files."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ANALYTICS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYTICS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "competition_results"
if str(ANALYTICS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYTICS_DIR))

from analytics_output import (  # noqa: E402
    build_improvement_suggestions,
    generate_full_report,
)
from match_analyzer import load_series_json  # noqa: E402
from replay_viewer import ReplayViewer  # noqa: E402


def _pct(wins: int, losses: int, draws: int = 0) -> str:
    total = wins + losses
    return "-" if not total else f"{wins / total * 100:.1f}%"


def _team_stat_rows(series):
    players = list(series.get_all_players().values())
    return sorted(players, key=lambda p: (p.team, -p.kills, p.name))


class AnalyticsViewer(tk.Tk):
    def __init__(self, results_dir: Path = RESULTS_DIR):
        super().__init__()
        self.results_dir = Path(results_dir)
        self.result_paths: list[Path] = []
        self.loaded_series = None
        self.title("Competition Analytics")
        self.geometry("1500x900")
        self.minsize(1050, 650)
        self.status_var = tk.StringVar(value="Ready")
        self._build_widgets()
        self.refresh_results()

    def _build_widgets(self):
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Refresh", command=self.refresh_results).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Analyze selected match", command=self.analyze_selected).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Replay selected match", command=self.open_replay).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Open results folder", command=self.open_results_folder).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        splitter = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        splitter.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(splitter, padding=4)
        right = ttk.Frame(splitter, padding=4)
        splitter.add(left, weight=1)
        splitter.add(right, weight=5)

        ttk.Label(left, text="Matches (newest first)").pack(anchor=tk.W)
        self.file_count_var = tk.StringVar()
        ttk.Label(left, textvariable=self.file_count_var).pack(anchor=tk.W)
        file_frame = ttk.Frame(left)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.file_tree = ttk.Treeview(file_frame, columns=("modified", "size"), show="tree headings", selectmode="browse")
        self.file_tree.heading("#0", text="File")
        self.file_tree.heading("modified", text="Modified")
        self.file_tree.heading("size", text="Size")
        self.file_tree.column("#0", width=320)
        self.file_tree.column("modified", width=145)
        self.file_tree.column("size", width=75, anchor=tk.E)
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.configure(yscrollcommand=scroll.set)
        self.file_tree.bind("<<TreeviewSelect>>", self._on_selection)
        self.file_tree.bind("<Double-1>", lambda _event: self.analyze_selected())

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self.overview_tab = ttk.Frame(self.tabs, padding=8)
        self.players_tab = ttk.Frame(self.tabs, padding=8)
        self.rounds_tab = ttk.Frame(self.tabs, padding=8)
        self.suggestions_tab = ttk.Frame(self.tabs, padding=8)
        self.report_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.overview_tab, text="Overview")
        self.tabs.add(self.players_tab, text="Scoreboard")
        self.tabs.add(self.rounds_tab, text="Rounds")
        self.tabs.add(self.suggestions_tab, text="Suggestions")
        self.tabs.add(self.report_tab, text="Full report")
        self.report_text = ScrolledText(self.report_tab, wrap=tk.NONE, font=("Consolas", 10), undo=False)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        self.report_text.configure(state=tk.DISABLED)
        self._show_empty_state()

    def _show_empty_state(self):
        for tab in (self.overview_tab, self.players_tab, self.rounds_tab, self.suggestions_tab):
            for child in tab.winfo_children():
                child.destroy()
        ttk.Label(self.overview_tab, text="Select a match and click Analyze selected match.").pack(anchor=tk.W)
        ttk.Label(self.players_tab, text="No match analyzed.").pack(anchor=tk.W)
        ttk.Label(self.rounds_tab, text="No match analyzed.").pack(anchor=tk.W)
        ttk.Label(self.suggestions_tab, text="No match analyzed.").pack(anchor=tk.W)

    def _set_report(self, text: str):
        self.report_text.configure(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert("1.0", text)
        self.report_text.configure(state=tk.DISABLED)

    def _make_tree(self, parent, columns, headings, widths=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for i, column in enumerate(columns):
            tree.heading(column, text=headings[i])
            tree.column(column, width=(widths[i] if widths else 100), anchor=tk.CENTER)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=yscroll.set)
        xscroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=tree.xview)
        xscroll.pack(fill=tk.X)
        tree.configure(xscrollcommand=xscroll.set)
        return tree

    def _populate_structured_views(self, series):
        for tab in (self.overview_tab, self.players_tab, self.rounds_tab, self.suggestions_tab):
            for child in tab.winfo_children():
                child.destroy()
        ttk.Label(self.overview_tab, text=f"{series.team1}  {series.team1_wins}  -  {series.team2_wins}  {series.team2}", font=("Segoe UI", 18, "bold")).pack(anchor=tk.W)
        ttk.Label(self.overview_tab, text=f"Winner: {series.winner or '-'}    Maps: {series.total_maps}    Total rounds: {series.total_rounds}").pack(anchor=tk.W, pady=(4, 12))
        map_tree = self._make_tree(self.overview_tab, ("map", "teams", "score", "winner", "attacker", "rounds"), ("Map", "Teams", "Score", "Winner", "Initial attacker", "Rounds"), (70, 360, 90, 180, 180, 80))
        for m in series.maps:
            map_tree.insert("", tk.END, values=(m.number, f"{m.team1} vs {m.team2}", f"{m.score1} - {m.score2}", m.winner, m.initial_attacker, m.total_rounds))

        ttk.Label(self.players_tab, text="Series aggregate", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W, pady=(0, 6))
        columns = ("team", "player", "role", "kda", "kd", "fights", "fight_wr", "1v1", "assists", "covers", "fkfd", "preaim")
        headings = ("Team", "Player", "Role", "K / D / A", "K/D", "Fights", "Fight win", "1v1 W/L", "Assists", "Covers", "FK / FD", "Preaim")
        player_tree = self._make_tree(self.players_tab, columns, headings, (150, 150, 100, 100, 65, 70, 80, 70, 70, 70, 70, 80))
        for p in _team_stat_rows(series):
            fights = p.gunfights_won + p.gunfights_lost + p.gunfights_draw
            preaim = "-" if not p.preaim_angle_count else f"{p.preaim_angle_sum / p.preaim_angle_count:.1f}°"
            player_tree.insert("", tk.END, values=(p.team, p.name, p.role or "-", f"{p.kills} / {p.deaths} / {p.assists}", f"{p.kd_ratio:.2f}", fights, _pct(p.gunfights_won, p.gunfights_lost), f"{p.one_v_one_won} / {p.one_v_one_lost}", p.assists, p.covers, f"{p.first_kills} / {p.first_deaths}", preaim))

        ttk.Label(self.rounds_tab, text="All recorded rounds", font=("Segoe UI", 13, "bold")).pack(anchor=tk.W, pady=(0, 6))
        round_tree = self._make_tree(self.rounds_tab, ("map", "round", "winner", "reason", "attacker", "site", "attack", "def_setup", "fake_score", "displaced", "plant", "defuse"), ("Map", "Round", "Winner", "Reason", "Attacker", "Site", "Attack tactic", "Defender setup", "Fake score", "Moved", "Plant", "Defuse"), (55, 65, 150, 150, 150, 65, 120, 150, 85, 65, 65, 65))
        for m in series.maps:
            for r in m.round_records:
                tactic = r.get("tactic", {}) or {}
                round_tree.insert("", tk.END, values=(m.number, r.get("round_number", "-"), r.get("winner", "-"), r.get("reason", r.get("win_reason", "-")), r.get("attacker_team", "-"), tactic.get("final_attack_site", r.get("site", "-")), tactic.get("attacker_strategy", r.get("attack_tactic", "-")), tactic.get("defender_initial_setup", r.get("def_setup", "-")), f"{float(tactic.get('fake_effect_score', 0.0) or 0.0):.1f}", tactic.get("defenders_displaced_from_final", 0), "Yes" if r.get("planted") else "No", "Yes" if r.get("defused") else "No"))

        ttk.Label(self.suggestions_tab, text="Improvement suggestions", font=("Segoe UI", 15, "bold")).pack(anchor=tk.W, pady=(0, 10))
        for suggestion in build_improvement_suggestions(series):
            ttk.Label(
                self.suggestions_tab,
                text=f"• {suggestion}",
                wraplength=1050,
                justify=tk.LEFT,
            ).pack(anchor=tk.W, fill=tk.X, pady=4)

    def refresh_results(self):
        self.file_tree.delete(*self.file_tree.get_children())
        self.result_paths = []
        if not self.results_dir.exists():
            self.file_count_var.set("Folder not found")
            self.status_var.set(str(self.results_dir))
            self._show_empty_state()
            return
        self.result_paths = sorted((p for p in self.results_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for i, path in enumerate(self.result_paths):
            stat = path.stat()
            self.file_tree.insert("", tk.END, iid=str(i), text=path.name, values=(datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"), f"{stat.st_size / 1024:.1f} KB"))
        self.file_count_var.set(f"{len(self.result_paths)} files")
        self.status_var.set("List refreshed")
        if self.result_paths:
            first = self.file_tree.get_children()[0]
            self.file_tree.selection_set(first)
            self.file_tree.focus(first)
        else:
            self._show_empty_state()

    def _selected_path(self):
        selected = self.file_tree.selection()
        if not selected:
            return None
        index = int(selected[0])
        return self.result_paths[index] if index < len(self.result_paths) else None

    def _on_selection(self, _event=None):
        path = self._selected_path()
        if path:
            self.status_var.set(f"Selected: {path.name}")

    def analyze_selected(self):
        path = self._selected_path()
        if path is None:
            messagebox.showinfo("Analyze", "Select a match first.")
            return
        self.status_var.set(f"Analyzing: {path.name}")
        self.update_idletasks()
        try:
            series = load_series_json(str(path))
            self.loaded_series = series
            self._populate_structured_views(series)
            self._set_report(f"File: {path.name}\nModified: {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M:%S}\n{'=' * 100}\n\n" + generate_full_report(series))
            self.tabs.select(self.overview_tab)
            self.status_var.set("Analysis complete")
        except Exception as exc:
            self._set_report(f"Analysis failed for {path}\n\n{exc}\n\n{traceback.format_exc()}")
            self.status_var.set("Analysis error")
            messagebox.showerror("Analysis error", str(exc))

    def open_replay(self):
        series = self.loaded_series
        if series is None:
            path = self._selected_path()
            if path is None:
                messagebox.showinfo("Replay", "Select and analyze a match first.")
                return
            try:
                series = load_series_json(str(path))
                self.loaded_series = series
            except Exception as exc:
                messagebox.showerror("Replay error", str(exc))
                return

        maps = [m for m in series.maps if m.replay_frames]
        if not maps:
            messagebox.showinfo(
                "Replay",
                "この試合にはリプレイデータがありません。新しい試合から記録されます。",
            )
            return
        # Open the first recorded map.  Map selection can be added without
        # changing the replay format because each map stores its own frames.
        map_data = maps[0]
        ReplayViewer(
            self,
            map_data.replay_frames,
            map_options=maps,
            title=f"Replay - Map {map_data.number} ({map_data.team1} vs {map_data.team2})",
        )

    def open_results_folder(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(self.results_dir))
        else:
            subprocess.Popen(["xdg-open", str(self.results_dir)])


def main():
    AnalyticsViewer().mainloop()


if __name__ == "__main__":
    main()
