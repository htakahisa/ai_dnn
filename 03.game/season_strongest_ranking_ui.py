"""Sortable season player standings loaded without blocking Tk."""

import math
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from season_strongest_ranking import load_season_rankings


COLUMNS = ("no", "name", "team", "role", "maps", "kd", "kda", "kills_per_round", "covers", "mvps", "one_v_one")
HEADINGS = ("No.", "選手名", "所属チーム", "ロール", "出場マップ数", "K/D", "K/D/A", "1ラウンド平均Kill", "カバー数", "MVP回数", "1v1 (WinRate)")


def sort_value(row, column):
    if column == "kda":
        values = tuple(row[key] for key in ("kills", "deaths", "assists"))
        return (values[0], -values[1], values[2]) if all(v is not None for v in values) else None
    if column == "one_v_one":
        won, lost = row["one_v_one_won"], row["one_v_one_lost"]
        return won / (won + lost) if won is not None and lost is not None and won + lost else None
    value = row[column]
    return value.casefold() if isinstance(value, str) else value


def ranked_rows(rows, role="すべて", column="kd", descending=True):
    selected = [row for row in rows if role == "すべて" or row["role"] == role]
    known = [row for row in selected if sort_value(row, column) is not None]
    missing = [row for row in selected if sort_value(row, column) is None]
    return sorted(known, key=lambda row: sort_value(row, column), reverse=descending) + missing


def display_values(row, number):
    def count(value):
        return "—" if value is None else str(value)

    kd, duel = row["kd"], sort_value(row, "one_v_one")
    return (number, row["name"], row["team"], row["role"], row["maps"],
            "—" if kd is None else "∞" if math.isinf(kd) else f"{kd:.2f}",
            " / ".join(count(row[key]) for key in ("kills", "deaths", "assists")),
            "—" if row["kills_per_round"] is None else f'{row["kills_per_round"]:.3f}',
            count(row["covers"]), count(row["mvps"]), "—" if duel is None else f"{duel:.1%}")


class SeasonStrongestRankingWindow(tk.Toplevel):
    def __init__(self, root, store, team_name):
        super().__init__(root)
        self.title(f"最強ランキング — {team_name}")
        self.geometry("1240x680")
        self.minsize(900, 450)
        self.save_path = store.path
        self.rows, self.errors = [], []
        self.column, self.descending = "kd", True
        self.maps = 0
        self.loading = False
        self._poll_id = None
        self._closed = False
        self.bind("<Destroy>", self._destroyed, add="+")
        self.role = tk.StringVar(self, value="すべて")
        self.status = tk.StringVar(self)
        ttk.Label(self, text=f"{team_name} — 最強ランキング", font=("Yu Gothic UI", 17, "bold")).pack(anchor="w", padx=16, pady=12)
        toolbar = ttk.Frame(self, padding=(16, 0))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="ロール").pack(side="left")
        self.roles = ttk.Combobox(toolbar, textvariable=self.role, state="readonly", width=18, values=("すべて",))
        self.roles.pack(side="left", padx=8)
        self.roles.bind("<<ComboboxSelected>>", lambda _: self.render())
        self.refresh_button = ttk.Button(toolbar, text="記録を再読み込み", command=self.reload)
        self.refresh_button.pack(side="right")
        self.error_button = ttk.Button(toolbar, text="読込エラー詳細", state="disabled", command=lambda: messagebox.showwarning("記録の読込エラー", "\n".join(self.errors), parent=self))
        self.error_button.pack(side="right", padx=8)
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=16, pady=8)
        frame = ttk.Frame(self, padding=(16, 0, 16, 12))
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, columns=COLUMNS, show="headings")
        for column, heading in zip(COLUMNS, HEADINGS):
            self.tree.heading(column, text=heading, command=(lambda c=column: self.sort(c)) if column != "no" else "")
            width = 55 if column == "no" else 180 if column == "team" else 150 if column in {"name", "kills_per_round", "one_v_one"} else 125
            self.tree.column(column, width=width, minwidth=width, stretch=False, anchor="w" if column in {"name", "team", "role"} else "center")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.reload()

    def _destroyed(self, event):
        if event.widget is self:
            self._closed = True
            if self._poll_id is not None:
                self.after_cancel(self._poll_id)
                self._poll_id = None

    def sort(self, column):
        self.descending = not self.descending if self.column == column else True
        self.column = column
        self.render()

    def render(self):
        self.tree.delete(*self.tree.get_children())
        rows = ranked_rows(self.rows, self.role.get(), self.column, self.descending)
        for number, row in enumerate(rows, 1):
            self.tree.insert("", "end", values=display_values(row, number))
        for column, heading in zip(COLUMNS, HEADINGS):
            self.tree.heading(column, text=heading + (" ▼" if self.descending else " ▲") if column == self.column else heading)
        if not self.loading:
            self.status.set(f"表示 {len(rows)} / {len(self.rows)}選手 — {self.maps}マップ / 読込エラー {len(self.errors)}件 / ロールは最多出場 / 記録不足は —")

    def reload(self):
        if self.loading:
            return
        self.loading = True
        self.refresh_button.configure(state="disabled")
        self.status.set("このセーブの戦績を集計中…")
        messages = queue.Queue()
        save_path = self.save_path

        def work():
            try:
                result = load_season_rankings(save_path, progress=lambda value: messages.put(("progress", value)))
                messages.put(("done", result))
            except Exception as exc:
                messages.put(("error", str(exc)))

        def poll():
            self._poll_id = None
            if self._closed:
                return
            while not messages.empty():
                kind, value = messages.get_nowait()
                if kind == "progress":
                    self.status.set(value)
                    continue
                self.loading = False
                self.refresh_button.configure(state="normal")
                if kind == "error":
                    self.status.set(f"集計エラー: {value}")
                    return
                self.rows, self.errors, self.maps = value["rows"], value["errors"], value["maps"]
                self.error_button.configure(state="normal" if self.errors else "disabled")
                roles = ["すべて", *sorted({row["role"] for row in self.rows})]
                self.roles.configure(values=roles)
                if self.role.get() not in roles:
                    self.role.set("すべて")
                self.render()
                return
            self._poll_id = self.after(100, poll)

        threading.Thread(target=work, daemon=True).start()
        self._poll_id = self.after(100, poll)
