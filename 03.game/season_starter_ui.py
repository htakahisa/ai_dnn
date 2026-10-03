"""Choose exactly five initial characters before starting a new season."""

import tkinter as tk
from tkinter import ttk

from realtime_season import ROSTER_SIZE, SeasonSaveError


class SeasonStarterMixin:
    def _build_starters(self):
        self.starter_host = host = ttk.Frame(self.root, padding=24)
        self.starter_summary = tk.StringVar(self.root)
        self.starter_details = tk.StringVar(self.root)
        ttk.Label(host, text="初期キャラを5人選択", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        ttk.Label(host, text="候補から選んだ5人だけを入手して、リアルタイムシーズンを始めます。\n"
                  "選ばなかった候補は所持しません。ゲーム開始後にスカウトで契約できます。",
                  wraplength=900).pack(anchor="w", pady=(12, 16))
        ttk.Label(host, textvariable=self.starter_summary, font=("Yu Gothic UI", 14, "bold")).pack(anchor="w", pady=(0, 12))
        table = ttk.Frame(host)
        table.pack(fill="both", expand=True)
        self.starter_players = tree = ttk.Treeview(table,
            columns=("chosen", "name", "role", "iq", "salary", "loyalty"), show="headings", selectmode="browse", height=12)
        for key, label, width in (("chosen", "選択", 80), ("name", "キャラ", 180), ("role", "ロール", 130),
                                  ("iq", "IQ", 70), ("salary", "月給（円）", 140), ("loyalty", "忠誠心 / 10", 100)):
            tree.heading(key, text=label)
            tree.column(key, width=width, minwidth=45)
        tree.tag_configure("chosen", background="#dff2e3")
        tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table, command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        tree.bind("<<TreeviewSelect>>", lambda _: self.preview_starter())
        tree.bind("<Double-1>", lambda _: self.toggle_starter())
        ttk.Label(host, textvariable=self.starter_details, wraplength=900).pack(anchor="w", pady=12)
        actions = ttk.Frame(host)
        actions.pack(fill="x", pady=(0, 12))
        self.starter_toggle_button = ttk.Button(actions, text="このキャラを選択", command=self.toggle_starter)
        self.starter_toggle_button.pack(side="left")
        self.starter_confirm_button = ttk.Button(actions, text="この5人を入手して開始", command=self.confirm_starters)
        self.starter_confirm_button.pack(side="right")
        ttk.Label(host, text="選択途中も自動保存します。所持金1000万円で開始し、初期の5人は基本月給の1年契約（忠誠心0なら6か月の短期契約）になります。",
                  wraplength=900).pack(anchor="w", pady=(0, 8))
        ttk.Label(host, textvariable=self.status, wraplength=900).pack(anchor="w")

    def refresh_starters(self):
        selected = self.starter_players.selection()
        self.starter_players.delete(*self.starter_players.get_children())
        for player in self.state.starter_candidates:
            chosen = player.name in self.state.starter_selection
            self.starter_players.insert("", "end", iid=player.name, values=(
                "選択済み" if chosen else "未選択", player.name, player.role, f"{player.iq:g}",
                f"{player.monthly_salary:,}", f"{player.loyalty:g}"), tags=("chosen",) if chosen else ())
        if selected and self.starter_players.exists(selected[0]):
            self.starter_players.selection_set(selected[0])
        count = len(self.state.starter_selection)
        self.starter_summary.set(f"{count} / {ROSTER_SIZE}人選択    " + "、".join(self.state.starter_selection))
        self.starter_confirm_button.configure(state="normal" if count == ROSTER_SIZE and self.state.starter_selection_pending else "disabled")
        self.preview_starter()

    def preview_starter(self):
        selected = self.starter_players.selection()
        player = next((p for p in self.state.starter_candidates if selected and p.name == selected[0]), None)
        if player is None:
            self.starter_details.set("候補を選ぶと能力を表示します。ダブルクリックでも選択・解除できます。")
            self.starter_toggle_button.configure(state="disabled")
            return
        self.starter_details.set(f"{player.name} / {player.role}  |  HS率 {player.hs_pct:.0%}  命中率 {player.hit_pct:.0%}  "
                                 f"回避率 {player.dodge_pct:.0%}  反応 {player.reaction:g}  "
                                 f"IQ {player.iq:g}  影響力 {player.influence:g}\n"
                                 f"メンタル {player.mental:g}  調子の波 {player.form_variance:g}  "
                                 f"月給 {player.monthly_salary:,}円  忠誠心 {player.loyalty:g}")
        chosen = player.name in self.state.starter_selection
        can_toggle = self.state.starter_selection_pending and (chosen or len(self.state.starter_selection) < ROSTER_SIZE)
        self.starter_toggle_button.configure(text="このキャラの選択を解除" if chosen else "このキャラを選択",
                                             state="normal" if can_toggle else "disabled")

    def toggle_starter(self):
        selected = self.starter_players.selection()
        if not selected or not self.state.starter_selection_pending:
            return
        name = selected[0]
        names = self.state.starter_selection
        names = tuple(n for n in names if n != name) if name in names else (*names, name)
        try:
            candidate = self.state.with_starter_selection(names)
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        self.commit(candidate, "初期キャラの選択を保存しました。")

    def confirm_starters(self):
        try:
            candidate = self.state.with_initial_selection()
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        if self.commit(candidate, "選んだ5人を入手しました。チーム編成からロスターを登録してください。"):
            self.show_home()
