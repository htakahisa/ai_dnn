"""Choose exactly five initial characters before starting a new season."""

import tkinter as tk
from tkinter import ttk

from realtime_season import ROSTER_SIZE, SeasonSaveError
from season.season_salary import SALARY_MODE_LABELS


class SeasonStarterMixin:
    def _build_starters(self):
        self.starter_host = host = ttk.Frame(self.root, padding=24)
        self.starter_summary = tk.StringVar(self.root)
        self.starter_details = tk.StringVar(self.root)
        self.starter_team_name = tk.StringVar(self.root, value=self.state.team_name)
        ttk.Label(host, text="初期キャラを5人選択", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        ttk.Label(host, text="候補から選んだ5人だけを入手して、リアルタイムシーズンを始めます。\n"
                  "選ばなかった候補は所持しません。ゲーム開始後にスカウトで契約できます。",
                  wraplength=900).pack(anchor="w", pady=(12, 16))
        naming = ttk.Frame(host)
        naming.pack(fill="x", pady=(0, 12))
        ttk.Label(naming, text="チーム名").pack(side="left")
        self.starter_name_entry = ttk.Entry(naming, textvariable=self.starter_team_name, width=32)
        self.starter_name_entry.pack(side="left", padx=8)
        self.starter_name_entry.bind("<FocusOut>", lambda _: self.save_starter_team_name())
        self.starter_name_entry.bind("<Return>", lambda _: self.save_starter_team_name())
        ttk.Label(naming, text="編成プリセットとは別の、シーズンを通して使う名前です。").pack(side="left")
        salary = ttk.Frame(host)
        salary.pack(fill="x", pady=(0, 8))
        ttk.Label(salary, text="給与モード").pack(side="left")
        self.starter_salary_mode = tk.StringVar(self.root)
        self.starter_salary_menu = ttk.Combobox(salary, textvariable=self.starter_salary_mode,
            values=tuple(SALARY_MODE_LABELS), state="readonly", width=25)
        self.starter_salary_menu.pack(side="left", padx=8)
        self.starter_salary_menu.bind("<<ComboboxSelected>>", self.change_starter_salary_mode)
        self.starter_salary_hint = tk.StringVar(self.root)
        ttk.Label(salary, textvariable=self.starter_salary_hint,
                  wraplength=560).pack(side="left")
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
        ttk.Label(host, text="選択途中も自動保存します。所持金1000万円で開始し、初期の5人は基本月給の6か月の短期契約になります。",
                  wraplength=900).pack(anchor="w", pady=(0, 8))
        ttk.Label(host, textvariable=self.status, wraplength=900).pack(anchor="w")

    def refresh_starters(self):
        minimum = self.state.salary_settings.min_games if self.state.salary_settings is not None else 10
        self.starter_salary_hint.set(f"開始後は変更できません。成績連動はコンペティション{minimum}マップ以上のK/Dが基準です。")
        self.starter_salary_mode.set(next(label for label, mode in SALARY_MODE_LABELS.items() if mode == self.state.salary_mode))
        self.starter_salary_menu.configure(state="readonly" if self.state.starter_selection_pending else "disabled")
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

    def change_starter_salary_mode(self, _event=None):
        try:
            candidate = self.state.with_salary_mode(SALARY_MODE_LABELS[self.starter_salary_mode.get()])
        except (ValueError, KeyError) as exc:
            self.status.set(str(exc))
            self.refresh_starters()
            return
        if not self.commit(candidate, "給与モードを保存しました。初期5人の確定後は変更できません。"):
            self.refresh_starters()

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
            candidate = self.state.with_team_name(self.starter_team_name.get()).with_initial_selection()
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        if self.commit(candidate, "選んだ5人を入手しました。チーム編成からロスターを登録してください。"):
            self.show_home()

    def save_starter_team_name(self):
        if not self.state.starter_selection_pending:
            return
        try:
            candidate = self.state.with_team_name(self.starter_team_name.get())
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        if candidate != self.state:
            self.commit(candidate, "チーム名を保存しました。初期キャラを5人選んでください。")
