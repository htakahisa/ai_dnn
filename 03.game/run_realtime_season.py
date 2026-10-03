"""Real-time season home, team editor, and pre-match preparation."""

import argparse
from datetime import datetime
import re
import tkinter as tk
from tkinter import messagebox, ttk

from realtime_season import DEFAULT_SAVE_PATH, ROSTER_SIZE, SeasonSaveError, SeasonStore
from season_scrim import ScrimJob, ai_options, build_scrim_request
from season_management_ui import SeasonManagementMixin
from season_competition_ui import SeasonCompetitionMixin
from season_starter_ui import SeasonStarterMixin
from season_rating_ui import SeasonRatingMixin
from season_monthly_ui import SeasonMonthlyMixin


class RealtimeSeasonApp(SeasonManagementMixin, SeasonCompetitionMixin, SeasonStarterMixin, SeasonRatingMixin, SeasonMonthlyMixin):
    def __init__(self, root, store, state):
        self.root = root
        self.store = store
        self.state = state
        root.title("リアルタイムシーズン")
        root.geometry("1120x800")
        root.minsize(960, 800)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.preset_name = tk.StringVar(root, value=state.preset_name)
        self.club_name = tk.StringVar(root, value=state.team_name)
        self.search = tk.StringVar(root)
        self.role = tk.StringVar(root, value="すべて")
        self.status = tk.StringVar(root, value="初期キャラ候補から5人を選んでください。選択途中も自動保存します。" if state.starter_selection_pending else
                                   "変更は自動保存されます。まずプレイヤー名を入力して所持選手を設定してください。" if not state.owned_players else
                                   "セーブデータを読み込みました。変更は自動保存されます。")
        self.summary = tk.StringVar(root)
        self.details = tk.StringVar(root, value="所持選手を選ぶと能力を表示します。")
        self.edit_team_choice = tk.StringVar(root)
        self.prep_team_choice = tk.StringVar(root)
        self.home_summary = tk.StringVar(root)
        self.home_selected = tk.StringVar(root)
        self.prep_hint = tk.StringVar(root)
        self.opponent_choice = tk.StringVar(root)
        self.own_ai_choice = tk.StringVar(root, value="ロジック")
        self.opponent_ai_choice = tk.StringVar(root, value="ロジック")
        self.own_igl = tk.StringVar(root)
        self.opponent_igl = tk.StringVar(root)
        self.own_spike = tk.StringVar(root)
        self.opponent_spike = tk.StringVar(root)
        self.scrim_render = tk.BooleanVar(root, value=True)
        self.scrim_side = tk.StringVar(root, value="攻撃")
        self.scrim_tick_ms = tk.StringVar(root, value="100")
        self.scrim_result = tk.StringVar(root, value="スクリムの対戦相手と使用チームを選択してください。")
        self.scrim_job = None
        self._scrim_after_id = None
        self._prepared_opponent_settings = None
        self._scrim_rating_context = None
        self.current_screen = "home"
        self._build()
        self._build_competitions()
        self._build_home()
        self._build_preparation()
        self._build_management()
        self._build_starters()
        self._build_ratings()
        self._build_monthly_events()
        self.search.trace_add("write", lambda *_: self.refresh_players())
        self.role.trace_add("write", lambda *_: self.refresh_players())
        self.refresh()
        self.show_screen("starter" if state.starter_selection_pending else "home")

    def _build(self):
        host = ttk.Frame(self.root, padding=16)
        self.editor_host = host
        ttk.Label(host, text="リアルタイムシーズン", font=("Yu Gothic UI", 20, "bold")).pack(anchor="w")
        navigation = ttk.Frame(host)
        navigation.pack(fill="x", pady=(4, 8))
        ttk.Button(navigation, text="ホームに戻る", command=self.show_home).pack(side="left")
        ttk.Button(navigation, text="スクリム準備へ", command=self.show_preparation).pack(side="left", padx=8)
        ttk.Label(host, text="編成プリセット  |  自チームの所持選手から5人を登録します。同じ選手を複数のプリセットで使えます。").pack(anchor="w", pady=(0, 12))
        team_bar = ttk.Frame(host)
        team_bar.pack(fill="x", pady=(0, 8))
        ttk.Label(team_bar, text="編集するプリセット").pack(side="left")
        self.edit_team_menu = ttk.Combobox(team_bar, textvariable=self.edit_team_choice, state="readonly", width=30)
        self.edit_team_menu.pack(side="left", padx=8)
        self.edit_team_menu.bind("<<ComboboxSelected>>", self.edit_saved_team)
        ttk.Button(team_bar, text="新しいプリセットを作る", command=self.new_team).pack(side="left")
        header = ttk.Frame(host)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="プリセット名").pack(side="left")
        entry = ttk.Entry(header, textvariable=self.preset_name, width=30)
        entry.pack(side="left", padx=8)
        entry.bind("<FocusOut>", lambda _: self.save_preset_name())
        entry.bind("<Return>", lambda _: self.save_preset_name())
        ttk.Button(header, text="プリセット名を保存", command=self.save_preset_name).pack(side="left")

        main = ttk.Frame(host)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)
        inventory = ttk.LabelFrame(main, text="所持選手", padding=10)
        inventory.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        inventory.columnconfigure(0, weight=1)
        inventory.rowconfigure(1, weight=1)
        filters = ttk.Frame(inventory)
        filters.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(filters, text="検索").pack(side="left")
        ttk.Entry(filters, textvariable=self.search, width=18).pack(side="left", padx=6)
        roles = ["すべて"] + sorted({p.role for p in self.state.owned_players})
        self.role_filter = ttk.Combobox(filters, textvariable=self.role, values=roles, state="readonly", width=12)
        self.role_filter.pack(side="left")
        self.players = ttk.Treeview(inventory, columns=("name", "role", "iq", "status"), show="headings", selectmode="browse")
        for key, label, width in (("name", "選手", 150), ("role", "ロール", 100), ("iq", "IQ", 50), ("status", "所属・編成", 110)):
            self.players.heading(key, text=label)
            self.players.column(key, width=width, minwidth=40, anchor="w")
        self.players.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(inventory, orient="vertical", command=self.players.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.players.configure(yscrollcommand=scroll.set)
        self.players.bind("<<TreeviewSelect>>", self.select_player)
        self.players.bind("<Double-1>", lambda _: self.add_player())
        self.add_button = ttk.Button(inventory, text="選択した選手をロスターに追加 →", command=self.add_player)
        self.add_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.delete_button = ttk.Button(inventory, text="選択した控え選手を所持選手から外す", command=self.delete_owned_player)
        self.delete_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        setup = ttk.LabelFrame(inventory, text="プレイヤー名を入力して所持選手を設定", padding=8)
        setup.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Label(setup, text="既存のプレイヤー名を改行・カンマ区切りで入力（例: Leo, Boaster, Derke）", wraplength=500).pack(anchor="w")
        self.name_input = tk.Text(setup, height=3, wrap="word", font=("Yu Gothic UI", 10))
        self.name_input.pack(fill="x", pady=6)
        ttk.Button(setup, text="入力した選手を所持選手に追加", command=self.register_players).pack(fill="x")
        if self.state.starter_candidates:
            setup.grid_remove()

        lineup = ttk.LabelFrame(main, text="ロスター", padding=10)
        lineup.grid(row=0, column=1, sticky="nsew")
        ttk.Label(lineup, textvariable=self.summary, font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", pady=(0, 8))
        self.roster = ttk.Treeview(lineup, columns=("slot", "name", "role"), show="headings", selectmode="browse", height=5)
        for key, label, width in (("slot", "枠", 35), ("name", "選手", 140), ("role", "ロール", 90)):
            self.roster.heading(key, text=label)
            self.roster.column(key, width=width, minwidth=30, anchor="w")
        self.roster.pack(fill="x")
        self.roster.bind("<<TreeviewSelect>>", lambda _: self.update_buttons())
        self.roster.bind("<Double-1>", lambda _: self.remove_player())
        self.remove_button = ttk.Button(lineup, text="選択した選手をロスターから外す", command=self.remove_player)
        self.remove_button.pack(fill="x", pady=10)
        ttk.Label(lineup, text="選手の詳細", font=("Yu Gothic UI", 11, "bold")).pack(anchor="w", pady=(6, 4))
        ttk.Label(lineup, textvariable=self.details, justify="left", wraplength=330).pack(anchor="w")
        self.confirm_button = ttk.Button(lineup, text="この5人で編成を確定", command=self.confirm)
        self.confirm_button.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(host, textvariable=self.status, wraplength=1000).pack(anchor="w", pady=(12, 4))
        ttk.Label(host, text=f"保存先: {self.store.path}", wraplength=1000).pack(anchor="w")

    def _build_home(self):
        self.home_host = host = ttk.Frame(self.root, padding=24)
        ttk.Label(host, text="リアルタイムシーズン", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        navigation = ttk.Frame(host)
        navigation.pack(fill="x", pady=(4, 8))
        ttk.Label(navigation, text="ホーム", font=("Yu Gothic UI", 14)).pack(side="left")
        ttk.Entry(navigation, textvariable=self.club_name, width=24).pack(side="left", padx=8)
        ttk.Button(navigation, text="チーム名を変更", command=self.rename_club).pack(side="left")
        ttk.Button(navigation, text="レーティング・スポンサー", command=lambda: self.show_screen("ratings")).pack(side="right")
        ttk.Button(navigation, text="月次イベント", command=lambda: self.show_screen("monthly")).pack(side="right", padx=8)
        self._build_calendar_home(host)
        ttk.Label(host, textvariable=self.home_summary, font=("Yu Gothic UI", 12)).pack(anchor="w", pady=(0, 12))
        actions = ttk.Frame(host)
        actions.pack(fill="x", pady=(0, 12))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        for index, (text, hint, command) in enumerate((
            ("チーム編成", "所持選手から5人の編成プリセットを保存", self.show_editor),
            ("スクリム", "相手と使用する編成を選んで練習試合", self.show_preparation),
            ("スカウト", "LFTとの契約・他チームからの引き抜き", lambda: self.show_screen("scout")),
            ("契約状況", "契約の経過・忠誠を確認して再契約", lambda: self.show_screen("contracts")),
        )):
            card = ttk.LabelFrame(actions, text=text, padding=10)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=(0, 12), pady=(0, 8))
            ttk.Label(card, text=hint, wraplength=400).pack(anchor="w", pady=(0, 12))
            ttk.Button(card, text=f"{text}へ", command=command).pack(fill="x")
        ttk.Label(host, text="シーズンの所属チーム", font=("Yu Gothic UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
        table = ttk.Frame(host)
        table.pack(fill="both", expand=True)
        self.home_teams = ttk.Treeview(table, columns=("kind", "name", "rating", "multiplier", "players"), show="headings", selectmode="browse", height=6)
        self.home_teams.heading("kind", text="区分")
        self.home_teams.heading("name", text="チーム名")
        self.home_teams.heading("players", text="所属選手（先頭5人がロスター）")
        self.home_teams.heading("rating", text="レート")
        self.home_teams.heading("multiplier", text="移籍金倍率")
        self.home_teams.column("rating", width=85, stretch=False)
        self.home_teams.column("multiplier", width=90, stretch=False)
        self.home_teams.column("kind", width=100, stretch=False)
        self.home_teams.column("name", width=200)
        self.home_teams.column("players", width=600)
        self.home_teams.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(table, command=self.home_teams.yview)
        scrollbar.pack(side="right", fill="y")
        self.home_teams.configure(yscrollcommand=scrollbar.set)
        ttk.Label(host, textvariable=self.home_selected, font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", pady=(20, 6))
        ttk.Label(host, textvariable=self.scrim_result, wraplength=1000).pack(anchor="w", pady=(0, 8))
        ttk.Label(host, textvariable=self.status, wraplength=1000).pack(anchor="w")

    def _build_preparation(self):
        self.preparation_host = host = ttk.Frame(self.root, padding=24)
        ttk.Label(host, text="スクリム — 試合前の準備", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        navigation = ttk.Frame(host)
        navigation.pack(fill="x", pady=(12, 20))
        ttk.Button(navigation, text="ホームに戻る", command=self.show_home).pack(side="left")
        ttk.Button(navigation, text="チーム編成へ", command=self.show_editor).pack(side="left", padx=8)
        ttk.Label(host, text="自分のチームと同じ舞台の相手を選択し、試合前の準備を整えてください。").pack(anchor="w", pady=(0, 12))
        chooser = ttk.Frame(host)
        chooser.pack(fill="x", pady=(0, 16))
        ttk.Label(chooser, text="使用する編成").pack(side="left")
        self.prep_team_menu = ttk.Combobox(chooser, textvariable=self.prep_team_choice, state="readonly", width=40)
        self.prep_team_menu.pack(side="left", padx=12)
        self.prep_team_menu.bind("<<ComboboxSelected>>", self.preview_preparation)
        self.prep_confirm_button = ttk.Button(chooser, text="この編成を使用", command=self.confirm_preparation)
        self.prep_confirm_button.pack(side="left")
        opponent_chooser = ttk.Frame(host)
        opponent_chooser.pack(fill="x", pady=(0, 12))
        ttk.Label(opponent_chooser, text="相手チーム").pack(side="left")
        self.opponent_menu = ttk.Combobox(opponent_chooser, textvariable=self.opponent_choice, state="readonly", width=40)
        self.opponent_menu.pack(side="left", padx=12)
        self.opponent_menu.bind("<<ComboboxSelected>>", self.preview_preparation)
        rosters = ttk.Frame(host)
        rosters.pack(fill="x")
        self.role_menus = {}
        options = ai_options()
        for key, title, igl_var, spike_var, ai_var in (
            ("own", "自分のチーム", self.own_igl, self.own_spike, self.own_ai_choice),
            ("opponent", "相手チーム", self.opponent_igl, self.opponent_spike, self.opponent_ai_choice),
        ):
            panel = ttk.LabelFrame(rosters, text=title, padding=8)
            panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
            tree = ttk.Treeview(panel, columns=("slot", "name", "role", "iq"), show="headings", height=5)
            for column, label, width in (("slot", "枠", 30), ("name", "選手", 145), ("role", "ロール", 110), ("iq", "IQ", 45)):
                tree.heading(column, text=label)
                tree.column(column, width=width, anchor="w")
            tree.pack(fill="x")
            if key == "own":
                self.prep_roster = tree
            else:
                self.opponent_roster = tree
            for label, variable, values in (("IGL", igl_var, ()), ("スパイク担当", spike_var, ()), ("AI", ai_var, tuple(options))):
                row = ttk.Frame(panel)
                row.pack(fill="x", pady=(6, 0))
                ttk.Label(row, text=label, width=12).pack(side="left")
                menu = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=24)
                menu.pack(side="left", fill="x", expand=True)
                menu.bind("<<ComboboxSelected>>", lambda _: self.update_scrim_start_state())
                if label != "AI":
                    self.role_menus[(key, label)] = menu
        settings = ttk.Frame(host)
        settings.pack(fill="x", pady=(18, 10))
        ttk.Label(settings, text="開始サイド").pack(side="left")
        ttk.Combobox(settings, textvariable=self.scrim_side, values=("攻撃", "防衛"), state="readonly", width=8).pack(side="left", padx=8)
        ttk.Checkbutton(settings, text="描画あり（通常の試合画面）", variable=self.scrim_render, command=self.update_scrim_start_state).pack(side="left", padx=12)
        ttk.Label(settings, text="1tick (ms)").pack(side="left", padx=(12, 5))
        ttk.Entry(settings, textvariable=self.scrim_tick_ms, width=8).pack(side="left")
        ttk.Label(host, text="描画なしの場合は別プロセスで高速シミュレーションします。").pack(anchor="w")
        buttons = ttk.Frame(host)
        buttons.pack(fill="x", pady=12)
        self.scrim_start_button = ttk.Button(buttons, text="スクリム開始", command=self.start_scrim)
        self.scrim_start_button.pack(side="left")
        self.scrim_cancel_button = ttk.Button(buttons, text="スクリムを中止", command=self.cancel_scrim, state="disabled")
        self.scrim_cancel_button.pack(side="left", padx=12)
        ttk.Label(host, textvariable=self.scrim_result, wraplength=1000).pack(anchor="w", pady=(0, 8))
        ttk.Label(host, textvariable=self.prep_hint, wraplength=1000).pack(anchor="w", pady=(8, 4))
        ttk.Label(host, textvariable=self.status, wraplength=1000).pack(anchor="w", pady=(8, 0))

    def show_screen(self, screen):
        if self.state.starter_selection_pending:
            if screen != "starter":
                self.status.set("初期キャラ候補から5人を選んで確定してください。")
            screen = "starter"
        elif screen == "starter":
            screen = "home"
        if self.current_screen == "editor" and screen != "editor" and not self.save_preset_name():
            return False
        hosts = {"home": self.home_host, "editor": self.editor_host, "preparation": self.preparation_host,
                 "scout": self.scout_host, "contracts": self.contracts_host, "competitions": self.competition_host,
                 "starter": self.starter_host, "ratings": self.ratings_host, "monthly": self.monthly_host}
        for host in hosts.values():
            host.pack_forget()
        hosts[screen].pack(fill="both", expand=True)
        self.current_screen = screen
        title = {"home": "ホーム", "editor": "チーム編成", "preparation": "スクリム準備",
                 "scout": "スカウト", "contracts": "契約状況", "competitions": "大会", "starter": "初期キャラ選択", "ratings": "レーティング", "monthly": "月次イベント"}[screen]
        self.root.title(f"リアルタイムシーズン — {title}")
        if screen == "preparation":
            self.refresh_preparation()
        elif screen in ("scout", "contracts"):
            self.refresh_management()
        elif screen == "competitions":
            self.refresh_competitions()
        elif screen == "starter":
            self.refresh_starters()
        elif screen == "ratings":
            self.refresh_ratings()
        elif screen == "monthly":
            self.refresh_monthly_events()
        return True

    def show_home(self):
        return self.show_screen("home")

    def show_editor(self):
        return self.show_screen("editor")

    def show_preparation(self):
        return self.show_screen("preparation")

    def refresh_home(self):
        self.home_summary.set(f"所持金: {self.state.money:,}円\n"
                              f"チーム: {self.state.team_name}    所持選手: {len(self.state.owned_players)}人    編成プリセット: {len(self.state.teams)}    他チーム: {len(self.state.opponent_teams)}\n"
                              f"スポンサー契約: {'有効' if self.state.sponsor_active else '停止'}    月額収入: {self.state.monthly_sponsor_income:,}円")
        self.home_teams.delete(*self.home_teams.get_children())
        self.home_teams.insert("", "end", iid=self.state.club_id, values=("自分", self.state.team_name,
            f"{self.state.rating(self.state.club_id):.3f}", "—", " / ".join(p.name for p in self.state.owned_players)))
        for team in self.state.opponent_teams:
            self.home_teams.insert("", "end", iid=team.id, values=("他チーム", team.name, f"{self.state.rating(team.id):.3f}", f"{team.transfer_multiplier:g}倍", " / ".join(team.members) or "選手なし"))
        selected = self.state.selected_team
        self.home_selected.set(f"使用する編成: {selected.name}" if selected else "使用する編成: 未選択（スクリムの準備で選択してください）")

    def refresh_preparation(self):
        self.prep_team_menu.configure(values=[team.name for team in self.state.teams], state="readonly" if self.state.teams else "disabled")
        selected = self.state.selected_team
        self.prep_team_choice.set(selected.name if selected else "")
        opponent_names = [team.name for team in self.state.opponent_teams]
        self.opponent_menu.configure(values=opponent_names, state="readonly" if opponent_names else "disabled")
        if self.opponent_choice.get() not in opponent_names:
            self.opponent_choice.set("")
        self.preview_preparation()

    def preview_preparation(self, _event=None):
        team = next((team for team in self.state.teams if team.name == self.prep_team_choice.get()), None)
        self.prep_roster.delete(*self.prep_roster.get_children())
        if team:
            for index, name in enumerate(team.roster):
                player = self.state.player(name)
                self.prep_roster.insert("", "end", values=(index + 1, name, player.role, f"{player.iq:g}"))
        self.prep_confirm_button.configure(state="normal" if team else "disabled")
        if not self.state.teams:
            hint = "編成プリセットがありません。チーム編成画面で5人を編成し、保存してください。"
        elif team and team.id == self.state.selected_team_id:
            hint = f"使用する編成は「{team.name}」です。選択内容は保存済みです。"
        else:
            hint = "編成を選択し、メンバーを確認して「この編成を使用」を押してください。"
        self.prep_hint.set(hint)
        opponent = next((club for club in self.state.opponent_teams if club.name == self.opponent_choice.get()), None)
        opponent_settings = (opponent.id, opponent.effective_igl, opponent.effective_carrier, opponent.ai) if opponent else None
        if opponent_settings != self._prepared_opponent_settings:
            self._prepared_opponent_settings = opponent_settings
            self.opponent_igl.set(opponent.effective_igl if opponent else "")
            self.opponent_spike.set(opponent.effective_carrier if opponent else "")
            labels = {key: label for label, key in ai_options().items()}
            self.opponent_ai_choice.set(labels[opponent.ai] if opponent else labels["default"])
        self.opponent_roster.delete(*self.opponent_roster.get_children())
        if opponent:
            for index, player in enumerate(opponent.players[:ROSTER_SIZE]):
                self.opponent_roster.insert("", "end", values=(index + 1, player.name, player.role, f"{player.iq:g}"))
        for key, players, igl_var, spike_var in (
            ("own", [self.state.player(name) for name in team.roster] if team else [], self.own_igl, self.own_spike),
            ("opponent", list(opponent.players[:ROSTER_SIZE]) if opponent else [], self.opponent_igl, self.opponent_spike),
        ):
            names = [p.name for p in players]
            for label in ("IGL", "スパイク担当"):
                self.role_menus[(key, label)].configure(values=names, state="readonly" if names else "disabled")
            if igl_var.get() not in names:
                igl_var.set(max(players, key=lambda p: p.iq).name if players else "")
            if spike_var.get() not in names:
                spike_var.set(names[0] if names else "")
        self.update_scrim_start_state()

    def update_scrim_start_state(self):
        own = next((team for team in self.state.teams if team.name == self.prep_team_choice.get()), None)
        opponent = next((team for team in self.state.opponent_teams if team.name == self.opponent_choice.get()), None)
        user_without_render = not self.scrim_render.get() and "ユーザー操作" in (self.own_ai_choice.get(), self.opponent_ai_choice.get())
        contracts_ready = own is not None and all(self.state.can_play(name) for name in own.roster)
        ready = contracts_ready and opponent is not None and len(opponent.players) >= ROSTER_SIZE and not user_without_render and not self.match_running
        self.scrim_start_button.configure(state="normal" if ready else "disabled")
        self.scrim_cancel_button.configure(state="normal" if self.scrim_job is not None else "disabled")
        if hasattr(self, "advance_month_button"):
            self.advance_month_button.configure(state="disabled" if self.match_running else "normal")
        if hasattr(self, "competition_list"):
            self.preview_competition()
        if not self.state.opponent_teams and self.state.teams:
            self.prep_hint.set("相手チームがありません。realtime_season_teams.py で設定し、--import-season-teams で反映してください。")
        elif user_without_render:
            self.prep_hint.set("ユーザー操作を使う場合は描画ありを選んでください。")
        elif own and not contracts_ready:
            self.prep_hint.set("契約が終了した選手がいます。契約状況画面で再契約するか、チームを編成し直してください。")
        elif opponent is not None and len(opponent.players) < ROSTER_SIZE:
            self.prep_hint.set("相手チームは所属選手が5人未満です。所属設定で選手を補充してください。")

    def start_scrim(self):
        if self.match_running:
            return
        own = next((team for team in self.state.teams if team.name == self.prep_team_choice.get()), None)
        opponent = next((team for team in self.state.opponent_teams if team.name == self.opponent_choice.get()), None)
        try:
            options = ai_options()
            request = build_scrim_request(
                self.state, own.id if own else None, opponent.id if opponent else None,
                render=self.scrim_render.get(), own_ai=options[self.own_ai_choice.get()],
                opponent_ai=options[self.opponent_ai_choice.get()],
                initial_side="A" if self.scrim_side.get() == "攻撃" else "D",
                own_igl=self.own_igl.get(), opponent_igl=self.opponent_igl.get(),
                own_spike=self.own_spike.get(), opponent_spike=self.opponent_spike.get(),
                tick_time_ms=self.scrim_tick_ms.get(),
            )
            if not self.commit(self.state.with_selected_team(own.id), f"使用チームを「{own.name}」に設定しました。"):
                return
            self.scrim_job = ScrimJob(request, self.store.path.parent / "scrims")
            self._scrim_rating_context = (f"scrim:{self.scrim_job.directory.name}", own.id, opponent.id)
        except (OSError, ValueError, KeyError) as exc:
            self.scrim_result.set(f"試合を開始できません: {exc}")
            return
        mode = "通常の試合画面" if request["render"] else "描画なしで高速シミュレーション"
        self.scrim_result.set(f"{own.name} vs {opponent.name} — {mode}で実行中")
        self.update_scrim_start_state()
        self._scrim_after_id = self.root.after(100, self.poll_scrim)

    def poll_scrim(self):
        self._scrim_after_id = None
        if self.scrim_job is None:
            return
        result = self.scrim_job.poll()
        if result is None:
            self._scrim_after_id = self.root.after(100, self.poll_scrim)
            return
        if result.get("status") == "completed":
            if self._scrim_rating_context is not None:
                try:
                    result_id, own_id, opponent_id = self._scrim_rating_context
                    won = result["winner"] == result["own_team"]
                    candidate = self.state.with_rated_result(result_id, own_id, opponent_id, int(won), int(not won))
                    if not self.commit(candidate, "スクリム結果をレートに反映しました。"):
                        self._scrim_after_id = self.root.after(1000, self.poll_scrim)
                        return
                except (SeasonSaveError, KeyError) as exc:
                    self.scrim_result.set(f"スクリムのレートを更新できません: {exc}")
                    self.scrim_job = None
                    self.refresh()
                    return
            self.scrim_result.set(f"スクリム終了: {result['own_team']} {result['own_score']} - {result['opponent_score']} {result['opponent_team']}  勝者: {result['winner']}")
        elif result.get("status") == "cancelled":
            self.scrim_result.set("スクリムを中止しました。")
        else:
            self.scrim_result.set(f"スクリムでエラーが発生しました: {result.get('message', '詳細不明')}")
        self.scrim_job = None
        self._scrim_rating_context = None
        self.update_scrim_start_state()

    def cancel_scrim(self):
        if self.scrim_job is not None:
            self.scrim_job.cancel()
            self.scrim_result.set("スクリムを中止しています。")
            self.scrim_cancel_button.configure(state="disabled")

    def confirm_preparation(self):
        team = next((team for team in self.state.teams if team.name == self.prep_team_choice.get()), None)
        if team:
            self.commit(self.state.with_selected_team(team.id), f"使用チームを「{team.name}」に設定しました。")

    def new_team(self):
        if self.save_preset_name() and self.commit(self.state.with_new_team(), "新しいプリセットの編成を開始しました。"):
            self.preset_name.set(self.state.preset_name)

    def edit_saved_team(self, _event=None):
        team = next((team for team in self.state.teams if team.name == self.edit_team_choice.get()), None)
        if team and self.save_preset_name() and self.commit(self.state.with_editing_team(team.id), f"「{team.name}」の編成を読み込みました。"):
            self.preset_name.set(self.state.preset_name)

    def refresh(self):
        self.refresh_ratings()
        self.refresh_monthly_events()
        self.edit_team_menu.configure(values=[team.name for team in self.state.teams], state="readonly" if self.state.teams else "disabled")
        editing = self.state.team(self.state.editing_team_id)
        self.edit_team_choice.set(editing.name if editing else "")
        self.refresh_home()
        self.refresh_preparation()
        self.role_filter.configure(values=["すべて"] + sorted({p.role for p in self.state.owned_players}))
        if self.role.get() != "すべて" and self.role.get() not in {p.role for p in self.state.owned_players}:
            self.role.set("すべて")
        self.refresh_players()
        self.roster.delete(*self.roster.get_children())
        for index in range(ROSTER_SIZE):
            name = self.state.roster[index] if index < len(self.state.roster) else ""
            player = self.state.player(name)
            self.roster.insert("", "end", iid=str(index), values=(index + 1, name or "未登録", player.role if player else ""))
        count = len(self.state.roster)
        self.summary.set(f"{count} / {ROSTER_SIZE}人  —  " + ("編成完了" if self.state.roster_ready else f"あと{ROSTER_SIZE - count}人"))
        self.update_buttons()
        self.refresh_management()
        self.refresh_competitions()
        self.refresh_starters()

    def refresh_players(self):
        selected = self.players.selection()
        self.players.delete(*self.players.get_children())
        query = self.search.get().strip().casefold()
        for index, player in enumerate(self.state.owned_players):
            if query not in player.name.casefold() or self.role.get() not in ("すべて", player.role):
                continue
            owner = next((team for team in self.state.teams if player.name in team.roster), None)
            affiliation = "編成中" if player.name in self.state.roster else self.state.team_name
            self.players.insert("", "end", iid=str(index), values=(player.name, player.role, f"{player.iq:g}", affiliation))
        if selected and self.players.exists(selected[0]):
            self.players.selection_set(selected[0])
        self.select_player()

    def selected_player(self):
        selection = self.players.selection()
        return self.state.owned_players[int(selection[0])] if selection else None

    def select_player(self, _event=None):
        player = self.selected_player()
        if player is None:
            self.details.set("所持選手を選ぶと能力を表示します。")
        else:
            owner = next((team for team in self.state.teams if player.name in team.roster), None)
            self.details.set(
                f"{player.name}  /  {player.role}\n所属: {self.state.team_name}\n\n"
                f"HS率: {player.hs_pct:.0%}    命中率: {player.hit_pct:.0%}\n"
                f"回避率: {player.dodge_pct:.0%}    反応: {player.reaction:g}\n"
                f"IQ: {player.iq:g}    影響力: {player.influence:g}\n"
                f"メンタル: {player.mental:g}    調子の波: {player.form_variance:g}\n"
                f"基本月給: {player.monthly_salary:,}円    忠誠心: {player.loyalty:g}"
            )
        self.update_buttons()

    def update_buttons(self):
        player = self.selected_player()
        can_add = player is not None and player.name not in self.state.roster and not self.state.roster_ready and self.state.roster_owner(player.name) is None
        self.add_button.configure(state="normal" if can_add else "disabled")
        can_delete = player is not None and player.name not in self.state.roster and not self.state.player_in_saved_team(player.name)
        self.delete_button.configure(state="normal" if can_delete else "disabled")
        selection = self.roster.selection()
        can_remove = bool(selection) and int(selection[0]) < len(self.state.roster)
        self.remove_button.configure(state="normal" if can_remove else "disabled")
        self.confirm_button.configure(state="normal" if self.state.roster_ready else "disabled")

    def commit(self, candidate, message):
        try:
            self.store.save(candidate)
        except (OSError, SeasonSaveError) as exc:
            self.status.set("保存に失敗しました。もう一度操作してください。")
            messagebox.showerror("保存エラー", str(exc), parent=self.root)
            return False
        self.state = candidate
        self.status.set(f"{message}  自動保存済み ({datetime.now().astimezone():%H:%M:%S})")
        self.refresh()
        return True

    def add_player(self):
        player = self.selected_player()
        if player and player.name not in self.state.roster and not self.state.roster_ready:
            owner = self.state.roster_owner(player.name)
            if owner:
                self.status.set(f"{player.name}は「{owner.name}」に所属しているため、別チームに編成できません。")
                return
            self.commit(self.state.with_roster((*self.state.roster, player.name)), f"{player.name}を登録しました。")

    def register_players(self):
        if self.state.starter_candidates:
            self.status.set("初期キャラは選択画面で5人を入手します。追加の選手はスカウトから獲得してください。")
            return
        names = [name.strip() for name in re.split(r"[,、，\n\r]+", self.name_input.get("1.0", "end")) if name.strip()]
        if not names:
            self.status.set("追加するプレイヤー名を入力してください。")
            return
        try:
            candidate = self.state.with_added_players(names)
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        if self.commit(candidate, f"{len(names)}人を所持選手に追加しました。"):
            self.name_input.delete("1.0", "end")

    def delete_owned_player(self):
        player = self.selected_player()
        if player and player.name not in self.state.roster and not self.state.player_in_saved_team(player.name):
            self.players.selection_remove(*self.players.selection())
            self.commit(self.state.without_player(player.name), f"{player.name}を所持選手から外しました。")

    def remove_player(self):
        selection = self.roster.selection()
        if not selection or int(selection[0]) >= len(self.state.roster):
            return
        index = int(selection[0])
        name = self.state.roster[index]
        self.commit(self.state.with_roster(self.state.roster[:index] + self.state.roster[index + 1:]), f"{name}を控えに戻しました。")

    def save_preset_name(self):
        try:
            candidate = self.state.with_preset_name(self.preset_name.get())
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return False
        if candidate != self.state and not self.commit(candidate, "プリセット名を保存しました。"):
            return False
        self.preset_name.set(self.state.preset_name)
        return True

    def rename_club(self):
        if self.match_running:
            self.status.set("試合終了後にチーム名を変更してください。")
            return
        try:
            candidate = self.state.with_team_name(self.club_name.get())
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        self.commit(candidate, "チーム名を変更しました。編成プリセットとレートは引き継ぎます。")

    def confirm(self):
        if self.state.roster_ready and self.save_preset_name():
            try:
                candidate = self.state.with_confirmed_team()
            except SeasonSaveError as exc:
                self.status.set(str(exc))
                return
            self.commit(candidate, "5人の編成プリセットを保存しました。試合前の準備画面で選択できます。")

    def close(self):
        if self.current_screen != "editor" or self.save_preset_name():
            if self.scrim_job is not None:
                self.scrim_job.cancel()
            if self._scrim_after_id is not None:
                self.root.after_cancel(self._scrim_after_id)
            if self.competition_job is not None:
                self.competition_job.cancel()
            if self._competition_after_id is not None:
                self.root.after_cancel(self._competition_after_id)
            self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="リアルタイムシーズンのホーム・チーム編成・試合前の準備")
    parser.add_argument("--save-file", default=DEFAULT_SAVE_PATH, help="セーブファイルのパス")
    parser.add_argument("--import-season-teams", action="store_true", help="専用ファイルの他チーム所属設定をセーブに反映")
    parser.add_argument("--import-competitions", action="store_true", help="カレンダー・未登録の大会設定をセーブに反映")
    args = parser.parse_args(argv)
    root = tk.Tk()
    root.withdraw()
    store = SeasonStore(args.save_file)
    try:
        state = store.load_or_create()
        if args.import_season_teams:
            state = store.import_season_teams(state)
        if args.import_competitions:
            state = store.import_competitions(state)
    except (OSError, SeasonSaveError) as exc:
        messagebox.showerror("セーブデータを開けません", f"{exc}\n\n保存先: {store.path}\n既存のセーブデータは保持されています。", parent=root)
        root.destroy()
        return 1
    RealtimeSeasonApp(root, store, state)
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
