"""Real-time season home, team editor, and pre-match preparation."""

import argparse
from datetime import datetime
import re
import tkinter as tk
from tkinter import messagebox, ttk

from realtime_season import DEFAULT_SAVE_PATH, EXPIRED_ROSTER_WARNING, ROSTER_SIZE, SeasonSaveError, SeasonStore
from realtime_season_config import DEFAULT_TEAM_AI
from season.season_training import MAX_TRAINING_LEVEL
from season.season_scrim import ScrimJob, ai_options, build_scrim_request
from season.season_management_ui import SeasonManagementMixin
from season.season_competition_ui import SeasonCompetitionMixin
from season.season_starter_ui import SeasonStarterMixin
from season.season_rating_ui import SeasonRatingMixin
from season.season_monthly_ui import SeasonMonthlyMixin
from season.season_training_ui import SeasonTrainingMixin
from season.season_pair_familiarity_ui import SeasonPairFamiliarityMixin
from season.season_player_stats import player_combat_power, player_duel_power
from season.season_salary import SalaryMode
from season.season_profiles import SeasonProfiles
from season.season_profiles_ui import choose_season_profile
from season.season_strongest_ranking_ui import SeasonStrongestRankingWindow


class RealtimeSeasonApp(SeasonManagementMixin, SeasonCompetitionMixin, SeasonStarterMixin, SeasonRatingMixin, SeasonMonthlyMixin, SeasonTrainingMixin, SeasonPairFamiliarityMixin):
    def __init__(self, root, store, state, profiles=None):
        self.root = root
        self.store = store
        self.state = state
        self.profiles = profiles
        root.title("リアルタイムシーズン")
        root.geometry("1120x800")
        root.minsize(960, 800)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.preset_name = tk.StringVar(root, value=state.preset_name)
        self.preset_igl_choice = tk.StringVar(root)
        self.preset_carrier_choice = tk.StringVar(root)
        self.preset_ai_choice = tk.StringVar(root)
        self.club_name = tk.StringVar(root, value=state.team_name)
        self.search = tk.StringVar(root)
        self.role = tk.StringVar(root, value="すべて")
        self.status = tk.StringVar(root, value="初期キャラ候補から5人を選んでください。選択途中も自動保存します。" if state.starter_selection_pending else
                                   "変更は自動保存されます。まずプレイヤー名を入力して所持選手を設定してください。" if not state.owned_players else
                                   "セーブデータを読み込みました。変更は自動保存されます。")
        self.summary = tk.StringVar(root)
        self.details = tk.StringVar(root, value="所持選手を選ぶと能力を表示します。")
        self.editor_contract_warning = tk.StringVar(root)
        self.edit_team_choice = tk.StringVar(root)
        self.prep_team_choice = tk.StringVar(root)
        self.home_summary = tk.StringVar(root)
        self.home_payroll = tk.StringVar(root)
        self.home_selected = tk.StringVar(root)
        self.prep_hint = tk.StringVar(root)
        self.opponent_choice = tk.StringVar(root)
        default_ai_label = next(label for label, key in ai_options().items() if key == DEFAULT_TEAM_AI)
        self.own_ai_choice = tk.StringVar(root, value=default_ai_label)
        self.opponent_ai_choice = tk.StringVar(root, value=default_ai_label)
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
        self._prepared_own_settings = None
        self._scrim_rating_context = None
        self._scrim_participants = None
        self.current_screen = "home"
        self._build()
        self._build_competitions()
        self._build_home()
        self._build_preparation()
        self._build_management()
        self._build_starters()
        self._build_ratings()
        self._build_monthly_events()
        self._build_training()
        self._build_pair_familiarity()
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

        settings = ttk.Frame(host)
        settings.pack(fill="x", pady=(0, 6))
        self.preset_settings_menus = {}
        for key, label, variable, width in (
            ("igl", "IGL", self.preset_igl_choice, 18),
            ("carrier", "キャリアー", self.preset_carrier_choice, 18),
            ("ai", "AI", self.preset_ai_choice, 28),
        ):
            ttk.Label(settings, text=label).pack(side="left")
            menu = ttk.Combobox(settings, textvariable=variable, state="readonly", width=width)
            menu.pack(side="left", padx=(6, 12))
            menu.bind("<<ComboboxSelected>>", self.save_preset_settings)
            self.preset_settings_menus[key] = menu
        ttk.Label(host, text="設定は自動保存。自動選択のIGLはIQ最大、キャリアーはロスター先頭。新規・入れ替え中の編成は5人で確定してください。",
                  wraplength=1000).pack(anchor="w", pady=(0, 8))

        self.editor_contract_banner = ttk.Frame(host)
        ttk.Label(self.editor_contract_banner, textvariable=self.editor_contract_warning,
                  foreground="#b71c1c", wraplength=770).pack(side="left", fill="x", expand=True)
        self.editor_renew_button = ttk.Button(self.editor_contract_banner, text="契約状況で再契約",
                                              command=self.show_expired_contracts)
        self.editor_renew_button.pack(side="right", padx=(8, 0))

        self.editor_main = main = ttk.Frame(host)
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
        self.players.tag_configure("contract_expired", foreground="#b71c1c", background="#ffebee")
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
        player_links = ttk.Frame(lineup)
        player_links.pack(anchor="w", pady=6)
        ttk.Button(player_links, text="相棒TOP5", command=lambda: self.show_player_pairs(
            self.selected_player().name) if self.selected_player() else None).pack(side="left")
        ttk.Button(player_links, text="チーム別忠誠", command=lambda: self.show_player_loyalties(
            self.selected_player().name) if self.selected_player() else None).pack(side="left", padx=(8, 0))
        ttk.Button(lineup, text="自チームのスタメン練度", command=lambda: self.show_team_pairs(self.state.club_id)).pack(anchor="w")
        self.confirm_button = ttk.Button(lineup, text="この5人で編成を確定", command=self.confirm)
        self.confirm_button.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(host, textvariable=self.status, wraplength=1000).pack(anchor="w", pady=(12, 4))
        ttk.Label(host, text=f"保存先: {self.store.path}", wraplength=1000).pack(anchor="w")

    def _build_home(self):
        self.home_host = host = ttk.Frame(self.root, padding=18)
        ttk.Label(host, text="リアルタイムシーズン", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        navigation = ttk.Frame(host)
        navigation.pack(fill="x", pady=(4, 8))
        ttk.Label(navigation, text="ホーム", font=("Yu Gothic UI", 14)).pack(side="left")
        ttk.Entry(navigation, textvariable=self.club_name, width=24).pack(side="left", padx=8)
        ttk.Button(navigation, text="チーム名を変更", command=self.rename_club).pack(side="left")
        if self.profiles is not None:
            ttk.Button(navigation, text="チームのロード・作成・削除", command=self.manage_profiles).pack(side="left", padx=8)
        ttk.Button(navigation, text="レーティング・スポンサー", command=lambda: self.show_screen("ratings")).pack(side="right")
        ttk.Button(navigation, text="月次イベント", command=lambda: self.show_screen("monthly")).pack(side="right", padx=8)
        self._build_calendar_home(host)
        ttk.Label(host, textvariable=self.home_summary, font=("Yu Gothic UI", 12)).pack(anchor="w", pady=(0, 4))
        self.home_payroll_label = ttk.Label(host, textvariable=self.home_payroll, foreground="#c62828",
                                           font=("Yu Gothic UI", 12, "bold"))
        self.home_payroll_label.pack(anchor="w", pady=(0, 8))
        self.transfer_banner = tk.Button(host, command=self.show_transfer_offers,
            font=("Yu Gothic UI", 16, "bold"), foreground="#b71c1c", background="#ffebee",
            activebackground="#ffcdd2", wraplength=950, cursor="hand2", relief="solid", borderwidth=1)
        self.home_actions = actions = ttk.Frame(host)
        actions.pack(fill="x", pady=(0, 12))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        for index, (text, hint, command) in enumerate((
            ("チーム編成", "所持選手から5人の編成プリセットを保存", self.show_editor),
            ("スクリム", "相手と使用する編成を選んで練習試合", self.show_preparation),
            ("スカウト", "LFTとの契約・他チームからの引き抜き", lambda: self.show_screen("scout")),
            ("契約状況", "契約の経過・忠誠を確認して再契約", lambda: self.show_screen("contracts")),
            ("研究", "選手のIQと研究レベルを強化", lambda: self.show_screen("research")),
            ("エイムラボ", "選手の命中率とエイムラボレベルを強化", lambda: self.show_screen("aim_lab")),
        )):
            card = ttk.LabelFrame(actions, text=text, padding=10)
            card.grid(row=index // 3, column=index % 3, sticky="nsew", padx=(0, 12), pady=(0, 8))
            ttk.Label(card, text=hint, wraplength=280).pack(anchor="w", pady=(0, 12))
            ttk.Button(card, text=f"{text}へ", command=command).pack(fill="x")
        pair_navigation = ttk.Frame(host)
        pair_navigation.pack(fill="x", pady=(0, 8))
        ttk.Label(pair_navigation, text="シーズンの所属チーム", font=("Yu Gothic UI", 13, "bold")).pack(side="left")
        ttk.Button(pair_navigation, text="名コンビ TOP20", command=self.show_pair_ranking).pack(side="right")
        ttk.Button(pair_navigation, text="選択チームの練度", command=self.show_selected_team_pairs).pack(side="right", padx=8)
        ttk.Button(pair_navigation, text="最強ランキング", command=self.show_strongest_ranking).pack(side="right")
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
        ttk.Label(host, textvariable=self.home_selected, font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", pady=(12, 6))
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
                 "starter": self.starter_host, "ratings": self.ratings_host, "monthly": self.monthly_host,
                 **self.training_hosts}
        for host in hosts.values():
            host.pack_forget()
        hosts[screen].pack(fill="both", expand=True)
        self.current_screen = screen
        title = {"home": "ホーム", "editor": "チーム編成", "preparation": "スクリム準備",
                 "scout": "スカウト", "contracts": "契約状況", "competitions": "大会", "starter": "初期キャラ選択", "ratings": "レーティング", "monthly": "月次イベント",
                 "research": "研究", "aim_lab": "エイムラボ"}[screen]
        self.root.title(f"リアルタイムシーズン — {title}")
        if screen == "editor":
            self.refresh()
        elif screen == "preparation":
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
        elif screen in self.training_hosts:
            self.refresh_training()
        return True

    def show_home(self):
        return self.show_screen("home")

    def show_editor(self):
        return self.show_screen("editor")

    def show_expired_contracts(self):
        player = self.selected_player()
        name = player.name if player and not self.state.can_play(player.name) else next(
            (p.name for p in self.state.owned_players if not self.state.can_play(p.name)), None)
        if name is not None and self.show_screen("contracts"):
            self.contracts_players.selection_set(name)
            self.contracts_players.see(name)
            self.refresh_offer("contracts")
            self.status.set(f"{name}の契約状況を開きました。契約条件を確認して再契約してください。")

    def show_preparation(self):
        return self.show_screen("preparation")

    def refresh_home(self):
        offers = self.state.pending_transfer_offers
        self.home_teams.configure(height=2 if offers else 6)
        if offers:
            self.transfer_banner.configure(text=f"他チームからこのチームの選手にオファーが来ています（{len(offers)}件）\nクリックして契約状況で確認")
            self.transfer_banner.pack(fill="x", pady=(0, 12), before=self.home_actions)
        else:
            self.transfer_banner.pack_forget()
        self.club_name.set(self.state.team_name)
        self.home_payroll.set(f"今月末の月給合計: {-self.state.monthly_payroll:,}円")
        world = self.state.world_level_settings
        rank, count = self.state.world_rank
        world_note = "（大会中は固定）" if self.state.active_tournaments else ""
        self.home_summary.set(f"所持金: {self.state.money:,}円    スカウト可能回数: {self.state.scout_allowance_text}\n"
                              f"チーム: {self.state.team_name}    所持選手: {len(self.state.owned_players)}人    編成プリセット: {len(self.state.teams)}    他チーム: {len(self.state.opponent_teams)}\n"
                              f"世界レベル: {world.level}{world_note}    ランキング: {rank} / {count}位（上位{self.state.world_top_percent:.2f}%）    敵倍率: {world.enemy_multiplier:g}倍\n"
                              f"スポンサー契約: {'有効' if self.state.sponsor_active else '停止'}    月額収入: {self.state.monthly_sponsor_income:,}円    "
                              f"給与: {'成績連動（K/D）' if self.state.salary_mode == SalaryMode.KD_DYNAMIC else '静的（従来）'}")
        if self.state.day_advance_pending:
            self.home_summary.set(self.home_summary.get() + "\n完了したアクションの1日進行待ち：大会への参加登録後、日付を進めてください。")
        self.home_teams.delete(*self.home_teams.get_children())
        self.home_teams.insert("", "end", iid=self.state.club_id, values=("自分", self.state.team_name,
            f"{self.state.rating(self.state.club_id):.3f}", f"{self.state.transfer_multiplier:g}倍", " / ".join(p.name for p in self.state.owned_players)))
        for team in self.state.opponent_teams:
            self.home_teams.insert("", "end", iid=team.id, values=("他チーム", team.name, f"{self.state.rating(team.id):.3f}", f"{team.transfer_multiplier:g}倍", " / ".join(team.members) or "選手なし"))
        selected = self.state.selected_team
        self.home_selected.set(f"使用する編成: {selected.name}" if selected else "使用する編成: 未選択（スクリムの準備で選択してください）")

    def show_transfer_offers(self):
        self.show_screen("contracts")
        if self.state.pending_transfer_offers:
            name = self.state.pending_transfer_offers[0].player_name
            self.contracts_players.selection_set(name)
            self.contracts_players.focus(name)
            self.contracts_players.see(name)
            self.refresh_offer("contracts")

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
        own_players = [self.state.player(name) for name in team.roster] if team else []
        own_settings = (team.id, team.roster, team.igl or max(own_players, key=lambda p: p.iq).name,
                        team.carrier or team.roster[0], team.ai) if team else None
        if own_settings != self._prepared_own_settings:
            self._prepared_own_settings = own_settings
            self.own_igl.set(own_settings[2] if team else "")
            self.own_spike.set(own_settings[3] if team else "")
            labels = {key: label for label, key in ai_options().items()}
            self.own_ai_choice.set(labels[team.ai] if team else labels[DEFAULT_TEAM_AI])
        opponent = next((club for club in self.state.opponent_teams if club.name == self.opponent_choice.get()), None)
        opponent_settings = (opponent.id, opponent.effective_igl, opponent.effective_carrier, opponent.ai) if opponent else None
        if opponent_settings != self._prepared_opponent_settings:
            self._prepared_opponent_settings = opponent_settings
            self.opponent_igl.set(opponent.effective_igl if opponent else "")
            self.opponent_spike.set(opponent.effective_carrier if opponent else "")
            labels = {key: label for label, key in ai_options().items()}
            self.opponent_ai_choice.set(labels[opponent.ai] if opponent else labels[DEFAULT_TEAM_AI])
        self.opponent_roster.delete(*self.opponent_roster.get_children())
        if opponent:
            for index, player in enumerate(opponent.players[:ROSTER_SIZE]):
                player = self.state.enemy_player(player)
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
            if not self.commit(self.state.with_selected_team(own.id), f"使用する編成を「{own.name}」に設定しました。"):
                return
            self.scrim_job = ScrimJob(request, self.store.path.parent / "scrims")
            self._scrim_rating_context = (f"scrim:{self.scrim_job.directory.name}", own.id, opponent.id)
            self._scrim_participants = {self.state.club_id: tuple(p["name"] for p in request["own"]["players"]),
                                        opponent.id: tuple(p["name"] for p in request["opponent"]["players"])}
        except (OSError, ValueError, KeyError) as exc:
            self.scrim_result.set(f"試合を開始できません: {exc}")
            return
        mode = "通常の試合画面" if request["render"] else "描画なしで高速シミュレーション"
        self.scrim_result.set(f"{own.name} vs {opponent.name} — {mode}で実行中")
        self.update_scrim_start_state()
        self.refresh_training()
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
                    previous = self.state
                    options = {"advance_day": False} if self.state.entry_deadline_tournaments else {}
                    if self._scrim_participants is not None:
                        options["participants"] = self._scrim_participants
                    candidate = self.state.with_scrim_result(result_id, own_id, opponent_id, int(won), int(not won), **options)
                    note = "1日進行を待っています。" if candidate.day_advance_pending else f"ゲーム内{candidate.date:%Y/%m/%d}へ1日進めました。"
                    if not self.commit(candidate, "スクリム結果を反映し、" + note):
                        self._scrim_after_id = self.root.after(1000, self.poll_scrim)
                        return
                    self.status.set(self.status.get() + self.monthly_event_notice(previous))
                except (SeasonSaveError, KeyError) as exc:
                    self.scrim_result.set(f"スクリム結果・日付を更新できません: {exc}")
                    self.scrim_job = None
                    self.refresh()
                    return
            self.scrim_result.set(f"スクリム終了: {result['own_team']} {result['own_score']} - {result['opponent_score']} {result['opponent_team']}  勝者: {result['winner']} / ゲーム内 {self.state.date:%Y/%m/%d}")
        elif result.get("status") == "cancelled":
            self.scrim_result.set("スクリムを中止しました。")
        else:
            self.scrim_result.set(f"スクリムでエラーが発生しました: {result.get('message', '詳細不明')}")
        self.scrim_job = None
        self._scrim_rating_context = None
        self._scrim_participants = None
        self.refresh()
        if result.get("status") == "completed":
            self.finish_action_day()

    def cancel_scrim(self):
        if self.scrim_job is not None:
            self.scrim_job.cancel()
            self.scrim_result.set("スクリムを中止しています。")
            self.scrim_cancel_button.configure(state="disabled")

    def confirm_preparation(self):
        team = next((team for team in self.state.teams if team.name == self.prep_team_choice.get()), None)
        if team:
            self.commit(self.state.with_selected_team(team.id), f"使用する編成を「{team.name}」に設定しました。")

    def new_team(self):
        if self.save_preset_name() and self.commit(self.state.with_new_team(), "新しいプリセットの編成を開始しました。"):
            self.preset_name.set(self.state.preset_name)

    def edit_saved_team(self, _event=None):
        team = next((team for team in self.state.teams if team.name == self.edit_team_choice.get()), None)
        if team and self.save_preset_name() and self.commit(self.state.with_editing_team(team.id), f"「{team.name}」の編成を読み込みました。"):
            self.preset_name.set(self.state.preset_name)

    def refresh(self):
        from season.season_contract_endings import settle_contract_endings
        settled = settle_contract_endings(self.state)
        if settled != self.state:
            if self.commit(settled, "契約終了・再契約猶予の状況を更新しました。"):
                return
        if self.state.unplayable_roster:
            if self.commit(self.state, "契約状況に合わせて編成を更新しました。"):
                return
        self.refresh_pair_windows()
        self.refresh_preset_settings()
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
        summary = ("契約終了選手を外してください" if self.state.unplayable_roster else
                   "編成完了" if self.state.roster_ready else f"あと{ROSTER_SIZE - count}人")
        self.summary.set(f"{count} / {ROSTER_SIZE}人  —  {summary}")
        self.update_buttons()
        self.refresh_management()
        self.refresh_competitions()
        self.refresh_starters()
        self.refresh_training()
        expired = tuple(p.name for p in self.state.owned_players if not self.state.can_play(p.name))
        if expired:
            names = "、".join(expired[:3]) + (f" ほか{len(expired) - 3}人" if len(expired) > 3 else "")
            self.editor_contract_warning.set(f"契約終了: {names}\n{EXPIRED_ROSTER_WARNING}。再契約すると編成に追加できます。")
            self.editor_contract_banner.pack(fill="x", pady=(0, 8), before=self.editor_main)
        else:
            self.editor_contract_banner.pack_forget()
            self.editor_contract_warning.set("")

    def refresh_players(self):
        selected = self.players.selection()
        self.players.delete(*self.players.get_children())
        query = self.search.get().strip().casefold()
        for index, player in enumerate(self.state.owned_players):
            if query not in player.name.casefold() or self.role.get() not in ("すべて", player.role):
                continue
            owner = next((team for team in self.state.teams if player.name in team.roster), None)
            expired = not self.state.can_play(player.name)
            affiliation = "契約終了・再契約が必要" if expired else "編成中" if player.name in self.state.roster else self.state.team_name
            self.players.insert("", "end", iid=str(index), values=(player.name, player.role, f"{player.iq:g}", affiliation),
                                tags=("contract_expired",) if expired else ())
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
            self.details.set(
                f"{player.name}  /  {player.role}\n所属: {self.state.team_name}\n\n"
                f"HS率: {player.hs_pct:.0%}    命中率: {player.hit_pct:.0%}\n"
                f"回避率: {player.dodge_pct:.0%}    反応: {player.reaction:g}\n"
                f"IQ: {player.iq:g}    影響力: {player.influence:g}\n"
                f"研究Lv: {player.research_level} / {MAX_TRAINING_LEVEL}    エイムラボLv: {player.aim_lab_level} / {MAX_TRAINING_LEVEL}\n"
                f"総合戦闘力: {player_combat_power(player):.2f}\n"
                f"撃ち合い戦闘力: {player_duel_power(player):.2f}\n"
                f"メンタル: {player.mental:g}    調子の波: {player.form_variance:g}\n"
                f"基本月給: {player.monthly_salary:,}円    忠誠心: {player.loyalty:g}"
            )
            if not self.state.can_play(player.name):
                self.details.set(self.details.get() + f"\n\n{EXPIRED_ROSTER_WARNING}。\n契約状況で再契約してください。")
        self.update_buttons()

    def update_buttons(self):
        player = self.selected_player()
        can_add = player is not None and self.state.can_play(player.name) and player.name not in self.state.roster and not self.state.roster_ready and self.state.roster_owner(player.name) is None
        self.add_button.configure(state="normal" if can_add else "disabled")
        can_delete = player is not None and player.name not in self.state.roster and not self.state.player_in_saved_team(player.name)
        self.delete_button.configure(state="normal" if can_delete else "disabled")
        selection = self.roster.selection()
        can_remove = bool(selection) and int(selection[0]) < len(self.state.roster)
        self.remove_button.configure(state="normal" if can_remove else "disabled")
        self.confirm_button.configure(state="normal" if self.state.roster_ready and not self.state.unplayable_roster else "disabled")

    def commit(self, candidate, message):
        try:
            from season.season_contract_endings import settle_contract_endings
            candidate = settle_contract_endings(candidate)
            expired = candidate.unplayable_roster
            if expired:
                candidate = candidate.with_playable_roster()
                message += f" {'、'.join(expired)}: {EXPIRED_ROSTER_WARNING}。ロスターから自動で外しました。契約状況で再契約してください。"
            self.store.save(candidate)
        except (OSError, SeasonSaveError) as exc:
            self.status.set("保存に失敗しました。もう一度操作してください。")
            messagebox.showerror("保存エラー", str(exc), parent=self.root)
            return False
        old_forced = {offer.id for offer in self.state.transfer_offers if offer.status == "forced"}
        old_news = {e.id for e in self.state.pair_news}
        new_news = [e for e in candidate.pair_news if e.id not in old_news]
        if new_news:
            message += f" ペアニュース{len(new_news)}件。「月次イベント」で確認できます。"
        forced = [offer for offer in candidate.transfer_offers if offer.status == "forced" and offer.id not in old_forced]
        for offer in forced:
            club = next(c for c in candidate.opponent_teams if c.id == offer.team_id)
            message += f" 忠誠が30未満になり、{offer.player_name}の{club.name}への移籍が強制成立しました（移籍金{offer.fee:,}円）。編成を確認してください。"
        if candidate.world_level != self.state.world_level:
            world = candidate.world_level_settings
            message += f" 世界レベルが{self.state.world_level}から{world.level}になりました。敵倍率{world.enemy_multiplier:g}倍・月額スポンサー資金{world.sponsor_funds:,}円。"
        if self.store.history_export_error:
            message += f" 履歴ファイルの出力に失敗しました: {self.store.history_export_error}。次の保存で再出力します。"
        self.state = candidate
        self.status.set(f"{message}  自動保存済み ({datetime.now().astimezone():%H:%M:%S})")
        self.refresh()
        return True

    def add_player(self):
        player = self.selected_player()
        if player is not None and not self.state.can_play(player.name):
            self.status.set(f"{player.name}: {EXPIRED_ROSTER_WARNING}。契約状況で再契約してください。")
            self.refresh()
            return
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

    def refresh_preset_settings(self):
        names = ("自動", *self.state.roster)
        for key in ("igl", "carrier"):
            self.preset_settings_menus[key].configure(values=names, state="readonly" if self.state.roster else "disabled")
        options = ai_options()
        self.preset_settings_menus["ai"].configure(values=tuple(options))
        self.preset_igl_choice.set(self.state.preset_igl or "自動")
        self.preset_carrier_choice.set(self.state.preset_carrier or "自動")
        self.preset_ai_choice.set(next(label for label, key in options.items() if key == self.state.preset_ai))

    def save_preset_settings(self, _event=None):
        try:
            candidate = self.state.with_preset_settings(
                igl=None if self.preset_igl_choice.get() == "自動" else self.preset_igl_choice.get(),
                carrier=None if self.preset_carrier_choice.get() == "自動" else self.preset_carrier_choice.get(),
                ai=ai_options()[self.preset_ai_choice.get()])
        except (SeasonSaveError, KeyError) as exc:
            self.status.set(str(exc))
            self.refresh_preset_settings()
            return False
        if candidate != self.state and not self.commit(candidate, "プリセットのIGL・キャリアー・AIを保存しました。"):
            self.refresh_preset_settings()
            return False
        return True

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

    def show_strongest_ranking(self):
        window = getattr(self, "_strongest_window", None)
        if window is not None and window.winfo_exists():
            window.lift()
            window.focus_set()
            return window
        self._strongest_window = SeasonStrongestRankingWindow(self.root, self.store, self.state.team_name)
        return self._strongest_window

    def manage_profiles(self):
        if self.match_running:
            self.status.set("試合終了後にチームを切り替えてください。")
            return
        if self.current_screen == "editor" and not self.save_preset_name():
            return
        selected = choose_season_profile(self.root, self.profiles, self.store.path)
        if selected is None:
            return
        store, state = selected
        for after_id in (self._scrim_after_id, self._competition_after_id):
            if after_id is not None:
                self.root.after_cancel(after_id)
        # Rebuild every screen and close detail windows so no callback retains
        # the previous team's state or prepared match settings.
        for child in self.root.winfo_children():
            child.destroy()
        self.__init__(self.root, store, state, self.profiles)

    def rename_club(self):
        if self.match_running:
            self.status.set("試合終了後にチーム名を変更してください。")
            return
        try:
            candidate = self.state.with_team_name(self.club_name.get())
            if getattr(self, "profiles", None) is not None and any(
                profile.path != self.store.path and profile.name.casefold() == candidate.team_name.casefold()
                for profile in self.profiles.list()
            ):
                raise SeasonSaveError("同じ名前のチームが既にあります。")
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        self.commit(candidate, "チーム名を変更しました。編成プリセットとレートは引き継ぎます。")

    def confirm(self):
        if self.state.unplayable_roster:
            self.refresh()
            return
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
    parser.add_argument("--save-file", help="セーブファイルを直接指定（省略時はチーム選択画面）")
    parser.add_argument("--import-season-teams", action="store_true", help="専用ファイルの他チーム所属設定をセーブに反映")
    parser.add_argument("--import-competitions", action="store_true", help="カレンダー・未登録の大会設定をセーブに反映")
    args = parser.parse_args(argv)
    root = tk.Tk()
    root.withdraw()
    profiles = SeasonProfiles()
    store = SeasonStore(args.save_file or DEFAULT_SAVE_PATH)
    try:
        if args.save_file:
            state = store.load_or_create()
        else:
            selected = choose_season_profile(root, profiles)
            if selected is None:
                root.destroy()
                return 0
            store, state = selected
        if args.import_season_teams:
            state = store.import_season_teams(state)
        if args.import_competitions:
            state = store.import_competitions(state)
    except (OSError, SeasonSaveError) as exc:
        messagebox.showerror("セーブデータを開けません", f"{exc}\n\n保存先: {store.path}\n既存のセーブデータは保持されています。", parent=root)
        root.destroy()
        return 1
    RealtimeSeasonApp(root, store, state, profiles)
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
