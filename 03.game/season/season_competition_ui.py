"""Season calendar badge, tournament entry, real series execution, and standings."""

import tkinter as tk
from tkinter import messagebox, ttk
from datetime import timedelta

from season.season_competitions import SeriesScore, next_match, parse_date, player_eliminated
from season.season_bracket_ui import SeasonBracketPanel
from season.season_scrim import ScrimJob
from season.season_series import build_series_request
from season.season_ratings import expected_score


STAGES = {"upper": "Upper", "lower": "Lower", "lower_final": "Lower Final", "grand_final": "Grand Final"}
CURRENT_ROSTER = "現在編集中の編成（不足枠は友達）"


class SeasonCompetitionMixin:
    @property
    def match_running(self):
        return self.scrim_job is not None or getattr(self, "competition_job", None) is not None

    def _build_competitions(self):
        self.competition_job = None
        self._competition_after_id = None
        self._competition_event_id = None
        self._competition_match = None
        self.calendar_label = tk.StringVar(self.root)
        self.phase_label = tk.StringVar(self.root)
        self.competition_info = tk.StringVar(self.root)
        self.competition_status = tk.StringVar(self.root, value="大会を選択して参加条件を確認してください。")
        self.competition_team = tk.StringVar(self.root)
        self.competition_roster_summary = tk.StringVar(self.root, value="出場ロスター：参加登録後に表示します。")
        self.competition_auto = tk.BooleanVar(self.root, value=False)
        self.competition_render = tk.BooleanVar(self.root, value=True)
        self.competition_tick_ms = tk.StringVar(self.root, value="100")
        self.competition_host = host = ttk.Frame(self.root, padding=20)
        ttk.Label(host, text="大会", font=("Yu Gothic UI", 22, "bold")).pack(anchor="w")
        bar = ttk.Frame(host)
        bar.pack(fill="x", pady=8)
        ttk.Button(bar, text="ホームに戻る", command=self.show_home).pack(side="left")
        ttk.Label(bar, textvariable=self.calendar_label, font=("Yu Gothic UI", 12, "bold")).pack(side="left", padx=16)
        ttk.Label(bar, textvariable=self.phase_label).pack(side="left")
        self.competition_next_day_button = ttk.Button(bar, text="1日進める", command=lambda: self.advance_calendar(1))
        self.competition_next_day_button.pack(side="right")
        self.competition_list = ttk.Treeview(host, columns=("name", "date", "count", "entry", "state"), show="headings", height=4, selectmode="browse")
        for key, label, width in (("name", "大会", 200), ("date", "開始日 ～ 終了予定日", 220), ("count", "参加数", 65),
                                  ("entry", "参加設定", 100), ("state", "状態", 150)):
            self.competition_list.heading(key, text=label)
            self.competition_list.column(key, width=width, minwidth=45)
        self.competition_list.pack(fill="x")
        self.competition_list.bind("<<TreeviewSelect>>", lambda _: self.preview_competition())
        ttk.Label(host, textvariable=self.competition_info, wraplength=900, justify="left").pack(anchor="w", pady=8)
        entry = ttk.Frame(host)
        entry.pack(fill="x", pady=(0, 8))
        ttk.Label(entry, text="参加する編成").pack(side="left")
        self.competition_team_menu = ttk.Combobox(entry, textvariable=self.competition_team, state="readonly", width=28)
        self.competition_team_menu.pack(side="left", padx=8)
        self.competition_enter_button = ttk.Button(entry, text="参加登録", command=self.enter_competition)
        self.competition_enter_button.pack(side="left")
        self.competition_decline_button = ttk.Button(entry, text="今回は参加しない", command=self.decline_competition)
        self.competition_decline_button.pack(side="left", padx=8)
        self.competition_entry_cancel_button = ttk.Button(entry, text="参加をキャンセル", command=self.cancel_competition_entry)
        self.competition_entry_cancel_button.pack(side="left", padx=8)
        ttk.Label(host, text="自分のAI・IGL・キャリアーは参加する編成の設定を使います。出場可能な選手が5人未満でも、不足枠を「友達」で補って参加できます。",
                  wraplength=900).pack(anchor="w", pady=(0, 8))
        ttk.Label(host, textvariable=self.competition_roster_summary, wraplength=950,
                  font=("Yu Gothic UI", 10, "bold")).pack(anchor="w", pady=(0, 6))
        self.competition_tabs = ttk.Notebook(host)
        self.competition_tabs.pack(fill="both", expand=True)
        self.competition_bracket = SeasonBracketPanel(self.competition_tabs, self.competition_status.set)
        self.competition_tabs.add(self.competition_bracket, text="トーナメント表")
        roster_tab = ttk.Frame(self.competition_tabs, padding=10)
        self.competition_tabs.add(roster_tab, text="出場ロスター")
        ttk.Label(roster_tab, text="出場する選手を5人まで選択してください（Ctrl・Shiftで複数選択）。不足する枠は初期能力の「友達」で補充します。",
                  wraplength=900).pack(anchor="w", pady=(0, 6))
        roster_list = ttk.Frame(roster_tab)
        roster_list.pack(fill="both", expand=True)
        self.competition_roster_players = ttk.Treeview(roster_list, columns=("name", "role", "iq"),
                                                       show="headings", selectmode="extended", height=5)
        for key, label in (("name", "出場可能な所持選手"), ("role", "ロール"), ("iq", "IQ")):
            self.competition_roster_players.heading(key, text=label)
        self.competition_roster_players.pack(side="left", fill="both", expand=True)
        roster_scroll = ttk.Scrollbar(roster_list, command=self.competition_roster_players.yview)
        roster_scroll.pack(side="right", fill="y")
        self.competition_roster_players.configure(yscrollcommand=roster_scroll.set)
        roster_actions = ttk.Frame(roster_tab)
        roster_actions.pack(fill="x", pady=(8, 0))
        self.competition_roster_apply_button = ttk.Button(roster_actions, text="選択した選手で出場", command=self.apply_competition_roster)
        self.competition_roster_apply_button.pack(side="left")
        self.competition_roster_preset_button = ttk.Button(roster_actions, text="上で選んだ編成を適用", command=self.apply_competition_preset)
        self.competition_roster_preset_button.pack(side="left", padx=8)
        result_tab = ttk.Frame(self.competition_tabs)
        self.competition_tabs.add(result_tab, text="結果・順位")
        self.competition_matches = ttk.Treeview(result_tab, columns=("stage", "teams", "score"), show="headings", height=6)
        for key, label, width in (("stage", "ステージ", 130), ("teams", "対戦 / 順位", 500), ("score", "結果 / 賞金", 170)):
            self.competition_matches.heading(key, text=label)
            self.competition_matches.column(key, width=width, minwidth=50)
        self.competition_matches.pack(fill="both", expand=True)
        settings = ttk.Frame(host)
        settings.pack(fill="x", pady=8)
        ttk.Checkbutton(settings, text="自分の試合を描画する", variable=self.competition_render).pack(side="left")
        ttk.Checkbutton(settings, text="日付を進めて連続実行", variable=self.competition_auto).pack(side="left", padx=8)
        ttk.Label(settings, text="1tick (ms)").pack(side="left")
        ttk.Entry(settings, textvariable=self.competition_tick_ms, width=6).pack(side="left", padx=6)
        self.competition_play_button = ttk.Button(settings, text="本日のシリーズを開始", command=self.start_competition_series)
        self.competition_play_button.pack(side="left", padx=8)
        self.competition_cancel_button = ttk.Button(settings, text="中止", command=self.cancel_competition_series)
        self.competition_cancel_button.pack(side="left")
        self.competition_forfeit_button = ttk.Button(settings, text="このシリーズを棄権", command=self.forfeit_competition_series)
        self.competition_forfeit_button.pack(side="left", padx=8)
        ttk.Label(host, textvariable=self.competition_status, wraplength=900).pack(anchor="w", pady=4)
        ttk.Label(host, textvariable=self.status, wraplength=900).pack(anchor="w")

    def _build_calendar_home(self, host):
        bar = ttk.Frame(host)
        bar.pack(fill="x", pady=(4, 12))
        # Native Tk vector drawing keeps the season icon crisp without image assets.
        self.phase_icon = tk.Canvas(bar, width=36, height=36, highlightthickness=0, background="#f0f0f0")
        self.phase_icon.pack(side="left")
        ttk.Label(bar, textvariable=self.calendar_label, font=("Yu Gothic UI", 16, "bold")).pack(side="left", padx=10)
        self.phase_badge = tk.Label(bar, textvariable=self.phase_label, font=("Yu Gothic UI", 11, "bold"), padx=12, pady=6)
        self.phase_badge.pack(side="left", padx=8)
        ttk.Button(bar, text="1日進める", command=lambda: self.advance_calendar(1)).pack(side="right")
        ttk.Button(bar, text="次の大会へ", command=self.advance_to_competition).pack(side="right", padx=8)
        ttk.Button(bar, text="大会へ", command=lambda: self.show_screen("competitions")).pack(side="right", padx=8)

    def refresh_competitions(self):
        self.calendar_label.set(f"ゲーム内 {self.state.date:%Y/%m/%d}")
        active = self.state.phase == "in_season"
        self.phase_label.set("インシーズン" if active else "オフシーズン")
        self.phase_badge.configure(bg="#d7f2df" if active else "#dce8f8", fg="#176039" if active else "#284c7b")
        icon = self.phase_icon
        icon.delete("all")
        color = "#238349" if active else "#426b9c"
        if active:
            icon.create_polygon(10, 5, 26, 5, 24, 20, 18, 25, 12, 20, fill=color, outline="")
            icon.create_arc(4, 6, 15, 21, start=90, extent=180, style="arc", outline=color, width=3)
            icon.create_arc(21, 6, 32, 21, start=270, extent=180, style="arc", outline=color, width=3)
            icon.create_line(18, 24, 18, 30, width=3, fill=color)
            icon.create_line(11, 31, 25, 31, width=3, fill=color)
        else:
            icon.create_oval(5, 5, 31, 31, fill=color, outline="")
            icon.create_rectangle(13, 11, 16, 25, fill="white", outline="")
            icon.create_rectangle(20, 11, 23, 25, fill="white", outline="")
        selected = self.competition_list.selection()
        self.competition_list.delete(*self.competition_list.get_children())
        for event in sorted(self.state.visible_tournaments, key=lambda event: parse_date(event.start_date)):
            run = self.state.tournament(event.id)
            status = "不参加" if run and run.declined else "完了" if run and run.completed else "参加登録済み" if run else "未登録"
            if self.state.date > parse_date(event.start_date) and run is None:
                status = "参加受付終了"
            elif run and not run.completed and self.state.date >= parse_date(event.start_date):
                status = "本日試合済み" if run.last_match_date == self.state.date.isoformat() else "本日の試合待ち"
            elif run and run.completed_date:
                status = f"終了 {run.completed_date}"
            if run and run.own_team_id is None and not run.declined:
                status = "自チーム不参加・" + status
            entry = "参加不可" if not event.allow_player_entry else "任意参加" if event.participation_optional else "強制参加"
            count = len(run.entrants) if run and run.entrants else event.team_count
            matches = 2 * count - 2 if event.format == "double_elimination" else count - 1
            end_date = parse_date(event.start_date) + timedelta(days=matches - 1)
            self.competition_list.insert("", "end", iid=event.id, values=(event.display_name,
                f"{event.start_date} ～ {end_date}", count, entry, status))
        if selected and self.competition_list.exists(selected[0]):
            self.competition_list.selection_set(selected[0])
        choices = [team.name for team in self.state.teams] + [CURRENT_ROSTER]
        self.competition_team_menu.configure(values=choices)
        if self.competition_team.get() not in choices:
            self.competition_team.set(self.state.selected_team.name if self.state.selected_team else (self.state.teams[0].name if self.state.teams else CURRENT_ROSTER))
        self.preview_competition()

    def selected_competition(self):
        selected = self.competition_list.selection()
        return self.state.tournament_definition(selected[0]) if selected else None

    def preview_competition(self):
        event = self.selected_competition()
        self.competition_bracket.show(event, self.state.tournament(event.id) if event else None)
        self.competition_matches.delete(*self.competition_matches.get_children())
        self.competition_cancel_button.configure(state="normal" if self.competition_job or self._competition_after_id else "disabled")
        self.competition_next_day_button.configure(state="disabled" if self.match_running else "normal")
        for button in (self.competition_enter_button, self.competition_decline_button, self.competition_play_button, self.competition_forfeit_button):
            button.configure(state="disabled")
        self.competition_entry_cancel_button.configure(state="disabled")
        self.competition_roster_apply_button.configure(state="disabled")
        self.competition_roster_preset_button.configure(state="disabled")
        self.competition_roster_players.delete(*self.competition_roster_players.get_children())
        self.competition_roster_summary.set("出場ロスター：参加登録後に表示します。")
        if event is None:
            self.competition_info.set("出現中の大会を選択してください。大会は専用Pythonファイルから設定できます。")
            return
        run = self.state.tournament(event.id)
        if run and run.own_team_id is not None and not run.declined:
            own = self.state.tournament_team(event.id)
            names = tuple(p.name for p in own.players)
            self.competition_roster_summary.set("出場ロスター：" + " / ".join(names)
                + f"（IGL: {own.igl} / キャリアー: {own.carrier}）")
            editable = not run.completed and not self.match_running and not player_eliminated(event, run)
            if editable:
                self.competition_roster_apply_button.configure(state="normal")
                self.competition_roster_preset_button.configure(state="normal")
                opponents = {p.name for t in run.entrants if t.id != run.own_team_id for p in t.players}
                for player in self.state.owned_players:
                    if self.state.can_play(player.name) and player.name not in opponents:
                        self.competition_roster_players.insert("", "end", iid=player.name,
                            values=(player.name, player.role, f"{player.iq:g}"))
                self.competition_roster_players.selection_set(tuple(
                    name for name in names if self.competition_roster_players.exists(name)))
            if not run.completed and not run.results and self.state.date < parse_date(event.start_date) and not self.match_running:
                self.competition_entry_cancel_button.configure(state="normal")
        prizes = " / ".join(f"{rank}位 {money:,}円" for rank, money in sorted(event.prizes.items())) or "賞金なし"
        count = len(run.entrants) if run and run.entrants else event.team_count
        matches = 2 * count - 2 if event.format == "double_elimination" else count - 1
        end_date = parse_date(event.start_date) + timedelta(days=matches - 1)
        deadline = parse_date(event.start_date)
        self.competition_info.set(f"{event.display_name}  |  {'ダブル' if event.format == 'double_elimination' else 'シングル'}エリミネーション\n"
            f"大会の敵ステータス倍率: {event.enemy_multiplier:g}倍（世界レベル・ペア練度の倍率と乗算）\n"
            f"先取マップ数: 通常{event.normal_maps_to_win} / Lower Final {event.lower_final_maps_to_win} / Grand Final {event.grand_final_maps_to_win}\n賞金: {prizes} / 全{matches}試合・終了予定日 {end_date}（自動計算）\n参加登録締切: {deadline}（開始日当日まで参加可能）")
        if (run is None or run.declined) and self.state.date <= parse_date(event.start_date) and not self.match_running:
            self.competition_enter_button.configure(state="normal")
            self.competition_enter_button.configure(text="参加登録" if event.allow_player_entry else "相手チームの大会を登録")
            self.competition_decline_button.configure(state="normal" if run is None and event.allow_player_entry and event.participation_optional else "disabled")
        if run and not run.declined:
            teams = {t.id: t.name for t in run.entrants}
            for score in run.results:
                self.competition_matches.insert("", "end", values=(score.match_id,
                    f"{teams[score.left_id]} vs {teams[score.right_id]}",
                    f"{score.left_wins} - {score.right_wins}" + ("（レート判定）" if score.decided_by_rating else "")))
            pending, ranking = next_match(event, run)
            if pending:
                self.competition_matches.insert("", "end", values=(STAGES[pending.stage],
                    f"次: {teams[pending.left]} vs {teams[pending.right]}", f"{pending.maps_to_win}マップ先取"))
                ready = (parse_date(event.start_date) <= self.state.date and run.last_match_date != self.state.date.isoformat()
                         and not self.match_running)
                eliminated = player_eliminated(event, run)
                npc_match = run.own_team_id not in (pending.left, pending.right)
                self.competition_play_button.configure(text="レート判定で大会を終了" if eliminated else
                                                       "本日の結果をレート抽選" if npc_match else "本日のシリーズを開始",
                                                       state="normal" if (ready or eliminated and not self.match_running) else "disabled")
                if run.own_team_id is None:
                    self.competition_play_button.configure(text="日付進行時に自動試合", state="disabled")
                self.competition_forfeit_button.configure(state="normal" if ready and run.own_team_id in (pending.left, pending.right) else "disabled")
                self.competition_info.set(self.competition_info.get() + ("\n本日の試合は終了しました。「1日進める」で翌日の試合へ進めます。"
                    if run.last_match_date == self.state.date.isoformat() else "\n1日1試合（シリーズ単位・相手同士の試合も含む）。全試合終了時に大会が自動終了します。"))
                if eliminated:
                    self.competition_info.set(self.competition_info.get() + "\n自チームは敗退済みです。残りは1試合につき1日進め、レート勝率の抽選で確定できます。")
                if npc_match and run.own_team_id is not None:
                    probability = expected_score(self.state.rating(pending.left), self.state.rating(pending.right))
                    self.competition_info.set(self.competition_info.get() +
                        f"\n他チーム同士はマップごとにレート勝率で抽選し、結果確定後に1日進めます。各マップの予測: {teams[pending.left]} {probability:.1%} / {teams[pending.right]} {1 - probability:.1%}")
                if run.own_team_id is None:
                    count = 2 * len(run.entrants) - 2 if event.format == "double_elimination" else len(run.entrants) - 1
                    self.competition_info.set(self.competition_info.get() +
                        f"\n自チームは不参加です。ライバル{len(run.entrants)}チーム・全{count}試合を1日1試合ずつ自動進行します。スカウト・育成・スクリムも利用できます。")
            else:
                for rank, team_id in enumerate(ranking, 1):
                    self.competition_matches.insert("", "end", values=(f"{rank}位", teams[team_id], f"{event.prizes.get(rank, 0):,}円"))
                self.competition_info.set(self.competition_info.get() + f"\n受け取り済み賞金: {run.prize_paid:,}円")
                if run.completed_date:
                    self.competition_info.set(self.competition_info.get() + f" / 実際の終了日: {run.completed_date}")
                if any(score.decided_by_rating for score in run.results):
                    self.competition_info.set(self.competition_info.get() + "\nレート判定の試合はシミュレーションを省略しています。自チーム敗退後も残りを自動判定して終了します。")

    def enter_competition(self):
        event = self.selected_competition()
        if event is None or self.match_running:
            return
        own = (None if self.competition_team.get() == CURRENT_ROSTER else
               next((t for t in self.state.teams if t.name == self.competition_team.get()), None))
        try:
            candidate = self.state.with_tournament_entry(event.id, own.id if own else None)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        if candidate.tournament(event.id) is None:
            self.status.set("出場可能な相手チームがいないため、参加登録を待っています。選手が5人揃ったチームが必要です。")
            return
        self.commit(candidate, f"{event.display_name}に参加登録しました。開催日から試合を進められます。")

    def decline_competition(self):
        event = self.selected_competition()
        if event is None or self.match_running:
            return
        try:
            candidate = self.state.with_declined_tournament(event.id)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.commit(candidate, f"{event.display_name}は不参加にしました。")

    def cancel_competition_entry(self):
        event = self.selected_competition()
        if event is None or self.match_running:
            return
        try:
            candidate = self.state.with_cancelled_tournament_entry(event.id)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.competition_auto.set(False)
        if self._competition_after_id is not None:
            self.root.after_cancel(self._competition_after_id)
            self._competition_after_id = None
        self.commit(candidate, f"{event.display_name}の参加をキャンセルしました。開始日まで再登録できます。")

    def apply_competition_roster(self):
        self._apply_competition_roster(self.competition_roster_players.selection())

    def apply_competition_preset(self):
        if self.competition_team.get() == CURRENT_ROSTER:
            self._apply_competition_roster(tuple(n for n in self.state.roster if self.state.can_play(n)),
                ai=self.state.preset_ai, igl=self.state.preset_igl, carrier=self.state.preset_carrier)
            return
        preset = next((t for t in self.state.teams if t.name == self.competition_team.get()), None)
        if preset:
            self._apply_competition_roster(tuple(n for n in preset.roster if self.state.can_play(n)),
                ai=preset.ai, igl=preset.igl, carrier=preset.carrier, preset_id=preset.id)

    def _apply_competition_roster(self, names, **settings):
        event = self.selected_competition()
        if event is None or self.match_running:
            return
        try:
            candidate = self.state.with_tournament_roster(event.id, names, **settings)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.commit(candidate, "大会の出場ロスターを更新しました。次のシリーズから反映します。")

    def confirm_entry_deadline(self):
        events = self.state.entry_deadline_tournaments
        if not events:
            return True
        names = "\n".join(f"・{event.display_name}" for event in events)
        if messagebox.askyesno("大会のエントリー期限", "エントリー期限が今日までの不参加の大会がありますが、明日に進んでよろしいですか？\n\n"
                              + names + "\n\n「いいえ」で大会画面を開きます。", parent=self.root):
            return True
        self.competition_auto.set(False)
        self.show_screen("competitions")
        self.competition_list.selection_set(events[0].id)
        self.competition_list.see(events[0].id)
        self.preview_competition()
        self.status.set("日付の進行を止めました。大会への参加登録ができます。" +
                        ("完了したアクションの結果は保存済みです。登録後に1日進めてください。"
                         if self.state.day_advance_pending else ""))
        return False

    def finish_action_day(self):
        if self.state.day_advance_pending:
            return self.advance_calendar(1)
        return True

    def advance_calendar(self, days):
        if self.match_running:
            self.status.set("試合終了後に日付を進めてください。")
            return False
        previous = self.state
        candidate = previous
        try:
            for _ in range(days):
                if candidate.entry_deadline_tournaments:
                    if candidate is not self.state and not self.commit(candidate, f"ゲーム内 {candidate.date:%Y/%m/%d}。本日が大会の参加登録締切です。"):
                        return False
                    self.status.set(self.status.get() + self.monthly_event_notice(previous))
                    previous = self.state
                    if not self.confirm_entry_deadline():
                        return False
                following = candidate.advance_days(1)
                if following.date == candidate.date:
                    break
                candidate = following
        except ValueError as exc:
            self.status.set(str(exc))
            return False
        message = f"ゲーム内 {candidate.date:%Y/%m/%d}。スポンサー収入・月給を反映した収支: {candidate.money - previous.money:+,}円。"
        if candidate.day_action_blocked:
            message += " 大会画面で本日の試合を進めてください。"
        left = [p.name for p in previous.owned_players if candidate.player(p.name) is None]
        if left:
            message += f" 退団: {'、'.join(left)}。該当チームは編成し直してください。"
        if candidate.money < 0:
            message += " 所持金が赤字です。新規契約には必要資金の確保が必要です。"
        if self.commit(candidate, message):
            self.status.set(self.status.get() + self.monthly_event_notice(previous))
            return candidate.date > previous.date
        return False

    def advance_to_competition(self):
        if self.state.pending_tournaments:
            self.show_screen("competitions")
            self.competition_list.selection_set(self.state.pending_tournaments[0].id)
            self.preview_competition()
            return
        future = [parse_date(e.start_date) for e in self.state.visible_tournaments if parse_date(e.start_date) > self.state.date
                  and not (self.state.tournament(e.id) and self.state.tournament(e.id).completed)]
        if not future:
            self.status.set("出現中の次の大会はありません。1日または1か月ずつ日付を進めてください。")
            return
        self.advance_calendar((min(future) - self.state.date).days)

    def start_competition_series(self, event_id=None):
        if self.match_running:
            return
        event = self.state.tournament_definition(event_id) if event_id else self.selected_competition()
        if event is None:
            return
        if self._competition_after_id is not None:
            self.root.after_cancel(self._competition_after_id)
            self._competition_after_id = None
        try:
            run = self.state.tournament(event.id)
            if run is not None and not run.completed and player_eliminated(event, run):
                candidate = self.state.with_tournament_rating_finish(event.id)
                updated = candidate.tournament(event.id)
                message = (f"残りの試合をレート判定で確定し、大会終了。賞金{updated.prize_paid:,}円を入金しました。"
                           if updated.completed else "日付の進行が止まりました。他大会の参加判断を確認してください。")
                if self.commit(candidate, message):
                    self.competition_status.set(message)
                    if self.state.entry_deadline_tournaments:
                        self.advance_calendar(1)
                return
            self.state.check_tournament_match_day(event.id)
            match, _ = next_match(event, run)
            if run.own_team_id not in (match.left, match.right):
                candidate = self.state.with_tournament_rating_result(event.id)
                if self.commit(candidate, "他チーム同士の試合をレート勝率で抽選し、結果と翌日の日付を保存しました。"):
                    self._after_competition_result(event.id)
                return
            from game_core import validate_tick_time_ms
            candidate = self.state.with_prepared_tournament_roster(event.id)
            if candidate != self.state and not self.commit(candidate, "本日の大会出場ロスターを保存しました。不足枠は友達が出場します。"):
                return
            request = build_series_request(self.state, event.id, render=self.competition_render.get(),
                                           tick_time_ms=validate_tick_time_ms(self.competition_tick_ms.get()))
            self._competition_event_id = event.id
            self._competition_match, _ = next_match(event, self.state.tournament(event.id))
            self.competition_job = ScrimJob(request, self.store.path.parent / "tournaments" / event.id)
        except (OSError, ValueError) as exc:
            self.competition_status.set(str(exc))
            return
        self.competition_status.set(f"{request['own']['name']} vs {request['opponent']['name']} — {request['maps_to_win']}マップ先取で実行中")
        self.refresh()
        self._competition_after_id = self.root.after(100, self.poll_competition_series)

    def poll_competition_series(self):
        self._competition_after_id = None
        if self.competition_job is None:
            return
        result = self.competition_job.poll()
        if result is None:
            self._competition_after_id = self.root.after(100, self.poll_competition_series)
            return
        if result.get("status") == "completed":
            try:
                score = SeriesScore(result["match_id"], result["left_id"], result["right_id"], result["left_wins"], result["right_wins"])
                candidate = self.state.with_tournament_result(self._competition_event_id, score)
                if not self.commit(candidate, "大会のシリーズ結果を保存しました。"):
                    self.competition_status.set("保存に失敗しました。結果を保持しているので再試行します。")
                    self._competition_after_id = self.root.after(1000, self.poll_competition_series)
                    return
            except (ValueError, KeyError) as exc:
                self.competition_status.set(f"大会結果を反映できません: {exc}")
                self.competition_job = None
                self.refresh()
                return
        else:
            self.competition_status.set("シリーズを中止しました。未完了のシリーズは最初から再開できます。" if result.get("status") == "cancelled"
                                        else f"試合エラー: {result.get('message', '詳細不明')}")
        self.competition_job = None
        self.refresh()
        if result.get("status") == "completed":
            self._after_competition_result(self._competition_event_id)

    def _after_competition_result(self, event_id):
        if not self.finish_action_day():
            return
        run = self.state.tournament(event_id)
        completion = "他チーム同士はレート勝率で抽選。" if any(s.decided_by_rating for s in run.results) else ""
        if run.completed:
            message = f"{run.completed_date} 大会終了。{completion}賞金{run.prize_paid:,}円を入金しました。"
        elif run.last_match_date == self.state.game_date:
            message = "本日のシリーズ終了。1日進めると次の試合を開始できます。"
        else:
            message = f"{self.state.game_date} 次の試合日へ進めました。"
        self.competition_status.set(message)
        self.refresh()
        if self.competition_auto.get() and not run.completed:
            if run.last_match_date == self.state.game_date:
                if not self.advance_calendar(1):
                    self.competition_status.set("日付の進行が止まりました。他大会の参加判断を確認してください。")
                    return
            # Yield between instant NPC results so long tournaments remain responsive.
            self._competition_after_id = self.root.after(0, lambda: self._start_next_competition_series(event_id))
            self.preview_competition()

    def _start_next_competition_series(self, event_id):
        self._competition_after_id = None
        if self.competition_auto.get():
            self.start_competition_series(event_id)

    def cancel_competition_series(self):
        self.competition_auto.set(False)
        if self.competition_job:
            self.competition_job.cancel()
        elif self._competition_after_id is not None:
            self.root.after_cancel(self._competition_after_id)
            self._competition_after_id = None
            self.preview_competition()

    def forfeit_competition_series(self):
        event = self.selected_competition()
        if event is None or self.match_running:
            return
        try:
            candidate = self.state.with_tournament_forfeit(event.id)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        run = candidate.tournament(event.id)
        self.commit(candidate, f"大会終了。賞金{run.prize_paid:,}円を入金しました。" if run.completed
                    else "本日のシリーズは棄権負けとして記録しました。1日進めると次の大会試合へ進めます。")
