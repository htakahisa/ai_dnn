"""Season calendar badge, tournament entry, real series execution, and standings."""

import tkinter as tk
from tkinter import ttk

from season_competitions import SeriesScore, next_match, parse_date
from season_scrim import ScrimJob, ai_options
from season_series import build_series_request


STAGES = {"upper": "Upper", "lower": "Lower", "lower_final": "Lower Final", "grand_final": "Grand Final"}


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
        self.competition_list = ttk.Treeview(host, columns=("name", "date", "count", "entry", "state"), show="headings", height=5, selectmode="browse")
        for key, label, width in (("name", "大会", 200), ("date", "開始日 ～ 終了予定日", 220), ("count", "参加数", 65),
                                  ("entry", "参加設定", 100), ("state", "状態", 150)):
            self.competition_list.heading(key, text=label)
            self.competition_list.column(key, width=width, minwidth=45)
        self.competition_list.pack(fill="x")
        self.competition_list.bind("<<TreeviewSelect>>", lambda _: self.preview_competition())
        ttk.Label(host, textvariable=self.competition_info, wraplength=900, justify="left").pack(anchor="w", pady=8)
        entry = ttk.Frame(host)
        entry.pack(fill="x", pady=(0, 8))
        ttk.Label(entry, text="参加チーム").pack(side="left")
        self.competition_team_menu = ttk.Combobox(entry, textvariable=self.competition_team, state="readonly", width=28)
        self.competition_team_menu.pack(side="left", padx=8)
        self.competition_enter_button = ttk.Button(entry, text="参加登録", command=self.enter_competition)
        self.competition_enter_button.pack(side="left")
        self.competition_decline_button = ttk.Button(entry, text="今回は参加しない", command=self.decline_competition)
        self.competition_decline_button.pack(side="left", padx=8)
        ttk.Label(host, text="自分のAI・IGL・キャリアーはスクリム準備画面の選択を使います。相手は所属設定を使います。",
                  wraplength=900).pack(anchor="w", pady=(0, 8))
        self.competition_matches = ttk.Treeview(host, columns=("stage", "teams", "score"), show="headings", height=6)
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
        for event in self.state.visible_tournaments:
            run = self.state.tournament(event.id)
            status = "不参加" if run and run.declined else "完了" if run and run.completed else "参加登録済み" if run else "未登録"
            if self.state.date > parse_date(event.end_date) and run is None:
                status = "参加受付終了"
            elif run and not run.completed and self.state.date >= parse_date(event.start_date):
                status = "本日試合済み" if run.last_match_date == self.state.date.isoformat() else "本日の試合待ち"
            elif run and run.completed_date:
                status = f"終了 {run.completed_date}"
            entry = "参加不可" if not event.allow_player_entry else "任意参加" if event.participation_optional else "強制参加"
            self.competition_list.insert("", "end", iid=event.id, values=(event.name,
                f"{event.start_date} ～ {event.end_date}", event.team_count, entry, status))
        if selected and self.competition_list.exists(selected[0]):
            self.competition_list.selection_set(selected[0])
        self.competition_team_menu.configure(values=[team.name for team in self.state.teams])
        if self.competition_team.get() not in [t.name for t in self.state.teams]:
            self.competition_team.set(self.state.selected_team.name if self.state.selected_team else (self.state.teams[0].name if self.state.teams else ""))
        self.preview_competition()

    def selected_competition(self):
        selected = self.competition_list.selection()
        return self.state.tournament_definition(selected[0]) if selected else None

    def preview_competition(self):
        event = self.selected_competition()
        self.competition_matches.delete(*self.competition_matches.get_children())
        self.competition_cancel_button.configure(state="normal" if self.competition_job else "disabled")
        self.competition_next_day_button.configure(state="disabled" if self.match_running else "normal")
        for button in (self.competition_enter_button, self.competition_decline_button, self.competition_play_button, self.competition_forfeit_button):
            button.configure(state="disabled")
        if event is None:
            self.competition_info.set("出現中の大会を選択してください。大会は専用Pythonファイルから設定できます。")
            return
        run = self.state.tournament(event.id)
        prizes = " / ".join(f"{rank}位 {money:,}円" for rank, money in sorted(event.prizes.items())) or "賞金なし"
        self.competition_info.set(f"{event.name}  |  {'ダブル' if event.format == 'double_elimination' else 'シングル'}エリミネーション\n"
            f"先取マップ数: 通常{event.normal_maps_to_win} / Lower Final {event.lower_final_maps_to_win} / Grand Final {event.grand_final_maps_to_win}\n賞金: {prizes}")
        if run is None and self.state.date <= parse_date(event.end_date) and not self.match_running:
            self.competition_enter_button.configure(state="normal")
            self.competition_enter_button.configure(text="参加登録" if event.allow_player_entry else "相手チームの大会を登録")
            self.competition_decline_button.configure(state="normal" if event.allow_player_entry and event.participation_optional else "disabled")
        if run and not run.declined:
            teams = {t.id: t.name for t in run.entrants}
            for score in run.results:
                self.competition_matches.insert("", "end", values=(score.match_id,
                    f"{teams[score.left_id]} vs {teams[score.right_id]}", f"{score.left_wins} - {score.right_wins}"))
            pending, ranking = next_match(event, run)
            if pending:
                self.competition_matches.insert("", "end", values=(STAGES[pending.stage],
                    f"次: {teams[pending.left]} vs {teams[pending.right]}", f"{pending.maps_to_win}マップ先取"))
                ready = (parse_date(event.start_date) <= self.state.date and run.last_match_date != self.state.date.isoformat()
                         and not self.match_running)
                self.competition_play_button.configure(state="normal" if ready else "disabled")
                self.competition_forfeit_button.configure(state="normal" if ready and run.own_team_id in (pending.left, pending.right) else "disabled")
                self.competition_info.set(self.competition_info.get() + ("\n本日の試合は終了しました。「1日進める」で翌日の試合へ進めます。"
                    if run.last_match_date == self.state.date.isoformat() else "\n1日1試合（シリーズ単位・相手同士の試合も含む）。全試合終了時に大会が自動終了します。"))
            else:
                for rank, team_id in enumerate(ranking, 1):
                    self.competition_matches.insert("", "end", values=(f"{rank}位", teams[team_id], f"{event.prizes.get(rank, 0):,}円"))
                self.competition_info.set(self.competition_info.get() + f"\n受け取り済み賞金: {run.prize_paid:,}円")
                if run.completed_date:
                    self.competition_info.set(self.competition_info.get() + f" / 実際の終了日: {run.completed_date}")

    def enter_competition(self):
        event = self.selected_competition()
        if event is None or self.match_running:
            return
        own = next((t for t in self.state.teams if t.name == self.competition_team.get()), None)
        try:
            igl = self.own_igl.get() if own and self.own_igl.get() in own.roster else None
            carrier = self.own_spike.get() if own and self.own_spike.get() in own.roster else None
            candidate = self.state.with_tournament_entry(event.id, own.id if own else None,
                own_ai=ai_options()[self.own_ai_choice.get()], igl=igl, carrier=carrier)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.commit(candidate, f"{event.name}に参加登録しました。開催日から試合を進められます。")

    def decline_competition(self):
        event = self.selected_competition()
        if event is None or self.match_running:
            return
        try:
            candidate = self.state.with_declined_tournament(event.id)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.commit(candidate, f"{event.name}は不参加にしました。")

    def advance_calendar(self, days):
        if self.match_running:
            self.status.set("試合終了後に日付を進めてください。")
            return
        try:
            candidate = self.state.advance_days(days)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        message = f"ゲーム内 {candidate.date:%Y/%m/%d}。スポンサー収入・月給を反映した収支: {candidate.money - self.state.money:+,}円。"
        if candidate.pending_tournaments:
            message += " 大会画面で参加判断または試合を進めてください。"
        previous = self.state
        if self.commit(candidate, message):
            self.status.set(self.status.get() + self.monthly_event_notice(previous))

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
        try:
            from game_core import validate_tick_time_ms
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
            run = self.state.tournament(self._competition_event_id)
            self.competition_status.set(f"{run.completed_date} 大会終了。賞金{run.prize_paid:,}円を入金しました。"
                if run.completed else "本日のシリーズ終了。1日進めると次の試合を開始できます。")
        else:
            self.competition_status.set("シリーズを中止しました。未完了のシリーズは最初から再開できます。" if result.get("status") == "cancelled"
                                        else f"試合エラー: {result.get('message', '詳細不明')}")
        self.competition_job = None
        self.refresh()
        run = self.state.tournament(self._competition_event_id)
        if result.get("status") == "completed" and self.competition_auto.get() and not run.completed:
            previous = self.state
            candidate = self.state.advance_days(1)
            if candidate.date == previous.date:
                self.competition_status.set("日付の進行が止まりました。他大会の参加判断を確認してください。")
                return
            if not self.commit(candidate, "大会の次の試合日へ進めました。"):
                self.competition_status.set("翌日の保存に失敗しました。日付を進めてから再開してください。")
                return
            self.status.set(self.status.get() + self.monthly_event_notice(previous))
            self.start_competition_series(self._competition_event_id)

    def cancel_competition_series(self):
        if self.competition_job:
            self.competition_job.cancel()
            self.competition_auto.set(False)

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
