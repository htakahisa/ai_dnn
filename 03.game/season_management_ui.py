"""Scouting and contract screens for the real-time season Tk application."""

import tkinter as tk
from tkinter import ttk

from realtime_season import CONTRACT_OPTIONS, SeasonSaveError, contract_terms
from season_competitions import add_months, parse_date


def game_calendar(month):
    return f"ゲーム内 {month // 12 + 1}年目{month % 12 + 1}月"


class SeasonManagementMixin:
    def _build_management(self):
        self.finance_summary = tk.StringVar(self.root)
        self.scout_search = tk.StringVar(self.root)
        self.scout_filter = tk.StringVar(self.root, value="LFTのみ")
        self.offer_kind = {}
        self.offer_months = {}
        self.offer_summary = {}
        self.offer_buttons = {}
        for screen, title in (("scout", "スカウト — LFT選手"), ("contracts", "契約状況")):
            host = ttk.Frame(self.root, padding=24)
            setattr(self, f"{screen}_host", host)
            ttk.Label(host, text=title, font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
            navigation = ttk.Frame(host)
            navigation.pack(fill="x", pady=(12, 16))
            ttk.Button(navigation, text="ホームに戻る", command=self.show_home).pack(side="left")
            other = "contracts" if screen == "scout" else "scout"
            ttk.Button(navigation, text="契約状況へ" if screen == "scout" else "スカウトへ",
                       command=lambda target=other: self.show_screen(target)).pack(side="left", padx=8)
            ttk.Label(host, textvariable=self.finance_summary, font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", pady=(0, 12))
            if screen == "scout":
                ttk.Label(host, text="LFTは移籍金なし。他チームの選手・控えも、全選手から移籍金を払って獲得できます。").pack(anchor="w")
                search = ttk.Frame(host)
                search.pack(fill="x", pady=8)
                ttk.Label(search, text="選手名で検索").pack(side="left")
                ttk.Entry(search, textvariable=self.scout_search, width=30).pack(side="left", padx=8)
                ttk.Combobox(search, textvariable=self.scout_filter, values=("LFTのみ", "全選手"), state="readonly", width=12).pack(side="left", padx=8)
                columns = (("name", "選手", 200), ("role", "ロール", 130), ("iq", "IQ", 65),
                           ("salary", "基本月給（円）", 120), ("loyalty", "忠誠心 / 10", 95),
                           ("affiliation", "所属", 150), ("fee", "移籍金（円）", 120))
            else:
                ttk.Label(host, text="契約終了後はここから再契約できます。チームへの忠誠が0以下の選手は再契約を断ります。",
                          wraplength=960).pack(anchor="w", pady=(0, 8))
                columns = (("name", "選手", 135), ("status", "状況", 95), ("kind", "契約", 85),
                           ("salary", "月給（円）", 110), ("progress", "経過 / 期間", 90), ("remaining", "残り", 65),
                           ("end", "終了月", 125), ("loyalty", "忠誠心 / 10", 95), ("team_loyalty", "チームへの忠誠", 115))
            table = ttk.Frame(host)
            table.pack(fill="both", expand=True)
            tree = ttk.Treeview(table, columns=tuple(c[0] for c in columns), show="headings", selectmode="browse", height=12)
            for key, label, width in columns:
                tree.heading(key, text=label)
                tree.column(key, width=width, minwidth=45, anchor="w")
            tree.pack(side="left", fill="both", expand=True)
            scrollbar = ttk.Scrollbar(table, command=tree.yview)
            scrollbar.pack(side="right", fill="y")
            tree.configure(yscrollcommand=scrollbar.set)
            setattr(self, f"{screen}_players", tree)
            tree.bind("<<TreeviewSelect>>", lambda _event, target=screen: self.refresh_offer(target))
            offer = ttk.LabelFrame(host, text="契約条件", padding=12)
            offer.pack(fill="x", pady=(12, 8))
            row = ttk.Frame(offer)
            row.pack(fill="x")
            self.offer_kind[screen] = kind = tk.StringVar(self.root, value="1年契約")
            self.offer_months[screen] = months = tk.StringVar(self.root, value="6")
            ttk.Label(row, text="契約の種類").pack(side="left")
            menu = ttk.Combobox(row, textvariable=kind, values=tuple(CONTRACT_OPTIONS), state="readonly", width=15)
            setattr(self, f"{screen}_kind_menu", menu)
            menu.pack(side="left", padx=8)
            ttk.Label(row, text="短期契約の月数").pack(side="left", padx=(12, 0))
            duration = ttk.Combobox(row, textvariable=months, values=tuple(str(n) for n in range(1, 7)), state="readonly", width=5)
            duration.pack(side="left", padx=8)
            self.offer_buttons[screen] = button = ttk.Button(row, text="契約して獲得" if screen == "scout" else "再契約する",
                                                            command=lambda target=screen: self.sign_selected_contract(target))
            button.pack(side="right")
            menu.bind("<<ComboboxSelected>>", lambda _event, target=screen: self.refresh_offer(target))
            duration.bind("<<ComboboxSelected>>", lambda _event, target=screen: self.refresh_offer(target))
            setattr(self, f"{screen}_duration_menu", duration)
            self.offer_summary[screen] = summary = tk.StringVar(self.root)
            ttk.Label(offer, textvariable=summary, wraplength=960).pack(anchor="w", pady=(8, 0))
            ttk.Label(host, text="必要資金は契約時の所持金チェックです。契約成立時には引き落としません。\n"
                      "短期は基本月給、1年は基本月給、2年は0.9倍、3年は0.8倍。短期契約は忠誠が0以下になると次の月の開始時に退団します。",
                      wraplength=960).pack(anchor="w", pady=(0, 8))
            ttk.Label(host, text="忠誠心は0〜10。0の選手は短期契約のみ。契約中は毎月、チームへの忠誠が (10 − 忠誠心) ÷ 10 下がります。10なら下がりません。",
                      wraplength=960).pack(anchor="w", pady=(0, 8))
            if screen == "contracts":
                self.advance_month_button = ttk.Button(host, text="ゲーム内の時間を1か月進める", command=self.advance_game_month)
                self.advance_month_button.pack(anchor="w", pady=(0, 8))
            ttk.Label(host, textvariable=self.status, wraplength=960).pack(anchor="w")
        self.scout_search.trace_add("write", lambda *_: self.refresh_scout())
        self.scout_filter.trace_add("write", lambda *_: self.refresh_scout())

    def refresh_management(self):
        self.finance_summary.set(f"ゲーム内 {self.state.date:%Y/%m/%d}    所持金: {self.state.money:,}円")
        self.refresh_scout()
        selected = self.contracts_players.selection()
        self.contracts_players.delete(*self.contracts_players.get_children())
        labels = {key: label for label, key in CONTRACT_OPTIONS.items()}
        for player in self.state.owned_players:
            contract = self.state.contract(player.name)
            status = "契約中" if contract.active(self.state.game_month) else "契約終了"
            self.contracts_players.insert("", "end", iid=player.name, values=(
                player.name, status, labels[contract.kind], f"{contract.monthly_salary:,}",
                f"{contract.elapsed(self.state.game_month)} / {contract.duration_months}月",
                f"{contract.remaining(self.state.game_month)}月", f"{add_months(parse_date(self.state.start_date), contract.end_month):%Y/%m/%d}",
                f"{player.loyalty:g}", f"{contract.team_loyalty:g}"))
        if selected and self.contracts_players.exists(selected[0]):
            self.contracts_players.selection_set(selected[0])
        self.refresh_offer("contracts")
        self.advance_month_button.configure(state="disabled" if self.match_running else "normal")

    def refresh_scout(self):
        selected = self.scout_players.selection()
        self.scout_players.delete(*self.scout_players.get_children())
        query = self.scout_search.get().strip().casefold()
        pool = self.state.lft_players if self.scout_filter.get() == "LFTのみ" else self.state.scout_players
        for player in pool:
            if query in player.name.casefold():
                self.scout_players.insert("", "end", iid=player.name, values=(
                    player.name, player.role, f"{player.iq:g}", f"{player.monthly_salary:,}", f"{player.loyalty:g}",
                    self.state.player_affiliation(player.name), f"{self.state.transfer_fee(player.name):,}"))
        if selected and self.scout_players.exists(selected[0]):
            self.scout_players.selection_set(selected[0])
        self.refresh_offer("scout")

    def offer_player(self, screen):
        selected = getattr(self, f"{screen}_players").selection()
        if not selected:
            return None
        if screen == "contracts":
            return self.state.player(selected[0])
        return next((p for p in self.state.scout_players if p.name == selected[0]), None)

    def refresh_offer(self, screen):
        player = self.offer_player(screen)
        options = ("短期契約",) if player is not None and player.loyalty == 0 else tuple(CONTRACT_OPTIONS)
        getattr(self, f"{screen}_kind_menu").configure(values=options)
        if self.offer_kind[screen].get() not in options:
            self.offer_kind[screen].set(options[0])
        kind = CONTRACT_OPTIONS[self.offer_kind[screen].get()]
        getattr(self, f"{screen}_duration_menu").configure(state="readonly" if kind == "short" else "disabled")
        if player is None:
            self.offer_summary[screen].set("選手を選択してください。")
            self.offer_buttons[screen].configure(state="disabled")
            return
        try:
            terms = contract_terms(player, kind, int(self.offer_months[screen].get()))
            contract = self.state.contract(player.name)
            fee = self.state.transfer_fee(player.name) if screen == "scout" else 0
            blocked = "試合中は契約を変更できません。" if self.match_running else (self.state.recruitment_blocked(player.name) if screen == "scout" else "")
            if not blocked and screen == "contracts" and contract.active(self.state.game_month):
                blocked = "現在の契約が終了してから再契約できます。"
            elif not blocked and contract is not None and contract.team_loyalty <= 0:
                blocked = "チームへの忠誠が0以下のため、再契約できません。"
            elif not blocked and self.state.money < terms.required_funds + fee:
                blocked = "必要資金に対して所持金が不足しています。"
            self.offer_summary[screen].set(f"{player.name}: 月給{terms.monthly_salary:,}円 × {terms.months}か月\n"
                                          f"移籍金: {fee:,}円（支払い）  契約条件: {terms.required_funds:,}円（支払い後の残金で確認）\n"
                                          f"必要資金合計: {fee + terms.required_funds:,}円  {blocked or '契約可能です。'}")
            self.offer_buttons[screen].configure(state="disabled" if blocked else "normal")
        except (SeasonSaveError, ValueError) as exc:
            self.offer_summary[screen].set(str(exc))
            self.offer_buttons[screen].configure(state="disabled")

    def sign_selected_contract(self, screen):
        if self.match_running:
            self.status.set("試合が終了してから契約してください。")
            return
        player = self.offer_player(screen)
        if player is None:
            return
        try:
            kind = CONTRACT_OPTIONS[self.offer_kind[screen].get()]
            action = self.state.with_scouted_player if screen == "scout" else self.state.with_renewed_contract
            candidate = action(player.name, kind, int(self.offer_months[screen].get()))
        except (SeasonSaveError, ValueError) as exc:
            self.status.set(str(exc))
            return
        fee = self.state.money - candidate.money
        self.commit(candidate, f"{player.name}と{self.offer_kind[screen].get()}を結びました。移籍金の支払い: {fee:,}円。")

    def advance_game_month(self):
        if self.match_running:
            self.status.set("試合が終了してからゲーム内の月を進めてください。")
            return
        try:
            candidate = self.state.advance_months(stop_for_tournaments=True)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        left = [p.name for p in self.state.owned_players if candidate.player(p.name) is None]
        months = candidate.game_month - self.state.game_month
        income = self.state.monthly_sponsor_income if months else 0
        message = f"ゲーム内 {candidate.date:%Y/%m/%d}へ進めました。スポンサー収入: {income:,}円。収支: {candidate.money - self.state.money:+,}円。"
        if candidate.pending_tournaments:
            message += " 開催中の大会で参加判断または試合を進めてください。"
        if left:
            message += f" 退団: {'、'.join(left)}。該当チームは編成し直してください。"
        if candidate.money < 0:
            message += " 所持金が赤字です。新規契約には必要資金の確保が必要です。"
        previous = self.state
        if self.commit(candidate, message):
            self.status.set(self.status.get() + self.monthly_event_notice(previous))
