"""Scouting and contract screens for the real-time season Tk application."""

import tkinter as tk
from tkinter import ttk

from realtime_season import CONTRACT_OPTIONS, SeasonSaveError
from season_competitions import add_months, parse_date
from season_player_stats import player_combat_power, player_duel_power


def game_calendar(month):
    return f"ゲーム内 {month // 12 + 1}年目{month % 12 + 1}月"


class SeasonManagementMixin:
    def _build_management(self):
        self.finance_summary = tk.StringVar(self.root)
        self.scout_search = tk.StringVar(self.root)
        self.scout_filter = tk.StringVar(self.root, value="LFTのみ")
        self.scout_sort = tk.StringVar(self.root, value="IQ")
        self.scout_sort_order = tk.StringVar(self.root, value="高い順")
        self.scout_details = tk.StringVar(self.root, value="選手を選択すると全ステータスを表示します。")
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
                ttk.Entry(search, textvariable=self.scout_search, width=24).pack(side="left", padx=8)
                ttk.Combobox(search, textvariable=self.scout_filter, values=("LFTのみ", "全選手"), state="readonly", width=12).pack(side="left", padx=8)
                ttk.Label(search, text="並び替え").pack(side="left")
                self.scout_sort_menu = ttk.Combobox(search, textvariable=self.scout_sort,
                    values=("IQ", "総合戦闘力", "撃ち合い戦闘力"), state="readonly", width=16)
                self.scout_sort_menu.pack(side="left", padx=8)
                ttk.Combobox(search, textvariable=self.scout_sort_order, values=("高い順", "低い順"),
                             state="readonly", width=7).pack(side="left")
                columns = (("name", "選手", 135), ("role", "ロール", 90), ("iq", "IQ", 55),
                           ("combat", "総合戦闘力", 95), ("duel", "撃ち合い戦闘力", 115),
                           ("salary", "基本月給（円）", 110), ("loyalty", "忠誠心 / 10", 90),
                           ("affiliation", "所属", 130), ("fee", "移籍金（円）", 110))
            else:
                ttk.Label(host, text="契約終了後はここから再契約できます。チームへの忠誠が0以下の選手は再契約を断ります。",
                          wraplength=960).pack(anchor="w", pady=(0, 8))
                columns = (("name", "選手", 135), ("status", "状況", 210), ("kind", "契約", 85),
                           ("salary", "月給（円）", 110), ("progress", "経過 / 期間", 90), ("remaining", "残り", 65),
                           ("end", "終了月", 125), ("loyalty", "忠誠心 / 10", 95), ("team_loyalty", "チームへの忠誠", 115))
            if screen == "scout":
                body = ttk.Frame(host)
                body.pack(fill="both", expand=True)
                body.columnconfigure(0, weight=1)
                body.columnconfigure(1, minsize=320)
                body.rowconfigure(0, weight=1)
                table = ttk.Frame(body)
                table.grid(row=0, column=0, sticky="nsew")
                details_panel = ttk.LabelFrame(body, text="選択中の選手 — 全ステータス", padding=12)
                details_panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
                ttk.Label(details_panel, textvariable=self.scout_details, justify="left", wraplength=290).pack(anchor="nw")
                ttk.Button(details_panel, text="選手詳細・相棒TOP5", command=lambda: self.show_player_pairs(
                    self.offer_player("scout").name) if self.offer_player("scout") else None).pack(anchor="w", pady=8)
            else:
                table = ttk.Frame(host)
                table.pack(fill="both", expand=True)
            table.columnconfigure(0, weight=1)
            table.rowconfigure(0, weight=1)
            tree = ttk.Treeview(table, columns=tuple(c[0] for c in columns), show="headings", selectmode="browse",
                               height=11 if screen == "contracts" else 12)
            for key, label, width in columns:
                tree.heading(key, text=label)
                tree.column(key, width=width, minwidth=45, anchor="w")
            tree.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(table, command=tree.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            tree.configure(yscrollcommand=scrollbar.set)
            if screen == "scout":
                horizontal = ttk.Scrollbar(table, orient="horizontal", command=tree.xview)
                horizontal.grid(row=1, column=0, sticky="ew")
                tree.configure(xscrollcommand=horizontal.set)
                tree.tag_configure("contract_available", foreground="#17643a", background="#e2f3e8")
                tree.tag_configure("contract_unavailable", foreground="#9f2222", background="#fbe6e6")
                for key, criterion in (("iq", "IQ"), ("combat", "総合戦闘力"), ("duel", "撃ち合い戦闘力")):
                    tree.heading(key, command=lambda target=criterion: self.change_scout_sort(target))
            setattr(self, f"{screen}_players", tree)
            if screen == "contracts":
                tree.tag_configure("incoming_offer", foreground="#b71c1c", background="#ffebee")
                self.incoming_offer_summary = tk.StringVar(self.root)
                incoming = ttk.LabelFrame(host, text="他チームからの移籍オファー", padding=8)
                incoming.pack(fill="x", pady=(8, 0))
                ttk.Label(incoming, textvariable=self.incoming_offer_summary, foreground="#b71c1c",
                          wraplength=950).pack(anchor="w")
                answers = ttk.Frame(incoming)
                answers.pack(fill="x", pady=(6, 0))
                self.accept_transfer_button = ttk.Button(answers, text="オファーを承認して移籍",
                                                        command=lambda: self.respond_to_transfer(True))
                self.accept_transfer_button.pack(side="left")
                self.reject_transfer_button = ttk.Button(answers, text="オファーを断る",
                                                        command=lambda: self.respond_to_transfer(False))
                self.reject_transfer_button.pack(side="left", padx=8)
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
            ttk.Label(host, text="契約金は契約後の月給の3倍を支払います。移籍金も別途支払い、残金で契約期間分の月給を払えるか確認します。\n"
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
        self.scout_sort.trace_add("write", lambda *_: self.refresh_scout_availability())
        self.scout_sort_order.trace_add("write", lambda *_: self.refresh_scout_availability())
        for screen in self.offer_kind:
            self.offer_kind[screen].trace_add("write", lambda *_, target=screen: self.refresh_offer(target))
            self.offer_months[screen].trace_add("write", lambda *_, target=screen: self.refresh_offer(target))

    def refresh_management(self):
        self.finance_summary.set(f"ゲーム内 {self.state.date:%Y/%m/%d}    所持金: {self.state.money:,}円")
        self.refresh_scout()
        selected = self.contracts_players.selection()
        self.contracts_players.delete(*self.contracts_players.get_children())
        labels = {key: label for label, key in CONTRACT_OPTIONS.items()}
        for player in self.state.owned_players:
            contract = self.state.contract(player.name)
            incoming = self.state.transfer_offer(player.name)
            status = "契約中" if contract.active(self.state.game_month) else "契約終了"
            deferred = self.state.contract_end_deferred(contract)
            if deferred:
                status = "出場中につき契約延期中"
            self.contracts_players.insert("", "end", iid=player.name, values=(
                f"● {player.name}" if incoming else player.name,
                status if deferred else "● オファーあり" if incoming else status, labels[contract.kind], f"{contract.monthly_salary:,}",
                f"{contract.elapsed(self.state.game_month)} / {contract.duration_months}月",
                f"{contract.remaining(self.state.game_month)}月", f"{add_months(parse_date(self.state.start_date), contract.end_month):%Y/%m/%d}",
                f"{player.loyalty:g}", f"{contract.team_loyalty:g}"), tags=("incoming_offer",) if incoming else ())
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
                player = self.state.displayed_player(player)
                self.scout_players.insert("", "end", iid=player.name, values=(
                    player.name, player.role, f"{player.iq:g}", f"{player_combat_power(player):.2f}",
                    f"{player_duel_power(player):.2f}", f"{player.monthly_salary:,}", f"{player.loyalty:g}",
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

    def offer_conditions(self, screen, player):
        kind = "short" if player.loyalty == 0 else CONTRACT_OPTIONS[self.offer_kind[screen].get()]
        terms = self.state.contract_terms(player, kind, int(self.offer_months[screen].get()))
        contract = self.state.contract(player.name)
        fee = self.state.transfer_fee(player.name) if screen == "scout" else 0
        blocked = "試合中は契約を変更できません。" if self.match_running else (self.state.recruitment_blocked(player.name) if screen == "scout" else "")
        if not blocked and screen == "scout":
            blocked = self.state.scout_blocked()
        if not blocked and screen == "contracts" and self.state.contract_end_deferred(contract):
            blocked = "出場中につき契約延期中です。大会終了後に再契約できます。"
        elif not blocked and screen == "contracts" and contract.active(self.state.game_month):
            blocked = "現在の契約が終了してから再契約できます。"
        elif not blocked and contract is not None and contract.team_loyalty <= 0:
            blocked = "チームへの忠誠が0以下のため、再契約できません。"
        elif not blocked and self.state.money < terms.total_required_funds + fee:
            blocked = "必要資金に対して所持金が不足しています。"
        return terms, fee, blocked

    def refresh_scout_availability(self):
        players = {p.name: p for p in self.state.scout_players}
        available, unavailable = [], []
        for name in self.scout_players.get_children():
            try:
                _, _, blocked = self.offer_conditions("scout", players[name])
            except (SeasonSaveError, ValueError):
                blocked = True
            self.scout_players.item(name, tags=("contract_unavailable" if blocked else "contract_available",))
            (unavailable if blocked else available).append(name)
        criterion = self.scout_sort.get()
        def sort_value(name):
            player = self.state.displayed_player(players[name])
            value = player.iq if criterion == "IQ" else (player_combat_power(player)
                if criterion == "総合戦闘力" else player_duel_power(player))
            return (-value if self.scout_sort_order.get() == "高い順" else value, name.casefold())
        available.sort(key=sort_value)
        unavailable.sort(key=sort_value)
        for index, name in enumerate((*available, *unavailable)):
            self.scout_players.move(name, "", index)

    def change_scout_sort(self, criterion):
        if self.scout_sort.get() == criterion:
            self.scout_sort_order.set("低い順" if self.scout_sort_order.get() == "高い順" else "高い順")
        else:
            self.scout_sort_order.set("高い順")
            self.scout_sort.set(criterion)

    def refresh_scout_details(self, player):
        if player is None:
            self.scout_details.set("選手を選択すると全ステータスを表示します。")
            return
        contract = self.state.contract(player.name)
        owner = self.state.opponent_owner(player.name)
        if contract is None and owner is not None:
            contract = next((c for c in owner.contracts if c.player_name == player.name), None)
        loyalty = f"{contract.team_loyalty:g}" if contract else "—"
        if owner is not None:
            player = self.state.enemy_player(player)
        world = (f"世界レベル補正: {self.state.world_level_settings.enemy_multiplier:g}倍（獲得後は補正前の能力）\n"
                 if owner is not None else "")
        self.scout_details.set(
            f"{player.name} / {player.role}\n所属: {self.state.player_affiliation(player.name)}\n{world}\n"
            f"総合戦闘力: {player_combat_power(player):.2f}\n撃ち合い戦闘力: {player_duel_power(player):.2f}\n\n"
            f"HS率: {player.hs_pct:.1%} / 命中率: {player.hit_pct:.1%}\n"
            f"回避率: {player.dodge_pct:.1%} / 反応: {player.reaction:g}\n"
            f"IQ: {player.iq:g} / 影響力: {player.influence:g}\n"
            f"メンタル: {player.mental:g} / 調子の波: {player.form_variance:g}\n\n"
            f"研究Lv: {player.research_level} / 10 / エイムラボLv: {player.aim_lab_level} / 10\n"
            f"基本月給: {player.monthly_salary:,}円\n忠誠心: {player.loyalty:g} / 10\nチームへの忠誠: {loyalty}")

    def refresh_offer(self, screen):
        player = self.offer_player(screen)
        if screen == "contracts":
            self.refresh_transfer_offer(player)
        options = ("短期契約",) if player is not None and player.loyalty == 0 else tuple(CONTRACT_OPTIONS)
        getattr(self, f"{screen}_kind_menu").configure(values=options)
        if self.offer_kind[screen].get() not in options:
            self.offer_kind[screen].set(options[0])
        kind = CONTRACT_OPTIONS[self.offer_kind[screen].get()]
        getattr(self, f"{screen}_duration_menu").configure(state="readonly" if kind == "short" else "disabled")
        if screen == "scout":
            self.refresh_scout_availability()
            self.refresh_scout_details(player)
        if player is None:
            self.offer_summary[screen].set("選手を選択してください。")
            self.offer_buttons[screen].configure(state="disabled")
            return
        try:
            terms, fee, blocked = self.offer_conditions(screen, player)
            self.offer_summary[screen].set(f"{player.name}: 月給{terms.monthly_salary:,}円 × {terms.months}か月\n"
                                          f"移籍金: {fee:,}円 / 契約金: {terms.signing_bonus:,}円（支払い）  残金条件: {terms.required_funds:,}円\n"
                                          f"必要資金合計: {fee + terms.total_required_funds:,}円  {blocked or '契約可能です。'}")
            self.offer_buttons[screen].configure(state="disabled" if blocked else "normal")
        except (SeasonSaveError, ValueError) as exc:
            self.offer_summary[screen].set(str(exc))
            self.offer_buttons[screen].configure(state="disabled")

    def refresh_transfer_offer(self, player):
        offer = self.state.transfer_offer(player.name) if player is not None else None
        if offer is None:
            self.incoming_offer_summary.set("赤い●の選手を選択するとオファーを確認できます。忠誠が30未満になると強制成立します。")
        else:
            club = next(c for c in self.state.opponent_teams if c.id == offer.team_id)
            loyalty = self.state.contract(player.name).team_loyalty
            self.incoming_offer_summary.set(f"● {player.name} ← {club.name} / 移籍金: {offer.fee:,}円 / チームへの忠誠: {loyalty:g}\n"
                                            "承認・拒否を選べます。拒否後も忠誠が30未満になると、強制的に移籍します。")
        enabled = offer is not None and not self.match_running
        self.accept_transfer_button.configure(state="normal" if enabled else "disabled")
        self.reject_transfer_button.configure(state="normal" if enabled else "disabled")

    def respond_to_transfer(self, accept):
        if self.match_running:
            self.status.set("試合が終了してからオファーに回答してください。")
            return
        player = self.offer_player("contracts")
        offer = self.state.transfer_offer(player.name) if player is not None else None
        if offer is None:
            return
        try:
            club = next(c for c in self.state.opponent_teams if c.id == offer.team_id)
            candidate = self.state.with_transfer_response(offer.id, accept)
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            return
        if candidate.player(player.name) is None:
            message = f"{player.name}が{club.name}へ移籍しました。移籍金{offer.fee:,}円を受け取りました。編成を確認してください。"
        elif any(o.id == offer.id and o.status == "cancelled" for o in candidate.transfer_offers):
            message = f"{club.name}が必要資金またはスカウト回数の条件を満たさなくなったため、{player.name}へのオファーは取り消されました。"
        else:
            message = f"{player.name}への{club.name}からのオファーを断りました。忠誠が30未満になると強制成立します。"
        self.commit(candidate, message)

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
            options = {"advance_day": False} if screen == "scout" and self.state.entry_deadline_tournaments else {}
            candidate = action(player.name, kind, int(self.offer_months[screen].get()), **options)
        except (SeasonSaveError, ValueError) as exc:
            self.status.set(str(exc))
            return
        fee = self.state.transfer_fee(player.name) if screen == "scout" else 0
        bonus = candidate.contract(player.name).monthly_salary * 3
        day_note = (" 1日進行を待っています。" if candidate.day_advance_pending else f" 1日経過しました（{candidate.game_date}）。") + f"スカウト: {candidate.scout_allowance_text}。" if screen == "scout" else ""
        if self.commit(candidate, f"{player.name}と{self.offer_kind[screen].get()}を結びました。移籍金: {fee:,}円 / 契約金: {bonus:,}円を支払いました。" + day_note):
            self.finish_action_day()

    def advance_game_month(self):
        if self.match_running:
            self.status.set("試合が終了してからゲーム内の月を進めてください。")
            return
        self.advance_calendar((add_months(self.state.date, 1) - self.state.date).days)
