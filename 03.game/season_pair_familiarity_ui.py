"""Touch-friendly, bounded pair views for the season's ttk application."""

import tkinter as tk
from tkinter import ttk

import realtime_season_pair_familiarity as settings
from season_pair_familiarity import familiarity, memberships, pair_key, stage, stage_index
from realtime_season import CONTRACT_OPTIONS, SeasonSaveError


class SeasonPairFamiliarityMixin:
    def _build_pair_familiarity(self):
        self._pair_windows = []
        self.home_teams.bind("<Double-1>", lambda _e: self.show_selected_team_pairs())

    def _pair_window(self, title, width, height):
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry(f"{width}x{height}")
        window.minsize(width, height)
        ttk.Label(window, text=title, font=("Yu Gothic UI", 17, "bold")).pack(anchor="w", padx=16, pady=12)
        return window

    def refresh_pair_windows(self):
        active = []
        for window, refresh in getattr(self, "_pair_windows", ()):
            if window.winfo_exists():
                refresh()
                active.append((window, refresh))
        self._pair_windows = active

    def show_selected_team_pairs(self):
        selected = self.home_teams.selection()
        return self.show_team_pairs(selected[0] if selected else self.state.club_id)

    def show_team_pairs(self, team_id):
        name = self.state.team_name if team_id == self.state.club_id else next(
            (c.name for c in self.state.opponent_teams if c.id == team_id), "チーム")
        window = self._pair_window(f"{name} — ペア練度", 840, 660)
        preset_choice = tk.StringVar(window)
        preset_menu = ttk.Combobox(window, textvariable=preset_choice, state="readonly")
        if team_id == self.state.club_id:
            preset_menu.pack(fill="x", padx=16, pady=(0, 8))
        table = ttk.Frame(window)
        table.pack(fill="x", padx=16)
        summary = tk.StringVar(window)
        detail = tk.StringVar(window, value="段階のセルをタップすると、通算日数と練度を表示します。")
        ttk.Label(window, textvariable=summary, font=("Yu Gothic UI", 12, "bold")).pack(anchor="w", padx=16, pady=8)
        ttk.Label(window, textvariable=detail, wraplength=800).pack(anchor="w", padx=16, pady=(0, 8))
        ttk.Label(window, text="所属全員（控えを含む）／選手を選んで「相棒TOP5」").pack(anchor="w", padx=16)
        members_host = ttk.Frame(window)
        members_host.pack(fill="both", expand=True, padx=16, pady=6)
        members = ttk.Treeview(members_host, columns=("name", "role"), show="headings", height=5, selectmode="browse")
        members.heading("name", text="選手")
        members.heading("role", text="スタメン / 控え")
        members.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(members_host, command=members.yview)
        scroll.pack(side="right", fill="y")
        members.configure(yscrollcommand=scroll.set)
        ttk.Button(window, text="選手詳細・相棒TOP5", command=lambda: self.show_player_pairs(
            members.selection()[0]) if members.selection() else None).pack(anchor="w", padx=16, pady=8)
        members.bind("<Double-1>", lambda _e: self.show_player_pairs(members.selection()[0]) if members.selection() else None)
        window.pair_cells = {}
        window.pair_detail = detail
        window.pair_summary = summary
        window.pair_members = members

        def refresh():
            if team_id == self.state.club_id:
                presets = {t.name: t.roster for t in self.state.teams}
                if len(self.state.roster) == 5:
                    presets["編集中の5人"] = self.state.roster
                preset_menu.configure(values=tuple(presets))
                if preset_choice.get() not in presets:
                    current = self.state.pair_starters(team_id)
                    preset_choice.set(next((n for n, roster in presets.items() if roster == current), next(iter(presets), "")))
                starters = presets.get(preset_choice.get(), ())
                names = tuple(p.name for p in self.state.owned_players)
            else:
                club = next((c for c in self.state.opponent_teams if c.id == team_id), None)
                starters, names = (club.roster, club.members) if club else ((), ())
            for widget in table.winfo_children():
                widget.destroy()
            window.pair_cells = {}
            for i, player in enumerate(starters):
                ttk.Label(table, text=player, wraplength=115).grid(row=0, column=i + 1, padx=2, pady=4)
                ttk.Label(table, text=player, wraplength=115).grid(row=i + 1, column=0, padx=2, pady=4)
                for j, other in enumerate(starters):
                    if j <= i:
                        ttk.Label(table, text="—" if j == i else "", width=12).grid(row=i + 1, column=j + 1)
                        continue
                    key = pair_key(player, other)
                    days = self.state.pair_days.get(key, 0)
                    value = familiarity(days)
                    index = stage_index(value)
                    message = f"{player} × {other}：{stage(days)} ／ 通算{days:,}日 ／ 練度{value:.2%}"
                    button = tk.Button(table, text=stage(days), width=12, height=2,
                        bg=settings.STAGE_COLORS[index], fg="white" if index >= 3 else "#172554",
                        command=lambda text=message: detail.set(text), cursor="hand2")
                    button.grid(row=i + 1, column=j + 1, padx=1, pady=1, sticky="nsew")
                    button.bind("<Enter>", lambda _e, text=message: detail.set(text))
                    window.pair_cells[key] = button
            c, m = self.state.pair_metrics(starters)
            summary.set(f"チーム練度 C：{c:.2%} ／ 能力倍率 M：{m:.4f}倍" +
                        ("（無効：試合には1倍で適用）" if not settings.pair_familiarity_enabled else "")
                        if c is not None else "スタメン5人が揃うと、チーム練度と倍率を表示します。")
            selected = members.selection()
            members.delete(*members.get_children())
            for player in names:
                members.insert("", "end", iid=player, values=(player, "スタメン" if player in starters else "控え"))
            if selected and members.exists(selected[0]):
                members.selection_set(selected[0])
        preset_menu.bind("<<ComboboxSelected>>", lambda _e: refresh())
        refresh()
        self._pair_windows.append((window, refresh))
        return window

    def show_player_pairs(self, player_name):
        window = self._pair_window(f"{player_name} — 相棒TOP5", 800, 450)
        current = tk.StringVar(window)
        ttk.Label(window, textvariable=current).pack(anchor="w", padx=16)
        tree = ttk.Treeview(window, columns=("name", "stage", "days", "same"), show="headings", height=5, selectmode="browse")
        for key, label, width in (("name", "相棒", 170), ("stage", "段階", 130), ("days", "通算日数", 90), ("same", "現在の所属", 290)):
            tree.heading(key, text=label)
            tree.column(key, width=width)
        tree.pack(fill="both", expand=True, padx=16, pady=10)
        detail = tk.StringVar(window, value="行をタップすると日数と練度を表示します。● は現在同じチームです。")
        ttk.Label(window, textvariable=detail, wraplength=760).pack(anchor="w", padx=16, pady=8)
        window.partner_tree = tree
        window.partner_detail = detail
        contract_info = tk.StringVar(window)
        ttk.Label(window, textvariable=contract_info, wraplength=760).pack(anchor="w", padx=16)
        pair_button = ttk.Button(window, text="ペア契約", command=lambda: self.sign_player_pair(player_name, tree.selection()[0]) if tree.selection() else None)
        window.pair_contract_button = pair_button
        def preview(_event=None):
            pair_button.pack_forget()
            contract_info.set("")
            selection = tree.selection()
            if selection:
                other = selection[0]
                days = self.state.pair_days.get(pair_key(player_name, other), 0)
                detail.set(f"{player_name} × {other}：{stage(days)} ／ 通算{days:,}日 ／ 練度{familiarity(days):.2%}")
                owner = self.state.opponent_owner(player_name)
                if owner is not None and owner == self.state.opponent_owner(other):
                    fee = self.state.transfer_fee(player_name) + self.state.transfer_fee(other)
                    kind = self.offer_kind["scout"].get()
                    contract_info.set(f"2人の移籍金合計：{fee:,}円（契約金・給与の資金条件は別途）。\n"
                                      f"契約条件はスカウト画面の「{kind}」。2回・2日を消費します。片方だけ成立した場合も保持します。\n"
                                      f"スカウト：{self.state.scout_allowance_text}。{self.state.scout_blocked()}")
                    pair_button.configure(text=f"ペア契約 — 移籍金合計 {fee:,}円", state="disabled" if self.match_running or self.state.scout_blocked() else "normal")
                    pair_button.pack(anchor="w", padx=16, pady=8)
        tree.bind("<<TreeviewSelect>>", preview)
        def refresh():
            owners = memberships(self.state)
            current.set(f"所属：{self.state.player_affiliation(player_name)}")
            selected = tree.selection()
            tree.delete(*tree.get_children())
            for other, days in self.state.pair_partners.get(player_name, ())[:settings.PARTNER_LIMIT]:
                same = owners.get(player_name) is not None and owners.get(player_name) == owners.get(other)
                tree.insert("", "end", iid=other, values=(other, stage(days), f"{days:,}日",
                    ("● 同じチーム：" if same else "") + self.state.player_affiliation(other)))
            if not tree.get_children():
                detail.set("通算1日以上一緒に所属した相棒はまだいません。")
            if selected and tree.exists(selected[0]):
                tree.selection_set(selected[0])
            preview()
        refresh()
        self._pair_windows.append((window, refresh))
        return window

    def sign_player_pair(self, first, second):
        if self.match_running:
            self.status.set("試合が終了してから契約してください。")
            return
        owner = self.state.opponent_owner(first)
        if owner is None or owner != self.state.opponent_owner(second) or first == second:
            self.status.set("同じライバルチームにいる2人を選択してください。")
            return
        signed = []
        for name in (first, second):
            try:
                kind = CONTRACT_OPTIONS[self.offer_kind["scout"].get()]
                options = {"advance_day": False} if self.state.entry_deadline_tournaments else {}
                player = next(p for p in self.state.scout_players if p.name == name)
                terms, _, _ = self.offer_conditions("scout", player)
                candidate = self.state.with_scouted_player(name, kind, terms.months, **options)
            except (SeasonSaveError, ValueError) as exc:
                self.status.set(f"{', '.join(signed)}のみ契約成立。{name}の契約は成立しませんでした：{exc}" if signed else str(exc))
                return
            if not self.commit(candidate, f"ペア契約：{name}を獲得しました。"):
                return
            signed.append(name)
            if not self.finish_action_day():
                return
        self.status.set(f"ペア契約成立：{first}と{second}を獲得しました。通算日数は引き継ぎます。")

    def show_pair_ranking(self):
        window = self._pair_window("リーグの名コンビ TOP20", 1020, 680)
        same_only = tk.BooleanVar(window, value=False)
        tree = ttk.Treeview(window, columns=("rank", "pair", "owner", "days", "stage"), show="headings", height=20, selectmode="browse")
        for key, label, width in (("rank", "順位", 55), ("pair", "ペア", 280), ("owner", "現所属", 370),
                                  ("days", "通算日数", 110), ("stage", "段階", 130)):
            tree.heading(key, text=label)
            tree.column(key, width=width)
        def refresh():
            owners = memberships(self.state)
            tree.delete(*tree.get_children())
            rows = (item for item in self.state.pair_ranking if not same_only.get() or
                    owners.get(item[0][0]) is not None and owners.get(item[0][0]) == owners.get(item[0][1]))
            from itertools import islice
            for rank, ((a, b), days) in enumerate(islice(rows, settings.RANKING_LIMIT), 1):
                same = owners.get(a) is not None and owners.get(a) == owners.get(b)
                owner = ("同じチーム：" + self.state.player_affiliation(a)) if same else (
                    f"{a}：{self.state.player_affiliation(a)} ／ {b}：{self.state.player_affiliation(b)}")
                tree.insert("", "end", iid=str(rank), values=(rank, f"{a} × {b}", owner, f"{days:,}日", stage(days)), tags=(str(days),))
        ttk.Checkbutton(window, text="現在同じチームのペアのみ", variable=same_only, command=refresh).pack(anchor="w", padx=16, pady=(0, 8))
        tree.pack(fill="both", expand=True, padx=16)
        detail = tk.StringVar(window, value="行をタップすると通算日数と練度を表示します。")
        def preview(_event=None):
            selection = tree.selection()
            if selection:
                days = int(tree.item(selection[0], "tags")[0])
                detail.set(f"{tree.item(selection[0], 'values')[1]} ／ 通算{days:,}日 ／ 練度{familiarity(days):.2%}")
        tree.bind("<<TreeviewSelect>>", preview)
        ttk.Label(window, textvariable=detail, wraplength=980).pack(anchor="w", padx=16, pady=12)
        window.ranking_tree, window.same_only = tree, same_only
        window.refresh_ranking = refresh
        refresh()
        self._pair_windows.append((window, refresh))
        return window
