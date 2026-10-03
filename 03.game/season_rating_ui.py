"""Season-local rating rankings and the monthly sponsor contract."""

import tkinter as tk
from tkinter import ttk


class SeasonRatingMixin:
    def _build_ratings(self):
        self.ratings_host = host = ttk.Frame(self.root, padding=24)
        ttk.Label(host, text="レーティング・スポンサー", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        ttk.Button(host, text="ホームに戻る", command=self.show_home).pack(anchor="w", pady=12)
        ttk.Label(host, text="このセーブだけのレート一覧です。1500開始。スクリムと大会の完了結果で更新します。",
                  wraplength=950).pack(anchor="w", pady=(0, 12))
        self.sponsor_enabled = tk.BooleanVar(self.root, value=self.state.sponsor_active)
        self.sponsor_summary = tk.StringVar(self.root)
        self.sponsor_button = ttk.Checkbutton(host, text="スポンサー契約を有効にする", variable=self.sponsor_enabled,
                                             command=self.change_sponsor_contract)
        self.sponsor_button.pack(anchor="w")
        self.sponsor_team_choice = tk.StringVar(self.root)
        row = ttk.Frame(host)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="スポンサーの契約チーム").pack(side="left")
        self.sponsor_team_menu = ttk.Label(row, textvariable=self.sponsor_team_choice)
        self.sponsor_team_menu.pack(side="left", padx=8)
        ttk.Label(host, textvariable=self.sponsor_summary, wraplength=950).pack(anchor="w", pady=(0, 16))
        self.ratings_table = tree = ttk.Treeview(host, columns=("rank", "team", "rating"), show="headings", height=16)
        for key, label, width in (("rank", "順位", 80), ("team", "チーム", 450), ("rating", "レート", 180)):
            tree.heading(key, text=label)
            tree.column(key, width=width)
        tree.pack(fill="both", expand=True)
        ttk.Label(host, textvariable=self.status, wraplength=950).pack(anchor="w", pady=12)

    def refresh_ratings(self):
        self.ratings_table.delete(*self.ratings_table.get_children())
        for rank, record in enumerate(self.state.rating_ranking, 1):
            self.ratings_table.insert("", "end", iid=record.team_id, values=(rank, record.team_name, f"{record.value:.3f}"))
        self.sponsor_enabled.set(self.state.sponsor_active)
        self.sponsor_button.configure(state="disabled" if self.match_running else "normal")
        team = self.state.sponsor_team
        self.sponsor_team_choice.set(team.name if team else "")
        text = f"対象: {team.name}（レート{self.state.rating(team.id):.3f}）" if team else "対象チーム未登録"
        world = self.state.world_level_settings
        rank, count = self.state.world_rank
        self.sponsor_summary.set(f"{text}  月額: {self.state.monthly_sponsor_income:,}円\n"
                                 f"世界レベル{world.level} / {rank}位・全{count}チーム（上位{self.state.world_top_percent:.2f}%） / 敵倍率{world.enemy_multiplier:g}倍\n"
                                 "世界レベルのスポンサー資金を毎月入金します。順位が変わると世界レベル・敵倍率・月額資金も上下します。編成プリセットを切り替えても共通です。")

    def change_sponsor_contract(self):
        if self.match_running:
            self.refresh_ratings()
            return
        self.commit(self.state.with_sponsor_contract(self.sponsor_enabled.get()), "スポンサー契約を更新しました。")

