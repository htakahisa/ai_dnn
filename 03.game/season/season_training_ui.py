"""Research and aim lab screens for owned season players."""

import tkinter as tk
from tkinter import ttk
from player_details_ui import shield_stats_text

from realtime_season import SeasonSaveError
from season.season_training import MAX_TRAINING_LEVEL, TRAINING_FIELDS


class SeasonTrainingMixin:
    def _build_training(self):
        self.training_hosts = {}
        self.training_players = {}
        self.training_summaries = {}
        self.training_buttons = {}
        for kind, (title, _, _) in TRAINING_FIELDS.items():
            self.training_hosts[kind] = host = ttk.Frame(self.root, padding=24)
            ttk.Label(host, text=title, font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
            navigation = ttk.Frame(host)
            navigation.pack(fill="x", pady=12)
            ttk.Button(navigation, text="ホームに戻る", command=self.show_home).pack(side="left")
            other = "aim_lab" if kind == "research" else "research"
            ttk.Button(navigation, text=f"{TRAINING_FIELDS[other][0]}へ",
                       command=lambda target=other: self.show_screen(target)).pack(side="left", padx=8)
            ability = "IQ" if kind == "research" else "命中率"
            if kind == "research":
                ability += "・チームへの忠誠"
            ttk.Label(host, text=f"所持選手を選び、資金を使って{ability}と{title}レベルを上げます。最大レベルは{MAX_TRAINING_LEVEL}です。実行すると1日経過します。契約中の全選手は毎日IQが0.1上がります。",
                      wraplength=950).pack(anchor="w", pady=(0, 12))
            table = ttk.Frame(host)
            table.pack(fill="both", expand=True)
            self.training_players[kind] = tree = ttk.Treeview(table,
                columns=("name", "role", "iq", "research", "hit", "aim_lab"), show="headings", height=13,
                selectmode="browse")
            for key, label, width in (("name", "選手", 170), ("role", "ロール", 130), ("iq", "IQ", 120),
                                      ("research", "研究レベル", 130), ("hit", "命中率", 130),
                                      ("aim_lab", "エイムラボレベル", 150)):
                tree.heading(key, text=label)
                tree.column(key, width=width, anchor="w")
            tree.pack(side="left", fill="both", expand=True)
            scroll = ttk.Scrollbar(table, command=tree.yview)
            scroll.pack(side="right", fill="y")
            tree.configure(yscrollcommand=scroll.set)
            tree.bind("<<TreeviewSelect>>", lambda _event, target=kind: self.refresh_training_offer(target))
            self.training_summaries[kind] = summary = tk.StringVar(self.root)
            ttk.Label(host, textvariable=summary, wraplength=950).pack(anchor="w", pady=12)
            self.training_buttons[kind] = button = ttk.Button(host, text=f"{title}する",
                                                              command=lambda target=kind: self.train_selected_player(target))
            button.pack(anchor="w", pady=(0, 12))
            ttk.Label(host, textvariable=self.status, wraplength=950).pack(anchor="w")

    def refresh_training(self):
        for kind, tree in self.training_players.items():
            selected = tree.selection()
            tree.delete(*tree.get_children())
            for player in self.state.owned_players:
                tree.insert("", "end", iid=player.name, values=(player.name, player.role, f"{player.iq:g}",
                            f"{player.research_level} / {MAX_TRAINING_LEVEL}", f"{player.hit_pct:.2%}",
                            f"{player.aim_lab_level} / {MAX_TRAINING_LEVEL}"))
            if selected and tree.exists(selected[0]):
                tree.selection_set(selected[0])
            self.refresh_training_offer(kind)

    def refresh_training_offer(self, kind):
        selected = self.training_players[kind].selection()
        player = self.state.player(selected[0]) if selected else None
        button = self.training_buttons[kind]
        button.configure(state="disabled")
        if player is None:
            self.training_summaries[kind].set(f"所持金: {self.state.money:,}円。育成する選手を選択してください。")
            return
        title, level_field, _ = TRAINING_FIELDS[kind]
        prefix = (f"{player.name} / {title}レベル: {getattr(player, level_field)} / {MAX_TRAINING_LEVEL} / 所持金: {self.state.money:,}円\n"
                  f"{shield_stats_text(player, compact=True)}\n")
        try:
            terms = self.state.training_terms(player.name, kind)
        except SeasonSaveError as exc:
            self.training_summaries[kind].set(prefix + str(exc))
            return
        growth = (f"IQ: {terms.before:g} → {terms.after:g}" if kind == "research"
                  else f"命中率: {terms.before:.2%} → {terms.after:.2%}")
        if kind == "research":
            loyalty = self.state.team_loyalty(player.name)
            growth += f" / チームへの忠誠: {loyalty:g} → {loyalty + terms.loyalty_gain:g}"
        blocked = "試合が終了してから育成してください。" if self.match_running else self.state.day_action_blocked or (
                  "所持金が不足しています。" if self.state.money < terms.cost else "")
        self.training_summaries[kind].set(prefix + f"次のレベル: {terms.next_level} / {growth}\n費用: {terms.cost:,}円 / 所要日数: 1日  {blocked}")
        button.configure(text=f"{title}する（{terms.cost:,}円）", state="disabled" if blocked else "normal")

    def train_selected_player(self, kind):
        if self.match_running:
            self.status.set("試合が終了してから育成してください。")
            return
        selected = self.training_players[kind].selection()
        if not selected:
            return
        try:
            terms = self.state.training_terms(selected[0], kind)
            options = {"advance_day": False} if self.state.entry_deadline_tournaments else {}
            candidate = self.state.with_trained_player(selected[0], kind, **options)
        except SeasonSaveError as exc:
            self.status.set(str(exc))
            self.refresh_training_offer(kind)
            return
        growth = (f"IQ {terms.before:g} → {terms.after:g}" if kind == "research"
                  else f"命中率 {terms.before:.2%} → {terms.after:.2%}")
        if kind == "research":
            growth += f"、チームへの忠誠 +{terms.loyalty_gain:g}"
        day_note = "1日進行を待っています。" if candidate.day_advance_pending else f"1日経過しました（{candidate.game_date}）。"
        if self.commit(candidate, f"{selected[0]}の{terms.title}がレベル{terms.next_level}になりました。育成で{growth}。費用: {terms.cost:,}円。{day_note}"):
            self.finish_action_day()
