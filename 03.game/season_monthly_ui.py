"""Saved news from other teams at each simulated month boundary."""

import tkinter as tk
from tkinter import ttk


EVENT_LABELS = {"recruitment": "LFT契約", "renewal": "再契約", "departure": "退団",
                "recruitment_unfilled": "補充見送り", "month_completed": "月次まとめ",
                "roster_return": "正規メンバー復帰", "transfer_offer": "移籍オファー"}


class SeasonMonthlyMixin:
    def _build_monthly_events(self):
        self.monthly_host = host = ttk.Frame(self.root, padding=24)
        ttk.Label(host, text="月次イベント", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        ttk.Button(host, text="ホームに戻る", command=self.show_home).pack(anchor="w", pady=12)
        ttk.Label(host, text="ゲーム内の月が切り替わると、他チームが契約更新や不足メンバーの補充を行います。",
                  wraplength=950).pack(anchor="w", pady=(0, 8))
        self.monthly_summary = tk.StringVar(self.root)
        self.monthly_detail = tk.StringVar(self.root)
        ttk.Label(host, textvariable=self.monthly_summary).pack(anchor="w", pady=(0, 12))
        table = ttk.Frame(host)
        table.pack(fill="both", expand=True)
        self.monthly_table = tree = ttk.Treeview(table, columns=("date", "kind", "team", "player", "message"), show="headings", height=18)
        for key, label, width in (("date", "日付", 100), ("kind", "出来事", 100), ("team", "チーム", 155),
                                  ("player", "選手", 130), ("message", "内容", 470)):
            tree.heading(key, text=label)
            tree.column(key, width=width, minwidth=60)
        tree.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table, command=tree.yview)
        scroll.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scroll.set)
        tree.bind("<<TreeviewSelect>>", self.preview_monthly_event)
        ttk.Label(host, textvariable=self.monthly_detail, wraplength=950).pack(anchor="w", pady=12)
        ttk.Label(host, textvariable=self.status, wraplength=950).pack(anchor="w")

    def refresh_monthly_events(self):
        selected = self.monthly_table.selection()
        self.monthly_table.delete(*self.monthly_table.get_children())
        for event in reversed(self.state.monthly_events):
            self.monthly_table.insert("", "end", iid=event.id, values=(event.date, EVENT_LABELS[event.kind],
                                      event.team_name, event.player_name or "—", event.message))
        count = sum(e.kind != "month_completed" for e in self.state.monthly_events)
        self.monthly_summary.set(f"ゲーム内 {self.state.date:%Y/%m/%d}    他チームの出来事: {count}件")
        if selected and self.monthly_table.exists(selected[0]):
            self.monthly_table.selection_set(selected[0])
        self.preview_monthly_event()

    def preview_monthly_event(self, _event=None):
        selected = self.monthly_table.selection()
        event = next((e for e in self.state.monthly_events if selected and e.id == selected[0]), None)
        self.monthly_detail.set(f"{event.date}  {event.team_name}\n{event.message}" if event else "行を選択すると出来事の全文を表示します。")

    def monthly_event_notice(self, previous):
        if self.state.monthly_events_through == previous.monthly_events_through:
            return ""
        events = self.state.monthly_events[len(previous.monthly_events):]
        count = sum(e.kind != "month_completed" for e in events)
        return f" 他チームの出来事: {count}件。「月次イベント」で確認できます。"
