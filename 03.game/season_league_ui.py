"""Initial chapter selection before opening a real-time season save."""

from tkinter import ttk

from season_leagues import configured_leagues


class SeasonChapterSelection:
    def __init__(self, root, on_select, *, state=None):
        root.title("リアルタイムシーズン — 章選択")
        root.minsize(640, 400)
        root.geometry("720x480")
        root.protocol("WM_DELETE_WINDOW", "")
        self.host = host = ttk.Frame(root, padding=32)
        host.pack(fill="both", expand=True)
        ttk.Label(host, text="参加するリーグを選択", font=("Yu Gothic UI", 24, "bold")).pack(anchor="w")
        ttk.Label(host, text="資金・選手・契約・忠誠・日付は全章で引き継ぎます。\nその章のフリーナ杯で優勝すると、次の章が表示されます。",
                  wraplength=630).pack(anchor="w", pady=(12, 24))
        self.buttons = {}
        for chapter, name in configured_leagues().items():
            if chapter not in ((1,) if state is None else state.unlocked_chapters):
                continue
            button = ttk.Button(host, text=f"第{chapter}章　{name}",
                                command=lambda chapter=chapter: on_select(chapter))
            button.pack(fill="x", pady=8, ipady=10)
            self.buttons[chapter] = button
