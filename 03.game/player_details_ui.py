"""Scrollable, selectable player descriptions bound to existing StringVars."""

import tkinter as tk
from tkinter import ttk


def shield_stats_text(player, *, compact=False):
    fields = (
        f"シールド: {player.shield_hp:g}HP",
        f"シールドピアサー: {'あり' if player.shield_piercer else 'なし'}",
        f"シールドクラッシュ: {player.shield_crash:g}HP",
    )
    return (" / " if compact else "\n").join(fields)


def readonly_details(parent, variable, *, height=14):
    host = ttk.Frame(parent)
    text = tk.Text(host, height=height, width=1, wrap="word", borderwidth=0,
                   highlightthickness=0, font=("Yu Gothic UI", 10),
                   background=ttk.Style(parent).lookup("TFrame", "background") or "SystemButtonFace")
    text.pack(side="left", fill="both", expand=True)
    scroll = ttk.Scrollbar(host, command=text.yview)
    scroll.pack(side="right", fill="y")
    text.configure(yscrollcommand=scroll.set)

    def refresh(*_args):
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", variable.get())
        text.yview_moveto(0)
        text.configure(state="disabled")

    trace = variable.trace_add("write", refresh)

    def cleanup(event):
        if event.widget == host:
            try:
                variable.trace_remove("write", trace)
            except tk.TclError:
                pass

    host.bind("<Destroy>", cleanup)
    host.text = text
    refresh()
    return host
