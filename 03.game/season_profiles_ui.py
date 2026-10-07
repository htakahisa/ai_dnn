"""Modal team selection, creation and deletion."""

import tkinter as tk
from tkinter import messagebox, ttk

from realtime_season import SeasonSaveError


def choose_season_profile(root, profiles, active_path=None):
    result = None
    window = tk.Toplevel(root)
    window.title("チームのセーブ管理")
    window.geometry("700x430")
    host = ttk.Frame(window, padding=16)
    host.pack(fill="both", expand=True)
    ttk.Label(host, text="チームを選んで読み込むか、新しいチームを作成してください。").pack(anchor="w")
    tree = ttk.Treeview(host, columns=("name", "type"), show="headings", selectmode="browse", height=9)
    tree.heading("name", text="チーム名")
    tree.heading("type", text="保存形式")
    tree.column("name", width=420)
    tree.column("type", width=160)
    tree.pack(fill="both", expand=True, pady=10)
    rows = []

    def refresh():
        nonlocal rows
        rows = profiles.list()
        tree.delete(*tree.get_children())
        for index, profile in enumerate(rows):
            kind = "旧形式（削除不可）" if profile.legacy else "チーム別"
            if profile.path == active_path:
                kind += " / 使用中"
            tree.insert("", "end", iid=str(index), values=(profile.name, kind))

    def selected():
        selection = tree.selection()
        return rows[int(selection[0])] if selection else None

    def run(action):
        nonlocal result
        try:
            result = action()
        except (OSError, SeasonSaveError) as exc:
            messagebox.showerror("セーブ管理", str(exc), parent=window)
            return
        window.destroy()

    def load():
        profile = selected()
        if profile:
            run(lambda: profiles.load(profile))

    def delete():
        profile = selected()
        if not profile:
            return
        if profile.path == active_path or profile.legacy:
            messagebox.showinfo("削除できません", "使用中のチームと旧形式のセーブは削除できません。別のチームを読み込んでから操作してください。", parent=window)
            return
        if messagebox.askyesno("チームを削除", f"「{profile.name}」のセーブ・履歴・試合ログを削除します。\nこの操作は取り消せません。\n\n保存先: {profile.path.parent}\n\n削除しますか？", parent=window):
            try:
                profiles.delete(profile)
            except (OSError, SeasonSaveError) as exc:
                messagebox.showerror("削除できません", str(exc), parent=window)
            refresh()

    buttons = ttk.Frame(host)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="選択したチームをロード", command=load).pack(side="left")
    ttk.Button(buttons, text="選択したチームを削除", command=delete).pack(side="left", padx=8)
    tree.bind("<Double-1>", lambda _: load())
    new = ttk.LabelFrame(host, text="新しいチーム", padding=10)
    new.pack(fill="x", pady=12)
    name = tk.StringVar(window)
    entry = ttk.Entry(new, textvariable=name, width=32)
    entry.pack(side="left", fill="x", expand=True)
    create_button = ttk.Button(new, text="作成して開始", state="disabled",
                               command=lambda: run(lambda: profiles.create(name.get())))
    create_button.pack(side="left", padx=8)
    name.trace_add("write", lambda *_: create_button.configure(
        state="normal" if name.get().strip() else "disabled"))
    ttk.Button(host, text="閉じる", command=window.destroy).pack(anchor="e")
    refresh()
    window.wait_visibility()
    window.grab_set()
    entry.focus_set()
    root.wait_window(window)
    return result
