"""ゲーム開始前のアタッカー編成・スパイク所持者選択画面。"""

import tkinter as tk
from character_stats import all_names

MAX_ROSTER = 5

# 💡追加: コントローラー選択の選択肢（表示名 -> 内部キー）
ATTACKER_CONTROLLER_OPTIONS = {
    "AI (学習済み)": "learning",
    "ロジック": "default",
    "ユーザー操作": "user",
}
DEFENDER_CONTROLLER_OPTIONS = {
    "AI (学習済み)": "learning_all",
    "ロジック": "default",
    "ユーザー操作": "user",
}


class RosterSelectScreen:
    def __init__(self, on_confirm, max_roster=MAX_ROSTER):
        self.on_confirm = on_confirm
        self.max_roster = max_roster
        self.roster = []
        self.char_buttons = {}
        self.spike_radio_buttons = {}

        # 必ず先にTkウィンドウを作る
        self.root = tk.Tk()
        self.root.title("キャラクター編成")
        self.root.geometry("420x760")

        # Tk生成後にStringVarを作る
        self.spike_var = tk.StringVar(master=self.root, value="")
        # 💡追加: コントローラー選択用の変数
        self.defender_ctrl_var = tk.StringVar(master=self.root, value=list(DEFENDER_CONTROLLER_OPTIONS.keys())[0])
        self.attacker_ctrl_var = tk.StringVar(master=self.root, value=list(ATTACKER_CONTROLLER_OPTIONS.keys())[0])

        # 💡追加: コントローラー選択セクション
        ctrl_frame = tk.LabelFrame(self.root, text="操作方法", padx=8, pady=6)
        ctrl_frame.pack(fill="x", padx=18, pady=(10, 0))

        tk.Label(ctrl_frame, text="Defender:").grid(row=0, column=0, sticky="w", pady=2)
        tk.OptionMenu(ctrl_frame, self.defender_ctrl_var, *DEFENDER_CONTROLLER_OPTIONS.keys()).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        tk.Label(ctrl_frame, text="Attacker:").grid(row=1, column=0, sticky="w", pady=2)
        tk.OptionMenu(ctrl_frame, self.attacker_ctrl_var, *ATTACKER_CONTROLLER_OPTIONS.keys()).grid(
            row=1, column=1, sticky="ew", padx=(6, 0)
        )

        ctrl_frame.grid_columnconfigure(1, weight=1)

        tk.Label(
            self.root,
            text=f"アタッカーを選択してください（{self.max_roster}人）",
            font=("Arial", 12, "bold"),
        ).pack(pady=(12, 6))

        list_frame = tk.Frame(self.root)
        list_frame.pack(fill="both", expand=True, padx=12)
        canvas = tk.Canvas(list_frame, height=280, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner_frame = tk.Frame(canvas)
        inner_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        for name in all_names():
            btn = tk.Button(inner_frame, text=name, width=34, anchor="w", command=lambda n=name: self.select_character(n))
            btn.pack(fill="x", padx=4, pady=2)
            self.char_buttons[name] = btn

        tk.Label(self.root, text="ロスター（選択中）", font=("Arial", 10, "bold")).pack(pady=(12, 0))
        self.roster_label = tk.Label(self.root, text="未選択", justify="left", fg="#333")
        self.roster_label.pack(pady=(2, 6))

        self.spike_frame = tk.LabelFrame(self.root, text="スパイク所持者", padx=8, pady=6)
        self.spike_frame.pack(fill="x", padx=18, pady=6)
        self.spike_hint = tk.Label(self.spike_frame, text="5人選択すると指定できます", fg="#666")
        self.spike_hint.pack(anchor="w")
        self.spike_buttons_frame = tk.Frame(self.spike_frame)
        self.spike_buttons_frame.pack(fill="x")

        tk.Button(self.root, text="選び直す", command=self.reset_roster).pack(pady=(4, 4))
        self.confirm_button = tk.Button(
            self.root, text="決定", font=("Arial", 12, "bold"), state="disabled", command=self.confirm
        )
        self.confirm_button.pack(pady=10)

    def select_character(self, name):
        if name in self.roster or len(self.roster) >= self.max_roster:
            return
        self.roster.append(name)
        self.char_buttons[name].config(state="disabled", relief="sunken")
        self.update_roster_ui()

    def rebuild_spike_choices(self):
        for widget in self.spike_buttons_frame.winfo_children():
            widget.destroy()
        if len(self.roster) != self.max_roster:
            self.spike_hint.config(text="5人選択すると指定できます")
            self.spike_var.set("")
            self.spike_holder = None
            return

        self.spike_hint.config(text="ラウンド開始時にスパイクを持つキャラクターを選択してください")
        if self.spike_var.get() not in self.roster:
            self.spike_var.set(self.roster[0])
        for name in self.roster:
            tk.Radiobutton(
                self.spike_buttons_frame,
                text=name,
                variable=self.spike_var,
                value=name,
                command=self.on_spike_changed,
                anchor="w",
            ).pack(fill="x")
        self.on_spike_changed()

    def on_spike_changed(self):
        value = self.spike_var.get()
        self.spike_holder = value if value in self.roster else None
        self.confirm_button.config(state="normal" if self.spike_holder else "disabled")

    def reset_roster(self):
        self.roster.clear()
        self.spike_holder = None
        self.spike_var.set("")
        for btn in self.char_buttons.values():
            btn.config(state="normal", relief="raised")
        self.confirm_button.config(state="disabled")
        self.update_roster_ui()

    def update_roster_ui(self):
        self.roster_label.config(text="\n".join(f"{i+1}. {n}" for i, n in enumerate(self.roster)) if self.roster else "未選択")
        self.rebuild_spike_choices()

    def confirm(self):
        roster = list(self.roster)
        spike_holder = self.spike_holder
        # 💡追加: 選択された表示名を内部キーに変換して渡す
        attacker_ctrl_key = ATTACKER_CONTROLLER_OPTIONS[self.attacker_ctrl_var.get()]
        defender_ctrl_key = DEFENDER_CONTROLLER_OPTIONS[self.defender_ctrl_var.get()]
        self.root.destroy()
        try:
            self.on_confirm(roster, spike_holder, attacker_ctrl_key, defender_ctrl_key)
        except TypeError:
            # 古いコールバックとの互換性
            try:
                self.on_confirm(roster, spike_holder)
            except TypeError:
                self.on_confirm(roster)

    def run(self):
        self.root.mainloop()