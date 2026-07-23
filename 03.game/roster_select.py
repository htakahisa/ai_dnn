"""ゲーム開始前の編成画面。

左右にスクロール可能な選手一覧、中央に選手詳細・選択状況・
スパイク・IGL、下部に固定の試合開始ボタンを配置する。
"""

import tkinter as tk

from character_stats import all_names, get_by_name
from game_core import calculate_combat_power
from party_presets import all_preset_names, get_preset


MAX_ROSTER = 5

ATTACKER_CONTROLLER_OPTIONS = {
    "AI ver1.0": "aiv1",
    "AI ver2.0": "aiv2",
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

        self.attacker_roster = []
        self.defender_roster = []
        self.char_buttons = {"A": {}, "D": {}}
        self.count_labels = {}
        self.roster_labels = {}
        self.team_summary_labels = {}

        self.root = tk.Tk()
        self.root.title("キャラクター編成")
        self.root.geometry("1260x820")
        self.root.minsize(980, 680)

        self.attacker_spike_var = tk.StringVar(master=self.root, value="")
        self.defender_spike_var = tk.StringVar(master=self.root, value="")
        # 旧コードとの互換用。現在の初期Attacker側を指す。
        self.spike_var = self.attacker_spike_var
        self.attacker_igl_var = tk.StringVar(master=self.root, value="")
        self.defender_igl_var = tk.StringVar(master=self.root, value="")
        self.attacker_ctrl_var = tk.StringVar(
            master=self.root,
            value=list(ATTACKER_CONTROLLER_OPTIONS.keys())[0],
        )
        self.defender_ctrl_var = tk.StringVar(
            master=self.root,
            value=list(DEFENDER_CONTROLLER_OPTIONS.keys())[0],
        )

        preset_names = all_preset_names()
        default_preset = preset_names[0] if preset_names else ""
        self.attacker_preset_var = tk.StringVar(
            master=self.root,
            value=default_preset,
        )
        self.defender_preset_var = tk.StringVar(
            master=self.root,
            value=default_preset,
        )

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._build_controller_bar()
        self._build_main_layout()
        self._build_footer()
        self.update_all_ui()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_controller_bar(self):
        frame = tk.LabelFrame(self.root, text="操作方法", padx=10, pady=7)
        frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))

        tk.Label(frame, text="Attacker:").grid(row=0, column=0, sticky="w")
        tk.OptionMenu(
            frame,
            self.attacker_ctrl_var,
            *ATTACKER_CONTROLLER_OPTIONS.keys(),
        ).grid(row=0, column=1, sticky="ew", padx=(7, 22))

        tk.Label(frame, text="Defender:").grid(row=0, column=2, sticky="w")
        tk.OptionMenu(
            frame,
            self.defender_ctrl_var,
            *DEFENDER_CONTROLLER_OPTIONS.keys(),
        ).grid(row=0, column=3, sticky="ew", padx=(7, 0))

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(3, weight=1)

        separator = tk.Frame(frame, height=1, bg="#bbbbbb")
        separator.grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(8, 7),
        )

        preset_names = all_preset_names()

        tk.Label(frame, text="Attackerプリセット:").grid(
            row=2, column=0, sticky="w"
        )
        attacker_menu = tk.OptionMenu(
            frame,
            self.attacker_preset_var,
            *preset_names,
        )
        attacker_menu.grid(row=2, column=1, sticky="ew", padx=(7, 8))
        tk.Button(
            frame,
            text="Attackerへ適用",
            command=lambda: self.apply_preset(
                "A", self.attacker_preset_var.get()
            ),
        ).grid(row=2, column=2, sticky="ew", padx=(0, 8))

        tk.Label(frame, text="Defenderプリセット:").grid(
            row=3, column=0, sticky="w", pady=(5, 0)
        )
        defender_menu = tk.OptionMenu(
            frame,
            self.defender_preset_var,
            *preset_names,
        )
        defender_menu.grid(
            row=3, column=1, sticky="ew", padx=(7, 8), pady=(5, 0)
        )
        tk.Button(
            frame,
            text="Defenderへ適用",
            command=lambda: self.apply_preset(
                "D", self.defender_preset_var.get()
            ),
        ).grid(row=3, column=2, sticky="ew", padx=(0, 8), pady=(5, 0))

        self.preset_status_label = tk.Label(
            frame,
            text="プリセットは5人・IGL・スパイク所持者をまとめて設定します",
            anchor="w",
            fg="#555555",
        )
        self.preset_status_label.grid(
            row=2,
            column=3,
            rowspan=2,
            sticky="nsew",
            padx=(5, 0),
        )

        if not preset_names:
            attacker_menu.config(state="disabled")
            defender_menu.config(state="disabled")

    def _build_main_layout(self):
        main = tk.Frame(self.root)
        main.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=4)
        main.grid_columnconfigure(2, weight=3)

        self._build_character_list(main, "A", 0, "ATTACKERS", "#c0392b")
        self._build_center_panel(main, 1)
        self._build_character_list(main, "D", 2, "DEFENDERS", "#218c4a")

    def _build_character_list(self, parent, team, column, title, accent):
        outer = tk.LabelFrame(parent, text=title, padx=7, pady=7)
        outer.grid(row=0, column=column, sticky="nsew", padx=5)
        outer.grid_rowconfigure(2, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self.count_labels[team] = tk.Label(
            outer,
            text=f"0 / {self.max_roster}",
            font=("Arial", 11, "bold"),
            fg=accent,
        )
        self.count_labels[team].grid(row=0, column=0, sticky="w")

        tk.Label(
            outer,
            text="クリックで追加／カーソルで詳細表示",
            anchor="w",
            fg="#555555",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 5))

        list_host = tk.Frame(outer)
        list_host.grid(row=2, column=0, sticky="nsew")
        list_host.grid_rowconfigure(0, weight=1)
        list_host.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(list_host, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            list_host,
            orient="vertical",
            command=canvas.yview,
        )
        inner = tk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind(
            "<Configure>",
            lambda _event, cv=canvas: cv.configure(scrollregion=cv.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event, cv=canvas, win=window_id: cv.itemconfigure(
                win, width=event.width
            ),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        def bind_wheel(_event, cv=canvas):
            cv.bind_all(
                "<MouseWheel>",
                lambda event: cv.yview_scroll(
                    int(-event.delta / 120), "units"
                ),
            )

        def unbind_wheel(_event, cv=canvas):
            cv.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", bind_wheel)
        canvas.bind("<Leave>", unbind_wheel)

        for name in all_names():
            button = tk.Button(
                inner,
                text=name,
                anchor="w",
                command=lambda n=name, t=team: self.select_character(t, n),
            )
            button.bind(
                "<Enter>",
                lambda _event, n=name: self._show_hovered_character(n),
            )
            button.bind(
                "<FocusIn>",
                lambda _event, n=name: self._show_hovered_character(n),
            )
            button.pack(fill="x", padx=2, pady=1)
            self.char_buttons[team][name] = button

    def _build_center_panel(self, parent, column):
        center = tk.Frame(parent)
        center.grid(row=0, column=column, sticky="nsew", padx=5)
        center.grid_columnconfigure(0, weight=1)

        self._build_hover_stats(center)
        self._build_selected_rosters(center)
        self._build_spike_section(center)
        self._build_igl_section(center)

    def _build_hover_stats(self, parent):
        frame = tk.LabelFrame(
            parent,
            text="カーソル中のプレイヤー",
            padx=9,
            pady=7,
        )
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.hover_name_label = tk.Label(
            frame,
            text="選手名にカーソルを合わせてください",
            anchor="w",
            font=("Arial", 11, "bold"),
        )
        self.hover_name_label.pack(fill="x")

        self.hover_stats_label = tk.Label(
            frame,
            text=(
                "命中率: -   回避率: -   HS率: -   IQ: -\n"
                "反応速度: -   ロール: -\n"
                "影響度: -   総合戦闘力: -"
            ),
            justify="left",
            anchor="w",
            font=("Arial", 10),
        )
        self.hover_stats_label.pack(fill="x", pady=(4, 0))

    def _build_selected_rosters(self, parent):
        frame = tk.LabelFrame(parent, text="選択中のパーティ", padx=8, pady=7)
        frame.grid(row=1, column=0, sticky="ew", pady=6)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        self._build_selected_team_box(frame, "A", 0, "ATTACKERS")
        self._build_selected_team_box(frame, "D", 1, "DEFENDERS")

    def _build_selected_team_box(self, parent, team, column, title):
        box = tk.LabelFrame(parent, text=title, padx=6, pady=5)
        box.grid(row=0, column=column, sticky="nsew", padx=4)

        self.roster_labels[team] = tk.Label(
            box,
            text="未選択",
            justify="left",
            anchor="nw",
            height=self.max_roster,
        )
        self.roster_labels[team].pack(fill="x")

        self.team_summary_labels[team] = tk.Label(
            box,
            text="影響度合計: 0\nIQ低下なし",
            justify="left",
            anchor="w",
            font=("Arial", 9, "bold"),
            fg="#2874a6",
        )
        self.team_summary_labels[team].pack(fill="x", pady=(4, 0))

        controls = tk.Frame(box)
        controls.pack(fill="x", pady=(5, 0))
        tk.Button(
            controls,
            text="1人戻す",
            command=lambda t=team: self.remove_last(t),
        ).pack(side="left")
        tk.Button(
            controls,
            text="全解除",
            command=lambda t=team: self.reset_roster(t),
        ).pack(side="right")

    def _build_spike_section(self, parent):
        self.spike_frame = tk.LabelFrame(
            parent,
            text="攻撃側になった時のスパイク所持者（両チーム分）",
            padx=8,
            pady=6,
        )
        self.spike_frame.grid(row=2, column=0, sticky="ew", pady=6)

        self.spike_hint = tk.Label(
            self.spike_frame,
            text=(
                f"各チームが{self.max_roster}人揃うと、"
                "1〜12R用と13R以降用をそれぞれ選択できます"
            ),
            fg="#666666",
            anchor="w",
            justify="left",
        )
        self.spike_hint.pack(fill="x")

        columns = tk.Frame(self.spike_frame)
        columns.pack(fill="x", pady=(4, 0))
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)

        self.attacker_spike_box = tk.LabelFrame(
            columns,
            text="1〜12R：初期ATTACKERチーム",
            padx=5,
            pady=3,
        )
        self.attacker_spike_box.grid(
            row=0, column=0, sticky="nsew", padx=(0, 4)
        )

        self.defender_spike_box = tk.LabelFrame(
            columns,
            text="13R以降：初期DEFENDERチーム",
            padx=5,
            pady=3,
        )
        self.defender_spike_box.grid(
            row=0, column=1, sticky="nsew", padx=(4, 0)
        )

    def _build_igl_section(self, parent):
        self.igl_frame = tk.LabelFrame(
            parent,
            text="IGL",
            padx=8,
            pady=6,
        )
        self.igl_frame.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        self.igl_hint = tk.Label(
            self.igl_frame,
            text=f"両チームを{self.max_roster}人ずつ選択してください",
            fg="#666666",
            anchor="w",
        )
        self.igl_hint.pack(fill="x")

        columns = tk.Frame(self.igl_frame)
        columns.pack(fill="x", pady=(4, 0))
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)

        self.attacker_igl_box = tk.LabelFrame(
            columns,
            text="ATTACKER IGL",
            padx=5,
            pady=3,
        )
        self.attacker_igl_box.grid(
            row=0, column=0, sticky="nsew", padx=(0, 4)
        )

        self.defender_igl_box = tk.LabelFrame(
            columns,
            text="DEFENDER IGL",
            padx=5,
            pady=3,
        )
        self.defender_igl_box.grid(
            row=0, column=1, sticky="nsew", padx=(4, 0)
        )

    def _build_footer(self):
        footer = tk.Frame(self.root)
        footer.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 10))
        footer.grid_columnconfigure(0, weight=1)

        self.validation_label = tk.Label(
            footer,
            text="",
            fg="#b03a2e",
            anchor="w",
        )
        self.validation_label.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        self.confirm_button = tk.Button(
            footer,
            text="決定して試合開始",
            font=("Arial", 12, "bold"),
            state="disabled",
            width=18,
            command=self.confirm,
        )
        self.confirm_button.grid(row=0, column=1, sticky="e")

    # ------------------------------------------------------------------
    # Stats and summaries
    # ------------------------------------------------------------------

    @staticmethod
    def _format_number(value):
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _show_hovered_character(self, name):
        stats = get_by_name(name)
        if stats is None:
            self.hover_name_label.config(text=name)
            self.hover_stats_label.config(text="ステータスを取得できませんでした")
            return

        combat_power = calculate_combat_power(
            float(stats.hs_pct) / 100.0,
            float(stats.dodge_pct) / 100.0,
            float(stats.iq),
            float(stats.hit_pct) / 100.0,
            float(stats.reaction),
        )

        self.hover_name_label.config(text=name)
        self.hover_stats_label.config(
            text=(
                f"命中率: {self._format_number(stats.hit_pct)}%   "
                f"回避率: {self._format_number(stats.dodge_pct)}%   "
                f"HS率: {self._format_number(stats.hs_pct)}%   "
                f"IQ: {self._format_number(stats.iq)}\n"
                f"反応速度: {self._format_number(stats.reaction)}   "
                f"ロール: {stats.role}\n"
                f"影響度: {self._format_number(stats.influence)}   "
                f"総合戦闘力: {self._format_number(combat_power)}"
            )
        )

    def _team_influence_summary(self, roster):
        total = 0.0
        for name in roster:
            stats = get_by_name(name)
            if stats is not None:
                total += max(0.0, float(stats.influence))
        penalty = max(0.0, (total - 300.0) / 10.0)
        return total, penalty

    # ------------------------------------------------------------------
    # Selection behavior
    # ------------------------------------------------------------------

    def _roster_for(self, team):
        return self.attacker_roster if team == "A" else self.defender_roster

    def _other_roster_for(self, team):
        return self.defender_roster if team == "A" else self.attacker_roster

    def apply_preset(self, team, preset_name):
        """指定したプリセットを片方のチームへ一括適用する。"""
        preset = get_preset(preset_name)
        team_label = "Attacker" if team == "A" else "Defender"

        if preset is None:
            self.preset_status_label.config(
                text=f"{team_label}: プリセットが見つかりません",
                fg="#b03a2e",
            )
            return

        errors = preset.validate(all_names(), expected_size=self.max_roster)
        if errors:
            self.preset_status_label.config(
                text=f"{preset.name}を適用できません: " + " / ".join(errors),
                fg="#b03a2e",
            )
            return

        roster = self._roster_for(team)
        roster.clear()
        roster.extend(preset.players)

        selected_spike_holder = (
            preset.spike_holder
            if preset.spike_holder in preset.players
            else preset.players[0]
        )

        if team == "A":
            self.attacker_igl_var.set(preset.igl)
            self.attacker_spike_var.set(selected_spike_holder)
        else:
            self.defender_igl_var.set(preset.igl)
            self.defender_spike_var.set(selected_spike_holder)

        self.update_all_ui()

        details = (
            f"{team_label}へ「{preset.name}」を適用しました"
            f"｜IGL: {preset.igl}"
        )
        spike_holder = (
            self.attacker_spike_var.get()
            if team == "A"
            else self.defender_spike_var.get()
        )
        details += f"｜攻撃時スパイク: {spike_holder}"
        if preset.description:
            details += f"｜{preset.description}"

        self.preset_status_label.config(text=details, fg="#2874a6")

    def select_character(self, team, name):
        roster = self._roster_for(team)

        # 同じパーティ内での重複だけを禁止する。
        # AttackerとDefenderは独立しているため、
        # 片方で選択済みでも、もう片方では選択可能。
        if name in roster or len(roster) >= self.max_roster:
            return

        roster.append(name)
        self.update_all_ui()

    def remove_last(self, team):
        roster = self._roster_for(team)
        if not roster:
            return

        removed = roster.pop()
        if team == "A":
            if self.attacker_spike_var.get() == removed:
                self.attacker_spike_var.set("")
            if self.attacker_igl_var.get() == removed:
                self.attacker_igl_var.set("")
        else:
            if self.defender_spike_var.get() == removed:
                self.defender_spike_var.set("")
            if self.defender_igl_var.get() == removed:
                self.defender_igl_var.set("")

        self.update_all_ui()

    def reset_roster(self, team):
        self._roster_for(team).clear()
        if team == "A":
            self.attacker_spike_var.set("")
            self.attacker_igl_var.set("")
        else:
            self.defender_spike_var.set("")
            self.defender_igl_var.set("")
        self.update_all_ui()

    # ------------------------------------------------------------------
    # UI refresh
    # ------------------------------------------------------------------

    def update_all_ui(self):
        self._update_roster_labels()
        self._update_character_buttons()
        self._rebuild_spike_choices()
        self._rebuild_igl_choices()
        self._update_confirm_state()

    def _update_roster_labels(self):
        for team in ("A", "D"):
            roster = self._roster_for(team)
            self.count_labels[team].config(
                text=f"{len(roster)} / {self.max_roster}"
            )

            roster_text = (
                "\n".join(
                    f"{index + 1}. {name}"
                    for index, name in enumerate(roster)
                )
                if roster
                else "未選択"
            )
            self.roster_labels[team].config(text=roster_text)

            total, penalty = self._team_influence_summary(roster)
            if penalty > 0:
                summary = (
                    f"影響度合計: {self._format_number(total)}（超過）\n"
                    f"1人あたりIQ -{self._format_number(penalty)}"
                )
                color = "#b03a2e"
            else:
                summary = (
                    f"影響度合計: {self._format_number(total)}\n"
                    "IQ低下なし"
                )
                color = "#2874a6"

            self.team_summary_labels[team].config(text=summary, fg=color)

    def _update_character_buttons(self):
        for team in ("A", "D"):
            roster = self._roster_for(team)
            full = len(roster) >= self.max_roster

            for name, button in self.char_buttons[team].items():
                own = name in roster

                # 無効化するのは、
                # 1. そのチームですでに選択済み
                # 2. そのチームが5人埋まっている
                # のどちらかだけ。
                #
                # 反対側のチームで選択されていても無効化しない。
                if own:
                    button.config(
                        state="disabled",
                        relief="sunken",
                        bg="#d6eaf8" if team == "A" else "#d5f5e3",
                    )
                elif full:
                    button.config(
                        state="disabled",
                        relief="raised",
                        bg="#eeeeee",
                    )
                else:
                    button.config(
                        state="normal",
                        relief="raised",
                        bg="SystemButtonFace",
                    )

    def _rebuild_spike_choices(self):
        """両チームについて、攻撃側になった時のスパイク所持者を設定する。"""
        for box in (self.attacker_spike_box, self.defender_spike_box):
            for widget in box.winfo_children():
                widget.destroy()

        attacker_ready = len(self.attacker_roster) == self.max_roster
        defender_ready = len(self.defender_roster) == self.max_roster

        if attacker_ready and defender_ready:
            hint = "両チームの攻撃時スパイク所持者を選択してください"
        elif attacker_ready:
            hint = "初期Attacker側の所持者を選択できます"
        elif defender_ready:
            hint = "初期Defender側の13R以降の所持者を選択できます"
        else:
            hint = (
                f"各チームが{self.max_roster}人揃うと、"
                "そのチームの所持者を選択できます"
            )
        self.spike_hint.config(text=hint)

        if attacker_ready:
            if self.attacker_spike_var.get() not in self.attacker_roster:
                self.attacker_spike_var.set(self.attacker_roster[0])
            for index, name in enumerate(self.attacker_roster):
                tk.Radiobutton(
                    self.attacker_spike_box,
                    text=name,
                    variable=self.attacker_spike_var,
                    value=name,
                    command=self._update_confirm_state,
                ).grid(
                    row=index // 2,
                    column=index % 2,
                    sticky="w",
                    padx=(0, 10),
                )
        else:
            self.attacker_spike_var.set("")
            tk.Label(
                self.attacker_spike_box,
                text=f"あと{self.max_roster - len(self.attacker_roster)}人",
                fg="#666666",
            ).grid(row=0, column=0, sticky="w")

        if defender_ready:
            if self.defender_spike_var.get() not in self.defender_roster:
                self.defender_spike_var.set(self.defender_roster[0])
            for index, name in enumerate(self.defender_roster):
                tk.Radiobutton(
                    self.defender_spike_box,
                    text=name,
                    variable=self.defender_spike_var,
                    value=name,
                    command=self._update_confirm_state,
                ).grid(
                    row=index // 2,
                    column=index % 2,
                    sticky="w",
                    padx=(0, 10),
                )
        else:
            self.defender_spike_var.set("")
            tk.Label(
                self.defender_spike_box,
                text=f"あと{self.max_roster - len(self.defender_roster)}人",
                fg="#666666",
            ).grid(row=0, column=0, sticky="w")

    def _rebuild_igl_choices(self):
        """各チームが5人揃った時点で、そのチームのIGLを選択可能にする。"""
        for box in (self.attacker_igl_box, self.defender_igl_box):
            for widget in box.winfo_children():
                widget.destroy()

        attacker_ready = len(self.attacker_roster) == self.max_roster
        defender_ready = len(self.defender_roster) == self.max_roster

        if attacker_ready and defender_ready:
            hint = "各チームからIGLを1人選択してください"
        elif attacker_ready:
            hint = (
                "AttackerのIGLを選択できます。"
                f" Defenderはあと"
                f"{self.max_roster - len(self.defender_roster)}人です"
            )
        elif defender_ready:
            hint = (
                "DefenderのIGLを選択できます。"
                f" Attackerはあと"
                f"{self.max_roster - len(self.attacker_roster)}人です"
            )
        else:
            hint = (
                f"各チームが{self.max_roster}人揃うと、"
                "そのチームのIGLを選択できます"
            )
        self.igl_hint.config(text=hint)

        if attacker_ready:
            # プリセットで設定されたIGLが有効なら、そのまま保持する。
            if self.attacker_igl_var.get() not in self.attacker_roster:
                self.attacker_igl_var.set(self.attacker_roster[0])

            for row, name in enumerate(self.attacker_roster):
                tk.Radiobutton(
                    self.attacker_igl_box,
                    text=name,
                    variable=self.attacker_igl_var,
                    value=name,
                    command=self._update_confirm_state,
                ).grid(row=row, column=0, sticky="w")
        else:
            self.attacker_igl_var.set("")
            tk.Label(
                self.attacker_igl_box,
                text=f"あと{self.max_roster - len(self.attacker_roster)}人",
                fg="#666666",
            ).grid(row=0, column=0, sticky="w")

        if defender_ready:
            # プリセットで設定されたIGLが有効なら、そのまま保持する。
            if self.defender_igl_var.get() not in self.defender_roster:
                self.defender_igl_var.set(self.defender_roster[0])

            for row, name in enumerate(self.defender_roster):
                tk.Radiobutton(
                    self.defender_igl_box,
                    text=name,
                    variable=self.defender_igl_var,
                    value=name,
                    command=self._update_confirm_state,
                ).grid(row=row, column=0, sticky="w")
        else:
            self.defender_igl_var.set("")
            tk.Label(
                self.defender_igl_box,
                text=f"あと{self.max_roster - len(self.defender_roster)}人",
                fg="#666666",
            ).grid(row=0, column=0, sticky="w")

    def _update_confirm_state(self):
        attackers_ready = len(self.attacker_roster) == self.max_roster
        defenders_ready = len(self.defender_roster) == self.max_roster
        attacker_spike_ready = (
            self.attacker_spike_var.get() in self.attacker_roster
        )
        defender_spike_ready = (
            self.defender_spike_var.get() in self.defender_roster
        )
        attacker_igl_ready = (
            self.attacker_igl_var.get() in self.attacker_roster
        )
        defender_igl_ready = (
            self.defender_igl_var.get() in self.defender_roster
        )

        if not attackers_ready:
            message = (
                f"アタッカーをあと"
                f"{self.max_roster - len(self.attacker_roster)}人選択してください"
            )
        elif not defenders_ready:
            message = (
                f"ディフェンダーをあと"
                f"{self.max_roster - len(self.defender_roster)}人選択してください"
            )
        elif not attacker_spike_ready:
            message = "初期Attacker側のスパイク所持者を選択してください"
        elif not defender_spike_ready:
            message = "初期Defender側の13R以降のスパイク所持者を選択してください"
        elif not attacker_igl_ready:
            message = "アタッカーのIGLを選択してください"
        elif not defender_igl_ready:
            message = "ディフェンダーのIGLを選択してください"
        else:
            message = "準備完了"

        self.validation_label.config(
            text=message,
            fg="#2874a6" if message == "準備完了" else "#b03a2e",
        )

        ready = (
            attackers_ready
            and defenders_ready
            and attacker_spike_ready
            and defender_spike_ready
            and attacker_igl_ready
            and defender_igl_ready
        )
        self.confirm_button.config(
            state="normal" if ready else "disabled"
        )

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def confirm(self):
        attacker_ctrl_key = ATTACKER_CONTROLLER_OPTIONS[
            self.attacker_ctrl_var.get()
        ]
        defender_ctrl_key = DEFENDER_CONTROLLER_OPTIONS[
            self.defender_ctrl_var.get()
        ]

        args = (
            list(self.attacker_roster),
            list(self.defender_roster),
            self.attacker_spike_var.get(),
            self.defender_spike_var.get(),
            attacker_ctrl_key,
            defender_ctrl_key,
            self.attacker_igl_var.get(),
            self.defender_igl_var.get(),
        )

        self.root.destroy()
        self.on_confirm(*args)

    def run(self):
        self.root.mainloop()
