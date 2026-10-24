"""Japanese desktop forms for teaching, training, and asking tactical questions."""

# 通常設定はここで変更。保存先・生成設定は tl1_ui_service_tactics.py 冒頭。
WINDOW_TITLE = "戦術ラボ — 自作言語モデル"
WINDOW_SIZE = "1180x850"
MIN_WINDOW_SIZE = (980, 740)
UI_FONT = "Yu Gothic UI"
UI_FONT_SIZE = 11
EVENT_POLL_MS = 100
CONVERSATION_DIRECTORY_NAME = "conversations"  # 教師データの保存先と同じ data/ 配下。

import queue
import re
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from tl1_ui_service_tactics import CHECKPOINT_NAMES, MAX_NEW_TOKENS, TeacherStore, WorkService
from tl1_dates_tactics import record_date, select_records
from tl1_conversations_tactics import ConversationStore

EXAMPLE = {
    "question": "設置後、人数有利なときはどう守ればよい？",
    "context": "攻撃側。Aサイトに設置済み。味方3人、敵1人。敵の現在位置は不明。",
    "answer": "敵を単独で追いかけず、解除を確認できる位置で守る。味方と互いに援護できる位置を取る。",
    "reason": "単独で交戦すると各個撃破される可能性があるため。解除を妨害できない位置なら移動する。",
    "group": "Aサイトポストプラント・人数有利",
}
FIELD_LABELS = {"question": "質問", "context": "状況・条件", "answer": "正解回答",
                "reason": "判断理由・例外条件", "group": "グループ名"}


class TacticsApp:
    def __init__(self, root, store=None, service=None, conversations=None):
        self.root = root
        self.store = store or TeacherStore()
        self.service = service or WorkService()
        self.conversations = conversations or ConversationStore(self.store.path.parent / CONVERSATION_DIRECTORY_NAME)
        self.thread = None
        self._pending_turn = None
        self._answer_metadata = {}
        self._refreshing_threads = False
        self._teaching_history = []
        self.rows = []
        self.edit_id = None
        self._original = {key: "" for key in FIELD_LABELS}
        self._refreshing = False
        self._closing = False
        self._after_id = None
        self._last_date = self.service.as_of_date
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*MIN_WINDOW_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._build()
        self._set_record()
        self.refresh()
        self.reload_threads(initial=True)
        self._poll()

    def _style(self):
        self.root.configure(background="#f2f5fa")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=(UI_FONT, UI_FONT_SIZE), background="#f2f5fa", foreground="#1d2b43")
        style.configure("TNotebook", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(24, 12), background="#e4eaf3")
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])
        style.configure("TButton", padding=(14, 8))
        style.configure("Primary.TButton", foreground="#ffffff", background="#245fc7")
        style.map("Primary.TButton", background=[("active", "#194da6"), ("disabled", "#9caac3")])
        style.configure("Title.TLabel", font=(UI_FONT, 22, "bold"))
        style.configure("Section.TLabel", font=(UI_FONT, 13, "bold"))
        style.configure("Muted.TLabel", foreground="#617087")
        style.configure("Treeview", rowheight=34, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=(UI_FONT, UI_FONT_SIZE, "bold"))
        style.configure("TProgressbar", background="#245fc7")

    @staticmethod
    def _text(parent, height=3, readonly=False):
        widget = ScrolledText(parent, height=height, wrap="word", undo=not readonly,
                              font=(UI_FONT, UI_FONT_SIZE), relief="solid", borderwidth=1,
                              background="#ffffff", foreground="#1d2b43",
                              insertbackground="#1d2b43", padx=10, pady=7)
        widget.bind("<Control-a>", lambda event: TacticsApp._select_all(widget))
        if readonly:
            widget.configure(state="disabled")
        return widget

    @staticmethod
    def _select_all(widget):
        widget.tag_add("sel", "1.0", "end-1c")
        return "break"

    @staticmethod
    def _write(widget, value, readonly=False):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        if readonly:
            widget.configure(state="disabled")

    def _build(self):
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(header, text="戦術ラボ", style="Title.TLabel").pack(side="left")
        self.summary = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.summary, style="Muted.TLabel").pack(side="right")
        strip = ttk.Frame(outer)
        strip.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(strip, text="戦術を教える → 学習する → 質問する",
                  style="Muted.TLabel").pack(side="left")
        controls = ttk.Frame(strip)
        controls.pack(side="right")
        self.today = tk.StringVar(value=f"今日：{self.service.as_of_date}（自動更新）")
        ttk.Label(controls, textvariable=self.today, style="Muted.TLabel").pack(side="left")
        self.tabs = ttk.Notebook(outer)
        self.tabs.grid(row=2, column=0, sticky="nsew")
        self.teacher_tab = ttk.Frame(self.tabs, padding=16)
        self.training_tab = ttk.Frame(self.tabs, padding=20)
        self.chat_tab = ttk.Frame(self.tabs, padding=20)
        self.tabs.add(self.teacher_tab, text="1  教師データ")
        self.tabs.add(self.training_tab, text="2  学習")
        self.tabs.add(self.chat_tab, text="3  質問")
        self._teacher_screen()
        self._training_screen()
        self._chat_screen()
        self.status = tk.StringVar(value="教師データを登録するか、一覧から選んで編集してください。")
        ttk.Label(outer, textvariable=self.status, wraplength=900).grid(row=3, column=0, sticky="ew", pady=(14, 0))

    def _teacher_screen(self):
        panes = ttk.Panedwindow(self.teacher_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        sidebar = ttk.Frame(panes, padding=(0, 0, 14, 0))
        form = ttk.Frame(panes, padding=(14, 0, 0, 0))
        panes.add(sidebar, weight=1)
        panes.add(form, weight=2)
        ttk.Label(sidebar, text="登録済みデータ", style="Section.TLabel").pack(anchor="w")
        self.filter = tk.StringVar()
        ttk.Entry(sidebar, textvariable=self.filter).pack(fill="x", pady=(10, 5))
        ttk.Label(sidebar, text="質問やグループ名で絞り込み", style="Muted.TLabel").pack(anchor="w")
        table = ttk.Frame(sidebar)
        table.pack(fill="both", expand=True, pady=10)
        self.tree = ttk.Treeview(table, columns=("group", "question"), show="headings", selectmode="browse")
        self.tree.heading("group", text="グループ")
        self.tree.heading("question", text="質問")
        self.tree.column("group", width=135, minwidth=80)
        self.tree.column("question", width=200, minwidth=100)
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.select_record)
        self.filter.trace_add("write", lambda *_: self._show_rows())
        controls = ttk.Frame(sidebar)
        controls.pack(fill="x")
        ttk.Button(controls, text="新しく教える", command=self.new_record).pack(side="left")
        ttk.Button(controls, text="再読込", command=self.reload).pack(side="right")
        form.columnconfigure(0, weight=1)
        self.edit_label = tk.StringVar(value="新しい教師データ")
        ttk.Label(form, textvariable=self.edit_label, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(form, text="自由に改行して入力できます。質問と正解回答は必須です。",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 8))
        self.fields = {}
        for index, key in enumerate(("question", "context", "answer", "reason")):
            row = 2 + index * 2
            label = FIELD_LABELS[key] + (" *" if key in ("question", "answer") else "（省略可）")
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=(5, 3))
            widget = self._text(form, height=2 if key == "question" else 3)
            widget.grid(row=row + 1, column=0, sticky="nsew")
            form.rowconfigure(row + 1, weight=1)
            self.fields[key] = widget
        ttk.Label(form, text="グループ名（日本語可・空欄なら自動）").grid(row=10, column=0, sticky="w", pady=(10, 3))
        self.group = tk.StringVar()
        self.group_combo = ttk.Combobox(form, textvariable=self.group)
        self.group_combo.grid(row=11, column=0, sticky="ew")
        ttk.Label(form, text="同じグループの更新は新しい日付で登録。最新日のデータを使用します。",
                  style="Muted.TLabel").grid(row=12, column=0, sticky="w", pady=(4, 6))
        classification = ttk.Frame(form)
        classification.grid(row=13, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(classification, text="適用開始日").pack(side="left", padx=(0, 6))
        self.valid_from = tk.StringVar()
        ttk.Entry(classification, textvariable=self.valid_from, width=13).pack(side="left")
        ttk.Button(classification, text="今日", command=lambda: self.valid_from.set(self.service.as_of_date)).pack(side="left", padx=6)
        ttk.Label(classification, text="YYYY-MM-DD／空欄なら共通", style="Muted.TLabel").pack(side="left")
        buttons = ttk.Frame(form)
        buttons.grid(row=14, column=0, sticky="ew")
        self.save_label = tk.StringVar(value="教師データを登録")
        self.save_button = ttk.Button(buttons, textvariable=self.save_label, command=self.save_record,
                                      style="Primary.TButton")
        self.save_button.pack(side="left")
        ttk.Button(buttons, text="入力例を見る", command=self.show_example).pack(side="right")
        self.teaching_history_label = tk.StringVar(value="")
        ttk.Label(form, textvariable=self.teaching_history_label, style="Muted.TLabel").grid(row=15, column=0, sticky="w", pady=(5, 0))

    def _training_screen(self):
        tab = self.training_tab
        ttk.Label(tab, text="登録した戦術をモデルに学習させる", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, text="共通と今日有効なデータを学習します。同じグループは、最新の適用開始日の内容を使います。",
                  style="Muted.TLabel").pack(anchor="w", pady=(10, 6))
        ttk.Label(tab, text="登録しただけでは回答に反映されません。追加・修正後は、ここから学習してください。",
                  style="Muted.TLabel").pack(anchor="w", pady=(0, 16))
        buttons = ttk.Frame(tab)
        buttons.pack(fill="x")
        self.train_button = ttk.Button(buttons, text="学習を開始", command=self.start_training,
                                       style="Primary.TButton")
        self.train_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="停止して保存", command=self.stop_work, state="disabled")
        self.stop_button.pack(side="left", padx=10)
        self.progress_text = tk.StringVar(value="待機中")
        ttk.Label(tab, textvariable=self.progress_text).pack(anchor="w", pady=(20, 8))
        self.progress = ttk.Progressbar(tab, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        ttk.Label(tab, text="学習ログ", style="Section.TLabel").pack(anchor="w", pady=(20, 8))
        self.log = self._text(tab, height=15, readonly=True)
        self.log.pack(fill="both", expand=True)
        ttk.Label(tab, text="学習回数やモデルの設定は、実行手順に記載したスクリプト冒頭の定数で変更できます。",
                  style="Muted.TLabel", wraplength=1000).pack(anchor="w", pady=(12, 0))

    def _chat_screen(self):
        panes = ttk.Panedwindow(self.chat_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        sidebar = ttk.Frame(panes, padding=(0, 0, 12, 0))
        tab = ttk.Frame(panes, padding=(12, 0, 0, 0))
        panes.add(sidebar, weight=1)
        panes.add(tab, weight=4)
        self.new_thread_button = ttk.Button(sidebar, text="＋ 新しい会話", command=self.new_thread, style="Primary.TButton")
        self.new_thread_button.pack(fill="x", pady=(0, 12))
        ttk.Label(sidebar, text="保存した会話", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        table = ttk.Frame(sidebar)
        table.pack(fill="both", expand=True)
        self.thread_tree = ttk.Treeview(table, columns=("title",), show="headings", selectmode="browse")
        self.thread_tree.heading("title", text="スレッド")
        self.thread_tree.column("title", width=215, minwidth=130)
        scroll = ttk.Scrollbar(table, command=self.thread_tree.yview)
        self.thread_tree.configure(yscrollcommand=scroll.set)
        self.thread_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.thread_tree.bind("<<TreeviewSelect>>", self.select_thread)
        self.reload_threads_button = ttk.Button(sidebar, text="会話を再読込", command=self.reload_threads)
        self.reload_threads_button.pack(fill="x", pady=(10, 0))
        self.delete_thread_button = ttk.Button(sidebar, text="選択した会話を削除", command=self.delete_thread, state="disabled")
        self.delete_thread_button.pack(fill="x", pady=(6, 0))
        tab.columnconfigure(0, weight=1)
        self.thread_title = tk.StringVar(value="新しい会話")
        ttk.Label(tab, textvariable=self.thread_title, style="Section.TLabel").grid(row=0, column=0, sticky="w")
        model_controls = ttk.Frame(tab)
        model_controls.grid(row=1, column=0, sticky="ew", pady=(8, 12))
        selection = ttk.Frame(model_controls)
        selection.pack(fill="x")
        ttk.Label(selection, text="使用モデル").pack(side="left", padx=(0, 8))
        self.model_choice = tk.StringVar(value=self.service.checkpoint_path.name)
        self.model_combo = ttk.Combobox(selection, textvariable=self.model_choice,
                                        values=CHECKPOINT_NAMES, state="readonly", width=12)
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", self.select_model)
        self.model_info = tk.StringVar(value=f"{self.service.checkpoint_path.name}（未読込）")
        ttk.Label(selection, textvariable=self.model_info, style="Muted.TLabel").pack(side="left", padx=10)
        ttk.Label(model_controls, text="best：評価損失が最小のモデル ／ latest：最後に保存したモデル",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(model_controls, text=f"最大 {MAX_NEW_TOKENS:,} トークン生成。会話を参照し、履歴は自動保存します。",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        self.answer = self._text(tab, height=14, readonly=True)
        self.answer.grid(row=2, column=0, sticky="nsew")
        self.answer.tag_configure("user", foreground="#245fc7", font=(UI_FONT, UI_FONT_SIZE, "bold"))
        self.answer.tag_configure("assistant", foreground="#1d2b43", font=(UI_FONT, UI_FONT_SIZE, "bold"))
        self.answer.tag_configure("muted", foreground="#617087")
        tab.rowconfigure(2, weight=1)
        self.teach_reply_button = ttk.Button(tab, text="この回答を修正して教師データにする", command=self.teach_reply)
        self.teach_reply_button.grid(row=3, column=0, sticky="e", pady=(6, 0))
        ttk.Label(tab, text="状況・条件（このスレッドで引き継ぎ／変更可）").grid(row=4, column=0, sticky="w", pady=(10, 4))
        self.context = self._text(tab, height=2)
        self.context.grid(row=5, column=0, sticky="ew")
        ttk.Label(tab, text="メッセージ（Ctrl＋Enterで送信）").grid(row=6, column=0, sticky="w", pady=(10, 4))
        self.question = self._text(tab, height=3)
        self.question.grid(row=7, column=0, sticky="ew")
        self.question.bind("<Control-Return>", lambda event: self._send_shortcut())
        buttons = ttk.Frame(tab)
        buttons.grid(row=8, column=0, sticky="ew", pady=(12, 0))
        self.ask_button = ttk.Button(buttons, text="送信", command=self.ask, style="Primary.TButton")
        self.ask_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="生成を停止", command=self.stop_work, state="disabled")
        self.cancel_button.pack(side="left", padx=10)
        self.chat_state = tk.StringVar(value="")
        ttk.Label(buttons, textvariable=self.chat_state, style="Muted.TLabel").pack(side="left", padx=10)

    def _send_shortcut(self):
        self.ask()
        return "break"

    def select_model(self, _event=None):
        try:
            self.service.select_checkpoint(self.model_choice.get())
            name = self.service.checkpoint_path.name
            self.model_info.set(f"{name}（未読込）")
            self.chat_state.set("")
            self._update_summary()
            self.status.set(f"使用モデルを {name} に切り替えました。次の送信時に読み込みます。" if
                            self.service.checkpoint_path.is_file() else
                            f"{name} はまだ保存されていません。別のモデルを選ぶか、学習してください。")
        except ValueError as exc:
            self.model_choice.set(self.service.checkpoint_path.name)
            self.status.set(str(exc))

    def _can_leave_thread(self):
        saved_context = self.thread["context"] if self.thread else ""
        dirty = self.question.get("1.0", "end-1c").strip() or self.context.get("1.0", "end-1c").strip() != saved_context
        return not dirty or messagebox.askokcancel("未送信の入力", "未送信の入力を破棄して移動しますか？", parent=self.root)

    def _thread_list(self):
        self._refreshing_threads = True
        self.thread_tree.delete(*self.thread_tree.get_children())
        threads = self.conversations.list_threads()
        for thread in threads:
            self.thread_tree.insert("", "end", iid=thread["id"], values=(thread["title"],))
        if self.thread and self.thread_tree.exists(self.thread["id"]):
            self.thread_tree.selection_set(self.thread["id"])
        self._refreshing_threads = False
        return threads

    def reload_threads(self, initial=False):
        if self.service.busy == "answer" or not initial and not self._can_leave_thread():
            return
        try:
            threads = self._thread_list()
            identifier = self.thread["id"] if self.thread else None
            self.thread = next((thread for thread in threads if thread["id"] == identifier), threads[0] if threads else None)
            if self.thread:
                self.thread_tree.selection_set(self.thread["id"])
            self._display_thread(restore_inputs=True)
        except (ValueError, OSError) as exc:
            self.status.set(f"会話を読み込めませんでした: {exc}")

    def new_thread(self):
        if self.service.busy == "answer" or not self._can_leave_thread():
            return
        try:
            self.thread = self.conversations.create()
            self._thread_list()
            self._display_thread(restore_inputs=True)
            self.question.focus_set()
        except (ValueError, OSError) as exc:
            self.status.set(f"会話を作成できませんでした: {exc}")

    def select_thread(self, _event=None):
        if self._refreshing_threads:
            return
        selected = self.thread_tree.selection()
        if not selected or self.thread and selected[0] == self.thread["id"]:
            return
        if self.service.busy == "answer" or not self._can_leave_thread():
            self.thread_tree.selection_remove(*selected)
            if self.thread:
                self.thread_tree.selection_set(self.thread["id"])
            return
        try:
            self.thread = self.conversations.read(selected[0])
            self._display_thread(restore_inputs=True)
        except (ValueError, OSError) as exc:
            self.status.set(f"会話を読み込めませんでした: {exc}")

    def delete_thread(self):
        if self.service.busy == "answer":
            self.status.set("回答生成が終わるか、生成を停止してから会話を削除してください。")
            return
        if not self.thread:
            self.status.set("削除する会話を一覧から選んでください。")
            return
        title = self.thread["title"]
        message = (f"「{title}」を削除しますか？\n\n"
                   "この会話の履歴と未送信の入力を削除します。元に戻せません。\n"
                   "登録済みの教師データと学習済みモデルは残ります。")
        if not messagebox.askyesno("会話を削除", message, parent=self.root, default="no"):
            return
        try:
            self.conversations.delete(self.thread)
            self.thread = None
            self._pending_turn = None
            self._answer_metadata = {}
            threads = self._thread_list()
            self.thread = threads[0] if threads else None
            if self.thread:
                self.thread_tree.selection_set(self.thread["id"])
            self._display_thread(restore_inputs=True)
            self.status.set(f"「{title}」を削除しました。")
        except (ValueError, OSError) as exc:
            self.status.set(f"会話を削除できませんでした: {exc}")

    def _display_thread(self, restore_inputs=False):
        self.thread_title.set(self.thread["title"] if self.thread else "新しい会話")
        self.answer.configure(state="normal")
        self.answer.delete("1.0", "end")
        if not self.thread or not self.thread["turns"]:
            self.answer.insert("end", "最初の質問を送信すると、このスレッドに会話が保存されます。", "muted")
        previous_context = ""
        for turn in self.thread["turns"] if self.thread else []:
            self.answer.insert("end", "あなた\n", "user")
            if turn["context"] and turn["context"] != previous_context:
                self.answer.insert("end", "状況：" + turn["context"] + "\n", "muted")
            previous_context = turn["context"]
            self.answer.insert("end", turn["question"] + "\n\n")
            self.answer.insert("end", "モデル\n", "assistant")
            text = turn["answer"] or ("生成中…" if turn["id"] == self._pending_turn else "回答は未完了です。")
            self.answer.insert("end", text + "\n")
            if turn["status"] == "error":
                self.answer.insert("end", "（生成できませんでした）\n", "muted")
            elif turn["status"] == "stopped":
                self.answer.insert("end", "（生成を停止しました）\n", "muted")
            self.answer.insert("end", "\n")
        self.answer.configure(state="disabled")
        self.answer.see("end")
        if restore_inputs:
            self._write(self.context, self.thread["context"] if self.thread else "")
            self._write(self.question, "")
            self.chat_state.set("")
        self._buttons()

    def _finish_turn(self, text, status):
        if self.thread and self._pending_turn:
            self.thread = self.conversations.finish_answer(self.thread, self._pending_turn, text, status, self._answer_metadata)
            self._pending_turn = None
            self._thread_list()
            self._display_thread()
        else:
            self._write(self.answer, text, readonly=True)

    def teach_reply(self):
        if self.service.busy == "answer" or not self.thread or not self._can_discard():
            return
        replies = [turn for turn in self.thread["turns"] if turn["status"] in ("completed", "stopped") and turn["answer"]]
        if not replies:
            self.status.set("教師データにする回答がありません。")
            return
        turn = replies[-1]
        history_ids = set(turn.get("used_history_ids", []))
        history = [{key: prior[key] for key in ("question", "context", "answer")}
                   for prior in self.thread["turns"] if prior["id"] in history_ids]
        self._set_record({"question": turn["question"], "context": turn["context"],
                          "answer": turn["answer"], "reason": "", "history": history})
        self.tabs.select(self.teacher_tab)
        self.status.set("回答をあなたの正解に修正してから登録してください。参照した会話履歴も学習対象になります。")

    def get_fields(self):
        return {**{key: widget.get("1.0", "end-1c").strip() for key, widget in self.fields.items()},
                "group": self.group.get().strip(), "valid_from": self.valid_from.get().strip(),
                "history": self._teaching_history}

    def _can_discard(self):
        return self.get_fields() == self._original or messagebox.askokcancel(
            "未保存の入力", "保存していない入力があります。入力を破棄して移動しますか？", parent=self.root)

    def _set_record(self, row=None):
        row = row or {}
        self.edit_id = row.get("id")
        for key, widget in self.fields.items():
            self._write(widget, row.get(key, ""))
            widget.edit_reset()
        self.group.set(row.get("group", ""))
        self.valid_from.set(record_date(row) or "")
        self._teaching_history = row.get("history", [])
        self.teaching_history_label.set(f"会話履歴 {len(self._teaching_history)} 往復も学習対象です。" if self._teaching_history else "")
        self._original = self.get_fields()
        self.edit_label.set("登録済みデータを編集" if self.edit_id else "新しい教師データ")
        self.save_label.set("変更を保存" if self.edit_id else "教師データを登録")
        if record_date(row) is None:
            self.status.set("旧環境名では日付を判定できません。適用開始日を設定して保存してください。")

    def _show_rows(self):
        self._refreshing = True
        self.tree.delete(*self.tree.get_children())
        query = self.filter.get().strip().casefold()
        for row in self.rows:
            effective = record_date(row)
            if query and query not in (row["group"] + " " + row["question"] + " " + (effective or "")).casefold():
                continue
            label = row["group"] if effective == "" else f"[{effective or '日付未設定'}] {row['group']}"
            self.tree.insert("", "end", iid=row["id"], values=(label, row["question"].replace("\n", " ")))
        if self.edit_id and self.tree.exists(self.edit_id):
            self.tree.selection_set(self.edit_id)
        self._refreshing = False

    def _update_summary(self):
        model_state = "モデルあり" if self.service.checkpoint_path.is_file() else "モデル未学習"
        usable = len(select_records(self.rows, self.service.as_of_date))
        self.summary.set(f"今日有効 {usable} / 全 {len(self.rows)} 件  ｜  {model_state}")

    def refresh(self):
        try:
            self.rows = self.store.read()
            self.group_combo.configure(values=sorted({row["group"] for row in self.rows}))
            self._show_rows()
            self._update_summary()
        except (ValueError, OSError, UnicodeError) as exc:
            self.status.set(f"読み込めませんでした: {exc}")

    def reload(self):
        if not self._can_discard():
            return
        self.refresh()
        row = next((row for row in self.rows if row["id"] == self.edit_id), None)
        self._set_record(row)

    def new_record(self):
        if self._can_discard():
            self._set_record()
            self.tree.selection_remove(*self.tree.selection())
            self.fields["question"].focus_set()

    def select_record(self, _event=None):
        if self._refreshing:
            return
        selection = self.tree.selection()
        if not selection or selection[0] == self.edit_id:
            return
        if not self._can_discard():
            self.tree.selection_remove(*selection)
            if self.edit_id and self.tree.exists(self.edit_id):
                self.tree.selection_set(self.edit_id)
            return
        row = next((row for row in self.rows if row["id"] == selection[0]), None)
        if row:
            self._set_record(row)

    def save_record(self):
        if self.service.busy == "training":
            self.status.set("学習が終わってから教師データを保存してください。")
            return
        fields = self.get_fields()
        if not fields["question"] or not fields["answer"]:
            self.status.set("質問と正解回答を入力してください。")
            return
        try:
            saved = self.store.save(fields, self.edit_id)
            self._set_record(saved)
            self.refresh()
            self.status.set("教師データを保存しました。「学習」画面で学習するとモデルに反映されます。")
        except (ValueError, OSError, UnicodeError) as exc:
            self.status.set(f"保存しませんでした: {exc}")

    def show_example(self):
        messagebox.showinfo("教師データの入力例", "\n\n".join(
            f"【{FIELD_LABELS[key]}】\n{value}" for key, value in EXAMPLE.items()) +
            "\n\nあなたが正しいと考える戦術を入力してください。", parent=self.root)

    def _buttons(self):
        busy = self.service.busy
        self.model_combo.configure(state="disabled" if busy else "readonly")
        for button in (self.train_button, self.ask_button):
            button.configure(state="disabled" if busy else "normal")
        self.save_button.configure(state="disabled" if busy == "training" else "normal")
        self.stop_button.configure(state="normal" if busy == "training" else "disabled")
        self.cancel_button.configure(state="normal" if busy == "answer" else "disabled")
        for button in (self.new_thread_button, self.reload_threads_button, self.teach_reply_button):
            button.configure(state="disabled" if busy == "answer" else "normal")
        self.delete_thread_button.configure(state="disabled" if busy == "answer" or not self.thread else "normal")

    def start_training(self):
        if self.get_fields() != self._original:
            self.tabs.select(self.teacher_tab)
            self.status.set("入力中の教師データを先に保存してください。")
            return
        try:
            self.service.start_training()
            self._write(self.log, "学習を開始しています…\n", readonly=True)
            self.progress.configure(value=0)
            self.progress_text.set("モデルとデータを準備中…")
            self.status.set("学習中です。停止ボタンで現在の学習状態を保存して終了できます。")
            self._buttons()
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))

    def ask(self):
        try:
            question = self.question.get("1.0", "end-1c").strip()
            context = self.context.get("1.0", "end-1c").strip()
            if not question:
                raise ValueError("質問を入力してください")
            if self.service.busy:
                raise ValueError("実行中の処理が終わるまでお待ちください")
            if not self.service.checkpoint_path.is_file():
                raise ValueError(f"学習済みモデルがありません（{self.service.checkpoint_path.name}）。別のモデルを選ぶか、「学習」画面から学習を開始してください")
            if not self.thread:
                self.thread = self.conversations.create()
            history = list(self.thread["turns"])
            self.thread, self._pending_turn = self.conversations.add_question(self.thread, question, context, self.service.as_of_date)
            self._answer_metadata = {}
            self.service.ask(context, question, history=history)
            self._write(self.question, "")
            self._thread_list()
            self._display_thread()
            self.chat_state.set("生成中…")
            self.status.set("回答を生成しています。入力した状況と質問に基づいて回答します。")
            self._buttons()
        except (ValueError, OSError) as exc:
            if self._pending_turn:
                try:
                    self._finish_turn(str(exc), "error")
                except (ValueError, OSError):
                    self._pending_turn = None
            self.status.set(str(exc))

    def stop_work(self):
        try:
            self.service.stop()
            self.status.set("処理を停止しています…")
            self.stop_button.configure(state="disabled")
            self.cancel_button.configure(state="disabled")
        except OSError as exc:
            self.status.set(f"停止要求を送れませんでした: {exc}")

    def _poll(self):
        date_today = self.service.as_of_date
        if date_today != self._last_date:
            self._last_date = date_today
            self.today.set(f"今日：{date_today}（自動更新）")
            self._update_summary()
            self.chat_state.set("")
            self.status.set("日付が変わったため、今日有効なデータに切り替えました。")
        while True:
            try:
                kind, value = self.service.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.log.configure(state="normal")
                self.log.insert("end", value + "\n")
                self.log.see("end")
                self.log.configure(state="disabled")
                match = re.match(r"\[進捗\] (\d+)/(\d+)", value)
                if match:
                    done, total = map(int, match.groups())
                    self.progress.configure(maximum=total, value=done)
                    self.progress_text.set(f"学習 {done:,} / {total:,} 回")
            elif kind == "model":
                self.model_info.set(f"{self.service.checkpoint_path.name} ／ 学習 {value:,} 回")
                self.chat_state.set(f"学習済み {value:,} 回")
            elif kind == "answer_meta":
                self._answer_metadata = value
            elif kind == "answer":
                text, ended, stopped = value
                save_error = None
                try:
                    self._finish_turn(text or "（回答を生成できませんでした）", "stopped" if stopped else "completed")
                except (ValueError, OSError) as exc:
                    save_error = str(exc)
                    self._pending_turn = None
                    self._write(self.answer, text + "\n\n会話を保存できませんでした。", readonly=True)
                note = "生成を停止しました。" if stopped else (
                    "回答を生成しました。" if ended else "生成できる長さの上限に達しました。")
                if self._answer_metadata.get("history_omitted"):
                    note += f" 長さ制限により古い会話 {self._answer_metadata['history_omitted']} 往復は参照していません。"
                if self._answer_metadata.get("history_outdated"):
                    note += " 情報更新前の回答や未完了の回答は参照していません。"
                if save_error:
                    note += f" 会話を保存できませんでした: {save_error}"
                self.chat_state.set(note)
                self.status.set(note)
            elif kind == "trained":
                self.model_info.set(f"{self.service.checkpoint_path.name}（未読込）")
                self.progress_text.set("停止しました" if value else "学習完了")
                self.status.set("学習を停止しました。" if value else "学習が完了しました。「質問」画面で質問できます。")
                self._update_summary()
            elif kind == "error":
                self.status.set(value)
                if self.service.busy == "answer":
                    self.chat_state.set("生成できませんでした")
                    try:
                        self._finish_turn(value, "error")
                    except (ValueError, OSError) as exc:
                        self._pending_turn = None
                        self.status.set(f"{value} / 会話の保存に失敗しました: {exc}")
                else:
                    self.progress_text.set("学習を完了できませんでした")
            elif kind == "finished":
                self.service.busy = None
                self._buttons()
        if self._closing and not self.service.busy:
            self.root.destroy()
            return
        self._after_id = self.root.after(EVENT_POLL_MS, self._poll)

    def close(self):
        if not self._can_discard() or not self._can_leave_thread():
            return
        if self.service.busy:
            self._closing = True
            self.stop_work()
            self.status.set("実行中の処理を停止してから画面を閉じます…")
        else:
            if self._after_id:
                self.root.after_cancel(self._after_id)
            self.root.destroy()


def main():
    root = tk.Tk()
    TacticsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
