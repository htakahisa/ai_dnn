"""Desktop data storage and background model/training work (no Tk calls)."""

from pathlib import Path

AI_DIR = Path(__file__).resolve().parent
DATA_PATH = AI_DIR / "data" / "teacher.jsonl"
CHECKPOINT_PATH = AI_DIR / "data" / "checkpoints" / "best.pt"
CHECKPOINT_NAMES = ("best.pt", "latest.pt")  # 画面で選択できるモデル。初期選択は CHECKPOINT_PATH。
TRAINING_SCRIPT = AI_DIR / "tl1_train_tactics.py"
TRAINING_LOG_PATH = AI_DIR / "logs" / "training_ui.log"
CPU_THREADS = 4
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
DEVICE = "auto"

from datetime import datetime, timezone
import hashlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from uuid import uuid4

from tl1_data_tactics import SCHEMA_VERSION, default_group, validate_record
from tl1_dates_tactics import checkpoint_issue, current_date, knowledge_signature, record_date, validate_date
from tl1_conversations_tactics import history_for_knowledge


class TeacherStore:
    def __init__(self, path=DATA_PATH):
        self.path = Path(path)
        self.version = None
        self.loaded = False

    def _bytes(self):
        return self.path.read_bytes() if self.path.exists() else b""

    def read(self):
        raw = self._bytes()
        rows, ids = [], set()
        for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = validate_record(json.loads(line))
                if row["id"] in ids:
                    raise ValueError("id が重複しています")
                rows.append(row)
                ids.add(row["id"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"教師データ {number} 行目: {exc}") from exc
        self.version = hashlib.sha256(raw).hexdigest()
        self.loaded = True
        return rows

    def save(self, fields, identifier=None):
        if not self.loaded:
            self.read()
        raw = self._bytes()
        if hashlib.sha256(raw).hexdigest() != self.version:
            raise ValueError("他の画面やスクリプトでデータが変更されました。「再読込」してから保存してください")
        rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
        existing = next((row for row in rows if row["id"] == identifier), None)
        if identifier and existing is None:
            raise ValueError("編集対象が見つかりません。「再読込」してください")
        now = datetime.now(timezone.utc).isoformat()
        row = dict(existing or {"schema_version": SCHEMA_VERSION, "id": uuid4().hex,
                                "created_at": now, "source": "user"})
        for key in ("context", "question", "answer", "reason", "group"):
            row[key] = fields.get(key, "").strip()
        row["group"] = row["group"] or default_group(row["context"], row["question"])
        row["valid_from"] = fields.get("valid_from", record_date(row) or "").strip()
        if row["valid_from"]:
            row["valid_from"] = validate_date(row["valid_from"])
        row["scope"] = "dated" if row["valid_from"] else "common"
        row.pop("environment_id", None)
        if "history" in fields:
            row["history"] = fields["history"]
        if existing:
            row["updated_at"] = now
        validate_record(row)
        for other in rows:
            if other["id"] != row["id"] and (
                other["context"].strip(), other["question"].strip()
            ) == (row["context"], row["question"]) and other.get("history", []) == row.get("history", []) and (
                record_date(other),
            ) == (record_date(row),):
                raise ValueError("同じ状況・質問が登録済みです。一覧からそのデータを選んで編集してください")
        if existing:
            rows = [row if other["id"] == identifier else other for other in rows]
        else:
            rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=self.path.parent, prefix=".teacher-",
                                             suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                for saved in rows:
                    file.write(json.dumps(saved, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            if hashlib.sha256(self._bytes()).hexdigest() != self.version:
                raise ValueError("保存中にデータが変更されました。「再読込」してやり直してください")
            temporary.replace(self.path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        self.read()
        return row


class WorkService:
    """Deliver worker results through a queue; only the UI thread touches widgets."""

    def __init__(self, data_path=DATA_PATH, checkpoint_path=CHECKPOINT_PATH,
                 training_script=TRAINING_SCRIPT, log_path=TRAINING_LOG_PATH,
                 date_provider=current_date):
        self.data_path = Path(data_path)
        self.base_checkpoint_path = Path(checkpoint_path)
        self._checkpoint_path = self.base_checkpoint_path
        self.date_provider = date_provider
        self.training_script = Path(training_script)
        self.log_path = Path(log_path)
        self.events = queue.Queue()
        self.busy = None
        self.thread = None
        self.cancel = threading.Event()
        self.stop_path = None
        self._model = None
        self._signature = None
        self._checkpoint = None

    @property
    def as_of_date(self):
        return validate_date(self.date_provider())

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    def select_checkpoint(self, name):
        if self.busy:
            raise ValueError("実行中の処理が終わってからモデルを切り替えてください")
        if name not in CHECKPOINT_NAMES:
            raise ValueError("best.pt または latest.pt を選択してください")
        path = self.base_checkpoint_path.with_name(name)
        if path != self._checkpoint_path:
            self._checkpoint_path = path
            self._model = self._signature = self._checkpoint = None

    def _start(self, kind, work):
        if self.busy:
            raise ValueError("実行中の処理が終わるまでお待ちください")
        self.busy = kind
        self.cancel.clear()

        def run():
            try:
                work()
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("finished", kind))

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def ask(self, context, question, history=None):
        if not question.strip():
            raise ValueError("質問を入力してください")
        if not self.checkpoint_path.is_file():
            raise ValueError(f"学習済みモデルがありません（{self.checkpoint_path.name}）。別のモデルを選ぶか、「学習」画面から学習を開始してください")
        as_of = self.as_of_date

        def work():
            import torch
            from tl1_model_tactics import fit_history, generate, load_checkpoint
            from tl1_train_tactics import resolve_device

            torch.set_num_threads(CPU_THREADS)
            stat = self.checkpoint_path.stat()
            signature = (str(self.checkpoint_path.resolve()), stat.st_mtime_ns, stat.st_size)
            if self._model is None or signature != self._signature:
                self._model, self._checkpoint = load_checkpoint(self.checkpoint_path, resolve_device(DEVICE))
                self._signature = signature
                self.events.put(("model", self._checkpoint["step"]))
            from tl1_data_tactics import load_records
            rows = load_records(self.data_path, as_of)
            issue = checkpoint_issue(self._checkpoint, rows, require_current_dates=True)
            if issue:
                raise ValueError(issue + "。「学習」画面で学習を実行してください")
            signature = knowledge_signature(rows)
            eligible = history_for_knowledge(history or [], signature)
            included, omitted = fit_history(context.strip(), question.strip(), eligible,
                                            self._model.config.max_length, MAX_NEW_TOKENS)
            metadata = {"as_of_date": as_of, "knowledge_signature": signature,
                        "used_history_ids": [turn.get("id", "") for turn in included],
                        "history_omitted": omitted, "history_outdated": len(history or []) - len(eligible)}
            self.events.put(("answer_meta", metadata))
            text, ended = generate(self._model, context.strip(), question.strip(),
                                   max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                   stop_event=self.cancel, history=included)
            if self.as_of_date != as_of:
                raise ValueError("日付が変わりました。もう一度質問してください")
            current_rows = load_records(self.data_path, as_of)
            issue = checkpoint_issue(self._checkpoint, current_rows, require_current_dates=True)
            if issue:
                raise ValueError(issue)
            if knowledge_signature(current_rows) != signature:
                raise ValueError("生成中に教師データが更新されました。もう一度質問してください")
            self.events.put(("answer", (text, ended, self.cancel.is_set())))

        self._start("answer", work)

    def start_training(self):
        from tl1_data_tactics import load_records

        as_of = self.as_of_date
        load_records(self.data_path, as_of)  # Use the newest effective date per group.
        if not self.training_script.is_file():
            raise ValueError("学習スクリプトが見つかりません")
        if self.busy:
            raise ValueError("実行中の処理が終わるまでお待ちください")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_path = self.log_path.parent / f"ui_stop_{uuid4().hex}.request"

        def work():
            # Drop the old model before training, freeing its GPU memory.
            self._model = self._signature = self._checkpoint = None
            if "torch" in sys.modules:
                sys.modules["torch"].cuda.empty_cache()
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"
            executable = Path(sys.executable)
            if executable.name.lower() == "pythonw.exe":
                executable = executable.with_name("python.exe")
            command = [str(executable), "-u", str(self.training_script),
                       "--data", str(self.data_path), "--output", str(self.base_checkpoint_path.parent),
                       "--as-of-date", as_of,
                       "--stop-file", str(self.stop_path)]
            try:
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n--- {datetime.now(timezone.utc).isoformat()} ---\n")
                    with subprocess.Popen(
                        command, cwd=self.training_script.parent, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    ) as process:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            self.events.put(("log", line.rstrip()))
                        code = process.wait()
                if code:
                    raise ValueError(f"学習を完了できませんでした（終了コード {code}）。下の学習ログをご確認ください")
                self.events.put(("trained", self.cancel.is_set()))
            finally:
                if self.stop_path.exists():
                    self.stop_path.unlink()

        self._start("training", work)

    def stop(self):
        self.cancel.set()
        if self.busy == "training" and self.stop_path:
            self.stop_path.write_text("stop\n", encoding="utf-8")
