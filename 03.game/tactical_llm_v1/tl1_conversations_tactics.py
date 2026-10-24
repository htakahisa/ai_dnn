"""Persistent chat threads, independent of teacher data and model weights."""

from pathlib import Path

CONVERSATION_DIR = Path(__file__).resolve().parent / "data" / "conversations"
THREAD_SCHEMA_VERSION = 1
DEFAULT_THREAD_TITLE = "新しい会話"
TITLE_MAX_LENGTH = 40

from datetime import datetime, timezone
import json
import os
import tempfile
from uuid import UUID, uuid4


def validate_history(history):
    if not isinstance(history, list):
        raise ValueError("会話履歴はリストにしてください")
    for turn in history:
        if not isinstance(turn, dict):
            raise ValueError("会話履歴の形式が不正です")
        for key in ("question", "context", "answer"):
            if not isinstance(turn.get(key), str) or key != "context" and not turn[key].strip():
                raise ValueError(f"会話履歴の {key} を確認してください")
    return history


def history_for_knowledge(turns, signature):
    return [turn for turn in turns if turn.get("status") == "completed" and turn.get("answer")
            and turn.get("knowledge_signature") == signature]


class ConversationStore:
    def __init__(self, directory=CONVERSATION_DIR):
        self.directory = Path(directory)

    def _path(self, identifier):
        try:
            identifier = UUID(identifier).hex
        except (ValueError, TypeError, AttributeError):
            raise ValueError("会話IDが不正です") from None
        return self.directory / f"{identifier}.json"

    @staticmethod
    def _validate(thread):
        if not isinstance(thread, dict) or thread.get("schema_version") != THREAD_SCHEMA_VERSION:
            raise ValueError("会話ファイルの形式が不正です")
        if not isinstance(thread.get("revision"), int) or not isinstance(thread.get("turns"), list):
            raise ValueError("会話ファイルの履歴が不正です")
        if not isinstance(thread.get("title"), str) or not isinstance(thread.get("context"), str):
            raise ValueError("会話ファイルのタイトル・状況が不正です")
        try:
            UUID(thread["id"])
            datetime.fromisoformat(thread["created_at"])
            datetime.fromisoformat(thread["updated_at"])
        except (KeyError, ValueError, TypeError, AttributeError):
            raise ValueError("会話ファイルのID・日時が不正です") from None
        ids = set()
        for turn in thread["turns"]:
            if not isinstance(turn, dict) or turn.get("status") not in ("pending", "completed", "stopped", "error"):
                raise ValueError("会話の回答状態が不正です")
            if not isinstance(turn.get("id"), str) or turn["id"] in ids:
                raise ValueError("会話のメッセージIDが不正です")
            ids.add(turn["id"])
            for key in ("question", "context", "answer"):
                if not isinstance(turn.get(key), str):
                    raise ValueError("会話の文章が不正です")
        return thread

    def read(self, identifier):
        thread = self._validate(json.loads(self._path(identifier).read_text(encoding="utf-8")))
        if UUID(thread["id"]).hex != UUID(identifier).hex:
            raise ValueError("会話IDとファイル名が一致しません")
        return thread

    def list_threads(self):
        if not self.directory.exists():
            return []
        threads = [self.read(path.stem) for path in self.directory.glob("*.json")]
        return sorted(threads, key=lambda thread: thread["updated_at"], reverse=True)

    def delete(self, thread):
        path = self._path(thread["id"])
        if path.resolve().parent != self.directory.resolve():
            raise ValueError("会話の保存先以外は削除できません")
        current = self.read(thread["id"])
        if current["revision"] != thread["revision"]:
            raise ValueError("他の画面で会話が変更されました。再読込してから削除してください")
        path.unlink()

    def create(self):
        now = datetime.now(timezone.utc).isoformat()
        thread = {"schema_version": THREAD_SCHEMA_VERSION, "id": uuid4().hex,
                  "revision": 0, "title": DEFAULT_THREAD_TITLE, "context": "",
                  "created_at": now, "updated_at": now, "turns": []}
        return self.save(thread, creating=True)

    def save(self, thread, *, creating=False):
        self._validate(thread)
        path = self._path(thread["id"])
        old = path.read_bytes() if path.exists() else None
        if creating and old is not None:
            raise ValueError("会話IDが既に存在します")
        if not creating and (old is None or json.loads(old)["revision"] != thread["revision"]):
            raise ValueError("他の画面で会話が変更されました。会話を再読込してください")
        updated = {**thread, "revision": thread["revision"] + 1,
                   "updated_at": datetime.now(timezone.utc).isoformat()}
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                json.dump(updated, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            current = path.read_bytes() if path.exists() else None
            if current != old:
                raise ValueError("保存中に会話が変更されました。再読込してください")
            temporary.replace(path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return updated

    def add_question(self, thread, question, context, as_of_date):
        if not question.strip():
            raise ValueError("質問を入力してください")
        turn = {"id": uuid4().hex, "question": question.strip(), "context": context.strip(),
                "answer": "", "status": "pending", "as_of_date": as_of_date,
                "created_at": datetime.now(timezone.utc).isoformat()}
        updated = {**thread, "context": turn["context"], "turns": [*thread["turns"], turn]}
        if not thread["turns"]:
            updated["title"] = turn["question"].splitlines()[0][:TITLE_MAX_LENGTH]
        return self.save(updated), turn["id"]

    def finish_answer(self, thread, turn_id, text, status, metadata=None):
        if status not in ("completed", "stopped", "error"):
            raise ValueError("回答の状態が不正です")
        if not any(turn["id"] == turn_id for turn in thread["turns"]):
            raise ValueError("回答先のメッセージが見つかりません")
        turns = [{**turn, **(metadata or {}), "answer": text, "status": status,
                  "answered_at": datetime.now(timezone.utc).isoformat()} if turn["id"] == turn_id else turn
                 for turn in thread["turns"]]
        return self.save({**thread, "turns": turns})
