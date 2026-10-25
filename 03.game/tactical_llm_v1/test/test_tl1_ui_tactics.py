"""Desktop workflows, data preservation, and background training tests."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))

from tl1_data_tactics import load_records
from tl1_ui_service_tactics import TeacherStore, WorkService
from tl1_ui_tactics import TacticsApp

FIELDS = {"question": "設置後はどう守る？", "context": "Aサイト。味方3人、敵1人。",
          "answer": "味方と援護できる位置で守る。", "reason": "単独交戦を避けるため。",
          "group": "Aサイトポストプラント"}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.path = Path(self.directory.name) / "teacher.jsonl"
        self.store = TeacherStore(self.path)
        self.store.read()

    def tearDown(self):
        self.directory.cleanup()

    def test_japanese_group_roundtrips_and_edit_preserves_identity(self):
        first = self.store.save(FIELDS)
        second = self.store.save({**FIELDS, "answer": "解除を確認できる位置で守る。"}, first["id"])
        rows = load_records(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(second["created_at"], first["created_at"])
        self.assertEqual(rows[0]["group"], "Aサイトポストプラント")
        self.assertEqual(rows[0]["reason"], FIELDS["reason"])

    def test_external_change_is_rejected_without_losing_either_record(self):
        first = self.store.save(FIELDS)
        external = TeacherStore(self.path)
        external.read()
        external.save({**FIELDS, "question": "別の質問", "group": "別グループ"})
        raw = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "変更されました"):
            self.store.save({**FIELDS, "answer": "変更"}, first["id"])
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(len(load_records(self.path)), 2)

    def test_duplicate_question_cannot_create_conflicting_answers(self):
        self.store.save(FIELDS)
        raw = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "登録済み"):
            self.store.save({**FIELDS, "answer": "別回答"})
        self.assertEqual(self.path.read_bytes(), raw)

    def test_empty_required_field_does_not_create_data_file(self):
        with self.assertRaises(ValueError):
            self.store.save({**FIELDS, "answer": ""})
        self.assertFalse(self.path.exists())


class DesktopTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        base = Path(self.directory.name)
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.directory.cleanup()
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()
        self.store = TeacherStore(base / "teacher.jsonl")
        self.day = ["2026-10-12"]
        self.service = WorkService(self.store.path, base / "models" / "best.pt",
                                   log_path=base / "ui.log", date_provider=lambda: self.day[0])
        self.app = TacticsApp(self.root, self.store, self.service)

    def tearDown(self):
        if hasattr(self, "app"):
            self.app._write(self.app.question, "")
            self.app._write(self.app.context, self.app.thread["context"] if self.app.thread else "")
            self.app.close()
        self.directory.cleanup()

    def test_form_buttons_save_and_reload_full_teacher_answer(self):
        for key, widget in self.app.fields.items():
            widget.insert("1.0", FIELDS[key])
        self.app.group.set(FIELDS["group"])
        self.app.save_button.invoke()
        self.root.update()
        rows = load_records(self.store.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], FIELDS["reason"])
        self.assertEqual(self.app.tree.item(rows[0]["id"])["values"][0], FIELDS["group"])
        self.assertEqual(self.app.save_label.get(), "変更を保存")
        self.app.new_record()
        self.app.tree.selection_set(rows[0]["id"])
        self.root.update()
        self.assertEqual({key: self.app.get_fields()[key] for key in FIELDS}, FIELDS)

    def test_question_without_model_shows_actionable_message(self):
        self.app.question.insert("1.0", "どう守る？")
        self.app.ask_button.invoke()
        self.assertIn("学習済みモデルがありません", self.app.status.get())
        self.assertIsNone(self.service.busy)
        self.assertFalse(self.store.path.exists())

    def test_model_selection_updates_path_and_blocks_switching_during_work(self):
        self.assertEqual(self.app.model_choice.get(), "best.pt")
        self.app.model_choice.set("latest.pt")
        self.app.model_combo.event_generate("<<ComboboxSelected>>")
        self.root.update()
        self.assertEqual(self.service.checkpoint_path.name, "latest.pt")
        self.assertIn("まだ保存されていません", self.app.status.get())
        self.app.question.insert("1.0", "どう守る？")
        self.app.ask_button.invoke()
        self.assertIn("latest.pt", self.app.status.get())
        for kind in ("answer", "training"):
            self.service.busy = kind
            self.app._buttons()
            self.assertEqual(str(self.app.model_combo["state"]), "disabled")
            self.app.model_choice.set("best.pt")
            self.app.select_model()
            self.assertEqual(self.service.checkpoint_path.name, "latest.pt")
            self.assertEqual(self.app.model_choice.get(), "latest.pt")
        self.service.busy = None
        self.app._buttons()
        self.assertEqual(str(self.app.model_combo["state"]), "readonly")

    def test_model_switch_preserves_conversation_and_unsent_inputs(self):
        self.app.new_thread()
        identifier = self.app.thread["id"]
        self.app._write(self.app.question, "その理由は？")
        self.app._write(self.app.context, "Aサイト")
        self.app.model_choice.set("latest.pt")
        self.app.select_model()
        self.assertEqual(self.app.thread["id"], identifier)
        self.assertEqual(self.app.question.get("1.0", "end-1c"), "その理由は？")
        self.assertEqual(self.app.context.get("1.0", "end-1c"), "Aサイト")

    def test_background_answer_and_reason_are_displayed_together(self):
        self.service.busy = "answer"
        text = "解除を確認できる位置で守る。\n判断理由：解除を妨害するため。"
        self.service.events.put(("answer", (text, True, False)))
        self.service.events.put(("finished", "answer"))
        deadline = time.monotonic() + 2
        while self.service.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertEqual(self.app.answer.get("1.0", "end-1c"), text)
        self.assertIsNone(self.service.busy)
        self.assertFalse(self.store.path.exists())

    def test_form_saves_effective_date_without_environment_name(self):
        for key, widget in self.app.fields.items():
            widget.insert("1.0", FIELDS[key])
        self.app.group.set(FIELDS["group"])
        self.app.valid_from.set("2026-10-12")
        self.app.save_button.invoke()
        row = self.store.read()[0]
        self.assertEqual(row["valid_from"], "2026-10-12")
        self.assertNotIn("environment_id", row)
        self.assertEqual(self.app.get_fields()["valid_from"], "2026-10-12")

    def test_open_window_updates_today_and_effective_record_count_at_midnight(self):
        self.store.save({**FIELDS, "valid_from": "2026-10-13"})
        self.app.refresh()
        self.assertIn("今日有効 0", self.app.summary.get())
        self.day[0] = "2026-10-13"
        deadline = time.monotonic() + 2
        while "2026-10-13" not in self.app.today.get() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertIn("2026-10-13", self.app.today.get())
        self.assertIn("今日有効 1", self.app.summary.get())

    def _send_mock_reply(self, question, reply, captures):
        def ask(context, question, history=None):
            captures.append({"context": context, "question": question, "history": history})
            self.service.busy = "answer"
            self.service.events.put(("answer_meta", {"knowledge_signature": "test-current",
                "used_history_ids": [turn["id"] for turn in history or []], "as_of_date": self.day[0]}))
            self.service.events.put(("answer", (reply, True, False)))
            self.service.events.put(("finished", "answer"))
        self.app._write(self.app.question, question)
        with patch.object(self.service, "ask", side_effect=ask):
            self.app.ask()
        deadline = time.monotonic() + 2
        while self.service.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertIsNone(self.service.busy)

    def test_followup_receives_prior_answer_and_thread_is_persisted(self):
        self.service.checkpoint_path.parent.mkdir(parents=True)
        self.service.checkpoint_path.write_bytes(b"mock checkpoint marker")
        captures = []
        self.app._write(self.app.context, "Aサイト")
        self._send_mock_reply("どう守る？", "味方と援護する。", captures)
        self._send_mock_reply("その理由は？", "単独交戦を避けるため。", captures)
        self.assertEqual(len(captures[1]["history"]), 1)
        self.assertEqual(captures[1]["history"][0]["answer"], "味方と援護する。")
        self.assertEqual(captures[1]["context"], "Aサイト")
        text = self.app.answer.get("1.0", "end-1c")
        self.assertIn("どう守る？", text)
        self.assertIn("その理由は？", text)
        thread = self.app.conversations.read(self.app.thread["id"])
        self.assertEqual(len(thread["turns"]), 2)
        self.assertFalse(self.store.path.exists())
        self.app.thread = None
        self.app.reload_threads(initial=True)
        self.assertEqual(self.app.context.get("1.0", "end-1c"), "Aサイト")
        self.assertIn("その理由は？", self.app.answer.get("1.0", "end-1c"))

    def test_new_thread_does_not_receive_previous_threads_history(self):
        self.service.checkpoint_path.parent.mkdir(parents=True)
        self.service.checkpoint_path.write_bytes(b"mock checkpoint marker")
        captures = []
        self._send_mock_reply("Aの質問", "Aの回答", captures)
        first_id = self.app.thread["id"]
        self.app.new_thread()
        self._send_mock_reply("Bの質問", "Bの回答", captures)
        self.assertNotEqual(self.app.thread["id"], first_id)
        self.assertEqual(captures[-1]["history"], [])
        self.assertEqual(self.app.conversations.read(first_id)["turns"][0]["answer"], "Aの回答")

    def test_corrected_reply_can_be_registered_with_its_conversation_history(self):
        self.service.checkpoint_path.parent.mkdir(parents=True)
        self.service.checkpoint_path.write_bytes(b"mock checkpoint marker")
        captures = []
        self._send_mock_reply("どう守る？", "援護する。", captures)
        self._send_mock_reply("その理由は？", "モデルの仮回答", captures)
        self.app.teach_reply()
        self.assertFalse(self.store.path.exists())
        self.app._write(self.app.fields["answer"], "あなたが確認した正解")
        self.app.save_record()
        row = load_records(self.store.path)[0]
        self.assertEqual(row["answer"], "あなたが確認した正解")
        self.assertEqual(row["history"][0]["answer"], "援護する。")

    def test_answer_save_failure_is_reported_instead_of_success_message(self):
        self.service.checkpoint_path.parent.mkdir(parents=True)
        self.service.checkpoint_path.write_bytes(b"mock checkpoint marker")
        with patch.object(self.app.conversations, "finish_answer", side_effect=OSError("保存先エラー")):
            self._send_mock_reply("質問", "回答", [])
        self.assertIn("保存できませんでした", self.app.status.get())
        self.assertIn("保存先エラー", self.app.status.get())

    def test_delete_selected_thread_requires_confirmation_and_preserves_other_thread(self):
        first = self.app.conversations.create()
        other = self.app.conversations.create()
        self.app.thread = first
        self.app._thread_list()
        self.app._display_thread(restore_inputs=True)
        with patch("tl1_ui_tactics.messagebox.askyesno", return_value=False):
            self.app.delete_thread_button.invoke()
        self.assertEqual(len(self.app.conversations.list_threads()), 2)
        with patch("tl1_ui_tactics.messagebox.askyesno", return_value=True) as confirmation:
            self.app.delete_thread_button.invoke()
        self.assertEqual(confirmation.call_count, 1)
        self.assertEqual(self.app.thread["id"], other["id"])
        self.assertEqual(len(self.app.conversations.list_threads()), 1)
        self.assertFalse(self.store.path.exists())

    def test_delete_last_thread_resets_chat_and_generation_blocks_deletion(self):
        self.app.new_thread()
        identifier = self.app.thread["id"]
        self.service.busy = "answer"
        with patch("tl1_ui_tactics.messagebox.askyesno", return_value=True) as confirmation:
            self.app.delete_thread()
        confirmation.assert_not_called()
        self.assertEqual(self.app.conversations.read(identifier)["id"], identifier)
        self.service.busy = None
        with patch("tl1_ui_tactics.messagebox.askyesno", return_value=True):
            self.app.delete_thread()
        self.assertIsNone(self.app.thread)
        self.assertEqual(self.app.conversations.list_threads(), [])
        self.assertEqual(self.app.context.get("1.0", "end-1c"), "")
        self.assertEqual(str(self.app.delete_thread_button["state"]), "disabled")


class BackgroundTests(unittest.TestCase):
    def test_real_background_training_and_inference_use_temporary_data_only(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            store = TeacherStore(base / "teacher.jsonl")
            store.save({**FIELDS, "context": "", "question": "Q", "answer": "守る。", "reason": ""})
            wrapper = base / "tiny_training.py"
            wrapper.write_text(
                "import sys\nfrom pathlib import Path\nsys.path.insert(0, " + repr(str(AI_DIR)) + ")\n"
                "from tl1_train_tactics import parse_args, train\nargs = parse_args()\n"
                "args.steps = 2\nargs.width = 32\nargs.layers = 1\nargs.heads = 4\n"
                "args.max_length = 64\nargs.dropout = 0.0\nargs.device = 'cpu'\n"
                "args.log = Path(args.output).parent / 'training.jsonl'\ntrain(args)\n",
                encoding="utf-8")
            service = WorkService(store.path, base / "models" / "best.pt", wrapper, base / "ui.log")
            service.select_checkpoint("latest.pt")
            raw = store.path.read_bytes()
            service.start_training()
            service.thread.join(45)
            self.assertFalse(service.thread.is_alive())
            events = []
            while not service.events.empty():
                events.append(service.events.get_nowait())
            self.assertFalse([event for event in events if event[0] == "error"], events)
            self.assertTrue(service.checkpoint_path.is_file())
            self.assertIn(("trained", False), events)
            self.assertTrue(any("[進捗]" in event[1] for event in events if event[0] == "log"))
            self.assertFalse(service.stop_path.exists())
            from tl1_model_tactics import generate, load_checkpoint
            with patch("tl1_model_tactics.load_checkpoint", wraps=load_checkpoint) as load, \
                 patch("tl1_model_tactics.generate", wraps=generate) as generation:
                for name in ("best.pt", "best.pt", "latest.pt", "best.pt"):
                    service.busy = None
                    service.select_checkpoint(name)
                    service.ask("", "Q")
                    service.thread.join(30)
                    self.assertFalse(service.thread.is_alive())
                    events = []
                    while not service.events.empty():
                        events.append(service.events.get_nowait())
                    self.assertFalse([event for event in events if event[0] == "error"], events)
                    self.assertTrue(any(event[0] == "answer" for event in events))
                self.assertEqual([call.args[0].name for call in load.call_args_list],
                                 ["best.pt", "latest.pt", "best.pt"])
                self.assertTrue(all(call.kwargs["max_new_tokens"] == 1024
                                    for call in generation.call_args_list))
            self.assertEqual(store.path.read_bytes(), raw)

    def test_stop_request_saves_completed_step_before_exiting(self):
        import torch
        from tl1_model_tactics import load_checkpoint
        from tl1_train_tactics import train

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            store = TeacherStore(base / "teacher.jsonl")
            store.save({**FIELDS, "context": "", "question": "Q", "answer": "守る。", "reason": ""})
            args = argparse.Namespace(data=store.path, output=base / "models", log=base / "training.jsonl",
                                      steps=10, batch_size=1, learning_rate=0.003, eval_interval=10,
                                      validation_fraction=0.2, seed=42, max_length=64, width=32,
                                      layers=1, heads=4, dropout=0.0, device="cpu", resume=True,
                                      overwrite=False, stop_file=base / "stop.request")
            original = torch.optim.AdamW.step

            def step_then_stop(optimizer, *values, **kwargs):
                result = original(optimizer, *values, **kwargs)
                args.stop_file.write_text("stop", encoding="utf-8")
                return result

            with patch.object(torch.optim.AdamW, "step", step_then_stop), contextlib.redirect_stdout(io.StringIO()):
                train(args)
            _, checkpoint = load_checkpoint(args.output / "latest.pt")
            self.assertEqual(checkpoint["step"], 1)


if __name__ == "__main__":
    unittest.main()
