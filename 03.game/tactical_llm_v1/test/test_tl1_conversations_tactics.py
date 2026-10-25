"""Multi-turn conditioning, persistent threads, and stale-history prevention."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))

import torch

from tl1_conversations_tactics import ConversationStore, history_for_knowledge
from tl1_data_tactics import load_records
from tl1_dates_tactics import KNOWLEDGE_POLICY_VERSION, checkpoint_issue, data_manifest
from tl1_model_tactics import (ModelConfig, TacticalLM, batch_examples, example_tokens,
                              fit_history, generate, loss_sum, prompt_tokens)
from tl1_ui_service_tactics import TeacherStore


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.base = Path(self.directory.name)
        self.store = ConversationStore(self.base / "conversations")

    def tearDown(self):
        self.directory.cleanup()

    def test_multiple_turns_survive_restart_with_context_and_reason(self):
        thread = self.store.create()
        thread, first = self.store.add_question(thread, "どう守る？", "Aサイト", "2026-10-12")
        thread = self.store.finish_answer(thread, first, "援護する。\n判断理由：単独交戦を避ける。", "completed",
                                          {"knowledge_signature": "current"})
        thread, second = self.store.add_question(thread, "その理由は？", thread["context"], "2026-10-12")
        thread = self.store.finish_answer(thread, second, "人数有利を維持するため。", "completed",
                                          {"knowledge_signature": "current"})
        reopened = ConversationStore(self.base / "conversations").read(thread["id"])
        self.assertEqual(len(reopened["turns"]), 2)
        self.assertEqual(reopened["context"], "Aサイト")
        self.assertIn("判断理由", reopened["turns"][0]["answer"])
        self.assertEqual(reopened["title"], "どう守る？")
        self.assertFalse((self.base / "teacher.jsonl").exists())

    def test_threads_are_isolated_and_concurrent_edits_do_not_overwrite_answers(self):
        a, b = self.store.create(), self.store.create()
        old_a = dict(a)
        a, turn = self.store.add_question(a, "Aの質問", "Aの状況", "2026-10-12")
        with self.assertRaisesRegex(ValueError, "変更されました"):
            self.store.add_question(old_a, "古い画面の質問", "", "2026-10-12")
        self.assertEqual(self.store.read(a["id"])["turns"][0]["id"], turn)
        self.assertEqual(self.store.read(b["id"])["turns"], [])

    def test_invalid_thread_id_cannot_address_other_files(self):
        for identifier in ("../teacher", "../../outside.json", "", None):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.store.read(identifier)

    def test_old_information_and_incomplete_answers_are_not_used_as_history(self):
        turns = [{"status": "completed", "answer": "旧順位", "knowledge_signature": "old"},
                 {"status": "completed", "answer": "現順位", "knowledge_signature": "current"},
                 {"status": "stopped", "answer": "途中", "knowledge_signature": "current"},
                 {"status": "pending", "answer": "", "knowledge_signature": "current"}]
        self.assertEqual(history_for_knowledge(turns, "current"), [turns[1]])

    def test_pending_question_is_preserved_after_interruption(self):
        thread, turn = self.store.add_question(self.store.create(), "未完了の質問", "状況", "2026-10-12")
        reopened = self.store.read(thread["id"])
        self.assertEqual(reopened["turns"][0]["id"], turn)
        self.assertEqual(reopened["turns"][0]["status"], "pending")
        self.assertEqual(history_for_knowledge(reopened["turns"], "current"), [])

    def test_delete_thread_keeps_other_threads_teacher_data_and_model(self):
        thread, turn = self.store.add_question(self.store.create(), "質問", "状況", "2026-10-12")
        thread = self.store.finish_answer(thread, turn, "正解回答", "completed")
        other = self.store.create()
        teacher = TeacherStore(self.base / "teacher.jsonl")
        row = teacher.save({"question": "追加質問", "context": "", "answer": "正解", "reason": "",
                           "group": "共通", "history": [{key: thread["turns"][0][key]
                                                          for key in ("question", "context", "answer")}]})
        model = self.base / "model.pt"
        model.write_bytes(b"model marker")
        teacher_bytes = teacher.path.read_bytes()
        self.store.delete(thread)
        self.assertEqual([saved["id"] for saved in self.store.list_threads()], [other["id"]])
        self.assertEqual(teacher.path.read_bytes(), teacher_bytes)
        self.assertEqual(model.read_bytes(), b"model marker")
        self.assertEqual(load_records(teacher.path)[0]["history"], row["history"])
        with self.assertRaises(FileNotFoundError):
            self.store.read(thread["id"])

    def test_stale_thread_cannot_delete_newer_conversation(self):
        original = self.store.create()
        updated, turn = self.store.add_question(original, "別の画面で送信した質問", "", "2026-10-12")
        with self.assertRaisesRegex(ValueError, "変更されました"):
            self.store.delete(original)
        self.assertEqual(self.store.read(original["id"])["turns"][0]["id"], turn)

    def test_history_limits_drop_whole_old_turns_and_preserve_current_question(self):
        history = [{"question": str(i), "context": "", "answer": "守る。" * 8} for i in range(3)]
        included, omitted = fit_history("今の状況", "現在の質問", history, 150, 20)
        self.assertGreater(omitted, 0)
        self.assertEqual(included[-1], history[-1])
        self.assertLessEqual(len(prompt_tokens("今の状況", "現在の質問", included)) + 20, 150)
        self.assertEqual(len(history), 3)
        with self.assertRaises(ValueError):
            fit_history("長い状況" * 30, "質問", history, 64, 20)

    def test_teacher_can_learn_same_followup_with_distinct_history(self):
        store = TeacherStore(self.base / "teacher.jsonl")
        fields = {"question": "次は？", "context": "", "answer": "左", "reason": "", "group": "followup",
                  "history": [{"question": "A", "context": "", "answer": "左"}]}
        row = store.save(fields)
        other = store.save({**fields, "answer": "右", "history": [{"question": "B", "context": "", "answer": "右"}]})
        self.assertEqual(len(load_records(store.path)), 2)
        checkpoint = {"knowledge_policy_version": KNOWLEDGE_POLICY_VERSION, "data_manifest": data_manifest([row, other])}
        self.assertIsNotNone(checkpoint_issue(checkpoint, [{**row, "history": other["history"]}, other]))

    def test_real_training_conditions_followup_on_previous_conversation(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        rows = [{"id": "a", "question": "次は？", "context": "", "answer": "左", "reason": "",
                 "history": [{"question": "A", "context": "", "answer": "左"}]},
                {"id": "b", "question": "次は？", "context": "", "answer": "右", "reason": "",
                 "history": [{"question": "B", "context": "", "answer": "右"}]}]
        model = TacticalLM(ModelConfig(128, 32, 1, 4, 0))
        inputs, labels = batch_examples([example_tokens(row, 128) for row in rows], "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        for _ in range(180):
            model.train()
            optimizer.zero_grad()
            loss = loss_sum(model(inputs), labels) / (labels != -100).sum()
            loss.backward()
            optimizer.step()
        for row in rows:
            text, ended = generate(model, "", row["question"], history=row["history"], max_new_tokens=16)
            self.assertEqual(text, row["answer"])
            self.assertTrue(ended)


if __name__ == "__main__":
    unittest.main()
