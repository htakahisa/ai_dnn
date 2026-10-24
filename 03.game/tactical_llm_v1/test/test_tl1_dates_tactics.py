"""Effective dates, replacement, midnight updates, and model history tests."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))

from tl1_data_tactics import load_records, validate_record
from tl1_dates_tactics import (KNOWLEDGE_POLICY_VERSION, checkpoint_issue, current_date,
                              data_manifest, record_date, select_records, validate_date)
from tl1_ui_service_tactics import TeacherStore, WorkService


def fields(answer="Aチーム", effective="2026-10-10"):
    return {"question": "最強のチームは？", "context": "リテイク成功率で比較する。",
            "answer": answer, "reason": "成功率が最大。", "group": "リテイク最強",
            "valid_from": effective}


class DatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.base = Path(self.directory.name)
        self.store = TeacherStore(self.base / "teacher.jsonl")

    def tearDown(self):
        self.directory.cleanup()

    def _args(self, max_length=2048, as_of="2026-10-12"):
        return argparse.Namespace(data=self.store.path, output=self.base / "models", log=self.base / "training.jsonl",
                                  steps=1, batch_size=1, learning_rate=0.003, eval_interval=1,
                                  validation_fraction=0.2, seed=42, max_length=max_length, width=32,
                                  layers=1, heads=4, dropout=0.0, device="cpu", resume=True,
                                  overwrite=False, as_of_date=as_of)

    def test_date_format_and_calendar_are_validated(self):
        self.assertEqual(validate_date("2026-10-12"), "2026-10-12")
        for value in ("2026-02-30", "20261012", "2026-W42-1", "環境v1", "", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_date(value)

    def test_latest_group_data_replaces_old_even_when_question_and_context_change(self):
        old = self.store.save(fields())
        new = self.store.save({**fields("Bチーム", "2026-10-12"), "context": "最新の評価結果で比較。",
                               "question": "今最強なのは？"})
        self.assertEqual(len(load_records(self.store.path)), 2)
        self.assertEqual([row["id"] for row in load_records(self.store.path, "2026-10-11")], [old["id"]])
        self.assertEqual([row["id"] for row in load_records(self.store.path, "2026-10-12")], [new["id"]])

    def test_previous_day_remains_valid_until_update_and_future_information_is_withheld(self):
        old = self.store.save(fields())
        new = self.store.save(fields("Bチーム", "2026-10-15"))
        self.assertEqual([row["id"] for row in load_records(self.store.path, "2026-10-14")], [old["id"]])
        self.assertEqual([row["id"] for row in load_records(self.store.path, "2026-10-15")], [new["id"]])

    def test_common_and_unrelated_group_knowledge_is_not_expired(self):
        self.store.save(fields())
        self.store.save(fields("Bチーム", "2026-10-12"))
        common = self.store.save({**fields(effective=""), "question": "どう援護する？", "group": "援護の基本"})
        other = self.store.save({**fields(effective="2026-10-09"), "question": "攻撃最強は？", "group": "攻撃最強"})
        ids = {row["id"] for row in load_records(self.store.path, "2026-10-20")}
        self.assertIn(common["id"], ids)
        self.assertIn(other["id"], ids)
        self.assertEqual(len(ids), 3)

    def test_all_examples_on_latest_date_are_retained_as_a_group(self):
        self.store.save(fields())
        a = self.store.save(fields("Bチーム", "2026-10-12"))
        b = self.store.save({**fields("Bチーム", "2026-10-12"), "question": "リテイクで一番強いのは？"})
        self.assertEqual({row["id"] for row in load_records(self.store.path, "2026-10-12")}, {a["id"], b["id"]})

    def test_midnight_changes_selection_without_changing_model_path(self):
        self.store.save(fields())
        latest = self.store.save(fields("Bチーム", "2026-10-13"))
        today = ["2026-10-12"]
        service = WorkService(self.store.path, self.base / "models" / "best.pt", date_provider=lambda: today[0])
        path = service.checkpoint_path
        old_rows = load_records(self.store.path, service.as_of_date)
        checkpoint = {"knowledge_policy_version": KNOWLEDGE_POLICY_VERSION, "data_manifest": data_manifest(old_rows)}
        today[0] = "2026-10-13"
        new_rows = load_records(self.store.path, service.as_of_date)
        self.assertEqual([row["id"] for row in new_rows], [latest["id"]])
        self.assertEqual(service.checkpoint_path, path)
        self.assertIsNotNone(checkpoint_issue(checkpoint, new_rows))

    def test_unchanged_selection_on_next_day_can_resume_same_model(self):
        self.store.save(fields())
        rows = load_records(self.store.path, "2026-10-12")
        checkpoint = {"knowledge_policy_version": KNOWLEDGE_POLICY_VERSION, "data_manifest": data_manifest(rows)}
        self.assertIsNone(checkpoint_issue(checkpoint, load_records(self.store.path, "2026-10-13")))

    def test_new_date_knowledge_requires_training_before_answers(self):
        old = self.store.save(fields(effective=""))
        checkpoint = {"knowledge_policy_version": KNOWLEDGE_POLICY_VERSION, "data_manifest": data_manifest([old])}
        new = self.store.save({**fields(), "question": "別の質問", "group": "別グループ"})
        self.assertIsNone(checkpoint_issue(checkpoint, [old, new]))
        self.assertIn("まだ学習", checkpoint_issue(checkpoint, [old, new], require_current_dates=True))

    def test_legacy_environment_label_is_preserved_but_not_used_without_a_date(self):
        common = self.store.save(fields(effective=""))
        legacy = {**common, "id": "legacy", "question": "旧情報", "scope": "environment", "environment_id": "初期環境"}
        legacy.pop("valid_from")
        validate_record(legacy)
        self.assertIsNone(record_date(legacy))
        self.assertEqual(select_records([common, legacy], "2026-10-12"), [common])

    def test_common_only_v1_checkpoint_is_adopted_without_retraining(self):
        row = self.store.save(fields(effective=""))
        old_fields = {key: row[key] for key in ("context", "question", "answer", "reason", "group")}
        old_fields.update(scope="common", environment_id="")
        old_hash = hashlib.sha256(json.dumps(old_fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        checkpoint = {"knowledge_policy_version": 1, "data_manifest": {row["id"]: old_hash}}
        self.assertIsNone(checkpoint_issue(checkpoint, [row]))
        self.assertEqual(checkpoint["data_manifest"], data_manifest([row]))
        self.assertEqual(checkpoint["knowledge_policy_version"], KNOWLEDGE_POLICY_VERSION)

    def test_changed_teacher_is_rejected_and_retrained_with_old_model_preserved(self):
        from tl1_model_tactics import load_checkpoint
        from tl1_train_tactics import train

        row = self.store.save({**fields(effective=""), "context": "", "question": "Q", "answer": "A", "reason": ""})
        args = self._args(max_length=64)
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
            path = args.output / "latest.pt"
            old_bytes = path.read_bytes()
            changed = self.store.save({**row, "answer": "B"}, row["id"])
            train(args)
        _, checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["step"], 1)
        self.assertEqual(checkpoint["data_manifest"], data_manifest([changed]))
        archived = list((args.output / "archive").glob("*/latest.pt"))
        self.assertEqual(archived[0].read_bytes(), old_bytes)

    def test_daily_training_resumes_until_new_effective_information_replaces_old(self):
        from tl1_model_tactics import load_checkpoint
        from tl1_train_tactics import train

        old = self.store.save({**fields(effective="2026-10-12"), "context": "", "question": "Q", "answer": "A", "reason": ""})
        new = self.store.save({**fields(effective="2026-10-14"), "context": "", "question": "Q", "answer": "B", "reason": ""})
        args = self._args(max_length=64)
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
            args.as_of_date = "2026-10-13"
            train(args)
            _, continued = load_checkpoint(args.output / "latest.pt")
            args.as_of_date = "2026-10-14"
            train(args)
        _, refreshed = load_checkpoint(args.output / "latest.pt")
        self.assertEqual(continued["step"], 2)
        self.assertEqual(set(continued["data_manifest"]), {old["id"]})
        self.assertEqual(refreshed["step"], 1)
        self.assertEqual(set(refreshed["data_manifest"]), {new["id"]})

    def test_long_teacher_trains_and_length_errors_do_not_archive_models(self):
        from tl1_model_tactics import example_tokens, load_checkpoint
        from tl1_train_tactics import train

        row = self.store.save({**fields(effective=""), "context": "", "question": "Q", "answer": "守る。" * 160, "reason": ""})
        self.assertGreater(len(example_tokens(row, 2048)[0]), 1024)
        args = self._args()
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        path = args.output / "latest.pt"
        raw = path.read_bytes()
        _, checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["model_config"]["max_length"], 2048)
        args.max_length = 64
        with self.assertRaisesRegex(ValueError, "MAX_SEQUENCE_LENGTH"):
            train(args)
        self.assertEqual(path.read_bytes(), raw)
        self.assertFalse((args.output / "archive").exists())

    def test_sequence_length_change_archives_old_model(self):
        from tl1_model_tactics import load_checkpoint
        from tl1_train_tactics import train

        self.store.save({**fields(effective=""), "context": "", "question": "Q", "answer": "守る。", "reason": ""})
        args = self._args(max_length=64)
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
            previous = (args.output / "latest.pt").read_bytes()
            args.max_length = 128
            train(args)
        _, checkpoint = load_checkpoint(args.output / "latest.pt")
        self.assertEqual(checkpoint["model_config"]["max_length"], 128)
        archived = list((args.output / "archive").glob("*/latest.pt"))
        self.assertEqual(archived[0].read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
