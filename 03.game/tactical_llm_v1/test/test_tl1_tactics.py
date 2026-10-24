"""Learning, leakage prevention, and checkpoint lifecycle tests."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR))

import torch

from tl1_data_tactics import default_group, load_records, scenario_assignments, split_records
from tl1_model_tactics import (ANSWER, BYTE_OFFSET, EOS, ModelConfig, TacticalLM,
                              batch_examples, encode, example_tokens, generate,
                              load_checkpoint, loss_sum, prompt_tokens)
from tl1_train_tactics import evaluate_loss, train


def record(identifier, context="", question="どうする？", answer="守る。", group=None):
    return {"schema_version": 1, "id": identifier, "context": context,
            "question": question, "answer": answer, "reason": "",
            "group": group or default_group(context, question)}


class TacticalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_utf8_roundtrip_including_unseen_characters(self):
        text = "Aサイトを守る。🛡️\n未知の漢字：鬱"
        self.assertEqual(bytes(value - BYTE_OFFSET for value in encode(text)).decode(), text)

    def test_future_answer_cannot_change_prompt_predictions(self):
        torch.manual_seed(4)
        model = TacticalLM(ModelConfig(64, 32, 1, 4, 0)).eval()
        prompt = prompt_tokens("", "Q")
        first = torch.tensor([prompt + encode("attack")])
        second = torch.tensor([prompt + encode("defend")])
        torch.testing.assert_close(model(first)[:, :len(prompt)], model(second)[:, :len(prompt)])

    def test_supervision_includes_first_answer_byte_and_excludes_prompt(self):
        row = record("test")
        inputs, labels = example_tokens(row, 128)
        prompt = prompt_tokens(row["context"], row["question"])
        self.assertEqual(inputs[len(prompt) - 1], ANSWER)
        self.assertTrue(all(value == -100 for value in labels[:len(prompt) - 1]))
        self.assertEqual(labels[len(prompt) - 1:], encode(row["answer"]) + [EOS])

    def test_long_example_is_rejected_without_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "MAX_SEQUENCE_LENGTH"):
            example_tokens(record("long", answer="守る" * 20), 32)

    def test_real_gradient_training_learns_japanese_response(self):
        torch.manual_seed(7)
        model = TacticalLM(ModelConfig(64, 32, 1, 4, 0))
        row = record("synthetic_test_only")
        examples = [example_tokens(row, 64)]
        before = evaluate_loss(model, examples, 1, "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs, labels = batch_examples(examples, "cpu")
        for _ in range(100):
            model.train()
            optimizer.zero_grad()
            loss = loss_sum(model(inputs), labels) / (labels != -100).sum()
            loss.backward()
            optimizer.step()
        after = evaluate_loss(model, examples, 1, "cpu")
        self.assertLess(after, before / 5)
        answer, ended = generate(model, "", row["question"], max_new_tokens=30)
        self.assertEqual(answer, row["answer"])
        self.assertTrue(ended)

    def test_generation_masks_role_tokens_and_preserves_valid_utf8(self):
        expected = encode("守る。") + [EOS]
        prefix_length = len(prompt_tokens("", "Q"))

        class ForcedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.config = ModelConfig(max_length=64)

            def forward(self, inputs):
                scores = torch.full((1, inputs.shape[1], 262), -20.0)
                scores[:, -1, ANSWER] = 100  # Must always be masked.
                scores[:, -1, expected[inputs.shape[1] - prefix_length]] = 50
                return scores

        answer, ended = generate(ForcedModel(), "", "Q", max_new_tokens=20)
        self.assertEqual(answer, "守る。")
        self.assertTrue(ended)


class TeacherDataTests(unittest.TestCase):
    def test_same_scenario_stays_together_even_with_different_group_names(self):
        rows = [record("1", "同じ状況", "質問1", group="a"),
                record("2", "同じ状況", "質問2", group="b"),
                record("3", "別の状況", group="c")]
        training, validation, assignment = split_records(rows, 0.2, 42)
        self.assertEqual(assignment["a"], assignment["b"])
        self.assertTrue(training)
        self.assertTrue(validation)
        self.assertFalse({row["context"] for row in training} &
                         {row["context"] for row in validation})

    def test_resume_keeps_existing_holdout_when_new_groups_arrive(self):
        rows = [record(str(i), f"状況{i}", group=f"g{i}") for i in range(5)]
        _, _, original = split_records(rows, 0.2, 42)
        _, _, continued = split_records(rows + [record("new", "新状況", group="new")],
                                        0.2, 42, original)
        self.assertTrue(all(continued[key] == value for key, value in original.items()))

    def test_resume_rejects_new_link_between_training_and_validation(self):
        rows = [record("1", "同じ", "質問1", group="a"),
                record("2", "同じ", "質問2", group="b")]
        with self.assertRaisesRegex(ValueError, "結合"):
            split_records(rows, 0.2, 42, {"a": "train", "b": "validation"})

    def test_renaming_removed_scenario_cannot_move_training_history_into_holdout(self):
        old_rows = [record("old", "学習済み", group="old")]
        scenarios = scenario_assignments(old_rows, {"old": "train"})
        new_rows = [record("new", "学習済み", group="renamed"),
                    record("other", "別", group="other")]
        _, _, assignment = split_records(new_rows, 0.2, 42,
                                         {"old": "train", "other": "validation"}, scenarios)
        self.assertEqual(assignment["renamed"], "train")

    def test_heldout_generation_report_uses_only_withheld_scenarios(self):
        import tl1_eval_tactics

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            data = base / "teacher.jsonl"
            rows = [record(str(i), f"状況{i}", group=f"g{i}") for i in range(4)]
            data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                            encoding="utf-8")
            args = argparse.Namespace(data=data, output=base / "models", log=base / "training.jsonl",
                                      steps=2, batch_size=1, learning_rate=0.003, eval_interval=2,
                                      validation_fraction=0.25, seed=42, max_length=64, width=32,
                                      layers=1, heads=4, dropout=0.0, device="cpu", resume=True,
                                      overwrite=False)
            report = base / "evaluation.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                train(args)
                argv = ["tl1_eval_tactics.py", "--data", str(data), "--checkpoint",
                        str(args.output / "best.pt"), "--report", str(report), "--device", "cpu"]
                with patch.object(sys, "argv", argv), patch.object(tl1_eval_tactics, "MAX_NEW_TOKENS", 8):
                    tl1_eval_tactics.main()
            lines = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(lines[0]["validation_count"], 1)
            self.assertIsNotNone(lines[0]["validation_loss"])
            self.assertEqual(len(lines), 2)
            _, checkpoint = load_checkpoint(args.output / "best.pt")
            expected = {row["id"] for row in rows
                        if checkpoint["group_assignment"][row["group"]] == "validation"}
            self.assertEqual({line["id"] for line in lines[1:]}, expected)

    def test_single_scenario_does_not_pretend_to_have_validation(self):
        training, validation, _ = split_records([record("one")], 0.2, 42)
        self.assertEqual(len(training), 1)
        self.assertEqual(validation, [])

    def test_conflicting_teacher_answers_fail_with_line_number(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "teacher.jsonl"
            rows = [record("a"), record("b", answer="攻める。")]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2:.*異なる正解"):
                load_records(path)

    def test_training_resume_uses_own_checkpoint_and_accumulates_steps(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            data = base / "teacher.jsonl"
            data.write_text(json.dumps(record("synthetic_test_only"), ensure_ascii=False) + "\n",
                            encoding="utf-8")
            args = argparse.Namespace(data=data, output=base / "models", log=base / "training.jsonl",
                                      steps=2, batch_size=1, learning_rate=0.003, eval_interval=2,
                                      validation_fraction=0.2, seed=42, max_length=64, width=32,
                                      layers=1, heads=4, dropout=0.0, device="cpu", resume=True,
                                      overwrite=False)
            with contextlib.redirect_stdout(io.StringIO()):
                path = train(args)
                first_model, first = load_checkpoint(path)
                train(args)
                second_model, second = load_checkpoint(path)
            self.assertEqual(first["step"], 2)
            self.assertEqual(second["step"], 4)
            self.assertIsNone(second["metrics"]["validation_loss"])
            self.assertEqual(first["group_assignment"], second["group_assignment"])
            self.assertFalse(torch.equal(first_model.token.weight, second_model.token.weight))
            self.assertEqual(len(args.log.read_text(encoding="utf-8").splitlines()), 2)
            args.resume = False
            with self.assertRaisesRegex(ValueError, "既存モデル"):
                train(args)


if __name__ == "__main__":
    unittest.main()
