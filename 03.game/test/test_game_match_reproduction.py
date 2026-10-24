"""Repeat actual matches, including private RNGs, GUI ticks and series state."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_competition_manager as manager
from party_presets import get_preset
from run_game import VisualFPSBattle
from simulation_runtime import deterministic_match, resolve_random_seed, MatchRandomState

TEST_SEED = 20261014
TEST_TEAMS = ("Fnatic2023", "EG2023")
RUNTIME_MATCHES = (
    ("Ghost Champions", "ghost_champions_v1", "Gorigons", "concon_v1"),
    ("Furina Classic", "frc_v1", "Fnatic2023", "fnatic_2"),
    ("Fnatic2023", "toru_ai_v4", "EG2023", "fnatic_v3"),
    ("Ghost Champions", "ghost_champions_v3", "Omoko Gaming", "omoko_gaming_v1"),
)
PROJECT_DIR = Path(__file__).resolve().parents[1]


class MatchReproductionTests(unittest.TestCase):
    def play(self, seed=TEST_SEED, **kwargs):
        return manager.play_map(*(get_preset(name) for name in TEST_TEAMS),
                                1, seed, False, "fnatic_v3", "default", **kwargs)

    def test_full_match_repeats_after_unrelated_randomness(self):
        first = self.play()
        random.seed(999)
        random.random()
        np.random.rand(100)
        torch.rand(100)
        second = self.play()
        self.assertTrue(first.replay_frames)
        self.assertEqual(manager.match_result_digest(first), manager.match_result_digest(second))
        different = self.play(seed=TEST_SEED + 1)
        self.assertNotEqual(first.replay_frames, different.replay_frames)

    def test_cpu_scope_and_private_seeds_restore_after_exception(self):
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        expected = random.getstate(), np.random.get_state(), torch.get_rng_state()
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with deterministic_match(TEST_SEED):
                seeds = [resolve_random_seed(namespace="test") for _ in range(2)]
                self.assertNotEqual(*seeds)
                self.assertEqual(resolve_random_seed(77, namespace="test"), 77)
                self.assertEqual(torch.get_num_threads(), 1)
                raise RuntimeError("test failure")
        self.assertEqual(random.getstate(), expected[0])
        np.testing.assert_equal(np.random.get_state(), expected[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), expected[2]))
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        with deterministic_match(TEST_SEED):
            self.assertEqual(seeds, [resolve_random_seed(namespace="test") for _ in range(2)])
        self.assertIsNone(resolve_random_seed(namespace="test"))

    def test_private_controller_rngs_are_seeded(self):
        from concon_v1.co1_attacker_controller import ConconRoundAttackerController
        from ghost_champions_v1_macro import LearningDefenderOpeningMacroGCRuntime
        for factory in (lambda: ConconRoundAttackerController(map_names=("A1",), selection_path=None),
                        LearningDefenderOpeningMacroGCRuntime):
            samples = []
            for _ in range(2):
                with deterministic_match(TEST_SEED), redirect_stdout(io.StringIO()):
                    controller = factory()
                    samples.append([controller.rng.random() for _ in range(8)])
            self.assertEqual(*samples)

    def test_private_rng_creation_between_callbacks_cannot_shift_match_seeds(self):
        with deterministic_match(TEST_SEED):
            state = MatchRandomState()
            with state.activate():
                first = resolve_random_seed(namespace="lazy_controller")
            for _ in range(7):
                resolve_random_seed(namespace="lazy_controller")
            with state.activate():
                second = resolve_random_seed(namespace="lazy_controller")
        with deterministic_match(TEST_SEED):
            self.assertEqual([first, second], [resolve_random_seed(namespace="lazy_controller") for _ in range(2)])

    def test_runtime_matches_repeat_with_private_and_learned_controllers(self):
        with patch("battle_logic.WINNING_ROUNDS", 1), patch(
                "ghost_champions_v3.preplant.gc3_learn_attacker_preplant.SAVE_MATCH_HISTORY", False):
            for team1, ai1, team2, ai2 in RUNTIME_MATCHES:
                for map_number in (1, 2):
                    with self.subTest(ai1=ai1, ai2=ai2, map_number=map_number):
                        arguments = (get_preset(team1), get_preset(team2), map_number,
                                     TEST_SEED, False, ai1, ai2)
                        first = manager.play_map(*arguments)
                        second = manager.play_map(*arguments)
                        self.assertEqual(manager.match_result_digest(first), manager.match_result_digest(second))

    def test_live_gc3_journals_preserve_reproduction_and_keep_separate_files(self):
        with tempfile.TemporaryDirectory() as directory, patch("battle_logic.WINNING_ROUNDS", 1), patch(
                "ghost_champions_v3.preplant.gc3_learn_attacker_preplant.SAVE_MATCH_HISTORY", True), patch(
                "ghost_champions_v3.preplant.gc3_learn_attacker_preplant.HISTORY_LOG_ROOT", Path(directory)):
            first = manager.play_map(get_preset("Ghost Champions"), get_preset("Omoko Gaming"),
                                     1, TEST_SEED, False, "ghost_champions_v3", "omoko_gaming_v1")
            saved = json.loads(json.dumps(asdict(first)))
            second = manager.reproduce_map_result(saved)
            self.assertGreaterEqual(len(list(Path(directory).glob("*.jsonl.gz"))), 2)
            self.assertEqual(manager.match_result_digest(first), manager.match_result_digest(second))

    def test_saved_second_map_restores_fatigue_and_history_after_json(self):
        fatigue = {"Leo": .35, "Demon1": .21}
        history = {"teams": {"Fnatic2023": {"attack_rounds": [
            {"map": 1, "round": 1, "contacts": {"Demon1": {"pos": (12, 3), "region": "A"}}}
        ], "defence_rounds": [{"map": 1, "round": 1, "site": "B"}]}}}
        initial_history = json.dumps(history)
        with patch("battle_logic.WINNING_ROUNDS", 2):
            first = manager.play_map(*(get_preset(name) for name in TEST_TEAMS),
                2, TEST_SEED, False, "fnatic_v3", "default", 1, 0, 2, fatigue, history,
                shield_abilities_enabled=False)
            saved = json.loads(json.dumps(asdict(first)))
            second = manager.reproduce_map_result(saved)
        self.assertEqual(json.dumps(first.reproduction["series_tactical_state"]), initial_history)
        self.assertEqual(manager.match_result_digest(first), manager.match_result_digest(second))
        self.assertEqual(manager.compact_map_result(first)["reproduction"], first.reproduction)
        self.assertEqual(manager.compact_map_model(first).reproduction, first.reproduction)
        with self.assertRaisesRegex(ValueError, "開始状態"):
            manager.reproduce_map_result({"seed": TEST_SEED})

    def test_rendered_ticks_ignore_ui_randomness_and_match_headless(self):
        real_tk = tk.Tk
        def hidden_root(*args, **kwargs):
            root = real_tk(*args, **kwargs)
            root.withdraw()
            return root
        def run_rendered(game):
            try:
                while not game.match_over:
                    # Simulate unrelated callbacks between every real GUI tick.
                    random.random()
                    np.random.rand()
                    torch.rand(1)
                    game.step_tick()
                    game.draw()
                    if game.round_over and not game.match_over:
                        game._cancel_match_callback()
                        game.init_round()
            finally:
                game.root.destroy()
        with patch("battle_logic.WINNING_ROUNDS", 2):
            headless = self.play()
            with patch.object(manager.tk, "Tk", side_effect=hidden_root), \
                 patch.object(VisualFPSBattle, "run", run_rendered), redirect_stdout(io.StringIO()):
                rendered = manager.play_map(*(get_preset(name) for name in TEST_TEAMS),
                    1, TEST_SEED, True, "fnatic_v3", "default")
        self.assertEqual(manager.match_result_digest(headless), manager.match_result_digest(rendered))

    def test_matches_across_fresh_processes_and_python_hash_seeds(self):
        script = '''
import run_competition_manager as m
from party_presets import get_preset
from unittest.mock import patch
with patch("battle_logic.WINNING_ROUNDS", 2):
    r = m.play_map(get_preset("Fnatic2023"), get_preset("EG2023"),
                   1, 20261014, False, "fnatic_v3", "default")
import json
print(json.dumps(m.asdict(r), ensure_ascii=True))
'''
        hashes = []
        for hash_seed in ("1", "897"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            hashes.append(json.loads(result.stdout.decode().strip().splitlines()[-1]))
        def difference(left, right, path=""):
            if isinstance(left, dict) and isinstance(right, dict):
                for key in left:
                    if key not in right:
                        return path + "." + key + " missing"
                    found = difference(left[key], right[key], path + "." + key)
                    if found:
                        return found
            elif isinstance(left, list) and isinstance(right, list):
                if len(left) != len(right):
                    return f"{path}: lengths {len(left)} != {len(right)}"
                for index, (a, b) in enumerate(zip(left, right)):
                    found = difference(a, b, f"{path}[{index}]")
                    if found:
                        return found
            elif left != right:
                return f"{path}: {left!r} != {right!r}"
            return None
        self.assertIsNone(difference(*hashes))

    def test_real_playback_pause_speed_and_skip_preserve_saved_match(self):
        real_tk = tk.Tk
        original_run = VisualFPSBattle.run
        def hidden_root(*args, **kwargs):
            root = real_tk(*args, **kwargs)
            root.withdraw()
            return root
        def run_playback(game):
            game.root.after(0, game.toggle_pause)
            game.root.after(1, game.toggle_pause)
            game.root.after(2, lambda: game.set_tick_time_ms(1))
            game.root.after(3, lambda: game.start_fast_forward("map"))
            try:
                original_run(game)
            finally:
                game.root.destroy()
        with patch("battle_logic.WINNING_ROUNDS", 2):
            first = self.play()
            saved = json.loads(json.dumps(asdict(first)))
            with patch.object(manager.tk, "Tk", side_effect=hidden_root), \
                 patch.object(VisualFPSBattle, "run", run_playback), redirect_stdout(io.StringIO()):
                second = manager.reproduce_map_result(saved, render=True, tick_time_ms=1)
        self.assertEqual(manager.match_result_digest(first), manager.match_result_digest(second))


if __name__ == "__main__":
    unittest.main()
