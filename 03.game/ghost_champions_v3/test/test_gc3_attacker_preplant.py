"""Contracts for unified pre-plant BC, phase handoff and randomized fixed opponents."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.test.test_gc3_scratch import fixture
from ghost_champions_v3.gc3_engine_actions import MODES, decode_command
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_reward_phase import potential_difference
from ghost_champions_v3.preplant.gc3_teacher_attacker_actions import teacher_record, UnrepresentableAction
from ghost_champions_v3.preplant.gc3_learn_attacker_preplant import PreplantController, TwoPhaseAttackerController
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import save_shard, load_shard
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import bc_log_prob, train_rl, evaluate_preplant
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import MAJOR_TEAMS, choose_roster, PreplantRewards, play_preplant


def preplant_fixture():
    char, state = fixture()
    state.update(is_planted=False, planted_pos=None, spike_pos=None)
    return char, state


class PreplantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_roster_cycles_are_random_balanced_and_deterministic(self):
        codes = tuple(MAJOR_TEAMS)
        first = [choose_roster(200, i, codes) for i in range(21)]
        self.assertEqual(first, [choose_roster(200, i, codes) for i in range(21)])
        for start in (0, 7, 14):
            self.assertEqual(set(first[start:start+7]), set(codes))
        self.assertNotEqual(first[:7], first[7:14])

    def test_teacher_moves_and_plants_use_student_action_schema(self):
        char, state = preplant_fixture()
        record = teacher_record(char, state, list(char.pos))
        self.assertEqual(decode_command(char, state, record["selections"]), (list(char.pos), "MOVE"))
        char.has_spike = True
        char.pos = list(np.argwhere(state["grid"] == 2)[0])
        record = teacher_record(char, state, (list(char.pos), "PLANT"))
        self.assertEqual(decode_command(char, state, record["selections"])[1], "PLANT")
        self.assertEqual(record["phase"], "standard")
        self.assertFalse(record["hints"].any())

    def test_facing_bc_teaches_probability_of_entire_direction(self):
        char, state = preplant_fixture()
        record = teacher_record(char, state, (list(char.pos), {"facing":"E"}))
        model = FreshPolicy.fresh("attacker", 20)
        logp = bc_log_prob(model, [record])[0]
        context = model.features(torch.from_numpy(record["obs"]))
        mode = model.distribution(context, record, "mode")
        watch = model.distribution(context, record, "watch")
        expected = mode.log_prob(torch.tensor(MODES.index("WATCH")))+watch.probs[record["watch_options"]].sum().log()
        self.assertAlmostEqual(float(logp.detach()), float(expected.detach()), places=5)
        other = dict(record, selections=dict(record["selections"]))
        other["selections"]["watch"] = int(np.flatnonzero(record["watch_options"])[-1])
        self.assertAlmostEqual(float(bc_log_prob(model, [other])[0].detach()), float(logp.detach()), places=5)
        (-logp).backward()
        self.assertTrue(torch.isfinite(model.spatial.watch.bias.grad).all())
        self.assertIsNone(model.critic.weight.grad)  # BC has no invented critic targets.

    def test_physical_facing_lock_labels_equivalent_hold(self):
        char, state = preplant_fixture()
        char.facing_forced_this_tick = True
        record = teacher_record(char, state, (list(char.pos), {"facing":"N"}))
        self.assertEqual(MODES[record["selections"]["mode"]], "MOVE")
        self.assertIsNone(record["watch_options"])

    def test_unsupported_ability_parameters_are_not_silently_dropped(self):
        char, state = preplant_fixture()
        with self.assertRaisesRegex(UnrepresentableAction, "unsupported_payload"):
            teacher_record(char, state, (list(char.pos), {"ability":"SMOKE", "target":tuple(char.pos), "facing":"N"}))

    def test_compressed_dataset_round_trip_preserves_labels_and_masks(self):
        char, state = preplant_fixture()
        records = [teacher_record(char, state, list(char.pos)),
                   teacher_record(char, state, (list(char.pos), {"facing":"E"}))]
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"round.npz"
            save_shard(path, records)
            restored = load_shard(path)
            for left, right in zip(records, restored):
                np.testing.assert_array_equal(left["obs"], right["obs"])
                self.assertEqual(left["selections"], right["selections"])
                for kind in left["masks"]:
                    np.testing.assert_array_equal(left["masks"][kind], right["masks"][kind])
                np.testing.assert_array_equal(left["watch_options"], right["watch_options"])
            with self.assertRaises(FileExistsError):
                save_shard(path, records)

    def test_preplant_stops_recording_at_actual_plant_within_tick(self):
        char, state = preplant_fixture()
        policy = SimpleNamespace(role="attacker", select=lambda *args: self.fail("Post-plant decision recorded"))
        controller = PreplantController(policy=policy)
        state["is_planted"] = True
        self.assertEqual(controller.decide_move(char, state)[1], "HOLD")

    def test_two_phase_runtime_uses_exactly_two_shared_policies(self):
        char, state = preplant_fixture()
        before = SimpleNamespace(role="attacker", select=lambda *args: ((list(char.pos), "PRE"), {"selections":{}}))
        after = SimpleNamespace(role="attacker", select=lambda *args: ((list(char.pos), "POST"), {"selections":{}}))
        controller = TwoPhaseAttackerController(preplant_policy=before, guard_policy=after)
        self.assertEqual(controller.decide_move(char, state)[1], "PRE")
        char.has_spike = True
        self.assertEqual(controller.decide_move(char, state)[1], "PRE")
        state["spike_pos"] = (8, 5)
        self.assertEqual(controller.decide_move(char, state)[1], "PRE")
        state["is_planted"] = True
        self.assertEqual(controller.decide_move(char, state)[1], "POST")
        self.assertIs(controller.preplant.policy, before)
        self.assertIs(controller.postplant.policy, after)

    def test_rl_requires_bc_before_creating_run(self):
        with TemporaryDirectory() as tmp:
            config = dict(BC_CHECKPOINT=Path(tmp)/"missing.pt", RUN_PATH=Path(tmp)/"rl")
            with self.assertRaisesRegex(FileNotFoundError, "BC stage first"):
                train_rl(config)
            self.assertFalse(config["RUN_PATH"].exists())

    def test_preplant_progress_does_not_force_one_site_or_retriever(self):
        reward = PreplantRewards.__new__(PreplantRewards)
        char = SimpleNamespace(pos=[0, 0], has_spike=True, is_alive=True, plant_timer=0)
        reward.own = {"ally":char}
        reward.goal_distances = {1:np.array([[12, 1], [2, 0]])}
        reward.goal_provider = lambda:1
        reward.game = SimpleNamespace(spike_pos=None, grid=np.zeros((2, 2), dtype=int))
        reward.drop, reward.drop_distances = None, None
        before = reward.objective_potential()["site_progress"]
        char.pos = [1, 1]
        after = reward.objective_potential()["site_progress"]
        self.assertGreater(potential_difference(before, after, .99, False), 0)
        char.has_spike = False
        reward.game.spike_pos = (0, 0)
        self.assertGreater(reward.objective_potential()["retrieval_progress"], 0)
        self.assertEqual(reward.objective_potential()["site_progress"], 0)

    def test_opponent_ai_cannot_silently_vary_by_roster(self):
        with self.assertRaisesRegex(ValueError, "fixed default"):
            play_preplant("FNC", 1, policy=object(), fixed_opponent_ai="fnatic_v3")

    def test_preplant_launcher_settings_do_not_invalidate_guard_training(self):
        from ghost_champions_v3.gc3_training import source_hash
        previous = source_hash()
        read_bytes = Path.read_bytes
        launchers = {"gc3_collect_attacker_preplant.py", "gc3_train_attacker_preplant_bc.py",
                     "gc3_train_attacker_preplant_rl.py"}
        with patch.object(Path, "read_bytes", lambda path: b"different preplant settings" if path.name in launchers else read_bytes(path)):
            self.assertEqual(source_hash(), previous)

    def test_evaluation_does_not_consume_training_random_stream(self):
        import random
        from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import rng_state
        saved = rng_state()
        def fake(*args, **kwargs):
            random.random(); np.random.random(); torch.rand(1)
            return [], dict(success=True, planted=True, actual_round_success=True)
        config = dict(SEED=2, OPPONENT_CODES=("GC",), EVALUATION_ROUNDS_PER_OPPONENT=2, FIXED_OPPONENT_AI="default")
        with patch("ghost_champions_v3.preplant.gc3_train_attacker_pipeline.play_preplant", side_effect=fake):
            result = evaluate_preplant(None, config)
        current = rng_state()
        self.assertEqual(saved["python_rng"], current["python_rng"])
        self.assertEqual(saved["numpy_rng"], current["numpy_rng"])
        self.assertTrue(torch.equal(saved["torch_rng"], current["torch_rng"]))
        self.assertEqual(result["plant_rate"], 1.)


if __name__ == "__main__":
    unittest.main()
