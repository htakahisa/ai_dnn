"""Actual round credit, causal public navigation and zero-update transfer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_rollout import Recorder
from ghost_champions_v3.gc3_observation import ARENA_SHAPE, CELLS, CONTEXT_WIDTH, UNIT_WIDTH, BOARD_CHANNELS
from ghost_champions_v3.gc3_reward_phase import PhaseRewards, WIN_REWARD, LOSS_REWARD
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import PreplantRewards, credit_round_outcome, BATTLE_TIME_COST_PER_TICK
from ghost_champions_v3.preplant.gc3_policy_attacker_site import SiteHistoryPolicy, NAVIGATION_SCHEMA_HASH
from ghost_champions_v3.preplant.gc3_policy_attacker_history import load_preplant_policy
from ghost_champions_v3.preplant.gc3_features_attacker_navigation import navigation_features, public_layout
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory, command_features
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import initialize, rl_input_hash, optimize_history, rl_evaluation_rank
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash, load_shard, save_shard
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_plant_replaces_reset_penalty_without_paying_a_win(self):
        reward = PreplantRewards.__new__(PreplantRewards)
        reward.tick_goal = 0
        reward.tick_round_timer = 100
        reward.game = SimpleNamespace(round_timer=99, is_planted=True, round_over=False)
        reward.objectives = SimpleNamespace(finish_tick=lambda terminal:({"site_progress":0.}, {"planter":{"plant_complete":5.}}))
        with patch.object(PhaseRewards, "finish_tick", return_value=({"outcome":WIN_REWARD, "site_progress":-.2}, {})):
            shared, personal = reward.finish_tick(True, True)
        self.assertNotIn("outcome", shared)
        self.assertNotIn("round_outcome", shared)
        self.assertEqual(shared["site_progress"], 0.)
        self.assertNotIn("plant_complete",shared)
        self.assertEqual(personal["planter"]["plant_complete"],5.)
        self.assertEqual(shared["battle_time"], -BATTLE_TIME_COST_PER_TICK)

    def test_defuse_after_plant_is_negative_and_discounted_over_real_ticks(self):
        for success, expected in ((False, LOSS_REWARD), (True, WIN_REWARD)):
            recorder = Recorder()
            for name in ("carrier", "support"):
                recorder.record(name, dict(value=0.))
            recorder.reward(0., .9, False, components={})  # Actual plant tick.
            credit_round_outcome(recorder, 3, .9, success)
            rows = recorder.advantages(.95)
            self.assertEqual(len(rows), 2)  # No guard decisions entered PPO.
            for row in rows:
                self.assertAlmostEqual(row["reward"], expected*.9**3)
                self.assertAlmostEqual(row["return"], expected*.9**3)
                self.assertEqual(row["discount"], 0.)
                self.assertEqual(row["age"], 4)
                self.assertEqual(list(row["reward_components"]), ["round_outcome"])

    def test_preplant_terminal_result_is_paid_once_and_setup_has_no_time_cost(self):
        reward = PreplantRewards.__new__(PreplantRewards)
        reward.tick_goal = 0
        reward.tick_round_timer = 100
        reward.game = SimpleNamespace(round_timer=100, is_planted=False, round_over=True)
        reward.objectives = SimpleNamespace(finish_tick=lambda terminal:({}, {}))
        with patch.object(PhaseRewards, "finish_tick", return_value=({"outcome":LOSS_REWARD}, {})):
            shared, _ = reward.finish_tick(True, False)
        self.assertEqual(shared, {"round_outcome":LOSS_REWARD, "battle_time":0.})
        recorder = Recorder(); recorder.record("actor", dict(value=0.))
        recorder.reward(sum(shared.values()), .99, True, components=shared)
        credit_round_outcome(recorder, 0, .99, False)
        self.assertEqual(recorder.advantages(.95)[0]["return"], LOSS_REWARD)

    def test_best_model_prioritizes_wins_over_plants(self):
        losing = dict(actual_round_win_rate=0., plant_rate=1.)
        winning = dict(actual_round_win_rate=.5, plant_rate=.5)
        self.assertGreater(rl_evaluation_rank(winning), rl_evaluation_rank(losing))

    def sampled_rows(self, model):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        rows = []
        for _ in range(4):
            command, row = model.select(char, state, history)
            history.append(row["obs"], command_features(char, command, state), name=char.name)
            row.update(advantage=len(rows)-1.5, watch_options=None, **{"return":len(rows)/4})
            rows.append(row)
        return rows

    def test_zero_residual_preserves_old_policy_weights_context_and_data(self):
        old = SiteHistoryPolicy.fresh("attacker", 501)
        new = SiteHistoryPolicy.with_navigation(old)
        rows = self.sampled_rows(old)
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(old.contexts(rows), new.contexts(rows), rtol=0, atol=0)
        for a, b in zip(old.evaluate_records(rows), new.evaluate_records(rows)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"existing_labels.npz"
            save_shard(path, rows)
            restored = load_shard(path)
            self.assertTrue(torch.isfinite(new.evaluate_records(restored)[0]).all())
            checkpoint = Path(tmp)/"navigation.pt";new.save(checkpoint, stage="rl", trained=True)
            loaded, _ = load_preplant_policy(checkpoint)
            self.assertEqual(loaded.schema_hash, NAVIGATION_SCHEMA_HASH)
            torch.testing.assert_close(new.contexts(rows), loaded.contexts(rows), rtol=0, atol=0)

    def test_navigation_site_sampling_and_critic_never_read_their_own_site_label(self):
        model = SiteHistoryPolicy.with_navigation(SiteHistoryPolicy.fresh("attacker", 502))
        with torch.no_grad():
            torch.nn.init.normal_(model.navigation[-1].weight, std=.1)
        rows = self.sampled_rows(model)
        replay, _, values = model.evaluate_records(rows)
        torch.testing.assert_close(replay, torch.tensor([r["logp"] for r in rows]), rtol=3e-6, atol=3e-6)
        torch.testing.assert_close(values, torch.tensor([r["value"] for r in rows]), rtol=3e-6, atol=3e-6)
        a, b = dict(rows[0], goal_site=0), dict(rows[0], goal_site=1)
        contexts = model.contexts([a, b])
        pa = model.distribution(contexts[0], a, "site").probs
        pb = model.distribution(contexts[1], b, "site").probs
        torch.testing.assert_close(pa, pb, rtol=3e-6, atol=3e-6)
        self.assertTrue(bool((pa > 0).all()))
        va = model.critic(model.neutral_context(contexts[:1], [a]))
        vb = model.critic(model.neutral_context(contexts[1:], [b]))
        torch.testing.assert_close(va, vb, rtol=3e-6, atol=3e-6)

    def test_navigation_distinguishes_wall_and_ally_and_ignores_future_events(self):
        model = SiteHistoryPolicy.with_navigation(SiteHistoryPolicy.fresh("attacker", 503))
        rows = self.sampled_rows(model)
        row = dict(rows[0], obs=rows[0]["obs"].copy(), _navigation_features={})
        point = (12, 1); row["obs"][:2] = np.asarray(point)/ARENA_SHAPE
        board, units = public_layout(row["obs"])
        board[0, 11, 1] = 1.; board[0, 13, 1] = 0.
        units[:5] = 0.
        units[0, :4] = [1, 0, 1, 1]
        units[0, 4:6] = np.asarray((13, 1))/ARENA_SHAPE
        features = navigation_features(row, 0).copy()
        self.assertEqual(features[8:10].tolist(), [1., 0.])   # North: wall.
        self.assertEqual(features[16:18].tolist(), [0., 1.]) # South: ally.
        changed = dict(row, _navigation_features={})
        row["history"].events[-1]["obs"][:2] = 0.
        np.testing.assert_array_equal(features, navigation_features(changed, 0))

    def test_navigation_PPO_gradients_reach_branch_without_optimizer_updates(self):
        model = SiteHistoryPolicy.with_navigation(SiteHistoryPolicy.fresh("attacker", 504))
        rows = self.sampled_rows(model)
        anchor = SiteHistoryPolicy.fresh("attacker", 505).requires_grad_(False)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        optimizer = SimpleNamespace(zero_grad=model.zero_grad, step=lambda:None)
        config = dict(PPO_EPOCHS=1, BATCH_SIZE=4, PPO_CLIP=.2, ENTROPY_WEIGHT=.01, BC_KL_WEIGHT=.1)
        self.assertTrue(np.isfinite(optimize_history(model, anchor, optimizer, rows, config)))
        self.assertGreater(float(model.navigation[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(torch.equal(v, before[k]) for k,v in model.state_dict().items()))

    def test_new_run_transfers_old_step_but_never_edits_old_run_or_optimizer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); bc=root/"bc.pt";old=root/"old.pt";guard=root/"guard.pt"
            model = SiteHistoryPolicy.fresh("attacker", 506)
            model.save(bc, stage="bc", trained=True)
            model.save(old, stage="rl", trained=True, step=32484, bc_checkpoint_hash=file_hash(bc))
            guard.write_bytes(b"frozen-guard")
            old_hash = file_hash(old)
            config = dict(RESUME=False, SEED=507, INITIAL_CHECKPOINT=old, BC_CHECKPOINT=bc,
                POSTPLANT_CHECKPOINT=guard, SITE_GOAL_POLICY=True, NAVIGATION_FEATURES=True, LEARNING_RATE=.0001)
            run = root/"new";run.mkdir()
            policy, optimizer, saved = initialize(config, run, "rl", rl_input_hash(config, file_hash(bc)))
            self.assertEqual(policy.schema_hash, NAVIGATION_SCHEMA_HASH)
            self.assertEqual(len(optimizer.state), 0)
            self.assertIsNone(saved)
            self.assertEqual(file_hash(old), old_hash)
            self.assertEqual(json.loads((run/"manifest.json").read_text())["inherited_step"], 32484)


if __name__ == "__main__":
    unittest.main()
