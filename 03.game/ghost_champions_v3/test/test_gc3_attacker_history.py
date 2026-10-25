"""Causality, complete memory, roster consistency and BC readiness contracts."""
import copy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.gc3_observation import encode, map_hints, OBS_DIM
from ghost_champions_v3.preplant.gc3_observation_attacker_history import (
    MatchHistory, encode_public, command_features, ACTION_WIDTH)
from ghost_champions_v3.preplant.gc3_policy_attacker_history import HistoryPolicy
from ghost_champions_v3.preplant.gc3_teacher_attacker_actions import teacher_record
from ghost_champions_v3.preplant.gc3_teacher_attacker_actions import RecordingTeacher
from ghost_champions_v3.preplant.gc3_teacher_attacker_roster import RosterTeacher
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import save_shard, load_shard, dataset_manifest
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import bc_log_prob, train_rl, optimize_history
from ghost_champions_v3.preplant.gc3_learn_attacker_preplant import PreplantController


def decision(history, char, state, command=None):
    command = command if command is not None else list(char.pos)
    row = teacher_record(char, state, command)
    row.update(obs=encode_public(char, state, history), history=history, history_index=len(history.events))
    history.append(row["obs"], command_features(char, command, state), name=char.name)
    return row


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observation_reuses_gc3_and_contains_allied_spike_ownership(self):
        char, state = preplant_fixture()
        ally = copy.copy(char)
        ally.name, ally.has_spike = "other", False
        state["chars"].append(ally)
        before = encode_public(char, state)
        np.testing.assert_array_equal(before[:OBS_DIM], encode(char, state, map_hints(char, state)))
        ally.has_spike = True
        after = encode_public(char, state)
        self.assertFalse(np.array_equal(before, after))
        np.testing.assert_array_equal(before[:OBS_DIM], after[:OBS_DIM])

    def test_hidden_enemy_does_not_leak_into_history_observation(self):
        char, state = preplant_fixture()
        enemy = copy.copy(char)
        enemy.name, enemy.team, enemy.position_known = "hidden", "D", False
        state["chars"].append(enemy)
        before = encode_public(char, state)
        enemy.pos, enemy.hp, enemy.has_spike = [20, 30], 1, True
        np.testing.assert_array_equal(before, encode_public(char, state))

    def test_current_label_and_future_events_cannot_change_a_past_prediction(self):
        char, state = preplant_fixture()
        history = MatchHistory()
        history.begin_round()
        first = decision(history, char, state)
        second = decision(history, char, state)
        model = HistoryPolicy.fresh("attacker", 5)
        with torch.no_grad():
            before = model.contexts([second]).clone()
            history.events[1]["action"][:] = 9  # Current action is still unknown to this query.
            decision(history, char, state)
            history.events[-1]["obs"][:] = 90
            after = model.contexts([second])
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertEqual(first["history_index"], 0)

    def test_previous_round_and_oldest_event_remain_in_attention(self):
        char, state = preplant_fixture()
        history = MatchHistory()
        history.begin_round()
        for _ in range(300):
            decision(history, char, state)
        history.finish_round(dict(success=True, planted=True, reason="planted"))
        history.begin_round()
        row = decision(history, char, state)
        self.assertEqual(len(history.results), 1)
        self.assertEqual(row["obs"][-3:].tolist(), [1., 1., 1.])
        model = HistoryPolicy.fresh("attacker", 6)
        with torch.no_grad():
            model.key.weight.zero_(); model.key.bias.zero_()
            before = model.contexts([row]).clone()
            history.events[0]["action"][:] = 10
            after = model.contexts([row])
        self.assertGreater(float((before-after).abs().max()), 1e-6)

    def test_sequence_roundtrip_retains_prefix_boundaries_and_skipped_events(self):
        char, state = preplant_fixture()
        history = MatchHistory()
        history.begin_round()
        first = decision(history, char, state)
        # This public event has no BC label and must still survive.
        history.append(first["obs"], np.zeros(ACTION_WIDTH), name="unlabelled")
        history.finish_round(dict(success=True, planted=True, reason="planted"))
        history.begin_round()
        second = decision(history, char, state, (list(char.pos), {"facing":"E"}))
        with TemporaryDirectory() as tmp:
            shard = Path(tmp)/"match.npz"
            save_shard(shard, [first, second])
            restored = load_shard(shard)
            self.assertIs(restored[0]["history"], restored[1]["history"])
            self.assertEqual(restored[1]["history_index"], 3)
            self.assertEqual(len(restored[1]["history"].events), 4)
            self.assertEqual(restored[1]["history"].results, history.results)
            archive = Path(tmp)/"history.npz"
            history.save(archive)
            self.assertEqual(len(MatchHistory.load(archive).events), 4)

    def test_facing_marginal_and_history_receive_gradients_without_optimizer_updates(self):
        char, state = preplant_fixture()
        history = MatchHistory()
        history.begin_round()
        decision(history, char, state)
        row = decision(history, char, state, (list(char.pos), {"facing":"E"}))
        model = HistoryPolicy.fresh("attacker", 7)
        loss = -bc_log_prob(model, [row]).mean()
        loss.backward()
        self.assertIsNone(model.critic.weight.grad)
        self.assertGreater(float(model.action_features.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(model.query.weight.grad).all())
        other = dict(row, selections=dict(row["selections"]))
        other["selections"]["watch"] = int(np.flatnonzero(row["watch_options"])[-1])
        torch.testing.assert_close(bc_log_prob(model, [row]), bc_log_prob(model, [other]))

    def test_round_reset_preserves_memory_and_persists_before_next_decision(self):
        char, state = preplant_fixture()
        model = HistoryPolicy.fresh("attacker", 8)
        with TemporaryDirectory() as tmp:
            history = MatchHistory(Path(tmp)/"match.jsonl.gz")
            controller = PreplantController(policy=model, history=history)
            controller.reset_round()
            controller.decide_move(char, state)
            controller.reset_round()
            self.assertEqual(len(history.events), 1)
            controller.decide_move(char, state)
            with gzip.open(history.journal_path, "rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual([r["round_id"] for r in records], [0, 1])
            self.assertEqual(len(history.events), 2)

    def test_zero_plant_bc_cannot_start_ppo_or_create_run(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/"bc.pt"
            HistoryPolicy.fresh("attacker", 9).save(checkpoint, stage="bc", trained=True)
            config = dict(BC_CHECKPOINT=checkpoint, TORCH_THREADS=1, MIN_BC_PLANT_RATE=.5,
                          RUN_PATH=Path(tmp)/"rl", LOG_PATH=Path(tmp)/"logs"/"rl")
            with patch("ghost_champions_v3.preplant.gc3_train_attacker_pipeline.evaluate_preplant",
                       return_value=dict(plant_rate=0., success_rate=0.)):
                with self.assertRaisesRegex(RuntimeError, "PPO was not started"):
                    train_rl(config)
            self.assertFalse(config["RUN_PATH"].exists())

    def test_teacher_alias_changes_only_names_and_returns_real_ally_target(self):
        from party_presets import get_preset
        own = [SimpleNamespace(name=name, team="A", hp=77, ability_name="RECON")
               for name in get_preset("Ghost Champions").players]
        received = []
        teacher = SimpleNamespace(set_game=lambda game:None, decide_move=lambda char,state:
            (received.append((char, state)) or ([0, 0], {"ability":"DANCE", "target_name":state["chars"][1].name})))
        adapter = RosterTeacher(teacher)
        adapter.set_game(SimpleNamespace(chars=own))
        command = adapter.decide_move(own[0], dict(chars=own))
        self.assertEqual(command[1]["target_name"], own[1].name)
        self.assertEqual(received[0][0].hp, own[0].hp)
        self.assertEqual(received[0][0].ability_name, own[0].ability_name)
        self.assertEqual([c.name for c in received[0][1]["chars"]], list(get_preset("Gorigons").players))

    def test_dagger_labels_teacher_but_stores_executed_student_action(self):
        char, state = preplant_fixture()
        teacher = SimpleNamespace(decide_move=lambda c,s:(list(c.pos), {"facing":"E"}))
        student = SimpleNamespace(select=lambda c,s,h:((list(c.pos), "MOVE"), {}))
        history = MatchHistory()
        history.begin_round()
        recorder = RecordingTeacher(teacher, history=history, executing_policy=student)
        command = recorder.decide_move(char, state)
        self.assertEqual(command[1], "MOVE")
        from ghost_champions_v3.gc3_engine_actions import MODES
        self.assertEqual(MODES[recorder.rows[0]["selections"]["mode"]], "WATCH")
        self.assertEqual(history.events[0]["command"][1], "MOVE")

    def test_live_attack_defense_controllers_share_round_boundaries(self):
        from ghost_champions_v3.preplant.gc3_history_attacker_match import ObservedDefender
        history = MatchHistory()
        history.live = True
        view = SimpleNamespace(current_round=1, analytics_tracker=SimpleNamespace(round_records=[]))
        controller = PreplantController(policy=HistoryPolicy.fresh("attacker", 10), history=history)
        defender = ObservedDefender(SimpleNamespace(set_game=lambda game:None, reset_round=lambda:None), history)
        controller.set_game(view); defender.set_game(view)
        controller.reset_round(); defender.reset_round()
        self.assertEqual(history.round_id, 0)
        view.analytics_tracker.round_records.append(dict(winner="defender", planted=False, reason="time_expired"))
        controller.on_round_finished(dict(success=False, planted=False, reason="time_expired"))
        view.current_round = 2
        defender.reset_round(); controller.reset_round()
        self.assertEqual(history.round_id, 1)
        self.assertEqual(len(history.results), 1)
        self.assertFalse(history.results[0]["success"])

    def test_ppo_causal_replay_and_anchor_gradients_with_zero_weight_updates(self):
        char, state = preplant_fixture()
        history = MatchHistory()
        history.begin_round()
        model = HistoryPolicy.fresh("attacker", 11)
        rows = []
        for index in range(4):
            command, row = model.select(char, state, history)
            history.append(row["obs"], command_features(char, command, state), name=char.name)
            row.update(advantage=float(index)-1.5, **{"return":float(index)/4})
            rows.append(row)
        with torch.no_grad():
            replay, _, _ = model.evaluate_records(rows)
        torch.testing.assert_close(replay, torch.tensor([r["logp"] for r in rows]), atol=3e-6, rtol=3e-6)
        anchor = HistoryPolicy.fresh("attacker", 12).requires_grad_(False)
        initial = {k:v.clone() for k,v in model.state_dict().items()}
        optimizer = SimpleNamespace(zero_grad=lambda:model.zero_grad(), step=lambda:None)
        config = dict(PPO_EPOCHS=1, BATCH_SIZE=4, PPO_CLIP=.2, ENTROPY_WEIGHT=.01, BC_KL_WEIGHT=.1)
        loss = optimize_history(model, anchor, optimizer, rows, config)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(torch.isfinite(model.action_features.weight.grad).all())
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in model.state_dict().items()))

    def test_new_checkpoints_reject_old_flat_models_and_keep_history_predictions(self):
        from ghost_champions_v3.gc3_policy import FreshPolicy
        char, state = preplant_fixture()
        history = MatchHistory()
        history.begin_round()
        row = decision(history, char, state)
        model = HistoryPolicy.fresh("attacker", 13)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"new.pt"
            model.save(path, stage="bc", trained=False)
            restored, metadata = HistoryPolicy.load(path)
            torch.testing.assert_close(model.contexts([row]), restored.contexts([row]))
            self.assertEqual(metadata["policy_type"], "gc3_public_match_history")
            old = Path(tmp)/"old.pt"
            FreshPolicy.fresh("attacker", 14).save(old)
            with self.assertRaisesRegex(ValueError, "old flat BC/RL cannot resume"):
                HistoryPolicy.load(old)


if __name__ == "__main__":
    unittest.main()
