"""Relative movement, causal self memory and reuse of existing demonstrations."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.test.test_gc3_attacker_history import decision
from ghost_champions_v3.gc3_observation import ARENA_SHAPE, OBS_DIM
from ghost_champions_v3.gc3_engine_actions import legal_actions
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory, command_features
from ghost_champions_v3.preplant.gc3_policy_attacker_history import HistoryPolicy, DATA_SCHEMA_HASH, SCHEMA_HASH, MOVE_OFFSETS
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import save_shard, file_hash, dataset_manifest
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import reused_dagger_shards, bc_batches


class RelativeMovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_direction_probabilities_translate_and_remain_sampled(self):
        char, state = preplant_fixture()
        model = HistoryPolicy.fresh("attacker", 103)
        context = torch.zeros(model.hidden)
        probabilities = []
        for point in ((15, 15), (20, 30)):
            obs = np.zeros(OBS_DIM+13, dtype=np.float32)
            obs[:2] = np.array(point)/ARENA_SHAPE
            cells = [np.ravel_multi_index((point[0]+dr, point[1]+dc), ARENA_SHAPE) for dr,dc in MOVE_OFFSETS]
            mask = np.zeros(np.prod(ARENA_SHAPE), dtype=bool)
            mask[cells] = True
            record = dict(obs=obs, masks={"position":mask}, phase="standard")
            dist = model.distribution(context, record, "position")
            probabilities.append(dist.probs[cells])
            self.assertTrue(bool((dist.probs[cells] > 0).all()))
            self.assertEqual(float(dist.probs[~mask].sum().detach()), 0.)
            self.assertGreater(len(set(dist.sample((100,)).tolist())), 1)
        # The 1144-cell normalization changes reduction order after translation.
        torch.testing.assert_close(probabilities[0], probabilities[1], rtol=1e-6, atol=1e-8)

    def test_nonlocal_mask_rejected_instead_of_silently_reinterpreted(self):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        row = decision(history, char, state)
        row["masks"]["position"][np.ravel_multi_index((20, 30), ARENA_SHAPE)] = True
        model = HistoryPolicy.fresh("attacker", 104)
        with self.assertRaisesRegex(ValueError, "nonlocal"):
            model.distribution(model.contexts([row])[0], row, "position")

    def test_recent_self_action_is_directly_available_beside_all_history(self):
        char, state = preplant_fixture()
        ally = copy.copy(char); ally.name = "other"
        state["chars"].append(ally)
        history = MatchHistory(); history.begin_round()
        decision(history, char, state)
        decision(history, ally, state)
        query = decision(history, char, state)
        model = HistoryPolicy.fresh("attacker", 105)
        inputs = []
        hook = model.memory_merge[0].register_forward_pre_hook(lambda _,args:inputs.append(args[0].detach().clone()))
        try:
            with torch.no_grad():
                model.contexts([query])
                expected = model.event_features([history.events[0]])[0]
                torch.testing.assert_close(inputs[-1][0, -model.hidden:], expected)
                history.events[query["history_index"]]["action"][:] = 90
                model.contexts([query])
                torch.testing.assert_close(inputs[-1], inputs[-2])
        finally:
            hook.remove()

    def test_existing_data_schema_survives_model_change_and_query_shuffle(self):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        rows = [decision(history, char, state) for _ in range(12)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_shard(root/"match.npz", rows)
            manifest = dict(complete=True, schema_hash=DATA_SCHEMA_HASH, history=True,
                shards=[dict(file="match.npz", split=split, sha256=file_hash(root/"match.npz"))
                        for split in ("train", "validation")])
            (root/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            verified, _ = dataset_manifest(root)
            before = file_hash(root/"match.npz")
            shuffled = next(bc_batches(root, verified, "train", 12, True))
            self.assertEqual(sorted(r["history_index"] for r in shuffled), list(range(12)))
            for row in shuffled:
                np.testing.assert_array_equal(row["obs"], row["history"].events[row["history_index"]]["obs"])
            self.assertEqual(file_hash(root/"match.npz"), before)
            self.assertNotEqual(DATA_SCHEMA_HASH, SCHEMA_HASH)

    def test_old_history_model_remains_available_for_explicit_comparison(self):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        row = decision(history, char, state)
        old = HistoryPolicy("attacker", legacy=True)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"old.pt"
            old.save(path, stage="bc", trained=True)
            restored, _ = HistoryPolicy.load(path)
            self.assertTrue(restored.legacy)
            torch.testing.assert_close(old.contexts([row]), restored.contexts([row]))

    def test_reused_dagger_requires_same_dataset_and_unchanged_training_shards(self):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        rows = [decision(history, char, state)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_shard(root/"extra.npz", rows)
            shard = dict(root=str(root), file="extra.npz", sha256=file_hash(root/"extra.npz"))
            HistoryPolicy("attacker", legacy=True).save(root/"latest.pt", stage="bc", dataset_hash="same", dagger_shards=[shard])
            (root/"manifest.json").write_text(json.dumps(dict(input_hash="same")), encoding="utf-8")
            self.assertEqual(reused_dagger_shards(dict(REUSE_DAGGER_RUN_PATH=root), "same"), [shard])
            with self.assertRaisesRegex(ValueError, "same teacher dataset"):
                reused_dagger_shards(dict(REUSE_DAGGER_RUN_PATH=root), "different")
            with (root/"extra.npz").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "shard changed"):
                reused_dagger_shards(dict(REUSE_DAGGER_RUN_PATH=root), "same")


if __name__ == "__main__":
    unittest.main()
