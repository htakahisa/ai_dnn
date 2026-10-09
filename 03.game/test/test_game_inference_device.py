"""Device overrides and migration of GPU-tagged saved inference cases."""
import io
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from simulation_runtime import cpu_inference, resolve_inference_device
from concon_v1.co1_retake_cases import CaseUnpickler, _restore_inference_device


class InferenceDeviceTests(unittest.TestCase):
    def test_all_learned_opponent_factories_forward_explicit_cpu(self):
        from run_game import _build_team_ai
        factories = {
            "touyama_gaming_v2": ("run_game.Tv2TouyamaAttackerController", "run_game.Tv2TouyamaDefenderController"),
            "omoko_gaming_v1": ("run_game.Ov1AttackerController", "run_game.Ov1DefenderController"),
            "gc_v1": ("run_game.GhostChampionsV1AttackerController", "run_game.GhostChampionsV1DefenderController"),
            "toru_ai_v3.1": ("run_game.MultiRoleAttackerController", "run_game.MultiRoleDefenderController"),
            "frc_v1": ("frc_v1.controller.FrcAttackerController", "frc_v1.controller.FrcDefenderController"),
        }
        for key, (attacker, defender) in factories.items():
            with self.subTest(opponent=key), patch(attacker) as attack, patch(defender) as defend:
                team = _build_team_ai(key, device="cpu")
                team.attacker_factory()
                team.defender_factory()
                self.assertEqual(attack.call_args.kwargs["device"], "cpu")
                self.assertEqual(defend.call_args.kwargs["device"], "cpu")

    def test_explicit_device_wins_over_match_default(self):
        with cpu_inference():
            self.assertEqual(resolve_inference_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(resolve_inference_device(), torch.device("cpu"))
            self.assertEqual(resolve_inference_device("auto"), torch.device("cpu"))

    def test_cpu_does_not_query_cuda_and_auto_keeps_existing_selection(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA queried")):
            self.assertEqual(resolve_inference_device("cpu").type, "cpu")
            self.assertEqual(resolve_inference_device(default="cpu").type, "cpu")
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_inference_device("auto").type, "cuda")
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_inference_device("auto").type, "cpu")

    def test_saved_cuda_tensor_maps_to_cpu_and_cache_clones_stay_independent(self):
        key = "a" * 64
        token = ("tensor", key, 0, "cuda:0")

        class TokenPickler(pickle.Pickler):
            def persistent_id(self, obj):
                return token if isinstance(obj, torch.Tensor) else None

        stream = io.BytesIO()
        TokenPickler(stream).dump([torch.ones(2)])
        with tempfile.TemporaryDirectory() as directory:
            tensor_dir = Path(directory)
            torch.save(torch.nn.Parameter(torch.tensor([1., 2.])), tensor_dir / (key + ".pt"))
            cache = {}
            with patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA initialized")):
                first = CaseUnpickler(io.BytesIO(stream.getvalue()), tensor_dir, cache, device="cpu").load()[0]
                second = CaseUnpickler(io.BytesIO(stream.getvalue()), tensor_dir, cache, device="cpu").load()[0]
            self.assertEqual(first.device.type, "cpu")
            self.assertIsInstance(first, torch.nn.Parameter)
            with torch.no_grad():
                first.add_(5)
            torch.testing.assert_close(second, torch.tensor([1., 2.]))
            self.assertEqual(next(iter(cache))[1], "cpu")

    def test_saved_controller_devices_and_legacy_missing_fields_are_restored(self):
        old = SimpleNamespace(model=torch.nn.Linear(2, 2))
        current = SimpleNamespace(device=torch.device("cuda:0"), policy_net=torch.nn.Linear(2, 2))
        opening = SimpleNamespace(device="cuda", _requested_device="cuda")
        game = SimpleNamespace(controllers=[old, current, opening])
        old.game = game  # Real saved cases contain cycles and shared controllers.
        _restore_inference_device(game, "cpu")
        for controller in game.controllers:
            self.assertEqual(controller.device, torch.device("cpu"))
        self.assertEqual(opening._requested_device, torch.device("cpu"))

    def test_legacy_case_without_override_uses_its_saved_model_device(self):
        old = SimpleNamespace(model=torch.nn.Linear(2, 2))
        current = SimpleNamespace(device="cuda:1")
        _restore_inference_device(SimpleNamespace(controllers=[old, current]))
        self.assertEqual(old.device, torch.device("cpu"))
        self.assertEqual(current.device, "cuda:1")


if __name__ == "__main__":
    unittest.main()
