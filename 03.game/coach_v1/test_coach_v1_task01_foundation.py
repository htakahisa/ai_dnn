import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from map_data import NEW_MAZE_STR
from party_presets import PARTY_PRESETS

from coach_v1.common.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointMetadata,
    build_checkpoint_metadata,
    build_checkpoint_payload,
    validate_checkpoint_compatibility,
)
from coach_v1.common.constants import (
    CHARACTER_CHECKPOINT_PATHS,
    COACH_CHECKPOINT_PATHS,
    DEFAULT_RUNTIME_PATHS,
    FACING_DELTAS,
    FIXED_ROSTER_NAMES,
    MAP_COLUMNS,
    MAP_ROWS,
    RuntimePaths,
)
from coach_v1.common.hashing import (
    canonical_json_sha256,
    load_hashed_json_config,
    load_json_config,
    map_sha256,
)
from coach_v1.common.types import (
    Facing,
    ModelTarget,
    MovementAction,
    ObjectiveAction,
    Side,
    TacticalIntent,
)
from coach_v1.common.versions import (
    CHARACTER_ACTION_VERSION,
    CHARACTER_OBSERVATION_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    COACH_ACTION_VERSION,
    COACH_OBSERVATION_VERSION,
)


ROOT = Path(__file__).resolve().parent


class CoachV1Task01FoundationTest(unittest.TestCase):
    def _metadata(self, **overrides):
        values = {
            "target": ModelTarget.coach(Side.ATTACKER),
            "map_hash": map_sha256(NEW_MAZE_STR),
            "watch_points_hash": canonical_json_sha256({"points": []}),
            "model_config": {"hidden_size": 64},
            "training_seed": 7,
            "training_step": 100,
            "created_at_utc": "2026-09-24T00:00:00Z",
        }
        values.update(overrides)
        return build_checkpoint_metadata(**values)

    def test_fixed_contract_constants_match_current_project(self):
        rows = NEW_MAZE_STR.strip().splitlines()
        self.assertEqual((MAP_ROWS, MAP_COLUMNS), (len(rows), len(rows[0])))
        self.assertEqual(PARTY_PRESETS["Gorigons"].players, FIXED_ROSTER_NAMES)
        self.assertEqual(
            ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            [facing.value for facing in Facing],
        )
        self.assertEqual((-1, 1), FACING_DELTAS[Facing.NE])

    def test_action_and_intent_contracts_are_complete(self):
        self.assertEqual(
            ["STAY", "MOVE_N", "MOVE_E", "MOVE_S", "MOVE_W"],
            [action.value for action in MovementAction],
        )
        self.assertEqual(
            ["NONE", "PLANT", "DEFUSE"],
            [action.value for action in ObjectiveAction],
        )
        self.assertEqual(9, len(TacticalIntent))

    def test_versions_and_separate_checkpoint_destinations_are_frozen(self):
        self.assertEqual("coach-checkpoint-v1", CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual("coach-observation-v1", COACH_OBSERVATION_VERSION)
        self.assertEqual("coach-action-v1", COACH_ACTION_VERSION)
        self.assertEqual("character-observation-v1", CHARACTER_OBSERVATION_VERSION)
        self.assertEqual("character-action-v1", CHARACTER_ACTION_VERSION)
        self.assertNotEqual(
            COACH_CHECKPOINT_PATHS["attacker"],
            COACH_CHECKPOINT_PATHS["defender"],
        )
        self.assertEqual(5, len(CHARACTER_CHECKPOINT_PATHS))
        self.assertEqual(ROOT / "logs", DEFAULT_RUNTIME_PATHS.logs)

    def test_runtime_directories_are_only_created_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            paths = RuntimePaths(base / "checkpoints", base / "logs", base / "reports")
            self.assertFalse(paths.logs.exists())
            paths.ensure_exists()
            self.assertTrue(all(dataclasses.astuple(paths)[i].is_dir() for i in range(3)))

    def test_hashes_are_canonical_and_sensitive_to_real_map_edits(self):
        self.assertEqual(
            map_sha256(NEW_MAZE_STR),
            map_sha256(NEW_MAZE_STR.replace("\n", "\r\n")),
        )
        self.assertNotEqual(
            map_sha256(NEW_MAZE_STR),
            map_sha256(NEW_MAZE_STR.replace("0", "1", 1)),
        )
        self.assertEqual(
            canonical_json_sha256({"b": 2, "a": [1]}),
            canonical_json_sha256({"a": [1], "b": 2}),
        )

    def test_json_configuration_loading_and_hashing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text('{"name":"警戒","points":[]}', encoding="utf-8")
            config, digest = load_hashed_json_config(path)
            self.assertEqual("警戒", config["name"])
            self.assertEqual(canonical_json_sha256(config), digest)

            path.write_text('{"same":1,"same":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_json_config(path)

            path.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                load_json_config(path)

    def test_checkpoint_metadata_round_trip_and_payload(self):
        metadata = self._metadata()
        restored = CheckpointMetadata.from_dict(metadata.to_dict())
        self.assertEqual(metadata, restored)
        payload = build_checkpoint_payload(
            metadata=metadata,
            model_state_dict={"weight": 1},
            optimizer_state_dict={"state": {}},
        )
        self.assertEqual(metadata.to_dict(), payload["metadata"])
        self.assertIn("optimizer_state_dict", payload)

    def test_checkpoint_mismatch_is_never_silently_accepted(self):
        loaded = self._metadata()
        defender = self._metadata(target=ModelTarget.coach(Side.DEFENDER))
        with self.assertRaisesRegex(CheckpointCompatibilityError, "target_id"):
            validate_checkpoint_compatibility(loaded, defender)

        changed_map = dataclasses.replace(loaded, map_hash="0" * 64)
        with self.assertRaisesRegex(CheckpointCompatibilityError, "map_hash"):
            validate_checkpoint_compatibility(loaded, changed_map)

    def test_metadata_rejects_wrong_versions_on_compatibility_check(self):
        expected = self._metadata()
        with self.assertRaisesRegex(
            CheckpointCompatibilityError, "observation version"
        ):
            dataclasses.replace(expected, observation_version="coach-observation-v2")

    def test_foundation_has_no_actor_observation_or_enemy_truth_reference(self):
        common_dir = ROOT / "common"
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in common_dir.glob("*.py")
        )
        forbidden = (
            'game_state["chars"]',
            "PerceivedGameView.real_game",
            "PerceivedCharacter.real_character",
            "enemy_position",
            "critic_observation",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, sources)

    def test_common_modules_have_no_torch_or_core_import_dependency(self):
        common_dir = ROOT / "common"
        for path in common_dir.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("import torch", source)
                self.assertNotIn("from game_core", source)
                self.assertNotIn("from battle_logic", source)


if __name__ == "__main__":
    unittest.main()
