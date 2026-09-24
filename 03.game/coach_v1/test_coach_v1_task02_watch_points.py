import copy
from pathlib import Path
import unittest

from map_data import NEW_MAZE_STR

from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.common.hashing import canonical_json_sha256, load_json_config, map_sha256
from coach_v1.common.types import Facing, Side
from coach_v1.common.watch_points import (
    ALLOWED_SITUATIONS,
    WATCH_POINTS_SCHEMA_VERSION,
    WatchPointConfigError,
    load_watch_points,
    render_watch_points,
    validate_watch_points,
)
from coach_v1.convert_watch_points_map import (
    extract_watch_positions,
    load_marked_map,
    convert_map_to_config,
)


ROOT = Path(__file__).resolve().parent
EXPECTED_CONFIG_HASH = "2c5ff118249917e926f0c7c8aba206642aaa5a45a2d33b29cbfa8dcb9aee31b7"
MARKED_MAP_PATH = ROOT / "config" / "watch_points_map.py"


class CoachV1Task02WatchPointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_json_config(WATCH_POINTS_CONFIG_PATH)
        cls.config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        cls.map_rows = NEW_MAZE_STR.strip().splitlines()

    def _changed(self):
        return copy.deepcopy(self.raw)

    def test_real_configuration_matches_fixed_map_and_hash(self):
        self.assertEqual(WATCH_POINTS_SCHEMA_VERSION, self.config.schema_version)
        self.assertEqual((26, 44), (self.config.map_rows, self.config.map_columns))
        self.assertEqual(map_sha256(NEW_MAZE_STR), self.config.map_hash)
        self.assertEqual(EXPECTED_CONFIG_HASH, self.config.config_hash)
        self.assertEqual(canonical_json_sha256(self.raw), self.config.config_hash)
        self.assertEqual(35, len(self.config.points))

    def test_json_coordinates_are_generated_from_marked_map(self):
        marked_map = load_marked_map(MARKED_MAP_PATH)
        positions = extract_watch_positions(marked_map, NEW_MAZE_STR)
        self.assertEqual(35, len(positions))
        self.assertEqual(set(positions), {point.position for point in self.config.points})
        generated = convert_map_to_config(marked_map, NEW_MAZE_STR, self.raw)
        self.assertEqual(generated, self.raw)

    def test_all_points_are_unique_walkable_and_fully_typed(self):
        point_ids = [point.point_id for point in self.config.points]
        positions = [point.position for point in self.config.points]
        self.assertEqual(len(point_ids), len(set(point_ids)))
        self.assertEqual(len(positions), len(set(positions)))
        for point in self.config.points:
            with self.subTest(point=point.point_id):
                row, column = point.position
                self.assertNotEqual("1", self.map_rows[row][column])
                self.assertIsInstance(point.facing, Facing)
                self.assertGreaterEqual(point.importance, 1)
                self.assertLessEqual(point.importance, 5)
                self.assertGreaterEqual(point.random_radius, 1)
                self.assertLessEqual(point.random_radius, 3)
                self.assertTrue(point.tags)

    def test_both_sides_and_all_phase_situations_are_registered(self):
        self.assertTrue(self.config.for_side(Side.ATTACKER))
        self.assertTrue(self.config.for_side("defender"))
        for situation in ALLOWED_SITUATIONS:
            with self.subTest(situation=situation):
                self.assertTrue(self.config.for_situation(situation))

    def test_visualizer_displays_coordinates_facing_and_filter(self):
        rendered = render_watch_points(
            self.config, NEW_MAZE_STR, side="attacker", situation="guard"
        )
        self.assertIn("00: ", rendered)
        self.assertIn("facing: ↑=N ↗=NE →=E ↘=SE ↓=S ↙=SW ←=W ↖=NW", rendered)
        self.assertIn("upper_left_junction (2, 17) ↙", rendered)
        self.assertNotIn("attacker_spawn_exit (22, 18)", rendered)
        self.assertIn(f"config_sha256: {EXPECTED_CONFIG_HASH}", rendered)

    def test_rejects_outside_wall_and_duplicate_positions(self):
        cases = []
        outside = self._changed()
        outside["points"][0]["position"] = [26, 0]
        cases.append(("outside map", outside))
        wall = self._changed()
        wall["points"][0]["position"] = [0, 0]
        cases.append(("is a wall", wall))
        duplicate_position = self._changed()
        duplicate_position["points"][1]["position"] = duplicate_position["points"][0][
            "position"
        ]
        duplicate_position["points"][1]["facing"] = duplicate_position["points"][0][
            "facing"
        ]
        cases.append(("duplicate watch-point position", duplicate_position))

        for message, raw in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                WatchPointConfigError, message
            ):
                validate_watch_points(raw, NEW_MAZE_STR)

    def test_rejects_duplicate_id_and_invalid_point_fields(self):
        cases = []
        duplicate_id = self._changed()
        duplicate_id["points"][1]["id"] = duplicate_id["points"][0]["id"]
        cases.append(("duplicate watch-point id", duplicate_id))
        invalid_facing = self._changed()
        invalid_facing["points"][0]["facing"] = "UP"
        cases.append(("facing must be", invalid_facing))
        blocked_facing = self._changed()
        blocked_facing["points"][0]["facing"] = "N"
        cases.append(("facing points immediately", blocked_facing))
        invalid_importance = self._changed()
        invalid_importance["points"][0]["importance"] = 6
        cases.append(("importance must be in range", invalid_importance))
        invalid_radius = self._changed()
        invalid_radius["points"][0]["random_radius"] = 0
        cases.append(("random_radius must be in range", invalid_radius))
        invalid_tag = self._changed()
        invalid_tag["points"][0]["tags"] = ["use_smoke_here"]
        cases.append(("unsupported values", invalid_tag))
        side_mismatch = self._changed()
        side_mismatch["points"][0]["sides"] = ["defender"]
        cases.append(("situations do not match sides", side_mismatch))

        for message, raw in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                WatchPointConfigError, message
            ):
                validate_watch_points(raw, NEW_MAZE_STR)

    def test_rejects_unknown_keys_and_changed_map(self):
        extra_key = self._changed()
        extra_key["points"][0]["ability_command"] = "smoke"
        with self.assertRaisesRegex(WatchPointConfigError, "invalid keys"):
            validate_watch_points(extra_key, NEW_MAZE_STR)

        changed_map = NEW_MAZE_STR.replace("0", "1", 1)
        with self.assertRaisesRegex(WatchPointConfigError, "map metadata does not match"):
            validate_watch_points(self.raw, changed_map)

    def test_task02_has_no_actor_observation_or_enemy_truth_input(self):
        task02_sources = [
            ROOT / "common" / "watch_points.py",
            ROOT / "validate_watch_points.py",
            ROOT / "convert_watch_points_map.py",
        ]
        sources = "\n".join(path.read_text(encoding="utf-8") for path in task02_sources)
        forbidden = (
            'game_state["chars"]',
            "PerceivedGameView.real_game",
            "PerceivedCharacter.real_character",
            "enemy_position",
            "enemy_pos",
            "critic_observation",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, sources)

        config_text = WATCH_POINTS_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertNotIn("enemy", config_text.lower())

    def test_task02_does_not_import_or_modify_core_behavior(self):
        common_source = (ROOT / "common" / "watch_points.py").read_text(
            encoding="utf-8"
        )
        for module_name in (
            "abilities_los",
            "battle_logic",
            "game_core",
            "map_data_defender_setup",
            "run_game",
        ):
            with self.subTest(module=module_name):
                self.assertNotIn(f"import {module_name}", common_source)
                self.assertNotIn(f"from {module_name}", common_source)


if __name__ == "__main__":
    unittest.main()
