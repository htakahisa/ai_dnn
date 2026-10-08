"""Player status reaches the policy without reading hidden live enemy state."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from game_core import Character
from roster_utils import roster_information
from ghost_champions_v2.config import load_config
from ghost_champions_v2.tactics import AttackerTactics
from ghost_champions_v2.controller import GhostChampionsV2AttackerController
from ghost_champions_v2.rl.controller import LearnedAttackerController
from ghost_champions_v2.rl.observation import (
    ObservationEncoder, OBS_DIM, LEGACY_V3_BLOCKS,
)
from ghost_champions_v2.rl.unit_observation import (
    UNIT_FIELDS, UNIT_WIDTH, SELF_FIELDS, LIVE_STATS, action_capabilities,
    player_abilities_status, self_capabilities,
)
from ghost_champions_v2.rl.actions import ACTIONS, candidates
from ghost_champions_v2.rl.rollout import Recorder
from ghost_champions_v2.rl.policy import ResidualPolicy
from ghost_champions_v2.tools.upgrade_observation_checkpoint import upgraded_policy


class UnitObservationTests(unittest.TestCase):
    def setUp(self):
        self.own = Character("Chronicle", "A", (5, 5), "white", "red", ultimate_points=8)
        self.ally = Character("Leo", "A", (5, 3), "white", "red")
        self.enemy = Character("Demon1", "D", (5, 8), "white", "green")
        self.enemy.position_known = True
        self.state = dict(grid=np.zeros((20, 20), int), chars=[self.own, self.ally, self.enemy],
                          battle_tick=10, is_planted=False, smoke_cells=set())
        self.state.update(roster_information(self.state["chars"], "A"))
        self.tactics = AttackerTactics(load_config())
        self.tactics.observe(self.own, self.state)
        self.proposal = ([5, 6], "MOVE")

    def rows(self):
        return player_abilities_status(self.own, self.state).reshape(10, UNIT_WIDTH)

    def row(self, index):
        return dict(zip(UNIT_FIELDS, self.rows()[index]))

    def encode(self):
        return ObservationEncoder().encode(self.own, self.state, self.tactics, self.proposal)

    def test_allied_and_visible_enemy_abilities_hp_shields_and_debuffs_reach_input(self):
        self.own.erosion_curse, self.own.fate_loom = 3, 2
        self.own.shield_hp, self.own.max_shield_hp = 25, 60
        self.own.shield_piercer, self.own.shield_crash = True, 15
        self.enemy.hp, self.enemy.max_hp = 40, 65
        self.enemy.shield_hp, self.enemy.max_shield_hp = 7, 80
        self.enemy.ability_seal_remaining, self.enemy.fate_loom_remaining = 4, 3
        self.enemy.fate_max_hp_lost = 35
        self.enemy.ultimate_points = 2
        self.ally.shield_hp = 17
        before = self.encode()
        own, ally, enemy = self.row(0), self.row(1), self.row(5)
        self.assertEqual(own["is_self"], 1)
        for key, expected in dict(erosion_curse=.3, fate_loom=.2, shield_hp=.25,
                                  max_shield_hp=.6, shield_piercer=1, shield_crash=.15,
                                  ability_FLASH=1, ultimate_TUNNEL=1).items():
            self.assertAlmostEqual(own[key], expected)
        self.assertAlmostEqual(ally["shield_hp"], .17)
        for key, expected in dict(hp=.4, max_hp=.65, shield_hp=.07, max_shield_hp=.8,
                                  ability_seal_remaining=.4, fate_loom_remaining=.3,
                                  fate_max_hp_lost=.35, ultimate_points=.2, status_known=1,
                                  can_move=0, can_use_ability=0, ability_SMOKE=1).items():
            self.assertAlmostEqual(enemy[key], expected)
        self.enemy.shield_hp = 0
        self.assertFalse(np.array_equal(before, self.encode()))
        self.assertEqual(before.shape, (1475,))
        self.assertEqual(OBS_DIM, 1475)
        self.assertTrue(np.isfinite(before).all())

    def test_unseen_live_state_cannot_change_any_input_but_base_traits_are_public(self):
        self.enemy.position_known = False
        self.state["enemy_roster"][0]["stats"].update(erosion_curse=5, fate_loom=7,
                                                      shield_hp=80, shield_piercer=True, shield_crash=20)
        before = self.encode()
        row = self.row(5)
        self.assertEqual(row["status_known"], 0)
        self.assertAlmostEqual(row["base_erosion_curse"], .5)
        self.assertAlmostEqual(row["base_fate_loom"], .7)
        self.assertAlmostEqual(row["base_shield_hp"], .8)
        self.assertEqual(row["base_shield_piercer"], 1)
        start = UNIT_FIELDS.index("status_known")
        self.assertTrue(np.all(self.rows()[5, start:] == 0))
        for key, _ in LIVE_STATS:
            setattr(self.enemy, key, 999)
        self.enemy.ability_name, self.enemy.ultimate_name = "ASH", "BALEMOON"
        self.enemy.active_awakenings = {"private": {"ticks_remaining": 50}}
        self.enemy.pos = [2, 2]
        np.testing.assert_array_equal(before, self.encode())
        self.state["enemy_roster"][0]["stats"]["fate_loom"] = 1
        self.assertFalse(np.array_equal(before, self.encode()))

    def test_unknown_enemy_live_attributes_are_never_read(self):
        forbidden = {key for key, _ in LIVE_STATS} | {"ability_name", "ultimate_name", "active_awakenings"}
        class Hidden:
            name, team, base_name, position_known, is_alive = "Demon1", "D", "Demon1", False, True
            def __getattribute__(self, key):
                if key in forbidden:
                    raise AssertionError("hidden live field read: " + key)
                return object.__getattribute__(self, key)
        self.state["chars"][-1] = Hidden()
        self.assertEqual(self.row(5)["status_known"], 0)

    def test_roster_name_and_base_name_only_entries_load_catalog_traits(self):
        self.state["enemy_roster"] = [dict(base_name="Demon1"), "Ethan"]
        self.state["chars"] = [self.own]
        self.assertEqual(self.row(5)["ability_SMOKE"], 1)
        self.assertEqual(self.row(6)["ability_RECON"], 1)
        self.assertEqual(self.row(5)["status_known"], 0)

    def test_roster_slots_survive_missing_units_and_are_order_independent(self):
        before = self.rows()
        self.state["chars"].reverse()
        self.state["ally_roster"].reverse()
        np.testing.assert_array_equal(before, self.rows())
        self.state["chars"] = [self.own]
        rows = self.rows()
        self.assertEqual(rows[1, UNIT_FIELDS.index("present")], 1)
        self.assertEqual(rows[5, UNIT_FIELDS.index("present")], 1)
        self.assertEqual(rows[1, UNIT_FIELDS.index("status_known")], 0)
        self.assertTrue(np.all(rows[2:5] == 0))
        self.assertTrue(np.all(rows[6:] == 0))

    def test_each_block_reason_and_ability_inventory_change_own_capabilities(self):
        normal = action_capabilities(self.own, self.state)
        self.assertTrue(all(normal[key] for key in ("can_move", "can_use_ability", "can_use_ultimate")))
        for field, expected in (
            ("ability_seal_remaining", (True, False, False)),
            ("fate_loom_remaining", (False, True, True)),
            ("movement_disabled_remaining", (False, False, False)),
            ("electric_remaining", (False, True, True)),
        ):
            with self.subTest(field=field):
                setattr(self.own, field, 3)
                values = dict(zip(SELF_FIELDS, self_capabilities(self.own, self.state)))
                self.assertEqual(tuple(bool(values[k]) for k in ("can_move", "can_use_ability", "can_use_ultimate")), expected)
                self.assertAlmostEqual(values[field], .3)
                setattr(self.own, field, 0)
        self.own.flash_charges = 0
        self.assertFalse(action_capabilities(self.own, self.state)["can_use_ability"])
        self.own.ultimate_points = 0
        self.assertFalse(action_capabilities(self.own, self.state)["can_use_ultimate"])

    def test_ramp_final_tick_matches_engine_and_own_escape_channel_blocks_actions(self):
        self.own.electric_remaining = 1
        self.own.electric_applied_tick = 6
        self.assertTrue(action_capabilities(self.own, self.state)["can_move"])
        self.state["battle_tick"] = 9
        self.assertFalse(action_capabilities(self.own, self.state)["can_move"])
        self.own.electric_remaining = 0
        self.state["ally_escape_portals"] = [{"owner": self.own.name, "remaining_ticks": 4}]
        values = dict(zip(SELF_FIELDS, self_capabilities(self.own, self.state)))
        self.assertEqual((values["can_move"], values["can_use_ability"], values["can_turn"]), (0, 0, 0))
        self.assertAlmostEqual(values["escape_remaining"], .4)
        self.assertEqual(values["escape_channeling"], 1)
        self.state["visible_escape_channel_owners"] = [self.enemy.name]
        self.assertEqual(self.row(5)["can_move"], 0)
        self.assertEqual(self.row(5)["can_use_ability"], 0)

    def test_masks_remove_blocked_moves_casts_and_keep_rooted_plant_legal(self):
        self.own.fate_loom_remaining = 3
        choices, mask = candidates(self.own, self.state, self.tactics, self.proposal)
        self.assertFalse(mask[4:8].any())
        self.assertTrue(mask[8:16].all())
        self.assertEqual(choices[1][0], self.own.pos)
        self.own.ability_seal_remaining = 3
        self.state.update(is_planted=True, planted_pos=(5, 5), defender_defuse_info={"Demon1": (1, 6)})
        cast = (list(self.own.pos), {"ability": "FLASH", "target": (5, 8)})
        choices, mask = candidates(self.own, self.state, self.tactics, cast)
        self.assertFalse(mask[2])
        self.assertFalse(mask[16:].any())
        self.assertTrue(mask[1])
        self.state["is_planted"] = False
        self.own.has_spike = True
        self.state["grid"][5, 5] = 2
        _, mask = candidates(self.own, self.state, self.tactics, ([5, 5], "PLANT"))
        self.assertEqual(np.flatnonzero(mask).tolist(), [3])

    def test_teacher_records_legal_hold_instead_of_a_sealed_cast(self):
        controller = LearnedAttackerController.__new__(LearnedAttackerController)
        controller.tactics = self.tactics
        controller.encoder = ObservationEncoder()
        controller.policy = None
        controller.recorder = Recorder()
        self.own.ability_seal_remaining = 2
        cast = (list(self.own.pos), {"ability": "FLASH", "target": (5, 8)})
        with patch.object(GhostChampionsV2AttackerController, "decide_move", return_value=cast):
            result = controller.decide_move(self.own, self.state)
        row = controller.recorder.rows[0]
        self.assertEqual(result, ([5, 5], "HOLD"))
        self.assertEqual(row["action"], 1)
        self.assertEqual(row["teacher"], 1)
        self.assertTrue(row["mask"][row["teacher"]])
        self.assertEqual(row["obs"].shape, (OBS_DIM,))

    def test_forced_facing_disables_rotation_without_disabling_movement(self):
        self.own.facing_forced_this_tick = True
        flags = action_capabilities(self.own, self.state)
        self.assertTrue(flags["can_move"])
        self.assertFalse(flags["can_turn"])
        _, mask = candidates(self.own, self.state, self.tactics, self.proposal)
        self.assertTrue(mask[0])
        self.assertTrue(mask[4:8].any())
        self.assertFalse(mask[8:16].any())

    def test_dead_serenade_is_ready_until_sealed(self):
        self.ally.ultimate_name = "SERENADE"
        self.ally.ultimate_cost = self.ally.ultimate_points = 4
        self.ally.is_alive = False
        self.assertTrue(action_capabilities(self.ally, self.state)["can_use_ultimate"])
        self.ally.ability_seal_remaining = 1
        self.assertFalse(action_capabilities(self.ally, self.state)["can_use_ultimate"])

    def test_v3_model_is_rejected_and_explicit_viewing_upgrade_preserves_predictions(self):
        schema = dict(version=3, dim=811, blocks=LEGACY_V3_BLOCKS,
                      enemy_roster_version=1, public_effect_version=1)
        model = ResidualPolicy(hidden=16)
        weights = model.state_dict()
        weights["features.0.weight"] = weights["features.0.weight"][:, :811].clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v3.pt"
            torch.save(dict(schema=schema, schema_hash=hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                            actions=ACTIONS, hidden=16, model_state_dict=weights), path)
            with self.assertRaisesRegex(ValueError, "v4"):
                ResidualPolicy.load(path)
            upgraded, metadata = upgraded_policy(path)
            x = torch.randn(3, OBS_DIM)
            expected = torch.tanh(torch.nn.functional.linear(x[:, :811], weights["features.0.weight"], weights["features.0.bias"]))
            expected = model.features[2:](expected)
            torch.testing.assert_close(upgraded.features(x), expected)
            self.assertFalse(metadata["unit_status_trained"])
            self.assertFalse(metadata["verified"])


if __name__ == "__main__":
    unittest.main()
