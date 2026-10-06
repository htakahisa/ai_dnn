"""Guard casts must execute in the game, including with existing checkpoints."""

import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from concon_v1.co1_guard_battle_training import GuardBattleEnv
from concon_v1.co1_guard_common import ABILITIES, LEGACY_ACTION_DIM, ULTIMATE_ACTION
from concon_v1.co1_learn_guard import ConconGuardController
from concon_v1.co1_train_guard import make_checkpoint, OPPONENTS, START_MODES


class GuardUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def environment(self, site="L"):
        env = GuardBattleEnv(map_name=site, opponents=["touyama_v2"])
        env.reset("hold", attacker_count=5, defender_count=5)
        for char in env.attackers:
            char.smoke_charges = char.flash_charges = char.recon_charges = 0
            char.ultimate_points = 0
        return env

    def state(self, env):
        return dict(grid=env.game.grid, chars=env.attackers, is_planted=True,
                    planted_pos=env.game.planted_pos, battle_tick=0, detonate_timer=50)

    def test_each_ability_executes_without_additional_training(self):
        for site in ("L", "R"):
            for ability in ABILITIES:
                with self.subTest(site=site, ability=ability):
                    env = self.environment(site)
                    char = next(char for char in env.attackers if char.ability_name == ability)
                    attribute = ability.lower() + "_charges"
                    setattr(char, attribute, 1)
                    state = self.state(env)
                    _, mask, _ = env.controller.policy_inputs(char, state)
                    self.assertFalse(mask[:40].any())
                    _, payload = env.controller.decide_move(char, state)
                    self.assertEqual(payload["ability"], ability)
                    self.assertTrue(env.game.execute_ai_ability(char, payload))
                    self.assertEqual(getattr(char, attribute), 0)
                    _, mask, _ = env.controller.policy_inputs(char, state)
                    self.assertTrue(mask[:40].any())

    def test_every_roster_ultimate_executes_on_both_maps(self):
        for site in ("L", "R"):
            env = self.environment(site)
            for char in env.attackers:
                with self.subTest(site=site, ultimate=char.ultimate_name):
                    char.ultimate_points = char.ultimate_cost
                    state = self.state(env)
                    _, mask, _ = env.controller.policy_inputs(char, state)
                    self.assertTrue(mask[ULTIMATE_ACTION:].any())
                    _, payload = env.controller.decide_move(char, state)
                    self.assertEqual(payload["ultimate"], char.ultimate_name)
                    self.assertTrue(env.game.execute_ai_ultimate(char, payload))
                    self.assertEqual(char.ultimate_points, 0)

    def test_ability_and_ultimate_remain_candidates_at_same_priority(self):
        env = self.environment()
        char = env.attackers[2]  # MONITOR does not move the caster.
        char.recon_charges = 1
        char.ultimate_points = char.ultimate_cost
        _, mask, _ = env.controller.policy_inputs(char, self.state(env))
        self.assertFalse(mask[:40].any())
        self.assertTrue(mask[40:ULTIMATE_ACTION].any())
        self.assertTrue(mask[ULTIMATE_ACTION:].any())
        with torch.no_grad():
            for parameter in env.model.parameters():
                parameter.zero_()
            env.model.head[-1].bias[ULTIMATE_ACTION:] = 100
        _, payload = env.controller.decide_move(char, self.state(env))
        self.assertEqual(payload["ultimate"], "MONITOR")
        env.game.execute_ai_ultimate(char, payload)
        _, payload = env.controller.decide_move(char, self.state(env))
        self.assertEqual(payload["ability"], "RECON")

    def test_legacy_output_weights_preserved_when_loading_and_casting(self):
        env = self.environment()
        checkpoint = make_checkpoint(env.model, env.scenario, 123, OPPONENTS, START_MODES)
        checkpoint["n_actions"] = LEGACY_ACTION_DIM
        for key in ("head.2.weight", "head.2.bias"):
            checkpoint["model_state_dict"][key] = checkpoint["model_state_dict"][key][:LEGACY_ACTION_DIM]
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        controller = ConconGuardController(checkpoint_bytes=buffer.getvalue())
        for key, value in checkpoint["model_state_dict"].items():
            loaded = controller.model.state_dict()[key]
            self.assertTrue(torch.equal(value, loaded[:LEGACY_ACTION_DIM] if key.startswith("head.2.") else loaded))
        char = env.attackers[2]
        char.ultimate_points = char.ultimate_cost
        _, payload = controller.decide_move(char, self.state(env))
        self.assertEqual(payload["ultimate"], "MONITOR")

    def test_actual_tick_casts_with_full_exploration_and_records_learning(self):
        env = self.environment()
        for char in env.attackers:
            if char.ability_name in ABILITIES:
                setattr(char, char.ability_name.lower() + "_charges", 1)
        env.step(epsilon=1.0)
        self.assertEqual(len(env.decisions), 5)
        for char in env.attackers:
            if char.ability_name not in ABILITIES:  # HUNT is a passive on-kill heal.
                continue
            self.assertEqual(getattr(char, char.ability_name.lower() + "_charges"), 0)
            self.assertGreaterEqual(env.pending[env.indices[char.name]]["action"], 40)


if __name__ == "__main__":
    unittest.main()
