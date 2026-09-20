import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_attacker_gc_real_curriculum as trainer


class CarryRoutePriorityIntegrationTest(unittest.TestCase):
    def test_carry_route_priority_respects_env_overrides(self):
        os.environ["GC_TRAINING_MODE"] = "carry_route_priority"
        os.environ["GC_CARRY_SITE_ENTRY_BONUS"] = "12.5"
        os.environ["GC_PLANT_PROGRESS_BONUS"] = "7.0"
        os.environ["GC_SITE_HOLD_BONUS"] = "3.5"
        os.environ["GC_CARRY_NO_ENTRY_PENALTY"] = "9.0"
        os.environ["GC_TIMEOUT_PENALTY"] = "6.0"
        os.environ["GC_SPIKE_DROP_PENALTY"] = "4.5"
        os.environ["GC_ROUTE_STALL_PENALTY"] = "5.5"
        os.environ["GC_TEAM_COLLISION_PENALTY"] = "2.5"
        os.environ["GC_GUARD_SUPPRESSION_FACTOR"] = "0.25"

        cfg = trainer.current_carry_route_priority()

        self.assertEqual(cfg.carry_site_entry_bonus, 12.5)
        self.assertEqual(cfg.plant_progress_bonus, 7.0)
        self.assertEqual(cfg.site_hold_bonus, 3.5)
        self.assertEqual(cfg.carry_no_entry_penalty, 9.0)
        self.assertEqual(cfg.timeout_penalty, 6.0)
        self.assertEqual(cfg.spike_drop_penalty, 4.5)
        self.assertEqual(cfg.route_stall_penalty, 5.5)
        self.assertEqual(cfg.team_collision_penalty, 2.5)
        self.assertEqual(cfg.guard_suppression_factor, 0.25)

        reward = cfg.reward(
            carry_site_entry=True,
            plant_progress=0.5,
            site_hold=True,
            carry_no_entry=True,
            timeout=True,
            spike_drop=True,
            route_stall=True,
            team_collision=True,
            escort_support=True,
            guard_active=True,
        )

        self.assertLess(reward, 0)


if __name__ == "__main__":
    unittest.main()
