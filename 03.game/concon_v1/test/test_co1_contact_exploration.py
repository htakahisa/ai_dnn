"""Real enemy contact must take priority over route exploration."""

import contextlib
import io
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from concon_v1 import co1_learn_attacker as learning
from concon_v1.co1_battle_training import BattleRouteEnv


class ContactExplorationTests(unittest.TestCase):
    def test_a2_a3_exploration_stops_and_actually_fires_on_contact(self):
        for map_name in ("A2", "A3"):
            with self.subTest(map=map_name), contextlib.redirect_stdout(io.StringIO()):
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                env = BattleRouteEnv(seed=0, opponents=["gc_v1"], map_name=map_name)
                contact_action = learning.preplant_contact_action
                stopped = {}
                stop_count = shot_count = 0

                def trace_contact(char, *args, **kwargs):
                    result = contact_action(char, *args, **kwargs)
                    if (isinstance(result, tuple) and isinstance(result[1], dict)
                            and "facing" in result[1]):
                        self.assertEqual(tuple(result[0]), tuple(char.pos))
                        stopped[char.name] = tuple(char.pos)
                    return result

                with patch.object(learning, "preplant_contact_action", side_effect=trace_contact):
                    while not env.done:
                        stopped.clear()
                        env.step(epsilon=1.0)
                        for name, position in stopped.items():
                            index = env.attacker_indices[name]
                            self.assertFalse(env.policy_action_applied[index],
                                             "Exploration overrode enemy-contact stopping")
                            self.assertEqual(tuple(env.attackers[index].pos), position)
                        stop_count += len(stopped)
                        for shot in env.game.last_shots:
                            if shot["shooter"].name in stopped:
                                self.assertFalse(shot["shooter"].moved_this_tick)
                                shot_count += 1
                self.assertGreater(stop_count, 0, "No enemy contact was exercised")
                self.assertGreater(shot_count, 0, "No actual shot while stopped was exercised")


if __name__ == "__main__":
    unittest.main()
