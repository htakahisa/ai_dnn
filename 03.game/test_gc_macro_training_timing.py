"""Plant timing, horizon and shared Fake-route regression tests."""
import unittest
from unittest.mock import patch
from test_gc_macro_coordination import training, unit


class MacroTimingTests(unittest.TestCase):
    def plant_env(self):
        env = training.MacroEnv.__new__(training.MacroEnv)
        pos = tuple(training.PLANT_CELLS[0])
        carrier = unit("Absol", pos, True)
        env.attackers = [carrier]
        env.tick = env.macro_step = 0
        env.current_strategy = "A_RUSH"
        env._default_a_scout_name = env._default_b_scout_name = None
        env.target_site = training.side_of_pos(pos)
        env.targets = {carrier.name: pos}
        env.assignment = {carrier.name: (env.target_site, "SITE", "MAIN")}
        env.done = env.success = env.planted = False
        env._reset_plant_progress()
        return env, carrier

    def action_tick(self, env):
        env._move_attackers_one_tick()
        env.tick += 1
        return env._check_plant()

    def test_four_stationary_ticks_required(self):
        env, carrier = self.plant_env()
        pos = carrier.pos
        for i in range(training.PLANT_REQUIRED_TICKS - 1):
            self.assertFalse(self.action_tick(env))
            self.assertEqual(env._plant_progress, i + 1)
            self.assertEqual(carrier.pos, pos)
            self.assertFalse(env._check_plant())
            self.assertEqual(env._plant_progress, i + 1)
        self.assertTrue(self.action_tick(env))

    def test_arrival_is_not_a_plant_action(self):
        env, carrier = self.plant_env()
        goal = carrier.pos
        carrier.pos = next((goal[0] + dr, goal[1] + dc)
                           for dr, dc in training.CARDINAL
                           if training.walkable((goal[0] + dr, goal[1] + dc))
                           and (goal[0] + dr, goal[1] + dc) not in training.PLANT_CELLS)
        with patch.object(training, "target_for_side", return_value=goal):
            self.assertFalse(self.action_tick(env))
        self.assertEqual(carrier.pos, goal)
        self.assertEqual(env._plant_progress, 0)
        for _ in range(3):
            self.assertFalse(self.action_tick(env))
        self.assertTrue(self.action_tick(env))

    def test_death_resets_progress(self):
        env, carrier = self.plant_env()
        self.action_tick(env)
        carrier.is_alive = False
        self.assertFalse(env._check_plant())
        self.assertEqual(env._plant_progress, 0)

    def test_departure_and_handoff_reset_progress(self):
        env, carrier = self.plant_env()
        self.action_tick(env)
        carrier.has_spike = False
        replacement = unit("cover", carrier.pos, True)
        env.attackers.append(replacement)
        env.tick += 1
        env._plant_action_holder = replacement.name
        self.assertFalse(env._check_plant())
        self.assertEqual(env._plant_progress, 1)
        replacement.pos = tuple(training.ATTACKER_SPAWNS[0])
        self.assertFalse(env._check_plant())
        self.assertEqual(env._plant_progress, 0)

    def test_last_tick_completion_but_not_late_completion(self):
        env, _ = self.plant_env()
        env.tick = training.ROUND_DURATION_TICKS - training.PLANT_REQUIRED_TICKS
        for _ in range(training.PLANT_REQUIRED_TICKS - 1):
            self.assertFalse(self.action_tick(env))
        self.assertTrue(self.action_tick(env))
        self.assertEqual(env.tick, training.ROUND_DURATION_TICKS)
        env, _ = self.plant_env()
        env.tick = training.ROUND_DURATION_TICKS - training.PLANT_REQUIRED_TICKS + 1
        for _ in range(training.PLANT_REQUIRED_TICKS):
            self.assertFalse(self.action_tick(env))

    def test_horizon_reaches_exactly_real_round_duration(self):
        env = training.MacroEnv()
        env.reset(forced_strategy="A_RUSH", forced_curriculum_mode="FREE")
        with patch.object(env, "_move_attackers_one_tick"), \
                patch.object(env, "_move_defenders_one_tick"), \
                patch.object(env, "_resolve_contact"), \
                patch.object(env, "_check_plant", return_value=False):
            for _ in range(training.MAX_MACRO_STEPS):
                _, _, done, _ = env.step(training.STRATEGY_TO_INDEX["A_RUSH"])
                if done:
                    break
        self.assertEqual(env.tick, training.ROUND_DURATION_TICKS)
        self.assertEqual(env.reason, "timeout")

    def test_fake_main_targets_close_stable_and_legal_after_handoff(self):
        for strategy in ("FAKE_A_TO_B", "FAKE_B_TO_A"):
            env = training.MacroEnv()
            env.reset(forced_strategy=strategy, forced_curriculum_mode="FREE")
            main = [a for a in env.attackers if a.name not in env._fake_group_names]
            self.assertEqual(len(env._fake_group_names), 2)
            self.assertEqual(len(main), 3)
            carrier = env._carrier()
            for phase in ("SELL", "ROTATE", "EXECUTE"):
                env._set_fake_phase_targets(phase)
                goals = [env.targets[a.name] for a in main]
                self.assertEqual(len(set(goals)), len(goals))
                self.assertTrue(all(training.nearest_distance(
                    p, [env.targets[carrier.name]]) <= 3 for p in goals))
                if phase != "SELL":
                    first = dict(env.targets)
                    env._set_fake_phase_targets(phase)
                    self.assertEqual(env.targets, first)
            self.assertIn(env.targets[carrier.name], training.PLANT_CELLS)
            carrier.is_alive = False
            replacement = next(a for a in main if a is not carrier)
            replacement.has_spike = True
            env._update_fake_option_phase()
            self.assertIn(env.targets[replacement.name], training.PLANT_CELLS)

    def test_full_episodes_terminate_with_finite_rewards(self):
        for strategy in ("DEFAULT", "A_SPLIT", "B_SPLIT", "FAKE_A_TO_B", "FAKE_B_TO_A"):
            env = training.MacroEnv()
            env.reset(forced_strategy=strategy, forced_curriculum_mode="FREE")
            for _ in range(training.MAX_MACRO_STEPS):
                obs, reward, done, _ = env.step(training.STRATEGY_TO_INDEX[strategy])
                self.assertTrue(training.np.isfinite(obs).all())
                self.assertTrue(training.np.isfinite(reward))
                self.assertLessEqual(env.tick, training.ROUND_DURATION_TICKS)
                if done:
                    break
            self.assertTrue(env.done)


if __name__ == "__main__":
    unittest.main()
