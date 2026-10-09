"""高速移動が分岐・目的地を通り越さず、往復から復帰することを検証。"""

import unittest
from types import SimpleNamespace

from controllers import BaseController
from test_ultimate_system import UltimateTestGame, make_character


class FastMovementTests(unittest.TestCase):
    def fixture(self, name="A-Train", speed=3, team="A"):
        game = UltimateTestGame()
        actor = make_character(name, team, (4, 1))
        actor.move_steps_per_tick = speed
        game.chars = [actor]
        return game, actor

    def move(self, game, actor, direction=(0, 1), payload=None):
        def decide(char, state):
            pos = [char.pos[i] + direction[i] for i in (0, 1)]
            return (pos, "MOVE", payload) if payload else pos
        controller = SimpleNamespace(decide_move=decide)
        game.attacker_controller = game.defender_controller = controller
        game.move_character(actor)

    def navigate(self, game, actor, goal):
        navigator = BaseController()
        def decide(char, state):
            return navigator.move_towards_target(
                char.pos, goal, state["grid"], chars=state["chars"], moving_char=char
            )
        game.attacker_controller = game.defender_controller = SimpleNamespace(decide_move=decide)
        game.move_character(actor)

    def test_all_three_characters_keep_speed_on_straights(self):
        for name, speed in (("Homelander", 2), ("A-Train", 3), ("おもこ", 3)):
            for team in ("A", "D"):
                with self.subTest(name=name, team=team):
                    game, actor = self.fixture(name, speed, team)
                    self.move(game, actor)
                    self.assertEqual(actor.pos, [4, 1 + speed])

    def test_stop_at_branch_even_if_corridor_continues(self):
        game, actor = self.fixture()
        game.grid[:] = 1
        game.grid[4, :] = 0
        game.grid[2:4, 2] = 0
        self.move(game, actor)
        self.assertEqual(actor.pos, [4, 2])
        self.move(game, actor, (-1, 0))
        self.assertEqual(actor.pos, [2, 2])

    def test_path_stops_at_turn_in_open_space(self):
        game, actor = self.fixture()
        actor.pos = [1, 1]
        self.navigate(game, actor, (2, 3))
        self.assertEqual(actor.pos, [2, 1])
        self.navigate(game, actor, (2, 3))
        self.assertEqual(actor.pos, [2, 3])

    def test_path_stops_exactly_at_goal(self):
        game, actor = self.fixture()
        self.navigate(game, actor, (4, 3))
        self.assertEqual(actor.pos, [4, 3])
        self.navigate(game, actor, (4, 3))
        self.assertEqual(actor.pos, [4, 3])

    def test_reversal_reaches_previously_skipped_goal(self):
        for name, speed in (("Homelander", 2), ("A-Train", 3), ("おもこ", 3)):
            with self.subTest(name=name):
                game, actor = self.fixture(name, speed)
                goal = (4, 2)
                def decide(char, state):
                    delta = (goal[1] > char.pos[1]) - (goal[1] < char.pos[1])
                    return [4, char.pos[1] + delta]
                game.attacker_controller = SimpleNamespace(decide_move=decide)
                for _ in range(5):
                    game.move_character(actor)
                self.assertEqual(tuple(actor.pos), goal)

    def test_full_speed_returns_after_slow_period(self):
        game, actor = self.fixture()
        actor.pos = [4, 8]
        self.move(game, actor, (0, -1))
        self.move(game, actor)
        self.assertEqual(actor.pos, [4, 6])
        for _ in range(3):
            self.move(game, actor)
        self.assertEqual(actor.pos, [4, 9])
        # 軸を変えて、過去の位置に戻らない直線で速度の復帰を確認する。
        self.move(game, actor, (-1, 0))
        self.assertEqual(actor.pos, [1, 9])

    def test_wall_and_occupancy_stop_each_extra_step(self):
        for obstacle in ("wall", "character"):
            with self.subTest(obstacle=obstacle):
                game, actor = self.fixture()
                if obstacle == "wall":
                    game.grid[4, 3] = 1
                else:
                    game.chars.append(make_character("Leo", "D", (4, 3)))
                game._build_occupancy_counts()
                self.move(game, actor)
                self.assertEqual(actor.pos, [4, 2])
                self.assertFalse(game._is_position_occupied(actor, (4, 1), (4, 2)))
                game._clear_occupancy_counts()

    def test_revisiting_recent_endpoint_slows_even_without_reversal(self):
        game, actor = self.fixture()
        actor.pos = [4, 4]
        for direction in ((0, 1), (-1, 0), (0, -1)):
            self.move(game, actor, direction)
        self.assertEqual(actor.pos, [1, 4])
        self.move(game, actor, (1, 0))
        self.assertEqual(actor.pos, [2, 4])

    def test_teleport_clears_slowdown_history(self):
        game, actor = self.fixture()
        self.move(game, actor)
        self.move(game, actor, (0, -1))
        actor.pos = [1, 6]
        self.move(game, actor)
        self.assertEqual(actor.pos, [1, 9])

    def test_explicit_step_limit_is_preserved(self):
        game, actor = self.fixture()
        self.move(game, actor, payload={"move_step_limit": 1})
        self.assertEqual(actor.pos, [4, 2])

    def test_orb_and_spike_are_not_skipped(self):
        for objective in ("orb", "spike", "plant", "defuse"):
            with self.subTest(objective=objective):
                game, actor = self.fixture(team="D" if objective == "defuse" else "A")
                if objective == "orb":
                    game.available_orbs = {(4, 2)}
                elif objective == "spike":
                    game.spike_pos = (4, 2)
                elif objective == "plant":
                    actor.has_spike = True
                    game.grid[4, 2] = 2
                else:
                    game.is_planted = True
                    game.planted_pos = (4, 3)
                self.move(game, actor)
                self.assertEqual(actor.pos, [4, 2])

    def test_navigation_limit_does_not_leak_to_next_decision(self):
        game, actor = self.fixture()
        self.navigate(game, actor, (4, 2))
        self.move(game, actor)
        self.assertEqual(actor.pos, [4, 5])


if __name__ == "__main__":
    unittest.main()
