"""Regression checks for one-round tactical simulation playback."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from game_core import Character, get_all_character_names
from tactical_simulator import (
    TacticalSimulator,
    create_sample_retake_scenario,
    get_character_resource_profile,
)


class TacticalSimulatorPlaybackTest(unittest.TestCase):
    def test_editor_limits_match_character_resources(self):
        for name in get_all_character_names():
            char = Character(name, "A", (7, 3), "white", "blue")
            profile = get_character_resource_profile(name)
            self.assertEqual(profile["ability"], char.ability_name, name)
            self.assertEqual(profile["ultimate"], char.ultimate_name, name)
            self.assertEqual(profile["ultimate_cost"], char.ultimate_cost, name)
            self.assertEqual(
                profile["max_charges"],
                char.smoke_charges + char.flash_charges + char.recon_charges,
                name,
            )

    def test_player_resources_are_initialized_and_can_be_spent(self):
        scenario = create_sample_retake_scenario()
        scenario.planted_pos = (7, 4)
        scenario.attackers = [{
            "name": "Xdll", "pos": (7, 3), "facing": "E",
            "ability_charges": 1, "ultimate_points": 8,
        }]
        scenario.defenders = [{
            "name": "Absol", "pos": (9, 3), "facing": "N",
            "ability_charges": 0, "ultimate_points": 5,
        }]
        with redirect_stdout(StringIO()):
            simulator = TacticalSimulator(
                scenario, attacker_ai_name="default", defender_ai_name="default"
            )

        attacker, defender = simulator.chars
        self.assertEqual(get_character_resource_profile("Xdll")["max_charges"], 2)
        self.assertEqual((attacker.ability_name, attacker.recon_charges), ("RECON", 1))
        self.assertEqual((attacker.ultimate_name, attacker.ultimate_points), ("MONITOR", 8))
        self.assertEqual((defender.flash_charges, defender.ultimate_points), (0, 5))
        self.assertTrue(simulator.execute_ai_ability(
            attacker, {"ability": "RECON", "target": (8, 3)}
        ))
        self.assertEqual(attacker.recon_charges, 0)
        self.assertTrue(simulator.execute_ai_ultimate(attacker, {"ultimate": "MONITOR"}))
        self.assertEqual(attacker.ultimate_points, 0)

    def test_gc_retake_with_partial_rosters_skips_defender_setup(self):
        scenario = create_sample_retake_scenario()
        scenario.attackers = [{"name": "Absol", "pos": (14, 9), "facing": "NW"}]
        scenario.defenders = [{"name": "Xdll", "pos": (12, 8), "facing": "E"}]
        scenario.detonate_timer = 3
        with redirect_stdout(StringIO()):
            simulator = TacticalSimulator(
                scenario,
                attacker_ai_name="ghost_champions_v1",
                defender_ai_name="ghost_champions_v1",
            )
            self.assertFalse(simulator.defender_setup_phase.active)
            result = simulator.run({}, max_ticks=5)

        self.assertEqual(result.winner, "attackers")
        self.assertEqual(result.total_ticks, 3)
        self.assertEqual(simulator.current_round, 1)

    def test_real_battle_stops_when_spike_detonates(self):
        scenario = create_sample_retake_scenario()
        scenario.detonate_timer = 1
        simulator = TacticalSimulator(
            scenario, attacker_ai_name="default", defender_ai_name="default"
        )

        result = simulator.run({}, max_ticks=5)

        self.assertEqual(result.winner, "attackers")
        self.assertEqual(result.total_ticks, 1)
        self.assertEqual(simulator.current_round, 1)
        self.assertTrue(simulator.round_over)
        self.assertFalse(simulator.simulation_active)
        self.assertTrue(result.replay_frames[-1]["round_over"])

    def test_run_stops_on_terminal_tick_without_starting_another_round(self):
        simulator = TacticalSimulator.__new__(TacticalSimulator)
        simulator.scenario = SimpleNamespace(scenario_name="one_round")
        simulator.simulation_active = True
        simulator.round_over = False
        simulator.match_over = False
        simulator.current_round = 1
        simulator.total_ticks = 0
        simulator.battle_tick = 0
        simulator.attacker_wins = 0
        simulator.defender_wins = 0
        simulator.winner = None
        simulator.replay_frames = []
        simulator._record_replay_frame = lambda: simulator.replay_frames.append(
            (simulator.current_round, simulator.round_over)
        )
        simulator._apply_player_actions = lambda handlers: None
        simulator._build_occupancy_counts = lambda: None
        simulator._clear_occupancy_counts = lambda: None
        simulator._move_order = lambda: []
        simulator._generate_player_stats = lambda: []

        def end_round():
            simulator.battle_tick += 1
            simulator.attacker_wins += 1
            simulator.round_over = True
            simulator._record_replay_frame()  # check_match_winner's terminal snapshot
            simulator.current_round += 1
            simulator.init_round()

        simulator.process_battle = end_round
        observed_ticks = []
        result = simulator.run({}, max_ticks=10, on_tick=lambda sim: observed_ticks.append(sim.total_ticks))

        self.assertEqual(result.winner, "attackers")
        self.assertEqual(result.total_ticks, 1)
        self.assertEqual(simulator.battle_tick, 1)
        self.assertEqual(simulator.current_round, 1)
        self.assertTrue(simulator.round_over)
        self.assertTrue(simulator.match_over)
        self.assertFalse(simulator.simulation_active)
        self.assertEqual(observed_ticks, [1])
        self.assertEqual(result.replay_frames, [(1, False), (1, True)])
        self.assertFalse(simulator.step({}))
        self.assertEqual(simulator.total_ticks, 1)


if __name__ == "__main__":
    unittest.main()
