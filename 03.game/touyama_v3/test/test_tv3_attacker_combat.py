"""Combat causes, public decisions, actual shooting physics, and reward credits."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace as NS
import unittest

from touyama_v3.test.test_tv3_attacker_entry_utility import Room
from touyama_v3.tv3_attacker_combat import AttackerCombatCoach, CombatAdvice, effective_utility, plant_ready
from touyama_v3.tv3_attacker_combat_audit import CombatAudit
from touyama_v3.tv3_train_attacker_plant import terminal_bonus, apply_terminal_bonus, utility_effect_credit, best_rank
from touyama_v3.tv3_train_attacker_analysis import summarize


class CombatTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Room()
        self.coach = AttackerCombatCoach(self.scenario)
        self.ally = NS(name="ally", slot=0, alive=True, position=(5, 3), hp=100., max_hp=100.,
                       blind=0, facing="W", forced_facing=False, has_spike=True)
        self.snapshot = NS(tick=1, allies=(self.ally,), sightings=(), enemies=(NS(enemy_id=0, alive=True, name="enemy"),),
                           visible_cells=(), smoke_cells=(), effects=(), spike_dropped=None)

    def test_choose_locally_shootable_enemy_instead_of_first_team_sighting(self):
        self.scenario.grid[5, 2] = 1
        self.snapshot.sightings = (NS(enemy_id=0, position=(5, 1)), NS(enemy_id=1, position=(5, 6)))
        advice = self.coach.advise(self.snapshot, self.ally, (5, 4), (5, 7))
        self.assertEqual(advice.target, (5, 6))
        self.assertEqual(advice.position, self.ally.position)
        self.assertEqual(advice.facing, "E")
        self.assertEqual(advice.reason, "stop_shoot")

    def test_forced_facing_is_honored_and_fresh_memory_is_used(self):
        self.ally.forced_facing = True
        advice = self.coach.advise(self.snapshot, self.ally, (5, 4), (5, 7), {0: ((5, 6), 0, (0, 0))})
        self.assertEqual(advice.reason, "stop_shoot")
        self.assertEqual(advice.facing, "W")
        self.snapshot.visible_cells = ((5, 6),)
        advice = self.coach.advise(self.snapshot, self.ally, (5, 4), (5, 7), {0: ((5, 6), 0, (0, 0))})
        self.assertEqual(advice.reason, "advance")  # A visible empty last-seen cell is not an enemy.

    def test_blind_actor_retreats_to_public_cover_without_omniscience(self):
        self.scenario.grid[5, 4] = 1
        self.ally.position, self.ally.blind = (4, 3), 5
        self.snapshot.sightings = (NS(enemy_id=0, position=(5, 6)),)
        advice = self.coach.advise(self.snapshot, self.ally, (4, 4), (5, 6))
        self.assertEqual(advice.reason, "cover_retreat")
        self.assertNotEqual(advice.position, (4, 4))

    def test_seven_tick_old_sighting_still_prevents_running_into_known_fire(self):
        self.snapshot.tick = 8
        advice = self.coach.advise(self.snapshot, self.ally, (5, 4), (5, 7), {0: ((5, 6), 1, (0, 0))})
        self.assertEqual(advice.reason, "stop_shoot")
        self.assertEqual(advice.position, self.ally.position)
        self.snapshot.tick = 22
        self.assertEqual(self.coach.advise(self.snapshot, self.ally, (5, 4), (5, 7),
                                         {0: ((5, 6), 1, (0, 0))}).reason, "advance")

    def test_isolated_corner_peek_waits_for_support_without_waiting_forever(self):
        self.scenario.grid[5, 4] = 1
        self.snapshot.sightings = (NS(enemy_id=0, position=(5, 6)),)
        first = self.coach.advise(self.snapshot, self.ally, (4, 3), (5, 6))
        self.assertEqual(first.reason, "assemble")
        self.assertEqual(first.position, (5, 3))
        self.snapshot.tick = 8
        after = self.coach.advise(self.snapshot, self.ally, (4, 3), (5, 6))
        self.assertEqual(after.position, (4, 3))

    def test_teammate_facing_away_does_not_count_as_ready_support(self):
        self.scenario.grid[5, 4] = 1
        self.snapshot.sightings = (NS(enemy_id=0, position=(5, 6)),)
        buddy = NS(slot=1, alive=True, position=(6, 3), blind=0, facing="W")
        self.snapshot.allies = (self.ally, buddy)
        advice = self.coach.advise(self.snapshot, self.ally, (4, 3), (5, 6))
        self.assertEqual(advice.supporters, 0)
        self.assertEqual(advice.reason, "assemble")
        buddy.facing = "E"
        advice = self.coach.advise(self.snapshot, self.ally, (4, 3), (5, 6))
        self.assertEqual(advice.supporters, 1)
        self.assertEqual(advice.position, (4, 3))

    def test_planting_under_fire_requires_healthy_carrier_and_covering_allies(self):
        advice = CombatAdvice(self.ally.position, "E", "stop_shoot", (5, 6), 1, 1, 0, 0)
        self.assertFalse(plant_ready(self.ally, advice))
        advice.supporters = 1
        self.assertFalse(plant_ready(self.ally, advice))
        advice.supporters = 2
        self.assertTrue(plant_ready(self.ally, advice))
        self.ally.hp = 20.
        self.assertFalse(plant_ready(self.ally, advice))
        self.ally.hp, self.ally.blind = 100., 2
        self.assertFalse(plant_ready(self.ally, advice))

    def test_reactive_recon_endpoint_requires_actual_projected_coverage(self):
        self.snapshot.sightings = (NS(enemy_id=0, position=(2, 2)),)
        self.assertFalse(effective_utility(self.scenario, self.snapshot, self.ally,
                         {"ability": "RECON", "target": (5, 7)}))

    def test_actual_engine_moving_shot_has_lower_chance_than_stopped_shot(self):
        from test.test_ultimate_system import UltimateTestGame, make_character
        from game_core import SHOOT_INTERVAL_TICKS
        game = UltimateTestGame()
        own = make_character("Xdll", "A", (4, 4), 99)
        foe = make_character("Demon1", "D", (4, 7))
        own.facing, foe.facing = "E", "W"
        own.hp = foe.hp = 100000.
        own.accuracy, foe.accuracy = .9, 0.
        own.hs_rate = foe.hs_rate = foe.dodge_rate = 0.
        game.chars = [own, foe]
        chances = []
        for moving in (True, False):
            game.battle_tick = SHOOT_INTERVAL_TICKS
            own.moved_this_tick = moving
            own.stopped_after_move_this_tick = False
            own.shoot_timer = foe.shoot_timer = 0
            game._resolve_all_shots()
            shots = [s for s in game.last_shots if s["shooter"] is own]
            self.assertTrue(shots)
            chances.append(shots[0]["hit_chance"])
        self.assertGreater(chances[1], chances[0])

    def test_actual_engine_drone_targets_are_audited_without_character_facing(self):
        from abilities_los import MonitorDrone
        from test.test_ultimate_system import UltimateTestGame, make_character
        from game_core import SHOOT_INTERVAL_TICKS
        game = UltimateTestGame()
        own = make_character("Xdll", "A", (4, 4), 99)
        foe = make_character("Demon1", "D", (4, 7))
        own.facing, foe.facing = "E", "W"
        own.hp = foe.hp = 100000.
        own.accuracy = foe.accuracy = 1.
        own.hs_rate = foe.hs_rate = own.dodge_rate = foe.dodge_rate = 0.
        own_drone, foe_drone = MonitorDrone(own, 0), MonitorDrone(foe, 0)
        own_drone.pos, foe_drone.pos = [4, 6], [4, 5]
        game.chars, game.monitor_drones = [own, foe], [own_drone, foe_drone]
        self.ally.name, self.ally.position, self.ally.facing = own.name, tuple(own.pos), own.facing
        self.ally.charges, self.ally.ability_name = own.flash_charges, own.ability_name
        self.snapshot.enemies = (NS(enemy_id=0, alive=True, name=foe.name),)
        audit = CombatAudit()
        audit.before(self.snapshot, {own.name: (own.pos, {"facing": "E"})})
        game.battle_tick = SHOOT_INTERVAL_TICKS
        game._resolve_all_shots()
        self.assertEqual(len(game.last_shots), 2)
        self.assertTrue(all(shot["target"].is_ultimate_drone for shot in game.last_shots))
        audit.after(game)
        self.assertEqual(audit.counts["ally_shots"], 1)
        self.assertEqual(audit.counts["incoming_shots"], 0)
        self.assertFalse(audit.deaths)
        self.assertTrue(all(shot["target_facing"] is None and shot["target_kind"] == "drone"
                            for shot in audit.context[-1]["shots"]))

    def test_death_audit_identifies_unseen_rear_moving_and_unsupported(self):
        self.ally.charges, self.ally.ability_name = 1, "FLASH"
        before = NS(**vars(self.snapshot))
        own = NS(name="ally", team="A", pos=[5, 3], hp=0., is_alive=False,
                 moved_this_tick=True, facing="E", flash_charges=1)
        foe = NS(name="enemy", team="D", pos=[5, 1], hp=100., is_alive=True,
                 moved_this_tick=False, facing="E")
        game = NS(chars=[own, foe], last_shots=[{"shooter": foe, "target": own,
            "hit": True, "damage": 100., "hit_chance": .9}],
            check_cell_line_of_sight=lambda *a, **kw: True)
        audit = CombatAudit()
        audit.before(before, {"ally": ([5, 3], {"facing": "E"})})
        audit.after(game)
        flags = audit.report()["deaths"][0]["flags"]
        self.assertIn("source_unseen_before_action", flags)
        self.assertIn("source_outside_facing", flags)
        self.assertIn("moving_on_death_tick", flags)
        self.assertIn("no_supporting_ally_line", flags)

    def test_dead_player_does_not_get_later_success_and_survivors_are_rewarded(self):
        transitions = [[None, 0, -2., None, None, 1.], [None, 0, 0., None, None, 0.]]
        terminal = {"planted": True, "alive": 1}
        apply_terminal_bonus(transitions, {"dead": 0, "alive": 1}, {"dead": False, "alive": True}, terminal, 5, 0.)
        self.assertEqual(transitions[0][2], -2.)
        self.assertGreater(transitions[1][2], 0.)
        self.assertGreater(terminal_bonus({"planted": True, "alive": 4}, 5, 0.), terminal_bonus(terminal, 5, 0.))

    def test_viable_plants_rank_above_suicidal_plants_and_actual_effect_gets_credit(self):
        safe = {"plant_rate": .5, "viable_plant_rate": .5, "mean_alive": 3., "mean_damage": 200.,
                "ability_reserve_rate": .2, "mean_ability_uses": 2., "mean_plant_ticks": 50.}
        unsafe = {**safe, "plant_rate": .9, "viable_plant_rate": .1, "mean_alive": 1.}
        self.assertGreater(best_rank(safe), best_rank(unsafe))
        empty = {"ability": "RECON", "affected_enemies": [], "blocked_lines": 0}
        self.assertLess(utility_effect_credit(empty), 0.)
        self.assertGreater(utility_effect_credit({**empty, "affected_enemies": [1, 2]}), 0.)

    def test_alive_players_with_almost_no_hp_are_not_counted_as_viable_plants(self):
        r = {"planted": True, "alive": 3, "enemy_alive": 5, "initial_alive": 5, "initial_enemy_alive": 5,
             "initial_abilities": 2, "uses": 0, "remaining_abilities_alive": 2,
             "damage": 450., "tick": 30, "reason": "planted", "remaining_hp": 15., "initial_team_hp": 500.}
        metrics = summarize([r])
        self.assertEqual(metrics["plant_rate"], 1.)
        self.assertEqual(metrics["survivor_plant_rate"], 1.)
        self.assertEqual(metrics["viable_plant_rate"], 0.)
        self.assertEqual(summarize([{**r, "remaining_hp": 200.}])["viable_plant_rate"], 1.)


if __name__ == "__main__":
    unittest.main()
