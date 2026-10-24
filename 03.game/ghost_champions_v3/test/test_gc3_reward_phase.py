"""Outcome priority, useful effects, causal credit and non-farmable shaping."""
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import random
import unittest
import numpy as np
import torch
from abilities_los import AbilityLosMixin
from ghost_champions_v3.gc3_observation import ARENA_SHAPE
from ghost_champions_v3.gc3_reward_phase import (
    PhaseRewards, potential_difference, WIN_REWARD, LOSS_REWARD, DAMAGE_DEALT_PER_HP,
    REJECTED_ABILITY_PENALTY, NEW_RECON_TARGET_REWARD, NEW_FLASH_TARGET_REWARD,
    SMOKE_PROTECTION_MAX_PER_CAST,
)
from ghost_champions_v3.gc3_rollout import Recorder
from ghost_champions_v3.gc3_training import normalize_launch_settings
from ghost_champions_v3.tools.gc3_training_metrics import record_training_metrics


def unit(name, team, point, facing):
    return SimpleNamespace(name=name, team=team, pos=list(point), facing=facing, is_alive=True,
                           hp=100., shield_hp=0., kills=0, plant_timer=0, defuse_timer=0,
                           blind_remaining=0, reveal_remaining=0, los_revealed=False,
                           sees_through_smoke=False, orb_collect_timer=0)


class Game(AbilityLosMixin):
    def __init__(self):
        self.grid = np.zeros(ARENA_SHAPE, dtype=np.int32)
        self.height, self.width = self.grid.shape
        self.chars = [unit("own", "A", (5, 5), "E"), unit("ally", "A", (6, 5), "E"),
                      unit("enemy", "D", (5, 9), "W"), unit("other", "D", (6, 9), "W")]
        self.planted_pos = (5, 8)
        self.in_defender_setup_phase = False
        self.smokes, self.last_shots = [], []
        self.analytics_tracker = SimpleNamespace(stats={}, record_contribution=lambda *args:None)
        self.cast_calls = 0

    def execute_ai_ability(self, owner, command):
        self.cast_calls += 1
        if command.get("reject"):
            return False
        if command.get("ability") == "DANCE":
            self.chars[1].hp = min(100., self.chars[1].hp+50)
        return True


class EntropyFixture:
    def evaluate_records(self, rows):
        return torch.zeros(len(rows)), torch.zeros(len(rows)), torch.zeros(len(rows))


class PhaseRewardTests(unittest.TestCase):
    def test_potential_rewards_telescope_even_when_episode_lengths_differ(self):
        gamma = .99
        for values in ([.2, .8, .2], [.2, .8, .1, .7, .2], [.2, .2, .2, .2]):
            rewards = [potential_difference(values[i], values[i+1], gamma, i==len(values)-2)
                       for i in range(len(values)-1)]
            self.assertAlmostEqual(sum(gamma**i*r for i,r in enumerate(rewards)), -values[0])
        self.assertEqual(potential_difference(.2, .9, 1., False)+potential_difference(.9, .2, 1., False), 0.)

    def test_terminal_success_and_failure_are_distinct_shared_rewards(self):
        for success, expected in [(True, WIN_REWARD), (False, LOSS_REWARD)]:
            game = Game()
            rewards = PhaseRewards(game, "postplant", .99)
            rewards.begin_tick()
            shared, personal = rewards.finish_tick(True, success)
            self.assertEqual(shared["outcome"], expected)
            for name in rewards.own:
                self.assertAlmostEqual(personal[name]["deployment"], -rewards.before["potential"][name]["deployment"])

    def test_actual_damage_caps_overkill_and_credits_the_shooter_only(self):
        game = Game(); own, ally, enemy, _ = game.chars
        enemy.hp = 20.
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick()
            game.analytics_tracker.record_contribution(own, enemy, 1, "damage")
            game.last_shots = [dict(shooter=own,target=enemy,hit=True,damage=160)]
            enemy.hp = 0.; enemy.is_alive = False; own.kills += 1
            _, personal = rewards.finish_tick(False, False)
        self.assertAlmostEqual(personal[own.name]["damage_dealt"], 20*DAMAGE_DEALT_PER_HP)
        self.assertNotIn("damage_dealt", personal[ally.name])
        self.assertGreater(personal[own.name]["kill"], 0.)

    def test_multiple_shooters_share_actual_loss_without_double_counting(self):
        game = Game(); own, ally, enemy, _ = game.chars
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick()
            for shooter, damage in [(own, 40), (ally, 20)]:
                game.analytics_tracker.record_contribution(shooter, enemy, 1, "damage")
                game.last_shots.append(dict(shooter=shooter,target=enemy,hit=True,damage=damage))
            enemy.hp = 30.
            _, personal = rewards.finish_tick(False, False)
        dealt = [personal[name]["damage_dealt"] for name in (own.name, ally.name)]
        self.assertAlmostEqual(sum(dealt), 70*DAMAGE_DEALT_PER_HP)
        self.assertAlmostEqual(dealt[0]/dealt[1], 2.)

    def test_damage_taken_and_death_are_personal_not_penalties_for_every_ally(self):
        game = Game(); own, ally, _, _ = game.chars
        rewards = PhaseRewards(game, "postplant", .99)
        rewards.begin_tick(); own.hp = 0.; own.is_alive = False
        _, personal = rewards.finish_tick(False, False)
        self.assertLess(personal[own.name]["damage_taken"], 0.)
        self.assertLess(personal[own.name]["death"], 0.)
        self.assertEqual(personal[ally.name]["damage_taken"], 0.)
        self.assertEqual(personal[ally.name]["death"], 0.)

    def test_recon_novelty_is_once_per_enemy_for_the_team(self):
        game = Game(); own, ally, enemy, _ = game.chars
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick()
            game.analytics_tracker.record_contribution(own, enemy, 1, "recon")
            game.analytics_tracker.record_contribution(ally, enemy, 1, "recon")
            _, personal = rewards.finish_tick(False, False)
            self.assertEqual(personal[own.name]["recon_new_target"], NEW_RECON_TARGET_REWARD)
            self.assertNotIn("recon_new_target", personal[ally.name])
            rewards.begin_tick()
            game.analytics_tracker.record_contribution(own, enemy, 2, "recon")
            _, personal = rewards.finish_tick(False, False)
            self.assertNotIn("recon_new_target", personal[own.name])

    def test_recon_of_an_already_revealed_enemy_does_not_pay(self):
        game = Game(); own, _, enemy, _ = game.chars
        enemy.reveal_remaining = 3
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick()
            game.analytics_tracker.record_contribution(own, enemy, 1, "recon")
            _, personal = rewards.finish_tick(False, False)
        self.assertNotIn("recon_new_target", personal[own.name])

    def test_flash_requires_new_effect_and_is_not_repeatable(self):
        game = Game(); own, _, enemy, _ = game.chars
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick(); enemy.blind_remaining = 10
            game.analytics_tracker.record_contribution(own, enemy, 1, "flash")
            _, personal = rewards.finish_tick(False, False)
            self.assertEqual(personal[own.name]["flash_effect"], NEW_FLASH_TARGET_REWARD)
            rewards.begin_tick()
            game.analytics_tracker.record_contribution(own, enemy, 2, "flash")
            _, personal = rewards.finish_tick(False, False)
            self.assertNotIn("flash_effect", personal[own.name])

    def test_empty_cast_has_no_positive_use_reward_and_rejected_cast_is_penalized(self):
        game = Game(); own = game.chars[0]
        original = game.execute_ai_ability
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick()
            self.assertTrue(game.execute_ai_ability(own, {"ability":"FLASH"}))
            self.assertFalse(game.execute_ai_ability(own, {"ability":"RECON", "reject":True}))
            _, personal = rewards.finish_tick(False, False)
        self.assertEqual(personal[own.name]["rejected_ability"], REJECTED_ABILITY_PENALTY)
        self.assertNotIn("flash_effect", personal[own.name])
        self.assertEqual(game.cast_calls, 2)
        self.assertEqual(game.execute_ai_ability, original)

    def test_healing_rewards_only_the_actual_hp_restored(self):
        game = Game(); own, ally, _, _ = game.chars
        ally.hp = 90.
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick()
            game.execute_ai_ability(own, {"ability":"DANCE"})
            game.execute_ai_ability(own, {"ability":"DANCE"})
            _, personal = rewards.finish_tick(False, False)
        self.assertAlmostEqual(personal[own.name]["healing"], .02)

    def test_smoke_rewards_real_protection_once_and_not_casting_in_empty_space(self):
        game = Game(); own, ally, enemy, _ = game.chars
        smoke = dict(owner=own.name, team="A", cells={(5,7),(6,7)},remaining_ticks=25)
        game.smokes.append(smoke)
        rewards = PhaseRewards(game, "postplant", .99)
        rewards.begin_tick(); _, personal = rewards.finish_tick(False, False)
        self.assertGreater(personal[own.name]["smoke_protection"], 0.)
        self.assertLessEqual(personal[own.name]["smoke_protection"], SMOKE_PROTECTION_MAX_PER_CAST)
        rewards.begin_tick(); _, personal = rewards.finish_tick(False, False)
        self.assertNotIn("smoke_protection", personal[own.name])
        game = Game(); game.smokes.append(dict(owner="own",cells={(20,20)},remaining_ticks=25))
        rewards = PhaseRewards(game, "postplant", .99)
        rewards.begin_tick(); _, personal = rewards.finish_tick(False, False)
        self.assertNotIn("smoke_protection", personal["own"])

    def test_smoke_does_not_get_credit_when_another_smoke_already_blocks_the_line(self):
        game = Game()
        game.smokes = [dict(owner="own",cells={(5,7),(6,7)},remaining_ticks=25),
                       dict(owner="enemy",cells={(5,7),(6,7)},remaining_ticks=25)]
        rewards = PhaseRewards(game, "postplant", .99)
        rewards.begin_tick(); _, personal = rewards.finish_tick(False, False)
        self.assertNotIn("smoke_protection", personal["own"])

    def test_objective_progress_is_negative_and_interrupting_it_is_positive(self):
        for phase, field, component in [("postplant","defuse_timer","defuse_pressure"),
                                        ("search","plant_timer","plant_pressure")]:
            game = Game()
            enemy = game.chars[2] if phase=="postplant" else game.chars[0]
            rewards = PhaseRewards(game, phase, 1.)
            rewards.begin_tick(); setattr(enemy, field, 2)
            forward, _ = rewards.finish_tick(False, False)
            rewards.begin_tick(); setattr(enemy, field, 0)
            interrupted, _ = rewards.finish_tick(False, False)
            self.assertLess(forward[component], 0.)
            self.assertGreater(interrupted[component], 0.)
            self.assertAlmostEqual(forward[component]+interrupted[component], 0.)

    def test_component_logging_matches_discounted_personal_and_team_rewards(self):
        recorder = Recorder()
        for name in ("own", "ally"):
            recorder.record(name,dict(value=0.,phase="postplant"))
        recorder.reward(0.,.9,False,components={"outcome":0.})
        recorder.reward(5.,.9,True,components={"outcome":5.},individual_components={"own":{"damage_dealt":.2}})
        rows = recorder.advantages(.95)
        for row in rows:
            row["reward_version"] = "reward-test-v1"
        self.assertAlmostEqual(rows[0]["reward"], .9*5.2)
        self.assertAlmostEqual(rows[1]["reward"], .9*5.)
        with TemporaryDirectory() as folder:
            metrics = record_training_metrics(folder,0,1,EntropyFixture(),rows,2)
            logged = [json.loads(line) for line in (Path(folder)/"rollout_transitions.jsonl").read_text().splitlines()]
            for row in logged:
                self.assertAlmostEqual(sum(row["reward_components"].values()),row["reward"])
                self.assertEqual(row["reward_version"], "reward-test-v1")
            self.assertEqual(metrics["reward_version"], "reward-test-v1")
            self.assertAlmostEqual(sum(metrics["reward_component_means"].values()), metrics["reward_mean"])

    def test_preaim_pays_for_turning_to_a_visible_enemy_but_not_for_hidden_positions(self):
        game = Game(); own, _, enemy, _ = game.chars
        enemy.los_revealed = True
        rewards = PhaseRewards(game, "postplant", .99)
        own.facing = "W"; before = rewards.personal_potential(own)["preaim"]
        own.facing = "E"; after = rewards.personal_potential(own)["preaim"]
        self.assertGreater(potential_difference(before, after, .99, False), 0.)
        enemy.los_revealed = False
        self.assertEqual(rewards.personal_potential(own)["preaim"], 0.)
        enemy.los_revealed = True; game.grid[5, 7] = 1
        self.assertEqual(rewards.personal_potential(own)["preaim"], 0.)

    def test_invalid_breakdown_cannot_be_written_as_the_ai_reward(self):
        rows = [dict(phase="postplant",reward=1.,return_=0.,advantage=0.,value=0.,discount=0.,age=1.,
                     reward_components={"damage_dealt":.2})]
        rows[0]["return"] = rows[0].pop("return_")
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                record_training_metrics(folder,0,1,EntropyFixture(),rows,1)
            self.assertFalse((Path(folder)/"rollout_transitions.jsonl").exists())

    def test_named_resume_steps_and_run_do_not_change_task_fingerprint(self):
        original = b'RESUME = False\nTRAINING_STEPS = 20000\nRUN_NAME = "first"\nSEED = 42\n'
        changed = b'RESUME = True\nTRAINING_STEPS = 40000\nRUN_NAME = "second"\nSEED = 42\n'
        self.assertEqual(normalize_launch_settings(original), normalize_launch_settings(changed))
        self.assertNotEqual(normalize_launch_settings(original), normalize_launch_settings(changed.replace(b'42',b'43')))

    def test_observing_rewards_does_not_advance_random_generators(self):
        game = Game()
        states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with PhaseRewards(game, "postplant", .99) as rewards:
            rewards.begin_tick(); rewards.finish_tick(False, False)
        self.assertEqual(random.getstate(),states[0])
        current = np.random.get_state()
        self.assertEqual(current[0],states[1][0]); np.testing.assert_array_equal(current[1],states[1][1])
        self.assertEqual(current[2:],states[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),states[2]))
