"""Real plant actions, discounted cancel exploits and completed micro objectives."""
import io
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import numpy as np
from game_core import PLANT_REQUIRED_TICKS
from test.test_ultimate_system import UltimateTestGame, FixedController, make_character
from ghost_champions_v3.preplant.gc3_reward_attacker_objectives import (
    MicroObjectives, SPIKE_PICKUP_REWARD, PLANT_PROGRESS_REWARD_PER_TICK,
    PLANT_COMPLETE_REWARD, PLANT_CANCEL_EXTRA_PENALTY,
)
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import PreplantRewards, preplant_reward_settings
from ghost_champions_v3.gc3_rollout import Recorder, engine_directory


class ObjectiveRewardTests(unittest.TestCase):
    def fixture(self, gamma=.99):
        game = UltimateTestGame()
        char = make_character("Demon1", "A", (2, 2))
        char.has_spike = True
        game.chars = [char]
        game.grid[2, 2] = 2
        game.attacker_controller = FixedController("PLANT")
        objective = MicroObjectives(game, gamma, lambda:0,
            lambda position, goal:1./(1+abs(position[1]-2)/12), retrieval_weight=.5, distance_scale=12)
        return game, char, objective

    def tick(self, game, char, reward, *, action=None, terminal=False):
        if action is not None:
            game.attacker_controller.action = action
        reward.begin_tick()
        before = dict(timer=int(char.plant_timer), has_spike=bool(char.has_spike),
                      planted=game.is_planted, pos=tuple(char.pos))
        game.move_character(char)
        reward.observe_move(char, before)
        game.battle_tick += 1
        return reward.finish_tick(terminal)

    def test_actual_completion_credits_all_four_ticks_and_one_completion(self):
        game, char, reward = self.fixture()
        progress = completion = 0.
        for _ in range(PLANT_REQUIRED_TICKS):
            shared, personal = self.tick(game, char, reward)
            progress += personal[char.name]["plant_progress"]
            completion += personal[char.name].get("plant_complete", 0.)
            self.assertEqual(shared["site_progress"], 0.)
            self.assertNotIn("plant_cancel", personal[char.name])
        self.assertTrue(game.is_planted)
        self.assertEqual(char.plant_timer, 0)
        self.assertFalse(char.has_spike)
        self.assertEqual(progress, PLANT_REQUIRED_TICKS*PLANT_PROGRESS_REWARD_PER_TICK)
        self.assertEqual(completion, PLANT_COMPLETE_REWARD)
        shared, personal = self.tick(game, char, reward)
        self.assertEqual(shared.get("plant_complete", 0.), 0.)
        self.assertEqual(sum(personal.get(char.name, {}).values()), 0.)
        self.assertEqual(reward.plants, 1)

    def test_invalid_plant_and_wait_and_enemy_do_not_receive_progress(self):
        for invalid in ("floor", "no_spike", "enemy", "wait"):
            with self.subTest(invalid=invalid):
                game, char, reward = self.fixture()
                if invalid == "floor": game.grid[2, 2] = 0
                if invalid == "no_spike": char.has_spike = False
                if invalid == "enemy": char.team = "D"
                shared, personal = self.tick(game, char, reward, action="MOVE" if invalid=="wait" else "PLANT")
                self.assertEqual(sum(personal.get(char.name, {}).values()), 0.)
                self.assertEqual(shared.get("plant_complete", 0.), 0.)

    def test_each_repeated_partial_plant_cancel_cycle_is_negative_when_discounted(self):
        for gamma in (1., .99, .9):
            for steps in range(1, PLANT_REQUIRED_TICKS):
                with self.subTest(gamma=gamma, steps=steps):
                    game, char, reward = self.fixture(gamma)
                    for cycle in range(4):
                        values = []
                        for _ in range(steps):
                            _, personal = self.tick(game, char, reward, action="PLANT")
                            values.append(sum(personal[char.name].values()))
                        _, personal = self.tick(game, char, reward, action="MOVE")
                        values.append(sum(personal[char.name].values()))
                        total = sum(gamma**i*value for i, value in enumerate(values))
                        self.assertAlmostEqual(total, -PLANT_CANCEL_EXTRA_PENALTY*gamma**steps)
                        self.assertLess(total, 0.)
                    self.assertEqual(reward.cancellations, 4)
                    _, personal = self.tick(game, char, reward, action="MOVE")
                    self.assertEqual(sum(personal.get(char.name, {}).values()), 0.)

    def test_waiting_before_cancel_cannot_hide_the_discounted_refund(self):
        game, char, reward = self.fixture()
        _, personal = self.tick(game, char, reward)
        first = personal[char.name]["plant_progress"]
        # A temporarily preserved timer may not reduce the refund's real value.
        for _ in range(100):
            reward.begin_tick(); game.battle_tick += 1; reward.finish_tick(False)
        _, personal = self.tick(game, char, reward, action="MOVE")
        cancel = personal[char.name]["plant_cancel"]
        self.assertAlmostEqual(first+.99**101*cancel, -PLANT_CANCEL_EXTRA_PENALTY*.99**101)

    def test_death_and_timeout_settle_unfinished_attempts_once(self):
        for kind in ("death", "timeout"):
            game, char, reward = self.fixture()
            self.tick(game, char, reward)
            reward.begin_tick()
            if kind == "death": char.is_alive=False; char.plant_timer=0
            game.battle_tick += 1
            _, personal = reward.finish_tick(kind=="timeout")
            self.assertAlmostEqual(personal[char.name]["plant_cancel"],
                                   -(PLANT_PROGRESS_REWARD_PER_TICK/.99+PLANT_CANCEL_EXTRA_PENALTY))
            reward.begin_tick(); game.battle_tick += 1
            _, personal = reward.finish_tick(True)
            self.assertNotIn("plant_cancel", personal.get(char.name, {}))

    def test_cancel_then_complete_keeps_only_the_successful_attempt(self):
        game, char, reward = self.fixture(1.)
        total = 0.
        for action in ("PLANT", "PLANT", "MOVE")+("PLANT",)*PLANT_REQUIRED_TICKS:
            shared, personal = self.tick(game, char, reward, action=action)
            total += sum(shared.values())+sum(personal.get(char.name, {}).values())
        self.assertAlmostEqual(total, PLANT_COMPLETE_REWARD+PLANT_REQUIRED_TICKS*PLANT_PROGRESS_REWARD_PER_TICK-PLANT_CANCEL_EXTRA_PENALTY)

    def test_pickup_is_personal_once_and_has_no_retrieval_reset_penalty(self):
        game, char, reward = self.fixture()
        char.has_spike=False; char.pos=[2, 1]; game.spike_pos=(2, 2)
        reward.begin_tick()
        char.pos=[2, 2]; char.has_spike=True; game.spike_pos=None
        reward.observe_before_shots(); reward.observe_before_shots()
        shared, personal = reward.finish_tick(False)
        self.assertGreater(personal[char.name]["retrieval_progress"], 0.)
        self.assertNotIn("retrieval_progress",shared)
        self.assertEqual(personal[char.name]["spike_pickup"], SPIKE_PICKUP_REWARD)
        reward.begin_tick()
        shared, personal = reward.finish_tick(False)
        self.assertEqual(shared.get("retrieval_progress", 0.), 0.)
        self.assertNotIn(char.name, personal)
        self.assertEqual(reward.pickups, 1)

    def test_pickup_then_death_still_counts_and_same_cell_later_drop_is_new(self):
        game, char, reward = self.fixture()
        char.has_spike=False; game.spike_pos=(2, 2)
        reward.begin_tick(); char.has_spike=True; game.spike_pos=None
        reward.observe_before_shots()
        char.is_alive=False; char.has_spike=False; game.spike_pos=(2, 2)
        _, personal = reward.finish_tick(True)
        self.assertEqual(personal[char.name]["spike_pickup"], SPIKE_PICKUP_REWARD)
        ally=make_character("Leo","A",(2, 3)); game.chars.append(ally); reward.own[ally.name]=ally
        reward.begin_tick(); ally.pos=[2, 2]; ally.has_spike=True; game.spike_pos=None
        shared, personal=reward.finish_tick(False)
        self.assertGreater(personal[ally.name]["retrieval_progress"], 0.)
        self.assertEqual(personal[ally.name]["spike_pickup"], SPIKE_PICKUP_REWARD)

    def test_site_and_retrieval_backtracking_do_not_repeat_approach_reward(self):
        for dropped in (False, True):
            game, char, reward = self.fixture()
            char.pos=[2, 0]
            if dropped: char.has_spike=False; game.spike_pos=(2, 2)
            values=[]
            key="retrieval_progress" if dropped else "site_progress"
            for x in (1, 0, 1, 0, 1):
                reward.begin_tick(); char.pos=[2, x]; game.battle_tick+=1
                shared, personal=reward.finish_tick(False)
                values.append((personal.get(char.name,{}) if dropped else shared).get(key,0.))
            self.assertGreater(values[0],0.)
            self.assertEqual(values[1:],[0.]*4)

    def test_new_site_choice_or_drop_is_not_paid_as_movement(self):
        game, char, reward = self.fixture()
        reward.goal_provider=lambda:None
        reward.begin_tick(); reward.goal_provider=lambda:0
        shared,_=reward.finish_tick(False)
        self.assertEqual(shared["site_progress"],0.)
        reward.begin_tick(); char.has_spike=False; game.spike_pos=(2,2)
        shared,_=reward.finish_tick(False)
        self.assertEqual(shared.get("retrieval_progress",0.),0.)

    def real_game(self):
        with engine_directory(), redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle
            from team_ai import DualRoleTeamAI
            from map_data import NEW_MAZE_STR
            from party_presets import get_preset
            team=DualRoleTeamAI("objective-test",lambda:FixedController("PLANT"),lambda:FixedController("MOVE"))
            game=VisualFPSBattle(NEW_MAZE_STR,team,team,headless=True,disable_side_swap=True,
                                attacker_roster=list(get_preset("Fnatic2023").players),
                                defender_roster=list(get_preset("Ghost Champions").players))
        game.stop_after_round=True
        game.defender_setup_phase.finish()
        char=next(c for c in game.chars if c.team=="A")
        for unit in game.chars:
            if unit.team=="A":
                unit.has_spike=False
                if unit is not char: unit.is_alive=False
        point=tuple(map(int,np.argwhere(game.grid==2)[0]))
        char.pos=list(point); char.has_spike=True
        return game,char,point

    def test_real_rollout_reward_wiring_completion_components_and_frozen_guard_boundary(self):
        game,char,point=self.real_game()
        goal=int(point[1]>=game.width/2)
        recorder=Recorder();progress=completion=0.
        with PreplantRewards(game,.99,lambda:goal) as reward, patch.object(game,"_resolve_all_shots",wraps=game._resolve_all_shots):
            for _ in range(PLANT_REQUIRED_TICKS):
                recorder.record(char.name,dict(value=0.))
                reward.begin_tick(); game._simulate_tick()
                shared,personal=reward.finish_tick(game.is_planted,False)
                recorder.reward(sum(shared.values()),.99,False,components=shared,individual_components=personal)
                progress+=personal[char.name].get("plant_progress",0.)
                completion+=personal[char.name].get("plant_complete",0.)
                self.assertNotIn("plant_complete",shared)
                self.assertEqual(shared["site_progress"],0.)
                self.assertNotIn("outcome",shared)
                self.assertNotIn("round_outcome",shared)
        self.assertTrue(game.is_planted)
        self.assertEqual(progress,1.)
        self.assertEqual(completion,PLANT_COMPLETE_REWARD)
        for row in recorder.rows:
            self.assertAlmostEqual(row["reward"],sum(row["reward_components"].values()))

    def test_real_engine_pickup_before_shooting_then_death_is_credited(self):
        game,char,point=self.real_game()
        char.has_spike=False; game.spike_pos=point
        game.attacker_controller.inner_controller.action="MOVE"
        original=game._resolve_all_shots
        def die(*args,**kwargs):
            self.assertTrue(char.has_spike)
            char.is_alive=False; char.has_spike=False; game.spike_pos=point
        game._resolve_all_shots=die
        with PreplantRewards(game,.99,lambda:0) as reward:
            reward.begin_tick(); game._simulate_tick()
            shared,personal=reward.finish_tick(game.round_over,False)
            self.assertEqual(personal[char.name]["spike_pickup"],SPIKE_PICKUP_REWARD)
            self.assertGreaterEqual(shared.get("retrieval_progress",0.),0.)
        self.assertIs(game._resolve_all_shots,die)
        game._resolve_all_shots=original

    def test_reward_manifest_contains_all_event_coefficients(self):
        settings=preplant_reward_settings()
        self.assertEqual(settings["SPIKE_PICKUP_REWARD"],SPIKE_PICKUP_REWARD)
        self.assertEqual(settings["PLANT_COMPLETE_REWARD"],PLANT_COMPLETE_REWARD)
        self.assertEqual(settings["PLANT_CANCEL_EXTRA_PENALTY"],PLANT_CANCEL_EXTRA_PENALTY)


if __name__=="__main__":unittest.main()
