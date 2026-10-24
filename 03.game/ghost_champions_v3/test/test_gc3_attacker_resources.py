"""Real engine orb credit, remaining-point carry and safe RL weight continuation."""
from collections import defaultdict, Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from test.test_ultimate_system import UltimateTestGame, FixedController, make_character
from game_core import ORB_COLLECT_REQUIRED_TICKS
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import PreplantRewards, ORB_ULTIMATE_REWARD_PER_POINT
from ghost_champions_v3.preplant.gc3_state_attacker_resources import MatchResources
from ghost_champions_v3.preplant.gc3_policy_attacker_site import SiteHistoryPolicy
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import initialize, rl_input_hash
from ghost_champions_v3.tools.gc3_metrics_monitor_data import snapshot


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def orb_fixture(self, points=0, team="A"):
        game=UltimateTestGame()
        char=make_character("Demon1",team,(2,2),points)
        game.chars=[char];game.available_orbs={(2,2)}
        game.attacker_controller=game.defender_controller=FixedController("COLLECT_ORB")
        reward=PreplantRewards.__new__(PreplantRewards)
        reward.game=game;reward.original_move=game.move_character
        reward.pending=defaultdict(Counter);reward.completed_orbs=0;reward.orb_points_gained=0
        return game,char,reward

    def test_only_completion_rewards_real_points_and_cannot_repeat(self):
        game,char,reward=self.orb_fixture()
        for _ in range(ORB_COLLECT_REQUIRED_TICKS-1):
            reward.move_character(char)
            self.assertEqual(reward.completed_orbs,0)
            self.assertEqual(sum(reward.pending[char.name].values()),0)
        reward.move_character(char)
        self.assertEqual(reward.completed_orbs,1)
        self.assertEqual(reward.orb_points_gained,2)
        self.assertAlmostEqual(reward.pending[char.name]["orb_ultimate_gain"],2*ORB_ULTIMATE_REWARD_PER_POINT)
        # A later ULT spend cannot erase already-earned collection credit.
        char.ultimate_points=0
        reward.move_character(char)
        self.assertEqual(reward.orb_points_gained,2)
        self.assertEqual(reward.completed_orbs,1)

    def test_interruption_resets_channel_without_reward(self):
        game,char,reward=self.orb_fixture()
        reward.move_character(char);reward.move_character(char)
        game.attacker_controller.action="MOVE"
        reward.move_character(char)
        self.assertEqual(char.orb_collect_timer,0)
        self.assertEqual(reward.completed_orbs,0)
        self.assertEqual(reward.orb_points_gained,0)

    def test_full_and_partial_capacity_credit_only_available_gain(self):
        for free in (0,1):
            game,char,reward=self.orb_fixture()
            char.ultimate_points=char.ultimate_cost-free
            for _ in range(ORB_COLLECT_REQUIRED_TICKS):reward.move_character(char)
            self.assertEqual(reward.completed_orbs,1)
            self.assertEqual(reward.orb_points_gained,free)
            self.assertAlmostEqual(reward.pending[char.name]["orb_ultimate_gain"],free*ORB_ULTIMATE_REWARD_PER_POINT)

    def test_enemy_collection_does_not_reward_the_student(self):
        game,char,reward=self.orb_fixture(team="D")
        for _ in range(ORB_COLLECT_REQUIRED_TICKS):reward.move_character(char)
        self.assertEqual(char.ultimate_points,2)
        self.assertEqual(reward.completed_orbs,0)
        self.assertEqual(dict(reward.pending),{})

    def test_actual_remaining_points_survive_death_spend_and_roster_aliases(self):
        own=make_character("Demon1","A",(2,2),3)
        enemy=make_character("Demon1","D",(2,3),2)
        enemy.name="D:Demon1"
        game=UltimateTestGame();game.chars=[own,enemy];game.round_over=True
        own.is_alive=False
        resources=MatchResources();resources.capture(game,"GC")
        own.ultimate_points=enemy.ultimate_points=0
        resources.restore(game,"GC")
        self.assertEqual((own.ultimate_points,enemy.ultimate_points),(3,2))
        self.assertEqual(game.match_stats[own.name]["ultimate_points"],3)
        # ULT consumption is carried as zero, rather than lifetime earned points.
        own.ultimate_points=0;resources.capture(game,"GC")
        own.ultimate_points=4;resources.restore(game,"GC")
        self.assertEqual(own.ultimate_points,0)
        # A new match resets both sides; another enemy roster has separate state.
        resources.restore(game,"FNC")
        self.assertEqual(enemy.ultimate_points,0)
        own.ultimate_points=4;MatchResources().restore(game,"GC")
        self.assertEqual(own.ultimate_points,0)
        game.round_over=False
        with self.assertRaisesRegex(RuntimeError,"actual round"):resources.capture(game,"GC")

    def test_warm_start_keeps_weights_and_step_without_old_optimizer(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);bc=root/"bc.pt";previous=root/"rl.pt";guard=root/"guard.pt"
            guard.write_bytes(b"frozen guard checkpoint identity")
            model=SiteHistoryPolicy.fresh("attacker",88)
            model.save(bc,stage="bc",trained=True)
            model.save(previous,stage="rl",trained=True,step=11438,bc_checkpoint_hash=file_hash(bc))
            run=root/"new";run.mkdir()
            config=dict(RESUME=False,SEED=89,INITIAL_CHECKPOINT=previous,BC_CHECKPOINT=bc,
                        POSTPLANT_CHECKPOINT=guard,SITE_GOAL_POLICY=True,LEARNING_RATE=.0001)
            input_hash=rl_input_hash(config,file_hash(bc))
            policy,optimizer,saved=initialize(config,run,"rl",input_hash)
            self.assertIsNone(saved)
            self.assertEqual(len(optimizer.state),0)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,policy.state_dict()[key],rtol=0,atol=0)
            initial=torch.load(run/"initial.pt",weights_only=True)
            self.assertEqual(initial["step"],11438)
            manifest=json.loads((run/"manifest.json").read_text())
            self.assertEqual(manifest["initialization"],"rl_warm_start")
            guard.write_bytes(b"changed guard")
            self.assertNotEqual(input_hash,rl_input_hash(config,file_hash(bc)))

    def test_graph_counts_only_new_decisions_on_top_of_inherited_step(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"initial_evaluation.json").write_text(json.dumps(dict(step=11438,plant_rate=.5)))
            (root/"evaluation_12438.json").write_text(json.dumps(dict(step=12438,plant_rate=.6)))
            (root/"episodes.jsonl").write_text(json.dumps(dict(decisions=1000,success=True))+"\n")
            result=snapshot(root)
            self.assertEqual(result["steps"],[12438])
            self.assertEqual(result["step"],12438)
            self.assertEqual([r["step"] for r in result["evaluations"]],[11438,12438])


if __name__=="__main__":
    unittest.main()
