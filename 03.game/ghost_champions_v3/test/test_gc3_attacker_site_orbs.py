"""Site-specific rewards, causal learned plans and the completed-orb RL gate."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.preplant.gc3_policy_attacker_site import SiteHistoryPolicy
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory, command_features
from ghost_champions_v3.preplant.gc3_teacher_attacker_actions import RecordingTeacher, teacher_record, setup_executed_command
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import save_shard, load_shard
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import PreplantRewards
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import train_rl, optimize_history, bc_log_prob
from ghost_champions_v3.gc3_engine_actions import MODES, legal_actions, decode_command
from ghost_champions_v3.gc3_reward_phase import potential_difference


class SiteOrbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_approaching_B_rewards_progress_even_while_moving_away_from_A(self):
        reward = PreplantRewards.__new__(PreplantRewards)
        carrier = SimpleNamespace(is_alive=True, has_spike=True, pos=[0, 0], plant_timer=0)
        reward.own = {"c":carrier}
        reward.game = SimpleNamespace(spike_pos=None)
        reward.goal_provider = lambda:1
        reward.goal_distances = {0:np.array([[1, 2, 3]]), 1:np.array([[3, 2, 1]])}
        before = reward.objective_potential()["site_progress"]
        carrier.pos = [0, 1]
        after = reward.objective_potential()["site_progress"]
        self.assertGreater(potential_difference(before, after, .99, False), 0)
        reward.goal_provider = lambda:0
        before = reward.objective_potential()["site_progress"]
        carrier.pos = [0, 2]
        after = reward.objective_potential()["site_progress"]
        self.assertLess(potential_difference(before, after, .99, False), 0)

    def test_site_is_sampled_once_and_shared_until_the_next_round(self):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        model = SiteHistoryPolicy.fresh("attacker", 32)
        command, first = model.select(char, state, history)
        self.assertIn("site", first["selections"])
        history.append(first["obs"], command_features(char, command, state), name=char.name)
        _, second = model.select(char, state, history)
        self.assertNotIn("site", second["selections"])
        self.assertEqual(first["goal_site"], second["goal_site"])
        history.begin_round()
        _, third = model.select(char, state, history)
        self.assertIn("site", third["selections"])

    def test_current_site_label_is_not_visible_to_the_site_decision(self):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        model = SiteHistoryPolicy.fresh("attacker", 33)
        _, row = model.select(char, state, history)
        a = dict(row, goal_site=0); b = dict(row, goal_site=1)
        context = model.contexts([a,b])
        pa = model.distribution(context[0], a, "site").probs
        pb = model.distribution(context[1], b, "site").probs
        torch.testing.assert_close(pa,pb,atol=1e-6,rtol=1e-6)
        self.assertTrue(bool((pa>0).all()))

    def test_own_plan_and_first_choice_survive_teacher_shard_roundtrip(self):
        char, state = preplant_fixture()
        history = MatchHistory(); history.begin_round()
        teacher = SimpleNamespace(target_site=1, decide_move=lambda c,s:(list(c.pos),"MOVE"))
        recorder = RecordingTeacher(teacher,history=history)
        recorder.decide_move(char,state); recorder.decide_move(char,state)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"labels.npz"
            save_shard(path,recorder.rows)
            restored=load_shard(path)
            self.assertEqual(restored[0]["selections"]["site"],1)
            self.assertNotIn("site",restored[1]["selections"])
            self.assertEqual(restored[1]["goal_site"],1)
            self.assertEqual(restored[0]["history"].events[0]["goal_site"],1)

    def test_orb_teacher_labels_use_the_real_collection_action(self):
        char,state=preplant_fixture()
        state["available_orbs"]={tuple(char.pos)}
        teacher=SimpleNamespace(decide_move=lambda c,s:(list(c.pos),"COLLECT_ORB"))
        recorder=RecordingTeacher(teacher)
        recorder.reset_round = lambda:None
        recorder.history.begin_round()
        recorder.decide_move(char,state)
        self.assertEqual(recorder.rows[0]["selections"]["mode"],MODES.index("COLLECT_ORB"))

    def test_setup_labels_ignore_disabled_channels_and_blocked_moves(self):
        char,state=preplant_fixture()
        state["defender_setup_active"]=True
        command=setup_executed_command(char,state,(list(char.pos),"COLLECT_ORB"))
        row=teacher_record(char,state,command)
        self.assertEqual(row["selections"]["mode"],MODES.index("MOVE"))
        self.assertEqual(decode_command(char,state,row["selections"])[0],list(char.pos))
        command=setup_executed_command(char,state,([-1,-1],{"facing":"N"}))
        self.assertEqual(command,(list(char.pos),"MOVE"))

    def test_blocked_live_movement_preserves_executed_facing_into_a_wall(self):
        char,state=preplant_fixture()
        grid=state["grid"]
        char.pos=[5,5]
        grid[5,5]=0; grid[5,6]=1
        row=teacher_record(char,state,([5,6],{"facing":"E"}))
        self.assertEqual(row["selections"]["mode"],MODES.index("WATCH"))
        command=decode_command(char,state,row["selections"])
        self.assertEqual(command,([5,5],{"facing":"E"}))

    def test_checkpoint_transfer_and_new_serialization_preserve_existing_weights(self):
        from ghost_champions_v3.preplant.gc3_policy_attacker_history import HistoryPolicy, load_preplant_policy
        initial=HistoryPolicy.fresh("attacker",37)
        model=SiteHistoryPolicy.from_history(initial)
        for key,value in initial.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"site.pt"
            model.save(path,stage="bc",trained=True)
            restored,_=load_preplant_policy(path)
            self.assertIsInstance(restored,SiteHistoryPolicy)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)

    def test_no_completed_orb_prevents_rl_even_when_plant_rate_is_good(self):
        with TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/"bc.pt"
            SiteHistoryPolicy.fresh("attacker",34).save(checkpoint,stage="bc",trained=True)
            config=dict(BC_CHECKPOINT=checkpoint,SITE_GOAL_POLICY=True,TORCH_THREADS=1,
                MIN_BC_PLANT_RATE=.01,MIN_BC_ORB_COLLECTIONS=1,
                RUN_PATH=Path(tmp)/"rl",LOG_PATH=Path(tmp)/"logs/rl")
            with patch("ghost_champions_v3.preplant.gc3_train_attacker_pipeline.evaluate_preplant",
                       return_value=dict(plant_rate=.6,success_rate=.7,orb_collections=0)):
                with self.assertRaisesRegex(RuntimeError,"completed orbs=0.*PPO was not started"):
                    train_rl(config)
            self.assertFalse(config["RUN_PATH"].exists())

    def test_site_ppo_replays_joint_sampling_with_zero_optimizer_updates(self):
        char,state=preplant_fixture()
        history=MatchHistory();history.begin_round()
        model=SiteHistoryPolicy.fresh("attacker",35)
        rows=[]
        for i in range(4):
            command,row=model.select(char,state,history)
            history.append(row["obs"],command_features(char,command,state),name=char.name)
            row.update(advantage=i-1.5,**{"return":i/4})
            rows.append(row)
        logp,_,values=model.evaluate_records(rows)
        torch.testing.assert_close(logp,torch.tensor([r["logp"] for r in rows]),atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(values,torch.tensor([r["value"] for r in rows]),atol=3e-6,rtol=3e-6)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        anchor=SiteHistoryPolicy.fresh("attacker",36).requires_grad_(False)
        optimizer=SimpleNamespace(zero_grad=model.zero_grad,step=lambda:None)
        config=dict(PPO_EPOCHS=1,BATCH_SIZE=4,PPO_CLIP=.2,ENTROPY_WEIGHT=.01,BC_KL_WEIGHT=.1)
        self.assertTrue(np.isfinite(optimize_history(model,anchor,optimizer,rows,config)))
        self.assertTrue(torch.isfinite(model.site.weight.grad).all())
        self.assertTrue(all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()))

    def test_setup_attacker_uses_the_engine_permission_mask(self):
        char,state=preplant_fixture()
        state["defender_setup_active"]=True
        from map_data_defender_setup import get_setup_mask
        grid=state["grid"]
        allowed=np.asarray(get_setup_mask())==0
        point=next(tuple(p) for p in np.argwhere(allowed & (grid!=1))
                   if any(0<=p[0]+dr<grid.shape[0] and 0<=p[1]+dc<grid.shape[1] and allowed[p[0]+dr,p[1]+dc]
                          and grid[p[0]+dr,p[1]+dc]!=1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]))
        char.pos=list(point)
        state["chars"]=[char]
        mask,_=legal_actions(char,state)
        self.assertGreater(int(mask["position"].sum()),1)
        destinations=np.argwhere(mask["position"].reshape(grid.shape))
        self.assertTrue(all(allowed[tuple(p)] for p in destinations))
        self.assertFalse(mask["mode"][MODES.index("COLLECT_ORB")])


if __name__ == "__main__":
    unittest.main()
