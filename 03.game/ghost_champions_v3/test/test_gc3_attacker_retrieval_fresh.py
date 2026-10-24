"""Public drop inputs, stochastic replay, personal credit and fresh initialization."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory, encode_public, command_features
from ghost_champions_v3.preplant.gc3_policy_attacker_history import preplant_legal_actions
from ghost_champions_v3.preplant.gc3_policy_attacker_site import SiteHistoryPolicy,RETRIEVAL_SCHEMA_HASH,NAVIGATION_SCHEMA_HASH
from ghost_champions_v3.preplant.gc3_features_attacker_navigation import dropped_spike_features,DROP_FEATURE_WIDTH
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import initialize,policy_schema,train_rl
from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy,MICRO_INITIALIZATION_SEED
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.test import test_gc3_attacker_objective_rewards as objective_tests


class FreshRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def drop_record(self):
        char,state=preplant_fixture()
        state['grid'][:]=0
        char.pos=[5,5]
        for unit in state['chars']:unit.has_spike=False
        state['spike_pos']=(5,7)
        state['grid'][5,6]=1
        masks,_=preplant_legal_actions(char,state)
        return char,state,dict(obs=encode_public(char,state),masks=masks)

    def test_drop_distances_go_around_walls_and_do_not_choose_an_action(self):
        _,_,row=self.drop_record()
        masks={key:value.copy() for key,value in row['masks'].items()}
        features=dropped_spike_features(row)
        self.assertEqual(features.shape,(DROP_FEATURE_WIDTH,))
        self.assertEqual(features[:3].tolist(),[1.,1.,1.])
        self.assertAlmostEqual(features[3],1/(1+4/12),places=6)
        self.assertGreater(features[9],0.) # North shortens the walking route.
        self.assertLess(features[18],0.) # East is a wall, not a shortcut.
        for key,mask in masks.items():np.testing.assert_array_equal(mask,row['masks'][key])

    def test_carried_and_planted_spike_do_not_look_like_a_drop(self):
        char,state,_=self.drop_record()
        # Planted observations validate the candidate map against the real arena.
        state['grid']=preplant_fixture()[1]['grid']
        for kind in ('carried','planted','absent'):
            char.has_spike=kind=='carried'
            state['is_planted']=kind=='planted'
            state['planted_pos']=(5,7) if kind=='planted' else None
            state['spike_pos']=None if kind=='absent' else (5,7)
            self.assertFalse(dropped_spike_features(dict(obs=encode_public(char,state))).any())

    def test_drop_in_reported_wall_exposes_search_neighbours_without_true_coordinates(self):
        char,state,_=self.drop_record()
        state['spike_pos']=(5,6)
        feature=dropped_spike_features(dict(obs=encode_public(char,state)))
        self.assertEqual(feature[:3].tolist(),[1.,0.,1.])
        self.assertEqual(feature[3],1.) # Our cell neighbours the reported wall.

    def test_new_fresh_bc_loads_no_model_and_matches_runtime_initialization(self):
        with TemporaryDirectory() as tmp:
            config=dict(RESUME=False,SEED=MICRO_INITIALIZATION_SEED,INITIAL_CHECKPOINT=None,
                REQUIRE_FRESH_INITIALIZATION=True,SITE_GOAL_POLICY=True,NAVIGATION_FEATURES=True,
                RETRIEVAL_FEATURES=True,LEARNING_RATE=.0001)
            with patch('ghost_champions_v3.preplant.gc3_train_attacker_pipeline.load_preplant_policy',side_effect=AssertionError('Older model loaded')):
                model,optimizer,saved=initialize(config,Path(tmp),'bc','existing teacher dataset')
            self.assertIsNone(saved)
            self.assertEqual(len(optimizer.state),0)
            self.assertEqual(model.schema_hash,RETRIEVAL_SCHEMA_HASH)
            self.assertEqual(module_digest([model]),module_digest([fresh_micro_policy()]))
            self.assertEqual(json.loads((Path(tmp)/'manifest.json').read_text())['initialization'],'scratch')
            different=SiteHistoryPolicy.fresh('attacker',MICRO_INITIALIZATION_SEED+1,navigation=True,retrieval=True)
            self.assertNotEqual(module_digest([different]),module_digest([model]))
            with self.assertRaisesRegex(ValueError,'older checkpoint'):
                initialize(dict(config,INITIAL_CHECKPOINT=Path(tmp)/'old.pt'),Path(tmp),'bc','data')

    def test_retrieval_policy_sample_replay_and_site_choice_are_causal(self):
        char,state,_=self.drop_record()
        model=fresh_micro_policy()
        with torch.no_grad():
            torch.nn.init.normal_(model.retrieval_navigation[-1].weight,std=.1)
        history=MatchHistory();history.begin_round()
        rows=[]
        for _ in range(5):
            command,row=model.select(char,state,history)
            history.append(row['obs'],command_features(char,command,state),name=char.name)
            rows.append(row)
        logp,_,values=model.evaluate_records(rows)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in rows]),atol=5e-6,rtol=5e-6)
        torch.testing.assert_close(values,torch.tensor([r['value'] for r in rows]),atol=5e-6,rtol=5e-6)
        a,b=dict(rows[0],goal_site=0),dict(rows[0],goal_site=1)
        contexts=model.contexts([a,b])
        torch.testing.assert_close(model.distribution(contexts[0],a,'site').probs,model.distribution(contexts[1],b,'site').probs,atol=5e-6,rtol=5e-6)

    def test_old_checkpoint_remains_readable_and_new_shape_is_strict(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.pt'
            old=SiteHistoryPolicy.fresh('attacker',99,navigation=True)
            old.save(path,trained=True,stage='rl')
            loaded,_=SiteHistoryPolicy.load(path)
            self.assertEqual(loaded.schema_hash,NAVIGATION_SCHEMA_HASH)
            self.assertIsNone(loaded.retrieval_navigation)
            fresh=fresh_micro_policy();fresh.save(Path(tmp)/'new.pt',trained=False,stage='bc')
            loaded,_=SiteHistoryPolicy.load(Path(tmp)/'new.pt')
            self.assertEqual(module_digest([fresh]),module_digest([loaded]))
            self.assertEqual(policy_schema(dict(SITE_GOAL_POLICY=True,NAVIGATION_FEATURES=True,RETRIEVAL_FEATURES=True)),RETRIEVAL_SCHEMA_HASH)

    def test_personal_recovery_progress_does_not_pay_stationary_allies(self):
        game,char,reward=objective_tests.ObjectiveRewardTests().fixture()
        from test.test_ultimate_system import make_character
        ally=make_character('Leo','A',(3,1));game.chars.append(ally);reward.own[ally.name]=ally
        char.has_spike=False;char.pos=[2,0];game.spike_pos=(2,2)
        reward.begin_tick();char.pos=[2,1]
        shared,personal=reward.finish_tick(False)
        self.assertNotIn('retrieval_progress',shared)
        self.assertGreater(personal[char.name]['retrieval_progress'],0)
        self.assertNotIn('retrieval_progress',personal.get(ally.name,{}))

    def test_completion_is_paid_to_planter_only(self):
        game,char,reward=objective_tests.ObjectiveRewardTests().fixture()
        from test.test_ultimate_system import make_character
        ally=make_character('Leo','A',(3,1));game.chars.append(ally);reward.own[ally.name]=ally
        for _ in range(4):shared,personal=objective_tests.ObjectiveRewardTests().tick(game,char,reward)
        self.assertEqual(personal[char.name]['plant_complete'],5.)
        self.assertNotIn('plant_complete',shared)
        self.assertNotIn('plant_complete',personal.get(ally.name,{}))

    def test_new_family_never_falls_back_to_historical_checkpoints(self):
        from ghost_champions_v3.preplant import gc3_checkpoint_attacker_micro as checkpoints
        with TemporaryDirectory() as tmp,patch.object(checkpoints,'AI_ROOT',Path(tmp)):
            historical=Path(tmp)/'data/attacker_preplant_objectives_v1/best.pt'
            historical.parent.mkdir(parents=True);historical.touch()
            self.assertIsNone(checkpoints.active_micro_checkpoint())
            bc=Path(tmp)/'data'/checkpoints.MICRO_BC_RUN/'best.pt'
            bc.parent.mkdir(parents=True);bc.touch()
            self.assertEqual(checkpoints.active_micro_checkpoint(),bc)
            rl=Path(tmp)/'data'/checkpoints.MICRO_RL_RUN/'latest.pt'
            rl.parent.mkdir(parents=True);rl.touch()
            self.assertEqual(checkpoints.active_micro_checkpoint(),rl)
            best=rl.with_name('best.pt');best.touch()
            self.assertEqual(checkpoints.active_micro_checkpoint(),best)

    def test_normal_runtime_uses_new_micro_and_fresh_weights_until_bc_exists(self):
        import ghost_champions_v3 as gc3
        from ghost_champions_v3.preplant.gc3_learn_attacker_preplant import TwoPhaseAttackerController
        from ghost_champions_v3.preplant import gc3_checkpoint_attacker_micro as checkpoints
        with patch.object(checkpoints,'active_micro_checkpoint',return_value=None),patch(
                'ghost_champions_v3.preplant.gc3_learn_attacker_preplant.SAVE_MATCH_HISTORY',False):
            team=gc3.build_team_ai()
        self.assertFalse(gc3.USE_ATTACKER_MACRO)
        self.assertIs(team.attacker_factory.func,TwoPhaseAttackerController)
        self.assertIsNone(team.attacker_factory.keywords['preplant_checkpoint'])
        self.assertEqual(module_digest([team.attacker_factory.keywords['preplant_policy']]),module_digest([fresh_micro_policy()]))

    def test_rl_starts_at_zero_from_only_new_bc_and_rejects_old_anchors(self):
        from ghost_champions_v3.preplant.gc3_data_attacker_preplant import source_signature
        with TemporaryDirectory() as tmp:
            root=Path(tmp);bc=root/'bc';bc.mkdir();rl=root/'rl';rl.mkdir()
            fresh=fresh_micro_policy()
            # A temporary synthetic fixture, never written into the real runs.
            fresh.save(bc/'best.pt',stage='bc',trained=True,step=1234)
            (bc/'manifest.json').write_text(json.dumps(dict(initialization='scratch',source_hash=source_signature())))
            config=dict(RESUME=False,SEED=3,INITIAL_CHECKPOINT=None,REQUIRE_FRESH_BC=True,
                SITE_GOAL_POLICY=True,NAVIGATION_FEATURES=True,RETRIEVAL_FEATURES=True,
                BC_CHECKPOINT=bc/'best.pt',LEARNING_RATE=.0001,TORCH_THREADS=1,
                MIN_BC_PLANT_RATE=.5,MIN_BC_ORB_COLLECTIONS=1,
                RUN_PATH=root/'blocked',LOG_PATH=root/'logs/blocked')
            model,optimizer,saved=initialize(config,rl,'rl','new bc')
            self.assertIsNone(saved);self.assertFalse(optimizer.state)
            self.assertEqual(module_digest([model]),module_digest([fresh]))
            initial=torch.load(rl/'initial.pt',weights_only=True)
            self.assertEqual(initial['step'],0)
            with patch('ghost_champions_v3.preplant.gc3_train_attacker_pipeline.evaluate_preplant',return_value=dict(plant_rate=0.,orb_collections=0)):
                with self.assertRaisesRegex(RuntimeError,'PPO was not started'):train_rl(config)
            self.assertFalse(config['RUN_PATH'].exists())
            (bc/'manifest.json').write_text(json.dumps(dict(initialization='gc3_bc_finetune',source_hash=source_signature())))
            with patch('ghost_champions_v3.preplant.gc3_train_attacker_pipeline.evaluate_preplant',side_effect=AssertionError('Old BC evaluated')):
                with self.assertRaisesRegex(ValueError,'older BC/RL weights'):train_rl(config)

    def test_personal_drop_progress_is_bounded_even_with_backtracking(self):
        game,char,reward=objective_tests.ObjectiveRewardTests().fixture()
        char.has_spike=False;char.pos=[2,0];game.spike_pos=(2,2)
        reward.retrieval_weight=6.
        earned=0.
        for column in (1,0,1,2,1,0,1,2):
            reward.begin_tick();char.pos=[2,column]
            _,personal=reward.finish_tick(False)
            earned+=personal.get(char.name,{}).get('retrieval_progress',0.)
            game.battle_tick+=1
        self.assertGreater(earned,0.)
        self.assertLessEqual(earned,6.)
        self.assertAlmostEqual(earned,6.-6./(1.+2/12.))


if __name__=='__main__':unittest.main()
