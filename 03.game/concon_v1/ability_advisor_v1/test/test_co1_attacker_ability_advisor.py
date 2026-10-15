"""History privacy, supported advice, per-point routing, and match-level validation."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import io
import json
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from concon_v1.ability_advisor_v1.co1_learn_attacker_ability_advisor import (
    FixedAbilityAdvisor, MatchMemory, Advice, AbilityJudge, build_features,
    feature_names, group_key, relevant_enemy_positions,
    VERSION, LABEL_VERSION, WALL_LOS_VERSION,
)
from concon_v1.ability_advisor_v1.co1_collect_attacker_ability_advisor import RecordingAdvisor
from concon_v1.ability_advisor_v1.co1_train_attacker_ability_advisor import (
    split_matches, classify_report, safe_group,
)
from concon_v1.co1_attacker_scenarios import get_scenario
from concon_v1.co1_attacker_controller import ConconAttackerController


def actor(name='caster',team='A',pos=(8,2),known=True):
    return SimpleNamespace(name=name,team=team,pos=list(pos),position_known=known,
        is_alive=True,facing='E',flash_charges=1,recon_charges=1,smoke_charges=1,
        ability_name='SMOKE',hp=100,max_hp=100,has_spike=True)


class AbilityAdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        scenario = get_scenario('A1')
        char = actor()
        enemy = actor('enemy','D',(-1,-1),False)
        state = dict(grid=np.zeros_like(scenario.grid),chars=[char,enemy],battle_tick=25)
        memory = MatchMemory()
        memory.observe(scenario,char,state,'omoko_v1')
        return scenario,char,enemy,state,memory

    def model_advisor(self,scenario):
        advisor = FixedAbilityAdvisor(None)
        advisor.model = AbilityJudge()
        with torch.no_grad():
            for p in advisor.model.parameters():
                p.zero_()
            advisor.model.layers[-1].bias.fill_(-10)
        advisor.bind_route(scenario,'carry-test')
        key = group_key(scenario,'omoko_v1','RECON',(8,8))
        advisor.checkpoint = dict(scenarios={'A1':dict(signature=scenario.signature,carry_sha256='carry-test')},
                                 groups={key:dict(validated=True)},skip_threshold=.1)
        return advisor

    def test_hidden_positions_and_facing_do_not_change_features_or_history(self):
        scenario,char,enemy,state,memory = self.fixture()
        first,_ = build_features(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),memory)
        enemy.pos,enemy.facing = [8,8],'W'
        memory.observe(scenario,char,state,'omoko_v1')
        second,direct = build_features(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),memory)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(len(first),len(feature_names()))
        self.assertFalse(direct)
        self.assertFalse(memory.current['sightings'])

    def test_only_completed_rounds_enter_history_and_other_routes_are_retained(self):
        scenario,char,enemy,state,memory = self.fixture()
        enemy.position_known,enemy.pos = True,[8,8]
        memory.observe(scenario,char,state,'omoko_v1')
        self.assertFalse(any(memory.features((8,8),'RECON',25,scenario.grid.shape)))
        memory.start_round('A2')
        history = memory.features((8,8),'RECON',25,scenario.grid.shape)
        self.assertEqual(history[0],1)
        self.assertEqual(history[1:5],[1,0,0,0])  # Previous A1 while current route is A2.
        self.assertEqual(history[8],1)
        for _ in range(8):
            memory.start_round('A3')
        self.assertEqual(len(memory.rounds),5)

    def test_new_match_and_opponent_clear_history(self):
        scenario,char,_,state,_ = self.fixture()
        advisor = FixedAbilityAdvisor(None)
        advisor.start_round('A1',1)
        advisor.observe(scenario,char,state,'omoko_v1')
        advisor.start_round('A2',2)
        self.assertEqual(len(advisor.memory.rounds),1)
        advisor.start_round('A1',1)
        self.assertFalse(advisor.memory.rounds)
        advisor.observe(scenario,char,state,'omoko_v1')
        advisor.start_round('A2',2)
        advisor.observe(scenario,char,state,'gc_v2')
        self.assertFalse(advisor.memory.rounds)

    def test_history_records_confirmed_charge_use_not_rejected_requests(self):
        scenario,char,_,state,_ = self.fixture()
        advisor = FixedAbilityAdvisor(None)
        advisor.observe(scenario,char,state,'omoko_v1')
        game = SimpleNamespace(_projectile_path=lambda s,t:[s,t])
        advisor.advise(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),game)
        state['battle_tick'] += 1
        advisor.observe(scenario,char,state,'omoko_v1')
        self.assertFalse(advisor.memory.current['uses'])
        advisor.advise(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),game)
        char.recon_charges -= 1
        state['battle_tick'] += 1
        advisor.observe(scenario,char,state,'omoko_v1')
        self.assertIn(('RECON',(8,8)),advisor.memory.current['uses'])

    def test_supported_model_saves_but_observed_enemy_always_uses(self):
        scenario,char,enemy,state,_ = self.fixture()
        advisor = self.model_advisor(scenario)
        advisor.observe(scenario,char,state,'omoko_v1')
        game = SimpleNamespace(_projectile_path=lambda s,t:[s,t])
        self.assertFalse(advisor.advise(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),game).use)
        enemy.position_known,enemy.pos = True,[8,8]
        self.assertTrue(advisor.advise(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),game).use)

    def test_unknown_group_stale_carry_and_changed_map_all_use(self):
        scenario,char,_,state,_ = self.fixture()
        advisor = self.model_advisor(scenario)
        advisor.observe(scenario,char,state,'omoko_v1')
        game = SimpleNamespace(_projectile_path=lambda s,t:[s,t])
        self.assertTrue(advisor.advise(scenario,char,state,'unknown','RECON',(8,8),(8,8),game).use)
        advisor.bind_route(scenario,'new-carry')
        self.assertTrue(advisor.advise(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),game).use)
        advisor.bind_route(scenario,'carry-test')
        changed = replace(scenario,signature='new-map')
        self.assertTrue(advisor.advise(changed,char,state,'omoko_v1','RECON',(8,8),(8,8),game).use)

    def test_teacher_labels_respect_flash_walls_recon_radius_and_smoke_lane(self):
        grid = np.zeros((12,12),dtype=np.int32)
        char,enemy = actor(pos=(5,0)),actor('enemy','D',(5,8))
        self.assertTrue(relevant_enemy_positions(grid,[char,enemy],'A','SMOKE',(5,4)))
        self.assertTrue(relevant_enemy_positions(grid,[char,enemy],'A','RECON',(5,4)))
        self.assertFalse(relevant_enemy_positions(grid,[char,enemy],'A','RECON',(5,3)))
        grid[:,6] = 1
        self.assertFalse(relevant_enemy_positions(grid,[char,enemy],'A','FLASH',(5,4)))

    def test_hidden_teacher_changes_label_but_never_features(self):
        scenario,char,_,state,_ = self.fixture()
        stream = io.StringIO()
        advisor = RecordingAdvisor(stream,'test','test:match','omoko_v1')
        teacher_enemy = actor('teacher','D',(8,8))
        advisor.teacher_game = SimpleNamespace(grid=state['grid'],chars=[char,teacher_enemy],
                                               current_round=1,_smoke_cells=lambda:set())
        advisor.observe(scenario,char,state,'omoko_v1')
        game = SimpleNamespace(_projectile_path=lambda s,t:[s,t])
        advisor.advise(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),game)
        teacher_enemy.pos = [0,0]
        advisor.advise(scenario,char,state,'omoko_v1','RECON',(8,8),(8,8),game)
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([r['label'] for r in rows],[1,0])
        self.assertEqual(rows[0]['features'],rows[1]['features'])

    def test_checkpoint_load_preserves_rng_and_feature_contract(self):
        scenario,_,_,_,_ = self.fixture()
        adviser = self.model_advisor(scenario)
        checkpoint = dict(adviser.checkpoint,version=VERSION,label_version=LABEL_VERSION,
                          wall_los_version=WALL_LOS_VERSION,feature_names=feature_names(),
                          model_state_dict=adviser.model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'adviser.pt'
            torch.save(checkpoint,path)
            before = torch.get_rng_state().clone()
            loaded = FixedAbilityAdvisor(path)
            self.assertIsNotNone(loaded.model)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_suppressed_point_does_not_hide_next_point_or_spend_resource(self):
        original = get_scenario('A1')
        points = tuple((int(r),int(c)) for r,c in np.argwhere(original.grid != 1))[:2]
        scenario = replace(original,smoke_points=points,smoke_trigger_bfs_distance=1000)
        class SelectSecond(FixedAbilityAdvisor):
            def advise(self,scenario,char,state,opponent,ability,point,aim,game):
                return Advice(point == points[1],'test')
        char = actor(pos=points[0])
        route = SimpleNamespace(scenario=scenario,set_game=lambda game:None)
        controller = ConconAttackerController(route_controller=route,ability_advisor=SelectSecond(None))
        controller.set_game(SimpleNamespace(grid=scenario.grid,chars=[char],smokes=[]))
        _,payload = controller.decide_move(char,dict(grid=scenario.grid,chars=[char],battle_tick=0))
        self.assertEqual(payload['target'],points[1])
        self.assertEqual(char.smoke_charges,1)
        self.assertFalse(controller.fixed_smokes.used_points)

    def test_train_validation_test_keep_every_match_and_history_together(self):
        rows = [dict(match_id=f'm{i}',opponent='omoko_v1',round_number=r)
                for i in range(10) for r in range(1,13)]
        folds = split_matches(rows,0)
        sets = [{r['match_id'] for r in fold} for fold in folds.values()]
        self.assertTrue(all(not sets[a].intersection(sets[b]) for a in range(3) for b in range(a)))
        self.assertTrue(all(sum(r['match_id'] == match for r in fold) == 12
                            for fold in folds.values() for match in {r['match_id'] for r in fold}))

    def test_no_validation_support_or_missed_enemies_disables_saving(self):
        rows = [dict(group='g',label=int(i == 0),match_id=str(i),round_number=1) for i in range(25)]
        report = classify_report(rows,[.01]*25,.1)['g']
        self.assertFalse(safe_group(report,20,.05))
        report = classify_report(rows,[.9]+[.01]*24,.1)['g']
        self.assertTrue(safe_group(report,20,.05))
        self.assertFalse(safe_group(report,30,.05))


if __name__ == '__main__':
    unittest.main()
