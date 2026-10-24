"""Failure timing and positive plant-quality rewards; no optimizer steps."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from touyama_v3.tv3_train_attacker_plant import (
    terminal_bonus,apply_terminal_bonus,PLANT_FAILURE_REWARD,PLANT_TIMEOUT_REWARD,warm_start_source,
    warm_start_schema_compatible,
)
from touyama_v3.tv3_learn_attacker_plant import GAMMA


class PlantTimeoutRewardTests(unittest.TestCase):
    def test_v13_to_v14_initialization_requires_identical_observations_and_geometry(self):
        from touyama_v3.tv3_scenario import Scenario
        from touyama_v3.tv3_learn_attacker_plant import policy_schema
        scenario=Scenario()
        old,new=policy_schema(scenario,13),policy_schema(scenario,14)
        self.assertTrue(warm_start_schema_compatible(old,new))
        self.assertFalse(warm_start_schema_compatible(new,old))
        self.assertFalse(warm_start_schema_compatible(old,{**new,'obs_dim':1}))
        self.assertFalse(warm_start_schema_compatible(old,{**new,'scenario':'changed'}))
        v16=policy_schema(scenario,16)
        self.assertTrue(warm_start_schema_compatible(old,v16))
        self.assertFalse(warm_start_schema_compatible(old,{**v16,'scenario':'changed'}))
        self.assertFalse(warm_start_schema_compatible(v16,old))
        v18=policy_schema(scenario,18)
        self.assertTrue(warm_start_schema_compatible(v16,v18))
        self.assertFalse(warm_start_schema_compatible(v16,{**v18,'obs_dim':1}))
        self.assertFalse(warm_start_schema_compatible(v16,{**v18,'scenario':'changed'}))
        self.assertFalse(warm_start_schema_compatible(v18,v16))

    def reward(self,alive,enemy,planted=True,reason='planted'):
        return terminal_bonus(dict(planted=planted,alive=alive,enemy_alive=enemy,reason=reason,
            initial_team_hp=500.,remaining_hp=alive*100.),5,.5)

    def test_every_plant_is_positive_and_man_ratio_improves_it(self):
        self.assertGreater(self.reward(1,5),0.)
        self.assertGreater(self.reward(1,4),self.reward(1,5))
        self.assertGreater(self.reward(3,4),self.reward(1,4))
        # Same surviving team, health, resources; enemy count is the only change.
        self.assertGreater(self.reward(3,2),self.reward(3,3))
        self.assertGreater(self.reward(3,3),self.reward(3,4))
        self.assertGreater(self.reward(3,2),self.reward(5,5))

    def test_relative_remaining_abilities_break_same_player_quality_ties(self):
        terminal=dict(planted=True,alive=3,enemy_alive=2,remaining_hp=300.,initial_team_hp=500.,
            remaining_abilities_alive=2,enemy_remaining_abilities_alive=1)
        good=terminal_bonus(terminal,5,0.)
        bad=terminal_bonus({**terminal,'enemy_remaining_abilities_alive':3},5,1.)
        self.assertGreater(good,bad)

    def test_resource_bonus_never_overrides_better_player_ratio_in_five_v_five(self):
        from fractions import Fraction
        cases=[(a,e,Fraction(a,a+e)) for a in range(1,6) for e in range(6)]
        for a,e,ratio in cases:
            for b,f,other in cases:
                if ratio>other:
                    good=terminal_bonus(dict(planted=True,alive=a,enemy_alive=e),5,0.)
                    bad=terminal_bonus(dict(planted=True,alive=b,enemy_alive=f),5,1.)
                    self.assertGreater(good,bad)

    def test_positive_plant_is_better_than_waiting_to_timeout(self):
        self.assertLess(PLANT_TIMEOUT_REWARD,PLANT_FAILURE_REWARD)
        self.assertGreater(self.reward(1,5),self.reward(5,5,False,'timeout'))
        self.assertEqual(self.reward(5,5,False,'timeout'),PLANT_TIMEOUT_REWARD)
        self.assertEqual(self.reward(1,5,False,'attacker_eliminated'),PLANT_FAILURE_REWARD)

    def test_delaying_failure_never_discounts_away_terminal_penalty(self):
        for reason,expected in [('timeout',PLANT_TIMEOUT_REWARD),('attacker_eliminated',PLANT_FAILURE_REWARD)]:
            for steps in (1,20,60,100):
                rows=[[None,0,0.,None,None,0.] for _ in range(steps)]
                terminal=dict(planted=False,alive=1,enemy_alive=5,reason=reason)
                apply_terminal_bonus(rows,{'player':steps-1},{'player':True},terminal,5,0.,{'player':steps})
                discounted=sum(GAMMA**i*r[2] for i,r in enumerate(rows))
                self.assertAlmostEqual(discounted,expected,places=10)

    def test_dead_actor_does_not_receive_later_success_or_living_timeout_cost(self):
        rows=[[None,0,0.,None,None,1.],[None,0,0.,None,None,0.]]
        apply_terminal_bonus(rows,{'dead':0,'alive':1},{'dead':False,'alive':True},
            dict(planted=False,alive=1,enemy_alive=5,reason='timeout'),5,0.,{'dead':1,'alive':1})
        self.assertEqual(rows[0][2],PLANT_FAILURE_REWARD)
        self.assertEqual(rows[1][2],PLANT_TIMEOUT_REWARD)

    def test_preplant_enemy_elimination_is_a_win_not_a_failed_plant(self):
        self.assertGreater(self.reward(3,0,False,'defender_eliminated'),0.)
        self.assertGreater(terminal_bonus(dict(planted=False,alive=3,enemy_alive=0,
            reason='timeout',preplant_round_won=True),5,0.),0.)
        self.assertEqual(terminal_bonus(dict(planted=False,alive=3,enemy_alive=0,
            reason='defender_eliminated',preplant_round_won=False),5,0.),PLANT_FAILURE_REWARD)

    def test_warm_start_archives_exact_weights_and_rejects_changed_dependencies(self):
        import tempfile,hashlib,torch
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'best.pt'; archive=Path(temporary)/'sources'
            state=dict(schema={'version':13},analysis_hash='fixed',opponent='frc_v1',
                model={'weights':torch.ones(2)},completed_sets=90)
            torch.save(state,source)
            original=source.read_bytes()
            loaded,digest=warm_start_source(source,{'version':13},'fixed','frc_v1',archive)
            self.assertEqual(digest,hashlib.sha256(original).hexdigest())
            self.assertEqual((archive/'frc_v1'/f'{digest}.pt').read_bytes(),original)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(loaded['completed_sets'],90)
            self.assertEqual(warm_start_source(source,{'version':13},'changed','frc_v1',archive),(None,None))


if __name__=='__main__':
    unittest.main()
