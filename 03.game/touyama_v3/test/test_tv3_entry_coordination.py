"""Public timing, firing lanes and low-damage selection; no training execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace
import unittest
import numpy as np
from grid_lines import line_cells
from touyama_v3.tv3_attacker_entry_coordination import EntryCoordinator, entry_features, shared_fire_score, firing_lanes
from touyama_v3.tv3_train_attacker_plant import best_rank, low_damage_summary, terminal_bonus


class CoordinationTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((11,11),dtype=np.int32)
        self.grid[5,5] = 2
        self.scenario = SimpleNamespace(grid=self.grid, clear=lambda p,q: True,
                                        _line_cells=line_cells)
        self.route = SimpleNamespace(site='L',cells=((5,2),(5,3),(5,4),(5,5)))
        self.allies = (
            SimpleNamespace(slot=0,position=(5,4),alive=True,blind=0,facing='E'),
            SimpleNamespace(slot=1,position=(1,5),alive=True,blind=0,facing='S'))
        self.snapshot = SimpleNamespace(allies=self.allies,sightings=(),smoke_cells=(),tick=0,round_timer=60)

    def test_main_waits_for_other_entry_and_wait_is_bounded(self):
        coordinator = EntryCoordinator(self.scenario)
        flanks = {1: ((3,5),(4,5))}
        first = coordinator.observe(self.snapshot,self.route,flanks)
        self.assertTrue(first['wait'])
        self.assertEqual(first['ready'],{0})
        self.snapshot.tick=6
        self.assertFalse(coordinator.observe(self.snapshot,self.route,flanks)['wait'])

    def test_both_entries_ready_release_and_short_timer_does_not_wait(self):
        coordinator=EntryCoordinator(self.scenario)
        flanks={1: ((3,5),(4,5))}
        self.allies[1].position=(2,5)
        state=coordinator.observe(self.snapshot,self.route,flanks)
        self.assertTrue(state['all_ready'])
        self.assertFalse(state['wait'])
        self.allies[1].position=(1,5)
        self.snapshot.round_timer=10
        self.assertFalse(coordinator.observe(self.snapshot,self.route,flanks)['wait'])

    def test_facing_blind_smoke_and_ally_bodies_limit_shared_fire(self):
        target=(5,6)
        self.allies[1].position=(4,6)
        self.snapshot.sightings=(SimpleNamespace(position=target),)
        self.assertEqual(len(firing_lanes(self.scenario,self.snapshot,target)),2)
        self.assertGreater(shared_fire_score(self.scenario,self.snapshot),0)
        self.allies[1].facing='N'
        self.assertEqual(shared_fire_score(self.scenario,self.snapshot),0)
        self.allies[1].facing='S'
        self.allies[1].blind=1
        self.assertEqual(shared_fire_score(self.scenario,self.snapshot),0)
        self.allies[1].blind=0
        self.snapshot.smoke_cells=((5,5),)
        self.assertEqual(shared_fire_score(self.scenario,self.snapshot),0)
        self.snapshot.smoke_cells=()
        self.allies[1].position=(5,5)
        self.assertEqual(len(firing_lanes(self.scenario,self.snapshot,target)),1)

    def test_three_distinct_lanes_outscore_one_and_two(self):
        target=(5,6)
        self.allies[1].position=(4,6)
        self.snapshot.sightings=(SimpleNamespace(position=target),)
        two=shared_fire_score(self.scenario,self.snapshot)
        third=SimpleNamespace(slot=2,position=(6,6),alive=True,blind=0,facing='N')
        self.snapshot.allies=(*self.allies,third)
        self.assertEqual(len(firing_lanes(self.scenario,self.snapshot,target)),3)
        self.assertGreater(shared_fire_score(self.scenario,self.snapshot),two)

    def test_public_flash_and_readiness_are_visible_in_features(self):
        base=entry_features(self.scenario,self.snapshot,self.allies[0],(5,5),None)
        flashed=entry_features(self.scenario,self.snapshot,self.allies[0],(5,5),(5,6))
        self.assertEqual(len(base),12)
        self.assertEqual(base[0],0)
        self.assertEqual(flashed[0],1)
        self.assertNotEqual(base,flashed)

    def test_low_damage_plants_rank_above_high_success_with_heavy_losses(self):
        base={'plant_rate':1.,'viable_plant_rate':1.,'mean_alive':4.,'mean_damage':10.,
              'ability_reserve_rate':.5,'mean_ability_uses':2.,'mean_plant_ticks':40.}
        safe={**base,'low_damage_plant_rate':.9}
        damaged={**base,'low_damage_plant_rate':.1,'ability_reserve_rate':1.}
        self.assertGreater(best_rank(safe),best_rank(damaged))
        records=[{'planted':True,'alive':5,'remaining_hp':500,'initial_team_hp':500},
                 {'planted':True,'alive':4,'remaining_hp':390,'initial_team_hp':500},
                 {'planted':False,'alive':5,'remaining_hp':500,'initial_team_hp':500}]
        self.assertEqual(low_damage_summary(records)['low_damage_plants'],1)
        terminal={'planted':True,'alive':5,'initial_team_hp':500,'remaining_hp':500}
        self.assertGreater(terminal_bonus(terminal,5,0.),
                           terminal_bonus({**terminal,'remaining_hp':200},5,1.))


if __name__=='__main__':
    unittest.main()
