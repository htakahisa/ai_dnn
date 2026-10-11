"""Match-winning plant conditions rather than pristine HP or empty survival."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from touyama_v3.tv3_train_attacker_plant import plant_advantage_summary,best_rank


class PlantBestSelectionTests(unittest.TestCase):
    def row(self,alive,enemy,planted=True,own_abilities=2,enemy_abilities=1):
        return dict(planted=planted,alive=alive,enemy_alive=enemy,
            remaining_abilities_alive=own_abilities,enemy_remaining_abilities_alive=enemy_abilities)

    def rank(self,rows,**extra):
        return best_rank(dict(plant_advantage_summary(rows),mean_damage=200.,mean_plant_ticks=60.,**extra))

    def test_one_v_four_is_not_a_competitive_success_and_timeouts_are_not_advantaged_plants(self):
        summary=plant_advantage_summary([self.row(1,4),self.row(5,2,False),self.row(3,2),self.row(3,3)])
        self.assertEqual(summary['advantaged_plant_rate'],.25)
        self.assertEqual(summary['competitive_plant_rate'],.5)
        self.assertEqual(summary['disadvantaged_plants'],1)
        self.assertGreater(self.rank([self.row(3,2)]),self.rank([self.row(5,2,False)]))
        self.assertGreater(self.rank([self.row(3,3)]),self.rank([self.row(1,4)]))

    def test_man_advantage_then_competitive_plant_then_abilities(self):
        # First priority beats a higher raw/competitive success rate.
        self.assertGreater(self.rank([self.row(3,2),self.row(5,2,False)]),
            self.rank([self.row(3,3),self.row(3,3)]))
        # With the same advantaged success, equal-count plants beat losing plants.
        self.assertGreater(self.rank([self.row(3,2),self.row(3,3)]),
            self.rank([self.row(3,2),self.row(1,4)]))
        # Remaining ordinary charges compare only living teams at effective plants.
        self.assertGreater(self.rank([self.row(3,2,own_abilities=3,enemy_abilities=1)]),
            self.rank([self.row(3,2,own_abilities=1,enemy_abilities=3)]))

    def test_missing_enemy_resources_are_unknown_not_zero(self):
        row=self.row(3,2); row.pop('enemy_remaining_abilities_alive')
        summary=plant_advantage_summary([row])
        self.assertIsNone(summary['ability_advantage_plant_rate'])
        with self.assertRaisesRegex(ValueError,'re-evaluate'):
            best_rank({**summary,'mean_damage':200.,'mean_plant_ticks':60.})

    def test_low_damage_and_raw_failure_plants_do_not_override_priorities(self):
        rows=[self.row(3,2)]
        self.assertEqual(self.rank(rows,low_damage_plant_rate=0.,plant_rate=.5),
                         self.rank(rows,low_damage_plant_rate=1.,plant_rate=1.))


if __name__=='__main__':
    unittest.main()
