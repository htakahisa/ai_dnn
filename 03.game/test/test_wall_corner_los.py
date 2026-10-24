"""Wall-corner rays must agree for the engine, shooting and public visibility."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
from abilities_los import AbilityLosMixin
from grid_lines import wall_line_cells, line_cells
from grid_visibility import visible_cells, _wall_rays


class Geometry(AbilityLosMixin):
    def __init__(self,grid):
        self.grid = grid
        self.smokes = []
        self.chars = []


class CornerLosTests(unittest.TestCase):
    def test_reported_two_wall_layout_blocks_in_both_directions(self):
        game = Geometry(np.array([[0,0],[0,1],[0,1]]))
        a = NS(pos=[0,1],reveal_remaining=0,sees_through_smoke=False)
        b = NS(pos=[2,0],reveal_remaining=10,sees_through_smoke=False)
        for start,end in ((a,b),(b,a)):
            self.assertFalse(game.check_cell_line_of_sight(start.pos,end.pos))
            self.assertFalse(game.check_line_of_sight(start,end))
            self.assertFalse(game.check_shot_line_of_sight(start,end))
        self.assertNotIn((2,0),visible_cells(game.grid,[((0,1),(1,-1))],()))

    def test_exact_corner_is_blocked_but_open_diagonal_is_visible(self):
        game = Geometry(np.array([[0,1],[0,0]]))
        self.assertFalse(game.check_cell_line_of_sight((0,0),(1,1)))
        game.grid[0,1]=0
        self.assertTrue(game.check_cell_line_of_sight((0,0),(1,1)))
        self.assertTrue(game.check_cell_line_of_sight((1,1),(0,0)))

    def test_all_ray_directions_are_symmetric_and_batch_matches(self):
        shape=(7,9)
        for origin in ((0,0),(2,7),(6,8)):
            rays=_wall_rays(shape,origin)
            for r in range(shape[0]):
                for c in range(shape[1]):
                    target=(r,c)
                    expected=set(wall_line_cells(origin,target))
                    self.assertEqual(expected,set(wall_line_cells(target,origin)))
                    actual={(int(n)//shape[1],int(n)%shape[1]) for n in rays[:,r*shape[1]+c]}
                    self.assertEqual(expected,actual)

    def test_projectile_path_is_preserved(self):
        self.assertEqual(line_cells((0,1),(2,0)),[(0,1),(1,0),(2,0)])
        self.assertIn((1,1),wall_line_cells((0,1),(2,0)))


if __name__=='__main__':
    unittest.main()
